# -*- coding: utf-8 -*-
"""Multi-step kernel-fusion benchmark for Double-Gyre FTLE.

V1 launches one RK4 kernel per physical time step.  V2 keeps each particle's
state in registers for ``chunk_steps`` consecutive RK4 steps before writing it
back to global memory.  A scan over chunk sizes quantifies the effect of fewer
kernel launches and less global-memory traffic.

This file depends on the two files from the previous stages being in the same
directory:

    double_gyre_ftle_baseline.py
    double_gyre_ftle_cuda_v1.py

Recommended A100 scan:

    python double_gyre_ftle_cuda_v2.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --threads 256 \
        --chunk-list 1,2,5,10,20,50 \
        --warmup-runs 1 --repeats 5 --skip-cpu
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from numba import cuda

from double_gyre_ftle_baseline import DoubleGyreConfig, run_reference
from double_gyre_ftle_cuda_v1 import (
    _rhs,
    error_metrics,
    ftle_kernel,
    initial_state,
    state_metrics,
)


@cuda.jit
def rk4_chunk_kernel(
    x,
    y,
    f11,
    f12,
    f21,
    f22,
    t_start,
    dt,
    chunk_steps,
    amplitude,
    epsilon,
    omega,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Advance one particle through several RK4 steps before writing state."""
    i = cuda.grid(1)
    if i >= x.size:
        return

    # These scalars are private to the thread.  Keeping them live across the
    # loop is the central V2 optimization: no global state traffic per step.
    x0 = x[i]
    y0 = y[i]
    f11_0 = f11[i]
    f12_0 = f12[i]
    f21_0 = f21[i]
    f22_0 = f22[i]

    for local_step in range(chunk_steps):
        t = t_start + local_step * dt

        k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs(
            t,
            x0,
            y0,
            f11_0,
            f12_0,
            f21_0,
            f22_0,
            amplitude,
            epsilon,
            omega,
        )

        half_dt = 0.5 * dt
        k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs(
            t + half_dt,
            x0 + half_dt * k1x,
            y0 + half_dt * k1y,
            f11_0 + half_dt * k1f11,
            f12_0 + half_dt * k1f12,
            f21_0 + half_dt * k1f21,
            f22_0 + half_dt * k1f22,
            amplitude,
            epsilon,
            omega,
        )

        k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs(
            t + half_dt,
            x0 + half_dt * k2x,
            y0 + half_dt * k2y,
            f11_0 + half_dt * k2f11,
            f12_0 + half_dt * k2f12,
            f21_0 + half_dt * k2f21,
            f22_0 + half_dt * k2f22,
            amplitude,
            epsilon,
            omega,
        )

        k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs(
            t + dt,
            x0 + dt * k3x,
            y0 + dt * k3y,
            f11_0 + dt * k3f11,
            f12_0 + dt * k3f12,
            f21_0 + dt * k3f21,
            f22_0 + dt * k3f22,
            amplitude,
            epsilon,
            omega,
        )

        sixth_dt = dt / 6.0
        x0 = x0 + sixth_dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y0 = y0 + sixth_dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        f11_0 = f11_0 + sixth_dt * (
            k1f11 + 2.0 * k2f11 + 2.0 * k3f11 + k4f11
        )
        f12_0 = f12_0 + sixth_dt * (
            k1f12 + 2.0 * k2f12 + 2.0 * k3f12 + k4f12
        )
        f21_0 = f21_0 + sixth_dt * (
            k1f21 + 2.0 * k2f21 + 2.0 * k3f21 + k4f21
        )
        f22_0 = f22_0 + sixth_dt * (
            k1f22 + 2.0 * k2f22 + 2.0 * k3f22 + k4f22
        )

        if x0 < x_min:
            x0 = x_min
        elif x0 > x_max:
            x0 = x_max

        if y0 < y_min:
            y0 = y_min
        elif y0 > y_max:
            y0 = y_max

    x[i] = x0
    y[i] = y0
    f11[i] = f11_0
    f12[i] = f12_0
    f21[i] = f21_0
    f22[i] = f22_0


def parse_chunk_list(text: str) -> list[int]:
    chunks = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("all --chunk-list values must be positive integers")
        if value not in chunks:
            chunks.append(value)
    if not chunks:
        raise ValueError("--chunk-list must contain at least one value")
    if 1 not in chunks:
        raise ValueError("--chunk-list must include 1 as the unfused GPU baseline")
    # The skip-CPU mode uses chunk=1 as its numerical reference, so evaluate
    # that configuration first even if the CLI list was given out of order.
    return [1] + [value for value in chunks if value != 1]


def compile_warmup(dtype: np.dtype, cfg: DoubleGyreConfig, threads: int) -> None:
    """Compile the chunk and FTLE kernels on a tiny input; no timing is used."""
    state = initial_state(2, 2, dtype, cfg)
    device_state = tuple(cuda.to_device(array) for array in state)
    d_ftle = cuda.device_array(state[0].size, dtype=dtype)
    scalar = dtype.type
    blocks = (state[0].size + threads - 1) // threads

    rk4_chunk_kernel[blocks, threads](
        *device_state,
        scalar(cfg.t0),
        scalar(0.01),
        1,
        scalar(cfg.amplitude),
        scalar(cfg.epsilon),
        scalar(cfg.omega),
        scalar(cfg.x_min),
        scalar(cfg.x_max),
        scalar(cfg.y_min),
        scalar(cfg.y_max),
    )
    ftle_kernel[blocks, threads](*device_state[2:], scalar(0.01), d_ftle)
    cuda.synchronize()


def run_chunk_once(
    host_state: tuple[np.ndarray, ...],
    dt: float,
    tau: float,
    chunk_steps: int,
    threads: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
    copy_state: bool,
) -> tuple[np.ndarray, dict[str, float], tuple[np.ndarray, ...] | None]:
    """Run one complete trajectory for one selected fusion chunk size."""
    scalar = dtype.type
    particles = host_state[0].size
    total_steps = int(round(tau / dt))
    blocks = (particles + threads - 1) // threads

    end_to_end_start = perf_counter()
    h2d_start = perf_counter()
    device_state = tuple(cuda.to_device(array) for array in host_state)
    d_ftle = cuda.device_array(particles, dtype=dtype)
    cuda.synchronize()
    h2d_seconds = perf_counter() - h2d_start

    start_event = cuda.event(timing=True)
    end_event = cuda.event(timing=True)
    start_event.record()

    completed_steps = 0
    launches = 0
    while completed_steps < total_steps:
        current_chunk = min(chunk_steps, total_steps - completed_steps)
        t_start = cfg.t0 + completed_steps * dt
        rk4_chunk_kernel[blocks, threads](
            *device_state,
            scalar(t_start),
            scalar(dt),
            current_chunk,
            scalar(cfg.amplitude),
            scalar(cfg.epsilon),
            scalar(cfg.omega),
            scalar(cfg.x_min),
            scalar(cfg.x_max),
            scalar(cfg.y_min),
            scalar(cfg.y_max),
        )
        completed_steps += current_chunk
        launches += 1

    ftle_kernel[blocks, threads](*device_state[2:], scalar(tau), d_ftle)
    end_event.record()
    end_event.synchronize()
    kernel_seconds = cuda.event_elapsed_time(start_event, end_event) / 1000.0

    d2h_start = perf_counter()
    ftle = d_ftle.copy_to_host()
    cuda.synchronize()
    d2h_seconds = perf_counter() - d2h_start
    end_to_end_seconds = perf_counter() - end_to_end_start

    final_state = None
    if copy_state:
        final_state = tuple(array.copy_to_host() for array in device_state)
        cuda.synchronize()

    metrics = {
        "chunk_steps": chunk_steps,
        "rk4_kernel_launches": launches,
        "kernel_seconds": float(kernel_seconds),
        "h2d_seconds": float(h2d_seconds),
        "d2h_seconds": float(d2h_seconds),
        "end_to_end_seconds": float(end_to_end_seconds),
        "kernel_particle_steps_per_second": float(
            particles * total_steps / kernel_seconds
        ),
    }
    return ftle, metrics, final_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=2001)
    parser.add_argument("--ny", type=int, default=1001)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--chunk-list", type=str, default="1,2,5,10,20,50")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cuda_v2_chunk_scan.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chunks = parse_chunk_list(args.chunk_list)

    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must both be at least 2")
    if args.dt <= 0.0 or args.tau <= 0.0:
        raise ValueError("--dt and --tau must be positive")
    if args.threads < 1 or args.threads > 1024:
        raise ValueError("--threads must be in [1, 1024]")
    if args.warmup_runs < 1 or args.repeats < 1:
        raise ValueError("--warmup-runs and --repeats must both be positive")

    total_steps = int(round(args.tau / args.dt))
    actual_tau = total_steps * args.dt
    if not np.isclose(actual_tau, args.tau, rtol=0.0, atol=1e-12):
        raise ValueError("--tau must be an integer multiple of --dt")
    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check LD_LIBRARY_PATH and the NVIDIA driver")

    dtype = np.dtype(args.dtype)
    cfg = DoubleGyreConfig()
    host_state = initial_state(args.nx, args.ny, dtype, cfg)
    particles = host_state[0].size

    print("Double-Gyre FTLE CUDA V2: multi-step fusion scan")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, dtype={dtype.name}"
    )
    print(
        f"threads={args.threads}, chunks={chunks}, "
        f"warmup_runs={args.warmup_runs}, repeats={args.repeats}"
    )
    print("Compiling CUDA kernels...")
    compile_warmup(dtype, cfg, args.threads)

    cpu_ftle = None
    cpu_metrics = None
    if not args.skip_cpu:
        print("Running CPU reference...")
        _, _, snapshots, cpu_metrics = run_reference(
            args.nx,
            args.ny,
            args.dt,
            [actual_tau],
            dtype,
            cfg,
        )
        cpu_ftle = snapshots[actual_tau].ravel()

    rows = []
    chunk_one_ftle = None

    for chunk_steps in chunks:
        print(f"\nchunk_steps={chunk_steps}")
        print("  full-size warmup...", end="", flush=True)
        for _ in range(args.warmup_runs):
            run_chunk_once(
                host_state,
                args.dt,
                actual_tau,
                chunk_steps,
                args.threads,
                dtype,
                cfg,
                copy_state=False,
            )
        print(" done")

        measured = []
        last_ftle = None
        last_state = None
        for repeat in range(args.repeats):
            copy_state = repeat == args.repeats - 1
            last_ftle, metrics, final_state = run_chunk_once(
                host_state,
                args.dt,
                actual_tau,
                chunk_steps,
                args.threads,
                dtype,
                cfg,
                copy_state=copy_state,
            )
            if final_state is not None:
                last_state = final_state
            measured.append(metrics)
            print(
                f"  repeat={repeat + 1}: kernel={metrics['kernel_seconds']:.6f}s, "
                f"end_to_end={metrics['end_to_end_seconds']:.6f}s"
            )

        kernel_seconds = median(item["kernel_seconds"] for item in measured)
        end_to_end_seconds = median(
            item["end_to_end_seconds"] for item in measured
        )
        h2d_seconds = median(item["h2d_seconds"] for item in measured)
        d2h_seconds = median(item["d2h_seconds"] for item in measured)

        row = {
            "chunk_steps": chunk_steps,
            "rk4_kernel_launches": math.ceil(total_steps / chunk_steps),
            "gpu_kernel_seconds_median": float(kernel_seconds),
            "gpu_end_to_end_seconds_median": float(end_to_end_seconds),
            "gpu_h2d_seconds_median": float(h2d_seconds),
            "gpu_d2h_seconds_median": float(d2h_seconds),
            "gpu_kernel_particle_steps_per_second": float(
                particles * total_steps / kernel_seconds
            ),
        }
        row.update(state_metrics(last_state, cfg))

        if chunk_steps == 1:
            chunk_one_ftle = last_ftle.copy()

        reference = cpu_ftle if cpu_ftle is not None else chunk_one_ftle
        row.update(error_metrics(reference, last_ftle))
        rows.append(row)

    baseline_kernel = next(
        row["gpu_kernel_seconds_median"]
        for row in rows
        if row["chunk_steps"] == 1
    )
    baseline_e2e = next(
        row["gpu_end_to_end_seconds_median"]
        for row in rows
        if row["chunk_steps"] == 1
    )

    for row in rows:
        row["kernel_speedup_vs_chunk1"] = baseline_kernel / row[
            "gpu_kernel_seconds_median"
        ]
        row["end_to_end_speedup_vs_chunk1"] = baseline_e2e / row[
            "gpu_end_to_end_seconds_median"
        ]
        if cpu_metrics is not None:
            row["kernel_speedup_vs_cpu"] = (
                cpu_metrics["compute_seconds"] / row["gpu_kernel_seconds_median"]
            )

    best = min(rows, key=lambda row: row["gpu_kernel_seconds_median"])
    report = {
        "configuration": {
            "nx": args.nx,
            "ny": args.ny,
            "particles": particles,
            "steps": total_steps,
            "dt": args.dt,
            "tau": actual_tau,
            "dtype": dtype.name,
            "threads": args.threads,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "error_reference": "cpu" if cpu_ftle is not None else "gpu_chunk1",
        },
        "results": rows,
        "best_chunk_steps": best["chunk_steps"],
        "best_kernel_seconds": best["gpu_kernel_seconds_median"],
        "best_kernel_speedup_vs_chunk1": best["kernel_speedup_vs_chunk1"],
    }

    print("\nChunk-fusion summary")
    print(
        "chunk  launches  kernel(s)  throughput(particle-steps/s)  "
        "speedup  rel_L2_error"
    )
    for row in rows:
        print(
            f"{row['chunk_steps']:5d}  "
            f"{row['rk4_kernel_launches']:8d}  "
            f"{row['gpu_kernel_seconds_median']:9.6f}  "
            f"{row['gpu_kernel_particle_steps_per_second']:28.3f}  "
            f"{row['kernel_speedup_vs_chunk1']:7.3f}  "
            f"{row['relative_l2_ftle_error']:.3e}"
        )

    print(
        f"best chunk_steps={best['chunk_steps']}, "
        f"kernel speedup={best['kernel_speedup_vs_chunk1']:.3f}x"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    tolerance = 1e-8 if dtype == np.dtype("float64") else 1e-4
    for row in rows:
        if not row["gpu_finite"] or row["gpu_max_boundary_error"] > 0.0:
            raise RuntimeError(
                f"chunk_steps={row['chunk_steps']} produced invalid GPU state"
            )
        if row["relative_l2_ftle_error"] > tolerance:
            raise RuntimeError(
                f"chunk_steps={row['chunk_steps']} failed correctness: "
                f"relative L2 error {row['relative_l2_ftle_error']:.3e}"
            )


if __name__ == "__main__":
    main()
