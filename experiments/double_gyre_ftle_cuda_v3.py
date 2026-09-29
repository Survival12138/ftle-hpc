# -*- coding: utf-8 -*-
"""Strict-vs-fastmath A/B benchmark for Double-Gyre FTLE on CUDA.

V2 established multi-step fusion and the block-size scan selected 64 threads
for the A100 test case.  V3 changes exactly one optimization dimension: the
RK4 integration kernel is compiled either with strict floating-point semantics
or with Numba-CUDA ``fastmath=True``.  FTLE post-processing stays strict in
both cases, so the comparison isolates the trajectory/deformation integration.

The strict result is the numerical reference for the fastmath result.  This is
an optimization experiment, not permission to silently weaken accuracy.

Recommended A100 run:

    python double_gyre_ftle_cuda_v3.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --threads 64 --chunk-steps 50 \
        --warmup-runs 1 --repeats 5 \
        --output results/cuda_v3_fastmath.json

Keep these files in the same directory:

    double_gyre_ftle_baseline.py
    double_gyre_ftle_cuda_v1.py
    double_gyre_ftle_cuda_v2.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
from numba import cuda

from double_gyre_ftle_baseline import DoubleGyreConfig
from double_gyre_ftle_cuda_v1 import (
    error_metrics,
    ftle_kernel,
    initial_state,
    state_metrics,
)
from double_gyre_ftle_cuda_v2 import rk4_chunk_kernel


@cuda.jit(device=True, inline="always", fastmath=True)
def _rhs_fast(
    t,
    x,
    y,
    f11,
    f12,
    f21,
    f22,
    amplitude,
    epsilon,
    omega,
):
    """Fastmath version of the Double-Gyre state and tangent-linear RHS."""
    sin_omega_t = math.sin(omega * t)
    a = epsilon * sin_omega_t
    b = 1.0 - 2.0 * epsilon * sin_omega_t

    f = a * x * x + b * x
    dfdx = 2.0 * a * x + b
    d2fdx2 = 2.0 * a

    pi_f = math.pi * f
    pi_y = math.pi * y
    sin_f = math.sin(pi_f)
    cos_f = math.cos(pi_f)
    sin_y = math.sin(pi_y)
    cos_y = math.cos(pi_y)

    u = -math.pi * amplitude * sin_f * cos_y
    v = math.pi * amplitude * cos_f * sin_y * dfdx

    ux = -(math.pi * math.pi) * amplitude * cos_f * cos_y * dfdx
    uy = (math.pi * math.pi) * amplitude * sin_f * sin_y
    vx = math.pi * amplitude * sin_y * (
        -math.pi * sin_f * dfdx * dfdx + cos_f * d2fdx2
    )
    vy = (math.pi * math.pi) * amplitude * cos_f * cos_y * dfdx

    df11 = ux * f11 + uy * f21
    df12 = ux * f12 + uy * f22
    df21 = vx * f11 + vy * f21
    df22 = vx * f12 + vy * f22
    return u, v, df11, df12, df21, df22


@cuda.jit(fastmath=True)
def rk4_chunk_fastmath_kernel(
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
    """Advance one particle for ``chunk_steps`` using fastmath RK4."""
    i = cuda.grid(1)
    if i >= x.size:
        return

    x0 = x[i]
    y0 = y[i]
    f11_0 = f11[i]
    f12_0 = f12[i]
    f21_0 = f21[i]
    f22_0 = f22[i]

    for local_step in range(chunk_steps):
        t = t_start + local_step * dt

        k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs_fast(
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
        k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs_fast(
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

        k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs_fast(
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

        k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs_fast(
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


def _launch_integrator(
    kernel: Any,
    blocks: int,
    threads: int,
    device_state: tuple[Any, ...],
    scalar: Any,
    cfg: DoubleGyreConfig,
    t_start: float,
    dt: float,
    chunk_steps: int,
) -> None:
    kernel[blocks, threads](
        *device_state,
        scalar(t_start),
        scalar(dt),
        chunk_steps,
        scalar(cfg.amplitude),
        scalar(cfg.epsilon),
        scalar(cfg.omega),
        scalar(cfg.x_min),
        scalar(cfg.x_max),
        scalar(cfg.y_min),
        scalar(cfg.y_max),
    )


def compile_warmup(dtype: np.dtype, cfg: DoubleGyreConfig, threads: int) -> None:
    """Compile strict, fastmath, and FTLE kernels on a tiny input."""
    host_state = initial_state(2, 2, dtype, cfg)
    scalar = dtype.type

    for kernel in (rk4_chunk_kernel, rk4_chunk_fastmath_kernel):
        device_state = tuple(cuda.to_device(array) for array in host_state)
        blocks = (host_state[0].size + threads - 1) // threads
        _launch_integrator(
            kernel,
            blocks,
            threads,
            device_state,
            scalar,
            cfg,
            cfg.t0,
            0.01,
            1,
        )

    d_ftle = cuda.device_array(host_state[0].size, dtype=dtype)
    ftle_kernel[blocks, threads](*device_state[2:], scalar(0.01), d_ftle)
    cuda.synchronize()


def run_once(
    kernel: Any,
    host_state: tuple[np.ndarray, ...],
    dt: float,
    tau: float,
    chunk_steps: int,
    threads: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
    copy_state: bool,
) -> tuple[np.ndarray, dict[str, float], tuple[np.ndarray, ...] | None]:
    """Run one full trajectory with a selected integration kernel."""
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

    integration_start = cuda.event(timing=True)
    integration_end = cuda.event(timing=True)
    all_kernels_end = cuda.event(timing=True)
    integration_start.record()

    completed_steps = 0
    launches = 0
    while completed_steps < total_steps:
        current_chunk = min(chunk_steps, total_steps - completed_steps)
        t_start = cfg.t0 + completed_steps * dt
        _launch_integrator(
            kernel,
            blocks,
            threads,
            device_state,
            scalar,
            cfg,
            t_start,
            dt,
            current_chunk,
        )
        completed_steps += current_chunk
        launches += 1

    integration_end.record()
    ftle_kernel[blocks, threads](*device_state[2:], scalar(tau), d_ftle)
    all_kernels_end.record()
    all_kernels_end.synchronize()

    integration_seconds = (
        cuda.event_elapsed_time(integration_start, integration_end) / 1000.0
    )
    ftle_seconds = (
        cuda.event_elapsed_time(integration_end, all_kernels_end) / 1000.0
    )

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
        "rk4_kernel_launches": launches,
        "integration_seconds": float(integration_seconds),
        "ftle_seconds": float(ftle_seconds),
        "all_kernel_seconds": float(integration_seconds + ftle_seconds),
        "h2d_seconds": float(h2d_seconds),
        "d2h_seconds": float(d2h_seconds),
        "end_to_end_seconds": float(end_to_end_seconds),
        "particle_steps_per_second": float(
            particles * total_steps / integration_seconds
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
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--error-tolerance",
        type=float,
        default=None,
        help="relative L2 FTLE tolerance; default: 1e-8 (float64), 1e-4 (float32)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cuda_v3_fastmath.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must both be at least 2")
    if args.dt <= 0.0 or args.tau <= 0.0:
        raise ValueError("--dt and --tau must be positive")
    if args.threads < 1 or args.threads > 1024:
        raise ValueError("--threads must be in [1, 1024]")
    if args.chunk_steps < 1:
        raise ValueError("--chunk-steps must be positive")
    if args.warmup_runs < 1 or args.repeats < 1:
        raise ValueError("--warmup-runs and --repeats must both be positive")
    if args.error_tolerance is not None and args.error_tolerance < 0.0:
        raise ValueError("--error-tolerance must be non-negative")

    total_steps = int(round(args.tau / args.dt))
    actual_tau = total_steps * args.dt
    if not np.isclose(actual_tau, args.tau, rtol=0.0, atol=1e-12):
        raise ValueError("--tau must be an integer multiple of --dt")
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; check LD_LIBRARY_PATH and the NVIDIA driver"
        )

    dtype = np.dtype(args.dtype)
    tolerance = args.error_tolerance
    if tolerance is None:
        tolerance = 1e-8 if dtype == np.dtype("float64") else 1e-4

    cfg = DoubleGyreConfig()
    host_state = initial_state(args.nx, args.ny, dtype, cfg)
    particles = host_state[0].size

    print("Double-Gyre FTLE CUDA V3: strict vs fastmath")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, dtype={dtype.name}"
    )
    print(
        f"threads={args.threads}, chunk_steps={args.chunk_steps}, "
        f"warmup_runs={args.warmup_runs}, repeats={args.repeats}"
    )
    print(f"fastmath relative-L2 tolerance={tolerance:.3e}")
    print("Compiling CUDA kernels...")
    compile_warmup(dtype, cfg, args.threads)

    variants = (
        ("strict", rk4_chunk_kernel),
        ("fastmath", rk4_chunk_fastmath_kernel),
    )
    rows: list[dict[str, Any]] = []
    strict_ftle = None

    for name, kernel in variants:
        print(f"\nmode={name}")
        print("  full-size warmup...", end="", flush=True)
        for _ in range(args.warmup_runs):
            run_once(
                kernel,
                host_state,
                args.dt,
                actual_tau,
                args.chunk_steps,
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
            last_ftle, metrics, final_state = run_once(
                kernel,
                host_state,
                args.dt,
                actual_tau,
                args.chunk_steps,
                args.threads,
                dtype,
                cfg,
                copy_state=copy_state,
            )
            measured.append(metrics)
            if final_state is not None:
                last_state = final_state
            print(
                f"  repeat={repeat + 1}: "
                f"integration={metrics['integration_seconds']:.6f}s, "
                f"end_to_end={metrics['end_to_end_seconds']:.6f}s"
            )

        integration_seconds = median(
            item["integration_seconds"] for item in measured
        )
        ftle_seconds = median(item["ftle_seconds"] for item in measured)
        all_kernel_seconds = median(
            item["all_kernel_seconds"] for item in measured
        )
        end_to_end_seconds = median(
            item["end_to_end_seconds"] for item in measured
        )

        row: dict[str, Any] = {
            "mode": name,
            "rk4_kernel_launches": math.ceil(total_steps / args.chunk_steps),
            "gpu_integration_seconds_median": float(integration_seconds),
            "gpu_ftle_seconds_median": float(ftle_seconds),
            "gpu_all_kernel_seconds_median": float(all_kernel_seconds),
            "gpu_end_to_end_seconds_median": float(end_to_end_seconds),
            "gpu_h2d_seconds_median": float(
                median(item["h2d_seconds"] for item in measured)
            ),
            "gpu_d2h_seconds_median": float(
                median(item["d2h_seconds"] for item in measured)
            ),
            "gpu_particle_steps_per_second": float(
                particles * total_steps / integration_seconds
            ),
        }
        row.update(state_metrics(last_state, cfg))

        if name == "strict":
            strict_ftle = last_ftle.copy()
            row.update(error_metrics(strict_ftle, last_ftle))
        else:
            row.update(error_metrics(strict_ftle, last_ftle))
        rows.append(row)

    strict_row, fast_row = rows
    fast_row["integration_speedup_vs_strict"] = (
        strict_row["gpu_integration_seconds_median"]
        / fast_row["gpu_integration_seconds_median"]
    )
    fast_row["all_kernel_speedup_vs_strict"] = (
        strict_row["gpu_all_kernel_seconds_median"]
        / fast_row["gpu_all_kernel_seconds_median"]
    )
    fast_row["end_to_end_speedup_vs_strict"] = (
        strict_row["gpu_end_to_end_seconds_median"]
        / fast_row["gpu_end_to_end_seconds_median"]
    )

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
            "chunk_steps": args.chunk_steps,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "error_reference": "strict_gpu",
            "relative_l2_error_tolerance": tolerance,
        },
        "results": rows,
        "fastmath_passed": bool(
            fast_row["gpu_finite"]
            and fast_row["gpu_max_boundary_error"] == 0.0
            and fast_row["relative_l2_ftle_error"] <= tolerance
        ),
        "fastmath_integration_speedup": fast_row[
            "integration_speedup_vs_strict"
        ],
    }

    print("\nStrict-vs-fastmath summary")
    print("mode      integration(s)  throughput(particle-steps/s)  rel_L2_error")
    for row in rows:
        print(
            f"{row['mode']:8s}  "
            f"{row['gpu_integration_seconds_median']:14.6f}  "
            f"{row['gpu_particle_steps_per_second']:28.3f}  "
            f"{row['relative_l2_ftle_error']:.3e}"
        )
    print(
        "fastmath integration speedup = "
        f"{report['fastmath_integration_speedup']:.3f}x"
    )
    print(
        "fastmath max absolute error  = "
        f"{fast_row['max_abs_ftle_error']:.3e}"
    )
    print(
        "fastmath max |det(F)-1|      = "
        f"{fast_row['gpu_max_abs_detF_minus_1']:.3e}"
    )
    print(f"fastmath correctness passed = {report['fastmath_passed']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    for row in rows:
        if not row["gpu_finite"] or row["gpu_max_boundary_error"] > 0.0:
            raise RuntimeError(f"{row['mode']} produced invalid GPU state")
    if fast_row["relative_l2_ftle_error"] > tolerance:
        raise RuntimeError(
            "fastmath failed correctness: relative L2 error "
            f"{fast_row['relative_l2_ftle_error']:.3e} > {tolerance:.3e}"
        )


if __name__ == "__main__":
    main()
