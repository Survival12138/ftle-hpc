# -*- coding: utf-8 -*-
"""Time-coefficient precomputation benchmark for Double-Gyre FTLE.

V4 removes particle-redundant evaluations of ``sin(omega * t)`` from the RK4
integration kernel.  For every RK4 stage time, the Double-Gyre coefficients

    a(t) = epsilon * sin(omega * t)
    b(t) = 1 - 2 * epsilon * sin(omega * t)

are computed once on the GPU and kept in two small device arrays.  During the
trajectory integration every particle reads the same cached coefficients.

The script performs a controlled A/B comparison:

    strict      : the V2 strict-FP64 chunk kernel
    precomputed : the V4 strict-FP64 coefficient-lookup kernel

Both variants use the same particle grid, chunk size, block size, FTLE kernel,
initial condition, and CUDA-event timing.  Measured runs alternate their order
to reduce clock/temperature/order bias.  The original strict result is the
numerical reference.

Recommended A100 run:

    python double_gyre_ftle_cuda_v4.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --threads 64 --chunk-steps 50 \
        --warmup-runs 1 --repeats 5 \
        --output results/cuda_v4_time_coefficients.json

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


@cuda.jit
def precompute_time_coefficients_kernel(
    stage_times,
    a_values,
    b_values,
    epsilon,
    omega,
):
    """Evaluate the time-only Double-Gyre coefficients once per RK4 stage."""
    i = cuda.grid(1)
    if i >= stage_times.size:
        return

    sin_omega_t = math.sin(omega * stage_times[i])
    a = epsilon * sin_omega_t
    a_values[i] = a
    b_values[i] = 1.0 - 2.0 * a


@cuda.jit(device=True, inline="always")
def _rhs_precomputed(
    x,
    y,
    f11,
    f12,
    f21,
    f22,
    amplitude,
    a,
    b,
):
    """Double-Gyre RHS using precomputed time-dependent coefficients."""
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


@cuda.jit
def rk4_chunk_precomputed_kernel(
    x,
    y,
    f11,
    f12,
    f21,
    f22,
    a_values,
    b_values,
    step_offset,
    dt,
    chunk_steps,
    amplitude,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Advance one particle using cached coefficients for each RK4 stage."""
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
        # Three coefficient entries are stored for each physical step:
        # t_n, t_n + dt/2, and t_n + dt.  k2 and k3 share the midpoint.
        coefficient_index = 3 * (step_offset + local_step)
        a0 = a_values[coefficient_index]
        b0 = b_values[coefficient_index]
        a_half = a_values[coefficient_index + 1]
        b_half = b_values[coefficient_index + 1]
        a1 = a_values[coefficient_index + 2]
        b1 = b_values[coefficient_index + 2]

        k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs_precomputed(
            x0,
            y0,
            f11_0,
            f12_0,
            f21_0,
            f22_0,
            amplitude,
            a0,
            b0,
        )

        half_dt = 0.5 * dt
        k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs_precomputed(
            x0 + half_dt * k1x,
            y0 + half_dt * k1y,
            f11_0 + half_dt * k1f11,
            f12_0 + half_dt * k1f12,
            f21_0 + half_dt * k1f21,
            f22_0 + half_dt * k1f22,
            amplitude,
            a_half,
            b_half,
        )

        k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs_precomputed(
            x0 + half_dt * k2x,
            y0 + half_dt * k2y,
            f11_0 + half_dt * k2f11,
            f12_0 + half_dt * k2f12,
            f21_0 + half_dt * k2f21,
            f22_0 + half_dt * k2f22,
            amplitude,
            a_half,
            b_half,
        )

        k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs_precomputed(
            x0 + dt * k3x,
            y0 + dt * k3y,
            f11_0 + dt * k3f11,
            f12_0 + dt * k3f12,
            f21_0 + dt * k3f21,
            f22_0 + dt * k3f22,
            amplitude,
            a1,
            b1,
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


def build_stage_times(
    total_steps: int,
    dt: float,
    chunk_steps: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
) -> np.ndarray:
    """Build t, t+dt/2, t+dt values matching the chunked baseline."""
    scalar = dtype.type
    dt_scalar = scalar(dt)
    half_dt = scalar(0.5) * dt_scalar
    stage_times = np.empty(3 * total_steps, dtype=dtype)

    for chunk_start in range(0, total_steps, chunk_steps):
        # V2 calculates this value on the host for each kernel launch, then
        # casts it to the selected dtype before passing it to the device.
        t_start = scalar(cfg.t0 + chunk_start * dt)
        current_chunk = min(chunk_steps, total_steps - chunk_start)

        for local_step in range(current_chunk):
            step = chunk_start + local_step
            t = scalar(t_start + scalar(local_step) * dt_scalar)
            base = 3 * step
            stage_times[base] = t
            stage_times[base + 1] = scalar(t + half_dt)
            stage_times[base + 2] = scalar(t + dt_scalar)

    return stage_times


def compile_warmup(dtype: np.dtype, cfg: DoubleGyreConfig, threads: int) -> None:
    """JIT-compile every kernel on tiny arrays before any measurement."""
    scalar = dtype.type
    host_state = initial_state(2, 2, dtype, cfg)
    blocks = (host_state[0].size + threads - 1) // threads

    # Compile the coefficient setup kernel.
    stage_times = np.asarray(
        [cfg.t0, cfg.t0 + 0.005, cfg.t0 + 0.01], dtype=dtype
    )
    d_stage_times = cuda.to_device(stage_times)
    d_a = cuda.device_array(stage_times.size, dtype=dtype)
    d_b = cuda.device_array(stage_times.size, dtype=dtype)
    coefficient_blocks = (stage_times.size + threads - 1) // threads
    precompute_time_coefficients_kernel[coefficient_blocks, threads](
        d_stage_times,
        d_a,
        d_b,
        scalar(cfg.epsilon),
        scalar(cfg.omega),
    )

    # Compile the original strict chunk kernel.
    strict_state = tuple(cuda.to_device(array) for array in host_state)
    rk4_chunk_kernel[blocks, threads](
        *strict_state,
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

    # Compile the precomputed-coefficient chunk kernel.
    precomputed_state = tuple(cuda.to_device(array) for array in host_state)
    rk4_chunk_precomputed_kernel[blocks, threads](
        *precomputed_state,
        d_a,
        d_b,
        0,
        scalar(0.01),
        1,
        scalar(cfg.amplitude),
        scalar(cfg.x_min),
        scalar(cfg.x_max),
        scalar(cfg.y_min),
        scalar(cfg.y_max),
    )

    d_ftle = cuda.device_array(host_state[0].size, dtype=dtype)
    ftle_kernel[blocks, threads](
        *precomputed_state[2:], scalar(0.01), d_ftle
    )
    cuda.synchronize()


def prepare_coefficients(
    total_steps: int,
    dt: float,
    chunk_steps: int,
    threads: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
) -> tuple[Any, Any, dict[str, float | int]]:
    """Create the small coefficient tables once and keep them on the GPU."""
    setup_start = perf_counter()
    stage_times = build_stage_times(total_steps, dt, chunk_steps, dtype, cfg)
    host_build_seconds = perf_counter() - setup_start

    device_start = perf_counter()
    d_stage_times = cuda.to_device(stage_times)
    d_a = cuda.device_array(stage_times.size, dtype=dtype)
    d_b = cuda.device_array(stage_times.size, dtype=dtype)
    blocks = (stage_times.size + threads - 1) // threads
    scalar = dtype.type
    precompute_time_coefficients_kernel[blocks, threads](
        d_stage_times,
        d_a,
        d_b,
        scalar(cfg.epsilon),
        scalar(cfg.omega),
    )
    cuda.synchronize()
    device_setup_seconds = perf_counter() - device_start
    total_setup_seconds = perf_counter() - setup_start

    metrics: dict[str, float | int] = {
        "coefficient_entries": int(stage_times.size),
        "coefficient_bytes": int(2 * stage_times.size * dtype.itemsize),
        "host_stage_time_build_seconds": float(host_build_seconds),
        "device_coefficient_setup_seconds": float(device_setup_seconds),
        "total_coefficient_setup_seconds": float(total_setup_seconds),
    }
    return d_a, d_b, metrics


def run_once(
    mode: str,
    host_state: tuple[np.ndarray, ...],
    d_a: Any,
    d_b: Any,
    dt: float,
    tau: float,
    chunk_steps: int,
    threads: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
    copy_state: bool,
) -> tuple[np.ndarray, dict[str, float | int], tuple[np.ndarray, ...] | None]:
    """Run one strict or precomputed full trajectory from a fresh state."""
    if mode not in ("strict", "precomputed"):
        raise ValueError(f"unsupported mode: {mode}")

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

        if mode == "strict":
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
        else:
            rk4_chunk_precomputed_kernel[blocks, threads](
                *device_state,
                d_a,
                d_b,
                completed_steps,
                scalar(dt),
                current_chunk,
                scalar(cfg.amplitude),
                scalar(cfg.x_min),
                scalar(cfg.x_max),
                scalar(cfg.y_min),
                scalar(cfg.y_max),
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
        # Diagnostic copies are intentionally outside the reported end-to-end
        # time.  A production FTLE run only needs the FTLE array.
        final_state = tuple(array.copy_to_host() for array in device_state)
        cuda.synchronize()

    metrics: dict[str, float | int] = {
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


def summarize_mode(
    mode: str,
    measured: list[dict[str, float | int]],
    ftle: np.ndarray,
    final_state: tuple[np.ndarray, ...],
    reference_ftle: np.ndarray,
    particles: int,
    total_steps: int,
    chunk_steps: int,
    cfg: DoubleGyreConfig,
) -> dict[str, Any]:
    """Reduce repetitions to medians and append correctness diagnostics."""
    integration_seconds = median(
        float(item["integration_seconds"]) for item in measured
    )
    row: dict[str, Any] = {
        "mode": mode,
        "rk4_kernel_launches": math.ceil(total_steps / chunk_steps),
        "gpu_integration_seconds_median": float(integration_seconds),
        "gpu_ftle_seconds_median": float(
            median(float(item["ftle_seconds"]) for item in measured)
        ),
        "gpu_all_kernel_seconds_median": float(
            median(float(item["all_kernel_seconds"]) for item in measured)
        ),
        "gpu_end_to_end_seconds_median": float(
            median(float(item["end_to_end_seconds"]) for item in measured)
        ),
        "gpu_h2d_seconds_median": float(
            median(float(item["h2d_seconds"]) for item in measured)
        ),
        "gpu_d2h_seconds_median": float(
            median(float(item["d2h_seconds"]) for item in measured)
        ),
        "gpu_particle_steps_per_second": float(
            particles * total_steps / integration_seconds
        ),
    }
    row.update(state_metrics(final_state, cfg))
    row.update(error_metrics(reference_ftle, ftle))
    return row


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
        default=Path("results/cuda_v4_time_coefficients.json"),
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

    print("Double-Gyre FTLE CUDA V4: time-coefficient precomputation")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, dtype={dtype.name}"
    )
    print(
        f"threads={args.threads}, chunk_steps={args.chunk_steps}, "
        f"warmup_runs={args.warmup_runs}, repeats={args.repeats}"
    )
    print(f"precomputed relative-L2 tolerance={tolerance:.3e}")

    print("Compiling CUDA kernels...")
    compile_warmup(dtype, cfg, args.threads)

    print("Preparing time-coefficient tables...")
    d_a, d_b, coefficient_metrics = prepare_coefficients(
        total_steps,
        args.dt,
        args.chunk_steps,
        args.threads,
        dtype,
        cfg,
    )
    print(
        f"coefficient_entries={coefficient_metrics['coefficient_entries']}, "
        f"bytes={coefficient_metrics['coefficient_bytes']}, "
        "one_time_setup="
        f"{coefficient_metrics['total_coefficient_setup_seconds']:.6f}s"
    )

    print("Full-size warmup...")
    for _ in range(args.warmup_runs):
        for mode in ("strict", "precomputed"):
            run_once(
                mode,
                host_state,
                d_a,
                d_b,
                args.dt,
                actual_tau,
                args.chunk_steps,
                args.threads,
                dtype,
                cfg,
                copy_state=False,
            )
    print("Full-size warmup done")

    measured: dict[str, list[dict[str, float | int]]] = {
        "strict": [],
        "precomputed": [],
    }
    last_ftle: dict[str, np.ndarray] = {}
    last_state: dict[str, tuple[np.ndarray, ...]] = {}

    print("Running alternating measured repetitions...")
    for repeat in range(args.repeats):
        # Alternate the order to reduce persistent first/second-run bias.
        order = (
            ("strict", "precomputed")
            if repeat % 2 == 0
            else ("precomputed", "strict")
        )
        print(f"repeat_pair={repeat + 1}, order={order[0]}->{order[1]}")

        for mode in order:
            ftle, metrics, final_state = run_once(
                mode,
                host_state,
                d_a,
                d_b,
                args.dt,
                actual_tau,
                args.chunk_steps,
                args.threads,
                dtype,
                cfg,
                copy_state=repeat == args.repeats - 1,
            )
            measured[mode].append(metrics)
            last_ftle[mode] = ftle
            if final_state is not None:
                last_state[mode] = final_state
            print(
                f"  {mode:11s}: "
                f"integration={float(metrics['integration_seconds']):.6f}s, "
                f"end_to_end={float(metrics['end_to_end_seconds']):.6f}s"
            )

    strict_ftle = last_ftle["strict"]
    strict_row = summarize_mode(
        "strict",
        measured["strict"],
        strict_ftle,
        last_state["strict"],
        strict_ftle,
        particles,
        total_steps,
        args.chunk_steps,
        cfg,
    )
    precomputed_row = summarize_mode(
        "precomputed",
        measured["precomputed"],
        last_ftle["precomputed"],
        last_state["precomputed"],
        strict_ftle,
        particles,
        total_steps,
        args.chunk_steps,
        cfg,
    )

    precomputed_row["integration_speedup_vs_strict"] = (
        strict_row["gpu_integration_seconds_median"]
        / precomputed_row["gpu_integration_seconds_median"]
    )
    precomputed_row["all_kernel_speedup_vs_strict"] = (
        strict_row["gpu_all_kernel_seconds_median"]
        / precomputed_row["gpu_all_kernel_seconds_median"]
    )
    precomputed_row["end_to_end_speedup_vs_strict"] = (
        strict_row["gpu_end_to_end_seconds_median"]
        / precomputed_row["gpu_end_to_end_seconds_median"]
    )

    correctness_passed = bool(
        precomputed_row["gpu_finite"]
        and precomputed_row["gpu_max_boundary_error"] == 0.0
        and precomputed_row["relative_l2_ftle_error"] <= tolerance
    )
    recommended_mode = (
        "precomputed"
        if correctness_passed
        and precomputed_row["integration_speedup_vs_strict"] > 1.0
        else "strict"
    )

    rows = [strict_row, precomputed_row]
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
            "measurement_order": "alternating_pairs",
            "error_reference": "strict_gpu",
            "relative_l2_error_tolerance": tolerance,
        },
        "coefficient_setup": coefficient_metrics,
        "results": rows,
        "precomputed_passed": correctness_passed,
        "precomputed_integration_speedup": precomputed_row[
            "integration_speedup_vs_strict"
        ],
        "recommended_mode": recommended_mode,
    }

    print("\nStrict-vs-precomputed summary")
    print("mode         integration(s)  throughput(particle-steps/s)  rel_L2_error")
    for row in rows:
        print(
            f"{row['mode']:11s}  "
            f"{row['gpu_integration_seconds_median']:14.6f}  "
            f"{row['gpu_particle_steps_per_second']:28.3f}  "
            f"{row['relative_l2_ftle_error']:.3e}"
        )
    print(
        "precomputed integration speedup = "
        f"{report['precomputed_integration_speedup']:.3f}x"
    )
    print(
        "precomputed max absolute error  = "
        f"{precomputed_row['max_abs_ftle_error']:.3e}"
    )
    print(
        "precomputed max |det(F)-1|      = "
        f"{precomputed_row['gpu_max_abs_detF_minus_1']:.3e}"
    )
    print(f"precomputed correctness passed = {correctness_passed}")
    print(f"recommended mode               = {recommended_mode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    for row in rows:
        if not row["gpu_finite"] or row["gpu_max_boundary_error"] > 0.0:
            raise RuntimeError(f"{row['mode']} produced invalid GPU state")
    if precomputed_row["relative_l2_ftle_error"] > tolerance:
        raise RuntimeError(
            "precomputed kernel failed correctness: relative L2 error "
            f"{precomputed_row['relative_l2_ftle_error']:.3e} "
            f"> {tolerance:.3e}"
        )


if __name__ == "__main__":
    main()
