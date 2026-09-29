# -*- coding: utf-8 -*-
"""Low-live-range RK4 benchmark for Double-Gyre FTLE on CUDA.

V5 showed that forcing the V4 kernel below 96 registers spills values to local
memory and slows the kernel.  V6 instead changes the RK4 expression so that
the compiler can naturally end each stage derivative's lifetime earlier.

The V4 kernel keeps k1, k2, k3, and k4 until the final weighted update.  The V6
kernel accumulates the RK4 weighted sum after every stage and reuses one set of
derivative variable names.  It does not set ``max_registers`` and preserves
the classical RK4 stages and weights.

Controlled variants:

    v4_original : original V4 precomputed-coefficient kernel
    accumulator : V6 stage-by-stage RK4 accumulation

Recommended A100 run:

    python double_gyre_ftle_cuda_v6.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --threads 64 --chunk-steps 50 \
        --warmup-runs 1 --repeats 6 \
        --output results/cuda_v6_accumulator.json

Keep the baseline, V1, V2, and V4 scripts in the same directory.
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
from double_gyre_ftle_cuda_v4 import (
    _rhs_precomputed,
    precompute_time_coefficients_kernel,
    prepare_coefficients,
    rk4_chunk_precomputed_kernel,
)


@cuda.jit
def rk4_chunk_accumulator_kernel(
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
    """V4 integration with stage derivatives accumulated immediately."""
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
        coefficient_index = 3 * (step_offset + local_step)
        a0 = a_values[coefficient_index]
        b0 = b_values[coefficient_index]
        a_half = a_values[coefficient_index + 1]
        b_half = b_values[coefficient_index + 1]
        a1 = a_values[coefficient_index + 2]
        b1 = b_values[coefficient_index + 2]

        # k1: initialize the six weighted-sum accumulators.
        kx, ky, kf11, kf12, kf21, kf22 = _rhs_precomputed(
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
        sum_x = kx
        sum_y = ky
        sum_f11 = kf11
        sum_f12 = kf12
        sum_f21 = kf21
        sum_f22 = kf22

        # k2: the k1 values are needed only to form this stage.  Reusing the
        # derivative names lets their previous SSA values die after the call.
        half_dt = 0.5 * dt
        kx, ky, kf11, kf12, kf21, kf22 = _rhs_precomputed(
            x0 + half_dt * kx,
            y0 + half_dt * ky,
            f11_0 + half_dt * kf11,
            f12_0 + half_dt * kf12,
            f21_0 + half_dt * kf21,
            f22_0 + half_dt * kf22,
            amplitude,
            a_half,
            b_half,
        )
        sum_x = sum_x + 2.0 * kx
        sum_y = sum_y + 2.0 * ky
        sum_f11 = sum_f11 + 2.0 * kf11
        sum_f12 = sum_f12 + 2.0 * kf12
        sum_f21 = sum_f21 + 2.0 * kf21
        sum_f22 = sum_f22 + 2.0 * kf22

        # k3: k2 dies after forming this stage and updating the accumulators.
        kx, ky, kf11, kf12, kf21, kf22 = _rhs_precomputed(
            x0 + half_dt * kx,
            y0 + half_dt * ky,
            f11_0 + half_dt * kf11,
            f12_0 + half_dt * kf12,
            f21_0 + half_dt * kf21,
            f22_0 + half_dt * kf22,
            amplitude,
            a_half,
            b_half,
        )
        sum_x = sum_x + 2.0 * kx
        sum_y = sum_y + 2.0 * ky
        sum_f11 = sum_f11 + 2.0 * kf11
        sum_f12 = sum_f12 + 2.0 * kf12
        sum_f21 = sum_f21 + 2.0 * kf21
        sum_f22 = sum_f22 + 2.0 * kf22

        # k4: k3 dies after this call; then k4 is folded into the sums.
        kx, ky, kf11, kf12, kf21, kf22 = _rhs_precomputed(
            x0 + dt * kx,
            y0 + dt * ky,
            f11_0 + dt * kf11,
            f12_0 + dt * kf12,
            f21_0 + dt * kf21,
            f22_0 + dt * kf22,
            amplitude,
            a1,
            b1,
        )
        sum_x = sum_x + kx
        sum_y = sum_y + ky
        sum_f11 = sum_f11 + kf11
        sum_f12 = sum_f12 + kf12
        sum_f21 = sum_f21 + kf21
        sum_f22 = sum_f22 + kf22

        sixth_dt = dt / 6.0
        x0 = x0 + sixth_dt * sum_x
        y0 = y0 + sixth_dt * sum_y
        f11_0 = f11_0 + sixth_dt * sum_f11
        f12_0 = f12_0 + sixth_dt * sum_f12
        f21_0 = f21_0 + sixth_dt * sum_f21
        f22_0 = f22_0 + sixth_dt * sum_f22

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


def _resource_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if len(value) != 1:
            return None
        value = next(iter(value.values()))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def query_kernel_resources(kernel: Any) -> dict[str, int | None]:
    """Query resources with compatibility for numba-cuda 0.30.x."""
    compiled_overloads = list(kernel.overloads.values())
    compiled = compiled_overloads[-1] if compiled_overloads else None

    def query(method_name: str, attribute_name: str) -> int | None:
        method = getattr(kernel, method_name, None)
        if method is not None:
            try:
                normalized = _resource_value(method())
                if normalized is not None:
                    return normalized
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                pass
        if compiled is not None:
            return _resource_value(getattr(compiled, attribute_name, None))
        return None

    return {
        "actual_registers_per_thread": query(
            "get_regs_per_thread", "regs_per_thread"
        ),
        "local_memory_bytes_per_thread": query(
            "get_local_mem_per_thread", "local_mem_per_thread"
        ),
    }


def compile_warmup(
    variants: list[dict[str, Any]],
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
    threads: int,
) -> None:
    """Compile both integration kernels and support kernels on tiny arrays."""
    scalar = dtype.type
    host_state = initial_state(2, 2, dtype, cfg)
    blocks = (host_state[0].size + threads - 1) // threads

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

    last_state = None
    for variant in variants:
        device_state = tuple(cuda.to_device(array) for array in host_state)
        variant["kernel"][blocks, threads](
            *device_state,
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
        last_state = device_state

    d_ftle = cuda.device_array(host_state[0].size, dtype=dtype)
    ftle_kernel[blocks, threads](*last_state[2:], scalar(0.01), d_ftle)
    cuda.synchronize()

    for variant in variants:
        variant.update(query_kernel_resources(variant["kernel"]))


def run_once(
    kernel: Any,
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
        kernel[blocks, threads](
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


def summarize_variant(
    variant: dict[str, Any],
    measured: list[dict[str, float | int]],
    ftle: np.ndarray,
    reference_ftle: np.ndarray,
    final_state_metrics: dict[str, float | bool],
    particles: int,
    total_steps: int,
    chunk_steps: int,
) -> dict[str, Any]:
    integration_seconds = median(
        float(item["integration_seconds"]) for item in measured
    )
    row: dict[str, Any] = {
        "mode": variant["mode"],
        "actual_registers_per_thread": variant.get(
            "actual_registers_per_thread"
        ),
        "local_memory_bytes_per_thread": variant.get(
            "local_memory_bytes_per_thread"
        ),
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
    row.update(final_state_metrics)
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
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--error-tolerance",
        type=float,
        default=None,
        help="relative L2 FTLE tolerance; default: 1e-8 (float64), 1e-4 (float32)",
    )
    parser.add_argument(
        "--selection-min-speedup",
        type=float,
        default=1.01,
        help="minimum speedup required to recommend the accumulator kernel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cuda_v6_accumulator.json"),
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
    if args.selection_min_speedup < 1.0:
        raise ValueError("--selection-min-speedup must be at least 1.0")

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
    variants = [
        {"mode": "v4_original", "kernel": rk4_chunk_precomputed_kernel},
        {"mode": "accumulator", "kernel": rk4_chunk_accumulator_kernel},
    ]

    print("Double-Gyre FTLE CUDA V6: RK4 live-range refactor")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, dtype={dtype.name}"
    )
    print(
        f"threads={args.threads}, chunk_steps={args.chunk_steps}, "
        f"warmup_runs={args.warmup_runs}, repeats={args.repeats}"
    )
    print(f"relative-L2 tolerance={tolerance:.3e}")

    print("Compiling CUDA kernel variants...")
    compile_warmup(variants, dtype, cfg, args.threads)
    print("Compiled resource usage")
    print("mode          actual_regs  local_bytes/thread")
    for variant in variants:
        print(
            f"{variant['mode']:12s}  "
            f"{str(variant.get('actual_registers_per_thread')):11s}  "
            f"{str(variant.get('local_memory_bytes_per_thread')):18s}"
        )

    print("Preparing V4 time-coefficient tables...")
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
        for variant in variants:
            run_once(
                variant["kernel"],
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
        variant["mode"]: [] for variant in variants
    }
    last_ftle: dict[str, np.ndarray] = {}
    last_state_metrics: dict[str, dict[str, float | bool]] = {}

    print("Running alternating measured repetitions...")
    for repeat in range(args.repeats):
        order = variants if repeat % 2 == 0 else list(reversed(variants))
        print(
            f"repeat_pair={repeat + 1}, order="
            + "->".join(item["mode"] for item in order)
        )
        for variant in order:
            mode = variant["mode"]
            ftle, metrics, final_state = run_once(
                variant["kernel"],
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
                last_state_metrics[mode] = state_metrics(final_state, cfg)
            print(
                f"  {mode:12s}: "
                f"integration={float(metrics['integration_seconds']):.6f}s, "
                f"end_to_end={float(metrics['end_to_end_seconds']):.6f}s"
            )

    reference_ftle = last_ftle["v4_original"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        mode = variant["mode"]
        rows.append(
            summarize_variant(
                variant,
                measured[mode],
                last_ftle[mode],
                reference_ftle,
                last_state_metrics[mode],
                particles,
                total_steps,
                args.chunk_steps,
            )
        )

    baseline, accumulator = rows
    for row in rows:
        row["integration_speedup_vs_v4"] = (
            baseline["gpu_integration_seconds_median"]
            / row["gpu_integration_seconds_median"]
        )
        row["end_to_end_speedup_vs_v4"] = (
            baseline["gpu_end_to_end_seconds_median"]
            / row["gpu_end_to_end_seconds_median"]
        )
        row["correctness_passed"] = bool(
            row["gpu_finite"]
            and row["gpu_max_boundary_error"] == 0.0
            and row["relative_l2_ftle_error"] <= tolerance
        )

    recommended_mode = "v4_original"
    if (
        accumulator["correctness_passed"]
        and accumulator["integration_speedup_vs_v4"]
        >= args.selection_min_speedup
    ):
        recommended_mode = "accumulator"

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
            "error_reference": "v4_original_gpu",
            "relative_l2_error_tolerance": tolerance,
            "selection_minimum_speedup": args.selection_min_speedup,
        },
        "coefficient_setup": coefficient_metrics,
        "results": rows,
        "accumulator_speedup": accumulator["integration_speedup_vs_v4"],
        "accumulator_passed": accumulator["correctness_passed"],
        "recommended_mode": recommended_mode,
    }

    print("\nV4-vs-accumulator summary")
    print(
        "mode          regs  local_B  integration(s)  speedup  rel_L2_error"
    )
    for row in rows:
        print(
            f"{row['mode']:12s}  "
            f"{str(row['actual_registers_per_thread']):4s}  "
            f"{str(row['local_memory_bytes_per_thread']):7s}  "
            f"{row['gpu_integration_seconds_median']:14.6f}  "
            f"{row['integration_speedup_vs_v4']:7.3f}  "
            f"{row['relative_l2_ftle_error']:.3e}"
        )
    print(
        "accumulator max absolute error = "
        f"{accumulator['max_abs_ftle_error']:.3e}"
    )
    print(
        "accumulator max |det(F)-1|     = "
        f"{accumulator['gpu_max_abs_detF_minus_1']:.3e}"
    )
    print(f"accumulator correctness passed = {accumulator['correctness_passed']}")
    print(f"recommended mode               = {recommended_mode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    for row in rows:
        if not row["correctness_passed"]:
            raise RuntimeError(
                f"{row['mode']} failed correctness: relative L2 error "
                f"{row['relative_l2_ftle_error']:.3e}"
            )


if __name__ == "__main__":
    main()
