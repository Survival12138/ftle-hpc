# -*- coding: utf-8 -*-
"""Register-cap scan for the precomputed Double-Gyre FTLE CUDA kernel.

V4 reduced redundant time-dependent trigonometric work, but Nsight Compute
still reported 96 registers per thread and register-limited occupancy.  V5
compiles the same strict-FP64 RK4 kernel body with several ``max_registers``
requests and measures the tradeoff between occupancy and register spilling.

Only one optimization dimension changes between variants:

    unlimited, maxr88, maxr80, maxr72, maxr64

All variants share the same V4 coefficient tables, block size, chunk size,
initial condition, FTLE post-processing, and CUDA-event timing.  Measurement
order is cyclically rotated so that, with five repetitions, every variant
occupies every run position once.

Recommended A100 run:

    python double_gyre_ftle_cuda_v5.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --threads 64 --chunk-steps 50 \
        --register-list 88,80,72,64 \
        --warmup-runs 1 --repeats 5 \
        --output results/cuda_v5_register_scan.json

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
)


def _rk4_chunk_precomputed_body(
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
    """V4 kernel body, decorated with different register limits in V5."""
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


def parse_register_list(text: str) -> list[int]:
    limits: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > 255:
            raise ValueError("all register limits must be in [1, 255]")
        if value not in limits:
            limits.append(value)
    if not limits:
        raise ValueError("--register-list must contain at least one value")
    return limits


def build_kernel_variants(register_limits: list[int]) -> list[dict[str, Any]]:
    """Compile dispatchers lazily; actual device compilation happens at launch."""
    variants: list[dict[str, Any]] = [
        {
            "mode": "unlimited",
            "requested_max_registers": None,
            "kernel": cuda.jit(_rk4_chunk_precomputed_body),
        }
    ]
    for limit in register_limits:
        variants.append(
            {
                "mode": f"maxr{limit}",
                "requested_max_registers": limit,
                "kernel": cuda.jit(max_registers=limit)(
                    _rk4_chunk_precomputed_body
                ),
            }
        )
    return variants


def _resource_value(value: Any) -> int | None:
    """Normalize scalar or one-entry dispatcher resource-query results."""
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
    """Query ptxas-selected registers and local-memory bytes per thread."""
    compiled_overloads = list(kernel.overloads.values())
    compiled = compiled_overloads[-1] if compiled_overloads else None

    def query(method_name: str, attribute_name: str) -> int | None:
        method = getattr(kernel, method_name, None)
        if method is not None:
            try:
                # With no explicit signature, numba-cuda returns either a
                # scalar (specialized dispatcher) or a one-entry mapping.
                result = method()
                normalized = _resource_value(result)
                if normalized is not None:
                    return normalized
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                pass

        # Compatibility fallback for numba-cuda versions whose public query
        # requires a numba Signature rather than an overload-key tuple.
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
    """Compile coefficient, RK4 variants, and FTLE kernels on tiny arrays."""
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
    """Run one complete trajectory for one register-cap variant."""
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
        # Diagnostic state copies are outside the end-to-end timing.
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
        "requested_max_registers": variant["requested_max_registers"],
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
    parser.add_argument("--register-list", type=str, default="88,80,72,64")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
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
        help="minimum speedup required to recommend a capped variant",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cuda_v5_register_scan.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    register_limits = parse_register_list(args.register_list)

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
    variants = build_kernel_variants(register_limits)
    modes = [variant["mode"] for variant in variants]

    print("Double-Gyre FTLE CUDA V5: max_registers scan")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, dtype={dtype.name}"
    )
    print(
        f"threads={args.threads}, chunk_steps={args.chunk_steps}, "
        f"modes={modes}, warmup_runs={args.warmup_runs}, "
        f"repeats={args.repeats}"
    )
    print(f"relative-L2 tolerance={tolerance:.3e}")

    print("Compiling CUDA kernel variants...")
    compile_warmup(variants, dtype, cfg, args.threads)
    print("Compiled resource usage")
    print("mode         requested  actual_regs  local_bytes/thread")
    for variant in variants:
        requested = variant["requested_max_registers"]
        requested_text = "none" if requested is None else str(requested)
        actual = variant.get("actual_registers_per_thread")
        local_bytes = variant.get("local_memory_bytes_per_thread")
        print(
            f"{variant['mode']:11s}  {requested_text:9s}  "
            f"{str(actual):11s}  {str(local_bytes):18s}"
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
        mode: [] for mode in modes
    }
    last_ftle: dict[str, np.ndarray] = {}
    last_state_metrics: dict[str, dict[str, float | bool]] = {}

    print("Running cyclically rotated measured repetitions...")
    for repeat in range(args.repeats):
        shift = repeat % len(variants)
        order = variants[shift:] + variants[:shift]
        print(
            f"repeat_round={repeat + 1}, order="
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
                f"  {mode:11s}: "
                f"integration={float(metrics['integration_seconds']):.6f}s, "
                f"end_to_end={float(metrics['end_to_end_seconds']):.6f}s"
            )

    reference_ftle = last_ftle["unlimited"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        mode = variant["mode"]
        row = summarize_variant(
            variant,
            measured[mode],
            last_ftle[mode],
            reference_ftle,
            last_state_metrics[mode],
            particles,
            total_steps,
            args.chunk_steps,
        )
        rows.append(row)

    baseline = rows[0]
    baseline_integration = baseline["gpu_integration_seconds_median"]
    baseline_e2e = baseline["gpu_end_to_end_seconds_median"]
    baseline_local = baseline["local_memory_bytes_per_thread"]

    for row in rows:
        row["integration_speedup_vs_unlimited"] = (
            baseline_integration / row["gpu_integration_seconds_median"]
        )
        row["end_to_end_speedup_vs_unlimited"] = (
            baseline_e2e / row["gpu_end_to_end_seconds_median"]
        )
        row["correctness_passed"] = bool(
            row["gpu_finite"]
            and row["gpu_max_boundary_error"] == 0.0
            and row["relative_l2_ftle_error"] <= tolerance
        )
        local_bytes = row["local_memory_bytes_per_thread"]
        row["local_memory_increased_vs_unlimited"] = bool(
            local_bytes is not None
            and baseline_local is not None
            and local_bytes > baseline_local
        )

    valid_rows = [row for row in rows if row["correctness_passed"]]
    fastest = min(
        valid_rows,
        key=lambda row: row["gpu_integration_seconds_median"],
    )
    recommended = baseline
    if (
        fastest["mode"] != "unlimited"
        and fastest["integration_speedup_vs_unlimited"]
        >= args.selection_min_speedup
    ):
        recommended = fastest

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
            "register_limits": register_limits,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "measurement_order": "cyclic_rotation",
            "error_reference": "unlimited_gpu",
            "relative_l2_error_tolerance": tolerance,
            "selection_minimum_speedup": args.selection_min_speedup,
        },
        "coefficient_setup": coefficient_metrics,
        "results": rows,
        "fastest_passing_mode": fastest["mode"],
        "fastest_passing_speedup": fastest[
            "integration_speedup_vs_unlimited"
        ],
        "recommended_mode": recommended["mode"],
    }

    print("\nRegister-cap summary")
    print(
        "mode         req  actual  local_B  integration(s)  speedup  "
        "rel_L2_error"
    )
    for row in rows:
        requested = row["requested_max_registers"]
        requested_text = "none" if requested is None else str(requested)
        print(
            f"{row['mode']:11s}  "
            f"{requested_text:4s}  "
            f"{str(row['actual_registers_per_thread']):6s}  "
            f"{str(row['local_memory_bytes_per_thread']):7s}  "
            f"{row['gpu_integration_seconds_median']:14.6f}  "
            f"{row['integration_speedup_vs_unlimited']:7.3f}  "
            f"{row['relative_l2_ftle_error']:.3e}"
        )
    print(f"fastest passing mode = {fastest['mode']}")
    print(
        "fastest speedup      = "
        f"{fastest['integration_speedup_vs_unlimited']:.3f}x"
    )
    print(f"recommended mode     = {recommended['mode']}")

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
