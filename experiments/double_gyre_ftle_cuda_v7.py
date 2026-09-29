# -*- coding: utf-8 -*-
"""Strict-FP64 sincospi benchmark for Double-Gyre FTLE on CUDA.

V7 changes only the particle-dependent trigonometric evaluation in the V4
precomputed-coefficient kernel.  The baseline evaluates ``sin(pi*x)`` and
``cos(pi*x)`` separately; the candidate uses the paired libdevice function
``sincospi(x)``.  RK4 storage, launch geometry, coefficient tables, and FTLE
post-processing remain identical to V4.

Recommended A100 run:

    python double_gyre_ftle_cuda_v7.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --threads 64 --chunk-steps 50 \
        --warmup-runs 1 --repeats 6 \
        --output results/cuda_v7_sincospi.json

Keep the baseline, V1, V2, V4, and V6 scripts in the same directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda
from numba.cuda import libdevice

from double_gyre_ftle_baseline import DoubleGyreConfig
from double_gyre_ftle_cuda_v1 import initial_state, state_metrics
from double_gyre_ftle_cuda_v4 import (
    prepare_coefficients,
    rk4_chunk_precomputed_kernel,
)
from double_gyre_ftle_cuda_v6 import (
    compile_warmup,
    run_once,
    summarize_variant,
)


@cuda.jit(device=True, inline="always")
def _rhs_sincospi(
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
    """Double-Gyre RHS using paired strict-FP64 sin(pi*x)/cos(pi*x)."""
    f = a * x * x + b * x
    dfdx = 2.0 * a * x + b
    d2fdx2 = 2.0 * a

    # libdevice.sincospi(z) returns (sin(pi*z), cos(pi*z)).
    sin_f, cos_f = libdevice.sincospi(f)
    sin_y, cos_y = libdevice.sincospi(y)

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
def rk4_chunk_sincospi_kernel(
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
    """V4 RK4 kernel with only the RHS trigonometric implementation changed."""
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

        k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs_sincospi(
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
        k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs_sincospi(
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

        k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs_sincospi(
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

        k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs_sincospi(
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
        help="minimum speedup required to recommend sincospi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cuda_v7_sincospi.json"),
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
        {"mode": "sincospi", "kernel": rk4_chunk_sincospi_kernel},
    ]

    print("Double-Gyre FTLE CUDA V7: strict-FP64 sincospi")
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

    baseline, sincospi_row = rows
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
        sincospi_row["correctness_passed"]
        and sincospi_row["integration_speedup_vs_v4"]
        >= args.selection_min_speedup
    ):
        recommended_mode = "sincospi"

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
        "sincospi_speedup": sincospi_row["integration_speedup_vs_v4"],
        "sincospi_passed": sincospi_row["correctness_passed"],
        "recommended_mode": recommended_mode,
    }

    print("\nV4-vs-sincospi summary")
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
        "sincospi max absolute error = "
        f"{sincospi_row['max_abs_ftle_error']:.3e}"
    )
    print(
        "sincospi max |det(F)-1|     = "
        f"{sincospi_row['gpu_max_abs_detF_minus_1']:.3e}"
    )
    print(f"sincospi correctness passed = {sincospi_row['correctness_passed']}")
    print(f"recommended mode            = {recommended_mode}")

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
