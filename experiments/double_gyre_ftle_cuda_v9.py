# -*- coding: utf-8 -*-
"""Mixed-precision trigonometry benchmark for Double-Gyre FTLE on CUDA.

V9 compares the accepted V7 strict-FP64 kernel against a controlled mixed-
precision candidate.  In the candidate, particle positions, deformation-
gradient entries, time coefficients, RK4 arithmetic, and FTLE post-processing
remain FP64.  Only the two paired trigonometric evaluations in each RHS call
use ``libdevice.sincospif`` after explicit FP32 input conversion.

Recommended A100 run:

    python double_gyre_ftle_cuda_v9.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --threads 64 --chunk-steps 50 \
        --warmup-runs 1 --repeats 6 \
        --relative-l2-tolerance 1e-5 \
        --max-abs-tolerance 1e-4 \
        --det-tolerance 1e-7 \
        --output results/cuda_v9_mixed_trig.json

Keep the baseline, V1, V4, V6, and V7 scripts in the same directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda, float32
from numba.cuda import libdevice

from double_gyre_ftle_baseline import DoubleGyreConfig
from double_gyre_ftle_cuda_v1 import initial_state, state_metrics
from double_gyre_ftle_cuda_v4 import prepare_coefficients
from double_gyre_ftle_cuda_v6 import (
    compile_warmup,
    run_once,
    summarize_variant,
)
from double_gyre_ftle_cuda_v7 import rk4_chunk_sincospi_kernel


@cuda.jit(device=True, inline="always")
def _rhs_mixed_trig32(
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
    """FP64 Double-Gyre RHS with only sincospi evaluated in FP32."""
    f = a * x * x + b * x
    dfdx = 2.0 * a * x + b
    d2fdx2 = 2.0 * a

    # These explicit casts are the only precision reduction in V9.
    sin_f, cos_f = libdevice.sincospif(float32(f))
    sin_y, cos_y = libdevice.sincospif(float32(y))

    # The FP32 trig results are promoted when combined with FP64 state and
    # parameters.  Velocity, Jacobian, RK4 stages, and accumulators stay FP64.
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
def rk4_chunk_mixed_trig32_kernel(
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
    """V7 RK4 kernel with only particle-dependent trig evaluated in FP32."""
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

        k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs_mixed_trig32(
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
        k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs_mixed_trig32(
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

        k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs_mixed_trig32(
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

        k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs_mixed_trig32(
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
    parser.add_argument(
        "--dtype",
        choices=("float64",),
        default="float64",
        help="V9 keeps all state and RK4 arithmetic in float64",
    )
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--relative-l2-tolerance",
        type=float,
        default=1e-5,
        help="maximum accepted relative L2 FTLE error",
    )
    parser.add_argument(
        "--max-abs-tolerance",
        type=float,
        default=1e-4,
        help="maximum accepted pointwise absolute FTLE error",
    )
    parser.add_argument(
        "--det-tolerance",
        type=float,
        default=1e-7,
        help="maximum accepted max |det(F)-1|",
    )
    parser.add_argument(
        "--selection-min-speedup",
        type=float,
        default=1.01,
        help="minimum integration speedup required to recommend mixed trig",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cuda_v9_mixed_trig.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must both be at least 2")
    if args.dt <= 0.0 or args.tau <= 0.0:
        raise ValueError("--dt and --tau must be positive")
    if args.threads < 32 or args.threads > 1024 or args.threads % 32 != 0:
        raise ValueError("--threads must be a multiple of 32 in [32, 1024]")
    if args.chunk_steps < 1:
        raise ValueError("--chunk-steps must be positive")
    if args.warmup_runs < 1 or args.repeats < 1:
        raise ValueError("--warmup-runs and --repeats must both be positive")
    if (
        args.relative_l2_tolerance < 0.0
        or args.max_abs_tolerance < 0.0
        or args.det_tolerance < 0.0
    ):
        raise ValueError("all correctness tolerances must be non-negative")
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

    dtype = np.dtype("float64")
    cfg = DoubleGyreConfig()
    host_state = initial_state(args.nx, args.ny, dtype, cfg)
    particles = host_state[0].size
    variants: list[dict[str, Any]] = [
        {"mode": "strict_fp64_trig", "kernel": rk4_chunk_sincospi_kernel},
        {"mode": "mixed_fp32_trig", "kernel": rk4_chunk_mixed_trig32_kernel},
    ]

    print("Double-Gyre FTLE CUDA V9: controlled mixed-precision trig")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, state_dtype={dtype.name}"
    )
    print(
        f"threads={args.threads}, chunk_steps={args.chunk_steps}, "
        f"warmup_runs={args.warmup_runs}, repeats={args.repeats}"
    )
    print(
        "acceptance thresholds: "
        f"relative_L2<={args.relative_l2_tolerance:.3e}, "
        f"max_abs<={args.max_abs_tolerance:.3e}, "
        f"max_|det(F)-1|<={args.det_tolerance:.3e}"
    )

    print("Compiling CUDA kernel variants...")
    compile_warmup(variants, dtype, cfg, args.threads)
    print("Compiled resource usage")
    print("mode               actual_regs  local_bytes/thread")
    for variant in variants:
        print(
            f"{variant['mode']:18s}  "
            f"{str(variant.get('actual_registers_per_thread')):11s}  "
            f"{str(variant.get('local_memory_bytes_per_thread')):18s}"
        )

    print("Preparing FP64 time-coefficient tables...")
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
            + "->".join(variant["mode"] for variant in order)
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
                f"  {mode:18s}: "
                f"integration={float(metrics['integration_seconds']):.6f}s, "
                f"end_to_end={float(metrics['end_to_end_seconds']):.6f}s"
            )

    reference_mode = "strict_fp64_trig"
    reference_ftle = last_ftle[reference_mode]
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

    strict_row, mixed_row = rows
    for row in rows:
        row["integration_speedup_vs_strict"] = (
            strict_row["gpu_integration_seconds_median"]
            / row["gpu_integration_seconds_median"]
        )
        row["end_to_end_speedup_vs_strict"] = (
            strict_row["gpu_end_to_end_seconds_median"]
            / row["gpu_end_to_end_seconds_median"]
        )
        row["relative_l2_passed"] = bool(
            row["relative_l2_ftle_error"] <= args.relative_l2_tolerance
        )
        row["max_abs_passed"] = bool(
            row["max_abs_ftle_error"] <= args.max_abs_tolerance
        )
        row["determinant_passed"] = bool(
            row["gpu_max_abs_detF_minus_1"] <= args.det_tolerance
        )
        row["correctness_passed"] = bool(
            row["gpu_finite"]
            and row["gpu_max_boundary_error"] == 0.0
            and row["relative_l2_passed"]
            and row["max_abs_passed"]
            and row["determinant_passed"]
        )

    recommended_mode = reference_mode
    if (
        mixed_row["correctness_passed"]
        and mixed_row["integration_speedup_vs_strict"]
        >= args.selection_min_speedup
    ):
        recommended_mode = "mixed_fp32_trig"

    report = {
        "configuration": {
            "nx": args.nx,
            "ny": args.ny,
            "particles": particles,
            "steps": total_steps,
            "dt": args.dt,
            "tau": actual_tau,
            "state_dtype": dtype.name,
            "strict_trig_dtype": "float64",
            "mixed_trig_dtype": "float32",
            "threads": args.threads,
            "chunk_steps": args.chunk_steps,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "measurement_order": "alternating_pairs",
            "error_reference": reference_mode,
            "relative_l2_error_tolerance": args.relative_l2_tolerance,
            "max_absolute_error_tolerance": args.max_abs_tolerance,
            "max_abs_detF_minus_1_tolerance": args.det_tolerance,
            "selection_minimum_speedup": args.selection_min_speedup,
        },
        "coefficient_setup": coefficient_metrics,
        "results": rows,
        "mixed_trig_integration_speedup": mixed_row[
            "integration_speedup_vs_strict"
        ],
        "mixed_trig_passed": mixed_row["correctness_passed"],
        "recommended_mode": recommended_mode,
    }

    print("\nStrict-FP64-vs-mixed-trig summary")
    print(
        "mode                regs  local_B  integration(s)  speedup  "
        "rel_L2_error  max_abs_error  det_error  passed"
    )
    for row in rows:
        print(
            f"{row['mode']:18s}  "
            f"{str(row['actual_registers_per_thread']):4s}  "
            f"{str(row['local_memory_bytes_per_thread']):7s}  "
            f"{row['gpu_integration_seconds_median']:14.6f}  "
            f"{row['integration_speedup_vs_strict']:7.3f}  "
            f"{row['relative_l2_ftle_error']:.3e}  "
            f"{row['max_abs_ftle_error']:.3e}  "
            f"{row['gpu_max_abs_detF_minus_1']:.3e}  "
            f"{row['correctness_passed']}"
        )
    print(
        "mixed trig integration speedup = "
        f"{mixed_row['integration_speedup_vs_strict']:.3f}x"
    )
    print(f"mixed relative-L2 passed    = {mixed_row['relative_l2_passed']}")
    print(f"mixed max-absolute passed   = {mixed_row['max_abs_passed']}")
    print(f"mixed determinant passed    = {mixed_row['determinant_passed']}")
    print(f"mixed correctness passed    = {mixed_row['correctness_passed']}")
    print(f"recommended mode            = {recommended_mode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    if not strict_row["correctness_passed"]:
        raise RuntimeError("strict FP64 reference failed invariant checks")


if __name__ == "__main__":
    main()
