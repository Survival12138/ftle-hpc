# -*- coding: utf-8 -*-
"""Thread-block-size scan for the V7 Double-Gyre FTLE CUDA kernel.

V8 keeps the accepted V7 numerical path unchanged: strict floating-point
arithmetic, precomputed time coefficients, paired libdevice ``sincospi``, and
multi-step RK4 fusion.  Only the number of CUDA threads per block changes.

Recommended A100 run:

    python double_gyre_ftle_cuda_v8.py \
        --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --dtype float64 --chunk-steps 50 \
        --thread-list 32,64,128,256,512 \
        --reference-threads 64 --warmup-runs 1 --repeats 5 \
        --output results/cuda_v8_block_scan.json

Keep the baseline, V1, V4, V6, and V7 scripts in the same directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numba import cuda

from double_gyre_ftle_baseline import DoubleGyreConfig
from double_gyre_ftle_cuda_v1 import initial_state, state_metrics
from double_gyre_ftle_cuda_v4 import prepare_coefficients
from double_gyre_ftle_cuda_v6 import (
    compile_warmup,
    run_once,
    summarize_variant,
)
from double_gyre_ftle_cuda_v7 import rk4_chunk_sincospi_kernel


def parse_thread_list(text: str) -> list[int]:
    """Parse a comma-separated list while preserving order and uniqueness."""
    values: list[int] = []
    for field in text.split(","):
        field = field.strip()
        if not field:
            continue
        value = int(field)
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("thread list must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=2001)
    parser.add_argument("--ny", type=int, default=1001)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument(
        "--thread-list",
        type=parse_thread_list,
        default=parse_thread_list("32,64,128,256,512"),
        help="comma-separated CUDA block sizes",
    )
    parser.add_argument(
        "--reference-threads",
        type=int,
        default=64,
        help="block size used as the speed and error reference",
    )
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
        help="minimum speedup required to replace the reference block size",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cuda_v8_block_scan.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must both be at least 2")
    if args.dt <= 0.0 or args.tau <= 0.0:
        raise ValueError("--dt and --tau must be positive")
    if args.chunk_steps < 1:
        raise ValueError("--chunk-steps must be positive")
    if args.warmup_runs < 1 or args.repeats < 1:
        raise ValueError("--warmup-runs and --repeats must both be positive")
    if args.error_tolerance is not None and args.error_tolerance < 0.0:
        raise ValueError("--error-tolerance must be non-negative")
    if args.selection_min_speedup < 1.0:
        raise ValueError("--selection-min-speedup must be at least 1.0")
    for threads in args.thread_list:
        if threads < 32 or threads > 1024 or threads % 32 != 0:
            raise ValueError(
                "every --thread-list value must be a multiple of 32 in [32, 1024]"
            )
    if args.reference_threads not in args.thread_list:
        raise ValueError("--reference-threads must appear in --thread-list")

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

    # The compiled kernel resource counts are invariant with respect to the
    # launch block size.  Compile once using the reference configuration.
    kernel_variant: dict[str, Any] = {
        "mode": "sincospi",
        "kernel": rk4_chunk_sincospi_kernel,
    }

    print("Double-Gyre FTLE CUDA V8: V7 thread-block scan")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, dtype={dtype.name}"
    )
    print(
        f"thread_list={args.thread_list}, reference_threads={args.reference_threads}, "
        f"chunk_steps={args.chunk_steps}, warmup_runs={args.warmup_runs}, "
        f"repeats={args.repeats}"
    )
    print(f"relative-L2 tolerance={tolerance:.3e}")

    print("Compiling V7 sincospi kernel...")
    compile_warmup(
        [kernel_variant], dtype, cfg, args.reference_threads
    )
    print(
        "compiled resources: "
        f"registers/thread={kernel_variant.get('actual_registers_per_thread')}, "
        "local_bytes/thread="
        f"{kernel_variant.get('local_memory_bytes_per_thread')}"
    )

    print("Preparing time-coefficient tables...")
    d_a, d_b, coefficient_metrics = prepare_coefficients(
        total_steps,
        args.dt,
        args.chunk_steps,
        args.reference_threads,
        dtype,
        cfg,
    )
    print(
        f"coefficient_entries={coefficient_metrics['coefficient_entries']}, "
        f"bytes={coefficient_metrics['coefficient_bytes']}, "
        "one_time_setup="
        f"{coefficient_metrics['total_coefficient_setup_seconds']:.6f}s"
    )

    print("Full-size warmup for every block size...")
    for _ in range(args.warmup_runs):
        for threads in args.thread_list:
            run_once(
                rk4_chunk_sincospi_kernel,
                host_state,
                d_a,
                d_b,
                args.dt,
                actual_tau,
                args.chunk_steps,
                threads,
                dtype,
                cfg,
                copy_state=False,
            )
    print("Full-size warmup done")

    measured: dict[int, list[dict[str, float | int]]] = {
        threads: [] for threads in args.thread_list
    }
    last_ftle: dict[int, np.ndarray] = {}
    last_state_metrics: dict[int, dict[str, float | bool]] = {}

    print("Running cyclically rotated measured repetitions...")
    number_of_variants = len(args.thread_list)
    for repeat in range(args.repeats):
        shift = repeat % number_of_variants
        order = args.thread_list[shift:] + args.thread_list[:shift]
        print(
            f"repeat={repeat + 1}, order="
            + "->".join(str(threads) for threads in order)
        )
        for threads in order:
            ftle, metrics, final_state = run_once(
                rk4_chunk_sincospi_kernel,
                host_state,
                d_a,
                d_b,
                args.dt,
                actual_tau,
                args.chunk_steps,
                threads,
                dtype,
                cfg,
                copy_state=repeat == args.repeats - 1,
            )
            measured[threads].append(metrics)
            last_ftle[threads] = ftle
            if final_state is not None:
                last_state_metrics[threads] = state_metrics(final_state, cfg)
            print(
                f"  threads={threads:4d}: "
                f"integration={float(metrics['integration_seconds']):.6f}s, "
                f"end_to_end={float(metrics['end_to_end_seconds']):.6f}s"
            )

    reference_ftle = last_ftle[args.reference_threads]
    rows: list[dict[str, Any]] = []
    for threads in args.thread_list:
        variant = {
            "mode": f"threads_{threads}",
            "actual_registers_per_thread": kernel_variant.get(
                "actual_registers_per_thread"
            ),
            "local_memory_bytes_per_thread": kernel_variant.get(
                "local_memory_bytes_per_thread"
            ),
        }
        row = summarize_variant(
            variant,
            measured[threads],
            last_ftle[threads],
            reference_ftle,
            last_state_metrics[threads],
            particles,
            total_steps,
            args.chunk_steps,
        )
        row["threads"] = threads
        rows.append(row)

    reference_row = next(
        row for row in rows if row["threads"] == args.reference_threads
    )
    for row in rows:
        row["integration_speedup_vs_reference"] = (
            reference_row["gpu_integration_seconds_median"]
            / row["gpu_integration_seconds_median"]
        )
        row["end_to_end_speedup_vs_reference"] = (
            reference_row["gpu_end_to_end_seconds_median"]
            / row["gpu_end_to_end_seconds_median"]
        )
        row["correctness_passed"] = bool(
            row["gpu_finite"]
            and row["gpu_max_boundary_error"] == 0.0
            and row["relative_l2_ftle_error"] <= tolerance
        )

    valid_rows = [row for row in rows if row["correctness_passed"]]
    if not valid_rows:
        raise RuntimeError("all thread-block configurations failed correctness")
    fastest_row = min(
        valid_rows, key=lambda row: row["gpu_integration_seconds_median"]
    )
    recommended_threads = args.reference_threads
    if (
        fastest_row["integration_speedup_vs_reference"]
        >= args.selection_min_speedup
    ):
        recommended_threads = int(fastest_row["threads"])

    report = {
        "configuration": {
            "nx": args.nx,
            "ny": args.ny,
            "particles": particles,
            "steps": total_steps,
            "dt": args.dt,
            "tau": actual_tau,
            "dtype": dtype.name,
            "thread_list": args.thread_list,
            "reference_threads": args.reference_threads,
            "chunk_steps": args.chunk_steps,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "measurement_order": "cyclic_rotation",
            "error_reference": f"gpu_threads_{args.reference_threads}",
            "relative_l2_error_tolerance": tolerance,
            "selection_minimum_speedup": args.selection_min_speedup,
        },
        "coefficient_setup": coefficient_metrics,
        "results": rows,
        "fastest_observed_threads": int(fastest_row["threads"]),
        "fastest_observed_integration_speedup_vs_reference": float(
            fastest_row["integration_speedup_vs_reference"]
        ),
        "recommended_threads": recommended_threads,
    }

    print("\nV7 thread-block scan summary")
    print(
        "threads  regs  local_B  integration(s)  throughput(particle-steps/s)  "
        "speedup  rel_L2_error  passed"
    )
    for row in rows:
        print(
            f"{row['threads']:7d}  "
            f"{str(row['actual_registers_per_thread']):4s}  "
            f"{str(row['local_memory_bytes_per_thread']):7s}  "
            f"{row['gpu_integration_seconds_median']:14.6f}  "
            f"{row['gpu_particle_steps_per_second']:28.3f}  "
            f"{row['integration_speedup_vs_reference']:7.3f}  "
            f"{row['relative_l2_ftle_error']:.3e}  "
            f"{row['correctness_passed']}"
        )
    print(f"fastest observed threads = {fastest_row['threads']}")
    print(
        "fastest speedup vs reference = "
        f"{fastest_row['integration_speedup_vs_reference']:.3f}x"
    )
    print(f"recommended threads      = {recommended_threads}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    failed = [row for row in rows if not row["correctness_passed"]]
    if failed:
        details = ", ".join(
            f"threads={row['threads']} (rel_L2={row['relative_l2_ftle_error']:.3e})"
            for row in failed
        )
        raise RuntimeError(f"thread-block correctness failure: {details}")


if __name__ == "__main__":
    main()
