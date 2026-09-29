# -*- coding: utf-8 -*-
"""Multi-GPU process-per-device driver for optimized Double-Gyre FTLE.

The global Cartesian grid is decomposed into contiguous y-row slabs.  Every
worker owns complete rows, selects exactly one logical CUDA device, and reuses
the validated kernels from ``double_gyre_ftle_cuda_optimized.py``.  Double-Gyre
particles are independent, so this first multi-GPU version needs no halo
exchange.

Examples
--------
Single-device smoke test on the current A100 server:

    python double_gyre_ftle_multi_gpu.py \
        --devices 0 --mode mixed --verify \
        --nx 241 --ny 121 --dt 0.01 --tau 1 \
        --warmup-runs 1 --repeats 3 --no-save

Four-GPU run (device IDs are logical CUDA-visible IDs):

    python double_gyre_ftle_multi_gpu.py \
        --devices 0,1,2,3 --mode mixed --verify \
        --nx 4001 --ny 2001 --dt 0.01 --tau 10 \
        --threads 64 --chunk-steps 50 \
        --warmup-runs 2 --repeats 7 \
        --output results/multi_gpu_ftle.npz \
        --metrics results/multi_gpu_metrics.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import tempfile
import traceback
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np


def parse_devices(text: str) -> list[int]:
    """Parse unique logical CUDA device IDs while preserving order."""
    devices: list[int] = []
    for field in text.split(","):
        field = field.strip()
        if not field:
            continue
        try:
            device = int(field)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid CUDA device {field!r}"
            ) from exc
        if device < 0:
            raise argparse.ArgumentTypeError("CUDA device IDs must be >= 0")
        if device in devices:
            raise argparse.ArgumentTypeError(
                f"duplicate CUDA device ID {device}"
            )
        devices.append(device)
    if not devices:
        raise argparse.ArgumentTypeError("--devices must not be empty")
    return devices


def partition_rows(ny: int, partitions: int) -> list[tuple[int, int]]:
    """Return balanced, contiguous half-open y-row ranges."""
    if ny < 1:
        raise ValueError("ny must be positive")
    if partitions < 1:
        raise ValueError("partitions must be positive")
    if partitions > ny:
        raise ValueError("number of devices cannot exceed ny")

    quotient, remainder = divmod(ny, partitions)
    ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(partitions):
        row_count = quotient + (1 if rank < remainder else 0)
        stop = start + row_count
        ranges.append((start, stop))
        start = stop
    return ranges


def initial_state_rows(
    nx: int,
    ny: int,
    row_start: int,
    row_stop: int,
    dtype: np.dtype,
    cfg: Any,
) -> tuple[np.ndarray, ...]:
    """Construct one row slab with the exact global-grid coordinates."""
    if not (0 <= row_start < row_stop <= ny):
        raise ValueError("invalid row range")
    x_grid = np.linspace(cfg.x_min, cfg.x_max, nx, dtype=dtype)
    y_global = np.linspace(cfg.y_min, cfg.y_max, ny, dtype=dtype)
    x0, y0 = np.meshgrid(x_grid, y_global[row_start:row_stop])
    x = np.ascontiguousarray(x0.ravel())
    y = np.ascontiguousarray(y0.ravel())
    f11 = np.ones_like(x)
    f12 = np.zeros_like(x)
    f21 = np.zeros_like(x)
    f22 = np.ones_like(x)
    return x, y, f11, f12, f21, f22


def _device_name(cuda: Any) -> str:
    name = cuda.current_context().device.name
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def _verification_sufficient_statistics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    reference64 = reference.astype(np.float64, copy=False)
    difference = candidate.astype(np.float64, copy=False) - reference64
    return {
        "count": int(reference64.size),
        "sum_abs_error": float(np.sum(np.abs(difference))),
        "sum_squared_error": float(np.dot(difference, difference)),
        "sum_squared_reference": float(np.dot(reference64, reference64)),
        "max_abs_error": float(np.max(np.abs(difference))),
    }


def aggregate_invariants(
    rank_results: list[dict[str, Any]],
) -> dict[str, float | bool]:
    total_particles = sum(int(result["particles"]) for result in rank_results)
    weighted_det_error = sum(
        int(result["particles"])
        * float(result["invariants"]["mean_abs_detF_minus_1"])
        for result in rank_results
    )
    return {
        "finite": bool(
            all(bool(result["invariants"]["finite"]) for result in rank_results)
        ),
        "max_boundary_error": float(
            max(
                float(result["invariants"]["max_boundary_error"])
                for result in rank_results
            )
        ),
        "mean_abs_detF_minus_1": float(weighted_det_error / total_particles),
        "max_abs_detF_minus_1": float(
            max(
                float(result["invariants"]["max_abs_detF_minus_1"])
                for result in rank_results
            )
        ),
    }


def aggregate_ftle_statistics(
    rank_results: list[dict[str, Any]],
) -> dict[str, float | bool]:
    total_particles = sum(int(result["particles"]) for result in rank_results)
    weighted_sum = sum(
        int(result["particles"]) * float(result["ftle_statistics"]["mean"])
        for result in rank_results
    )
    return {
        "finite": bool(
            all(
                bool(result["ftle_statistics"]["finite"])
                for result in rank_results
            )
        ),
        "minimum": float(
            min(
                float(result["ftle_statistics"]["minimum"])
                for result in rank_results
            )
        ),
        "maximum": float(
            max(
                float(result["ftle_statistics"]["maximum"])
                for result in rank_results
            )
        ),
        "mean": float(weighted_sum / total_particles),
    }


def aggregate_verification(
    rank_results: list[dict[str, Any]],
    relative_l2_tolerance: float,
    max_abs_tolerance: float,
) -> dict[str, Any]:
    statistics = [
        result.get("verification_statistics") for result in rank_results
    ]
    if any(item is None for item in statistics):
        return {
            "performed": False,
            "reference_mode": "strict",
            "passed": None,
        }

    typed_statistics = [item for item in statistics if item is not None]
    count = sum(int(item["count"]) for item in typed_statistics)
    sum_abs_error = sum(
        float(item["sum_abs_error"]) for item in typed_statistics
    )
    sum_squared_error = sum(
        float(item["sum_squared_error"]) for item in typed_statistics
    )
    sum_squared_reference = sum(
        float(item["sum_squared_reference"]) for item in typed_statistics
    )
    max_abs_error = max(
        float(item["max_abs_error"]) for item in typed_statistics
    )
    relative_l2 = (
        float(np.sqrt(sum_squared_error / sum_squared_reference))
        if sum_squared_reference > 0.0
        else float("nan")
    )
    relative_passed = bool(relative_l2 <= relative_l2_tolerance)
    max_abs_passed = bool(max_abs_error <= max_abs_tolerance)
    return {
        "performed": True,
        "reference_mode": "strict",
        "relative_l2_ftle_error": relative_l2,
        "max_abs_ftle_error": float(max_abs_error),
        "mean_abs_ftle_error": float(sum_abs_error / count),
        "relative_l2_passed": relative_passed,
        "max_abs_passed": max_abs_passed,
        "passed": bool(relative_passed and max_abs_passed),
    }


def _worker(
    rank: int,
    device_id: int,
    row_start: int,
    row_stop: int,
    settings: dict[str, Any],
    barrier: Any,
    result_queue: Any,
    shard_directory: str | None,
) -> None:
    """Own one CUDA context and solve one contiguous grid slab."""
    from numba import cuda
    import double_gyre_ftle_cuda_optimized as optimized

    try:
        cuda.select_device(device_id)
        dtype = np.dtype("float64")
        cfg = optimized.DoubleGyreConfig()
        host_state = initial_state_rows(
            settings["nx"],
            settings["ny"],
            row_start,
            row_stop,
            dtype,
            cfg,
        )
        particles = int(host_state[0].size)
        kernel = (
            optimized.rk4_chunk_strict_kernel
            if settings["mode"] == "strict"
            else optimized.rk4_chunk_mixed_kernel
        )

        print(
            f"[rank {rank}] device={device_id}, rows=[{row_start},{row_stop}), "
            f"particles={particles}",
            flush=True,
        )
        resources = optimized.compile_warmup(dtype, cfg, settings["threads"])
        d_a, d_b, coefficient_metrics = optimized.prepare_coefficients(
            settings["total_steps"],
            settings["dt"],
            settings["chunk_steps"],
            settings["threads"],
            dtype,
            cfg,
        )

        for _ in range(settings["warmup_runs"]):
            optimized.run_once(
                kernel,
                host_state,
                d_a,
                d_b,
                settings["dt"],
                settings["tau"],
                settings["chunk_steps"],
                settings["threads"],
                dtype,
                cfg,
                copy_state=False,
            )

        barrier.wait()
        runs: list[dict[str, float | int]] = []
        distributed_run_seconds: list[float] = []
        for repeat in range(settings["repeats"]):
            barrier.wait()
            distributed_start = perf_counter()
            _, run_metrics, _ = optimized.run_once(
                kernel,
                host_state,
                d_a,
                d_b,
                settings["dt"],
                settings["tau"],
                settings["chunk_steps"],
                settings["threads"],
                dtype,
                cfg,
                copy_state=False,
            )
            barrier.wait()
            if rank == 0:
                distributed_run_seconds.append(
                    float(perf_counter() - distributed_start)
                )
            runs.append(run_metrics)
            print(
                f"[rank {rank}] repeat={repeat + 1}, "
                f"integration={run_metrics['integration_seconds']:.6f}s, "
                f"end_to_end={run_metrics['end_to_end_seconds']:.6f}s",
                flush=True,
            )

        # Materialize final state outside measured repetitions so every timed
        # repetition has the same H2D/kernel/FTLE/D2H work.
        final_ftle, materialization_run, final_state = optimized.run_once(
            kernel,
            host_state,
            d_a,
            d_b,
            settings["dt"],
            settings["tau"],
            settings["chunk_steps"],
            settings["threads"],
            dtype,
            cfg,
            copy_state=True,
        )
        if final_state is None:
            raise RuntimeError("final state was not materialized")

        verification_statistics = None
        if settings["mode"] == "mixed" and settings["verify"]:
            strict_ftle, _, _ = optimized.run_once(
                optimized.rk4_chunk_strict_kernel,
                host_state,
                d_a,
                d_b,
                settings["dt"],
                settings["tau"],
                settings["chunk_steps"],
                settings["threads"],
                dtype,
                cfg,
                copy_state=False,
            )
            verification_statistics = _verification_sufficient_statistics(
                strict_ftle, final_ftle
            )

        shard_path = None
        if shard_directory is not None:
            shard_path = str(Path(shard_directory) / f"rank_{rank:04d}.npz")
            np.savez(
                shard_path,
                ftle=final_ftle,
                x_final=final_state[0],
                y_final=final_state[1],
            )

        rank_result = {
            "status": "ok",
            "rank": rank,
            "device_id": device_id,
            "device_name": _device_name(cuda),
            "row_start": row_start,
            "row_stop": row_stop,
            "rows": row_stop - row_start,
            "particles": particles,
            "compiled_resources": resources[settings["mode"]],
            "coefficient_setup": coefficient_metrics,
            "performance": optimized.summarize_runs(
                runs, particles, settings["total_steps"]
            ),
            "run_metrics": runs,
            "distributed_run_seconds": distributed_run_seconds,
            "materialization_run": materialization_run,
            "ftle_statistics": {
                "finite": bool(np.all(np.isfinite(final_ftle))),
                "minimum": float(np.min(final_ftle)),
                "maximum": float(np.max(final_ftle)),
                "mean": float(np.mean(final_ftle)),
            },
            "invariants": optimized.state_metrics(final_state, cfg),
            "verification_statistics": verification_statistics,
            "shard_path": shard_path,
        }
        result_queue.put(rank_result)
    except BaseException as exc:
        try:
            barrier.abort()
        except BaseException:
            pass
        result_queue.put(
            {
                "status": "error",
                "rank": rank,
                "device_id": device_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            cuda.close()
        except BaseException:
            pass


def _collect_worker_results(
    processes: list[Any],
    result_queue: Any,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while len(messages) < len(processes):
        try:
            messages.append(result_queue.get(timeout=1.0))
        except queue.Empty:
            reported_ranks = {int(message["rank"]) for message in messages}
            finished_without_result = [
                rank
                for rank, process in enumerate(processes)
                if not process.is_alive() and rank not in reported_ranks
            ]
            if finished_without_result:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                break
            if all(not process.is_alive() for process in processes):
                break
    for process in processes:
        process.join()

    errors = [message for message in messages if message["status"] != "ok"]
    missing = len(processes) - len(messages)
    bad_exit_codes = [
        (index, process.exitcode)
        for index, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if errors or missing or bad_exit_codes:
        details = [
            f"rank {item['rank']} on device {item['device_id']}: "
            f"{item['error']}\n{item['traceback']}"
            for item in errors
        ]
        if missing:
            details.append(f"{missing} worker(s) exited without a result")
        if bad_exit_codes:
            details.append(f"non-zero worker exits: {bad_exit_codes}")
        raise RuntimeError("multi-GPU worker failure:\n" + "\n".join(details))
    return sorted(messages, key=lambda item: int(item["rank"]))


def _save_global_output(
    path: Path,
    nx: int,
    ny: int,
    cfg: Any,
    rank_results: list[dict[str, Any]],
) -> float:
    save_start = perf_counter()
    ftle = np.empty((ny, nx), dtype=np.float64)
    x_final = np.empty((ny, nx), dtype=np.float64)
    y_final = np.empty((ny, nx), dtype=np.float64)
    for result in rank_results:
        row_slice = slice(int(result["row_start"]), int(result["row_stop"]))
        row_count = int(result["rows"])
        with np.load(result["shard_path"]) as shard:
            ftle[row_slice] = shard["ftle"].reshape(row_count, nx)
            x_final[row_slice] = shard["x_final"].reshape(row_count, nx)
            y_final[row_slice] = shard["y_final"].reshape(row_count, nx)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_grid=np.linspace(cfg.x_min, cfg.x_max, nx, dtype=np.float64),
        y_grid=np.linspace(cfg.y_min, cfg.y_max, ny, dtype=np.float64),
        ftle=ftle,
        x_final=x_final,
        y_final=y_final,
    )
    return float(perf_counter() - save_start)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=parse_devices("0"),
        help="comma-separated logical CUDA device IDs",
    )
    parser.add_argument("--mode", choices=("strict", "mixed"), default="strict")
    parser.add_argument("--nx", type=int, default=2001)
    parser.add_argument("--ny", type=int, default=1001)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--dtype", choices=("float64",), default="float64")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--relative-l2-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--det-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/double_gyre_ftle_multi_gpu.npz"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/double_gyre_ftle_multi_gpu_metrics.json"),
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> tuple[int, float]:
    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must both be at least 2")
    if len(args.devices) > args.ny:
        raise ValueError("number of devices cannot exceed --ny")
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
    total_steps = int(round(args.tau / args.dt))
    actual_tau = total_steps * args.dt
    if not np.isclose(actual_tau, args.tau, rtol=0.0, atol=1e-12):
        raise ValueError("--tau must be an integer multiple of --dt")
    return total_steps, actual_tau


def main() -> None:
    args = build_parser().parse_args()
    total_steps, actual_tau = validate_args(args)
    import double_gyre_ftle_cuda_optimized as optimized

    row_ranges = partition_rows(args.ny, len(args.devices))
    particles = args.nx * args.ny
    settings = {
        "mode": args.mode,
        "nx": args.nx,
        "ny": args.ny,
        "dt": args.dt,
        "tau": actual_tau,
        "total_steps": total_steps,
        "threads": args.threads,
        "chunk_steps": args.chunk_steps,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "verify": args.verify,
    }

    print("Double-Gyre FTLE multi-GPU V1")
    print(
        f"mode={args.mode}, devices={args.devices}, gpu_count={len(args.devices)}"
    )
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, steps={total_steps}, "
        f"dt={args.dt}, tau={actual_tau}"
    )
    print(
        f"threads={args.threads}, chunk_steps={args.chunk_steps}, "
        f"warmup_runs={args.warmup_runs}, repeats={args.repeats}, "
        f"verify={args.verify}"
    )
    print(f"row decomposition={row_ranges}")
    if len(args.devices) == 1:
        print("NOTE: one device validates the multi-process path but not scaling")
    if args.mode == "mixed" and not args.verify:
        print("WARNING: use --verify for a new mixed-mode configuration")

    context = mp.get_context("spawn")
    barrier = context.Barrier(len(args.devices))
    result_queue = context.Queue()
    cfg = optimized.DoubleGyreConfig()

    with tempfile.TemporaryDirectory(prefix="ftle_multi_gpu_") as temp_dir:
        shard_directory = None if args.no_save else temp_dir
        processes = []
        for rank, (device_id, row_range) in enumerate(
            zip(args.devices, row_ranges)
        ):
            process = context.Process(
                target=_worker,
                args=(
                    rank,
                    device_id,
                    row_range[0],
                    row_range[1],
                    settings,
                    barrier,
                    result_queue,
                    shard_directory,
                ),
                name=f"ftle-gpu-{device_id}",
            )
            process.start()
            processes.append(process)

        rank_results = _collect_worker_results(processes, result_queue)
        distributed_runs = rank_results[0]["distributed_run_seconds"]
        if len(distributed_runs) != args.repeats:
            raise RuntimeError("rank 0 did not record every distributed repeat")
        compute_wall_median = float(median(distributed_runs))
        distributed_performance = {
            "compute_wall_seconds_median": compute_wall_median,
            "particle_steps_per_second": float(
                particles * total_steps / compute_wall_median
            ),
            "per_repeat_wall_seconds": distributed_runs,
            "slowest_rank_integration_seconds_median": float(
                max(
                    result["performance"]["integration_seconds_median"]
                    for result in rank_results
                )
            ),
            "slowest_rank_end_to_end_seconds_median": float(
                max(
                    result["performance"]["end_to_end_seconds_median"]
                    for result in rank_results
                )
            ),
        }

        invariants = aggregate_invariants(rank_results)
        ftle_statistics = aggregate_ftle_statistics(rank_results)
        invariants_passed = bool(
            invariants["finite"]
            and ftle_statistics["finite"]
            and invariants["max_boundary_error"] == 0.0
            and invariants["max_abs_detF_minus_1"] <= args.det_tolerance
        )
        verification = aggregate_verification(
            rank_results,
            args.relative_l2_tolerance,
            args.max_abs_tolerance,
        )
        if verification["performed"]:
            verification["passed"] = bool(
                verification["passed"] and invariants_passed
            )
        elif args.mode == "strict":
            verification["reason"] = "strict mode is the reference"

        save_seconds = 0.0
        if not args.no_save:
            save_seconds = _save_global_output(
                args.output, args.nx, args.ny, cfg, rank_results
            )

        report_rank_results = []
        for result in rank_results:
            report_rank_results.append(
                {
                    key: value
                    for key, value in result.items()
                    if key
                    not in (
                        "status",
                        "shard_path",
                        "verification_statistics",
                        "distributed_run_seconds",
                    )
                }
            )

        report = {
            "configuration": {
                "mode": args.mode,
                "devices": args.devices,
                "gpu_count": len(args.devices),
                "decomposition": "contiguous_y_rows",
                "row_ranges": row_ranges,
                "nx": args.nx,
                "ny": args.ny,
                "particles": particles,
                "steps": total_steps,
                "dt": args.dt,
                "tau": actual_tau,
                "state_dtype": "float64",
                "trig_dtype": (
                    "float64" if args.mode == "strict" else "float32"
                ),
                "threads": args.threads,
                "chunk_steps": args.chunk_steps,
                "warmup_runs": args.warmup_runs,
                "repeats": args.repeats,
                "relative_l2_error_tolerance": args.relative_l2_tolerance,
                "max_absolute_error_tolerance": args.max_abs_tolerance,
                "max_abs_detF_minus_1_tolerance": args.det_tolerance,
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
            },
            "distributed_performance": distributed_performance,
            "rank_results": report_rank_results,
            "ftle_statistics": ftle_statistics,
            "invariants": {**invariants, "passed": invariants_passed},
            "verification": verification,
            "output": {
                "npz_saved": not args.no_save,
                "npz_path": (
                    None if args.no_save else str(args.output.resolve())
                ),
                "npz_save_seconds": save_seconds,
                "materialization_run_timed": False,
            },
        }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMulti-GPU summary")
    print(f"GPU count                    = {len(args.devices)}")
    print(f"devices                      = {args.devices}")
    print(
        "distributed compute median   = "
        f"{distributed_performance['compute_wall_seconds_median']:.6f} s"
    )
    print(
        "aggregate particle-steps/s   = "
        f"{distributed_performance['particle_steps_per_second']:.3f}"
    )
    print(
        "max |det(F)-1|               = "
        f"{invariants['max_abs_detF_minus_1']:.3e}"
    )
    print(f"invariants passed            = {invariants_passed}")
    if verification["performed"]:
        print(
            "relative L2 vs strict        = "
            f"{verification['relative_l2_ftle_error']:.3e}"
        )
        print(
            "max absolute vs strict       = "
            f"{verification['max_abs_ftle_error']:.3e}"
        )
        print(f"verification passed          = {verification['passed']}")
    if not args.no_save:
        print(f"result saved                 = {args.output.resolve()}")
    print(f"metrics saved                = {args.metrics.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    if not invariants_passed:
        raise RuntimeError("state invariant checks failed")
    if verification["performed"] and not verification["passed"]:
        raise RuntimeError("mixed mode failed strict-FP64 verification")


if __name__ == "__main__":
    main()
