# -*- coding: utf-8 -*-
"""Reproducible single-GPU benchmark driver for Double-Gyre FTLE.

The driver runs the original V1 CUDA baseline plus the standalone optimized
strict and mixed modes.  It writes per-run logs, a machine-readable JSON
report, and a compact CSV summary.

Example:

    python benchmarks/bench_single_gpu.py \
        --grids 241x121,501x251,1001x501,2001x1001 \
        --tau 10 --dt 0.01 --warmup-runs 2 --repeats 7 \
        --output results/single_gpu_benchmark.json \
        --csv results/single_gpu_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_grids(text: str) -> list[tuple[int, int]]:
    grids: list[tuple[int, int]] = []
    for field in text.split(","):
        field = field.strip().lower()
        if not field:
            continue
        try:
            nx_text, ny_text = field.split("x", maxsplit=1)
            grid = (int(nx_text), int(ny_text))
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(
                f"invalid grid {field!r}; expected NXxNY"
            ) from exc
        if grid[0] < 2 or grid[1] < 2:
            raise argparse.ArgumentTypeError("grid dimensions must be >= 2")
        if grid not in grids:
            grids.append(grid)
    if not grids:
        raise argparse.ArgumentTypeError("grid list must not be empty")
    return grids


def parse_modes(text: str) -> list[str]:
    modes: list[str] = []
    for field in text.split(","):
        mode = field.strip().lower()
        if not mode:
            continue
        if mode not in ("strict", "mixed"):
            raise argparse.ArgumentTypeError(
                "modes must be a comma-separated subset of strict,mixed"
            )
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("mode list must not be empty")
    return modes


def run_command(
    command: list[str],
    log_path: Path,
) -> dict[str, Any] | None:
    """Stream a command, retain its log, and parse BENCHMARK_JSON if present."""
    print("\n$ " + shlex.join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark: dict[str, Any] | None = None

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("failed to capture subprocess output")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            if line.startswith("BENCHMARK_JSON="):
                benchmark = json.loads(line.split("=", maxsplit=1)[1])
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"benchmark command failed with exit code {return_code}; "
            f"see {log_path}"
        )
    return benchmark


def query_gpu() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first_line = completed.stdout.strip().splitlines()[0]
    name, driver, memory_mib = [item.strip() for item in first_line.split(",")]
    return {
        "name": name,
        "driver_version": driver,
        "memory_mib": memory_mib,
    }


def v1_row(
    nx: int,
    ny: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    compute_seconds = float(report["gpu_kernel_seconds_median"])
    return {
        "implementation": "v1_baseline",
        "mode": "float64",
        "nx": nx,
        "ny": ny,
        "particles": int(report["particles"]),
        "steps": int(report["steps"]),
        "compute_seconds": compute_seconds,
        "end_to_end_seconds": float(report["gpu_end_to_end_seconds_median"]),
        "particle_steps_per_second": float(
            report["gpu_kernel_particle_steps_per_second"]
        ),
        "speedup_vs_v1": 1.0,
        "relative_l2_vs_strict": None,
        "max_abs_vs_strict": None,
        "verification_passed": None,
    }


def optimized_row(
    mode: str,
    report: dict[str, Any],
    v1_seconds: float | None,
) -> dict[str, Any]:
    configuration = report["configuration"]
    performance = report["performance"]
    verification = report["verification"]
    compute_seconds = float(performance["all_kernel_seconds_median"])
    particle_steps = int(configuration["particles"]) * int(
        configuration["steps"]
    )
    return {
        "implementation": "optimized",
        "mode": mode,
        "nx": int(configuration["nx"]),
        "ny": int(configuration["ny"]),
        "particles": int(configuration["particles"]),
        "steps": int(configuration["steps"]),
        "compute_seconds": compute_seconds,
        "end_to_end_seconds": float(
            performance["end_to_end_seconds_median"]
        ),
        "particle_steps_per_second": float(particle_steps / compute_seconds),
        "speedup_vs_v1": (
            None if v1_seconds is None else float(v1_seconds / compute_seconds)
        ),
        "relative_l2_vs_strict": (
            verification.get("relative_l2_ftle_error")
            if verification["performed"]
            else 0.0 if mode == "strict" else None
        ),
        "max_abs_vs_strict": (
            verification.get("max_abs_ftle_error")
            if verification["performed"]
            else 0.0 if mode == "strict" else None
        ),
        "verification_passed": (
            verification.get("passed")
            if verification["performed"]
            else True if mode == "strict" else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grids",
        type=parse_grids,
        default=parse_grids("241x121,501x251,1001x501,2001x1001"),
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes("strict,mixed"),
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--v1-threads", type=int, default=256)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--skip-v1",
        action="store_true",
        help="benchmark only optimized modes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/single_gpu_benchmark.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/single_gpu_benchmark.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dt <= 0.0 or args.tau <= 0.0:
        raise ValueError("--dt and --tau must be positive")
    if args.repeats < 1 or args.warmup_runs < 1:
        raise ValueError("--repeats and --warmup-runs must be positive")

    optimized_script = REPOSITORY_ROOT / "double_gyre_ftle_cuda_optimized.py"
    v1_script = REPOSITORY_ROOT / "double_gyre_ftle_cuda_v1.py"
    if not optimized_script.is_file():
        raise FileNotFoundError(optimized_script)
    if not args.skip_v1 and not v1_script.is_file():
        raise FileNotFoundError(v1_script)

    output_path = (
        args.output
        if args.output.is_absolute()
        else REPOSITORY_ROOT / args.output
    )
    csv_path = args.csv if args.csv.is_absolute() else REPOSITORY_ROOT / args.csv
    run_directory = output_path.parent / "single_gpu_runs"
    rows: list[dict[str, Any]] = []

    for nx, ny in args.grids:
        v1_seconds: float | None = None
        if not args.skip_v1:
            command = [
                sys.executable,
                str(v1_script),
                "--nx",
                str(nx),
                "--ny",
                str(ny),
                "--dt",
                str(args.dt),
                "--tau",
                str(args.tau),
                "--dtype",
                "float64",
                "--threads",
                str(args.v1_threads),
                "--repeats",
                str(args.repeats),
                "--skip-cpu",
            ]
            parsed = run_command(
                command,
                run_directory / f"v1_{nx}x{ny}.log",
            )
            if parsed is None:
                raise RuntimeError("V1 did not emit BENCHMARK_JSON")
            row = v1_row(nx, ny, parsed)
            rows.append(row)
            v1_seconds = float(row["compute_seconds"])

        for mode in args.modes:
            metrics_path = run_directory / f"optimized_{mode}_{nx}x{ny}.json"
            command = [
                sys.executable,
                str(optimized_script),
                "--mode",
                mode,
                "--nx",
                str(nx),
                "--ny",
                str(ny),
                "--dt",
                str(args.dt),
                "--tau",
                str(args.tau),
                "--threads",
                str(args.threads),
                "--chunk-steps",
                str(args.chunk_steps),
                "--warmup-runs",
                str(args.warmup_runs),
                "--repeats",
                str(args.repeats),
                "--no-save",
                "--metrics",
                str(metrics_path),
            ]
            if mode == "mixed":
                command.append("--verify")
            run_command(
                command,
                run_directory / f"optimized_{mode}_{nx}x{ny}.log",
            )
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(optimized_row(mode, report, v1_seconds))

    report = {
        "configuration": {
            "grids": [f"{nx}x{ny}" for nx, ny in args.grids],
            "modes": args.modes,
            "dt": args.dt,
            "tau": args.tau,
            "threads": args.threads,
            "v1_threads": args.v1_threads,
            "chunk_steps": args.chunk_steps,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "v1_included": not args.skip_v1,
        },
        "gpu": query_gpu(),
        "results": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fieldnames = [
        "implementation",
        "mode",
        "nx",
        "ny",
        "particles",
        "steps",
        "compute_seconds",
        "end_to_end_seconds",
        "particle_steps_per_second",
        "speedup_vs_v1",
        "relative_l2_vs_strict",
        "max_abs_vs_strict",
        "verification_passed",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSingle-GPU benchmark summary")
    print(
        "implementation       grid       compute(s)  throughput(psteps/s)  "
        "speedup_vs_v1"
    )
    for row in rows:
        label = f"{row['implementation']}_{row['mode']}"
        speedup = row["speedup_vs_v1"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.3f}x"
        print(
            f"{label:20s} "
            f"{row['nx']:4d}x{row['ny']:<4d} "
            f"{row['compute_seconds']:11.6f}  "
            f"{row['particle_steps_per_second']:21.3f}  "
            f"{speedup_text}"
        )
    print(f"saved JSON: {output_path.resolve()}")
    print(f"saved CSV : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
