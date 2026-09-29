# -*- coding: utf-8 -*-
"""Build the final FTLE-HPC summary and A800 multi-GPU figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMMUNICATION_FILES = {
    "host_staged": "a800_host_staged_formal.json",
    "ipc_sync": "a800_ipc_sync_formal.json",
    "ipc_overlap": "a800_ipc_overlap_formal.json",
}

COMMUNICATION_LABELS = {
    "host_staged": "Host staged",
    "ipc_sync": "IPC sync",
    "ipc_overlap": "IPC overlap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/final_performance_summary.json"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/multi_gpu_communication_summary.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("assets/a800_multi_gpu_summary.png"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def critical_rank_metric(report: dict, key: str) -> float:
    """Return the slowest-rank median for a distributed phase."""
    return max(item["performance"][key] for item in report["rank_results"])


def communication_rows(results_dir: Path) -> list[dict]:
    rows = []
    reports = {
        mode: load_json(results_dir / filename)
        for mode, filename in COMMUNICATION_FILES.items()
    }
    host_halo = critical_rank_metric(
        reports["host_staged"], "halo_exchange_seconds_median"
    )
    host_phase = critical_rank_metric(
        reports["host_staged"], "communication_ftle_phase_seconds_median"
    )

    for mode, report in reports.items():
        correctness = report["correctness"]
        halo_seconds = critical_rank_metric(
            report, "halo_exchange_seconds_median"
        )
        phase_seconds = critical_rank_metric(
            report, "communication_ftle_phase_seconds_median"
        )
        rows.append(
            {
                "mode": mode,
                "label": COMMUNICATION_LABELS[mode],
                "wall_ms": (
                    report["distributed_performance"][
                        "compute_wall_seconds_median"
                    ]
                    * 1.0e3
                ),
                "throughput_billion_particle_steps_per_second": (
                    report["distributed_performance"][
                        "particle_steps_per_second"
                    ]
                    / 1.0e9
                ),
                "integration_ms": critical_rank_metric(
                    report, "integration_seconds_median"
                )
                * 1.0e3,
                "halo_us": halo_seconds * 1.0e6,
                "flowmap_ftle_us": critical_rank_metric(
                    report, "flowmap_ftle_seconds_median"
                )
                * 1.0e6,
                "communication_ftle_phase_us": phase_seconds * 1.0e6,
                "halo_speedup_vs_host": host_halo / halo_seconds,
                "phase_speedup_vs_host": host_phase / phase_seconds,
                "distributed_relative_l2_error": correctness[
                    "distributed_vs_global_flowmap"
                ]["relative_l2_error"],
                "flowmap_vs_direct_f_relative_l2_error": correctness[
                    "flowmap_vs_direct_F_discretization"
                ]["relative_l2_error"],
                "correctness_passed": correctness["distributed_passed"],
            }
        )
    return rows


def write_communication_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(results_dir: Path, communication: list[dict]) -> dict:
    single_gpu = load_json(results_dir / "single_gpu_benchmark.json")
    p2p = load_json(results_dir / "a800_p2p_nvlink.json")
    strong_scaling = load_json(
        results_dir / "a800_strong_scaling_summary.json"
    )

    largest_grid = max(
        single_gpu["results"], key=lambda item: item["particles"]
    )["particles"]
    largest_results = [
        item
        for item in single_gpu["results"]
        if item["particles"] == largest_grid
    ]
    peak_p2p = max(item["bandwidth_GBps"] for item in p2p["results"])
    ipc_sync = next(item for item in communication if item["mode"] == "ipc_sync")

    return {
        "single_gpu": {
            "gpu": single_gpu["gpu"],
            "largest_grid_particles": largest_grid,
            "largest_grid_results": largest_results,
        },
        "strong_scaling": strong_scaling,
        "p2p_nvlink": {
            "devices": p2p["devices"],
            "all_verified": p2p["all_verified"],
            "peak_measured_bandwidth_GBps": peak_p2p,
            "results": p2p["results"],
        },
        "communication": {
            "rows": communication,
            "recommended_mode": "ipc_sync",
            "ipc_sync_halo_speedup_vs_host": ipc_sync[
                "halo_speedup_vs_host"
            ],
            "ipc_sync_phase_speedup_vs_host": ipc_sync[
                "phase_speedup_vs_host"
            ],
        },
        "interpretation": {
            "total_wall_time_warning": (
                "Communication modes were measured in separate processes; "
                "compare halo and communication+FTLE phase metrics rather "
                "than total wall time, which is dominated by integration."
            ),
            "flowmap_warning": (
                "Distributed flow-map FTLE matches the monolithic flow-map "
                "reference. Its difference from direct-F is discretization "
                "error and is not a distributed communication error."
            ),
        },
    }


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_summary(
    path: Path,
    p2p: dict,
    strong_scaling: dict,
    communication: list[dict],
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(15.0, 4.5), constrained_layout=True
    )

    scaling_ax, bandwidth_ax, communication_ax = axes

    scaling = strong_scaling["results"]
    gpu_counts = [item["gpu_count"] for item in scaling]
    speedups = [item["speedup"] for item in scaling]
    scaling_ax.plot(
        gpu_counts,
        gpu_counts,
        linestyle="--",
        color="#777777",
        marker="o",
        label="Ideal",
    )
    scaling_ax.plot(
        gpu_counts,
        speedups,
        color="#4C78A8",
        marker="o",
        linewidth=2.4,
        label="Measured",
    )
    scaling_ax.set_xticks(gpu_counts)
    scaling_ax.set_ylim(0.9, 2.05)
    scaling_ax.set_xlabel("GPU count")
    scaling_ax.set_ylabel("Speedup (×)")
    scaling_ax.set_title("Direct-F strong scaling")
    scaling_ax.legend(frameon=False, loc="upper left")
    scaling_ax.annotate(
        f"{speedups[-1]:.3f}×\n{scaling[-1]['parallel_efficiency'] * 100:.1f}% efficiency",
        (gpu_counts[-1], speedups[-1]),
        xytext=(-8, -38),
        textcoords="offset points",
        ha="right",
        color="#4C78A8",
        fontweight="bold",
    )

    by_direction: dict[str, list[tuple[float, float]]] = {}
    for item in p2p["results"]:
        by_direction.setdefault(item["direction"], []).append(
            (
                item["bytes_per_transfer"] / (1024.0**2),
                item["bandwidth_GBps"],
            )
        )
    for direction, points in sorted(by_direction.items()):
        points.sort()
        bandwidth_ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=2.2,
            label=direction,
        )
    bandwidth_ax.axhline(
        200.0,
        color="#777777",
        linewidth=1.0,
        linestyle=":",
        label="NV8 link-rate sum",
    )
    bandwidth_ax.set_xscale("log", base=2)
    bandwidth_ax.set_ylim(0.0, 215.0)
    bandwidth_ax.set_xlabel("Transfer size (MiB)")
    bandwidth_ax.set_ylabel("Bandwidth (GB/s)")
    bandwidth_ax.set_title("CUDA P2P over NVLink")
    bandwidth_ax.legend(frameon=False, loc="lower right")

    labels = [item["label"] for item in communication]
    halo = [item["halo_us"] for item in communication]
    phase = [item["communication_ftle_phase_us"] for item in communication]
    positions = np.arange(len(labels))
    width = 0.36
    communication_ax.bar(
        positions - width / 2,
        halo,
        width,
        label="Halo exchange",
        color="#54A24B",
    )
    communication_ax.bar(
        positions + width / 2,
        phase,
        width,
        label="Communication + FTLE",
        color="#F58518",
    )
    communication_ax.set_xticks(positions, labels)
    communication_ax.set_ylabel("Critical-path time (μs)")
    communication_ax.set_title("Flow-map communication paths")
    communication_ax.legend(frameon=False, loc="upper center")
    for x_value, y_value in zip(positions - width / 2, halo):
        communication_ax.text(
            x_value,
            y_value + 14,
            f"{y_value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(
        "Double-Gyre FTLE multi-GPU performance — 2× NVIDIA A800 NV8",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        "2001×1001 grid, 1000 RK4 steps; communication bars use the slowest-rank median",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    communication = communication_rows(args.results_dir)
    summary = build_summary(args.results_dir, communication)
    write_communication_csv(args.summary_csv, communication)
    write_summary(args.summary_json, summary)
    plot_summary(
        args.plot,
        summary["p2p_nvlink"],
        summary["strong_scaling"],
        communication,
        args.dpi,
    )

    ipc_sync = next(item for item in communication if item["mode"] == "ipc_sync")
    print("Final FTLE-HPC summary")
    print(
        "2-GPU direct-F speedup      = "
        f"{summary['strong_scaling']['results'][-1]['speedup']:.3f}x"
    )
    print(
        "2-GPU parallel efficiency  = "
        f"{summary['strong_scaling']['results'][-1]['parallel_efficiency'] * 100:.1f}%"
    )
    print(
        "peak P2P bandwidth         = "
        f"{summary['p2p_nvlink']['peak_measured_bandwidth_GBps']:.2f} GB/s"
    )
    print(
        "IPC sync halo speedup      = "
        f"{ipc_sync['halo_speedup_vs_host']:.2f}x vs host staged"
    )
    print(
        "IPC sync phase speedup     = "
        f"{ipc_sync['phase_speedup_vs_host']:.2f}x vs host staged"
    )
    print(f"saved JSON={args.summary_json.resolve()}")
    print(f"saved CSV ={args.summary_csv.resolve()}")
    print(f"saved plot={args.plot.resolve()}")


if __name__ == "__main__":
    main()
