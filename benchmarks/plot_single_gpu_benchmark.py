# -*- coding: utf-8 -*-
"""Plot throughput and V1 speedup from the unified single-GPU benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SERIES = {
    ("v1_baseline", "float64"): ("V1 baseline FP64", "#4C78A8", "o"),
    ("optimized", "strict"): ("Optimized strict", "#F58518", "s"),
    ("optimized", "mixed"): ("Optimized mixed", "#54A24B", "^"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/single_gpu_benchmark.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/single_gpu_benchmark.png"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as input_file:
        for source in csv.DictReader(input_file):
            key = (source["implementation"], source["mode"])
            if key not in SERIES:
                continue
            rows.append(
                {
                    "key": key,
                    "particles": int(source["particles"]),
                    "throughput_billion": (
                        float(source["particle_steps_per_second"]) / 1.0e9
                    ),
                    "speedup": float(source["speedup_vs_v1"]),
                }
            )
    if not rows:
        raise ValueError(f"no recognized benchmark rows in {path}")
    return rows


def particle_tick(value: float, _position: float) -> str:
    if value >= 1.0e6:
        return f"{value / 1.0e6:g}M"
    if value >= 1.0e3:
        return f"{value / 1.0e3:g}k"
    return f"{value:g}"


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["key"]].append(row)
    for points in grouped.values():
        points.sort(key=lambda item: item["particles"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )
    figure, (throughput_ax, speedup_ax) = plt.subplots(
        1, 2, figsize=(12.0, 4.5), constrained_layout=True
    )

    for key, (label, color, marker) in SERIES.items():
        points = grouped.get(key, [])
        if not points:
            continue
        x = [point["particles"] for point in points]
        throughput = [point["throughput_billion"] for point in points]
        throughput_ax.plot(
            x,
            throughput,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )
        if key[0] == "optimized":
            speedup = [point["speedup"] for point in points]
            speedup_ax.plot(
                x,
                speedup,
                label=label,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=6,
            )
            speedup_ax.annotate(
                f"{speedup[-1]:.3f}×",
                (x[-1], speedup[-1]),
                xytext=(-4, 8),
                textcoords="offset points",
                ha="right",
                color=color,
                fontweight="bold",
            )

    for axis in (throughput_ax, speedup_ax):
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(particle_tick))
        axis.set_xlabel("Particles")

    throughput_ax.set_title("Throughput scaling")
    throughput_ax.set_ylabel("Billion particle-steps/s")
    throughput_ax.set_ylim(bottom=0.0)
    throughput_ax.legend(frameon=False, loc="lower right")

    speedup_ax.set_title("Speedup over V1 baseline")
    speedup_ax.set_ylabel("Speedup (×)")
    speedup_ax.axhline(1.0, color="#777777", linewidth=1.0, linestyle=":")
    speedup_ax.set_ylim(bottom=0.0)
    speedup_ax.legend(frameon=False, loc="upper right")

    figure.suptitle(
        "Double-Gyre FTLE single-GPU performance — NVIDIA A100",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "FP64 state and RK4; mixed mode uses FP32 trigonometric evaluation only",
        ha="center",
        color="#555555",
        fontsize=9,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
