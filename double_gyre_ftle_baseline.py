# -*- coding: utf-8 -*-
"""Minimal CPU reference for Double-Gyre FTLE.

This file intentionally contains only the compute path that will later be
ported and optimized:

    initial particle grid
        -> RK4 particle advection
        -> deformation-gradient update dF/dt = grad(u) @ F
        -> C = F.T @ F
        -> largest eigenvalue
        -> FTLE

Ridge extraction, structural-convergence metrics, plotting, pandas, SciPy,
and legacy diagnostics are deliberately kept outside this baseline.  The
resulting NumPy implementation is the correctness oracle for future
Numba-CUDA/CUDA C++ and multi-GPU versions.

Examples
--------
Quick correctness run:
    python double_gyre_ftle_baseline.py --nx 81 --ny 41 --dt 0.02 --taus 1,2

Reference run matching the original spatial resolution:
    python double_gyre_ftle_baseline.py --nx 241 --ny 121 --dt 0.01 --taus 10

Do not include file output in kernel timing:
    python double_gyre_ftle_baseline.py --taus 10 --no-save
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class DoubleGyreConfig:
    amplitude: float = 0.1
    epsilon: float = 0.25
    period: float = 10.0
    x_min: float = 0.0
    x_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 1.0
    t0: float = 0.0

    @property
    def omega(self) -> float:
        return 2.0 * np.pi / self.period


def velocity_and_gradient(
    t: float,
    x: np.ndarray,
    y: np.ndarray,
    cfg: DoubleGyreConfig,
) -> tuple[np.ndarray, ...]:
    """Return u, v, du/dx, du/dy, dv/dx, dv/dy analytically."""
    a = cfg.epsilon * np.sin(cfg.omega * t)
    b = 1.0 - 2.0 * cfg.epsilon * np.sin(cfg.omega * t)

    f = a * x * x + b * x
    dfdx = 2.0 * a * x + b
    d2fdx2 = 2.0 * a

    pi_f = np.pi * f
    pi_y = np.pi * y
    sin_f = np.sin(pi_f)
    cos_f = np.cos(pi_f)
    sin_y = np.sin(pi_y)
    cos_y = np.cos(pi_y)

    u = -np.pi * cfg.amplitude * sin_f * cos_y
    v = np.pi * cfg.amplitude * cos_f * sin_y * dfdx

    ux = -(np.pi**2) * cfg.amplitude * cos_f * cos_y * dfdx
    uy = (np.pi**2) * cfg.amplitude * sin_f * sin_y
    vx = np.pi * cfg.amplitude * sin_y * (
        -np.pi * sin_f * dfdx * dfdx + cos_f * d2fdx2
    )
    vy = (np.pi**2) * cfg.amplitude * cos_f * cos_y * dfdx
    return u, v, ux, uy, vx, vy


def rhs(
    t: float,
    x: np.ndarray,
    y: np.ndarray,
    f11: np.ndarray,
    f12: np.ndarray,
    f21: np.ndarray,
    f22: np.ndarray,
    cfg: DoubleGyreConfig,
) -> tuple[np.ndarray, ...]:
    """Right-hand side for particle positions and deformation gradient."""
    u, v, ux, uy, vx, vy = velocity_and_gradient(t, x, y, cfg)
    df11 = ux * f11 + uy * f21
    df12 = ux * f12 + uy * f22
    df21 = vx * f11 + vy * f21
    df22 = vx * f12 + vy * f22
    return u, v, df11, df12, df21, df22


def rk4_step(
    t: float,
    dt: float,
    state: tuple[np.ndarray, ...],
    cfg: DoubleGyreConfig,
) -> tuple[np.ndarray, ...]:
    """Advance all particles by one classical fourth-order Runge-Kutta step."""
    k1 = rhs(t, *state, cfg)
    state2 = tuple(value + 0.5 * dt * slope for value, slope in zip(state, k1))
    k2 = rhs(t + 0.5 * dt, *state2, cfg)
    state3 = tuple(value + 0.5 * dt * slope for value, slope in zip(state, k2))
    k3 = rhs(t + 0.5 * dt, *state3, cfg)
    state4 = tuple(value + dt * slope for value, slope in zip(state, k3))
    k4 = rhs(t + dt, *state4, cfg)

    out = tuple(
        value + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
        for value, s1, s2, s3, s4 in zip(state, k1, k2, k3, k4)
    )

    # Double-Gyre outer boundaries are invariant.  Clipping only removes tiny
    # round-off excursions and matches the original reference implementation.
    x, y, f11, f12, f21, f22 = out
    x = np.clip(x, cfg.x_min, cfg.x_max)
    y = np.clip(y, cfg.y_min, cfg.y_max)
    return x, y, f11, f12, f21, f22


def ftle_from_deformation_gradient(
    f11: np.ndarray,
    f12: np.ndarray,
    f21: np.ndarray,
    f22: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Compute FTLE using the closed-form largest eigenvalue of 2-D C=F^T F."""
    c11 = f11 * f11 + f21 * f21
    c12 = f11 * f12 + f21 * f22
    c22 = f12 * f12 + f22 * f22

    trace = c11 + c22
    discriminant = np.sqrt(np.maximum((c11 - c22) ** 2 + 4.0 * c12 * c12, 0.0))
    lambda_max = np.maximum(0.5 * (trace + discriminant), 1e-30)
    return 0.5 * np.log(lambda_max) / abs(tau)


def parse_taus(text: str, dt: float) -> list[float]:
    taus = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not taus or taus[0] <= 0.0:
        raise ValueError("--taus must contain one or more positive values")

    aligned = []
    for tau in taus:
        steps = max(1, int(round(tau / dt)))
        actual_tau = steps * dt
        if not np.isclose(actual_tau, tau, rtol=0.0, atol=1e-12):
            raise ValueError(f"tau={tau} is not an integer multiple of dt={dt}")
        aligned.append(actual_tau)
    return aligned


def run_reference(
    nx: int,
    ny: int,
    dt: float,
    taus: list[float],
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray], dict[str, float]]:
    """Run the NumPy CPU reference and return all requested FTLE snapshots."""
    x_grid = np.linspace(cfg.x_min, cfg.x_max, nx, dtype=dtype)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, ny, dtype=dtype)
    x0, y0 = np.meshgrid(x_grid, y_grid)

    x = x0.ravel().copy()
    y = y0.ravel().copy()
    f11 = np.ones_like(x)
    f12 = np.zeros_like(x)
    f21 = np.zeros_like(x)
    f22 = np.ones_like(x)
    state = (x, y, f11, f12, f21, f22)

    checkpoints = {int(round(tau / dt)): tau for tau in taus}
    max_step = max(checkpoints)
    snapshots: dict[float, np.ndarray] = {}
    ftle_seconds = 0.0

    integration_start = perf_counter()
    t = cfg.t0
    for step in range(1, max_step + 1):
        state = rk4_step(t, dt, state, cfg)
        t = cfg.t0 + step * dt

        if step in checkpoints:
            tau = checkpoints[step]
            ftle_start = perf_counter()
            ftle = ftle_from_deformation_gradient(*state[2:], tau)
            snapshots[tau] = ftle.reshape(ny, nx).copy()
            ftle_seconds += perf_counter() - ftle_start

    integration_seconds = perf_counter() - integration_start - ftle_seconds

    x, y, f11, f12, f21, f22 = state
    determinant = f11 * f22 - f12 * f21
    finite = all(np.all(np.isfinite(field)) for field in state)
    boundary_error = max(
        float(np.maximum(cfg.x_min - x, 0.0).max(initial=0.0)),
        float(np.maximum(x - cfg.x_max, 0.0).max(initial=0.0)),
        float(np.maximum(cfg.y_min - y, 0.0).max(initial=0.0)),
        float(np.maximum(y - cfg.y_max, 0.0).max(initial=0.0)),
    )

    metrics = {
        "particles": int(nx * ny),
        "steps": int(max_step),
        "particle_steps": int(nx * ny * max_step),
        "integration_seconds": float(integration_seconds),
        "ftle_seconds": float(ftle_seconds),
        "compute_seconds": float(integration_seconds + ftle_seconds),
        "particle_steps_per_second": float(nx * ny * max_step / integration_seconds),
        "finite": bool(finite),
        "max_boundary_error": boundary_error,
        "mean_abs_detF_minus_1": float(np.mean(np.abs(determinant - 1.0))),
        "max_abs_detF_minus_1": float(np.max(np.abs(determinant - 1.0))),
    }
    return x_grid, y_grid, snapshots, metrics


def save_results(
    path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    snapshots: dict[float, np.ndarray],
    metrics: dict[str, float],
) -> float:
    """Save an uncompressed baseline file so I/O cost is explicit and small."""
    path.parent.mkdir(parents=True, exist_ok=True)
    taus = np.asarray(sorted(snapshots), dtype=np.float64)
    ftle = np.stack([snapshots[tau] for tau in taus], axis=0)
    start = perf_counter()
    np.savez(
        path,
        x=x_grid,
        y=y_grid,
        taus=taus,
        ftle=ftle,
        metrics_json=json.dumps(metrics),
    )
    return perf_counter() - start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=241)
    parser.add_argument("--ny", type=int, default=121)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--taus",
        type=str,
        default="10",
        help="comma-separated positive integration windows; each must align with dt",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=Path("double_gyre_ftle_baseline.npz"))
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must both be at least 2")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")

    dtype = np.dtype(args.dtype)
    taus = parse_taus(args.taus, args.dt)
    cfg = DoubleGyreConfig()

    print("Double-Gyre FTLE CPU reference")
    print(f"grid={args.nx}x{args.ny}, particles={args.nx * args.ny}")
    print(f"dt={args.dt}, taus={taus}, dtype={dtype.name}")

    x_grid, y_grid, snapshots, metrics = run_reference(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        taus=taus,
        dtype=dtype,
        cfg=cfg,
    )

    for tau, ftle in snapshots.items():
        print(
            f"tau={tau:g}: FTLE min={ftle.min():.8f}, "
            f"max={ftle.max():.8f}, mean={ftle.mean():.8f}"
        )

    metrics["save_seconds"] = 0.0
    if not args.no_save:
        metrics["save_seconds"] = save_results(
            args.output, x_grid, y_grid, snapshots, metrics
        )
        print(f"saved={args.output.resolve()}")

    print(f"integration_seconds={metrics['integration_seconds']:.6f}")
    print(f"ftle_seconds={metrics['ftle_seconds']:.6f}")
    print(f"particle_steps_per_second={metrics['particle_steps_per_second']:.3f}")
    print(f"mean_abs_detF_minus_1={metrics['mean_abs_detF_minus_1']:.3e}")
    print(f"max_abs_detF_minus_1={metrics['max_abs_detF_minus_1']:.3e}")
    print("BENCHMARK_JSON=" + json.dumps(metrics, sort_keys=True))

    if not metrics["finite"] or metrics["max_boundary_error"] > 0.0:
        raise RuntimeError("correctness check failed: non-finite value or boundary violation")


if __name__ == "__main__":
    main()
