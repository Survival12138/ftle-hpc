# -*- coding: utf-8 -*-
"""Correctness-first Numba-CUDA baseline for Double-Gyre FTLE.

V1 intentionally launches one RK4 kernel per time step and one FTLE kernel at
the end.  Particle state remains resident on the GPU for the whole trajectory.
This version is the simple GPU baseline against which later kernel fusion and
multi-step-per-launch variants will be compared.

Run on the A100 server:

    python double_gyre_ftle_cuda_v1.py \
        --nx 241 --ny 121 --dt 0.01 --tau 10 \
        --dtype float64 --repeats 3

The script reports:
  * CPU reference time and particle-step throughput;
  * GPU kernel-only time measured with CUDA events;
  * GPU end-to-end time including allocations/transfers;
  * CPU/GPU FTLE error and deformation-gradient consistency.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import median
from time import perf_counter

import numpy as np
from numba import cuda

from double_gyre_ftle_baseline import DoubleGyreConfig, run_reference


@cuda.jit(device=True)
def _rhs(
    t,
    x,
    y,
    f11,
    f12,
    f21,
    f22,
    amplitude,
    epsilon,
    omega,
):
    """Double-Gyre trajectory and deformation-gradient RHS for one particle."""
    sin_omega_t = math.sin(omega * t)
    a = epsilon * sin_omega_t
    b = 1.0 - 2.0 * epsilon * sin_omega_t

    f = a * x * x + b * x
    dfdx = 2.0 * a * x + b
    d2fdx2 = 2.0 * a

    pi_f = math.pi * f
    pi_y = math.pi * y
    sin_f = math.sin(pi_f)
    cos_f = math.cos(pi_f)
    sin_y = math.sin(pi_y)
    cos_y = math.cos(pi_y)

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
def rk4_step_kernel(
    x,
    y,
    f11,
    f12,
    f21,
    f22,
    t,
    dt,
    amplitude,
    epsilon,
    omega,
    x_min,
    x_max,
    y_min,
    y_max,
):
    """Advance one RK4 step; one CUDA thread owns one particle."""
    i = cuda.grid(1)
    if i >= x.size:
        return

    x0 = x[i]
    y0 = y[i]
    f11_0 = f11[i]
    f12_0 = f12[i]
    f21_0 = f21[i]
    f22_0 = f22[i]

    k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs(
        t,
        x0,
        y0,
        f11_0,
        f12_0,
        f21_0,
        f22_0,
        amplitude,
        epsilon,
        omega,
    )

    half_dt = 0.5 * dt
    k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs(
        t + half_dt,
        x0 + half_dt * k1x,
        y0 + half_dt * k1y,
        f11_0 + half_dt * k1f11,
        f12_0 + half_dt * k1f12,
        f21_0 + half_dt * k1f21,
        f22_0 + half_dt * k1f22,
        amplitude,
        epsilon,
        omega,
    )

    k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs(
        t + half_dt,
        x0 + half_dt * k2x,
        y0 + half_dt * k2y,
        f11_0 + half_dt * k2f11,
        f12_0 + half_dt * k2f12,
        f21_0 + half_dt * k2f21,
        f22_0 + half_dt * k2f22,
        amplitude,
        epsilon,
        omega,
    )

    k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs(
        t + dt,
        x0 + dt * k3x,
        y0 + dt * k3y,
        f11_0 + dt * k3f11,
        f12_0 + dt * k3f12,
        f21_0 + dt * k3f21,
        f22_0 + dt * k3f22,
        amplitude,
        epsilon,
        omega,
    )

    sixth_dt = dt / 6.0
    x_new = x0 + sixth_dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y0 + sixth_dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

    if x_new < x_min:
        x_new = x_min
    elif x_new > x_max:
        x_new = x_max

    if y_new < y_min:
        y_new = y_min
    elif y_new > y_max:
        y_new = y_max

    x[i] = x_new
    y[i] = y_new
    f11[i] = f11_0 + sixth_dt * (
        k1f11 + 2.0 * k2f11 + 2.0 * k3f11 + k4f11
    )
    f12[i] = f12_0 + sixth_dt * (
        k1f12 + 2.0 * k2f12 + 2.0 * k3f12 + k4f12
    )
    f21[i] = f21_0 + sixth_dt * (
        k1f21 + 2.0 * k2f21 + 2.0 * k3f21 + k4f21
    )
    f22[i] = f22_0 + sixth_dt * (
        k1f22 + 2.0 * k2f22 + 2.0 * k3f22 + k4f22
    )


@cuda.jit
def ftle_kernel(f11, f12, f21, f22, tau, ftle):
    """Compute final FTLE; one CUDA thread owns one particle."""
    i = cuda.grid(1)
    if i >= ftle.size:
        return

    a11 = f11[i]
    a12 = f12[i]
    a21 = f21[i]
    a22 = f22[i]

    c11 = a11 * a11 + a21 * a21
    c12 = a11 * a12 + a21 * a22
    c22 = a12 * a12 + a22 * a22

    trace = c11 + c22
    disc_arg = (c11 - c22) * (c11 - c22) + 4.0 * c12 * c12
    if disc_arg < 0.0:
        disc_arg = 0.0
    lambda_max = 0.5 * (trace + math.sqrt(disc_arg))
    if lambda_max < 1e-30:
        lambda_max = 1e-30

    ftle[i] = 0.5 * math.log(lambda_max) / abs(tau)


def initial_state(
    nx: int,
    ny: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
) -> tuple[np.ndarray, ...]:
    """Create the same structure-of-arrays initial state as the CPU reference."""
    x_grid = np.linspace(cfg.x_min, cfg.x_max, nx, dtype=dtype)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, ny, dtype=dtype)
    x0, y0 = np.meshgrid(x_grid, y_grid)
    x = np.ascontiguousarray(x0.ravel())
    y = np.ascontiguousarray(y0.ravel())
    f11 = np.ones_like(x)
    f12 = np.zeros_like(x)
    f21 = np.zeros_like(x)
    f22 = np.ones_like(x)
    return x, y, f11, f12, f21, f22


def warm_up_cuda(dtype: np.dtype, cfg: DoubleGyreConfig, threads: int) -> None:
    """Compile both kernels before any benchmark timing."""
    state = initial_state(2, 2, dtype, cfg)
    device_state = tuple(cuda.to_device(array) for array in state)
    d_ftle = cuda.device_array(state[0].size, dtype=dtype)
    scalar = dtype.type
    blocks = (state[0].size + threads - 1) // threads

    rk4_step_kernel[blocks, threads](
        *device_state,
        scalar(cfg.t0),
        scalar(0.01),
        scalar(cfg.amplitude),
        scalar(cfg.epsilon),
        scalar(cfg.omega),
        scalar(cfg.x_min),
        scalar(cfg.x_max),
        scalar(cfg.y_min),
        scalar(cfg.y_max),
    )
    ftle_kernel[blocks, threads](*device_state[2:], scalar(0.01), d_ftle)
    cuda.synchronize()


def run_gpu_once(
    host_state: tuple[np.ndarray, ...],
    dt: float,
    tau: float,
    threads: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
) -> tuple[np.ndarray, dict[str, float], tuple[np.ndarray, ...]]:
    """Run one measured GPU repetition from freshly initialized state."""
    scalar = dtype.type
    n = host_state[0].size
    steps = int(round(tau / dt))
    blocks = (n + threads - 1) // threads

    end_to_end_start = perf_counter()
    h2d_start = perf_counter()
    device_state = tuple(cuda.to_device(array) for array in host_state)
    d_ftle = cuda.device_array(n, dtype=dtype)
    cuda.synchronize()
    h2d_seconds = perf_counter() - h2d_start

    start_event = cuda.event(timing=True)
    end_event = cuda.event(timing=True)
    start_event.record()

    for step in range(steps):
        t = cfg.t0 + step * dt
        rk4_step_kernel[blocks, threads](
            *device_state,
            scalar(t),
            scalar(dt),
            scalar(cfg.amplitude),
            scalar(cfg.epsilon),
            scalar(cfg.omega),
            scalar(cfg.x_min),
            scalar(cfg.x_max),
            scalar(cfg.y_min),
            scalar(cfg.y_max),
        )

    ftle_kernel[blocks, threads](
        *device_state[2:], scalar(tau), d_ftle
    )
    end_event.record()
    end_event.synchronize()
    kernel_seconds = cuda.event_elapsed_time(start_event, end_event) / 1000.0

    d2h_start = perf_counter()
    ftle = d_ftle.copy_to_host()
    cuda.synchronize()
    d2h_seconds = perf_counter() - d2h_start
    end_to_end_seconds = perf_counter() - end_to_end_start

    # State copies are outside the end-to-end benchmark.  They are used only
    # for correctness diagnostics and are not required by a production run.
    final_state = tuple(array.copy_to_host() for array in device_state)
    cuda.synchronize()

    metrics = {
        "kernel_seconds": float(kernel_seconds),
        "h2d_seconds": float(h2d_seconds),
        "d2h_seconds": float(d2h_seconds),
        "end_to_end_seconds": float(end_to_end_seconds),
        "kernel_particle_steps_per_second": float(n * steps / kernel_seconds),
    }
    return ftle, metrics, final_state


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = np.linalg.norm(reference.astype(np.float64))
    relative_l2 = np.linalg.norm(diff) / denominator if denominator > 0.0 else np.nan
    return {
        "max_abs_ftle_error": float(np.max(np.abs(diff))),
        "mean_abs_ftle_error": float(np.mean(np.abs(diff))),
        "relative_l2_ftle_error": float(relative_l2),
    }


def state_metrics(state: tuple[np.ndarray, ...], cfg: DoubleGyreConfig) -> dict[str, float]:
    x, y, f11, f12, f21, f22 = state
    determinant = f11 * f22 - f12 * f21
    boundary_error = max(
        float(np.maximum(cfg.x_min - x, 0.0).max(initial=0.0)),
        float(np.maximum(x - cfg.x_max, 0.0).max(initial=0.0)),
        float(np.maximum(cfg.y_min - y, 0.0).max(initial=0.0)),
        float(np.maximum(y - cfg.y_max, 0.0).max(initial=0.0)),
    )
    return {
        "gpu_finite": bool(all(np.all(np.isfinite(array)) for array in state)),
        "gpu_max_boundary_error": boundary_error,
        "gpu_mean_abs_detF_minus_1": float(np.mean(np.abs(determinant - 1.0))),
        "gpu_max_abs_detF_minus_1": float(np.max(np.abs(determinant - 1.0))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=241)
    parser.add_argument("--ny", type=int, default=121)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="skip CPU reference and CPU/GPU error calculation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must both be at least 2")
    if args.dt <= 0.0 or args.tau <= 0.0:
        raise ValueError("--dt and --tau must be positive")
    if args.threads <= 0 or args.threads > 1024:
        raise ValueError("--threads must be in [1, 1024]")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    steps = int(round(args.tau / args.dt))
    actual_tau = steps * args.dt
    if not np.isclose(actual_tau, args.tau, rtol=0.0, atol=1e-12):
        raise ValueError("--tau must be an integer multiple of --dt")

    if not cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check LD_LIBRARY_PATH and the NVIDIA driver")

    dtype = np.dtype(args.dtype)
    cfg = DoubleGyreConfig()
    host_state = initial_state(args.nx, args.ny, dtype, cfg)
    particles = args.nx * args.ny

    print("Double-Gyre FTLE CUDA V1")
    print(f"grid={args.nx}x{args.ny}, particles={particles}, steps={steps}")
    print(
        f"dt={args.dt}, tau={actual_tau}, dtype={dtype.name}, "
        f"threads={args.threads}, repeats={args.repeats}"
    )

    print("Warming up CUDA JIT...")
    warm_up_cuda(dtype, cfg, args.threads)

    cpu_ftle = None
    cpu_metrics = None
    if not args.skip_cpu:
        print("Running CPU reference...")
        _, _, cpu_snapshots, cpu_metrics = run_reference(
            nx=args.nx,
            ny=args.ny,
            dt=args.dt,
            taus=[actual_tau],
            dtype=dtype,
            cfg=cfg,
        )
        cpu_ftle = cpu_snapshots[actual_tau].ravel()

    runs = []
    gpu_ftle = None
    final_state = None
    print("Running measured GPU repetitions...")
    for repeat in range(args.repeats):
        gpu_ftle, metrics, final_state = run_gpu_once(
            host_state=host_state,
            dt=args.dt,
            tau=actual_tau,
            threads=args.threads,
            dtype=dtype,
            cfg=cfg,
        )
        runs.append(metrics)
        print(
            f"repeat={repeat + 1}: kernel={metrics['kernel_seconds']:.6f}s, "
            f"end_to_end={metrics['end_to_end_seconds']:.6f}s"
        )

    kernel_seconds = median(run["kernel_seconds"] for run in runs)
    end_to_end_seconds = median(run["end_to_end_seconds"] for run in runs)
    h2d_seconds = median(run["h2d_seconds"] for run in runs)
    d2h_seconds = median(run["d2h_seconds"] for run in runs)

    summary = {
        "particles": particles,
        "steps": steps,
        "particle_steps": particles * steps,
        "dtype": dtype.name,
        "threads": args.threads,
        "repeats": args.repeats,
        "gpu_kernel_seconds_median": float(kernel_seconds),
        "gpu_end_to_end_seconds_median": float(end_to_end_seconds),
        "gpu_h2d_seconds_median": float(h2d_seconds),
        "gpu_d2h_seconds_median": float(d2h_seconds),
        "gpu_kernel_particle_steps_per_second": float(
            particles * steps / kernel_seconds
        ),
    }
    summary.update(state_metrics(final_state, cfg))

    if cpu_metrics is not None and cpu_ftle is not None:
        summary["cpu_seconds"] = cpu_metrics["compute_seconds"]
        summary["kernel_speedup_vs_cpu"] = (
            cpu_metrics["compute_seconds"] / kernel_seconds
        )
        summary["end_to_end_speedup_vs_cpu"] = (
            cpu_metrics["compute_seconds"] / end_to_end_seconds
        )
        summary.update(error_metrics(cpu_ftle, gpu_ftle))

    print("\nResults")
    print(f"GPU kernel median       = {kernel_seconds:.6f} s")
    print(f"GPU end-to-end median   = {end_to_end_seconds:.6f} s")
    print(
        "GPU throughput          = "
        f"{summary['gpu_kernel_particle_steps_per_second']:.3f} particle-steps/s"
    )

    if cpu_metrics is not None:
        print(f"CPU compute time        = {summary['cpu_seconds']:.6f} s")
        print(f"Kernel speedup          = {summary['kernel_speedup_vs_cpu']:.3f}x")
        print(f"End-to-end speedup      = {summary['end_to_end_speedup_vs_cpu']:.3f}x")
        print(f"Max absolute FTLE error = {summary['max_abs_ftle_error']:.3e}")
        print(f"Relative L2 FTLE error  = {summary['relative_l2_ftle_error']:.3e}")

    print(
        "GPU max |det(F)-1|      = "
        f"{summary['gpu_max_abs_detF_minus_1']:.3e}"
    )
    print("BENCHMARK_JSON=" + json.dumps(summary, sort_keys=True))

    if not summary["gpu_finite"] or summary["gpu_max_boundary_error"] > 0.0:
        raise RuntimeError("GPU correctness check failed: non-finite value or boundary violation")

    if cpu_ftle is not None:
        # CPU NumPy and GPU libdevice use different transcendental-function
        # implementations and may also differ through fused operations.  The
        # tolerance is strict enough to catch indexing/integration bugs while
        # allowing those expected floating-point differences to accumulate.
        tolerance = 1e-8 if dtype == np.dtype("float64") else 1e-4
        if summary["relative_l2_ftle_error"] > tolerance:
            raise RuntimeError(
                "CPU/GPU FTLE mismatch: relative L2 error "
                f"{summary['relative_l2_ftle_error']:.3e} > {tolerance:.1e}"
            )


if __name__ == "__main__":
    main()
