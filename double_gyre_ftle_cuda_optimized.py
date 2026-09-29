# -*- coding: utf-8 -*-
"""Standalone optimized CUDA solver for Double-Gyre FTLE.

The production entry point contains the accepted single-GPU optimizations:

* structure-of-arrays particle and deformation-gradient state;
* multi-step RK4 kernel fusion;
* precomputed time-only Double-Gyre coefficients;
* paired ``sincospi`` evaluation;
* optional FP32 trigonometry with FP64 state and RK4 accumulation.

``strict`` is the safe default and evaluates trigonometry in FP64.  ``mixed``
must be selected explicitly and can be checked against strict FP64 with
``--verify``.  The file is self-contained and does not import V1--V10.

Examples
--------
Strict scientific reference:

    python double_gyre_ftle_cuda_optimized.py \
        --mode strict --nx 2001 --ny 1001 --dt 0.01 --tau 10

Verified high-performance run:

    python double_gyre_ftle_cuda_optimized.py \
        --mode mixed --nx 2001 --ny 1001 --dt 0.01 --tau 10 \
        --threads 64 --chunk-steps 50 --warmup-runs 1 --repeats 5 \
        --verify \
        --output results/final_mixed.npz \
        --metrics results/final_mixed.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
from numba import cuda, float32
from numba.cuda import libdevice


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
        return 2.0 * math.pi / self.period


@cuda.jit
def precompute_time_coefficients_kernel(
    stage_times,
    a_values,
    b_values,
    epsilon,
    omega,
):
    """Evaluate a(t) and b(t) once for every RK4 stage time."""
    i = cuda.grid(1)
    if i >= stage_times.size:
        return

    sin_omega_t = math.sin(omega * stage_times[i])
    a = epsilon * sin_omega_t
    a_values[i] = a
    b_values[i] = 1.0 - 2.0 * a


@cuda.jit(device=True, inline="always")
def _rhs_strict(
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
    """FP64 Double-Gyre trajectory and deformation-gradient RHS."""
    f = a * x * x + b * x
    dfdx = 2.0 * a * x + b
    d2fdx2 = 2.0 * a

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


@cuda.jit(device=True, inline="always")
def _rhs_mixed(
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
    """FP64 RHS with only the paired trigonometric evaluation in FP32."""
    f = a * x * x + b * x
    dfdx = 2.0 * a * x + b
    d2fdx2 = 2.0 * a

    sin_f, cos_f = libdevice.sincospif(float32(f))
    sin_y, cos_y = libdevice.sincospif(float32(y))

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
def rk4_chunk_strict_kernel(
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
    """Advance one particle for a fused chunk using FP64 sincospi."""
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

        k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs_strict(
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
        k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs_strict(
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

        k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs_strict(
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

        k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs_strict(
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


@cuda.jit
def rk4_chunk_mixed_kernel(
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
    """Advance one particle for a fused chunk using FP32 sincospi."""
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

        k1x, k1y, k1f11, k1f12, k1f21, k1f22 = _rhs_mixed(
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
        k2x, k2y, k2f11, k2f12, k2f21, k2f22 = _rhs_mixed(
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

        k3x, k3y, k3f11, k3f12, k3f21, k3f22 = _rhs_mixed(
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

        k4x, k4y, k4f11, k4f12, k4f21, k4f22 = _rhs_mixed(
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


@cuda.jit
def ftle_kernel(f11, f12, f21, f22, tau, ftle):
    """Compute FTLE from the largest eigenvalue of C=F.T@F."""
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


def build_stage_times(
    total_steps: int,
    dt: float,
    chunk_steps: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
) -> np.ndarray:
    """Build t, t+dt/2, and t+dt in the same order as fused RK4."""
    scalar = dtype.type
    dt_scalar = scalar(dt)
    half_dt = scalar(0.5) * dt_scalar
    stage_times = np.empty(3 * total_steps, dtype=dtype)

    for chunk_start in range(0, total_steps, chunk_steps):
        t_start = scalar(cfg.t0 + chunk_start * dt)
        current_chunk = min(chunk_steps, total_steps - chunk_start)
        for local_step in range(current_chunk):
            step = chunk_start + local_step
            t = scalar(t_start + scalar(local_step) * dt_scalar)
            base = 3 * step
            stage_times[base] = t
            stage_times[base + 1] = scalar(t + half_dt)
            stage_times[base + 2] = scalar(t + dt_scalar)
    return stage_times


def prepare_coefficients(
    total_steps: int,
    dt: float,
    chunk_steps: int,
    threads: int,
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
) -> tuple[Any, Any, dict[str, float | int]]:
    setup_start = perf_counter()
    stage_times = build_stage_times(total_steps, dt, chunk_steps, dtype, cfg)
    host_build_seconds = perf_counter() - setup_start

    device_start = perf_counter()
    d_stage_times = cuda.to_device(stage_times)
    d_a = cuda.device_array(stage_times.size, dtype=dtype)
    d_b = cuda.device_array(stage_times.size, dtype=dtype)
    blocks = (stage_times.size + threads - 1) // threads
    scalar = dtype.type
    precompute_time_coefficients_kernel[blocks, threads](
        d_stage_times,
        d_a,
        d_b,
        scalar(cfg.epsilon),
        scalar(cfg.omega),
    )
    cuda.synchronize()
    device_setup_seconds = perf_counter() - device_start

    return d_a, d_b, {
        "coefficient_entries": int(stage_times.size),
        "coefficient_bytes": int(2 * stage_times.size * dtype.itemsize),
        "host_stage_time_build_seconds": float(host_build_seconds),
        "device_coefficient_setup_seconds": float(device_setup_seconds),
        "total_coefficient_setup_seconds": float(
            perf_counter() - setup_start
        ),
    }


def _resource_value(value: Any) -> int | None:
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
    """Query resources with compatibility for numba-cuda 0.30.x."""
    compiled_overloads = list(kernel.overloads.values())
    compiled = compiled_overloads[-1] if compiled_overloads else None

    def query(method_name: str, attribute_name: str) -> int | None:
        method = getattr(kernel, method_name, None)
        if method is not None:
            try:
                normalized = _resource_value(method())
                if normalized is not None:
                    return normalized
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                pass
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
    dtype: np.dtype,
    cfg: DoubleGyreConfig,
    threads: int,
) -> dict[str, dict[str, int | None]]:
    """Compile both production variants on tiny arrays before timing."""
    scalar = dtype.type
    state = initial_state(2, 2, dtype, cfg)
    blocks = (state[0].size + threads - 1) // threads
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

    for kernel in (rk4_chunk_strict_kernel, rk4_chunk_mixed_kernel):
        device_state = tuple(cuda.to_device(array) for array in state)
        kernel[blocks, threads](
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

    d_ftle = cuda.device_array(state[0].size, dtype=dtype)
    ftle_kernel[blocks, threads](
        *device_state[2:], scalar(0.01), d_ftle
    )
    cuda.synchronize()
    return {
        "strict": query_kernel_resources(rk4_chunk_strict_kernel),
        "mixed": query_kernel_resources(rk4_chunk_mixed_kernel),
    }


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
        final_state = tuple(array.copy_to_host() for array in device_state)
        cuda.synchronize()

    return ftle, {
        "rk4_kernel_launches": int(launches),
        "integration_seconds": float(integration_seconds),
        "ftle_seconds": float(ftle_seconds),
        "all_kernel_seconds": float(integration_seconds + ftle_seconds),
        "h2d_seconds": float(h2d_seconds),
        "d2h_seconds": float(d2h_seconds),
        "end_to_end_seconds": float(end_to_end_seconds),
        "particle_steps_per_second": float(
            particles * total_steps / integration_seconds
        ),
    }, final_state


def summarize_runs(
    runs: list[dict[str, float | int]],
    particles: int,
    total_steps: int,
) -> dict[str, float | int]:
    integration_seconds = median(
        float(run["integration_seconds"]) for run in runs
    )
    return {
        "rk4_kernel_launches": int(runs[0]["rk4_kernel_launches"]),
        "integration_seconds_median": float(integration_seconds),
        "ftle_seconds_median": float(
            median(float(run["ftle_seconds"]) for run in runs)
        ),
        "all_kernel_seconds_median": float(
            median(float(run["all_kernel_seconds"]) for run in runs)
        ),
        "h2d_seconds_median": float(
            median(float(run["h2d_seconds"]) for run in runs)
        ),
        "d2h_seconds_median": float(
            median(float(run["d2h_seconds"]) for run in runs)
        ),
        "end_to_end_seconds_median": float(
            median(float(run["end_to_end_seconds"]) for run in runs)
        ),
        "particle_steps_per_second": float(
            particles * total_steps / integration_seconds
        ),
    }


def state_metrics(
    state: tuple[np.ndarray, ...],
    cfg: DoubleGyreConfig,
) -> dict[str, float | bool]:
    x, y, f11, f12, f21, f22 = state
    determinant = f11 * f22 - f12 * f21
    boundary_error = max(
        float(np.maximum(cfg.x_min - x, 0.0).max(initial=0.0)),
        float(np.maximum(x - cfg.x_max, 0.0).max(initial=0.0)),
        float(np.maximum(cfg.y_min - y, 0.0).max(initial=0.0)),
        float(np.maximum(y - cfg.y_max, 0.0).max(initial=0.0)),
    )
    return {
        "finite": bool(all(np.all(np.isfinite(array)) for array in state)),
        "max_boundary_error": boundary_error,
        "mean_abs_detF_minus_1": float(np.mean(np.abs(determinant - 1.0))),
        "max_abs_detF_minus_1": float(np.max(np.abs(determinant - 1.0))),
    }


def error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    reference64 = reference.astype(np.float64)
    diff = candidate.astype(np.float64) - reference64
    denominator = np.linalg.norm(reference64)
    relative_l2 = (
        np.linalg.norm(diff) / denominator if denominator > 0.0 else np.nan
    )
    return {
        "max_abs_ftle_error": float(np.max(np.abs(diff))),
        "mean_abs_ftle_error": float(np.mean(np.abs(diff))),
        "relative_l2_ftle_error": float(relative_l2),
    }


def save_npz(
    path: Path,
    nx: int,
    ny: int,
    cfg: DoubleGyreConfig,
    ftle: np.ndarray,
    final_state: tuple[np.ndarray, ...],
) -> float:
    save_start = perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    x_grid = np.linspace(cfg.x_min, cfg.x_max, nx, dtype=np.float64)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, ny, dtype=np.float64)
    np.savez_compressed(
        path,
        x_grid=x_grid,
        y_grid=y_grid,
        ftle=ftle.reshape(ny, nx),
        x_final=final_state[0].reshape(ny, nx),
        y_final=final_state[1].reshape(ny, nx),
    )
    return perf_counter() - save_start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("strict", "mixed"), default="strict")
    parser.add_argument("--nx", type=int, default=2001)
    parser.add_argument("--ny", type=int, default=1001)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument(
        "--dtype",
        choices=("float64",),
        default="float64",
        help="state, coefficients, RK4, and FTLE remain float64",
    )
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="for mixed mode, compare the final FTLE with strict FP64",
    )
    parser.add_argument("--relative-l2-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--det-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/double_gyre_ftle.npz"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/double_gyre_ftle_metrics.json"),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="skip the compressed FTLE/flow-map NPZ output",
    )
    return parser


def validate_args(args: argparse.Namespace) -> tuple[int, float]:
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

    total_steps = int(round(args.tau / args.dt))
    actual_tau = total_steps * args.dt
    if not np.isclose(actual_tau, args.tau, rtol=0.0, atol=1e-12):
        raise ValueError("--tau must be an integer multiple of --dt")
    return total_steps, actual_tau


def main() -> None:
    args = build_parser().parse_args()
    total_steps, actual_tau = validate_args(args)
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; check LD_LIBRARY_PATH and the NVIDIA driver"
        )

    dtype = np.dtype("float64")
    cfg = DoubleGyreConfig()
    host_state = initial_state(args.nx, args.ny, dtype, cfg)
    particles = host_state[0].size
    selected_kernel = (
        rk4_chunk_strict_kernel
        if args.mode == "strict"
        else rk4_chunk_mixed_kernel
    )

    print("Double-Gyre FTLE CUDA optimized")
    print(
        f"mode={args.mode}, grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}"
    )
    print(
        f"dt={args.dt}, tau={actual_tau}, state_dtype={dtype.name}, "
        f"threads={args.threads}, chunk_steps={args.chunk_steps}"
    )
    print(
        f"warmup_runs={args.warmup_runs}, repeats={args.repeats}, "
        f"verify={args.verify}"
    )
    if args.mode == "mixed" and not args.verify:
        print(
            "WARNING: mixed mode is not being checked against strict FP64; "
            "use --verify for a new configuration"
        )

    print("Compiling CUDA kernels...")
    resources = compile_warmup(dtype, cfg, args.threads)
    selected_resources = resources[args.mode]
    print(
        "selected kernel resources: "
        f"registers/thread={selected_resources['actual_registers_per_thread']}, "
        "local_bytes/thread="
        f"{selected_resources['local_memory_bytes_per_thread']}"
    )

    print("Preparing time-coefficient tables...")
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
        run_once(
            selected_kernel,
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

    runs: list[dict[str, float | int]] = []
    final_ftle: np.ndarray | None = None
    final_state: tuple[np.ndarray, ...] | None = None
    print("Running measured repetitions...")
    for repeat in range(args.repeats):
        ftle, run_metrics, state = run_once(
            selected_kernel,
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
        runs.append(run_metrics)
        final_ftle = ftle
        if state is not None:
            final_state = state
        print(
            f"repeat={repeat + 1}: "
            f"integration={float(run_metrics['integration_seconds']):.6f}s, "
            f"end_to_end={float(run_metrics['end_to_end_seconds']):.6f}s"
        )

    if final_ftle is None or final_state is None:
        raise RuntimeError("measured run did not produce final output")

    performance = summarize_runs(runs, particles, total_steps)
    invariants = state_metrics(final_state, cfg)
    ftle_statistics = {
        "finite": bool(np.all(np.isfinite(final_ftle))),
        "minimum": float(np.min(final_ftle)),
        "maximum": float(np.max(final_ftle)),
        "mean": float(np.mean(final_ftle)),
    }
    invariants_passed = bool(
        invariants["finite"]
        and ftle_statistics["finite"]
        and invariants["max_boundary_error"] == 0.0
        and invariants["max_abs_detF_minus_1"] <= args.det_tolerance
    )

    verification: dict[str, Any] = {
        "performed": False,
        "reference_mode": "strict",
        "passed": None,
    }
    if args.mode == "mixed" and args.verify:
        print("Running strict-FP64 verification reference...")
        strict_ftle, strict_run_metrics, _ = run_once(
            rk4_chunk_strict_kernel,
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
        errors = error_metrics(strict_ftle, final_ftle)
        verification = {
            "performed": True,
            "reference_mode": "strict",
            "reference_run_metrics": strict_run_metrics,
            **errors,
            "relative_l2_passed": bool(
                errors["relative_l2_ftle_error"]
                <= args.relative_l2_tolerance
            ),
            "max_abs_passed": bool(
                errors["max_abs_ftle_error"] <= args.max_abs_tolerance
            ),
        }
        verification["passed"] = bool(
            verification["relative_l2_passed"]
            and verification["max_abs_passed"]
            and invariants_passed
        )
    elif args.mode == "strict":
        verification["reason"] = "strict mode is the reference"

    save_seconds = 0.0
    if not args.no_save:
        save_seconds = save_npz(
            args.output,
            args.nx,
            args.ny,
            cfg,
            final_ftle,
            final_state,
        )

    report = {
        "configuration": {
            "mode": args.mode,
            "nx": args.nx,
            "ny": args.ny,
            "particles": particles,
            "steps": total_steps,
            "dt": args.dt,
            "tau": actual_tau,
            "state_dtype": dtype.name,
            "trig_dtype": "float64" if args.mode == "strict" else "float32",
            "threads": args.threads,
            "chunk_steps": args.chunk_steps,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "relative_l2_error_tolerance": args.relative_l2_tolerance,
            "max_absolute_error_tolerance": args.max_abs_tolerance,
            "max_abs_detF_minus_1_tolerance": args.det_tolerance,
        },
        "compiled_resources": resources,
        "coefficient_setup": coefficient_metrics,
        "performance": performance,
        "run_metrics": runs,
        "ftle_statistics": ftle_statistics,
        "invariants": {**invariants, "passed": invariants_passed},
        "verification": verification,
        "output": {
            "npz_saved": not args.no_save,
            "npz_path": None if args.no_save else str(args.output.resolve()),
            "npz_save_seconds": float(save_seconds),
        },
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFinal summary")
    print(f"mode                         = {args.mode}")
    print(
        "integration median           = "
        f"{performance['integration_seconds_median']:.6f} s"
    )
    print(
        "end-to-end median            = "
        f"{performance['end_to_end_seconds_median']:.6f} s"
    )
    print(
        "particle-steps/s             = "
        f"{performance['particle_steps_per_second']:.3f}"
    )
    print(
        "max |det(F)-1|               = "
        f"{invariants['max_abs_detF_minus_1']:.3e}"
    )
    print(
        "FTLE min/max/mean            = "
        f"{ftle_statistics['minimum']:.6g} / "
        f"{ftle_statistics['maximum']:.6g} / "
        f"{ftle_statistics['mean']:.6g}"
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
