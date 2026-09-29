#!/usr/bin/env python3
"""Process-per-GPU Double-Gyre flow-map FTLE with CUDA IPC halo exchange.

Each worker owns exactly one CUDA context.  The workers export the allocations
containing their final flow-map coordinates with cudaIpcGetMemHandle, exchange
the 64-byte handles through the parent process, map the peer allocations with
cudaIpcOpenMemHandle, and copy the required boundary row over P2P/NVLink.

This is the stable process-based replacement for the earlier two-thread P2P
prototype.  The communication is synchronous in this version; overlap is a
separate optimization after this correctness baseline passes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import queue
import tempfile
import traceback
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
from cuda.bindings import runtime
from numba import cuda

import double_gyre_ftle_multi_gpu as multi_v1
from double_gyre_ftle_multi_gpu_p2p import (
    _error_metrics,
    _flowmap_ftle_numpy,
    flowmap_ftle_rows_kernel,
)


@cuda.jit
def flowmap_ftle_boundary_kernel(
    x_final,
    y_final,
    previous_x,
    previous_y,
    next_x,
    next_y,
    nx,
    local_rows,
    global_row_start,
    global_ny,
    dx,
    dy,
    tau,
    ftle,
):
    """Compute only the first and last local rows after halo arrival."""
    boundary_index = cuda.grid(1)
    if boundary_index >= 2 * nx:
        return
    boundary_row = boundary_index // nx
    column = boundary_index - boundary_row * nx
    local_row = 0 if boundary_row == 0 else local_rows - 1
    index = local_row * nx + column
    global_row = global_row_start + local_row

    if column == 0:
        right = index + 1
        dxdx = (x_final[right] - x_final[index]) / dx
        dydx = (y_final[right] - y_final[index]) / dx
    elif column == nx - 1:
        left = index - 1
        dxdx = (x_final[index] - x_final[left]) / dx
        dydx = (y_final[index] - y_final[left]) / dx
    else:
        left = index - 1
        right = index + 1
        dxdx = (x_final[right] - x_final[left]) / (2.0 * dx)
        dydx = (y_final[right] - y_final[left]) / (2.0 * dx)

    if global_row == 0:
        above = index + nx
        dxdy = (x_final[above] - x_final[index]) / dy
        dydy = (y_final[above] - y_final[index]) / dy
    elif global_row == global_ny - 1:
        below = index - nx
        dxdy = (x_final[index] - x_final[below]) / dy
        dydy = (y_final[index] - y_final[below]) / dy
    elif local_row == 0:
        above = index + nx
        dxdy = (x_final[above] - previous_x[column]) / (2.0 * dy)
        dydy = (y_final[above] - previous_y[column]) / (2.0 * dy)
    else:
        below = index - nx
        dxdy = (next_x[column] - x_final[below]) / (2.0 * dy)
        dydy = (next_y[column] - y_final[below]) / (2.0 * dy)

    c11 = dxdx * dxdx + dydx * dydx
    c12 = dxdx * dxdy + dydx * dydy
    c22 = dxdy * dxdy + dydy * dydy
    trace = c11 + c22
    discriminant = (c11 - c22) * (c11 - c22) + 4.0 * c12 * c12
    if discriminant < 0.0:
        discriminant = 0.0
    lambda_max = 0.5 * (trace + math.sqrt(discriminant))
    if lambda_max < 1e-30:
        lambda_max = 1e-30
    ftle[index] = 0.5 * math.log(lambda_max) / abs(tau)


def _runtime_check(result: Any, operation: str) -> Any:
    values = result if isinstance(result, tuple) else (result,)
    if not values:
        raise RuntimeError(f"{operation} returned no status")
    error = values[0]
    if error != runtime.cudaError_t.cudaSuccess:
        try:
            name_result = runtime.cudaGetErrorName(error)
            name = name_result[1]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
        except Exception:
            name = str(error)
        raise RuntimeError(f"{operation} failed: {name}")
    payload = values[1:]
    if not payload:
        return None
    return payload[0] if len(payload) == 1 else payload


def _device_pointer(array: Any, byte_offset: int = 0) -> int:
    return int(array.device_ctypes_pointer.value) + int(byte_offset)


def _ipc_handle_bytes(device_pointer: int) -> bytes:
    handle = _runtime_check(
        runtime.cudaIpcGetMemHandle(device_pointer), "cudaIpcGetMemHandle"
    )
    raw = bytes(handle.reserved)
    if len(raw) != 64:
        raise RuntimeError(f"unexpected CUDA IPC handle length: {len(raw)}")
    return raw


def _ipc_handle_from_bytes(raw: bytes) -> Any:
    if len(raw) != 64:
        raise ValueError(f"CUDA IPC handle must be 64 bytes, got {len(raw)}")
    handle = runtime.cudaIpcMemHandle_t()
    try:
        handle.reserved = raw
    except (AttributeError, TypeError, ValueError):
        # cuda-bindings exposes getPtr() on generated structs.  Keep this
        # fallback for versions where the fixed-size byte field is read-only.
        ctypes.memmove(int(handle.getPtr()), raw, len(raw))
    return handle


def _event_seconds(cuda: Any, start: Any, end: Any) -> float:
    return float(cuda.event_elapsed_time(start, end) / 1000.0)


def _integrate_once(
    cuda: Any,
    kernel: Any,
    device_state: tuple[Any, ...],
    d_a: Any,
    d_b: Any,
    compute_stream: Any,
    particles: int,
    args: argparse.Namespace,
    total_steps: int,
    cfg: Any,
) -> float:
    blocks = (particles + args.threads - 1) // args.threads
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record(compute_stream)
    completed_steps = 0
    while completed_steps < total_steps:
        current_chunk = min(args.chunk_steps, total_steps - completed_steps)
        kernel[blocks, args.threads, compute_stream](
            *device_state,
            d_a,
            d_b,
            completed_steps,
            np.float64(args.dt),
            current_chunk,
            np.float64(cfg.amplitude),
            np.float64(cfg.x_min),
            np.float64(cfg.x_max),
            np.float64(cfg.y_min),
            np.float64(cfg.y_max),
        )
        completed_steps += current_chunk
    end.record(compute_stream)
    end.synchronize()
    return _event_seconds(cuda, start, end)


def _exchange_mapped_halo(
    cuda: Any,
    rank: int,
    nx: int,
    local_device_state: tuple[Any, ...],
    previous_x: Any,
    previous_y: Any,
    next_x: Any,
    next_y: Any,
    peer_x_pointer: int,
    peer_y_pointer: int,
    peer_local_rows: int,
    communication_stream: Any,
) -> tuple[float, int]:
    row_bytes = nx * np.dtype(np.float64).itemsize
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record(communication_stream)

    if rank == 0:
        source_offset = 0
        destinations = (next_x, next_y)
    else:
        source_offset = (peer_local_rows - 1) * row_bytes
        destinations = (previous_x, previous_y)

    for destination, peer_base in zip(
        destinations, (peer_x_pointer, peer_y_pointer)
    ):
        _runtime_check(
            runtime.cudaMemcpyAsync(
                _device_pointer(destination),
                int(peer_base) + source_offset,
                row_bytes,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                communication_stream.handle,
            ),
            "cudaMemcpyAsync(IPC-mapped peer halo)",
        )

    end.record(communication_stream)
    end.synchronize()
    return _event_seconds(cuda, start, end), 2 * row_bytes


def _exchange_host_staged(
    rank: int,
    nx: int,
    local_rows: int,
    device_state: tuple[Any, ...],
    previous_x: Any,
    previous_y: Any,
    next_x: Any,
    next_y: Any,
    shared_host_halo: Any,
    barrier: Any,
) -> tuple[float, int]:
    """Exchange x/y boundary rows through pageable shared host memory."""
    host_halo = np.frombuffer(shared_host_halo, dtype=np.float64).reshape(
        2, 2, nx
    )
    own_offset = (local_rows - 1) * nx if rank == 0 else 0
    start = perf_counter()
    device_state[0][own_offset : own_offset + nx].copy_to_host(
        host_halo[rank, 0]
    )
    device_state[1][own_offset : own_offset + nx].copy_to_host(
        host_halo[rank, 1]
    )
    barrier.wait()

    peer = 1 - rank
    destination_x = next_x if rank == 0 else previous_x
    destination_y = next_y if rank == 0 else previous_y
    destination_x.copy_to_device(host_halo[peer, 0])
    destination_y.copy_to_device(host_halo[peer, 1])
    barrier.wait()
    return float(perf_counter() - start), 2 * nx * np.dtype(np.float64).itemsize


def _launch_mapped_halo_async(
    rank: int,
    nx: int,
    previous_x: Any,
    previous_y: Any,
    next_x: Any,
    next_y: Any,
    peer_x_pointer: int,
    peer_y_pointer: int,
    peer_local_rows: int,
    communication_stream: Any,
) -> tuple[Any, Any, int]:
    """Enqueue the IPC P2P copy and return timing events without waiting."""
    row_bytes = nx * np.dtype(np.float64).itemsize
    if rank == 0:
        source_offset = 0
        destinations = (next_x, next_y)
    else:
        source_offset = (peer_local_rows - 1) * row_bytes
        destinations = (previous_x, previous_y)

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record(communication_stream)
    for destination, peer_base in zip(
        destinations, (peer_x_pointer, peer_y_pointer)
    ):
        _runtime_check(
            runtime.cudaMemcpyAsync(
                _device_pointer(destination),
                int(peer_base) + source_offset,
                row_bytes,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                communication_stream.handle,
            ),
            "cudaMemcpyAsync(IPC overlap halo)",
        )
    end.record(communication_stream)
    return start, end, 2 * row_bytes


def _compute_flowmap_ftle_overlap(
    cuda_module: Any,
    rank: int,
    x_final: Any,
    y_final: Any,
    previous_x: Any,
    previous_y: Any,
    next_x: Any,
    next_y: Any,
    d_flowmap_ftle: Any,
    compute_stream: Any,
    communication_stream: Any,
    peer_x_pointer: int,
    peer_y_pointer: int,
    peer_local_rows: int,
    row_start: int,
    local_rows: int,
    args: argparse.Namespace,
    cfg: Any,
) -> tuple[float, float, float, int]:
    """Overlap P2P halo arrival with FTLE on rows not using the halo."""
    dx = (cfg.x_max - cfg.x_min) / (args.nx - 1)
    dy = (cfg.y_max - cfg.y_min) / (args.ny - 1)
    phase_start = perf_counter()
    copy_start, copy_end, halo_bytes = _launch_mapped_halo_async(
        rank,
        args.nx,
        previous_x,
        previous_y,
        next_x,
        next_y,
        peer_x_pointer,
        peer_y_pointer,
        peer_local_rows,
        communication_stream,
    )

    interior_rows = local_rows - 2
    interior_start = cuda_module.event(timing=True)
    interior_end = cuda_module.event(timing=True)
    interior_start.record(compute_stream)
    if interior_rows > 0:
        begin = args.nx
        end_index = (local_rows - 1) * args.nx
        blocks = (
            interior_rows * args.nx + args.threads - 1
        ) // args.threads
        # The boundary rows of the original slab become the halo rows of this
        # interior-only view, so no remote halo is read by this launch.
        flowmap_ftle_rows_kernel[blocks, args.threads, compute_stream](
            x_final[begin:end_index],
            y_final[begin:end_index],
            x_final[0 : args.nx],
            y_final[0 : args.nx],
            x_final[end_index : end_index + args.nx],
            y_final[end_index : end_index + args.nx],
            args.nx,
            interior_rows,
            row_start + 1,
            args.ny,
            np.float64(dx),
            np.float64(dy),
            np.float64(args.tau),
            d_flowmap_ftle[begin:end_index],
        )
    interior_end.record(compute_stream)

    copy_end.synchronize()
    halo_seconds = _event_seconds(cuda_module, copy_start, copy_end)

    boundary_start = cuda_module.event(timing=True)
    boundary_end = cuda_module.event(timing=True)
    boundary_start.record(compute_stream)
    boundary_blocks = (2 * args.nx + args.threads - 1) // args.threads
    flowmap_ftle_boundary_kernel[
        boundary_blocks, args.threads, compute_stream
    ](
        x_final,
        y_final,
        previous_x,
        previous_y,
        next_x,
        next_y,
        args.nx,
        local_rows,
        row_start,
        args.ny,
        np.float64(dx),
        np.float64(dy),
        np.float64(args.tau),
        d_flowmap_ftle,
    )
    boundary_end.record(compute_stream)
    boundary_end.synchronize()
    interior_seconds = _event_seconds(
        cuda_module, interior_start, interior_end
    )
    boundary_seconds = _event_seconds(
        cuda_module, boundary_start, boundary_end
    )
    return (
        halo_seconds,
        interior_seconds + boundary_seconds,
        float(perf_counter() - phase_start),
        halo_bytes,
    )


def _compute_flowmap_ftle(
    cuda: Any,
    x_final: Any,
    y_final: Any,
    previous_x: Any,
    previous_y: Any,
    next_x: Any,
    next_y: Any,
    d_flowmap_ftle: Any,
    compute_stream: Any,
    row_start: int,
    local_rows: int,
    args: argparse.Namespace,
    cfg: Any,
) -> float:
    particles = args.nx * local_rows
    blocks = (particles + args.threads - 1) // args.threads
    dx = (cfg.x_max - cfg.x_min) / (args.nx - 1)
    dy = (cfg.y_max - cfg.y_min) / (args.ny - 1)
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record(compute_stream)
    flowmap_ftle_rows_kernel[blocks, args.threads, compute_stream](
        x_final,
        y_final,
        previous_x,
        previous_y,
        next_x,
        next_y,
        args.nx,
        local_rows,
        row_start,
        args.ny,
        np.float64(dx),
        np.float64(dy),
        np.float64(args.tau),
        d_flowmap_ftle,
    )
    end.record(compute_stream)
    end.synchronize()
    return _event_seconds(cuda, start, end)


def _worker(
    rank: int,
    device_id: int,
    row_range: tuple[int, int],
    args: argparse.Namespace,
    total_steps: int,
    barrier: Any,
    handle_queue: Any,
    peer_queue: Any,
    result_queue: Any,
    shared_host_halo: Any,
    shard_directory: str,
) -> None:
    from numba import cuda
    import double_gyre_ftle_cuda_optimized as optimized

    peer_x_pointer = None
    peer_y_pointer = None
    try:
        cuda.select_device(device_id)
        _runtime_check(runtime.cudaSetDevice(device_id), "cudaSetDevice")
        cfg = optimized.DoubleGyreConfig()
        dtype = np.dtype(np.float64)
        resources = optimized.compile_warmup(dtype, cfg, args.threads)
        d_a, d_b, coefficient_setup = optimized.prepare_coefficients(
            total_steps,
            args.dt,
            args.chunk_steps,
            args.threads,
            dtype,
            cfg,
        )

        host_state = multi_v1.initial_state_rows(
            args.nx,
            args.ny,
            row_range[0],
            row_range[1],
            dtype,
            cfg,
        )
        local_rows = row_range[1] - row_range[0]
        if local_rows < 2:
            raise ValueError("each GPU must own at least two rows")
        particles = args.nx * local_rows
        compute_stream = cuda.stream()
        communication_stream = cuda.stream()
        device_state = tuple(
            cuda.to_device(array, stream=compute_stream) for array in host_state
        )
        compute_stream.synchronize()
        previous_x = cuda.device_array(args.nx, dtype=dtype)
        previous_y = cuda.device_array(args.nx, dtype=dtype)
        next_x = cuda.device_array(args.nx, dtype=dtype)
        next_y = cuda.device_array(args.nx, dtype=dtype)
        d_flowmap_ftle = cuda.device_array(particles, dtype=dtype)
        d_direct_ftle = cuda.device_array(particles, dtype=dtype)

        peer_local_rows = 0
        if args.communication != "host_staged":
            handle_queue.put(
                {
                    "status": "handle",
                    "rank": rank,
                    "device_id": device_id,
                    "local_rows": local_rows,
                    "x_handle": _ipc_handle_bytes(
                        _device_pointer(device_state[0])
                    ),
                    "y_handle": _ipc_handle_bytes(
                        _device_pointer(device_state[1])
                    ),
                }
            )
            peer_packet = peer_queue.get(timeout=120.0)
            peer_x_handle = _ipc_handle_from_bytes(peer_packet["x_handle"])
            peer_y_handle = _ipc_handle_from_bytes(peer_packet["y_handle"])
            peer_x_pointer = int(
                _runtime_check(
                    runtime.cudaIpcOpenMemHandle(
                        peer_x_handle, runtime.cudaIpcMemLazyEnablePeerAccess
                    ),
                    "cudaIpcOpenMemHandle(x)",
                )
            )
            peer_y_pointer = int(
                _runtime_check(
                    runtime.cudaIpcOpenMemHandle(
                        peer_y_handle, runtime.cudaIpcMemLazyEnablePeerAccess
                    ),
                    "cudaIpcOpenMemHandle(y)",
                )
            )
            peer_local_rows = int(peer_packet["local_rows"])
        kernel = (
            optimized.rk4_chunk_strict_kernel
            if args.mode == "strict"
            else optimized.rk4_chunk_mixed_kernel
        )

        run_metrics: list[dict[str, float | int]] = []
        total_runs = args.warmup_runs + args.repeats
        barrier.wait()
        for run_index in range(total_runs):
            for device_array, host_array in zip(device_state, host_state):
                device_array.copy_to_device(host_array, stream=compute_stream)
            compute_stream.synchronize()
            barrier.wait()

            wall_start = perf_counter()
            integration_seconds = _integrate_once(
                cuda,
                kernel,
                device_state,
                d_a,
                d_b,
                compute_stream,
                particles,
                args,
                total_steps,
                cfg,
            )
            barrier.wait()
            communication_phase_start = perf_counter()
            if args.communication == "host_staged":
                exchange_seconds, halo_bytes = _exchange_host_staged(
                    rank,
                    args.nx,
                    local_rows,
                    device_state,
                    previous_x,
                    previous_y,
                    next_x,
                    next_y,
                    shared_host_halo,
                    barrier,
                )
                ftle_seconds = _compute_flowmap_ftle(
                    cuda,
                    device_state[0],
                    device_state[1],
                    previous_x,
                    previous_y,
                    next_x,
                    next_y,
                    d_flowmap_ftle,
                    compute_stream,
                    row_range[0],
                    local_rows,
                    args,
                    cfg,
                )
                communication_phase_seconds = (
                    perf_counter() - communication_phase_start
                )
            elif args.communication == "ipc_sync":
                exchange_seconds, halo_bytes = _exchange_mapped_halo(
                    cuda,
                    rank,
                    args.nx,
                    device_state,
                    previous_x,
                    previous_y,
                    next_x,
                    next_y,
                    peer_x_pointer,
                    peer_y_pointer,
                    peer_local_rows,
                    communication_stream,
                )
                barrier.wait()
                ftle_seconds = _compute_flowmap_ftle(
                    cuda,
                    device_state[0],
                    device_state[1],
                    previous_x,
                    previous_y,
                    next_x,
                    next_y,
                    d_flowmap_ftle,
                    compute_stream,
                    row_range[0],
                    local_rows,
                    args,
                    cfg,
                )
                communication_phase_seconds = (
                    perf_counter() - communication_phase_start
                )
            else:
                (
                    exchange_seconds,
                    ftle_seconds,
                    communication_phase_seconds,
                    halo_bytes,
                ) = _compute_flowmap_ftle_overlap(
                    cuda,
                    rank,
                    device_state[0],
                    device_state[1],
                    previous_x,
                    previous_y,
                    next_x,
                    next_y,
                    d_flowmap_ftle,
                    compute_stream,
                    communication_stream,
                    peer_x_pointer,
                    peer_y_pointer,
                    peer_local_rows,
                    row_range[0],
                    local_rows,
                    args,
                    cfg,
                )
            barrier.wait()
            wall_seconds = perf_counter() - wall_start
            if run_index >= args.warmup_runs:
                run_metrics.append(
                    {
                        "integration_seconds": integration_seconds,
                        "halo_exchange_seconds": exchange_seconds,
                        "flowmap_ftle_seconds": ftle_seconds,
                        "communication_ftle_phase_seconds": (
                            communication_phase_seconds
                        ),
                        "compute_wall_seconds": wall_seconds,
                        "halo_bytes_received": halo_bytes,
                    }
                )

        blocks = (particles + args.threads - 1) // args.threads
        optimized.ftle_kernel[blocks, args.threads, compute_stream](
            *device_state[2:], np.float64(args.tau), d_direct_ftle
        )
        compute_stream.synchronize()
        final_state_host = tuple(array.copy_to_host() for array in device_state)
        flowmap_ftle_host = d_flowmap_ftle.copy_to_host()
        direct_ftle_host = d_direct_ftle.copy_to_host()

        shard_path = Path(shard_directory) / f"rank_{rank:04d}.npz"
        np.savez(
            shard_path,
            x_final=final_state_host[0],
            y_final=final_state_host[1],
            ftle_flowmap=flowmap_ftle_host,
            ftle_direct_F=direct_ftle_host,
        )

        barrier.wait()
        if args.communication != "host_staged":
            _runtime_check(
                runtime.cudaIpcCloseMemHandle(peer_x_pointer),
                "cudaIpcCloseMemHandle(x)",
            )
            peer_x_pointer = None
            _runtime_check(
                runtime.cudaIpcCloseMemHandle(peer_y_pointer),
                "cudaIpcCloseMemHandle(y)",
            )
            peer_y_pointer = None
            # Do not let either owner process free its exported allocation
            # until both peer mappings have been closed.
            barrier.wait()

        result_queue.put(
            {
                "status": "ok",
                "rank": rank,
                "device_id": device_id,
                "row_start": row_range[0],
                "row_stop": row_range[1],
                "particles": particles,
                "resources": resources[args.mode],
                "coefficient_setup": coefficient_setup,
                "run_metrics": run_metrics,
                "shard_path": str(shard_path),
            }
        )
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
        for pointer, label in (
            (peer_x_pointer, "x"),
            (peer_y_pointer, "y"),
        ):
            if pointer is not None:
                try:
                    runtime.cudaIpcCloseMemHandle(pointer)
                except BaseException:
                    pass
        try:
            cuda.close()
        except BaseException:
            pass


def _collect_results(processes: list[Any], result_queue: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while len(messages) < len(processes):
        try:
            messages.append(result_queue.get(timeout=1.0))
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break
    for process in processes:
        process.join()

    errors = [message for message in messages if message.get("status") != "ok"]
    bad_exits = [
        (index, process.exitcode)
        for index, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if errors or len(messages) != len(processes) or bad_exits:
        details = [
            f"rank {message.get('rank')}: {message.get('error')}\n"
            f"{message.get('traceback')}"
            for message in errors
        ]
        if len(messages) != len(processes):
            details.append("one or more workers exited without a result")
        if bad_exits:
            details.append(f"non-zero worker exits: {bad_exits}")
        raise RuntimeError("CUDA IPC worker failure:\n" + "\n".join(details))
    return sorted(messages, key=lambda item: int(item["rank"]))


def _parse_devices(text: str) -> tuple[int, int]:
    try:
        devices = tuple(
            int(value.strip()) for value in text.split(",") if value.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--devices must contain integers") from exc
    if len(devices) != 2 or devices[0] == devices[1] or min(devices) < 0:
        raise argparse.ArgumentTypeError(
            "--devices must contain two distinct non-negative IDs"
        )
    return devices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=_parse_devices, default=(0, 1))
    parser.add_argument("--mode", choices=("strict", "mixed"), default="mixed")
    parser.add_argument(
        "--communication",
        choices=("host_staged", "ipc_sync", "ipc_overlap"),
        default="ipc_sync",
    )
    parser.add_argument("--nx", type=int, default=2001)
    parser.add_argument("--ny", type=int, default=1001)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--distributed-max-abs-tolerance", type=float, default=1e-10
    )
    parser.add_argument(
        "--distributed-relative-l2-tolerance", type=float, default=1e-11
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/flowmap_ipc_ftle.npz")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("results/flowmap_ipc_metrics.json")
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.nx < 3 or args.ny < 4:
        raise ValueError("--nx >= 3 and --ny >= 4 are required")
    if args.threads < 32 or args.threads > 1024 or args.threads % 32 != 0:
        raise ValueError("--threads must be a multiple of 32 in [32, 1024]")
    if min(args.chunk_steps, args.warmup_runs, args.repeats) < 1:
        raise ValueError("chunk steps, warmups, and repeats must be positive")
    total_steps = int(round(args.tau / args.dt))
    if not np.isclose(total_steps * args.dt, args.tau, atol=1e-12, rtol=0.0):
        raise ValueError("--tau must be an integer multiple of --dt")

    row_ranges = multi_v1.partition_rows(args.ny, 2)
    particles = args.nx * args.ny
    print("Double-Gyre FTLE multi-GPU communication V2")
    print(f"devices={args.devices}, mode={args.mode}, rows={row_ranges}")
    print(
        f"grid={args.nx}x{args.ny}, particles={particles}, "
        f"steps={total_steps}, dt={args.dt}, tau={args.tau}"
    )
    print(f"communication={args.communication}, halo_width=1 row")

    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    handle_queue = context.Queue()
    peer_queues = [context.Queue(), context.Queue()]
    result_queue = context.Queue()
    shared_host_halo = context.RawArray("d", 4 * args.nx)

    with tempfile.TemporaryDirectory(prefix="ftle_cuda_ipc_") as temp_dir:
        processes = []
        for rank in range(2):
            process = context.Process(
                target=_worker,
                args=(
                    rank,
                    args.devices[rank],
                    row_ranges[rank],
                    args,
                    total_steps,
                    barrier,
                    handle_queue,
                    peer_queues[rank],
                    result_queue,
                    shared_host_halo,
                    temp_dir,
                ),
                name=f"ftle-ipc-gpu-{args.devices[rank]}",
            )
            process.start()
            processes.append(process)

        if args.communication != "host_staged":
            handle_packets: dict[int, dict[str, Any]] = {}
            try:
                while len(handle_packets) < 2:
                    packet = handle_queue.get(timeout=120.0)
                    handle_packets[int(packet["rank"])] = packet
                peer_queues[0].put(handle_packets[1])
                peer_queues[1].put(handle_packets[0])
            except BaseException:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join()
                raise

        rank_results = _collect_results(processes, result_queue)
        shards = []
        for result in rank_results:
            with np.load(result["shard_path"]) as shard:
                shards.append({name: shard[name].copy() for name in shard.files})

        x_final = np.concatenate([shard["x_final"] for shard in shards])
        y_final = np.concatenate([shard["y_final"] for shard in shards])
        flowmap_ftle = np.concatenate(
            [shard["ftle_flowmap"] for shard in shards]
        )
        direct_ftle = np.concatenate(
            [shard["ftle_direct_F"] for shard in shards]
        )

        if not args.no_save:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output,
                x_final=x_final.reshape(args.ny, args.nx),
                y_final=y_final.reshape(args.ny, args.nx),
                ftle_flowmap=flowmap_ftle.reshape(args.ny, args.nx),
                ftle_direct_F=direct_ftle.reshape(args.ny, args.nx),
            )

    cfg = __import__("double_gyre_ftle_cuda_optimized").DoubleGyreConfig()
    dx = (cfg.x_max - cfg.x_min) / (args.nx - 1)
    dy = (cfg.y_max - cfg.y_min) / (args.ny - 1)
    global_reference = _flowmap_ftle_numpy(
        x_final, y_final, args.nx, args.ny, dx, dy, args.tau
    )
    distributed_error = _error_metrics(global_reference, flowmap_ftle)
    discretization_error = _error_metrics(direct_ftle, flowmap_ftle)
    distributed_passed = bool(
        distributed_error["max_absolute_error"]
        <= args.distributed_max_abs_tolerance
        and distributed_error["relative_l2_error"]
        <= args.distributed_relative_l2_tolerance
    )

    rank_reports = []
    for result in rank_results:
        runs = result["run_metrics"]
        rank_reports.append(
            {
                "rank": result["rank"],
                "device_id": result["device_id"],
                "row_range": [result["row_start"], result["row_stop"]],
                "particles": result["particles"],
                "resources": result["resources"],
                "coefficient_setup": result["coefficient_setup"],
                "performance": {
                    "integration_seconds_median": float(
                        median(run["integration_seconds"] for run in runs)
                    ),
                    "halo_exchange_seconds_median": float(
                        median(run["halo_exchange_seconds"] for run in runs)
                    ),
                    "flowmap_ftle_seconds_median": float(
                        median(run["flowmap_ftle_seconds"] for run in runs)
                    ),
                    "communication_ftle_phase_seconds_median": float(
                        median(
                            run["communication_ftle_phase_seconds"]
                            for run in runs
                        )
                    ),
                    "compute_wall_seconds_median": float(
                        median(run["compute_wall_seconds"] for run in runs)
                    ),
                    "halo_bytes_received": int(runs[0]["halo_bytes_received"]),
                },
                "run_metrics": runs,
            }
        )

    distributed_wall = max(
        item["performance"]["compute_wall_seconds_median"]
        for item in rank_reports
    )
    report = {
        "configuration": {
            "devices": list(args.devices),
            "mode": args.mode,
            "decomposition": "contiguous_y_rows",
            "communication": args.communication,
            "halo_width_rows": 1,
            "nx": args.nx,
            "ny": args.ny,
            "particles": particles,
            "steps": total_steps,
            "dt": args.dt,
            "tau": args.tau,
            "threads": args.threads,
            "chunk_steps": args.chunk_steps,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
        },
        "distributed_performance": {
            "compute_wall_seconds_median": distributed_wall,
            "particle_steps_per_second": float(
                particles * total_steps / distributed_wall
            ),
        },
        "rank_results": rank_reports,
        "correctness": {
            "distributed_vs_global_flowmap": distributed_error,
            "distributed_passed": distributed_passed,
            "flowmap_vs_direct_F_discretization": discretization_error,
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMulti-GPU communication summary")
    print(f"communication mode          = {args.communication}")
    print(f"distributed wall median     = {distributed_wall:.6f} s")
    print(
        "aggregate particle-steps/s = "
        f"{report['distributed_performance']['particle_steps_per_second']:.3f}"
    )
    print(
        "distributed vs global max abs = "
        f"{distributed_error['max_absolute_error']:.3e}"
    )
    print(
        "distributed vs global rel L2  = "
        f"{distributed_error['relative_l2_error']:.3e}"
    )
    print(f"distributed correctness     = {distributed_passed}")
    print(
        "flow-map vs direct-F rel L2 = "
        f"{discretization_error['relative_l2_error']:.3e}"
    )
    for item in rank_reports:
        performance = item["performance"]
        print(
            f"rank {item['rank']}: integration="
            f"{performance['integration_seconds_median']:.6f}s, halo="
            f"{performance['halo_exchange_seconds_median'] * 1e6:.3f}us, "
            f"flowmap_ftle={performance['flowmap_ftle_seconds_median']:.6f}s, "
            f"comm+ftle_phase="
            f"{performance['communication_ftle_phase_seconds_median']:.6f}s"
        )
    print(f"metrics saved               = {args.metrics.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    if not distributed_passed:
        raise RuntimeError("distributed CUDA IPC result failed validation")


if __name__ == "__main__":
    main()
