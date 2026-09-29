#!/usr/bin/env python3
"""Measure CUDA peer-to-peer bandwidth between two GPUs.

This is the environment gate for the communication-aware FTLE path.  It uses
the CUDA Runtime API directly through ``cuda-bindings`` and therefore measures
device-to-device peer copies without staging through NumPy or host memory.

Example
-------
python bench_p2p_nvlink.py \
  --devices 0,1 \
  --sizes-mib 4,64,256,1024 \
  --target-gib 16 \
  --output results/a800_p2p_nvlink.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

from cuda.bindings import runtime


def _error_name(error: Any) -> str:
    try:
        result = runtime.cudaGetErrorName(error)
        value = result[1] if isinstance(result, tuple) else result
        return value.decode() if isinstance(value, bytes) else str(value)
    except Exception:
        return str(error)


def _check(result: Any, operation: str) -> Any:
    """Validate the first cuda-bindings return value and unwrap payloads."""
    values = result if isinstance(result, tuple) else (result,)
    if not values:
        raise RuntimeError(f"{operation} returned no status")
    error = values[0]
    if error != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {_error_name(error)}")
    payload = values[1:]
    if not payload:
        return None
    return payload[0] if len(payload) == 1 else payload


def _parse_devices(text: str) -> tuple[int, int]:
    try:
        devices = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--devices must contain two integers") from exc
    if len(devices) != 2 or devices[0] == devices[1] or min(devices) < 0:
        raise argparse.ArgumentTypeError(
            "--devices must contain two distinct non-negative IDs"
        )
    return devices


def _parse_sizes(text: str) -> list[int]:
    try:
        sizes = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--sizes-mib must contain integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("all transfer sizes must be positive")
    return sizes


def _device_name(device: int) -> str:
    properties = _check(
        runtime.cudaGetDeviceProperties(device), "cudaGetDeviceProperties"
    )
    name = properties.name
    if isinstance(name, bytes):
        return name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(name)


def _enable_peer(current: int, peer: int) -> None:
    _check(runtime.cudaSetDevice(current), "cudaSetDevice")
    result = runtime.cudaDeviceEnablePeerAccess(peer, 0)
    error = result[0] if isinstance(result, tuple) else result
    if error not in (
        runtime.cudaError_t.cudaSuccess,
        runtime.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
    ):
        raise RuntimeError(
            f"cudaDeviceEnablePeerAccess({current}->{peer}) failed: "
            f"{_error_name(error)}"
        )
    if error == runtime.cudaError_t.cudaErrorPeerAccessAlreadyEnabled:
        # Clear the sticky runtime error before subsequent API calls.
        runtime.cudaGetLastError()


def _allocate(device: int, nbytes: int) -> Any:
    _check(runtime.cudaSetDevice(device), "cudaSetDevice")
    return _check(runtime.cudaMalloc(nbytes), "cudaMalloc")


def _free(device: int, pointer: Any) -> None:
    _check(runtime.cudaSetDevice(device), "cudaSetDevice")
    _check(runtime.cudaFree(pointer), "cudaFree")


def _measure_direction(
    src_device: int,
    dst_device: int,
    src_pointer: Any,
    dst_pointer: Any,
    nbytes: int,
    repeats: int,
) -> dict[str, float | int | str]:
    _check(runtime.cudaSetDevice(dst_device), "cudaSetDevice")
    stream = _check(runtime.cudaStreamCreate(), "cudaStreamCreate")
    start = _check(runtime.cudaEventCreate(), "cudaEventCreate")
    stop = _check(runtime.cudaEventCreate(), "cudaEventCreate")
    try:
        for _ in range(3):
            _check(
                runtime.cudaMemcpyPeerAsync(
                    dst_pointer,
                    dst_device,
                    src_pointer,
                    src_device,
                    nbytes,
                    stream,
                ),
                "cudaMemcpyPeerAsync warmup",
            )
        _check(runtime.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

        _check(runtime.cudaEventRecord(start, stream), "cudaEventRecord(start)")
        for _ in range(repeats):
            _check(
                runtime.cudaMemcpyPeerAsync(
                    dst_pointer,
                    dst_device,
                    src_pointer,
                    src_device,
                    nbytes,
                    stream,
                ),
                "cudaMemcpyPeerAsync",
            )
        _check(runtime.cudaEventRecord(stop, stream), "cudaEventRecord(stop)")
        _check(runtime.cudaEventSynchronize(stop), "cudaEventSynchronize")
        elapsed_ms = float(
            _check(runtime.cudaEventElapsedTime(start, stop), "cudaEventElapsedTime")
        )
    finally:
        _check(runtime.cudaEventDestroy(start), "cudaEventDestroy(start)")
        _check(runtime.cudaEventDestroy(stop), "cudaEventDestroy(stop)")
        _check(runtime.cudaStreamDestroy(stream), "cudaStreamDestroy")

    elapsed_seconds = elapsed_ms / 1000.0
    transferred_bytes = nbytes * repeats
    return {
        "direction": f"{src_device}->{dst_device}",
        "bytes_per_transfer": int(nbytes),
        "repeats": int(repeats),
        "elapsed_seconds": elapsed_seconds,
        "bandwidth_GBps": float(transferred_bytes / elapsed_seconds / 1e9),
    }


def _verify_copy(
    src_device: int,
    dst_device: int,
    src_pointer: Any,
    dst_pointer: Any,
    nbytes: int,
) -> bool:
    sample_bytes = min(nbytes, 4096)
    host_pointer = _check(runtime.cudaMallocHost(sample_bytes), "cudaMallocHost")
    _check(runtime.cudaSetDevice(src_device), "cudaSetDevice")
    try:
        _check(runtime.cudaMemset(src_pointer, 0xA5, nbytes), "cudaMemset(src)")
        _check(runtime.cudaSetDevice(dst_device), "cudaSetDevice")
        _check(runtime.cudaMemset(dst_pointer, 0, nbytes), "cudaMemset(dst)")
        _check(
            runtime.cudaMemcpyPeer(
                dst_pointer, dst_device, src_pointer, src_device, nbytes
            ),
            "cudaMemcpyPeer verification",
        )
        _check(
            runtime.cudaMemcpy(
                host_pointer,
                dst_pointer,
                sample_bytes,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy D2H verification",
        )
        host_bytes = (ctypes.c_ubyte * sample_bytes).from_address(
            int(host_pointer)
        )
        return all(value == 0xA5 for value in host_bytes)
    finally:
        _check(runtime.cudaFreeHost(host_pointer), "cudaFreeHost")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=_parse_devices, default=(0, 1))
    parser.add_argument(
        "--sizes-mib", type=_parse_sizes, default=[4, 64, 256, 1024]
    )
    parser.add_argument(
        "--target-gib",
        type=float,
        default=16.0,
        help="approximately how many GiB to copy per measured direction and size",
    )
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("results/p2p_nvlink.json")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.target_gib <= 0.0:
        raise ValueError("--target-gib must be positive")
    if args.min_repeats < 1 or args.max_repeats < args.min_repeats:
        raise ValueError("invalid repeat limits")

    device_count = int(_check(runtime.cudaGetDeviceCount(), "cudaGetDeviceCount"))
    dev0, dev1 = args.devices
    if max(dev0, dev1) >= device_count:
        raise ValueError(
            f"requested devices {args.devices}, but only {device_count} visible"
        )

    can_01 = int(
        _check(runtime.cudaDeviceCanAccessPeer(dev0, dev1), "cudaDeviceCanAccessPeer")
    )
    can_10 = int(
        _check(runtime.cudaDeviceCanAccessPeer(dev1, dev0), "cudaDeviceCanAccessPeer")
    )
    if not (can_01 and can_10):
        raise RuntimeError(
            f"bidirectional P2P is unavailable: {dev0}->{dev1}={can_01}, "
            f"{dev1}->{dev0}={can_10}"
        )

    _enable_peer(dev0, dev1)
    _enable_peer(dev1, dev0)

    print("CUDA P2P / NVLink bandwidth benchmark")
    print(f"device {dev0}: {_device_name(dev0)}")
    print(f"device {dev1}: {_device_name(dev1)}")
    print(f"peer access: {dev0}->{dev1}=OK, {dev1}->{dev0}=OK")

    results: list[dict[str, float | int | str | bool]] = []
    target_bytes = int(args.target_gib * (1024**3))
    for size_mib in args.sizes_mib:
        nbytes = size_mib * 1024**2
        repeats = max(
            args.min_repeats,
            min(args.max_repeats, max(1, target_bytes // nbytes)),
        )
        print(
            f"testing {size_mib} MiB, repeats={repeats}: allocating...",
            flush=True,
        )
        pointer0 = _allocate(dev0, nbytes)
        pointer1 = _allocate(dev1, nbytes)
        try:
            print("  verifying 0->1...", flush=True)
            verified_01 = _verify_copy(dev0, dev1, pointer0, pointer1, nbytes)
            print("  timing 0->1...", flush=True)
            result_01 = _measure_direction(
                dev0, dev1, pointer0, pointer1, nbytes, repeats
            )
            print("  verifying 1->0...", flush=True)
            verified_10 = _verify_copy(dev1, dev0, pointer1, pointer0, nbytes)
            print("  timing 1->0...", flush=True)
            result_10 = _measure_direction(
                dev1, dev0, pointer1, pointer0, nbytes, repeats
            )
        finally:
            _free(dev0, pointer0)
            _free(dev1, pointer1)

        result_01["verified"] = verified_01
        result_10["verified"] = verified_10
        results.extend((result_01, result_10))
        print(
            f"{size_mib:5d} MiB | {dev0}->{dev1}: "
            f"{result_01['bandwidth_GBps']:8.2f} GB/s | {dev1}->{dev0}: "
            f"{result_10['bandwidth_GBps']:8.2f} GB/s | "
            f"verified={verified_01 and verified_10}"
        )

    report = {
        "devices": [
            {"id": dev0, "name": _device_name(dev0)},
            {"id": dev1, "name": _device_name(dev1)},
        ],
        "peer_access": {f"{dev0}->{dev1}": bool(can_01), f"{dev1}->{dev0}": bool(can_10)},
        "configuration": {
            "sizes_mib": args.sizes_mib,
            "target_gib": args.target_gib,
            "min_repeats": args.min_repeats,
            "max_repeats": args.max_repeats,
        },
        "results": results,
        "all_verified": bool(all(result["verified"] for result in results)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved={args.output.resolve()}")
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    if not report["all_verified"]:
        raise RuntimeError("P2P copy verification failed")


if __name__ == "__main__":
    main()
