# -*- coding: utf-8 -*-
"""CPU-only structural tests for the multi-GPU decomposition driver."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import double_gyre_ftle_multi_gpu as multi_gpu  # noqa: E402


class MultiGpuStructureTests(unittest.TestCase):
    def test_row_partition_is_balanced_and_complete(self) -> None:
        ranges = multi_gpu.partition_rows(11, 4)
        self.assertEqual(ranges, [(0, 3), (3, 6), (6, 9), (9, 11)])
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 11)
        self.assertTrue(
            all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        )
        sizes = [stop - start for start, stop in ranges]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_sharded_initial_state_matches_single_gpu_layout(self) -> None:
        nx, ny = 9, 7
        dtype = np.dtype("float64")
        cfg = SimpleNamespace(x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0)
        x_grid = np.linspace(cfg.x_min, cfg.x_max, nx, dtype=dtype)
        y_grid = np.linspace(cfg.y_min, cfg.y_max, ny, dtype=dtype)
        x0, y0 = np.meshgrid(x_grid, y_grid)
        reference = (
            np.ascontiguousarray(x0.ravel()),
            np.ascontiguousarray(y0.ravel()),
            np.ones(nx * ny, dtype=dtype),
            np.zeros(nx * ny, dtype=dtype),
            np.zeros(nx * ny, dtype=dtype),
            np.ones(nx * ny, dtype=dtype),
        )
        shards = [
            multi_gpu.initial_state_rows(nx, ny, start, stop, dtype, cfg)
            for start, stop in multi_gpu.partition_rows(ny, 3)
        ]
        reconstructed = tuple(
            np.concatenate([shard[field] for shard in shards])
            for field in range(6)
        )
        for expected, actual in zip(reference, reconstructed):
            np.testing.assert_array_equal(expected, actual)

    def test_global_verification_uses_sufficient_statistics(self) -> None:
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        candidate = np.array([1.0, 2.1, 2.9, 4.2])
        rank_results = []
        for rank, indices in enumerate((slice(0, 2), slice(2, 4))):
            rank_results.append(
                {
                    "rank": rank,
                    "verification_statistics": (
                        multi_gpu._verification_sufficient_statistics(
                            reference[indices], candidate[indices]
                        )
                    ),
                }
            )
        metrics = multi_gpu.aggregate_verification(
            rank_results,
            relative_l2_tolerance=1.0,
            max_abs_tolerance=1.0,
        )
        difference = candidate - reference
        expected_relative_l2 = np.linalg.norm(difference) / np.linalg.norm(
            reference
        )
        expected_max_abs = np.max(np.abs(difference))
        expected_mean_abs = np.mean(np.abs(difference))
        self.assertAlmostEqual(
            metrics["relative_l2_ftle_error"],
            expected_relative_l2,
        )
        self.assertAlmostEqual(
            metrics["max_abs_ftle_error"], expected_max_abs
        )
        self.assertAlmostEqual(
            metrics["mean_abs_ftle_error"], expected_mean_abs
        )

    def test_device_parser_rejects_duplicates(self) -> None:
        self.assertEqual(multi_gpu.parse_devices("0,2,1"), [0, 2, 1])
        with self.assertRaises(Exception):
            multi_gpu.parse_devices("0,0")


if __name__ == "__main__":
    unittest.main()
