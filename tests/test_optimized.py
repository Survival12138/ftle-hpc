# -*- coding: utf-8 -*-
"""GPU regression tests for the standalone optimized Double-Gyre solver.

Run from the repository root:

    python tests/test_optimized.py -v

The suite uses a small grid so CUDA JIT compilation dominates its wall time.
It skips cleanly when CUDA is unavailable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import double_gyre_ftle_cuda_optimized as optimized  # noqa: E402
from double_gyre_ftle_baseline import run_reference  # noqa: E402
from numba import cuda  # noqa: E402


class OptimizedCudaRegressionTests(unittest.TestCase):
    """Compare CPU, strict CUDA, and mixed CUDA on one deterministic case."""

    @classmethod
    def setUpClass(cls) -> None:
        if not cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

        cls.nx = 41
        cls.ny = 21
        cls.dt = 0.02
        cls.tau = 1.0
        cls.threads = 64
        cls.chunk_steps = 10
        cls.dtype = np.dtype("float64")
        cls.cfg = optimized.DoubleGyreConfig()
        cls.total_steps = int(round(cls.tau / cls.dt))
        cls.host_state = optimized.initial_state(
            cls.nx,
            cls.ny,
            cls.dtype,
            cls.cfg,
        )

        cls.resources = optimized.compile_warmup(
            cls.dtype,
            cls.cfg,
            cls.threads,
        )
        cls.d_a, cls.d_b, cls.coefficient_metrics = (
            optimized.prepare_coefficients(
                cls.total_steps,
                cls.dt,
                cls.chunk_steps,
                cls.threads,
                cls.dtype,
                cls.cfg,
            )
        )

        cls.strict_ftle, cls.strict_run, cls.strict_state = optimized.run_once(
            optimized.rk4_chunk_strict_kernel,
            cls.host_state,
            cls.d_a,
            cls.d_b,
            cls.dt,
            cls.tau,
            cls.chunk_steps,
            cls.threads,
            cls.dtype,
            cls.cfg,
            copy_state=True,
        )
        cls.mixed_ftle, cls.mixed_run, cls.mixed_state = optimized.run_once(
            optimized.rk4_chunk_mixed_kernel,
            cls.host_state,
            cls.d_a,
            cls.d_b,
            cls.dt,
            cls.tau,
            cls.chunk_steps,
            cls.threads,
            cls.dtype,
            cls.cfg,
            copy_state=True,
        )

        _, _, cpu_snapshots, cls.cpu_metrics = run_reference(
            nx=cls.nx,
            ny=cls.ny,
            dt=cls.dt,
            taus=[cls.tau],
            dtype=cls.dtype,
            cfg=cls.cfg,
        )
        cls.cpu_ftle = cpu_snapshots[cls.tau].ravel()

    def test_output_shape_and_finiteness(self) -> None:
        expected_shape = (self.nx * self.ny,)
        self.assertEqual(self.strict_ftle.shape, expected_shape)
        self.assertEqual(self.mixed_ftle.shape, expected_shape)
        self.assertTrue(np.all(np.isfinite(self.strict_ftle)))
        self.assertTrue(np.all(np.isfinite(self.mixed_ftle)))

    def test_strict_cuda_matches_cpu_reference(self) -> None:
        errors = optimized.error_metrics(self.cpu_ftle, self.strict_ftle)
        self.assertLessEqual(errors["relative_l2_ftle_error"], 1e-10)
        self.assertLessEqual(errors["max_abs_ftle_error"], 1e-10)

    def test_mixed_cuda_stays_within_project_tolerances(self) -> None:
        errors = optimized.error_metrics(self.strict_ftle, self.mixed_ftle)
        self.assertLessEqual(errors["relative_l2_ftle_error"], 1e-5)
        self.assertLessEqual(errors["max_abs_ftle_error"], 1e-4)

    def test_state_invariants(self) -> None:
        for state in (self.strict_state, self.mixed_state):
            metrics = optimized.state_metrics(state, self.cfg)
            self.assertTrue(metrics["finite"])
            self.assertEqual(metrics["max_boundary_error"], 0.0)
            self.assertLessEqual(metrics["max_abs_detF_minus_1"], 1e-7)

    def test_kernel_launch_count_and_resources(self) -> None:
        expected_launches = (
            self.total_steps + self.chunk_steps - 1
        ) // self.chunk_steps
        self.assertEqual(
            self.strict_run["rk4_kernel_launches"], expected_launches
        )
        self.assertEqual(
            self.mixed_run["rk4_kernel_launches"], expected_launches
        )
        for mode in ("strict", "mixed"):
            self.assertIn("actual_registers_per_thread", self.resources[mode])
            self.assertIn("local_memory_bytes_per_thread", self.resources[mode])

    def test_coefficient_table_size(self) -> None:
        self.assertEqual(
            self.coefficient_metrics["coefficient_entries"],
            3 * self.total_steps,
        )
        self.assertEqual(
            self.coefficient_metrics["coefficient_bytes"],
            2 * 3 * self.total_steps * self.dtype.itemsize,
        )


if __name__ == "__main__":
    unittest.main()
