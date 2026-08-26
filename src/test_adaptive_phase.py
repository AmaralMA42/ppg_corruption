"""Fast tests for the adaptive phase-space engine."""

import unittest
import numpy as np

from adaptive_phase import AdaptiveConfig, EffortTier, PointResult, adaptive_phase_sweep, choose_next_effort
from run_phase_adaptive import _stable_seed


def synthetic_evaluator(x, y, effort):
    boundary = 3.45 + 0.18 * np.sin(8.0 * y)
    z = 1.0 / (1.0 + np.exp(-(x - boundary) / 0.035))
    p = np.exp(-((x - (3.25 + 0.15 * y)) / 0.11) ** 2)
    p *= np.exp(-((y - 0.42) / 0.18) ** 2)
    p = min(0.75, 0.65 * p)
    mean = np.array([(1-z)*(1-p), z*(1-p), p], dtype=float)
    mean /= mean.sum()
    variance = np.full(3, 0.001 + 0.012*np.exp(-((x-boundary)/0.08)**2))
    return PointResult(x, y, "r", "alpha", mean, np.full(3, 0.01), variance, np.full(3, 0.001), effort, 123)


class AdaptivePhaseTests(unittest.TestCase):
    def setUp(self):
        self.acfg = AdaptiveConfig(
            initial_nx=5, initial_ny=5, max_depth=3, max_points=450,
            pilot=EffortTier(2, 20), confirm=EffortTier(3, 30), critical=EffortTier(4, 40),
            sem_tol=0.03, critical_sem_tol=0.02,
            composition_range_tol=0.18, center_error_tol=0.05,
            variance_tol=0.005, use_historical_prior=False,
        )

    def test_seed_is_order_independent(self):
        p = {"r": 3.3, "alpha": 0.4, "sigma": 1.05}
        self.assertEqual(_stable_seed(12345, p), _stable_seed(12345, dict(reversed(list(p.items())))))

    def test_budget_and_local_refinement(self):
        points, cells, _ = adaptive_phase_sweep(
            "r", (2.8, 4.0), "alpha", (0.0, 0.7), synthetic_evaluator,
            acfg=self.acfg, verbose=False,
        )
        self.assertLessEqual(len(points), self.acfg.max_points)
        self.assertEqual(max(c.depth for c in cells), self.acfg.max_depth)
        near = sum(abs(p.x - (3.45 + 0.18*np.sin(8*p.y))) < 0.14 for p in points.values())
        self.assertGreater(near, 20)
        # Uniform depth-3 would use 2113 vertices+centers for this starting grid.
        self.assertLess(len(points), 0.25 * 2113)

    def test_statistical_effort_is_separated(self):
        base = dict(
            x=3.3, y=0.4, param_x="r", param_y="alpha",
            mean=np.array([0.3, 0.3, 0.4]), variance=np.zeros(3),
            variance_sem=np.zeros(3), effort=self.acfg.pilot, base_seed=1,
        )
        noisy = PointResult(sem=np.full(3, 0.08), drift=np.zeros(3), **base)
        e1 = choose_next_effort(noisy, self.acfg)
        self.assertGreater(e1.samples, self.acfg.pilot.samples)
        self.assertEqual(e1.steps, self.acfg.pilot.steps)

        drifting = PointResult(sem=np.zeros(3), drift=np.full(3, 0.09), **base)
        e2 = choose_next_effort(drifting, self.acfg)
        self.assertEqual(e2.samples, self.acfg.pilot.samples)
        self.assertGreater(e2.steps, self.acfg.pilot.steps)


if __name__ == "__main__":
    unittest.main()
