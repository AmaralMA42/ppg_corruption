"""Validation gates for the adaptive phase-map driver.

The quick gate is a software-equivalence test at L=100; it is not a physical
convergence claim.  Scientific phase maps should use the reference L=200 (or
an explicitly validated larger size) and the adaptive presets.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import numpy as np

from adaptive_phase import EffortTier
from run_phase_adaptive import (
    AdaptivePointCache,
    PhysicalPointEvaluator,
    _extract_point_result,
    _run_batches_compat,
    _stable_seed,
    phase_reference_config,
)


def direct_result(cfg, r, sigma, alpha, effort):
    seed = _stable_seed(cfg.seed, {"r": r, "sigma": sigma, "alpha": alpha})
    local = replace(cfg, r=r, sigma=sigma, alpha=alpha,
                    amostras=effort.samples, total_passos=effort.steps, seed=seed)
    batch = _run_batches_compat(local)
    return _extract_point_result(batch, local, r, alpha, "r", "alpha", effort, seed)


def quick_wrapper_gate(atol=1e-12):
    # Keep physical L, but short trajectories because this gate tests wrapper
    # identity only. Never use its observables as science.
    cfg = replace(phase_reference_config(), L=100)
    effort = EffortTier(1, 80)
    r, sigma, alpha = 3.3, 1.05, 0.35
    direct = direct_result(cfg, r, sigma, alpha, effort)
    with tempfile.TemporaryDirectory(prefix="cg_gate_") as tmp:
        base = replace(cfg, sigma=sigma)
        wrapped = PhysicalPointEvaluator(base, "r", "alpha", AdaptivePointCache(Path(tmp)))(r, alpha, effort)
    diffs = [
        np.max(np.abs(direct.mean - wrapped.mean)),
        np.max(np.abs(direct.variance - wrapped.variance)),
        np.max(np.abs(direct.drift - wrapped.drift)),
    ]
    return max(map(float, diffs)) <= atol, max(map(float, diffs))


def main():
    ok, diff = quick_wrapper_gate()
    print(f"quick wrapper gate: {'PASS' if ok else 'FAIL'} maxdiff={diff:.3e}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
