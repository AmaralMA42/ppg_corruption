"""Adaptive phase-diagram driver for the spatial Corruption Game.

Scientific design rules
-----------------------
* The microscopic model stays in ``core_simulation.py``.
* ``L`` is NOT an adaptive variable.  Phase-map runs reject L < 100.
* The supplied project snapshot is used as the reference geometry:
  L=200, random C/D/P initial condition, K=0.1, G=5, c=1,
  sigma=1.05, alpha=0.35 and the final 30% used for steady averages.
* Parameter-space density, sample count and trajectory length may adapt.
* Heuristic "no flips for N steps" early stopping is disabled here until it
  has a separate scientific validation.  True absorbing-state optimization
  can be added later without changing the phase sampler.
* Cache keys include a fingerprint of the simulation kernel, preventing stale
  results from silently surviving code changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from adaptive_phase import AdaptiveConfig, EffortTier, PointResult, adaptive_phase_sweep
from config import SimulationConfig
from run_sampling import run_batches

try:
    from run_sampling import create_sample_pool
except ImportError:
    create_sample_pool = None

MIN_PHYSICAL_L = 100
REFERENCE_L = 200
REFERENCE_AVG_FRACTION = 0.70


def _cfg_field_names() -> set[str]:
    return {f.name for f in fields(SimulationConfig)}


def phase_reference_config() -> SimulationConfig:
    """Return the project phase-map reference configuration when supported."""
    cfg = SimulationConfig()
    available = _cfg_field_names()
    target = {
        "L": REFERENCE_L,
        "percent_avg_MC": REFERENCE_AVG_FRACTION,
        "cond_ini": 0,
        "k": 0.1,
        "G": 5,
        "c": 1.0,
        "r": 3.3,
        "sigma": 1.05,
        "alpha": 0.35,
        # The existing zero-activity-window stop is a heuristic, not a proof
        # of absorption. Keep it off for scientific phase maps.
        "absorbing_window": 0,
        "make_plots": False,
        "compute_time_analysis": False,
    }
    updates = {k: v for k, v in target.items() if k in available}
    return replace(cfg, **updates)


def validate_physical_scale(cfg: SimulationConfig) -> None:
    if int(cfg.L) < MIN_PHYSICAL_L:
        raise ValueError(
            f"Adaptive phase maps require L >= {MIN_PHYSICAL_L}; got L={cfg.L}. "
            "Small-L runs are allowed only in synthetic/software tests, never "
            "as scientific scouts for this project."
        )


def kernel_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("core_simulation.py", "run_sampling.py", "config.py"):
        path = root / name
        h.update(name.encode())
        if path.exists():
            h.update(path.read_bytes())
        else:
            h.update(b"<missing>")
    return h.hexdigest()[:20]


class AdaptivePointCache:
    """Persistent cache keyed by physics, effort and simulation code."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(payload: Mapping) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def path_for(self, payload: Mapping) -> Path:
        return self.root / f"{self._hash(payload)}.json"

    def load(self, payload: Mapping) -> PointResult | None:
        path = self.path_for(payload)
        if not path.exists():
            return None
        return PointResult.from_jsonable(json.loads(path.read_text(encoding="utf-8")))

    def save(self, payload: Mapping, result: PointResult) -> None:
        path = self.path_for(payload)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result.to_jsonable(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


def _stable_seed(base_seed: int | None, params: Mapping[str, float]) -> int:
    seed0 = 0 if base_seed is None else int(base_seed)
    payload = {"seed": seed0, **{k: float(v) for k, v in sorted(params.items())}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _sem(values: np.ndarray, axis: int) -> np.ndarray:
    n = values.shape[axis]
    if n <= 1:
        shape = list(values.shape)
        del shape[axis]
        return np.zeros(shape, dtype=float)
    return np.std(values, axis=axis, ddof=1) / math.sqrt(n)


def _run_batches_compat(cfg: SimulationConfig, pool=None):
    sig = inspect.signature(run_batches)
    kwargs = {"base_seed": cfg.seed}
    if "absorbing_window" in sig.parameters:
        # Scientific default: do not equate a quiet window with absorption.
        kwargs["absorbing_window"] = 0
    if "pool" in sig.parameters and pool is not None:
        kwargs["pool"] = pool
    return run_batches(cfg.L, cfg.amostras, cfg.total_passos, cfg.simulation_params(), **kwargs)


def _extract_point_result(batch_result, cfg, x, y, param_x, param_y, effort, seed):
    estrat_t = np.asarray(batch_result[0], dtype=float)
    tail = estrat_t[:, :, cfg.passos_media:]
    if tail.shape[2] < 2:
        raise ValueError("steady-state window too short; increase total MCS")
    sample_mean = np.mean(tail, axis=2)
    sample_var = np.var(tail, axis=2, ddof=1)
    mean = np.mean(sample_mean, axis=1)
    sem = _sem(sample_mean, axis=1)
    variance = np.mean(sample_var, axis=1)
    variance_sem = _sem(sample_var, axis=1)

    half = max(1, tail.shape[2] // 2)
    first = np.mean(tail[:, :, :half], axis=2)
    second = np.mean(tail[:, :, half:], axis=2) if half < tail.shape[2] else first
    drift = np.mean(np.abs(second - first), axis=1)

    absorbed_fraction = math.nan
    if len(batch_result) >= 7:
        absorbed_at = np.asarray(batch_result[6])
        absorbed_fraction = float(np.mean(absorbed_at < cfg.total_passos))

    return PointResult(
        x=x, y=y, param_x=param_x, param_y=param_y,
        mean=mean, sem=sem, variance=variance, variance_sem=variance_sem,
        effort=effort, base_seed=seed, absorbed_fraction=absorbed_fraction,
        drift=drift,
    )


class PhysicalPointEvaluator:
    def __init__(self, base_cfg, param_x, param_y, cache, pool=None):
        if param_x not in {"r", "alpha", "sigma"} or param_y not in {"r", "alpha", "sigma"}:
            raise ValueError("phase parameters must be r, alpha or sigma")
        validate_physical_scale(base_cfg)
        self.base_cfg = base_cfg
        self.param_x = param_x
        self.param_y = param_y
        self.cache = cache
        self.pool = pool
        self.kernel_id = kernel_fingerprint()

    def _payload(self, x, y, effort):
        vals = {
            "r": float(self.base_cfg.r),
            "alpha": float(self.base_cfg.alpha),
            "sigma": float(self.base_cfg.sigma),
        }
        vals[self.param_x] = float(x)
        vals[self.param_y] = float(y)
        return {
            "model": "corruption_game_spatial_square_fermi",
            "kernel_fingerprint": self.kernel_id,
            **vals,
            "L": int(self.base_cfg.L),
            "G": int(self.base_cfg.G),
            "c": float(self.base_cfg.c),
            "k": float(self.base_cfg.k),
            "cond_ini": int(self.base_cfg.cond_ini),
            "percent_avg_MC": float(self.base_cfg.percent_avg_MC),
            "absorbing_window": 0,
            "samples": int(effort.samples),
            "steps": int(effort.steps),
            "seed_root": self.base_cfg.seed,
        }

    def __call__(self, x: float, y: float, effort: EffortTier) -> PointResult:
        payload = self._payload(x, y, effort)
        cached = self.cache.load(payload)
        if cached is not None:
            return cached

        point_params = {k: payload[k] for k in ("r", "alpha", "sigma")}
        seed = _stable_seed(self.base_cfg.seed, point_params)
        updates = {
            self.param_x: float(x), self.param_y: float(y),
            "amostras": int(effort.samples), "total_passos": int(effort.steps),
            "seed": seed,
        }
        available = _cfg_field_names()
        if "absorbing_window" in available:
            updates["absorbing_window"] = 0
        if "make_plots" in available:
            updates["make_plots"] = False
        if "compute_time_analysis" in available:
            updates["compute_time_analysis"] = False
        cfg = replace(self.base_cfg, **updates)
        batch = _run_batches_compat(cfg, pool=self.pool)
        result = _extract_point_result(batch, cfg, x, y, self.param_x, self.param_y, effort, seed)
        self.cache.save(payload, result)
        return result


@contextmanager
def shared_sample_pool(max_samples: int):
    if create_sample_pool is None:
        yield None
        return
    with create_sample_pool(max_samples) as pool:
        yield pool


def recommended_bounds(pair: str):
    """Broad reference ranges; AMR, not manual cropping, saves the work."""
    if pair == "r_alpha":
        return "r", (1.0, 4.5), "alpha", (0.0, 0.60)
    if pair == "r_sigma":
        return "r", (1.0, 4.5), "sigma", (1.0, 3.50)
    if pair == "alpha_sigma":
        return "alpha", (0.0, 0.60), "sigma", (1.0, 3.50)
    raise ValueError("pair must be r_alpha, r_sigma or alpha_sigma")


def adaptive_preset(name: str) -> AdaptiveConfig:
    # L is deliberately absent: all presets run on the same physical system.
    if name == "scout":
        return AdaptiveConfig(
            initial_nx=7, initial_ny=7, max_depth=2, max_points=500,
            pilot=EffortTier(3, 600), confirm=EffortTier(5, 1200),
            critical=EffortTier(8, 2400),
            sem_tol=0.050, critical_sem_tol=0.030,
            drift_tol=0.050, critical_drift_tol=0.030,
        )
    if name == "refine":
        return AdaptiveConfig(
            initial_nx=7, initial_ny=7, max_depth=4, max_points=2500,
            pilot=EffortTier(4, 800), confirm=EffortTier(8, 1600),
            critical=EffortTier(12, 3200),
            sem_tol=0.035, critical_sem_tol=0.020,
            drift_tol=0.035, critical_drift_tol=0.020,
        )
    if name == "validation":
        return AdaptiveConfig(
            initial_nx=9, initial_ny=9, max_depth=4, max_points=3500,
            pilot=EffortTier(10, 1200), confirm=EffortTier(15, 2400),
            critical=EffortTier(20, 4000),
            sem_tol=0.025, critical_sem_tol=0.015,
            drift_tol=0.025, critical_drift_tol=0.015,
        )
    raise ValueError("preset must be scout, refine or validation")


def save_adaptive_result(base_cfg, param_x, param_y, points, cells, diagnostics, acfg, stem):
    out_dir = Path(base_cfg.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(points.values(), key=lambda p: (p.x, p.y))
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        x=np.array([p.x for p in ordered]), y=np.array([p.y for p in ordered]),
        mean=np.vstack([p.mean for p in ordered]), sem=np.vstack([p.sem for p in ordered]),
        variance=np.vstack([p.variance for p in ordered]), variance_sem=np.vstack([p.variance_sem for p in ordered]),
        drift=np.vstack([p.drift for p in ordered]),
        samples=np.array([p.effort.samples for p in ordered]),
        steps=np.array([p.effort.steps for p in ordered]),
        base_seed=np.array([p.base_seed for p in ordered], dtype=np.uint32),
        absorbed_fraction=np.array([p.absorbed_fraction for p in ordered]),
    )
    manifest = {
        "param_x": param_x, "param_y": param_y,
        "kernel_fingerprint": kernel_fingerprint(),
        "scientific_min_L": MIN_PHYSICAL_L,
        "reference_L": REFERENCE_L,
        "base_config": {
            k: v for k, v in asdict(base_cfg).items()
            if isinstance(v, (str, int, float, bool)) or v is None
        },
        "adaptive_config": asdict(acfg),
        "n_points": len(points), "n_leaf_cells": len(cells),
        "points": [p.to_jsonable() for p in ordered],
        "cells": [{**asdict(c), "diagnostics": asdict(diagnostics[c])} for c in cells],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, json_path


def plot_adaptive_result(base_cfg, param_x, param_y, points, stem, show_points=True):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    ordered = sorted(points.values(), key=lambda p: (p.x, p.y))
    x = np.array([p.x for p in ordered]); y = np.array([p.y for p in ordered])
    mean = np.vstack([p.mean for p in ordered]); variance = np.vstack([p.variance for p in ordered])
    triang = mtri.Triangulation(x, y)
    fig_dir = Path(base_cfg.figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    def render(data, prefix, suffix, limits=None):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
        for i, ax in enumerate(axes):
            kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
            im = ax.tripcolor(triang, data[:, i], shading="gouraud", **kwargs)
            if show_points:
                ax.scatter(x, y, s=4, alpha=0.35)
            fig.colorbar(im, ax=ax)
            ax.set(xlabel=param_x, ylabel=param_y, title=f"{prefix} {'CDP'[i]}")
        path = fig_dir / f"{stem}_{suffix}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
        return path

    return (
        render(mean, "steady fraction", "mean", (0.0, 1.0)),
        render(variance, "temporal variance", "variance"),
    )


def run_pair(pair, base_cfg=None, acfg=None, stem=None, verbose=True):
    base_cfg = phase_reference_config() if base_cfg is None else base_cfg
    validate_physical_scale(base_cfg)
    acfg = adaptive_preset("scout") if acfg is None else acfg
    param_x, x_bounds, param_y, y_bounds = recommended_bounds(pair)
    cache = AdaptivePointCache(Path(base_cfg.data_dir) / "adaptive_phase_cache_v2")
    max_samples = max(acfg.pilot.samples, acfg.confirm.samples, acfg.critical.samples)
    with shared_sample_pool(max_samples) as pool:
        evaluator = PhysicalPointEvaluator(base_cfg, param_x, param_y, cache, pool=pool)
        points, cells, diagnostics = adaptive_phase_sweep(
            param_x, x_bounds, param_y, y_bounds, evaluator,
            acfg=acfg, verbose=verbose,
        )
    stem = stem or f"adaptive_v2_{param_x}_{param_y}"
    data_paths = save_adaptive_result(base_cfg, param_x, param_y, points, cells, diagnostics, acfg, stem)
    figure_paths = plot_adaptive_result(base_cfg, param_x, param_y, points, stem)
    return points, cells, diagnostics, data_paths, figure_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validated adaptive Corruption Game phase sweep")
    parser.add_argument("pair", choices=["r_alpha", "r_sigma", "alpha_sigma"])
    parser.add_argument("--preset", choices=["scout", "refine", "validation"], default="scout")
    parser.add_argument("--L", type=int)
    parser.add_argument("--r", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--percent-avg", type=float)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = phase_reference_config()
    updates = {k: getattr(args, k) for k in ("L", "r", "alpha", "sigma") if getattr(args, k) is not None}
    if args.percent_avg is not None:
        updates["percent_avg_MC"] = args.percent_avg
    if updates:
        cfg = replace(cfg, **updates)
    validate_physical_scale(cfg)

    acfg = adaptive_preset(args.preset)
    au = {k: getattr(args, k) for k in ("max_points", "max_depth") if getattr(args, k) is not None}
    if au:
        acfg = replace(acfg, **au)
    run_pair(args.pair, cfg, acfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
