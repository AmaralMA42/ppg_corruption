"""Adaptive phase-diagram driver for the spatial Corruption Game.

The microscopic model is imported from the existing repository. This file
adds caching, deterministic seeds, simulation summaries, plotting and a small
CLI around the generic adaptive engine in ``adaptive_phase.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
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


class AdaptivePointCache:
    """Persistent cache keyed by the complete physical point and effort."""

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
        kwargs["absorbing_window"] = getattr(cfg, "absorbing_window", 0)
    if "pool" in sig.parameters and pool is not None:
        kwargs["pool"] = pool
    return run_batches(cfg.L, cfg.amostras, cfg.total_passos, cfg.simulation_params(), **kwargs)


def _extract_point_result(batch_result, cfg, x, y, param_x, param_y, effort, seed):
    estrat_t = np.asarray(batch_result[0], dtype=float)
    tail = estrat_t[:, :, cfg.passos_media:]
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
        self.base_cfg = base_cfg
        self.param_x = param_x
        self.param_y = param_y
        self.cache = cache
        self.pool = pool

    def _payload(self, x, y, effort):
        vals = {"r": float(self.base_cfg.r), "alpha": float(self.base_cfg.alpha), "sigma": float(self.base_cfg.sigma)}
        vals[self.param_x] = float(x)
        vals[self.param_y] = float(y)
        return {
            "model": "corruption_game_spatial_square_fermi",
            **vals,
            "L": int(self.base_cfg.L), "G": int(self.base_cfg.G),
            "c": float(self.base_cfg.c), "k": float(self.base_cfg.k),
            "cond_ini": int(self.base_cfg.cond_ini),
            "percent_avg_MC": float(self.base_cfg.percent_avg_MC),
            "samples": int(effort.samples), "steps": int(effort.steps),
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
        if hasattr(self.base_cfg, "make_plots"):
            updates["make_plots"] = False
        if hasattr(self.base_cfg, "compute_time_analysis"):
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
    if pair == "r_alpha":
        return "r", (2.3, 4.2), "alpha", (0.0, 0.70)
    if pair == "r_sigma":
        return "r", (2.3, 4.2), "sigma", (0.92, 1.45)
    if pair == "alpha_sigma":
        return "alpha", (0.0, 0.70), "sigma", (0.92, 1.45)
    raise ValueError("pair must be r_alpha, r_sigma or alpha_sigma")


def adaptive_preset(name: str) -> AdaptiveConfig:
    if name == "scout":
        return AdaptiveConfig(
            initial_nx=7, initial_ny=7, max_depth=2, max_points=500,
            pilot=EffortTier(2, 300), confirm=EffortTier(4, 700), critical=EffortTier(8, 1400),
            sem_tol=0.050, critical_sem_tol=0.030, drift_tol=0.060, critical_drift_tol=0.035,
        )
    if name == "refine":
        return AdaptiveConfig(
            initial_nx=7, initial_ny=7, max_depth=4, max_points=2500,
            pilot=EffortTier(3, 500), confirm=EffortTier(6, 1200), critical=EffortTier(12, 2500),
            sem_tol=0.035, critical_sem_tol=0.020, drift_tol=0.040, critical_drift_tol=0.025,
        )
    if name == "validation":
        return AdaptiveConfig(
            initial_nx=9, initial_ny=9, max_depth=4, max_points=3500,
            pilot=EffortTier(4, 800), confirm=EffortTier(10, 2000), critical=EffortTier(20, 4000),
            sem_tol=0.025, critical_sem_tol=0.015, drift_tol=0.030, critical_drift_tol=0.018,
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
        samples=np.array([p.effort.samples for p in ordered]), steps=np.array([p.effort.steps for p in ordered]),
        base_seed=np.array([p.base_seed for p in ordered], dtype=np.uint32),
        absorbed_fraction=np.array([p.absorbed_fraction for p in ordered]),
    )
    manifest = {
        "param_x": param_x, "param_y": param_y,
        "base_config": {k: v for k, v in asdict(base_cfg).items() if isinstance(v, (str, int, float, bool)) or v is None},
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

    return render(mean, "steady fraction", "mean", (0.0, 1.0)), render(variance, "temporal variance", "variance")


def run_pair(pair, base_cfg=None, acfg=None, stem=None, verbose=True):
    base_cfg = SimulationConfig() if base_cfg is None else base_cfg
    acfg = adaptive_preset("scout") if acfg is None else acfg
    param_x, x_bounds, param_y, y_bounds = recommended_bounds(pair)
    cache = AdaptivePointCache(Path(base_cfg.data_dir) / "adaptive_phase_cache")
    max_samples = max(acfg.pilot.samples, acfg.confirm.samples, acfg.critical.samples)
    with shared_sample_pool(max_samples) as pool:
        evaluator = PhysicalPointEvaluator(base_cfg, param_x, param_y, cache, pool=pool)
        points, cells, diagnostics = adaptive_phase_sweep(param_x, x_bounds, param_y, y_bounds, evaluator, acfg=acfg, verbose=verbose)
    stem = stem or f"adaptive_{param_x}_{param_y}"
    data_paths = save_adaptive_result(base_cfg, param_x, param_y, points, cells, diagnostics, acfg, stem)
    figure_paths = plot_adaptive_result(base_cfg, param_x, param_y, points, stem)
    return points, cells, diagnostics, data_paths, figure_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Corruption Game phase sweep")
    parser.add_argument("pair", choices=["r_alpha", "r_sigma", "alpha_sigma"])
    parser.add_argument("--preset", choices=["scout", "refine", "validation"], default="scout")
    parser.add_argument("--L", type=int); parser.add_argument("--r", type=float)
    parser.add_argument("--alpha", type=float); parser.add_argument("--sigma", type=float)
    parser.add_argument("--max-points", type=int); parser.add_argument("--max-depth", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = SimulationConfig()
    updates = {k: getattr(args, k) for k in ("L", "r", "alpha", "sigma") if getattr(args, k) is not None}
    if updates:
        cfg = replace(cfg, **updates)
    acfg = adaptive_preset(args.preset)
    au = {k: getattr(args, k) for k in ("max_points", "max_depth") if getattr(args, k) is not None}
    if au:
        acfg = replace(acfg, **au)
    run_pair(args.pair, cfg, acfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
