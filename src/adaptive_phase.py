"""Generic adaptive mesh refinement for two-parameter phase diagrams.

The module is deliberately independent of the microscopic simulation.  A
caller supplies an evaluator ``(x, y, effort) -> PointResult``.  This keeps the
physics kernel unchanged and makes the adaptive logic directly testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class EffortTier:
    samples: int
    steps: int


@dataclass(frozen=True)
class AdaptiveConfig:
    initial_nx: int = 7
    initial_ny: int = 7
    max_depth: int = 3
    max_points: int = 1200

    # Spatial refinement: phase changes, nonlinear interpolation and dynamics.
    composition_range_tol: float = 0.20
    center_error_tol: float = 0.075
    variance_tol: float = 2.5e-3
    phase_extinction_threshold: float = 0.025

    # Statistical escalation: samples and trajectory length are distinct.
    sem_tol: float = 0.035
    critical_sem_tol: float = 0.020
    drift_tol: float = 0.040
    critical_drift_tol: float = 0.025

    pilot: EffortTier = EffortTier(samples=3, steps=450)
    confirm: EffortTier = EffortTier(samples=6, steps=1000)
    critical: EffortTier = EffortTier(samples=10, steps=1800)

    # Historical information only forces a minimum mesh depth.  It never
    # overrides data-driven stopping at deeper levels.
    use_historical_prior: bool = True
    prior_min_depth: int = 1
    round_digits: int = 12


@dataclass
class PointResult:
    x: float
    y: float
    param_x: str
    param_y: str
    mean: np.ndarray
    sem: np.ndarray
    variance: np.ndarray
    variance_sem: np.ndarray
    effort: EffortTier
    base_seed: int
    absorbed_fraction: float = math.nan
    drift: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def phase_signature(self, extinction_threshold: float) -> tuple:
        dominant = int(np.argmax(self.mean))
        extant = tuple(bool(v >= extinction_threshold) for v in self.mean)
        return dominant, extant

    def to_jsonable(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "param_x": self.param_x,
            "param_y": self.param_y,
            "mean": self.mean.tolist(),
            "sem": self.sem.tolist(),
            "variance": self.variance.tolist(),
            "variance_sem": self.variance_sem.tolist(),
            "effort": asdict(self.effort),
            "base_seed": self.base_seed,
            "absorbed_fraction": self.absorbed_fraction,
            "drift": self.drift.tolist(),
        }

    @classmethod
    def from_jsonable(cls, data: Mapping) -> "PointResult":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            param_x=str(data["param_x"]),
            param_y=str(data["param_y"]),
            mean=np.asarray(data["mean"], dtype=float),
            sem=np.asarray(data["sem"], dtype=float),
            variance=np.asarray(data["variance"], dtype=float),
            variance_sem=np.asarray(data["variance_sem"], dtype=float),
            effort=EffortTier(**data["effort"]),
            base_seed=int(data["base_seed"]),
            absorbed_fraction=float(data.get("absorbed_fraction", math.nan)),
            drift=np.asarray(data.get("drift", [0.0, 0.0, 0.0]), dtype=float),
        )


@dataclass(frozen=True)
class Cell:
    x0: float
    x1: float
    y0: float
    y1: float
    depth: int

    @property
    def xm(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def ym(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    def split(self) -> tuple["Cell", "Cell", "Cell", "Cell"]:
        xm, ym, d = self.xm, self.ym, self.depth + 1
        return (
            Cell(self.x0, xm, self.y0, ym, d),
            Cell(xm, self.x1, self.y0, ym, d),
            Cell(self.x0, xm, ym, self.y1, d),
            Cell(xm, self.x1, ym, self.y1, d),
        )


@dataclass
class CellDiagnostics:
    refine: bool
    reasons: tuple[str, ...]
    composition_range: float
    center_error: float
    max_variance: float
    max_sem: float
    prior_level: int


def historical_prior_level(param_x: str, param_y: str, x: float, y: float) -> int:
    """Project-informed search prior from preserved phase diagnostics.

    This is a sampling prior, not a phase-boundary claim.
    """

    vals = {param_x: x, param_y: y}
    if {param_x, param_y} == {"r", "alpha"}:
        r, a = vals["r"], vals["alpha"]
        if 2.85 <= r <= 3.75 and 0.15 <= a <= 0.62:
            return 1
        if 3.0 <= r <= 3.8 and 0.0 <= a <= 0.08:
            return 1

    if {param_x, param_y} == {"r", "sigma"}:
        r, s = vals["r"], vals["sigma"]
        if 2.85 <= r <= 3.85 and 0.98 <= s <= 1.15:
            return 1

    if {param_x, param_y} == {"alpha", "sigma"}:
        a, s = vals["alpha"], vals["sigma"]
        if 0.05 <= a <= 0.62 and 0.98 <= s <= 1.20:
            return 1
        if a < 0.75:
            crossover = 1.0 / max(1e-12, 1.0 - a)
            if abs(s - crossover) <= 0.06:
                return 1
    return 0


def statistical_flags(result: PointResult, acfg: AdaptiveConfig) -> tuple[bool, bool, bool]:
    """Return (sample_uncertainty, time_uncertainty, severe)."""

    max_sem = float(np.max(result.sem))
    max_var = float(np.max(result.variance))
    max_drift = float(np.max(result.drift))
    sample_uncertainty = max_sem >= acfg.sem_tol
    time_uncertainty = max_drift >= acfg.drift_tol or max_var >= acfg.variance_tol
    severe = (
        max_sem >= 2.0 * acfg.sem_tol
        or max_drift >= 2.0 * acfg.drift_tol
        or max_var >= 2.5 * acfg.variance_tol
    )
    return sample_uncertainty, time_uncertainty, severe


def choose_next_effort(result: PointResult, acfg: AdaptiveConfig) -> EffortTier | None:
    """Escalate samples and MCS independently whenever possible."""

    sample_uncertainty, time_uncertainty, severe = statistical_flags(result, acfg)
    if not sample_uncertainty and not time_uncertainty:
        return None
    target = acfg.critical if severe else acfg.confirm
    effort = EffortTier(
        samples=target.samples if sample_uncertainty else result.effort.samples,
        steps=target.steps if time_uncertainty else result.effort.steps,
    )
    return None if effort == result.effort else effort


def _point_key(x: float, y: float, digits: int) -> tuple[float, float]:
    return round(float(x), digits), round(float(y), digits)


def _cell_points(cell: Cell) -> tuple[tuple[float, float], ...]:
    return (
        (cell.x0, cell.y0),
        (cell.x1, cell.y0),
        (cell.x0, cell.y1),
        (cell.x1, cell.y1),
        (cell.xm, cell.ym),
    )


def diagnose_cell(
    cell: Cell,
    points: Mapping[tuple[float, float], PointResult],
    param_x: str,
    param_y: str,
    acfg: AdaptiveConfig,
) -> CellDiagnostics:
    vals = [
        points[_point_key(x, y, acfg.round_digits)]
        for x, y in _cell_points(cell)
    ]
    corners, center = vals[:4], vals[4]
    comp = np.vstack([p.mean for p in vals])
    composition_range = float(np.max(np.ptp(comp, axis=0)))
    bilinear_center = np.mean(np.vstack([p.mean for p in corners]), axis=0)
    center_error = float(np.sum(np.abs(center.mean - bilinear_center)))
    max_variance = float(max(np.max(p.variance) for p in vals))
    max_sem = float(max(np.max(p.sem) for p in vals))
    signatures = {
        p.phase_signature(acfg.phase_extinction_threshold)
        for p in vals
    }

    prior_level = 0
    if acfg.use_historical_prior:
        prior_level = historical_prior_level(param_x, param_y, cell.xm, cell.ym)

    reasons: list[str] = []
    if len(signatures) > 1:
        reasons.append("phase_signature")
    if composition_range >= acfg.composition_range_tol:
        reasons.append("composition_gradient")
    if center_error >= acfg.center_error_tol:
        reasons.append("interpolation_error")
    if max_variance >= acfg.variance_tol:
        reasons.append("temporal_variance")
    if prior_level > 0 and cell.depth < acfg.prior_min_depth:
        reasons.append("historical_prior")

    return CellDiagnostics(
        refine=bool(reasons) and cell.depth < acfg.max_depth,
        reasons=tuple(reasons),
        composition_range=composition_range,
        center_error=center_error,
        max_variance=max_variance,
        max_sem=max_sem,
        prior_level=prior_level,
    )


def _evaluate_with_escalation(
    evaluator: Callable[[float, float, EffortTier], PointResult],
    x: float,
    y: float,
    acfg: AdaptiveConfig,
) -> PointResult:
    result = evaluator(x, y, acfg.pilot)
    effort = choose_next_effort(result, acfg)
    if effort is None:
        return result

    result = evaluator(x, y, effort)
    if (
        float(np.max(result.sem)) >= acfg.critical_sem_tol
        or float(np.max(result.drift)) >= acfg.critical_drift_tol
        or float(np.max(result.variance)) >= 2.5 * acfg.variance_tol
    ) and result.effort != acfg.critical:
        result = evaluator(x, y, acfg.critical)
    return result


def _initial_cells(x_bounds, y_bounds, nx: int, ny: int) -> list[Cell]:
    xs = np.linspace(x_bounds[0], x_bounds[1], nx)
    ys = np.linspace(y_bounds[0], y_bounds[1], ny)
    return [
        Cell(float(xs[i]), float(xs[i + 1]), float(ys[j]), float(ys[j + 1]), 0)
        for i in range(nx - 1)
        for j in range(ny - 1)
    ]


def adaptive_phase_sweep(
    param_x: str,
    x_bounds: tuple[float, float],
    param_y: str,
    y_bounds: tuple[float, float],
    evaluator: Callable[[float, float, EffortTier], PointResult],
    *,
    acfg: AdaptiveConfig = AdaptiveConfig(),
    verbose: bool = True,
) -> tuple[dict[tuple[float, float], PointResult], list[Cell], dict[Cell, CellDiagnostics]]:
    """Breadth-first quadtree refinement with a hard point budget."""

    if param_x == param_y:
        raise ValueError("param_x and param_y must differ")
    if acfg.initial_nx < 2 or acfg.initial_ny < 2:
        raise ValueError("initial_nx and initial_ny must be >= 2")
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError("parameter bounds must be increasing")

    points: dict[tuple[float, float], PointResult] = {}
    cells = _initial_cells(x_bounds, y_bounds, acfg.initial_nx, acfg.initial_ny)

    def ensure_point(x: float, y: float) -> PointResult:
        key = _point_key(x, y, acfg.round_digits)
        if key not in points:
            if len(points) >= acfg.max_points:
                raise RuntimeError("adaptive sweep reached max_points")
            result = _evaluate_with_escalation(evaluator, float(x), float(y), acfg)
            points[key] = result
            if verbose:
                print(
                    f"[{len(points):4d}] {param_x}={x:.6g}, {param_y}={y:.6g} "
                    f"mean={np.round(result.mean, 3)} "
                    f"sem={np.max(result.sem):.3g} "
                    f"var={np.max(result.variance):.3g} "
                    f"drift={np.max(result.drift):.3g} "
                    f"effort={result.effort.samples}x{result.effort.steps}",
                    flush=True,
                )
        return points[key]

    while True:
        next_cells: list[Cell] = []
        refined_any = False

        # Evaluate the complete current generation before reserving next-level
        # points.  This guarantees a clean stop at max_points.
        for cell in cells:
            for x, y in _cell_points(cell):
                ensure_point(x, y)

        reserved_keys: set[tuple[float, float]] = set()
        for cell in cells:
            diag = diagnose_cell(cell, points, param_x, param_y, acfg)
            if not diag.refine:
                next_cells.append(cell)
                continue

            children = cell.split()
            required = {
                _point_key(x, y, acfg.round_digits)
                for child in children
                for x, y in _cell_points(child)
            }
            new_keys = required.difference(points).difference(reserved_keys)
            if len(points) + len(reserved_keys) + len(new_keys) > acfg.max_points:
                next_cells.append(cell)
                continue

            next_cells.extend(children)
            reserved_keys.update(new_keys)
            refined_any = True
            if verbose:
                print(
                    f" refine depth={cell.depth} "
                    f"[{cell.x0:.4g},{cell.x1:.4g}]x"
                    f"[{cell.y0:.4g},{cell.y1:.4g}] "
                    f"reasons={','.join(diag.reasons)}",
                    flush=True,
                )

        cells = next_cells
        if not refined_any:
            break

    diagnostics: dict[Cell, CellDiagnostics] = {}
    for cell in cells:
        for x, y in _cell_points(cell):
            ensure_point(x, y)
        diagnostics[cell] = diagnose_cell(cell, points, param_x, param_y, acfg)
    return points, cells, diagnostics
