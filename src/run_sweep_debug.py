"""
Debug version of run_sweep with memory monitoring and profiling.
Helps identify memory leaks and bottlenecks.
"""

import numpy as np
import time
import psutil
import os
import gc
from pathlib import Path
from run_sampling import run_batches
from config import SimulationConfig
from plotting import plot_sweep_1d, plot_trajectories_vs_time, plot_variance_vs_param
from dataclasses import replace, dataclass


@dataclass
class SweepConfig:
    param_name: str
    start: float
    stop: float
    n_points: int

    def values(self):
        return np.linspace(self.start, self.stop, self.n_points)


class MemoryMonitor:
    """Track memory usage during execution."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_mb = self.get_memory_mb()
        self.checkpoints = []

    def get_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def checkpoint(self, label):
        """Record memory usage at a checkpoint."""
        current = self.get_memory_mb()
        delta = current - self.initial_mb
        print(f"[MEM] {label:40s}: {current:8.1f} MB (Δ{delta:+7.1f} MB)")
        self.checkpoints.append((label, current, delta))

    def report(self):
        """Print memory report."""
        print("\n=== MEMORY REPORT ===")
        for label, mb, delta in self.checkpoints:
            print(f"  {label:40s}: {mb:8.1f} MB (Δ{delta:+7.1f} MB)")


cfg = SimulationConfig()
monitor = MemoryMonitor()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

monitor.checkpoint("STARTUP")


def temporal_variance(estrat_t, start):
    """Variância temporal após termalização."""
    return np.var(estrat_t[:, :, start:], axis=2)


def sweep_1d_DEBUG(cfg, param_name, values, observable_fn, monitor):
    """
    OPTIMIZED: Only keeps essential data, not full trajectories.
    """
    means = []
    sems = []
    vars_means = []
    vars_sems = []
    varp_means = []
    varp_sems = []

    for i, v in enumerate(values):
        print(f"  [{i+1}/{len(values)}] {param_name} = {v:.3f}")
        monitor.checkpoint(f"  Before sim: {param_name}={v:.3f}")

        cfg_v = replace(cfg, **{param_name: v})
        params = cfg_v.simulation_params()

        # RUN SIMULATION
        estrat_t, _, payavg_t, _ = run_batches(
            cfg_v.L,
            cfg_v.amostras,
            cfg_v.total_passos,
            params,
            base_seed=cfg_v.seed
        )

        monitor.checkpoint(f"  After sim:  {param_name}={v:.3f}")
        print(f"      estrat_t shape: {estrat_t.shape}, size: {estrat_t.nbytes/1024/1024:.1f} MB")

        # COMPUTE STATS (only what we need)
        steady_strat = np.mean(estrat_t[:, :, cfg_v.passos_media:], axis=2)
        var_strat = temporal_variance(estrat_t, cfg_v.passos_media)
        steady_pay = np.mean(payavg_t[:, :, cfg_v.passos_media:], axis=2)
        var_pay = temporal_variance(payavg_t, cfg_v.passos_media)

        obs = observable_fn(steady_strat, steady_pay)

        var_mean = np.mean(var_strat, axis=1)
        var_sem = np.std(var_strat, axis=1, ddof=1) / np.sqrt(cfg_v.amostras)
        varp_mean = np.mean(var_pay, axis=1)
        varp_sem = np.std(var_pay, axis=1, ddof=1) / np.sqrt(cfg_v.amostras)

        means.append(np.mean(obs, axis=1))
        sems.append(np.std(obs, axis=1, ddof=1) / np.sqrt(cfg_v.amostras))
        vars_means.append(var_mean)
        vars_sems.append(var_sem)
        varp_means.append(varp_mean)
        varp_sems.append(varp_sem)

        # CLEANUP
        del estrat_t, payavg_t, steady_strat, steady_pay, var_strat, var_pay, obs
        gc.collect()

        monitor.checkpoint(f"  After cleanup: {param_name}={v:.3f}")

    return (
        np.array(means),
        np.array(sems),
        np.array(vars_means),
        np.array(vars_sems),
        np.array(varp_means),
        np.array(varp_sems),
    )


def obs_fraction(steady_strat, steady_pay):
    return steady_strat


def obs_payoff(steady_strat, steady_pay):
    return steady_pay


def start_sweep(cfg, sweep, data_dir, figures_dir, monitor):
    print(f"\n{'='*60}")
    print(f"=== SWEEP: {sweep.param_name.upper()} ===")
    print(f"r={cfg.r}, sigma={cfg.sigma}, alpha={cfg.alpha}")
    print(f"{'='*60}")

    values = sweep.values()

    mean, sem, vars_mean, vars_sem, varp_mean, varp_sem = sweep_1d_DEBUG(
        cfg,
        sweep.param_name,
        values,
        observable_fn=obs_fraction,
        monitor=monitor
    )

    monitor.checkpoint(f"Completed sweep: {sweep.param_name}")

    # PLOT & SAVE
    try:
        plot_sweep_1d(
            values,
            mean,
            sem,
            ["C", "D", "P"],
            sweep.param_name,
            cfg
        )
        print(f"  ✓ Plot saved")
    except Exception as e:
        print(f"  ✗ Plot failed: {e}")

    # Save data
    np.savetxt(
        data_dir / f"vs_{sweep.param_name}.dat",
        np.column_stack([
            values,
            mean[:, 0], mean[:, 1], mean[:, 2],
            sem[:, 0], sem[:, 1], sem[:, 2],
        ]),
        header=f"{sweep.param_name} C_mean D_mean P_mean C_sem D_sem P_sem"
    )
    print(f"  ✓ Data saved")

    # Cleanup large arrays
    del mean, sem, vars_mean, vars_sem, varp_mean, varp_sem
    gc.collect()
    monitor.checkpoint(f"After cleanup: {sweep.param_name}")


def main():
    start = time.time()

    sweeps = [
        SweepConfig("r", cfg.r_start, cfg.r_stop, cfg.r_npoints),
        SweepConfig("sigma", cfg.sig_start, cfg.sig_stop, cfg.sig_npoints),
        SweepConfig("alpha", cfg.alpha_start, cfg.alpha_stop, cfg.alpha_npoints),
    ]

    for sweep in sweeps:
        start_sweep(cfg, sweep, DATA_DIR, FIGURES_DIR, monitor)
        gc.collect()  # Aggressive cleanup between sweeps

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    monitor.report()
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


