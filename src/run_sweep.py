import numpy as np
import time
from pathlib import Path
from run_sampling import run_batches
from config import SimulationConfig
from plotting import  plot_vs_r, plot_sweep_1d
from dataclasses import replace

cfg = SimulationConfig()
L = cfg.L
amostras = cfg.amostras
total_passos = cfg.total_passos
passos_media = cfg.passos_media
seed = cfg.seed
params = cfg.simulation_params()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def sweep_r(r_values, FIGURES_DIR=FIGURES_DIR):
    results_mean = []
    results_sem = []
    for r_val in r_values:
        print(f"Rodando r = {r_val:.2f}")

        cfg_var = replace(cfg, r=r_val)
        params = cfg_var.simulation_params()

        estrat_t, _, payavg_t, _ = run_batches(
            L, amostras, total_passos, params, base_seed=seed
        )

        steady_state = np.mean(estrat_t[:, :, passos_media:], axis=2)

        mean = np.mean(steady_state, axis=1)
        std = np.std(steady_state, axis=1, ddof=1)
        sem = std / np.sqrt(amostras)

        results_mean.append(mean)
        results_sem.append(sem)
    return np.array(results_mean), np.array(results_sem)

def run_simulation(cfg):
    params = cfg.simulation_params()

    return run_batches(
        cfg.L,
        cfg.amostras,
        cfg.total_passos,
        params,
        base_seed=cfg.seed
    )

def sweep_1d(cfg, param_name, values, observable_fn):
    means = []
    sems = []

    for v in values:
        print(f"{param_name} = {v}")

        cfg_v = replace(cfg, **{param_name: v})

        estrat_t, _, payavg_t, _ = run_simulation(cfg_v)

        steady = np.mean(estrat_t[:, :, cfg_v.passos_media:], axis=2)

        obs = observable_fn(steady)

        means.append(np.mean(obs, axis=1))
        sems.append(np.std(obs, axis=1, ddof=1) / np.sqrt(cfg_v.amostras))

    return np.array(means), np.array(sems)

def obs_fraction(steady_state):
    return steady_state

def obs_payoff(steady_payoff):
    return steady_payoff


def update_config(cfg, **kwargs):
    return replace(cfg, **kwargs)

def main():
    start = time.time()

    r_values = np.linspace(cfg.r_start, cfg.r_stop, cfg.r_npoints)

    mean, sem = sweep_r(r_values)

    plot_vs_r(r_values, mean, sem, FIGURES_DIR, cfg)

    mean, sem = sweep_1d(
        cfg,
        "r",
        r_values,
        observable_fn=obs_fraction
    )

    plot_sweep_1d(r_values, mean, sem, ["C", "D", "P"], "r")


    np.savetxt(
        DATA_DIR / "vs_r.dat",
        np.column_stack([
            r_values,
            mean[:, 0], mean[:, 1], mean[:, 2],
            sem[:, 0], sem[:, 1], sem[:, 2],
        ]),
        header="r C_mean D_mean P_mean C_sem D_sem P_sem"
    )

    print(f"Tempo total: {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
