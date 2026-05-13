import numpy as np
import time
from pathlib import Path
from run_sampling import run_batches
from config import SimulationConfig
from plotting import plot_sweep_1d, plot_trajectories_vs_time, plot_variance_vs_param
from dataclasses import replace
from run_sweep import run_simulation
from config_sweep import SweepConfig
from plotting import plot_heatmap, plot_heatmap_3
printatudo = True
cfg = SimulationConfig()


def obs_C(steady):
    return steady  # depois você pega índice 0

def obs_fraction_C(steady):
    # steady: (3, amostras)
    return np.mean(steady[0])  # média sobre amostras

def obs_index(idx):
    def f(steady):
        return np.mean(steady[idx])
    return f

def sweep_2d(cfg, param_x, values_x, param_y, values_y, observable_fn):
    Z = np.zeros((len(values_x), len(values_y)))

    for i, vx in enumerate(values_x):
        for j, vy in enumerate(values_y):
            print(f"{param_x}={vx}, {param_y}={vy}")

            cfg_v = replace(cfg, **{param_x: vx, param_y: vy})

            estrat_t, _, payavg_t, _ = run_simulation(cfg_v)

            steady = np.mean(estrat_t[:, :, cfg_v.passos_media:], axis=2)

            # exemplo: fração de cooperadores
            Z[i, j] = observable_fn(steady)

    return Z

def sweep_2d_all(cfg, param_x, values_x, param_y, values_y):
    Z = np.zeros((len(values_x), len(values_y), 3))  # C, D, P
    Z_var = np.zeros((len(values_x), len(values_y), 3))  # C, D, P

    for i, vx in enumerate(values_x):
        for j, vy in enumerate(values_y):
            print(f"{param_x}={vx}, {param_y}={vy}")

            cfg_v = replace(cfg, **{param_x: vx, param_y: vy})

            estrat_t, _, _, _ = run_simulation(cfg_v)

            steady = np.mean(
                estrat_t[:, :, cfg_v.passos_media:], axis=2
            )  # (3, amostras
            Z[i, j, :] = np.mean(steady, axis=1)  # média nas amostras

            var = np.var(
                estrat_t[:, :, cfg_v.passos_media:], axis=2, ddof=1
            )  # (3, amostras)

            Z_var[i, j, :] = np.mean(var, axis=1)

    return Z, Z_var

def main():
    start = time.time()
#    r_vals = np.linspace(cfg.r_start, cfg.r_stop, cfg.r_npoints)
#    sigma_vals = np.linspace(cfg.sig_start, cfg.sig_stop, cfg.sig_npoints)
#    obs_C = obs_index(0)
#    obs_D = obs_index(1)
#    obs_P = obs_index(2)
#    Z = sweep_2d(cfg, "r", r_vals, "sigma", sigma_vals, obs_C)
#    plot_heatmap(r_vals, sigma_vals, Z, "r", "sigma", "Fração de C")

    if cfg.phaseport in ['alpha', 'both']:
        r_vals = np.linspace(cfg.r_start, cfg.r_stop, cfg.r_npoints)
        alpha_vals = np.linspace(cfg.alpha_start, cfg.alpha_stop, cfg.alpha_npoints)

        Z, Z_var = sweep_2d_all(cfg, "r", r_vals, "alpha", alpha_vals)

        plot_heatmap_3(r_vals, alpha_vals, Z, "r", "alpha")

        plot_heatmap_3(r_vals, alpha_vals, Z_var, "r", "alpha variance", 'variance')

    if cfg.phaseport in ['sigma', 'both']:
        r_vals = np.linspace(cfg.r_start, cfg.r_stop, cfg.r_npoints)
        sigma_vals = np.linspace(cfg.sig_start, cfg.sig_stop, cfg.sig_npoints)

        Z, Z_var = sweep_2d_all(cfg, "r", r_vals, "sigma", sigma_vals)

        plot_heatmap_3(r_vals, sigma_vals, Z, "r", " sigma")

        plot_heatmap_3(r_vals, sigma_vals, Z_var, "r", "sigma variance", 'variance')



    print(f"Tempo: {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
