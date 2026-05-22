import numpy as np
import time
from pathlib import Path
from run_sampling import run_batches
from config import SimulationConfig
from plotting import plot_sweep_1d, plot_trajectories_vs_time, plot_variance_vs_param
from dataclasses import replace, dataclass
#from config_sweep import SweepConfig

@dataclass
class SweepConfig:
    param_name: str
    start: float
    stop: float
    n_points: int

    def values(self):
        return np.linspace(self.start, self.stop, self.n_points)

cfg = SimulationConfig()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

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
    traj_strat = []   # ← novo
    traj_pay = []     # ← novo
    vars_means = []
    vars_sems = []
    varp_means = []
    varp_sems = []

    for v in values:
        print(f"{param_name} = {v}")

        cfg_v = replace(cfg, **{param_name: v})

        estrat_t, _, payavg_t, _ = run_simulation(cfg_v)

        # amostrador da amostra zero!!
        traj_strat.append(estrat_t[:, 0, :])
        traj_pay.append(payavg_t[:, 0, :])

        steady_strat = np.mean(estrat_t[:, :, cfg_v.passos_media:], axis=2)
        var_strat = temporal_variance(estrat_t, cfg_v.passos_media)
        steady_pay = np.mean(payavg_t[:, :, cfg_v.passos_media:], axis=2)
        var_pay = temporal_variance(payavg_t, cfg_v.passos_media)

        obs = observable_fn(steady_strat, steady_pay)

        var_mean = np.mean(var_strat, axis=1)
        var_sem = sem_across_samples(var_strat)
        varp_mean = np.mean(var_pay, axis=1)
        varp_sem = sem_across_samples(var_pay)

        means.append(np.mean(obs, axis=1))
        sems.append(sem_across_samples(obs))
        vars_means.append(var_mean)
        vars_sems.append(var_sem)
        varp_means.append(varp_mean)
        varp_sems.append(varp_sem)


    return (
        np.array(means),
        np.array(sems),
        np.array(traj_strat),  # (n_values, 3, T)
        np.array(traj_pay),
        np.array(vars_means),
        np.array(vars_sems),
        np.array(varp_means),
        np.array(varp_sems),
    )



def obs_fraction(steady_strat, steady_pay):
    return steady_strat

def obs_payoff(steady_strat, steady_pay):
    return steady_pay


def temporal_variance(estrat_t, start):
    """
    Variância temporal após termalização.

    estrat_t shape: (3, amostras, tempo)
    return: (3, amostras)
    """
    return np.var(estrat_t[:, :, start:], axis=2)

def sem_across_samples(values):
    """
    Erro padrao entre amostras.

    values shape: (3, amostras)
    return: (3,)
    """
    n_samples = values.shape[1]
    if n_samples <= 1:
        return np.zeros(values.shape[0])
    return np.std(values, axis=1, ddof=1) / np.sqrt(n_samples)

def start_sweep(cfg, sweep, data_dir, figures_dir):
    print(f"\n=== Sweepgeral em {sweep.param_name} ===")
    print(
        f"r={cfg.r}, sigma={cfg.sigma}, alpha={cfg.alpha}, "
        f"k={cfg.k}, G={cfg.G}, c={cfg.c}"
    )

    values = sweep.values()

    mean, sem, traj_strat, traj_pay, vars_mean, vars_sem, varp_mean, varp_sem = sweep_1d(
        cfg,
        sweep.param_name,
        values,
        observable_fn=obs_fraction
    )



    # plot
    plot_sweep_1d(
        values,
        mean,
        sem,
        ["C", "D", "P"],
        sweep.param_name,
        cfg
    )

    plot_trajectories_vs_time(
        values,
        traj_strat,
        sweep.param_name,
        ylabel="ρ"
    )

    plot_trajectories_vs_time(
        values,
        traj_pay,
        sweep.param_name,
        ylabel="Payoff"
    )

    plot_trajectories_vs_time(
        values,
        traj_pay,
        sweep.param_name,
        ylabel="Payoff",
        step = 10
    )

    plot_variance_vs_param(values, vars_mean, vars_sem, ["C", "D", "P"], sweep.param_name)
    plot_variance_vs_param(values, varp_mean, varp_sem, ["Cpay", "D", "P"], sweep.param_name)


    # salvar dados
    np.savetxt(
        data_dir / f"vs_{sweep.param_name}.dat",
        np.column_stack([
            values,
            mean[:, 0], mean[:, 1], mean[:, 2],
            sem[:, 0], sem[:, 1], sem[:, 2],
        ]),
            header=f"{sweep.param_name} C_mean D_mean P_mean C_sem D_sem P_sem"
        )


def main():
    start = time.time()

 #   r_values = np.linspace(cfg.r_start, cfg.r_stop, cfg.r_npoints)
 #   mean, sem = sweep_1d(
 #       cfg,
 #       'r',
 #       r_values,
 #       observable_fn=obs_fraction
 #   )
 #   plot_vs_r(r_values, mean, sem, FIGURES_DIR, cfg)

#    param_name = cfg.param_name
#    values = np.linspace(cfg.alpha_start, cfg.alpha_stop, cfg.alpha_npoints)
#    mean, sem = sweep_1d(
#        cfg,
#        param_name,
#        values,
#        observable_fn=obs_fraction
#    )
#    plot_sweep_1d(values, mean, sem, ["C", "D", "P"], param_name)

#    param_name2 = cfg.param_name2
#    valuessi = np.linspace(cfg.sig_start, cfg.sig_stop, cfg.sig_npoints)
#    mean, sem = sweep_1d(
    #    cfg,
   #     param_name2,
  #      valuessi,
 #       observable_fn=obs_fraction
#    )
#    plot_sweep_1d(valuessi, mean, sem, ["Csig", "D", "P"], param_name2)

#    np.savetxt(
#        DATA_DIR / "vs_r.dat",
#        np.column_stack([
#            values,
#            mean[:, 0], mean[:, 1], mean[:, 2],
#            sem[:, 0], sem[:, 1], sem[:, 2],
#        ]),
#        header="r C_mean D_mean P_mean C_sem D_sem P_sem"
#    )




###############################################################################

    #config2

    sweeps = [
        SweepConfig("r", cfg.r_start, cfg.r_stop, cfg.r_npoints),
        SweepConfig("sigma", cfg.sig_start, cfg.sig_stop, cfg.sig_npoints),
        SweepConfig("alpha", cfg.alpha_start, cfg.alpha_stop, cfg.alpha_npoints),
    ]

    for sweep in sweeps:
        start_sweep(cfg, sweep, DATA_DIR, FIGURES_DIR)


    print(f"Tempo total: {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
