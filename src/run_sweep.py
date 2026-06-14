import time
from dataclasses import dataclass, replace

import numpy as np

from config import SimulationConfig
from plotting import (
    plot_activity_trajectories_vs_time,
    plot_metric_vs_param,
    plot_sweep_1d,
    plot_trajectories_vs_time,
    plot_variance_vs_param,
)
from run_sampling import run_batches
from time_analysis import analyze_strategy_scalar_metrics
from utils import config_metadata, load_npz_result, save_npz_result


@dataclass
class SweepConfig:
    param_name: str
    start: float
    stop: float
    n_points: int

    def values(self):
        return np.linspace(self.start, self.stop, self.n_points)


cfg = SimulationConfig()


def run_simulation(cfg):
    params = cfg.simulation_params()

    return run_batches(
        cfg.L,
        cfg.amostras,
        cfg.total_passos,
        params,
        base_seed=cfg.seed,
        absorbing_window=cfg.absorbing_window,
    )


def temporal_variance(estrat_t, start):
    """
    Variancia temporal apos termalizacao.

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


def obs_fraction(steady_strat, steady_pay):
    return steady_strat


def obs_payoff(steady_strat, steady_pay):
    return steady_pay


def sweep_1d(cfg, param_name, values, observable_fn):
    means = []
    sems = []
    traj_strat = []
    traj_pay = []
    traj_activity = []
    vars_means = []
    vars_sems = []
    varp_means = []
    varp_sems = []
    dominant_freq_means = []
    dominant_freq_sems = []
    dominant_power_means = []
    dominant_power_sems = []
    dominant_period_means = []
    dominant_period_sems = []
    peak_ratio_means = []
    peak_ratio_sems = []

    for v in values:
        print(f"{param_name} = {v}")

        cfg_v = replace(cfg, **{param_name: v})
        estrat_t, _, payavg_t, _, activity_t, _, absorbed_at = run_simulation(cfg_v)

        traj_strat.append(estrat_t[:, 0, :])
        traj_pay.append(payavg_t[:, 0, :])
        traj_activity.append(activity_t[0, :])

        steady_strat = np.mean(estrat_t[:, :, cfg_v.passos_media:], axis=2)
        steady_pay = np.mean(payavg_t[:, :, cfg_v.passos_media:], axis=2)
        var_strat = temporal_variance(estrat_t, cfg_v.passos_media)
        var_pay = temporal_variance(payavg_t, cfg_v.passos_media)
        scalar_analysis = None
        if cfg_v.compute_time_analysis:
            scalar_analysis = analyze_strategy_scalar_metrics(
                estrat_t,
                start=cfg_v.passos_media,
                absorbed_at=absorbed_at,
            )

        obs = observable_fn(steady_strat, steady_pay)

        means.append(np.mean(obs, axis=1))
        sems.append(sem_across_samples(obs))
        if scalar_analysis is None:
            vars_means.append(np.mean(var_strat, axis=1))
            vars_sems.append(sem_across_samples(var_strat))
            dominant_freq_means.append(np.full(3, np.nan))
            dominant_freq_sems.append(np.full(3, np.nan))
            dominant_power_means.append(np.full(3, np.nan))
            dominant_power_sems.append(np.full(3, np.nan))
            dominant_period_means.append(np.full(3, np.nan))
            dominant_period_sems.append(np.full(3, np.nan))
            peak_ratio_means.append(np.full(3, np.nan))
            peak_ratio_sems.append(np.full(3, np.nan))
        else:
            vars_means.append(scalar_analysis["variance_mean"])
            vars_sems.append(scalar_analysis["variance_sem"])
            dominant_freq_means.append(scalar_analysis["dominant_freq_mean"])
            dominant_freq_sems.append(scalar_analysis["dominant_freq_sem"])
            dominant_power_means.append(scalar_analysis["dominant_power_mean"])
            dominant_power_sems.append(scalar_analysis["dominant_power_sem"])
            dominant_period_means.append(scalar_analysis["dominant_period_mean"])
            dominant_period_sems.append(scalar_analysis["dominant_period_sem"])
            peak_ratio_means.append(scalar_analysis["peak_ratio_mean"])
            peak_ratio_sems.append(scalar_analysis["peak_ratio_sem"])
        varp_means.append(np.mean(var_pay, axis=1))
        varp_sems.append(sem_across_samples(var_pay))

    return (
        np.array(means),
        np.array(sems),
        np.array(traj_strat),
        np.array(traj_pay),
        np.array(traj_activity),
        np.array(vars_means),
        np.array(vars_sems),
        np.array(varp_means),
        np.array(varp_sems),
        np.array(dominant_freq_means),
        np.array(dominant_freq_sems),
        np.array(dominant_power_means),
        np.array(dominant_power_sems),
        np.array(dominant_period_means),
        np.array(dominant_period_sems),
        np.array(peak_ratio_means),
        np.array(peak_ratio_sems),
    )


def plot_sweep_results(
    values,
    mean,
    sem,
    traj_strat,
    traj_pay,
    traj_activity,
    vars_mean,
    vars_sem,
    varp_mean,
    varp_sem,
    param_name,
    cfg,
    dominant_period_mean=None,
    dominant_period_sem=None,
    peak_ratio_mean=None,
    peak_ratio_sem=None,
):
    plot_sweep_1d(values, mean, sem, ["C", "D", "P"], param_name, cfg)
    plot_trajectories_vs_time(values, traj_strat, param_name, ylabel="rho")
    plot_trajectories_vs_time(values, traj_pay, param_name, ylabel="Payoff")
    plot_trajectories_vs_time(values, traj_pay, param_name, ylabel="Payoff", step=10)
    plot_activity_trajectories_vs_time(values, traj_activity, param_name)
    plot_variance_vs_param(values, vars_mean, vars_sem, ["C", "D", "P"], param_name)
    plot_variance_vs_param(values, varp_mean, varp_sem, ["Cpay", "D", "P"], param_name)
    if dominant_period_mean is not None and np.any(np.isfinite(dominant_period_mean)):
        plot_metric_vs_param(
            values,
            dominant_period_mean,
            dominant_period_sem,
            ["C", "D", "P"],
            param_name,
            "Periodo dominante (MCS)",
            "Periodo dominante pos-transiente",
        )
    if peak_ratio_mean is not None and np.any(np.isfinite(peak_ratio_mean)):
        plot_metric_vs_param(
            values,
            peak_ratio_mean,
            peak_ratio_sem,
            ["C", "D", "P"],
            param_name,
            "Peak ratio",
            "Concentracao espectral no pico dominante",
            ylim=(0, 1),
        )


def plot_saved_sweep(path, cfg=cfg):
    arrays, metadata = load_npz_result(path)
    plot_sweep_results(
        arrays["values"],
        arrays["mean"],
        arrays["sem"],
        arrays["traj_strat"],
        arrays["traj_pay"],
        arrays["traj_activity"],
        arrays["vars_mean"],
        arrays["vars_sem"],
        arrays["varp_mean"],
        arrays["varp_sem"],
        metadata["param_name"],
        cfg,
        arrays.get("dominant_period_mean"),
        arrays.get("dominant_period_sem"),
        arrays.get("peak_ratio_mean"),
        arrays.get("peak_ratio_sem"),
    )
    return metadata


def start_sweep(cfg, sweep):
    print(f"\n=== Sweep em {sweep.param_name} ===")
    print(
        f"r={cfg.r}, sigma={cfg.sigma}, alpha={cfg.alpha}, "
        f"k={cfg.k}, G={cfg.G}, c={cfg.c}"
    )

    values = sweep.values()
    (
        mean,
        sem,
        traj_strat,
        traj_pay,
        traj_activity,
        vars_mean,
        vars_sem,
        varp_mean,
        varp_sem,
        dominant_freq_mean,
        dominant_freq_sem,
        dominant_power_mean,
        dominant_power_sem,
        dominant_period_mean,
        dominant_period_sem,
        peak_ratio_mean,
        peak_ratio_sem,
    ) = sweep_1d(
        cfg,
        sweep.param_name,
        values,
        observable_fn=obs_fraction,
    )

    metadata = config_metadata(cfg, "sweep_1d", param_name=sweep.param_name)
    arrays_to_save = {
        "values": values,
        "mean": mean,
        "sem": sem,
        "traj_strat": traj_strat,
        "traj_pay": traj_pay,
        "traj_activity": traj_activity,
        "vars_mean": vars_mean,
        "vars_sem": vars_sem,
        "varp_mean": varp_mean,
        "varp_sem": varp_sem,
    }
    if cfg.compute_time_analysis:
        arrays_to_save.update({
            "dominant_freq_mean": dominant_freq_mean,
            "dominant_freq_sem": dominant_freq_sem,
            "dominant_power_mean": dominant_power_mean,
            "dominant_power_sem": dominant_power_sem,
            "dominant_period_mean": dominant_period_mean,
            "dominant_period_sem": dominant_period_sem,
            "peak_ratio_mean": peak_ratio_mean,
            "peak_ratio_sem": peak_ratio_sem,
        })

    output_file = save_npz_result(
        cfg,
        "sweep_1d",
        f"sweep_{sweep.param_name}",
        metadata=metadata,
        **arrays_to_save,
    )
    print(f"Dados salvos em: {output_file}")

    if cfg.make_plots:
        plot_sweep_results(
            values,
            mean,
            sem,
            traj_strat,
            traj_pay,
            traj_activity,
            vars_mean,
            vars_sem,
            varp_mean,
            varp_sem,
            sweep.param_name,
            cfg,
            dominant_period_mean,
            dominant_period_sem,
            peak_ratio_mean,
            peak_ratio_sem,
        )


def main():
    start = time.time()

    sweeps = [
        SweepConfig("r", cfg.r_start, cfg.r_stop, cfg.r_npoints),
        SweepConfig("sigma", cfg.sig_start, cfg.sig_stop, cfg.sig_npoints),
        SweepConfig("alpha", cfg.alpha_start, cfg.alpha_stop, cfg.alpha_npoints),
    ]

    for sweep in sweeps:
        start_sweep(cfg, sweep)

    print(f"Tempo total: {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
