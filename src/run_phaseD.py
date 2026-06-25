import numpy as np
import time
from dataclasses import replace

from config import SimulationConfig
from run_sweep import run_simulation
from run_sampling import create_sample_pool
from plotting import plot_heatmap_3
from time_analysis import analyze_strategy_scalar_metrics
from utils import config_metadata, load_npz_result, save_npz_result

cfg = SimulationConfig()

def sweep_2d_all(cfg, param_x, values_x, param_y, values_y, pool=None):
    Z = np.zeros((len(values_x), len(values_y), 3))  # C, D, P
    Z_var = np.zeros((len(values_x), len(values_y), 3))  # C, D, P
    Z_var_sem = np.full((len(values_x), len(values_y), 3), np.nan)
    Z_period = np.full((len(values_x), len(values_y), 3), np.nan)
    Z_period_sem = np.full((len(values_x), len(values_y), 3), np.nan)
    Z_peak_ratio = np.full((len(values_x), len(values_y), 3), np.nan)
    Z_peak_ratio_sem = np.full((len(values_x), len(values_y), 3), np.nan)

    for i, vx in enumerate(values_x):
        for j, vy in enumerate(values_y):
            print(f"{param_x}={vx}, {param_y}={vy}")

            cfg_v = replace(cfg, **{param_x: vx, param_y: vy})

            estrat_t, _, _, _, _, _, absorbed_at = run_simulation(cfg_v, pool=pool)

            steady = np.mean(
                estrat_t[:, :, cfg_v.passos_media:], axis=2
            )  # (3, amostras
            Z[i, j, :] = np.mean(steady, axis=1)  # média nas amostras

            var = np.var(
                estrat_t[:, :, cfg_v.passos_media:], axis=2, ddof=1
            )  # (3, amostras)

            Z_var[i, j, :] = np.mean(var, axis=1)

            if cfg_v.compute_time_analysis:
                scalar_analysis = analyze_strategy_scalar_metrics(
                    estrat_t,
                    start=cfg_v.passos_media,
                    absorbed_at=absorbed_at,
                )
                Z_var[i, j, :] = scalar_analysis["variance_mean"]
                Z_var_sem[i, j, :] = scalar_analysis["variance_sem"]
                Z_period[i, j, :] = scalar_analysis["dominant_period_mean"]
                Z_period_sem[i, j, :] = scalar_analysis["dominant_period_sem"]
                Z_peak_ratio[i, j, :] = scalar_analysis["peak_ratio_mean"]
                Z_peak_ratio_sem[i, j, :] = scalar_analysis["peak_ratio_sem"]

    return Z, Z_var, Z_var_sem, Z_period, Z_period_sem, Z_peak_ratio, Z_peak_ratio_sem

def phase_values(cfg, param_name):
    if param_name == "r":
        return np.linspace(cfg.r_start, cfg.r_stop, cfg.r_npoints)
    if param_name == "alpha":
        return np.linspace(cfg.alpha_start, cfg.alpha_stop, cfg.alpha_npoints)
    if param_name == "sigma":
        return np.linspace(cfg.sig_start, cfg.sig_stop, cfg.sig_npoints)
    raise ValueError(f"Parametro de fase desconhecido: {param_name}")

def selected_phase_pairs(cfg):
    phaseport = cfg.phaseport
    phase_pairs = {
        "r_alpha": ("r", "alpha"),
        "r_sigma": ("r", "sigma"),
        "alpha_sigma": ("alpha", "sigma"),
    }

    if phaseport == "all":
        return list(phase_pairs.values())
    if phaseport not in phase_pairs:
        valid = ", ".join([*phase_pairs.keys(), "all"])
        raise ValueError(f"phaseport invalido: {phaseport}. Use um de: {valid}")
    return [phase_pairs[phaseport]]

def run_phase_pair(cfg, param_x, param_y, pool=None):
    if pool is None:
        with create_sample_pool(cfg.amostras) as local_pool:
            return run_phase_pair(cfg, param_x, param_y, pool=local_pool)

    values_x = phase_values(cfg, param_x)
    values_y = phase_values(cfg, param_y)

    Z, Z_var, Z_var_sem, Z_period, Z_period_sem, Z_peak_ratio, Z_peak_ratio_sem = sweep_2d_all(
        cfg,
        param_x,
        values_x,
        param_y,
        values_y,
        pool=pool,
    )

    metadata = config_metadata(
        cfg,
        "phase_2d",
        param_x=param_x,
        param_y=param_y,
    )
    arrays_to_save = {
        "values_x": values_x,
        "values_y": values_y,
        "Z": Z,
        "Z_var": Z_var,
    }
    if cfg.compute_time_analysis:
        arrays_to_save.update({
            "Z_var_sem": Z_var_sem,
            "Z_period": Z_period,
            "Z_period_sem": Z_period_sem,
            "Z_peak_ratio": Z_peak_ratio,
            "Z_peak_ratio_sem": Z_peak_ratio_sem,
        })

    output_file = save_npz_result(
        cfg,
        "phase_2d",
        f"phase_{param_x}_{param_y}",
        metadata=metadata,
        **arrays_to_save,
    )
    print(f"Dados salvos em: {output_file}")

    if cfg.make_plots:
        plot_heatmap_3(values_x, values_y, Z, param_x, param_y, cfg=cfg)
        plot_heatmap_3(values_x, values_y, Z_var, param_x, param_y, "variance", cfg=cfg)
        if cfg.compute_time_analysis:
            plot_heatmap_3(values_x, values_y, Z_period, param_x, param_y, "period", cfg=cfg)
            plot_heatmap_3(values_x, values_y, Z_peak_ratio, param_x, param_y, "peak_ratio", cfg=cfg)

    return output_file

def plot_saved_phase(path, cfg=cfg):
    arrays, metadata = load_npz_result(path)
    param_x = metadata["param_x"]
    param_y = metadata["param_y"]

    plot_heatmap_3(arrays["values_x"], arrays["values_y"], arrays["Z"], param_x, param_y, cfg=cfg)
    plot_heatmap_3(
        arrays["values_x"],
        arrays["values_y"],
        arrays["Z_var"],
        param_x,
        param_y,
        "variance",
        cfg=cfg,
    )
    if "Z_period" in arrays:
        plot_heatmap_3(
            arrays["values_x"],
            arrays["values_y"],
            arrays["Z_period"],
            param_x,
            param_y,
            "period",
            cfg=cfg,
        )
    if "Z_peak_ratio" in arrays:
        plot_heatmap_3(
            arrays["values_x"],
            arrays["values_y"],
            arrays["Z_peak_ratio"],
            param_x,
            param_y,
            "peak_ratio",
            cfg=cfg,
        )
    return metadata

def main():
    start = time.time()

    with create_sample_pool(cfg.amostras) as pool:
        for param_x, param_y in selected_phase_pairs(cfg):
            run_phase_pair(cfg, param_x, param_y, pool=pool)

    print(f"Tempo: {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
