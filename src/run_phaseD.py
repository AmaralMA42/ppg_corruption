import numpy as np
import time
from dataclasses import replace

from config import SimulationConfig
from run_sweep import run_simulation
from plotting import plot_heatmap_3
from utils import config_metadata, load_npz_result, save_npz_result

cfg = SimulationConfig()

def sweep_2d_all(cfg, param_x, values_x, param_y, values_y):
    Z = np.zeros((len(values_x), len(values_y), 3))  # C, D, P
    Z_var = np.zeros((len(values_x), len(values_y), 3))  # C, D, P

    for i, vx in enumerate(values_x):
        for j, vy in enumerate(values_y):
            print(f"{param_x}={vx}, {param_y}={vy}")

            cfg_v = replace(cfg, **{param_x: vx, param_y: vy})

            estrat_t, _, _, _, _, _, _ = run_simulation(cfg_v)

            steady = np.mean(
                estrat_t[:, :, cfg_v.passos_media:], axis=2
            )  # (3, amostras
            Z[i, j, :] = np.mean(steady, axis=1)  # média nas amostras

            var = np.var(
                estrat_t[:, :, cfg_v.passos_media:], axis=2, ddof=1
            )  # (3, amostras)

            Z_var[i, j, :] = np.mean(var, axis=1)

    return Z, Z_var

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

def run_phase_pair(cfg, param_x, param_y):
    values_x = phase_values(cfg, param_x)
    values_y = phase_values(cfg, param_y)

    Z, Z_var = sweep_2d_all(cfg, param_x, values_x, param_y, values_y)

    metadata = config_metadata(
        cfg,
        "phase_2d",
        param_x=param_x,
        param_y=param_y,
    )
    output_file = save_npz_result(
        cfg,
        "phase_2d",
        f"phase_{param_x}_{param_y}",
        metadata=metadata,
        values_x=values_x,
        values_y=values_y,
        Z=Z,
        Z_var=Z_var,
    )
    print(f"Dados salvos em: {output_file}")

    plot_heatmap_3(values_x, values_y, Z, param_x, param_y)
    plot_heatmap_3(values_x, values_y, Z_var, param_x, f"{param_y} variance", "variance")

    return output_file

def plot_saved_phase(path):
    arrays, metadata = load_npz_result(path)
    param_x = metadata["param_x"]
    param_y = metadata["param_y"]

    plot_heatmap_3(arrays["values_x"], arrays["values_y"], arrays["Z"], param_x, param_y)
    plot_heatmap_3(
        arrays["values_x"],
        arrays["values_y"],
        arrays["Z_var"],
        param_x,
        f"{param_y} variance",
        "variance",
    )
    return metadata

def main():
    start = time.time()

    for param_x, param_y in selected_phase_pairs(cfg):
        run_phase_pair(cfg, param_x, param_y)

    print(f"Tempo: {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
