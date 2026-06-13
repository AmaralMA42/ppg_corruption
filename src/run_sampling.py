import numpy as np
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from config import SimulationConfig
from core_simulation import monte_carlo_single, inicia_vizinhos
from plotting import (
    plota_atividade,
    plota_autocorrelacao,
    plota_frequencia_dominante,
    plota_media_com_erro,
    plota_payoff_por_estrategia,
    plota_potencia_dominante,
    plota_todas_amostras,
    plota_variancia_temporal,
)
from time_analysis import analyze_strategy_timeseries
from utils import config_metadata, load_npz_result, save_npz_result


cfg = SimulationConfig()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def _worker(params, total_jog, total_passos, L, seed, absorbing_window):
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, L)

    return monte_carlo_single(
        viz,
        params,
        total_jog,
        total_passos,
        L,
        seed,
        absorbing_window=absorbing_window,
    )


def run_batches(L, amostras, total_passos, params, base_seed=0, absorbing_window=0):
    total_jog = L * L

    rng = np.random.SeedSequence(base_seed)
    child_seeds = rng.spawn(amostras)

    args = [
        (
            params,
            total_jog,
            total_passos,
            L,
            int(s.generate_state(1, dtype=np.uint32)[0]),
            absorbing_window,
        )
        for s in child_seeds
    ]

    nproc = min(cpu_count(), amostras)
    with Pool(nproc) as pool:
        results = pool.starmap(_worker, args)

    estrat_t = np.array([r[0] for r in results]).transpose(1,0,2)
    payavg_t = np.array([r[1] for r in results]).transpose(1,0,2)
    activity_t = np.array([r[2] for r in results])
    absorbed_at = np.array([r[3] for r in results], dtype=np.int64)


    return (
        estrat_t,
        np.mean(estrat_t, axis=1),
        payavg_t,
        np.mean(payavg_t, axis=1),
        activity_t,
        np.mean(activity_t, axis=0),
        absorbed_at,
    )

def plot_saved_sampling(path, cfg=cfg):
    arrays, metadata = load_npz_result(path)
    plota_todas_amostras(arrays["estrat_t"], arrays["estrat_medio"], cfg)
    plota_payoff_por_estrategia(arrays["payavg_t"], arrays["payavg_medio"], cfg)
    plota_atividade(arrays["activity_t"], arrays["activity_medio"], cfg)
    plota_media_com_erro(arrays["steady_state"], cfg)
    plota_media_com_erro(arrays["steady_payoff"], cfg, tipo="payoff")
    if "temporal_variance_samples" in arrays:
        plota_variancia_temporal(
            arrays["temporal_variance_samples"],
            arrays["temporal_variance_mean"],
            cfg,
        )
    if "autocorr_mean" in arrays:
        plota_autocorrelacao(arrays["autocorr_mean"], cfg)
    if "dominant_freq_samples" in arrays:
        plota_frequencia_dominante(
            arrays["dominant_freq_samples"],
            arrays["dominant_freq_mean"],
            cfg,
        )
    if "dominant_power_samples" in arrays:
        plota_potencia_dominante(
            arrays["dominant_power_samples"],
            arrays["dominant_power_mean"],
            cfg,
        )
    return metadata


def main():
    start = time.time()
    params = cfg.simulation_params()

    estrat_t, estrat_medio, payavg_t, payavg_medio, activity_t, activity_medio, absorbed_at = run_batches(
        cfg.L,
        cfg.amostras,
        cfg.total_passos,
        params,
        base_seed=cfg.seed,
        absorbing_window=cfg.absorbing_window,
    )

    plota_todas_amostras(estrat_t, estrat_medio, cfg)
    plota_payoff_por_estrategia(payavg_t, payavg_medio, cfg)
    plota_atividade(activity_t, activity_medio, cfg)
    # Media temporal apos a termalizacao.

    steady_state = np.mean(estrat_t[:, :, cfg.passos_media:], axis=2)
    steady_payoff = np.mean(payavg_t[:, :, cfg.passos_media:], axis=2)
    time_analysis = analyze_strategy_timeseries(
        estrat_t,
        start=cfg.passos_media,
        absorbed_at=absorbed_at,
    )

    plota_media_com_erro(steady_state, cfg, )
    plota_media_com_erro(steady_payoff, cfg, tipo='payoff')
    plota_variancia_temporal(
        time_analysis["variance_samples"],
        time_analysis["variance_mean"],
        cfg,
    )
    plota_autocorrelacao(time_analysis["autocorr_mean"], cfg)
    plota_frequencia_dominante(
        time_analysis["dominant_freq_samples"],
        time_analysis["dominant_freq_mean"],
        cfg,
    )
    plota_potencia_dominante(
        time_analysis["dominant_power_samples"],
        time_analysis["dominant_power_mean"],
        cfg,
    )

    metadata = config_metadata(cfg, "sampling")
    output_file = save_npz_result(
        cfg,
        "sampling",
        "sampling",
        metadata=metadata,
        estrat_t=estrat_t,
        estrat_medio=estrat_medio,
        payavg_t=payavg_t,
        payavg_medio=payavg_medio,
        activity_t=activity_t,
        activity_medio=activity_medio,
        absorbed_at=absorbed_at,
        steady_state=steady_state,
        steady_payoff=steady_payoff,
        temporal_variance_samples=time_analysis["variance_samples"],
        temporal_variance_mean=time_analysis["variance_mean"],
        autocorr_samples=time_analysis["autocorr_samples"],
        autocorr_mean=time_analysis["autocorr_mean"],
        dominant_freq_samples=time_analysis["dominant_freq_samples"],
        dominant_freq_mean=time_analysis["dominant_freq_mean"],
        dominant_power_samples=time_analysis["dominant_power_samples"],
        dominant_power_mean=time_analysis["dominant_power_mean"],
    )
    print(f"Dados salvos em: {output_file}")

    print(f"Tempo: {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()
