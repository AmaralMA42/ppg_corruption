import numpy as np
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from config import SimulationConfig
from core_simulation import monte_carlo_single, inicia_vizinhos
from plotting import plota_payoff_por_estrategia, plota_todas_amostras, plota_media_com_erro
from utils import save_data


cfg = SimulationConfig()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def _worker(params, total_jog, total_passos, L, seed):
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, L)

    return monte_carlo_single(viz, params, total_jog, total_passos, L, seed)


def run_batches(L, amostras, total_passos, params, base_seed=0):
    total_jog = L * L

    rng = np.random.SeedSequence(base_seed)
    child_seeds = rng.spawn(amostras)

    args = [
        (params, total_jog, total_passos, L, int(s.generate_state(1, dtype=np.uint32)[0]))
        for s in child_seeds
    ]

    nproc = min(cpu_count(), amostras)
    with Pool(nproc) as pool:
        results = pool.starmap(_worker, args)

    estrat_t = np.array([r[0] for r in results]).transpose(1,0,2)
    payavg_t = np.array([r[1] for r in results]).transpose(1,0,2)


    return (
        estrat_t,
        np.mean(estrat_t, axis=1),
        payavg_t,
        np.mean(payavg_t, axis=1),
    )


def main():
    start = time.time()

    estrat_t, estrat_medio, payavg_t, payavg_medio = run_batches(
        cfg.L, cfg.amostras, cfg.total_passos, cfg.simulation_params(), base_seed=cfg.seed
    )

    plota_todas_amostras(estrat_t, estrat_medio, cfg)
    plota_payoff_por_estrategia(payavg_t, payavg_medio, cfg)
    # Media temporal apos a termalizacao.

    steady_state = np.mean(estrat_t[:, :, cfg.passos_media:], axis=2)
    steady_payoff = np.mean(payavg_t[:, :, cfg.passos_media:], axis=2)

    plota_media_com_erro(steady_state, cfg)
    plota_media_com_erro(steady_payoff, cfg, tipo='payoff')

    save_data(steady_state, cfg.simulation_params())

    print(f"Tempo: {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()
