import numpy as np
import time
from core_simulation import run_simulation
from utils import save_data
from plotting import plot_all_trajectories

def thermal_average(estrat_t, passos_media):
    return np.mean(estrat_t[:, :, passos_media:], axis=2)

def main():
    # parâmetros
    L = 100
    amostras = 20
    total_passos = 1000
    passos_media = int(0.9 * total_passos)

    r = 3.3
    G = 5
    c = 1
    sigma = 1.05
    alpha = 0.4
    k = 0.1

    params = np.array([r, G, c, sigma, k, alpha])

    start = time.time()

    estrat_t, estrat_medio = run_simulation(
        L, amostras, total_passos, params, cond_ini=0
    )
    plot_all_trajectories(estrat_t, estrat_medio)
    # 🔥 média pós-termalização
    steady_state = thermal_average(estrat_t, passos_media)

    save_data(steady_state, params)

    print(f"Tempo: {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()