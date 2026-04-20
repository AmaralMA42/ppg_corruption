import matplotlib.pyplot as plt
import numpy as np
from core_simulation import run_simulation
from plotting import plot_all_trajectories

def main():
    # parâmetros
    L = 100
    amostras = 20
    total_passos = 5000
    passos_media = int(0.9 * total_passos)

    r = 3.3
    G = 5
    c = 1
    sigma = 1.05
    alpha = 0.3
    k = 0.1

    params = np.array([r, G, c, sigma, k, alpha])

    estrat_t, estrat_medio = run_simulation(
        L, amostras, total_passos, params, cond_ini=0
    )

    plot_all_trajectories(estrat_t, estrat_medio)

if __name__ == "__main__":
    main()