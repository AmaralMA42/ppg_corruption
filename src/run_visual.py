from pathlib import Path
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from config import SimulationConfig
from core_simulation import (
    atualiza_total_estrat,
    calc_frac_and_payoff,
    calcula_payoff,
    inicia_estrategias,
    inicia_vizinhos,
    seed_numba,
)
from plotting import plota_payoff_por_estrategia, plota_todas_amostras, plota_atividade
from utils import config_metadata, load_npz_result, save_npz_result


cfg = SimulationConfig()
amostras = 1

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def monte_carlo_image(viz, params, estrategia, payoff, estrat_t, payavg_t, activity_t, imagegrid, payoff_grid, grad_grid, var_grid, total_jog, total_passos, cfg):
    amo=0
    if cfg.seed is not None:
        seed_numba(cfg.seed)

    L = cfg.L
    framerate = cfg.framerate
    fpsgif = cfg.fpsgif
    passo_filma_inicio = cfg.passo_filma_inicio
    create_snapshot = cfg.create_snapshot
    absorbing_window = cfg.absorbing_window
    # Inicio das estratégias para amostra
    cond_ini = params[6]
    #Variáveis da imagem
    vmin = -0.5  # ajuste baseado no seu modelo
    vmax = 8.5

    # Figura A e B
    figA, (axA1, axA2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    figB, (axB1, axB2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    frames_A = []
    frames_B = []

    imA1 = axA1.imshow(imagegrid, vmin=0, vmax=2, cmap='brg')
    imA2 = axA2.imshow(payoff_grid, cmap='viridis', vmin=vmin, vmax=vmax)

    imB1 = axB1.imshow(grad_grid, cmap='turbo', interpolation='bicubic') # divergente
    imB2 = axB2.imshow(var_grid, cmap='cividis', interpolation='bicubic') # variancia

    figA.colorbar(imA1, ax=axA1)
    figA.colorbar(imA2, ax=axA2)
    figB.colorbar(imB1, ax=axB1)
    figB.colorbar(imB2, ax=axB2)

    inicia_estrategias(estrategia, total_jog, L, cond_ini)
    # inicia os payoffs de cada sítio
    for i in range(total_jog):
        payoff[i] = calcula_payoff(i, estrategia, viz, params)

    # Começo da simulação de Monte-Carlo
    for passo_atual in range(0, total_passos):
        # calcular todas estatísticas da população no passo atual
        frac, pay_strat = calc_frac_and_payoff(estrategia, payoff, total_jog)
        for count2 in range(3):
            payavg_t[count2, amo, passo_atual] = pay_strat[count2]  # C D P
            estrat_t[count2, amo, passo_atual] = frac[count2]

        if create_snapshot:
            if passo_atual % framerate == 0 and passo_atual >= passo_filma_inicio:
                imagegrid[:] = estrategia.reshape(L, L)
                payoff_grid[:] = payoff.reshape(L, L)
                gx, gy = np.gradient(payoff_grid)
                grad_grid[:] = np.sqrt(gx ** 2 + gy ** 2)
                mean = uniform_filter(payoff_grid, size=3)
                mean_sq = uniform_filter(payoff_grid ** 2, size=3)
                var_grid[:] = mean_sq - mean ** 2
                # ========================
                # FIGURA A (estratégia + payoff)
                # ========================
                imA1.set_data(imagegrid)
                imA2.set_data(payoff_grid)

                imB1.set_data(grad_grid)
                imB2.set_data(var_grid)
                imB1.set_clim(0, np.percentile(grad_grid, 99))
                imB2.set_clim(0, np.percentile(var_grid, 99))

                axA1.set_title(f"Estratégia | mcs={passo_atual}")
                axA2.set_title(f"Payoff | mcs={passo_atual}")

                figA.canvas.draw()
                frameA = np.asarray(figA.canvas.buffer_rgba()).copy()
                frames_A.append(Image.fromarray(frameA))

                figA.savefig(
                    FIGURES_DIR / f"str_{passo_atual:05d}.png",
                    bbox_inches='tight',
                    pad_inches=0.02,
                    dpi=150
                )

                # ========================
                # FIGURA B (gradiente + variância)
                # ========================

                axB1.set_title(f"|∇p| | mcs={passo_atual}")
                axB2.set_title(f"Var(p) local | mcs={passo_atual}")

                figB.canvas.draw()
                frameB = np.asarray(figB.canvas.buffer_rgba()).copy()
                frames_B.append(Image.fromarray(frameB))

                figB.savefig(
                    FIGURES_DIR / f"VAR_{passo_atual:05d}.png",
                    bbox_inches='tight',
                    pad_inches=0.02,
                    dpi=150
                )


        # Etapa de atualização da estratégia
        activity = atualiza_total_estrat(estrategia, payoff, viz, params, total_jog)  # atualiza cada rede individualmente
        activity_t[amo, passo_atual] = activity / total_jog

        if absorbing_window > 0 and passo_atual + 1 >= absorbing_window:
            start = passo_atual + 1 - absorbing_window
            if np.sum(activity_t[amo, start:passo_atual + 1]) == 0:
                for future in range(passo_atual + 1, total_passos):
                    estrat_t[:, amo, future] = estrat_t[:, amo, passo_atual]
                    payavg_t[:, amo, future] = payavg_t[:, amo, passo_atual]
                    activity_t[amo, future] = 0
                break

    if create_snapshot and frames_A:
        duration_ms = int(1000 / max(fpsgif, 1))

        frames_A[0].save(
            FIGURES_DIR / "anima_A.gif",
            save_all=True,
            append_images=frames_A[1:],
            duration=duration_ms,
            loop=0,
        )

    if create_snapshot and frames_B:
        duration_ms = int(1000 / max(fpsgif, 1))

        frames_B[0].save(
            FIGURES_DIR / "anima_B.gif",
            save_all=True,
            append_images=frames_B[1:],
            duration=duration_ms,
            loop=0,
        )
    plt.close(figA)
    plt.close(figB)
    return estrat_t, payavg_t, activity_t


def plot_saved_visual(path, cfg=cfg):
    arrays, metadata = load_npz_result(path)
    plota_todas_amostras(arrays["estrat_t"], arrays["estrat_medio_t"], cfg)
    plota_payoff_por_estrategia(arrays["payavg_t"], arrays["payavg_medio_t"], cfg)
    plota_atividade(arrays["activity_t"], arrays["activity_medio_t"], cfg)
    return metadata







def main():
    L = cfg.L
    total_passos = cfg.total_passos
    total_jog = cfg.total_jog
    params = cfg.simulation_params()

    # Definição de variáveis do jogo
    estrategia = np.zeros(total_jog, dtype=int)
    payoff = np.zeros(total_jog)
    estrat_t = np.zeros((3, amostras, total_passos))
    payavg_t = np.zeros((3, amostras, total_passos))
    activity_t = np.zeros((amostras, total_passos))
    # Definição de vizinhos
    viz = np.zeros((total_jog, 4), dtype=int)  # matriz contendo os vizinhos 0=cima,1=direita,2=baixo,3=esquerda
    imagegrid = np.zeros((L, L))
    payoff_grid = np.zeros((L, L))
    grad_grid = np.zeros((L, L))
    var_grid = np.zeros((L, L))
   # 0 Copera 1 Deserta 2 CorruPto
    inicia_vizinhos(viz, total_jog, L)


# SIMULAÇÂO!!!
    estrat_t, payavg_t, activity_t = monte_carlo_image(viz, params, estrategia, payoff, estrat_t, payavg_t, activity_t, imagegrid, payoff_grid, grad_grid, var_grid, total_jog, total_passos, cfg)

    estrat_medio_t = np.mean(estrat_t, axis=1)
    payavg_medio_t = np.mean(payavg_t, axis=1)
    activity_medio_t = np.mean(activity_t, axis=0)

    plota_todas_amostras(estrat_t, estrat_medio_t,cfg)
    plota_payoff_por_estrategia(payavg_t, payavg_medio_t,cfg)
    plota_atividade(activity_t, activity_medio_t, cfg)

    metadata = config_metadata(cfg, "visual")
    output_file = save_npz_result(
        cfg,
        "visual",
        "visual",
        metadata=metadata,
        estrat_t=estrat_t,
        estrat_medio_t=estrat_medio_t,
        payavg_t=payavg_t,
        payavg_medio_t=payavg_medio_t,
        activity_t=activity_t,
        activity_medio_t=activity_medio_t,
    )
    print(f"Dados salvos em: {output_file}")

if __name__ == "__main__":
    main()
