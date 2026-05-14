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
)
from plotting import plota_payoff_por_estrategia, plota_todas_amostras


cfg = SimulationConfig()
#amostras = cfg.amostras
amostras = 1

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# todo importante, unificar monte carlo pra imagens e para simulações
#@jit(nopython=True, fastmath=True,  parallel=False)
def monte_carlo_image(viz, params, estrategia, payoff, estrat_t, payavg_t, imagegrid, payoff_grid, grad_grid, var_grid, total_jog, total_passos, cfg):
    amo=0
    L = cfg.L
    framerate = cfg.framerate
    fpsgif = cfg.fpsgif
    passo_filma_inicio = cfg.passo_filma_inicio
    create_snapshot = cfg.create_snapshot
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

##    figA.colorbar(axA1.imshow(imagegrid, vmin=0, vmax=2, cmap='brg'), ax=axA1)
##    figA.colorbar(axA2.imshow(payoff_grid, cmap='viridis', vmin=vmin, vmax=vmax), ax=axA2)

##    figB.colorbar(axB1.imshow(grad_grid, cmap='inferno'), ax=axB1)
##    figB.colorbar(axB2.imshow(var_grid, cmap='magma'), ax=axB2)

#    im1 = ax1.imshow(imagegrid, vmin=0, vmax=2, cmap='brg', interpolation='none')
#    im2 = ax2.imshow(payoff_grid, cmap='viridis', vmin=vmin, vmax=vmax)
    imA1 = axA1.imshow(imagegrid, vmin=0, vmax=2, cmap='brg')
    imA2 = axA2.imshow(payoff_grid, cmap='viridis', vmin=vmin, vmax=vmax)

    imB1 = axB1.imshow(grad_grid, cmap='turbo', interpolation='bicubic') # divergente
    imB2 = axB2.imshow(var_grid, cmap='cividis', interpolation='bicubic') # variancia

    figA.colorbar(imA1, ax=axA1)
    figA.colorbar(imA2, ax=axA2)
    figB.colorbar(imB1, ax=axB1)
    figB.colorbar(imB2, ax=axB2)
#    plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
#    plt.colorbar(im2, ax=ax2)
#    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.92, wspace=0.22)



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
#                var_grid[:] =  (payoff_grid - payoff_grid.mean())**2
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
        atualiza_total_estrat(estrategia, payoff, viz, params, total_jog)  # atualiza cada rede individualmente

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
    return estrat_t, payavg_t








def main():
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    L = cfg.L
    total_passos = cfg.total_passos
    total_jog = cfg.total_jog
    params = cfg.simulation_params()

    # Definição de variáveis do jogo
    estrategia = np.zeros(total_jog, dtype=int)
    payoff = np.zeros(total_jog)
    estrat_t = np.zeros((3, amostras, total_passos))
    payavg_t = np.zeros((3, amostras, total_passos))
    # Definição de vizinhos
    viz = np.zeros((total_jog, 4), dtype=int)  # matriz contendo os vizinhos 0=cima,1=direita,2=baixo,3=esquerda
    imagegrid = np.zeros((L, L))
    payoff_grid = np.zeros((L, L))
    grad_grid = np.zeros((L, L))
    var_grid = np.zeros((L, L))
   # 0 Copera 1 Deserta 2 CorruPto
    inicia_vizinhos(viz, total_jog, L)


# SIMULAÇÂO!!!
    estrat_t, payavg_t = monte_carlo_image(viz, params, estrategia, payoff, estrat_t, payavg_t, imagegrid, payoff_grid, grad_grid, var_grid, total_jog, total_passos, cfg)

    estrat_medio_t = np.mean(estrat_t, axis=1)
    payavg_medio_t = np.mean(payavg_t, axis=1)

    plota_todas_amostras(estrat_t, estrat_medio_t,cfg)
    plota_payoff_por_estrategia(payavg_t, payavg_medio_t,cfg)
#    imprime_dados(estrat_medio_t, total_passos, start_time,cfg)


#    plota_todas_amostras(estrat_t, estrat_medio_t)
#    plota_payoff_por_estrategia(payavg_t, payavg_medio_t)
#    imprime_dados(estrat_medio_t, total_passos, start_time)

if __name__ == "__main__":
    main()
