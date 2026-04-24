import matplotlib.pyplot as plt
from config import SimulationConfig
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# @jit(nopython=True, fastmath=True,  parallel=False) Numba não compatível com impressão aparentemente
def imprime_dados(estrat_medio_t, total_passos, start_time,cfg):
    L = cfg.L
    amostras = cfg.amostras
    total_passos = cfg.total_passos
    passos_media = cfg.passos_media
    seed = cfg.seed
    k = cfg.k
    r = cfg.r
    G = cfg.G
    c = cfg.c
    sigma = cfg.sigma
    alpha = cfg.alpha
    creat_snapshot = cfg.create_snapshot
    deldata = cfg.deldata0
    framerate = cfg.framerate
    fpsgif = cfg.fpsgif
    passo_filma_inicio = cfg.passo_filma_inicio
    cond_ini = cfg.cond_ini
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    FIGURES_DIR = ROOT_DIR / "figures"
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    arquivo2 = open(DATA_DIR / f"subpop_r{r}_sigma{sigma}.dat", "w+")
    for cont in range(0, total_passos):
        arquivo2.write(f"{cont} {estrat_medio_t[0, cont]:.4f} {estrat_medio_t[1, cont]:.4f} "
                       f"{estrat_medio_t[2, cont]:.4f} \n")
    print(f"----- {(time.time() - start_time):.4f} seconds or {((time.time() - start_time) / 60.0):.4f} minutes-----")
    arquivo2.close()


def plota_dados(estrat_medio_t,cfg):

    plt.figure()
    plt.title('Evolução média das subpopulações')
    plt.xlabel('Número de Passos')
    plt.ylabel(r'$<\rho >$')
    plt.plot(estrat_medio_t[0, :], label='C')
    plt.plot(estrat_medio_t[1, :], label='D')
    plt.plot(estrat_medio_t[2, :], label='P')
    plt.ylim(0, 1)
    # salvar parametros nos graficos
    param_text = (
        f"L={cfg.L}\n"
        f"r={cfg.r}\n"
        f"$\\sigma$={cfg.sigma}\n"
        f"$\\alpha$={cfg.alpha}\n"
        f"k={cfg.k}"
    )

    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    plt.legend()
    plt.show()

def plota_todas_amostras(estrat_t, estrat_medio_t,cfg):
    plt.figure()

    # trajetórias individuais
    for i in range(estrat_t.shape[1]):
        plt.plot(estrat_t[0, i, :], alpha=0.2, linestyle='-')
        plt.plot(estrat_t[1, i, :], alpha=0.2, linestyle='--')
        plt.plot(estrat_t[2, i, :], alpha=0.2, linestyle=':')

    # média (destacada)
    plt.plot(estrat_medio_t[0, :], label='C (média)', linewidth=2., color='blue')
    plt.plot(estrat_medio_t[1, :], label='D (média)', linewidth=2., color='red')
    plt.plot(estrat_medio_t[2, :], label='P (média)', linewidth=2., color='green')

    plt.ylim(0, 1)
    plt.xlabel('Número de Passos')
    plt.ylabel(r'$\langle \rho \rangle$')

    # parâmetros discretos
    param_text = (
        f"L={cfg.L}\n"
        f"r={cfg.r}\n"
        f"$\\sigma$={cfg.sigma}\n"
        f"$\\alpha$={cfg.alpha}\n"
        f"k={cfg.k}"
    )

    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )

    plt.legend()
    plt.show()

def plota_payoff_por_estrategia(payavg_t, payavg_medio_t, cfg):
    plt.figure()

    # trajetórias individuais
    for i in range(payavg_t.shape[1]):
        plt.plot(payavg_t[0, i, :], alpha=0.2, linestyle='-')
        plt.plot(payavg_t[1, i, :], alpha=0.2, linestyle='--')
        plt.plot(payavg_t[2, i, :], alpha=0.2, linestyle=':')

    # média (destacada)
    plt.plot(payavg_medio_t[0, :], label='C payoff', linewidth=2.)
    plt.plot(payavg_medio_t[1, :], label='D payoff', linewidth=2.)
    plt.plot(payavg_medio_t[2, :], label='P payoff', linewidth=2.)

    plt.xlabel('Número de Passos')
    plt.ylabel(r'$\langle \Pi \rangle$')

    param_text = (
        f"L={cfg.L}\n"
        f"r={cfg.r}\n"
        f"$\\sigma$={cfg.sigma}\n"
        f"$\\alpha$={cfg.alpha}\n"
        f"k={cfg.k}"
    )

    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )

    plt.legend()
    plt.show()



def plota_media_com_erro(steady_state, cfg, tipo='estraegia'):
    amostras = steady_state.shape[1]

    mean = np.mean(steady_state, axis=1)
    std = np.std(steady_state, axis=1, ddof=1)
    sem = std / np.sqrt(amostras)

    labels = ["C", "D", "P"]
    x = np.arange(3)

    plt.figure(figsize=(6,4))
    plt.errorbar(x, mean, yerr=sem, fmt='o', capsize=5)
    plt.xticks(x, labels)
    if tipo == 'estrategia':
        plt.ylabel("Fração média (steady state)")
        plt.title("Média + erro entre amostras")
        plt.ylim(0, 1)
    elif tipo == 'payoff' :
        plt.ylabel("payoff médio (steady state)")
        plt.title("Média + erro entre amostras")
    plt.grid(alpha=0.3)
    plt.tight_layout()

#    plt.savefig(cfg.figures_dir / "media_erro.png", dpi=150)
    plt.show()


def plot_vs_r(r_values, mean, sem, output_dir,cfg):
    labels = ["C", "D", "P"]
    colors = ["blue", "red", "green"]
    param_text = (
        f"L={cfg.L}\n"
        f"$\\sigma$={cfg.sigma}\n"
        f"$\\alpha$={cfg.alpha}\n"
        f"k={cfg.k}"
    )

    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    plt.figure(figsize=(6,4))

    for i in range(3):
        plt.errorbar(
            r_values,
            mean[:, i],
            yerr=sem[:, i],
            label=labels[i],
            capsize=3,
            color=colors[i]
        )

    plt.xlabel("r")
    plt.ylabel("Fração média")
    plt.title("Dependência em r")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "vs_r.png", dpi=150)
    plt.show()

def plot_sweep_1d(x, mean, sem, labels, xlabel, cfg):
    colors = ["blue", "red", "green"]
    param_text = (
        f"L={cfg.L}\n"
        f"$\\sigma$={cfg.sigma}\n"
        f"$\\alpha$={cfg.alpha}\n"
        f"k={cfg.k}"
    )

    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    plt.figure(figsize=(6,4))

    for i, label in enumerate(labels):
        plt.errorbar(x, mean[:, i], yerr=sem[:, i], label=label, capsize=3)

    plt.title(f"Dependencia em ={xlabel}\n")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel("observável médio")
    plt.show()

def plot_trajectories_vs_time(values, traj, param_name, ylabel, step=1):
    """
    traj: (n_values, 3, T)
    """

    T = traj.shape[2]
    x = np.arange(T)

    labels = ["C", "D", "P"]
    colors = ["blue", "red", "green"]

    plt.figure(figsize=(8,5))

    for i, val in enumerate(values[::step]):
        for s in range(3):
            plt.plot(
                x,
                traj[i, s],
                color=colors[s],
                alpha=0.3,
                label=f"{labels[s]}, {param_name}={val:.2f}" if i == 0 else None
            )

    plt.xlabel("Monte Carlo step")
    plt.ylabel(ylabel)
    plt.title(f"Evolução temporal (varrendo {param_name})")

    # legenda só uma vez
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_variance_vs_param(values, var_mean, var_sem, labels, param_name):
    plt.figure(figsize=(6,4))

    for i in range(3):
        plt.errorbar(
            values,
            var_mean[:, i],
            yerr=var_sem[:, i],
            label=labels[i],
            capsize=3
        )

    plt.xlabel(param_name)
    plt.ylabel("Variância temporal")
    plt.title("Flutuações temporais (indicador de ciclos)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()