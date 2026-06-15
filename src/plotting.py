import matplotlib.pyplot as plt
import numpy as np

STRATEGY_LABELS = ["C", "D", "P"]

STRATEGY_COLORS = {
    "C": "Blues",
    "D": "Reds",
    "P": "Greens",
}

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


def plota_atividade(activity_t, activity_medio_t, cfg):
    plt.figure()

    for i in range(activity_t.shape[0]):
        plt.plot(activity_t[i, :], alpha=0.25, color="gray")

    plt.plot(activity_medio_t, label="A(t) media", linewidth=2.0, color="black")

    param_text = (
        f"L={cfg.L}\n"
        f"r={cfg.r}\n"
        f"$\\sigma$={cfg.sigma}\n"
        f"$\\alpha$={cfg.alpha}\n"
        f"k={cfg.k}"
    )

    plt.text(
        0.98, 0.98, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )

    plt.xlabel("Numero de Passos")
    plt.ylabel("A(t) / N")
    plt.title("Atividade normalizada por MCS")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plota_variancia_temporal(variance_samples, variance_mean, cfg):
    labels = ["C", "D", "P"]
    colors = ["blue", "red", "green"]
    markers = ["o", "s", "^"]
    x = np.arange(3)

    plt.figure(figsize=(6,4))

    for i in range(3):
        samples = variance_samples[i, :]
        valid = np.isfinite(samples)
        if np.any(valid):
            jitter = np.linspace(-0.08, 0.08, np.sum(valid))
            plt.scatter(
                np.full(np.sum(valid), x[i]) + jitter,
                samples[valid],
                alpha=0.35,
                color=colors[i],
                marker=markers[i],
            )
        if np.isfinite(variance_mean[i]):
            plt.scatter(x[i], variance_mean[i], color=colors[i], edgecolor="black", marker=markers[i], s=80, zorder=3)

    plt.xticks(x, labels)
    plt.ylabel("Variancia temporal")
    plt.title("Variancia temporal pos-transiente")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plota_autocorrelacao(autocorr_mean, cfg):
    labels = ["C", "D", "P"]
    colors = ["blue", "red", "green"]
    tau = np.arange(autocorr_mean.shape[1])

    plt.figure(figsize=(7,4))

    for i in range(3):
        valid = np.isfinite(autocorr_mean[i, :])
        if np.any(valid):
            plt.plot(tau[valid], autocorr_mean[i, valid], label=labels[i], color=colors[i])

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("Lag temporal tau (MCS)")
    plt.ylabel("C(tau) normalizada")
    plt.title("Autocorrelacao temporal media")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plota_periodo_dominante(period_samples, period_mean, cfg):
    labels = ["C", "D", "P"]
    colors = ["blue", "red", "green"]
    markers = ["D", "P", "X"]
    x = np.arange(3)

    plt.figure(figsize=(6,4))

    for i in range(3):
        samples = period_samples[i, :]
        valid = np.isfinite(samples)
        if np.any(valid):
            jitter = np.linspace(-0.08, 0.08, np.sum(valid))
            plt.scatter(
                np.full(np.sum(valid), x[i]) + jitter,
                samples[valid],
                alpha=0.35,
                color=colors[i],
                marker=markers[i],
            )
        if np.isfinite(period_mean[i]):
            plt.scatter(x[i], period_mean[i], color=colors[i], edgecolor="black", marker=markers[i], s=80, zorder=3)

    plt.xticks(x, labels)
    plt.ylabel("Periodo dominante (MCS)")
    plt.title("Periodo dominante por estrategia")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plota_peak_ratio(peak_ratio_samples, peak_ratio_mean, cfg):
    labels = ["C", "D", "P"]
    colors = ["blue", "red", "green"]
    markers = ["v", "<", ">"]
    x = np.arange(3)

    plt.figure(figsize=(6,4))

    for i in range(3):
        samples = peak_ratio_samples[i, :]
        valid = np.isfinite(samples)
        if np.any(valid):
            jitter = np.linspace(-0.08, 0.08, np.sum(valid))
            plt.scatter(
                np.full(np.sum(valid), x[i]) + jitter,
                samples[valid],
                alpha=0.35,
                color=colors[i],
                marker=markers[i],
            )
        if np.isfinite(peak_ratio_mean[i]):
            plt.scatter(x[i], peak_ratio_mean[i], color=colors[i], edgecolor="black", marker=markers[i], s=80, zorder=3)

    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Peak ratio")
    plt.title("Concentracao espectral no pico dominante")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plota_media_com_erro(steady_state, cfg, tipo='estrategia'):
    amostras = steady_state.shape[1]

    mean = np.mean(steady_state, axis=1)
    if amostras <= 1:
        sem = np.zeros_like(mean)
    else:
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

    plt.figure(figsize=(6,4))
    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )

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
    param_text = (
        f"L={cfg.L}\n"
        f"$\\sigma$={cfg.sigma}\n"
        f"$\\alpha$={cfg.alpha}\n"
        f"k={cfg.k}"
    )

    plt.figure(figsize=(6,4))
    plt.text(
        0.98, 0.02, param_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )

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

    step = max(int(step), 1)
    T = traj.shape[2]
    x = np.arange(T)

    labels = ["C", "D", "P"]
    colors = ["blue", "red", "green"]

    plt.figure(figsize=(8,5))

    for idx in range(0, len(values), step):
        val = values[idx]
        for s in range(3):
            plt.plot(
                x,
                traj[idx, s],
                color=colors[s],
                alpha=0.3,
                label=f"{labels[s]}, {param_name}={val:.2f}" if idx == 0 else None
            )

    plt.xlabel("Monte Carlo step")
    plt.ylabel(ylabel)
    plt.title(f"Evolução temporal (varrendo {param_name})")

    # legenda só uma vez
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_activity_trajectories_vs_time(values, traj_activity, param_name, step=1):
    step = max(int(step), 1)
    T = traj_activity.shape[1]
    x = np.arange(T)

    plt.figure(figsize=(8,5))

    for idx in range(0, len(values), step):
        plt.plot(
            x,
            traj_activity[idx],
            alpha=0.45,
            label=f"{param_name}={values[idx]:.2f}" if idx == 0 else None
        )

    plt.xlabel("Monte Carlo step")
    plt.ylabel("A(t) / N")
    plt.title(f"Atividade normalizada temporal (varrendo {param_name})")
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


def plot_metric_vs_param(values, metric_mean, metric_sem, labels, param_name, ylabel, title, ylim=None):
    plt.figure(figsize=(6,4))

    for i in range(3):
        plt.errorbar(
            values,
            metric_mean[:, i],
            yerr=metric_sem[:, i],
            label=labels[i],
            capsize=3
        )

    plt.xlabel(param_name)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_heatmap(X, Y, Z, xlabel, ylabel, title=""):
    plt.figure(figsize=(6, 5))

    im = plt.imshow(
        Z.T,
        origin='lower',
        aspect='auto',
        extent=[X[0], X[-1], Y[0], Y[-1]],
    )

    plt.colorbar(im, label=r"$\rho_C$")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_heatmap_3(X, Y, Z, xlabel, ylabel, mode="fraction"):
    labels = ["C", "D", "P"]
    if mode == "fraction":
        cmaps = ["Blues", "Reds", "Greens"]
        vmin, vmax = 0, 1
    elif mode == "variance":
        cmaps = ["magma", "magma", "magma"]
        vmin = np.nanmin(Z) if np.any(np.isfinite(Z)) else 0
        vmax = np.nanmax(Z) if np.any(np.isfinite(Z)) else 1
    elif mode == "period":
        cmaps = ["viridis", "viridis", "viridis"]
        vmin = np.nanmin(Z) if np.any(np.isfinite(Z)) else 0
        vmax = np.nanmax(Z) if np.any(np.isfinite(Z)) else 1
    elif mode == "peak_ratio":
        cmaps = ["plasma", "plasma", "plasma"]
        vmin, vmax = 0, 1
    else:
        cmaps = ["viridis", "viridis", "viridis"]
        vmin = np.nanmin(Z) if np.any(np.isfinite(Z)) else 0
        vmax = np.nanmax(Z) if np.any(np.isfinite(Z)) else 1

    if vmin == vmax:
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for k in range(3):
        im = axes[k].imshow(
            Z[:, :, k].T,
            origin='lower',
            aspect='auto',
            extent=[X[0], X[-1], Y[0], Y[-1]],
            cmap=cmaps[k],
            vmin=vmin, vmax=vmax
        )

        axes[k].set_title(f"ρ_{labels[k]} ({mode})")
        axes[k].set_xlabel(xlabel)
        axes[k].set_ylabel(ylabel)

        fig.colorbar(im, ax=axes[k])

    plt.tight_layout()
    plt.show()
