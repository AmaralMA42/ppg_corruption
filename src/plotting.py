import matplotlib.pyplot as plt
import numpy as np

STRATEGY_LABELS = ["C", "D", "P"]
STRATEGY_LINE_COLORS = ["blue", "red", "green"]

STRATEGY_COLORS = {
    "C": "Blues",
    "D": "Reds",
    "P": "Greens",
}


def _free_range(cfg):
    return bool(getattr(cfg, "freerange", False))


def _print_param(cfg):
    return bool(getattr(cfg, "print_param", True))


def _format_param_value(value):
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def param_box_text(cfg, varying_params=()):
    if cfg is None:
        return ""

    varying = set(varying_params or ())
    params = [
        ("L", "L"),
        ("T", "total_passos"),
        ("amostras", "amostras"),
        ("r", "r"),
        (r"$\sigma$", "sigma"),
        (r"$\alpha$", "alpha"),
        ("k", "k"),
        ("G", "G"),
        ("c", "c"),
    ]
    lines = []
    for label, attr in params:
        if attr in varying or not hasattr(cfg, attr):
            continue
        lines.append(f"{label}={_format_param_value(getattr(cfg, attr))}")
    return "\n".join(lines)


def add_param_box(ax, cfg, varying_params=()):
    if cfg is None or not _print_param(cfg):
        return False

    text = param_box_text(cfg, varying_params=varying_params)
    if not text:
        return False

    ax.text(
        1.02,
        0.5,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    return True


def add_figure_param_box(fig, cfg, varying_params=()):
    if cfg is None or not _print_param(cfg):
        return False

    text = param_box_text(cfg, varying_params=varying_params)
    if not text:
        return False

    fig.text(
        0.86,
        0.5,
        text,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    return True


def finish_layout(fig, has_param_box=False):
    if has_param_box:
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        fig.tight_layout()


def finite_minmax(values, fallback=(0.0, 1.0)):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return fallback

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin == vmax:
        delta = max(abs(vmin) * 1e-6, 1e-12)
        return vmin - delta, vmax + delta
    return vmin, vmax


def plota_dados(estrat_medio_t, cfg):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("Evolucao media das subpopulacoes")
    ax.set_xlabel("Numero de Passos")
    ax.set_ylabel(r"$<\rho >$")
    ax.plot(estrat_medio_t[0, :], label="C")
    ax.plot(estrat_medio_t[1, :], label="D")
    ax.plot(estrat_medio_t[2, :], label="P")
    if not _free_range(cfg):
        ax.set_ylim(0, 1)
    ax.legend()
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_todas_amostras(estrat_t, estrat_medio_t, cfg):
    fig = plt.figure()
    ax = plt.gca()

    for i in range(estrat_t.shape[1]):
        ax.plot(estrat_t[0, i, :], alpha=0.2, linestyle="-")
        ax.plot(estrat_t[1, i, :], alpha=0.2, linestyle="--")
        ax.plot(estrat_t[2, i, :], alpha=0.2, linestyle=":")

    ax.plot(estrat_medio_t[0, :], label="C (media)", linewidth=2.0, color="blue")
    ax.plot(estrat_medio_t[1, :], label="D (media)", linewidth=2.0, color="red")
    ax.plot(estrat_medio_t[2, :], label="P (media)", linewidth=2.0, color="green")

    if not _free_range(cfg):
        ax.set_ylim(0, 1)
    ax.set_xlabel("Numero de Passos")
    ax.set_ylabel(r"$\langle \rho \rangle$")
    ax.legend()
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_payoff_por_estrategia(payavg_t, payavg_medio_t, cfg):
    fig = plt.figure()
    ax = plt.gca()

    for i in range(payavg_t.shape[1]):
        ax.plot(payavg_t[0, i, :], alpha=0.2, linestyle="-")
        ax.plot(payavg_t[1, i, :], alpha=0.2, linestyle="--")
        ax.plot(payavg_t[2, i, :], alpha=0.2, linestyle=":")

    ax.plot(payavg_medio_t[0, :], label="C payoff", linewidth=2.0)
    ax.plot(payavg_medio_t[1, :], label="D payoff", linewidth=2.0)
    ax.plot(payavg_medio_t[2, :], label="P payoff", linewidth=2.0)

    ax.set_xlabel("Numero de Passos")
    ax.set_ylabel(r"$\langle \Pi \rangle$")
    ax.legend()
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_atividade(activity_t, activity_medio_t, cfg):
    fig = plt.figure()
    ax = plt.gca()

    for i in range(activity_t.shape[0]):
        ax.plot(activity_t[i, :], alpha=0.25, color="gray")

    ax.plot(activity_medio_t, label="A(t) media", linewidth=2.0, color="black")
    if not _free_range(cfg):
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Numero de Passos")
    ax.set_ylabel("A(t) / N")
    ax.set_title("Atividade normalizada por MCS")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_variancia_temporal(variance_samples, variance_mean, cfg):
    markers = ["o", "s", "^"]
    x = np.arange(3)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for i in range(3):
        samples = variance_samples[i, :]
        valid = np.isfinite(samples)
        if np.any(valid):
            jitter = np.linspace(-0.08, 0.08, np.sum(valid))
            ax.scatter(
                np.full(np.sum(valid), x[i]) + jitter,
                samples[valid],
                alpha=0.35,
                color=STRATEGY_LINE_COLORS[i],
                marker=markers[i],
            )
        if np.isfinite(variance_mean[i]):
            ax.scatter(
                x[i],
                variance_mean[i],
                color=STRATEGY_LINE_COLORS[i],
                edgecolor="black",
                marker=markers[i],
                s=80,
                zorder=3,
            )

    if not _free_range(cfg):
        ax.set_ylim(bottom=0)
    ax.set_xticks(x, STRATEGY_LABELS)
    ax.set_ylabel("Variancia temporal")
    ax.set_title("Variancia temporal pos-transiente")
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_autocorrelacao(autocorr_mean, cfg):
    tau = np.arange(autocorr_mean.shape[1])

    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    for i in range(3):
        valid = np.isfinite(autocorr_mean[i, :])
        if np.any(valid):
            ax.plot(tau[valid], autocorr_mean[i, valid], label=STRATEGY_LABELS[i], color=STRATEGY_LINE_COLORS[i])

    if not _free_range(cfg):
        ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Lag temporal tau (MCS)")
    ax.set_ylabel("C(tau) normalizada")
    ax.set_title("Autocorrelacao temporal media")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_fft_power(freqs, power_mean, cfg, power_sem=None):
    freqs = np.asarray(freqs, dtype=float)
    max_freq = getattr(cfg, "fft_max_freq", None)
    freq_mask = np.isfinite(freqs)
    if max_freq is not None:
        freq_mask &= freqs <= max_freq

    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    for i in range(3):
        valid = freq_mask & np.isfinite(power_mean[i, :])
        if not np.any(valid):
            continue

        ax.plot(
            freqs[valid],
            power_mean[i, valid],
            label=STRATEGY_LABELS[i],
            color=STRATEGY_LINE_COLORS[i],
        )

        if power_sem is not None:
            sem = np.asarray(power_sem[i, :], dtype=float)
            sem_valid = valid & np.isfinite(sem)
            if np.any(sem_valid):
                lower = np.maximum(power_mean[i, sem_valid] - sem[sem_valid], 0.0)
                upper = power_mean[i, sem_valid] + sem[sem_valid]
                ax.fill_between(
                    freqs[sem_valid],
                    lower,
                    upper,
                    color=STRATEGY_LINE_COLORS[i],
                    alpha=0.15,
                    linewidth=0,
                )

    if not _free_range(cfg):
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequencia (1/MCS)")
    ax.set_ylabel("Potencia FFT")
    ax.set_title("Espectro de potencia medio pos-transiente")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_periodo_dominante(period_samples, period_mean, cfg):
    markers = ["D", "P", "X"]
    x = np.arange(3)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for i in range(3):
        samples = period_samples[i, :]
        valid = np.isfinite(samples)
        if np.any(valid):
            jitter = np.linspace(-0.08, 0.08, np.sum(valid))
            ax.scatter(
                np.full(np.sum(valid), x[i]) + jitter,
                samples[valid],
                alpha=0.35,
                color=STRATEGY_LINE_COLORS[i],
                marker=markers[i],
            )
        if np.isfinite(period_mean[i]):
            ax.scatter(
                x[i],
                period_mean[i],
                color=STRATEGY_LINE_COLORS[i],
                edgecolor="black",
                marker=markers[i],
                s=80,
                zorder=3,
            )

    if not _free_range(cfg):
        ax.set_ylim(bottom=0)
    ax.set_xticks(x, STRATEGY_LABELS)
    ax.set_ylabel("Periodo dominante (MCS)")
    ax.set_title("Periodo dominante por estrategia")
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_peak_ratio(peak_ratio_samples, peak_ratio_mean, cfg):
    markers = ["v", "<", ">"]
    x = np.arange(3)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for i in range(3):
        samples = peak_ratio_samples[i, :]
        valid = np.isfinite(samples)
        if np.any(valid):
            jitter = np.linspace(-0.08, 0.08, np.sum(valid))
            ax.scatter(
                np.full(np.sum(valid), x[i]) + jitter,
                samples[valid],
                alpha=0.35,
                color=STRATEGY_LINE_COLORS[i],
                marker=markers[i],
            )
        if np.isfinite(peak_ratio_mean[i]):
            ax.scatter(
                x[i],
                peak_ratio_mean[i],
                color=STRATEGY_LINE_COLORS[i],
                edgecolor="black",
                marker=markers[i],
                s=80,
                zorder=3,
            )

    if not _free_range(cfg):
        ax.set_ylim(0, 1)
    ax.set_xticks(x, STRATEGY_LABELS)
    ax.set_ylabel("Peak ratio")
    ax.set_title("Concentracao espectral no pico dominante")
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plota_media_com_erro(steady_state, cfg, tipo="estrategia"):
    amostras = steady_state.shape[1]
    mean = np.mean(steady_state, axis=1)
    if amostras <= 1:
        sem = np.zeros_like(mean)
    else:
        sem = np.std(steady_state, axis=1, ddof=1) / np.sqrt(amostras)

    x = np.arange(3)
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.errorbar(x, mean, yerr=sem, fmt="o", capsize=5)
    ax.set_xticks(x, STRATEGY_LABELS)

    if tipo == "estrategia":
        ax.set_ylabel("Fracao media (steady state)")
        ax.set_title("Media + erro entre amostras")
        if not _free_range(cfg):
            ax.set_ylim(0, 1)
    elif tipo == "payoff":
        ax.set_ylabel("payoff medio (steady state)")
        ax.set_title("Media + erro entre amostras")

    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg))
    plt.show()


def plot_vs_r(r_values, mean, sem, output_dir, cfg):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for i in range(3):
        ax.errorbar(
            r_values,
            mean[:, i],
            yerr=sem[:, i],
            label=STRATEGY_LABELS[i],
            capsize=3,
            color=STRATEGY_LINE_COLORS[i],
        )

    if not _free_range(cfg):
        ax.set_ylim(0, 1)
    ax.set_xlabel("r")
    ax.set_ylabel("Fracao media")
    ax.set_title("Dependencia em r")
    ax.legend()
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg, varying_params=("r",)))
    plt.savefig(output_dir / "vs_r.png", dpi=150)
    plt.show()


def plot_sweep_1d(x, mean, sem, labels, xlabel, cfg):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for i, label in enumerate(labels):
        ax.errorbar(x, mean[:, i], yerr=sem[:, i], label=label, capsize=3)

    if not _free_range(cfg):
        ax.set_ylim(0, 1)
    ax.set_title(f"Dependencia em {xlabel}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("observavel medio")
    finish_layout(fig, add_param_box(ax, cfg, varying_params=(xlabel,)))
    plt.show()


def plot_trajectories_vs_time(values, traj, param_name, ylabel, cfg=None, step=1):
    step = max(int(step), 1)
    t = np.arange(traj.shape[2])

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for idx in range(0, len(values), step):
        val = values[idx]
        for s in range(3):
            ax.plot(
                t,
                traj[idx, s],
                color=STRATEGY_LINE_COLORS[s],
                alpha=0.3,
                label=f"{STRATEGY_LABELS[s]}, {param_name}={val:.2f}" if idx == 0 else None,
            )

    if ylabel == "rho" and cfg is not None and not _free_range(cfg):
        ax.set_ylim(0, 1)
    ax.set_xlabel("Monte Carlo step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Evolucao temporal (varrendo {param_name})")
    ax.legend()
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg, varying_params=(param_name,)))
    plt.show()


def plot_activity_trajectories_vs_time(values, traj_activity, param_name, cfg=None, step=1):
    step = max(int(step), 1)
    t = np.arange(traj_activity.shape[1])

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for idx in range(0, len(values), step):
        ax.plot(
            t,
            traj_activity[idx],
            alpha=0.45,
            label=f"{param_name}={values[idx]:.2f}" if idx == 0 else None,
        )

    if cfg is not None and not _free_range(cfg):
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Monte Carlo step")
    ax.set_ylabel("A(t) / N")
    ax.set_title(f"Atividade normalizada temporal (varrendo {param_name})")
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg, varying_params=(param_name,)))
    plt.show()


def plot_variance_vs_param(values, var_mean, var_sem, labels, param_name, cfg=None):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for i in range(3):
        ax.errorbar(values, var_mean[:, i], yerr=var_sem[:, i], label=labels[i], capsize=3)

    if cfg is not None and not _free_range(cfg):
        ax.set_ylim(bottom=0)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Variancia temporal")
    ax.set_title("Flutuacoes temporais (indicador de ciclos)")
    ax.legend()
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg, varying_params=(param_name,)))
    plt.show()


def plot_metric_vs_param(values, metric_mean, metric_sem, labels, param_name, ylabel, title, ylim=None, cfg=None):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for i in range(3):
        ax.errorbar(values, metric_mean[:, i], yerr=metric_sem[:, i], label=labels[i], capsize=3)

    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None and (cfg is None or not _free_range(cfg)):
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(alpha=0.3)
    finish_layout(fig, add_param_box(ax, cfg, varying_params=(param_name,)))
    plt.show()


def plot_heatmap(X, Y, Z, xlabel, ylabel, title="", cfg=None, mode="fraction"):
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    vmin, vmax = heatmap_limits(Z, mode, cfg)

    im = ax.imshow(
        Z.T,
        origin="lower",
        aspect="auto",
        extent=[X[0], X[-1], Y[0], Y[-1]],
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(im, ax=ax, label=r"$\rho_C$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finish_layout(fig, add_param_box(ax, cfg, varying_params=(xlabel, ylabel)))
    plt.show()


def heatmap_limits(Z, mode, cfg):
    if _free_range(cfg):
        return finite_minmax(Z)
    if mode in ("fraction", "peak_ratio"):
        return 0, 1
    return finite_minmax(Z)


def heatmap_cmaps(mode):
    if mode == "fraction":
        return ["Blues", "Reds", "Greens"]
    if mode == "variance":
        return ["turbo", "turbo", "turbo"]
    if mode == "period":
        return ["Spectral_r", "Spectral_r", "Spectral_r"]
    if mode == "peak_ratio":
        return ["plasma", "plasma", "plasma"]
    return ["viridis", "viridis", "viridis"]


def heatmap_colorbar_label(mode):
    if mode == "fraction":
        return "fracao media"
    if mode == "variance":
        return "variancia temporal"
    if mode == "period":
        return "periodo dominante (MCS)"
    if mode == "peak_ratio":
        return "peak ratio"
    return mode


def plot_heatmap_3(X, Y, Z, xlabel, ylabel, mode="fraction", cfg=None, varying_params=None):
    cmaps = heatmap_cmaps(mode)
    vmin, vmax = heatmap_limits(Z, mode, cfg)
    if varying_params is None:
        varying_params = (xlabel, ylabel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for k in range(3):
        panel_vmin, panel_vmax = finite_minmax(Z[:, :, k]) if _free_range(cfg) else (vmin, vmax)
        im = axes[k].imshow(
            Z[:, :, k].T,
            origin="lower",
            aspect="auto",
            extent=[X[0], X[-1], Y[0], Y[-1]],
            cmap=cmaps[k],
            vmin=panel_vmin,
            vmax=panel_vmax,
        )

        axes[k].set_title(f"rho_{STRATEGY_LABELS[k]} ({mode})")
        axes[k].set_xlabel(xlabel)
        axes[k].set_ylabel(ylabel)
        fig.colorbar(im, ax=axes[k], label=heatmap_colorbar_label(mode))

    has_param_box = add_figure_param_box(fig, cfg, varying_params=varying_params)
    finish_layout(fig, has_param_box)
    plt.show()
