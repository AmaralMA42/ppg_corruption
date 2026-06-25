from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from config import SimulationConfig
from core_simulation import inicia_vizinhos, monte_carlo_single
from plotting import plota_atividade, plota_payoff_por_estrategia, plota_todas_amostras
from utils import config_metadata, load_npz_result, save_npz_result


cfg = SimulationConfig()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def finite_clim(values, fallback=(0.0, 1.0)):
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


def positive_percentile_clim(values, percentile=99):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return 0.0, 1.0

    vmax = float(np.nanpercentile(values[valid], percentile))
    return 0.0, max(vmax, 1e-12)


class VisualRecorder:
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self.enabled = bool(cfg.create_snapshot)
        self.output_dir = Path(output_dir or cfg.figures_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.L = cfg.L
        self.framerate = max(int(cfg.framerate), 1)
        self.fpsgif = max(int(cfg.fpsgif), 1)
        self.passo_filma_inicio = int(cfg.passo_filma_inicio)

        self.frames_A = []
        self.frames_B = []
        self.figA = None
        self.figB = None

        if self.enabled:
            self._setup_figures()

    def _setup_figures(self):
        self.imagegrid = np.zeros((self.L, self.L))
        self.payoff_grid = np.zeros((self.L, self.L))
        self.grad_grid = np.zeros((self.L, self.L))
        self.var_grid = np.zeros((self.L, self.L))

        self.figA, (self.axA1, self.axA2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
        self.figB, (self.axB1, self.axB2) = plt.subplots(1, 2, figsize=(8.6, 4.2))

        self.imA1 = self.axA1.imshow(self.imagegrid, vmin=0, vmax=2, cmap="brg")
        self.imA2 = self.axA2.imshow(self.payoff_grid, cmap="viridis", vmin=-0.5, vmax=8.5)
        self.imB1 = self.axB1.imshow(self.grad_grid, cmap="turbo", interpolation="bicubic")
        self.imB2 = self.axB2.imshow(self.var_grid, cmap="cividis", interpolation="bicubic")

        self.figA.colorbar(self.imA1, ax=self.axA1)
        self.figA.colorbar(self.imA2, ax=self.axA2)
        self.figB.colorbar(self.imB1, ax=self.axB1)
        self.figB.colorbar(self.imB2, ax=self.axB2)

    def __call__(self, passo, estrategia, payoff):
        if not self.enabled:
            return
        if passo < self.passo_filma_inicio or passo % self.framerate != 0:
            return

        self.imagegrid[:] = estrategia.reshape(self.L, self.L)
        self.payoff_grid[:] = payoff.reshape(self.L, self.L)

        gx, gy = np.gradient(self.payoff_grid)
        self.grad_grid[:] = np.sqrt(gx ** 2 + gy ** 2)

        mean = uniform_filter(self.payoff_grid, size=3)
        mean_sq = uniform_filter(self.payoff_grid ** 2, size=3)
        self.var_grid[:] = mean_sq - mean ** 2

        self.imA1.set_data(self.imagegrid)
        self.imA2.set_data(self.payoff_grid)
        self.imB1.set_data(self.grad_grid)
        self.imB2.set_data(self.var_grid)

        if getattr(self.cfg, "freerange", False):
            self.imA2.set_clim(*finite_clim(self.payoff_grid, fallback=(-0.5, 8.5)))
        self.imB1.set_clim(*positive_percentile_clim(self.grad_grid))
        self.imB2.set_clim(*positive_percentile_clim(self.var_grid))

        self.axA1.set_title(f"Estrategia | mcs={passo}")
        self.axA2.set_title(f"Payoff | mcs={passo}")
        self.axB1.set_title(f"|grad p| | mcs={passo}")
        self.axB2.set_title(f"Var(p) local | mcs={passo}")

        self._capture_frame(passo)

    def _capture_frame(self, passo):
        self.figA.canvas.draw()
        frame_A = np.asarray(self.figA.canvas.buffer_rgba()).copy()
        self.frames_A.append(Image.fromarray(frame_A))
        self.figA.savefig(
            self.output_dir / f"str_{passo:05d}.png",
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=150,
        )

        self.figB.canvas.draw()
        frame_B = np.asarray(self.figB.canvas.buffer_rgba()).copy()
        self.frames_B.append(Image.fromarray(frame_B))
        self.figB.savefig(
            self.output_dir / f"VAR_{passo:05d}.png",
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=150,
        )

    def close(self):
        if not self.enabled:
            return

        duration_ms = int(1000 / self.fpsgif)
        if self.frames_A:
            self.frames_A[0].save(
                self.output_dir / "anima_A.gif",
                save_all=True,
                append_images=self.frames_A[1:],
                duration=duration_ms,
                loop=0,
            )
        if self.frames_B:
            self.frames_B[0].save(
                self.output_dir / "anima_B.gif",
                save_all=True,
                append_images=self.frames_B[1:],
                duration=duration_ms,
                loop=0,
            )

        plt.close(self.figA)
        plt.close(self.figB)


def run_visual_simulation(cfg, output_dir=None):
    params = cfg.simulation_params()
    total_jog = cfg.total_jog
    viz = np.zeros((total_jog, 4), dtype=np.int32)
    inicia_vizinhos(viz, total_jog, cfg.L)

    recorder = VisualRecorder(cfg, output_dir=output_dir)
    callback = recorder if cfg.create_snapshot else None

    try:
        estrat_single, payavg_single, activity_single, absorbed_at = monte_carlo_single(
            viz,
            params,
            total_jog,
            cfg.total_passos,
            cfg.L,
            cfg.seed,
            callback=callback,
            absorbing_window=cfg.absorbing_window,
        )
    finally:
        recorder.close()

    estrat_t = estrat_single[:, np.newaxis, :]
    payavg_t = payavg_single[:, np.newaxis, :]
    activity_t = activity_single[np.newaxis, :]
    absorbed_at = np.array([absorbed_at], dtype=np.int64)

    return (
        estrat_t,
        np.mean(estrat_t, axis=1),
        payavg_t,
        np.mean(payavg_t, axis=1),
        activity_t,
        np.mean(activity_t, axis=0),
        absorbed_at,
    )


def plot_saved_visual(path, cfg=cfg):
    arrays, metadata = load_npz_result(path)
    plota_todas_amostras(arrays["estrat_t"], arrays["estrat_medio_t"], cfg)
    plota_payoff_por_estrategia(arrays["payavg_t"], arrays["payavg_medio_t"], cfg)
    plota_atividade(arrays["activity_t"], arrays["activity_medio_t"], cfg)
    return metadata


def main():
    (
        estrat_t,
        estrat_medio_t,
        payavg_t,
        payavg_medio_t,
        activity_t,
        activity_medio_t,
        absorbed_at,
    ) = run_visual_simulation(cfg)

    if cfg.make_plots:
        plota_todas_amostras(estrat_t, estrat_medio_t, cfg)
        plota_payoff_por_estrategia(payavg_t, payavg_medio_t, cfg)
        plota_atividade(activity_t, activity_medio_t, cfg)

    metadata = config_metadata(cfg, "visual", visual_amostras=1)
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
        absorbed_at=absorbed_at,
    )
    print(f"Dados salvos em: {output_file}")


if __name__ == "__main__":
    main()
