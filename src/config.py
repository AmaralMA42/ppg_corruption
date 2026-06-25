from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    L: int = 80
    amostras: int = 5
    total_passos: int = 500
    percent_avg_MC: float = 0.6  # percent final a tirar media termica
    cond_ini: int = 0   #  0 rand, 1 lines, 2 pizza, 3 ????, 4 C&D
    k: float = 0.1
    r: float = 2.7
    G: int = 5
    c: float = 1.0
    sigma: float = 1.02
    alpha: float = 0.25
    r_start: float = 0.0
    r_stop: float = 4.0
    r_npoints: int = 20
    alpha_start: float = 0.0
    alpha_stop: float = 0.35
    alpha_npoints: int = 30
    sig_start: float = 1.0
    sig_stop: float = 3
    sig_npoints: int = 20

    phaseport: str = 'r_sigma'  # r_alpha, r_sigma, alpha_sigma, all

    framerate: int = 10
    fpsgif: int = 2
    passo_filma_inicio: int = 0
    seed: int | None = 12345
    create_snapshot: bool = True
    absorbing_window: int = 10
    make_plots: bool = True
    compute_time_analysis: bool = True
    compress_output: bool = False
    freerange: bool = True
    print_param: bool = True
    fft_max_freq: float= 0.2# | None = None

    @property
    def total_jog(self) -> int:
        return self.L * self.L

    @property
    def passos_media(self) -> int:
        return int(self.percent_avg_MC * self.total_passos)

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "figures"

    def simulation_params(self) -> np.ndarray:
        return np.array(
            [self.r, self.G, self.c, self.sigma, self.k, self.alpha, self.cond_ini],
            dtype=np.float64
        )
