from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    L: int = 50
    amostras: int = 2
    total_passos: int = 100
    percent_avg_MC: float = 0.9  # percent final a tirar media termica
    seed: int | None = 12345
    create_snapshot: bool = True
    deldata: bool = True
    framerate: int = 100
    fpsgif: int = 10
    passo_filma_inicio: int = 0
    cond_ini: int = 0
    k: float = 0.1
    r: float = 3.3
    G: int = 5
    c: float = 1.0
    sigma: float = 1.05
    alpha: float = 0.40

    r_start: float = 1.0
    r_stop: float = 6.0
    r_npoints: int = 10

    param_name = 'alpha'
    alpha_start: float = 0.0
    alpha_stop: float = 1.0
    alpha_npoints: int = 10

    param_name2 = 'sigma'
    sig_start: float = 0.75
    sig_stop: float = 2.0
    sig_npoints: int = 10


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
