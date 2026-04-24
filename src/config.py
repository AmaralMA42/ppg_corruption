from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    L: int = 300
    amostras: int = 1
    total_passos: int = 1000
    percent_avg_MC: float = 0.9  # percent final a tirar media termica
    seed: int | None = 12345
    create_snapshot: bool = True
    deldata: bool = True
    framerate: int = 10
    fpsgif: int = 10
    passo_filma_inicio: int = 0
    cond_ini: int = 0
    k: float = 0.1
    r: float = 3.3
    G: int = 5
    c: float = 1.0
    sigma: float = 1.05
    alpha: float = 0.46


    param_name = 'r'
    r_start: float = 1.0
    r_stop: float = 5.6
    r_npoints: int = 60

    param_name = 'alpha'
    alpha_start: float = 0.1
    alpha_stop: float = 0.6
    alpha_npoints: int = 40

    param_name2 = 'sigma'
    sig_start: float = 0.95
    sig_stop: float = 1.1
    sig_npoints: int = 20


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
