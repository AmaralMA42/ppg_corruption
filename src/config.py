from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    L: int = 100
    amostras: int = 1
    total_passos: int = 1000
    percent_avg_MC: float = 0.9  # percent final a tirar media termica
    seed: int | None = 12345
    create_snapshot: bool = True
    deldata: bool = True
    framerate: int = 10
    fpsgif: int = 10
    passo_filma_inicio: int = 0
    cond_ini: int = 4
    k: float = 0.1
    r: float = 3.6
    G: int = 5
    c: float = 1.0
    sigma: float = 1.02
    alpha: float = 0.3


#    param_name = 'r'
    r_start: float = 3.8
    r_stop: float = 4.2
    r_npoints: int = 10

#    param_name = 'alpha'
    alpha_start: float = 0.1
    alpha_stop: float = 0.25
    alpha_npoints: int = 2

#    param_name2 = 'sigma'
    sig_start: float = 1.0
    sig_stop: float = 1.1
    sig_npoints: int = 2

    phaseport: str = 'both'  #sigma, alpha, ou both

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
