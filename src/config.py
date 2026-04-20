from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    L: int = 200
    amostras: int = 50
    total_passos: int = 500
    seed: int | None = 12345
    create_snapshot: bool = False
    deldata: bool = True
    framerate: int = 30
    fpsgif: int = 5
    passo_filma_inicio: int = 0
    cond_ini: int = 0
    k: float = 0.1
    r: float = 3.3
    G: int = 5
    c: float = 1.0
    sigma: float = 1.05
    alpha: float = 0.4

    @property
    def total_jog(self) -> int:
        return self.L * self.L

    @property
    def passos_media(self) -> int:
        return int(0.9 * self.total_passos)

    def simulation_params(self) -> np.ndarray:
        return np.array([self.r, self.G, self.c, self.sigma, self.k, self.alpha])
