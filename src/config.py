from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class SimulationConfig:
    """Configuration class for evolutionary game theory simulations on a 2D lattice."""
    # Game parameters
    k: float = 0.1
    r: float = 3.2
    G: int = 5
    c: float = 1.0
    sigma: float = 1.02
    alpha: float = 0.3
    # Initial conditions
    cond_ini: int = 2
    # Lattice and sampling parameters
    L: int = 100
    amostras: int = 20
    # Simulation timing
    total_passos: int = 1600
    percent_avg_MC: float = 0.9  # percent final a tirar media termica
    seed: int | None = 12345
    # Output control
    create_snapshot: bool = True
    deldata: bool = True
    framerate: int = 10
    fpsgif: int = 10
    passo_filma_inicio: int = 0

    phaseport: str = 'both'  # Phase portrait parameter: 'sigma', 'alpha', or 'both'
    # Parameter sweep ranges for r
    r_start: float = 2.5
    r_stop: float = 4.5
    r_npoints: int = 30
    # Parameter sweep ranges for alpha
    alpha_start: float = 0.1
    alpha_stop: float = 0.25
    alpha_npoints: int = 30
    # Parameter sweep ranges for sigma
    sig_start: float = 1.0
    sig_stop: float = 1.1
    sig_npoints: int = 20

    @property
    def total_jog(self) -> int:
        """Total number of agents on the lattice (L * L)."""
        return self.L * self.L

    @property
    def passos_media(self) -> int:
        """Number of steps to skip for thermal averaging."""
        return int(self.percent_avg_MC * self.total_passos)

    @property
    def root_dir(self) -> Path:
        """Root directory of the project."""
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        """Data directory for storing simulation results."""
        return self.root_dir / "data"

    @property
    def figures_dir(self) -> Path:
        """Figures directory for storing plots and images."""
        return self.root_dir / "figures"

    def simulation_params(self) -> np.ndarray:
        """Convert configuration to numpy array for simulation functions.

        Returns:
            Array [r, G, c, sigma, k, alpha, cond_ini] as float64.
        """
        return np.array(
            [self.r, self.G, self.c, self.sigma, self.k, self.alpha, self.cond_ini],
            dtype=np.float64
        )
