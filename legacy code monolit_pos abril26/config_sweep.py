from dataclasses import dataclass
import numpy as np

@dataclass
class SweepConfig:
    param_name: str
    start: float
    stop: float
    n_points: int

    def values(self):
        return np.linspace(self.start, self.stop, self.n_points)