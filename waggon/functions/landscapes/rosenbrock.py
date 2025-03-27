import numpy as np
from ..base import FunctionV2


class Rosenbrock(FunctionV2):
    def __init__(self, dim=20, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.domain = np.tile([-2, 2], reps=(dim, 1))
        self.name = f"Rosenbrock (dim={dim})"
        self.glob_min = np.ones((1, dim))
        self.f_min = 0.0
    
    def func(self, x: np.ndarray) -> np.ndarray:
        return np.sum(
            100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2,
            axis=-1,
            keepdims=True
        )
