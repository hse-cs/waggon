import numpy as np
from ..base import FunctionV2


class Ackley(FunctionV2):
    def __init__(self, dim=2, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.domain = np.tile([-5, 5], reps=(dim, 1))
        self.glob_min = np.zeros((1, dim))
        self.f_min = 0.0
        self.name = f"Ackley (dim={dim})"
    
    def func(self, x: np.ndarray) -> np.ndarray:
        a, b, c, d = 20.0, 0.2, 2 * np.pi, self.dim
        x_sq_sum = np.square(x).sum(axis=-1, keepdims=True)
        x_cos_sum = np.cos(c * x).sum(axis=-1, keepdims=True)
        
        y = -a * np.exp(-b * np.sqrt(x_sq_sum / d))
        y -= np.exp(x_cos_sum / d)
        y += a + np.exp(1.0)
        return y
