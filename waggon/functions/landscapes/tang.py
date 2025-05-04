import numpy as np
from ..base import FunctionV2


class StyblinskyTang(FunctionV2):
    """
    d-dimensional Styblinsky-Tang function
    """
    def __init__(self, dim=20, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.domain = np.tile([-5, 5], reps=(dim, 1))
        self.name = f"Styblinski-Tang (dim={self.dim})"
        self.glob_min = np.full((1, dim), fill_value=-2.903534)
        self.f_min = 0.0
        self.sigma = 1.0
    
    def func(self, x: np.ndarray) -> np.ndarray:
        return np.sum(
            (x ** 4) - 16.0 * (x ** 2) + 5.0 * x,
            axis=-1,
            keepdims=True
        )  + 39.16617 * self.dim
