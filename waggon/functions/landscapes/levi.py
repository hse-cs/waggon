import numpy as np
from ..base import FunctionV2


class Levi(FunctionV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim = 2
        self.domain = np.tile([-4, 6], reps=(2, 1))
        self.name = "Lévi"
        self.glob_min = np.ones((1, 2))
        self.f_min = 0.0
    
    def func(self, x: np.ndarray) -> np.ndarray:
        y = np.sin(3 * np.pi * x[..., 0]) ** 2
        y += ((x[..., 0] - 1) ** 2) * (1 + np.sin(3 * np.pi * x[..., 1]) ** 2)
        y += ((x[..., 1] - 1) ** 2) * (1 + np.sin(2 * np.pi * x[..., 1]) ** 2)
        return np.expand_dims(y, -1)
