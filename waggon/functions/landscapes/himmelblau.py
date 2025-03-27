import numpy as np
from ..base import FunctionV2


class Himmelblau(FunctionV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim = 2
        self.domain = np.tile([-5, 5], reps=(2, 1))
        self.name = "Himmelblau"
        self.glob_min = np.array([
            [3.0, 2.0], 
            [-2.805118, 3.131312], 
            [-3.779310, -3.283186], 
            [3.584428, -1.848126]
        ])
        self.f_min = 0.0
    
    def func(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(
            (x[..., 0] ** 2 + x[..., 1] - 11.0) ** 2 + (x[..., 0] + x[..., 1] ** 2 - 7.0) ** 2,
            axis=-1
        )
