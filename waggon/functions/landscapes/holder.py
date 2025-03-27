import numpy as np
from ..base import FunctionV2


class Holder(FunctionV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim = 2
        self.domain = np.tile([-10, 10], reps=(2, 1))
        self.name = "Holder"
        self.glob_min = np.array([
            [8.05502, 9.66459], 
            [-8.05502, -9.66459], 
            [-8.05502, 9.66459], 
            [8.05502, -9.66459]
        ])
        self.f_min = 0.0
    
    def func(self, x: np.ndarray) -> np.ndarray:
        y = -np.abs(
            np.sin(x[..., 0]) * np.cos(x[..., 1]) * np.exp(np.abs(1 - np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2) / np.pi))
        ) + 19.2085
        return np.expand_dims(y, -1)
