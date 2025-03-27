import numpy as np
from ..base import FunctionV2


class ThreeHumpCamel(FunctionV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim = 2
        self.domain = np.tile([-5, 5], reps=(2, 1))
        self.glob_min = np.zeros((1, self.dim))
        self.f_min = 0.0
        self.name = "Three-hump camel function"
    
    def func(self, x: np.ndarray) -> np.ndarray:
        xc = x[..., 0]
        yc = x[..., 1]
        res = 2.0 * (xc ** 2) - 1.05 * (xc ** 4) + (1 / 6) * (xc ** 6) + (xc * yc) + (yc ** 2)
        return np.expand_dims(res, axis=-1)
    