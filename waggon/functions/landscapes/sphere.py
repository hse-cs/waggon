import numpy as np
from ..base import FunctionV2


class Sphere(FunctionV2):
    def __init__(self, dim=1, **kwargs):
        super().__init__(**kwargs)
       
        self.dim = dim
        self.domain = np.tile([-10, 10], reps=(dim, 1))
        self.glob_min = np.zeros((1, dim))
        self.name = f"Sphere (dim={dim})"
    
    def func(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, f"Input must got 2d-input, but it has shape {x.shape} with {x.ndim} dims"
        assert x.shape[1] == self.dim, f"Mismatch between function ({self.dim}) and input ({x.shape[-1]}) dimensionalities"
        return np.square(x).sum(axis=1, keepdims=True)
