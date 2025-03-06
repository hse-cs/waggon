import numpy as np
import scipy.linalg
from ..base import FunctionV2


class SubmanifoldRosenbrock(FunctionV2):
    def __init__(self, dim=20, sub_dim=8, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.sub_dim = sub_dim
        self.domain = np.tile([-10, 10], reps=(dim, 1))
        self.name = f"SubmanifoldRosenbrock (dim={dim}, subdim={sub_dim})"

        A = np.random.randn(dim, sub_dim)
        Q, _ = np.linalg.qr(A)
        self.Q = Q

        b = np.ones(sub_dim)

        x_min, _, _, _ = scipy.linalg.lstsq(Q.T, b)
        self.glob_min = np.expand_dims(x_min, 0)

    def func(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2
        assert x.shape[1] == self.dim
        
        y = np.matmul(x, self.Q)
        return np.sum(
            100.0 * (y[..., 1:] - (y[..., :-1]) ** 2.0) ** 2.0 + (1.0 - y[..., :-1]) ** 2,
            axis=-1,
            keepdims=True
        )
