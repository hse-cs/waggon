import numpy as np
import scipy.linalg
from ..base import FunctionV2
from ..utils import fixed_numpy_seed


class SubmanifoldRosenbrock(FunctionV2):
    """
    Submanifold Rosenbrock problem function

    Parameters
    ----------
    dim: int, default 20
        Dimensionality of the function
    sub_dim: int, default 8
        Sub-dimensionality of the function
    seed: int | None, default None
        Seed for mapping matrix generation
    **kwargs
        Arguments passed to FunctionV2 base class
    
    Notes
    -----
    In case `seed=None`, NumPy automatically generates a seed 
    based on the system entropy source.
    """
    def __init__(self, dim=20, sub_dim=8, seed=None, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.sub_dim = sub_dim
        self.domain = np.tile([-10, 10], reps=(dim, 1))
        self.name = f"SubmanifoldRosenbrock (dim={dim}, subdim={sub_dim})"

        with fixed_numpy_seed(seed):
            A = np.random.randn(dim, sub_dim)
        Q, _ = np.linalg.qr(A)
        self.Q = Q

        b = np.ones(sub_dim)

        x_min, _, _, _ = scipy.linalg.lstsq(Q.T, b)
        self.glob_min = np.expand_dims(x_min, 0)
        self.f_min = 0.0
        self.sigma = 1.0

    def func(self, x: np.ndarray) -> np.ndarray:
        y = np.matmul(x, self.Q)
        return np.sum(
            100.0 * (y[..., 1:] - (y[..., :-1]) ** 2.0) ** 2.0 + (1.0 - y[..., :-1]) ** 2,
            axis=-1,
            keepdims=True
        )