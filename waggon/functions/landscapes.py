import numpy as np
import scipy.linalg
from .base import FunctionV2


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


class Sphere(FunctionV2):
    def __init__(self, dim=1, **kwargs):
        super().__init__(**kwargs)
       
        self.dim = dim
        self.domain = np.tile([-10, 10], reps=(dim, 1))
        self.glob_min = np.zeros((1, dim))
        self.f_min = 0.0
        self.name = f"Sphere (dim={dim})"
    
    def func(self, x: np.ndarray) -> np.ndarray:
        return np.square(x).sum(axis=1, keepdims=True)


class SubmanifoldRosenbrock(FunctionV2):
    def __init__(self, dim=20, sub_dim=8, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.sub_dim = sub_dim
        self.sigma = 1.0
        self.domain = np.tile([-10, 10], reps=(dim, 1))
        self.name = f"SubmanifoldRosenbrock (dim={dim}, subdim={sub_dim})"

        np.random.seed(kwargs['seed'])
        A = np.random.randn(dim, sub_dim)
        Q, _ = np.linalg.qr(A)
        self.Q = Q
        
        b = np.ones(sub_dim)

        x_min, _, _, _ = scipy.linalg.lstsq(Q.T, b)
        self.glob_min = np.expand_dims(x_min, 0)
        self.f_min = 0.0

    def func(self, x: np.ndarray) -> np.ndarray:
        y = np.matmul(x, self.Q)
        return np.sum(
            100.0 * (y[..., 1:] - (y[..., :-1]) ** 2.0) ** 2.0 + (1.0 - y[..., :-1]) ** 2,
            axis=-1,
            keepdims=True
        )

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


class StyblinskyTang(FunctionV2):
    '''
    d-dimensional Styblinsky-Tang function.
    '''
    def __init__(self, dim=20, **kwargs):
        super(StyblinskyTang, self).__init__(**kwargs)

        self.dim      = dim
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = f'Styblinski-Tang ({self.dim} dim.)'
        self.glob_min = np.ones(self.dim).reshape(1, -1) * -2.903534

        self.f        = lambda x: np.sum(
            x ** 4.0 - 16.0 * x ** 2.0 + 5.0 * x + 39.16617 * self.dim, 
            axis=-1, 
        )
        self.f_min    = 0.0
        self.sigma    = lambda x: np.abs(x[:, 0] * np.sin(x[:, 1] - (-2.903534))) if 'sigma' in kwargs else lambda x: 1e-1
    
    def __call__(self, x):
        if self.log_transform:
            if self.sigma(np.zeros((1, self.dim)))[0] == 1e-1:
                return np.log(self.f(x) + self.log_eps) 
            else:
                return np.log(self.f(x) + self.sigma(x) + self.log_eps) 
        else:
            if self.sigma(np.zeros((1, self.dim)))[0] == 1e-1:
                return self.f(x)
            else:
                return self.f(x) + self.sigma(x)

    def sample(self, x):

        y = np.random.normal(self.__call__(x[0, :].reshape(1, -1)), self.sigma(x), (self.n_obs, 1))
        X = x[0, :]*np.ones((self.n_obs, 1))
        
        for i in range(1, x.shape[0]):
            y_ = np.random.normal(self.__call__(x[i, :].reshape(1, -1)), self.sigma(x), (self.n_obs, 1))
            X_ = x[i, :]*np.ones((self.n_obs, 1))

            y = np.concatenate((y, y_))
            X = np.concatenate((X, X_))
        
        return X, y
