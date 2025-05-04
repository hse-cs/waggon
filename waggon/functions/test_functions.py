import numpy as np
import scipy.linalg

from .base import Function
from .utils import fixed_numpy_seed


class three_hump_camel(Function):
    '''
    Three Hump Camel function.
    '''
    def __init__(self, **kwargs):
        super(three_hump_camel, self).__init__(**kwargs)
        
        self.dim      = 2
        self.domain   = np.array([[-5, 5], [-5, 5]])
        self.name     = 'Three Hump Camel'
        self.glob_min = np.zeros(self.dim).reshape(1, -1)
        self.f_min    = 0.0
        self.f        = lambda x: 2 * x[:, 0]**2 - 1.05 * x[:, 0]**4 + x[:, 0]**6 / 6 + x[:, 0]*x[:, 1] + x[:, 1]**2


class rosenbrock(Function):
    '''
    d-dimensional Rosenbrock function.
    '''
    def __init__(self, dim=20, **kwargs):
        super(rosenbrock, self).__init__(**kwargs)

        self.dim      = dim
        self.domain   = np.array([self.dim*[-2, 2]]).reshape(self.dim, 2)
        self.name     = f'Rosenbrock ({self.dim} dim.)'
        self.glob_min = np.ones(self.dim).reshape(1, -1)
        self.f_min    = 0.0
        self.f        = lambda x: np.sum(np.array([100 * (x[:, i+1] - x[:, i] ** 2)**2 + (1 - x[:, i])**2 for i in range(self.dim - 1)]), axis=0)


class ackley(Function):
    '''
    Ackley function.
    '''
    def __init__(self, dim=2, **kwargs):
        super(ackley, self).__init__(**kwargs)

        self.dim      = dim
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = 'Ackley'
        self.glob_min = np.zeros(self.dim).reshape(1, -1)
        self.f_min    = 0.0
        self.f        = lambda x: -20 * np.exp(-0.2*np.sqrt((1./self.dim) * (np.sum(x**2, axis=1)))) -  np.exp((1./self.dim) * (np.sum(np.cos(2*np.pi*x), axis=1))) + np.e + 20


class levi(Function):
    '''
    Lévi function.
    '''
    def __init__(self, **kwargs):
        super(levi, self).__init__(**kwargs)

        self.dim      = 2
        self.domain   = np.array([self.dim*[-4, 6]]).reshape(self.dim, 2)
        self.name     = 'Lévi'
        self.glob_min = np.ones(self.dim).reshape(1, -1)
        self.f_min    = 0.0
        self.f        = lambda x: (np.sin(3*np.pi*x[:, 0]))**2 + ((x[:, 0] - 1)**2) * (1 + (np.sin(3*np.pi*x[:, 1]))**2) + ((x[:, 1] - 1)**2) * (1 + (np.sin(2*np.pi*x[:, 1]))**2)

    
    def sample(self, x):
        X, y = None, None
        
        for i in range(x.shape[0]):
            s1 = 0.04 - 0.03 * np.square(np.sin(3 * np.pi * x[i, 1]))
            s2 = 0.001 + 0.03 * np.square(np.sin(3 * np.pi * x[i, 1]))
            g1 = np.random.normal(self.__call__(x[i, :].reshape(1, -1))-0.05, s1, (self.n_obs//2, 1))
            g2 = np.random.normal(self.__call__(x[i, :].reshape(1, -1))+0.05, s2, (self.n_obs//2, 1))
            y_ = np.concatenate((g1, g2), axis=0)
            X_ = x[i, :]*np.ones((self.n_obs, 1))
        
            if i:
                X = np.concatenate((X, X_))
                y = np.concatenate((y, y_))
            else:
                X = X_
                y = y_
        
        return X, y


class himmelblau(Function):
    '''
    Himmelblau function.
    '''
    def __init__(self, **kwargs):
        super(himmelblau, self).__init__(**kwargs)

        self.dim      = 2
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = 'Himmelblau'
        self.glob_min = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
        self.f_min    = 0.0
        self.f        = lambda x: (x[:, 0]**2 + x[:, 1] - 11)**2 + (x[:, 0] + x[:, 1]**2 - 7)**2 


class holder(Function):
    '''
    Hölder function.
    '''
    def __init__(self, **kwargs):
        super(holder, self).__init__(**kwargs)

        self.dim      = 2
        self.domain   = np.array([self.dim*[-10, 10]]).reshape(self.dim, 2)
        self.name     = 'Holder'
        self.glob_min = np.array([[8.05502, 9.66459], [-8.05502, -9.66459], [-8.05502, 9.66459], [8.05502, -9.66459]])
        self.f_min    = 0.0
        self.f        = lambda x: -np.abs(np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.exp(np.abs(1 - np.sqrt(x[:, 0]**2 + x[:, 1]**2)/np.pi))) + 19.2085


class submanifold_rosenbrock(Function):
    """
    Submanifold Rosenbrock problem function
    """
    def __init__(self, dim=20, sub_dim=8, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.sub_dim = sub_dim
        self.sigma = 1.0
        self.domain = np.array([self.dim*[-10, 10]]).reshape(self.dim, 2)
        
        self.name = f"Submanifold Rosenbrock (dim={dim}, subdim={sub_dim})"
        
        with fixed_numpy_seed(seed):
            A = np.random.randn(self.dim, self.sub_dim)
        Q, _ = np.linalg.qr(A)
        b = np.ones(sub_dim)

        x_min, _, _, _ = scipy.linalg.lstsq(Q.T, b)

        self.glob_min = np.expand_dims(x_min, 0)
        self.Q = Q
        self.f = lambda x: np.sum(
            100 * (x @ self.Q[:, 1:] - (x @ self.Q[:, :-1])**2)**2 + (1 - (x @ self.Q[:, :-1])) ** 2,
            axis=-1
        )


class tang(Function):
    '''
    d-dimensional Styblinsky-Tang function.
    '''
    def __init__(self, dim=20, **kwargs):
        super(tang, self).__init__(**kwargs)

        self.dim      = dim
        self.sigma    = 1.0
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = f'Styblinski-Tang ({self.dim} dim.)'
        self.glob_min = np.ones(self.dim).reshape(1, -1) * -2.903534

        self.f        = lambda x: np.sum(
            x ** 4.0 - 16.0 * x ** 2.0 + 5.0 * x, 
            axis=-1, 
        ) + 39.16617 * self.dim
        
        self.f_min    = 0.0
        # self.sigma    = lambda x: np.abs(x[:, 0] * np.sin(x[:, 1] - (-2.903534))) if 'sigma' in kwargs else lambda x: 1e-1
    
    # def __call__(self, x):
    #     if self.log_transform:
    #         if self.sigma(np.zeros((1, self.dim)))[0] == 1e-1:
    #             return np.log(self.f(x) + self.log_eps) 
    #         else:
    #             return np.log(self.f(x) + self.sigma(x) + self.log_eps) 
    #     else:
    #         if self.sigma(np.zeros((1, self.dim)))[0] == 1e-1:
    #             return self.f(x)
    #         else:
    #             return self.f(x) + self.sigma(x)

    # def sample(self, x):

    #     y = np.random.normal(self.__call__(x[0, :].reshape(1, -1)), self.sigma(x), (self.n_obs, 1))
    #     X = x[0, :]*np.ones((self.n_obs, 1))
        
    #     for i in range(1, x.shape[0]):
    #         y_ = np.random.normal(self.__call__(x[i, :].reshape(1, -1)), self.sigma(x), (self.n_obs, 1))
    #         X_ = x[i, :]*np.ones((self.n_obs, 1))

    #         y = np.concatenate((y, y_))
    #         X = np.concatenate((X, X_))
        
    #     return X, y