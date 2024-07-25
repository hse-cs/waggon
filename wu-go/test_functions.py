import numpy as np


class function():
    def __init__(self, **kwargs):
        self.dim           = kwargs['dim'] if 'dim' in kwargs else 1
        self.domain        = kwargs['domain'] if 'domain' in kwargs else np.array([-10, 10])
        self.name          = kwargs['name'] if 'name' in kwargs else 'squre'
        self.glob_min      = kwargs['glob_min'] if 'glob_min' in kwargs else np.zeros(self.dim)
        self.f             = kwargs['f'] if 'f' in kwargs else lambda x: x**2
        self.log_transform = kwargs['log_transform'] if 'log_transform' in kwargs else True
        self.log_eps       = kwargs['log_eps'] if 'log_eps' in kwargs else 1e-8
        self.sigma         = kwargs['sigma'] if 'sigma' in kwargs else 1e-1
        self.n_obs         = kwargs['n_obs'] if 'n_obs' in kwargs else 100
    
    def __call__(self, x):
        if not self.log_transform:
            return self.f(x)
        else:
            return np.log(self.f(x) + self.log_eps)

    def sample(self, x): # TODO: change to any distribution

        y = np.random.normal(self.__call__(x[0, :].reshape(1, -1)), self.sigma, (self.n_obs, 1))
        X = x[0, :]*np.ones((self.n_obs, 1))
        
        for i in range(1, x.shape[0]):
            y_ = np.random.normal(self.__call__(x[i, :].reshape(1, -1)), self.sigma, (self.n_obs, 1))
            X_ = x[i, :]*np.ones((self.n_obs, 1))

            y = np.concatenate((y, y_))
            X = np.concatenate((X, X_))
        
        return X, y


class three_hump_camel(function):
    def __init__(self):
        super().__init__()
        self.dim      = 2
        self.domain   = np.array([[-5, 5], [-5, 5]])
        self.name     = 'Three Hump Camel'
        self.glob_min = np.zeros(self.dim)
        self.f        = lambda x: 2 * x[:, 0]**2 - 1.05 * x[:, 0]**4 + x[:, 0]**6 / 6 + x[:, 0]*x[:, 1] + x[:, 1]**2


class rosenbrock(function):
    def __init__(self, dim=20):
        super().__init__()
        self.dim      = dim
        self.domain   = np.array([self.dim*[-2, 2]]).reshape(self.dim, 2)
        self.name     = f'Rosenbrock ({self.dim} dim.)'
        self.glob_min = np.ones(self.dim)
        self.f        = lambda x: np.sum(np.array([100 * (x[:, i+1] - x[:, i] ** 2)**2 + (1 - x[:, i])**2 for i in range(self.dim - 1)]), axis=0).squeeze()


class tang(function):
    def __init__(self, dim=20):
        super().__init__()
        self.dim      = dim
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = f'Styblinski-Tang ({self.dim} dim.)'
        self.glob_min = np.ones(self.dim) * -2.903534
        self.f        = lambda x: np.sum(x**4 - 16*x**2 + 5*x + 39.16617*self.dim, axis=1).squeeze()


class ackley(function):
    def __init__(self):
        super().__init__()
        self.dim      = 2
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = 'Ackley'
        self.glob_min = np.array([0, 0])
        self.f        = lambda x: -20 * np.exp(-0.2*np.sqrt(0.5*(x[:, 0]**2 + x[:, 1]**2))) -  np.exp(0.5*(np.cos(2*np.pi**x[:, 0]) + np.cos(2*np.pi*x[:, 1]))) + np.e + 20


class levi(function):
    def __init__(self):
        super().__init__()
        self.dim      = 2
        self.domain   = np.array([self.dim*[-4, 6]]).reshape(self.dim, 2)
        self.name     = 'Lévi'
        self.glob_min = np.ones(self.dim)
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


class himmelblau(function):
    def __init__(self):
        super().__init__()
        self.dim      = 2
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = 'Himmelblau'
        self.glob_min = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
        self.f        = lambda x: (x[:, 0]**2 + x[:, 1] - 11)**2 + (x[:, 0] + x[:, 1]**2 - 7)**2 


class holder(function):
    def __init__(self):
        super().__init__()
        self.dim      = 2
        self.domain   = np.array([self.dim*[-10, 10]]).reshape(self.dim, 2)
        self.name     = 'Holder'
        self.glob_min = np.array([[8.05502, 9.66459], [-8.05502, -9.66459], [-8.05502, 9.66459], [8.05502, -9.66459]])
        self.f        = lambda x: -np.abs(np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.exp(np.abs(1 - np.sqrt(x[:, 0]**2 + x[:, 1]**2)/np.pi))) + 19.2085
