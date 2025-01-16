import numpy as np

from .base import Function


class three_hump_camel(Function):
    '''
    Three Hump Camel function.
    '''
    def __init__(self, **kwargs):
        super(three_hump_camel, self).__init__()
        
        self.dim      = 2
        self.domain   = np.array([[-5, 5], [-5, 5]])
        self.name     = 'Three Hump Camel'
        self.glob_min = np.zeros(self.dim).reshape(1, -1)
        self.f        = lambda x: 2 * x[:, 0]**2 - 1.05 * x[:, 0]**4 + x[:, 0]**6 / 6 + x[:, 0]*x[:, 1] + x[:, 1]**2


class rosenbrock(Function):
    '''
    d-dimensional Rosenbrock function.
    '''
    def __init__(self, dim=20, **kwargs):
        super(rosenbrock, self).__init__()

        self.dim      = dim
        self.domain   = np.array([self.dim*[-2, 2]]).reshape(self.dim, 2)
        self.name     = f'Rosenbrock ({self.dim} dim.)'
        self.glob_min = np.ones(self.dim).reshape(1, -1)
        self.f        = lambda x: np.sum(np.array([100 * (x[:, i+1] - x[:, i] ** 2)**2 + (1 - x[:, i])**2 for i in range(self.dim - 1)]), axis=0).squeeze()


class tang(Function):
    '''
    d-dimensional Styblinsky-Tang function.
    '''
    def __init__(self, dim=20, **kwargs):
        super(tang, self).__init__()

        self.dim      = dim
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = f'Styblinski-Tang ({self.dim} dim.)'
        self.glob_min = np.ones(self.dim).reshape(1, -1) * -2.903534
        self.f        = lambda x: np.sum(x**4 - 16*x**2 + 5*x + 39.16617*self.dim, axis=1).squeeze()


class ackley(Function):
    '''
    Ackley function.
    '''
    def __init__(self, **kwargs):
        super(ackley, self).__init__()

        self.dim      = 2
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = 'Ackley'
        self.glob_min = np.zeros(self.dim).reshape(1, -1)
        self.f        = lambda x: -20 * np.exp(-0.2*np.sqrt(0.5*(x[:, 0]**2 + x[:, 1]**2))) -  np.exp(0.5*(np.cos(2*np.pi**x[:, 0]) + np.cos(2*np.pi*x[:, 1]))) + np.e + 20


class levi(Function):
    '''
    Lévi function.
    '''
    def __init__(self, **kwargs):
        super().__init__()

        self.dim      = 2
        self.domain   = np.array([self.dim*[-4, 6]]).reshape(self.dim, 2)
        self.name     = 'Lévi'
        self.glob_min = np.ones(self.dim).reshape(1, -1)
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
        super(himmelblau, self).__init__()

        self.dim      = 2
        self.domain   = np.array([self.dim*[-5, 5]]).reshape(self.dim, 2)
        self.name     = 'Himmelblau'
        self.glob_min = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])
        self.f        = lambda x: (x[:, 0]**2 + x[:, 1] - 11)**2 + (x[:, 0] + x[:, 1]**2 - 7)**2 


class holder(Function):
    '''
    Hölder function.
    '''
    def __init__(self, **kwargs):
        super(holder, self).__init__()

        self.dim      = 2
        self.domain   = np.array([self.dim*[-10, 10]]).reshape(self.dim, 2)
        self.name     = 'Holder'
        self.glob_min = np.array([[8.05502, 9.66459], [-8.05502, -9.66459], [-8.05502, 9.66459], [8.05502, -9.66459]])
        self.f        = lambda x: -np.abs(np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.exp(np.abs(1 - np.sqrt(x[:, 0]**2 + x[:, 1]**2)/np.pi))) + 19.2085


class submanifold(Function):
    """
    Submanifold Rosenbrock function.
    """
    def __init__(self, dim=20, sub_dim=4, **kwargs):
        super(submanifold, self).__init__()
        self.dim = dim
        self.sub_dim = sub_dim
        self.domain = np.array([[-10, 10]] * self.dim).reshape(self.dim, 2)
        self.name = f'Submanifold Rosenbrock ({self.dim}/{self.sub_dim} dim.)'
        self.glob_min = np.zeros(self.dim).reshape(1, -1)

        A, _ = np.linalg.qr(np.random.randn(self.dim, self.sub_dim))
        self.f = lambda x: np.sum((x @ A - (x @ A)**2)**2 + (1 - (x @ A)**2), axis=1)


class nonlinear_submanifold(submanifold):
    """
    Nonlinear Submanifold Hump Problem
    """
    def __init__(self, dim=40, sub_dim=2, **kwargs):
        super(nonlinear_submanifold, self).__init__()
        self.name = f"Nonlinear Submanifold Hump Problem ({self.dim}/{self.sub_dim})"
        self.glob_min = np.zeros(self.dim).reshape(1, -1)

        A, _ = np.linalg.qr(np.random.randn(self.sub_dim, self.dim)) # A ∈ R^(16 x 40)
        B, _ = np.linalg.qr(np.random.randn(2, self.sub_dim)) # B ∈ R^(2 x 16)

        self.non_lin_f = np.tanh

        self.three_hump_camel = lambda x: 2 * x[:, 0]**2 - 1.05 * x[:, 0]**4 + x[:, 0]**6 / 6 + x[:, 0]*x[:, 1] + x[:, 1]**2

        self.f = lambda x: self.three_hump_camel(B @ self.non_lin_f(A @ x))


class optimal_control(Function):
    def __init__(self, T, dt=1.0, z0=0.8, z_ref=0.7, variance_range=(0.01, 0.1), n_obs=100, **kwargs):
        super().__init__(dim=1, n_obs=n_obs, variance_range=variance_range)
        self.T = T
        self.dt = dt
        self.z0 = z0
        self.z_ref = z_ref
        self.variance_range = variance_range  

    def state_dynamics(self, z, x):
        return z + self.dt * (z**3 - x)

    def objective_function(self, z_trajectory):
        return (1 / self.T) * np.sum((z_trajectory - self.z_ref) ** 2)

    def __call__(self, x_sequence):
        x_sequence = np.asarray(x_sequence).flatten()  
        if len(x_sequence) != self.T:
            raise ValueError(f"Expected x_sequence of length {self.T}, but got {len(x_sequence)}")

        z_trajectory = np.zeros(self.T + 1)
        z_trajectory[0] = self.z0
        for t in range(self.T):
            z_trajectory[t + 1] = self.state_dynamics(z_trajectory[t], x_sequence[t])
        return self.objective_function(z_trajectory)

    def sample(self, x):

        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None] 
        if x.shape[0] != self.T:
            raise ValueError(f"Expected x with shape ({self.T}, n), but got {x.shape}")

        y = []
        X = []
        for t in range(self.T):
            min_var, max_var = self.variance_range
            variance = min_var + (max_var - min_var) * (np.sin(2 * np.pi * t / self.T) ** 2)
            noise = np.random.normal(0, np.sqrt(variance), self.n_obs)

            x_t = x[t, 0] * np.ones(self.n_obs)
            y_t = self.__call__(x_t) + noise

            X.append(x_t.reshape(-1, 1))
            y.append(y_t.reshape(-1, 1))

        return np.vstack(X), np.vstack(y)