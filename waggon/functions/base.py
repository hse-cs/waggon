import numpy as np


class Function:
    def __init__(self, **kwargs):
        '''
        Base class for a synthetic black box.

        Parameters
        ----------
        dim : int, default = 1
            Dimensionality of the parameter space
        
        domain : np.array of shape (dim, 2), default = np.array([-10, 10])
            Contains bounds of the parameter space with the first column containing the smallest bound, and
            the second column containing the largest bound for the corresponding dimensions.
        
        name : str, default = 'parabola'
            Name of the function.
        
        glob_min : np.array of shape (dim,), default = np.zeros(dim)
            Point of global minimum of the function.
        
        f : Callable, default = lambda x: x**2
            The experiment function
        
        log_transform : bool, default = True
            Whether a log transformation is applied when calling the function or not.
        
        log_eps : float, default = 1e-8
            Small real value to prevent log(0) when log_transform is True.
        
        sigma : float, default = 1e-1
            Standard deviation of the output distribution.
        
        n_obs : int, default = 100
            Number of observations to sample for each point of the parameter space.
        '''
        super(Function, self).__init__()

        self.dim           = kwargs['dim'] if 'dim' in kwargs else 1 # TODO: add q-dimensional output?
        self.domain        = kwargs['domain'] if 'domain' in kwargs else np.array([[-10, 10]])
        self.name          = kwargs['name'] if 'name' in kwargs else 'parabola'
        self.glob_min      = kwargs['glob_min'] if 'glob_min' in kwargs else np.zeros(self.dim)
        self.f             = kwargs['f'] if 'f' in kwargs else lambda x: x**2
        self.log_transform = kwargs['log_transform'] if 'log_transform' in kwargs else True
        self.log_eps       = kwargs['log_eps'] if 'log_eps' in kwargs else 1e-8
        self.sigma         = kwargs['sigma'] if 'sigma' in kwargs else 1e-1
        self.n_obs         = kwargs['n_obs'] if 'n_obs' in kwargs else 1
    
    def __call__(self, x):
        '''
        Call of the black-box function

        Parameters
        ----------
        x : np.array of shape (n_samples, func.dim)
            Argument for which the function is called.
        
        Returns
        -------
        self.f(x) : np.array of shape (n_samples, func.dim)
            Black-box function values.
        '''
        if not self.log_transform:
            return self.f(x)
        else:
            return np.log(self.f(x) + self.log_eps)

    def sample(self, x): # TODO: change to any distribution
        '''
        Sample from the black-box response distribution

        Parameters
        ----------
        x : np.array of shape (n_samples, func.dim)
            Argument for which the function is called.
        
        Returns
        -------
        X : np.array of shape (n_samples * func.n_obs, func.dim)
            Training input points. Correspond to input parameter x.

        y : np.array of shape (n_samples * func.n_obs, func.dim)
            Target values of the black-box function.
        '''

        y = np.random.normal(self.__call__(x[0, :].reshape(1, -1)), self.sigma, (self.n_obs, 1))
        X = x[0, :]*np.ones((self.n_obs, 1))
        
        for i in range(1, x.shape[0]):
            y_ = np.random.normal(self.__call__(x[i, :].reshape(1, -1)), self.sigma, (self.n_obs, 1))
            X_ = x[i, :]*np.ones((self.n_obs, 1))

            y = np.concatenate((y, y_))
            X = np.concatenate((X, X_))
        
        return X, y
