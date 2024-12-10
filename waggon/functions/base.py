import numpy as np


class Function:
    def __init__(self, **kwargs):
        '''
        Base class for a synthetic black box.

        Parameters
        ----------
        dim : int, default = 1
            Dimensionality of the parameter space (Input dimension)
        
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

        self.dim           = kwargs.get('dim', 1)  # TODO: add q-dimensional output?
        self.domain        = kwargs.get('domain', np.array([[-10, 10]]))
        self.name          = kwargs.get('name', 'parabola')
        self.glob_min      = kwargs.get('glob_min', np.zeros(self.dim))
        self.f             = kwargs.get('f', lambda x: x ** 2)
        self.log_transform = kwargs.get('log_transform', True)
        self.log_eps       = kwargs.get('log_eps', 1e-8)
        self.sigma         = kwargs.get('sigma', 1e-1)
        self.n_obs         = kwargs.get('n_obs', 1)
    
    def __call__(self, x):
        '''
        Call of the black-box function

        Parameters
        ----------
        x : np.array of shape (n_samples, func.dim)
            Argument for which the function is called.
        
        Returns
        -------
        self.f(x) : np.array of shape (n_samples, 1)
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

        n_samples, _ = x.shape
        n_obs = self.n_obs

        input_dim = self.dim
        output_dim = 1     

        X = np.expand_dims(x, axis=1) * np.ones((1, n_obs, 1))
        X = X.reshape(-1, input_dim)

        mu = self.__call__(X) # shape of [n_samples * n_obs, output_dim]
        y = np.random.normal(mu, self.sigma)
        
        if y.ndim == 1:
            # Crutch. It will be removed in the future
            y = np.expand_dims(y, axis=-1)
        
        return X, y
