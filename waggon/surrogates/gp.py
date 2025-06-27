from .base import Surrogate

import GPy
import numpy as np

class GP(Surrogate):
    def __init__(self, **kwargs):
        super(GP, self).__init__()

        self.name     = 'GP'
        self.model    = kwargs['model'] if 'model' in kwargs else None
        self.kernel   = kwargs['kernel'] if 'kernel' in kwargs else None
        self.mean     = kwargs['mean'] if 'mean' in kwargs else None
        self.verbose  = kwargs['verbose'] if 'verbose' in kwargs else 1
    
    def fit(self, X, y):

        if self.model is None:
            
            if self.kernel is None:
                self.kernel = GPy.kern.Matern32(input_dim=X.shape[-1], lengthscale=1.0)

            self.model = GPy.models.GPRegression(X.astype(np.float128), y.astype(np.float128),
                                                 kernel = self.kernel,
                                                 mean_function = self.mean)
        self.mu, self.std = np.mean(y), np.std(y)
        y = (y - self.mu) / (self.std + 1e-8)
        
        self.model.set_XY(X=X.astype(np.float128) , Y=y.astype(np.float128))
        
        self.model.optimize(optimizer='lbfgsb')

    def predict(self, X):

        f, var = self.model.predict(X.astype(np.float128))
        std = np.sqrt(var)

        f += self.mu
        std *= self.std

        return f.astype(np.float64), std.astype(np.float64)
