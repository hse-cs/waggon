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

            self.model = GPy.models.GPRegression(X, y,
                                                 kernel = self.kernel,
                                                 mean_function = self.mean)
        
        self.model.set_XY(X=X, Y=y)
        
        self.model.optimize(messages=True if self.verbose > 1 else False)

    def predict(self, X):

        f, var = self.model.predict(X)
        std = np.sqrt(var)

        return f, std
