from .base import Surrogate

import GPy
import numpy as np

class GP(Surrogate):
    def __init__(self, **kwargs):
        super(GP, self).__init__()

        self.name     = 'GP'
        self.model    = kwargs['model'] if 'model' in kwargs else None
        self.n_epochs = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
        self.verbose  = kwargs['verbose'] if 'verbose' in kwargs else 1
    
    def fit(self, X, y):

        if self.model is None:
            self.model = GPy.models.GPRegression(X, y)
        
        self.model.set_XY(X=X, Y=y)
        
        self.model.optimize(max_iters=self.n_epochs, messages=True if self.verbose > 1 else False)

    def predict(self, X):

        f, var = self.model.predict(X)
        std = np.sqrt(var)

        return f, std
