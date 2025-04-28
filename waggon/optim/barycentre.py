import gc
import numpy as np
from .surrogate import SurrogateOptimiser


class BarycentreSurrogateOptimiser(SurrogateOptimiser):
    def __init__(self, func, surr, acqf, **kwargs):
        super(BarycentreSurrogateOptimiser, self).__init__(func, surr, acqf, **kwargs)
        self.acqf.n_epochs = self.surr.n_epochs

    def predict(self, X, y):

        self.surr.save_epoch = int(self.surr.n_epochs * self.acqf.wp)

        self.surr.fit(X, y)
        self.acqf.surr = [self.surr]
        
        x0 = None
        if self.num_opt_start == 'fmin':
            x0 = X[np.argmin(y)]
        
        next_x = self.numerical_search(x0=x0)

        del self.acqf.surr
        gc.collect()
        
        if next_x in X:
            next_x += np.random.normal(0, self.jitter, 1)
        
        return np.array([next_x])


class EnsembleBarycentreSurrogateOptimiser(SurrogateOptimiser):
    def __init__(self, func, surr, acqf, **kwargs):
        super(EnsembleBarycentreSurrogateOptimiser, self).__init__(func, surr, acqf, **kwargs)
        
        for surr in self.surr:
            surr.verbose   = self.verbose
        self.acqf.verbose   = self.verbose
        self.candidates     = None
    

    def predict(self, X, y, n_pred=1):
        
        surrs = []
        
        for surr in self.surr:
            
            surr.fit(X, y)
            surrs.append(surr)
        
        self.acqf.surr = surrs
        
        self.acqf.y = y.reshape(y.shape[0]//self.func.n_obs, self.func.n_obs)
        self.acqf.conds = X[::self.func.n_obs]
        
        next_x = []
        for _ in range(n_pred):
            next_x.append(self.numerical_search([1]))
        
        del surrs
        gc.collect()

        return np.array(next_x)