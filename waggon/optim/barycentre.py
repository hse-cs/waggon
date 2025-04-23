import gc
import numpy as np
from .surrogate import SurrogateOptimiser


class BarycentreSurrogateOptimiser(SurrogateOptimiser):
    def __init__(self, func, surr, acqf, **kwargs):
        super(BarycentreSurrogateOptimiser, self).__init__(func, surr, acqf, **kwargs)
        
        self.surr.verbose   = self.verbose
        self.acqf.verbose   = self.verbose
        self.candidates     = None
        self.surr_n_epochs  = surr.n_epochs
    

    def predict(self, X, y, n_pred=1):
        
        if self.surr.n_epochs == 1:
            del self.acqf.surr
            gc.collect()

        preds = []
        
        for i in range(self.surr_n_epochs):
            self.surr.n_epochs = 1
            self.surr.fit(X, y)
            preds.append(self.surr)
        
        self.acqf.surr = preds
        
        self.acqf.y = y.reshape(y.shape[0]//self.func.n_obs, self.func.n_obs)
        self.acqf.conds = X[::self.func.n_obs]
        
        next_x = []
        for _ in range(n_pred):
            next_x.append(self.numerical_search([1]))
        
        del preds
        gc.collect()

        return np.array(next_x)


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