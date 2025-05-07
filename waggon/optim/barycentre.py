import gc
import numpy as np
from .surrogate import SurrogateOptimiser


class BarycentreSurrogateOptimiser(SurrogateOptimiser):
    def __init__(self, func, surr, acqf, **kwargs):
        super(BarycentreSurrogateOptimiser, self).__init__(func, surr, acqf, **kwargs)
        self.clear_surr = kwargs['clear_surr'] if 'clear_surr' in kwargs else False
        self.surr.models_dir = f'models_{self.seed}_{"robust" if self.acqf.robust else "optimist"}'
        self.surr.checkpoints = np.arange(109, self.surr.n_epochs, 10)
    
    def get_lip(self, X, y):
        idx = np.array(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[0]))).T.reshape(-1, 2)
        idx = idx[idx[:, 0] != idx[:, 1]]
        L = np.max(np.linalg.norm(y[idx[:, 0]] - y[idx[:, 1]], axis=-1) / np.linalg.norm(X[idx[:, 0]] - X[idx[:, 1]], axis=-1))
        return L

    def predict(self, X, y):
        
        for j in range(self.predict_runs):

            self.surr.fit(X, y)
            self.acqf.L = self.get_lip(X, y)
            self.acqf.y_mu = self.surr.y_mu.item()
            self.acqf.surr = []
            for epoch in self.surr.checkpoints:
                self.acqf.surr.append(self.surr.load_model(epoch=epoch, return_model=True))
            
            next_x = self.numerical_search(x0=X[np.argmin(y)])

            if not np.any(np.linalg.norm(X - next_x, axis=-1) < 1e-6):
                break
        
        if j == self.predict_runs - 1:
            next_x += np.random.normal(0, self.eps, 1)
        
        if self.clear_surr:
            del self.acqf.surr
            gc.collect()
        
        return np.array([next_x])


class EnsembleBarycentreSurrogateOptimiser(SurrogateOptimiser):
    def __init__(self, func, surr, acqf, **kwargs):
        super(EnsembleBarycentreSurrogateOptimiser, self).__init__(func, surr, acqf, **kwargs)
        
        for surr in self.surr:
            surr.verbose   = self.verbose
        self.acqf.verbose   = self.verbose
    
    def get_lip(self, X, y):
        idx = np.array(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[0]))).T.reshape(-1, 2)
        idx = idx[idx[:, 0] != idx[:, 1]]
        L = np.max(np.linalg.norm(y[idx[:, 0]] - y[idx[:, 1]], axis=-1) / np.linalg.norm(X[idx[:, 0]] - X[idx[:, 1]], axis=-1))
        return L

    def predict(self, X, y):
        
        surrs = []
        
        for surr in self.surr:
            surr.fit(X, y)
            surrs.append(surr)
        
        self.acqf.surr = surrs
        self.acqf.L = self.get_lip(X, y)
        
        x0 = None
        if self.num_opt_start == 'fmin':
            x0 = X[np.argmin(y)]
        
        next_x = self.numerical_search(x0=x0)
        
        del self.acqf.surr
        gc.collect()

        return np.array([next_x])
