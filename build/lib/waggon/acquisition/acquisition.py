import numpy as np
from scipy.stats import norm, energy_distance as Wdist


class Acquisition():
    def __init__(self):
        self.y         = None
        self.name      = None
        self.surr      = None
    
    def __call__(self, x):
        pass


class EI(Acquisition):
    def __init__(self):
        super().__init__()
        self.name = 'EI'
    
    def __call__(self, x, **kwargs):
        mu, std = self.surr.predict(x.reshape(1, -1), **kwargs)

        z_ = np.min(self.y) - mu
        z  = z_ / (std + 1e-8)
        z_prob, z_dens = norm.cdf(z), norm.pdf(z)

        EI = z_ * z_prob + std * z_dens

        return -1.0 * EI


class CB(Acquisition):
    def __init__(self, kappa=2.0, minimise=True):
        super().__init__()
        self.name     = 'LCB' if minimise else 'UCB'
        self.kappa    = kappa
        self.minimise = minimise
    
    def __call__(self, x, **kwargs):
        mu, std = self.surr.predict(x.reshape(1, -1), **kwargs)

        if self.minimise:
            return mu - self.kappa * std # LCB
        else:
            return mu + self.kappa * std # UCB


class WU(Acquisition):
    def __init__(self, n_obs=100, kappa=2.0, minimise=True):
        super().__init__()
        self.name     = 'WU'
        self.n_obs    = n_obs
        self.kappa    = kappa
        self.minimise = minimise
    
    def __call__(self, candidates):
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        
        if candidates.shape[0] == self.y.shape[0] * self.y.shape[1]:
            y_gen = self.surr.sample(candidates)
        else:
            y_gen = np.concatenate([self.surr.sample(candidates) for _ in range(self.y.shape[1])])
        
        y_gen = y_gen.reshape(y_gen.shape[0]//self.y.shape[1], self.y.shape[1])

        mu = np.mean(y_gen, axis=-1)
        wu = [[Wdist(self.y[i, :], y_gen[j, :]) for i in range(self.y.shape[0])] for j in range(y_gen.shape[0])]
        wu = np.min(wu, axis=-1)

        if self.minimise:
            return mu - self.kappa * wu
        else:
            return mu + self.kappa * wu
