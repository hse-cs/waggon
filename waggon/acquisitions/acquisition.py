import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, energy_distance as Wdist

from .base import Acquisition


class EI(Acquisition):
    def __init__(self):
        '''
        Expected Improvement (EI) acquisition function.
        '''
        super(EI, self).__init__()
        
        self.name = 'EI'
    
    def __call__(self, x, **kwargs):
        '''
        Parameters
        ----------
        x : np.array of shape (n_samples * func.n_obs, func.dim)
            Candidate points.
        
        kwargs : dict
            Keyword arguments of the surrogate model.
        
        Returns
        -------
        EI : np.array of shape (n_samples, 1)
            EI values. They are multiplied by -1.0 for optimisation perposes.
        '''
        mu, std = self.surr.predict(x, **kwargs)

        z_ = np.min(self.y) - mu
        z  = z_ / (std + 1e-8)
        z_prob, z_dens = norm.cdf(z), norm.pdf(z)

        EI = z_ * z_prob + std * z_dens

        return -1.0 * EI


class CB(Acquisition):
    def __init__(self, kappa=2.0, minimise=True):
        '''
        Confidence Bound (CB) type acquisition function.

        Parameters
        ----------
        kappa : float, default = 2.0
            Exploration v. Exploitation coefficient.
        
        minimise : bool, default = True
            Whether the objective is minimised or not.
            If True, the acquisition function is Lower Confidence Bound (LCB).
            If False, the acquisition function is Upper Confidence Bound (UCB).
        '''
        super(CB, self).__init__()

        self.name     = 'LCB' if minimise else 'UCB'
        self.kappa    = kappa
        self.minimise = minimise
    
    def __call__(self, x, **kwargs):
        '''
        Parameters
        ----------
        x : np.array of shape (n_candidates * func.n_obs, func.dim)
            Candidates points.
        
        kwargs : dict
            Keyword arguments of the surrogate model.
        
        Returns
        -------
        regret : np.array of shape (n_candidates, 1)
            CB values.
        '''
        mu, std = self.surr.predict(x, **kwargs)

        if self.minimise:
            regret = mu - self.kappa * std # LCB
        else:
            regret = mu + self.kappa * std # UCB
        
        return regret


class WU(Acquisition):
    def __init__(self, kappa=2.0, minimise=True):
        '''
        Wasserstein Uncertainty based regret.

        Parameters
        ----------
        kappa : float, default = 2.0
            Exploration v. Exploitation coefficient.
        
        minimise : bool, default = True
            Whether the objective is minimised or not.
        '''
        super(WU, self).__init__()

        self.name     = 'WU'
        self.kappa    = kappa
        self.minimise = minimise
    
    def __call__(self, x):
        '''
        Parameters
        ----------
        x : np.array of shape (n_samples * func.n_obs, func.dim)
            Candidate points.
        
        Returns
        -------
        regret : np.array of shape (n_candidates, 1)
            Wasserstein uncertainty based regret.
        '''
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[0] == self.y.shape[0] * self.y.shape[1]:
            y_gen = self.surr.sample(x)
        else:
            y_gen = np.concatenate([self.surr.sample(x) for _ in range(self.y.shape[1])])
        
        y_gen = y_gen.reshape(y_gen.shape[0]//self.y.shape[1], self.y.shape[1])

        mu = np.mean(y_gen, axis=-1)
        wu = [[Wdist(self.y[i, :], y_gen[j, :]) for i in range(self.y.shape[0])] for j in range(y_gen.shape[0])]
        wu = np.min(wu, axis=-1)

        if self.minimise:
            regret = mu - self.kappa * wu
        else:
            regret = mu + self.kappa * wu
        
        return regret


class WU_IDW(Acquisition):
    def __init__(self, kappa=2.0, minimise=True, power=1.0):
        '''
        Inverse distance weighted Wasserstein Uncertainty based regret.

        Parameters
        ----------
        kappa : float, default = 2.0
            Exploration v. Exploitation coefficient.
        
        minimise : bool, default = True
            Whether the objective is minimised or not.
        '''
        super(WU_IDW, self).__init__()

        self.name     = 'WU-IDW'
        self.kappa    = kappa
        self.minimise = minimise
        self.power    = power

    def __call__(self, x):
        '''
        Parameters
        ----------
        x : np.array of shape (n_samples * func.n_obs, func.dim)
            Candidate points.
        
        Returns
        -------
        regret : np.array of shape (n_candidates, 1)
            Wasserstein uncertainty based regret.
        '''
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[0] == self.y.shape[0] * self.y.shape[1]:
            y_gen = self.surr.sample(x)
        else:
            y_gen = np.concatenate([self.surr.sample(x) for _ in range(self.y.shape[1])])
        
        y_gen = y_gen.reshape(y_gen.shape[0]//self.y.shape[1], self.y.shape[1])

        mu = np.mean(y_gen, axis=-1)
        wu = [[Wdist(self.y[i, :], y_gen[j, :]) for i in range(self.y.shape[0])] for j in range(y_gen.shape[0])]
        
        dist = cdist(self.conds, x)

        weights = 1.0 / (dist + 1e-12)**self.power
        
        weights /= weights.sum(axis=0)
        
        std = np.dot(wu, weights)
        
        std = np.diag(std)

        if self.minimise:
            regret = mu - self.kappa * std
        else:
            regret = mu + self.kappa * std
        
        return regret
