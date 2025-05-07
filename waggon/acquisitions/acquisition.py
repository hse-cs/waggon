import torch
import gpytorch
import numpy as np
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.stats import norm, energy_distance as Wdist

from joblib import Parallel, delayed

import multiprocessing
multiprocessing.set_start_method('spawn')

from .base import Acquisition


class EI(Acquisition):
    def __init__(self, log_transform=True):
        '''
        Expected Improvement (EI) acquisition function.
        '''
        super(EI, self).__init__()
        
        self.name = 'EI'
        self.log_transform = log_transform
    
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
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        mu, std = self.surr.predict(x, **kwargs)

        z_ = np.min(self.y) - mu
        z  = z_ / (std + 1e-8)
        z_prob, z_dens = norm.cdf(z), norm.pdf(z)

        EI = z_ * z_prob + std * z_dens

        if self.log_transform:
            return -1.0 * np.log(EI + 1e-6)
        else:
            return -1.0 * EI


class CB(Acquisition):
    def __init__(self, kappa=1.0, minimise=True):
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
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
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



class PI(Acquisition):
    def __init__(self, xi=0.01, minimise=True):
        """
        Probability of Improvement (PI) acquisition function.

        Parameters
        ----------
        xi : float, default=0.01
            Exploration vs exploitation trade-off parameter.
        
        minimise : bool, default=True
            Whether the objective is minimised or maximised.
        """
        super(PI, self).__init__()
        self.name = 'PI'
        self.xi = xi
        self.minimise = minimise

    def __call__(self, x, **kwargs):
        """
        Calculate the PI acquisition value for candidate points.

        Parameters
        ----------
        x : np.array of shape (n_candidates, func.dim)
            Candidate points.

        Returns
        -------
        PI : np.array of shape (n_candidates,)
            Probability of Improvement values.
        """
        mu, std = self.surr.predict(x, **kwargs)
        best_y = np.min(self.y) if self.minimise else np.max(self.y)
        z = (best_y - mu - self.xi) / (std + 1e-8)
        pi = norm.cdf(z)
        return pi


class ES(Acquisition):
    def __init__(self):
        """
        Entropy Search (ES) acquisition function.
        """
        super(ES, self).__init__()
        self.name = "EntropySearch"

    def __call__(self, x, **kwargs):
        """
        Compute the reduction in entropy of the posterior over the location of the global minimum.

        Parameters:
        ----------
        x : np.array of shape (n_samples, func.dim)
            Candidate points.

        Returns:
        -------
        es : np.array
            The entropy reduction at each candidate point.
        """
        mu, std = self.surr.predict(x, **kwargs)
        
        p_min = norm.cdf((np.min(self.y) - mu) / (std + 1e-8))
        entropy = p_min * np.log(p_min + 1e-8) + (1 - p_min) * np.log(1 - p_min + 1e-8)
        
        return entropy


class KG(Acquisition):
    def __init__(self):
        """
        Knowledge Gradient (KG) acquisition function.
        """
        super(KG, self).__init__()
        self.name = "KnowledgeGradient"

    def __call__(self, x, **kwargs):
        """
        Compute the Knowledge Gradient (KG) for candidate points.

        Parameters:
        ----------
        x : np.array of shape (n_samples, func.dim)
            Candidate points.

        Returns:
        -------
        kg : np.array
            KG values at the candidate points.
        """
        mu, std = self.surr.predict(x, **kwargs)
        
        delta = np.min(self.y) - mu
        z = delta / (std + 1e-8)
        kg = std * (z * norm.cdf(z) + norm.pdf(z))

        return kg


class OTUCB(Acquisition):
    def __init__(self, robust=False, parallel=0):
        super(OTUCB, self).__init__()
        
        self.name = 'OTUCB'
        self.surr = None
        self.robust = robust
        self.verbose = 1
        self.parallel = parallel
        self.L = 1.0
        self.y_mu = 0.0
        self.w = 0.1 * np.ones(10).reshape(-1, 1)
        
    def __call__(self, x):
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        mu = []
        x = torch.tensor(x).float()
        
        for surr in self.surr:
            with torch.no_grad(), gpytorch.settings.fast_computations():
                mu.append(surr.likelihood(surr(x)).mean[0, 0, :].detach().numpy())
        
        mu = self.w * (np.array(mu) + self.y_mu)
        eps = np.std(mu)
        mu = np.sum(mu, axis=0)
        
        if self.robust:
            return mu + self.L * eps
        else:
            return mu - self.L * eps


class GP_OTUCB(Acquisition):
    def __init__(self, wf='u', ws='h', wp=0.8, robust=False, parallel=0):
        super(GP_OTUCB, self).__init__()
        
        self.name = 'GP_OTUCB'
        self.surr = None
        self.robust = robust
        self.verbose = 1
        self.wf = wf
        self.parallel = parallel
        self.L = 1.0
    
    def __single_pred(self, surr):
        return surr.predict(self.x)[0]
        
    def __call__(self, x):
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        if self.parallel:
            mu = np.array(Parallel(n_jobs=self.parallel, prefer="threads")(delayed(self.__single_pred)(surr) for surr in self.surr))
        else:
            mu = []
            for surr in self.surr:
                mu.append(surr.predict(x)[0])
            mu = np.array(mu)
        
        if self.wf == 'l':
            w = np.linspace(1e-2, 1, mu.shape[1])
        elif self.wf == 'c':
            w = np.sin(np.linspace(1e-2, 1.57, mu.shape[1]))**2
        elif self.wf == 's':
            w = expit(np.linspace(-3, 3, mu.shape[1]))
        elif self.wf == 'e':
            w = np.exp(np.linspace(-2, 2, mu.shape[1]))
        else:
            w = np.ones(mu.shape[1])
        
        w /= np.sum(w)
        
        mu, eps = np.sum(w * mu, axis=0), np.std(mu)
        
        if self.robust:
            return mu + self.L * eps
        else:
            return mu - self.L * eps
