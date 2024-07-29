import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from scipy.optimize import minimize
from waggon.utils import get_olhs_num
from sklearn.base import BaseEstimator


class Optimiser(BaseEstimator):
    def __init__(self, func, surr, acqf, **kwargs):
        self.func           = func
        self.surr           = surr
        self.acqf           = acqf
        self.num_opt        = kwargs['num_opt'] if 'num_opt' in kwargs else False
        self.fix_candidates = False if self.num_opt else (kwargs['fix_candidates'] if 'fix_candidates' in kwargs else True)
        self.max_iter       = kwargs['max_iter'] if 'max_iter' in kwargs else 100 
        self.plot_res       = kwargs['plot_res'] if 'plot_res' in kwargs else False 
        self.opt_eps        = kwargs['opt_eps'] if 'opt_eps' in kwargs else 1e-1
        self.n_candidates   = kwargs['n_candidates'] if 'n_candidates' in kwargs else (1 if self.num_opt else 101**2)
        self.olhs           = kwargs['olhs'] if 'olhs' in kwargs else True
        self.lhs_seed       = kwargs['lhs_seed'] if 'lhs_seed' in kwargs else None
        self.verbosity      = kwargs['verbosity'] if 'verbosity' in kwargs else 1
        self.candidates     = None
    
    def fit(self, X, y, **kwargs):
        self.surr.fit(X, y, **kwargs)
    
    def create_candidates(self, N=None):
        if N is None:
            N = self.n_candidates

        if self.olhs:
            N = max(N, get_olhs_num(self.func.dim)[0])
            strength = 2
        else:
            strength = 1

        lhs_       = qmc.LatinHypercube(d=self.func.domain.shape[0], scramble=True, strength=strength, seed=self.lhs_seed)
        candidates = lhs_.random(N)
        candidates = qmc.scale(candidates, self.func.domain[:, 0], self.func.domain[:, 1])
        return candidates
    
    def numerical_search(self):
        best_x = None
        best_acqf = np.inf

        candidates = self.create_candidates()

        for x0 in candidates:

            opt_res = minimize(fun=self.acqf, x0=x0, bounds=self.func.domain, method='L-BFGS-B')

            if opt_res.fun < best_acqf:
                best_acqf = opt_res.fun
                best_x = opt_res.x
        
        return best_x
    
    def direct_search(self):
        
        if self.fix_candidates:
            if self.candidates is None:
                self.candidates = self.create_candidates()
        else:
            self.candidates = self.create_candidates()
        
        acqf_values = self.acqf(self.candidates)
        best_x = self.candidates[np.argmin(acqf_values)]
        
        return best_x

    def predict(self):

        if self.num_opt:
            next_x = self.numerical_search()
        else:
            next_x = self.direct_search()

        return np.array([next_x])
    
    def optimise(self, X=None, y=None):

        if X is None:
            X = self.create_candidates(N=1)
            X, y = self.func.sample(X)

        self.res, self.params = np.array([[np.min(self.func(X))]]), np.array([X[np.argmin(self.func(X)), :]])

        if self.verbosity == 0:
            opt_loop = range(self.max_iter)
        else:
            opt_loop = tqdm(range(self.max_iter), desc='Optimisation loop', leave=True, position=0)

        for i in opt_loop:
            
            self.fit(X, y, verbosity=self.verbosity)
            
            self.acqf.y = y.reshape(y.shape[0]//self.func.n_obs, self.func.n_obs)
            self.acqf.surr = self.surr

            next_x = self.predict()
            next_f = np.array([self.func(next_x)])

            if next_f <= self.res[-1, :]:
                self.res = np.concatenate((self.res, next_f))
                self.params = np.concatenate((self.params, next_x))
            else:
                self.res = np.concatenate((self.res, self.res[-1, :].reshape(1, -1)))
                self.params = np.concatenate((self.params, self.params[-1, :].reshape(1, -1)))

            X_, y_ = self.func.sample(next_x)
            
            X = np.concatenate((X, X_))
            y = np.concatenate((y, y_))

            if np.linalg.norm(self.res[-1]) <= self.opt_eps:
                print('Experiment finished successfully')
                break
        
        if np.linalg.norm(self.res[-1]) > self.opt_eps:
            print('Experiment failed')
