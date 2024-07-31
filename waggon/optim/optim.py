import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


class Optimiser(BaseEstimator):
    def __init__(self, func, surr, acqf, **kwargs):
        '''
        Black-box optimiser.

        Parameter
        ----------
        func : waggon.functions.Function #TODO: callable
            Black-box function to be optimised.
        
        surr : # TODO: base surrogate class
            Surrogate model for the black-box function
        
        acqf : waggon.acquisition.Acquisition
            Acquisition function defining the optimisation strategy
        
        max_iter : int, default = 100
            Maximum number of optimisation loop iterations.
        
        opt_eps : float, default = 1e-1
            Epsilon-solution criterion value.

        num_opt : bool, default = False
            Whether the acquisition function is optimised numerically or not. If False,
            the search of the next best parameters is done via direct search.

        fix_candidates : bool, default = False if num_opt else True
            Whether the candidate points should be fixed or not.
        
        n_candidates : int, default = 1 if num_opt else 101**2
            Number of candidates points.
        
        olhs : bool, default = True
            Whether orthogonal Latin hypercube sampling (LHS) is used for choosing candidate points.
            If False, simple LHS is used.

        lhs_seed : int, default = None
            Controls the randomness of candidates sampled via (orthogonal) LHS.
        
        verbose : int, default = 1
            Controls verbosity when fitting and predicting. By default only a progress bar over the
            optimisation loop is displayed.
        '''
        self.func           = func
        self.surr           = surr
        self.acqf           = acqf
        self.max_iter       = kwargs['max_iter'] if 'max_iter' in kwargs else 100
        self.opt_eps        = kwargs['opt_eps'] if 'opt_eps' in kwargs else 1e-1
        self.num_opt        = kwargs['num_opt'] if 'num_opt' in kwargs else False
        self.fix_candidates = False if self.num_opt else (kwargs['fix_candidates'] if 'fix_candidates' in kwargs else True)
        self.n_candidates   = kwargs['n_candidates'] if 'n_candidates' in kwargs else (1 if self.num_opt else 101**2)
        self.olhs           = kwargs['olhs'] if 'olhs' in kwargs else True
        self.lhs_seed       = kwargs['lhs_seed'] if 'lhs_seed' in kwargs else None
        self.verbose        = kwargs['verbose'] if 'verbose' in kwargs else 1
        self.candidates     = None
    
    def fit(self, X, y, **kwargs):
        '''
        Fit the surrogate model on a training set (X, y)

        Parameters
        ----------
        X : np.array of shape (n_samples * func.n_obs, func.dim)
            Training input points.

        y : np.array of shape (n_samples * func.n_obs, func.dim)
            Target values of the black-box function.

        kwargs : dict, default = None
            Keyword arguments of the surrogate model
        '''
        self.surr.fit(X, y, **kwargs)
    
    def create_candidates(self, N=None):
        '''
        Creates candidate points among which the next best parameters will be selected.
        Also used for selecting the initial points of the optimisation process.

        Parameters
        ----------
        N : int, default = self.n_candidates
            Number of points to sample

        Returns
        -------
        candidates : np.array of shape (N, func.dim)
            Candidates points.
        '''

        N = self.n_candidates if N is None else N

        if self.olhs:
            N = max(N, _get_olhs_num(self.func.dim)[0])
            strength = 2
        else:
            strength = 1

        lhs_       = qmc.LatinHypercube(d=self.func.domain.shape[0], scramble=True, strength=strength, seed=self.lhs_seed)
        candidates = lhs_.random(N)
        candidates = qmc.scale(candidates, self.func.domain[:, 0], self.func.domain[:, 1])
        return candidates
    
    def numerical_search(self):
        '''
        Numerical optimisation of the acquisition function.

        Returns
        -------
        best_x : np.array of shape (1, func.dim) # TODO: check return type
            Predicted optimum of the acquisition function.
        '''
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
        '''
        Direct search for the optimum of the acquisition function.

        Returns
        -------
        best_x : np.array of shape (1, func.dim) # TODO: check return type
            Predicted optimum of the acquisition function
        '''
        
        if self.fix_candidates:
            if self.candidates is None:
                self.candidates = self.create_candidates()
        else:
            self.candidates = self.create_candidates()
        
        acqf_values = self.acqf(self.candidates)
        best_x = self.candidates[np.argmin(acqf_values)]
        
        return best_x

    def predict(self):
        '''
        Predicts the next best set of parameter values by optimising the acquisition function.

        Returns
        -------
        next_x : np.array of shape (1, func.dim)
        '''

        if self.num_opt:
            next_x = self.numerical_search()
        else:
            next_x = self.direct_search()

        return np.array([next_x])
    
    def optimise(self, X=None, y=None):
        '''
        Runs the optimisation of the black-box function.

        Parameters
        ----------
        X : np.array of shape (n_samples * func.n_obs, func.dim), default = None
            Training input points. If None, an initial set of points and a corresponding training set
            will be created via self.create_candidates and func.sample functions respectively. Values will be created
            for both X and y.

        y : np.array of shape (n_samples * func.n_obs, func.dim), default = None
            Target values of the black-box function.
        '''

        if X is None:
            X = self.create_candidates(N=1) # TODO: krivo, ispravit'
            X, y = self.func.sample(X)

        self.res, self.params = np.array([[np.min(self.func(X))]]), np.array([X[np.argmin(self.func(X)), :]])

        if self.verbose == 0:
            opt_loop = range(self.max_iter)
        else:
            opt_loop = tqdm(range(self.max_iter), desc='Optimisation loop', leave=True, position=0)

        for i in opt_loop:
            
            self.fit(X, y, verbose=self.verbose)
            
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


_PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])

def _get_olhs_num(n):
    '''
    Private function to select the number of sampling points for orthogonal Latin hypercube sampling.
    '''
    return _PRIMES[_PRIMES ** 2 > n]**2