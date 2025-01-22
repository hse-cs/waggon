import os
import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc

from .utils import _get_olhs_num
from ..functions import Function


class Optimiser(object):
    def __init__(self, **kwargs):
        super(Optimiser, self).__init__() # TODO: fix desc
        '''
        Black-box optimiser.

        Parameter
        ----------
        func : Callable, waggon.functions.Function
            Black-box function to be optimised.
        
        max_iter : int, default = 100
            Maximum number of optimisation loop iterations.
        
        eps : float, default = 1e-1
            Epsilon-solution criterion value.
        
        error_type : {'x', 'f'}, default = 'x'
            Optimisation error type - either in argument, 'x', or function, 'f'.
        
        eq_cons : dict, default = None
            Equality-type constraints for numerical optimisation.
        
        ineq_cons : dict, default = None
            Ineqaulity-type constraints for numerical optimisation.

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
        
        save_results : bool, default = True
            Whether results are saved or not.
        '''
        self.func           = kwargs.get('func', Function())
        self.max_iter       = kwargs.get('max_iter', 100)
        self.eps            = kwargs.get('eps', 1e-1)
        self.error_type     = kwargs.get('error_type', 'x')
        self.fix_candidates = kwargs.get('fix_candidates', True)
        self.n_candidates   = kwargs.get('n_candidates', 1)
        self.olhs           = kwargs.get('olhs', True)
        self.lhs_seed       = kwargs.get('lhs_seed', None)
        self.verbose        = kwargs.get('verbose', 1)
        self.save_results   = kwargs.get('save_results', True)
        self.surr.verbose   = self.verbose
        self.candidates     = None
    
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
        N = _get_olhs_num(self.func.dim)[0] if N == -1 else N

        if self.olhs:
            N = max(N, _get_olhs_num(self.func.dim)[0])
            strength = 2
        else:
            strength = 1

        lhs_       = qmc.LatinHypercube(d=self.func.domain.shape[0], scramble=True, strength=strength, seed=self.lhs_seed)
        candidates = lhs_.random(N)
        candidates = qmc.scale(candidates, self.func.domain[:, 0], self.func.domain[:, 1])
        return candidates
    
    def predict(self):
        pass
    
    def optimise(self, X=None, y=None, **kwargs):
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

        self.errors = []
        
        if X is None:
            X = self.create_candidates(N=-1)
            X, y = self.func.sample(X)
            self.res = np.array([[np.min(self.func(X))]])
            self.params = np.array([X[np.argmin(self.func(X)), :]])
        else:
            self.res = np.array([np.min(y)])
            self.params = np.array([X[np.argmin(y), :]])

        if self.verbose == 0:
            opt_loop = range(self.max_iter)
        else:
            opt_loop = tqdm(range(self.max_iter), desc='Optimisation loop started...', leave=True, position=0)

        for _ in opt_loop:

            next_x = self.predict(X, y)
            next_f = np.array([self.func(next_x)])

            if next_f <= self.res[-1, :]:
                self.res = np.concatenate((self.res, next_f.reshape(1, -1)))
                self.params = np.concatenate((self.params, next_x.reshape(1, -1)))
            else:
                self.res = np.concatenate((self.res, self.res[-1, :].reshape(1, -1)))
                self.params = np.concatenate((self.params, self.params[-1, :].reshape(1, -1)))

            X_, y_ = self.func.sample(next_x)
            
            X = np.concatenate((X, X_))
            y = np.concatenate((y, y_))

            if self.error_type == 'x':
                error = np.min(np.linalg.norm(self.func.glob_min - X, ord=2, axis=-1), axis=-1)
            elif self.error_type == 'f':
                error = np.min(np.linalg.norm(self.func(self.func.glob_min) - y, ord=2, axis=-1), axis=-1)
            
            self.errors.append(error)
            
            if self.verbose > 0:
                opt_loop.set_description(f"Optimisation error: {error:.4f}")
            
            if error <= self.eps:
                print('Experiment finished successfully')
                break
        
        if error > self.eps:
            print('Experiment failed')
        
        if self.save_results:
            self._save()
    
    def _save(self, base_dir='test_results'):

        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)

        with open(f'{base_dir}/{time.strftime("%d.%m-%H:%M:%S")}.pkl', 'wb') as f:
            pickle.dump(self.res, f)
