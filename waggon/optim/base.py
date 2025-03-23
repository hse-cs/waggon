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
        super(Optimiser, self).__init__()
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
        self.func           = kwargs['func'] if 'func' in kwargs else Function()
        self.max_iter       = kwargs['max_iter'] if 'max_iter' in kwargs else 100
        self.eps            = kwargs['eps'] if 'eps' in kwargs else 1e-4
        self.error_type     = kwargs['error_type'] if 'error_type' in kwargs else 'f'
        self.fix_candidates = kwargs['fix_candidates'] if 'fix_candidates' in kwargs else True
        self.n_candidates   = kwargs['n_candidates'] if 'n_candidates' in kwargs else 1
        self.olhs           = kwargs['olhs'] if 'olhs' in kwargs else True
        self.lhs_seed       = kwargs['lhs_seed'] if 'lhs_seed' in kwargs else None
        self.verbose        = kwargs['verbose'] if 'verbose' in kwargs else 1
        self.save_results   = kwargs['save_results'] if 'save_results' in kwargs else True
        self.plot_results   = kwargs['plot_results'] if 'plot_results' in kwargs else False
        self.candidates     = None

        if self.func.log_transform:
            transform = lambda x: np.exp(x)
        else:
            transform = lambda x: x

        if self.error_type == 'f':
            if self.func.f_min is None:
                self.error = lambda x: np.min(np.linalg.norm(transform(self.func(self.func.glob_min)) - transform(x), ord=2, axis=-1), axis=-1)
            else:
                self.error = lambda x: np.min(np.linalg.norm(self.func.f_min - transform(x), ord=2, axis=-1), axis=-1)
        else:
            self.error = lambda x: np.min(np.linalg.norm(self.func.glob_min - x, ord=2, axis=-1), axis=-1)
    
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
        N = _get_olhs_num((self.func.dim - 1)**2)[0] if N == -1 else N

        if self.olhs:
            N = max(N, _get_olhs_num((self.func.dim - 1)**2)[0])
            strength = 2
        else:
            strength = 1

        lhs_       = qmc.LatinHypercube(d=self.func.domain.shape[0], scramble=True, strength=strength, seed=self.lhs_seed)
        candidates = lhs_.random(N)
        candidates = qmc.scale(candidates, self.func.domain[:, 0], self.func.domain[:, 1])
        return candidates
    
    def predict(self):
        pass
    
    def optimise(self, X=None, y=None, N=None):
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
            X = self.create_candidates(N=-1 if N is None else N)
            X, y = self.func.sample(X)
            self.res = np.array([[np.min(self.func(X))]])
            self.params = np.array([X[np.argmin(self.func(X)), :]])
        else:
            self.res = np.array([np.min(y)])
            self.params = np.array([X[np.argmin(y), :]])

        if self.verbose == 0:
            opt_loop = range(self.max_iter)
        else:
            opt_loop = tqdm(range(self.max_iter), desc="Optimisation started...", leave=True, position=0)

        for _ in opt_loop:

            next_x = self.predict(X, y)
            next_f = np.array([self.func(next_x)])

            if self.plot_results:
                self.plot_iteration_results(np.unique(X, axis=0), next_x[0])

            if next_f <= self.res[-1, :]:
                self.res = np.concatenate((self.res, next_f.reshape(1, -1)))
                self.params = np.concatenate((self.params, next_x.reshape(1, -1)))
            else:
                self.res = np.concatenate((self.res, self.res[-1, :].reshape(1, -1)))
                self.params = np.concatenate((self.params, self.params[-1, :].reshape(1, -1)))

            X_, y_ = self.func.sample(next_x)
            X = np.concatenate((X, X_))
            y = np.concatenate((y, y_))

            error = self.error(y) if self.error_type == 'f' else self.error(X)
            self.errors.append(error)
            
            if self.verbose > 0:
                opt_loop.set_description(f"Optimisation error: {error:.4f}")
            
            if error <= self.eps:
                print('Experiment finished successfully!')
                break
        
        if error > self.eps:
            print('Experiment failed')
        
        if self.save_results:
            self._save()
    
    def _save(self, base_dir='test_results'):

        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)

        with open(f'{base_dir}/{time.strftime("%d_%m_%H_%M_%S")}.pkl', 'wb') as f:
            pickle.dump(self.res, f)
    
    def plot_iteration_results(self):
        pass
