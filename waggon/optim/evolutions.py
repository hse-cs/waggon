import time
import pickle

import numpy as np

from tqdm import tqdm

from .base import Optimiser
from .utils import create_dir


class DifferentialEvolutionOptimizer(Optimiser):
    """
    Differential evolution black-box function optimizer

    Parameters
    ----------
    """

    def __init__(self, func, **kwargs):
        super(DifferentialEvolutionOptimizer).__init__()

        self.func         = func
        self.max_iter     = kwargs.get('max_iter', 1000)
        self.eps          = kwargs.get('eps', 1e-1)
        self.error_type   = kwargs.get('error_type', 'x')
        self.n_candidates = kwargs.get('n_candidates', 11 ** 2)
        self.olhs         = kwargs.get('olhs', True)
        self.lhs_seed     = kwargs.get('lhs_seed', None)
        self.verbose      = kwargs.get('verbose', 1)
        self.save_results = kwargs.get('save_results', True)
        self.candidates   = self.create_candidates(self.n_candidates)

        # Evolution parameters
        self.mutation_rate  = kwargs.get('mutation_rate', 1.0)
        self.crossover_rate = kwargs.get('crossover_rate', 0.5)

        # Experiments parameters
        self.errors = []
        self.res = None
        self.res = None

    def predict(self, *args, **kwargs) -> np.ndarray:
        y_func = self.func(self.candidates).flatten() # shape of [n_candidates]
        return self.candidates[np.argmin(y_func)]
    
    def evolution_step(self):
        X = self.candidates
        rng = np.random.default_rng()

        # Mutation
        Z = np.apply_along_axis(
            func1d = lambda x: rng.choice(X, size=3, replace=False),
            axis = -1,
            arr = X
        )

        R = self.mutation_rate
        coeffs = np.expand_dims([1.0, R, -R], axis=[-1, 0]) # shape of [1, 3, 1]
        Z = np.sum(Z * coeffs, axis=-2)

        # Crossover
        p = self.crossover_rate
        mask = rng.uniform(0, 1, Z.shape)
        mask = (mask < p).astype(bool)
        Z[mask] = X[mask]

        # Selection
        y_prev = self.func(X) # shape of [n_candidates, 1]
        y_next = self.func(Z) # shape of [n_candidates, 1]

        if y_prev.ndim == 1:
            y_prev = np.expand_dims(y_prev, -1)
        
        if y_next.ndim == 1:
            y_next = np.expand_dims(y_next, -1)
        
        self.candidates = np.where(
            (y_prev <= y_next), X, Z
        )

    def optimise(self, X=None, y=None, **kwargs):
        if X is None:
            self.candidates = self.create_candidates(self.n_candidates)
        else:
            self.candidates = X
        
        if self.verbose == 0:
            opt_loop = range(self.max_iter)
        else:
            opt_loop = tqdm(
                range(self.max_iter), 
                desc='Optimization loop started...', 
                leave=True, 
                position=0
            )
        
        self.errors = []
        

        for _ in opt_loop:
            # TO DO: add res and params logging
            self.evolution_step()

            x_cand = self.candidates
            y_pred = self.func(x_cand).flatten() # works only for 1-d output
            
            x_glob_min = self.func.glob_min
            y_glob_min = self.func(self.func.glob_min)

            if self.error_type == 'x':
                error = np.min(
                    np.linalg.norm(x_glob_min - x_cand, axis=-1),
                    axis = -1
                )
            elif self.error_type == "f":
                error = np.min(
                    np.linalg.norm(y_glob_min - y_pred, axis=-1),
                    axis = -1
                )
            else:
                raise ValueError(f"Unsupported error type: {self.error_type}")
            
            self.errors.append(error)

            if error <= self.eps:
                print('Experiment finished successfully')
                break
        
        if error > self.eps:
            print('Experiment failed')
        
        if self.save_results:
            self._save()
