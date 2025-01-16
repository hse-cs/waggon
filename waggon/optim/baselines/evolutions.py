import time
import pickle

import numpy as np

from tqdm import tqdm

from ..base import Optimiser
from ..utils import create_dir


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
        self.error_type   = kwargs.get('error_type', 'f')
        self.n_candidates = kwargs.get('n_candidates', 11 ** 2)
        self.olhs         = kwargs.get('olhs', True)
        self.lhs_seed     = kwargs.get('lhs_seed', None)
        self.verbose      = kwargs.get('verbose', 1)
        self.save_results = kwargs.get('save_results', True)
        self.candidates   = self.create_candidates()

        # Evolution parameters
        self.mutation_rate  = kwargs.get('mutation_rate', 1.0)
        self.crossover_rate = kwargs.get('crossover_rate', 0.5)

        # Experiments parameters
        self.errors = None
        self.res = None
        self.params = None

    def predict(self, X=None, y=None) -> np.ndarray:
        """
        Make one step of evolution and returns next best parameters.

        Parameters
        ----------
        X: Any, default = None
            Ignored
        y: Any, default = None
            Ignored
        
        Returns
        -------
        x_next: np.ndarray shape of [dim, ]
        """
        self.evolution_step()
        y_func = self.func(self.candidates).flatten() # shape of [n_candidates]
        return self.candidates[np.argmin(y_func)]
    
    def evolution_step(self):
        """
        Make evolution step: mutation, crossover, selecting.
        """
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
            X = self.create_candidates()
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

        for _ in opt_loop:
            self.predict()
            x_min, y_min, error = self.evaluate()

            # Log metrics            
            self.log_metrics(
                x_min = x_min,
                y_min = y_min,
                error = error
            )

            if error <= self.eps:
                print(f'Experiment finished successfully')
                break
        
        if error > self.eps:
            print('Experiment failed')
        
        if self.save_results:
            self._save()

    def evaluate(self) -> tuple[np.ndarray, np.ndarray, float]:
        x_cand = self.candidates
        y_pred = self.func(x_cand).flatten() # works only for 1-d output

        dim = self.func.dim
        x_glob_min = self.func.glob_min
        y_glob_min = self.func(self.func.glob_min.reshape(1, dim)).flatten()

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
    
        x_min = x_cand[np.argmin(y_pred)]
        y_min = np.min(y_pred)

        return x_min, y_min, error
    
    def log_metrics(self, **kwargs) -> None:
        """
        Save optimization results
        """
        error = kwargs.pop('error')

        if self.errors is None:
            self.errors = []
        self.errors.append(error)

        x_min = kwargs.pop('x_min')
        y_min = kwargs.pop('y_min')

        if self.params is None:
            dim = self.func.dim
            self.params = np.zeros((0, dim))
            self.res = np.zeros((0, 1))
            
            x_last = None
            y_last = np.inf
        else:
            x_last = self.params[-1]
            y_last = self.res[-1, 0]


        if y_last <= y_min:
            x_best, y_best = x_last, y_last
        else:
            x_best, y_best = x_min, y_min
        
        self.params = np.concatenate((
            self.params, np.expand_dims(x_best, 0)
        ))
        self.res = np.concatenate((
            self.res, np.expand_dims([y_best], 0)
        ))
