import numpy as np
import jax.numpy as jnp
from protes import protes

from ..base import Optimiser


class ProtesOptimiser(Optimiser):
    def __init__(self, func, K=10, constraint_func=None, penalty=1e6, **kwargs):
        """
        Initialize the PROTES Optimiser.

        Parameters:
        - func: The function to optimize.
        - K: Number of samples per iteration.
        - constraint_func: Function to evaluate constraints on candidates.
        - penalty: Penalty value for constraint violations.
        """
        super().__init__(**kwargs)
        self.func = func
        self.K = K
        self.constraint_func = constraint_func
        self.penalty = penalty
        self.verbose = kwargs.get("verbose", True)

        self.d = self.func.T
        self.n = 2  
        self.protes_model = self.initialize_protes()

    def initialize_protes(self):
        if self.verbose:
            print(f"Initializing PROTES with d={self.d}, n={self.n}, K={self.K}")
        
        return lambda f: protes(f=f, d=self.d, n=self.n, m=self.K, log=self.verbose)

    def f_batch(self, batch):
        func_values = jnp.array([self.func(x) for x in batch])

        if self.constraint_func is not None:
            penalties = jnp.array([self.constraint_func(x) for x in batch])
            func_values += penalties * self.penalty

        return func_values

    def predict(self):

        f_batch = lambda batch: self.f_batch(jnp.array(batch))
        try:
            best_candidate, best_value = self.protes_model(f=f_batch)
        except Exception as e:
            print(f"Error during PROTES optimization: {e}")
            raise

        if self.verbose:
            print(f"Best candidate: {best_candidate}, Best value: {best_value}")

        return best_candidate, best_value
