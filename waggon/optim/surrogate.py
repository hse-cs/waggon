import gc
import time
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from botorch.optim import optimize_acqf
# from joblib import Parallel, delayed

from .base import Optimiser
from .utils import create_dir


class SurrogateOptimiser(Optimiser):
    '''
    Surrogate based black-box optimiser.

    Parameter
    ----------
    func : Callable, waggon.functions.Function
        Black-box function to be optimised.
    
    surr : waggon.surrogates.Surrogate or waggon.surrogate.GenSurrogate
        Surrogate model for the black-box function.
    
    acqf : waggon.acquisitions.Acquisition
        Acquisition function defining the optimisation strategy.
    
    max_iter : int, default = 100
        Maximum number of optimisation loop iterations.
    
    eps : float, default = 1e-1
        Epsilon-solution criterion value.
    
    error_type : {'x', 'f'}, default = 'x'
        Optimisation error type - either in argument, 'x', or function, 'f'.
    
    num_opt : bool, default = False
        Whether the acquisition function is optimised numerically or not. If False,
        the search of the next best parameters is done via direct search.
    
    eq_cons : dict, default = None
        Equality-type constraints for numerical optimisation.
    
    ineq_cons : dict, default = None
        Ineqaulity-type constraints for numerical optimisation.

    fix_candidates : bool, default = False if num_opt else True
        Whether the candidate points should be fixed or not.
    
    n_candidates_ : int, default = 1 if num_opt else 101**2
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
    def __init__(self, func, surr, acqf, **kwargs):
        kwargs['func'] = func
        super(SurrogateOptimiser, self).__init__(**kwargs)
        
        self.func               = func
        self.surr               = surr
        self.acqf               = acqf
        
        self.n_candidates       = kwargs['n_candidates'] if 'n_candidates' in kwargs else 10201
        self.num_opt_start      = kwargs['num_opt_start'] if 'num_opt_start' in kwargs else 'grid'
        self.num_opt_disp       = kwargs['num_opt_disp'] if 'num_opt_disp' in kwargs else False
        self.num_opt_tol        = kwargs['num_opt_tol'] if 'num_opt_tol' in kwargs else 1e-8
        self.num_opt_candidates = kwargs['num_opt_candidates'] if 'num_opt_candidates' in kwargs else 128

        self.eq_cons            = kwargs['eq_cons'] if 'eq_cons' in kwargs else None
        self.ineq_cons          = kwargs['ineq_cons'] if 'ineq_cons' in kwargs else None

        self.jitter             = kwargs['jitter'] if 'jitter' in kwargs else 1e0
        self.parallel           = kwargs['parallel'] if 'parallel' in kwargs else 0
        
        if type(self.surr) == list:
            for s in self.surr:
                s.verbose = self.verbose
        else:
            self.surr.verbose   = self.verbose

    def predict(self, X, y):
        
        self.surr.fit(X, y)
        
        self.acqf.y = y.reshape(y.shape[0]//self.func.n_obs, self.func.n_obs)
        self.acqf.conds = X[::self.func.n_obs]
        self.acqf.model = self.surr
        
        next_x, _ = optimize_acqf(self.acqf, bounds=torch.Tensor(self.func.domain),
                                          q=1, num_restarts=200, raw_samples=1000)
        
        if hasattr(self, "clear_surr") and self.clear_surr:
            del self.acqf.surr
            gc.collect()
        
        return torch.Tensor([next_x])
    
    
    def _save(self, base_dir='test_results'):
        res_path = create_dir(self.func, self.acqf.name, self.surr.name, base_dir=base_dir)

        res = {'X': self.params,
               'y': self.res,
               'err': self.errors}

        with open(f'{res_path}/{time.strftime("%d_%m_%H_%M_%S")}.pkl', 'wb') as f:
            pickle.dump(res, f)
    

    def plot_iteration_results(self, X, next_x):
        '''
        For surrogate optimiser only.
        '''
        if self.func.dim == 1:
            inter_conds = np.linspace(self.func.domain[:, 0], self.func.domain[:, 1], 121)
        else:
            inter_conds = self.create_candidates(N=10201)
        
        # transform = lambda x: np.exp(x) if self.func.log_transform else lambda x: x

        mu, _ = self.surr.predict(inter_conds)
        mu = mu
        y_true = self.func(inter_conds)
        
        if self.func.dim == 1:
            plt.plot(inter_conds, mu, label=f'Pred: {np.mean((y_true - mu)**2):.2f}', c='cornflowerblue')
            plt.plot(inter_conds, y_true, label='True', c='orange')
            # plt.scatter(X, y, c='black')
            plt.legend()
        else:
            
            y_true, mu = y_true.reshape(101, 101), mu.reshape(101, 101)
            mse = (y_true - mu)**2

            ei = self.acqf(inter_conds).reshape(101, 101)
            x_pred = inter_conds[np.argmin(ei)]
            
            def single_2d_plot(axis, f, title):
                plt.subplot(axis)
                colormap = plt.contourf(f, locator=ticker.LinearLocator(), extent=self.func.domain.flatten(), vmin=np.min(f), vmax=np.max(f))
                plt.scatter(X[:, 0], X[:, 1], color='black')
                for gb in self.func.glob_min:
                    plt.scatter(gb[0], gb[1], color='red', marker='*')
                plt.scatter(x_pred[0], x_pred[1], color='cyan', marker='*')
                plt.scatter(next_x[0], next_x[1], color='magenta', marker='*')
                plt.title(title)
                plt.colorbar(colormap)
            
            plt.figure(figsize=(24, 6))

            single_2d_plot(141, y_true, 'True Function')
            single_2d_plot(142, mu, 'Estimated Function')
            single_2d_plot(143, mse, 'MSE')
            single_2d_plot(144, ei, 'EI')
            
            plt.tight_layout()
        
        plt.show()

