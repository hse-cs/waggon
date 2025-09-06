import os
import pickle
import numpy as np
import conv_nn as conv

from waggon.optim import SurrogateOptimiser
from waggon.surrogates import GP
from waggon.acquisitions import EI


def init_samples(filename: str, optimizer: SurrogateOptimiser):
    # samples already exist
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            samples = pickle.load(f)
        return samples[:, :-1], samples[:, -1].reshape(-1, 1)
    
    # make samples if they aren't exist
    X = optimizer.create_candidates()
    X, y = optimizer.func.sample(X)
    samples = np.concatenate((X, y), axis=1)
    
    # save samples
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)

    return X, y


opt = SurrogateOptimiser(
    func=conv.ConvNN(verbose=1),
    surr=GP(),
    acqf=EI(),
    error_type='f',
    num_opt_candidates=1,
    n_candidates=25,
    seed=2,
    max_iter=3,
    verbose=2
)

X, y = init_samples('benchmarks/conv_nn/init.pkl', opt)
result = opt.optimise(X, y)

print(opt.res)
print(opt.params)
