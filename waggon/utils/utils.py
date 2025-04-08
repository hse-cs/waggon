import os
import pickle
import contextlib
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results_for_plotting(func_dir, acqf_name, surr_name, base_dir='test_results', exp_transform=True):

    res_path = f'{base_dir}/{func_dir}/{acqf_name}/{surr_name}'
    
    f = []
    for (_, _, filenames) in os.walk(res_path):
        f.extend(filenames)
        break
    
    results = []
    
    if len(f) == 0:
        return None, None
    
    for file in f:
        with open(f'{res_path}/{file}', "rb") as input_file:
            results.append(np.squeeze(pickle.load(input_file)))
    
    res = np.zeros((len(results), len(max(results, key=len))))
    
    for i in range(len(results)):
        res[i, :results[i].shape[0]] = results[i]
        
        if results[i].shape[0] < res.shape[0]:
            res[i, results[i].shape[0]:] = results[i][-1] * np.ones(res.shape[1] - results[i].shape[0])
    
    if exp_transform:
        res = np.exp(res)
    
    return np.mean(res, axis=0), np.std(res, axis=0)


def plot_results(base_dir='test_results', figsize=(16, 8), **kwargs):

    funcs = []
    for (_, dirnames, _) in os.walk(base_dir):
        funcs.extend(dirnames)
        break
    
    nrows = int(np.round(np.sqrt(len(funcs))))
    nplots = len(funcs) if len(funcs) == nrows ** 2 else len(funcs) + 1
    ncols = int(np.round((nplots)/nrows))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    
    for i, func_dir in enumerate(funcs):

        r, c = i // ncols, i % ncols
        results = display(func_dir=func_dir, ax=ax if len(funcs) == 1 else (ax[c] if nrows == 1 else ax[r, c]), base_dir='test_results',
                              y_label=True if c == 0 else False,
                              x_label=True if ((r == nrows - 1) or ((r == nrows - 1) and (c == ncols - 1))) else False, kwargs=kwargs)

    handles = [mpatches.Patch(color=results[key]['color'], label=results[key]['label']) for key in results.keys()]

    if len(funcs) == nrows ** 2:
        fig.legend(handles=handles, bbox_to_anchor=(0.5, -0.1), loc='lower center', prop={'weight':'bold', 'size': 24})
    else:
        legend_ax = ax[-1] if nrows == 1 else ax[-1, -1]
        
        legend_ax.clear()
        legend_ax.set_axis_off()

        legend_ax.legend(handles=handles, loc='center', prop={'weight':'bold', 'size': 24})

    plt.tight_layout()
    if 'show' in kwargs:
        if kwargs['show']:
            plt.show()
    
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    fig.savefig('plots/res.jpeg', format='jpeg', dpi=200, bbox_inches='tight')


@contextlib.contextmanager
def fixed_random_seed(seed: int):
    """
    Context manager to set specified seed for random package

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


@contextlib.contextmanager
def fixed_numpy_seed(seed: int):
    """
    Context manager to set specified seed for NumPy

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def fixed_torch_seed(seed: int):
    """
    Context manager to set specified for Torch
    """
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(state)
        torch.cuda.set_rng_state_all(cuda_state)


def set_all_seed(seed: int):
    """
    Set specified seed for random, numpy, torch packages

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def fixed_all_seed(seed: int):
    """
    Context manager to set specified seed for random, numpy and torch packages

    Parameters
    ----------
    seed: int
        The seed value to set
    """
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all()

    set_all_seed(seed)

    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(torch_cuda_state)
