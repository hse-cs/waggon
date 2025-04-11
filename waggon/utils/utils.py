import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# from waggon.functions import FunctionV2 #creates circular import
from waggon.optim import Optimiser
from waggon.optim import SurrogateOptimiser


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


# def plot_func_2d(func, fig, ax, x_range, y_range, name=None, nx=150, ny=150, norm=None):
#     """
#     Plot colormap for 2-dimensional function
#     """
#     assert func.dim == 2

#     x_min, x_max = x_range
#     y_min, y_max = y_range

#     x_arr = np.linspace(x_min, x_max, nx)
#     y_arr = np.linspace(y_min, y_max, ny)

#     xs, ys = np.meshgrid(x_arr, y_arr)

#     x_inp = np.zeros((ny * nx, 2))
#     x_inp[:, 0] = xs.flatten()
#     x_inp[:, 1] = ys.flatten()

#     z_out = func(x_inp)
#     z_out = z_out.squeeze(1) if z_out.ndim == 2 else z_out
#     z_out = z_out.reshape((nx, ny), order='F')
#     z_out = np.transpose(z_out)

#     mesh = ax.pcolormesh(x_arr, y_arr, z_out, norm=norm)
#     cbar = fig.colorbar(mesh, ax=ax)

#     glob_min = func.glob_min.reshape(-1, 1)
#     ax.scatter(func.glob_min[:, 0], func.glob_min[:, 1], color='red', s=8)

#     if name is None:
#         title_name = f"{func.name} (new API)" if isinstance(func, FunctionV2) else f"{func.name} (old API)"
#     else:
#         title_name = name

#     ax.set_title(title_name)    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
