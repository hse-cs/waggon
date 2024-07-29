import os
import time
import pickle
import numpy as np
from optim import Optimiser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_dir(func, acqf_name, surr_name, base_dir='test_results'):
    
    res_path = base_dir
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    func_dir = func.name if func.dim == 2 else func.name + str(func.dim)

    res_path += f'/{func_dir}'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    res_path += f'/{acqf_name}'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    res_path += f'/{surr_name}'
    if not os.path.isdir(res_path):
        os.mkdir(res_path)
    
    return res_path


def save_results(optimiser, base_dir='test_results'):
    res_path = create_dir(optimiser.func, optimiser.acqf.name, optimiser.surrogate.name, base_dir=base_dir)

    with open(f'{res_path}/{time.strftime("%d.%m-%H:%M:%S")}.pkl', 'wb') as f:
        pickle.dump(optimiser.res, f)


def load_results_for_plotting(func_dir, acqf_name, surr_name, base_dir='test_results', epsilon=1e-1):

    res_path = f'{base_dir}/{func_dir}/{acqf_name}/{surr_name}'
    
    f = []
    for (dirpath, dirnames, filenames) in os.walk(res_path):
        f.extend(filenames)
        break
    
    results = []
    
    if len(f) == 0:
        return None, None
    
    for file in f:
        with open(f'{res_path}/{file}', "rb") as input_file:
            results.append(np.squeeze(pickle.load(input_file)))
    
    for i in range(len(results)):
        while len(results[i]) < len(max(results, key=len)):
            np.insert(results[i], -1, results[i][-1])
    
    return np.mean(results, axis=0), np.std(results, axis=0)


def display(source, ax=None, base_dir='test_results', y_label=False, x_label=False, title=None, **kwargs): # TODO: add option to plot single results

    if type(source) == 'str': # if function directory is given; used when plotting all results
        epsilon = kwargs['opt_eps'] if 'opt_eps' in kwargs else 1e-1
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 100

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        res_path = f'{base_dir}/{source}'

        results = {}

        acqfs = []
        for (_, dirnames, _) in os.walk(res_path):
            acqfs.extend(dirnames)
            break
        
        j = 0

        for acqf in acqfs:

            surrs = []
            for (_, dirnames, _) in os.walk(f'{res_path}/{acqf}/'):
                surrs.extend(dirnames)
                break

            for surr in surrs:

                results[surr] = {'res': load_results_for_plotting(source, acqf, surr, epsilon=epsilon),
                                        'color': f'C{j}', 'label': f'{surr} ({acqf})'}
                j += 1 # TODO: fix colouring; works fine when the ran models are the same across all experiments

    elif type(source) == Optimiser:
        results[f'{source.surr.name}'] = {'res': (source.res, np.ones(source.res.shape)),
                                          'color': 'C0', 'label': f'{source.surr.name} ({source.acqf.name})'}
        title = source.func.name if title is None else title

    y_lims = []
    
    ax.hlines(epsilon, 0, max_iter, color='grey', linestyles='--', linewidth=5)

    for key in results.keys():
        ax.plot(np.arange(results[key]['res'][0].shape[0]), results[key]['res'][0], '-', linewidth=3, label=key, color=results[key]['color'])
        ax.fill_between(np.arange(results[key]['res'][0].shape[0]),
                        results[key]['res'][0] - results[key]['res'][1],
                        results[key]['res'][0] + results[key]['res'][1],
                        color=results[key]['color'], alpha=0.2)
        y_lims.append(results[key]['res'][0] - results[key]['res'][1])
    
    ax.set_title(source if title is None else title, fontsize=20, weight='bold')
    
    if y_label: ax.set_ylabel('Distance to optimum', fontsize=16, weight='bold')
    if x_label: ax.set_xlabel('Number of simulator calls', fontsize=16, weight='bold')
    
    y_lim = min(epsilon, np.min(y_lims))
    ax.set_ylim(bottom=9e-1 * (y_lim if y_lim > 0 else epsilon))
    ax.tick_params(labelsize=16)
    ax.set_yscale('log')

    if type(source) == Optimiser:
        handles = [mpatches.Patch(color=results[key]['color'], label=results[key]['label']) for key in results.keys()]
        fig.legend(handles=handles, bbox_to_anchor=(0.5, -0.1), loc='lower center', prop={'weight':'bold', 'size': 24})
        plt.show()
    
    return results


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
