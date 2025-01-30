import os
import pickle
import numpy as np
from waggon.optim import Optimiser, SurrogateOptimiser
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


def display(source, ax=None, base_dir='test_results', y_label=False, x_label=False, title=None, **kwargs): # TODO: add option to plot single results

    results = {}
    # TODO: move args to saved results
    epsilon = kwargs['eps'] if 'eps' in kwargs else 1e-1
    max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 100

    if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

    if type(source) == str: # if function directory is given; used when plotting all results
        res_path = f'{base_dir}/{source}'

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

    elif type(source) in [Optimiser, SurrogateOptimiser]:
        results[f'{source.surr.name}'] = {'res': (np.array(source.errors), np.zeros_like(source.errors)),
                                          'color': 'C0', 'label': f'{source.surr.name} ({source.acqf.name})'}
        
        title = source.func.name if title is None else title
        epsilon = source.eps
        max_iter = source.max_iter

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
    
    try:
        y_lim = min(epsilon, np.min(y_lims))
    except ValueError:
        y_lim = -1.0
        pass
    
    ax.set_ylim(bottom=9e-1 * (y_lim if y_lim > 0 else epsilon))
    ax.tick_params(labelsize=16)
    ax.set_yscale('log')

    if type(source) in [Optimiser, SurrogateOptimiser]:
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


# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)

# X,Y = np.meshgrid(x, y)
# data = np.vstack([X.ravel(), Y.ravel()])
# plt.contourf(X, Y, d(data.T).reshape(100, 100))