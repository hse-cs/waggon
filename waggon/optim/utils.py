import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .base import Optimiser
from .surrogate import SurrogateOptimiser

from typing import Optional


def display(
    source: Optimiser | SurrogateOptimiser | str, 
    ax: Optional[matplotlib.axes.Axes] = None, 
    base_dir: str='test_results', 
    y_label: bool = False, 
    x_label: bool = False, 
    title: Optional[str] = None, 
    **kwargs
) -> dict:
    """
    Display optimization process results

    Parameters
    ----------
    source: Optimiser | SurrogateOptimiser | str 
        Source of optimization results (optimizer or directory path)
    ax: matplotlib.axes.Axes, optional
        Axes optimization results will drawn on
    base_dir: str, default 'test_results'
        Directory contains all results
    y_label: bool, default False
        If True label for y-axis will be displayed
    x_label: bool, default False
        If True label for x-axis will be displayed
    title: str, optional
        Axis title
    **kwargs
        Optimization process parameters
    
    Returns
    -------
    dict
        Optimization results
    """
    
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
