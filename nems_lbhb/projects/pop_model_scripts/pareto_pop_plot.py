import copy

import matplotlib.pyplot as plt
import numpy as np

#import importlib; importlib.reload(nems_lbhb.plots)  # if need to incorporate updated code w/o restarting entire kernel
import nems_lbhb.plots
import nems.db as nd
from nems.utils import ax_remove_box

from pop_model_utils import (SIG_TEST_MODELS, MODELGROUPS, POP_MODELGROUPS, PLOT_STAT, DOT_COLORS, DOT_MARKERS,
                             get_significant_cells)


def model_comp_pareto(batch, modelgroups, ax, cellids, nparms_modelgroups=None, dot_colors=None, dot_markers=None,
                      plot_stat='r_test', plot_medians=False, labeled_models=None, show_legend=True):

    if labeled_models is None:
        labeled_models = []
    if nparms_modelgroups is None:
        nparms_modelgroups = copy.copy(modelgroups)

    mean_cells_per_site = len(cellids)  # NAT4 dataset, so all cellids are used
    overall_min = 100
    overall_max = -100

    for k, modelnames in modelgroups.items():
        print(f"{k} len {len(modelnames)}")
        np_modelnames = nparms_modelgroups[k]
        b_ceiling = nd.batch_comp(batch, modelnames, cellids=cellids, stat=plot_stat)
        b_n = nd.batch_comp(batch, np_modelnames, cellids=cellids, stat='n_parms')
        if not plot_medians:
            model_mean = b_ceiling.mean()
        else:
            model_mean = b_ceiling.median()
        b_m = np.array(model_mean)

        n_parms = np.array([np.mean(b_n[m]) for m in np_modelnames])

        # don't divide by cells per site if only one cell was fit
        if ('_single' not in k) and (k != 'LN') and (k != 'stp'):
            n_parms = n_parms / mean_cells_per_site

        y_max = b_m.max() * 1.05
        y_min = b_m.min() * 0.9
        overall_max = max(overall_max, y_max)
        overall_min = min(overall_min, y_min)Ok

        ax.plot(n_parms, b_m, color=dot_colors[k], marker=dot_markers[k], label=k.split('_single')[0], markersize=6)
        for m in labeled_models:
            if m in modelnames:
                i = modelnames.index(m)
                ax.plot(n_parms[i], b_m[i], color='black', marker='o', fillstyle='none', markersize=10)

    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    if show_legend:
        ax.legend(handles, labels, loc='lower right', fontsize=8, frameon=False)
    ax.set_xlabel('Free parameters')
    ax.set_ylabel('Mean pred corr')
    ax.set_ylim((overall_min, overall_max))
    ax_remove_box(ax)

    return ax, b_ceiling, model_mean


if __name__ == '__main__':
    batches=[322, 323]
    # a1_sig_cells = get_significant_cells(322, sig_test_models)
    # peg_sig_cells = get_significant_cells(323, sig_test_models)
    # sig_cells = a1_sig_cells + peg_sig_cells

    means = []
    f, ax = plt.subplots(1,2, figsize=(10,4))
    xlims = []
    ylims = []
    for i, batch in enumerate(batches):
        sig_cells = get_significant_cells(batch, SIG_TEST_MODELS)
        _, b_ceiling, model_mean = model_comp_pareto(batch, MODELGROUPS, ax[i], sig_cells, nparms_modelgroups=POP_MODELGROUPS,
                                                     dot_colors=DOT_COLORS, dot_markers=DOT_MARKERS,
                                                     plot_stat=PLOT_STAT, plot_medians=True,
                                                     labeled_models=SIG_TEST_MODELS)
        means.append(model_mean)
        ax[i].set_title(f'pareto batch={batch}')
        xlims.extend(ax[i].get_xlim())
        ylims.extend(ax[i].get_ylim())

    xlims = np.array(xlims)
    ylims = np.array(ylims)
    min_x = xlims.min()
    max_x = xlims.max()
    min_y = ylims.min()
    max_y = ylims.max()
    for a in ax:
        a.set_xlim(min_x, max_x)
        a.set_ylim(min_y, max_y)
