import copy
import numpy as np

import nems_lbhb.plots
import nems.db as nd
from nems.utils import ax_remove_box

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (mplparams, SIG_TEST_MODELS, MODELGROUPS, POP_MODELGROUPS,
                                                                  PLOT_STAT, DOT_COLORS, DOT_MARKERS,
                                                                  get_significant_cells)
import matplotlib as mpl
mpl.rcParams.update(mplparams)
import matplotlib.pyplot as plt


def model_comp_pareto(batch, modelgroups, ax, cellids, nparms_modelgroups=None, dot_colors=None, dot_markers=None,
                      fill_styles=None, plot_stat='r_test', plot_medians=False, labeled_models=None, show_legend=True,
                      y_lim=None):

    if labeled_models is None:
        labeled_models = []
    if nparms_modelgroups is None:
        nparms_modelgroups = copy.copy(modelgroups)
    if fill_styles is None:
        fill_styles = {k:'full' for (k,v) in dot_colors.items()}
    mean_cells_per_site = len(cellids)  # NAT4 dataset, so all cellids are used
    overall_min = 100
    overall_max = -100

    all_model_means = []
    labeled_data = []
    for k, modelnames in modelgroups.items():
        np_modelnames = nparms_modelgroups[k]
        b_ceiling = nd.batch_comp(batch, modelnames, cellids=cellids, stat=plot_stat)
        b_n = nd.batch_comp(batch, np_modelnames, cellids=cellids, stat='n_parms')
        if not plot_medians:
            model_mean = b_ceiling.mean()
        else:
            model_mean = b_ceiling.median()
        b_m = np.array(model_mean)
        print(f"{k} modelcount {len(modelnames)} fits per model: {b_n.count().values}")
        n_parms = np.array([np.mean(b_n[m]) for m in np_modelnames])

        # don't divide by cells per site if only one cell was fit
        if ('single' not in k) and (k != 'LN') and (k != 'stp'):
            n_parms = n_parms / mean_cells_per_site

        y_max = b_m.max() * 1.05
        y_min = b_m.min() * 0.9
        overall_max = max(overall_max, y_max)
        overall_min = min(overall_min, y_min)

        ax.plot(n_parms, b_m, color=dot_colors[k], marker=dot_markers[k], label=k, markersize=4.5,
                fillstyle=fill_styles[k])
        for m in labeled_models:
            if m in modelnames:
                i = modelnames.index(m)
                labeled_data.append([n_parms[i], b_m[i]])

        all_model_means.append(model_mean)

    ax.plot(*list(zip(*labeled_data)), 's', color='black', marker='o', fillstyle='none', markersize=10)

    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    if show_legend:
        ax.legend(handles, labels, loc='lower right', fontsize=7, frameon=False)
    ax.set_xlabel('Free parameters per neuron')
    ax.set_ylabel('Median prediction correlation')
    if y_lim is None:
        ax.set_ylim((overall_min, overall_max))
    else:
        ax.set_ylim(*y_lim)
    ax_remove_box(ax)
    plt.tight_layout()

    return ax, b_ceiling, all_model_means, labels


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
