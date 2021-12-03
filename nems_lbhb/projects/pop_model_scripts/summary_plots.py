import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nems.utils import ax_remove_box
import nems.db as nd
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
    SIG_TEST_MODELS, get_significant_cells, set_equal_axes, PLOT_STAT, DOT_COLORS, ALL_FAMILY_MODELS
)


# TODO: 1) color scatter plot by significant difference
#       2) color bar plot by modelname
#       3) fix diagonal line on scatter, not showing up
#       4) truncate lims at 1.0 for r_ceiling? (a few cells over)


def scatter_bar(batches, modelnames, axes=None):
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 6))
    else:
        ax1, ax2 = axes

    cellids = [get_significant_cells(batch, SIG_TEST_MODELS, as_list=True) for batch in batches]
    r_values = [nd.batch_comp(batch, modelnames, cellids=cells, stat=PLOT_STAT) for batch, cells in zip(batches, cellids)]
    all_r_values = pd.concat(r_values)

    # NOTE: if SIG_TEST_MODELS changes, the 1, 0 indices will need to be updated
    # Scatter Plot -- LN vs c1dx2+d
    ax1.scatter(all_r_values[modelnames[1]], all_r_values[modelnames[0]], s=2, c='black')
    ax1.plot([[0, 0], [1, 1]], c='black', linestyle='dashed', linewidth=1)
    ax_remove_box(ax1)
    #set_equal_axes(ax1)
    ax1.set_ylim(0,1)
    ax1.set_xlim(0,1)
    ax1.set_xlabel('LN_pop prediction accuracy')
    ax1.set_ylabel('conv1Dx2 prediction accuracy')

    # Bar Plot -- Median for each model
    # NOTE: ordering of names is assuming ALL_FAMILY_MODELS is being used and has not changed.
    short_names = ['conv2d', 'conv1d', 'conv1dx2+d', 'LN_pop', 'dnn1_single']
    bar_colors = [DOT_COLORS[k] for k in short_names]
    ax2.bar(np.arange(0, len(modelnames)), all_r_values.median(axis=0).values, color=bar_colors,
            tick_label=short_names)
    ax_remove_box(ax2)
    ax2.set_ylabel('Median Prediction Accuracy')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation='45', ha='right')

    fig = plt.gcf()
    fig.tight_layout()

    return ax1, ax2


if __name__ == '__main__':
    modelnames = ALL_FAMILY_MODELS
    batches = [322, 323]
    ax1, ax2 = scatter_bar(batches, modelnames)
