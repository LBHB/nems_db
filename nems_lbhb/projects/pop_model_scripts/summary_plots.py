import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nems.utils import ax_remove_box
import nems.db as nd
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
    SIG_TEST_MODELS, get_significant_cells, set_equal_axes, PLOT_STAT
)


# TODO: 1) color scatter plot by significant difference
#       2) color bar plot by modelname
#       3) fix diagonal line on scatter, not showing up
#       4) truncate lims at 1.0 for r_ceiling? (a few cells over)


def scatter_bar(batches, modelnames, axes=None):
    if axes is None:
        _, (ax1, ax2) = plt.subplots(2, 1)
    else:
        ax1, ax2 = axes

    cellids = [get_significant_cells(batch, SIG_TEST_MODELS, as_list=True) for batch in batches]
    r_values = [nd.batch_comp(batch, modelnames, cellids=cells, stat=PLOT_STAT) for batch, cells in zip(batches, cellids)]
    all_r_values = pd.concat(r_values)

    # NOTE: if SIG_TEST_MODELS changes, the 1, 0 indices will need to be updated
    # Scatter Plot -- LN vs c1dx2+d
    ax1.scatter(all_r_values[modelnames[1]], all_r_values[modelnames[0]], s=2, c='black')
    ax1.plot([0, 0], [1, 1], c='black', linestyle='dashed', linewidth=1)
    ax_remove_box(ax1)
    set_equal_axes(ax1)

    # Bar Plot -- Median for each model
    ax2.bar(np.arange(0, len(modelnames)), all_r_values.median(axis=0).values)
    ax_remove_box(ax2)

    fig = plt.gcf()
    fig.tight_layout()

    return ax1, ax2


if __name__ == '__main__':
    modelnames = SIG_TEST_MODELS
    batches = [322, 323]
    ax1, ax2 = scatter_bar(batches, modelnames)
