import numpy as np
import scipy.stats as st
import seaborn as sns
import pandas as pd

import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 10,
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)
import matplotlib.pyplot as plt

import nems
import nems.db as nd
import nems.xform_helper as xhelp
import nems_lbhb.xform_wrappers as xwrap
import nems.epoch as ep
from nems.xforms import evaluate_step
import nems_lbhb.baphy_io as io

from pop_model_utils import (get_significant_cells, SIG_TEST_MODELS, MODELGROUPS, HELDOUT, MATCHED, PLOT_STAT, DOT_COLORS,
                             a1, peg, single_column_short, single_column_tall, column_and_half_short, column_and_half_tall)


def get_heldout_results(batch, significant_cells, short_names):
    modelgroups = {}
    for i, name in enumerate(short_names):
        modelgroups[name + '_hld'] = HELDOUT[i]
        modelgroups[name + '_mat'] = MATCHED[i]

    r_ceilings = {}
    for n in short_names:
        # sum is just to collapse cols
        heldout_r_ceiling = nd.batch_comp(batch, [modelgroups[n + '_hld']], cellids=significant_cells,
                                          stat=PLOT_STAT)[modelgroups[n + '_hld']]
        matched_r_ceiling = nd.batch_comp(batch, [modelgroups[n + '_mat']], cellids=significant_cells,
                                          stat=PLOT_STAT)[modelgroups[n + '_mat']]

        r_ceilings[n + '_hld'] = heldout_r_ceiling
        r_ceilings[n + '_mat'] = matched_r_ceiling

    return r_ceilings


def generate_heldout_plots(batch, batch_name, sig_test_models=SIG_TEST_MODELS, ax=None, hide_xaxis=False):

    significant_cells = get_significant_cells(batch, sig_test_models, as_list=True)
    #short_names = ['conv2d', 'conv1d', 'conv1dx2+d', 'LN_pop', 'dnn1']
    short_names = ['conv1dx2+d', 'LN_pop', 'dnn1']
    if len(short_names) != len(HELDOUT):
        raise ValueError('length of short_names must equal number of models in HELDOUT / MATCHED')
    r_ceilings = get_heldout_results(batch, significant_cells, short_names)

    r_ceilings['cellid'] = significant_cells
    reference_results = nd.batch_comp(batch, sig_test_models, cellids=significant_cells, stat=PLOT_STAT)
    reference_medians = [reference_results[m].median() for m in sig_test_models]

    heldout_names = [n + '_hld' for n in short_names]
    matched_names = [n + '_mat' for n in short_names]
    tests = [st.mannwhitneyu(r_ceilings[x], r_ceilings[y], alternative='two-sided')
             for x, y in zip(heldout_names, matched_names)]
    median_diffs = [r_ceilings[y].median() - r_ceilings[x].median() for x, y in zip(heldout_names, matched_names)]

    df = pd.DataFrame.from_dict(r_ceilings)
    value_vars = [col for sublist in zip(heldout_names, matched_names) for col in sublist]
    results = pd.melt(df, id_vars='cellid', value_vars=value_vars, value_name=PLOT_STAT)
    results = results.rename(columns={'variable': 'model'})
    results['hue_tag'] = np.zeros(results.shape[0],)
    for i, n in enumerate(short_names):
        results.loc[results['model'].str.contains(n), 'hue_tag'] = i

    if ax is None:
        _, ax = plt.subplots()
    else:
        plt.sca(ax)
    tres=results.loc[(results[PLOT_STAT]<1) & results[PLOT_STAT]>-0.05]
    #                  palette=[DOT_COLORS['conv1dx2+d'],DOT_COLORS['conv1dx2+d'],
    #                          DOT_COLORS['LN_pop'],DOT_COLORS['LN_pop'],
    #                          DOT_COLORS['dnn1_single'],DOT_COLORS['dnn1_single']],
    sns.stripplot(x='model', y=PLOT_STAT, hue='hue_tag', data=tres, zorder=0, order=value_vars, jitter=0.2, ax=ax,
                  palette=[DOT_COLORS['conv1dx2+d'],DOT_COLORS['LN_pop'],DOT_COLORS['dnn1_single']],
                  size=2)
    ax.legend_.remove()
    sns.boxplot(x='model', y=PLOT_STAT, data=tres, boxprops={'facecolor': 'None', 'linewidth': 1},
                     showcaps=False, showfliers=False, whiskerprops={'linewidth': 0}, order=value_vars, ax=ax)
    plt.title('%s' % batch_name)

    labels = [e.get_text() for e in ax.get_xticklabels()]
    ticks = ax.get_xticks()
    w = 0.1
    for idx, model in enumerate(labels):
        idx = labels.index(model)
        j = int(idx*0.5)
        plt.hlines(reference_medians[j], ticks[idx]-w, ticks[idx]+w, color='black', linewidth=2)

    ax.set_ylim(-0.05, 1)
    ax.set_xlabel('')
    plt.xticks(rotation=45, fontsize=6, ha='right')
    if hide_xaxis:
        ax.xaxis.set_visible(False)
    plt.tight_layout()

    return [p[1] for p in tests], significant_cells, r_ceilings, reference_medians, median_diffs


if __name__ == '__main__':

    fig5, axes3 = plt.subplots(1, 2, figsize=column_and_half_short, sharex=True, sharey=True)
    tests1, sig1, r1, m1, mds1 = generate_heldout_plots(a1, 'A1', ax=axes3[0])
    tests2, sig2, r2, m2, mds2 = generate_heldout_plots(peg, 'PEG', ax=axes3[1])
    axes3[1].set_ylabel('')

    print("\n\nheldout vs matched, Sig. tests for batch %d:" % a1)
    print(tests1)
    print("median diffs:")
    print(mds1)
    print("\n")
    print("heldout vs matched, Sig. tests for batch %d:" % peg)
    print(tests2)
    print("median diffs:")
    print(mds2)
