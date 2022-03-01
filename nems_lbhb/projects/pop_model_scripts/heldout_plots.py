import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd

import nems
import nems.db as nd
import nems.xform_helper as xhelp
import nems_lbhb.xform_wrappers as xwrap
import nems.epoch as ep
from nems.xforms import evaluate_step
import nems_lbhb.baphy_io as io

from pop_model_utils import (get_significant_cells, SIG_TEST_MODELS, MODELGROUPS, HELDOUT, MATCHED, PLOT_STAT)


def get_heldout_results(batch, significant_cells, short_names):
    modelgroups = {}
    for i, name in enumerate(short_names):
        modelgroups[name + '_heldout'] = HELDOUT[i]
        modelgroups[name + '_matched'] = MATCHED[i]

    r_ceilings = {}
    for n in short_names:
        # sum is just to collapse cols
        heldout_r_ceiling = nd.batch_comp(batch, [modelgroups[n + '_heldout']], cellids=significant_cells,
                                          stat=PLOT_STAT)[modelgroups[n + '_heldout']]
        matched_r_ceiling = nd.batch_comp(batch, [modelgroups[n + '_matched']], cellids=significant_cells,
                                          stat=PLOT_STAT)[modelgroups[n + '_matched']]

        r_ceilings[n + '_heldout'] = heldout_r_ceiling
        r_ceilings[n + '_matched'] = matched_r_ceiling

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

    heldout_names = [n + '_heldout' for n in short_names]
    matched_names = [n + '_matched' for n in short_names]
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
    sns.stripplot(x='model', y=PLOT_STAT, hue='hue_tag', data=results, zorder=0, order=value_vars, jitter=0.2, ax=ax,
                  size=3)
    ax.legend_.remove()
    sns.boxplot(x='model', y=PLOT_STAT, data=results, boxprops={'facecolor': 'None', 'linewidth': 2},
                     showcaps=False, showfliers=False, whiskerprops={'linewidth': 0}, order=value_vars, ax=ax)
    plt.title('%s' % batch_name)

    labels = [e.get_text() for e in ax.get_xticklabels()]
    ticks = ax.get_xticks()
    w = 0.1
    for idx, model in enumerate(labels):
        idx = labels.index(model)
        j = int(idx*0.5)
        plt.hlines(reference_medians[j], ticks[idx]-w, ticks[idx]+w)

    plt.ylim(None, 1)
    plt.xticks(rotation=45)
    if hide_xaxis:
        ax.xaxis.set_visible(False)
    plt.tight_layout()

    return [p[1] for p in tests], significant_cells, r_ceilings, reference_medians, median_diffs


if __name__ == '__main__':

    a1 = 322
    peg = 323

    print("Generating plots for batch %d" % a1)
    tests1, sig1, r1, m1, mds1 = generate_heldout_plots(a1, 'A1')

    print("Generating plots for batch %d" % peg)
    tests2, sig2, r2, m2, mds2 = generate_heldout_plots(peg, 'PEG')

    print("Sig. tests for batch %d:" % a1)
    print(tests1)
    print("median diffs:")
    print(mds1)
    print("\n")
    print("Sig. tests for batch %d:" % peg)
    print(tests2)
    print("median diffs:")
    print(mds2)
