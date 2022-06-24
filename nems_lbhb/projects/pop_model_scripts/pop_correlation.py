from pathlib import Path
import pickle
import requests

import numpy as np
import pandas as pd
import scipy.stats as st

import nems
import nems.db as nd
import nems.xform_helper as xhelp

from pop_model_utils import (mplparams, get_significant_cells, SIG_TEST_MODELS, EQUIVALENCE_MODELS_SINGLE,
                             EQUIVALENCE_MODELS_POP, DOT_COLORS, MODELGROUPS)

import matplotlib as mpl
mpl.rcParams.update(mplparams)
import matplotlib.pyplot as plt
import seaborn as sns


def sanity_check_LN(batch, modelnames, save_path=None, load_path=None):
    if load_path is not None:
        corrs = pd.read_pickle(load_path)
        return corrs
    else:
        cellids = []
        pop_vs_LN = []
    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    for cellid in significant_cells:
        # Load and evaluate each model, pull out validation pred signal for each one.
        contexts = [xhelp.load_model_xform(cellid, batch, m, eval_model=True)[1] for m in modelnames]
        preds = [c['val'].apply_mask()['pred'].as_continuous() for c in contexts]

        # Compute correlation between eaceh pair of models, append to running list.
        # 0: conv2d, 1: conv1dx2+d, 2: LN_pop,  # TODO: if EQUIVALENCE_MODELS changes, this needs to change as well
        pop_vs_LN.append(np.corrcoef(preds[0], preds[1])[0, 1])   # correlate conv1dx2+d with LN_pop
        cellids.append(cellid)

        # Convert to dataframe and save after each cell, in case there's a crash.
        corrs = {'cellid': cellids, 'pop_vs_LN': pop_vs_LN}
        corrs = pd.DataFrame.from_dict(corrs)
        corrs.set_index('cellid', inplace=True)
        if save_path is not None:
            corrs.to_pickle(save_path)

    return corrs


# Use this for first stage of fit (with all cellids in one recording)
def generate_psth_correlations_pop(batch, modelnames, save_path=None, load_path=None):
    if load_path is not None:
        corrs = pd.read_pickle(load_path)
        return corrs
    else:
        cellids = []
        c2d_c1d = []
        c2d_LN = []
        c1d_LN = []
    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    # Load and evaluate each model, pull out validation pred signal for each one.
    contexts = [xhelp.load_model_xform(significant_cells[0], batch, m, eval_model=True)[1] for m in modelnames]
    preds = [c['val'].apply_mask()['pred'] for c in contexts]
    chans = preds[0].chans  # not all the models load chans for some reason
    for i, _ in enumerate(preds[1:]):
        preds[i+1].chans = chans
    preds = [p.extract_channels(significant_cells).as_continuous() for p in preds]

    for i, cellid in enumerate(significant_cells):
        # Compute correlation between eaceh pair of models, append to running list.
        # 0: conv2d, 1: conv1dx2+d, 2: LN_pop,  # TODO: if EQUIVALENCE_MODELS_POP changes, this needs to change as well
        c2d_c1d.append(np.corrcoef(preds[0][i], preds[1][i])[0, 1])  # correlate conv2d with conv1dx2+d
        c2d_LN.append(np.corrcoef(preds[0][i], preds[2][i])[0, 1])   # correlate conv2d with LN_pop
        c1d_LN.append(np.corrcoef(preds[1][i], preds[2][i])[0, 1])   # correlate conv1dx2+d with LN_pop
        cellids.append(cellid)

    # Convert to dataframe and save after each cell, in case there's a crash.
    corrs = {'cellid': cellids, 'c2d_c1d': c2d_c1d, 'c2d_LN': c2d_LN, 'c1d_LN': c1d_LN}
    corrs = pd.DataFrame.from_dict(corrs)
    corrs.set_index('cellid', inplace=True)
    if save_path is not None:
        corrs.to_pickle(save_path)

    return corrs


# Use this for second stage of fit (single cell)
def generate_psth_correlations_single(batch, modelnames, save_path=None, load_path=None, test_limit=None, force_rerun=False,
                                      skip_new_cells=True):
    if load_path is not None:
        corrs = pd.read_pickle(load_path)
        cellids = corrs.index.values.tolist()
        c2d_c1d = corrs['c2d_c1d'].values.tolist()
        c2d_LN = corrs['c2d_LN'].values.tolist()
        c1d_LN = corrs['c1d_LN'].values.tolist()
    else:
        cellids = []
        c2d_c1d = []
        c2d_LN = []
        c1d_LN = []

    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)
    for cellid in significant_cells[:test_limit]:
        if (cellid in cellids) and (not force_rerun):
            #print(f'skipping cellid: {cellid}')
            continue
        if skip_new_cells:
            # Don't stop to add new correlations for cells that weren't included in
            # a previous analysis (like if new recordings have been done for the same batch).
            continue

        # Load and evaluate each model, pull out validation pred signal for each one.
        contexts = [xhelp.load_model_xform(cellid, batch, m, eval_model=True)[1] for m in modelnames]
        preds = [c['val'].apply_mask()['pred'].as_continuous() for c in contexts]

        # Compute correlation between eaceh pair of models, append to running list.
        # 0: conv2d, 1: conv1dx2+d, 2: LN_pop,  # TODO: if EQUIVALENCE_MODELS changes, this needs to change as well
        c2d_c1d.append(np.corrcoef(preds[0], preds[1])[0, 1])  # correlate conv2d with conv1dx2+d
        c2d_LN.append(np.corrcoef(preds[0], preds[2])[0, 1])   # correlate conv2d with LN_pop
        c1d_LN.append(np.corrcoef(preds[1], preds[2])[0, 1])   # correlate conv1dx2+d with LN_pop
        cellids.append(cellid)

        # Convert to dataframe and save after each cell, in case there's a crash.
        corrs = {'cellid': cellids, 'c2d_c1d': c2d_c1d, 'c2d_LN': c2d_LN, 'c1d_LN': c1d_LN}
        corrs = pd.DataFrame.from_dict(corrs)
        corrs.set_index('cellid', inplace=True)
        if save_path is not None:
            corrs.to_pickle(save_path)

    return corrs


def correlation_histogram(batch, batch_name, save_path=None, load_path=None, test_limit=None, force_rerun=False,
                          use_pop_models=False, ax=None, skip_new_cells=True, plot_LN=False, LN_save=None, LN_load=None):
    # Load correlations and significance tests
    if use_pop_models:
        correlations = generate_psth_correlations_pop(batch, EQUIVALENCE_MODELS_POP, save_path=save_path,
                                                      load_path=load_path)
    else:
        correlations = generate_psth_correlations_single(batch, EQUIVALENCE_MODELS_SINGLE, save_path=save_path,
                                                         load_path=load_path, test_limit=test_limit,
                                                         force_rerun=force_rerun, skip_new_cells=skip_new_cells)

    correlations = correlations.drop('c2d_LN', axis=1)
    stats_tests = st.wilcoxon(correlations['c2d_c1d'], correlations['c1d_LN'], alternative='two-sided')

    # Plot all distributions of correlations on common bins
    if ax is None:
        _, ax = plt.subplots()
    else:
        plt.sca(ax)
    bins = np.histogram(np.hstack([c.values for _, c in correlations.items()]), bins=20)[1]
    colors = [DOT_COLORS['2D CNN'], DOT_COLORS['pop LN']]  # 1D CNNx2 in common, so color by other model

    c1 = correlations['c2d_c1d']
    c2 = correlations['c1d_LN']
    if plot_LN:
        c3 = sanity_check_LN(batch, [MODELGROUPS['LN'][4], EQUIVALENCE_MODELS_SINGLE[2]], save_path=LN_save,
                             load_path=LN_load)
        ax.hist(c3, bins=bins, alpha=1.0, color='darkgray', edgecolor='black', linewidth=0.5,
                histtype='stepfilled')

    else:
        ax.hist(c1, bins=bins, alpha=1, color=DOT_COLORS['2D CNN'], edgecolor='black', linewidth=0.5,
                histtype='stepfilled')
        ax.hist(c2, bins=bins, alpha=1, color=DOT_COLORS['pop LN'], edgecolor='black', linewidth=0.5,
                histtype='stepfilled')
        ax.hist(c1, bins=bins, alpha=1, color=DOT_COLORS['2D CNN'], edgecolor='black', linewidth=0.5,
                histtype='stepfilled', fc='None', hatch='\\\\\\\\')

    plt.xlabel('PSTH Correlation')
    plt.ylabel('Count')

    #plt.title('%s' % batch_name)
    #plt.legend()
    plt.tight_layout()

    return correlations, stats_tests


if __name__ == '__main__':
    a1 = 322
    peg = 323
    a1_corr_path = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(a1) / 'corr_nat4.pkl'
    a1_corr_path_pop = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(a1) / 'corr_nat4_pop.pkl'
    a1_corr_path_LN = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(a1) / 'corr_nat4_LN_test.pkl'
    peg_corr_path = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(peg) / 'corr_nat4.pkl'
    peg_corr_path_pop = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(peg) / 'corr_nat4_pop.pkl'
    peg_corr_path_LN = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(peg) / 'corr_nat4_LN_test.pkl'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 6))

    # Use this version when re-running all cells, in case there are connection issues.
    # for i in range(10):
    #     try:
    #         # Use load_path=None for first run, change to load_path=a1_corr_path for subsequent loading
    #         a1_corr, a1_p, a1_t = correlation_histogram(a1, 'A1', save_path=a1_corr_path, test_limit=None, load_path=None)#a1_corr_path)
    #         break
    #     except ConnectionError:
    #         print(f'failed mysql connection x{i}')
    # for i in range(10):
    #     try:
    #         # Use load_path similar to above
    #         peg_corr, peg_p, peg_t = correlation_histogram(peg, 'PEG', save_path=peg_corr_path, test_limit=None, load_path=None)#peg_corr_path)
    #         break
    #     except requests.exceptions.ConnectionError:
    #         print(f'failed mysql connection x{i}')

    # For using first stage fit / pop models
    # a1_corr_pop, a1_p_pop, a1_t_pop = correlation_histogram(a1, 'A1', save_path=a1_corr_path_pop, load_path=None, use_pop_models=True)
    # peg_corr_pop, peg_p_pop, peg_t_pop = correlation_histogram(peg, 'PEG', save_path=peg_corr_path_pop, load_path=None, use_pop_models=True)

    # For debugging without overwriting
    #a1_corr, a1_p, a1_t = correlation_histogram(a1, 'A1', save_path=None, test_limit=None, load_path=a1_corr_path, force_rerun=True)
    # peg_corr, peg_p, peg_t = correlation_histogram(peg, 'PEG', save_path=None, test_limit=None, load_path=peg_corr_path, force_rerun=True)

    # To run plot when everything is done
    # a1_corr, a1_stats = correlation_histogram(a1, 'A1', save_path=a1_corr_path, load_path=a1_corr_path, force_rerun=False,
    #                                             skip_new_cells=True, ax=ax1)
    # peg_corr, peg_stats = correlation_histogram(peg, 'PEG', save_path=a1_corr_path, load_path=peg_corr_path, force_rerun=False,
    #                                                skip_new_cells=True, ax=ax2)
    # fig.tight_layout()

    # T-statistic, p-value
    # print("A1 sig tests: %s" % a1_stats)
    # print("PEG sig tests: %s" % peg_stats)

    # test LN vs pop_LN
    # a1_corr_LN = sanity_check_LN(a1, [MODELGROUPS['LN'][4], EQUIVALENCE_MODELS_SINGLE[2]],
    #                              save_path=a1_corr_path_LN, load_path=a1_corr_path_LN)
    # peg_corr_LN = sanity_check_LN(peg, [MODELGROUPS['LN'][4], EQUIVALENCE_MODELS_SINGLE[2]],
    #                               save_path=peg_corr_path_LN, load_path=peg_corr_path_LN)

    a1_corr, _ = correlation_histogram(a1, 'A1', save_path=a1_corr_path, load_path=a1_corr_path,
                                              ax=ax1, plot_LN=True, LN_save=a1_corr_path_LN, LN_load=a1_corr_path_LN)
    peg_corr, _ = correlation_histogram(peg, 'PEG', save_path=peg_corr_path, load_path=peg_corr_path,
                                                ax=ax2, plot_LN=True, LN_save=peg_corr_path_LN, LN_load=peg_corr_path_LN)
    fig.tight_layout()
    plt.show(block=True)
