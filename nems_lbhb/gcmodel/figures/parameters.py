import logging

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

import nems.xform_helper as xhelp
import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             get_valid_improvements,
                                             adjustFigAspect)
from nems_lbhb.gcmodel.figures.examples import improved_cells_to_list
from nems.metrics.stp import stp_magnitude
from nems_lbhb.gcmodel.magnitude import gc_magnitude
from nems_db.params import fitted_params_per_batch
import nems.modelspec as ms
from nems.modules.nonlinearity import _double_exponential

log = logging.getLogger(__name__)

plt.rcParams.update(params)  # loaded from definitions
_ALPHA = 0.3
_BINS = 30


def stp_distributions(batch, gc, stp, LN, combined, se_filter=True,
                      good_LN=0):

    gc_cells, stp_cells, both_cells = improved_cells_to_list(
            batch, gc, stp, LN, combined
            )
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter)
    cellids = df_r[LN] > good_LN
    stp_params = fitted_params_per_batch(289, stp, stats_keys=[], meta=[])
    stp_params_cells = stp_params.transpose().index.values.tolist()
    for c in stp_params_cells:
        if c not in cellids:
            cellids[c] = False

    # index keys are formatted like "2--stp.2--tau"
    mod_keys = stp.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'stp' in k:
            break
    tau_key = '%d--%s--tau' % (i, k)
    u_key = '%d--%s--u' % (i, k)

    all_taus = stp_params[stp_params.index==tau_key].transpose()[cellids].transpose()
    all_us = stp_params[stp_params.index==u_key].transpose()[cellids].transpose()
    dims = all_taus.values.flatten()[0].shape[0]

    # convert to dims x cells array instead of cells, array w/ multidim values
    sep_taus = _df_to_array(all_taus, dims)
    sep_us = _df_to_array(all_us, dims)
    stp_taus = all_taus[stp_cells]
    stp_us = all_us[stp_cells]
    stp_sep_taus = _df_to_array(stp_taus, dims)
    stp_sep_us = _df_to_array(stp_us, dims)

    fig, axes = plt.subplots(dims, 2, figsize=(12,12), sharex=True)
    hist_kwargs = {'alpha': _ALPHA, 'bins': _BINS}
    for d in range(dims):
        #tau col 0
        plt.sca(axes[d][0])
        plt.hist(sep_taus[d, :], **hist_kwargs)
        plt.hist(stp_sep_taus[d, :], **hist_kwargs, color='black')
        plt.title('tau %d' % d)
        if d == 0:
            plt.legend(['all cells', 'stp > gc'])

        #u col 1
        plt.sca(axes[d][1])
        plt.hist(sep_us[d, :], **hist_kwargs)
        plt.hist(stp_sep_us[d, :], **hist_kwargs, color='black')
        plt.title('u %d' % d)

    fig.suptitle('STP parameter distributions')


def gc_distributions(batch, gc, stp, LN, combined, se_filter=True, good_LN=0):

    gc_cells, stp_cells, both_cells = improved_cells_to_list(
            batch, gc, stp, LN, combined
            )
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter)
    cellids = df_r[LN] > good_LN
    gc_params = fitted_params_per_batch(289, gc, stats_keys=[], meta=[])
    gc_params_cells = gc_params.transpose().index.values.tolist()
    for c in gc_params_cells:
        if c not in cellids:
            cellids[c] = False

    # index keys are formatted like "4--dsig.d--kappa"
    mod_keys = gc.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'dsig' in k:
            break
    b_key = f'{i}--{k}--base'
    a_key = f'{i}--{k}--amplitude'
    s_key = f'{i}--{k}--shift'
    k_key = f'{i}--{k}--kappa'
    ka_key = k_key + '_mod'
    ba_key = b_key + '_mod'
    aa_key = a_key + '_mod'
    sa_key = s_key + '_mod'
    all_keys = [b_key, a_key, s_key, k_key, ba_key, aa_key, sa_key, ka_key]

    phi_dfs = [gc_params[gc_params.index==k].transpose()[cellids].transpose()
               for k in all_keys]
    sep_dfs = [df.values.flatten().astype(np.float64) for df in phi_dfs]
    gc_sep_dfs = [df[gc_cells].values.flatten().astype(np.float64)
                  for df in phi_dfs]
    diffs = [sep_dfs[i+1] - sep_dfs[i]
             for i, _ in enumerate(sep_dfs[:-1])
             if i % 2 == 0]
    gc_diffs = [gc_sep_dfs[i+1] - gc_sep_dfs[i]
                for i, _ in enumerate(gc_sep_dfs[:-1])
                if i % 2 == 0]

    fig, axes = plt.subplots(3, 4, figsize=(16,13), sharex=True, sharey=True)
    flatax = axes.flatten()
    hist_kwargs = {'alpha': _ALPHA, 'bins': _BINS}
    for i in range(12):
        plt.sca(flatax[i])
        if i < 8:
            plt.hist(sep_dfs[i], **hist_kwargs)
            plt.hist(gc_sep_dfs[i], **hist_kwargs, color='black')
            plt.title(all_keys[i].split('-')[-1])
            if i == 0:
                plt.legend(['all cells', 'gc > stp'])
        else:
            plt.hist(diffs[i-8], **hist_kwargs)
            plt.hist(gc_diffs[i-8], **hist_kwargs, color='black')
            plt.title(all_keys[i-8].split('-')[-1] + ' difference')

    fig.suptitle('GC parameter distributions')

def _df_to_array(df, dims):
    vals = df.values.flatten()
    array = np.array([np.array([vals[j][k] for j, v in enumerate(vals)])
                      for k in range(dims)])
    return array


def gd_ratio(batch, gc, stp, LN, combined, se_filter=True, good_LN=0, bins=60,
             truncate=False):
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids = df_r[LN] > good_LN
    gc_cells = (df_r[LN] > good_LN) & (df_r[gc] > df_r[LN])
    gc_params = fitted_params_per_batch(289, gc, stats_keys=[])
    # drop cellids that haven't been fit for all models
    gc_params_cells = gc_params.transpose().index.values.tolist()
    for c in gc_params_cells:
        if c not in cellids:
            cellids[c] = False
        if c not in gc_cells:
            gc_cells[c] = False

    # index keys are formatted like "4--dsig.d--kappa"
    mod_keys = gc.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'dsig' in k:
            break
    k_key = f'{i}--{k}--kappa'
    ka_key = k_key + '_mod'
    r_test_key = 'meta--r_test'
    all_keys = [k_key, ka_key, r_test_key]
    phi_dfs = [gc_params[gc_params.index==k].transpose()[cellids].transpose()
               for k in all_keys]
    sep_dfs = [df.values.flatten().astype(np.float64) for df in phi_dfs]
    gc_dfs = [gc_params[gc_params.index==k].transpose()[gc_cells].transpose()
               for k in all_keys]
    gc_sep_dfs = [df.values.flatten().astype(np.float64) for df in gc_dfs]
    low, high, r_test = sep_dfs
    gc_low, gc_high, gc_r = gc_sep_dfs

    # need to fix the range for comparison to rabinowitz ratio -
    # scale negative infinity for our kappa to 0
    # and scale positive infinity to 2 just to make the range easy to work with.
    # the raw values shouldn't really matter since we're taking a ratio
#    abs_max_kappa = np.nanmax(np.abs(np.concatenate([low, high])))
#    low = low / abs_max_kappa + 1
#    high = high / abs_max_kappa + 1

    if truncate:
        log.warning('This option should only be used when dealing with old '
                    'fits where kappa had a lower bound of 0.')
        high[high < 0.1] = 0.1
        gc_high[gc_high < 0.1] = 0.1

    #ratio = low/high
    ratio = np.exp(low - high)
    #gc_ratio = gc_low/gc_high
    gc_ratio = np.exp(low - high)

    if truncate:
        ratio[ratio > 10] = 10
        gc_ratio[gc_ratio > 10] = 10

    fig1 = plt.figure(figsize=figsize)
    plt.hist(ratio, bins=bins)
    plt.title('e^low/e^high')

    fig2 = plt.figure(figsize=figsize)
    plt.hist(gc_ratio, bins=bins)
    plt.title('e^low/e^high, gc improvements only')

    fig3 = plt.figure(figsize=figsize)
    plt.scatter(ratio, r_test)
    plt.title('e^low/e^high vs r_test')

    fig4 = plt.figure(figsize=figsize)
    plt.scatter(gc_ratio, gc_r)
    plt.title('e^low/e^high vs r_test, gc improvements only')


def gain_by_contrast_slopes(batch, gc, stp, LN, combined, se_filter=True,
                            good_LN=0, bins=60, truncate=False):

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids = df_r[LN] > good_LN
    gc_cells = (df_r[LN] > good_LN) & (df_r[gc] > df_r[LN])
    gc_params = fitted_params_per_batch(289, gc, stats_keys=[])
    # drop cellids that haven't been fit for all models
    gc_params_cells = gc_params.transpose().index.values.tolist()
    for c in gc_params_cells:
        if c not in cellids:
            cellids[c] = False
        if c not in gc_cells:
            gc_cells[c] = False

    # index keys are formatted like "4--dsig.d--kappa"
    mod_keys = gc.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'dsig' in k:
            break
    k_key = f'{i}--{k}--kappa'
    ka_key = k_key + '_mod'
    r_test_key = 'meta--r_test'
    all_keys = [k_key, ka_key, r_test_key]
    phi_dfs = [gc_params[gc_params.index==k].transpose()[cellids].transpose()
               for k in all_keys]
    sep_dfs = [df.values.flatten().astype(np.float64) for df in phi_dfs]
    gc_dfs = [gc_params[gc_params.index==k].transpose()[gc_cells].transpose()
               for k in all_keys]
    gc_sep_dfs = [df.values.flatten().astype(np.float64) for df in gc_dfs]
    low, high, r_test = sep_dfs
    gc_low, gc_high, gc_r = gc_sep_dfs

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot([np.zeros(low.shape), np.ones(high.shape)], [low, high],
                 color='black', alpha=0.2)
    axes[1].hist(high-low, bins=60)

    fig1, axes1 = plt.subplots(1, 2, figsize=figsize)
    axes1[0].plot([np.zeros(gc_low.shape), np.ones(gc_high.shape)],
                  [gc_low, gc_high], color='black', alpha=0.2)
    axes1[1].hist(gc_high-gc_low, bins=60)
    plt.title('gc only')

    plt.figure(figsize=figsize)
    plt.scatter(high-low, r_test)

    plt.figure(figsize=figsize)
    plt.scatter(gc_high-gc_low, gc_r)
    plt.title('gc only')



def dynamic_sigmoid_range(cellid, batch, modelname):
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    modelspec = ctx['modelspec']
    val = ctx['val']
    modelspec.recording = val
    lows = {k: v for k, v in modelspec[-1]['phi'].items()
            if '_mod' not in k}
    highs = {k[:-4]: v for k, v in modelspec[-1]['phi'].items()
             if '_mod' in k}
    for k in lows:
        if k not in highs:
            highs[k] = lows[k]
    val_before_dsig = ms.evaluate(val, modelspec, stop=-1)
    pred_before_dsig = val_before_dsig['pred'].as_continuous().T
    low_out = _double_exponential(pred_before_dsig, **lows)
    high_out = _double_exponential(pred_before_dsig, **highs)

    fig = plt.figure(figsize=figsize)
    plt.scatter(pred_before_dsig, low_out, color='blue', s=0.7, alpha=0.6)
    plt.scatter(pred_before_dsig, high_out, color='red', s=0.7, alpha=0.6)
