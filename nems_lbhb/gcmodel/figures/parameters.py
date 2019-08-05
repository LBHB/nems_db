import logging
import os

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import nems.xform_helper as xhelp
import nems.db as nd
import nems.epoch as ep
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             get_valid_improvements,
                                             adjustFigAspect)
from nems_lbhb.gcmodel.figures.examples import improved_cells_to_list
from nems.metrics.stp import stp_magnitude
from nems_lbhb.gcmodel.magnitude import gc_magnitude
from nems_lbhb.gcmodel.figures.soundstats import silence_duration
from nems_db.params import fitted_params_per_batch
import nems.modelspec as ms
from nems.modules.nonlinearity import _double_exponential, _saturated_rectifier

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
             use_exp=True):
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    #cellids = df_r[LN] > good_LN
    cellids = df_r[LN] > df_e[LN]*2
    gc_LN_SE = (df_e[gc] + df_e[LN])
    #stp_LN_SE = (df_e[stp] + df_e[LN])
    gc_cells = cellids & ((df_r[gc] - df_r[LN]) > gc_LN_SE)
    #stp_cells = (df_r[LN] > good_LN) & ((df_r[stp] - df_r[LN]) > stp_LN_SE)
    #both_cells = gc_cells & stp_cells
    LN_cells = cellids & np.logical_not(gc_cells)
    #stp_cells = stp_cells & np.logical_not(both_cells)
    meta = ['r_test', 'ctmax_val', 'ctmax_est', 'ctmin_val', 'ctmin_est']
    gc_params = fitted_params_per_batch(289, gc, stats_keys=[], meta=meta)
    # drop cellids that haven't been fit for all models
    gc_params_cells = gc_params.transpose().index.values.tolist()
    for c in gc_params_cells:
        if c not in LN_cells:
            LN_cells[c] = False
        if c not in gc_cells:
            gc_cells[c] = False
#        if c not in stp_cells:
#            stp_cells[c] = False
#        if c not in both_cells:
#            both_cells[c] = False

    # index keys are formatted like "4--dsig.d--kappa"
    mod_keys = gc.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'dsig' in k:
            break
    k_key = f'{i}--{k}--kappa'
    ka_key = k_key + '_mod'
    meta_keys = ['meta--' + k for k in meta]
    all_keys = [k_key, ka_key] + meta_keys
    phi_dfs = [gc_params[gc_params.index==k].transpose()[LN_cells].transpose()
               for k in all_keys]
    sep_dfs = [df.values.flatten().astype(np.float64) for df in phi_dfs]
    gc_dfs = [gc_params[gc_params.index==k].transpose()[gc_cells].transpose()
               for k in all_keys]
    gc_sep_dfs = [df.values.flatten().astype(np.float64) for df in gc_dfs]
#    stp_dfs = [gc_params[gc_params.index==k].transpose()[stp_cells].transpose()
#               for k in all_keys]
#    stp_sep_dfs = [df.values.flatten().astype(np.float64) for df in stp_dfs]
#    both_dfs = [gc_params[gc_params.index==k].transpose()[both_cells].transpose()
#               for k in all_keys]
#    both_sep_dfs = [df.values.flatten().astype(np.float64) for df in both_dfs]
    low, high, r_test, ctmax_val, ctmax_est, ctmin_val, ctmin_est = sep_dfs
    gc_low, gc_high, gc_r, gc_ctmax_val, \
        gc_ctmax_est, gc_ctmin_val, gc_ctmin_est = gc_sep_dfs
#    stp_low, stp_high, stp_r, stp_ctmax_val, \
#        stp_ctmax_est, stp_ctmin_val, stp_ctmin_est = stp_sep_dfs
#    both_low, both_high, both_r, both_ctmax_val, \
#        both_ctmax_est, both_ctmin_val, both_ctmin_est = both_sep_dfs

    ctmax = np.maximum(ctmax_val, ctmax_est)
    gc_ctmax = np.maximum(gc_ctmax_val, gc_ctmax_est)
    ctmin = np.minimum(ctmin_val, ctmin_est)
    gc_ctmin = np.minimum(gc_ctmin_val, gc_ctmin_est)
#    stp_ctmax = np.maximum(stp_ctmax_val, stp_ctmax_est)
#    stp_ctmin = np.minimum(stp_ctmin_val, stp_ctmin_est)
#    both_ctmax = np.maximum(both_ctmax_val, both_ctmax_est)
#    both_ctmin = np.minimum(both_ctmin_val, both_ctmin_est)

    k_low = low + (high - low)*ctmin
    k_high = low + (high - low)*ctmax
    gc_k_low = gc_low + (gc_high - gc_low)*gc_ctmin
    gc_k_high = gc_low + (gc_high - gc_low)*gc_ctmax
#    stp_k_low = stp_low + (stp_high - stp_low)*stp_ctmin
#    stp_k_high = stp_low + (stp_high - stp_low)*stp_ctmax
#    both_k_low = both_low + (both_high - both_low)*both_ctmin
#    both_k_high = both_low + (both_high - both_low)*both_ctmax

    if use_exp:
        k_low = np.exp(k_low)
        k_high = np.exp(k_high)
        gc_k_low = np.exp(gc_k_low)
        gc_k_high = np.exp(gc_k_high)
#        stp_k_low = np.exp(stp_k_low)
#        stp_k_high = np.exp(stp_k_high)
#        both_k_low = np.exp(both_k_low)
#        both_k_high = np.exp(both_k_high)

    ratio = k_low / k_high
    gc_ratio = gc_k_low / gc_k_high
#    stp_ratio = stp_k_low / stp_k_high
#    both_ratio = both_k_low / both_k_high

    fig1, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=figsize)
    ax1.hist(ratio, bins=bins)
    ax1.set_title('all cells')
    ax2.hist(gc_ratio, bins=bins)
    ax2.set_title('gc')
#    ax3.hist(stp_ratio, bins=bins)
#    ax3.set_title('stp')
    if not use_exp:
        title = 'k_low / k_high'
    else:
        title = 'e^(k_low - k_high)'
    fig1.suptitle(title)


    fig3 = plt.figure(figsize=figsize)
    plt.scatter(ratio, r_test)
    plt.title('low/high vs r_test')

    fig4 = plt.figure(figsize=figsize)
    plt.scatter(gc_ratio, gc_r)
    plt.title('low/high vs r_test, gc improvements only')


def gain_by_contrast_slopes(batch, gc, stp, LN, combined, se_filter=True,
                            good_LN=0, bins=60, use_exp=True):

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    #cellids = df_r[LN] > good_LN
    cellids = df_r[LN] > df_e[LN]*2
    gc_LN_SE = (df_e[gc] + df_e[LN])
#    stp_LN_SE = (df_e[stp] + df_e[LN])
    gc_cells = (cellids) & ((df_r[gc] - df_r[LN]) > gc_LN_SE)
#    stp_cells = (df_r[LN] > good_LN) & ((df_r[stp] - df_r[LN]) > stp_LN_SE)
#    both_cells = gc_cells & stp_cells
#    gc_cells = gc_cells & np.logical_not(both_cells)
#    stp_cells = stp_cells & np.logical_not(both_cells)
    LN_cells = cellids & np.logical_not(gc_cells)# | stp_cells | both_cells)
    meta = ['r_test', 'ctmax_val', 'ctmax_est', 'ctmin_val', 'ctmin_est']
    gc_params = fitted_params_per_batch(289, gc, stats_keys=[], meta=meta)
    # drop cellids that haven't been fit for all models
    gc_params_cells = gc_params.transpose().index.values.tolist()
    for c in gc_params_cells:
        if c not in LN_cells:
            LN_cells[c] = False
        if c not in gc_cells:
            gc_cells[c] = False
#        if c not in stp_cells:
#            stp_cells[c] = False
#        if c not in both_cells:
#            both_cells[c] = False

    # index keys are formatted like "4--dsig.d--kappa"
    mod_keys = gc.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'dsig' in k:
            break
    k_key = f'{i}--{k}--kappa'
    ka_key = k_key + '_mod'
    meta_keys = ['meta--' + k for k in meta]
    all_keys = [k_key, ka_key] + meta_keys
    phi_dfs = [gc_params[gc_params.index==k].transpose()[LN_cells].transpose()
               for k in all_keys]
    sep_dfs = [df.values.flatten().astype(np.float64) for df in phi_dfs]
    gc_dfs = [gc_params[gc_params.index==k].transpose()[gc_cells].transpose()
               for k in all_keys]
    gc_sep_dfs = [df.values.flatten().astype(np.float64) for df in gc_dfs]
#    stp_dfs = [gc_params[gc_params.index==k].transpose()[stp_cells].transpose()
#               for k in all_keys]
#    stp_sep_dfs = [df.values.flatten().astype(np.float64) for df in stp_dfs]
#    both_dfs = [gc_params[gc_params.index==k].transpose()[both_cells].transpose()
#               for k in all_keys]
#    both_sep_dfs = [df.values.flatten().astype(np.float64) for df in both_dfs]
    low, high, r_test, ctmax_val, ctmax_est, ctmin_val, ctmin_est = sep_dfs
    gc_low, gc_high, gc_r, gc_ctmax_val, \
        gc_ctmax_est, gc_ctmin_val, gc_ctmin_est = gc_sep_dfs
#    stp_low, stp_high, stp_r, stp_ctmax_val, \
#        stp_ctmax_est, stp_ctmin_val, stp_ctmin_est = stp_sep_dfs
#    both_low, both_high, both_r, both_ctmax_val, \
#        both_ctmax_est, both_ctmin_val, both_ctmin_est = both_sep_dfs

    ctmax = np.maximum(ctmax_val, ctmax_est)
    gc_ctmax = np.maximum(gc_ctmax_val, gc_ctmax_est)
    ctmin = np.minimum(ctmin_val, ctmin_est)
    gc_ctmin = np.minimum(gc_ctmin_val, gc_ctmin_est)
#    stp_ctmax = np.maximum(stp_ctmax_val, stp_ctmax_est)
#    stp_ctmin = np.minimum(stp_ctmin_val, stp_ctmin_est)
#    both_ctmax = np.maximum(both_ctmax_val, both_ctmax_est)
#    both_ctmin = np.minimum(both_ctmin_val, both_ctmin_est)
    ct_range = ctmax - ctmin
    gc_ct_range = gc_ctmax - gc_ctmin
#    stp_ct_range = stp_ctmax - stp_ctmin
#    both_ct_range = both_ctmax - both_ctmin
    gain = (high - low)*ct_range
    gc_gain = (gc_high - gc_low)*gc_ct_range
    # test hyp. that gc_gains are more negative than LN
    gc_LN_p = st.mannwhitneyu(gc_gain, gain, alternative='two-sided')[1]
    med_gain = np.median(gain)
    gc_med_gain = np.median(gc_gain)
#    stp_gain = (stp_high - stp_low)*stp_ct_range
#    both_gain = (both_high - both_low)*both_ct_range

    k_low = low + (high - low)*ctmin
    k_high = low + (high - low)*ctmax
    gc_k_low = gc_low + (gc_high - gc_low)*gc_ctmin
    gc_k_high = gc_low + (gc_high - gc_low)*gc_ctmax
#    stp_k_low = stp_low + (stp_high - stp_low)*stp_ctmin
#    stp_k_high = stp_low + (stp_high - stp_low)*stp_ctmax
#    both_k_low = both_low + (both_high - both_low)*both_ctmin
#    both_k_high = both_low + (both_high - both_low)*both_ctmax

    if use_exp:
        k_low = np.exp(k_low)
        k_high = np.exp(k_high)
        gc_k_low = np.exp(gc_k_low)
        gc_k_high = np.exp(gc_k_high)
#        stp_k_low = np.exp(stp_k_low)
#        stp_k_high = np.exp(stp_k_high)
#        both_k_low = np.exp(both_k_low)
#        both_k_high = np.exp(both_k_high)

#    fig = plt.figure(figsize=figsize)#, axes = plt.subplots(1, 2, figsize=figsize)
#    #axes[0].plot([ctmin, ctmax], [k_low, k_high], color='black', alpha=0.5)
#    plt.hist(high-low, bins=bins, color='black', alpha=0.5)
#
#    #axes[0].plot([gc_ctmin, gc_ctmax], [gc_k_low, gc_k_high], color='red',
#    #              alpha=0.3)
#    plt.hist(gc_high-gc_low, bins=bins, color='red', alpha=0.3)
#
#    #axes[0].plot([stp_ctmin, stp_ctmax], [stp_k_low, stp_k_high], color='blue',
#    #              alpha=0.3)
#    plt.hist(stp_high-stp_low, bins=bins, color='blue', alpha=0.3)
#    plt.xlabel('gain slope')
#    plt.ylabel('count')
#    plt.title(f'raw counts, LN > {good_LN}')
#    plt.legend([f'LN, {len(low)}', f'gc, {len(gc_low)}', f'stp, {len(stp_low)}',
#                f'Both, {len(both_low)}'])

    smallest_slope = min(np.min(gain), np.min(gc_gain))#, np.min(stp_gain),
                         #np.min(both_gain))
    largest_slope = max(np.max(gain), np.max(gc_gain))#, np.max(stp_gain),
                        #np.max(both_gain))
    slope_range = (smallest_slope, largest_slope)
    bins = np.linspace(smallest_slope, largest_slope, bins)
    bar_width = bins[1]-bins[0]
    axis_locs = bins[:-1]
    hist = np.histogram(gain, bins=bins, range=slope_range)
    gc_hist = np.histogram(gc_gain, bins=bins, range=slope_range)
#    stp_hist = np.histogram(stp_gain, bins=bins, range=slope_range)
#    both_hist = np.histogram(both_gain, bins=bins, range=slope_range)
    raw = hist[0]
    gc_raw = gc_hist[0]
#    stp_raw = stp_hist[0]
#    both_raw = both_hist[0]
    #prop_hist = hist[0] / np.sum(hist[0])
    #prop_gc_hist = gc_hist[0] / np.sum(gc_hist[0])
#    prop_stp_hist = stp_hist[0] / np.sum(stp_hist[0])
#    prop_both_hist = both_hist[0] / np.sum(both_hist[0])


    fig1 = plt.figure(figsize=figsize)
    plt.bar(axis_locs, raw, width=bar_width, color='gray', alpha=0.8)
    plt.bar(axis_locs, gc_raw, width=bar_width, color='maroon', alpha=0.8,
            bottom=raw)
#    plt.bar(axis_locs, stp_raw, width=bar_width, color='teal', alpha=0.8,
#            bottom=raw+gc_raw)
#    plt.bar(axis_locs, both_raw, width=bar_width, color='goldenrod', alpha=0.8,
#            bottom=raw+gc_raw+stp_raw)
    plt.xlabel('gain slope')
    plt.ylabel('count')
    plt.title(f'raw counts, LN > {good_LN}')
    plt.legend([f'LN, {len(low)}, md={med_gain:.4f}',
                f'gc, {len(gc_low)}, md={gc_med_gain:.4f}, p={gc_LN_p:.4f}'])
                #, f'stp, {len(stp_low)}',
                #f'both, {len(both_low)}'])

#    fig2 = plt.figure(figsize=figsize)
#    plt.bar(axis_locs, prop_hist, width=bar_width, color='gray', alpha=0.8)
#    plt.bar(axis_locs, prop_gc_hist, width=bar_width, bottom=prop_hist,
#            color='maroon', alpha=0.8)
##    plt.bar(axis_locs, prop_stp_hist, width=bar_width,
##            bottom=prop_hist+prop_gc_hist, color='teal', alpha=0.8)
##    plt.bar(axis_locs, prop_both_hist, width=bar_width,
##            bottom=prop_hist+prop_gc_hist+prop_stp_hist, color='goldenrod',
##            alpha=0.8)
#    plt.xlabel('gain slope')
#    plt.ylabel('proportion within category')
#    plt.title(f'proportions, LN > {good_LN}')
#    plt.legend([f'LN, {len(low)}', f'gc, {len(gc_low)}'])#, f'stp, {len(stp_low)}',
#                #f'both, {len(both_low)}'])

    # Not really seeing anything from these so far, so commenting out for now.
#    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
#
#    r1, p1 = st.pearsonr(high-low, r_test)
#    ax1.scatter(gain, r_test)
#    ax1.set_xlabel('gain slope')
#    ax1.set_ylabel('r_test')
#    ax1.set_title(f'LN, r={r1:.2f}, p={p1:.2f}')
#
#    r2, p2 = st.pearsonr(gc_gain, gc_r)
#    ax2.scatter(gc_high-gc_low, gc_r)
#    ax2.set_xlabel('gain slope')
#    ax2.set_ylabel('r_test')
#    ax2.set_title(f'gc, r={r2:.2f}, p={p2:.2f}')
#
#    r3, p3 = st.pearsonr(stp_gain, stp_r)
#    ax3.scatter(stp_high-stp_low, stp_r)
#    ax3.set_xlabel('gain slope')
#    ax3.set_ylabel('r_test')
#    ax3.set_title(f'stp, r={r3:.2f}, p={p3:.2f}')
#
#    r4, p4 = st.pearsonr(both_gain, both_r)
#    ax4.scatter(both_high-both_low, both_r)
#    ax4.set_xlabel('gain slope')
#    ax4.set_ylabel('r_test')
#    ax4.set_title(f'both, r={r4:.2f}, p={p4:.2f}')
#
#    fig3.suptitle('gain slope vs r_test')
#    fig3.tight_layout()


def dynamic_sigmoid_range(cellid, batch, modelname, plot=True,
                          show_min_max=True):

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    modelspec = ctx['modelspec']
    modelspec.recording = ctx['val']
    val = ctx['val'].apply_mask()
    ctpred = val['ctpred'].as_continuous().flatten()
    val_before_dsig = ms.evaluate(val, modelspec, stop=-1)
    pred_before_dsig = val_before_dsig['pred'].as_continuous().flatten()
    lows = {k: v for k, v in modelspec[-1]['phi'].items()
            if '_mod' not in k}
    highs = {k[:-4]: v for k, v in modelspec[-1]['phi'].items()
             if '_mod' in k}
    for k in lows:
        if k not in highs:
            highs[k] = lows[k]
    # re-sort keys to make sure they're in the same order
    lows = {k: lows[k] for k in sorted(lows)}
    highs = {k: highs[k] for k in sorted(highs)}

    ctmax_val = np.max(ctpred)
    ctmin_val = np.min(ctpred)
    ctmax_idx = np.argmax(ctpred)
    ctmin_idx = np.argmin(ctpred)

    thetas = list(lows.values())
    theta_mods = list(highs.values())
    for t, t_m, k in zip(thetas, theta_mods, list(lows.keys())):
        lows[k] = t + (t_m - t)*ctmin_val
        highs[k] = t + (t_m - t)*ctmax_val

    low_out = _double_exponential(pred_before_dsig, **lows).flatten()
    high_out = _double_exponential(pred_before_dsig, **highs).flatten()

    if plot:
        fig = plt.figure(figsize=figsize)
        plt.scatter(pred_before_dsig, low_out, color='blue', s=0.7, alpha=0.6)
        plt.scatter(pred_before_dsig, high_out, color='red', s=0.7, alpha=0.6)

        if show_min_max:
            max_pred = pred_before_dsig[ctmax_idx]
            min_pred = pred_before_dsig[ctmin_idx]
            plt.scatter(min_pred, low_out[ctmin_idx], facecolors=None,
                        edgecolors='blue', s=60)
            plt.scatter(max_pred, high_out[ctmax_idx], facecolors=None,
                        edgecolors='red', s=60)

    return pred_before_dsig, low_out, high_out


def dynamic_sigmoid_distribution(cellid, batch, modelname, sample_every=10,
                                 alpha=0.1):

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    modelspec = ctx['modelspec']
    val = ctx['val'].apply_mask()
    modelspec.recording = val
    val_before_dsig = ms.evaluate(val, modelspec, stop=-1)
    pred_before_dsig = val_before_dsig['pred'].as_continuous().T
    ctpred = val_before_dsig['ctpred'].as_continuous().T

    lows = {k: v for k, v in modelspec[-1]['phi'].items()
            if '_mod' not in k}
    highs = {k[:-4]: v for k, v in modelspec[-1]['phi'].items()
             if '_mod' in k}
    for k in lows:
        if k not in highs:
            highs[k] = lows[k]
    # re-sort keys to make sure they're in the same order
    lows = {k: lows[k] for k in sorted(lows)}
    highs = {k: highs[k] for k in sorted(highs)}
    thetas = list(lows.values())
    theta_mods = list(highs.values())

    fig = plt.figure(figsize=figsize)
    for i in range(int(len(pred_before_dsig)/sample_every)):
        try:
            ts = {}
            for t, t_m, k in zip(thetas, theta_mods, list(lows.keys())):
                ts[k] = t + (t_m - t)*ctpred[i*sample_every]
            y = _double_exponential(pred_before_dsig, **ts)
            plt.scatter(pred_before_dsig, y, color='black', alpha=alpha,
                       s=0.01)
        except IndexError:
            # Will happen on last attempt if array wasn't evenly divisible
            # by sample_every
            pass
    t_max = {}
    t_min = {}
    for t, t_m, k in zip(thetas, theta_mods, list(lows.keys())):
        t_max[k] = t + (t_m - t)*np.nanmax(ctpred)
        t_min[k] = t + (t_m - t)*np.nanmin(ctpred)
    max_out = _double_exponential(pred_before_dsig, **t_max)
    min_out = _double_exponential(pred_before_dsig, **t_min)
    plt.scatter(pred_before_dsig, max_out, color='red', s=0.1)
    plt.scatter(pred_before_dsig, min_out, color='blue', s=0.1)


def dynamic_sigmoid_differences(batch, modelname, bins=60, test_limit=None,
                                save_path=None, load_path=None,
                                use_quartiles=False):

    if load_path is None:
        cellids = nd.get_batch_cells(batch, as_list=True)
        ratios = []
        for cellid in cellids[:test_limit]:
            xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
            val = ctx['val'].apply_mask()
            ctpred = val['ctpred'].as_continuous().flatten()
            pred_after = val['pred'].as_continuous().flatten()
            val_before = ms.evaluate(val, ctx['modelspec'], stop=-1)
            pred_before = val_before['pred'].as_continuous().flatten()
            median_ct = np.nanmedian(ctpred)
            if use_quartiles:
                low = np.percentile(ctpred, 25)
                high = np.percentile(ctpred, 75)
                low_mask = (ctpred >= low) & (ctpred < median_ct)
                high_mask = ctpred >= high
            else:
                low_mask = ctpred < median_ct
                high_mask = ctpred >= median_ct

            indices = np.argsort(pred_before)
            high_indices = indices[high_mask]
            high = pred_after[high_indices]
            low_indices = indices[low_mask]
            low = pred_after[low_indices]
            ratio = np.nanmean((low - high)/(np.abs(low) + np.abs(high)))
            ratios.append(ratio)

        ratios = np.array(ratios)
        if save_path is not None:
            np.save(save_path, ratios)
    else:
        ratios = np.load(load_path)

    plt.figure(figsize=figsize)
    plt.hist(ratios, bins=bins, color=[wsu_gray_light], edgecolor='black',
             linewidth=1)
    #plt.rc('text', usetex=True)
    #plt.xlabel(r'\texit{\frac{low-high}{\left|high\right|+\left|low\right|}}')
    plt.xlabel('(low - high)/(|low| + |high|)')
    plt.ylabel('cell count')
    plt.title("difference of low-contrast output and high-contrast output\n"
              "positive means low-contrast has higher firing rate on average")


def dynamic_sigmoid_pred_matched(cellid, batch, modelname):
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    modelspec = ctx['modelspec']
    modelspec.recording = ctx['val']
    val = ctx['val'].apply_mask()
    ctpred = val['ctpred'].as_continuous().flatten()
    # HACK
    # this really shouldn't happen.. but for some reason some  of the
    # batch 263 cells are getting nans, so temporary fix.
    ctpred[np.isnan(ctpred)] = 0
    pred_after_dsig = val['pred'].as_continuous().flatten()
    val_before_dsig = ms.evaluate(val, modelspec, stop=-1)
    pred_before_dsig = val_before_dsig['pred'].as_continuous().flatten()

    fig = plt.figure(figsize=figsize)
    plt.scatter(pred_before_dsig, pred_after_dsig, c=ctpred, s=2,
                alpha=0.75, cmap=plt.get_cmap('plasma'))
    plt.title(cellid)
    plt.xlabel('pred in')
    plt.ylabel('pred out')
    plt.colorbar()

    return fig


def save_pred_matched_batch(batch, modelname, save_path, test_limit=None):
    cells = nd.get_batch_cells(batch, as_list=True)
    for cellid in cells[:test_limit]:
        try:
            fig = dynamic_sigmoid_pred_matched(cellid, batch, modelname)
            full_path = os.path.join(save_path, str(batch), cellid)
            fig.savefig(full_path, format='pdf')
            plt.close(fig)
        except:
            # model probably not fit for that cell
            continue


def mean_prior_used(batch, modelname):
    choices = []
    cells = nd.get_batch_cells(batch, as_list=True)
    for i, c in enumerate(cells[400:500]):
        if 25 % (i+1) == 0:
            print('cell %d/%d\n' % (i, len(cells)))
        try:
            xfspec, ctx = xhelp.load_model_xform(c, batch, modelname,
                                                 eval_model=False)
            modelspec = ctx['modelspec']
            choices.append(modelspec.meta.get('best_random_idx', 0))
        except ValueError:
            # no result
            continue

    if choices:
        choices = np.array(choices).flatten()
        mean_count = np.sum(choices == 0)
        proportion = mean_count / len(choices)
        print('proportion mean prior used: %.4f' % proportion)
    else:
        print('no results found')


def random_condition_convergence(cellid, batch, modelname,
                                 separate_figures=True):
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    meta = ctx['modelspec'].meta
    rcs = meta['random_conditions']
    best_idx = meta['best_random_idx']
    keys = list(rcs[0][0].keys())

    if not separate_figures:
        plt.figure(figsize=figsize)
        colors = [np.random.rand(3,) for k in keys]
        for initial, final in rcs:
            starts = list(initial.values())
            ends = list(final.values())
            for i, k in enumerate(keys):
                plt.plot([0, 1], [np.asscalar(starts[i]), np.asscalar(ends[i])],
                         c=colors[i])
        plt.legend(keys)

    else:
        for k in keys:
            plt.figure(figsize=figsize)
            for i, (initial, final) in enumerate(rcs):
                start = initial[k]
                end = final[k]
                if i == 0:
                    color = 'blue'
                    label = 'mean'
                elif i == best_idx:
                    color = 'red'
                    label = 'best'
                else:
                    color = 'black'
                    label = None
                plt.plot([0, 1], np.concatenate((start,end)), color=color,
                         label=label)
            plt.legend()
            plt.title("%s, best_idx: %d" % (k, best_idx))
