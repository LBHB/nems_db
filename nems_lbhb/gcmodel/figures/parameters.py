import logging
import os

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import nems.xform_helper as xhelp
import nems.db as nd
import nems.epoch as ep
from nems.utils import find_module
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             get_valid_improvements,
                                             adjustFigAspect,
                                             improved_cells_to_list,
                                             is_outlier)
from nems_lbhb.gcmodel.figures.respstats import _binned_xvar, _binned_yavg
from nems.metrics.stp import stp_magnitude
from nems_lbhb.gcmodel.magnitude import gc_magnitude
from nems_lbhb.gcmodel.figures.soundstats import silence_duration
from nems_db.params import fitted_params_per_batch
import nems.modelspec as ms
from nems.modules.nonlinearity import _double_exponential, _saturated_rectifier
from nems_lbhb.gcmodel.figures.definitions import *

log = logging.getLogger(__name__)

plt.rcParams.update(params)  # loaded from definitions


def stp_distributions(batch, gc, stp, LN, combined, se_filter=True,
                      good_ln=0, log_scale=False):

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter=se_filter,
                                                          as_lists=False)
    _, _, _, _, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           good_ln=good_ln)

    stp_params = fitted_params_per_batch(289, stp, stats_keys=[], meta=[])
    stp_params_cells = stp_params.transpose().index.values.tolist()
    for cell in stp_params_cells:
        if cell not in cellids:
            cellids[cell] = False
    not_c = list(set(stp_params.transpose()[cellids].index.values) - set(c))

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
    #sep_taus = _df_to_array(all_taus, dims).mean(axis=0)
    #sep_us = _df_to_array(all_us, dims).mean(axis=0)
    #med_tau = np.median(sep_taus)
    #med_u = np.median(sep_u)
    sep_taus = _df_to_array(all_taus[not_c], dims).mean(axis=0)
    sep_us = _df_to_array(all_us[not_c], dims).mean(axis=0)
    sep_taus = sep_taus[~is_outlier(sep_taus)]
    sep_us = sep_us[~is_outlier(sep_us)]
    med_tau = np.median(sep_taus)
    med_u = np.median(sep_us)

    stp_taus = all_taus[c]
    stp_us = all_us[c]
    stp_sep_taus = _df_to_array(stp_taus, dims).mean(axis=0)
    stp_sep_us = _df_to_array(stp_us, dims).mean(axis=0)
    stp_sep_taus = stp_sep_taus[~is_outlier(stp_sep_taus)]
    stp_sep_us = stp_sep_us[~is_outlier(stp_sep_us)]
    stp_med_tau = np.median(stp_sep_taus)
    stp_med_u = np.median(stp_sep_us)

    tau_t, tau_p = st.ttest_ind(sep_taus, stp_sep_taus)
    u_t, u_p = st.ttest_ind(sep_us, stp_sep_us)


    fig, (a1, a2) = plt.subplots(2, 1)
    color = model_colors['LN']
    imp_color = model_colors['max']
    stp_label = 'STP ++ (%d)' % len(c)
    total_cells = len(c) + len(not_c)
    bin_count = 30
    hist_kwargs = {'bins': bin_count, 'linewidth': 1, 'color': [color, imp_color],
                   'label': ['not imp', 'stp imp']}


    plt.sca(a1)
    weights = [np.ones(len(sep_taus))/len(sep_taus),
               np.ones(len(stp_sep_taus))/len(stp_sep_taus)]
    if log_scale:
        lower_bound = min(sep_taus.min(), stp_sep_taus.min())
        upper_bound = max(sep_taus.max(), stp_sep_taus.max())
        bins = np.logspace(lower_bound, upper_bound, bin_count+1)
        hist_kwargs['bins'] = bins
    plt.hist([sep_taus, stp_sep_taus], weights=weights, **hist_kwargs)
#    plt.hist(sep_taus, color=color, **hist_kwargs, alpha=0.6)
#    plt.hist(stp_sep_taus, color=stp_color, label=stp_label,
#             **hist_kwargs, alpha=0.6)
    a1.axes.axvline(med_tau, color=color, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    a1.axes.axvline(stp_med_tau, color=imp_color, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    plt.title('tau,  sig diff?:  p=%.4E' % tau_p)
    plt.xlabel('tau (ms)')
    plt.legend()


    plt.sca(a2)
    weights = [np.ones(len(sep_us))/len(sep_us),
               np.ones(len(stp_sep_us))/len(stp_sep_us)]
    if log_scale:
        lower_bound = min(sep_us.min(), stp_sep_us.min())
        upper_bound = max(sep_us.max(), stp_sep_us.max())
        bins = np.logspace(lower_bound, upper_bound, bin_count+1)
        hist_kwargs['bins'] = bins
    plt.hist([sep_us, stp_sep_us], weights=weights, **hist_kwargs)
#    plt.hist(sep_us, color=color, **hist_kwargs, alpha=0.6)
#    plt.hist(stp_sep_us, color=stp_color,
#             label='STP ++', **hist_kwargs, alpha=0.6)
    a2.axes.axvline(med_u, color=color, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    a2.axes.axvline(stp_med_u, color=imp_color, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    plt.title('u,  sig diff?:  p=%.4E' % u_p)
    plt.xlabel('u (fractional change in gain \nper unit of stimulus amplitude)')
    plt.ylabel('proportion within group')


    fig.suptitle("STP parameter distributions,  n: %d\n"
                 "n stp imp:  %d\n"
                 "n not imp:  %d\n"
                 % (total_cells, len(c), len(not_c)))
    return fig


def gc_distributions(batch, gc, stp, LN, combined, se_filter=True, good_ln=0,
                     log_scale=False):
    if log_scale:
        raise NotImplementedError('log scale not implemented yet for gc dist')

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          as_lists=False)
    _, _, _, _, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           good_ln=good_ln)

    gc_params = fitted_params_per_batch(289, gc, stats_keys=[], meta=[])
    gc_params_cells = gc_params.transpose().index.values.tolist()
    for cell in gc_params_cells:
        if cell not in cellids:
            cellids[cell] = False
    not_c = list(set(gc_params.transpose()[cellids].index.values) - set(c))

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
    all_keys = [b_key, ba_key, a_key, aa_key, s_key, sa_key, k_key, ka_key]

    phi_dfs = [gc_params[gc_params.index==k].transpose()[cellids].transpose()
               for k in all_keys]
    sep_dfs = [df[not_c].values.flatten().astype(np.float64) for df in phi_dfs]
    gc_sep_dfs = [df[c].values.flatten().astype(np.float64)
                  for df in phi_dfs]

    # removing extreme outliers b/c kept getting one or two cells with
    # values that were multiple orders of magnitude different than all others
    diffs = [sep_dfs[i+1] - sep_dfs[i]
             for i, _ in enumerate(sep_dfs[:-1])
             if i % 2 == 0]
    diffs = [d[~is_outlier(d)] for d in diffs]
    gc_diffs = [gc_sep_dfs[i+1] - gc_sep_dfs[i]
                for i, _ in enumerate(gc_sep_dfs[:-1])
                if i % 2 == 0]
    gc_diffs = [d[~is_outlier(d)] for d in gc_diffs]
    medians = [np.median(d) for d in diffs]
    gc_medians = [np.median(d) for d in gc_diffs]


    ts, ps = zip(*[st.ttest_ind(diff, gc_diff)
                   for diff, gc_diff in zip(diffs, gc_diffs)])

    fig1, (a1, a3) = plt.subplots(2, 1)
    fig2, (a2, a4) = plt.subplots(2, 1)
    color = model_colors['LN']
    c_color = model_colors['max']
    gc_label = 'GC ++ (%d)' % len(c)
    total_cells = len(c) + len(not_c)
    hist_kwargs = {'bins': 30, 'color': [color, c_color],
                   'label': ['no imp', 'sig imp']}

    # b a s k
    for i, ax in zip([0, 1, 2, 3], [a1, a3, a2, a4]):
#        ax.hist(diffs[i], color=color, **hist_kwargs, alpha=0.6)
#        ax.hist(gc_diffs[i], color=gc_color, label=gc_label, **hist_kwargs,
#                alpha=0.6)
        weights = [np.ones(len(diffs[i]))/len(diffs[i]),
                   np.ones(len(gc_diffs[i]))/len(gc_diffs[i])]
        ax.hist([diffs[i], gc_diffs[i]], weights=weights, **hist_kwargs)
        ax.axes.axvline(medians[i], color=color, linewidth=2,
                        linestyle='dashed', dashes=dash_spacing)
        ax.axes.axvline(gc_medians[i], color=c_color, linewidth=2,
                        linestyle='dashed', dashes=dash_spacing)
        if (i == 0) or (i == 2):
            ax.legend()

    a1.set_title('base,   sig diff?  %.4E' % ps[0])
    a2.set_title('amplitude,   sig diff?  %.4E' % ps[1])
    a3.set_title('shift,   sig diff?  %.4E' % ps[2])
    a3.set_xlabel('fractional change in parameter\nper unit contrast')
    a3.set_ylabel('proportion within group')
    a4.set_title('kappa,   sig diff?  %.4E' % ps[3])

    title = ("GC parameter differences, total n: %d\n"
             "n gc improved:  %d\n"
             "n not improved: %d\n"
             % (total_cells, len(c), len(not_c)))
    fig1.suptitle(title)
    fig2.suptitle(title)
    return fig1, fig2


def _df_to_array(df, dims):
    vals = df.values.flatten()
    array = np.array([np.array([vals[j][k] for j, v in enumerate(vals)])
                      for k in range(dims)])
    return array


def gd_ratio(batch, gc, stp, LN, combined, se_filter=True, good_LN=0, bins=30,
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

    fig1, ((ax1), (ax2)) = plt.subplots(1, 2, )
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


    fig3 = plt.figure()
    plt.scatter(ratio, r_test)
    plt.title('low/high vs r_test')

    fig4 = plt.figure()
    plt.scatter(gc_ratio, gc_r)
    plt.title('low/high vs r_test, gc improvements only')


def gain_by_contrast_slopes(batch, gc, stp, LN, combined, se_filter=True,
                            good_LN=0, bins=30, use_exp=True):

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

#    fig = plt.figure()#, axes = plt.subplots(1, 2, )
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


    fig1 = plt.figure()
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

#    fig2 = plt.figure()
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
#    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, )
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
        fig = plt.figure()
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

    fig = plt.figure()
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


def dynamic_sigmoid_differences(batch, modelname, hist_bins=60, test_limit=None,
                                save_path=None, load_path=None,
                                use_quartiles=False, avg_bin_count=20):

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

            # TODO: do some kind of binning here since the two vectors
            # don't actually overlap in x axis
            mean_before, bin_masks = _binned_xvar(pred_before, avg_bin_count)
            low = _binned_yavg(pred_after, low_mask, bin_masks)
            high = _binned_yavg(pred_after, high_mask, bin_masks)

            ratio = np.nanmean((low - high)/(np.abs(low) + np.abs(high)))
            ratios.append(ratio)

        ratios = np.array(ratios)
        if save_path is not None:
            np.save(save_path, ratios)
    else:
        ratios = np.load(load_path)

    plt.figure()
    plt.hist(ratios, bins=hist_bins, color=[wsu_gray_light], edgecolor='black',
             linewidth=1)
    #plt.rc('text', usetex=True)
    #plt.xlabel(r'\texit{\frac{low-high}{\left|high\right|+\left|low\right|}}')
    plt.xlabel('(low - high)/(|low| + |high|)')
    plt.ylabel('cell count')
    plt.title("difference of low-contrast output and high-contrast output\n"
              "positive means low-contrast has higher firing rate on average")


def dynamic_sigmoid_pred_matched(cellid, batch, modelname, include_phi=True):
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

    fig = plt.figure(figsize=(12, 7))
    plasma = plt.get_cmap('plasma')
    plt.scatter(pred_before_dsig, pred_after_dsig, c=ctpred, s=2,
                alpha=0.75, cmap=plasma)
    plt.title(cellid)
    plt.xlabel('pred in')
    plt.ylabel('pred out')

    if include_phi:
        dsig_phi = modelspec.phi[-1]
        phi_string = '\n'.join(['%s:  %.4E' % (k, v) for k, v in dsig_phi.items()])
        thetas = list(dsig_phi.keys())[0:-1:2]
        mods = list(dsig_phi.keys())[1::2]
        weights = {k: (dsig_phi[mods[i]] - dsig_phi[thetas[i]])
                   for i, k in enumerate(thetas)}
        weights_string = 'weights:\n' + '\n'.join(['%s:  %.4E' % (k, v)
                                                   for k, v in weights.items()])
        fig.text(0.775, 0.9, phi_string, va='top', ha='left')
        fig.text(0.775, 0.1, weights_string, va='bottom', ha='left')
        plt.subplots_adjust(right=0.775, left=0.075)

    plt.colorbar()
    return fig

# if going back to discrete colors for pred_matched sigmoid, can use this:
#        plasma = plt.get_cmap('plasma')
#        c1, c2, c3, c4 = [plasma(n) for n in [.1, .4, .7, .9]]


def stp_sigmoid_pred_matched(cellid, batch, modelname, LN, include_phi=True):
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    ln_spec, ln_ctx = xhelp.load_model_xform(cellid, batch, LN)
    modelspec = ctx['modelspec']
    modelspec.recording = ctx['val']
    val = ctx['val'].apply_mask()
    ln_modelspec = ln_ctx['modelspec']
    ln_modelspec.recording = ln_ctx['val']
    ln_val = ln_ctx['val'].apply_mask()

    pred_after_NL = val['pred'].as_continuous().flatten() # with stp
    val_before_NL = ms.evaluate(ln_val, ln_modelspec, stop=-1)
    pred_before_NL = val_before_NL['pred'].as_continuous().flatten() # no stp

    stp_idx = find_module('stp', modelspec)
    val_before_stp = ms.evaluate(val, modelspec, stop=stp_idx)
    val_after_stp = ms.evaluate(val, modelspec, stop=stp_idx+1)
    pred_before_stp = val_before_stp['pred'].as_continuous().mean(axis=0).flatten()
    pred_after_stp = val_after_stp['pred'].as_continuous().mean(axis=0).flatten()
    stp_effect = (pred_after_stp - pred_before_stp)/(pred_after_stp + pred_before_stp)

    fig = plt.figure()
    plasma = plt.get_cmap('plasma')
    plt.scatter(pred_before_NL, pred_after_NL, c=stp_effect, s=2,
                alpha=0.75, cmap=plasma)
    plt.title(cellid)
    plt.xlabel('pred in (no stp)')
    plt.ylabel('pred out (with stp)')

    if include_phi:
        stp_phi = modelspec.phi[stp_idx]
        phi_string = '\n'.join(['%s:  %.4E' % (k, v) for k, v in stp_phi.items()])
        fig.text(0.775, 0.9, phi_string, va='top', ha='left')
        plt.subplots_adjust(right=0.775, left=0.075)
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


def filtered_pred_matched_batch(batch, gc, stp, LN, combined, save_path,
                                good_ln=0.0, test_limit=None, stat='r_ceiling',
                                replace_existing=True):

    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           good_ln=good_ln)
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    if stat == 'r_ceiling':
        df = df_c
    else:
        df = df_r

    tags = ['either', 'neither', 'gc', 'stp', 'combined']
    a = list(set(a) - set(e))
    for cells, tag in zip([e, a, g, s, c], tags):
        _sigmoid_sub_batch(cells[:test_limit], df, tag, stat, batch, gc, stp, LN,
                           combined, save_path,
                           replace_existing=replace_existing)


def _sigmoid_sub_batch(cells, df, tag, stat, batch, gc, stp, LN, combined,
                       save_path, replace_existing=True):
    for cellid in cells:
        full_path = os.path.join(save_path, str(batch), tag, cellid)
        if not replace_existing:
            if os.path.exists(full_path):
                print('skipping existing result for:   %s' % cellid)
                continue

        try:
            fig = dynamic_sigmoid_pred_matched(cellid, batch, gc)
            gc_r = df[gc][cellid]
            stp_r = df[stp][cellid]
            LN_r = df[LN][cellid]
            combined_r = df[combined][cellid]
        except:
            # model probably not fit for that cell
            continue

        parent_directory = '/'.join(full_path.split('/')[:-1])
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory, mode=0o777)
        fig.suptitle("model performances, %s:\n"
                     "gc: %.4f  |stp: %.4f  |LN: %.4f  |comb.: %.4f"
                     % (stat, gc_r, stp_r, LN_r, combined_r))
        fig.savefig(full_path, format='pdf', dpi=fig.dpi)
        plt.close(fig)


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
        plt.figure()
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
            plt.figure()
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
