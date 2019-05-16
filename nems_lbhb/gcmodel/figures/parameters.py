import numpy as np
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
