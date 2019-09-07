import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

import nems.db as nd
from nems_db.params import fitted_params_per_batch
from nems_lbhb.gcmodel.figures.utils import improved_cells_to_list, is_outlier
from nems_lbhb.gcmodel.figures.parameters import _df_to_array
from nems_lbhb.gcmodel.figures.definitions import *


plt.rcParams.update(params)


def per_cell_group(batch, gc, stp, LN, combined, equivalence_path,
                   drop_outliers=True):
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           as_lists=False)
    # 4 groups: neither improve, both improve, gc only, stp only
    neither = a & np.logical_not(e)
    both = g & s
    gc_only = g & np.logical_not(s)
    stp_only = s & np.logical_not(g)

    args = (batch, gc, stp, LN, combined, equivalence_path, drop_outliers)
    f1, f2 = kitchen_sink(*args, cell_mask=neither, mask_name='neither')
    f3, f4 = kitchen_sink(*args, cell_mask=both, mask_name='both')
    f5, f6 = kitchen_sink(*args, cell_mask=gc_only, mask_name='gc')
    f7, f8 = kitchen_sink(*args, cell_mask=stp_only, mask_name='stp')
    f9, f10 = kitchen_sink(*args, cell_mask=None, mask_name='all')

    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10


def kitchen_sink(batch, gc, stp, LN, combined, equivalence_path,
                 drop_outliers=True, cell_mask=None, mask_name=''):
    # 0.  Get auditory-responsive cells
    _, a, _, _, _ = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           as_lists=False)
    a_list = a[a == True].index.values.tolist()

    # 1.  load batch parameters (shouldn't need to load models)
    stp_params = fitted_params_per_batch(289, stp, stats_keys=[],
                                         meta=['r_test'], manual_cellids=a_list)
    gc_params = fitted_params_per_batch(289, gc, stats_keys=[], meta=['r_test'],
                                        manual_cellids=a_list)
    LN_params = fitted_params_per_batch(289, LN, stats_keys=[], meta=['r_test'],
                                        manual_cellids=a_list)


    df = pd.read_pickle(equivalence_path)
    equivalence = df.sort_index()['equivalence'].values
#    for c in gc_params_cells:
#        if c not in LN_cells:
#            LN_cells[c] = False

    # assemble each attribute as a vector
    # index keys are formatted like "2--stp.2--tau"
    mod_keys = stp.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'stp' in k:
            break
    tau_key = '%d--%s--tau' % (i, k)
    u_key = '%d--%s--u' % (i, k)

    mod_keys = gc.split('_')[1]
    for i, k in enumerate(mod_keys.split('-')):
        if 'dsig' in k:
            break
    b_key = f'{i}--{k}--base'
    a_key = f'{i}--{k}--amplitude'
    s_key = f'{i}--{k}--shift'
    k_key = f'{i}--{k}--kappa'
    ba_key = b_key + '_mod'
    aa_key = a_key + '_mod'
    sa_key = s_key + '_mod'
    ka_key = k_key + '_mod'

    stp_keys = [tau_key, u_key]
    gc_keys = [b_key, a_key, s_key, k_key, ba_key, aa_key, sa_key,
               ka_key]
    stp_dfs = [stp_params[stp_params.index==k].transpose().sort_index()[a]
               for k in stp_keys]
    gc_dfs = [gc_params[gc_params.index==k].transpose().sort_index()[a]\
              .astype(np.float64).values.flatten()
              for k in gc_keys]
    r_dfs = [df[df.index=='meta--r_test'].transpose().sort_index()[a]
             for df in [gc_params, stp_params, LN_params]]

    diffs = [gc_dfs[i+1] - gc_dfs[i]
             for i, _ in enumerate(gc_dfs[:-1])
             if i % 2 == 0]
    for i, k in enumerate(gc_keys):
        if '_mod' in k:
            gc_keys[i] = k[:-3] + 'diff'
    #gc_dfs = gc_dfs[:4] + diffs
    gc_dfs = diffs
    gc_keys = gc_keys[4:]


    dims = 3
    gc_vs_LN = (r_dfs[0] - r_dfs[2]).values.astype(np.float64).flatten()
    stp_vs_LN = (r_dfs[1] - r_dfs[2]).values.astype(np.float64).flatten()
    to_corr = [gc_vs_LN, stp_vs_LN, equivalence]
    to_corr.extend([df for df in gc_dfs])
    to_corr.extend([_df_to_array(df, dims).mean(axis=0) for df in stp_dfs])

    replace = []
    if cell_mask is not None:
        for v in to_corr:
            replace.append(v[cell_mask])
        to_corr = replace

    replace = []
    if drop_outliers:
        # drop any cells that are an outlier for at least one of the variables
        out = np.zeros_like(to_corr[0], dtype='bool')
        for v in to_corr:
            out = out | is_outlier(v)
        for v in to_corr:
            replace.append(v[~out])
        to_corr = replace

    n_cells = len(to_corr[0])

    matrix = np.vstack(to_corr)
    labels = ['gc_vs_LN', 'stp_vs_LN', 'equivalence']
    for k in gc_keys + stp_keys:
        labels.append(k.split('-')[-1])

    corr = np.corrcoef(matrix)
    fig1, ax = plt.subplots()
    plt.imshow(corr)
    plt.colorbar()
    ax.set_xticks(np.arange(len(labels)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fig1.suptitle("Correlations,  mask:%s\n"
                  "n: %d\n"
                  "outliers dropped?:  %s"
                  % (mask_name, n_cells, drop_outliers))

    for i in range(len(corr)):
        for j in range(len(corr)):
            v = str('%.3f'%corr[i, j])
            ax.text(j, i, v, ha='center', va='center', color='w')

    ps = np.empty_like(corr)
    p_correction = ps.shape[0]  # do a bonferroni correction since it's easy
    for i in range(len(ps)):
        for j in range(len(ps)):
            r, p = st.pearsonr(matrix[i], matrix[j])
            ps[i][j] = p*p_correction

    fig2, ax = plt.subplots()
    plt.imshow(ps)
    plt.colorbar()
    ax.set_xticks(np.arange(len(labels)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(corr)):
        for j in range(len(corr)):
            v = str('%.1E'%ps[i, j])
            ax.text(j, i, v, size=12,
                    ha='center', va='center', color='w')
    fig2.suptitle("P-values * %d (bonferroni correction)\n"
                  "mask:%s\n"
                  "n: %d\n"
                  "outliers dropped?:  %s"
                  % (p_correction, mask_name, n_cells, drop_outliers))

    return fig1, fig2
