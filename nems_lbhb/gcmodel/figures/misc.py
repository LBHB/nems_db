import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import nems.db as nd
import nems.xform_helper as xhelp
import nems.xforms as xf
from nems_db.params import fitted_params_per_batch
from nems_lbhb.gcmodel.figures.utils import adjustFigAspect, forceAspect
from nems.utils import find_module


# TODO: Need to clean these up, some can probably be deleted.
#       Others haven't been updated in a while.


# Overlay of prediction from STP versus prediction from GC for sample cell(s)
def example_pred_overlay(cellid, batch, model1, model2):
    '''
    model1: GC
    model2: STP

    '''
    xfspec1, ctx1 = xhelp.load_model_xform(cellid, batch, model1)
    xfspec2, ctx2 = xhelp.load_model_xform(cellid, batch, model2)
    plt.figure()
    #xf.plot_timeseries(ctx1, 'resp', cutoff=500)
    xf.plot_timeseries(ctx1, 'pred', cutoff=(200, 500))
    xf.plot_timeseries(ctx2, 'pred', cutoff=(200, 500))
    plt.legend([#'resp',
                'gc',
                'stp'])


# Average values for fitted contrast parameters, to compare to paper
# -- use param extraction functions
# Scatter of full model versus ".k.s" model
def mean_contrast_variables(batch, modelname):

    df1 = fitted_params_per_batch(batch, modelname, mod_key='fn')

    amplitude_mods = df1[df1.index.str.contains('amplitude_mod')]
    base_mods = df1[df1.index.str.contains('base_mod')]
    kappa_mods = df1[df1.index.str.contains('kappa_mod')]
    shift_mods = df1[df1.index.str.contains('shift_mod')]

    avg_amp = amplitude_mods['mean'][0]
    avg_base = base_mods['mean'][0]
    avg_kappa = kappa_mods['mean'][0]
    avg_shift = shift_mods['mean'][0]

    max_amp = amplitude_mods['max'][0]
    max_base = base_mods['max'][0]
    max_kappa = kappa_mods['max'][0]
    max_shift = shift_mods['max'][0]

#    raw_amp = amplitude_mods.values[0][5:]
#    raw_base = base_mods.values[0][5:]
#    raw_kappa = kappa_mods.values[0][5:]
#    raw_shift = shift_mods.values[0][5:]

    print("Mean amplitude_mod: %.06f\n"
          "Mean base_mod: %.06f\n"
          "Mean kappa_mod: %.06f\n"
          "Mean shift_mod: %.06f\n" % (
                  avg_amp, avg_base, avg_kappa, avg_shift
                  ))

    # Better way to tell which ones are being modulated?
    # Can't really tell just from the average.
    print("ratio of max: %.06f, %.06f, %.06f, %.06f" % (
            avg_amp/max_amp, avg_base/max_base,
            avg_kappa/max_kappa, avg_shift/max_shift))


def gd_ratio(cellid, batch, modelname):

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname,
                                         eval_model=False)
    mspec = ctx['modelspec']
    dsig_idx = find_module('dynamic_sigmoid', mspec)
    phi = mspec[dsig_idx]['phi']

    return phi['kappa_mod']/phi['kappa']


def gd_scatter(batch, model1, model2, se_filter=True, gd_threshold=0,
               param='kappa', log_gd=False):

    df_r = nd.batch_comp(batch, [model1, model2],
                         stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2],
                         stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    gc_test = df_r[model1]
    gc_se = df_e[model1]
    ln_test = df_r[model2]
    ln_se = df_e[model2]


    if se_filter:
        # Remove if performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (ln_test > ln_se*2))
    else:
        # Set to series w/ all True, so none are skipped
        good_cells = (gc_test != np.nan)

    df1 = fitted_params_per_batch(batch, model1, stats_keys=[])
    df2 = fitted_params_per_batch(batch, model2, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    cellids = [c for c in cellids if c in good_cells]
    nrows = len(df1.index.values.tolist())

    df1_cells = df1.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df2.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df1[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df2[c] = nan_series
            df2_nans += 1

    print("# missing cells: %d, %d" % (df1_nans, df2_nans))

    # Force same cellid order now that missing cols are filled in
    df1 = df1[cellids]; df2 = df2[cellids];

    gc_vs_ln = df1.loc['meta--r_test'].values / df2.loc['meta--r_test'].values
    gc_vs_ln = gc_vs_ln.astype('float32')

    kappa_mod = df1[df1.index.str.contains('%s_mod'%param)]
    kappa = df1[df1.index.str.contains('%s$'%param)]
    gd_ratio = (np.abs(kappa_mod.values / kappa.values)).astype('float32').flatten()

    ff = np.isfinite(gc_vs_ln) & np.isfinite(gd_ratio)
    gc_vs_ln = gc_vs_ln[ff]
    gd_ratio = gd_ratio[ff]
    if log_gd:
        gd_ratio = np.log(gd_ratio)

    # drop cells with excessively large/small gd_ratio or gc_vs_ln
    gcd_big = gd_ratio > 10
    gc_vs_ln_big = gc_vs_ln > 10
    gc_vs_ln_small = gc_vs_ln < 0.1
    keep = ~gcd_big & ~gc_vs_ln_big & ~gc_vs_ln_small
    gd_ratio = gd_ratio[keep]
    gc_vs_ln = gc_vs_ln[keep]

    r = np.corrcoef(gc_vs_ln, gd_ratio)[0, 1]
    n = gc_vs_ln.size


    # Separately do the same comparison but only with cells that had a
    # Gd ratio at least a little greater than 1 (i.e. had *some* GC effect)
    gd_ratio2 = copy.deepcopy(gd_ratio)
    gc_vs_ln2 = copy.deepcopy(gc_vs_ln)
    if log_gd:
        gd_threshold = np.log(gd_threshold)
    thresholded = (gd_ratio2 > gd_threshold)
    gd_ratio2 = gd_ratio2[thresholded]
    gc_vs_ln2 = gc_vs_ln2[thresholded]

    r2 = np.corrcoef(gc_vs_ln2, gd_ratio2)[0, 1]
    n2 = gc_vs_ln2.size

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 9))

    ax1.scatter(gd_ratio, gc_vs_ln, c='black', s=1)
    ax1.set_ylabel("GC/LN R")
    ax1.set_xlabel("Gd ratio")
    ax1.set_title("Performance Improvement vs Gd ratio\nr: %.02f, n: %d"
                  % (r, n))

    ax2.hist(gd_ratio, bins=30, histtype='bar', color=['gray'])
    ax2.set_title('Gd ratio distribution')
    ax2.set_xlabel('Gd ratio')
    ax2.set_ylabel('Count')

    ax3.scatter(gd_ratio2, gc_vs_ln2, c='black', s=1)
    ax3.set_ylabel("GC/LN R")
    ax3.set_xlabel("Gd ratio")
    ax3.set_title("Same, only cells w/ Gd > %.02f\nr: %.02f, n: %d"
                  % (gd_threshold, r2, n2))

    ax4.hist(gd_ratio2, bins=30, histtype='bar', color=['gray'])
    ax4.set_title('Gd ratio distribution, only Gd > %.02f' % gd_threshold)
    ax4.set_xlabel('Gd ratio')
    ax4.set_ylabel('Count')

    fig.suptitle('param: %s'%param)
    fig.tight_layout()
