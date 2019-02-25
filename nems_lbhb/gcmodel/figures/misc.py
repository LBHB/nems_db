import matplotlib.pyplot as plt

import nems.xform_helper as xhelp
import nems.xforms as xf
from nems_db.params import fitted_params_per_batch
from .utils import adjustFigAspect


# TODO: Need to clean these up, some can probably be deleted.
#       Others haven't been updated in a while.


# Overlay of prediction from STP versus prediction from GC for sample cell(s)
def example_pred_overlay(cellid=good_cell, batch, model1, model2):
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
def mean_contrast_variables(modelname):

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


def continuous_contrast_improvements():
    df_full = fitted_params_per_batch(batch, gc_model_full, stats_keys=[])
    df_cont = fitted_params_per_batch(batch, gc_model_cont, stats_keys=[])
    df_stp = fitted_params_per_batch(batch, stp_model, stats_keys=[])
    df_ln = fitted_params_per_batch(batch, ln_model, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    nrows = len(df_full.index.values.tolist())

    df1_cells = df_full.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df_cont.loc['meta--r_test'].index.values.tolist()[5:]
    df3_cells = df_ln.loc['meta--r_test'].index.values.tolist()[5:]
    df4_cells = df_stp.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0
    df3_nans = 0
    df4_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df_full[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df_cont[c] = nan_series
            df2_nans += 1
        if c not in df3_cells:
            df_ln[c] = nan_series
            df3_nans += 1
        if c not in df4_cells:
            df_stp[c] = nan_series
            df4_nans += 1

    print("# missing cells: %d, %d, %d, %d" % (df1_nans, df2_nans, df3_nans,
                                               df4_nans))

    # Force same cellid order now that cols are filled in
    df_full = df_full[cellids]
    df_cont = df_cont[cellids]
    df_ln = df_ln[cellids]
    df_stp = df_stp[cellids]

    # Only look at cells that did better than linear for binary model
    full_vs_ln = df_full.loc['meta--r_test'].values - \
            df_ln.loc['meta--r_test'].values
    cont_vs_ln = df_cont.loc['meta--r_test'].values - \
            df_ln.loc['meta--r_test'].values
    full_vs_ln = full_vs_ln.astype('float32')
    cont_vs_ln = cont_vs_ln.astype('float32')

    better = full_vs_ln > 0
    #full_vs_ln = full_vs_ln[better]
    #cont_vs_ln = cont_vs_ln[better]

    # which cells got further improvement by keeping contrast continuous?
    cont_improve = (cont_vs_ln - full_vs_ln) > 0
    cont_vs_full = cont_vs_ln[cont_improve]

    # Keep indices so can extract cellid names
    cont_improve = (cont_vs_ln - full_vs_ln) > 0
    cont_better = np.logical_and(better, cont_improve)

    cont_cells = celldata['cellid'][cont_better].tolist()
    full_cells = celldata['cellid'][np.logical_not(cont_better)].tolist()

    df_full = df_full[full_cells]
    df_cont = df_cont[cont_cells]
    df_stp_full = df_stp[full_cells]
    df_stp_cont = df_stp[cont_cells]
    df_ln_full = df_ln[full_cells]
    df_ln_cont = df_ln[cont_cells]

    df_full_r = (df_full.loc['meta--r_test'].values
                 - df_ln_full.loc['meta--r_test']).astype('float32')
    df_cont_r = (df_cont.loc['meta--r_test'].values
                 - df_ln_cont.loc['meta--r_test']).astype('float32')
    df_stp_full_r = (df_stp_full.loc['meta--r_test'].values
                     - df_ln_full.loc['meta--r_test']).astype('float32')
    df_stp_cont_r = (df_stp_cont.loc['meta--r_test'].values
                     - df_ln_cont.loc['meta--r_test']).astype('float32')

    ff = np.isfinite(df_full_r) & np.isfinite(df_stp_full_r)
    df_full_r = df_full_r[ff]
    df_stp_full_r = df_stp_full_r[ff]
    ff = np.isfinite(df_cont_r) & np.isfinite(df_stp_cont_r)
    df_cont_r = df_cont_r[ff]
    df_stp_cont_r = df_stp_cont_r[ff]

    r1 = np.corrcoef(df_full_r, df_stp_full_r)[0, 1]
    r2 = np.corrcoef(df_cont_r, df_stp_cont_r)[0, 1]

    fig = plt.figure()
    plt.scatter(df_full_r, df_stp_full_r)
    plt.title('full, r: %.04f'%r1)
    adjustFigAspect(fig, aspect=1)

    fig = plt.figure()
    plt.scatter(df_cont_r, df_stp_cont_r)
    plt.title('cont, r: %.04f'%r2)
    adjustFigAspect(fig, aspect=1)


def gd_ratio(cellid=good_cell, modelname=gc_cont_full):

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    mspec = ctx['modelspec']
    dsig_idx = find_module('dynamic_sigmoid', mspec)
    phi = mspec[dsig_idx]['phi']

    return phi['kappa_mod']/phi['kappa']


def gd_scatter(batch=289, model1=gc_cont_full, model2=ln_model):

    df1 = fitted_params_per_batch(batch, model1, stats_keys=[])
    df2 = fitted_params_per_batch(batch, model2, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
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

    gc_vs_ln = df1.loc['meta--r_test'].values - df2.loc['meta--r_test'].values
    gc_vs_ln = gc_vs_ln.astype('float32')

    kappa_mod = df1[df1.index.str.contains('kappa_mod')]
    kappa = df1[df1.index.str.contains('kappa$')]
    gd_ratio = (kappa_mod.values / kappa.values).astype('float32').flatten()


    # For testing: Some times kappa is so small that the ratio ends up
    # throwing the scale off so far that the plot is worthless.
    # But majority of the time the ratio is less than 5ish, so try rectifying:
    gd_ratio[gd_ratio > 5] = 5
    gd_ratio[gd_ratio < -5] = -5
    # Then normalize to -1 to 1 scale for easier comparison to r value
    gd_ratio /= 5


    ff = np.isfinite(gc_vs_ln) & np.isfinite(gd_ratio)
    gc_vs_ln = gc_vs_ln[ff]
    gd_ratio = gd_ratio[ff]

    r = np.corrcoef(gc_vs_ln, gd_ratio)[0, 1]
    n = gc_vs_ln.size

    y_max = np.max(gd_ratio)
    y_min = np.min(gd_ratio)
    x_max = np.max(gc_vs_ln)
    x_min = np.min(gc_vs_ln)

    abs_max = max(np.abs(y_max), np.abs(x_max), np.abs(y_min), np.abs(x_min))
    abs_max *= 1.15

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(gc_vs_ln, gd_ratio)
    plt.xlabel("GC - LN model")
    plt.ylabel("Gd ratio")
    plt.title("r: %.02f, n: %d" % (r, n))
    gca = plt.gca()
    gca.axes.axhline(0, color='black', linewidth=1, linestyle='dashed')
    gca.axes.axvline(0, color='black', linewidth=1, linestyle='dashed')
    plt.ylim(ymin=(-1)*abs_max, ymax=abs_max)
    plt.xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)
