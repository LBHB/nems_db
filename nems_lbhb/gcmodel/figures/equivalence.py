import numpy as np
import matplotlib.pyplot as plt

import nems.xform_helper as xhelp
import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import (get_valid_improvements,
                                             adjustFigAspect)
from nems.utils import find_module
from nems.metrics.stp import stp_magnitude
from nems_lbhb.gcmodel.magnitude import gc_magnitude
from nems_db.params import fitted_params_per_batch


def equivalence_scatter(batch, model1, model2, model3, se_filter=True,
                        ln_filter=False, ratio_filter=False,
                        threshold=2.5, manual_cellids=None):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    df_r = nd.batch_comp(batch, [model1, model2, model3], stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3], stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    gc_test = df_r[model1]
    gc_se = df_e[model1]
    stp_test = df_r[model2]
    stp_se = df_e[model2]
    ln_test = df_r[model3]
    ln_se = df_e[model3]

    if ln_filter:
        # Remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2))
    else:
        # Set to series w/ all True, so none are skipped
        good_cells = (gc_test != np.nan)

    if se_filter:
        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se))
    else:
        # Set to series w/ all False, so none are skipped
        bad_cells = (gc_test == np.nan)

    keep = good_cells & ~bad_cells
    cellids = df_r[keep].index.values.tolist()

    if ratio_filter:
        # Ex: for threshold = 2.5
        # Only use cellids where performance for gc/stp was within 2.5x
        # of LN performance (or where LN within 2.5x of gc/stp) to filter
        # outliers.
        c1 = get_valid_improvements(model1=model1, threshold=threshold)
        c2 = get_valid_improvements(model1=model2, threshold=threshold)
        cellids = list(set(c1) & set(c2) & set(cellids))

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids

    gc_test = df_r[model1][cellids]
    stp_test = df_r[model2][cellids]
    ln_test = df_r[model3][cellids]

    gc_vs_ln = gc_test.values - ln_test.values
    stp_vs_ln = stp_test.values - ln_test.values
    gc_vs_ln = gc_vs_ln.astype('float32')
    stp_vs_ln = stp_vs_ln.astype('float32')

    ff = np.isfinite(gc_vs_ln) & np.isfinite(stp_vs_ln)
    gc_vs_ln = gc_vs_ln[ff]
    stp_vs_ln = stp_vs_ln[ff]
    r = np.corrcoef(gc_vs_ln, stp_vs_ln)[0, 1]
    n = gc_vs_ln.size

    y_max = np.max(stp_vs_ln)
    y_min = np.min(stp_vs_ln)
    x_max = np.max(gc_vs_ln)
    x_min = np.min(gc_vs_ln)

    abs_max = max(np.abs(y_max), np.abs(x_max), np.abs(y_min), np.abs(x_min))
    abs_max *= 1.15

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(gc_vs_ln, stp_vs_ln, c='black', s=1)
    plt.xlabel("GC - LN model")
    plt.ylabel("STP - LN model")
    plt.title("Performance Improvements over LN\nr: %.02f, n: %d" % (r, n))
    gca = plt.gca()
    gca.axes.axhline(0, color='black', linewidth=1, linestyle='dashed')
    gca.axes.axvline(0, color='black', linewidth=1, linestyle='dashed')
    plt.ylim(ymin=(-1)*abs_max, ymax=abs_max)
    plt.xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)

    return fig


def equivalence_histogram(batch, model1, model2, model3, se_filter=True,
                          ln_filter=False, test_limit=None):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    df_r = nd.batch_comp(batch, [model1, model2, model3], stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3], stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    gc_test = df_r[model1]
    gc_se = df_e[model1]
    stp_test = df_r[model2]
    stp_se = df_e[model2]
    ln_test = df_r[model3]
    ln_se = df_e[model3]

    if se_filter:
        # Remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2))
    else:
        # Set to series w/ all True, so none are skipped
        good_cells = (gc_test != np.nan)

    if ln_filter:
        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se))
    else:
        # Set to series w/ all False, so none are skipped
        bad_cells = (gc_test == np.nan)

    keep = good_cells & ~bad_cells
    cellids = df_r[keep].index.values.tolist()

    rs = []
    for c in cellids[:test_limit]:
        xf1, ctx1 = xhelp.load_model_xform(c, batch, model1)
        xf2, ctx2 = xhelp.load_model_xform(c, batch, model2)
        xf3, ctx3 = xhelp.load_model_xform(c, batch, model3)

        gc = ctx1['val'].apply_mask()['pred'].as_continuous()
        stp = ctx2['val'].apply_mask()['pred'].as_continuous()
        ln = ctx3['val'].apply_mask()['pred'].as_continuous()

        ff = np.isfinite(gc) & np.isfinite(stp) & np.isfinite(ln)
        rs.append(np.corrcoef(gc[ff]-ln[ff], stp[ff]-ln[ff])[0, 1])

    rs = np.array(rs)
    md = np.nanmedian(rs)

    n_cells = len(cellids)
    fig = plt.figure(figsize=(6, 6))
    plt.hist(rs, bins=30, range=[-0.5, 1], histtype='bar', color=['gray'])
    plt.plot(np.array([0,0]), np.array(fig.axes[0].get_ylim()), 'k--')
    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.title('Equivalence of Change in Prediction Relative to LN Model')

    return fig


def gc_vs_stp_strengths(batch, model1, model2, model3, se_filter=True,
                        ln_filter=False, test_limit=None):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    df_r = nd.batch_comp(batch, [model1, model2, model3], stat='r_test')
    df_e = nd.batch_comp(batch, [model1, model2, model3], stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    gc_test = df_r[model1]
    gc_se = df_e[model1]
    stp_test = df_r[model2]
    stp_se = df_e[model2]
    ln_test = df_r[model3]
    ln_se = df_e[model3]

    if se_filter:
        # Remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2))
    else:
        # Set to series w/ all True, so none are skipped
        good_cells = (gc_test != np.nan)

    if ln_filter:
        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se))
    else:
        # Set to series w/ all False, so none are skipped
        bad_cells = (gc_test == np.nan)

    keep = good_cells & ~bad_cells
    cellids = df_r[keep].index.values.tolist()[:test_limit]

    gc_test = gc_test[cellids]
    stp_test = stp_test[cellids]
    ln_test = ln_test[cellids]

    df1 = fitted_params_per_batch(batch, model1, stats_keys=[])

    base_mod_df = df1[df1.index.str.contains('base_mod$')]
    base_df = df1[df1.index.str.contains('base$')]

    amp_mod_df = df1[df1.index.str.contains('amplitude_mod$')]
    amp_df = df1[df1.index.str.contains('amplitude$')]

    shift_mod_df = df1[df1.index.str.contains('shift_mod$')]
    shift_df = df1[df1.index.str.contains('shift$')]

    kappa_mod_df = df1[df1.index.str.contains('kappa_mod$')]
    kappa_df = df1[df1.index.str.contains('kappa$')]


    df2 = fitted_params_per_batch(batch, model2, stats_keys=[])
    tau_df = df2[df2.index.str.contains('tau$')]
    u_df = df2[df2.index.str.contains('-u$')]

    gcs = []
    stps = []
    for c in cellids:
        b = base_df[c].values[0]
        b_m = base_mod_df[c].values[0]
        a = amp_df[c].values[0]
        a_m = amp_mod_df[c].values[0]
        s = shift_df[c].values[0]
        s_m = shift_mod_df[c].values[0]
        k = kappa_df[c].values[0]
        k_m = kappa_mod_df[c].values[0]

        gc = gc_magnitude(b, b_m, a, a_m, s, s_m, k, k_m)
        gcs.append(gc)

        tau = tau_df[c].values[0]
        u = u_df[c].values[0]
        stp = abs(stp_magnitude(tau, u)[0])
        stps.append(stp)

    stps_arr = np.mean(np.array(stps), axis=1)
    gcs_arr = np.array(gcs)
    # Normalize both to 0-1 so they're on the same scale
    stps_arr /= np.abs(stps_arr.max())
    gcs_arr /= np.abs(gcs_arr.max())

    fig = plt.figure(figsize=(4, 4))
    plt.scatter(gcs_arr, stps_arr, c='black', s=1)
    plt.xlabel('GC Magnitude')
    plt.ylabel('STP Magnitude')
    fig.tight_layout()
