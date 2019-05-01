import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import mplcursors

import nems.xform_helper as xhelp
import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import (get_valid_improvements,
                                             adjustFigAspect)
from nems.metrics.stp import stp_magnitude
from nems_lbhb.gcmodel.magnitude import gc_magnitude
from nems_db.params import fitted_params_per_batch

params = {
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1,
        'font.weight': 'bold',
        'font.size': 16,
        }
plt.rcParams.update(params)


def equivalence_scatter(batch, model1, model2, model3, model4, se_filter=True,
                        ln_filter=False, ratio_filter=False, threshold=2.5,
                        manual_cellids=None, enable_hover=False):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    df_r = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='se_test')
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
    gc_stp_test = df_r[model4]
    gc_stp_se = df_e[model4]

    if se_filter:
        # Remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2) & (gc_stp_test > gc_stp_se*2))
    else:
        # Set to series w/ all True, so none are skipped
        good_cells = (gc_test != np.nan)

    if ln_filter:
        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se) |
                     (gc_stp_test+gc_stp_se < ln_test-ln_se))
    else:
        # Set to series w/ all False, so none are skipped
        bad_cells = (gc_test == np.nan)

    keep = good_cells & ~bad_cells
    cellids = df_r[keep].index.values.tolist()
    under_chance = df_r[~good_cells].index.values.tolist()
    less_LN = df_r[bad_cells].index.values.tolist()

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
    #r = np.corrcoef(gc_vs_ln, stp_vs_ln)[0, 1]
    r2, p = st.pearsonr(gc_vs_ln, stp_vs_ln)
    # TODO: compute p manually? the scipy documentation isn't
    #       very clear on how they calculate it, something to do with
    #       a beta function
    n = gc_vs_ln.size

    y_max = np.max(stp_vs_ln)
    y_min = np.min(stp_vs_ln)
    x_max = np.max(gc_vs_ln)
    x_min = np.min(gc_vs_ln)

    abs_max = max(np.abs(y_max), np.abs(x_max), np.abs(y_min), np.abs(x_min))
    abs_max *= 1.15

    fig = plt.figure(figsize=(12, 12))
    ax = plt.gca()
    scatter = ax.scatter(gc_vs_ln, stp_vs_ln, c=wsu_gray, s=20)
    ax.set_xlabel("GC - LN model")
    ax.set_ylabel("STP - LN model")
    ax.set_title("Performance Improvements over LN\nr: %.02f, p: %.2E, n: %d\n"
              % (r2, p, n))
    ax.axes.axhline(0, color='black', linewidth=1, linestyle='dashed')
    ax.axes.axvline(0, color='black', linewidth=1, linestyle='dashed')
    ax.set_ylim(ymin=(-1)*abs_max, ymax=abs_max)
    ax.set_xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)
    if enable_hover:
        mplcursors.cursor(ax, hover=True).connect(
                "add",
                lambda sel: sel.annotation.set_text(cellids[sel.target.index])
                )

    return fig


def equivalence_histogram(batch, model1, model2, model3, model4, se_filter=True,
                          ln_filter=False, test_limit=None, alpha=0.05,
                          save_path=None, load_path=None):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    if load_path is None:
        df_r = nd.batch_comp(batch, [model1, model2, model3, model4],
                             stat='r_ceiling')
        df_e = nd.batch_comp(batch, [model1, model2, model3, model4],
                             stat='se_test')
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
        gc_stp_test = df_r[model4]
        gc_stp_se = df_e[model4]

        if se_filter:
            # Remove is performance not significant at all
            good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                         (ln_test > ln_se*2) & (gc_stp_test > gc_stp_se*2))
        else:
            # Set to series w/ all True, so none are skipped
            good_cells = (gc_test != np.nan)

        if ln_filter:
            # Remove if performance significantly worse than LN
            bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                         (stp_test+stp_se < ln_test-ln_se) |
                         (gc_stp_test+gc_stp_se < ln_test-ln_se))
        else:
            # Set to series w/ all False, so none are skipped
            bad_cells = (gc_test == np.nan)

        keep = good_cells & ~bad_cells
        cellids = df_r[keep].index.values.tolist()
        under_chance = df_r[~good_cells].index.values.tolist()
        less_LN = df_r[bad_cells].index.values.tolist()

        rs = []
        #ks = []
        for c in cellids[:test_limit]:
            xf1, ctx1 = xhelp.load_model_xform(c, batch, model1)
            xf2, ctx2 = xhelp.load_model_xform(c, batch, model2)
            xf3, ctx3 = xhelp.load_model_xform(c, batch, model3)

            gc = ctx1['val'].apply_mask()['pred'].as_continuous()
            stp = ctx2['val'].apply_mask()['pred'].as_continuous()
            ln = ctx3['val'].apply_mask()['pred'].as_continuous()

            ff = np.isfinite(gc) & np.isfinite(stp) & np.isfinite(ln)
            gcff = gc[ff]
            stpff = stp[ff]
            lnff = ln[ff]
            rs.append(np.corrcoef(gcff-lnff, stpff-lnff)[0, 1])
            #p = st.ks_2samp(gcff-lnff, stpff-lnff)[1]
            #ks.append(D)
    #        if p == 0:
    #            ks.append(1e-5)
    #        else:
    #            ks.append(p)

        rs = np.array(rs)
        if save_path is not None:
            np.save(save_path, rs)
    else:
        rs = np.load(load_path)

    md = np.nanmedian(rs)
    #mks = np.nanmedian(ks)
    #logks = -1*np.log10(ks)

    #n_samps = gcff.shape[-1]
    #d_threshold = (np.sqrt(-0.5*np.log(alpha)))*np.sqrt((2*n_samps)/n_samps**2)
    n_cells = rs.shape[0]
    fig = plt.figure(figsize=(12, 12))
    plt.hist(rs, bins=30, range=[-0.5, 1], histtype='bar', color=[wsu_gray_light])
    plt.plot(np.array([0,0]), np.array(fig.axes[0].get_ylim()), 'k--')
    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.title('Equivalence of Change in Prediction Relative to LN Model')

    # TODO: maybe not working as intended? or maybe it is and the p values
    #       are just tiny, but end up with all < 0.00001
    #       Ask SVD about continuing with this.
#    fig2 = plt.figure(figsize=(12, 12))
#    plt.hist(logks, bins=30, range=[0, 5], histtype='bar',
#                                    color=['gray'])
#    plt.plot(np.array([-np.log10(0.05), -np.log10(0.05)]),
#             np.array(fig.axes[0].get_ylim()), 'k--')
#    plt.text(0.05, 0.95, 'n = %d' % n_cells,
#             ha='left', va='top', transform=fig.axes[0].transAxes)
#    plt.xlabel('-log p')
#    plt.title('Kolmolgorov-Smirnov Test\nBetween Prediction Changes Relative to LN Model')

    return fig#, fig2


def residual_histogram(batch, model1, model2, model3, model4, se_filter=True,
                       ln_filter=False, test_limit=None, alpha=0.05,
                       save_path=None, load_path=None):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    if load_path is None:
        df_r = nd.batch_comp(batch, [model1, model2, model3, model4],
                             stat='r_ceiling')
        df_e = nd.batch_comp(batch, [model1, model2, model3, model4],
                             stat='se_test')
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
        gc_stp_test = df_r[model4]
        gc_stp_se = df_e[model4]

        if se_filter:
            # Remove is performance not significant at all
            good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                         (ln_test > ln_se*2) & (gc_stp_test > gc_stp_se*2))
        else:
            # Set to series w/ all True, so none are skipped
            good_cells = (gc_test != np.nan)

        if ln_filter:
            # Remove if performance significantly worse than LN
            bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                         (stp_test+stp_se < ln_test-ln_se) |
                         (gc_stp_test+gc_stp_se < ln_test-ln_se))
        else:
            # Set to series w/ all False, so none are skipped
            bad_cells = (gc_test == np.nan)

        keep = good_cells & ~bad_cells
        cellids = df_r[keep].index.values.tolist()
        under_chance = df_r[~good_cells].index.values.tolist()
        less_LN = df_r[bad_cells].index.values.tolist()

        rs = []
        ks = []
        for c in cellids[:test_limit]:
            xf1, ctx1 = xhelp.load_model_xform(c, batch, model1)
            xf2, ctx2 = xhelp.load_model_xform(c, batch, model2)
            xf3, ctx3 = xhelp.load_model_xform(c, batch, model3)

            gc = ctx1['val'].apply_mask()['pred'].as_continuous()
            stp = ctx2['val'].apply_mask()['pred'].as_continuous()
            ln = ctx3['val'].apply_mask()['pred'].as_continuous()
            resp = ctx3['val'].apply_mask()['resp'].as_continuous()

            ff = (np.isfinite(gc) & np.isfinite(stp)
                  & np.isfinite(ln) & np.isfinite(resp))
            gcff = gc[ff]
            stpff = stp[ff]
            lnff = ln[ff]
            respff = resp[ff]

            gc_err = np.abs(gcff-respff)
            stp_err = np.abs(stpff-respff)
            ln_err = np.abs(lnff-respff)

            rs.append(np.corrcoef(gc_err-ln_err, stp_err-ln_err)[0, 1])
            p = st.ks_2samp(gc_err-ln_err, stp_err-ln_err)[1]
            #ks.append(D)
            if p == 0:
                ks.append(1e-10)
            else:
                ks.append(p)

        rs = np.array(rs)
        if save_path is not None:
            np.save(save_path, rs)
    else:
        rs = np.load(load_path)

    md = np.nanmedian(rs)
    #mks = np.nanmedian(ks)
    logks = -1*np.log10(ks)

    #n_samps = gcff.shape[-1]
    #d_threshold = (np.sqrt(-0.5*np.log(alpha)))*np.sqrt((2*n_samps)/n_samps**2)
    n_cells = rs.shape[0]
    fig = plt.figure(figsize=(12, 12))
    plt.hist(rs, bins=30, range=[-0.5, 1], histtype='bar', color=[wsu_gray_light])
    plt.plot(np.array([0,0]), np.array(fig.axes[0].get_ylim()), 'k--')
    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.title('Equivalence of Change in Magnitude of Error Relative to LN Model')

    # TODO: maybe not working as intended? or maybe it is and the p values
    #       are just tiny, but end up with all < 0.00001
    #       Ask SVD about continuing with this.
    fig2 = plt.figure(figsize=(12, 12))
    plt.hist(logks, bins=30, range=[0, 10], histtype='bar',
                                    color=[wsu_gray_light])
    plt.plot(np.array([-np.log10(0.05), -np.log10(0.05)]),
             np.array(fig.axes[0].get_ylim()), 'k--')
    plt.text(0.05, 0.95, 'n = %d' % n_cells,
             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('-log p')
    plt.title('Kolmolgorov-Smirnov Test\nBetween Changes in Magnitude of Error Relative to LN Model')

    return fig, fig2

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
