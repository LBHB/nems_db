import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import mplcursors

import nems.xform_helper as xhelp
import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             get_valid_improvements,
                                             adjustFigAspect,
                                             improved_cells_to_list)
from nems.metrics.stp import stp_magnitude
from nems_lbhb.gcmodel.magnitude import gc_magnitude
from nems_db.params import fitted_params_per_batch
from nems_lbhb.gcmodel.figures.definitions import *

plt.rcParams.update(params)  # loaded from definitions


def equivalence_scatter(batch, gc, stp, LN, combined, se_filter=True,
                        LN_filter=False, manual_cellids=None,
                        plot_stat='r_ceiling', enable_hover=False,
                        add_combined=False, color_improvements=False):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          LN_filter)
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           as_lists=False)

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    gc_test = plot_df[gc][cellids]
    stp_test = plot_df[stp][cellids]
    ln_test = plot_df[LN][cellids]
    combined_test = plot_df[combined][cellids]

    gc_vs_ln = gc_test.values - ln_test.values
    stp_vs_ln = stp_test.values - ln_test.values
    combined_vs_ln = combined_test.values - ln_test.values

    if color_improvements:
        gc_test_gc_cells = plot_df[gc][cellids][g].values
        stp_test_gc_cells = plot_df[stp][cellids][g].values
        ln_test_gc_cells = plot_df[LN][cellids][g].values
        gc_test_stp_cells = plot_df[gc][cellids][s].values
        stp_test_stp_cells = plot_df[stp][cellids][s].values
        ln_test_stp_cells = plot_df[LN][cellids][s].values

        gc_vs_ln_gc_cells = gc_test_gc_cells - ln_test_gc_cells
        stp_vs_ln_gc_cells = stp_test_gc_cells - ln_test_gc_cells
        gc_vs_ln_stp_cells = gc_test_stp_cells - ln_test_stp_cells
        stp_vs_ln_stp_cells = stp_test_stp_cells - ln_test_stp_cells


#    gc_vs_ln = gc_vs_ln.astype('float32')
#    stp_vs_ln = stp_vs_ln.astype('float32')
#    combined_vs_ln = combined_vs_ln.astype('float32')
#
#    ff = np.isfinite(gc_vs_ln) & np.isfinite(stp_vs_ln) & np.isfinite(combined_vs_ln)
#    gc_vs_ln = gc_vs_ln[ff]
#    stp_vs_ln = stp_vs_ln[ff]
#    combined_vs_ln = combined_vs_ln[ff]

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

    fig = plt.figure()
    ax = plt.gca()
    ax.axes.axhline(0, color='black', linewidth=2, linestyle='dashed', dashes=dash_spacing)
    ax.axes.axvline(0, color='black', linewidth=2, linestyle='dashed', dashes=dash_spacing)
    ax.scatter(gc_vs_ln, stp_vs_ln, c=wsu_gray, s=20)
    if color_improvements:
        ax.scatter(gc_vs_ln_gc_cells, stp_vs_ln_gc_cells, c=model_colors['gc'])
        ax.scatter(gc_vs_ln_stp_cells, stp_vs_ln_stp_cells, c=model_colors['stp'])
    ax.set_xlabel("GC - LN model")
    ax.set_ylabel("STP - LN model")
    ax.set_title("Performance Improvements over LN\nr: %.02f, p: %.2E, n: %d\n"
              % (r2, p, n))
    ax.set_ylim(ymin=(-1)*abs_max, ymax=abs_max)
    ax.set_xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)
    if enable_hover:
        mplcursors.cursor(ax, hover=True).connect(
                "add",
                lambda sel: sel.annotation.set_text(cellids[sel.target.index])
                )

    if add_combined:
        fig2 = plt.figure()
        ax = plt.gca()
        ax.axes.axhline(0, color='black', linewidth=2, linestyle='dashed', dashes=dash_spacing)
        ax.axes.axvline(0, color='black', linewidth=2, linestyle='dashed', dashes=dash_spacing)
        ax.scatter(combined_vs_ln, stp_vs_ln, c=wsu_crimson, s=20,
                    alpha=0.3, label='combined vs stp')
        ax.scatter(gc_vs_ln, combined_vs_ln, c='goldenrod', s=20,
             alpha=0.3, label='gc vs combined')
        ax.legend()
        ax.set_xlabel("GC - LN model")
        ax.set_ylabel("STP - LN model")
        ax.set_title("Performance Improvements over LN\nr: %.02f, p: %.2E, n: %d\n"
                  % (r2, p, n))
        ax.set_ylim(ymin=(-1)*abs_max, ymax=abs_max)
        ax.set_xlim(xmin=(-1)*abs_max, xmax=abs_max)
        adjustFigAspect(fig, aspect=1)
        if enable_hover:
            mplcursors.cursor(ax, hover=True).connect(
                    "add",
                    lambda sel: sel.annotation.set_text(cellids[sel.target.index])
                    )

    plt.tight_layout()
    return fig


def equivalence_histogram(batch, gc, stp, LN, combined, se_filter=True,
                          LN_filter=False, test_limit=None, alpha=0.05,
                          save_path=None, load_path=None):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    if load_path is None:
        df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
        cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                              LN, combined,
                                                              se_filter,
                                                              LN_filter)

        rs = []
        #ks = []
        for c in cellids[:test_limit]:
            xf1, ctx1 = xhelp.load_model_xform(c, batch, gc)
            xf2, ctx2 = xhelp.load_model_xform(c, batch, stp)
            xf3, ctx3 = xhelp.load_model_xform(c, batch, LN)

            gc_pred = ctx1['val'].apply_mask()['pred'].as_continuous()
            stp_pred = ctx2['val'].apply_mask()['pred'].as_continuous()
            ln_pred = ctx3['val'].apply_mask()['pred'].as_continuous()

            ff = np.isfinite(gc_pred) & np.isfinite(stp_pred) & np.isfinite(ln_pred)
            gcff = gc_pred[ff]
            stpff = stp_pred[ff]
            lnff = ln_pred[ff]
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
    fig = plt.figure()
    plt.hist(rs, bins=30, range=[-0.5, 1], histtype='bar', color=[wsu_gray_light],
             edgecolor='black', linewidth=1)
    plt.plot(np.array([0,0]), np.array(fig.axes[0].get_ylim()), 'k--',
             linewidth=2, dashes=dash_spacing)
    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.title('Equivalence of Change in Prediction Relative to LN Model')

    # TODO: maybe not working as intended? or maybe it is and the p values
    #       are just tiny, but end up with all < 0.00001
    #       Ask SVD about continuing with this.
#    fig2 = plt.figure()
#    plt.hist(logks, bins=30, range=[0, 5], histtype='bar',
#                                    color=['gray'])
#    plt.plot(np.array([-np.log10(0.05), -np.log10(0.05)]),
#             np.array(fig.axes[0].get_ylim()), 'k--')
#    plt.text(0.05, 0.95, 'n = %d' % n_cells,
#             ha='left', va='top', transform=fig.axes[0].transAxes)
#    plt.xlabel('-log p')
#    plt.title('Kolmolgorov-Smirnov Test\nBetween Prediction Changes Relative to LN Model')

    plt.tight_layout()
    return fig#, fig2


def equivalence_effect_size(batch, gc, stp, LN, combined, se_filter=True,
                            LN_filter=False, save_path=None, load_path=None,
                            test_limit=None):

    if load_path is None:
        e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                               se_filter=se_filter,
                                               LN_filter=LN_filter)
        equivs = []
        gcs = []
        stps = []
        effects = []
        for cell in a[:test_limit]:
            xf1, ctx1 = xhelp.load_model_xform(cell, batch, gc)
            xf2, ctx2 = xhelp.load_model_xform(cell, batch, stp)
            xf3, ctx3 = xhelp.load_model_xform(cell, batch, LN)

            gc_pred = ctx1['val'].apply_mask()['pred'].as_continuous()
            stp_pred = ctx2['val'].apply_mask()['pred'].as_continuous()
            ln_pred = ctx3['val'].apply_mask()['pred'].as_continuous()

            ff = np.isfinite(gc_pred) & np.isfinite(stp_pred) & np.isfinite(ln_pred)
            gcff = gc_pred[ff]
            stpff = stp_pred[ff]
            lnff = ln_pred[ff]

            equivs.append(np.corrcoef(gcff-lnff, stpff-lnff)[0, 1])
            this_gc = np.corrcoef(gcff, lnff)[0,1]
            this_stp = np.corrcoef(stpff, lnff)[0,1]
            gcs.append(this_gc)
            stps.append(this_stp)
            effects.append(1 - 0.5*(this_gc+this_stp))

        results = {'cellid': a[:test_limit], 'equivalence': equivs,
                   'effect_size': effects, 'corr_gc_LN': gcs,
                   'corr_stp_LN': stps}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    equivalence = df['equivalence'].values
    effect_size = df['effect_size'].values

    fig1 = plt.figure()
    plt.scatter(effect_size, equivalence, s=20, color='black')
    plt.ylabel('Equivalence:  CC(GC-LN, STP-LN)')
    plt.xlabel('Effect size:  1 - 0.5*(CC(GC,LN) + CC(STP,LN))')
    plt.title("Equivalence of Change to Predicted PSTH\n"
              "vs Effect Size")
    plt.tight_layout()

    fig2 = plt.figure()
    md = np.nanmedian(equivalence)
    n_cells = equivalence.shape[0]
    plt.hist(equivalence, bins=30, range=[-0.5, 1], histtype='bar',
             color=[wsu_gray_light], edgecolor='black', linewidth=1)
    plt.plot(np.array([0,0]), np.array(fig2.axes[0].get_ylim()), 'k--',
             linewidth=2, dashes=dash_spacing)
    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
             ha='left', va='top', transform=fig2.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.title('Equivalence of Change in Prediction Relative to LN Model')

    return fig1, fig2


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
    fig = plt.figure()
    plt.hist(rs, bins=30, range=[-0.5, 1], histtype='bar', color=[wsu_gray_light])
    plt.plot(np.array([0,0]), np.array(fig.axes[0].get_ylim()), 'k--', dashes=dash_spacing)
    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.title('Equivalence of Change in Magnitude of Error Relative to LN Model')

    # TODO: maybe not working as intended? or maybe it is and the p values
    #       are just tiny, but end up with all < 0.00001
    #       Ask SVD about continuing with this.
    fig2 = plt.figure()
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
