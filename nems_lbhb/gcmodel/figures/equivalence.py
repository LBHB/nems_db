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
                                             improved_cells_to_list,
                                             is_outlier, drop_common_outliers)
from nems.metrics.stp import stp_magnitude
from nems_lbhb.gcmodel.magnitude import gc_magnitude
from nems_db.params import fitted_params_per_batch
from nems_lbhb.gcmodel.figures.definitions import *

plt.rcParams.update(params)  # loaded from definitions


def equivalence_scatter(batch, gc, stp, LN, combined, se_filter=True,
                        LN_filter=False, plot_stat='r_ceiling',
                        enable_hover=False, manual_lims=None,
                        drop_outliers=False, color_improvements=True,
                        xmodel='GC', ymodel='STP'):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=se_filter,
                                           LN_filter=LN_filter)
    improved = e
    not_improved = list(set(a) - set(e))
    models = [gc, stp, LN]
    gc_rel_imp, stp_rel_imp = _relative_score(plot_df, models, improved,
                                              drop_outliers=drop_outliers)
    gc_rel_not, stp_rel_not = _relative_score(plot_df, models, not_improved,
                                              drop_outliers=drop_outliers)
    gc_rel_all, stp_rel_all = _relative_score(plot_df, models, a,
                                              drop_outliers=drop_outliers)

    r_imp, p_imp = st.pearsonr(gc_rel_imp, stp_rel_imp)
    r_not, p_not = st.pearsonr(gc_rel_not, stp_rel_not)
    r_all, p_all = st.pearsonr(gc_rel_all, stp_rel_all)
    n_imp = len(improved)
    n_not = len(not_improved)
    n_all = len(a)

    y_max = np.max(stp_rel_all)
    y_min = np.min(stp_rel_all)
    x_max = np.max(gc_rel_all)
    x_min = np.min(gc_rel_all)

    fig = plt.figure()
    ax = plt.gca()
    ax.axes.axhline(0, color='black', linewidth=2, linestyle='dashed',
                    dashes=dash_spacing)
    ax.axes.axvline(0, color='black', linewidth=2, linestyle='dashed',
                    dashes=dash_spacing)
    if color_improvements:
        ax.scatter(gc_rel_not, stp_rel_not, c=model_colors['LN'], s=15)
        ax.scatter(gc_rel_imp, stp_rel_imp, edgecolors=model_colors['max'],
                   facecolors='none', s=40, linewidth=2)
    else:
        ax.scatter(gc_rel_all, stp_rel_all, c='black', s=20)
    ax.set_xlabel("%s - LN model" % xmodel)
    ax.set_ylabel("%s - LN model" % ymodel)
    ax.set_title("Performance Improvements over LN\n"
                 "dropped outliers?:  %s\n"
                 "all cells:  r: %.02f, p: %.2E, n: %d\n"
                 "improved:  r: %.02f, p: %.2E, n: %d\n"
                 "not imp:  r: %.02f, p: %.2E, n: %d\n"
                 % (drop_outliers, r_all, p_all, n_all, r_imp, p_imp, n_imp,
                    r_not, p_not, n_not))
    plt.legend()

    if manual_lims is not None:
        ax.set_ylim(*manual_lims)
        ax.set_xlim(*manual_lims)
    else:
        upper = max(y_max, x_max)
        lower = min(y_min, x_min)
        upper_lim = np.ceil(10*upper)/10
        lower_lim = np.floor(10*lower)/10
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_xlim(lower_lim, upper_lim)

    if enable_hover:
        mplcursors.cursor(ax, hover=True).connect(
                "add",
                lambda sel: sel.annotation.set_text(a[sel.target.index])
                )

    plt.tight_layout()
    return fig


def _relative_score(df, models, mask, drop_outliers=False):
    g, s, L = models
    gc = df[g][mask]
    stp = df[s][mask]
    LN = df[L][mask]
    gc_rel = gc.values - LN.values
    stp_rel = stp.values - LN.values

    if drop_outliers:
        gc_rel, stp_rel = drop_common_outliers(gc_rel, stp_rel)

    return gc_rel, stp_rel


def equivalence_histogram(batch, gc, stp, LN, combined, se_filter=True,
                          LN_filter=False, test_limit=None, alpha=0.05,
                          save_path=None, load_path=None):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined)
    improved = e
    not_improved = list(set(a) - set(e))

    if load_path is None:
        df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
        cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc,
                                                              stp, LN, combined,
                                                              se_filter,
                                                              LN_filter)

        rs = []
        for c in a[:test_limit]:
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

        blank = np.full_like(rs, np.nan)
        results = {'cellid': cellids[:test_limit], 'equivalence': rs,
                   'effect_size': blank, 'corr_gc_LN': blank,
                   'corr_stp_LN': blank}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    rs = df['equivalence'].values

    imp = np.array(improved)
    not_imp = np.array(not_improved)
    imp_mask = np.isin(a, imp)
    not_mask = np.isin(a, not_imp)
    rs_not = rs[not_mask]
    rs_imp = rs[imp_mask]
    md_not = np.nanmedian(rs_not)
    md_imp = np.nanmedian(rs_imp)
    t, p = st.ttest_ind(rs_not, rs_imp)
    n_not = len(not_improved)
    n_imp = len(improved)

    colors = [model_colors['LN'], model_colors['max']]
    weights = [np.ones(len(rs_not))/len(rs_not),
               np.ones(len(rs_imp))/len(rs_imp)]

    #n_cells = rs.shape[0]
    fig, ax = plt.subplots()
    plt.hist([rs_not, rs_imp], bins=30, range=[-0.5, 1], weights=weights,
             color=colors, edgecolor='black', linewidth=1)
    ax.axes.axvline(md_not, color=model_colors['LN'], linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    ax.axes.axvline(md_imp, color=model_colors['max'], linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
#    plt.plot(np.array([0,0]), np.array(fig.axes[0].get_ylim()), 'k--',
#             linewidth=2, dashes=dash_spacing)

#    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
#             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.ylabel('proportion within group')
    plt.title('Equivalence of Change in Prediction Relative to LN Model\n'
              'n not imp:  %d,  md:  %.2f\n'
              'n sig. imp:  %d,  md:  %.2f\n'
              'p:  %.4E'
              % (n_not, md_not, n_imp, md_imp, p))

    plt.tight_layout()
    return fig


def equivalence_effect_size(batch, gc, stp, LN, combined, se_filter=True,
                            LN_filter=False, save_path=None, load_path=None,
                            test_limit=None, color_improvements=False):

    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=se_filter,
                                           LN_filter=LN_filter)

    if load_path is None:
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
    r, p = st.pearsonr(effect_size, equivalence)

    fig1, ax = plt.subplots(1,1)
    ax.axes.axhline(0, color='black', linewidth=2, linestyle='dashed',
                    dashes=dash_spacing)

    extra_title_lines = []
    if color_improvements:
        improved = e
        not_improved = list(set(a) - set(e))
        equivalence_imp = df['equivalence'][improved].values
        equivalence_not = df['equivalence'][not_improved].values
        effectsize_imp = df['effect_size'][improved].values
        effectsize_not = df['effect_size'][not_improved].values
        r_imp, p_imp = st.pearsonr(effectsize_imp, equivalence_imp)
        r_not, p_not = st.pearsonr(effectsize_not, equivalence_not)
        n_imp = len(improved)
        n_not = len(not_improved)
        lines = ["improved cells,  r:  %.4f,    p:  %.4E" % (r_imp, p_imp),
                 "not improved,  r:  %.4f,    p:  %.4E" % (r_not, p_not)]
        extra_title_lines.extend(lines)

        plt.scatter(effectsize_not, equivalence_not, s=15,
                    color=model_colors['LN'], label='no imp')
        plt.scatter(effectsize_imp, equivalence_imp, s=40, facecolors='none',
                    edgecolors=model_colors['max'], linewidth=2,
                    label='sig. imp.')
        plt.legend()
    else:
        plt.scatter(effect_size, equivalence, s=20, color='black')

    plt.ylabel('Equivalence:  CC(GC-LN, STP-LN)')
    plt.xlabel('Effect size:  1 - 0.5*(CC(GC,LN) + CC(STP,LN))')
    title = ("Equivalence of Change to Predicted PSTH\n"
             "vs Effect Size\n"
             "all cells,  r:  %.4f,    p:  %.4E" % (r, p))
    for ln in extra_title_lines:
        title += "\n%s" % ln
    plt.title(title)
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
