import copy

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import stats, linalg
import matplotlib.pyplot as plt
import mplcursors

import nems.xform_helper as xhelp
import nems.xforms as xforms
import nems.db as nd
import nems.epoch as ep
from nems.utils import ax_remove_box
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
                        xmodel='GC', ymodel='STP', legend=False,
                        self_equiv=False, self_eq_models=[],
                        show_highlights=False,
                        exclude_low_snr=False,
                        snr_path=None):
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
    improved = c
    not_improved = list(set(a) - set(c))
    if exclude_low_snr:
        snr_df = pd.read_pickle(snr_path)
        med_snr = snr_df['snr'].median()
        high_snr = snr_df.loc[snr_df['snr'] >= med_snr]
        high_snr_cells = high_snr.index.values.tolist()
        improved = list(set(improved) & set(high_snr_cells))
        not_improved = list(set(not_improved) & set(high_snr_cells))
        a = list(set(a) & set(high_snr_cells))
        # check number of animals included
        prefixes = [s[:3] for s in a]
        n_animals = len(list(set(prefixes)))
        print('n_animals: %d' % n_animals)

    models = [gc, stp, LN]
    gc_rel_imp, stp_rel_imp = _relative_score(plot_df, models, improved)
    gc_rel_not, stp_rel_not = _relative_score(plot_df, models, not_improved)
    gc_rel_all, stp_rel_all = _relative_score(plot_df, models, a)


    # LN, STP, GC
    cells_to_highlight = ['TAR010c-40-1', 'AMT005c-20-1', 'TAR009d-22-1']
    cells_to_plot_gc_rel = []
    cells_to_plot_stp_rel = []
    for c in cells_to_highlight:
        stp_rel = plot_df[stp][c] - plot_df[LN][c]
        gc_rel = plot_df[gc][c] - plot_df[LN][c]
        cells_to_plot_gc_rel.append(gc_rel)
        cells_to_plot_stp_rel.append(stp_rel)

    # compute corr. before dropping outliers (only dropping for visualization)
    r_imp, p_imp = st.pearsonr(gc_rel_imp, stp_rel_imp)
    r_not, p_not = st.pearsonr(gc_rel_not, stp_rel_not)
    r_all, p_all = st.pearsonr(gc_rel_all, stp_rel_all)

    if self_equiv:
        stp1, stp2, gc1, gc2, LN1, LN2 = self_eq_models
        _, ga, _, _, _ = improved_cells_to_list(batch, gc1, gc2, LN1, LN2,
                                                as_lists=True)
        _, sa, _, _, _ = improved_cells_to_list(batch, stp1, stp2, LN1, LN2,
                                                as_lists=True)
        aa = list(set(ga) & set(sa))

        if exclude_low_snr:
            snr_df = pd.read_pickle(snr_path)
            med_snr = snr_df['snr'].median()
            high_snr = snr_df.loc[snr_df['snr'] >= med_snr]
            high_snr_cells = high_snr.index.values.tolist()
            aa = list(set(aa) & set(high_snr_cells))

        df_r_eq = nd.batch_comp(batch, [gc1, gc2, stp1, stp2, LN1, LN2],
                                stat=plot_stat)
        df_r_eq.dropna(axis=0, how='any', inplace=True)
        df_r_eq.sort_index(inplace=True)
        df_r_eq = df_r_eq[df_r_eq.index.isin(aa)]

        gc1_rel_imp = df_r_eq[gc1].values - df_r_eq[LN1].values
        gc2_rel_imp = df_r_eq[gc2].values - df_r_eq[LN2].values
        stp1_rel_imp = df_r_eq[stp1].values - df_r_eq[LN1].values
        stp2_rel_imp = df_r_eq[stp2].values - df_r_eq[LN2].values
        r_gceq, p_gceq = st.pearsonr(gc1_rel_imp, gc2_rel_imp)
        r_stpeq, p_stpeq = st.pearsonr(stp1_rel_imp, stp2_rel_imp)
        n_eq = gc1_rel_imp.size

        # compute on same subset for full estimation data
        # to compare to cross-set
        gc_subset1 = df_r[gc][ga].values
        gc_subset2 = df_r[gc][sa].values
        stp_subset1 = df_r[stp][ga].values
        stp_subset2 = df_r[stp][sa].values
        LN_subset1 = df_r[LN][ga].values
        LN_subset2 = df_r[LN][sa].values
        gc_rel1 = gc_subset1 - LN_subset1
        gc_rel2 = gc_subset2 - LN_subset2
        stp_rel1 = stp_subset1 - LN_subset1
        stp_rel2 = stp_subset2 - LN_subset2
        r_sub1, p_sub1 = st.pearsonr(gc_rel1, stp_rel1)
        r_sub2, p_sub2 = st.pearsonr(gc_rel2, stp_rel2)


    gc_rel_imp, stp_rel_imp = drop_common_outliers(gc_rel_imp, stp_rel_imp)
    gc_rel_not, stp_rel_not = drop_common_outliers(gc_rel_not, stp_rel_not)
    gc_rel_all, stp_rel_all = drop_common_outliers(gc_rel_all, stp_rel_all)

    n_imp = len(improved)
    n_not = len(not_improved)
    n_all = len(a)

    y_max = np.max(stp_rel_all)
    y_min = np.min(stp_rel_all)
    x_max = np.max(gc_rel_all)
    x_min = np.min(gc_rel_all)

    fig = plt.figure()
    ax = plt.gca()
    ax.axes.axhline(0, color='black', linewidth=1, linestyle='dashed',
                    dashes=dash_spacing)
    ax.axes.axvline(0, color='black', linewidth=1, linestyle='dashed',
                    dashes=dash_spacing)
    if color_improvements:
        ax.scatter(gc_rel_not, stp_rel_not, c=model_colors['LN'], s=small_scatter)
        ax.scatter(gc_rel_imp, stp_rel_imp, c=model_colors['max'], s=big_scatter)
        if show_highlights:
            for i, (g, s) in enumerate(zip(cells_to_plot_gc_rel, cells_to_plot_stp_rel)):
                plt.text(g, s, str(i+1), fontsize=12, color='black')
    else:
        ax.scatter(gc_rel_all, stp_rel_all, c='black', s=big_scatter)

    if legend:
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
    plt.axes().set_aspect('equal')

    if enable_hover:
        mplcursors.cursor(ax, hover=True).connect(
                "add",
                lambda sel: sel.annotation.set_text(a[sel.target.index])
                )

    plt.tight_layout()

    fig2 = plt.figure(figsize=text_fig)
    text = ("Performance Improvements over LN\n"
            "batch: %d\n"
            "dropped outliers?:  %s\n"
            "all cells:  r: %.2E, p: %.2E, n: %d\n"
            "improved:  r: %.2E, p: %.2E, n: %d\n"
            "not imp:  r: %.2E, p: %.2E, n: %d\n"
            "x: %s - LN\n"
            "y: %s - LN"
            % (batch, drop_outliers, r_all, p_all, n_all, r_imp, p_imp, n_imp,
               r_not, p_not, n_not, xmodel, ymodel))
    if self_equiv:
        text += ("\n\nSelf Equivalence, n:  %d\n"
                 "gc,  r: %.2Ef, p: %.2E\n"
                 "stp, r: %.2Ef, p: %.2E\n"
                 "sub1, r: %.2E, p: %.2E\n"
                 "sub2, r: %.2E, p: %.2E\n" %
                 (n_eq, r_gceq, p_gceq, r_stpeq, p_stpeq, r_sub1, p_sub1,
                  r_sub2, p_sub2))
    plt.text(0.1, 0.5, text)
    ax_remove_box(ax)

    return fig, fig2


def _relative_score(df, models, mask):
    g, s, L = models
    gc = df[g][mask]
    stp = df[s][mask]
    LN = df[L][mask]
    gc_rel = gc.values - LN.values
    stp_rel = stp.values - LN.values

    return gc_rel, stp_rel


def self_equivalence_data(batch, m1, m2, LN1, LN2, save_path=None,
                          load_path=None, cellids=None):

    e, a, g, s, c = improved_cells_to_list(batch, m1, m2, LN1, LN2,
                                           as_lists=True)
    if load_path is None:
        eqs = []
        for c in a:
            xf1, ctx1 = xhelp.load_model_xform(c, batch, m1)
            xf2, ctx2 = xhelp.load_model_xform(c, batch, m2)
            xf3, ctx3 = xhelp.load_model_xform(c, batch, LN1)
            xf4, ctx4 = xhelp.load_model_xform(c, batch, LN2)

            test_pred1 = ctx1['val'].apply_mask()['pred'].as_continuous().flatten()
            test_pred2 = ctx2['val'].apply_mask()['pred'].as_continuous().flatten()
            LN_pred1 = ctx3['val'].apply_mask()['pred'].as_continuous().flatten()
            LN_pred2 = ctx4['val'].apply_mask()['pred'].as_continuous().flatten()

            C1 = np.hstack((np.expand_dims(test_pred1, 0).transpose(),
                            np.expand_dims(test_pred2, 0).transpose(),
                            np.expand_dims(LN_pred1, 0).transpose()))
            p1 = partial_corr(C1)[0,1]

            C2 = np.hstack((np.expand_dims(test_pred1, 0).transpose(),
                            np.expand_dims(test_pred2, 0).transpose(),
                            np.expand_dims(LN_pred2, 0).transpose()))
            p2 = partial_corr(C2)[0,1]
            eqs.append(0.5*(p1+p2))

        results = {'cellid': a, 'equivalence': eqs}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)

    else:
        df = pd.read_pickle(load_path)

    # don't look for cells that the analysis didn't
    # get saved for
    saved_cells = df.index.values.tolist()
    a = list(set(a) & set(saved_cells))
    df = df.loc[a]
    if cellids is not None:
        df = df.loc[cellids].reindex(cellids)
    eqs = df['equivalence'].values

    return eqs


def plot_self_equivalence(batch, stp1, stp2, gc1, gc2, LN1, LN2,
                          stp_load, gc_load, axes=None):
    eqs1 = self_equivalence_data(batch, stp1, stp2, LN1, LN2, load_path=stp_load)
    eqs2 = self_equivalence_data(batch, gc1, gc2, LN1, LN2, load_path=gc_load)
    eqs = np.hstack([eqs1, eqs2])
    weights1 = [np.ones(len(eqs))/len(eqs)]

    if axes is not None:
        a1 = axes
    else:
        _, a1 = plt.subplots(1,1)

    a1.hist(eqs, bins=30, range=[-0.5, 1], weights=weights1,
            fc='gray', edgecolor='black', linewidth=1, alpha=0.6)
    a1.axes.axvline(np.median(eqs), color='black', linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    if axes is None:
        ax_remove_box(a1)
        plt.tight_layout()

    return a1


def _get_self_equivs(batch, stp1, stp2, gc1, gc2, LN1, LN2, stp_load, gc_load,
                     cellids=None):
    eqs1 = self_equivalence_data(batch, stp1, stp2, LN1, LN2, load_path=stp_load,
                                 cellids=cellids)
    eqs2 = self_equivalence_data(batch, gc1, gc2, LN1, LN2, load_path=gc_load,
                                 cellids=cellids)
    return eqs1, eqs2


def equivalence_histogram(batch, gc, stp, LN, combined, se_filter=True,
                          LN_filter=False, test_limit=None, alpha=0.05,
                          save_path=None, load_path=None,
                          equiv_key='partial_corr',
                          effect_key='performance_effect',
                          self_equiv=False, self_kwargs={}, eq_models=[],
                          cross_kwargs={}, cross_models=[],
                          use_median=True, exclude_low_snr=False,
                          snr_path=None, adjust_scores=True,
                          use_log_ratios=False):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined)
    _, cellids, _, _, _ = improved_cells_to_list(batch, gc, stp, LN, combined,
                                                 as_lists=False)
    improved = c
    not_improved = list(set(a) - set(c))

    if load_path is None:
        df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)

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
        results = {'cellid': a[:test_limit], 'equivalence': rs,
                   'effect_size': blank, 'corr_gc_LN': blank,
                   'corr_stp_LN': blank}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    df = df[cellids]
    rs = df[equiv_key].values
    if exclude_low_snr:
        snr_df = pd.read_pickle(snr_path)
        med_snr = snr_df['snr'].median()
        high_snr = snr_df.loc[snr_df['snr'] >= med_snr]
        high_snr_cells = high_snr.index.values.tolist()
        improved = list(set(improved) & set(high_snr_cells))
        not_improved = list(set(not_improved) & set(high_snr_cells))

    imp = np.array(improved)
    not_imp = np.array(not_improved)
    imp_mask = np.isin(a, imp)
    not_mask = np.isin(a, not_imp)
    rs_not = rs[not_mask]
    rs_imp = rs[imp_mask]
    md_not = np.nanmedian(rs_not)
    md_imp = np.nanmedian(rs_imp)
    u, p = st.mannwhitneyu(rs_not, rs_imp, alternative='two-sided')
    n_not = len(not_improved)
    n_imp = len(improved)

    if self_equiv:
        stp1, stp2, gc1, gc2, LN1, LN2 = eq_models
        g1, s2, s1, g2, L1, L2 = cross_models
        _, ga, _, _, _ = improved_cells_to_list(batch, gc1, gc2, LN1, LN2,
                                                as_lists=True)
        _, sa, _, _, _ = improved_cells_to_list(batch, stp1, stp2, LN1, LN2,
                                                as_lists=True)
        aa = list(set(ga) & set(sa))
        if exclude_low_snr:
            snr_df = pd.read_pickle(snr_path)
            med_snr = snr_df['snr'].median()
            high_snr = snr_df.loc[snr_df['snr'] >= med_snr]
            high_snr_cells = high_snr.index.values.tolist()
            aa = list(set(aa) & set(high_snr_cells))

        eqs_stp, eqs_gc = _get_self_equivs(**self_kwargs, cellids=aa)
        md_stpeq = np.nanmedian(eqs_stp)
        md_gceq = np.nanmedian(eqs_gc)
        n_eq = eqs_stp.size
        u_gceq, p_gceq = st.mannwhitneyu(eqs_gc, rs_imp, alternative='two-sided')
        u_stpeq, p_stpeq = st.mannwhitneyu(eqs_stp, rs_imp, alternative='two-sided')

        eqs_x1, eqs_x2 = _get_self_equivs(**cross_kwargs, cellids=aa)
        md_x1 = np.nanmedian(eqs_x1)
        md_x2 = np.nanmedian(eqs_x2)

        sub1 = df.index.isin(ga) & df.index.isin(aa)
        sub2 = df.index.isin(sa) & df.index.isin(aa)
        eqs_sub1 = df[equiv_key][sub1].values
        eqs_sub2 = df[equiv_key][sub2].values
        md_sub1 = np.nanmedian(eqs_sub1)
        md_sub2 = np.nanmedian(eqs_sub2)

        if adjust_scores:
            if use_median:
                md_avg1 = 0.5*(md_x1 + md_x2)
                md_avg2 = 0.5*(md_sub1 + md_sub2)
                ratio = md_avg2 / md_avg1
                md_stpeq *= ratio
                md_gceq *= ratio

            else:
                eqs_avg1 = 0.5*(eqs_x1 + eqs_x2)  # between-model, halved est
                eqs_avg2 = 0.5*(eqs_sub1 + eqs_sub2)  # between-model, full data
                ratios = np.abs(eqs_avg2 / eqs_avg1)
                if use_log_ratios:
                    log_ratios = np.abs(np.log(ratios))
                    log_ratios /= log_ratios.max()
                    stp_scaled = eqs_stp + (1-eqs_stp) * log_ratios
                    gc_scaled = eqs_gc + (1-eqs_gc) * log_ratios
                else:
                    stp_scaled = eqs_stp * ratios
                    gc_scaled = eqs_gc * ratios
                md_stpeq = np.nanmedian(stp_scaled)
                md_gceq = np.nanmedian(gc_scaled)

    not_color = model_colors['LN']
    imp_color = model_colors['max']
    weights1 = [np.ones(len(rs_not))/len(rs_not)]
    weights2 = [np.ones(len(rs_imp))/len(rs_imp)]

    #n_cells = rs.shape[0]
    fig1, (a1, a2) = plt.subplots(2, 1)

    a1.hist(rs_not, bins=30, range=[-0.5, 1], weights=weights1,
            fc=faded_LN, edgecolor=dark_LN, linewidth=1)
    a2.hist(rs_imp, bins=30, range=[-0.5, 1], weights=weights2,
            fc=faded_max, edgecolor=dark_max, linewidth=1)

    a1.axes.axvline(md_not, color=dark_LN, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    a1.axes.axvline(md_imp, color=dark_max, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    a2.axes.axvline(md_not, color=dark_LN, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)
    a2.axes.axvline(md_imp, color=dark_max, linewidth=2,
                    linestyle='dashed', dashes=dash_spacing)

    if self_equiv:
        a1.axes.annotate('', xy=(md_gceq, 0), xycoords='data',
                         xytext=(md_gceq, 0.07), textcoords='data',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        a1.axes.annotate('', xy=(md_stpeq, 0), xycoords='data',
                         xytext=(md_stpeq, 0.07), textcoords='data',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        a2.axes.annotate('', xy=(md_gceq, 0), xycoords='data',
                         xytext=(md_gceq, 0.07), textcoords='data',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        a2.axes.annotate('', xy=(md_stpeq, 0), xycoords='data',
                         xytext=(md_stpeq, 0.07), textcoords='data',
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    ymin1, ymax1 = a1.get_ylim()
    ymin2, ymax2 = a2.get_ylim()
    ymax = max(ymax1, ymax2)
    a1.set_ylim(0, ymax)
    a2.set_ylim(0, ymax)

    ax_remove_box(a1)
    ax_remove_box(a2)
    plt.tight_layout()


    if equiv_key == 'equivalence':
        x_text = 'equivalence, CC(GC-LN, STP-LN)\n'
    elif equiv_key == 'partial_corr':
        x_text = 'equivalence, partial correlation\n'
    else:
        x_text = 'unknown equivalence key'
    fig3 = plt.figure(figsize=text_fig)
    text2 = ("hist: equivalence of changein prediction relative to LN model\n"
             "batch: %d\n"
             "x: %s"
             "y: cell fraction\n"
             "n not imp:  %d,  md:  %.2f\n"
             "n sig. imp:  %d,  md:  %.2f\n"
             "st.mannwhitneyu:  u:  %.4E p:  %.4E"
             % (batch, x_text, n_not, md_not, n_imp, md_imp, u, p))
    if self_equiv:
        text2 += ("\n\nSelf equivalence, n: %d\n"
                  "stp:  md:  %.2f,  u:  %.4E   p  %.4E\n"
                  "gc:   md:  %.2f,  u:  %.4E   p  %.4E\n"
                  "md sub1: %.2E\n"
                  "md sub2: %.2E\n"
                  "md x1: %.2E\n"
                  "md x2: %.2E"
                  % (n_eq, md_stpeq, u_stpeq, p_stpeq, md_gceq, u_gceq, p_gceq,
                     md_sub1, md_sub2, md_x1, md_x2))
    plt.text(0.1, 0.5, text2)

    return fig1, fig3


def equivalence_effect_size(batch, gc, stp, LN, combined, se_filter=True,
                            LN_filter=False, save_path=None, load_path=None,
                            test_limit=None, only_improvements=False,
                            legend=False, effect_key='performance_effect',
                            equiv_key='partial_corr', enable_hover=False,
                            plot_stat='r_ceiling', highlight_cells=None):

    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=se_filter,
                                           LN_filter=LN_filter)
    _, cellids, _, _, _ = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=se_filter,
                                           LN_filter=LN_filter, as_lists=False)

    if load_path is None:
        equivs = []
        partials = []
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

            C = np.hstack((np.expand_dims(gcff, 0).transpose(),
                           np.expand_dims(stpff, 0).transpose(),
                           np.expand_dims(lnff, 0).transpose()))
            partials.append(partial_corr(C)[0,1])

            equivs.append(np.corrcoef(gcff-lnff, stpff-lnff)[0, 1])
            this_gc = np.corrcoef(gcff, lnff)[0,1]
            this_stp = np.corrcoef(stpff, lnff)[0,1]
            gcs.append(this_gc)
            stps.append(this_stp)
            effects.append(1 - 0.5*(this_gc+this_stp))

        df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)

        if plot_stat == 'r_ceiling':
            plot_df = df_c
        else:
            plot_df = df_r

        models = [gc, stp, LN]
        gc_rel_all, stp_rel_all = _relative_score(plot_df, models, a)

        results = {'cellid': a[:test_limit], 'equivalence': equivs,
                   'effect_size': effects, 'corr_gc_LN': gcs,
                   'corr_stp_LN': stps, 'partial_corr': partials,
                   'performance_effect':0.5*(gc_rel_all[:test_limit]
                                             + stp_rel_all[:test_limit])}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    df = df[cellids]
    equivalence = df[equiv_key].values
    effect_size = df[effect_key].values
    r, p = st.pearsonr(effect_size, equivalence)
    improved = c
    not_improved = list(set(a) - set(c))


    fig1, ax = plt.subplots(1,1)
    ax.axes.axhline(0, color='black', linewidth=1, linestyle='dashed',
                    dashes=dash_spacing)
    ax_remove_box(ax)

    extra_title_lines = []
    if only_improvements:
        equivalence_imp = df[equiv_key][improved].values
        equivalence_not = df[equiv_key][not_improved].values
        effectsize_imp = df[effect_key][improved].values
        effectsize_not = df[effect_key][not_improved].values

        r_imp, p_imp = st.pearsonr(effectsize_imp, equivalence_imp)
        r_not, p_not = st.pearsonr(effectsize_not, equivalence_not)
        n_imp = len(improved)
        n_not = len(not_improved)
        lines = ["improved cells,  r:  %.4f,    p:  %.4E" % (r_imp, p_imp),
                 "not improved,  r:  %.4f,    p:  %.4E" % (r_not, p_not)]
        extra_title_lines.extend(lines)

#        plt.scatter(effectsize_not, equivalence_not, s=small_scatter,
#                    color=model_colors['LN'], label='no imp')
        plt.scatter(effectsize_imp, equivalence_imp, s=big_scatter,
                    color=model_colors['max'], label='sig. imp.')
        if enable_hover:
            mplcursors.cursor(ax, hover=True).connect(
                    "add",
                    lambda sel: sel.annotation.set_text(
                            improved[sel.target.index])
                    )
        if legend:
            plt.legend()
    else:
        plt.scatter(effect_size, equivalence, s=big_scatter, color='black')
        if enable_hover:
            mplcursors.cursor(ax, hover=True).connect(
                    "add",
                    lambda sel: sel.annotation.set_text(a[sel.target.index])
                    )

    if highlight_cells is not None:
        highlights = [df[df.index == h] for h in highlight_cells]
        equiv_highlights = [df[equiv_key].values for df in highlights]
        effect_highlights = [df[effect_key].values for df in highlights]

        for eq, eff in zip(equiv_highlights, effect_highlights):
            plt.scatter(eff, eq, s=big_scatter*10, facecolors='none',
                        edgecolors='black', linewidths=1)

    #plt.ylabel('Equivalence:  CC(GC-LN, STP-LN)')
    #plt.xlabel('Effect size:  1 - 0.5*(CC(GC,LN) + CC(STP,LN))')
    if equiv_key == 'equivalence':
        y_text = 'equivalence, CC(GC-LN, STP-LN)\n'
    elif equiv_key == 'partial_corr':
        y_text = 'equivalence, partial correlation\n'
    else:
        y_text = 'unknown equivalence key'

    if effect_key == 'effect_size':
        x_text = 'effect size: 1 - 0.5*(CC(GC,LN) + CC(STP,LN))'
    elif effect_key == 'performance_effect':
        x_text = 'effect size: 0.5*(rGC-rLN + rSTP-rLN)'
    else:
        x_text = 'unknown effect key'

    text = ("scatter: Equivalence of Change to Predicted PSTH\n"
            "batch: %d\n"
            "vs Effect Size\n"
            "all cells,  r:  %.4f,    p:  %.4E\n"
            "y: %s"
            "x: %s"
            % (batch, r, p, y_text, x_text))
    for ln in extra_title_lines:
        text += "\n%s" % ln

    plt.tight_layout()


#    fig2 = plt.figure()
#    md = np.nanmedian(equivalence)
#    n_cells = equivalence.shape[0]
#    plt.hist(equivalence, bins=30, range=[-0.5, 1], histtype='bar',
#             color=model_colors['combined'], edgecolor='black', linewidth=1)
#    plt.plot(np.array([0,0]), np.array(fig2.axes[0].get_ylim()), 'k--',
#             linewidth=2, dashes=dash_spacing)
#    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
#             ha='left', va='top', transform=fig2.axes[0].transAxes)
    #plt.xlabel('CC, GC-LN vs STP-LN')
    #plt.title('Equivalence of Change in Prediction Relative to LN Model')

    fig3 = plt.figure()
    plt.text(0.1, 0.75, text, va='top')
    #plt.text(0.1, 0.25, text2, va='top')


    return fig1, fig3


def equiv_effect_cells(batch, gc, stp, LN, combined, save_path=None,
                       load_path=None,
                       test_limit=None, only_improvements=False,
                       effect_key='performance_effect', equiv_key='partial_corr',
                       plot_stat='r_ceiling'):

    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined)
    improved = c
    df = pd.read_pickle(load_path)

    equivalence_imp = df[equiv_key][improved]
    effectsize_imp = df[effect_key][improved]
    print("effect size above 0.15:")
    big_effect = effectsize_imp[effectsize_imp > 0.15].index.values.tolist()
    print(big_effect)
    print("\nequivalence above 0.5:")
    big_equiv = equivalence_imp[equivalence_imp > 0.5].index.values.tolist()
    print(big_equiv)
    print("\nequivalence below 0.1:")
    small_equiv = equivalence_imp[equivalence_imp < 0.1].index.values.tolist()
    print(small_equiv)

    return big_effect, big_equiv, small_equiv


def equiv_vs_self(cellid, batch, modelname, LN_model, random_seed=1234):
    # evaluate old fit just to get est/val already split up
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)

    # further divide est into two datasets
    # (how to do this?  pick from epochs randomly?)
    est = ctx['est']
    val = ctx['val']
    epochs = est['stim'].epochs
    stims = np.array(ep.epoch_names_matching(epochs, 'STIM_'))
    indices = np.linspace(0, len(stims)-1, len(stims), dtype=np.int)

    st0 = np.random.get_state()
    np.random.seed(random_seed)
    set1_idx = np.random.choice(indices, round(len(stims)/2),
                                replace=False)
    np.random.set_state(st0)

    mask = np.zeros_like(stims, np.bool)
    mask[set1_idx] = True
    set1_stims = stims[mask].tolist()
    set2_stims = stims[~mask].tolist()

    est1, est2 = est.split_by_epochs(set1_stims, set2_stims)

    # re-fit on the smaller est sets
    # (will have to re-fit LN model as well?)
    # also have to remove -sev- from modelname and add est-val in manually
    ctx1 = {'est': est1, 'val': val.copy()}
    ctx2 = {'est': est2, 'val': val.copy()}
    LN_ctx1 = copy.deepcopy(ctx1)
    LN_ctx2 = copy.deepcopy(ctx2)
#    modelname = modelname.replace('-sev', '')
#    LN_model = LN_model.replace('-sev', '')
    tm = 'none_'+'_'.join(modelname.split('_')[1:])
    lm = 'none_'+'_'.join(LN_model.split('_')[1:])

    # test model, est1
    xfspec = xhelp.generate_xforms_spec(modelname=tm)
    ctx, _ = xforms.evaluate(xfspec, context=ctx1)
    test_pred1 = ctx['val']['pred'].as_continuous().flatten()

    # test model, est2
    xfspec = xhelp.generate_xforms_spec(modelname=tm)
    ctx, _ = xforms.evaluate(xfspec, context=ctx2)
    test_pred2 = ctx['val']['pred'].as_continuous().flatten()

    # LN model, est1
    xfspec = xhelp.generate_xforms_spec(modelname=lm)
    ctx, _ = xforms.evaluate(xfspec, context=ctx1)
    LN_pred1 = ctx['val']['pred'].as_continuous().flatten()

    # LN model, est2
    xfspec = xhelp.generate_xforms_spec(modelname=lm)
    ctx, _ = xforms.evaluate(xfspec, context=ctx2)
    LN_pred2 = ctx['val']['pred'].as_continuous().flatten()

    # test equivalence on the new fits
    C1 = np.hstack((np.expand_dims(test_pred1, 0).transpose(),
                    np.expand_dims(test_pred2, 0).transpose(),
                    np.expand_dims(LN_pred1, 0).transpose()))
    p1 = partial_corr(C1)[0,1]


    C2 = np.hstack((np.expand_dims(test_pred1, 0).transpose(),
                    np.expand_dims(test_pred2, 0).transpose(),
                    np.expand_dims(LN_pred2, 0).transpose()))
    p2 = partial_corr(C2)[0,1]

    return 0.5*(p1+p2)


# https://gist.github.com/fabianp/9396204419c7b638d38f
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def partial_corr_noise_test():
    C1 = np.ones((100, 3))  # true random noise
    C3 = C1.copy()
    # samples [0,1), shift and scale to [-0.05, 0.05)
    noise1 = (np.random.rand(100,) - 0.5)/10
    noise2 = (np.random.rand(100,) - 0.5)/10
    C1[:,0] += noise1
    C1[:,1] += noise2


    # noise plus some mostly matching bits (should be high partial correlation)
    # - how much is the noise dragging it down?
    C2 = C1.copy()
    point_increase_one = np.zeros(100,)
    point_increase_one[30] = 0.5
    point_increase_one[31] = 0.4
    point_increase_two = np.zeros(100,)
    point_increase_two[30:33] = 0.3
    point_increase_two[60] = 0.2
    C2[:,0] += point_increase_one
    C2[:,1] += point_increase_two

    # just the matching bits (should be even higher partial correlation)
    C3[:,0] += point_increase_one
    C3[:,1] += point_increase_two

    p_cor1 = partial_corr(C1)[0,1]
    p_cor2 = partial_corr(C2)[0,1]
    p_cor3 = partial_corr(C3)[0,1]

    fig, (a1,a2, a3) = plt.subplots(3, 1, sharey=True, figsize=(6,8))
    for i in range(3):
        a1.plot(C1[:,i], label=f'variable {i}')
        a2.plot(C2[:,i])
        a3.plot(C3[:,i])
    a1.set_title(f'random noise, pc: {p_cor1}')
    a2.set_title(f'point noise, pc: {p_cor2}')
    a3.set_title(f'point, pc: {p_cor3}')
    plt.sca(a1)
    plt.legend()
    plt.tight_layout()

    return C1, C2, C3
