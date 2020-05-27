import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.patheffects as pe
import numpy as np
import scipy.stats as st

import nems.db as nd
from nems.utils import ax_remove_box
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             get_valid_improvements,
                                             adjustFigAspect,
                                             improved_cells_to_list)
from nems_lbhb.gcmodel.figures.definitions import *

#gc_color = wsu_gray
#stp_color = wsu_gray
#ln_color = wsu_gray_light
#gc_stp_color = ohsu_navy

plt.rcParams.update(params) # loaded from definitions
dropped_cell_color = wsu_crimson
scatter_color = wsu_gray


def single_scatter(batch, gc, stp, LN, combined, compare, plot_stat='r_ceiling',
                   legend=False):
    all_batch_cells = nd.get_batch_cells(batch, as_list=True)
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=True, LN_filter=False,
                                           as_lists=True)

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    improved = c
    not_improved = list(set(a) - set(c))
    models = [gc, stp, LN, combined]
    names = ['gc', 'stp', 'LN', 'combined']
    m1 = models[compare[0]]
    m2 = models[compare[1]]
    name1 = names[compare[0]]
    name2 = names[compare[1]]
    n_batch = len(all_batch_cells)
    n_all = len(a)
    n_imp = len(improved)
    n_not_imp = len(not_improved)

    m1_scores = plot_df[m1][not_improved]
    m1_scores_improved = plot_df[m1][improved]
    m2_scores = plot_df[m2][not_improved]
    m2_scores_improved = plot_df[m2][improved]

    fig = plt.figure()
    plt.plot([0,1],[0,1], color='black', linewidth=1, linestyle='dashed',
             dashes=dash_spacing)
    plt.scatter(m1_scores, m2_scores, s=small_scatter, label='no imp.',
                color=model_colors['LN'])
                #color='none',
                #edgecolors='black', linewidth=0.35)
    plt.scatter(m1_scores_improved, m2_scores_improved, s=big_scatter, label='sig. imp.',
                color=model_colors['max'])
                #color='none',
                #edgecolors='black', linewidth=0.35)
    ax_remove_box()

    if legend:
        plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.axes().set_aspect('equal')

    fig2 = plt.figure(figsize=text_fig)
    plt.text(0.1, 0.5, "batch %d\n"
             "%d/%d  auditory/total cells\n"
             "%d no improvements\n"
             "%d at least one improvement\n"
             "stat: %s, x: %s, y: %s"
             % (batch, n_all, n_batch, n_not_imp, n_imp, plot_stat,
                name1, name2))

    return fig, fig2


def performance_scatters(batch, gc, stp, LN, combined,
                         se_filter=True, LN_filter=False, manual_cellids=None,
                         plot_stat='r_ceiling', show_dropped=True,
                         color_improvements=False):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

    '''
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
#    cellids, under_chance, less_LN = get_filtered_cellids(batch, gc, stp,
#                                                          LN, combined,
#                                                          se_filter,
#                                                          LN_filter)
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=se_filter,
                                           LN_filter=LN_filter)
    cellids = a

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids


    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    gc_test = plot_df[gc][cellids]
    gc_test_under_chance = plot_df[gc][under_chance]
    stp_test = plot_df[stp][cellids]
    stp_test_under_chance = plot_df[stp][under_chance]
    ln_test = plot_df[LN][cellids]
    ln_test_under_chance = plot_df[LN][under_chance]
    gc_stp_test = plot_df[combined][cellids]
    gc_stp_test_under_chance = plot_df[combined][under_chance]

    # Row 1 (vs LN)
    fig1, ax = plt.subplots(1,1)
    ax.scatter(ln_test, gc_test, c=scatter_color, s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(gc_test_under_chance, ln_test_under_chance, c=dropped_cell_color, s=20)
    #ax.scatter(gc_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('LN vs GC')
    ax.set_ylabel('GC')
    ax.set_xlabel('LN')

    fig2, ax = plt.subplots(1,1)
    ax.scatter(ln_test, stp_test, c=scatter_color, s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(stp_test_under_chance, ln_test_under_chance, c=dropped_cell_color, s=20)
    #ax.scatter(stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('LN vs STP')
    ax.set_ylabel('STP')
    ax.set_xlabel('LN')

    fig3, ax = plt.subplots(1, 1)
    ax.scatter(ln_test, gc_stp_test, c=scatter_color, s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(ln_test_under_chance, gc_stp_test_under_chance, c=dropped_cell_color, s=20)
    #ax.scatter(gc_stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('LN vs GC + STP')
    ax.set_ylabel('GC + STP')
    ax.set_xlabel('LN')

    # Row 2 (head-to-head)
    fig4, ax = plt.subplots(1, 1)
    ax.scatter(gc_test, stp_test, c=scatter_color, s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(gc_test_under_chance, stp_test_under_chance, c=dropped_cell_color, s=20)
    #ax.scatter(gc_test_less_LN, stp_test_less_LN, c='blue', s=20)
    ax.set_title('GC vs STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('STP')

    fig5, ax = plt.subplots(1, 1)
    ax.scatter(gc_test, gc_stp_test, c=scatter_color, s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(gc_test_under_chance, gc_stp_test_under_chance, c=dropped_cell_color, s=20)
    #ax.scatter(gc_test_less_LN, gc_stp_test_less_LN, c='blue', s=20)
    ax.set_title('GC vs GC + STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('GC + STP')

    fig6, ax = plt.subplots(1, 1)
    ax.scatter(stp_test, gc_stp_test, c=scatter_color, s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(stp_test_under_chance, gc_stp_test_under_chance, c=dropped_cell_color, s=20)
    #ax.scatter(stp_test_less_LN, gc_stp_test_less_LN, c='blue', s=20)
    ax.set_title('STP vs GC + STP')
    ax.set_xlabel('STP')
    ax.set_ylabel('GC + STP')

    #plt.tight_layout()

    return fig1, fig2, fig3, fig4, fig5, fig6


def gc_stp_scatter(batch, gc, stp, LN, combined,
                   se_filter=True, LN_filter=False, manual_cellids=None,
                   plot_stat='r_ceiling'):

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          LN_filter)

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids


    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    n_cells = len(cellids)
    n_under_chance = len(under_chance) if under_chance != cellids else 0
    n_less_LN = len(less_LN) if less_LN != cellids else 0

    gc_test = plot_df[gc][cellids]
    gc_test_under_chance = plot_df[gc][under_chance]

    stp_test = plot_df[stp][cellids]
    stp_test_under_chance = plot_df[stp][under_chance]

    fig = plt.figure()
    plt.scatter(gc_test, stp_test, c=wsu_gray, s=20)
    ax = fig.axes[0]
    plt.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=1, dashes=dash_spacing)
    plt.scatter(gc_test_under_chance, stp_test_under_chance, c=dropped_cell_color, s=20)
    plt.title('GC vs STP')
    plt.xlabel('GC')
    plt.ylabel('STP')
    if se_filter:
        plt.text(0.90, -0.05, 'all = %d' % (n_cells+n_under_chance+n_less_LN),
                ha='right', va='bottom')
        plt.text(0.90, 0.00, 'n = %d' % n_cells, ha='right', va='bottom')
        plt.text(0.90, 0.05, 'uc = %d' % n_under_chance, ha='right', va='bottom',
                color=dropped_cell_color)


def combined_vs_max(batch, gc, stp, LN, combined, se_filter=True,
                    LN_filter=False, plot_stat='r_ceiling',
                    legend=False):

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
#    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
#                                                          LN, combined,
#                                                          se_filter,
#                                                          LN_filter)
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=se_filter,
                                           LN_filter=LN_filter)
    improved = c
    not_improved = list(set(a) - set(c))

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    gc_not = plot_df[gc][not_improved]
    gc_imp = plot_df[gc][improved]
    #gc_test_under_chance = plot_df[gc][under_chance]
    stp_not = plot_df[stp][not_improved]
    stp_imp = plot_df[stp][improved]
    #stp_test_under_chance = plot_df[stp][under_chance]
    #ln_test = plot_df[LN][cellids]
    gc_stp_not = plot_df[combined][not_improved]
    gc_stp_imp = plot_df[combined][improved]
    max_not = np.maximum(gc_not, stp_not)
    max_imp = np.maximum(gc_imp, stp_imp)
    #gc_stp_test_rel = gc_stp_test - ln_test
    #max_test_rel = np.maximum(gc_test, stp_test) - ln_test

    fig1 = plt.figure()
    c_not = model_colors['LN']
    c_imp = model_colors['max']
    plt.scatter(max_not, gc_stp_not, c=c_not, s=small_scatter, label='no imp.')
    plt.scatter(max_imp, gc_stp_imp, c=c_imp, s=big_scatter, label='sig. imp.')
    ax = fig1.axes[0]
    plt.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=1,
             dashes=dash_spacing)
    if legend:
        plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.axes().set_aspect('equal')
    ax_remove_box()

    fig2 = plt.figure(figsize=text_fig)
    text = ("batch: %d\n"
            "x: Max(GC,STP)\n"
            "y: GC+STP\n" % batch)
    plt.text(0.1, 0.5, text)


    return fig1, fig2


def performance_bar(batch, gc, stp, LN, combined, se_filter=True,
                    LN_filter=False, manual_cellids=None, abbr_yaxis=False,
                    plot_stat='r_ceiling', y_adjust=0.05, manual_y=None,
                    only_improvements=False, show_text_labels=False):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

    '''

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
#    cellids, under_chance, less_LN = get_filtered_cellids(batch, gc, stp,
#                                                          LN, combined,
#                                                          se_filter,
#                                                          LN_filter)
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           se_filter=se_filter,
                                           LN_filter=LN_filter)
    cellids = a

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids
    elif only_improvements:
        e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                               as_lists=True)
        cellids = e

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    n_cells = len(cellids)
    gc_test = plot_df[gc][cellids]
    stp_test = plot_df[stp][cellids]
    ln_test = plot_df[LN][cellids]
    gc_stp_test = plot_df[combined][cellids]
    #max_test = np.maximum(gc_test, stp_test)

    gc = np.median(gc_test.values)
    stp = np.median(stp_test.values)
    ln = np.median(ln_test.values)
    gc_stp = np.median(gc_stp_test.values)
    #maximum = np.median(max_test)
    largest = max(gc, stp, ln, gc_stp)#, maximum)

    colors = [model_colors[k] for k in ['LN', 'gc', 'stp', 'combined']]#, 'max']]
    #fig = plt.figure(figsize=(15, 12))
    fig = plt.figure()
    plt.bar([1, 2, 3, 4], [ln, gc, stp, gc_stp],# maximum],
            color=colors,
            edgecolor="black", linewidth=1)
    plt.xticks([1, 2, 3, 4, 5], ['LN', 'GC', 'STP', 'GC+STP'])#, 'Max(GC,STP)'])
    if abbr_yaxis:
        if manual_y:
            lower, upper = manual_y
        else:
            lower = np.floor(10*min(gc, stp, ln, gc_stp))/10
            upper = np.ceil(10*max(gc, stp, ln, gc_stp))/10 + y_adjust
        plt.ylim(ymin=lower, ymax=upper)
    else:
        plt.ylim(ymax=largest*1.4)
    common_kwargs = {'color': 'white', 'horizontalalignment': 'center'}
    if abbr_yaxis:
        y_text = 0.5*(lower + min(gc, stp, ln, gc_stp))
    else:
        y_text = 0.2
    if show_text_labels:
        for i, m in enumerate([ln, gc, stp, gc_stp]):#, maximum]):
            t = plt.text(i+1, y_text, "%0.04f" % m, **common_kwargs)
        #t.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, xmax - 0.35)
    ax_remove_box()
    plt.tight_layout()

    fig2 = plt.figure(figsize=text_fig)
    text = "Median Performance, batch: %d\, n:%d" % (batch, n_cells)
    plt.text(0.1, 0.5, text)

    return fig, fig2


def performance_table(batch1, gc, stp, LN, combined, batch2,
                      plot_stat='r_ceiling', height_scaling=3):
    # 4 tables: batch 289 and 263 all cells / improved cells
    df_r1, df_c1, df_e1 = get_dataframes(batch1, gc, stp, LN, combined)
    cellids1, under_chance1, less_LN1 = get_filtered_cellids(df_r1, df_e1, gc,
                                                             stp, LN, combined)

    df_r2, df_c2, df_e2 = get_dataframes(batch2, gc, stp, LN, combined)
    cellids2, under_chance2, less_LN2 = get_filtered_cellids(df_r2, df_e2, gc,
                                                             stp, LN, combined)
    e1, a1, _, _, _ = improved_cells_to_list(batch1, gc, stp, LN, combined)
    e2, a2, _, _, _ = improved_cells_to_list(batch2, gc, stp, LN, combined)

    if plot_stat == 'r_ceiling':
        df1 = df_c1
        df2 = df_c2
    else:
        df1 = df_r1
        df2 = df_c2

    models = [LN, gc, stp, combined]
    a289, a289_stats = _make_table(df1, df_e1, models, a1)
    a263, a263_stats = _make_table(df2, df_e2, models, a2)
    i289, i289_stats = _make_table(df1, df_e1, models, e1)
    i263, i263_stats = _make_table(df2, df_e2, models, e2)
    model_names = ['LN', 'GC', 'STP', 'GC+STP', 'Max(GC,STP)']

    fig1, ((a1, a2), (a3, a4)) = plt.subplots(2,2)
    fig2, ((a5, a6), (a7, a8)) = plt.subplots(2,2)
    fig1.patch.set_visible(False)
    fig2.patch.set_visible(False)
    iters = zip([a1, a2, a3, a4],
                [a5, a6, a7, a8],
                [a289, a263, i289, i263],
                [a289_stats, a263_stats, i289_stats, i263_stats],
                ['Natural stimuli, all cells', 'Voc. in noise, all cells',
                 'Natural stimuli, nonlinear cells',
                 'Voc. in noise, nonlinear cells'])

    for ax1, ax2, table, stats, title in iters:
        ax1.axis('off')
        ax1.axis('tight')
        table1 = ax1.table(cellText=table, colLabels=model_names, rowLabels=model_names,
                         loc='center', cellLoc='center', rowLoc='center')
        table1_cells = table1.properties()['child_artists']
        for c1 in table1_cells:
            current_height1 = c1.get_height()
            c1.set_height(current_height1*height_scaling)
        ax1.set_title(title)

        ax2.axis('off')
        ax2.axis('tight')
        row_labels = ['mean', 'median', 'std err']
        table_text = np.empty((len(row_labels), len(model_names)), dtype='U7')
        for i, _ in enumerate(row_labels):
            for j, _ in enumerate(model_names):
                s = stats[j][i]
                text = '%.5f' % s
                table_text[i][j] = text
        table2 = ax2.table(cellText=table_text, colLabels=model_names,
                         rowLabels=row_labels, loc='center', cellLoc='center',
                         rowLoc='center')
        table2_cells = table2.properties()['child_artists']
        for c2 in table2_cells:
            current_height2 = c2.get_height()
            c2.set_height(current_height2*height_scaling)
        ax2.set_title(title)

    fig1.tight_layout()
    fig2.tight_layout()


    return fig2, fig1


def _make_table(df, df_e, models, cells):
    vals = []
    stats = []
    for m in models:
        v = df[m][cells].values
        vals.append(v)
        mean = np.mean(v)
        median = np.median(v)
        se = st.sem(v)
        stats.append((mean, median, se))

    max_v = np.maximum(vals[0], vals[1])  # max(gc, stp)
    vals.append(max_v)
    max_mean = np.mean(max_v)
    max_median = np.median(max_v)
    max_se = st.sem(max_v)
    stats.append((max_mean, max_median, max_se))

    table = np.empty((len(vals), len(vals)), dtype='U7')
    for i, v_one in enumerate(vals):
        for j, v_two in enumerate(vals):
            if j == i:
                # if indices equal, on diagonal so no comparison
                table[i][j] = '       '
            elif j > i:
                # if j is larger below diagonal so no comparison
                table[i][j] = '       '
            else:
                # if j is smaller, above diagonal so run wilcoxon test and
                # get p-value and round to nearest power of 10 for small p's
                p = st.wilcoxon(v_one, v_two)[1]
                if p < 0.001:
                    p = int(np.ceil(np.log10(p)))
                    if p <= -100:
                        p = -99
                    if p <= -10:
                        pad = ''
                    else:
                        pad = '0'
                    text = '< 1E%s%d' % (pad, p)
                else:
                    text = '  %.3f' % p
                table[i][j] = text

    return table, stats


def relative_bar_comparison(batch1, batch2, gc, stp, LN, combined,
                            se_filter=True, ln_filter=False,
                            plot_stat='r_ceiling', only_improvements=False,
                            good_ln=0.0):
    raise ValueError('fix cellid filters before using me')

    df_r1, df_c1, df_e1 = get_dataframes(batch1, gc, stp, LN, combined)
    cellids1, under_chance1, less_LN1 = get_filtered_cellids(df_r1, df_e1, gc,
                                                             stp, LN, combined,
                                                             se_filter,
                                                             ln_filter)

    df_r2, df_c2, df_e2 = get_dataframes(batch2, gc, stp, LN, combined)
    cellids2, under_chance2, less_LN2 = get_filtered_cellids(df_r2, df_e2, gc,
                                                             stp, LN, combined,
                                                             se_filter,
                                                             ln_filter)

    if only_improvements:
        # only use cells for which there was a significant improvement
        # to one or more models
        e1, n1, g1, s1, c1 = improved_cells_to_list(batch1, gc, stp, LN,
                                                    combined, good_ln=good_ln)
        filter1 = list(set(e1) | set(g1) | set(s1) | set(c1))

        e2, n2, g2, s2, c2 = improved_cells_to_list(batch2, gc, stp, LN,
                                                    combined, good_ln=good_ln)
        filter2 = list(set(e2) | set(g2) | set(s2) | set(c2))

        cellids1 = [c for c in cellids1 if c in filter1]
        cellids2 = [c for c in cellids2 if c in filter2]


    if plot_stat == 'r_ceiling':
        plot_df1 = df_c1
        plot_df2 = df_c2
    else:
        plot_df1 = df_r1
        plot_df2 = df_r2

    n_cells1 = len(cellids1)
    gc_test1 = plot_df1[gc][cellids1]
    stp_test1 = plot_df1[stp][cellids1]
    ln_test1 = plot_df1[LN][cellids1]
    gc_stp_test1 = plot_df1[combined][cellids1]

    n_cells2 = len(cellids2)
    gc_test2 = plot_df2[gc][cellids2]
    stp_test2 = plot_df2[stp][cellids2]
    ln_test2 = plot_df2[LN][cellids2]
    gc_stp_test2 = plot_df2[combined][cellids2]

    gc_rel1 = gc_test1 - ln_test1
    stp_rel1 = stp_test1 - ln_test1
    gc_stp_rel1 = gc_stp_test1 - ln_test1
    gc1 = np.mean(gc_rel1.values)
    stp1 = np.mean(stp_rel1.values)
    gc_stp1 = np.mean(gc_stp_rel1.values)
    #largest1 = max(gc1, stp1, gc_stp1)

    gc_rel2 = gc_test2 - ln_test2
    stp_rel2 = stp_test2 - ln_test2
    gc_stp_rel2 = gc_stp_test2 - ln_test2
    gc2 = np.mean(gc_rel2.values)
    stp2 = np.mean(stp_rel2.values)
    gc_stp2 = np.mean(gc_stp_rel2.values)
    #largest2 = max(gc2, stp2, gc_stp2)

    fig = plt.figure(figsize=(15, 12))
    plt.bar([1, 2, 3, 4, 5, 6], [gc1, stp1, gc_stp1, gc2, stp2, gc_stp2],
            #color=['purple', 'green', 'gray', 'blue'])
            color=[gc_color, stp_color, gc_stp_color,
                   gc_color, stp_color, gc_stp_color],
            edgecolor="black", linewidth=2)
    plt.xticks([1, 2, 3, 4, 5], ['GC %d' % batch1, 'STP', 'GC+STP',
                                 'GC %d' % batch2, 'STP', 'GC+STP'])
#    if abbr_yaxis:
#        lower = np.floor(10*min(gc, stp, ln, gc_stp))/10
#        upper = np.ceil(10*max(gc, stp, ln, gc_stp))/10 + y_adjust
#        plt.ylim(ymin=lower, ymax=upper)
#    else:
#        plt.ylim(ymax=largest*1.4)

    common_kwargs = {'color': 'white', 'horizontalalignment': 'center'}
#    if abbr_yaxis:
#        y_text = 0.5*(lower + min(gc, stp, ln, gc_stp))
#    else:
    y_text = 0.2
    plt.text(1, y_text, "%0.04f" % gc1, **common_kwargs)
    plt.text(2, y_text, "%0.04f" % stp1, **common_kwargs)
    plt.text(3, y_text, "%0.04f" % gc_stp1, **common_kwargs)
    plt.text(4, y_text, "%0.04f" % gc2, **common_kwargs)
    plt.text(5, y_text, "%0.04f" % stp2, **common_kwargs)
    plt.text(6, y_text, "%0.04f" % gc_stp2, **common_kwargs)
    plt.title("Mean Relative (to LN) Performance for GC, STP, and GC+STP\n"
              "batch %d, n: %d   vs   batch %d, n: %d"
              % (batch1, n_cells1, batch2, n_cells2))

    return fig



def significance(batch, gc, stp, LN, combined,
                 manual_cellids=None, plot_stat='r_ceiling',
                 include_legend=True, only_improvements=False):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

    '''
    # NOTE: The comparison of max(gc, stp) to gc/stp should be
    #       ignored. They're known to be different by definition
    #       and there's a bug in the scipy code that causes the
    #       W-statistic to be reported as 0.
    #       This happens because all of the differences are one-sided
    #       and the scipy code takes the minimum of either positive
    #       or negative differences, one of which will always be 0

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined)
    cellids = a

    gc_test = df_r[gc][cellids]
    stp_test = df_r[stp][cellids]
    ln_test = df_r[LN][cellids]
    gc_stp_test = df_r[combined][cellids]
    max_test = np.maximum(gc_test, stp_test)

    modelnames = ['GC', 'STP', 'LN', 'GC + STP', 'Max(GC,STP)']
    models = {'GC': gc_test, 'STP': stp_test, 'LN': ln_test,
              'GC + STP': gc_stp_test, 'Max(GC,STP)': max_test}
    array = np.ndarray(shape=(len(modelnames), len(modelnames)), dtype=float)

    for i, m_one in enumerate(modelnames):
        for j, m_two in enumerate(modelnames):
            # get series of values corresponding to selected measure
            # for each model
            series_one = models[m_one]
            series_two = models[m_two]
            # TODO: no reason to convert these to lists anymore?
            first = series_one.tolist()
            second = series_two.tolist()
            w, p = st.wilcoxon(first, second)
            if j == i:
                # if indices equal, on diagonal so no comparison
                array[i][j] = 0.00
            elif j > i:
                # if j is larger, below diagonal so get mean difference
                array[i][j] = w
            else:
                # if j is smaller, above diagonal so run t-test and
                # get p-value
                array[i][j] = p

    xticks = range(len(modelnames))
    yticks = xticks
    minor_xticks = np.arange(-0.5, len(modelnames), 1)
    minor_yticks = np.arange(-0.5, len(modelnames), 1)

    fig = plt.figure(figsize=(12,12))
    ax = plt.gca()

    # ripped from stackoverflow. adds text labels to the grid
    # at positions i,j (model x model)  with text z (value of array at i, j)
    for (i, j), z in np.ndenumerate(array):
        if j == i:
            color="#EBEBEB"
        elif j > i:
            color="#368DFF"
        else:
            if array[i][j] < 0.001:
                color="#74E572"
            elif array[i][j] < 0.01:
                color="#59AF57"
            elif array[i][j] < 0.05:
                color="#397038"
            else:
                color="#ABABAB"

        ax.add_patch(mpatch.Rectangle(
                xy=(j-0.5, i-0.5), width=1.0, height=1.0, angle=0.0,
                facecolor=color, edgecolor='black',
                ))
        if j == i:
            # don't draw text for diagonal
            continue
#        formatting = '{:.04f}'
#        if z <= 0.0001:
#            formatting = '{:.2E}'
        formatting = '{:.2E}'
        ax.text(
                j, i, formatting.format(z), ha='center', va='center',
                )

    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticks(yticks)
    ax.set_yticklabels(modelnames, fontsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(modelnames, fontsize=10, rotation="vertical")
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(minor_xticks, minor=True)
    ax.grid(b=False)
    ax.grid(which='minor', color='b', linestyle='-', linewidth=0.75)
    title = "Wilcoxon Signed Test\nOnly improvements?:  %s" % only_improvements
    ax.set_title(title, ha='center', fontsize = 14)

    if include_legend:
        blue_patch = mpatch.Patch(
                color='#368DFF', label='W statistic', edgecolor='black'
                )
        p001_patch = mpatch.Patch(
                color='#74E572', label='P < 0.001', edgecolor='black'
                )
        p01_patch = mpatch.Patch(
                color='#59AF57', label='P < 0.01', edgecolor='black'
                )
        p05_patch = mpatch.Patch(
                color='#397038', label='P < 0.05', edgecolor='black'
                )
        nonsig_patch = mpatch.Patch(
                color='#ABABAB', label='Not Significant', edgecolor='black',
                )

        plt.legend(
                #bbox_to_anchor=(0., 1.02, 1., .102), ncol=2,
                bbox_to_anchor=(1.05, 1), ncol=1,
                loc=2, handles=[
                        p05_patch, p01_patch, p001_patch,
                        nonsig_patch, blue_patch,
                        ]
                )
    plt.tight_layout()

    return fig
