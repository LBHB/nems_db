import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
import scipy.stats as st

import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             get_valid_improvements,
                                             adjustFigAspect)


#gc_color = '#69657C'
#stp_color = '#394B5E'
#ln_color = '#62838C'
#gc_stp_color = '#215454'

gc_color = wsu_gray
stp_color = wsu_gray
ln_color = wsu_gray_light
gc_stp_color = ohsu_navy

plt.rcParams.update(params) # loaded from definitions

def performance_scatters(batch, gc, stp, LN, combined,
                         se_filter=True, LN_filter=False, manual_cellids=None,
                         plot_stat='r_ceiling'):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

    '''
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

    gc_test = plot_df[gc][cellids]
    gc_test_under_chance = plot_df[gc][under_chance]
    stp_test = plot_df[stp][cellids]
    stp_test_under_chance = plot_df[stp][under_chance]
    ln_test = plot_df[LN][cellids]
    ln_test_under_chance = plot_df[LN][under_chance]
    gc_stp_test = plot_df[combined][cellids]
    gc_stp_test_under_chance = plot_df[combined][under_chance]

    # Row 1 (vs LN)
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.scatter(ln_test, gc_test, c='black', s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(gc_test_under_chance, ln_test_under_chance, c=wsu_crimson, s=20)
    #ax.scatter(gc_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('LN vs GC')
    ax.set_ylabel('GC')
    ax.set_xlabel('LN')
#    if ln_filter or se_filter:
#        ax.text(0.90, -0.10, 'all = %d' % (n_cells+n_under_chance+n_less_LN),
#                ha='right', va='bottom')
#    ax.text(0.90, 0.00, 'n = %d' % n_cells, ha='right', va='bottom')
#    if se_filter:
#        ax.text(0.90, 0.10, 'uc = %d' % n_under_chance, ha='right', va='bottom',
#                color='red')
#    if ln_filter:
#        ax.text(0.90, 0.20, '<ln = %d' % n_less_LN, ha='right', va='bottom',
#                color='blue')

    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.scatter(ln_test, stp_test, c='black', s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(stp_test_under_chance, ln_test_under_chance, c=wsu_crimson, s=20)
    #ax.scatter(stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('LN vs STP')
    ax.set_ylabel('STP')
    ax.set_xlabel('LN')

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(gc_stp_test, ln_test, c='black', s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(gc_stp_test_under_chance, ln_test_under_chance, c=wsu_crimson, s=20)
    #ax.scatter(gc_stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('GC + STP vs LN')
    ax.set_xlabel('GC + STP')
    ax.set_ylabel('LN')

    # Row 2 (head-to-head)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(gc_test, stp_test, c='black', s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(gc_test_under_chance, stp_test_under_chance, c=wsu_crimson, s=20)
    #ax.scatter(gc_test_less_LN, stp_test_less_LN, c='blue', s=20)
    ax.set_title('GC vs STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('STP')

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(gc_test, gc_stp_test, c='black', s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(gc_test_under_chance, gc_stp_test_under_chance, c=wsu_crimson, s=20)
    #ax.scatter(gc_test_less_LN, gc_stp_test_less_LN, c='blue', s=20)
    ax.set_title('GC vs GC + STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('GC + STP')

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(stp_test, gc_stp_test, c='black', s=20)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=2, dashes=dash_spacing)
    ax.scatter(stp_test_under_chance, gc_stp_test_under_chance, c=wsu_crimson, s=20)
    #ax.scatter(stp_test_less_LN, gc_stp_test_less_LN, c='blue', s=20)
    ax.set_title('STP vs GC + STP')
    ax.set_xlabel('STP')
    ax.set_ylabel('GC + STP')

    #plt.tight_layout()

    #return fig


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

    fig = plt.figure(figsize=figsize)
    plt.scatter(gc_test, stp_test, c=wsu_gray, s=20)
    ax = fig.axes[0]
    plt.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=1, dashes=dash_spacing)
    plt.scatter(gc_test_under_chance, stp_test_under_chance, c=wsu_crimson, s=20)
    plt.title('GC vs STP')
    plt.xlabel('GC')
    plt.ylabel('STP')
    if se_filter:
        plt.text(0.90, -0.05, 'all = %d' % (n_cells+n_under_chance+n_less_LN),
                ha='right', va='bottom')
        plt.text(0.90, 0.00, 'n = %d' % n_cells, ha='right', va='bottom')
        plt.text(0.90, 0.05, 'uc = %d' % n_under_chance, ha='right', va='bottom',
                color=wsu_crimson)


def combined_vs_max(batch, gc, stp, LN, combined, se_filter=True,
                    LN_filter=False, plot_stat='r_ceiling'):

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          LN_filter)


    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    gc_test = plot_df[gc][cellids]
    gc_test_under_chance = plot_df[gc][under_chance]
    stp_test = plot_df[stp][cellids]
    stp_test_under_chance = plot_df[stp][under_chance]
    ln_test = plot_df[LN][cellids]
    gc_stp_test = plot_df[combined][cellids]
    max_test = np.maximum(gc_test, stp_test)
    gc_stp_test_rel = gc_stp_test - ln_test
    max_test_rel = np.maximum(gc_test, stp_test) - ln_test

    fig1 = plt.figure(figsize=figsize)
    plt.scatter(max_test, gc_stp_test, c=wsu_gray, s=20)
    plt.scatter(gc_test_under_chance, stp_test_under_chance, c=wsu_crimson, s=20)
    ax = fig1.axes[0]
    plt.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=2, dashes=dash_spacing)
    plt.title('Absolute')
    plt.ylabel('GC+STP')
    plt.xlabel('Max GC or STP')

    fig2 = plt.figure(figsize=figsize)
    plt.scatter(max_test_rel, gc_stp_test_rel, c=wsu_gray, s=20)
    plt.scatter(gc_test_under_chance, stp_test_under_chance, c=wsu_crimson, s=20)
    ax = fig2.axes[0]
    plt.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=2, dashes=dash_spacing)
    plt.title('Relative')
    plt.ylabel('GC+STP')
    plt.xlabel('Max GC or STP')


def performance_bar(batch, gc, stp, LN, combined, se_filter=True,
                    LN_filter=False, manual_cellids=None, abbr_yaxis=False,
                    plot_stat='r_ceiling', y_adjust=0.05):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

    '''

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
    gc_test = plot_df[gc][cellids]
    gc_se = df_e[gc][cellids]
    stp_test = plot_df[stp][cellids]
    stp_se = df_e[stp][cellids]
    ln_test = plot_df[LN][cellids]
    ln_se = df_e[LN][cellids]
    gc_stp_test = plot_df[combined][cellids]
    gc_stp_se = df_e[combined][cellids]
    max_test = np.maximum(gc_test, stp_test)

    gc = np.median(gc_test.values)
    stp = np.median(stp_test.values)
    ln = np.median(ln_test.values)
    gc_stp = np.median(gc_stp_test.values)
    maximum = np.median(max_test)
    largest = max(gc, stp, ln, gc_stp, maximum)

    # TODO: double check that this is valid, to just take mean of errors
#    gc_sem = np.median(gc_se.values)
#    stp_sem = np.median(stp_se.values)
#    ln_sem = np.median(ln_se.values)
#    gc_stp_sem = np.median(gc_stp_se.values)

    fig = plt.figure(figsize=(15, 12))
    plt.bar([1, 2, 3, 4, 5], [ln, gc, stp, gc_stp, maximum],
            #color=['purple', 'green', 'gray', 'blue'])
            color=[ln_color, gc_color, stp_color, gc_stp_color, gc_stp_color],
            edgecolor="black", linewidth=2)
    plt.xticks([1, 2, 3, 4, 5], ['LN', 'GC', 'STP', 'GC+STP', 'Max(GC,STP)'])
    if abbr_yaxis:
        lower = np.floor(10*min(gc, stp, ln, gc_stp))/10
        upper = np.ceil(10*max(gc, stp, ln, gc_stp))/10 + y_adjust
        plt.ylim(ymin=lower, ymax=upper)
    else:
        plt.ylim(ymax=largest*1.4)
#    plt.errorbar([1, 2, 3, 4], [gc, stp, ln, gc_stp], yerr=[gc_sem, stp_sem,
#                 ln_sem, gc_stp_sem], fmt='none', ecolor='black')
    common_kwargs = {'color': 'white', 'horizontalalignment': 'center'}
    if abbr_yaxis:
        y_text = 0.5*(lower + min(gc, stp, ln, gc_stp))
    else:
        y_text = 0.2
    plt.text(1, y_text, "%0.04f" % ln, **common_kwargs)
    plt.text(2, y_text, "%0.04f" % gc, **common_kwargs)
    plt.text(3, y_text, "%0.04f" % stp, **common_kwargs)
    plt.text(4, y_text, "%0.04f" % gc_stp, **common_kwargs)
    plt.text(5, y_text, "%0.04f" % maximum, **common_kwargs)
    plt.title("Median Performance for LN, GC, STP, GC+STP and Max(Gc,STP),\n"
              "n: %d" % n_cells)

    return fig


def significance(batch, model1, model2, model3, model4, se_filter=True,
                 ln_filter=False, ratio_filter=False, threshold=2.5,
                 manual_cellids=None):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

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
    gc_stp_test = df_r[model4][cellids]
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
            if j == i:
                # if indices equal, on diagonal so no comparison
                array[i][j] = 0.00
            elif j > i:
                # if j is larger, below diagonal so get mean difference
                mean_one = np.mean(series_one)
                mean_two = np.mean(series_two)
                array[i][j] = abs(mean_one - mean_two)
            else:
                # if j is smaller, above diagonal so run t-test and
                # get p-value
                first = series_one.tolist()
                second = series_two.tolist()
                array[i][j] = st.wilcoxon(first, second)[1]

    xticks = range(len(modelnames))
    yticks = xticks
    minor_xticks = np.arange(-0.5, len(modelnames), 1)
    minor_yticks = np.arange(-0.5, len(modelnames), 1)

    fig = plt.figure(figsize=(len(modelnames),len(modelnames)))
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
        formatting = '{:.04f}'
        if z <= 0.0001:
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
    ax.set_title("Wilcoxon Signed Test", ha='center', fontsize = 14)

    blue_patch = mpatch.Patch(
            color='#368DFF', label='Mean Difference', edgecolor='black'
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
