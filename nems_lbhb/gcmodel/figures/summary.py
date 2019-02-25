import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
import scipy.stats as st

import nems.db as nd
from .utils import get_valid_improvements


gc_color = '#69657C'
stp_color = '#394B5E'
ln_color = '#62838C'
gc_stp_color = '#215454'


# Scatter comparisons of overall model performance (similar to web ui)
# For:
# LN versus GC     GC_STP vs GC
# LN versus STP    GC_STP vs STP
# GC versus STP    GC_STP vs LN
def performance_scatters(batch, model1, model2, model3, model4,
                         se_filter=True, ln_filter=False, ratio_filter=False,
                         threshold=2.5, manual_cellids=None):
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

    if not se_filter:
        under_chance = np.array([True]*len(df_r[model1]))
        less_LN = copy.deepcopy(under_chance)

    n_cells = len(cellids)
    n_under_chance = len(under_chance) if under_chance != cellids else 0
    n_less_LN = len(less_LN) if less_LN != cellids else 0

    gc_test = df_r[model1][cellids]
    gc_test_under_chance = df_r[model1][under_chance]
    gc_test_less_LN = df_r[model1][less_LN]

    stp_test = df_r[model2][cellids]
    stp_test_under_chance = df_r[model2][under_chance]
    stp_test_less_LN = df_r[model2][less_LN]

    ln_test = df_r[model3][cellids]
    ln_test_under_chance = df_r[model3][under_chance]
    ln_test_less_LN = df_r[model3][less_LN]

    gc_stp_test = df_r[model4][cellids]
    gc_stp_test_under_chance = df_r[model4][under_chance]
    gc_stp_test_less_LN = df_r[model4][less_LN]

    fig, axes = plt.subplots(2, 3)

    # Row 1 (vs LN)
    ax = axes[0][0]
    ax.scatter(gc_test, ln_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_test_under_chance, ln_test_under_chance, c='red', s=1)
    ax.scatter(gc_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('GC vs LN')
    ax.set_xlabel('GC')
    ax.set_ylabel('LN')
    ax.text(0.90, -0.10, 'all = %d' % (n_cells+n_under_chance+n_less_LN),
            ha='right', va='bottom')
    ax.text(0.90, 0.00, 'n = %d' % n_cells, ha='right', va='bottom')
    ax.text(0.90, 0.10, 'uc = %d' % n_under_chance, ha='right', va='bottom',
            color='red')
    ax.text(0.90, 0.20, '<ln = %d' % n_less_LN, ha='right', va='bottom',
            color='blue')

    ax = axes[0][1]
    ax.scatter(stp_test, ln_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(stp_test_under_chance, ln_test_under_chance, c='red', s=1)
    ax.scatter(stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('STP vs LN')
    ax.set_xlabel('STP')
    ax.set_ylabel('LN')

    ax = axes[0][2]
    ax.scatter(gc_stp_test, ln_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_stp_test_under_chance, ln_test_under_chance, c='red', s=1)
    ax.scatter(gc_stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('GC + STP vs LN')
    ax.set_xlabel('GC + STP')
    ax.set_ylabel('LN')

    # Row 2 (head-to-head)
    ax = axes[1][0]
    ax.scatter(gc_test, stp_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_test_under_chance, stp_test_under_chance, c='red', s=1)
    ax.scatter(gc_test_less_LN, stp_test_less_LN, c='blue', s=1)
    ax.set_title('GC vs STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('STP')

    ax = axes[1][1]
    ax.scatter(gc_test, gc_stp_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_test_under_chance, gc_stp_test_under_chance, c='red', s=1)
    ax.scatter(gc_test_less_LN, gc_stp_test_less_LN, c='blue', s=1)
    ax.set_title('GC vs GC + STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('GC + STP')

    ax = axes[1][2]
    ax.scatter(stp_test, gc_stp_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(stp_test_under_chance, gc_stp_test_under_chance, c='red', s=1)
    ax.scatter(stp_test_less_LN, gc_stp_test_less_LN, c='blue', s=1)
    ax.set_title('STP vs GC + STP')
    ax.set_xlabel('STP')
    ax.set_ylabel('GC + STP')

    plt.tight_layout()

    return fig

def performance_bar(batch, model1, model2, model3, model4, se_filter=True,
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

    n_cells = len(cellids)
    gc_test = df_r[model1][cellids]
    gc_se = df_e[model1][cellids]
    stp_test = df_r[model2][cellids]
    stp_se = df_e[model2][cellids]
    ln_test = df_r[model3][cellids]
    ln_se = df_e[model3][cellids]
    gc_stp_test = df_r[model4][cellids]
    gc_stp_se = df_e[model4][cellids]

    gc = np.median(gc_test.values)
    stp = np.median(stp_test.values)
    ln = np.median(ln_test.values)
    gc_stp = np.median(gc_stp_test.values)
    largest = max(gc, stp, ln, gc_stp)

    # TODO: double check that this is valid, to just take mean of errors
    gc_sem = np.median(gc_se.values)
    stp_sem = np.median(stp_se.values)
    ln_sem = np.median(ln_se.values)
    gc_stp_sem = np.median(gc_stp_se.values)

    fig = plt.figure()
    plt.bar([1, 2, 3, 4], [gc, stp, ln, gc_stp],
            #color=['purple', 'green', 'gray', 'blue'])
            color=[gc_color, stp_color, ln_color, gc_stp_color])
    plt.xticks([1, 2, 3, 4], ['GC', 'STP', 'LN', 'GC + STP'])
    plt.ylim(ymax=largest*1.4)
    plt.errorbar([1, 2, 3, 4], [gc, stp, ln, gc_stp], yerr=[gc_sem, stp_sem,
                 ln_sem, gc_stp_sem], fmt='none', ecolor='black')
    common_kwargs = {'color': 'white', 'horizontalalignment': 'center'}
    plt.text(1, 0.2, "%0.04f" % gc, **common_kwargs)
    plt.text(2, 0.2, "%0.04f" % stp, **common_kwargs)
    plt.text(3, 0.2, "%0.04f" % ln, **common_kwargs)
    plt.text(4, 0.2, "%0.04f" % gc_stp, **common_kwargs)
    plt.title("Median Performance for GC, STP, LN, and GC + STP models,\n"
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

    modelnames = ['GC', 'STP', 'LN', 'GC + STP']
    models = {'GC': gc_test, 'STP': stp_test, 'LN': ln_test,
              'GC + STP': gc_stp_test}
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
