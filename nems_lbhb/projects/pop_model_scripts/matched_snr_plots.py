from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

import nems
import nems.db as nd
import nems.xform_helper as xhelp
import nems_lbhb.xform_wrappers as xwrap
import nems.epoch as ep

from pop_model_utils import (mplparams, get_significant_cells, get_rceiling_correction, SIG_TEST_MODELS, snr_by_batch,
                             NAT4_A1_SITES, NAT4_PEG_SITES, PLOT_STAT, DOT_COLORS,
                             figures_base_path, a1, peg, int_path, HELDOUT_CROSSBATCH, HELDOUT,
                             single_column_short, single_column_tall, column_and_half_short, column_and_half_tall)

import matplotlib as mpl
mpl.rcParams.update(mplparams)
import matplotlib.pyplot as plt
import seaborn as sns


def load_snr_and_cellids(a1, peg, a1_snr_path, peg_snr_path):
    a1_significant_cells = get_significant_cells(a1, SIG_TEST_MODELS, as_list=True)
    a1_results = nd.batch_comp(batch=a1, modelnames=SIG_TEST_MODELS, stat=PLOT_STAT, cellids=a1_significant_cells)
    a1_index = a1_results.index.values
    peg_significant_cells = get_significant_cells(peg, SIG_TEST_MODELS, as_list=True)
    peg_results = nd.batch_comp(batch=peg, modelnames=SIG_TEST_MODELS, stat=PLOT_STAT, cellids=peg_significant_cells)
    peg_index = peg_results.index.values

    a1_snr_df = snr_by_batch(a1, 'ozgf.fs100.ch18', load_path=a1_snr_path).loc[a1_index]
    peg_snr_df = snr_by_batch(peg, 'ozgf.fs100.ch18', load_path=peg_snr_path).loc[peg_index]
    a1_snr = a1_snr_df.values.flatten()
    a1_cellids = a1_snr_df.index
    peg_snr = peg_snr_df.values.flatten()
    peg_idx = np.argsort(peg_snr_df.values, axis=None)
    peg_snr_sample = peg_snr[peg_idx]
    peg_cellids = peg_snr_df.index[peg_idx].values

    return a1_snr, peg_snr_sample, a1_snr_df, peg_snr_df, a1_cellids, peg_cellids


def get_matched_snr_cells(a1_snr, peg_snr, a1_cellids, peg_cellids):
    # force "exact" distribution match for given histogram bins
    bins = np.histogram(np.hstack((peg_snr, a1_snr)), bins=40)[1]
    bin_assignments = np.digitize(peg_snr, bins, right=True)
    a1_matched_cellids = []
    peg_matched_cellids = []
    for i, (bin_idx, snr) in enumerate(zip(bin_assignments, peg_snr)):
        # look for a1 cell with minimum snr difference *that is also in the same bin*
        # and then add both peg cell and matching a1 cell to cellid lists
        a1_subset = ((a1_snr > bins[bin_idx-1]) & (a1_snr <= bins[bin_idx]))
        if a1_subset.sum() > 0:
            diffs = snr - a1_snr[a1_subset]
            min_idx = np.argmin(np.abs(diffs))  # index within subset
            min_cell_idx = np.argwhere(a1_subset)[min_idx][0]  # index within full list of cellids
            a1_matched_cellids.append(a1_cellids[min_cell_idx])
            peg_matched_cellids.append(peg_cellids[i])
        else:
            # if no such match, exclude cell
            continue

    return a1_matched_cellids, peg_matched_cellids, bins


def get_matched_snr_mapping(a1_snr, peg_snr, a1_snr_df, peg_snr_df, a1_cellids, peg_cellids):
    modelnames = SIG_TEST_MODELS

    a1_matched_cellids, peg_matched_cellids, _ = get_matched_snr_cells(a1_snr, peg_snr, a1_cellids, peg_cellids)
    cell_map_df = pd.DataFrame(data={'A1_cellid': a1_matched_cellids,
                                     'A1_snr': a1_snr_df.snr.loc[a1_matched_cellids].values,
                                     'PEG_cellid': peg_matched_cellids,
                                     'PEG_snr': peg_snr_df.snr.loc[peg_matched_cellids].values})

    return cell_map_df


def plot_matched_snr(a1, peg, a1_snr_path, peg_snr_path, plot_sanity_check=True, ax=None, inset_ax=None):
    modelnames = SIG_TEST_MODELS

    # Load model performance results for a1 and peg
    a1_significant_cells = get_significant_cells(a1, modelnames, as_list=True)
    a1_results = nd.batch_comp(batch=a1, modelnames=modelnames, stat=PLOT_STAT, cellids=a1_significant_cells)
    a1_index = a1_results.index.values
    a1_medians = [a1_results[m].median() for m in modelnames]

    peg_significant_cells = get_significant_cells(peg, modelnames, as_list=True)
    peg_results = nd.batch_comp(batch=peg, modelnames=modelnames, stat=PLOT_STAT, cellids=peg_significant_cells)
    peg_index = peg_results.index.values
    peg_medians = [peg_results[m].median() for m in modelnames]

    a1_snr_df = snr_by_batch(a1, 'ozgf.fs100.ch18', load_path=a1_snr_path).loc[a1_index]
    peg_snr_df = snr_by_batch(peg, 'ozgf.fs100.ch18', load_path=peg_snr_path).loc[peg_index]

    # put peg snr in increasing order
    a1_snr = a1_snr_df.values.flatten()
    a1_median_snr = np.median(a1_snr)
    a1_cellids = a1_snr_df.index.values
    peg_snr = peg_snr_df.values.flatten()
    peg_median_snr = np.median(peg_snr)
    peg_idx = np.argsort(peg_snr_df.values, axis=None)
    peg_cellids = peg_snr_df.index[peg_idx].values
    peg_snr_sample = peg_snr[peg_idx]
    test_snr = st.mannwhitneyu(a1_snr, peg_snr, alternative='two-sided')

    # force "exact" distribution match for given histogram bins
    a1_matched_cellids, peg_matched_cellids, bins = get_matched_snr_cells(a1_snr, peg_snr_sample, a1_cellids, peg_cellids)
    a1_matched_snr_df = a1_snr_df.loc[a1_matched_cellids]
    peg_matched_snr_df = peg_snr_df.loc[peg_matched_cellids]

    a1_snr_df2 = a1_snr_df.rename(columns={'snr': 'A1'})
    peg_snr_df2 = peg_snr_df.rename(columns={'snr': 'PEG'})
    combined_df = a1_snr_df2.join(peg_snr_df2, how='outer')

    combined_df = combined_df.reset_index(0)
    combined_df = pd.melt(combined_df, value_vars=['A1', 'PEG'], value_name='SNR', id_vars='cellid')
    combined_df['matched'] = 0
    combined_df['matched'].loc[combined_df.cellid.isin(a1_matched_cellids + peg_matched_cellids)] = 1
    combined_df = combined_df.sort_values(by='matched')

    a1_matched_snr = a1_matched_snr_df.values
    peg_matched_snr = peg_matched_snr_df.values
    a1_median_snr_matched = np.median(a1_matched_snr)
    peg_median_snr_matched = np.median(peg_matched_snr)

    if plot_sanity_check:
        # Sanity check: these should definitely be the same
        fig = plt.figure()
        plt.hist(peg_matched_snr, bins=bins, alpha=0.5, label='peg')
        plt.hist(a1_matched_snr, bins=bins, alpha=0.3, label='a1')
        plt.title('sanity check: these should completely overlap')
        plt.legend()



    # Filter by matched cellids,
    # then combine into single dataframe with columns for cellid, a1/peg, modelname, PLOT_STAT
    short_names = ['CNN 1Dx2', 'pop LN', 'CNN single']
    a1_short = [s + ' A1' for s in short_names]
    a1_rename = {k: v for k, v in zip(modelnames, a1_short)}
    a1_results = a1_results.rename(columns=a1_rename)
    a1_matched_results = a1_results.loc[a1_matched_cellids].reset_index(level=0)
    a1_removed_results = a1_results.loc[~a1_results.index.isin(a1_matched_cellids)].reset_index(level=0)
    #a1_full_results = a1_results.reset_index(level=0)

    peg_short = [s + ' PEG' for s in short_names]
    peg_rename = {k: v for k, v in zip(modelnames, peg_short)}
    peg_results = peg_results.rename(columns=peg_rename)
    peg_matched_results = peg_results.loc[peg_matched_cellids].reset_index(level=0)
    peg_removed_results = peg_results[~peg_results.index.isin(peg_matched_cellids)].reset_index(level=0)

    # Test significance after matching distributions
    test_c1 = st.mannwhitneyu(a1_matched_results[a1_short[0]], peg_matched_results[peg_short[0]], alternative='two-sided')
    test_LN = st.mannwhitneyu(a1_matched_results[a1_short[1]], peg_matched_results[peg_short[1]], alternative='two-sided')
    test_dnn = st.mannwhitneyu(a1_matched_results[a1_short[2]], peg_matched_results[peg_short[2]], alternative='two-sided')

    results_matched = pd.concat([a1_matched_results, peg_matched_results], axis=0)
    results_removed = pd.concat([a1_removed_results, peg_removed_results], axis=0)
    alternating_columns = [col for sublist in zip(a1_short, peg_short) for col in sublist]

    results_matched = pd.melt(results_matched, id_vars='cellid', value_vars=a1_short + peg_short, value_name=PLOT_STAT)
    results_matched = results_matched.rename(columns={'variable': 'model'})
    results_matched['hue_tag'] = np.zeros(results_matched.shape[0], )
    for i, n in enumerate(short_names):
        results_matched.loc[results_matched['model'].str.contains(n), 'hue_tag'] = i

    results_removed = pd.melt(results_removed, id_vars='cellid', value_vars=a1_short + peg_short, value_name=PLOT_STAT)
    results_removed = results_removed.rename(columns={'variable': 'model'})
    results_removed['hue_tag'] = np.zeros(results_removed.shape[0], )
    for i, n in enumerate(short_names):
        results_removed.loc[results_removed['model'].str.contains(n), 'hue_tag'] = i

    if ax is None:
        _, ax = plt.subplots()
    else:
        plt.sca(ax)
    jitter = 0.2
    palette = {0: DOT_COLORS['1D CNNx2'], 1: DOT_COLORS['pop LN'], 2: DOT_COLORS['CNN single']}

    # plot removed cells "under" the remaining ones
    #tres = results_removed.loc[(results_removed[PLOT_STAT]<1) & results_removed[PLOT_STAT]>-0.05]
    # sns.stripplot(x='model', y=PLOT_STAT, data=tres, zorder=0, order=alternating_columns,
    #                    color='gray', alpha=0.5, size=2, jitter=jitter, hue='hue_tag', palette=palette, ax=ax)
    #ax.legend_.remove()

    tres=results_matched.loc[(results_matched[PLOT_STAT]<1) & results_matched[PLOT_STAT]>-0.05]
    sns.stripplot(x='model', y=PLOT_STAT, data=tres, zorder=0, order=alternating_columns,
                       jitter=jitter, hue='hue_tag', palette=palette, ax=ax, size=2)
    ax.legend_.remove()
    sns.boxplot(x='model', y=PLOT_STAT, data=results_matched, boxprops={'facecolor': 'None', 'linewidth': 1},
                     showcaps=False, showfliers=False, whiskerprops={'linewidth': 0}, order=alternating_columns, ax=ax)

    labels = [e.get_text() for e in ax.get_xticklabels()]
    ticks = ax.get_xticks()
    w = 0.1
    for idx, model in enumerate(labels):
        idx = labels.index(model)
        j = int(idx*0.5)
        if idx % 2 == 0:
            plt.hlines(a1_medians[j], ticks[idx]-w, ticks[idx]+w, color='black', linewidth=2)
        else:
            plt.hlines(peg_medians[j], ticks[idx]-w, ticks[idx]+w, color='black', linewidth=2)

    ax.set(ylim=(None,1))
    plt.xticks(rotation=45, fontsize=6, ha='right')
    plt.tight_layout()

    # Plot the original distributions and the matched distribution, to visualize what was removed.
    if inset_ax is None:
        _, inset_ax = plt.subplots()
    else:
        plt.sca(inset_ax)

    # Histogram version
    # bins=bins -- this would be the actual bins used for matching the distributions, but it's a bit too fine
    # for the figure size we ended up at.
    plt.hist(a1_snr, bins=bins, label='a1', histtype='stepfilled', edgecolor='black', color='white')#DOT_COLORS['pop LN'])
    plt.hist(peg_snr, bins=bins, label='peg', histtype='stepfilled', edgecolor='black', color='lightgray')#DOT_COLORS['2D CNN'],)
    plt.hist(a1_matched_snr, bins=bins, label='matched', histtype='stepfilled', edgecolor='black', fill=False,
             hatch='\\\\\\')
    plt.legend()
    plt.ylabel('Number of neurons')
    plt.xlabel('SNR')


    # Strip plot version
    # sns.stripplot(x='variable', y='SNR', data=combined_df, hue='matched', palette={0: 'lightgray', 1: 'black'},
    #               size=2, ax=inset_ax, jitter=jitter)
    # inset_ax.legend_.remove()
    #sns.stripplot(data=combined_df_matched, palette={'A1': 'black', 'PEG': 'black'}, size=2, ax=inset_ax, jitter=jitter)

    plt.tight_layout()

    return (test_c1, test_LN, test_dnn, test_snr,
            a1_results.median(), a1_matched_results.median(), peg_results.median(), peg_matched_results.median(),
            a1_median_snr, a1_median_snr_matched, peg_median_snr, peg_median_snr_matched)


def plot_heldout_a1_vs_peg(a1_snr_path, peg_snr_path, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(single_column_short))

    a1_snr, peg_snr, a1_snr_df, peg_snr_df, a1_cellids, peg_cellids = load_snr_and_cellids(322, 323, a1_snr_path, peg_snr_path)
    cell_map_df = get_matched_snr_mapping(a1_snr, peg_snr, a1_snr_df, peg_snr_df, a1_cellids, peg_cellids)
    #a1_matched_cellids = cell_map_df.A1_cellid
    peg_matched_cellids = cell_map_df.PEG_cellid
    #a1_r = nd.batch_comp(322, HELDOUT[:1], stat=PLOT_STAT).loc[a1_matched_cellids]
    peg_r = nd.batch_comp(323, [HELDOUT_CROSSBATCH], stat=PLOT_STAT).loc[peg_matched_cellids]
    peg_r2 = nd.batch_comp(323, HELDOUT[:1], stat=PLOT_STAT).loc[peg_matched_cellids]
    #test_crossbatch = st.wilcoxon(a1_r.values.flatten(), peg_r.values.flatten(), alternative='two-sided')
    test_within_peg = st.wilcoxon(peg_r.values.flatten(), peg_r2.values.flatten(), alternative='two-sided')

    #a1_r = a1_r.rename(columns={HELDOUT[0]: 'A1'})
    peg_r = peg_r.rename(columns={HELDOUT_CROSSBATCH: 'cross-batch held'})
    peg_r2 = peg_r2.rename(columns={HELDOUT[0]: 'PEG held'})
    combined_r = peg_r.join(peg_r2, how='outer')
    #combined_df = combined_r.join(peg_snr_df, how='left')
    #combined_df = combined_df.fillna(peg_snr_df.loc[peg_matched_cellids])
    #combined_df[PLOT_STAT] = np.nan
    #combined_df.r_ceiling.loc[a1_matched_cellids] = combined_df.A1.loc[a1_matched_cellids]
    #combined_df.r_ceiling.loc[peg_matched_cellids] = combined_df.PEG.loc[peg_matched_cellids]
    #combined_df['variable'] = 'A1'
    #combined_df['variable'].loc[peg_matched_cellids] = 'PEG'
    #combined_df = combined_df.drop(columns='A1')
    #combined_df = combined_df.drop(columns='PEG')
    #combined_df = combined_df.sort_values(by='snr')
    #combined_df = combined_df.reset_index(0)

    #sns.stripplot(x='variable', y=PLOT_STAT, data=combined_df, color=DOT_COLORS['1D CNNx2'],
                  #hue='snr', palette=f'dark:{DOT_COLORS["1D CNNx2"]}',
                  #size=2, ax=ax, zorder=0)
    sns.stripplot(data=combined_r, color=DOT_COLORS['1D CNNx2'], size=2, ax=ax, zorder=0)
    plt.xticks(rotation=45, fontsize=6, ha='right')
    sns.boxplot(data=combined_r, boxprops={'facecolor': 'None', 'linewidth': 1},
                     showcaps=False, showfliers=False, whiskerprops={'linewidth': 0}, ax=ax)
    #ax.legend_.remove()  # tried to color by snr but it's just really hard to see, and no pattern anyway
    plt.tight_layout()

    return peg_r, peg_r2, test_within_peg

if __name__ == '__main__':
    a1_snr_path = int_path / str(a1) / 'snr_nat4.csv'
    peg_snr_path = int_path / str(peg) / 'snr_nat4.csv'

    peg_r, peg_r2, test_within = plot_heldout_a1_vs_peg(a1_snr_path, peg_snr_path)
    plt.show(block=True)

    fig9a, ax4a = plt.subplots(figsize=single_column_short)
    fig9b, ax4b = plt.subplots(figsize=single_column_short)  # but actually resize manually in illustrator, as needed.
    test_c1, test_LN, test_dnn, test_snr, a1_md, a1_md_matched, peg_md, \
        peg_md_matched, a1_md_snr, a1_md_snr_matched, peg_md_snr, peg_md_snr_matched = plot_matched_snr(
            a1, peg, a1_snr_path, peg_snr_path, plot_sanity_check=False, ax=ax4a, inset_ax=ax4b
    )
    plt.show(block=True)

    #a1_snr_path = int_path  / str(a1) / 'snr_nat4.pkl'
    #peg_snr_path = int_path  / str(peg) / 'snr_nat4.pkl'
    #
    #u_c1, p_c1, u_LN, p_LN, u_dnn, p_dnn = plot_matched_snr(a1, peg, a1_snr_path, peg_snr_path)

    print("Sig. tests:\n"
          "conv1Dx2: %s\n"
          "LN: %s\n"
          "dnn1_single: %s\n"% (test_c1, test_LN, test_dnn))

    # # Generate mapping csv for use elsewhere
    # cell_map_df = get_matched_snr_mapping(a1, peg, a1_snr_path, peg_snr_path)
    # cell_map_df.to_csv(int_path / 'snr_subset_map.csv')
