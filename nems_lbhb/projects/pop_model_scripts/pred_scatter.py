from pathlib import Path
import pickle
import requests
import datetime
import numpy as np
import pandas as pd

import scipy.stats as st

import nems
import nems.db as nd
import nems.xform_helper as xhelp
from nems_lbhb.analysis.statistics import arrays_to_p

from pop_model_utils import mplparams, get_significant_cells, SIG_TEST_MODELS, \
    ALL_FAMILY_MODELS, POP_MODELS, PLOT_STAT, DOT_COLORS, shortnames, base_path, a1, peg, \
    single_column_short, single_column_tall, column_and_half_short, column_and_half_tall
import matplotlib as mpl
mpl.rcParams.update(mplparams)
import matplotlib.pyplot as plt
import seaborn as sns

batch_str = {a1: 'A1', peg: 'PEG'}

def scatter_groups(groups, colors, add_diagonal=True, ax=None, scatter_kwargs=None, labels=None):

    if ax is None:
        fig, ax = plt.subplots()
    if scatter_kwargs is None:
        scatter_kwargs = {}

    if add_diagonal:
        ax.plot([0,1], [0,1], c='black', linestyle='dashed')
    if labels is None:
        labels = ['_nolabel' for c in colors]
    for g, color, label in zip(groups, colors, labels):
        vg = (g[0]>0) & (g[0]<1) & (g[1]>0) & (g[1]<1)
        ax.scatter(g[0][vg], g[1][vg], c=color, s=1, label=label, **scatter_kwargs)
    ax.set_aspect('equal')

    return ax

def plot_pred_scatter(batch, modelnames, labels=None, colors=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if labels is None:
        labels = ['model 1','model 2']
    if colors is None:
        colors = ['black', 'black']

    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    sig_scores = nd.batch_comp(batch, modelnames, cellids=significant_cells, stat='r_test')
    se_scores = nd.batch_comp(batch, modelnames, cellids=significant_cells, stat='se_test')
    ceiling_scores = nd.batch_comp(batch, modelnames, cellids=significant_cells, stat=PLOT_STAT)
    nonsig_cells = list(set(sig_scores.index) - set(significant_cells))

    # figure out units with significant differences between models
    sig = (sig_scores[modelnames[1]] - se_scores[modelnames[1]] > sig_scores[modelnames[0]] + se_scores[modelnames[0]]) | \
          (sig_scores[modelnames[0]] - se_scores[modelnames[0]] > sig_scores[modelnames[1]] + se_scores[modelnames[1]])
    group1 = (ceiling_scores.loc[~sig,modelnames[0]].values, ceiling_scores.loc[~sig,modelnames[1]].values)
    group2 = (ceiling_scores.loc[sig,modelnames[0]].values, ceiling_scores.loc[sig,modelnames[1]].values)
    n_nonsig = group1[0].size
    n_sig = group2[0].size

    scatter_groups([group1, group2], ['lightgray', 'black'], ax=ax, labels=['N.S.', 'p < 0.5'])
    #ax.set_title(f'{batch_str[batch]} {PLOT_STAT}')
    ax.set_xlabel(f'{labels[0]}\n(median r={ceiling_scores[modelnames[0]].median():.3f})', color=colors[0])
    ax.set_ylabel(f'{labels[1]}\n(median r={ceiling_scores[modelnames[1]].median():.3f})', color=colors[1])

    return fig, n_sig, n_nonsig


def plot_conv_scatters(batch):
    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    sig_scores = nd.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat='r_test')
    se_scores = nd.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat='se_test')
    ceiling_scores = nd.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat=PLOT_STAT)
    nonsig_cells = list(set(sig_scores.index) - set(significant_cells))

    fig, ax = plt.subplots(1,3,figsize=(12,4))

    # LN vs DNN-single
    plot_pred_scatter(batch, [ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[4]], labels=['1Dx2-CNN','pop-LN'], ax=ax[0])

    # LN vs conv1dx2+d
    plot_pred_scatter(batch, [ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[2]], labels=['1Dx2-CNN','pop-LN'], ax=ax[0])

    # conv2d vs conv1dx2+d
    sig = (sig_scores[ALL_FAMILY_MODELS[0]] - se_scores[ALL_FAMILY_MODELS[0]] > sig_scores[ALL_FAMILY_MODELS[2]] + se_scores[ALL_FAMILY_MODELS[2]]) | \
        (sig_scores[ALL_FAMILY_MODELS[2]] - se_scores[ALL_FAMILY_MODELS[2]] > sig_scores[ALL_FAMILY_MODELS[0]] + se_scores[ALL_FAMILY_MODELS[0]])
    group3 = (ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[0]].values, ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[2]].values)
    group4 = (ceiling_scores.loc[sig,ALL_FAMILY_MODELS[0]].values, ceiling_scores.loc[sig,ALL_FAMILY_MODELS[2]].values)
    scatter_groups([group3, group4], ['lightgray', 'black'], ax=ax[2])
    ax[2].set_title('batch %d, %s, conv2d vs conv1dx2+d' % (batch, PLOT_STAT))
    ax[2].set_xlabel(f'DNN (2D conv) pred. correlation ({ceiling_scores[ALL_FAMILY_MODELS[0]].mean():.3f})', color=colors[0])
    ax[2].set_ylabel(f'DNN (1D conv) pred. correlation ({ceiling_scores[ALL_FAMILY_MODELS[2]].mean():.3f})', color=colors[1])

    plt.tight_layout()

    return fig

def scatter_titan(batch):
    PLOT_STAT = 'r_test'
    TITAN_MODEL = 'ozgf.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_tfinit.n.mc50.lr1e3.es20-newtf.n.mc100.lr1e4.exa'
    REFERENCE_MODEL = 'ozgf.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4'
    MODELS = [REFERENCE_MODEL, TITAN_MODEL]

    significant_cells = get_significant_cells(batch, MODELS, as_list=True)

    sig_scores = nd.batch_comp(batch, MODELS, cellids=significant_cells, stat='r_test')
    se_scores = nd.batch_comp(batch, MODELS, cellids=significant_cells, stat='se_test')
    ceiling_scores = nd.batch_comp(batch, MODELS, cellids=significant_cells, stat=PLOT_STAT)
    nonsig_cells = list(set(sig_scores.index) - set(significant_cells))

    fig, ax = plt.subplots(1,1,figsize=(4,4))

    # LN vs DNN-single
    sig = (sig_scores[MODELS[1]] - se_scores[MODELS[1]] > sig_scores[MODELS[0]] + se_scores[MODELS[0]]) | \
          (sig_scores[MODELS[0]] - se_scores[MODELS[0]] > sig_scores[MODELS[1]] + se_scores[MODELS[1]])
    group1 = (ceiling_scores.loc[~sig,MODELS[0]].values, ceiling_scores.loc[~sig,MODELS[1]].values)
    group2 = (ceiling_scores.loc[sig,MODELS[0]].values, ceiling_scores.loc[sig,MODELS[1]].values)

    scatter_groups([group1, group2], ['lightgray', 'black'], ax=ax)
    ax.set_title('batch %d, %s, Conv1d vs Titan' % (batch, PLOT_STAT))
    ax.set_xlabel(f'Single stim ({ceiling_scores[MODELS[0]].mean():.3f})', color='orange')
    ax.set_ylabel(f'Titan ({ceiling_scores[MODELS[1]].mean():.3f})', color='lightgreen')

    return fig

def bar_mean(batch, modelnames, stest=SIG_TEST_MODELS, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    cellids = get_significant_cells(batch, stest, as_list=True)
    r_values = nd.batch_comp(batch, modelnames, cellids=cellids, stat=PLOT_STAT)

    # Bar Plot -- Median for each model
    # NOTE: ordering of names is assuming ALL_FAMILY_MODELS is being used and has not changed.
    bar_colors = [DOT_COLORS[k] for k in shortnames]
    medians = r_values.median(axis=0).values
    ax.bar(np.arange(0, len(modelnames)), medians, color=bar_colors, edgecolor='black', linewidth=1,
           tick_label=shortnames)
    ax.set_ylabel('Median prediction\ncorrelation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation='45', ha='right')


    # Test significance for all comparisons
    stats_results = {}
    reduced_modelnames = modelnames.copy()
    reduced_shortnames = shortnames.copy()
    for m1, s1 in zip(modelnames, shortnames):
        i = 0
        reduced_modelnames.pop(i)
        reduced_shortnames.pop(i)
        i += 1
        # compare each model to every other model
        for m2, s2 in zip(reduced_modelnames, reduced_shortnames):
            stats_test = st.wilcoxon(r_values[m1], r_values[m2], alternative='two-sided')
            #stats_test = arrays_to_p(r_values[m1], r_values[m2], cellids, twosided=True)
            key = f'{s1} vs {s2}'
            stats_results[key] = stats_test

    return ax, medians, stats_results


if __name__ == '__main__':

    f, ax = plt.subplots(2, 2, figsize=column_and_half_tall, sharey='row')
    plot_pred_scatter(a1, [ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[2]], labels=['pop LN','1D CNN'],
                      colors=[DOT_COLORS['LN_pop'],DOT_COLORS['conv1dx2+d']], ax=ax[0,0])
    plot_pred_scatter(a1, [ALL_FAMILY_MODELS[2], ALL_FAMILY_MODELS[0]], labels=['1D CNN','2D CNN'],
                      colors=[DOT_COLORS['conv1dx2+d'],DOT_COLORS['conv2d']], ax=ax[0,1])
    bar_mean(a1, ALL_FAMILY_MODELS, stest=SIG_TEST_MODELS, ax=ax[1,0])
    ax[1,0].set_title('A1')
    bar_mean(peg, ALL_FAMILY_MODELS, stest=SIG_TEST_MODELS, ax=ax[1,1])
    ax[1,1].set_title('PEG')
    ax[1,1].set_ylabel('')
    f.tight_layout()
