from pathlib import Path
import pickle
import requests
import datetime
import numpy as np
import pandas as pd

import scipy.stats as st

import nems
import nems0.db as nd
import nems0.xform_helper as xhelp
from nems_lbhb.analysis.statistics import arrays_to_p
from nems0.metrics.mi import mutual_information

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

def load_pred(batch, cellids, modelnames):

    pred_data = []
    for i,cellid in enumerate(cellids):
        # LN
        xf0,ctx0 = xhelp.load_model_xform(cellid=cellid,batch=batch,modelname=modelnames[0])
        #CNN
        xf1,ctx1 = xhelp.load_model_xform(cellid=cellid,batch=batch,modelname=modelnames[1],
                                          eval_model=False)
        ctx1['val'] = ctx1['modelspec'].evaluate(rec=ctx0['val'].copy())

        pred_data.append({})
        pred_data[i]['cellid'] = cellid
        pred_data[i]['pred0'] = ctx0['val']['pred'].as_continuous()
        pred_data[i]['pred1'] = ctx1['val']['pred'].as_continuous()
        pred_data[i]['resp'] = ctx0['val']['resp'].as_continuous()

    return pred_data

def plot_xc_v_mi(batch=322, labels=None, colors=None, reload=True):

    if labels is None:
        labels = ['Prediction correlation','Mutual information']
    if colors is None:
        colors = ['black', 'black']
    df = pd.DataFrame()
    existing_cellids=[]
    if reload:
        try:
            df = pd.read_csv('xc_mi.csv', index_col=0)
            existing_cellids = df.index.to_list()
        except:
            print('initializing df')

    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)
    #significant_cells = significant_cells[:50]
    significant_cells = [c for c in significant_cells if c not in existing_cellids]
    modelnames = [SIG_TEST_MODELS[1], SIG_TEST_MODELS[0]]
    pred_data = load_pred(batch, significant_cells, modelnames)

    for i, p in enumerate(pred_data):
        cellid = p['cellid']
        df.loc[cellid,'mi0'], n0 = mutual_information(p['pred0'].flatten(), p['resp'].flatten(), L=12)
        df.loc[cellid,'mi1'], n1 = mutual_information(p['pred1'].flatten(), p['resp'].flatten(), L=12)
        df.loc[cellid,'xc0'] = np.corrcoef(p['pred0'].flatten(), p['resp'].flatten())[0,1]
        df.loc[cellid,'xc1'] = np.corrcoef(p['pred1'].flatten(), p['resp'].flatten())[0,1]

    df.to_csv('xc_mi.csv')

    fig, ax = plt.subplots(2,2, sharex='row', sharey='row')
    ax[0,0].scatter(df['xc0'],df['mi0'], s=3)
    ax[0,0].set_title(f"LN: r={np.corrcoef(df['xc0'],df['mi0'])[0,1]:.3f}")
    ax[0,1].scatter(df['xc1'],df['mi1'], s=3)
    ax[0,1].set_title(f"CNN: r={np.corrcoef(df['xc1'],df['mi1'])[0,1]:.3f}")
    ax[1,0].set_visible(False)
    ax[1,1].scatter(df['xc1']-df['xc0'], df['mi1']-df['mi0'], s=3)
    ax[1,1].set_title(f"improvement: r={np.corrcoef(df['xc1']-df['xc0'],df['mi1']-df['mi0'])[0,1]:.3f}")
    ax[1,1].set_xlabel('xc1 - xc0')
    ax[1,1].set_ylabel('mi1 - mi0')

    return fig


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
    plot_xc_v_mi()

    if 0:
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
