from pathlib import Path
import pickle
import requests
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

import nems
import nems.db as nd
import nems.xform_helper as xhelp

from pop_model_utils import get_significant_cells, SIG_TEST_MODELS, \
    ALL_FAMILY_MODELS, POP_MODELS, PLOT_STAT, shortnames, base_path


def plot_conv_scatters(batch):
    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    sig_scores = nd.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat='r_test')
    se_scores = nd.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat='se_test')
    ceiling_scores = nd.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat=PLOT_STAT)
    nonsig_cells = list(set(sig_scores.index) - set(significant_cells))

    fig, ax = plt.subplots(1,3,figsize=(12,4))

    # LN vs DNN-single
    sig = (sig_scores[ALL_FAMILY_MODELS[4]] - se_scores[ALL_FAMILY_MODELS[4]] > sig_scores[ALL_FAMILY_MODELS[3]] + se_scores[ALL_FAMILY_MODELS[3]]) | \
        (sig_scores[ALL_FAMILY_MODELS[3]] - se_scores[ALL_FAMILY_MODELS[3]] > sig_scores[ALL_FAMILY_MODELS[4]] + se_scores[ALL_FAMILY_MODELS[4]])
    group1 = (ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[3]].values, ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[4]].values)
    group2 = (ceiling_scores.loc[sig,ALL_FAMILY_MODELS[3]].values, ceiling_scores.loc[sig,ALL_FAMILY_MODELS[4]].values)

    scatter_groups([group1, group2], ['lightgray', 'black'], ax=ax[0])
    ax[0].set_title('batch %d, %s, LN vs DNN-single' % (batch, PLOT_STAT))
    ax[0].set_xlabel(f'LN pred. corr. ({ceiling_scores[ALL_FAMILY_MODELS[3]].mean():.3f})', color='orange')
    ax[0].set_ylabel(f'DNN single pred. corr. ({ceiling_scores[ALL_FAMILY_MODELS[4]].mean():.3f})', color='lightgreen')

    # LN vs conv1dx2+d
    sig = (sig_scores[ALL_FAMILY_MODELS[2]] - se_scores[ALL_FAMILY_MODELS[2]] > sig_scores[ALL_FAMILY_MODELS[3]] + se_scores[ALL_FAMILY_MODELS[3]]) | \
        (sig_scores[ALL_FAMILY_MODELS[3]] - se_scores[ALL_FAMILY_MODELS[3]] > sig_scores[ALL_FAMILY_MODELS[2]] + se_scores[ALL_FAMILY_MODELS[2]])
    group1 = (ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[3]].values, ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[2]].values)
    group2 = (ceiling_scores.loc[sig,ALL_FAMILY_MODELS[3]].values, ceiling_scores.loc[sig,ALL_FAMILY_MODELS[2]].values)

    scatter_groups([group1, group2], ['lightgray', 'black'], ax=ax[1])
    ax[1].set_title('batch %d, %s, LN vs conv1dx2+d' % (batch, PLOT_STAT))
    ax[1].set_xlabel(f'LN pred. correlation ({ceiling_scores[ALL_FAMILY_MODELS[3]].mean():.3f})', color='orange')
    ax[1].set_ylabel(f'DNN (1D conv x 2) pred. corr. ({ceiling_scores[ALL_FAMILY_MODELS[2]].mean():.3f})', color='purple')

    # conv2d vs conv1dx2+d
    sig = (sig_scores[ALL_FAMILY_MODELS[0]] - se_scores[ALL_FAMILY_MODELS[0]] > sig_scores[ALL_FAMILY_MODELS[2]] + se_scores[ALL_FAMILY_MODELS[2]]) | \
        (sig_scores[ALL_FAMILY_MODELS[2]] - se_scores[ALL_FAMILY_MODELS[2]] > sig_scores[ALL_FAMILY_MODELS[0]] + se_scores[ALL_FAMILY_MODELS[0]])
    group3 = (ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[0]].values, ceiling_scores.loc[~sig,ALL_FAMILY_MODELS[2]].values)
    group4 = (ceiling_scores.loc[sig,ALL_FAMILY_MODELS[0]].values, ceiling_scores.loc[sig,ALL_FAMILY_MODELS[2]].values)
    scatter_groups([group3, group4], ['lightgray', 'black'], ax=ax[2])
    ax[2].set_title('batch %d, %s, conv2d vs conv1dx2+d' % (batch, PLOT_STAT))
    ax[2].set_xlabel(f'DNN (2D conv) pred. correlation ({ceiling_scores[ALL_FAMILY_MODELS[0]].mean():.3f})', color='green')
    ax[2].set_ylabel(f'DNN (1D conv) pred. correlation ({ceiling_scores[ALL_FAMILY_MODELS[2]].mean():.3f})', color='purple')

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

def scatter_groups(groups, colors, add_diagonal=True, ax=None, scatter_kwargs=None):
    if ax is None:
        ax = plt.gca()
    if scatter_kwargs is None:
        scatter_kwargs = {}

    if add_diagonal:
        ax.plot([0,1], [0,1], c='black', linestyle='dashed')
    for g, color in zip(groups, colors):
        ax.scatter(g[0], g[1], c=color, s=5, **scatter_kwargs)

    return ax

a1 = 322
peg = 323
f1=plot_conv_scatters(322)
f2=plot_conv_scatters(323)
f3=scatter_titan(322)

f1.savefig(base_path / 'A1_pred_scatter.pdf')
f2.savefig(base_path / 'PEG_pred_scatter.pdf')
f3.savefig(base_path / 'titan_pred_scatter.pdf')
