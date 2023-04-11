from pathlib import Path
import pickle
import requests
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

import scipy.stats as st

import nems0
from nems0 import db
import nems0.xform_helper as xhelp
from nems_lbhb.analysis.statistics import arrays_to_p
from nems0.metrics.mi import mutual_information
from nems0.metrics.loglike import likelihood_poisson

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import \
    mplparams, get_significant_cells, SIG_TEST_MODELS, \
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

    sig_scores = db.batch_comp(batch, modelnames, cellids=significant_cells, stat='r_test')
    se_scores = db.batch_comp(batch, modelnames, cellids=significant_cells, stat='se_test')
    ceiling_scores = db.batch_comp(batch, modelnames, cellids=significant_cells, stat=PLOT_STAT)
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

def compare_metrics(batch=322, labels=None, colors=None, reload=True, L=10):
    """
    Supplemental figure - compare different evaluation metrics on LN and best CNN models
    Metrics: variance explained (r2  -- square of predxc used in the rest of the paper),
             mutual information (MI),
             Poisson log-likelihood (LL)
    :param batch:
    :param labels:
    :param colors:
    :param reload:
    :param L:
    :return:
    """
    if labels is None:
        labels = ['Prediction correlation','Mutual information']
    if colors is None:
        colors = ['black', 'black']
    df = pd.DataFrame()
    existing_cellids=[]
    if reload:
        try:
            df = pd.read_csv(f'xc_mi_L{L}.csv', index_col=0)
            existing_cellids = df.loc[np.isfinite(df['ll0'])].index.to_list()
        except:
            print('initializing df')

    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)
    #significant_cells = significant_cells[:50]

    modelnames = [ALL_FAMILY_MODELS[3],ALL_FAMILY_MODELS[2]]
    rc = db.batch_comp(batch, modelnames, cellids=significant_cells, stat='r_ceiling',
                       shortnames=['rc0','rc1'])
    rc.loc[rc['rc0']>1,'rc0']=1
    rc.loc[rc['rc1']>1,'rc1']=1
    significant_cells = [c for c in significant_cells if c not in existing_cellids]
    modelnames = [SIG_TEST_MODELS[1], SIG_TEST_MODELS[0]]

    for i, cellid in enumerate(significant_cells):
        p = load_pred(batch, [cellid], modelnames)[0]

        df.loc[cellid,'mi0'], n0 = mutual_information(p['pred0'].flatten(), p['resp'].flatten(), L=L)
        df.loc[cellid,'mi1'], n1 = mutual_information(p['pred1'].flatten(), p['resp'].flatten(), L=L)
        df.loc[cellid,'xc0'] = np.corrcoef(p['pred0'].flatten(), p['resp'].flatten())[0,1]
        df.loc[cellid,'xc1'] = np.corrcoef(p['pred1'].flatten(), p['resp'].flatten())[0,1]
        df.loc[cellid,'ll0'] = likelihood_poisson(x1=p['pred0'], x2=p['resp'])
        df.loc[cellid,'ll1'] = likelihood_poisson(x1=p['pred1'], x2=p['resp'])

        df.to_csv(f'xc_mi_L{L}.csv')

    df = df.dropna()
    print("df.shape", df.shape)
    df = df.merge(rc, how='inner', left_index=True, right_index=True)
    df['cc0'] = df['xc0']**2
    df['cc1'] = df['xc1']**2
    df['ccdiff'] = df['cc1']-df['cc0']
    df['midiff'] = df['mi1']-df['mi0']
    df['lldiff'] = df['ll1']-df['ll0']

    regopts = {'scatter_kws': {'s': 3, 'color': 'gray'},
               'line_kws': {'color': 'black', 'linestyle': '--', 'lw': 1},
               'fit_reg': True}
    fig, ax = plt.subplots(4,2, sharex='row',  sharey='row', figsize=(4,8))
    sns.regplot(data=df, x='cc0', y='mi0', ax=ax[0, 0], **regopts)
    cc, p = st.pearsonr(df['cc0'], df['mi0'])
    ax[0, 0].set_title(f"LN: r={cc:.3f} p={p:.3e}")
    ax[0, 0].set_xlabel('predxc**2')
    ax[0, 0].set_ylabel('MI')
    sns.regplot(data=df, x='cc1', y='mi1', ax=ax[0,1], **regopts)
    cc, p = st.pearsonr(df['cc1'], df['mi1'])
    ax[0, 1].set_title(f"CNN: r={cc:.3f} p={p:.3e}")
    ax[0, 1].set_xlabel('predxc**2')
    #ax[1, 0].set_visible(False)
    sns.regplot(data=df, x='ccdiff', y='midiff', ax=ax[1,1], **regopts)
    cc, p = st.pearsonr(df['ccdiff'], df['midiff'])
    ax[1, 1].set_title(f"improvement: r={cc:.3f} p={p:.3e}")
    ax[1, 1].set_xlabel('xc1^2 - xc0^2')
    ax[1, 1].set_ylabel('mi1 - mi0')

    sns.regplot(data=df, x='cc0', y='ll0', ax=ax[2, 0], **regopts)
    cc, p = st.pearsonr(df['cc0'], df['ll0'])
    print (cc, p, np.corrcoef(df['cc0'], df['ll0'])[0,1])
    ax[2, 0].set_title(f"LN: r={cc:.3f} p={p:.3e}")
    ax[2, 0].set_xlabel('predxc**2')
    ax[2, 0].set_ylabel('LL')
    sns.regplot(data=df, x='cc1', y='ll1', ax=ax[2, 1], **regopts)
    cc, p = st.pearsonr(df['cc1'], df['ll1'])
    ax[2, 1].set_title(f"CNN: r={cc:.3f} p={p:.3e}")
    ax[2, 1].set_xlabel('predxc**2')
    #ax[3, 0].set_visible(False)
    sns.regplot(data=df, x='ccdiff', y='lldiff', ax=ax[3, 1], **regopts)
    cc, p = st.pearsonr(df['ccdiff'], df['lldiff'])
    ax[3, 1].set_title(f"improvement: r={cc:.3f} p={p:.3e}")
    ax[3, 1].set_xlabel('xc1^2 - xc0^2')
    ax[3, 1].set_ylabel('ll1 - ll0')
    plt.tight_layout()

    fig2,ax=plt.subplots(1,3, figsize=(5,3))


    stats = ['Prediction correlation','Mutual information','Log likelihood']
    col0s = ["rc0","mi0","ll0"]
    col1s = ["rc1","mi1","ll1"]
    models = ['pop-LN','1Dx2-CNN']
    for a, s, c0, c1 in zip(ax, stats, col0s, col1s):

        dfmean = [df.loc[:,[c0]],df.loc[:,[c1]]]
        dfmean[0].columns=[s]
        dfmean[1].columns=[s]
        dfmean[0]['model'] = models[0]
        dfmean[1]['model'] = models[1]
        res = st.wilcoxon(dfmean[0][s].values, dfmean[1][s].values)


        dfmean=pd.concat(dfmean)
        sns.boxplot(data=dfmean, x='model', y=s, ax=a)
        #sns.boxplot(data=dfmean, x='model', y=s, estimator=np.median, ax=a)

        a.set_title(f'p={res.pvalue:.2e}')
    plt.tight_layout()

    return fig, fig2


def plot_conv_scatters(batch):
    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    sig_scores = db.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat='r_test')
    se_scores = db.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat='se_test')
    ceiling_scores = db.batch_comp(batch, ALL_FAMILY_MODELS, cellids=significant_cells, stat=PLOT_STAT)
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

    sig_scores = db.batch_comp(batch, MODELS, cellids=significant_cells, stat='r_test')
    se_scores = db.batch_comp(batch, MODELS, cellids=significant_cells, stat='se_test')
    ceiling_scores = db.batch_comp(batch, MODELS, cellids=significant_cells, stat=PLOT_STAT)
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
    r_values = db.batch_comp(batch, modelnames, cellids=cellids, stat=PLOT_STAT)

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

def plot_other_fitters(area='A1', dropna=False, exstr=None, do_plot=True, use_r_ceiling=True):
    dataroot = f'/auto/users/lbhb/data/'
    if area.startswith('PEG'):
        batch = 323
    else:
        batch = 322
    if (exstr is None):
        if area=='A1gtgram':
            exstr='lo5'
        else:
            exstr=''
    files = [f'2filt{exstr}',
             f'3filt{exstr}',
             f'lnp']
    dNIM2 = pd.read_csv(f'{dataroot}{area}/r_table_{files[0]}.csv')
    dNIM2['siteid'] = dNIM2['Cellid'].apply(db.get_siteid)
    dNIM2['chan'] = dNIM2['Cellid'].apply(lambda x: int(x.split("-")[1]))
    dNIM2['unit'] = dNIM2['Cellid'].apply(lambda x: int(x.split("-")[2]))
    dNIM3 = pd.read_csv(f'{dataroot}{area}/r_table_{files[1]}.csv')
    dNIM3['siteid'] = dNIM3['Cellid'].apply(db.get_siteid)
    dNIM3['chan'] = dNIM3['Cellid'].apply(lambda x: int(x.split("-")[1]))
    dNIM3['unit'] = dNIM3['Cellid'].apply(lambda x: int(x.split("-")[2]))
    dLNP = pd.read_csv(f'{dataroot}{area}/r_table_{files[2]}.csv')
    dLNP['siteid'] = dLNP['Cellid'].apply(db.get_siteid)
    dLNP['chan'] = dLNP['Cellid'].apply(lambda x: int(x.split("-")[1]))
    dLNP['unit'] = dLNP['Cellid'].apply(lambda x: int(x.split("-")[2]))
    cellids = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    dNEMS = db.batch_comp(batch, ALL_FAMILY_MODELS, cellids=cellids)
    dNEMS.columns = shortnames
    dNEMS = dNEMS.reset_index()
    dNEMS['siteid'] = dNEMS['cellid'].apply(db.get_siteid)
    dNEMS['chan'] = dNEMS['cellid'].apply(lambda x: int(x.split("-")[1]))
    dNEMS['unit'] = dNEMS['cellid'].apply(lambda x: int(x.split("-")[2]))

    d = dNEMS.merge(dNIM2, how='left', on = ['siteid','chan','unit'])
    d = d.merge(dNIM3, how='left', on = ['siteid','chan','unit'], suffixes=('_2','_3'))
    d = d.merge(dLNP, how='left', on = ['siteid','chan','unit'])
    if dropna:
        d=d.dropna()

    if use_r_ceiling:
        pop_LN = ALL_FAMILY_MODELS[3]
        rt = db.batch_comp(batch, [pop_LN], cellids=cellids, stat='r_test')
        rc = db.batch_comp(batch, [pop_LN], cellids=cellids, stat='r_ceiling')

        cfrac=rc / rt
        cfrac=cfrac.reset_index()
        cfrac.columns=['cellid','cfrac']

        d = d.merge(cfrac,how='inner',on='cellid')
        models = ['1Dx2-CNN', '2D-CNN', '1D-CNN', 'single-CNN', 'pop-LN',
                  'GLM_2', 'NIM_2', 'NIM_3', 'iSTAC', 'CBF', 'RBF']
        for c in models:
            d[c] *= d['cfrac']

        
    
    if do_plot:
        f,ax=plt.subplots(1,2, sharey=True,figsize=(12,5))
        sns.stripplot(data=d[models], s=2, ax=ax[0])
        sns.barplot(data=d[models], estimator=np.median, ax=ax[0])
        ax[0].set_ylabel('median')
        sns.stripplot(data=d[models], s=2, ax=ax[1])
        sns.barplot(data=d[models], estimator=np.mean, ax=ax[1])
        ax[1].set_ylabel('mean')

        for i,m in enumerate(models):
            y=ax[0].get_ylim()[1]
            n=np.sum(np.isfinite(d[m]))
            ax[0].text(i,y,f"{np.nanmedian(d[m]):.3f}/{n}", ha='center', rotation=90)
            ax[1].text(i, y, f"{np.nanmean(d[m]):.3f}/{n}", ha='center', rotation=90)
        plt.suptitle(f"{area} ({','.join(files)})")
        plt.tight_layout()
        
    return d

"""
batch=323
dNEMS = db.batch_comp(batch, ALL_FAMILY_MODELS)
dNEMS.columns=shortnames

"""

if __name__ == '__main__':
    fig,fig2 = compare_metrics(L=10)
    full_path = (base_path / 'figS_performance_metrics_scatter').with_suffix('.pdf')
    #fig.savefig(full_path, format='pdf', dpi=300)
    full_path = (base_path / 'figS_performance_metrics_box').with_suffix('.pdf')
    #fig2.savefig(full_path, format='pdf', dpi=300)

    #plot_other_fitters()
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
