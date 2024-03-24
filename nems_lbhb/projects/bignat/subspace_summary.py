from os.path import basename, join
import logging
import os
import io
import importlib

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from nems0.utils import escaped_split, escaped_join, get_setting
from nems0.registry import KeywordRegistry, xforms_lib
from nems0 import xform_helper, xforms, db

from nems.layers import filter
from nems import Model
from nems.metrics import correlation
from nems.preprocessing import split
from nems.models.dataset import DataSet
from nems.models import LN
from nems0.initializers import init_nl_lite

from nems0.registry import xform, scan_for_kw_defs
from nems.layers.tools import require_shape, pop_shape
from nems_lbhb.projects.bignat.bnt_defaults import POP_MODELS,SIG_TEST_MODELS
import nems_lbhb.plots as nplt
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
from nems_lbhb.plots import histscatter2d, histmean2d, scatter_comp
from nems_lbhb.analysis import dstrf, depth
from nems_lbhb import baphy_io
from nems_lbhb.projects.bignat import clustering_helpers

log = logging.getLogger(__name__)


CB_color_cycle = nplt.CB_color_cycle

# set up paths, batches, modelnames

batch=322
batch=343

siteids, cellids = db.get_batch_sites(batch)
save_figs=False

#task='predsum'
task='localsim'

figpath='/home/svd/Documents/onedrive/projects/subspace_models/'


chcount=32
first_lin=".nl"
#first_lin="" # first first "dPC" dim to be a linear STRF
ss="95"
#ss=""

if batch==343:
    # 2024-03-22
    # 32 channel stimulus with 15 pcs
    load_kw = 'gtgram.fs100.ch32-ld-norm.l1-sev'
    fit_kw = f"lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss{ss}{first_lin}"
    fit_kw_lin = 'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4'
    modelnames = [
        f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.s-wc.70x1x80.l2:4-fir.10x1x80-relu.80.s-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
        f'{load_kw}_wc.Nx1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_{fit_kw_lin}',
    ]
    shortnames = [f'CNN32-j8p15s{ss}{first_lin}','LN32-j8r5']

elif batch in [322,323]:
    load_kw = 'gtgram.fs100.ch32-ld-norm.l1-sev'
    fit_kw = 'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss95.nl'
    fit_kw_lin = 'lite.tf.init.lr1e3.t3.es20.jk5.rb3-lite.tf.lr1e4.t5e4'
    modelnames = [
        #f'{load_kw}_wc.Nx1x50.g-fir.15x1x50-relu.50.s-wc.50x1x60.l2:4-fir.10x1x60-relu.60.s-wc.60x80.l2:4-relu.80.s-wc.80xR.l2:4-dexp.R_{fit_kw}',
        #f'{load_kw}_wc.Nx1x100.g-fir.25x1x100-wc.100xR.l2:4-dexp.R_{fit_kw_lin}',
        f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.s-wc.70x1x80.l2:4-fir.10x1x80-relu.80.s-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
        f'{load_kw}_wc.Nx1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_{fit_kw_lin}',
    ]
    #shortnames=['CNN sm','LN sm']
    shortnames=['CNN lg','LN lg']

modelname=modelnames[0]

###
### Load performance and cell type/depth info
###

df_allcells = db.batch_comp(batch=batch,modelnames=modelnames,shortnames=shortnames, stat='r_test')
df_test = db.batch_comp(batch=batch,modelnames=modelnames,shortnames=shortnames)
dff = db.batch_comp(batch=batch,modelnames=modelnames,shortnames=shortnames, stat='r_floor')

df_allcells['siteid']=df_allcells.index
df_allcells['siteid']=df_allcells['siteid'].apply(db.get_siteid)
siteids = list(df_allcells['siteid'].unique())
for siteid in siteids:
    cellid = [c for c in cellids if c.startswith(siteid)][0]
    xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=False)
    for cellid,sspredxc,sm,em in zip(ctx['modelspec'].meta['cellids'], ctx['modelspec'].meta['sspredxc'],
                                    ctx['modelspec'].meta['spont_mean'], ctx['modelspec'].meta['evoked_mean']):
        df_allcells.loc[cellid,'sspredxc']=sspredxc
        df_allcells.loc[cellid,'spont_mean']=sm
        df_allcells.loc[cellid,'evoked_mean']=em

df_test['sspredxc']=df_allcells['sspredxc']
dff['sspredxc']=dff[shortnames[0]]
shortnames.append('sspredxc')

g=(df_test>dff).sum(axis=1)==df_test.shape[1]
df_allcells['goodpred']=g

df_allcells = df_allcells.loc[np.isfinite(df_allcells[shortnames].sum(axis=1))]

# change in r_test for CNN vs. LN
df_allcells['improve']=(df_allcells[shortnames[0]]-df_allcells[shortnames[1]]) / (df_allcells[shortnames].sum(axis=1))
shortlist=shortnames+['improve']

if batch==343:
    sw_thresh=0.35
else:
    sw_thresh=0.4

#dfinfo = None
try:
    assert (dfinfo is not None)
except:
    dfinfo = depth.get_depth_details(siteids, sw_thresh=sw_thresh)

df = df_allcells.merge(dfinfo, how='left', left_index=True, right_index=True, suffixes=('','_2'))

types=['NS','RS','ND','RD']

if task=='predsim':
    print(f"Excluding {(1-df['area'].isin(['A1','PEG'])).sum()}/{df.shape[0]} cells outside of A1/PEG")
    d_ = df.loc[df['area'].isin(['A1','PEG'])]

    f,ax=plt.subplots(1,3,figsize=(9,3))

    d_.groupby(['area','narrow','layer'])[shortnames[:2]].median().plot.bar(ax=ax[0])
    ax[0].set_ylabel('Median predxc')
    d_.groupby(['area','narrow','layer'])[shortnames[:1]].count().plot.bar(ax=ax[1])
    ax[1].set_ylabel('Number of cells')
    #dfa1 = df.loc[(df['area']=='A1') & df['layer'].isin(['13','44','56']) & df['goodpred']]
    dfa1 = d_.loc[ df['layer'].isin(['13','44','56']) & df['goodpred']]
    dfa1 = dfa1.groupby(['layer','narrow'])[shortnames[:1]].median().unstack(-1)
    dfa1.columns=['Regular','Narrow']
    dfa1.plot.bar(ax=ax[2], color=['darkgray','red'])
    ax[2].set_ylabel('Median pred. corr.')
    ax[2].set_xlabel('Cortical layer')
    ax[2].set_xticklabels(['1-3','4','5-6'])

    if save_figs:
        f.savefig(f"{figpath}predcomp_layers_{batch}_{shortnames[0]}_{shortnames[1]}.pdf")

    f,axs=plt.subplots(2,3,figsize=(6,5))#, sharex='col',sharey='col')
    for ax,area in zip(axs,['A1','PEG']):
        d_ = df.loc[df['goodpred']&(df['area']==area)]
        nplt.scatter_comp(d_[shortnames[1]],d_[shortnames[0]],
                          n1=shortnames[1],n2=shortnames[0],hist_range=[0,1], ax=ax[0], s=2);
        ax[0].set_title(area)
        nplt.scatter_comp(d_['sspredxc'], d_[shortnames[0]],
                          n1='Subspace',n2=shortnames[0],hist_range=[0,1], ax=ax[1], s=2);

        d_[[shortnames[1], shortnames[0], 'sspredxc']].median(numeric_only=True).plot.bar(ax=ax[2])
        ds=d_.groupby('siteid')[shortnames].mean(numeric_only=True)
        r01=stats.wilcoxon(ds[shortnames[1]],ds[shortnames[0]])
        r12=stats.wilcoxon(ds[shortnames[0]],ds[shortnames[2]])
        ax[2].set_ylabel('Median pred. corr.')
        ax[2].set_title(f"p01={r01.pvalue:.2e}, p12={r12.pvalue:.2e}")
        print(f"{area} {shortnames[1]} vs. {shortnames[0]} p={r01.pvalue:.3e}")
        print(f"{area} {shortnames[0]} vs. subspace p={r12.pvalue:.3e}")

    axs[0,2].set_xticklabels([])
    plt.tight_layout()

    if save_figs:
        f.savefig(f"{figpath}sspredsum_{batch}_{shortnames[0]}_{shortnames[1]}.pdf")

    dc = df.copy()
    dc=dc.loc[dc['area']=='A1']
    dc['Layer group'] = (np.floor(dc['depth'].astype(float)/100)*100)
    dc=dc.loc[(dc['area']=='A1') & (dc['Layer group']>-900)  & (dc['Layer group']<1200)]
    dc['type'] = "RS"
    dc.loc[dc['narrow'],'type']='NS'
    f,ax = plt.subplots(2,1, sharex='col', figsize=(6,4))
    dc.groupby(['Layer group','type'])[[shortnames[1]]].count().unstack(-1).plot.bar(ax=ax[0], color=['red', 'darkgray'])
    dc.groupby(['Layer group','type'])[[shortnames[1]]].median().unstack(-1).plot.bar(ax=ax[1], color=['red','darkgray'])
    ax[0].set_ylabel('N units')
    ax[1].set_ylabel('Median LN STRF pred corr')

    if save_figs:
        f.savefig(f"{figpath}predcomp_layer_detail_{batch}_{shortnames[1]}.pdf")

    f,ax =plt.subplots(1,2,figsize=(8,4))
    d_ = df.copy()
    d_['mean_rate'] = (d_['spont_mean']+d_['evoked_mean']) **0.5
    for i in range(4):
        sns.regplot(d_.loc[d_['sorted_class']==i], y=shortnames[0], x='mean_rate', scatter_kws={'s':3}, ax=ax[0], label=types[i])
        #print(d_.loc[d_['sorted_class']==i,['mean_rate',shortnames[0],shortnames[1]]].mean())

    ax[0].legend()
    #sns.scatterplot(d_, y='mean_rate', x=shortnames[0], hue='sorted_class', ax=ax[0])
    sns.scatterplot(d_, x='depth', y='mean_rate', hue='sorted_class', s=3, ax=ax[1])
    d_.groupby('sorted_class')[['mean_rate',shortnames[0],shortnames[1]]].mean()

elif task == 'localsim':
    a1siteids = [s for s in siteids if df.loc[df['siteid'] == s, 'area'].values[0] == 'A1']
    pegsiteids = [s for s in siteids if df.loc[df['siteid'] == s, 'area'].values[0] == 'PEG']
    log.info(f"A1 sites: {len(a1siteids)} PEG sites: {len(pegsiteids)}")

    f, ax = plt.subplots(1, 2, figsize=(4, 2))
    ax[0].hist((df['evoked_mean'] + df['spont_mean']) * 100, bins=np.linspace(0, 5, 20))
    ax[1].hist(df[shortnames[0]], bins=np.linspace(0, 1, 20))
    rate_bounds = [0, 0.75, 1.5, 4, 30]
    predxc_bounds = [-1, 0.1, 0.25, 0.5, 1]

    r_test_thr = 0.15
    dpc_var = 0.75
    verbose = False
    mdict = {}
    pc = 0
    use_pcs = None

    # groupby='rate'
    # groupby='pred'
    # ctypes = ['VL','L','H','VH']

    groupby = 'spike'
    ctypes = ['NS', 'RS', 'DN', 'DR']

    dtypes = ['d' + t for t in ctypes]
    df_site_sim = pd.DataFrame(columns=['area', 'siteid1', 'siteid2', 'site_rel'] + ctypes + dtypes)

    full_sim_dict = {}
    for area in ['A1', 'PEG']:
        full_sim_dict[area] = []
        area_siteids = [s for s in siteids if df.loc[df['siteid'] == s, 'area'].values[0] == area]
        for ss1, siteid1 in enumerate(area_siteids):
            if mdict.get(siteid1, None) is None:
                cellid1 = [c for c in cellids if c.startswith(siteid1)][0]
                xfspec1, ctx1 = xform_helper.load_model_xform(cellid=cellid1, batch=batch, modelname=modelname,
                                                              eval_model=False, verbose=False)
                modelspec1 = ctx1['modelspec']
                mdict[siteid1] = modelspec1
            else:
                modelspec1 = mdict[siteid1]

            goodcellids1 = [c for o, c in enumerate(modelspec1.meta['cellids']) if
                            (modelspec1.meta['r_test'][o, 0] > r_test_thr)]
            gg1 = np.array(
                [o for o, c in enumerate(modelspec1.meta['cellids']) if (modelspec1.meta['r_test'][o, 0] > r_test_thr)])
            cell_count1 = len(gg1)

            # sorted_class1 = df.loc[goodcellids1,'sorted_class'].values
            sorted_class1 = get_sorted_class(goodcellids1, groupby, df)
            si1 = np.argsort(sorted_class1)
            goodcellids1 = [goodcellids1[i] for i in si1]
            gg1 = gg1[si1]
            class1 = sorted_class1[si1]

            # site dpc similarity for each cell
            dpc_all = modelspec1.meta['dpc_all']
            dpc_mag_all = modelspec1.meta['dpc_mag_all'] ** 2
            dpc = modelspec1.meta['dpc'][gg1]
            dpc_mag = modelspec1.meta['dpc_mag'][:, gg1] ** 2
            pc_count = dpc_mag_all.shape[0]

            # dpc_var=0.9
            dmag = dpc_mag / dpc_mag.sum(axis=0, keepdims=True)
            dall = dpc_mag_all[:, 0] / dpc_mag_all[:, 0].sum()
            dsum = np.cumsum(dmag, axis=0)
            cellcount = dsum.shape[1]
            dm = np.argmax(dsum > dpc_var, axis=0)

            full_sim = np.zeros((cell_count1, pc_count))
            for pci in range(pc_count):
                for i in range(cell_count1):
                    ref_pc = dpc_all[0, [pci]].flatten()
                    test_pc = dpc[i, :(dm[i] + 1)]
                    x = np.array([np.corrcoef(ref_pc, t.flatten())[0, 1] for t in test_pc])
                    full_sim[i, pci] = ((x ** 2).sum())
            lmean = np.zeros((4, pc_count)) * np.nan
            for i in np.unique(class1):
                lmean[i] = full_sim[class1 == i].mean(axis=0)
                lmean[i] /= lmean[i].sum()
            full_sim_dict[area].append(lmean)

            # compare with other sites
            for ss2, siteid2 in enumerate(area_siteids[(ss1 % 2):(ss1 + 1):2]):
                cellid2 = [c for c in cellids if c.startswith(siteid2)][0]
                xfspec2, ctx2 = xform_helper.load_model_xform(cellid=cellid2, batch=batch, modelname=modelname,
                                                              eval_model=False, verbose=False)
                modelspec2 = ctx2['modelspec']
                if mdict.get(siteid2, None) is None:
                    cellid1 = [c for c in cellids if c.startswith(siteid2)][0]
                    xfspec2, ctx2 = xform_helper.load_model_xform(cellid=cellid2, batch=batch, modelname=modelname,
                                                                  eval_model=False, verbose=False)
                    modelspec2 = ctx2['modelspec']
                    mdict[siteid2] = modelspec2
                else:
                    modelspec2 = mdict[siteid2]

                goodcellids2 = [c for o, c in enumerate(modelspec2.meta['cellids']) if
                                (modelspec2.meta['r_test'][o, 0] > r_test_thr)]
                gg2 = np.array(
                    [o for o, c in enumerate(modelspec2.meta['cellids']) if (modelspec2.meta['r_test'][o, 0] > r_test_thr)])
                cell_count2 = len(gg2)

                # sorted_class2 = df.loc[goodcellids2,'sorted_class'].values
                sorted_class2 = get_sorted_class(goodcellids2, groupby, df)

                si2 = np.argsort(sorted_class2)
                goodcellids2 = [goodcellids2[i] for i in si2]
                gg2 = gg2[si2]
                class2 = sorted_class2[si2]

                if ss2 == 0:
                    clss = " ".join([f"{c}: {(sorted_class1 == i).sum()}" for i, c in enumerate(ctypes)])
                    print(f"{area} Site1 {siteid1} good cells: {cell_count1}/{len(modelspec1.meta['cellids'])}: {clss}")
                print(f"     Site2 {siteid2} good cells: {cell_count2}/{len(modelspec2.meta['cellids'])}")
                if (cell_count1 > 0) & (cell_count2 > 0):
                    dpc1 = modelspec1.meta['dpc'][gg1]
                    dpc2 = modelspec2.meta['dpc'][gg2]

                    if use_pcs is None:
                        dpc_mag1 = modelspec1.meta['dpc_mag'][:, gg1] ** 2
                        dpc_mag1 = dpc_mag1 / dpc_mag1.sum(axis=0, keepdims=True)
                        dsum1 = np.cumsum(dpc_mag1, axis=0)
                        pcpc1 = np.array([int(np.min(np.where(dsum1[:, i] > dpc_var)[0]) + 1) for i in range(cell_count1)])

                        dpc_mag2 = modelspec2.meta['dpc_mag'][:, gg2] ** 2
                        dpc_mag2 = dpc_mag2 / dpc_mag2.sum(axis=0, keepdims=True)
                        dsum2 = np.cumsum(dpc_mag2, axis=0)
                        pcpc2 = np.array([int(np.min(np.where(dsum2[:, i] > dpc_var)[0]) + 1) for i in range(cell_count2)])
                    else:
                        pcpc1 = np.zeros_like(gg1) + use_pcs
                        pcpc2 = np.zeros_like(gg2) + use_pcs

                    cc = np.zeros((cell_count1, cell_count2))
                    for g1 in range(cell_count1):
                        for g2 in range(cell_count2):
                            cc[g1, g2] = dpc_dist(dpc1[g1], dpc2[g2], p1count=pcpc1[g1], p2count=pcpc2[g2], metric='dcc')

                    cccat = np.zeros((4, 4)) * np.nan
                    pccat = np.zeros(4) * np.nan
                    for cl1 in range(4):

                        for cl2 in range(4):
                            g1 = (class1 == cl1)
                            g2 = (class2 == cl2)

                            if g1.sum() + g2.sum() > 0:
                                pccat[cl1] = np.mean(np.concatenate([pcpc1[g1], pcpc2[g2]]))

                            c_ = cc[g1][:, g2].copy()
                            if (cl1 == cl2) & (siteid1 == siteid2):
                                np.fill_diagonal(c_, np.nan)
                                if c_.shape[0] > 1:
                                    cccat[cl1, cl2] = np.nanmean(c_)
                            else:
                                if np.any(c_):
                                    cccat[cl1, cl2] = np.nanmean(c_)
                    df_site_sim.loc[pc, 'area'] = area
                    df_site_sim.loc[pc, 'siteid1'] = siteid1
                    df_site_sim.loc[pc, 'siteid2'] = siteid2
                    if (siteid1 == siteid2):
                        df_site_sim.loc[pc, 'site_rel'] = 'same'
                    else:
                        df_site_sim.loc[pc, 'site_rel'] = 'diff'
                    for i, t in enumerate(ctypes):
                        df_site_sim.loc[pc, t] = cccat[i, i]
                        df_site_sim.loc[pc, dtypes[i]] = pccat[i]
                    pc += 1

                    if verbose:
                        f, ax = plt.subplots(1, 2, figsize=(4, 2))
                        ax[0].imshow(cc, aspect='equal', vmin=0, vmax=0.9)
                        os1 = 0
                        os2 = 0
                        for cl in range(3):
                            os1 += (class1 == cl).sum()
                            os2 += (class2 == cl).sum()
                            ax[0].axhline(os1 - 0.5, color='white', ls='--')
                            ax[0].axvline(os2 - 0.5, color='white', ls='--')
                        ax[0].set_title(f"{siteid1} v. {siteid2}")
                        ax[1].plot(ctypes, np.diag(cccat))
                        plt.tight_layout()

    f,ax=plt.subplots(1,2, figsize=(6,2), sharex=True, sharey=True)

    full_sim = np.nanmean(np.stack(full_sim_dict['A1'], axis=2),axis=2).T
    ax[0].plot(full_sim-full_sim.mean(axis=1,keepdims=True), label=ctypes);
    ax[0].set_title(f'A1 - {groupby}')
    ax[0].set_xlabel('Site dPC dimension')
    ax[0].set_ylabel('Mean per-cell sim.')
    full_sim = np.nanmean(np.stack(full_sim_dict['PEG'], axis=2),axis=2).T
    ax[1].plot(full_sim-full_sim.mean(axis=1,keepdims=True), label=ctypes);
    ax[1].set_title(f'PEG - {groupby}')
    ax[1].set_xlabel('Site dPC dimension')
    ax[1].legend()

    f,ax=plt.subplots(4,2,figsize=(4,3), sharex=True)
    bins=np.linspace(0,1,16)
    for ai,area in enumerate(['A1','PEG']):
        d_aa = df_site_sim.loc[(df_site_sim['area']==area),ctypes]
        d_cc = df_site_sim.loc[(df_site_sim['area']==area) & (df_site_sim['site_rel']=='same'),ctypes]
        for a,t in zip(ax,ctypes):
            a[ai].hist(np.sqrt(d_aa[t].dropna().astype(float)), bins=bins, label='diff')
            a[ai].hist(np.sqrt(d_cc[t].dropna().astype(float)), bins=bins, label='same')
            if (ai==1) & (t==ctypes[0]):
                a[ai].legend()
            if ai==0:
                a[ai].set_ylabel(t)
    f.suptitle(f"{shortnames[0]} dpc_var={dpc_var} r_test_thr={r_test_thr} groupby={groupby}")

    f, ax = plt.subplots(2, 3, figsize=(6, 5))
    for a, area in enumerate(['A1', 'PEG']):
        for i, rel in enumerate(['same', 'diff']):
            d_cc = df_site_sim.loc[(df_site_sim['area'] == area) & (df_site_sim['site_rel'] == rel), ctypes]

            msum = d_cc.median()
            esum = d_cc.sem()

            # sns.stripplot(d_cc, s=2, ax=ax[a,i])
            ax[a, i].plot(ctypes, d_cc.T, lw=0.5, color='gray')
            ax[a, i].errorbar(ctypes, msum, esum, color='k')
            ax[a, i].set_title(f"{area}, {rel} site")
            ax[a, i].set_ylim([0, 1])

        d_n = df_site_sim.loc[(df_site_sim['area'] == area) & (df_site_sim['site_rel'] == 'same'), dtypes]
        msum = d_n.mean()
        esum = d_n.sem()
        ax[a, 2].errorbar(ctypes, msum, esum, color='k')
        ax[a, 2].set_title("mean dimcount")
        ax[a, 2].set_ylim([0, 4])
    f.suptitle(f"{shortnames[0]} dpc_var={dpc_var} r_test_thr={r_test_thr} groupby={groupby}")
    plt.tight_layout()

log.info(f'Subspace_summary done for task {task}')