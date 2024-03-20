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
from sklearn.cross_decomposition import CCA

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


# set up paths, batches, modelnames
figpath='/home/svd/Documents/onedrive/projects/subspace_models/examples/'

batch=343
siteids, cellids = db.get_batch_sites(batch)

# remove sites without a complete test set

first_lin=False
ss=""
load_kw = 'gtgram.fs100.ch32-ld-norm.l1-sev'
if first_lin:
    fit_kw = 'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss95'
else:
    fit_kw = f'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss{ss}.nl'
fit_kw_lin = 'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4'
modelnames = [
    f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.s-wc.70x1x80.l2:4-fir.10x1x80-relu.80.s-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
    # f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80.l2:4-fir.10x1x80-relu.80.f-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
    f'{load_kw}_wc.Nx1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_{fit_kw_lin}',
    # f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80.l2:4-fir.10x1x80-relu.80.f-wc.80x100.l2:4-relu.100.f-wc.100xR.l2:4-dexp.R_{fit_kw}',
]
modelname=modelnames[0]
shortnames = [f'CNN1d32-j8p15s{ss}', 'LN32-j8r5']

# depth info
df = db.batch_comp(batch, modelnames, shortnames=shortnames)

df_cellinfo = depth.get_depth_details(siteids, verbose=False)
df = df.merge(df_cellinfo, how='inner', left_index=True, right_index=True)
types = ['NS','RS','ND','RD']

siteid="PRN018a"
siteid="CLT033c"


plot_siteids = ["CLT029c", "CLT033c", "PRN022a",
                "PRN055a", "LMD052a", "PRN013c"]
plot_siteids = ["PRN055a", "LMD052a", "PRN013c"]
for siteid in plot_siteids:
    cellid = [c for c in cellids if c.startswith(siteid)][0]
    xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True,
                                                verbose=False)

    modelspec = ctx['modelspec']
    cell_list = modelspec.meta['cellids']
    for i,(r,c) in enumerate(zip(modelspec.meta['r_test'][:,0],cell_list)):
        print(f"{i} {c} {r:.3f}")

    plt.close('all')
    rs_cellids=[c for o,c in enumerate(modelspec.meta['cellids']) if (df.loc[c,'celltype']=='RS') & (modelspec.meta['r_test'][o, 0]>0.4)]
    ns_cellids=[c for o,c in enumerate(modelspec.meta['cellids']) if (df.loc[c,'celltype']=='NS') & (modelspec.meta['r_test'][o, 0]>0.2)]
    rd_cellids=[c for o,c in enumerate(modelspec.meta['cellids']) if (df.loc[c,'celltype']=='RD') & (modelspec.meta['r_test'][o, 0]>0.4)]
    nd_cellids=[c for o,c in enumerate(modelspec.meta['cellids']) if (df.loc[c,'celltype']=='ND') & (modelspec.meta['r_test'][o, 0]>0.2)]

    importlib.reload(dstrf)
    cell_lists = [ns_cellids, rs_cellids, nd_cellids, rd_cellids]
    for t, cell_list in zip(types, cell_lists):
        if len(cell_list)>0:
            f = dstrf.plot_dpc_rows(cell_list=cell_list[:4], df=df,
                                    title=f"{siteid} - {t}", emax=200000, **ctx)
            f.savefig(f"{figpath}{siteid}_{t}_rows.jpg")

            f = dstrf.plot_dpc_all_3d(cell_list=cell_list[:6], use_dpc_all=True,
                                      title=f"{siteid} - {t}", **ctx)
            f.savefig(f"{figpath}{siteid}_{t}_3d.jpg")



raise ValueError('done')

# pick which cells to display

goodcellids, goodlabel = ns_cellids, 'NS'


import importlib

plt.close('all')

importlib.reload(dstrf)




maxrows=15

cell_list=rs_cellids
rec=ctx['est']
orange = [rec['resp'].chans.index(c) for c in cell_list]
modelspec=ctx['modelspec']
X = rec['stim'].as_continuous().T
Y = dstrf.project_to_subspace(modelspec=modelspec, X=X, out_channels=orange,
                              use_dpc_all=False, verbose=True)

X = rec['stim'].as_continuous().T
Y = project_to_subspace(modelspec=modelspec, X=X, out_channels=orange,
                        use_dpc_all=False, verbose=True)

dpc = modelspec.meta['dpc'].copy()
dpc_mag = modelspec.meta['dpc_mag'].copy()
dpc_mag_sh = modelspec.meta['dpc_mag_sh']
dpc_mag_e = modelspec.meta['dpc_mag_e']
pc_count = dpc.shape[1]

spont=modelspec.meta['spont_mean']
rows = np.min([len(orange), maxrows])
f = plt.figure(figsize=(10, rows * 1))
gs = f.add_gridspec(rows + 1, 10)
ax = np.zeros((rows + 1, 8), dtype='O')
for r in range(rows + 1):
    for c in range(8):
        if (c < 7) & (r > 0):
            ax[r, c] = f.add_subplot(gs[r, c])
        elif (c >= 7):
            ax[r, c] = f.add_subplot(gs[r, c:])

# top row, just plot stim in one panel
T1 = 50
T2 = 550
ss = ctx['val']['stim'].as_continuous()[:, T1:T2]
ax[0, -1].imshow(ss, aspect='auto', origin='lower', cmap='gray_r')
ax[0, -1].set_yticklabels([])
ax[0, -1].set_xticklabels([])

for j_, (oi, cellid) in enumerate(zip(orange[:rows], cell_list[:rows])):
    j = j_ + 1
    print(oi, cellid)

    pred = rec['pred'].as_continuous().T[:, oi]
    r = rec['resp'].as_continuous().T[:, oi]

    pcp = 3
    for i in range(pcp):
        cc = np.corrcoef(Y[j_, i], r)[0, 1]
        if cc < 0:
            Y[j_, i] = -Y[j_, i]
            dpc[oi, i] = -dpc[oi, i]
    if 0 & (df is not None):
        mwf = df.loc[cellid, 'mwf']
        if df.loc[cellid, 'narrow']:
            ax[j, 0].plot(mwf, 'r', lw=1)
        else:
            ax[j, 0].plot(mwf, 'gray', lw=0.5)
        ax[j, 0].set_xticklabels([])
        ax[j, 0].set_yticklabels([])
    else:
        ax[j, 0].set_axis_off()

    dp = dpc_mag[:, oi] / dpc_mag[:, oi].sum()
    p1rat = dp[1] / dp[0]

    ax[j, 1].plot(np.arange(1, len(dpc_mag) + 1), dp, 'o-', markersize=2)
    if dpc_mag_sh is not None:
        de = dpc_mag_e[:, oi] / dpc_mag_sh[:, oi].sum() * np.sqrt(50)
        dsh = dpc_mag_sh[:, oi] / dpc_mag_sh[:, oi].sum()
        for pp in range(1, pc_count):
            de[pp:] = de[pp:] / dsh[pp:].sum() * (1 - dp[:pp].sum())
            dsh[pp:] = dsh[pp:] / dsh[pp:].sum() * (1 - dp[:pp].sum())
        ax[j, 1].errorbar(np.arange(1, len(dsh) + 1), dsh, de, color='gray', lw=0.5)
    ax[j, 1].set_ylabel(f"{cellid}")
    ax[j, 1].set_xticks(np.arange(1, pc_count + 1))
    ax[j, 1].set_xticklabels([])
    ax[j, 1].set_yticklabels([])

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
              'interpolation': 'none'}
    ymin, ymax = 1, 0
    for i in range(pcp):
        d = dpc[oi, i]
        d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]
        if i == 1:
            d *= p1rat

        ax[j, i + 2].imshow(np.fliplr(d), **imopts)
        ax[j, i + 2].set_xticklabels([])
        ax[j, i + 2].set_yticklabels([])

    # PC0 vs. PC1 heatmap
    ac, bc, Zresp, N = histmean2d(Y[j_, 0, :], Y[j_, 1, :], r, bins=20, ax=ax[j, -3], spont=spont[oi], ex_pct=0.025, Z=None)
    ac, bc, Zresp, N = histmean2d(Y[j_, 0, :], Y[j_, 2, :], r, bins=20, ax=ax[j, -2], spont=spont[oi], ex_pct=0.025, Z=None)
    ax[j, -2].set_yticklabels([])
    ax[j, -2].set_xticklabels([])
    ax[j, -3].set_yticklabels([])
    ax[j, -3].set_xticklabels([])

    # snippet of resp/pred PSTH
    rr = ctx['val']['resp'].as_continuous()[oi, T1:T2]
    pp = ctx['val']['pred'].as_continuous()[oi, T1:T2]
    pp = pp - pp.mean()
    r0 = rr.mean()
    pp = pp / pp.std() * (rr - r0).std()
    pp = pp + r0
    ax[j, -1].plot(pp, color='gray', lw=0.5)
    ax[j, -1].plot(rr, color='black', lw=0.5)
    ax[j, -1].set_yticklabels([])
    ax[j, -1].set_xticklabels([])
    ax[j, -1].set_xlim([0, T2 - T1])
    yl = ax[j, -1].get_ylim()
    depth = df.loc[cellid, 'depth']
    sw = df.loc[cellid, 'sw']
    ax[j, -1].text(0, yl[1], f"r={modelspec.meta['r_test'][oi, 0]:.3f} depth={depth} sw={sw:.2f}")

"""
r_test_thr=0.3
pc_mag_thr=0.01

site_list = siteids[::4]
site_data = []
for siteid in site_list:

    cellid = [c for c in cellids if c.startswith(siteid)][0]
    xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True, verbose=False)

    modelspec = ctx['modelspec']
    out_channels=np.arange(len(modelspec.meta['cellids']))
    goodcellids = [c for o,c in zip(out_channels,modelspec.meta['cellids']) \
                   if (modelspec.meta['r_test'][o, 0]>r_test_thr)]
    g = np.array([i for i,c in enumerate(modelspec.meta['cellids']) if c in goodcellids])
    print(f"{siteid} {len(goodcellids)}/{len(out_channels)} hi-pred units (r_test>{r_test_thr:.2f}) {g.sum()}:{len(goodcellids)} hi pc2")
    cc1 = modelspec.meta['r_test'][g,0]
    cc2 = modelspec.meta['sspredxc'][g]
    print(f"mean good predxc. orig {cc1.mean():.3f} ss: {cc2.mean():.3f}")

    dpcz = modelspec.meta['dpc_all']
    dpc_magz = modelspec.meta['dpc_mag_all']
    dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, 0]
    fir = filter.FIR(shape=dpcz.shape)
    fir['coefficients'] = np.flip(dpcz, axis=0)

    val = ctx['val']
    resp = val['resp']
    stim_epochs = resp.epochs.loc[resp.epochs.name.str.startswith('STIM_'),'name'].unique().tolist()
    s=val['stim'].extract_epochs(stim_epochs)
    X = np.concatenate([s_[0,:,:2000].T for k,s_ in s.items()], axis=0)
    Y = fir.evaluate(X)

    r = val['resp'].extract_epochs(stim_epochs)
    r = np.concatenate([r_[0,:,:2000].T for k,r_ in r.items()], axis=0)
    p = val['pred'].extract_epochs(stim_epochs)
    p = np.concatenate([p_[0,:,:2000].T for k,p_ in p.items()], axis=0)

    res = {'siteid': siteid, 'cellid': cellid,
           'modelspec': modelspec,
           'stim_epochs': stim_epochs,
           'val': val, 'X': X, 'Y': Y,
           'r': r, 'p': p,
           'goodcellids': goodcellids, 'g': g}
    site_data.append(res)



pc_count = dpcz.shape[1]
pc_use=10
res = dstrf.subspace_model_fit(pc_count=pc_use, dpc_var=0.8, out_channels=g,
                               use_dpc_all=True, single_fit=True,
                               return_all=True, units_per_layer=15, **ctx)

plt.close('all')
sc=len(site_data)
f,ax = plt.subplots(sc, sc, sharex=True, sharey=True)

for ii in range(len(site_data)):
    for jj in range(len(site_data)):
        pc = 5
        smag1 = site_data[ii]['Y'][:,:pc]
        smag2 = site_data[jj]['Y'][:,:pc]
        d = np.concatenate([smag1, smag2], axis=1)
        shi = d.sum(axis=1)>0
        cchi = np.corrcoef(d[shi,:].T) ** 2

        ax[ii,jj].imshow(cchi[pc:][:,:pc], aspect='equal', vmin=0, vmax=1)


            #ax[1].plot(d[:,[0,5]])
print('in site (inspace->outspace) vs out site (inspace->outspace)')
inout = []
for ii in range(len(site_data)):
    for jj in range(len(site_data)):
        if ii!=jj:

            smag = np.abs(site_data[ii]['Y'][:,:5]).sum(axis=1)
            g1 = site_data[ii]['g']
            g2 = site_data[jj]['g']
            d = np.concatenate([site_data[ii]['r'][:, g1],site_data[jj]['r'][:, g2]], axis=1)


            cc = np.corrcoef(d.T)
            msm = np.median(smag[smag>0.001])
            msm = np.median(smag)
            sshi = (smag>msm)
            sslo = (smag>=0.000) # & (smag<=msm)
            cchi = np.corrcoef(d[sshi,:].T)
            cclo = np.corrcoef(d[sslo,:].T)
            cchi1 = cchi[:len(g1)][:,:len(g1)]
            cclo1 = cclo[:len(g1)][:,:len(g1)]
            cchi2 = cchi[len(g1):][:,len(g1):]
            cclo2 = cclo[len(g1):][:,len(g1):]

            mcc1hi = (cchi1[np.triu_indices(cchi1.shape[0], k=1)]**2).mean()
            mcc1lo = (cclo1[np.triu_indices(cclo1.shape[0], k=1)]**2).mean()
            mcc2hi = (cchi2[np.triu_indices(cchi2.shape[0], k=1)]**2).mean()
            mcc2lo = (cclo2[np.triu_indices(cclo2.shape[0], k=1)]**2).mean()

            if 0:
                f,ax=plt.subplots(1,2, figsize=(6,3))

                ax[0].imshow(cclo, aspect='equal', vmin=-0.8,vmax=0.8)
                ax[0].set_title(f"N sslo={sslo.sum()} {mcc1lo:.3f} {mcc2lo:.3f}")
                ax[1].imshow(cchi, aspect='equal', vmin=-0.8,vmax=0.8)
                ax[1].set_title(f"N sshi={sshi.sum()} {mcc1hi:.3f} {mcc2hi:.3f}")

                f.suptitle(f"{site_data[ii]['siteid']} ({len(g1)}) v {site_data[jj]['siteid']} ({len(g2)})")

            print(f"{site_data[ii]['siteid']} ({len(g1)}) {mcc1hi:.3f}->{mcc1lo:.3f} v {site_data[jj]['siteid']} ({len(g2)})  {mcc2hi:.3f}->{mcc2lo:.3f}")
            inout.append(np.array([mcc1hi, mcc1lo, mcc2hi, mcc2lo]))

inoutsum = np.stack(inout,axis=0)
f,ax=plt.subplots(1,2)

hist_range = [inoutsum.min(), inoutsum.max()]

nplt.scatter_comp(inoutsum[:,0],inoutsum[:,1],n1='in site space', n2='out site space',
                  hist_range=hist_range, ax=ax[0])
nplt.scatter_comp(inoutsum[:,2],inoutsum[:,3],n1='in ctrl space', n2='out ctrl space',
                  hist_range=hist_range, ax=ax[1])
"""