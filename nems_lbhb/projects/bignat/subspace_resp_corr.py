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
from nems0.utils import shrinkage, smooth

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

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# set up paths, batches, modelnames

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

task = 'single_dstrf'

if task == 'single_dstrf':

    ####
    #### demo dSTRF code
    ####

    figpath='/home/svd/Documents/onedrive/projects/subspace_models/single_dstrf/'

    cell_list = ['CLT033c-010-1', 'CLT033c-029-1', 'CLT033c-063-1']
    cell_list = ["PRN022a-251-2", "PRN022a-252-1", "PRN022a-281-1", "PRN022a-289-1"]
    cell_list = ["PRN024a-255-1", "PRN024a-271-1"]
    cell_list = ["PRN027b-314-1", "PRN027b-322-1", "PRN027b-322-2", "PRN027b-338-1", "PRN027b-355-1"]
    cell_list = ["PRN029a-310-1", "PRN029a-317-1", "PRN029a-318-1", "PRN029a-319-1"]
    cell_list = ["PRN043a-322-4", "PRN043a-363-2"]
    cell_list = ["PRN048a-251-1", "PRN048a-303-1"]
    cell_list = ["PRN050a-220-2", "PRN050a-227-1", "PRN050a-246-1"]
    cell_list = ["CLT032c-021-2", "CLT032c-024-1", "CLT032c-045-2", "CLT032c-051-1"]
    cell_list = ["LMD052a-A-501-1", "LMD052a-A-539-1","LMD052a-A-584-1"]
    cell_list = ["PRN013c-271-2", "PRN013c-270-2","PRN013c-313-1","PRN013c-327-1"]
    #cell_list = ["PRN023a-260-2"]
    cell_list = ["CLT029c-014-1", "CLT029c-024-1"]

    cellid = cell_list[0]
    siteid = cellid.split("-")[0]
    #cellid = [c for c in cellids if c.startswith(siteid)][0]
    xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True,
                                                verbose=False)
    xfspec2, ctx2 = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelnames[1], eval_model=False,
                                                  verbose=False)
    modelspec = ctx['modelspec']
    modelspec2 = ctx2['modelspec']

    val=ctx['val']
    est=ctx['est']
    D=20
    modelspec_list = ctx['modelspec_list']
    timestep=30

    t_indexes = np.arange(295, val['stim'].shape[1], timestep)[:10]
    t_indexes = np.array([100,103,122,150,200,294,340,380,420,460])
    tcount = len(t_indexes)

    out_channels = [[i for i, c in enumerate(modelspec.meta['cellids']) if c == c0][0] for c0 in cell_list]
    cellcount = len(out_channels)

    stim = {'input': val['stim'].as_continuous().T}
    if 'dlc' in val.signals.keys():
        stim['dlc']=val['dlc'].as_continuous().T
    sig = 'input'
    dstrfs = []

    for mi, m in enumerate(modelspec_list):
        log.info(f"Computing dSTRF {mi + 1}/{len(modelspec_list)} at {tcount} points (timestep={timestep})")

        d = m.dstrf(stim, D=D, out_channels=out_channels, t_indexes=t_indexes, reset_backend=False)
        dstrfs.append(d[sig].astype(np.float32))  # to save memory

    dstrfs = np.stack(dstrfs, axis=1)

    # hack, zero-out top channel
    dstrfs[:,:,:,-1]=0

    s = np.std(dstrfs, axis=(2, 3, 4), keepdims=True)
    dstrfs /= s
    dstrfs /= np.max(np.abs(dstrfs), axis=(1, 2, 3, 4), keepdims=True)

    mdstrf = dstrfs.mean(axis=1, keepdims=True)
    if len(modelspec_list)>1:
        sdstrf = dstrfs.std(axis=1, keepdims=True)
        sdstrf[sdstrf == 0] = 1

        mzdstrf = shrinkage(mdstrf, sdstrf, sigrat=1)
        mdstrf = mzdstrf/np.max(np.abs(mzdstrf), axis=(2,3,4), keepdims=True)

    dpc = modelspec.meta.get('dpc',None)
    if dpc is None:
        pc_count=15
        import nems.tools.dstrf as dtools
        d = dtools.compute_dpcs(mdstrf[:, 0], pc_count=pc_count, as_dict=True)
        dpc = d[sig]['pcs']
        dpc_mag = d[sig]['pc_mag']
    else:
        dpc = dpc[out_channels].copy()
        dpc_mag = modelspec.meta['dpc_mag'][:,out_channels].copy()
        pc_count=dpc.shape[1]

    Y = dstrf.project_to_subspace(modelspec, X=stim['input'],
                                  out_channels=out_channels)
    emax=200000
    Ye = dstrf.project_to_subspace(modelspec, X=est['stim'].as_continuous()[:,:emax].T,
                                  out_channels=out_channels)

    # LN STRF -- modelspec2
    D=dpc.shape[-1]
    lnstrf = LN.LNpop_get_strf(modelspec2)[:,:D,:]
    lnstrf = np.moveaxis(lnstrf,(0,1,2),(1,2,0))[:,np.newaxis]

    lnpred = modelspec2.predict(stim['input'])[:,out_channels]

    importlib.reload(dstrf)
    Ylin = dstrf.project_to_subspace(modelspec2, dpc0=np.flip(lnstrf, axis=3), X=stim['input'], out_channels=out_channels)


    # plt.close('all')
    for o,cellid in enumerate(cell_list):
        oi = out_channels[o]
        cols=tcount
        pcp=np.min([3,pc_count])
        pred = est['pred'].as_continuous()[oi,:emax]
        spont = modelspec.meta['spont_mean'][oi]

        f = plt.figure(figsize=(10,6))

        sax = f.add_subplot(6,1,1)
        rax = f.add_subplot(6,1,2)
        jax = [f.add_subplot(9,1,4), f.add_subplot(9,1,5), f.add_subplot(9,1,6)]
        dax = [f.add_subplot(6, cols, 4*cols+i+1) for i in range(cols)]
        eax = f.add_subplot(6, cols, 5*cols+1)
        pax = [f.add_subplot(6, cols, 5*cols+i+2) for i in range(pcp)]
        hax = [f.add_subplot(6, cols, 5*cols+i*2+2+pcp) for i in range(1, pcp)]
        lax = f.add_subplot(6, cols, 6*cols)

        trange = np.arange(50,t_indexes[:cols].max()+50)
        tt=trange/val['resp'].fs

        sax.imshow(val['stim'].as_continuous()[:,trange], origin='lower', cmap='gray_r')
        sax.set_xticklabels([])

        # pred and actual
        p0 = lnpred[trange,o]
        rr = val['resp'].as_continuous()[oi,trange]
        p1 = val['pred'].as_continuous()[oi,trange]
        r0 = rr.mean()
        p1 = (p1-p1.mean()) / p1.std() * (rr - r0).std() + r0
        p0 = (p0-p0.mean()) / p0.std() * (rr - r0).std() + r0

        rax.plot(tt, p0, lw=0.5, color='darkgray')
        rax.plot(tt, rr, lw=0.5, color='gray')
        rax.plot(tt, p1, lw=0.5, color='k')
        rax.set_xlim([tt[0],tt[-1]])

        jax[0].plot(tt, Ylin[o, 0, trange], lw=1, color=CB_color_cycle[0])
        jax[0].set_xlim([tt[0], tt[-1]])

        for j in range(2):
            jax[j+1].plot(tt,Y[o,j,trange], lw=1, color=CB_color_cycle[j+1])
            jax[j+1].set_xlim([tt[0], tt[-1]])
            jax[j].set_xticklabels([])

        for i,t in enumerate(t_indexes[:cols]):
            rax.axvline(t/val['resp'].fs, lw=0.5, color='r')
            jax[0].axvline(t/val['resp'].fs, lw=0.5, color='r')
            jax[1].axvline(t/val['resp'].fs, lw=0.5, color='r')

        md = mdstrf[o,0,:cols]
        md = md/np.max(np.abs(md))
        for i in range(cols):
            dax[i].imshow(md[i], origin='lower', cmap='bwr', vmin=-0.8, vmax=0.8)
            dax[i].set_xticks([])
            dax[i].set_yticks([])

        dp = dpc_mag[:, o] / dpc_mag[:, o].sum()
        dpc_mag_sh=modelspec.meta['dpc_mag_sh']
        dpc_mag_e=modelspec.meta['dpc_mag_e']

        eax.plot(np.arange(1, len(dpc_mag) + 1), dp, 'o-', markersize=2)
        if dpc_mag_sh is not None:
            de = dpc_mag_e[:, oi] / dpc_mag_sh[:, oi].sum() * np.sqrt(50)
            dsh = dpc_mag_sh[:, oi] / dpc_mag_sh[:, oi].sum()
            for pp in range(1, pc_count):
                de[pp:] = de[pp:] / dsh[pp:].sum() * (1 - dp[:pp].sum())
                dsh[pp:] = dsh[pp:] / dsh[pp:].sum() * (1 - dp[:pp].sum())
            eax.errorbar(np.arange(1, len(dsh) + 1), dsh, de, color='gray', lw=0.5)
        eax.set_xticks(np.arange(1, pc_count + 1, 3))
        #eax.set_xticklabels([])
        #eax.set_yticklabels([])

        for i in range(pcp):
            d = dpc[o,i] / np.max(np.abs(dpc[o,i]))
            pax[i].imshow(d, origin='lower', cmap='bwr', vmin=-1, vmax=1)
            pax[i].set_xticks([])
            pax[i].set_yticks([])

            if i>0:
                histmean2d(Ye[o,0,:], Ye[o, i, :], pred, bins=20, ax=hax[i-1],
                           cmap='summer', spont=spont, ex_pct=0.025, minN=4)

                for aa,ti in enumerate(t_indexes[:10]):
                    hax[i-1].text(Y[o,0,ti], Y[o,i,ti], f"{aa}", fontsize=6, va="center", ha='center')
        d = np.flip(lnstrf[oi,0],axis=1) / np.max(np.abs(lnstrf[oi,0]))
        lax.imshow(d, origin='lower', cmap='bwr', vmin=-1, vmax=1)
        f.suptitle(f"{cellid} LN: {modelspec2.meta['r_test'][oi,0]:.3f} CNN: {modelspec.meta['r_test'][oi,0]:.3f}")
        plt.tight_layout()

        f.savefig(f"{figpath}{cellid}.jpg")

elif task == 'depth':

    ###
    ### Examples from different depths.
    ###

    figpath='/home/svd/Documents/onedrive/projects/subspace_models/examples/'

    # depth info
    df = db.batch_comp(batch, modelnames, shortnames=shortnames)

    df_cellinfo = depth.get_depth_details(siteids, verbose=False)
    df = df.merge(df_cellinfo, how='inner', left_index=True, right_index=True)
    types = ['NS','RS','ND','RD']

    plot_siteids = ["CLT029c", "CLT033c", "PRN022a",
                    "PRN055a", "LMD052a", "PRN013c"]
    for siteid in plot_siteids:
        cellid = [c for c in cellids if c.startswith(siteid)][0]
        xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True,
                                                    verbose=False)
        modelspec = ctx['modelspec']
        cell_list = modelspec.meta['cellids']
        r_test = modelspec.meta['r_test'][:, 0]

        si = np.argsort(-r_test)
        cell_list = [cell_list[i] for i in si]
        r_test = r_test[si]
        for r,c in zip(r_test,cell_list):
            print(f"{c} {r:.3f}")

        cell_lists = []
        for i,type in enumerate(types):
            cell_lists.append([c for c in cell_list if (df.loc[c,'celltype']==type)])

        plt.close('all')
        importlib.reload(dstrf)
        for t, cell_list in zip(types, cell_lists):
            if len(cell_list)>0:
                f = dstrf.plot_dpc_rows(cell_list=cell_list[:4], df=df,
                                        title=f"{siteid} - {t}", emax=200000, **ctx)
                f.savefig(f"{figpath}{siteid}_{t}_rows.jpg")

                f = dstrf.plot_dpc_all_3d(cell_list=cell_list[:6], use_dpc_all=True,
                                          title=f"{siteid} - {t}", **ctx)
                f.savefig(f"{figpath}{siteid}_{t}_3d.jpg")

elif task=='dump':
    # DUMP dPCS for best cells in all sites

    figpath='/home/svd/Documents/onedrive/projects/subspace_models/dump/'

    for siteid in siteids[40:]:
        cellid = [c for c in cellids if c.startswith(siteid)][0]
        xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True,
                                                    verbose=False)

        modelspec = ctx['modelspec']
        cell_list = modelspec.meta['cellids']
        r_test = modelspec.meta['r_test'][:,0]
        si = np.argsort(r_test)

        goodcellids=[c for o,c in enumerate(modelspec.meta['cellids']) if o in si[-6:]]

        importlib.reload(dstrf)
        f = dstrf.plot_dpc_rows(cell_list=goodcellids, df=df,
                                title=f"{siteid} - best pred", emax=200000, **ctx)
        f.savefig(f"{figpath}{siteid}_best_rows.jpg")

        #f = dstrf.plot_dpc_all_3d(cell_list=goodcellids, use_dpc_all=True,
        #                          title=f"{siteid} - best pred", **ctx)
        #f.savefig(f"{figpath}{siteid}_best_3d.jpg")


