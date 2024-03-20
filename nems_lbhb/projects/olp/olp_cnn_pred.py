import numpy as np
import os
import io
import logging
import time

import matplotlib.pyplot as plt
import sys, importlib
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns

import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join
from nems0 import db
from nems0 import get_setting
from nems0.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems0.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems0.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
#from nems_lbhb import baphy_experiment
from nems_lbhb import baphy_io

from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems_lbhb.projects.olp.OLP_get_epochs import get_rec_epochs, get_stim_type, generate_cc_dataframe
from nems_lbhb.projects.olp import OLP_get_epochs

log = logging.getLogger(__name__)


def add_noise(psth, reps=20, prat=1):
    p = psth.copy()
    p[p < 0] = 0
    r = np.random.poisson(prat * p[:, np.newaxis], (len(p), reps)) + \
        np.random.normal((1 - prat) * p[:, np.newaxis], (1 - prat) * p[:, np.newaxis] / 2, (len(p), reps))
    r[r < 0] = 0

    return r.mean(axis=1)

def fb_weights(rfg, rbg, rfgbg, spontbins=50):
    spont = np.concatenate([rfg[:spontbins], rbg[:spontbins], rfgbg[:spontbins]]).mean()
    rfg0 = rfg - spont
    rbg0 = rbg - spont
    rfgbg0 = rfgbg - spont
    y = rfgbg0 - rfg0 - rbg0
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), y, rcond=None)
    # weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), rfgbg0, rcond=None)

    return weights2 + 1


def pred_comp(batch=341, modelnames=None):
    if modelnames is None:
        if batch == 341:
            modelnames = [
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
            ]
        else:
            modelnames = [
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            ]
    df = db.batch_comp(batch,modelnames, stat='r_ceiling', shortnames=['CNN','LN'])
    dftest = db.batch_comp(batch,modelnames, stat='r_test', shortnames=['CNN','LN'])
    dffloor = db.batch_comp(batch,modelnames, stat='r_floor', shortnames=['CNN','LN'])
    df = df.merge(dftest, left_index=True, right_index=True, suffixes=['','_test'])
    df = df.merge(dffloor, left_index=True, right_index=True, suffixes=['','_floor'])
    df['keepidx']=(df['LN_test']>df['LN_floor']) & (df['CNN_test']>df['CNN_floor'])

    d_ = df.loc[df['keepidx']]
    f=plt.figure()
    ax=f.add_subplot(1,2,1)
    ax.scatter(d_['LN'],d_['CNN'],s=3,color='black')
    ax.plot([0,1],[0,1],'k--')

    ax=f.add_subplot(1,4,3)
    sns.barplot(d_[['LN','CNN']], ax=ax)

    print(f"f.savefig('/home/svd/Documents/onedrive/projects/olp/batch{batch}_pred_comp.pdf')")
    return f

def get_valid_olp_rows(cell_epoch_df, minresp=0.01, mingain=0.03, maxgain=2.0,
                       exclude_low_pred=True, AC_only=True):
    """
    Select a "good" subset of rows from cell_epoch_df, the output of compare_olp_preds
    :param cell_epoch_df: dataframe generated by compare_olp_preds
    :param minresp: minimum normalized response (mean evoked response after normalizing full response between 0 and 1
    :param mingain: minimum fg and bg gain for actual and predicted response
    :param maxgain: maximum fg and bg gain for actual and predicted
    :param exclude_low_pred: bool
       default True. throw out cells for which LN and CNN model predictions are below r_floor
    :param AC_only: bool
        default True. remove cells not in A1 or PEG (or AC or BRD)
    :return: pd.DataFrame
        subset of cell_epoch_df
    """

    rFB = (cell_epoch_df['rfg'] > minresp) & (cell_epoch_df['rbg'] > minresp) & \
          (cell_epoch_df['rwfg'] > mingain) & (cell_epoch_df['rwbg'] > mingain) & \
          (cell_epoch_df['rwfg'] < maxgain) & (cell_epoch_df['rwbg'] < maxgain) & \
          (cell_epoch_df['p1wfg'] > mingain) & (cell_epoch_df['p1wbg'] > mingain) & \
          (cell_epoch_df['p1wfg'] < maxgain) & (cell_epoch_df['p1wbg'] < maxgain) & \
          (cell_epoch_df['p2wfg'] > mingain) & (cell_epoch_df['p2wbg'] > mingain) & \
          (cell_epoch_df['p2wfg'] < maxgain) & (cell_epoch_df['p2wbg'] < maxgain)

    if exclude_low_pred:
        keepidx = (cell_epoch_df['LN'] > cell_epoch_df['LN_floor']) & \
                  (cell_epoch_df['CNN'] > cell_epoch_df['CNN_floor'])
        rFB = (rFB & keepidx)
    if AC_only:
        rFB = (rFB & cell_epoch_df['area'].isin(['A1', 'PEG', 'BRD', 'AC']))

    return cell_epoch_df.loc[rFB].copy()


def compare_olp_preds(siteid, batch=341, modelnames=None, verbose=False):
    """
    returns cell_epoch_df DataFrame with many columns
    
    'BG + FG', 'BG', 'FG' : FG/BG pairing info
    'Synth Type', 'Binaural Type', 'Dynamic Type', 'SNR': other OLP parameters
    'cellid', 'area', 'iso': unit name, area and isolation
    'cellstd' : std of response across all stimuli
    'rfg', 'rbg' : mean evoked fg, bg response
    'p1fg', 'p1bg' : mean response predicted by model 1 (CNN)
    'p2fg', 'p2bg' : mean response predicted by model 2 (LN)
    'rwfg', 'rwbg', 'p1wfg', 'p1wbg', 'p2wfg', 'p2wbg' : fg, bg weights for actual and predicted response
    'rw0','rwfg0', 'rwbg0' : actual weights for 1/2 of the reps--used to calculate noise levels in actual response
    'reps', number of times fgbg stimulus was presented
    'sf', : How predicted weights should be scaled to account for noise affecting weights of actual response
    'CNN', 'LN' : prediction correlation for each model
    'CNN_floor', 'LN_floor' : noise floor on prediction (shuffled in time). "good" models have prediction correlation> noise floor
    
    """
    if modelnames is None:
        modelnames = [
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
            ]

    siteids, cellids = db.get_batch_sites(batch=batch)
    cellid = [c for s,c in zip(siteids, cellids) if s==siteid]
    
    df = db.batch_comp(batch=batch, modelnames=modelnames, shortnames=["CNN","LN"])
    dffloor = db.batch_comp(batch=batch, modelnames=modelnames, shortnames=["CNN","LN"], stat='r_floor')
    df = df.merge(dffloor, how='inner', left_index=True, right_index=True, suffixes=('','_floor'))

    try:
        di = baphy_io.get_depth_info(siteid=siteid)
        df = df.merge(di, how='left', left_index=True, right_index=True)
    except:
        df['siteid'] = df['cellid'].apply(db.get_siteid)
        df['area'] = 'unknown'
    dfsite = df.loc[df['siteid']==siteid]
    print(dfsite.shape, dfsite['area'].unique(), dfsite.columns)
    
    xf,ctx=load_model_xform(cellid, batch, modelnames[0])
    xf2,ctx2=load_model_xform(cellid, batch, modelnames[1])

    if verbose:
        modelspec=ctx['modelspec']
        modelspec2=ctx2['modelspec']

        f,ax=plt.subplots(1,2,figsize=(12,4))
        score='r_test'
        ax[0].plot(modelspec.meta[score], label=f"m1: {modelspec.meta[score].mean():.3f}")
        ax[0].plot(modelspec2.meta[score], label=f"m2: {modelspec2.meta[score].mean():.3f}")
        ax[0].axvline(cid0, color='black', linestyle='--')
        ax[0].set_title(score);
        ax[0].set_xlabel('unit')
        ax[0].set_ylabel('pred xc')
        ax[0].legend(fontsize=8);
        score='r_fit'
        ax[1].plot(modelspec.meta[score], label=f"m1: {modelspec.meta[score].mean():.3f}")
        ax[1].plot(modelspec2.meta[score], label=f"m2: {modelspec2.meta[score].mean():.3f}")
        ax[1].axvline(cid0, color='black', linestyle='--')
        ax[1].set_title(score);
        ax[1].legend(fontsize=8);
        f.suptitle(siteid);

    rec = ctx['val']
    rec2 = ctx2['val']
    
    stim = rec['stim'].rasterize()
    resp = rec['resp'].rasterize()
    pred1 = rec['pred'].rasterize()
    pred2 = rec2['pred'].rasterize()
    fs = resp.fs

    epoch_df_all = OLP_get_epochs.get_rec_epochs(fs=100, rec=rec)

    if batch==341:
        epoch_df = epoch_df_all.loc[(epoch_df_all['Dynamic Type']=='fullBG/fullFG') & (epoch_df_all['SNR']==0) & (epoch_df_all['Synth Type']=='Unsynthetic')]
        if len(epoch_df)==0:
            epoch_df = epoch_df_all.loc[(epoch_df_all['Dynamic Type']=='fullBG/fullFG') & (epoch_df_all['SNR']==0) & (epoch_df_all['Synth Type']=='Non-RMS Unsynthetic')]

    elif batch==345:
        #epoch_df = epoch_df_all.loc[(epoch_df_all['Binaural Type']=='BG Ipsi, FG Contra')]
        epoch_df = epoch_df_all.loc[(epoch_df_all['Binaural Type']=='BG Ipsi, FG Contra') | (epoch_df_all['Binaural Type']=='BG Contra, FG Contra')]

    L = []
    for cellid in resp.chans:
        e = epoch_df.copy()
        e['cellid'] = cellid
        e['cellstd'] = resp.extract_channels([cellid]).as_continuous().std()

        L.append(e)
    cell_epoch_df = pd.concat(L, ignore_index=True)

    prefs = ['r', 'p1', 'p2']
    sigs = [resp, pred1, pred2]
    for pre in prefs:
        cell_epoch_df[pre + 'fg'] = np.nan
        cell_epoch_df[pre + 'bg'] = np.nan
        cell_epoch_df[pre + 'wfg'] = np.nan
        cell_epoch_df[pre + 'wbg'] = np.nan
    cell_epoch_df['rw0'] = np.nan
    cell_epoch_df['rwfg0'] = np.nan
    cell_epoch_df['rwbg0'] = np.nan
    cell_epoch_df['reps'] = 0
    cell_epoch_df.shape

    sp = int(fs * 0.5)
    for id, r in cell_epoch_df.iterrows():
        cellid = r['cellid']
        epoch_fg = r['FG']
        epoch_bg = r['BG']
        epoch_fgbg = r['BG + FG']

        cell_epoch_df.at[id, 'reps'] = int(ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_fgbg).shape[0])
        rfgbg1 = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_fgbg)[1::2, :, :].mean(axis=0)[0, :]
        rfgbg2 = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_fgbg)[::2, :, :].mean(axis=0)[0, :]
        spont = (rfgbg1[:sp].mean() + rfgbg2[:sp].mean()) / 2

        # weights0,_,_,_ = np.linalg.lstsq(np.stack([rfgbg1-spont, np.ones_like(rfgbg1)], axis=1), rfgbg2-spont, rcond=None)
        weights0 = fb_weights(rfgbg1, np.ones_like(rfgbg1) * spont, rfgbg2, sp)

        # special 1/2 trials computation of FG/BG weights
        rfg = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_fg)[::2, :, :].mean(axis=0)[0, :] / r[
            'cellstd']
        rbg = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_bg)[::2, :, :].mean(axis=0)[0, :] / r[
            'cellstd']
        rfgbg = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_fgbg)[::2, :, :].mean(axis=0)[0, :] / r[
            'cellstd']

        weights2 = fb_weights(rfg, rbg, rfgbg, spontbins=sp)
        cell_epoch_df.at[id, 'rwfg0'] = weights2[0]
        cell_epoch_df.at[id, 'rwbg0'] = weights2[1]
        cell_epoch_df.at[id, 'rw0'] = weights0[0]

        for ii, pre, sig in zip(range(len(prefs)), prefs, sigs):
            rfg = sig.extract_channels([cellid]).extract_epoch(epoch_fg).mean(axis=0)[0, :] / r['cellstd']
            rbg = sig.extract_channels([cellid]).extract_epoch(epoch_bg).mean(axis=0)[0, :] / r['cellstd']
            rfgbg = sig.extract_channels([cellid]).extract_epoch(epoch_fgbg).mean(axis=0)[0, :] / r['cellstd']

            if ii > 20:
                rfg = add_noise(rfg, reps=15)
                rbg = add_noise(rbg, reps=15)
                rfgbg = add_noise(rfgbg, reps=15)

            spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3
            weights2 = fb_weights(rfg, rbg, rfgbg, sp)

            cell_epoch_df.at[id, pre + 'wfg'] = weights2[0]
            cell_epoch_df.at[id, pre + 'wbg'] = weights2[1]
            cell_epoch_df.at[id, pre + 'fg'] = rfg[sp:-sp].mean() - spont
            cell_epoch_df.at[id, pre + 'bg'] = rbg[sp:-sp].mean() - spont

    # link to area labels
    cell_epoch_df = cell_epoch_df.merge(dfsite[['area','iso','CNN','LN','CNN_floor','LN_floor']], how='left', left_on='cellid',
                                        right_index=True)

    # figure out an adjustment to account for the effec tof noisy responses on the estimate for fg/bg weights
    # exlude outliers to make fit stable.
    d = get_valid_olp_rows(cell_epoch_df, minresp=0.1, mingain=0.1, maxgain=2.0, exclude_low_pred=False)

    ww, residual_sum, rank, singular_values = np.linalg.lstsq(
        np.stack([d['rwbg0'],np.zeros_like(d['rwbg'])], axis=1),
                 d['rwbg'], rcond=None)
    print(f"ratio wbg10 / wbg5: {ww[0]:.3f} const={ww[1]:.3f} rw0 mean: {cell_epoch_df['rw0'].mean():.3f}")

    sf = cell_epoch_df['rw0']*ww[0]
    sf[sf<0.1]=0.1
    cell_epoch_df['sf']=sf

    return cell_epoch_df, rec, rec2


def plot_olp_preds(cell_epoch_df, minresp=0.01, mingain=0.03, maxgain=2.0, exclude_low_pred=True):
    
    # original
    #minresp, mingain, maxgain = 0.05, 0, 2.0
    
    siteid = db.get_siteid(cell_epoch_df['cellid'].values[0])
    area = ",".join(list(cell_epoch_df['area'].unique()))
    original_N = cell_epoch_df.shape[0]

    cell_epoch_df = get_valid_olp_rows(cell_epoch_df, minresp=minresp, mingain=mingain, maxgain=maxgain, exclude_low_pred=False)

    rCC = cell_epoch_df['Binaural Type']=='BG Contra, FG Contra'
    rIC = cell_epoch_df['Binaural Type']=='BG Ipsi, FG Contra'
    labels=['BG Contra, FG Contra','BG Ipsi, FG Contra']
    print('valid n', cell_epoch_df.shape[0], 'out of', original_N, 'frac: ', np.round(cell_epoch_df.shape[0]/original_N,3))
    
    if rIC.sum()==0:
        rrset = [slice(None)]
        figlabels = ['mono']
    else:
        rrset = [rCC, rIC]
        figlabels = ['CC', 'CI']

    for rr, figlabel in zip(rrset, figlabels):
        f, axs = plt.subplots(2, 3, figsize=(8, 6))
        labels = ['CNN pred', 'LN pred']
        prefs = ['r', 'p1', 'p2']
        for pre, ax, label in zip(prefs[1:], axs, labels):
            ax[0].plot([0, 1], [0, 1], 'k--')
            ax[1].plot([0, 1], [0, 1], 'k--')
            ax[2].plot([-1.2, 1], [-1.2, 1], 'k--')

            fr = cell_epoch_df.loc[rr, 'rwfg']
            fp = cell_epoch_df.loc[rr, pre + 'wfg'] * cell_epoch_df.loc[rr, 'sf']

            br = cell_epoch_df.loc[rr, 'rwbg']
            bp = cell_epoch_df.loc[rr, pre + 'wbg'] * cell_epoch_df.loc[rr, 'sf']

            dr = fr - br
            dp = fp - bp
            ax[0].scatter(fr, fp, s=2)
            ax[0].set_title(f"r={np.corrcoef(fr, fp)[0, 1]:.3f} mse={np.std(fr-fp)/np.std(fr):.3f}")
            ax[0].set_xlim([-0.1, 1.5])
            ax[0].set_ylim([-0.1, 1.5])
            ax[0].set_ylabel(label + ' w_fg')
            ax[0].set_xlabel(f'Actual w_fg ({fr.mean():.3f})')

            ax[1].scatter(br, bp, s=2)
            ax[1].set_xlim([-0.1, 1.5])
            ax[1].set_ylim([-0.1, 1.5])
            ax[1].set_title(f"r={np.corrcoef(br, bp)[0, 1]:.3f} mse={np.std(br-bp)/np.std(br):.3f}")
            ax[1].set_ylabel(label + ' w_bg')
            ax[1].set_xlabel(f'Actual w_bg ({br.mean():.3f})')

            ax[2].scatter(dr, dp, s=2)
            ax[2].set_xlim([-1.2, 1])
            ax[2].set_ylim([-1.2, 1])
            ax[2].set_title(f"r={np.corrcoef(dr, dp)[0, 1]:.3f} mse={np.std(dr-dp)/np.std(dr):.3f}")
            ax[2].set_ylabel(f'{label} relative gain ({dp.mean():.3f})')
            ax[2].set_xlabel(f'Actual relative gain ({dr.mean():.3f})')
        f.suptitle(f'{siteid} {figlabel} ({area})')
        plt.tight_layout()
    
    return f


def olp_pred_example(cell_epoch_df, rec, rec2, cellid, estim):

    prefs = ['r', 'p2', 'p1']
    labels = ['CNN pred', 'LN pred']
    resp=rec['resp']
    stim=rec['stim']
    sigs = [rec['resp'], rec2['pred'], rec['pred']]

    ii=cell_epoch_df.loc[(cell_epoch_df['cellid']==cellid) &
                      (cell_epoch_df['BG + FG']==estim)].index.values

    efg = cell_epoch_df.loc[ii, 'FG'].values[0]
    ebg = cell_epoch_df.loc[ii, 'BG'].values[0]
    efgbg = cell_epoch_df.loc[ii, 'BG + FG'].values[0]


    # weights,residual_sum,rank,singular_values = np.linalg.lstsq(np.stack([rfg, rbg, np.ones_like(rfg)], axis=1), rfgbg,rcond=None)
    # print(np.round(weights,2))
    rfg = resp.extract_channels([cellid]).extract_epoch(efg).mean(axis=0)[0, :]
    rbg = resp.extract_channels([cellid]).extract_epoch(ebg).mean(axis=0)[0, :]
    rfgbg = resp.extract_channels([cellid]).extract_epoch(efgbg).mean(axis=0)[0, :]

    fs=rec['resp'].fs
    sp = int(fs * 0.5)
    spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3

    rfg0 = rfg - spont
    rbg0 = rbg - spont
    rfgbg0 = rfgbg - spont
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), rfgbg0, rcond=None)
    #print(np.round(weights2, 2))
    weights2 = fb_weights(rfg, rbg, rfgbg, sp)
    #print(np.round(weights2, 2))

    imopts = {'origin': 'lower', 'interpolation': 'none', 'cmap': 'gray_r', 'aspect': 'auto'}

    f, ax = plt.subplots(4, 3, sharey='row', figsize=(12, 6), sharex='row')

    sfg = stim.extract_epoch(efg).mean(axis=0)
    sbg = stim.extract_epoch(ebg).mean(axis=0)
    sfgbg = stim.extract_epoch(efgbg).mean(axis=0)

    ax[0, 0].imshow(sbg, **imopts)
    ax[0, 1].imshow(sfg, **imopts)
    ax[0, 2].imshow(sfgbg, **imopts)

    ax[0, 0].set_title(cellid + " " + ebg)
    ax[0, 1].set_title(efg)
    ax[0, 2].set_title(efgbg)
    for kk, pre, sig in zip(range(len(sigs)), prefs, sigs):

        rfg = sig.extract_channels([cellid]).extract_epoch(efg).mean(axis=0)[0, :]
        rbg = sig.extract_channels([cellid]).extract_epoch(ebg).mean(axis=0)[0, :]
        rfgbg = sig.extract_channels([cellid]).extract_epoch(efgbg).mean(axis=0)[0, :]

        if kk > 30:
            rfg = add_noise(rfg)
            rbg = add_noise(rbg)
            rfgbg = add_noise(rfgbg)

        spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3
        rfg0 = rfg - spont
        rbg0 = rbg - spont
        rfgbg0 = rfgbg - spont
        weights2 = fb_weights(rfg, rbg, rfgbg, sp)

        ax[kk + 1, 0].plot(rbg0)
        ax[kk + 1, 1].plot(rfg0)
        ax[kk + 1, 2].plot(rfgbg0, label='resp')
        ax[kk + 1, 2].plot((rfg0) * weights2[0] + (rbg0) * weights2[1],label='wsum')
        ax[kk + 1, 2].axhline(0, color='r', lw=0.5)
        ax[kk + 1, 2].set_title(f'{pre}wfg={weights2[0]:.2f} {pre}wbg={weights2[1]:.2f}', fontsize=10)
    ax[1,2].legend(frameon=False)
    plt.tight_layout()

if __name__ == '__main__':
    batch = 341
    siteids, cellids = db.get_batch_sites(batch=batch)

    if batch == 341:
        modelnames = [
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
        ]
    else:
        modelnames = [
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
        ]

    cid = 35
    siteid, cellid = siteids[cid], cellids[cid]

    cell_epoch_df, rec1, rec2 = compare_olp_preds(siteid, batch=batch, modelnames=modelnames,
                                                  verbose=False)

    f = plot_olp_preds(cell_epoch_df, minresp=0.01, mingain=0.01, maxgain=2.0, exclude_low_pred=True)

    d = get_valid_olp_rows(cell_epoch_df, minresp=0.1, mingain=0.1, maxgain=2.0, exclude_low_pred=False)
    d = d.loc[d['area'].isin(['A1','PEG'])]
    rCC = d['Binaural Type']=='BG Contra, FG Contra'
    rIC = d['Binaural Type']=='BG Ipsi, FG Contra'

    d_big_gain_diff = d.loc[(cell_epoch_df['rwfg'] < 0.3) & (cell_epoch_df['rwbg'] > 0.7)]

    for i, r in d_big_gain_diff.iterrows():
        cellid = r['cellid']
        estim = r['BG + FG']
        print(f"{cellid} {estim} FG/BG weights: actual: {r['rwfg']:.2f}/{r['rwbg']:.2f} CNN: {r['p1wfg']:.2f}/{r['p1wbg']:.2f} LN: {r['p2wfg']:.2f}/{r['p2wbg']:.2f}")
        olp_pred_example(cell_epoch_df, rec1, rec2, cellid, estim)


batch = 341
siteids = db.get_batch_sites(batch=batch)

if batch == 341:
    modelnames = [
        "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
        "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
    ]

siteid = 'PRN022a'
cell_epoch_df, rec1, rec2 = compare_olp_preds(siteid, batch=batch, modelnames=modelnames,
                                              verbose=False)
d = get_valid_olp_rows(cell_epoch_df, minresp=0.1, mingain=0.1, maxgain=2.0, exclude_low_pred=False)
d = d.loc[d['area'].isin(['A1', 'PEG'])]

d_big_gain_diff = d.loc[(cell_epoch_df['rwfg'] < 0.35) & (cell_epoch_df['rwbg'] > 0.65)]
dd = d_big_gain_diff[['BG + FG', 'cellid']]

for ww in range(len(dd)):
    estim, cellid = dd.iloc[ww]
    olp_pred_example_greg(d, rec1, rec2, cellid, estim)



# options
siteid = 'PRN022a'
cellid, estim = 'PRN022a-250-1', 'STIM_27Chainsaw-0-1_37WomanA-0-1'




import joblib as jl
path = '/auto/users/hamersky/OLP_models/today_model'
ddd = jl.load(path)
dff = ddd.loc[ddd['area'].isin(['A1'])]

avgs = plot_olp_preds_summary(dff, minresp=0.01, mingain=0.03, maxgain=2.0, sitewise=False)

def plot_olp_preds_summary(cell_epoch_df, minresp=0.01, mingain=0.03, maxgain=2.0, sitewise=False):
    area = ",".join(list(cell_epoch_df['area'].unique()))
    original_N = cell_epoch_df.shape[0]
    cell_epoch_df = get_valid_olp_rows(cell_epoch_df, minresp=minresp, mingain=mingain, maxgain=maxgain,
                                       exclude_low_pred=False)

    rCC = cell_epoch_df['Binaural Type'] == 'BG Contra, FG Contra'
    rIC = cell_epoch_df['Binaural Type'] == 'BG Ipsi, FG Contra'
    labels = ['BG Contra, FG Contra', 'BG Ipsi, FG Contra']
    print('valid n', cell_epoch_df.shape[0], 'out of', original_N, 'frac: ',
          np.round(cell_epoch_df.shape[0] / original_N, 3))

    if rIC.sum() == 0:
        rrset = [slice(None)]
        figlabels = ['mono']
    else:
        rrset = [rCC, rIC]
        figlabels = ['CC', 'CI']

    model_dict = {'p1': 'CNN', 'p2': 'LN'}
    sites = cell_epoch_df.siteid.unique().tolist()
    site_r_dict, site_br_dict, site_fr_dict = {}, {}, {}
    for sid in sites:
        site_df = cell_epoch_df.loc[cell_epoch_df.siteid==sid]
        rr = rrset[0]

        r_df, br_df, fr_df = {}, {}, {}
        for pre in prefs[1:]:
            fr = site_df.loc[rr, 'rwfg']
            fp = site_df.loc[rr, pre + 'wfg'] * site_df.loc[rr, 'sf']
            br = site_df.loc[rr, 'rwbg']
            bp = site_df.loc[rr, pre + 'wbg'] * site_df.loc[rr, 'sf']
            dr = fr - br
            dp = fp - bp
            r_df[f'{model_dict[pre]}'] = np.corrcoef(dr, dp)[0, 1]
            br_df[f'{model_dict[pre]}'] = np.corrcoef(br, bp)[0, 1]
            fr_df[f'{model_dict[pre]}'] = np.corrcoef(fr, fp)[0, 1]

        site_r_dict[sid] = (r_df['CNN'], r_df['LN'])
        site_br_dict[sid] = (br_df['CNN'], br_df['LN'])
        site_fr_dict[sid] = (fr_df['CNN'], fr_df['LN'])

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].plot([0, 1], [0, 1], 'k--')

    CNNs, LNs = [dd[0] for dd in site_r_dict.values()], [dd[1] for dd in site_r_dict.values()]
    bCNNs, bLNs = [dd[0] for dd in site_br_dict.values()], [dd[1] for dd in site_br_dict.values()]
    fCNNs, fLNs = [dd[0] for dd in site_fr_dict.values()], [dd[1] for dd in site_fr_dict.values()]

    if sitewise:
        LN_av, CNN_av = np.mean(LNs), np.mean(CNNs)
        bLN_av, bCNN_av = np.mean(bLNs), np.mean(bCNNs)
        fLN_av, fCNN_av = np.mean(fLNs), np.mean(fCNNs)

    else:
        rr = rrset[0]
        r_dict, br_dict, fr_dict = {}, {}, {}
        for pre in prefs[1:]:
            fr = cell_epoch_df.loc[rr, 'rwfg']
            fp = cell_epoch_df.loc[rr, pre + 'wfg'] * cell_epoch_df.loc[rr, 'sf']
            br = cell_epoch_df.loc[rr, 'rwbg']
            bp = cell_epoch_df.loc[rr, pre + 'wbg'] * cell_epoch_df.loc[rr, 'sf']
            dr = fr - br
            dp = fp - bp
            r_dict[f'{model_dict[pre]}'] = np.corrcoef(dr, dp)[0, 1]
            br_dict[f'{model_dict[pre]}'] = np.corrcoef(br, bp)[0, 1]
            fr_dict[f'{model_dict[pre]}'] = np.corrcoef(fr, fp)[0, 1]

        bLN_av, bCNN_av = br_dict['LN'], br_dict['CNN']
        fLN_av, fCNN_av = fr_dict['LN'], fr_dict['CNN']
        LN_av, CNN_av = r_dict['LN'], r_dict['CNN']

    returns = {'RG': {'LN': LN_av, 'CNN': CNN_av}, 'wBG': {'LN': bLN_av, 'CNN': bCNN_av},
               'wFG': {'LN': fLN_av, 'CNN': fCNN_av}}

    ax[0].scatter(LNs, CNNs, s=20, color='dimgrey', label='Recording site')
    ax[0].legend(fontsize=10)
    ax[0].set_xlabel('LN Performance (r)', fontweight='bold', fontsize=12)
    ax[0].set_ylabel('CNN Performance (r)', fontweight='bold', fontsize=12)
    # ax[0].set_title(f"r={np.corrcoef(LNs, CNNs)[0, 1]:.3f}", fontweight='bold', fontsize=12)



    pp = 1
    ax[1].bar([1-0.2, pp+0.2], [bLN_av, bCNN_av], color=['goldenrod', 'darkgoldenrod'], width=0.4)
    pp+=1
    ax[1].bar([pp-0.2, pp+0.2], [fLN_av, fCNN_av], color=['goldenrod', 'darkgoldenrod'], width=0.4)
    pp+=1.5
    ax[1].bar([pp-0.2], [LN_av], color=['goldenrod'], width=0.4, label='LN')
    ax[1].bar([pp+0.2], [CNN_av], color=['darkgoldenrod'], width=0.4, label='CNN')

    ax[1].legend(fontsize=12)
    ax[1].set_ylabel('Model Performance (r)', fontweight='bold', fontsize=12)
    ax[1].set_xticks([1, 2, 3.5])
    ax[1].set_xticklabels(['BG\nweight', 'FG\nweight', 'Relative\nGain'], fontsize=10, fontweight='bold')

    plt.tight_layout()

    return returns


def olp_pred_example_greg(cell_epoch_df, rec, rec2, cellid, estim, linsum=True):
    prefs = ['r', 'p2', 'p1']
    labels = ['CNN pred', 'LN pred']
    resp=rec['resp']
    stim=rec['stim']
    sigs = [rec['resp'], rec2['pred'], rec['pred']]

    ii=cell_epoch_df.loc[(cell_epoch_df['cellid']==cellid) &
                      (cell_epoch_df['BG + FG']==estim)].index.values

    efg = cell_epoch_df.loc[ii, 'FG'].values[0]
    ebg = cell_epoch_df.loc[ii, 'BG'].values[0]
    efgbg = cell_epoch_df.loc[ii, 'BG + FG'].values[0]

    kinds = ['BG:', 'FG:', 'Combo']
    names = [estim.split('_')[1].split('-')[0][2:], estim.split('_')[2].split('-')[0][2:], '']

    # weights,residual_sum,rank,singular_values = np.linalg.lstsq(np.stack([rfg, rbg, np.ones_like(rfg)], axis=1), rfgbg,rcond=None)
    # print(np.round(weights,2))
    rfg = resp.extract_channels([cellid]).extract_epoch(efg).mean(axis=0)[0, :]
    rbg = resp.extract_channels([cellid]).extract_epoch(ebg).mean(axis=0)[0, :]
    rfgbg = resp.extract_channels([cellid]).extract_epoch(efgbg).mean(axis=0)[0, :]

    fs=rec['resp'].fs
    sp = int(fs * 0.5)
    spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3

    rfg0 = rfg - spont
    rbg0 = rbg - spont
    rfgbg0 = rfgbg - spont
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), rfgbg0, rcond=None)
    #print(np.round(weights2, 2))
    weights2 = fb_weights(rfg, rbg, rfgbg, sp)
    #print(np.round(weights2, 2))

    imopts = {'origin': 'lower', 'interpolation': 'none', 'cmap': 'gray_r', 'aspect': 'auto'}

    label_dict = {'r': '', 'p2': 'LN', 'p1': 'CNN'}
    ylabel_dict = {'r': 'Actual\nResponse', 'p2': 'LN', 'p1': 'CNN'}

    f = plt.figure(figsize=(12, 6))
    psthBG = plt.subplot2grid((21, 18), (4, 0), rowspan=5, colspan=5)
    LNpsthBG = plt.subplot2grid((21, 18), (10, 0), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)
    CNNpsthBG = plt.subplot2grid((21, 18), (16, 0), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)

    psthFG = plt.subplot2grid((21, 18), (4, 6), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)
    LNpsthFG = plt.subplot2grid((21, 18), (10, 6), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)
    CNNpsthFG = plt.subplot2grid((21, 18), (16, 6), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)

    psthCM = plt.subplot2grid((21, 18), (4, 12), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)
    LNpsthCM = plt.subplot2grid((21, 18), (10, 12), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)
    CNNpsthCM = plt.subplot2grid((21, 18), (16, 12), rowspan=5, colspan=5, sharex=psthBG, sharey=psthBG)

    specBG = plt.subplot2grid((21, 18), (0, 0), rowspan=2, colspan=5)
    specFG = plt.subplot2grid((21, 18), (0, 6), rowspan=2, colspan=5)
    specCM = plt.subplot2grid((21, 18), (0, 12), rowspan=2, colspan=5)

    ax = [psthBG, psthFG, psthCM, LNpsthBG, LNpsthFG, LNpsthCM, CNNpsthBG, CNNpsthFG, CNNpsthCM,
          specBG, specFG, specCM]

    time = (np.arange(0, rfg0.shape[-1]) / 100) - 0.5

    aa = 0
    for kk, pre, sig in zip(range(len(sigs)), prefs, sigs):

        rfg = sig.extract_channels([cellid]).extract_epoch(efg).mean(axis=0)[0, :]
        rbg = sig.extract_channels([cellid]).extract_epoch(ebg).mean(axis=0)[0, :]
        rfgbg = sig.extract_channels([cellid]).extract_epoch(efgbg).mean(axis=0)[0, :]

        spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3
        rfg0 = rfg - spont
        rbg0 = rbg - spont
        rfgbg0 = rfgbg - spont
        weights2 = fb_weights(rfg, rbg, rfgbg, sp)

        ax[aa].plot(time, rbg0, color='deepskyblue')
        ax[aa+1].plot(time, rfg0, color='yellowgreen')
        if linsum==True:
            if aa == 0:
                ymin, ymax = ax[aa].get_ylim()
                ymin1, ymax1 = ax[aa + 1].get_ylim()
                maxi = np.max([ymax, ymax1])
            ax[aa+2].plot(time, (rbg0+rfg0), color='dimgray', ls='--', label='Linear Sum', lw=0.5)

        ax[aa+2].plot(time, rfgbg0, color='dimgray', label='Actual Response')
        ax[aa+2].plot(time, (rfg0) * weights2[0] + (rbg0) * weights2[1], color='orange', label='Weighted Pred')

        ax[aa+2].axhline(0, color='r', lw=0.5)
        # ax[aa+2].set_title(f'{pre}: wbg={weights2[1]:.2f}, wfg={weights2[0]:.2f}', fontsize=10, fontweight='bold')
        if aa == 0:
            ax[0].set_title(f'r: wbg={weights2[1]:.2f}, wfg={weights2[0]:.2f}', fontsize=10, fontweight='bold')
        if aa == 3:
            ax[1].set_title(f'LN: wbg={weights2[1]:.2f}, wfg={weights2[0]:.2f}', fontsize=10, fontweight='bold')
        if aa == 6:
            ax[2].set_title(f'CNN: wbg={weights2[1]:.2f}, wfg={weights2[0]:.2f}', fontsize=10, fontweight='bold')

        if kk==(len(sigs) - 1):
            ax[aa+1].set_xlabel('Time (s)', fontweight='bold', fontsize=12)

        ax[aa].set_ylabel(f'{ylabel_dict[pre]}', fontweight='bold', fontsize=12)

        aa+=3

    ax[2].legend(frameon=False, fontsize=6)
    ymin, ymax = ax[0].get_ylim()
    for ee in range(len(sigs)*3):
        ax[ee].vlines([0, 1], ymin, ymax, colors='black', linestyles=':', lw=0.75)
    ax[0].set_ylim(ymin, ymax)
    ax[0].set_xticks([0, 0.5, 1.0])
    ax[0].set_xlim(-0.2, 1.3)

    ymi, yma = ax[0].get_ylim()
    ax[0].set_ylim(ymi, maxi)

    sfg = stim.extract_epoch(efg).mean(axis=0)
    sbg = stim.extract_epoch(ebg).mean(axis=0)
    sfgbg = stim.extract_epoch(efgbg).mean(axis=0)

    specs = [sbg, sfg, sfgbg]

    cc = 0
    for ee, spec in zip(range(-3,0,1), specs):

        ax[ee].imshow(spec, aspect='auto', origin='lower', extent=[0, sbg.shape[1], 0, sbg.shape[0]],
                     cmap='gray_r')
        ax[ee].set_xticks([]), ax[ee].set_yticks([])
        ax[ee].set_xticklabels([]), ax[ee].set_yticklabels([])
        ax[ee].spines['top'].set_visible(False), ax[ee].spines['bottom'].set_visible(False)
        ax[ee].spines['left'].set_visible(False), ax[ee].spines['right'].set_visible(False)
        ax[ee].set_xlim(30, 180)
        ax[ee].set_title(f"{kinds[cc]} {names[cc]}", rotation=0, fontweight='bold', fontsize=12)
        cc+=1

        ymin, ymax = ax[ee].get_ylim()
        ax[ee].vlines([50, 150], ymin+0.1, ymax, color='black', lw=0.5)
        ax[ee].hlines([ymin+0.1,ymax], 50, 150, color='black', lw=0.5)

    f.suptitle(f'{cellid} -- {estim}')
    f.tight_layout()