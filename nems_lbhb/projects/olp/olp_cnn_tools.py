"""
author: SVD
date: 10/12/23

Analysis of CNN predictions of OLP relative gain
Currently for preliminary data for R01 BCP proposal

pred_comp() - prediction comparison scatter plot


"""

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
from scipy.stats import wilcoxon, pearsonr


import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join
from nems0 import db
from nems0 import get_setting
from nems0.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems0.utils import smooth

from nems_lbhb import baphy_io
from nems_lbhb.analysis import dstrf
from nems_lbhb.plots import histscatter2d, histmean2d, scatter_comp

from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems_lbhb.projects.olp.OLP_get_epochs import get_rec_epochs, get_stim_type, generate_cc_dataframe
from nems_lbhb.projects.olp import OLP_get_epochs
from nems_lbhb.gcmodel.figures.snr import compute_snr, compute_snr_multi


log = logging.getLogger(__name__)


def add_noise(psth, reps=20, prat=1):
    p = psth.copy()
    p[p < 0] = 0
    r = np.random.poisson(prat * p[:, np.newaxis], (len(p), reps)) + \
        np.random.normal((1 - prat) * p[:, np.newaxis], (1 - prat) * p[:, np.newaxis] / 2, (len(p), reps))
    r[r < 0] = 0

    return r.mean(axis=1)

def fb_weights(rfg,rbg,rfgbg, spontbins=50, smoothwin=None):

    spont = np.concatenate([rfg[:spontbins],rbg[:spontbins],rfgbg[:spontbins]]).mean()
    rfg0 = rfg[spontbins:]-spont
    rbg0 = rbg[spontbins:]-spont
    rfgbg0 = rfgbg[spontbins:]-spont
    if smoothwin is not None:
        rfg0 = smooth(rfg0, window_len=smoothwin, axis=0)
        rbg0 = smooth(rbg0, window_len=smoothwin, axis=0)
        rfgbg0 = smooth(rfgbg0, window_len=smoothwin, axis=0)

    y = rfgbg0 - rfg0 - rbg0
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), y, rcond=None)

    w = weights2 + 1
    pred = rfg0*w[0] + rbg0*w[1]
    if (pred.std()>0) and (y.std()>0):
        r = np.corrcoef(pred,y)[0,1]
    else:
        r = 0
    return w, r

def stim_snr(resp, frac_total=True):
    resp = resp.squeeze()
    if len(resp.shape)!=2:
        return np.nan

    products = np.dot(resp, resp.T)
    per_rep_snrs = []
    for i, _ in enumerate(resp):
        total_power = products[i, i]
        signal_powers = np.delete(products[i], i)
        if total_power == 0:
            rep_snr=0
        elif frac_total:
            rep_snr = np.nanmean(signal_powers) / total_power
        else:
            rep_snr = np.nanmean(signal_powers / (total_power - signal_powers))

        per_rep_snrs.append(rep_snr)

    return np.nanmean(per_rep_snrs)


def pred_comp(batch=341, modelnames=None):
    if modelnames is None:
        if batch == 341:
            modelnames = [
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-lvl.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
            ]
        else:
            modelnames = [
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
                "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            ]
    print(modelnames)
    df = db.batch_comp(batch=batch, modelnames=modelnames, shortnames=["CNN", "LN"])
    dfceil = db.batch_comp(batch=batch, modelnames=modelnames, shortnames=["CNN", "LN"], stat='r_ceiling')
    dffloor = db.batch_comp(batch=batch, modelnames=modelnames, shortnames=["CNN", "LN"], stat='r_floor')
    df = df.merge(dffloor, how='inner', left_index=True, right_index=True, suffixes=('', '_floor'))
    df = df.merge(dfceil, how='inner', left_index=True, right_index=True, suffixes=('', '_ceil'))
    df['goodidx'] = ((df['LN'] > df['LN_floor']) | (df['CNN'] > df['CNN_floor'])) & ~np.isnan(df['LN']) & ~np.isnan(
        df['CNN'])

    df_area = db.pd_query(f"select distinct sCellFile.cellid,sCellFile.area from sCellFile INNER JOIN sRunData on sCellFile.cellid=sRunData.cellid and sRunData.batch={batch}")
    df = df.merge(df_area, how='inner', left_index=True, right_on='cellid', suffixes=('', '_area'))
    df = df.loc[df.area.isin(['A1', 'PEG'])]
    print(f"Good count: {df['goodidx'].sum()}/{len(df)} 'good' units")
    d_ = df.loc[df['goodidx']]
    [w, p] = wilcoxon(d_['LN'], d_['CNN'])

    f = plt.figure(figsize=(3.5, 1.5))
    ax = f.add_subplot(1,2,1)

    df.loc[~df['goodidx']].plot.scatter(x='LN_ceil', y='CNN_ceil', s=2, ax=ax, color='gray')
    df.loc[df['goodidx']].plot.scatter(x='LN_ceil', y='CNN_ceil', s=2, ax=ax, color='black')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel(f"LN ({d_['LN_ceil'].mean():.3f})")
    ax.set_ylabel(f"CNN ({d_['CNN_ceil'].mean():.3f})")

    ax=f.add_subplot(1,4,3)
    sns.barplot(d_[['LN_ceil','CNN_ceil']], ax=ax)
    ax.set_xticklabels(['LN','CNN'])
    ax.set_ylabel('Mean pred. corr.')
    f.suptitle(f"{batch} n={len(d_)}/{len(df)} p={p:.3e}", fontsize=8)
    plt.tight_layout()
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
    'CNN', 'LN' : prediction correlation for each model (r_test)
    'CNN_floor', 'LN_floor' : r_floor on prediction (shuffled in time). "good" models have prediction correlation>noise floor
    
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
        df['cellid'] = df.index
        df['siteid'] = df['cellid'].apply(db.get_siteid)
        df['area'] = 'unknown'
        df['iso'] = 0
    dfsite = df.loc[df['siteid']==siteid]
    print(dfsite.shape, dfsite['area'].unique(), dfsite.columns)
    
    xf,ctx=load_model_xform(cellid, batch, modelnames[0], verbose=verbose)
    xf2,ctx2=load_model_xform(cellid, batch, modelnames[1], verbose=verbose)

    if verbose:
        modelspec=ctx['modelspec']
        modelspec2=ctx2['modelspec']

        f,ax=plt.subplots(1,2,figsize=(12,4))
        score='r_test'
        ax[0].plot(modelspec.meta[score], label=f"m1: {modelspec.meta[score].mean():.3f}")
        ax[0].plot(modelspec2.meta[score], label=f"m2: {modelspec2.meta[score].mean():.3f}")
        ax[0].set_title(score);
        ax[0].set_xlabel('unit')
        ax[0].set_ylabel('pred xc')
        ax[0].legend(fontsize=8);
        score='r_fit'
        ax[1].plot(modelspec.meta[score], label=f"m1: {modelspec.meta[score].mean():.3f}")
        ax[1].plot(modelspec2.meta[score], label=f"m2: {modelspec2.meta[score].mean():.3f}")
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
    cell_epoch_df['swfg'] = np.nan
    cell_epoch_df['swbg'] = np.nan
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
        weights0, predxc = fb_weights(rfgbg1, np.ones_like(rfgbg1) * spont, rfgbg2, sp)

        # special 1/2 trials computation of FG/BG weights
        rfg = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_fg)[::2, :, :].mean(axis=0)[0, :] / r[
            'cellstd']
        rbg = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_bg)[::2, :, :].mean(axis=0)[0, :] / r[
            'cellstd']
        rfgbg = ctx['rec']['resp'].extract_channels([cellid]).extract_epoch(epoch_fgbg)[::2, :, :].mean(axis=0)[0, :] / r[
            'cellstd']

        weights2, predxc = fb_weights(rfg, rbg, rfgbg, spontbins=sp)
        cell_epoch_df.at[id, 'rwfg0'] = weights2[0]
        cell_epoch_df.at[id, 'rwbg0'] = weights2[1]
        cell_epoch_df.at[id, 'rw0'] = weights0[0]

        sfg = ctx2['rec']['stim'].extract_epoch(epoch_fg)[0]
        sbg = ctx2['rec']['stim'].extract_epoch(epoch_bg)[0]
        sfgbg = ctx2['rec']['stim'].extract_epoch(epoch_fgbg)[0]
        sweights, spredxc = fb_weights(sfg.flatten(), sbg.flatten(), sfgbg.flatten(), sp)

        for ii, pre, sig in zip(range(len(prefs)), prefs, sigs):
            rfg = sig.extract_channels([cellid]).extract_epoch(epoch_fg).mean(axis=0)[0, :] / r['cellstd']
            rbg = sig.extract_channels([cellid]).extract_epoch(epoch_bg).mean(axis=0)[0, :] / r['cellstd']
            rfgbg = sig.extract_channels([cellid]).extract_epoch(epoch_fgbg).mean(axis=0)[0, :] / r['cellstd']

            if ii > 20:
                rfg = add_noise(rfg, reps=15)
                rbg = add_noise(rbg, reps=15)
                rfgbg = add_noise(rfgbg, reps=15)

            spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3
            weights2, predxc = fb_weights(rfg, rbg, rfgbg, sp)

            cell_epoch_df.at[id, pre + 'wfg'] = weights2[0]
            cell_epoch_df.at[id, pre + 'wbg'] = weights2[1]
            cell_epoch_df.at[id, pre + 'fg'] = rfg[sp:-sp].mean() - spont
            cell_epoch_df.at[id, pre + 'bg'] = rbg[sp:-sp].mean() - spont
            cell_epoch_df.at[id, 'swfg'] = sweights[0]
            cell_epoch_df.at[id, 'swbg'] = sweights[1]

    # link to area labels
    cell_epoch_df = cell_epoch_df.merge(dfsite[['area','iso','CNN','LN','CNN_floor','LN_floor']], how='left', left_on='cellid',
                                        right_index=True)

    # figure out an adjustment to account for the effec tof noisy responses on the estimate for fg/bg weights
    # exlude outliers to make fit stable.
    d = get_valid_olp_rows(cell_epoch_df, minresp=0.1, mingain=0.1, maxgain=2.0, exclude_low_pred=False)

    ww, residual_sum, rank, singular_values = np.linalg.lstsq(
        np.stack([d['rwbg0'],np.zeros_like(d['rwbg'])], axis=1),
                 d['rwbg'], rcond=None)
    log.info(f"ratio wbg10 / wbg5: {ww[0]:.3f} const={ww[1]:.3f} rw0 mean: {cell_epoch_df['rw0'].mean():.3f}")

    sf = cell_epoch_df['rw0']*ww[0]
    sf[sf<0.1]=0.1
    cell_epoch_df['sf']=sf

    if verbose:
        return cell_epoch_df, ctx, ctx2
    else:
        return cell_epoch_df, ctx, ctx2


def plot_olp_preds(cell_epoch_df, minresp=0.01, mingain=0.03, maxgain=2.0,
                   exclude_low_pred=True, fig_label=None, N=500, split_space=False):
    """
    returns:
       f - figure handle
       ccs - 3 x 3 correlation between stim [wfg, wbg, rdiff] X [specgram, ln, cnn]
    """
    # original
    #minresp, mingain, maxgain = 0.05, 0, 2.0
    if fig_label is None:
        fig_label = db.get_siteid(cell_epoch_df['cellid'].values[0])
    area = ",".join(list(cell_epoch_df['area'].unique()))
    original_N = cell_epoch_df.shape[0]

    cell_epoch_df = get_valid_olp_rows(cell_epoch_df, minresp=minresp, mingain=mingain, maxgain=maxgain, exclude_low_pred=exclude_low_pred)

    rCC = cell_epoch_df['Binaural Type']=='BG Contra, FG Contra'
    rIC = cell_epoch_df['Binaural Type']=='BG Ipsi, FG Contra'
    #labels=['BG Contra, FG Contra','BG Ipsi, FG Contra']
    print('valid n', cell_epoch_df.shape[0], 'out of', original_N, 'frac: ', np.round(cell_epoch_df.shape[0]/original_N,3))
    
    if split_space and (rIC.sum()>0):
        rrset = [rCC, rIC]
        figlabels = ['CC', 'CI']
    else:
        rrset = [slice(None)]
        figlabels = ['mono']

    prefs = ['r', 's', 'p2', 'p1']
    labels = ['Specgram', 'LN pred', 'CNN pred']
    ccs=np.zeros((3,len(labels)))
    ps=np.zeros((3,len(labels)))

    if cell_epoch_df.shape[0]==0:
        log.info('no valid results')
        f = None
        return f,ccs
    
    for rr, figlabel in zip(rrset, figlabels):
        f, axs = plt.subplots(len(labels)+1, 3, figsize=(6, 2*(len(labels)+1)))
        for row, (pre, ax, label) in enumerate(zip(prefs[1:], axs, labels)):
            ax[0].plot([0, 1], [0, 1], 'k--')
            ax[1].plot([0, 1], [0, 1], 'k--')
            ax[2].plot([-1.2, 1], [-1.2, 1], 'k--')

            SCALE_DOWN=True
            if SCALE_DOWN:
                fr = cell_epoch_df.loc[rr, 'rwfg']
                br = cell_epoch_df.loc[rr, 'rwbg']
                fp = cell_epoch_df.loc[rr, pre + 'wfg'] * cell_epoch_df.loc[rr, 'sf']
                bp = cell_epoch_df.loc[rr, pre + 'wbg'] * cell_epoch_df.loc[rr, 'sf']
            else:
                if type(rr) is slice:
                    rr = (cell_epoch_df['sf']>0.2)
                else:
                    rr = rr & (cell_epoch_df['sf']>0.2)
                fr = cell_epoch_df.loc[rr, 'rwfg'] / cell_epoch_df.loc[rr, 'sf']
                br = cell_epoch_df.loc[rr, 'rwbg'] / cell_epoch_df.loc[rr, 'sf']
                fp = cell_epoch_df.loc[rr, pre + 'wfg']
                bp = cell_epoch_df.loc[rr, pre + 'wbg']

            dr = fr - br
            dp = fp - bp

            ccs[0,row],ps[0,row] = pearsonr(fr, fp)
            ccs[1,row],ps[1,row] = pearsonr(br, bp)
            ccs[2,row],ps[2,row] = pearsonr(dr, dp)

            if len(fr)>N:
                idx = np.linspace(0, len(fr)-1, N, dtype=int)
            else:
                idx = np.arange(len(fr)).astype(int)
            ax[0].scatter(fr.iloc[idx], fp.iloc[idx], s=2)
            ax[0].set_title(f"r={ccs[0,row]:.3f} p={ps[0,row]:.2e} mse={np.std(fr-fp)/np.std(fr):.2f}")
            ax[0].set_xlim([-0.1, 1.5])
            ax[0].set_ylim([-0.1, 1.5])
            ax[0].set_ylabel(label + ' w_fg')
            ax[0].set_xlabel(f'Actual w_fg ({fr.mean():.3f})')

            ax[1].scatter(br.iloc[idx], bp.iloc[idx], s=2)
            ax[1].set_xlim([-0.1, 1.5])
            ax[1].set_ylim([-0.1, 1.5])
            ax[1].set_title(f"r={ccs[1,row]:.3f} p={ps[1,row]:.2e} mse={np.std(br-bp)/np.std(br):.2f}")
            ax[1].set_ylabel(label + ' w_bg')
            ax[1].set_xlabel(f'Actual w_bg ({br.mean():.3f})')

            ax[2].scatter(dr.iloc[idx], dp.iloc[idx], s=2, color='black')
            ax[2].set_xlim([-1.2, 1])
            ax[2].set_ylim([-1.2, 1])
            ax[2].set_title(f"r={ccs[2,row]:.3f} p={ps[2,row]:.2e} mse={np.std(dr-dp):.2f}")
            ax[2].set_ylabel(f'{label} relative gain ({dp.mean():.3f})')
            ax[2].set_xlabel(f'Actual relative gain ({dr.mean():.3f})')

        for col,lab in enumerate(['FG','BG','FG+BG']):
            axs[-1, col].bar(labels, ccs[col, :])
            axs[-1, col].set_title(lab)

        fr = cell_epoch_df.loc[rr, 'rwfg']
        br = cell_epoch_df.loc[rr, 'rwbg']
        fp1 = cell_epoch_df.loc[rr, 'p1wfg'] * cell_epoch_df.loc[rr, 'sf']
        bp1 = cell_epoch_df.loc[rr, 'p1wbg'] * cell_epoch_df.loc[rr, 'sf']
        fp2 = cell_epoch_df.loc[rr, 'p2wfg'] * cell_epoch_df.loc[rr, 'sf']
        bp2 = cell_epoch_df.loc[rr, 'p2wbg'] * cell_epoch_df.loc[rr, 'sf']
        dr=fr-br
        dp1=fp1-bp1
        dp2=fp2-bp2
        njacks=20
        chunksize = int(np.ceil(len(dr) / njacks / 10))
        chunkcount = int(np.ceil(len(dr) / chunksize / njacks))
        idx = np.zeros((chunkcount, njacks, chunksize))
        for jj in range(njacks):
            idx[:, jj, :] = jj
        idx = np.reshape(idx, [-1])[:len(dr)]
        jc = np.zeros((njacks,2))
        for jj in range(njacks):
            ff = (idx != jj)
            jc[jj,0] = np.corrcoef(dp1[ff], dr[ff])[0, 1]
            jc[jj,1] = np.corrcoef(dp2[ff], dr[ff])[0, 1]

        cc = np.nanmean(jc,axis=0)
        ee = np.nanstd(jc,axis=0) * np.sqrt(njacks - 1)
        print(cc,ee)
        print(cc[0]-cc[1], ee[0]+ee[1])
        f.suptitle(f"{fig_label} {figlabel} ({area}) (CNN: {cell_epoch_df['CNN'].mean():.3f}, LN: {cell_epoch_df['LN'].mean():.3f})")
        plt.tight_layout()
    
    return f,ccs


def olp_pred_example(cell_epoch_df, rec, rec2, cellid, estim):

    prefs = ['r', 'p2', 'p1']
    labels = ['CNN pred', 'LN pred']
    resp = rec['resp']
    stim = rec['stim']
    sigs = [rec['resp'], rec2['pred'], rec['pred']]

    ii = cell_epoch_df.loc[(cell_epoch_df['cellid']==cellid) &
                           (cell_epoch_df['BG + FG']==estim)].index.values

    efg = cell_epoch_df.loc[ii, 'FG'].values[0]
    ebg = cell_epoch_df.loc[ii, 'BG'].values[0]
    efgbg = cell_epoch_df.loc[ii, 'BG + FG'].values[0]


    # weights,residual_sum,rank,singular_values = np.linalg.lstsq(np.stack([rfg, rbg, np.ones_like(rfg)], axis=1), rfgbg,rcond=None)
    # print(np.round(weights,2))
    rfg = resp.extract_channels([cellid]).extract_epoch(efg).mean(axis=0)[0, :]
    rbg = resp.extract_channels([cellid]).extract_epoch(ebg).mean(axis=0)[0, :]
    rfgbg = resp.extract_channels([cellid]).extract_epoch(efgbg).mean(axis=0)[0, :]

    fs = rec['resp'].fs
    sp = int(fs * 0.5)
    spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3

    rfg0 = rfg - spont
    rbg0 = rbg - spont
    rfgbg0 = rfgbg - spont
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), rfgbg0, rcond=None)
    #print(np.round(weights2, 2))
    weights2, predxc = fb_weights(rfg, rbg, rfgbg, sp)
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
        weights2, predxc = fb_weights(rfg, rbg, rfgbg, sp)

        ax[kk + 1, 0].plot(rbg0)
        ax[kk + 1, 1].plot(rfg0)
        ax[kk + 1, 2].plot(rfgbg0, label='resp')
        ax[kk + 1, 2].plot((rfg0) * weights2[0] + (rbg0) * weights2[1],label='wsum')
        ax[kk + 1, 2].axhline(0, color='r', lw=0.5)
        ax[kk + 1, 2].set_title(f'{pre}wfg={weights2[0]:.2f} {pre}wbg={weights2[1]:.2f}', fontsize=10)

    ax[1, 2].legend(frameon=False)
    plt.tight_layout()
    return f

def load_all_sites(batch=341, modelnames=None, area="A1", show_plots=False):
    """
    Load all models in batch and compute rel gain predictions
    :param batch:
    :param modelnames:
    :param area:
    :return:
    """
    siteids, cellids = db.get_batch_sites(batch=batch, area=area)

    if (modelnames is None) and (batch == 341):
        modelnames = [
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-lvl.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
        ]
    elif (modelnames is None):
        modelnames = [
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
        ]

    dfs = []
    for cid, siteid in enumerate(siteids):
        try:
            cell_epoch_df, rec1, rec2 = compare_olp_preds(
                siteid, batch=batch, modelnames=modelnames, verbose=False)
            if show_plots:
                plot_olp_preds(cell_epoch_df, minresp=0.01, mingain=0.1, maxgain=1.1, exclude_low_pred=True)
            dfs.append(cell_epoch_df)
        except:
            print('bad site', siteid)
    df_all = pd.concat(dfs, ignore_index=True)
    if show_plots:
        plot_olp_preds(df_all, minresp=0.01, mingain=0.1, maxgain=1.1, exclude_low_pred=True)
    return df_all

def demo_site(batch=341, modelnames=None, cid=None, load_only=False):
    siteids, cellids = db.get_batch_sites(batch=batch)

    if (modelnames is None) and (batch == 341):
        modelnames = [
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-lvl.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
        ]
    elif (modelnames is None):
        modelnames = [
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
        ]
    if cid is None:
        # cid = 35
        # cid = 32
        cid = 11

    siteid, cellid = siteids[cid], cellids[cid]

    cell_epoch_df, rec1, rec2 = compare_olp_preds(siteid, batch=batch, modelnames=modelnames,
                                                  verbose=False)
    if load_only:
        return cell_epoch_df

    f = plot_olp_preds(cell_epoch_df, minresp=0.01, mingain=0.1, maxgain=1.1, exclude_low_pred=True)

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

    return cell_epoch_df

if __name__ == '__main__':
    batch = 341
    area = "A1"
    if batch == 341:
        modelnames = [
            #"gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18-ld-norm-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            #"gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-lvl.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
            "gtgram.fs100.ch18-ld-norm-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-lvl.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
        ]
    else:
        modelnames = [
            "gtgram.fs100.ch18.bin6-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            # "gtgram.fs100.ch18.bin6-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
            "gtgram.fs100.ch18.bin6-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-lvl.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
        ]
    # cell_epoch_df = demo_site(batch=batch, modelnames=None, load_only=True, cid=5)
    # f = plot_olp_preds(cell_epoch_df, minresp=0.05, mingain=0.01, maxgain=1.1,
    #                    exclude_low_pred=True, fig_label=area)

    df_all = load_all_sites(batch=batch, modelnames=modelnames, area=area, show_plots=False)
    f,ccs = plot_olp_preds(df_all, minresp=0.05, mingain=0.01, maxgain=1.15,
                       exclude_low_pred=True, fig_label=area, split_space=False)
    #f.savefig(f'/home/svd/Documents/onedrive/projects/olp/batch{batch}_relgain_comp_nolog.pdf')

    f=pred_comp(batch, modelnames)
    #f.savefig(f'/home/svd/Documents/onedrive/projects/olp/batch{batch}_pred_comp.pdf')


#############################################
### dSTRF functions
def triplet_dpc_subspace(cell_epoch_df, dim1=0, dim2=1, show_pred=True,
                         modelspec=None, est=None, val=None, cellid=None,
                         triplet_idx=None, **context):

    fgbgepochs = cell_epoch_df['BG + FG'].unique().tolist()
    if triplet_idx is None:
        triplet_idx = np.arange(len(fgbgepochs))
    else:
        fgbgepochs = [fgbgepochs[i] for i in triplet_idx]

    cid = [i for i, c in enumerate(modelspec.meta['cellids']) if c == cellid][0]
    log.info(f"Cell {cellid} index is {cid}")

    if show_pred:
        r = est['pred'].as_continuous()[cid]
    else:
        r = est['resp'].as_continuous()[cid]

    rec2 = dstrf.project_to_subspace(modelspec, est=est, val=None, rec=None, out_channels=[cid])['est']
    resp = rec2['resp']
    pred = rec2['pred']
    ss = rec2['subspace']

    f, ax = plt.subplots(len(fgbgepochs), 5, figsize=(10, 2 * len(fgbgepochs)), sharex='col', sharey='col')
    for jj, estim in enumerate(fgbgepochs):
        ii = cell_epoch_df.loc[(cell_epoch_df['cellid'] == cellid) &
                               (cell_epoch_df['BG + FG'] == estim)].index.values

        efg = cell_epoch_df.loc[ii, 'FG'].values[0]
        ebg = cell_epoch_df.loc[ii, 'BG'].values[0]
        efgbg = cell_epoch_df.loc[ii, 'BG + FG'].values[0]

        # weights,residual_sum,rank,singular_values = np.linalg.lstsq(np.stack([rfg, rbg, np.ones_like(rfg)], axis=1), rfgbg,rcond=None)
        # print(np.round(weights,2))
        rfg = resp.extract_channels([cellid]).extract_epoch(efg).mean(axis=0)[0, :]
        rbg = resp.extract_channels([cellid]).extract_epoch(ebg).mean(axis=0)[0, :]
        rfgbg = resp.extract_channels([cellid]).extract_epoch(efgbg).mean(axis=0)[0, :]

        pfg = pred.extract_channels([cellid]).extract_epoch(efg).mean(axis=0)[0, :]
        pbg = pred.extract_channels([cellid]).extract_epoch(ebg).mean(axis=0)[0, :]
        pfgbg = pred.extract_channels([cellid]).extract_epoch(efgbg).mean(axis=0)[0, :]

        sfg = ss.extract_epoch(efg).mean(axis=0)[0]
        sbg = ss.extract_epoch(ebg).mean(axis=0)[0]
        sfgbg = ss.extract_epoch(efgbg).mean(axis=0)[0]

        fs = resp.fs
        sp = int(fs * 0.5)
        spont = (rfg[:sp].mean() + rbg[:sp].mean() + rfgbg[:sp].mean()) / 3
        weights2, predxc = fb_weights(rfg, rbg, rfgbg, sp)

        Y = ss.as_continuous()[0]

        sig = [sfg, sbg, sfgbg]
        step = int(fs / 10)

        rsigs = [rfg, rbg, rfgbg]
        psigs = [pfg, pbg, pfgbg]
        pcols = ['g', 'b', 'k']
        Z = None
        for i in range(3):
            ax[jj, 0].plot(psigs[i], lw=0.5, color=pcols[i])
            ax[jj, 1].plot(rsigs[i], lw=0.5, color=pcols[i])
            ac, bc, Z, N = histmean2d(Y[dim1, :], Y[dim2, :], r, bins=20, ax=ax[jj, i + 2], spont=spont, ex_pct=0.025,
                                      Z=Z)
            ax[jj, i + 2].plot(sig[i][dim1], sig[i][dim2], color='w', lw=0.5)
            ax[jj, i + 2].plot(sig[i][dim1, (sp + step + 3)::step], sig[i][dim2, (sp + step + 4)::step], '.', color='r')
            ax[jj, i + 2].plot(sig[i][dim1, (sp + 3)], sig[i][dim2, (sp + 4)], '.', color='k')
            ax[jj, i + 2].set_yticklabels([])
        ax[jj, 2].set_title(f'{estim} wfg={weights2[0]:.2f} wbg={weights2[1]:.2f}')

        rwfg = cell_epoch_df.loc[ii, 'rwfg'].values[0]
        rwbg = cell_epoch_df.loc[ii, 'rwbg'].values[0]
        p1wfg = cell_epoch_df.loc[ii, 'p1wfg'].values[0]
        p1wbg = cell_epoch_df.loc[ii, 'p1wbg'].values[0]

        ax[jj, 0].text(ax[0, 0].get_xlim()[0], ax[0, 0].get_ylim()[1], f'pred({p1wfg:0.2f},{p1wbg:0.2f})', va='top')
        ax[jj, 1].text(ax[0, 1].get_xlim()[0], ax[0, 1].get_ylim()[1], f'resp({rwfg:0.2f},{rwbg:0.2f})', va='top')
        ax[jj, 2].text(ax[0, 2].get_xlim()[0], ax[0, 2].get_ylim()[1], 'fg', va='top')
        ax[jj, 3].text(ax[0, 3].get_xlim()[0], ax[0, 3].get_ylim()[1], 'bg', va='top')
        ax[jj, 4].text(ax[0, 4].get_xlim()[0], ax[0, 4].get_ylim()[1], 'fgbg', va='top')
    ax[0, 0].set_title(cellid)
    return f

def triplet_stimuli(cell_epoch_df, rec=None, triplet_idx=None, **context):

    fgbgepochs = cell_epoch_df['BG + FG'].unique().tolist()
    if triplet_idx is None:
        triplet_idx = np.arange(len(fgbgepochs))
    else:
        fgbgepochs = [fgbgepochs[i] for i in triplet_idx]

    stim = rec['stim']

    f, ax = plt.subplots(len(fgbgepochs), 3, figsize=(8, 1 * len(fgbgepochs)),
                         sharex=True, sharey=True)
    for jj, estim in enumerate(fgbgepochs):
        ii = cell_epoch_df.loc[(cell_epoch_df['BG + FG'] == estim)].index.values[0]

        efg = cell_epoch_df.loc[ii, 'FG']
        ebg = cell_epoch_df.loc[ii, 'BG']
        efgbg = cell_epoch_df.loc[ii, 'BG + FG']

        sfg = stim.extract_epoch(efg).mean(axis=0)
        sbg = stim.extract_epoch(ebg).mean(axis=0)
        sfgbg = stim.extract_epoch(efgbg).mean(axis=0)

        imopts = {'origin': 'lower', 'interpolation': 'none', 'cmap': 'gray_r', 'aspect': 'auto'}
        ax[jj, 0].imshow(sfg, **imopts)
        ax[jj, 1].imshow(sbg, **imopts)
        ax[jj, 2].imshow(sfgbg, **imopts)
        ax[jj, 0].set_title(efg.replace("STIM_",""))
        ax[jj, 1].set_title(ebg.replace("STIM_",""))

    plt.tight_layout()
    return f