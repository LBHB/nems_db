import numpy as np
import os
import io
import logging
import time

import matplotlib.pyplot as plt
import sys, importlib
import pandas as pd
from sklearn.linear_model import LinearRegression

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

    # figure out an adjustment to account for the effec tof noisy responses on the estimate for fg/bg weights
    rFB = (cell_epoch_df['rfg']>0.1) & (cell_epoch_df['rbg']>0.1) & \
        (cell_epoch_df['rwfg']>0.1) & (cell_epoch_df['rwbg']>0.1) & \
        (cell_epoch_df['rwfg']<2) & (cell_epoch_df['rwbg']<2) & \
        (cell_epoch_df['rwfg0']>0.1) & (cell_epoch_df['rwbg0']>0.1) & \
        (cell_epoch_df['rwfg0']<2) & (cell_epoch_df['rwbg0']<2)
    ww, residual_sum, rank, singular_values = np.linalg.lstsq(
        np.stack([cell_epoch_df.loc[rFB,'rwbg0'],np.zeros_like(cell_epoch_df.loc[rFB,'rwbg'])], axis=1),
                 cell_epoch_df.loc[rFB,'rwbg'], rcond=None)
    print(f"ratio wbg10 / wbg5: {ww[0]:.3f} const={ww[1]:.3f} rw0 mean: {cell_epoch_df['rw0'].mean():.3f}")
    sf = cell_epoch_df['rw0']*ww[0]
    sf[sf<0.1]=0.1
    cell_epoch_df['sf']=sf

    # link to area labels
    cell_epoch_df = cell_epoch_df.merge(dfsite[['area','iso','CNN','LN','CNN_floor','LN_floor']], how='left', left_on='cellid',
                                        right_index=True)

    return cell_epoch_df, rec, rec2


def plot_olp_preds(cell_epoch_df, minresp=0.01, mingain=0.03, maxgain=2.0, exclude_low_pred=True):
    
    # original
    #minresp, mingain, maxgain = 0.05, 0, 2.0
    
    siteid = db.get_siteid(cell_epoch_df['cellid'].values[0])
    area = ",".join(list(cell_epoch_df['area'].unique()))
    
    rFB = (cell_epoch_df['rfg']>minresp) & (cell_epoch_df['rbg']>minresp) & \
        (cell_epoch_df['rwfg']>mingain) & (cell_epoch_df['rwbg']>mingain) & \
        (cell_epoch_df['rwfg']<maxgain) & (cell_epoch_df['rwbg']<maxgain) & \
        (cell_epoch_df['p1wfg']>mingain) & (cell_epoch_df['p1wbg']>mingain) & \
        (cell_epoch_df['p1wfg']<maxgain) & (cell_epoch_df['p1wbg']<maxgain) & \
        (cell_epoch_df['p2wfg']>mingain) & (cell_epoch_df['p2wbg']>mingain) & \
        (cell_epoch_df['p2wfg']<maxgain) & (cell_epoch_df['p2wbg']<maxgain)
    
    if exclude_low_pred:
        keepidx= (cell_epoch_df['LN']>cell_epoch_df['LN_floor']) & (cell_epoch_df['CNN']>cell_epoch_df['CNN_floor'])
        rFB=(rFB & keepidx)
            
    rCC = cell_epoch_df['Binaural Type']=='BG Contra, FG Contra'
    rIC = cell_epoch_df['Binaural Type']=='BG Ipsi, FG Contra'
    labels=['BG Contra, FG Contra','BG Ipsi, FG Contra']
    print('valid n', rFB.sum(), 'out of', len(rFB), 'frac: ', np.round(rFB.mean(),3))
    
    if rIC.sum()==0:
        rrset=[rFB]
        figlabels=['mono']
    else:
        rrset= [rFB & rCC, rFB & rIC]
        figlabels=['CC','CI']
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