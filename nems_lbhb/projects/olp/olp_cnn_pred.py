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
from nems_lbhb import baphy_experiment
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems_lbhb.projects.olp.OLP_get_epochs import get_rec_epochs, get_stim_type, generate_cc_dataframe
from nems_lbhb.projects.olp import OLP_get_epochs

log = logging.getLogger(__name__)

batch=341
siteids, cellids = db.get_batch_sites(batch=batch)

cid, cid0 = 33, 40  # PRN017b
cid, cid0 = 20, 0  # PRN017b
#cid = 13
#cid0 = 0
siteid, cellid = siteids[cid], cellids[cid]

#modelnames = [
#    "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_prefit.b322.f.nf-lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
#    "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
#]
modelnames = [
    "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
    "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
]

#xf,ctx=fit_model_xform(cellid, batch, modelnames[0])

#raise ValueError('stopping')

xf,ctx=load_model_xform(cellid, batch, modelnames[0])
xf2,ctx2=load_model_xform(cellid, batch, modelnames[1])



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

resp = ctx['val']['resp'].rasterize()
pred1 = ctx['val']['pred'].rasterize()
pred2 = ctx2['val']['pred'].rasterize()
stim = ctx['val']['stim'].rasterize()
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


def add_noise(psth, reps=20, prat=1):
    p = psth.copy()
    p[p < 0] = 0
    r = np.random.poisson(prat * p[:, np.newaxis], (len(p), reps)) + \
        np.random.normal((1 - prat) * p[:, np.newaxis], (1 - prat) * p[:, np.newaxis] / 2, (len(p), reps))
    r[r < 0] = 0

    return r.mean(axis=1)


sp = int(fs * 0.5)


def fb_weights(rfg, rbg, rfgbg, spontbins=50):
    spont = np.concatenate([rfg[:spontbins], rbg[:spontbins], rfgbg[:spontbins]]).mean()
    rfg0 = rfg - spont
    rbg0 = rbg - spont
    rfgbg0 = rfgbg - spont
    y = rfgbg0 - rfg0 - rbg0
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), y, rcond=None)
    # weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), rfgbg0, rcond=None)

    return weights2 + 1


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

    weights2 = fb_weights(rfg, rbg, rfgbg, sp)
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

minresp=0.05
mingain=0.1
maxgain=2.0
rFB = (cell_epoch_df['rfg']>minresp) & (cell_epoch_df['rbg']>minresp) & \
    (cell_epoch_df['rwfg']>mingain) & (cell_epoch_df['rwbg']>mingain) & \
    (cell_epoch_df['rwfg']<maxgain) & (cell_epoch_df['rwbg']<maxgain) & \
    (cell_epoch_df['p1wfg']>mingain) & (cell_epoch_df['p1wbg']>mingain) & \
    (cell_epoch_df['p1wfg']<maxgain) & (cell_epoch_df['p1wbg']<maxgain) & \
    (cell_epoch_df['p2wfg']>mingain) & (cell_epoch_df['p2wbg']>mingain) & \
    (cell_epoch_df['p2wfg']<maxgain) & (cell_epoch_df['p2wbg']<maxgain)
print('(gain floored) valid n', rFB.sum(), 'out of', len(rFB))

f, ax = plt.subplots(1,2,figsize=(6,3))
ax[0].plot([-.2, 1],[-.2, 1], 'k--')
ax[0].scatter(cell_epoch_df.loc[rFB,'rwbg0'],cell_epoch_df.loc[rFB,'rwbg'], s=2, color='black')
ax[0].scatter(cell_epoch_df.loc[rFB,'rwfg0'],cell_epoch_df.loc[rFB,'rwfg'], s=2, color='lightgray')
ax[0].set_title(f'10/5 ratio: {ww[0]:.3f}',fontsize=8)
ax[0].set_xlabel('weights with 5 reps')
ax[0].set_ylabel('weights with 10 reps')

ax[1].plot([-.2, 1],[-.2, 1], 'k--')
ax[1].scatter(cell_epoch_df.loc[rFB,'rwbg'], cell_epoch_df.loc[rFB,'sf'], s=2, color='black')
ax[1].scatter(cell_epoch_df.loc[rFB,'rwfg'], cell_epoch_df.loc[rFB,'sf'], s=2, color='lightgray')
ax[1].set_xlabel('measured weight')
ax[1].set_ylabel('adjusted upper bound')
plt.tight_layout()

print('estimates of rw0 adjustment:')
print(np.median(cell_epoch_df.loc[rFB,'rwbg']+cell_epoch_df.loc[rFB,'rwfg'])/np.median(cell_epoch_df.loc[rFB,'rwbg0']+cell_epoch_df.loc[rFB,'rwfg0']))
print(np.median((cell_epoch_df.loc[rFB,'rwbg']+cell_epoch_df.loc[rFB,'rwfg'])/(cell_epoch_df.loc[rFB,'rwbg0']+cell_epoch_df.loc[rFB,'rwfg0'])))
print(cell_epoch_df['reps'].unique())

mingain=0
rFB = (cell_epoch_df['rfg']>minresp) & (cell_epoch_df['rbg']>minresp) & \
    (cell_epoch_df['rwfg']>mingain) & (cell_epoch_df['rwbg']>mingain) & \
    (cell_epoch_df['rwfg']<maxgain) & (cell_epoch_df['rwbg']<maxgain) & \
    (cell_epoch_df['p1wfg']>mingain) & (cell_epoch_df['p1wbg']>mingain) & \
    (cell_epoch_df['p1wfg']<maxgain) & (cell_epoch_df['p1wbg']<maxgain) & \
    (cell_epoch_df['p2wfg']>mingain) & (cell_epoch_df['p2wbg']>mingain) & \
    (cell_epoch_df['p2wfg']<maxgain) & (cell_epoch_df['p2wbg']<maxgain)
rCC = cell_epoch_df['Binaural Type']=='BG Contra, FG Contra'
rIC = cell_epoch_df['Binaural Type']=='BG Ipsi, FG Contra'
labels=['BG Contra, FG Contra','BG Ipsi, FG Contra']
print('valid n', rFB.sum(), 'out of', len(rFB), 'frac: ', np.round(rFB.mean(),3))


f, axs = plt.subplots(2, 3, figsize=(9, 6), sharex='row', sharey='row')
for l, a1, cb in zip(labels, axs, [rCC, rIC]):
    for pre, ax in zip(prefs, a1):
        ax.plot([-.2, 1], [-.2, 1], 'k--')
        if pre.startswith('r'):
            sf = 1.0
            sf = 1 / cell_epoch_df.loc[rFB & cb, 'sf']
        else:
            sf = cell_epoch_df.loc[rFB & cb, 'sf']
            sf = 1.0
        ax.scatter(cell_epoch_df.loc[rFB & cb, pre + 'wfg'] * sf, cell_epoch_df.loc[rFB & cb, pre + 'wbg'] * sf, s=2)
        ax.set_xlabel(f"{pre}wfg ({np.mean(cell_epoch_df.loc[rFB & cb, pre + 'wfg'] * sf):.3f})")
        ax.set_ylabel(f"{pre}wbg ({np.mean(cell_epoch_df.loc[rFB & cb, pre + 'wbg'] * sf):.3f})")
        ax.set_xlim([-0.3, 1.5])
        ax.set_ylim([-0.3, 1.5])

        print(l, pre, np.mean(cell_epoch_df.loc[rFB & cb, pre + 'wfg'] * sf),
              np.mean(cell_epoch_df.loc[rFB & cb, pre + 'wbg'] * sf))
    a1[1].set_title(l)
plt.tight_layout()

f, axs = plt.subplots(2, 3, figsize=(8, 6))
labels = ['CNN pred', 'LN pred']
for pre, ax, label in zip(prefs[1:], axs, labels):
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[2].plot([-1.2, 1], [-1.2, 1], 'k--')

    fr = cell_epoch_df.loc[rFB, 'rwfg']
    fp = cell_epoch_df.loc[rFB, pre + 'wfg'] * cell_epoch_df.loc[rFB, 'sf']

    br = cell_epoch_df.loc[rFB, 'rwbg']
    bp = cell_epoch_df.loc[rFB, pre + 'wbg'] * cell_epoch_df.loc[rFB, 'sf']

    dr = fr - br
    dp = fp - bp
    ax[0].scatter(fr, fp, s=2)
    ax[2].scatter(dr, dp, s=2)
    ax[0].set_title(f"r={np.corrcoef(fr, fp)[0, 1]:.3f}")
    ax[0].set_xlim([-0.1, 1.5])
    ax[0].set_ylim([-0.1, 1.5])
    ax[0].set_ylabel(label + ' w_fg')
    ax[0].set_xlabel('Actual w_fg')

    ax[1].scatter(br, bp, s=2)
    ax[1].set_xlim([-0.1, 1.5])
    ax[1].set_ylim([-0.1, 1.5])
    ax[1].set_title(f"r={np.corrcoef(br, bp)[0, 1]:.3f}")
    ax[1].set_ylabel(label + ' w_bg')
    ax[1].set_xlabel('Actual w_bg')

    ax[2].set_xlim([-1.2, 1])
    ax[2].set_ylim([-1.2, 1])
    ax[2].set_title(f"r={np.corrcoef(dr, dp)[0, 1]:.3f}")
    ax[2].set_ylabel(label + ' relative gain')
    ax[2].set_xlabel('Actual relative gain')
f.suptitle(siteid)
plt.tight_layout()