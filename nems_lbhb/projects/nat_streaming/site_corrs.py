import os
import io
import logging
import time
import sys, importlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns


import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join
from nems0 import get_setting
from nems0.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems0.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems0.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems0.modules.nonlinearity import _dlog

from nems_lbhb import baphy_experiment
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems_lbhb.projects.olp.OLP_get_epochs import get_rec_epochs, get_stim_type, generate_cc_dataframe
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems0.recording import load_recording
from nems_lbhb.projects.olp import OLP_get_epochs
import nems0.plots.api as nplt
from nems0 import db
from nems_lbhb.baphy_io import get_depth_info
import nems0.epoch as ep
from nems.models.LN import LN_reconstruction, CNN_reconstruction
from nems0.analysis.cluster import cluster_corr

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)

def fb_weights(rfg,rbg,rfgbg,spontbins=50):
    spont = np.concatenate([rfg[:spontbins],rbg[:spontbins],rfgbg[:spontbins]]).mean()
    rfg0 = rfg-spont
    rbg0 = rbg-spont
    rfgbg0 = rfgbg-spont
    y=rfgbg0-rfg0-rbg0
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), y, rcond=None)
    #weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), rfgbg0, rcond=None)

    return weights2+1

monostim=True
if monostim:
    batch=341
    loadkey = "gtgram.fs50.ch18"

    siteids,cellids = db.get_batch_sites(batch)
    #siteids = [siteids[35]]
    if batch == 341:
        siteids.remove('PRN004a')
        siteids.remove('PRN020b')
        siteids.remove('PRN031a')
        siteids.remove('PRN042a')
        siteids.remove('PRN045b')
        siteids.remove('CLT052d')
        siteids.remove('CLT043b')
        siteids.remove('ARM029a')
        siteids.remove('ARM027a')
else:
    # lab-specific code to load data from one experiment.
    loadkey = "gtgram.fs50.ch18.bin100"
    batch = 345
    siteids,cellids = db.get_batch_sites(batch)
    siteid = siteids[4]


cluster_count = 4
groupby = 'fgbg'
pc_count = 8

dfs = []

for siteid in siteids:
    uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
    rec = load_recording(uri)

    d=get_depth_info(siteid=siteid).reset_index()
    a1chans=[r['index'] for i,r in d.iterrows() if r.area in ['BS','PEG','A1']]
    keepchans = [c for c in a1chans if c in rec['resp'].chans]

    rec['resp'] = rec['resp'].rasterize().extract_channels(keepchans)

    # log compress and normalize stim
    fn = lambda x: _dlog(x, -1)
    rec['stim'] = rec['stim'].transform(fn, 'stim')
    rec['stim'] = rec['stim'].normalize('minmax')
    rec['resp'] = rec['resp'].normalize('minmax')


    epoch_df_all = OLP_get_epochs.get_rec_epochs(fs=50, rec=rec)
    epoch_df = epoch_df_all.loc[(epoch_df_all['Dynamic Type'] == 'fullBG/fullFG')]
    epoch_df = epoch_df.loc[(epoch_df['SNR']==0)]
    #epoch_df = epoch_df.loc[(epoch_df['SNR']==10)]
    epoch_df = epoch_df.loc[(epoch_df['Binaural Type'] == 'BG Contra, FG Contra') |
                            (epoch_df['Binaural Type'] == 'BG Ipsi, FG Contra')]
    #epoch_df = epoch_df_all.loc[(epoch_df_all['Binaural Type'] == 'BG Ipsi, FG Contra')]
    epoch_df = epoch_df.reset_index()

    resp = rec['resp']
    stim = rec['stim']

    if groupby=='bg':
        unique_bg = list(epoch_df['BG'].unique())
    elif groupby=='fgbg':
        unique_bg = list(epoch_df['BG + FG'].unique())
    else:
        raise ValueError('unknown groupby')

    for ebg in unique_bg:
        if groupby=='bg':
            bids = (epoch_df['BG']==ebg)
        elif groupby=='fgbg':
            bids = (epoch_df['BG + FG']==ebg)

        ebgs = epoch_df.loc[bids, 'BG'].to_list()
        efgs = epoch_df.loc[bids, 'FG'].to_list()
        efgbgs = epoch_df.loc[bids, 'BG + FG'].to_list()
        epoch_list = [efgbgs, efgs, ebgs]
        rfgbg = np.concatenate([resp.extract_epoch(e) for e in efgbgs],axis=2).mean(axis=0)
        rfg = np.concatenate([resp.extract_epoch(e) for e in efgs],axis=2).mean(axis=0)
        rbg = np.concatenate([resp.extract_epoch(e) for e in ebgs],axis=2).mean(axis=0)

        # norm_psth = psth - np.mean(psth[:,:25], axis=1, keepdims=True)
        norm_psth = rfgbg - np.mean(rfgbg, axis=1, keepdims=True)
        #norm_psth = rbg - np.mean(rbg, axis=1, keepdims=True)
        s = np.std(norm_psth, axis=1, keepdims=True)
        norm_psth /= (s + (s == 0))
        sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

        use_abs_corr = False
        if use_abs_corr:
            cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, use_abs=True, count=cluster_count, threshold=1.5)
        else:
            cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, count=cluster_count, threshold=1.85)

        pca = PCA(n_components=pc_count)
        pca = pca.fit(norm_psth.T)

        weights = np.zeros((rfg.shape[0],2))
        ccs = np.zeros((rfg.shape[0],2)) * np.nan
        cluster_count = idx_to_cluster_array.max()
        cluster_n = np.zeros(cluster_count)

        for i in range(rfg.shape[0]):
            weights[i] = fb_weights(rfg[i],rbg[i],rfgbg[i],spontbins=25)
            if (rfg[i].std()>0) & (rfg[i].std()>0) & (rfgbg[i].std()>0):
                ccs[i,0] = np.corrcoef(rfg[i], rfgbg[i])[0, 1]
                ccs[i,1] = np.corrcoef(rbg[i], rfgbg[i])[0, 1]

        mw=np.zeros((cluster_count,2))
        mcc=np.zeros((cluster_count,2))
        for c in range(cluster_count):
            cluster_n[c] = (idx_to_cluster_array == c + 1).sum()
            for ri, rsingle in enumerate([rfg, rbg]):
                mw[c,ri] = np.mean(weights[idx_to_cluster_array == c + 1, ri])
                mcc[c,ri] = np.mean(ccs[idx_to_cluster_array == c + 1, ri])
        cluster_cum_n = np.cumsum(cluster_n)

        mean_cc=np.zeros((cluster_count,3))
        cluster_cc = np.zeros((cluster_count,3))
        p_cc = np.zeros((pc_count,3))
        cluster_psth = np.zeros((cluster_count, 3, rfg.shape[1]))
        cluster_pc = np.zeros((pc_count, 3, rfg.shape[1]))

        # iterate through epochs, plotting data for each one in a different column
        for col, rsingle in enumerate([rfgbg, rfg, rbg]):

            # extract spike data for current epoch and cell (cid)
            norm_psth = rsingle - np.mean(rsingle, axis=1, keepdims=True)
            s = np.std(rsingle, axis=1, keepdims=True)
            norm_psth /= (s + (s == 0))
            sc = norm_psth @ norm_psth.T / norm_psth.shape[1]
            cluster_pc[:,col,:] = pca.transform(norm_psth.T).T

            # for display, compute relative to spont
            norm_psth = rsingle - np.mean(rsingle[:, :25], axis=1, keepdims=True)
            norm_psth /= (s + (s == 0))

            sc_sorted = sc[cc_idx, :][:, cc_idx]
            for c in range(cluster_count):
                cluster_psth[c, col, :] = norm_psth[(idx_to_cluster_array == c + 1), :].mean(axis=0)
                cc_sub = sc[(idx_to_cluster_array == c + 1), :][:, (idx_to_cluster_array == c + 1)]
                if len(cc_sub)>1:
                    mean_cc[c,col] = cc_sub[np.triu_indices(cc_sub.shape[0],k=1)].mean()
                else:
                    print(f'cluster_n == 1 for cluster {c}')

            if col > 0:
                for c in range(pc_count):
                    p_cc[c,col] = np.corrcoef(cluster_pc[c,col], cluster_pc[c,0])[0, 1]
                for c in range(cluster_count):
                    cluster_cc[c,col] = np.corrcoef(cluster_psth[c,col], cluster_psth[c,0])[0, 1]

        d = {'siteid': siteid, 'groupby': groupby, 'cluster_count': cluster_count,
             'estim': ebg, 'cid': np.arange(cluster_count), 'cluster_n': cluster_n,
             'mw_fg': mw[:,0], 'mw_bg': mw[:,1],
             'cc_fg': mcc[:,0], 'cc_bg': mcc[:,1],
             'clc_fg': cluster_cc[:,1], 'clc_bg': cluster_cc[:,2],
             'mean_sc_fgbg': mean_cc[:,0], 'mean_sc_fg': mean_cc[:,1], 'mean_sc_bg': mean_cc[:,2],
             }
        dfs.append(pd.DataFrame(d))

        single_plot = False
        if single_plot:
            # create figure with 4x3 subplots
            f, ax = plt.subplots(5, 3, figsize=(6, 8), sharex='row', sharey='row')

            # iterate through epochs, plotting data for each one in a different column
            for col, epochs in enumerate(epoch_list):

                # extract spike data for current epoch and cell (cid)
                raster = np.concatenate([resp.extract_epoch(e) for e in epochs], axis=2)
                psth = np.mean(raster, axis=0)
                norm_psth = psth - np.mean(psth, axis=1, keepdims=True)
                #norm_psth = psth - np.mean(psth[:, :25], axis=1, keepdims=True)
                s = np.std(psth, axis=1, keepdims=True)
                norm_psth /= (s + (s == 0))
                sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

                p = pca.transform(norm_psth.T).T

                # for display, compute relative to spont
                norm_psth = psth - np.mean(psth[:, :25], axis=1, keepdims=True)
                sc_sorted = sc[cc_idx, :][:, cc_idx]

                spec = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in epochs], axis=1)

                ax[0, col].imshow(spec, aspect='auto', cmap='gray_r', interpolation='none',
                                  origin='lower', extent=[-0.5, 1.5, 1, spec.shape[0]])
                ax[0, col].set_title(epochs[0].replace("STIM_", ""))
                ax[0, col].set_ylabel('Freq')

                if use_abs_corr:
                    ax[1, col].imshow(np.abs(sc_sorted), aspect='equal', cmap='gray_r', interpolation='none',
                                      vmin=0, vmax=1)
                else:
                    ax[1, col].imshow(sc_sorted, aspect='equal', cmap='gray_r', interpolation='none',
                                      vmin=0, vmax=1)
                ax[1, col].vlines(np.cumsum(cluster_n)[:-1] - 0.5, -0.5, sc_sorted.shape[1] - 0.5, lw=0.5, color='r')
                ax[1, col].hlines(np.cumsum(cluster_n)[:-1] - 0.5, -0.5, sc_sorted.shape[1] - 0.5, lw=0.5, color='r')
                ccstr = "\n".join([f" {i+1}: {cc:.3f}" for i,cc in enumerate(mean_cc[:,col])])
                ax[1,col].text(sc_sorted.shape[0], sc_sorted.shape[1], ccstr)

                extent = [0, cluster_psth.shape[2]/rec['resp'].fs, cluster_count+0.5, 0.5]
                ax[2, col].imshow(cluster_psth[:,col,:], aspect='auto', interpolation='none',
                                  origin='upper', vmin=0, extent=extent)
                extent = [0, cluster_psth.shape[2]/rec['resp'].fs, pc_count+0.5, 0.5]
                ax[4, col].imshow(cluster_pc[:,col,:], aspect='auto', interpolation='none',
                                  origin='upper', vmin=0, extent=extent)

                extent = [0, cluster_psth.shape[2]/rec['resp'].fs, psth.shape[0]+0.5, 0.5]
                ax[3, col].imshow(psth[cc_idx, :], aspect='auto', interpolation='none',
                                  origin='upper', cmap='gray_r', extent=extent)
                for c in range(cluster_count):
                    ax[3, col].axhline(y=cluster_cum_n[c]+0.5, ls='--', lw='0.5', color='red')

                if col > 0:
                    for c in range(cluster_count):
                        ax[2, col].text(cluster_psth.shape[2]/rec['resp'].fs, c+1,
                                        f" {cluster_cc[c,col]:.2f}", va='center', fontsize=7)
                    for c in range(pc_count):
                        ax[4, col].text(p.shape[1] / rec['resp'].fs, c + 1,
                                        f" {p_cc[c,col]:.2f}", va='center', fontsize=7)
                else:
                    p_cc_diff = p_cc[:,1] - p_cc[:,2]
                    cc_diff = cluster_cc[:,1] - cluster_cc[:,2]
                    tcol = {0: 'red', 1: 'blue'}

                    for c in range(cluster_count):
                        ax[2, 0].text(cluster_psth.shape[2] / rec['resp'].fs, c + 1,
                                      f" {p_cc_diff[c]:.2f} ({int(cluster_n[c])})",
                                      va='center', fontsize=7, color=tcol[p_cc_diff[c] > 0])

                        dmw = mw[c,0]-mw[c,1]
                        dcc = mcc[c,0]-mcc[c,1]

                        ax[3, 0].text(psth.shape[1]/rec['resp'].fs, cluster_cum_n[c],
                                      f" {dmw:.2f} {dcc:.2f}",
                                      va='bottom', fontsize=7, color=tcol[dmw > 0])

                    for c in range(pc_count):
                        ax[4, 0].text(p.shape[1] / rec['resp'].fs, c + 1,
                                      f" {p_cc_diff[c]:.2f}",
                                      va='center', fontsize=7, color=tcol[p_cc_diff[c] > 0])

                    ax[1, 0].set_ylabel('Unit (sorted)')
                    ax[2, 0].set_ylabel('Cluster')
            f.suptitle(f"{siteid} {ebg}: Signal correlation")
            plt.tight_layout()
        print(f"{siteid} {ebg}: {cluster_n}")

df = pd.concat(dfs, ignore_index=True)
df = df.loc[((df['cluster_n']>2) &
             np.isfinite(df['cc_fg']) & np.isfinite(df['cc_bg']) &
             (df['mw_fg']+df['mw_bg']>0.2) &
             (df['mw_fg']<1.0) & (df['mw_bg']<1.0))]
df['dmw'] = df['mw_fg']-df['mw_bg']
df['dcc'] = df['cc_fg']-df['cc_bg']

f,ax = plt.subplots(3,3, sharex='row', sharey='row')
sns.scatterplot(df, x='mw_fg', y='mw_bg', ax=ax[0,0], s=5)
sns.scatterplot(df, x='cc_fg', y='cc_bg', ax=ax[0,1], s=5)

sns.regplot(df, x='mean_sc_fgbg', y='dmw',
            fit_reg=True, ax=ax[1,0], scatter_kws={'s': 3})
ax[1,0].set_title(f"r={np.corrcoef(df['mean_sc_fgbg'], df['dmw'])[0,1]:.3}")
sns.regplot(df, x='mean_sc_fg', y='dmw',
            fit_reg=True, ax=ax[1,1], scatter_kws={'s': 3})
ax[1,1].set_title(f"r={np.corrcoef(df['mean_sc_fg'], df['dmw'])[0,1]:.3}")
sns.regplot(df, x='mean_sc_bg', y='dmw',
            fit_reg=True, ax=ax[1,2], scatter_kws={'s': 3})
ax[1,2].set_title(f"r={np.corrcoef(df['mean_sc_bg'], df['dmw'])[0,1]:.3}")
sns.regplot(df, x='mean_sc_fgbg', y='dcc',
            fit_reg=True, ax=ax[2,0], scatter_kws={'s': 3})
ax[2,0].set_title(f"r={np.corrcoef(df['mean_sc_fgbg'], df['dcc'])[0,1]:.3}")
sns.regplot(df, x='mean_sc_fg', y='dcc',
            fit_reg=True, ax=ax[2,1], scatter_kws={'s': 3})
ax[2,1].set_title(f"r={np.corrcoef(df['mean_sc_fg'], df['dcc'])[0,1]:.3}")
sns.regplot(df, x='mean_sc_bg', y='dcc',
            fit_reg=True, ax=ax[2,2], scatter_kws={'s': 3})
ax[2,2].set_title(f"r={np.corrcoef(df['mean_sc_bg'], df['dcc'])[0,1]:.3}")

plt.tight_layout()

raise ValueError("stopping")

for epoch_id in range(len(epoch_df)): # in range(5): #
    # names of epochs from this triad
    efg = epoch_df.loc[epoch_id, 'FG']
    ebg = epoch_df.loc[epoch_id, 'BG']
    efgbg = epoch_df.loc[epoch_id, 'BG + FG']
    epoch_list = [efgbg, efg, ebg]

    # cluster based on FG(?) responses
    # psth = np.concatenate([resp.extract_epoch(efgbg).mean(axis=0),
    #                       resp.extract_epoch(efg).mean(axis=0),
    #                       resp.extract_epoch(ebg).mean(axis=0)], axis=1)
    # psth = np.concatenate([resp.extract_epoch(efg).mean(axis=0),
    #                       resp.extract_epoch(ebg).mean(axis=0)], axis=1)
    psth = resp.extract_epoch(efgbg).mean(axis=0)

    # norm_psth = psth - np.mean(psth[:,:25], axis=1, keepdims=True)
    norm_psth = psth - np.mean(psth, axis=1, keepdims=True)
    s = np.std(psth, axis=1, keepdims=True)
    norm_psth /= (s + (s == 0))
    sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

    use_abs_corr = False
    if use_abs_corr:
        cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, use_abs=True, count=5, threshold=1.5)
    else:
        cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, count=5, threshold=1.85)

    # create figure with 4x3 subplots
    f, ax = plt.subplots(4, 3, figsize=(6, 6), sharex='row', sharey='row')

    # iterate through epochs, plotting data for each one in a different column
    for col, epoch in enumerate(epoch_list):

        # extract spike data for current epoch and cell (cid)
        raster = resp.extract_epoch(epoch)
        psth = np.mean(raster, axis=0)
        norm_psth = psth - np.mean(psth, axis=1, keepdims=True)
        #norm_psth = psth - np.mean(psth[:, :25], axis=1, keepdims=True)
        s = np.std(psth, axis=1, keepdims=True)
        norm_psth /= (s + (s == 0))
        sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

        # for display, compute relative to spont
        norm_psth = psth - np.mean(psth[:, :25], axis=1, keepdims=True)

        spec = stim.extract_epoch(epoch)[0, :, :]

        ax[0, col].imshow(spec, aspect='auto', cmap='gray_r', interpolation='none',
                          origin='lower', extent=[-0.5, 1.5, 1, spec.shape[0]])
        ax[0, col].set_title(epoch.replace("STIM_", ""))
        ax[0, col].set_ylabel('Freq')

        sc_sorted = sc[cc_idx, :][:, cc_idx]

        cluster_count = idx_to_cluster_array.max()
        cluster_psth = np.zeros((cluster_count, psth.shape[1]))
        cluster_n = np.zeros(cluster_count)
        mean_cc = np.zeros(cluster_count)
        for c in range(cluster_count):
            cluster_psth[c, :] = norm_psth[(idx_to_cluster_array == c + 1), :].mean(axis=0)
            cluster_n[c] = (idx_to_cluster_array == c + 1).sum()
            cc_sub = sc[(idx_to_cluster_array == c + 1), :][:, (idx_to_cluster_array == c + 1)]
            mean_cc[c] = cc_sub[np.triu_indices(cc_sub.shape[0],k=1)].mean()

        if col == 0:
            cluster_psth0 = cluster_psth
        cluster_cc = np.zeros(cluster_count)

        for c in range(cluster_count):
            cluster_cc[c] = np.corrcoef(cluster_psth[c], cluster_psth0[c])[0, 1]
        if col == 1:
            fg_cc = cluster_cc
        elif col == 2:
            bg_cc = cluster_cc
        if use_abs_corr:
            ax[1, col].imshow(np.abs(sc_sorted), aspect='equal', cmap='gray_r', interpolation='none',
                              vmin=0, vmax=1)
        else:
            ax[1, col].imshow(sc_sorted, aspect='equal', cmap='gray_r', interpolation='none',
                              vmin=0, vmax=1)
        ax[1, col].vlines(np.cumsum(cluster_n)[:-1] - 0.5, -0.5, sc_sorted.shape[1] - 0.5, lw=0.5, color='r')
        ax[1, col].hlines(np.cumsum(cluster_n)[:-1] - 0.5, -0.5, sc_sorted.shape[1] - 0.5, lw=0.5, color='r')
        ccstr = "\n".join([f" {i+1}: {cc:.3f}" for i,cc in enumerate(mean_cc)])
        ax[1,col].text(sc_sorted.shape[0],sc_sorted.shape[1],ccstr)

        extent = [0, cluster_psth.shape[1]/rec['resp'].fs, cluster_count+0.5, 0.5]
        ax[2, col].imshow(cluster_psth, aspect='auto', interpolation='none',
                          origin='upper', vmin=0, extent=extent)

        extent = [0, cluster_psth.shape[1]/rec['resp'].fs, psth.shape[0]+0.5, 0.5]
        ax[3, col].imshow(psth[cc_idx, :], aspect='auto', interpolation='none',
                          origin='upper', cmap='gray_r', extent=extent)

        if col > 0:
            for c in range(cluster_count):
                ax[2, col].text(cluster_psth.shape[1]/rec['resp'].fs, c+1,
                                f" {cluster_cc[c]:.2f}", va='center', fontsize=7)
        if col == 2:
            cc_diff = fg_cc - bg_cc
            for c in range(cluster_count):
                if cc_diff[c] > 0:
                    tcol = 'blue'
                else:
                    tcol = 'red'
                ax[2, 0].text(cluster_psth.shape[1]/rec['resp'].fs, c+1,
                              f" {cc_diff[c]:.2f} ({int(cluster_n[c])})",
                              va='center', fontsize=7, color=tcol)
        if col == 0:
            ax[1, 0].set_ylabel('Unit (sorted)')
            ax[2, 0].set_ylabel('Cluster')
    f.suptitle(f"{epoch_id}: Signal correlation")
    plt.tight_layout()
    print(efgbg)
    print(f"{epoch_id}: {cluster_n}")

