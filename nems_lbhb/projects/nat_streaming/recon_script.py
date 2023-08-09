import os
import io
import logging
import time
import sys, importlib

import numpy as np

import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 12,
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch

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
from nems.models import LN

log = logging.getLogger(__name__)

import importlib
importlib.reload(LN)

def get_cluster_data(rec, idx_to_cluster_array, cluster_ids):
    resp = rec['resp']
    stim = rec['stim']
    keep_cellids = [resp.chans[i] for i in range(len(idx_to_cluster_array)) if idx_to_cluster_array[i] in cluster_ids]

    resp2 = resp.extract_channels(keep_cellids)

    bnt_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_.*seq')
    if len(bnt_epochs)==0:
        bnt_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_00cat') + \
                     ep.epoch_names_matching(rec['resp'].epochs, 'STIM_cat')

    bnt_stim = np.stack([stim.extract_epoch(e)[0].T for e in bnt_epochs], axis=0)
    bnt_resp = np.stack([resp2.extract_epoch(e).mean(axis=0).T for e in bnt_epochs], axis=0)

    print("Resp shape:", bnt_resp.shape, "Stim shape:", bnt_stim.shape)

    return bnt_resp, bnt_stim, resp2

def nmse(a, b):
    """
    mse of a vs. b, normed by std of b
    :param a:
    :param b:
    :return:
    """
    return np.std(a - b) / np.std(b)

def corrcoef(a,b):
    return np.corrcoef(a.flatten(),b.flatten())[0,1]

batch = 345
siteids,cellids = db.get_batch_sites(batch)
# lab-specific code to load data from one experiment.
loadkey = "gtgram.fs50.ch18.bin6"
siteid = 'PRN015a'
siteid = 'PRN043a'
siteid = 'PRN050a'
siteid = 'PRN053a'
siteid = 'PRN051a'
siteid = 'PRN031a'

modeltype='LN'
if modeltype=='LN':
    outpath = '/auto/users/svd/projects/olp/reconstruction_LN/'
elif modeltype=='CNN':
    outpath = '/auto/users/svd/projects/olp/reconstruction_CNN/'


batch=341
loadkey = "gtgram.fs50.ch18"

siteid='PRN017a'
siteid='PRN022a'
siteid='PRN013c'
siteid='PRN015a'
siteid='PRN050a'
siteids,cellids = db.get_batch_sites(batch)

for siteid in siteids[36:]:
    uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
    rec = load_recording(uri)
    try:
        d=get_depth_info(siteid=siteid).reset_index()
        a1chans=[r['index'] for i,r in d.iterrows() if r.area in ['A1', 'PEG','BRD']]
        keepchans = [c for c in a1chans if c in rec['resp'].chans]
    except:
        keepchans = [c for c in rec['resp'].chans]
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
    if (epoch_df['Synth Type']=='Unsynthetic').sum()>0:
        epoch_df = epoch_df.loc[(epoch_df['Synth Type']=='Unsynthetic')]
    else:
        epoch_df = epoch_df.loc[(epoch_df['Synth Type']=='Non-RMS Unsynthetic')]


    epoch_df = epoch_df.reset_index()

    resp = rec['resp']
    stim = rec['stim']

    unique_bg = list(epoch_df['BG'].unique())
    try:
        df_recon = pd.read_csv(outpath+'df_recon.csv', index_col=0)
        df_recon = df_recon.loc[df_recon.siteid!=siteid].copy()
        df_recons=[df_recon]
    except:
        df_recons = []

    skip_recon = False

    for ebg in unique_bg:
        print(f"Focusing on bg={ebg}")
        bids = (epoch_df['BG']==ebg)

        ebgs = epoch_df.loc[bids, 'BG'].to_list()
        efgs = epoch_df.loc[bids, 'FG'].to_list()
        efgbgs = epoch_df.loc[bids, 'BG + FG'].to_list()
        epoch_list = [efgbgs, efgs, ebgs]
        rfgbg = np.concatenate([resp.extract_epoch(e) for e in efgbgs],axis=2).mean(axis=0)
        rfg = np.concatenate([resp.extract_epoch(e) for e in efgs],axis=2).mean(axis=0)
        rbg = np.concatenate([resp.extract_epoch(e) for e in ebgs],axis=2).mean(axis=0)

        # norm_psth = psth - np.mean(psth[:,:25], axis=1, keepdims=True)
        r=rfgbg
        #r=np.concatenate([rfgbg, rbg], axis=1)
        #r=np.concatenate([rbg], axis=1)
        norm_psth = r - np.mean(r, axis=1, keepdims=True)
        s = np.std(norm_psth, axis=1, keepdims=True)
        norm_psth /= (s + (s == 0))
        sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

        cluster_count = 3
        use_abs_corr = False
        if use_abs_corr:
            cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, use_abs=True, count=cluster_count, threshold=1.5)
        else:
            cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, count=cluster_count, threshold=1.85)

        # create figure with 4x3 subplots
        f1, ax = plt.subplots(4, 3, figsize=(6, 6), sharex='row', sharey='row')

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

            # for display, compute relative to spont
            norm_psth = psth - np.mean(psth[:, :25], axis=1, keepdims=True)

            spec = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in epochs], axis=1)

            ax[0, col].imshow(spec, aspect='auto', cmap='gray_r', interpolation='none',
                              origin='lower', extent=[-0.5, 1.5, 1, spec.shape[0]])
            ax[0, col].set_title(epochs[0].replace("STIM_", ""))
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

            ax[1, col].vlines(np.cumsum(cluster_n)[:-1] - 0.5, -0.5, sc_sorted.shape[1] - 0.5, lw=0.5, color='r')
            ax[1, col].hlines(np.cumsum(cluster_n)[:-1] - 0.5, -0.5, sc_sorted.shape[1] - 0.5, lw=0.5, color='r')
            ccstr = "\n".join([f" {i+1}: {cc:.3f}" for i,cc in enumerate(mean_cc)])
            ax[1,col].text(sc_sorted.shape[0],sc_sorted.shape[1],ccstr)

            extent = [0, cluster_psth.shape[1]/rec['resp'].fs, cluster_count+0.5, 0.5]
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
        f1.suptitle(f"{ebg}: Signal correlation")
        plt.tight_layout()
        f1.savefig(f"{outpath}{siteid}_{ebg}_clusters.jpg")
        print(f"{ebg}: {cluster_n}")

        if skip_recon:
            continue

        #raise ValueError('stopping')

        #
        # Fit decoder with different clusters of neurons
        #
        shuffle_count=11
        for shuffidx in range(shuffle_count):
            print(f"{ebg}: decoder fitting shuffle {shuffidx}")
            models = []
            resps = []
            cellcount = rec['resp'].shape[0]
            if shuffidx == 0:
                idx_cluster_map = idx_to_cluster_array
            else:
                idx_cluster_map = np.random.permutation(idx_to_cluster_array)
            cmin, cmax = idx_cluster_map.min(), idx_cluster_map.max()
            cluster_sets = [[c] for c in range(cmin, cmax+1)] + [[c for c in range(cmin, cmax+1)]]
            exclude_clusters = False
            if exclude_clusters:
                # leave out one cluster (instead of just fitting for that one cluster)
                call = np.arange(cmin,cmax+1)
                cluster_sets = [np.setdiff1d(call,c) for c in cluster_sets]
                cluster_sets[-1] = call

            for fidx, cluster_ids in enumerate(cluster_sets):
                print(f"Starting cluster {fidx}: {cluster_ids}")
                bnt_resp, bnt_stim, resp2 = get_cluster_data(rec, idx_cluster_map, cluster_ids)

                nl_kwargs = {'no_shift': False, 'no_offset': False, 'no_gain': False}
                if modeltype=='LN':
                    model = LN.LN_reconstruction(time_bins=19, channels=bnt_resp.shape[2],
                                                  out_channels=bnt_stim.shape[2],
                                                  rank=20, nl_kwargs=nl_kwargs)
                else:
                    model = LN.CNN_reconstruction(time_bins=15, channels=bnt_resp.shape[2],
                                                  out_channels=bnt_stim.shape[2],
                                                  L1=25, L2=0, nl_kwargs=nl_kwargs)
                model = model.fit_LBHB(bnt_resp, bnt_stim, cost_function='squared_error')
                models.append(model)
                resps.append(resp2)
                #raise ValueError("Pause")
            #
            #
            # Compare performance of decoders fit using different clusters
            #
            #plt.close('all')
            f2, ax = plt.subplots(len(cluster_sets)+1, 3, figsize=(8, 8), sharex=True, sharey=True)
            for col, epochs in enumerate(epoch_list):
                spec = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in epochs], axis=1).T
                ax[0, col].imshow(spec.T, aspect='auto', cmap='gray_r', interpolation='none',
                                  origin='lower', extent=[-0.5, 1.5, 1, spec.shape[1]], vmin=0, vmax=1)
                ax[0, col].set_title(epochs[0].replace("STIM_", ""), fontsize=8)
                if col == 0:
                    ax[0, col].set_ylabel('Freq')

            for fidx, cluster_ids in enumerate(cluster_sets):
                model = models[fidx]
                resp2 = resps[fidx]

                d={'siteid': siteid, 'ebg': ebg, 'cid': fidx}
                d['shuffidx'] = shuffidx
                d['cluster_ids'] = ",".join([str(c) for c in cluster_ids])
                d['n_units'] = resp2.shape[0]
                if fidx>=len(cluster_cc):
                    d['cluster_cc']=0
                else:
                    d['cluster_cc']=cluster_cc[fidx]

                spec_fg = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in efgs], axis=1).T
                spec_bg = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in ebgs], axis=1).T
                spec_fgbg = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in efgbgs], axis=1).T

                for col, epochs in enumerate(epoch_list):
                    raster = np.concatenate([resp2.extract_epoch(e) for e in epochs], axis=2)
                    psth = np.mean(raster, axis=0).T

                    #raster0 = np.concatenate([resps[-1].extract_epoch(e) for e in epochs], axis=2)
                    #psth0 = np.mean(raster0, axis=0).T
                    #recon0 = models[-1].predict(psth0)

                    spec = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in epochs], axis=1).T

                    recon = model.predict(psth)
                    #recon -= recon0

                    ax[fidx+1, col].imshow(recon.T, aspect='auto', cmap='gray_r', interpolation='none',
                                      origin='lower', extent=[-0.5, 1.5, 1, recon.shape[1]], vmin=0, vmax=1)
                    if col == 0:
                        recon0 = recon
                        d['Efg'] = nmse(recon, spec_fg)  # np.corrcoef(recon.flatten(), spec_fg.flatten())[0,1] #
                        d['Ebg'] = nmse(recon, spec_bg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
                        d['Efgbg'] = nmse(recon, spec_fgbg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
                        d['Cfg'] = corrcoef(recon, spec_fg)  # np.corrcoef(recon.flatten(), spec_fg.flatten())[0,1] #
                        d['Cbg'] = corrcoef(recon, spec_bg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
                        d['Cfgbg'] = corrcoef(recon, spec_fgbg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #

                        if len(cluster_ids)==1:
                            n_units = (np.isin(idx_to_cluster_array, cluster_ids)).sum()
                            ax[fidx+1, col].set_ylabel(f'{cluster_ids} n={n_units}')
                        else:
                            ax[fidx + 1, col].set_ylabel(f'{cluster_ids}')
                        ax[fidx+1, col].set_title(f"Efgbg={d['Efgbg']:.3f}, Efg={d['Efg']:.3f}, Ebg={d['Ebg']:.3f}", fontsize=8)
                    else:
                        E = nmse(recon, spec)  # np.corrcoef(recon.flatten(), spec.flatten())[0,1] #
                        Er = nmse(recon, recon0)  # np.corrcoef(recon.flatten(), recon0.flatten())[0,1] #
                        C = corrcoef(recon, spec)  # np.corrcoef(recon.flatten(), spec.flatten())[0,1] #
                        Cr = corrcoef(recon, recon0)  # np.corrcoef(recon.flatten(), recon0.flatten())[0,1] #
                        ax[fidx+1, col].set_title(f"E={E:.3f}, Er={Er:.3f}", fontsize=8)
                        if col==1:
                            d['Erfg'] = Er
                            d['Crfg'] = Cr
                        else:
                            d['Erbg'] = Er
                            d['Crbg'] = Cr
                df_recons.append(pd.DataFrame(d, index=[0]))

            f2.suptitle(f'{ebg.split("_")[1]} shuffidx={shuffidx} exclude={exclude_clusters}')
            plt.tight_layout()
            if shuffidx==0:
                f2.savefig(f"{outpath}{siteid}_{ebg}_recon_shuff{shuffidx}.jpg")
            #plt.close(f2)

        #plt.close(f1)

    df_recon = pd.concat(df_recons, ignore_index=True)
    df_recon.to_csv(outpath+'df_recon.csv')

    d = df_recon.loc[df_recon.siteid==siteid].copy()
    d['fg_rat'] = d['Efg']# /d['Efgbg']
    d['bg_rat'] = d['Ebg'] #/d['Efgbg']
    d['fg_crat'] = d['Cfg'] #/d['Cfgbg']
    d['bg_crat'] = d['Cbg'] #/d['Cfgbg']

    p=d[['ebg', 'cid', 'shuffidx', 'fg_rat', 'bg_rat', 'fg_crat', 'bg_crat']].groupby(['ebg', 'cid', 'shuffidx']).mean()
    p=p.reset_index(2)
    f3,ax=plt.subplots(2,1)
    ax[0].plot(p.loc[p.shuffidx == 0, ['fg_rat','bg_rat']].values)
    for sh in range(1,shuffle_count):
        ax[0].plot(p.loc[p.shuffidx==sh, ['fg_rat','bg_rat']].values,
                lw=0.5, color='gray')
    breaks = np.arange(cluster_count,(p['shuffidx']==0).sum(), cluster_count+1)
    yl = ax[0].get_ylim()
    ax[0].vlines(breaks, yl[0], yl[1], ls='--', color='gray')
    for i, e in enumerate(unique_bg):
        e_ = e.split("_")[1]
        ax[0].text(breaks[i], yl[1], e_, ha='right')

    ax[0].set_ylabel('MSE')

    ax[1].plot(p.loc[p.shuffidx == 0, ['fg_crat', 'bg_crat']].values)
    for sh in range(1, shuffle_count):
        ax[1].plot(p.loc[p.shuffidx == sh, ['fg_crat', 'bg_crat']].values,
                   lw=0.5, color='gray')
    yl = ax[1].get_ylim()
    ax[1].vlines(breaks, yl[0], yl[1], ls='--', color='gray')
    ax[1].legend(('FG ratio','BG ratio'))
    ax[1].set_ylabel('CC')
    f3.suptitle(siteid)
    f3.savefig(f"{outpath}{siteid}_summary_clusters.jpg")
