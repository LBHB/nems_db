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

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV

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
from nems0.utils import smooth

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

font_size=6
params = {'legend.fontsize': font_size,
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size}
plt.rcParams.update(params)

def fb_weights(rfg,rbg,rfgbg,spontbins=50):
    spont = np.concatenate([rfg[:spontbins],rbg[:spontbins],rfgbg[:spontbins]]).mean()
    rfg0 = rfg-spont
    rbg0 = rbg-spont
    rfgbg0 = rfgbg-spont
    y=rfgbg0-rfg0-rbg0
    weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), y, rcond=None)
    #weights2, residual_sum, rank, singular_values = np.linalg.lstsq(np.stack([rfg0, rbg0], axis=1), rfgbg0, rcond=None)

    return weights2+1

monostim=False
example=False
if example:
    batch = 345
    loadkey = "gtgram.fs50.ch36.bin100"
    siteids, cellids = db.get_batch_sites(batch)
    siteids=['CLT020a']
    groupby = 'fgbgboth'

elif monostim:
    batch = 341
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
    groupby = 'bg'

else:
    # lab-specific code to load data from one experiment.
    batch = 345
    loadkey = "gtgram.fs50.ch18.bin100"
    siteids,cellids = db.get_batch_sites(batch)
    groupby = 'bg'

outpath='/home/svd/Documents/onedrive/projects/olp/'
cluster_count0 = 3
pc_count0 = 8

dfs = []
dfdiscrims = []
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
    if (epoch_df['Synth Type'] == 'Unsynthetic').sum() > 0:
        epoch_df = epoch_df.loc[(epoch_df['Synth Type'] == 'Unsynthetic')]
    else:
        epoch_df = epoch_df.loc[(epoch_df['Synth Type'] == 'Non-RMS Unsynthetic')]

    epoch_df = epoch_df.reset_index()

    resp = rec['resp']
    stim = rec['stim']

    if groupby=='bg':
        unique_bg = list(epoch_df['BG'].unique())
    elif groupby == 'bgboth':
        unique_bg = list(epoch_df['BG'].str[:-6].unique())
    elif groupby == 'fgbg':
        unique_bg = list(epoch_df['BG + FG'].unique())
    elif groupby == 'fgbgboth':
        unique_bg = list(epoch_df['BG + FG'].unique())
        unique_bg = [u for u in unique_bg if '-2_' not in u]
    else:
        raise ValueError('unknown groupby')

    for ebg in unique_bg:
        if groupby=='bg':
            bids = (epoch_df['BG']==ebg)
        elif groupby == 'bgboth':
            bids = (epoch_df['BG'].str.startswith(ebg))
        elif groupby == 'fgbg':
            bids = (epoch_df['BG + FG'] == ebg)
        elif groupby=='fgbgboth':
            ebgci = ebg.replace("-1_", "-2_")
            bids = (epoch_df['BG + FG']==ebg) | (epoch_df['BG + FG']==ebgci)

        ebgs = epoch_df.loc[bids, 'BG'].to_list()
        efgs = epoch_df.loc[bids, 'FG'].to_list()
        efgbgs = epoch_df.loc[bids, 'BG + FG'].to_list()
        epoch_list = [efgbgs, efgs, ebgs]
        rfgbg = np.concatenate([resp.extract_epoch(e) for e in efgbgs],axis=2).mean(axis=0)
        rfg = np.concatenate([resp.extract_epoch(e) for e in efgs],axis=2).mean(axis=0)
        rbg = np.concatenate([resp.extract_epoch(e) for e in ebgs],axis=2).mean(axis=0)

        # norm_psth = psth - np.mean(psth[:,:25], axis=1, keepdims=True)
        norm_psth = rfgbg - np.mean(rfgbg, axis=1, keepdims=True)
        s = np.std(norm_psth, axis=1, keepdims=True)
        norm_psth /= (s + (s == 0))
        sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

        use_abs_corr = False
        if use_abs_corr:
            cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, use_abs=True, count=cluster_count0, threshold=1.5)
        else:
            cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True, count=cluster_count0, threshold=1.85)
        pc_count = np.min([pc_count0, norm_psth.shape[0]])
        pca = PCA(n_components=pc_count)
        pca = pca.fit(norm_psth.T)

        fit_weights_separately=True
        if fit_weights_separately:
            bid_set = epoch_df.loc[bids].index.to_list()
        else:
            bid_set = [epoch_df.loc[bids].index.to_list()[0]]

        psths = []
        for i, epochs in enumerate(epoch_list):
            col = 2 - i
            # extract spike data for current epoch and cell (cid)
            raster = np.concatenate([resp.extract_epoch(e) for e in epochs], axis=2)
            psths.append(np.mean(raster, axis=0))
        psth_all = np.stack(psths, axis=0)
        spont = np.mean(psth_all[:, :, :25], axis=(0, 2), keepdims=True)

        mean_cc_set = {}
        for bid in bid_set:
            if len(bid_set)>=0:
                ebgs = epoch_df.loc[[bid], 'BG'].to_list()
                efgs = epoch_df.loc[[bid], 'FG'].to_list()
                efgbgs = epoch_df.loc[[bid], 'BG + FG'].to_list()
                epoch_list = [efgbgs, efgs, ebgs]
                rfgbg = np.concatenate([resp.extract_epoch(e) for e in efgbgs], axis=2).mean(axis=0)
                rfg = np.concatenate([resp.extract_epoch(e) for e in efgs], axis=2).mean(axis=0)
                rbg = np.concatenate([resp.extract_epoch(e) for e in ebgs], axis=2).mean(axis=0)
                estim = efgbgs[0]
            else:
                estim = ebg
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
                        if (cluster_pc[c,col].std()>0) & (cluster_pc[c,0].std()>0):
                            p_cc[c,col] = np.corrcoef(cluster_pc[c,col], cluster_pc[c,0])[0, 1]
                    for c in range(cluster_count):
                        if (cluster_cc[c,col].std()>0) & (cluster_cc[c,0].std()>0):
                            cluster_cc[c,col] = np.corrcoef(cluster_psth[c,col], cluster_psth[c,0])[0, 1]

            d = {'siteid': siteid, 'groupby': groupby, 'cluster_count': cluster_count0,
                 'estim': estim, 'Binaural Type': epoch_df.loc[[bid],'Binaural Type'].values[0],
                 'cid': np.arange(cluster_count), 'cluster_n': cluster_n,
                 'mw_fg': mw[:,0], 'mw_bg': mw[:,1],
                 'cc_fg': mcc[:,0], 'cc_bg': mcc[:,1],
                 'clc_fg': cluster_cc[:,1], 'clc_bg': cluster_cc[:,2],
                 'mean_sc_fgbg': mean_cc[:,0], 'mean_sc_fg': mean_cc[:,1], 'mean_sc_bg': mean_cc[:,2],
                 }
            dfs.append(pd.DataFrame(d))
            mean_cc_set[estim]=mean_cc

            if bid==bid_set[0]:
                ww=np.stack((idx_to_cluster_array, weights[:,0]-weights[:,1]), axis=1)
                a1 = ww[:,1].argsort()
                a2 = ww[a1,0].argsort(kind='mergesort')
                cc_idx_new = a1[a2]
            cc_idx = cc_idx_new

            #if False and (resp.shape[0]>15) and \
            #        (('KitWhine' in estim) or ('KitHigh' in estim) or
            #         ('FightSqueak' in estim) or ('Gobble' in estim)):
            #if True:
            #if False:
            if (len(siteids)==1)  and \
                    (('KitWhine' in estim) or ('KitHigh' in estim) or
                     ('FightSqueak' in estim) or ('Gobble' in estim)):
                # create figure with 4x3 subplots
                f, ax = plt.subplots(2, 5, figsize=(7, 2.5), sharey='row', sharex='col')

                epoch_list = [efgbgs, ebgs, efgs]

                # iterate through epochs, plotting data for each one in a different column
                for i, epochs in enumerate(epoch_list):
                    col = 2 - i
                    raster = np.concatenate([resp.extract_epoch(e) for e in epochs], axis=2)
                    psth = np.mean(raster, axis=0)
                    norm_psth = psth - np.mean(psth, axis=1, keepdims=True)
                    s = np.std(psth, axis=1, keepdims=True)
                    norm_psth /= (s + (s == 0))
                    sc = norm_psth @ norm_psth.T / norm_psth.shape[1]
                    p = pca.transform(norm_psth.T).T

                    # for display, compute relative to spont
                    norm_psth = smooth(psth, window_len=5, axis=1) - spont[0]

                    pmax = np.abs(psth_all-spont).max(axis=(0,2), keepdims=True)[0]
                    pmax[pmax == 0] = 1
                    norm_psth /= pmax

                    sc_sorted = sc[cc_idx, :][:, cc_idx]

                    spec = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in epochs], axis=1)
                    if batch == 345:
                        m = int(spec.shape[0]/2)
                        smax=spec.max()
                        ax[0, col].imshow(spec[:m,:]**2, aspect='auto', cmap='gray_r', interpolation='none',
                                          origin='lower', extent=[-0.5, 1.5, 1, m+1], vmax=smax)
                        ax[0, col].imshow(spec[m:,:]**2, aspect='auto', cmap='gray_r', interpolation='none',
                                          origin='lower', extent=[-0.5, 1.5, m+2, m*2 + 1], vmax=smax)
                        ax[0, col].set_xlim([-0.5, 1.5])
                        ax[0, col].set_ylim([1, m*2+1])

                        ax[0, col].set_title(epochs[0].replace("STIM_", ""))
                    else:
                        m = spec.shape[0]
                        ax[0, col].imshow(spec**2, aspect='auto', cmap='gray_r', interpolation='none',
                                          origin='lower', extent=[-0.5, 1.5, 1, m+1])

                    if col==2:
                        extent = [0.5, psth.shape[0] + 0.5, psth.shape[0] + 0.5, 0.5]
                        relgain = weights[cc_idx,0]-weights[cc_idx,1]
                        relgain[relgain<-1]=-1
                        relgain[relgain>1]=1

                        edges = np.concatenate([[0],np.cumsum(cluster_n)]).astype(int)
                        for c in range(cluster_count):
                            ax[1, 3].plot(relgain[edges[c]:edges[c+1]],
                                          np.arange(edges[c],edges[c+1])+1, color='k', lw=0.5)
                        ax[1, 3].hlines(np.cumsum(cluster_n)[:-1] + 0.5, -1, 1, lw=0.5, color='r')
                        ax[1, 3].axvline(0, lw=0.5, color='k', ls='--')
                        ax[1,3].set_xlim([-2,2])
                        if use_abs_corr:
                            ax[1, 4].imshow(np.abs(sc_sorted), aspect='equal', cmap='gray_r', interpolation='none',
                                            vmin=0, vmax=1, extent=extent)
                        else:
                            ax[1, 4].imshow(sc_sorted, aspect='equal', cmap='gray_r', interpolation='none',
                                            vmin=0, vmax=1, extent=extent)
                        ax[1, 4].vlines(np.cumsum(cluster_n)[:-1] + 0.5, 0.5, sc_sorted.shape[1] + 0.5, lw=0.5, color='r')
                        ax[1, 4].hlines(np.cumsum(cluster_n)[:-1] + 0.5, 0.5, sc_sorted.shape[1] + 0.5, lw=0.5, color='r')
                        for c in range(cluster_count):
                            ax[1, 4].text(sc_sorted.shape[0], cluster_cum_n[c],
                                            f"{mean_cc[c, i]:.3f}", va='bottom', fontsize=6)

                    extent = [-0.5, 1.5, psth.shape[0]+0.5, 0.5]
                    ax[1, col].imshow(norm_psth[cc_idx, :], aspect='auto', interpolation='none',
                                      origin='upper', cmap='bwr', extent=extent, vmin=-1, vmax=1)
                    for c in range(cluster_count):
                        ax[1, col].axhline(y=cluster_cum_n[c]+0.5, ls='-', lw='0.5', color='red')

                    #extent = [-0.5, 1.5, cluster_count+0.5, 0.5]
                    #ax[2, col].imshow(cluster_psth[:,col,:], aspect='auto', interpolation='none',
                    #                  origin='upper', vmin=0, extent=extent)

                    tcol = {0: 'red', 1: 'blue'}
                    if col == 2:
                        p_cc_diff = p_cc[:,1] - p_cc[:,2]
                        cc_diff = cluster_cc[:, 1] - cluster_cc[:, 2]

                        for c in range(cluster_count):
                            relgain = mw[c,0]-mw[c,1]
                            dcc = mcc[c,0]-mcc[c,1]

                            ax[1, col].text(1.5, cluster_cum_n[c],
                                          f" {relgain:.2f}",
                                          va='bottom', fontsize=6, color=tcol[relgain > 0])

                            #ax[2, col].text(1.5, c + 1,
                            #              f" {p_cc_diff[c]:.2f} ({int(cluster_n[c])})",
                            #              va='center', fontsize=6, color=tcol[p_cc_diff[c] > 0])
                    else:
                        for c in range(cluster_count):
                            relgain = mw[c, 0] - mw[c, 1]
                            ax[1, col].text(1.5, cluster_cum_n[c],
                                            f" {mw[c, col]:.2f}",
                                            va='bottom', fontsize=6, color=tcol[relgain > 0])
                            #ax[2, col].text(1.5, c + 1,
                            #                f" {cluster_cc[c, col]:.2f}", va='center', fontsize=6)

                ax[0, 0].set_ylabel('Frequency')
                ax[1, 0].set_ylabel('Unit (sorted)')
                #ax[2, 0].set_ylabel('Cluster')
                ax[1, col].set_xlabel('Time (s)')
                ax[0, 3].set_axis_off()
                ax[1, 3].set_axis_off()
                ax[0, 4].set_axis_off()
                #ax[2, 3].set_axis_off()

                sparts = estim.split("_")
                sf = sparts[1].split("-")[0]
                sb = sparts[2].split("-")[0]
                if epoch_df.loc[bid,'Binaural Type']=='BG Ipsi, FG Contra':
                    ssp = "CI"
                else:
                    ssp = "CC"
                f.suptitle(f"{siteid} {sf}/{sb} {ssp}: Clusters")
                plt.tight_layout()

                print(f"{siteid} {estim}: {cluster_n}")
                outfile = f"{outpath}examples/clusters_{batch}_{siteid}_{sf}_{sb}_{ssp}_{cluster_count0}.pdf"
                print(f"saving clusters fit to {outfile}")
                f.savefig(outfile)

        ebgs = epoch_df.loc[bids, 'BG'].to_list()
        efgs = epoch_df.loc[bids, 'FG'].to_list()
        efgbgs = epoch_df.loc[bids, 'BG + FG'].to_list()
        epoch_list = [efgbgs, efgs, ebgs]
        rfgbg = np.stack([resp.extract_epoch(e) for e in efgbgs], axis=0)#.mean(axis=1, keepdims=True)
        rfg = np.stack([resp.extract_epoch(e) for e in efgs], axis=0)#.mean(axis=1, keepdims=True)
        rbg = np.stack([resp.extract_epoch(e) for e in ebgs], axis=0)#.mean(axis=1, keepdims=True)

        norm=True
        if norm:
            m = np.mean(rfgbg, axis=(0,1,3), keepdims=True)
            s = np.std(rfgbg, axis=(0,1,3), keepdims=True)
            s=(s+(s==0))
            norm_fgbg = (rfgbg - m) / s
            norm_fg = (rfg - m) / s
            norm_bg = (rbg - m) / s
        else:
            norm_fgbg=rfgbg
            norm_fg=rfg
            norm_bg=rbg

        shuffle_count=11
        ecount = rfgbg.shape[0]
        paircount = int(ecount*(ecount-1)/2)
        pred = np.zeros((shuffle_count,cluster_count, int(ecount*(ecount-1)/2), norm_fgbg.shape[1] * 2))
        paircc = np.zeros((shuffle_count, cluster_count, int(ecount*(ecount-1)/2), 3))
        s1=[[]] * paircount
        s2=[[]] * paircount
        for shuffidx in range(shuffle_count):
            if shuffidx == 0:
                idx_cluster_map = idx_to_cluster_array
            else:
                idx_cluster_map = np.random.permutation(idx_to_cluster_array)
            #idx_cluster_map = np.random.permutation(idx_to_cluster_array)
            cmin, cmax = idx_cluster_map.min(), idx_cluster_map.max()

            #cluster_sets = [[c] for c in range(cmin, cmax + 1)] \
            #               + [[c for c in range(cmin, cmax + 1)]]
            cluster_sets = [[c] for c in range(cmin, cmax + 1)]

            cluster_treatment='exclude'
            #cluster_treatment='include'
            if cluster_treatment=='exclude':
                # leave out one cluster (instead of just fitting for that one cluster)
                call = np.arange(cmin,cmax+1)
                cluster_sets = [np.setdiff1d(call,c) for c in cluster_sets]
                cluster_sets[-1] = call

            fgsim = np.zeros((cluster_count+1,ecount,ecount))
            bgsim = np.zeros((cluster_count+1,ecount,ecount))
            ffsim = np.zeros((cluster_count+1,ecount,ecount))
            bbsim = np.zeros((cluster_count+1,ecount,ecount))
            for fidx, cluster_ids in enumerate(cluster_sets):
                keepids = [i for i in range(len(idx_cluster_map)) if idx_cluster_map[i] in cluster_ids]
                #print(cluster_ids, keepids)
                ec = 0
                for jj in range(ecount):
                    for ii in range(0,jj):
                        s1[ec]=efgbgs[jj]
                        s2[ec]=efgbgs[ii]
                        x1 = smooth(norm_fg[ii,:,keepids,25:],window_len=3,axis=2)[:,:,2::5]
                        x2 = smooth(norm_fg[jj,:,keepids,25:],window_len=3,axis=2)[:,:,2::5]
                        #x1 = norm_fgbg[ii,:,keepids,25:75]#.mean(axis=2,keepdims=True)
                        #x2 = norm_fgbg[jj,:,keepids,25:75]#.mean(axis=2,keepdims=True)
                        X = np.concatenate([x1,x2], axis=1)
                        X = np.transpose(X,[1,0,2])
                        X = np.reshape(X,[X.shape[0], -1])
                        Y = np.concatenate([np.zeros(x1.shape[1]),
                                            np.ones(x2.shape[1])])
                        x1t = smooth(norm_fgbg[ii,:,keepids,25:],window_len=3,axis=2)[:,:,2::5]
                        x2t = smooth(norm_fgbg[jj,:,keepids,25:],window_len=3,axis=2)[:,:,2::5]
                        Xt = np.concatenate([x1t,x2t], axis=1)
                        Xt = np.transpose(Xt,[1,0,2])
                        Xt = np.reshape(Xt,[Xt.shape[0], -1])
                        Yt = np.concatenate([np.zeros(x1t.shape[1]),np.ones(x2t.shape[1])])
                        clf = LogisticRegression(random_state=0) #, max_iter=500, tol=1e-3)
                        #clf = RidgeClassifier(random_state=0, solver='svd')
                        clf.fit(X, Y)
                        pred[shuffidx, fidx, ec, :] = clf.predict_proba(Xt)[:, 0]

                        E = np.zeros((Xt.shape[0], Xt.shape[0])) * np.nan
                        for aa in range(Xt.shape[0]):
                            for bb in range(Xt.shape[0]):
                                if (aa!=bb) & (Xt[aa].std()>0) & (Xt[bb].std()>0):
                                    E[aa,bb] = np.corrcoef(Xt[aa], Xt[bb])[0,1]
                        #np.fill_diagonal(E, np.nan)

                        nt = x1t.shape[1]
                        paircc[shuffidx,fidx,ec,0] = np.nanmean(E[:nt,:nt]) # within fgA
                        paircc[shuffidx,fidx,ec,1] = np.nanmean(E[nt:,nt:]) # within fgB
                        paircc[shuffidx,fidx,ec,2] = (np.nanmean(E[:nt,:nt]) + np.nanmean(E[nt:,nt:]))/2 - \
                                                     np.nanmean(E[nt:,:nt]) # between fgA-fgB
                        #N = X.shape[0]
                        #for ex in range(N):
                        #    clf = LogisticRegression(random_state=0) #, max_iter=500, tol=1e-3)
                        #    #clf = RidgeClassifier(random_state=0, solver='svd')
                        #    fitidx = np.ones(N, dtype=bool)
                        #    fitidx[ex] = 0
                        #    clf.fit(X[fitidx], Y[fitidx])
                        #    #pred[shuffidx, fidx, ec, ex] = clf._predict_proba_lr(X[[ex]])[0, 0]
                        #    pred[shuffidx, fidx, ec, ex] = clf.predict_proba(X[[ex]])[0, 0]
                        ec += 1

        mpred = (np.mean(pred[:,:,:,:norm_fgbg.shape[1]], axis=3) + 1 -\
                np.mean(pred[:,:,  :, norm_fgbg.shape[1]:], axis=3))/2
        #mpred = np.concatenate((mpred[1:].max(axis=0,keepdims=True),mpred), axis=0)

        for cidx in range(cluster_count):
            if cidx<cluster_count:
                mcc=mean_cc_set[estim][cidx,0]
                mn=cluster_n[cidx]
            else:
                mcc=0
                mn=resp.shape[0]
            mp = mpred[0,cidx,:]
            mpmax = mpred[1:,cidx,:].max(axis=0)
            mpmin = mpred[1:,cidx,:].min(axis=0)
            pairccmax = paircc[1:,cidx,:].max(axis=0)
            pairccmin = paircc[1:,cidx,:].min(axis=0)
            if cluster_treatment=='exclude':
                sizes = weights.shape[0]-cluster_n
            else:
                sizes = cluster_n

            d=pd.DataFrame(
                {'siteid': siteid, 's1': s1, 's2': s2, 'cidx': cidx,
                 'mean_cc': mcc, 'cluster_n': sizes[cidx],
                 'mp': mp, 'mpmin': mpmin, 'mpmax': mpmax,
                 's1cc': paircc[0,cidx,:,0], 's1ccmin': pairccmin[:,0], 's1ccmax': pairccmax[:,0],
                 's2cc': paircc[0, cidx, :,1], 's2ccmin': pairccmin[:,1], 's2ccmax': pairccmax[:,1],
                 's1s2cc': paircc[0, cidx, :, 2], 's1s2ccmin': pairccmin[:, 2], 's1s2ccmax': pairccmax[:, 2],

                 })
            dfdiscrims.append(d)

        #f, ax = plt.subplots(1, cluster_count)
        #for i, a in enumerate(ax):
        #    a.imshow(mpred[:, i, :])
        #f.suptitle(f"{siteid} {ebg}")

        """
        t1 = np.reshape(norm_fgbg[ii,:,keepids], (norm_fgbg.shape[1],-1))
                    t2 = np.reshape(norm_fg[jj,:,keepids], (norm_fg.shape[1],-1))
                    t2a = np.reshape(norm_fg[ii,:,keepids], (norm_fg.shape[1],-1))
                    t3 = np.reshape(norm_bg[jj,:,keepids], (norm_bg.shape[1],-1))
                    t3a = np.reshape(norm_bg[ii,:,keepids], (norm_bg.shape[1],-1))
                    cc_ = t1 @ t2.T / len(keepids)**2
                    fgsim[fidx,ii,jj]=cc_.mean()
                    cc_ = t1 @ t3.T / len(keepids)**2
                    bgsim[fidx,ii,jj]=cc_.mean()
                    cc_ = t2a @ t2.T / len(keepids)**2
                    #np.fill_diagonal(cc_,np.nan)
                    ffsim[fidx,ii,jj]=np.nanmean(cc_)
                    cc_ = t3a @ t3.T / len(keepids)**2
                    #np.fill_diagonal(cc_,np.nan)
                    bbsim[fidx,ii,jj]=np.nanmean(cc_)
                    fgsim[fidx,ii,jj]=np.corrcoef(t1.flatten(), t2.flatten())[0,1]
                    bgsim[fidx,ii,jj]=np.corrcoef(t1.flatten(), t3.flatten())[0,1]
                    ffsim[fidx,ii,jj]=np.corrcoef(t2a.flatten(), t2.flatten())[0,1]
                    bbsim[fidx,ii,jj]=np.corrcoef(t3a.flatten(), t3.flatten())[0,1]
                fgsim[fidx,:,jj] /= fgsim[fidx,jj,jj]
                bgsim[fidx, :, jj] /= bgsim[fidx, jj,jj]
        f,ax = plt.subplots(1,4)
        ax[0].imshow(np.concatenate([fgsim[i] for i in range(cluster_count+1)], axis=0))
        ax[1].imshow(np.concatenate([bgsim[i] for i in range(cluster_count + 1)], axis=0))
        ax[2].imshow(np.concatenate([ffsim[i] for i in range(cluster_count + 1)], axis=0))
        ax[3].imshow(np.concatenate([bbsim[i] for i in range(cluster_count + 1)], axis=0))
        f.suptitle(f"{siteid} {ebg}")
        """

if len(siteids)==1:
    raise ValueError("stopping")

df = pd.concat(dfs, ignore_index=True)
df = df.loc[(df['cluster_n']>4) &
             np.isfinite(df['cc_fg']) & np.isfinite(df['cc_bg']) &
             (df['mw_fg']+df['mw_bg']>0.2) &
             (df['mw_fg']>0) & (df['mw_bg']>0) &
             (df['mw_fg']<1.0) & (df['mw_bg']<1.0)]
df['relgain'] = df['mw_fg']-df['mw_bg']
df['dcc'] = df['cc_fg']-df['cc_bg']

ci = df['Binaural Type']=='BG Ipsi, FG Contra'
cc = df['Binaural Type']=='BG Contra, FG Contra'

f, ax = plt.subplots(2, 2, figsize=(3,3), sharex='col', sharey='col')
sns.scatterplot(df.loc[cc], x='mw_fg', y='mw_bg', ax=ax[0,0], s=3)
ax[0,0].plot([0,1], [0,1], 'k--', lw=0.5)
ax[0,0].set_title(f"CC wfg={df.loc[cc,'mw_fg'].median():.3f} wbg={df.loc[cc,'mw_bg'].median():.3f}")
ax[0,0].set_xlabel('FG weight')
ax[0,0].set_ylabel('BG weight')
sns.scatterplot(df.loc[ci], x='mw_fg', y='mw_bg', ax=ax[1,0], s=3)
ax[1,0].plot([0,1], [0,1], 'k--', lw=0.5)
ax[1,0].set_title(f"CI wfg={df.loc[ci,'mw_fg'].median():.3f} wbg={df.loc[ci,'mw_bg'].median():.3f}")
ax[1,0].set_xlabel('FG weight')
ax[1,0].set_ylabel('BG weight')

#sns.scatterplot(df, x='cc_fg', y='cc_bg', ax=ax[0,1], s=5)
#ax[0,1].plot([0,1],[0,1],'k--', lw=0.5)

sns.regplot(df.loc[cc], x='mean_sc_fgbg', y='relgain',
            fit_reg=True, ax=ax[0,1], scatter_kws={'s': 1})
ax[0,1].set_title(f"CC r={np.corrcoef(df.loc[cc,'mean_sc_fgbg'], df.loc[cc,'relgain'])[0,1]:.3}")
ax[0,1].set_xlabel('Cluster correlation')
ax[0,1].set_ylabel('Mean rel FG gain')
sns.regplot(df.loc[ci], x='mean_sc_fgbg', y='relgain',
            fit_reg=True, ax=ax[1,1], scatter_kws={'s': 1})
ax[1,1].set_title(f"CI r={np.corrcoef(df.loc[ci,'mean_sc_fgbg'], df.loc[ci,'relgain'])[0,1]:.3}")
ax[1,1].set_xlabel('Cluster correlation')
ax[1,1].set_ylabel('Mean rel FG gain')

f.suptitle(f"groupby {groupby} - batch {batch} - clusters {cluster_count}")
plt.tight_layout()

if len(siteids)>1:
    outfile = f"{outpath}site_corrs_{batch}_{groupby}_{cluster_count0}_relgain.pdf"
    print(f"saving summ fit to {outfile}")
    f.savefig(outfile)



dfd = pd.concat(dfdiscrims, ignore_index=True)
dfd['mpdiff']=(dfd['mp']-dfd['mpmin'])/(dfd['mpmax']-dfd['mpmin'])

dfd['ccdiff']=(dfd['s1s2cc']-dfd['s1s2ccmin'])/(dfd['s1s2ccmax']-dfd['s1s2ccmin'])

dfd['ccnorm']=np.sqrt(dfd['s1cc']*dfd['s2cc'])
dfd['ccnormmin']=np.sqrt(dfd['s1ccmin']*dfd['s2ccmin'])
dfd['ccnormmax']=np.sqrt(dfd['s1ccmax']*dfd['s2ccmax'])
dfd['ccdiff3']=(dfd['ccnorm']-dfd['ccnormmin'])/(dfd['ccnormmax']-dfd['ccnormmin'])

dfd=dfd.merge(df[['siteid','cid','estim','relgain']].loc[df['cid']==0],
              how='left',left_on=['siteid','s1'], right_on=['siteid','estim'])
dfd=dfd.merge(df[['siteid','cid','estim','relgain']].loc[df['cid']==0],
              how='left',left_on=['siteid','s2'], right_on=['siteid','estim'], suffixes=('_1','_2'))
dfd['mrelgain']=(dfd['relgain_1']+dfd['relgain_2'])/2

dfd['CI'] = dfd['s1'].str.contains('0-1-2_')
if dfd['CI'].sum()>0:
    dfd['CC'] = dfd['s1'].str.contains('0-1-1_')
else:
    dfd['CC']=True

includeidx = (dfd['cidx']<=cluster_count) & (dfd['cluster_n']>12) &\
             (np.isfinite(dfd['ccdiff'])) & (np.isfinite(dfd['mrelgain'])) & \
             (np.abs(dfd['ccdiff'])<3) & (np.abs(dfd['mrelgain'])<3)

f=plt.figure()
ax=f.add_subplot(3, 1, 1)
dfd[['s1s2ccmin', 's1s2ccmax']].plot(lw=0.5, ax=ax)
dfd[['s1s2cc']].plot(color='red', ax=ax)
dfd[['mean_cc']].plot(ls='--', lw=0.5, ax=ax)

cc=includeidx & dfd['CC']
ci=includeidx & dfd['CI']

ax=f.add_subplot(3, 3, 4)
sns.regplot(dfd.loc[cc], x='mean_cc', y='ccdiff',
            fit_reg=True, ax=ax, scatter_kws={'s': 3})
ax.set_title(f"CC: r={np.corrcoef(dfd.loc[cc,'mean_cc'], dfd.loc[cc,'ccdiff'])[0,1]:.3}")

ax=f.add_subplot(3, 3, 5)
sns.regplot(dfd.loc[cc], x='mean_cc', y='s1s2cc',
            fit_reg=True, ax=ax, scatter_kws={'s': 3})
ax.set_title(f"normCC: r={np.corrcoef(dfd.loc[cc,'mean_cc'], dfd.loc[cc,'s1s2cc'])[0,1]:.3}")

ax=f.add_subplot(3, 3, 6)
sns.regplot(dfd.loc[cc], x='mrelgain', y='s1s2cc',
            fit_reg=True, ax=ax, scatter_kws={'s': 3})
ax.set_title(f"CC: r={np.corrcoef(dfd.loc[cc,'mrelgain'], dfd.loc[cc,'s1s2cc'])[0,1]:.3}")

ax=f.add_subplot(3, 3, 7)
sns.regplot(dfd.loc[ci], x='mean_cc', y='ccdiff',
            fit_reg=True, ax=ax, scatter_kws={'s': 3})
ax.set_title(f"CC: r={np.corrcoef(dfd.loc[ci,'mean_cc'], dfd.loc[ci,'ccdiff'])[0,1]:.3}")

ax=f.add_subplot(3, 3, 8)
sns.regplot(dfd.loc[ci], x='mean_cc', y='s1s2cc',
            fit_reg=True, ax=ax, scatter_kws={'s': 3})
ax.set_title(f"normCC: r={np.corrcoef(dfd.loc[ci,'mean_cc'], dfd.loc[ci,'s1s2cc'])[0,1]:.3}")

ax=f.add_subplot(3, 3, 9)
sns.regplot(dfd.loc[ci], x='mrelgain', y='s1s2cc',
            fit_reg=True, ax=ax, scatter_kws={'s': 3})
ax.set_title(f"CC: r={np.corrcoef(dfd.loc[ci,'mrelgain'], dfd.loc[ci,'s1s2cc'])[0,1]:.3}")

plt.tight_layout()


hi = np.sum(dfd['s1s2cc']>dfd['s1s2ccmax'])
lo = np.sum(dfd['s1s2cc']<dfd['s1s2ccmin'])
T=dfd.shape[0]
mid= T-hi-lo
print(f'cc lo {lo} {lo/T:.2f}/mid {mid}/hi {hi} {hi/T:.2f}' )

hi = np.sum(dfd['mp']>dfd['mpmax'])
lo = np.sum(dfd['mp']<dfd['mpmin'])
T=dfd.shape[0]
mid= T-hi-lo
print(f'decode lo {lo} {lo/T:.2f}/mid {mid}/hi {hi} {hi/T:.2f}' )
if len(siteids)>1:
    outfile = f"{outpath}site_corrs_{batch}_{groupby}_{cluster_count0}_decode.pdf"
    print(f"saving summ fit to {outfile}")
    f.savefig(outfile)

