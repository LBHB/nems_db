import os
import io
import logging
import time
import sys, importlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch
import json
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
from nems_lbhb.projects.nat_streaming.recon_tools import nmse, corrcoef, recon_site_stim, get_cluster_data

log = logging.getLogger(__name__)

cluster_count = 3
use_dlog=True
batch = 341
modeltype = 'LN'
cluster_treatment = "exclude"
reverse_sites = True

if os.uname()[1]=='agouti':
    #groupby = 'all'
    groupby = 'bg'
    cluster_count = 3
    #reverse_sites=True
elif os.uname()[1] == 'manatee':
    batch=345
    groupby = 'bg'
    cluster_count = 3
    reverse_sites = False
elif os.uname()[1] == 'hyena':
    groupby = 'bg'
    cluster_count = 3
    reverse_sites = False
elif os.uname()[1] == 'capybara':
    #groupby = 'fgbg'
    batch=345
    groupby = 'bg'
    cluster_count = 3
else:
    raise ValueError("Only runs on agouti, capybara, manatee")

if use_dlog:
    dlstr = ""
else:
    dlstr = "_nocomp"
if cluster_count > 3:
    cluster_str = f"_c{cluster_count}"
else:
    cluster_str = ""

shuffle_count=11


loadkey = "gtgram.fs50.ch18"
if len(sys.argv)>1:
    siteids=[sys.argv[1]]
else:
    siteids, cellids = db.get_batch_sites(batch)
    if batch==341:
        #siteids.remove('SLJ008a')
        siteids.remove('PRN004a')
        siteids.remove('PRN031a')
        siteids.remove('PRN042a')
        siteids.remove('PRN045b')
        siteids.remove('CLT052d')

    if reverse_sites:
        # start at one end of the site list
        siteids.sort(reverse=True)
    else:
        # or at the other
        siteids.sort()

    #siteid = siteids[0]

for siteid in siteids:
    plt.close('all')

    if len(sys.argv)>2:
        estim=sys.argv[2]
        unique_group=[estim]
    else:
        uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
        rec = load_recording(uri)

        epoch_df_all = OLP_get_epochs.get_rec_epochs(fs=50, rec=rec)
        epoch_df = epoch_df_all.loc[(epoch_df_all['Dynamic Type'] == 'fullBG/fullFG')]
        epoch_df = epoch_df.loc[(epoch_df['SNR'] == 0)]
        # epoch_df = epoch_df.loc[(epoch_df['SNR']==10)]
        epoch_df = epoch_df.loc[(epoch_df['Binaural Type'] == 'BG Contra, FG Contra') |
                                (epoch_df['Binaural Type'] == 'BG Ipsi, FG Contra')]
        # epoch_df = epoch_df_all.loc[(epoch_df_all['Binaural Type'] == 'BG Ipsi, FG Contra')]
        if (epoch_df['Synth Type'] == 'Unsynthetic').sum() > 0:
            epoch_df = epoch_df.loc[(epoch_df['Synth Type'] == 'Unsynthetic')]
        else:
            epoch_df = epoch_df.loc[(epoch_df['Synth Type'] == 'Non-RMS Unsynthetic')]
        epoch_df = epoch_df.reset_index()

        print(f"{siteid}: {epoch_df.shape}")

        if groupby == 'fg':
            unique_group = list(epoch_df['FG'].unique())
        elif groupby == 'bg':
            unique_group = list(epoch_df['BG'].unique())
        elif groupby == 'fgbg':
            unique_group = list(epoch_df['BG + FG'].unique())
        elif groupby == 'all':
            unique_group = ['STIM_']
        else:
            raise ValueError("groupby must be fg or bg")

    for estim in unique_group:
        df_recon = recon_site_stim(siteid, estim, cluster_count=cluster_count,
                        batch=batch, modeltype=modeltype, groupby=groupby,
                        shuffle_count=shuffle_count, force_rerun=False)

"""
if batch==341:
    siteids.remove('PRN004a')
    siteids.remove('PRN031a')
    siteids.remove('PRN042a')
    siteids.remove('PRN045b')
    siteids.remove('CLT052d')

for siteid in siteids:
    sql=f"SELECT * FROM aRecon WHERE siteid='{siteid}'" +\
        f" AND groupby='{groupby}' AND shuffidx={shuffle_count}" +\
        f" AND batch={batch} AND cluster_count={cluster_count}"
    dtest=db.pd_query(sql)

    try:
        df_recon = pd.read_csv(outpath / 'df_recon.csv', index_col=0)
        if ((df_recon['siteid']==siteid) & (df_recon['shuffidx']==shuffle_count-1)).sum()>0:
            print(f"{siteid} shuffidx>=10 already analyzed. Skipping")
            continue
        df_recon = df_recon.loc[df_recon.siteid!=siteid].copy()
        df_recons=[df_recon]
    except:
        df_recons = []
    print(f"Starting siteid {siteid}")

    uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
    rec = load_recording(uri)
    try:
        d=get_depth_info(siteid=siteid).reset_index()
        a1chans=[r['index'] for i,r in d.iterrows() if r.area in ['A1', 'PEG','BRD']]
        keepchans = [c for c in a1chans if c in rec['resp'].chans]
    except:
        keepchans = [c for c in rec['resp'].chans]
    rec['resp'] = rec['resp'].rasterize().extract_channels(keepchans)
    rec['stim'] = rec['stim'].rasterize()

    # log compress and normalize stim
    if use_dlog:
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

    print(f"{siteid}: {epoch_df.shape}")
    #continue

    resp = rec['resp']
    stim = rec['stim']

    skip_recon = False
    if groupby=='fg':
        unique_group = list(epoch_df['FG'].unique())
    elif groupby == 'bg':
        unique_group = list(epoch_df['BG'].unique())
    elif groupby == 'fgbg':
        unique_group = list(epoch_df['BG + FG'].unique())
    elif groupby=='all':
        unique_group = ['STIM_']
    else:
        raise ValueError("groupby must be fg or bg")
    break_list=[]
    for estim in unique_group:
        if groupby=='bg':
            print(f"Focusing on bg={estim}")
            bids = (epoch_df['BG']==estim)
            estim_label = estim.split("_")[1]
        elif groupby == 'fg':
            print(f"Focusing on fg={estim}")
            bids = (epoch_df['FG'] == estim)
            estim_label = estim.split("_")[2]
        elif groupby == 'fgbg':
            print(f"Focusing on epoch={estim}")
            bids = (epoch_df['BG + FG'] == estim)
            estim_label = estim.replace("STIM_","")
        else:
            print(f"Single clustering for all stim")
            bids = (epoch_df['BG + FG'].str.startswith(estim))
            estim_label = 'ALL'


        ebgs = epoch_df.loc[bids, 'BG'].to_list()
        efgs = epoch_df.loc[bids, 'FG'].to_list()
        efgbgs = epoch_df.loc[bids, 'BG + FG'].to_list()
        epoch_list = [efgbgs, efgs, ebgs]
        rfgbg = np.concatenate([resp.extract_epoch(e).mean(axis=0,keepdims=True) for e in efgbgs],axis=2).mean(axis=0)
        rfg = np.concatenate([resp.extract_epoch(e).mean(axis=0,keepdims=True) for e in efgs],axis=2).mean(axis=0)
        rbg = np.concatenate([resp.extract_epoch(e).mean(axis=0,keepdims=True) for e in ebgs],axis=2).mean(axis=0)

        # norm_psth = psth - np.mean(psth[:,:25], axis=1, keepdims=True)
        r=rfgbg
        #r=np.concatenate([rfgbg, rbg], axis=1)
        #r=np.concatenate([rbg], axis=1)
        norm_psth = r - np.mean(r, axis=1, keepdims=True)
        s = np.std(norm_psth, axis=1, keepdims=True)
        norm_psth /= (s + (s == 0))
        sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

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
            raster = np.concatenate([resp.extract_epoch(e).mean(axis=0,keepdims=True) for e in epochs], axis=2)
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
        f1.suptitle(f"{siteid} {estim}: Signal correlation")
        plt.tight_layout()
        f1.savefig(outpath / f"{siteid}_{estim}_clusters.jpg")
        print(f"{estim}: {cluster_n}")

        if skip_recon:
            continue

        #raise ValueError('stopping')

        #
        # Fit decoder with different clusters of neurons
        #
        for shuffidx in range(shuffle_count):
            print(f"{estim}: decoder fitting shuffle {shuffidx}")
            models = []
            resps = []
            cellcount = rec['resp'].shape[0]
            if shuffidx == 0:
                idx_cluster_map = idx_to_cluster_array
            else:
                idx_cluster_map = np.random.permutation(idx_to_cluster_array)
            cmin, cmax = idx_cluster_map.min(), idx_cluster_map.max()

            min_cluster_units = 0
            if min_cluster_units>0:
                cluster_sets = [[c] for c in range(cmin, cmax+1)
                                if cluster_n[c-1]>min_cluster_units] \
                               + [[c for c in range(cmin, cmax+1)]]
                cc_diff = [c for i,c in enumerate(cc_diff)
                           if cluster_n[i]>min_cluster_units]
                cluster_cc = [c for i,c in enumerate(cluster_cc)
                           if cluster_n[i]>min_cluster_units]
            if shuffidx==0:
                break_list.append(len(cluster_sets))
            if cluster_treatment=='exclude':
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
                    model = LN.CNN_reconstruction(time_bins=17, channels=bnt_resp.shape[2],
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
                if col == 0:
                    ax[0, col].set_ylabel('Freq')
                    vmax = spec.max()
                ax[0, col].imshow(spec.T**2, aspect='auto', cmap='gray_r', interpolation='none',
                                  origin='lower', extent=[-0.5, 1.5, 1, spec.shape[1]], vmin=0, vmax=vmax)
                ax[0, col].set_title(epochs[0].replace("STIM_", ""), fontsize=8)

            for fidx, cluster_ids in enumerate(cluster_sets):
                model = models[fidx]
                resp2 = resps[fidx]
                if fidx<len(cluster_sets)-1:
                    cid=fidx
                else:
                    cid=cluster_count
                d={'siteid': siteid, 'estim': estim, 'cid': cid}
                d['shuffidx'] = shuffidx
                d['cluster_ids'] = ",".join([str(c) for c in cluster_ids])
                d['n_units'] = resp2.shape[0]
                d['total_units'] = resp.shape[0]
                if fidx>=len(cluster_cc):
                    d['cluster_cc']=0
                    d['cc_diff']=0
                else:
                    d['cluster_cc']=cluster_cc[fidx]
                    d['cc_diff']=cc_diff[fidx]
                spec_fg = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in efgs], axis=1).T
                spec_bg = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in ebgs], axis=1).T
                spec_fgbg = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in efgbgs], axis=1).T
                for col, epochs in enumerate(epoch_list):
                    raster = np.concatenate([resp2.extract_epoch(e).mean(axis=0,keepdims=True) for e in epochs], axis=2)
                    psth = np.mean(raster, axis=0).T

                    #raster0 = np.concatenate([resps[-1].extract_epoch(e) for e in epochs], axis=2)
                    #psth0 = np.mean(raster0, axis=0).T
                    #recon0 = models[-1].predict(psth0)

                    spec = np.concatenate([stim.extract_epoch(e)[0, :, :] for e in epochs], axis=1).T

                    recon = model.predict(psth)
                    #recon -= recon0

                    ax[fidx+1, col].imshow(recon.T**2, aspect='auto', cmap='gray_r', interpolation='none',
                                      origin='lower', extent=[-0.5, 1.5, 1, recon.shape[1]], vmin=0, vmax=vmax)
                    if col == 0:
                        recon0 = recon
                        d['Efg'] = nmse(recon, spec_fg)  # np.corrcoef(recon.flatten(), spec_fg.flatten())[0,1] #
                        d['Ebg'] = nmse(recon, spec_bg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
                        d['Efgbg'] = nmse(recon, spec_fgbg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
                        d['Cfg'] = corrcoef(recon, spec_fg)  # np.corrcoef(recon.flatten(), spec_fg.flatten())[0,1] #
                        d['Cbg'] = corrcoef(recon, spec_bg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
                        d['Cfgbg'] = corrcoef(recon, spec_fgbg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #

                        # compute SNR of fg in reconstruction
                        from scipy.stats import linregress
                        from sklearn import linear_model
                        X = np.stack([np.ones_like(spec_fg.flatten()), spec_fg.flatten(), spec_bg.flatten()], axis=1)**2
                        Y = spec_fgbg.flatten()[:,np.newaxis]**2
                        Yest = recon.flatten()[:,np.newaxis]**2

                        m = linear_model.LinearRegression()
                        m.fit(X,Y)
                        m2 = linear_model.LinearRegression()
                        m2.fit(X,Yest)
                        d['wFGact']=m.coef_[0,1]
                        d['wBGact']=m.coef_[0,2]
                        d['wFG']=m2.coef_[0,1]
                        d['wBG']=m2.coef_[0,2]

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
                        if col == 1:
                            d['Erfg'] = Er
                            d['Crfg'] = Cr
                        else:
                            d['Erbg'] = Er
                            d['Crbg'] = Cr
                df_recons.append(pd.DataFrame(d, index=[0]))

            f2.suptitle(f'{estim} shuffidx={shuffidx} treatment={cluster_treatment}')
            plt.tight_layout()
            if shuffidx==0:
                f2.savefig(outpath / f"{siteid}_{estim}_recon_shuff{shuffidx}.jpg")

            plt.close(f2)

    df_recon = pd.concat(df_recons, ignore_index=True)
    df_recon.to_csv(outpath / 'df_recon.csv')
    for i,r in df_recon.iloc[:10].iterrows():
        res = r[['cluster_ids', 'n_units',
                'cluster_cc', 'Efg', 'Ebg', 'Efgbg', 'Cfg', 'Cbg', 'Cfgbg', 'Erfg',
                'Crfg', 'Erbg', 'Crbg']].to_dict()
        res = json.dumps(res)
        sql=f"INSERT INTO aRecon (siteid, estim, cid, shuffidx, batch, groupby, cluster_count, results)" +\
            f" VALUES ('{siteid}','{estim}',{cid},{shuffidx},{batch},'{groupby}',{cluster_count}, '{res}')"
        db.sql_command(sql)

    d=db.pd_query("SELECT * FROM aRecon")

    d = df_recon.loc[df_recon.siteid==siteid].copy()
    d['fg_rat'] = d['Efg']# /d['Efgbg']
    d['bg_rat'] = d['Ebg'] #/d['Efgbg']
    d['fg_crat'] = d['Cfg'] #/d['Cfgbg']
    d['bg_crat'] = d['Cbg'] #/d['Cfgbg']

    p=d[['estim', 'cid', 'shuffidx', 'fg_rat', 'bg_rat', 'fg_crat', 'bg_crat']].groupby(['estim', 'cid', 'shuffidx']).mean()
    p=p.reset_index(2)
    f3,ax=plt.subplots(2,1)
    ax[0].plot(p.loc[p.shuffidx == 0, ['fg_rat','bg_rat']].values)
    for sh in range(1,shuffle_count):
        ax[0].plot(p.loc[p.shuffidx == sh, ['fg_rat']].values,
                   lw=0.5, color='lightblue')
        ax[0].plot(p.loc[p.shuffidx == sh, ['bg_rat']].values,
                   lw=0.5, color='orange')
    breaks = np.cumsum(np.array(break_list))-1
    #breaks = np.arange(cluster_count,(p['shuffidx']==0).sum(), cluster_count+1)
    yl = ax[0].get_ylim()
    ax[0].vlines(breaks, yl[0], yl[1], ls='--', color='gray')
    for i, e in enumerate(unique_group):
        if groupby=='bg':
            e_ = e.split("_")[1]
        elif groupby == 'fg':
            e_ = e.split("_")[2]
        elif groupby == 'all':
            e_ = e.split("_")[-1]
        else:
            e_ = e.split("_")[1] + "\n" + e.split("_")[2]
        ax[0].text(breaks[i], yl[1], e_, ha='right', va='top', fontsize=6)

    ax[0].set_ylabel('MSE')

    ax[1].plot(p.loc[p.shuffidx == 0, ['fg_crat', 'bg_crat']].values)
    for sh in range(1, shuffle_count+1):
        ax[1].plot(p.loc[p.shuffidx == sh, ['fg_crat']].values,
                   lw=0.5, color='lightblue')
        ax[1].plot(p.loc[p.shuffidx == sh, ['bg_crat']].values,
                   lw=0.5, color='orange')
    yl = ax[1].get_ylim()
    ax[1].vlines(breaks, yl[0], yl[1], ls='--', color='gray')
    ax[1].legend(('FG ratio','BG ratio'))
    ax[1].set_ylabel('CC')
    f3.suptitle(siteid)

    f3.savefig(outpath / f"{siteid}_summary_clusters.jpg")
    plt.close(f3)
    plt.close(f1)
"""
