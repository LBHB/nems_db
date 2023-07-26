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
from nems.models.LN import LN_reconstruction

log = logging.getLogger(__name__)

def cluster_corr(corr_array, inplace=False, return_indices=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to each other

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 1.75
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    if return_indices:
        return idx, idx_to_cluster_array

    return corr_array[idx, :][:, idx]

loadkey = "psth.fs50"
batch=345
siteids,cellids = db.get_batch_sites(batch)

siteids = [s for s in siteids if s.startswith("PRN")]
recs = []
epoch_dfs = []

for siteid in siteids:

    uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
    rec = load_recording(uri)

    d = get_depth_info(siteid=siteid).reset_index()
    acchans = [r['index'] for i, r in d.iterrows() if (r.area=='A1') | (r.area=='PEG')]
    keepchans = [c for c in acchans if c in rec['resp'].chans]
    rec['resp'] = rec['resp'].rasterize().extract_channels(keepchans)

    epoch_df_all = OLP_get_epochs.get_rec_epochs(fs=50, rec=rec)
    epoch_df = epoch_df_all.loc[(epoch_df_all['Binaural Type'] == 'BG Contra, FG Contra')]
    #epoch_df = epoch_df_all.loc[(epoch_df_all['Binaural Type'] == 'BG Ipsi, FG Contra')]
    epoch_df = epoch_df.loc[(epoch_df['Dynamic Type'] == 'fullBG/fullFG')]
    epoch_df = epoch_df.loc[(epoch_df['SNR'] == 0)]
    # epoch_df = epoch_df.loc[(epoch_df['SNR']==10)]
    epoch_df = epoch_df.reset_index()
    epoch_df['siteid'] = siteid
    epoch_dfs.append(epoch_df)
    recs.append(rec)

epoch_df = pd.concat(epoch_dfs, ignore_index=True)

counts = epoch_df.groupby(['BG + FG']).count()[['siteid']].reset_index()

gg = (counts['siteid']>1) & \
   ((counts['BG + FG'].str.contains('Kit')) |
   (counts['BG + FG'].str.contains('Fight')))
print(counts.loc[gg])
elist = counts.loc[gg, 'BG + FG'].to_list()

#plt.close('all')
for e in elist:
    use_siteids = epoch_df.loc[epoch_df['BG + FG']==e, 'siteid'].to_list()
    sids = [i for i,s in enumerate(siteids) if s in use_siteids]

    epoch_id = epoch_df.loc[epoch_df['BG + FG']==e].index.to_list()[0]
    efg=epoch_df.loc[epoch_id,'FG']
    ebg=epoch_df.loc[epoch_id,'BG']
    efgbg=epoch_df.loc[epoch_id,'BG + FG']
    epoch_list = [efgbg, efg, ebg ]
    epoch=epoch_list[0]

    # cluster based on BG+FG responses
    psth = np.concatenate(
        [recs[sid]['resp'].extract_epoch(efgbg).mean(axis=0) for sid in sids],
        axis=0)

    # cluster based on separate BG, FG responses
    #fpsth = np.concatenate(
    #    [recs[sid]['resp'].extract_epoch(efg).mean(axis=0) for sid in sids],
    #    axis=0)
    #bpsth = np.concatenate(
    #    [recs[sid]['resp'].extract_epoch(ebg).mean(axis=0) for sid in sids],
    #    axis=0)
    #psth = np.concatenate([fpsth,bpsth], axis=1)

    # norm_psth = psth - np.mean(psth[:,:25], axis=1, keepdims=True)
    norm_psth = psth - np.mean(psth, axis=1, keepdims=True)
    s = np.std(psth, axis=1, keepdims=True)
    norm_psth /= (s + (s == 0))
    sc = norm_psth @ norm_psth.T / norm_psth.shape[1]
    cc_idx, idx_to_cluster_array = cluster_corr(sc, return_indices=True)
    cluster_count=idx_to_cluster_array.max()

    f,ax = plt.subplots(3, 3, figsize=(6,4), sharex='row', sharey='row')

    for col,epoch in enumerate(epoch_list):
        psth = np.concatenate(
            [recs[sid]['resp'].extract_epoch(epoch).mean(axis=0) for sid in sids],
            axis=0)
        norm_psth = psth - np.mean(psth, axis=1, keepdims=True)
        s = np.std(psth, axis=1, keepdims=True)
        norm_psth /= (s + (s == 0))
        sc = norm_psth @ norm_psth.T / norm_psth.shape[1]
        sc_sorted=sc[cc_idx,:][:,cc_idx]

        cluster_psth = np.zeros((cluster_count,psth.shape[1]))
        cluster_n = np.zeros(cluster_count)
        for c in range(cluster_count):
            cluster_psth[c,:]=norm_psth[(idx_to_cluster_array==c+1),:].mean(axis=0)
            cluster_n[c]=(idx_to_cluster_array==c+1).sum()
        if col==0:
            cluster_psth0=cluster_psth
        cluster_cc=np.zeros(cluster_count)
        for c in range(cluster_count):
            cluster_cc[c]=np.corrcoef(cluster_psth[c],cluster_psth0[c])[0,1]
        if col==1:
            fg_cc=cluster_cc
        elif col==2:
            bg_cc=cluster_cc
        ax[1,col].imshow(sc_sorted, aspect='equal', cmap='gray_r', interpolation='none',
                         origin='lower', vmin=0, vmax=1)
        ax[1,col].vlines(np.cumsum(cluster_n)[:-1]-0.5, -0.5, sc_sorted.shape[1]-0.5, lw=0.5, color='r')
        ax[1,col].hlines(np.cumsum(cluster_n)[:-1]-0.5, -0.5, sc_sorted.shape[1]-0.5, lw=0.5, color='r')

        ax[2,col].imshow(cluster_psth, aspect='auto', interpolation='none',
                         origin='lower', vmin=0)
        if col>0:
            for c in range(cluster_count):
                ax[2,col].text(cluster_psth.shape[1],c,
                               f"{cluster_cc[c]:.2f}", va='center', fontsize=7)
        if col==2:
            cc_diff = fg_cc - bg_cc
            for c in range(cluster_count):
                if cc_diff[c]>0:
                    tcol='blue'
                else:
                    tcol='red'
                ax[2,0].text(cluster_psth.shape[1],c,
                             f"{cc_diff[c]:.2f} ({int(cluster_n[c])})",
                             va='center', fontsize=7,color=tcol)
        if col==0:
            ax[1,0].set_ylabel('Unit (sorted)')
            ax[2,0].set_ylabel('Cluster')

    f.suptitle(f"{efgbg}: {cluster_n}")
    plt.tight_layout()
    print(f"{efgbg} {epoch_id}:")
    print(f"cluster sizes: {cluster_n}")
    print(f"        sites: {use_siteids}")

#    bnt_epochs=ep.epoch_names_matching(recs[sid]['resp'].epochs, 'STIM_.*seq')
#    print(len(bnt_epochs))
