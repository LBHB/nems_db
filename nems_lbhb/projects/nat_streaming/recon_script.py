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
log = logging.getLogger(__name__)


import importlib
from nems.models import LN

importlib.reload(LN)


def get_cluster_data(rec, idx_to_cluster_array, cluster_ids):
    resp = rec['resp']
    stim = rec['stim']
    keep_cellids = [resp.chans[i] for i in range(len(idx_to_cluster_array)) if idx_to_cluster_array[i] in cluster_ids]

    resp2 = resp.extract_channels(keep_cellids)

    bnt_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_.*seq')

    bnt_stim = np.stack([stim.extract_epoch(e)[0].T for e in bnt_epochs], axis=0)
    bnt_resp = np.stack([resp2.extract_epoch(e).mean(axis=0).T for e in bnt_epochs], axis=0)

    print("Resp shape:", bnt_resp.shape, "Stim shape:", bnt_stim.shape)

    return bnt_resp, bnt_stim, resp2


def fit_recon(bnt_resp, bnt_stim, time_bins=15, **modelargs):
    print(f"{modelargs}")
    nl_kwargs = {'no_shift': False, 'no_offset': False, 'no_gain': False}
    # model = LN_reconstruction(time_bins=15, channels=bnt_resp.shape[2],
    #                          out_channels=bnt_stim.shape[2], rank=rank,
    #                          nl_kwargs=nl_kwargs)
    model = LN.CNN_reconstruction(time_bins=time_bins, channels=bnt_resp.shape[2],
                                  out_channels=bnt_stim.shape[2],
                                  nl_kwargs=nl_kwargs, **modelargs)

    cost_function = 'squared_error'
    fitter = 'tf'
    fitter_options = {'cost_function': cost_function,  # 'nmse'
                      'early_stopping_tolerance': 5e-3,
                      'validation_split': 0,
                      'learning_rate': 1e-2, 'epochs': 3000
                      }
    fitter_options2 = {'cost_function': cost_function,
                       'early_stopping_tolerance': 1e-4,
                       'validation_split': 0,
                       'learning_rate': 1e-3, 'epochs': 8000
                       }

    model = model.sample_from_priors()
    model = model.sample_from_priors()

    model.layers[-1].skip_nonlinearity()
    model = model.fit(input=bnt_resp, target=bnt_stim, backend=fitter,
                      fitter_options=fitter_options, batch_size=None)
    model.layers[-1].unskip_nonlinearity()
    log.info('Fit stage 2: with static output nonlinearity')
    model = model.fit(input=bnt_resp, target=bnt_stim, backend=fitter,
                      verbose=0, fitter_options=fitter_options2, batch_size=None)

    return model


siteids,cellids = db.get_batch_sites(345)
# lab-specific code to load data from one experiment.
loadkey = "gtgram.fs50.ch18.bin6"
siteid = 'PRN015a'
siteid = 'PRN043a'
siteid = 'PRN050a'
siteid = 'PRN053a'
siteid = 'PRN051a'
siteid = 'PRN031a'

batch = 345
uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
#uri = '/auto/data/nems_db/recordings/344/CLT008a_4f9e060a5ec7a8dae28df42df445e5fadb3313d1.tgz'
rec = load_recording(uri)

d=get_depth_info(siteid=siteid).reset_index()
a1chans=[r['index'] for i,r in d.iterrows() if r.area=='A1']
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

epoch_name = "STIM_04Rain-0-1-2_01FightSqueak-0-1-1"
epoch_name = "STIM_17Tuning-0-1-1_01FightSqueak-0-1-1"
epoch_name = "STIM_17Tuning-0-1-2_01FightSqueak-0-1-1"
epoch_name = "STIM_19Blender-0-1-2_01FightSqueak-0-1-1"
epoch_name = "STIM_04Rain-0-1-2_01FightSqueak-0-1-1"
epoch_name = "STIM_19Blender-0-1-2_05KitWhine-0-1-1"
epoch_name = "STIM_12Film-0-1-2_05KitWhine-0-1-1"
epoch_name = "STIM_12Film-0-1-1_05KitWhine-0-1-1"
epoch_name = "STIM_19Blender-0-1-1_05KitWhine-0-1-1"
epoch_name = "STIM_28Coffee-0-1-1_05KitWhine-0-1-1"
epoch_name = "STIM_28Coffee-0-1-1_02Gobble-0-1-1"
epoch_id = epoch_df.loc[epoch_df['BG + FG']==epoch_name].index.values[0]

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
f, ax = plt.subplots(4, 3, figsize=(8, 8), sharex='row', sharey='row')

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

#raise(ValueError("Stopping"))

cmin, cmax = idx_to_cluster_array.min(), idx_to_cluster_array.max()

# create list of
models = []
resps = []
cellcount = rec['resp'].shape[0]

cluster_sets = [[c] for c in range(cmin, cmax+1)] + [[c for c in range(cmin, cmax+1)]]
exclude_clusters = False
if exclude_clusters:
    # leave out one cluster (instead of just fitting for that one cluster)
    call = np.arange(cmin,cmax+1)
    cluster_sets = [np.setdiff1d(call,c) for c in cluster_sets]
    cluster_sets[-1] = call
#cluster_sets = [cluster_sets[1], cluster_sets[-1]]
for fidx, cluster_ids in enumerate(cluster_sets):
    print(f"Starting cluster {fidx}: {cluster_ids}")
    bnt_resp, bnt_stim, resp2 = get_cluster_data(rec, idx_to_cluster_array, cluster_ids)

    #model = fit_recon(bnt_resp, bnt_stim, L1=20, L2=10)
    nl_kwargs = {'no_shift': False, 'no_offset': False, 'no_gain': False}
    # model = LN_reconstruction(time_bins=15, channels=bnt_resp.shape[2],
    #                          out_channels=bnt_stim.shape[2], rank=rank,
    #                          nl_kwargs=nl_kwargs)
    model = LN.CNN_reconstruction(time_bins=15, channels=bnt_resp.shape[2],
                                  out_channels=bnt_stim.shape[2],
                                  nl_kwargs=nl_kwargs, L1=15, L2=0)
    model = model.fit_LBHB(bnt_resp, bnt_stim)
    models.append(model)
    resps.append(resp2)

#plt.close('all')
f, ax = plt.subplots(len(cluster_sets)+1, 3, figsize=(6, 10), sharex=True, sharey=True)
for col, epoch in enumerate(epoch_list):
    spec = stim.extract_epoch(epoch)[0].T
    ax[0, col].imshow(spec.T, aspect='auto', cmap='gray_r', interpolation='none',
                      origin='lower', extent=[-0.5, 1.5, 1, spec.shape[1]], vmin=0, vmax=1)
    ax[0, col].set_title(epoch.replace("STIM_", ""), fontsize=8)
    if col == 0:
        ax[0, col].set_ylabel('Freq')

for fidx, cluster_ids in enumerate(cluster_sets):
    model = models[fidx]
    resp2 = resps[fidx]

    spec_fg = stim.extract_epoch(efg)[0].T
    spec_bg = stim.extract_epoch(ebg)[0].T
    spec_fgbg = stim.extract_epoch(efgbg)[0].T

    for col, epoch in enumerate(epoch_list):
        raster = resp2.extract_epoch(epoch)
        psth = np.mean(raster, axis=0).T
        spec = stim.extract_epoch(epoch)[0].T

        recon = model.predict(psth)
        ax[fidx+1, col].imshow(recon.T, aspect='auto', cmap='gray_r', interpolation='none',
                          origin='lower', extent=[-0.5, 1.5, 1, recon.shape[1]], vmin=0, vmax=1)
        if col == 0:
            recon0 = recon
            Efg = np.std(recon - spec_fg) / np.std(spec_fg)  # np.corrcoef(recon.flatten(), spec_fg.flatten())[0,1] #
            Ebg = np.std(recon - spec_bg) / np.std(spec_bg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
            Efgbg = np.std(recon - spec_fgbg) / np.std(spec_fgbg)  # np.corrcoef(recon.flatten(), spec_bg.flatten())[0,1] #
            if len(cluster_ids)==1:
                n_units = (np.isin(idx_to_cluster_array, cluster_ids)).sum()
                ax[fidx+1, col].set_ylabel(f'{cluster_ids} n={n_units}')
            else:
                ax[fidx + 1, col].set_ylabel(f'{cluster_ids}')
            ax[fidx+1, col].set_title(f"Efgbg={Efgbg:.3f}, Efg={Efg:.3f}, Ebg={Ebg:.3f}", fontsize=8)
        else:
            E = np.std(recon - spec) / np.std(spec)  # np.corrcoef(recon.flatten(), spec.flatten())[0,1] #
            Er = np.std(recon - recon0) / np.std(recon0)  # np.corrcoef(recon.flatten(), recon0.flatten())[0,1] #
            ax[fidx+1, col].set_title(f"E={E:.3f}, Er={Er:.3f}", fontsize=8)

f.suptitle(f'{efgbg.replace("STIM_", "")} abs={use_abs_corr}')
plt.tight_layout()
