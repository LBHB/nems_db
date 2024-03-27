import json

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from nems0 import db, preprocessing
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb import baphy_io
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
import nems_lbhb.projects.freemoving.decoder_tools as dec
from nems_lbhb.projects.freemoving import free_model
from pathlib import Path
import zarr

batch=348
dlc_chans=8
rasterfs=100
siteids, cellids = db.get_batch_sites(batch=batch)
plot=True
siteids = siteids[::-1]
f, ax = plt.subplots(4, 4)
ax = ax.flatten()
f1, ax1 = plt.subplots(4, 4)
ax1 = ax1.flatten()
f2, ax2 = plt.subplots(4, 4)
ax2 = ax2.flatten()
f3, ax3 = plt.subplots(4, 4)
ax3 = ax3.flatten()

for i, siteid in enumerate(siteids):
    rec = dec.load_free_data(siteid, batch=batch, rasterfs=rasterfs,
                                    dlc_chans=dlc_chans, compute_position=False, remove_layer='', layer_perms=0, return_raw=True)
    rec = rec[0]
    if i <= 16:
        trial_target_epochs = dec.dlc_epoch_join(rec, targets=['TRIAL', 'TRIAL'], ref=['start', 'start'],
                                                        offset=['-0.5', '+0.5'], plot=plot, ax=ax[i])
        ax[i].set_title(f"{siteid} - TRIAL start")
        ax[i].legend()
        ls_target_epochs = dec.dlc_epoch_join(rec, targets=['LICK , HIT', 'LICK , HIT'], ref=['start', 'start'],
                                                        offset=['-0.5', '+0.5'], plot=plot, ax=ax1[i])
        ax1[i].set_title(f"{siteid} - LICK, HIT")
        ax1[i].legend()

    if (i > 16) and (i <= 32):
        trial_target_epochs = dec.dlc_epoch_join(rec, targets=['TRIAL', 'TRIAL'], ref=['start', 'start'],
                                                 offset=['-0.5', '+0.5'], plot=plot, ax=ax2[i-16])
        ax2[i-16].set_title(f"{siteid} - TRIAL start")
        ax2[i-16].legend()
        ls_target_epochs = dec.dlc_epoch_join(rec, targets=['LICK , HIT', 'LICK , HIT'], ref=['start', 'start'],
                                              offset=['-0.5', '+0.5'], plot=plot, ax=ax3[i-16])
        ax3[i-16].set_title(f"{siteid} - LICK, HIT")
        ax3[i-16].legend()

    np_location, np_hb, dlc_within_np = dec.dlc_within_radius_new(rec, target = ['TRIAL', 'TRIAL'], ref=['start', 'start'], radius=0.015, offset=['-0.5', '0.5'], plot=True)
    ls_location, ls_hb, dlc_within_ls = dec.dlc_within_radius_new(rec, target = ['LICK , HIT', 'LICK , HIT'], ref=['start', 'start'], radius=0.015, offset=['0.0', '+1.0'], plot=True)
    np_entry_ind = np.where(np.array([int(dlc_within_np[i + 1]) - int(dlc_within_np[i]) for i in range(len(dlc_within_np) - 1)]) == 1)[0]
    np_exit_ind = np.where(np.array([int(dlc_within_np[i + 1]) - int(dlc_within_np[i]) for i in range(len(dlc_within_np) - 1)]) == -1)[0]
    ls_entry_ind = np.where(np.array([int(dlc_within_ls[i + 1]) - int(dlc_within_ls[i]) for i in range(len(dlc_within_ls) - 1)]) == 1)[0]
    ls_exit_ind = np.where(np.array([int(dlc_within_ls[i + 1]) - int(dlc_within_ls[i]) for i in range(len(dlc_within_ls) - 1)]) == -1)[0]
    np_entry_ts = np.array([np.round(ind/rasterfs, decimals=2) for ind in np_entry_ind])
    np_epochs = rec.epochs.loc[rec.epochs['name'].str.startswith("TRIAL")].reset_index()
    trial_np_ts_diffs = []
    for trial_ts in np_epochs['start']:
        np_ent = np.where(np.sign(np_entry_ts-trial_ts) == -1)[0][-1]
        trial_np_ts_diffs.append(trial_ts-np_entry_ts[np_ent])
    f, ax = plt.sublots(1,1)
    ax.plot(np.arange(len(trial_np_ts_diffs)), trial_np_ts_diffs)
    ax.set_ylim(0, 1)
    ax.set_ylabel('trial ts - np entry (dlc) (s)')
    ax.set_xlabel('trial')

    np_paths = []
    np_data = []
    event_logs = []
    trial_logs = []
    np_threshold = 2.5
    np_timestamps = []
    np_timestamps_ind = []
    np_end_timestamps = []
    np_end_timestamps_ind = []
    for file in rec.meta['files']:
        np_file = Path(file, 'np_contact_analog.zarr')
        event_log_path = Path(file, 'event_log.csv')
        event_log = pd.read_csv(event_log_path)
        trial_log_path = Path(file, 'trial_log.csv')
        trial_log = pd.read_csv(trial_log_path)
        np_trace = zarr.load(np_file)
        np_rasterfs = zarr.open(np_file).attrs['fs']
        np_thresholded = (np_trace[0, :] > np_threshold)*1
        nps = np_thresholded[1:] - np_thresholded[:-1]
        np_ts_ind = np.where(nps==1)[0]
        np_ts = np_ts_ind/np_rasterfs
        np_end_ts_ind = np.where(nps==-1)[0]
        np_end_ts = np_end_ts_ind/np_rasterfs
        event_log.loc['event' == 'np_start']
        np_data.append(np_trace)
        np_paths.append(np_file)
        event_logs.append(event_log)
        trial_logs.append(trial_log)
        np_timestamps.append(np_ts)
        np_timestamps_ind.append(np_ts_ind)
        np_end_timestamps.append(np_end_ts)
        np_end_timestamps_ind.append(np_end_ts_ind)



    # np_np_start = []
    # np_np_end = []
    # np_ls_start = []
    # np_ls_end = []
    # for np_ent in np_entry_ind:
    #     test = np.argmin(abs(np_exit_ind - np_ent))

bp = []