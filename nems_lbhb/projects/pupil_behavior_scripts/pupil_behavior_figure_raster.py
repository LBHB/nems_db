#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import pandas as pd
from scipy.signal import decimate

import nems0.db as nd
import nems_db.params
import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems_lbhb.xform_wrappers as nw
import nems0.db as nd
import nems0.recording as recording
import nems0.epoch as ep
import nems0.preprocessing as preproc
import nems0.plots.api as nplt


def psth_per_file(rec):

    raise NotImplementedError

    resp = rec['resp'].rasterize()

    file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")

    epoch_regex = "^STIM_"
    stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

    r = []
    max_rep_id = np.zeros(len(file_epochs))
    for f in file_epochs:

        r.append(resp.as_matrix(stim_epochs, overlapping_epoch=f) * resp.fs)

    repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
    max_rep_id, = np.where(repcount == np.max(repcount))

    t = np.arange(r.shape[-1]) / resp.fs

    plt.figure()

    ax = plt.subplot(3, 1, 1)
    nplt.plot_spectrogram(s[max_rep_id[-1],0,:,:], fs=stim.fs, ax=ax,
                          title="cell {} - stim".format(cellid))

    ax = plt.subplot(3, 1, 2)
    nplt.raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster')

    ax = plt.subplot(3, 1, 3);
    nplt.psth_from_raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster',
                          ylabel='spk/s')

    plt.tight_layout()


# PSTH figures
batch = 307
fh = plt.figure()
outpath = "{}/selectivity_{}/".format(
        '/auto/users/svd/docs/current/pupil_behavior',batch)
csv_file = outpath+"selectivity.csv"

try:
    d_cells = pd.read_csv(csv_file)
except:
    d_cells = nd.get_batch_cells(batch)

    d_cells['spont']=0
    d_cells['ref_onset']=0
    d_cells['ref_sust']=0
    d_cells['ref_offset']=0
    d_cells['tar_onset']=0
    d_cells['tar_sust']=0
    d_cells['tar_offset']=0
    d_cells['ref_mean'] = '0'
    d_cells['tar_mean'] = '0'

d_cells = d_cells.set_index(['cellid'])
#cellids = list(d_cells.index)
loader = "psth.fs200-st.pupbeh"

for cellid in d_cells[d_cells['spont']==0].index:
    print(cellid)

    #cellid = "TAR010c-06-1"
    #cellid = "bbl102d-13-1"
    uri = nw.generate_recording_uri(cellid, batch, loader)

    rec = recording.load_recording(uri)
    rec = preproc.mask_all_but_correct_references(rec, balance_rep_count=False)
    fs = rec['resp'].fs

    # TODO move code below to
    # stateplots.psth_per_file(rec)

    file_epochs = ep.epoch_names_matching(rec.epochs, "^FILE_")

    epoch_regex = "^STIM_"
    stim_epochs = ep.epoch_names_matching(rec.epochs, epoch_regex)
    tar_regex = "^TAR_"
    tar_epochs = ep.epoch_names_matching(rec.epochs, tar_regex)

    d = rec['resp'].get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d))
    d = rec['resp'].get_epoch_bounds('PostStimSilence')
    dd = np.diff(d)
    PostStimSilence = round(np.min(dd[dd > 0]), 3)
    prebins = int(PreStimSilence*fs)
    postbins = int(PostStimSilence*fs)
    onsetbin = prebins+int((PreStimSilence+0.1)*fs)

    r = []
    tar = []
    max_rep_id = []
    max_tar_id = []

    spont = []
    ref_onset = []
    ref_sust = []
    ref_offset = []
    ref_mean = []
    tar_onset = []
    tar_sust = []
    tar_offset = []
    tar_mean = []

    for i, f in enumerate(file_epochs):

        trec = rec.copy()
        trec = trec.and_mask(f)

        resp = trec['resp'].rasterize()
        m = trec['mask']

        _r = resp.as_matrix(stim_epochs, mask=m)
        if _r.size > 0:
            r.append(_r)
            repcount = np.sum(np.isfinite(_r[:, :, 0, 0]), axis=1)
            n, = np.where(repcount == np.max(repcount))
            n = n[-1]
            max_rep_id.append(n)
            offsetbin = _r.shape[3]-postbins

            spont.append(np.nanmean(_r[n,:,0,:prebins])*fs)
            ref_onset.append(np.nanmean(_r[n,:,0,prebins:onsetbin])*fs)
            ref_sust.append(np.nanmean(_r[n,:,0,onsetbin:offsetbin])*fs)
            ref_offset.append(np.nanmean(_r[n,:,0,offsetbin:(offsetbin+postbins)])*fs)
            ref_mean.append(decimate(np.nanmean(_r[n,:,0,:], axis=0),2))

            trec = trec.create_mask(f)
            _t = resp.as_matrix(tar_epochs, mask=trec['mask'])
            tar.append(_t)
            repcount = np.sum(np.isfinite(_t[:, :, 0, 0]), axis=1)
            if (len(repcount) >= 3) and (repcount[2]>0):
                n = 2
            elif (len(repcount) == 2) and (repcount[0]==0):
                n = 1
            else:
                n = 0
            max_tar_id.append(n)
            #n, = np.where(repcount == np.max(repcount))
            #max_tar_id.append(n[-1])
            tar_onset.append(np.nanmean(_t[n,:,0,prebins:onsetbin])*fs)
            tar_sust.append(np.nanmean(_t[n,:,0,onsetbin:offsetbin])*fs)
            tar_offset.append(np.nanmean(_t[n,:,0,offsetbin:(offsetbin+postbins)])*fs)
            tar_mean.append(decimate(np.nanmean(_t[n,:,0,:], axis=0),2))

    d_cells.loc[cellid,'spont'] = np.nanmean(np.array(spont)[:3])
    d_cells.loc[cellid,'ref_onset'] = np.nanmean(np.array(ref_onset)[:3])
    d_cells.loc[cellid,'ref_sust'] = np.nanmean(np.array(ref_sust)[:3])
    d_cells.loc[cellid,'ref_offset'] = np.nanmean(np.array(ref_offset)[:3])
    d_cells.loc[cellid,'tar_onset'] = np.nanmean(np.array(tar_onset)[:3])
    d_cells.loc[cellid,'tar_sust'] = np.nanmean(np.array(tar_sust)[:3])
    d_cells.loc[cellid,'tar_offset'] = np.nanmean(np.array(tar_offset)[:3])

    d_cells.at[cellid,'ref_mean'] = list(np.nanmean(np.stack(ref_mean[:3]), axis=0))
    d_cells.at[cellid,'tar_mean'] = list(np.nanmean(np.stack(tar_mean[:3]), axis=0))

    fh.clf()

    for i, _r in enumerate(r):
        # set up time axis
        t = np.arange(_r.shape[-1]) / resp.fs - PreStimSilence - 0.5/resp.fs
        bincount = t.size

        # plot target raster and PSTH
        _t = tar[i][max_tar_id[i], :, 0, :bincount]

        ax = plt.subplot(4, len(r), i+1)
        nplt.raster(t, _t, ax=ax, title=cellid, ylabel='tar')
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.plot(np.array([0, 0]), ylim, 'g--')
        ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs), ylim, 'g--')

        ax = plt.subplot(4, len(r), len(r)+i+1)
        title="{:.1f}/{:.1f}/{:.1f}".format(tar_onset[i],tar_sust[i],tar_offset[i])
        nplt.psth_from_raster(t, _t*fs, ax=ax, title=title,
                              ylabel='spk/s', binsize=10)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.plot(np.array([0, 0]), ylim, 'k--')
        ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs*10), ylim, 'k--')

        # plot reference raster
        med_rep=3

        ax = plt.subplot(4, len(r), len(r)*2+i+1)
        rall = _r[max_rep_id[i], :, 0, :]
        #rall = np.reshape(rall, [rall.shape[0]*med_rep, rall.shape[2]])
        nplt.raster(t, rall, ax=ax, ylabel='ref', title=file_epochs[i])
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.plot(np.array([0, 0]), ylim, 'g--')
        ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs), ylim, 'g--')

        # plot reference PSTH
        ax = plt.subplot(4, len(r), len(r)*3+i+1)
        rall = _r[max_rep_id[i], :, 0, :] * fs
        #rall = np.reshape(rall, [rall.shape[0] * med_rep, rall.shape[2]])
        title="{:.1f}/{:.1f}/{:.1f}".format(ref_onset[i],ref_sust[i],ref_offset[i])
        nplt.psth_from_raster(t, rall, ax=ax, title=title,
                              ylabel='spk/s', binsize=10)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.plot(np.array([0, 0]), ylim, 'k--')
        ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence + 0.5/resp.fs*10), ylim, 'k--')

    plt.tight_layout()
    #plt.show()
    #plt.pause(0.05)
    fh.savefig(outpath+cellid+".pdf")
    d_cells.to_csv(outpath+"selectivity.csv")