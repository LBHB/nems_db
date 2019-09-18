import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sci
import statsmodels.formula.api as smf
import matplotlib.collections as clt
import re
import pylab as pl

from nems_lbhb.pupil_behavior_scripts.mod_per_state import get_model_results_per_state_model
from nems_lbhb.pupil_behavior_scripts.mod_per_state import aud_vs_state
from nems_lbhb.pupil_behavior_scripts.mod_per_state import hlf_analysis
from nems_lbhb.pupil_behavior_scripts.mod_per_state import beh_only_plot
from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
import nems_lbhb.io as nio
import common
import nems.epoch as ep

from nems_lbhb.strf import strf
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording
from nems.xforms import load_recordings
from nems_lbhb.strf.torc_subfunctions import interpft, strfplot, get_strf_tuning, \
    strf_torc_pred, strf_est_core

# SPECIFY pup+beh models
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
#basemodel = "-ref-psthfr.s_sdexp.S"
basemodel = "-ref-psthfr.s_stategain.S"

#df.pd.read_csv('pup_beh_processed.csv')
df=pd.read_csv('d_307_pb.csv')

cellids=np.unique(df['cellid'].values)
batch=307
loadkey="psth.fs100.pup"

cellids = [c for c in cellids if not c.startswith("AMT018")]
cellids = [c for c in cellids if not c.startswith("AMT020")]
#cellids = ['BRT026c-02-1']

PERFILE = True

for c, cellid in enumerate(cellids[63:]):
    plt.close('all')
    uri = generate_recording_uri(cellid, batch, loadkey=loadkey)

    ctx = load_recordings([uri], cellid=cellid)
    rec = ctx['rec']

    if PERFILE:
        e = rec.epochs
        masks = e[e['name'].str.startswith("FILE_")]['name'].values.tolist()
    else:
        masks = ['PASSIVE_EXPERIMENT', 'ACTIVE_EXPERIMENT']

    mfilename = rec.meta['files'][0]
    globalparams, exptparams, exptevents = nio.baphy_parm_read(mfilename)

    print("Fitting single STRF for all data to get tuning properties")
    res_all = strf.tor_tuning(exptparams, cellid, rec=rec, fs=1000, plot=False)
    StimParams = res_all.StimParams
    bf, lat, offlat, bfidx, latbin, durbin = get_strf_tuning(
        res_all.STRF, res_all.STRF_error, StimParams)

    res = []
    active = []
    exptparams_set = []
    mdisp=[]
    for m in masks:
        _rec = rec.copy()
        _rec = _rec.create_mask(m)
        if ("_a_" in m) or ("ACTIVE" in m):
            _rec = _rec.and_mask('HIT_TRIAL')
        _rec = _rec.remove_masked_epochs()
        _m = m.replace('FILE_', '')

        mfilenames = [f for f in rec.meta['files'] if _m in f]
        if len(mfilenames) == 0:
            if m == 'PASSIVE_EXPERIMENT':
                mfilenames = [f for f in rec.meta['files'] if "_p_" in f]
            else:
                mfilenames = [f for f in rec.meta['files'] if "_a_" in f]
            if len(mfilenames) == 0:
                mfilenames = rec.meta['files']
        mfilename = mfilenames[0]

        globalparams, exptparams, exptevents = nio.baphy_parm_read(mfilename)
        active.append(exptparams['BehaveObjectClass'] != 'Passive')
        exptparams_set.append(exptparams)
        mdisp.append(_m)
        res.append(strf.tor_tuning(exptparams, cellid, rec=_rec, fs=1000, plot=False))

    mm = np.max(np.abs(np.concatenate([r.STRF for r in res])))
    aidxs = np.argwhere(np.array(active))[:,0]
    pidxs = np.argwhere(~np.array(active))[:,0]

    fig, axs = plt.subplots(aidxs.size, 3, figsize=(8,2*aidxs.size))
    if axs.ndim == 1:
        axs = axs[np.newaxis, :]
    for i, aidx in enumerate(aidxs):
        pidx = pidxs[np.argmin(np.abs(pidxs-aidx))]

        strf_diff = res[aidx].STRF - res[pidx].STRF
        mmd = np.max(np.abs(strf_diff))

        StimParams = res[aidx].StimParams
        #bf, lat, offlat, bfidx, latbin, durbin = get_strf_tuning(
        #    res[aidx].STRF, res[aidx].STRF_error, StimParams)

        TarFreq = exptparams_set[aidx]['TrialObject'][1]['TargetHandle'][1]['Frequencies']
        if (type(TarFreq) is np.ndarray) or (type(TarFreq) is list):
            TarFreq = TarFreq[0]

        tarplotidx = int(np.min([14, np.log2(TarFreq/StimParams['lfreq'])*3]))/3+0.5/3
        taridx = int(np.min([14, np.log2(TarFreq / StimParams['lfreq'])*3]))
        bfplotidx = int(np.min([14, np.log2(bf/StimParams['lfreq'])*3]))/3+0.5/3
        bfidx = int(np.min([14, np.log2(bf / StimParams['lfreq'])*3]))
        latidx = np.argmax(np.abs(res[aidx].STRF[bfidx,:] + res[pidx].STRF[bfidx,:]))
        latplotidx = latidx/res[aidx].STRF.shape[1]*StimParams['basep'] + 5
        #bfidx = int(np.round(np.min([14, bfidx]))-1)
        meanlatidx=int(np.round(latbin+durbin/2)-1)

        delta = strf_diff[bfidx, latidx]
        deltafrac = delta/res[pidx].STRF[bfidx,latidx]
        tdelta = strf_diff[taridx, latidx]
        tdeltafrac = tdelta/res[pidx].STRF[bfidx,latidx]

        print('{} BF delta frac (bf,lat {:d},{:d})= {:.3f}/{:.3f} = {:.3f}'.format(
            cellid, bfidx, meanlatidx, delta, res[pidx].STRF[bfidx,latidx], deltafrac))
        print('{} tar delta frac (bf,lat {:d},{:d})= {:.3f}/{:.3f} = {:.3f}'.format(
            cellid, taridx, meanlatidx, tdelta, res[pidx].STRF[taridx,latidx], tdeltafrac))

        [freqticks, _] = strfplot(res[aidx].STRF, StimParams['lfreq'], StimParams['basep'], 0, StimParams['octaves'], axs=axs[i,0])
        axs[i,0].plot(latplotidx,tarplotidx,'ro')
        axs[i,0].plot(latplotidx,bfplotidx,'kx')
        axs[i,0].get_images()[0].set_clim((-mm,mm))
        axs[i,0].set_title('{} (snr {:.3})'.format(mdisp[aidx], res[aidx].Signal_to_Noise))

        [freqticks, _] = strfplot(res[pidx].STRF, StimParams['lfreq'], StimParams['basep'], 0, StimParams['octaves'], axs=axs[i,1])
        axs[i,1].plot(latplotidx,tarplotidx,'ro')
        axs[i,1].plot(latplotidx,bfplotidx,'kx')
        axs[i,1].get_images()[0].set_clim((-mm,mm))
        axs[i,1].set_title('{} (snr {:.3f})'.format(mdisp[pidx], res[pidx].Signal_to_Noise))

        [freqticks, _] = strfplot(strf_diff, StimParams['lfreq'], StimParams['basep'], 0, StimParams['octaves'], axs=axs[i,2])
        axs[i,2].plot(latplotidx,tarplotidx,'ro')
        axs[i,2].plot(latplotidx,bfplotidx,'kx')
        axs[i,2].get_images()[0].set_clim((-mmd,mmd))
        axs[i,2].set_title('{} act-pas dBF={:.3f} dtar={:.3f}'.format(cellid,deltafrac,tdeltafrac))

    if res[aidx].Signal_to_Noise<0.8 and res[pidx].Signal_to_Noise<0.8:
        print('saving low SNR')
        plt.savefig('/auto/users/svd/docs/current/pupil_behavior/strf/low_snr/strf_ap_'+cellid+'.png')
    else:
        print('saving High SNR')
        plt.savefig('/auto/users/svd/docs/current/pupil_behavior/strf/strf_ap_'+cellid+'.png')

