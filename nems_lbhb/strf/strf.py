import nems_lbhb.baphy as nb
import nems_lbhb.io as nio
from nems import epoch as ep
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import signal as sgn
import collections

from nems_lbhb.strf.torc_subfunctions import interpft, strfplot, strf_torc_pred, strf_est_core

####Sample Data- works as test#####
# mfilename = "/auto/data/daq/Amanita/AMT005/AMT005c05_p_TOR.m"
# cellid = 'AMT005c-12-1'
###################################

def tor_tuning(mfilename,cellid,plot=False):
    '''
    Creates STRF from stimulus and response
    :param mfilename: File with your data in it
    :param cellid: Name of cell
    :param fs: sampling frequency, default 1000
    :param plot: If True, makes a nice plot of your data
    :return: named tuple with important data that would be found in plot (strf, bestfreq, snr, onset/offset latency)
    '''
    fs = 1000
    rec = nb.baphy_load_recording_file(mfilename=mfilename, cellid=cellid,fs=fs, stim=False)
    globalparams, exptparams, exptevents = nio.baphy_parm_read(mfilename)
    signal = rec['resp'].rasterize(fs=fs)

    # Pick only TORC epochs, find them, extract them
    epoch_regex = "^STIM_TORC_.*"         #pick all epochs that have STIM_TORC
    epochs_to_extract = ep.epoch_names_matching(signal.epochs, epoch_regex)
    r = signal.extract_epochs(epochs_to_extract)

    # Transform r to have dimensions time x repetitions x torc recordings
    all_arr = list()
    for val in r.values():
        fval = np.swapaxes(np.squeeze(val),0, 1)
        all_arr.append(fval)
    stacked = np.stack(all_arr, axis=2)      #rasters

    TorcObject = exptparams["TrialObject"][1]["ReferenceHandle"][1]

    # Process response signal to eliminate bins of silence before and after stimulus
    PreStimbin = int(TorcObject['PreStimSilence']*fs)
    PostStimbin = int(TorcObject['PostStimSilence']*fs)
    numbin = stacked.shape[0]
    stacked = stacked[PreStimbin:(numbin-PostStimbin),:,:]

    INC1stCYCLE = 0
    [strf0,snr,stim,strfemp,StimParams] = strf_est_core(stacked, TorcObject, exptparams, fs, INC1stCYCLE, 16)

    pred = strf_torc_pred(stim, strf0)

    if INC1stCYCLE:
        FirstStimTime = 0
    else:
        FirstStimTime = 250

    numreps = stacked.shape[1]
    numstims = stacked.shape[2]
    [_,stimT,_] = stim.shape
    basep = StimParams['basep']

    stackeduse = stacked[FirstStimTime:,:,:]
    cyclesperrep = int(stackeduse.shape[0] / basep)
    totalreps = numreps * cyclesperrep

    stackeduse = np.reshape(stackeduse, [int(basep),totalreps,numstims], order='F').copy()

    jackcount = 16
    jstrf = np.zeros((strf0.shape[0],strf0.shape[1],jackcount))
    jackstep = totalreps / jackcount
    mm = int(np.round(totalreps / 2))
    xc = np.expand_dims(np.zeros(jackcount),axis=1)

    for jj in range(jackcount):
        estidx = range(mm) + np.round((jj) * jackstep) + 1
        estidx = (np.remainder(estidx - 1, totalreps)).astype(int)
        validx = (np.setdiff1d(range(totalreps), estidx)).astype(int)
        tr = np.expand_dims(np.nanmean(stackeduse[:,estidx,:], 1),axis=1)
        trval = np.nanmean(stackeduse[:,validx,:],1)

        [jstrf[:,:,jj],_,_,_,_] = strf_est_core(tr,TorcObject,exptparams,fs,1)
        jpred = strf_torc_pred(stim,jstrf[:,:,jj])

        trval2 = np.zeros(pred.shape)
        for ii in range(trval.shape[1]):
            trval2[:,ii] = sgn.resample_poly(trval[:,ii],stimT,basep).copy()
        trvalravel = np.ravel(trval2,order='F').copy()
        jpredravel = np.ravel(jpred,order='F').copy()
        xc[jj] = np.corrcoef(trvalravel[:],jpredravel[:])[0,1]

    linpred = np.mean(xc)

    maxoct = int(np.log2(StimParams['hfreq']/StimParams['lfreq']))
    stepsize2 = maxoct / strf0.shape[0]

    smooth = [100,strf0.shape[1]]
    strfsmooth = interpft(strf0, smooth[0], 0)
    strfempsmooth = interpft(strfemp, smooth[0], 0)

    ff = np.exp(np.linspace(np.log(StimParams['lfreq']),np.log(StimParams['hfreq']),strfsmooth.shape[0]))

    mm = np.mean(strfsmooth[:,:7] * (1*(strfsmooth[:,:7] > 0)), 1)
    if sum(np.abs(mm)) > 0:
        bfidx = int(sum(((mm == np.max(mm)).ravel().nonzero())))
        bf = np.round(ff[bfidx])
        bfshiftbins = (maxoct / 2 - np.log2(bf / StimParams['lfreq'])) / stepsize2
    else:
        bfidx = 1
        bf = 0
        bfshiftbins = 0

    mmneg = np.mean(strfsmooth[:,:7] * (1*(strfsmooth[:,:7] < 0)), 1)
    if sum(np.abs(mm)) > 0:
        wfidx = int(sum(((mmneg == np.min(mmneg)).ravel().nonzero())))
        wf = np.round(ff[wfidx])
        wfshiftbins = (maxoct / 2 - np.log2(wf / StimParams['lfreq'])) / stepsize2
    else:
        wfidx = 1
        wf = 0
        wfshiftbins = 0

    if -mmneg[wfidx] > mm[bfidx]:
        # If stronger negative component, calculate latency with neg
        shiftbins = wfshiftbins
        irsmooth = -interpft(strfsmooth[wfidx, :], 250)
        irempsmooth = interpft(strfempsmooth[wfidx], 250, 0)
    else:
        # Use positives
        shiftbins = bfshiftbins
        irsmooth = interpft(strfsmooth[bfidx, :], 250)
        irempsmooth = interpft(strfempsmooth[bfidx], 250)

    mb = 0
    # Find significantly modulated time bins
    sigmod = np.asarray((irsmooth-mb > irempsmooth*2).ravel().nonzero())
    sigmod = sigmod[np.logical_and(sigmod>=7,sigmod<124)]

    if len(sigmod) > 3:
        latbin = sigmod[0]
        dd = np.concatenate([np.diff(sigmod),[41]])
        durbin = sigmod[np.min((dd[0:] > 40).ravel().nonzero())]
        lat = int(np.round(latbin * 1000 / fs))
        offlat = int(np.round(durbin * 1000 / fs))
        print("onset/offset latency:", lat, offlat)
    else:
        latbin = 0
        lat = 0
        durbin = 0
        offlat = 0
        print('no significant onset latency\n')

    # Plot code
    if plot:
        fig,axs = plt.subplots(1,3)

        [freqticks,_] = strfplot(strf0, StimParams['lfreq'], StimParams['basep'], 1, StimParams['octaves'], axs=axs[0])

        [ylow,yhigh] = axs[0].get_ylim()
        [_,xhigh] = axs[0].get_xlim()

        ydiff = yhigh - ylow
        ym = ylow + ydiff/2
        ybf = ym - shiftbins / strf0.shape[0] * ydiff
        axs[0].hlines(ybf,0,xhigh,linestyle='dashed')
        axs[0].vlines(latbin,0,yhigh,linestyle='dashed')
        axs[0].vlines(durbin,0,yhigh,linestyle='dashed')
        axs[0].set_title('%s - BF %d Hz' % (os.path.basename(mfilename),bf),fontweight='bold')
        axs[0].set_xlabel('SNR %.2f linxc %.2f' % (snr,linpred))

        #move to next subplot
        axs[1].set_ylim(np.min(irsmooth),np.max(irsmooth))
        axs[1].set_xlim(0,len(irsmooth))
        if np.all(strfempsmooth[:] == 0):
            axs[1].plot(irsmooth)
        else:
            axs[1].errorbar(range(len(irsmooth)),irsmooth,irempsmooth,alpha=0.3)
        axs[1].hlines(mb,0,len(irsmooth),linestyle='dashed')
        axs[1].vlines(latbin,0,(np.max(irsmooth)+np.max(irempsmooth)),linestyle='dashed')
        axs[1].vlines(durbin,0,(np.max(irsmooth)+np.max(irempsmooth)),linestyle='dashed')
        axs[1].set_title('On/Off Lat %d/%d ms' % (lat, offlat),fontweight='bold')

        #move to next subplot
        [u,s,v] = np.linalg.svd(strfsmooth)
        axs[2].set_xlim(0,u.shape[0])
        axs[2].set_xticks(np.linspace(0,u.shape[0],6))
        axs[2].set_xticklabels(freqticks)
        axs[2].plot(ndi.filters.gaussian_filter(u[:,0],5))
        axs[2].set_title('Frequency Tuning',fontweight='bold')
        axs[2].set_xlabel('Frequency (Hz)')
        axs[2].set_ylabel('Gain (a.u.)')

    tor_tuning_output = collections.namedtuple('STRF_Data',['STRF','Best_Frequency_Hz','Signal_to_Noise','Onset_Latency_ms','Offset_Latency_ms'])

    return tor_tuning_output(strf0, bf, snr, lat, offlat)