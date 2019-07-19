import nems_lbhb.baphy as nb
import nems_lbhb.io as nio
from nems import epoch as ep
import os

import matplotlib.pyplot as plt
import numpy as np

from nems_lbhb.strf.strf import tor_tuning
from nems_lbhb.strf.torc_subfunctions import strfplot
from nems_lbhb.strf.tensor_subfunctions import strf_plot_prepare,model_strf,strf_input_gen

####Sample Data- works as test#####
# mfilename = "/auto/data/daq/Amanita/AMT005/AMT005c05_p_TOR.m"
# cellid = 'AMT005c-12-1' #one being used in Matlab
# fs=1000
###########################

def strf_tensor(mfilename,cellid,plot=False,linalg=False,real=False):
    '''
     Creates matrices used as features and labels of tensor
     :param mfilename: File with your data in it
     :param cellid: Name of cell
     :param fs: sampling frequency, default 1000
     :param linalg: if True, will perform and display strf generated from inputs using linear algebra as check
     :param real: if True, will run full tor_tuning function to give actual strf output for given input
     :return: matrix of stimulus, with time delay & matrix of response
     '''
    fs=1000
    rec = nb.baphy_load_recording_file(mfilename=mfilename, cellid=cellid, fs=fs, stim=False)
    globalparams, exptparams, exptevents = nio.baphy_parm_read(mfilename)
    signal = rec['resp'].rasterize(fs=fs)

    epoch_regex = "^STIM_TORC_.*"  # pick all epochs that have STIM_TORC_...
    epochs_to_extract = ep.epoch_names_matching(signal.epochs, epoch_regex)  # find those epochs
    r = signal.extract_epochs(
        epochs_to_extract)  # extract them, r.keys() yields names of TORCS that can be looked through as dic r['name']...can be np.squeeze(0, np.mean(

    all_arr = list()
    for val in r.values():
        fval = np.swapaxes(np.squeeze(val), 0, 1)
        all_arr.append(fval)
    stacked = np.stack(all_arr,axis=2)

    TorcObject = exptparams["TrialObject"][1]["ReferenceHandle"][1]
    PreStimbin = int(TorcObject['PreStimSilence'] * fs)
    PostStimbin = int(TorcObject['PostStimSilence'] * fs)
    numbin = stacked.shape[0]
    stacked = stacked[PreStimbin:(numbin - PostStimbin), :, :]
    stimall,avgResp,Params = strf_input_gen(stacked,TorcObject,exptparams,fs)

    fitH = model_strf(stimall,avgResp)

    if plot == True:
        #bf = strf_plot_prepare(fitH,Params)
        [_,_] = strfplot(fitH, Params['lfreq'], Params['basep'], 1, Params['octaves'])
        plt.title('%s - Linear Regression' % (os.path.basename(mfilename)), fontweight='bold')

    if linalg == True:
        #based on idea that H = Y*Xtranspose*inverse of correlation matrix
        X = stimall
        XT = np.swapaxes(stimall, 0, 1)
        Y = np.swapaxes(np.expand_dims(avgResp, axis=1), 0, 1)
        C = np.dot(X, XT)
        Cinv = np.linalg.pinv(C)

        H = np.reshape((np.dot(np.dot(Y, XT), Cinv)),(15,25),order='F')

        #bfl = strf_plot_prepare(H,Params)
        [_,_] = strfplot(H, Params['lfreq'], Params['basep'], 1, Params['octaves'])
        plt.title('%s - Control' % (os.path.basename(mfilename)), fontweight='bold')

    if real == True:
        _ = tor_tuning(mfilename,cellid,fs,plot=True)

    return fitH