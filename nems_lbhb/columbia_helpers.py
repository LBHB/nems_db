"""
Helper functions for fitting models to ECoG data from Columbia group

requires hdf5storage
"""

import os
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from hdf5storage import loadmat, savemat
import matplotlib.pyplot as plt

import nems.analysis.api
import nems.initializers
import nems.preprocessing as preproc
import nems.recording as recording
import nems.uri
from nems.fitters.api import scipy_minimize
from nems.signal import RasterizedSignal
from nems.registry import xform, xmodule

log = logging.getLogger(__name__)


dbase_dir = "/auto/users/svd/projects/pop_models/ecog/"
dbase_name = "out_sample_highgamma.mat"

def load_data(path, dbname, downsample=1, compress=1/3):
    mdata = loadmat(os.path.join(path, dbname))

    # Patient identifier
    subject = mdata["subject"][0]
    # Electrode anatomical locations
    elec_loc = [str(e[0]) for e in mdata["loc_destrieux"][0]]
    # Make unique electrode labels
    elec_id  = [f"{subject}-i{e:04d}" for e in range(len(elec_loc))]

    # Read two separate tasks
    hank_stories  = mdata["out"][0]["hank_stories"][0] # 19 trials with different lengths, used for estimation
    repetitions   = mdata["out"][0]["repetitions"][0]  # 8 trials with same length, repeated 5 or 6 times, used for validation

    # Unique trial names
    est_trial = [str(s[0]) for t in (hank_stories,) for s in t["name"][0]]
    val_trial = [str(s[0]) for t in (repetitions,) for s in t["name"][0]]

    # Data sampling rate
    fs = int(np.unique([f for t in (hank_stories, repetitions) for f in t["dataf"][0]]))

    # Estimation set stimulus and response (optional: spectrogram downsampled and compressed)
    X = [x for t in (hank_stories,) for x in t["stim"][0]]
    X = [x**compress if downsample==1 else x[::downsample]**compress for x in X]
    Y = [y for t in (hank_stories,) for y in t["resp"][0]]

    # Validation set stimulus and response
    X_val = [x for t in (repetitions,) for x in t["stim"][0]]
    X_val = [x**compress if downsample==1 else x[::downsample]**compress for x in X_val]
    Y_val = [y for t in (repetitions,) for y in t["resp"][0]]

    return elec_id, elec_loc, fs, est_trial, X, Y, val_trial, X_val, Y_val


def to_recording(X, Y, trials, chans, locs=None, repetition=False, fs=1, recname="ecog"):
    # Create epochs for each trial
    epoch, offset = [], 0
    for i, name in enumerate(trials):
        length = X[i].shape[1] / fs
        epoch.append([offset, offset+1.0, "PreSilence"])
        epoch.append([offset+1.0, offset+length-1.0, "TrialStimulus"])
        epoch.append([offset+1.0, offset+length-1.0, "Trial_" + name])
        epoch.append([offset+length-1.0, offset+length, "PostSilence"])
        offset += length
    epoch = pd.DataFrame(epoch, columns=("start", "end", "name"))

    signals = dict()

    # Concatenate all trials, with silence padding
    stim = np.concatenate(X, 1)
    signals["stim"] = nems.signal.RasterizedSignal(
        fs, stim, "stim", recname, epochs=epoch, segments=None)

    resp = np.concatenate(Y, 1)
    resp = resp.mean(-1) if repetition else resp
    signals["resp"] = nems.signal.RasterizedSignal(
        fs, resp, "resp", recname, chans=chans, meta={"type": "ECoG", "locs": locs})

    return nems.recording.Recording(signals, name=recname)

@xform
def ldcol(loadkey, recording_uri=None, cellid=None):
    '''
    Keyword for loading Columbia ecog data.

    options:
    dN : downsample by factor N (default N=4)
    '''

    options = loadkey.split('.')[1:]

    d = {}
    if recording_uri is not None:
        d['recording_uri_list'] = [recording_uri]
    d['downsample'] = 4
    if cellid is not None:
        d['cellid'] = cellid

    for op in options:
        if op[:1] == 'd':
            d['downsample'] = int(op[1:])

    xfspec = [['nems_lbhb.columbia_helpers.load_columbia_data', d]]

    return xfspec


def load_columbia_data(recording_uri_list=None, cellid=None, downsample=4, **options):
    """
    Load and format data as NEMS recording

    converted to xforms-friendly format (SVD 2020-03-24)
    """

    # this could be replaced with a "for" loop over multiple files. then you'd need to concatenate_time the recordings.
    if recording_uri_list is not None:
       recording_uri = recording_uri_list[0]
    else:
       recording_uri = dbase_dir + dbase_name

    # load Matlab data files
    recname = "samplerec"
    elec_id, elec_loc, fs, est_trial, X, Y, val_trial, X_val, Y_val = load_data("", recording_uri, downsample=downsample)


    # check if cellid exists in recording
    if cellid not in elec_id:
        raise ValueError("cellid not found")

    print(f"Selecting channel {cellid} from {str(elec_id)}")

    # create NEMS-format recording objects from the raw data
    est = to_recording(X, Y, est_trial, elec_id, elec_loc, fs=fs)
    val = to_recording(X_val, Y_val, val_trial, elec_id, elec_loc, repetition=True, fs=fs)

    # Pick out first electrode for testing
    estx = nems.recording.Recording({"stim": est.signals["stim"],
                                     "resp": est.signals["resp"].extract_channels([cellid])})
    valx = nems.recording.Recording({"stim": val.signals["stim"],
                                     "resp": val.signals["resp"].extract_channels([cellid])})
    return {'est': estx, 'val': valx}

