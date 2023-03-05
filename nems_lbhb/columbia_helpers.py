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

import nems0.analysis.api
import nems0.initializers
import nems0.preprocessing as preproc
import nems0.recording as recording
import nems0.uri
from nems0.fitters.api import scipy_minimize
from nems0.signal import RasterizedSignal
from nems0.registry import xform, xmodule

log = logging.getLogger(__name__)


#dbase_dir = "/tf/data/"
#dbase_name = "out_045_052_058_059_063_highgamma_legacy.mat"
dbase_dir = "/auto/users/svd/projects/pop_models/ecog/"
dbase_name = "out_highgamma_Stephen.mat"


def to_recording(X, Y, trials, fs, labels, locs=None, before=1.0, after=1.0, recname="ecog"):
    # Mask inter-trial silence
    for i in range(len(Y)):
        Y[i][:, :100], Y[i][:, -100:] = np.nan, np.nan

    # Create epochs based on trials
    epoch, offset = [], 0
    for i, name in enumerate(trials):
        length = X[i].shape[1] / fs
        epoch.append([offset, offset+before, "SILENCE_PRE"])
        epoch.append([offset+before, offset+length-after, "TRIAL"])
        epoch.append([offset+before, offset+length-after, "TRIAL_" + name])
        epoch.append([offset+length-after, offset+length, "SILENCE_POST"])
        offset += length
    epoch = pd.DataFrame(epoch, columns=('start', 'end', 'name'))

    # Concatenate all trials
    X = np.concatenate(X, 1)
    Y = np.concatenate(Y, 1)
    Y = Y.mean(-1) if Y.ndim == 3 else Y

    # Create NEMS-format recording
    return nems0.recording.Recording({
        "stim": nems0.signal.RasterizedSignal(
            fs, X, "stim", recname, epochs=epoch, segments=None),
        "resp": nems0.signal.RasterizedSignal(
            fs, Y, "resp", recname, chans=labels, meta={"type": "ECoG", "locs": locs})},
        name=recname)


def load_data(path, downsample=4, compress=1/3, merge_reps=True):
    mdata = loadmat(path)

    groups = mdata["groups"][0]
    labels = [e.item() for e in mdata["ilabel"][0]]

    # Read all tasks
    hank_stories  = mdata["out"][0]["hank_stories"][0]   # 19 trials with different lengths, used for estimation
    short_stories = mdata["out"][0]["short_stories"][0]  # 35 trials with different lengths, used for estimation
    repetitions   = mdata["out"][0]["repetitions"][0]    # 8 trials with same length, repeated 5 or 6 times, used for validation

    # Group estimation and validation trials
    est = [t for s in (hank_stories, short_stories) for t in s]
    val = [t for s in (repetitions,) for t in s]

    # Unique trial names
    trial_est = [t["name"].item() for t in est]
    trial_val = [t["name"].item() for t in val]

    # Data sampling rate
    fs = int(np.unique([t["dataf"].item() for t in est + val]))

    # Estimation set stimulus and response (optional: spectrogram downsampled and compressed)
    X_est = [t["aud"] for t in est]
    Y_est = [t["resp"] for t in est]

    # Validation set stimulus and response
    X_val = [t["aud"] for t in val]
    if merge_reps:
        Y_val = [t["resp"] for t in val]
    else:
        Y_val = ([t["resp"][..., 0] for t in val],
                 [t["resp"][..., 1] for t in val])

    # Compress and downsample stimuli
    X_est = [x**compress if downsample==1 else x[::downsample]**compress for x in X_est]
    X_val = [x**compress if downsample==1 else x[::downsample]**compress for x in X_val]

    # Convert to NEMS-format recordings
    est = to_recording(X_est, Y_est, trial_est, fs, labels, locs=groups, recname="ecog_est")
    if merge_reps:
        val = to_recording(X_val, Y_val, trial_val, fs, labels, locs=groups, recname="ecog_val")
    else:
        val = (to_recording(X_val, Y_val[0], trial_val, fs, labels, locs=groups, recname="ecog_val"),
               to_recording(X_val, Y_val[1], trial_val, fs, labels, locs=groups, recname="ecog_val"))

    return groups, labels, est, val


@xform()
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
    groups, labels, est, val = load_data(recording_uri, downsample)


    # check if cellid exists in recording
    if cellid not in labels:
        print("Labels available: ", labels)
        raise ValueError("cellid not found")

    print(f"Selecting channel {cellid}")

    # Pick out specified electrode
    est = nems0.recording.Recording({"stim": est.signals["stim"],
                                    "resp": est.signals["resp"].extract_channels([cellid])})
    val = nems0.recording.Recording({"stim": val.signals["stim"],
                                    "resp": val.signals["resp"].extract_channels([cellid])})

    return {'est': est, 'val': val}
