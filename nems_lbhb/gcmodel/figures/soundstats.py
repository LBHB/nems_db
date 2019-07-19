import copy

import numpy as np
import matplotlib.pyplot as plt

import nems.recording
import nems.db as nd
import nems.epoch as ep
import nems_lbhb.xform_wrappers as xwrap


def spectrogram_mean_sd(spectrogram, max_db_scale=65, pre_log_floor=1):
    summed = np.sum(spectrogram, axis=0)
    summed[summed <= 1] = 1
    log_spec = np.log2(summed)
    scaled_spec = log_spec * (max_db_scale/log_spec.max())
    mean = np.nanmean(scaled_spec)
    sd = np.nanstd(scaled_spec)

    return mean, sd


def mean_sd_per_stim_by_cellid(cellid, batch, loadkey='ozgf.fs100.ch18',
                               max_db_scale=65, pre_log_floor=1):
    rec_path = xwrap.generate_recording_uri(cellid, batch, loadkey=loadkey)
    rec = nems.recording.load_recording(rec_path)
    stim = copy.deepcopy(rec['stim'].as_continuous())
    fs = rec['stim'].fs
    epochs = rec.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    pre_silence = _silence_duration(epochs, 'PreStimSilence')
    post_silence = _silence_duration(epochs, 'PostStimSilence')

    results = {}
    for s in stim_epochs:
        row = epochs[epochs.name == s]
        start = int((row['start'].values[0] + pre_silence)*fs)
        end = int((row['end'].values[0] - post_silence)*fs)
        results[s] = spectrogram_mean_sd(stim[:, start:end],
                                         max_db_scale=max_db_scale,
                                         pre_log_floor=pre_log_floor)

    return results


def scatter_soundstats(results):
    means = []
    sds = []
    for k, (mean, sd) in results.items():
        means.append(mean)
        sds.append(sd)

    plt.scatter(sds, means)
    plt.ylabel('mean level')
    plt.xlabel('std')


def _silence_duration(epochs, prepost):
    start = epochs[epochs.name == prepost]['start']
    end = epochs[epochs.name == prepost]['end']
    duration = (end.values - start.values).flatten()[0]
    return duration


def relative_gain_by_batch(batch, loadkey='ozgf.fs100.ch18'):
    # get cellids list
    cellids = nd.get_batch_cells(batch)

    # load stim/resp for full batch
    recs = {c: nems.recording.load_recording(
                       xwrap.generate_recording_uri(c, batch, loadkey))
            for c in cellids}

    # break up into epochs by stim, remove pre/post silence
    sigs = {c: stim_resp_per_epoch(r) for c, r in recs.items()}

    # calc. stim means and sds for all cells, stims
    stim_m_sd = {c: spectrogram_mean_sd(s[0]) for c, s in sigs.items()}

    # pick set of reference stims (sd close to 9)


    # within each cell:

        # within each (non-reference) stim:

            # stack by repetitions

            # calculate signal power for response

            # calculate relative gain

    # average across cells

    # 3d plot sd, mean, gain of each stim


def stim_resp_per_epoch(rec):
    stim = copy.deepcopy(rec['stim'].as_continuous())
    resp = copy.deepcopy(rec['resp'].as_continuous())
    fs = rec['stim'].fs
    epochs = rec.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    pre_silence = _silence_duration(epochs, 'PreStimSilence')
    post_silence = _silence_duration(epochs, 'PostStimSilence')

    stims = []
    resps = []
    for s in stim_epochs:
        row = epochs[epochs.name == s]
        start = int((row['start'].values[0] + pre_silence)*fs)
        end = int((row['end'].values[0] - post_silence)*fs)

        stims.append(stim[:, start:end])
        resps.append(resp[start:end])

    return stims, resps
