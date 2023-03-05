import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nems0.recording
from nems0.utils import ax_remove_box
import nems0.db as nd
import nems0.epoch as ep
import nems_lbhb.xform_wrappers as xwrap
from nems_lbhb.gcmodel.figures.definitions import *

plt.rcParams.update(params)


def spectrogram_mean_sd(spectrogram, max_db_scale=65, pre_log_floor=1):
    summed = np.sum(spectrogram, axis=0)
    summed[summed <= 1] = 1
    log_spec = np.log2(summed)
    scaled_spec = log_spec * (max_db_scale/log_spec.max())
    mean = np.nanmean(scaled_spec)
    sd = np.nanstd(scaled_spec)

    return mean, sd


def mean_sd_per_stim_by_cellid(cellid, batch, loadkey='ozgf.fs100.ch18',
                               max_db_scale=65, pre_log_floor=1,
                               stims_to_skip=[]):
    rec_path = xwrap.generate_recording_uri(cellid, batch, loadkey=loadkey)
    rec = nems0.recording.load_recording(rec_path)
    stim = copy.deepcopy(rec['stim'].as_continuous())
    fs = rec['stim'].fs
    epochs = rec.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    stim_epochs = [s for s in stim_epochs if s not in stims_to_skip]
    pre_silence = silence_duration(epochs, 'PreStimSilence')
    post_silence = silence_duration(epochs, 'PostStimSilence')

    results = {}
    for s in stim_epochs:
        row = epochs[epochs.name == s]
        start = int((row['start'].values[0] + pre_silence)*fs)
        end = int((row['end'].values[0] - post_silence)*fs)
        results[s] = spectrogram_mean_sd(stim[:, start:end],
                                         max_db_scale=max_db_scale,
                                         pre_log_floor=pre_log_floor)

    return results


def mean_sd_per_stim_by_batch(batch, loadkey='ozgf.fs100.ch18', max_db_scale=65,
                              pre_log_floor=1, test_limit=None, save_path=None,
                              load_path=None, manual_lims=False):
    if load_path is None:
        cellids = nd.get_batch_cells(batch, as_list=True)
        batch_results = {}
        stims_to_skip = []
        for c in cellids[:test_limit]:
            # wastes some time calculating repeat stims, but oh well...
            # doesn't take that long anyway
            results = mean_sd_per_stim_by_cellid(c, batch, loadkey, max_db_scale,
                                                 pre_log_floor, stims_to_skip)
            batch_results.update(results)
            stims_to_skip = list(batch_results.keys())
        df_dict = {'stim': list(batch_results.keys()),
                   'stats': list(batch_results.values())}
        df = pd.DataFrame.from_dict(df_dict)
        df.set_index('stim', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)
        stims = df.index.values.tolist()
        stats = df['stats'].values.tolist()
        batch_results = {k: v for k, v in zip(stims, stats)}

    fig = scatter_soundstats(batch_results, manual_lims=manual_lims)
    fig2 = plt.figure()
    text = ("x: mean level (db SPL)\n"
            "y: std (dB SPL)\n"
            "batch: %d\n" % batch)
    plt.text(0.1, 0.5, text)
    return fig, fig2


def scatter_soundstats(results, legend=False, manual_lims=False):
    # HACK to get separate markers for batch 263 noisy vs clean dsounds
    voc_in_noise = False
    for k in results:
        if '0dB' in k:
            voc_in_noise = True
            break

    if voc_in_noise:
        clean_means = []
        clean_sds = []
        noisy_means = []
        noisy_sds = []
        for k, (mean, sd) in results.items():
            if '0dB' in k:
                noisy_means.append(mean)
                noisy_sds.append(sd)
            else:
                clean_means.append(mean)
                clean_sds.append(sd)
        fig = plt.figure(figsize=small_fig)
        plt.scatter(clean_sds, clean_means, color='black',
                    s=big_scatter, label='clean')
        plt.scatter(noisy_sds, noisy_means, color=model_colors['combined'],
                    s=small_scatter, label='noisy')
        if legend:
            plt.legend()
        if manual_lims:
            plt.xlim(0,32)
            plt.ylim(0,65)

    else:
        means = []
        sds = []
        for k, (mean, sd) in results.items():
            means.append(mean)
            sds.append(sd)

        fig = plt.figure(figsize=small_fig)
        plt.scatter(sds, means, color='black', s=big_scatter)
        if manual_lims:
            plt.xlim(0,32)
            plt.ylim(0,65)

    plt.tight_layout()
    ax_remove_box()

    return fig


def silence_duration(epochs, prepost):
    start = epochs[epochs.name == prepost]['start']
    end = epochs[epochs.name == prepost]['end']
    duration = (end.values - start.values).flatten()[0]
    return duration


def relative_gain_by_batch(batch, loadkey='ozgf.fs100.ch18'):
    # get cellids list
    cellids = nd.get_batch_cells(batch)

    # load stim/resp for full batch
    recs = {c: nems0.recording.load_recording(
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
