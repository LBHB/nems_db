import matplotlib.pyplot as plt
import numpy as np

import nems.epoch as ep
from nems.plots.utils import ax_remove_box
from nems.recording import load_recording
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.gcmodel.contrast import make_contrast_signal
from nems_lbhb.gcmodel.drc import rec_from_DRC
from nems_lbhb.gcmodel.figures.soundstats import silence_duration
from nems_lbhb.gcmodel.figures.definitions import *

plt.rcParams.update(params)


# TODO: this name and doc aren't accurate anymore. it just plots
#       stim, contrast, and summed contrast for nat and voc sets
def test_DRC_with_contrast(ms=30, normalize=True, fs=100, bands=1,
                           percentile=70, n_segments=8, example_batch=289,
                           example_cell='TAR010c-13-1', voc_batch=263,
                           voc_cell='tul034b-b1'):
    '''
    Plot a sample DRC stimulus next to assigned contrast
    and calculated contrast.
    '''
    loadkey = 'ozgf.fs%d.ch18' % fs
    nat_plot_bins = 1050
    voc_plot_bins = 1110
    nat_seconds = np.arange(0, nat_plot_bins)/fs
    voc_seconds = np.arange(0, voc_plot_bins)/fs
    recording_uri = generate_recording_uri(cellid=example_cell,
                                           batch=example_batch,
                                           loadkey=loadkey, stim=True)
    nat_rec = load_recording(recording_uri)
    nat_rec = make_contrast_signal(nat_rec, name='continuous', continuous=True,
                                   ms=ms, percentile=percentile, bands=bands,
                                   normalize=normalize)

    epochs = nat_rec['resp'].epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    pre_silence = silence_duration(epochs, 'PreStimSilence')
    post_silence = silence_duration(epochs, 'PostStimSilence')
    indices = np.arange(20, dtype=np.int32)
    for s in stim_epochs:
        row = epochs[epochs.name == s]
        st = row['start'].values[0]
        end = row['end'].values[0]
        stim_start = int((st + pre_silence)*fs)
        stim_end = int((end - post_silence)*fs)
        indices = np.append(indices, np.arange(stim_start-20, stim_end+20,
                                   dtype=np.int32))

    nat_stim = nat_rec['stim'].as_continuous()[:, indices][:, :nat_plot_bins]
    nat_contrast = nat_rec['continuous'].as_continuous()[:, indices][:, :nat_plot_bins]
    nat_summed = np.sum(nat_contrast, axis=0)
    nat_summed /= np.max(nat_summed) # norm 0 to 1 just to match axes


    voc_rec_uri = generate_recording_uri(cellid=voc_cell, batch=voc_batch,
                                         loadkey=loadkey, stim=True)
    voc_rec = load_recording(voc_rec_uri)
    voc_rec = make_contrast_signal(voc_rec, name='continuous', continuous=True,
                                   ms=ms, percentile=percentile, bands=bands,
                                   normalize=normalize)

    # Force voc and noise to be interleaved for visualization
    epochs = voc_rec.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    vocs = sorted([s for s in stim_epochs if '0dB' not in s])
    noise = sorted(list(set(stim_epochs) - set(vocs)))
    indices = np.array([], dtype=np.int32)
    for v, n in zip(vocs, noise):
        voc_row = epochs[epochs.name == v]
        voc_start = int(voc_row['start'].values[0]*fs)
        voc_end = int(voc_row['end'].values[0]*fs)
        indices = np.append(indices, np.arange(voc_start, voc_end,
                                               dtype=np.int32))

        noise_row = epochs[epochs.name == n]
        noise_start = int(noise_row['start'].values[0]*fs)
        noise_end = int(noise_row['end'].values[0]*fs)
        indices = np.append(indices, np.arange(noise_start, noise_end,
                                               dtype=np.int32))


    voc_stim = voc_rec['stim'].as_continuous()[:, indices][:, :voc_plot_bins]
    voc_contrast = voc_rec['continuous'].as_continuous()[:, indices][:, :voc_plot_bins]
    voc_summed = np.sum(voc_contrast, axis=0)
    voc_summed /= np.max(voc_summed) # norm 0 to 1 just to match axes


    fig, ((a2,a3), (a5,a6), (a8,a9)) = plt.subplots(3,2, figsize=(3.5,4))

    # Natural Sound
    plt.sca(a2)
    #plt.title('Nat. Sound')
    plt.imshow(nat_stim, cmap=spectrogram_cmap, aspect='auto', origin='lower')
    a2.get_xaxis().set_visible(False)
    a2.get_yaxis().set_visible(False)
    ax_remove_box(a2)
    #plt.ylabel('Freq. Channel')

    plt.sca(a5)
    #plt.title('Contrast')
    plt.imshow(nat_contrast, aspect='auto', origin='lower', cmap=contrast_cmap)
    a5.get_xaxis().set_visible(False)
    a5.get_yaxis().set_visible(False)
    ax_remove_box(a5)

    plt.sca(a8)
    #plt.title('Summed')
    plt.plot(nat_seconds, nat_summed, color='black', linewidth=0.5)
    a8.set_ylim(-0.1, 1.1)
    a8.get_yaxis().set_visible(False)
    #plt.xlabel('Time (s)')
    #plt.ylabel('Summed Contrast (A.U.)')
    a8.set_xlim(nat_seconds.min(), nat_seconds.max())
    ax_remove_box(a8)


    # Voc in noise
    plt.sca(a3)
    #plt.title('Voc. in Noise')
    plt.imshow(voc_stim, cmap=spectrogram_cmap, aspect='auto', origin='lower')
    a3.get_xaxis().set_visible(False)
    a3.get_yaxis().set_visible(False)
    ax_remove_box(a3)

    plt.sca(a6)
    #plt.title('Continuous Calculated Contrast')
    plt.imshow(voc_contrast, aspect='auto', origin='lower', cmap=contrast_cmap)
    a6.get_xaxis().set_visible(False)
    a6.get_yaxis().set_visible(False)
    ax_remove_box(a6)

    plt.sca(a9)
    #plt.title('Summed')
    plt.plot(voc_seconds, voc_summed, color='black', linewidth=0.5)
    a9.set_ylim(-0.1, 1.1)
    #a9.get_yaxis().tick_right()
    a9.get_yaxis().set_visible(False)
    a9.set_xlim(voc_seconds.min(), voc_seconds.max())
    ax_remove_box(a9)


    fig2 = plt.figure()
    text = ("top: spectrogram\n"
            "middle: contrast\n"
            "bottom: summed contrast\n"
            "left: 289, right: 263\n"
            "x: Time (s)\n"
            "summed 0 to 1, contrast arb., spec freq increasing")
    plt.text(0.1, 0.5, text)

    return fig, fig2
