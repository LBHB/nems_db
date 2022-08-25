from nems.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.projects.olp.OLP_analysis as olp


def intro_figure_spectrograms():
    '''Plots the spectrograms in a format I use for my intro slides (first panel of APAN/SFN
    poster where a few spectrograms(piano, violin, bass) are shown separately and then
    over top of each other. Moved from scratch file 2022_08_25'''
    path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set3/'
    piano_path = path + 'cat268_rec1_classicalsolo_haydn_piano-sonata-53_24sec_excerpt1.wav'
    bass_path = path + 'cat20_rec1_acoustic_bass_gillespie_bass_solo_excerpt1.wav'
    violin_path = path + 'cat394_rec1_violin_excerpt1.wav'
    paths = [piano_path, violin_path, bass_path]

    fig, ax = plt.subplots(5, 1, figsize=(7.5, 10))
    specs = []
    for nn, pth in enumerate(paths):
        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, 48, 0, 12000)
        ax[nn].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                      cmap='gray_r')
        ax[nn].set_yticks([]), ax[nn].set_xticks([])
        specs.append(spec)

    comb = specs[0] + specs[1] + specs[2]
    ax[4].imshow(comb, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[3].set_yticks([]), ax[3].set_xticks([])
    ax[3].spines['top'].set_visible(False), ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['left'].set_visible(False), ax[3].spines['right'].set_visible(False)
    ax[4].set_yticks([]), ax[4].set_xticks([])
    ax[4].set_ylabel('Frequency (Hz)'), ax[4].set_xlabel('Time (s)')
    ax[1].set_ylabel('Frequency (Hz)'), ax[2].set_xlabel('Time (s)')


def intro_figure_waveforms():
    '''Same figure as above except plots some grey waveforms instead of the spectrograms.
    Added 2022_02_07 for WIP. Moved here from scratch file 2022_08_25.'''
    path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set3/'
    piano_path = path + 'cat268_rec1_classicalsolo_haydn_piano-sonata-53_24sec_excerpt1.wav'
    bass_path = path + 'cat20_rec1_acoustic_bass_gillespie_bass_solo_excerpt1.wav'
    violin_path = path + 'cat394_rec1_violin_excerpt1.wav'
    paths = [piano_path, violin_path, bass_path]

    fig, ax = plt.subplots(5, 1, figsize=(7.5, 10), sharey=True)
    waves = []
    for nn, pth in enumerate(paths):
        sfs, W = wavfile.read(pth)
        ax[nn].plot(W, color='dimgrey', lw=1)
        ax[nn].set_yticks([]), ax[nn].set_xticks([])
        waves.append(W)
    comb = waves[0] + waves[1] + waves[2]
    ax[4].plot(comb, color='dimgrey', lw=1)
    ax[3].set_yticks([]), ax[3].set_xticks([])
    ax[3].spines['top'].set_visible(False), ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['left'].set_visible(False), ax[3].spines['right'].set_visible(False)
    ax[4].set_yticks([]), ax[4].set_xticks([])
    ax[4].set_ylabel('Frequency (Hz)'), ax[4].set_xlabel('Time (s)')
    ax[1].set_ylabel('Frequency (Hz)'), ax[2].set_xlabel('Time (s)')


def methods_figure_waveforms():
    '''Plots colorful waveforms of BG, FG, BG+FG using marmoset vocalizations for now. Moved
    from scratch file 2022_08_25'''
    path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
    BG_path = path + 'Background2/10Wind.wav'
    FG_path = path + 'Foreground3/07Chirp.wav'

    paths, wavs, colors = [BG_path, FG_path], [], ['deepskyblue', 'yellowgreen', 'dimgray']
    tags = ['Sound Texture -\nBackground (BG)', 'Marmoset Vocalization -\nForeground (FG)',
            'BG+FG Combination']

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6))
    for nn, pth in enumerate(paths):
        sfs, W = wavfile.read(pth)
        ax[nn].plot(W, color=colors[nn])

        ax[nn].spines['top'].set_visible(False), ax[nn].spines['bottom'].set_visible(False)
        ax[nn].spines['left'].set_visible(False), ax[nn].spines['right'].set_visible(False)
        ax[nn].set_yticks([]), ax[nn].set_xticks([])
        ax[nn].set_title(f"{tags[nn]}", fontweight='bold', fontsize=18)

        wavs.append(W)

    comb = wavs[0] + wavs[1]
    ax[2].plot(comb, color=colors[2])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)
    ax[2].set_yticks([]), ax[2].set_xticks([])
    ax[2].set_title(f"{tags[2]}", fontweight='bold', fontsize=18)
    fig.tight_layout()


def normalized_linear_error_figure(parmfile='/auto/data/daq/Tabor/TBR012/TBR012a14_p_OLP.m'):
    '''Panel 3 of APAN/SFN 2021 posters. Outputs three plots, two of which are spectrograms
    above with resultant PSTH and NLE below, the other is for a given site the summary
    heat map. Moved from scratch file 2022_08_25.'''
    response, params = olp._response_params(parmfile)
    olp.psth_comp_figure([0, 1, 2], 0, 12, response, params, 2, True)
    olp.psth_comp_figure([0, 1, 2], 0, 20, response, params, 2, True)
    olp.z_heatmaps_onepairs_figure([0, 1, 2], 0, response, params, tags=[12, 20], sigma=2, arranged=True)


def plot_electrode_shank():
    '''Makes a little graphic of the 64ch arrays. Moved froms scratch file 2022_08_25.'''
    from nems_lbhb.plots import plot_weights_64D
    plot_weights_64D(np.zeros(64),[f'AMT001a-{x}-1' for x in range(64)])