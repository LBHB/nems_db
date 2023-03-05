from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.projects.olp.OLP_analysis as olp
import nems_lbhb.projects.olp.OLP_plot_helpers as oph
import seaborn as sb
import scipy.ndimage.filters as sf
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.projects.olp.OLP_helpers as ohel
import copy
from scipy import stats
import glob
import pandas as pd

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))


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


def intro_figure_spectrograms_wip():
    '''Updated the figure I like to use as an intro to make it relateable to the people at a WIP.
    Plots the spectrograms in a format I use for my intro slides (first panel of APAN/SFN
    poster where a few spectrograms(piano, violin, bass) are shown separately and then
    over top of each other. Moved from scratch file 2022_08_25'''
    path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@NaturalSounds/'
    chips_path = path + 'sounds_set3/cat102_rec1_crumpling_paper_excerpt1.wav'
    can_path = path + 'sounds_set4/cat588_rec1_can_opener_excerpt1.wav'
    voice_path = path + 'sounds_set3/00cat232_rec1_man_speaking_excerpt1.wav'
    paths = [chips_path, can_path, voice_path]

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


def intro_figure_spectrograms_two():
    '''2022_11_24. Plots two spectrograms and then the overlapping, useful for my experiment schematic.'''
    path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
    piano_path = path + 'Background2/03Insect Buzz.wav'
    bass_path = path + 'Foreground3/01Fight Squeak.wav'
    paths = [piano_path, bass_path]

    fig, ax = plt.subplots(5, 1, figsize=(7.5, 10))
    specs = []
    for nn, pth in enumerate(paths):
        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, 48, 0, 12000)
        ax[nn].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                      cmap='gray_r')
        ax[nn].set_yticks([]), ax[nn].set_xticks([])
        specs.append(spec)
        ax[nn].spines['top'].set_visible(True), ax[nn].spines['bottom'].set_visible(True)
        ax[nn].spines['left'].set_visible(True), ax[nn].spines['right'].set_visible(True)

    comb = specs[0] + specs[1]
    ax[4].imshow(comb, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[3].set_yticks([]), ax[3].set_xticks([])
    ax[3].spines['top'].set_visible(False), ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['left'].set_visible(False), ax[3].spines['right'].set_visible(False)
    ax[4].set_yticks([]), ax[4].set_xticks([])
    ax[4].spines['top'].set_visible(True), ax[4].spines['bottom'].set_visible(True)
    ax[4].spines['left'].set_visible(True), ax[4].spines['right'].set_visible(True)
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


def methods_figure_waveforms(animal='Ferret'):
    '''Plots colorful waveforms of BG, FG, BG+FG using marmoset vocalizations for now. Moved
    from scratch file 2022_08_25'''
    path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
    BG_path = path + 'Background2/10Wind.wav'
    FG_path = path + 'Foreground3/07Chirp.wav'

    paths, wavs, colors = [BG_path, FG_path], [], ['deepskyblue', 'yellowgreen', 'dimgray']
    if animal=='Ferret':
        tags = ['Sound Texture - Background (BG)', 'Transient - Foreground (FG)',
            'BG+FG Combination']
    elif animal=='Marmoset':
        tags = ['Sound Texture -\nBackground (BG)', 'Marmoset Vocalization -\nForeground (FG)',
            'BG+FG Combination']

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,4))
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


def plot_linear_model_pieces_helper(df, cellid, bg, fg):
    '''Plots the pieces that I rearrange to make the little schematic of the weighted model.
    Previously it was these two functions just chilling in OLP_plot_helpers with no clear
    connection between them. So to simplify stuff, put this in one spot. Makes panel 5 of
    APAN/SFN 2021 Posters.'''
    cell_string = oph.get_cellstring(cellid, bg, fg, df)
    oph.plot_model_diagram_parts(cell_string, df)


def plot_modspec_stats(sound_df):
    '''Uses the dataframe of all the used sounds and plots the modulation spectra
    collapses across time and freq in the left columns, the cumulative sum in the
    center column, and the median value in the rightmost. With BG on top row and
    FG on bottom row. Figure was used in my DAC in May 2022, cleaned up and made
    work with sound_df 2022_08_26.'''
    tbins, fbins = 100, 48
    wt = np.fft.fftshift(np.fft.fftfreq(tbins, 1 / tbins))
    wf = np.fft.fftshift(np.fft.fftfreq(fbins, 1 / 6))
    wt2 = wt[50:70]
    wf2 = wf[24:]

    bgs, fgs = sound_df.loc[sound_df['type'] == 'BG'], sound_df.loc[sound_df['type'] == 'FG']

    f, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax = axes.ravel()
    for aa in bgs['avgwt']:
        ax[0].plot(wt2, aa, color='deepskyblue')
    for aa in fgs['avgwt']:
        ax[0].plot(wt2, aa, color='yellowgreen')
    ax[0].set_xlabel('wt (Hz)', fontweight='bold', fontsize=8)
    ax[0].set_ylabel('Average', fontweight='bold', fontsize=8)

    for aa in bgs['cumwt']:
        ax[1].plot(aa, color='deepskyblue')
    for aa in fgs['cumwt']:
        ax[1].plot(aa, color='yellowgreen')
    ax[1].set_ylabel('Cumulative Sum', fontweight='bold', fontsize=8)
    ax[1].set_xlabel('wt (Hz)', fontweight='bold', fontsize=8)
    ax[1].set_title('BGs', fontsize=10, fontweight='bold')

    ax[2].boxplot([bgs['t50'], fgs['t50']], labels=['BG', 'FG'])
    ax[2].set_ylabel('Median', fontweight='bold', fontsize=8)

    for aa in bgs['avgft']:
        ax[3].plot(wf2, aa, color='deepskyblue')
    for aa in fgs['avgft']:
        ax[3].plot(wf2, aa, color='yellowgreen')
    ax[3].set_xlabel('wf (cycles/s)', fontweight='bold', fontsize=8)
    ax[3].set_ylabel('Average', fontweight='bold', fontsize=8)

    for aa in bgs['cumft']:
        ax[4].plot(aa, color='deepskyblue')
    for aa in fgs['cumft']:
        ax[4].plot(aa, color='yellowgreen')
    ax[4].set_ylabel('Cumulative Sum', fontweight='bold', fontsize=8)
    ax[4].set_xlabel('wf (cycles/s)', fontweight='bold', fontsize=8)
    ax[4].set_title('FGs', fontsize=10, fontweight='bold')

    ax[5].boxplot([bgs['f50'], fgs['f50']], labels=['BG', 'FG'])
    ax[5].set_ylabel('Median', fontweight='bold', fontsize=8)
    f.tight_layout()


def plot_modspec_stats_bar(sound_df):
    '''Takes your sound_df and plots wt and wf as a bar plot where color differentiates
    BG and FG. Moved from OLP_analysis_main 2022_08_31'''
    fig, ax = plt.subplots(2, 1, figsize=(5, 8))

    sb.barplot(x='name', y='t50', palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
               data=sound_df, ci=68, ax=ax[0], errwidth=1)
    ax[0].set_xticklabels(sound_df.name, rotation=90, fontweight='bold', fontsize=7)
    ax[0].set_ylabel('wt (Hz)', fontweight='bold', fontsize=12)
    ax[0].spines['top'].set_visible(True), ax[0].spines['right'].set_visible(True)
    ax[0].set(xlabel=None)

    sb.barplot(x='name', y='f50',
               palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
               data=sound_df, ax=ax[1])
    ax[1].set_xticklabels(sound_df.name, rotation=90, fontweight='bold', fontsize=7)
    ax[1].set_ylabel('wf (cycles/s)', fontweight='bold', fontsize=12)
    ax[1].spines['top'].set_visible(True), ax[1].spines['right'].set_visible(True)
    ax[1].set(xlabel=None)

    fig.tight_layout()


def plot_mod_spec(idx, df, lfreq=100, hfreq=2400, fbins=48,
                  tbins=100, t=1):
    '''Moved from OLP_Binaural_plot 2022_08_26. Plots a spectrogram above and
    the modulation power spectrum below. You have to pick a numerical index for
    the spectrum. Could change it so you type a name, but meh for now.'''
    row = df.iloc[idx]
    spec = row['spec']
    mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))

    # oct = np.log2(hfreq/lfreq)
    # fmod = (bins/oct) / 2
    tmod = (tbins/t) / 2
    xbound = tmod*0.4

    wt = np.fft.fftshift(np.fft.fftfreq(tbins, 1/tbins))
    wf = np.fft.fftshift(np.fft.fftfreq(fbins, 1/6))

    f, ax = plt.subplots(2, 1, figsize=(5, 5))
    ax[0].imshow(spec, aspect='auto', origin='lower')
    ax[1].imshow(np.sqrt(mod), aspect='auto', origin='lower',
                 extent=(wt[0]+0.5, wt[-1]+0.5, wf[0], wf[-1]))
    ax[1].set_xlim(-xbound, xbound)
    ax[1].set_ylim(0,np.max(wf))
    ax[1].set_ylabel("wf (cycles/s)", fontweight='bold', fontsize=10)
    ax[1].set_xlabel("wt (Hz)", fontweight='bold', fontsize=10)
    ax[0].set_title(f"{row['name']}", fontweight='bold', fontsize=14)


def plot_sorted_modspecs(weight_df, sound_df, by='self'):
    '''Prepares the sound_df in the order of however you choose to sort the weights. This can
    be specified by either putting 'self' or 'effect' for by, which will either sort the sounds
    in ascending order of the weight they are given or descending order of the weight they cause
    in a paired sound, respectively. Moved from OLP_analysis_main 2022_08_26'''
    quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=0.03, quad_return=3)
    quad = quad.copy()
    sound_df = sound_df.copy()

    bgsub = quad[['BG', 'weightsA', 'weightsB']].copy()
    fgsub = quad[['FG', 'weightsB', 'weightsA']].copy()

    bgsub.rename(columns={'BG':'name', 'weightsA':'selfweight', 'weightsB':'effectweight'}, inplace=True)
    fgsub.rename(columns={'FG':'name', 'weightsB':'selfweight', 'weightsA':'effectweight'}, inplace=True)
    weights = pd.concat([bgsub, fgsub], axis=0)
    means = weights.groupby('name').agg('mean')
    selfy = weights.groupby('name').agg(selfweight=('selfweight',np.mean)).reset_index()
    effect = weights.groupby('name').agg(effectweight=('effectweight',np.mean)).reset_index()

    fn = lambda x: x[2:].replace(' ', '')
    sound_df['sound'] = sound_df.name.apply(fn)
    sound_df.rename(columns={'name':'fullname', 'sound':'name'}, inplace=True)

    selfsounds = selfy.merge(sound_df, on='name').sort_values('selfweight')
    effectsounds = effect.merge(selfsounds, on='name').sort_values('effectweight', ascending=False)
    self_sort = selfsounds.name
    effect_sort = effectsounds.name

    if by == 'self':
        sorted_modspecs(self_sort, selfsounds, 'selfweight')
    elif by == 'effect':
        sorted_modspecs(effect_sort, effectsounds, 'effectweight')
    else:
        raise ValueError(f"Can't use {by}, must use 'self' or 'effect'.")


def sorted_modspecs(sorted, sounds, label):
    '''Takes a sorted dataframe (either by ascending sound weight or descending weight that sound
    causes in a paired sound) and plots them going across rows, spectrogram above, modspec below.
    Can only handle a maximum of 39 sounds for now... 2022_08_26. '''
    w, h, t = 13, 3, 1
    tbins, fbins, lfreq, hfreq = 100, 48, 100, 24000

    tmod = (tbins / t) / 2
    xbound = tmod * 0.4
    wt = np.fft.fftshift(np.fft.fftfreq(tbins, 1 / tbins))
    wf = np.fft.fftshift(np.fft.fftfreq(fbins, 1 / 6))

    ##plot mod specs in order ascending of their weight
    f, axes = plt.subplots(h*2, w, figsize=(18,8))
    ax = axes.ravel()
    AX = list(np.arange(0,13)) + list(np.arange(26,39)) + list(np.arange(52,65))

    for aa, snd in zip(AX, sorted.to_list()):
        row = sounds.loc[sounds.name == snd]
        spec = row['spec'].values[0]
        mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))
        # if you want to add back in plotting normalized, it looks bad though...
        # mod = row['normmod'].values[0]

        ax[aa].imshow(spec, aspect='auto', origin='lower')
        ax[aa+13].imshow(np.sqrt(mod), aspect='auto', origin='lower',
                     extent=(wt[0]+0.5, wt[-1]+0.5, wf[0], wf[-1]))
        ax[aa].set_yticks([]), ax[aa].set_xticks([])
        ax[aa+13].set_xlim(-xbound, xbound)
        ax[aa+13].set_ylim(0,np.max(wf))
        if aa == 0 or aa == 13 or aa == 26:
            ax[aa+13].set_ylabel("wf (cycles/s)", fontweight='bold', fontsize=6)
        if aa >= 52:
            ax[aa+13].set_xlabel("wt (Hz)", fontweight='bold', fontsize=6)
        ax[aa].set_title(f"{row['name'].values[0]}: {np.around(row[label].values[0], 3)}",
                         fontweight='bold', fontsize=8)
    # If there are more axes than sounds, make the other subplots go away
    if len(AX) > len(sorted):
        diff = len(AX) - len(sorted)
        for ss in range(1,diff+1):
            ax[AX[-ss]].spines['top'].set_visible(False), ax[AX[-ss]].spines['bottom'].set_visible(False)
            ax[AX[-ss]].spines['left'].set_visible(False), ax[AX[-ss]].spines['right'].set_visible(False)
            ax[AX[-ss]].set_yticks([]), ax[AX[-ss]].set_xticks([])
            ax[AX[-ss]+13].spines['top'].set_visible(False), ax[AX[-ss]+13].spines['bottom'].set_visible(False)
            ax[AX[-ss]+13].spines['left'].set_visible(False), ax[AX[-ss]+13].spines['right'].set_visible(False)
            ax[AX[-ss]+13].set_yticks([]), ax[AX[-ss]+13].set_xticks([])



def psths_with_specs(df, cellid, bg, fg, batch=340, bin_kind='11', synth_kind='N',
                     sigma=None, error=True):
    '''Makes panel three of APAN 2021 poster and NGP 2022 poster, this is a better way than the way
    normalized_linear_error_figure in this file does it, which relies on an old way of loading and
    saving the data that doesn't use DFs and is stupid. The other way also only works with marmoset
    and maybe early ferret data, but definitely not binaural and synthetics. Use this, it's better.
    It does lack the linear error stat though, but that's not important anymore 2022_09_01.'''

    # Make figure bones. Could add another spectrogram below, there's space.
    f = plt.figure(figsize=(8, 6))
    psth = plt.subplot2grid((18, 3), (4, 0), rowspan=5, colspan=6)
    specA = plt.subplot2grid((18, 3), (0, 0), rowspan=2, colspan=6)
    specB = plt.subplot2grid((18, 3), (2, 0), rowspan=2, colspan=6)
    ax = [specA, specB, psth]

    tags = ['Background (BG)', 'Foreground (FG)', 'BG+FG Combo']
    colors = ['deepskyblue','yellowgreen','dimgray']

    #Get this particular row out for some stuff down the road
    row = df.loc[(df.cellid==cellid) & (df.BG==bg) & (df.FG==fg) & (df.kind==bin_kind)
                 & (df.synth_kind==synth_kind)].squeeze()

    # Load response
    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    epo = row.epoch
    epochs = [f"STIM_{epo.split('_')[1]}_null", f"STIM_null_{epo.split('_')[2]}", epo]

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100
    fs = 100
    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    r = norm_spont.extract_epochs(epochs)
    ls = np.squeeze(np.nanmean(r[epochs[0]] + r[epochs[1]],axis=0))

    # Some plotting calculations
    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs ) - prestim
    dur = manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['Duration']

    # Plot the three response lines
    for (cnt, kk) in enumerate(r.keys()):
        plot_resp = r[kk]
        mean_resp = np.squeeze(np.nanmean(plot_resp, axis=0))
        if sigma:
            ax[2].plot(time, sf.gaussian_filter1d(mean_resp, sigma) * rec['resp'].fs,
                                               color=colors[cnt], label=f"{tags[cnt]}")
        if not sigma:
            ax[2].plot(time, mean_resp * rec['resp'].fs, color=colors[cnt], label=f"{tags[cnt]}")
        if error:
            sem = np.squeeze(stats.sem(plot_resp, axis=0, nan_policy='omit'))
            ax[2].fill_between(time, sf.gaussian_filter1d((mean_resp - sem) * rec['resp'].fs, sigma),
                            sf.gaussian_filter1d((mean_resp + sem) * rec['resp'].fs, sigma),
                               alpha=0.4, color=colors[cnt])
    # Plot the linear sum line
    if sigma:
        ax[2].plot(time, sf.gaussian_filter1d(ls * rec['resp'].fs, sigma), color='dimgray',
                ls='--', label='Linear Sum')
    if not sigma:
        ax[2].plot(time, ls * rec['resp'].fs, color='dimgray', ls='--', label='Linear Sum')
    ax[2].set_xlim(-0.2, (dur + 0.3))        # arbitrary window I think is nice
    ymin, ymax = ax[2].get_ylim()

    ax[2].set_ylabel('spk/s', fontweight='bold', size=10)
    ax[2].legend(loc='upper right', fontsize=6)
    ax[2].vlines([0, dur], ymin, ymax, colors='black', linestyles=':')
    ax[2].set_ylim(ymin, ymax)
    ax[2].set_xlabel('Time (s)', fontweight='bold', size=10)
    ax[2].spines['top'].set_visible(True), ax[2].spines['right'].set_visible(True)
    # ax[2].vlines(params['SilenceOnset'], ymax * .9, ymax, colors='black', linestyles='-', lw=0.25)

    ax[0].set_title(f"{cellid}", fontweight='bold', size=12)
    xmin, xmax = ax[2].get_xlim()

    # Spectrogram part
    folder_ids = [int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['BG_Folder'][-1]),
            int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['FG_Folder'][-1])]

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Foreground{folder_ids[1]}/*.wav'))
    bg_path = [bb for bb in bg_dir if epo.split('_')[1].split('-')[0][:2] in bb][0]
    fg_path = [ff for ff in fg_dir if epo.split('_')[2].split('-')[0][:2] in ff][0]

    xf = 100
    low, high = xmin * xf, xmax * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[0].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[0].set_xlim(low, high)
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xticklabels([]), ax[0].set_yticklabels([])
    ax[0].spines['top'].set_visible(False), ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False), ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel(f"BG: {row.BG}", rotation=0, fontweight='bold',
                     size=8, labelpad=-35)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[1].set_xticklabels([]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)
    ax[1].set_ylabel(f"FG: {row.FG}", rotation=0, fontweight='bold',
                     size=8, labelpad=-35)

    # This just makes boxes around only the important part of the spec axis. So it all lines up.
    ymin, ymax = ax[1].get_ylim()
    ax[0].vlines([0, 1*rec['resp'].fs], ymin, ymax, color='black', lw=0.5)
    ax[0].hlines([ymin+2,ymax], 0, 1*rec['resp'].fs, color='black', lw=0.5)
    ax[1].vlines([0, 1*rec['resp'].fs], ymin, ymax, color='black', lw=0.5)
    ax[1].hlines([ymin+1,ymax], 0, 1*rec['resp'].fs, color='black', lw=0.5)


def response_heatmaps_comparison(df, site, bg, fg, cellid=None, batch=340, bin_kind='11',
                                 synth_kind='N', sigma=None, example=False, sort=True):
    '''Takes out the BG, FG, combo, diff psth heatmaps from the interactive plot and makes it it's own
    figure. You provide the weight_df, site, and sounds, and then optionally you can throw a cellid
    or list of them in and it'll go ahead and only label those on the y axis so you can better see
    it. Turn example to True if you'd like it to be a more generically titled than using the actually
    epoch names, which are not good for posters. Added 2022_09_01

    Added sorting for the difference panel which will in turn sort all other panels 2022_09_07. Also,
    added mandatory normalization of responses by the max for each unit across the three epochs.'''
    df['expt_num'] = [int(aa[4:6]) for aa in df['cellid']]
    if synth_kind == 'n/a':
        all_cells = df.loc[(df['cellid'].str.contains(site)) & (df.BG == bg) & (df.FG == fg) & (df.kind == bin_kind)]
    else:
        all_cells = df.loc[(df['cellid'].str.contains(site)) & (df.BG==bg) & (df.FG==fg) & (df.kind==bin_kind)
                     & (df.synth_kind==synth_kind)]
    print(f"all_cells is {len(all_cells)}")

    manager = BAPHYExperiment(cellid=all_cells.cellid.unique()[0], batch=batch)
    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    resp = copy.copy(rec['resp'].rasterize())
    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    rec['resp'].fs, fs = 100, 100
    font_size=5

    epo = list(all_cells.epoch)[0]
    epochs = [f"STIM_{epo.split('_')[1]}_null", f"STIM_null_{epo.split('_')[2]}", epo]
    r = norm_spont.extract_epochs(epochs)
    resp_plot = np.stack([np.nanmean(aa, axis=0) for aa in list(r.values())])

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / rec['resp'].fs) - prestim
    dur = manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['Duration']

    # Adding code to normalize each cell to the max of that cell to any of the epochs
    for nn in range(resp_plot.shape[1]):
        # max_val = np.max(np.abs(resp_plot[:,nn,int(prestim*fs):int((prestim+dur)*fs)]))
        max_val = np.max(np.abs(resp_plot[:,nn,:]))
        resp_plot[:,nn,:] = resp_plot[:,nn,:] / max_val

    # Get difference array before smoothing
    ls_array = resp_plot[0,:,:] + resp_plot[1,:,:]
    # diff_array = resp_plot[2,:,:] - resp_plot[1,:,:]
    diff_array = resp_plot[2,:,:] - ls_array

    num_ids = [cc[8:] for cc in all_cells.cellid.tolist()]
    if sort == True:
        sort_array = diff_array[:,int(prestim*fs):int((prestim+dur)*fs)]
        means = list(np.nanmean(sort_array, axis=1))
        indexes = list(range(len(means)))
        sort_df = pd.DataFrame(list(zip(means, indexes)), columns=['mean', 'idx'])
        sort_df = sort_df.sort_values('mean', ascending=False)
        sort_list = sort_df.idx
        diff_array = diff_array[sort_list, :]
        resp_plot = resp_plot[:, sort_list, :]
        num_array = np.asarray(num_ids)
        num_ids = list(num_array[sort_list])

    if cellid:
        if isinstance(cellid, list):
            if len(cellid[0].split('-')) == 3:
                cellid = [aa[8:] for aa in cellid]
            elif len(cellid[0].split('-')) == 2:
                cellid = [aa for aa in cellid]
        num_ids = [ii if ii in cellid else '' for ii in num_ids]
        font_size=8

    # Smooth if you have given it a sigma by which to smooth
    if sigma:
        resp_plot = sf.gaussian_filter1d(resp_plot, sigma, axis=2)
        diff_array = sf.gaussian_filter1d(diff_array, sigma, axis=1)
    # Get the min and max of the array, find the biggest magnitude and set max and min
    # to the abs and -abs of that so that the colormap is centered at zero
    cmax, cmin = np.max(resp_plot), np.min(resp_plot)
    biggest = np.maximum(np.abs(cmax),np.abs(cmin))
    cmax, cmin = np.abs(biggest), -np.abs(biggest)

    # Plot BG, FG, Combo
    fig, axes = plt.subplots(figsize=(9, 12))
    BGheat = plt.subplot2grid((11, 6), (0, 0), rowspan=2, colspan=5)
    FGheat = plt.subplot2grid((11, 6), (2, 0), rowspan=2, colspan=5)
    combheat = plt.subplot2grid((11, 6), (4, 0), rowspan=2, colspan=5)
    diffheat = plt.subplot2grid((11, 6), (7, 0), rowspan=2, colspan=5)
    cbar_main = plt.subplot2grid((11, 6), (2, 5), rowspan=2, colspan=1)
    cbar_diff = plt.subplot2grid((11, 6), (7, 5), rowspan=2, colspan=1)
    ax = [BGheat, FGheat, combheat, diffheat, cbar_main, cbar_diff]

    for (ww, qq) in enumerate(range(0,len(epochs))):
        dd = ax[qq].imshow(resp_plot[ww, :, :], vmin=cmin, vmax=cmax,
                           cmap='bwr', aspect='auto', origin='lower',
                           extent=[time[0], time[-1], 0, len(all_cells)])
        ax[qq].vlines([int(prestim), int(prestim+dur)], ymin=0, ymax=len(all_cells),
                      color='black', lw=1, ls=':')
        # ax[qq].set_ylabel('Unit', fontweight='bold', fontsize=8)
        ax[qq].set_yticks([*range(0, len(all_cells))])
        ax[qq].set_yticklabels(num_ids, fontsize=font_size, fontweight='bold')
        ax[qq].set_xlim(-0.2, (dur + 0.3))  # arbitrary window I think is nice
        if example == True:
            titles = [f"BG - {bg}", f"FG - {fg}", f"Combo\nBG+FG"]
            ax[qq].set_ylabel(f"{titles[ww]}", fontsize=12, fontweight='bold', rotation=90,
                              horizontalalignment='center') # , labelpad=40)
            ax[0].set_title(f'Site {all_cells.iloc[0].cellid[:7]} Responses', fontweight='bold', fontsize=12)
        else:
            ax[qq].set_title(f"{epochs[ww]}", fontsize=8, fontweight='bold')
            ax[qq].set_ylabel('Unit', fontweight='bold', fontsize=8)
        ax[qq].spines['top'].set_visible(True), ax[qq].spines['right'].set_visible(True)
    ax[2].set_xlabel('Time (s)', fontweight='bold', fontsize=9)
    ax[0].set_xticks([]), ax[1].set_xticks([])
    # Add the colorbar to the axis to the right of these, the diff will get separate cbar
    fig.colorbar(dd, ax=ax[4], aspect=7)
    ax[4].spines['top'].set_visible(False), ax[4].spines['right'].set_visible(False)
    ax[4].spines['bottom'].set_visible(False), ax[4].spines['left'].set_visible(False)
    ax[4].set_yticks([]), ax[4].set_xticks([])

    # Plot the difference heatmap with its own colorbar
    dmax, dmin = np.max(diff_array), np.min(diff_array)
    biggestd = np.maximum(np.abs(dmax),np.abs(dmin))
    # dmax, dmin = np.abs(biggestd), -np.abs(biggestd)
    dmax, dmin = 1, -1
    ddd = ax[3].imshow(diff_array, vmin=dmin, vmax=dmax,
                           cmap='PuOr', aspect='auto', origin='lower',
                           extent=[time[0], time[-1], 0, len(all_cells)])
    ax[3].vlines([0, int(dur)], ymin=0, ymax=len(all_cells),
                 color='black', lw=1, ls=':')
    ax[3].set_xlim(-0.2, (dur + 0.3))  # arbitrary window I think is nice
    ax[3].set_yticks([*range(0, len(all_cells))])
    ax[3].set_ylabel('Unit', fontweight='bold', fontsize=9)
    ax[3].set_yticklabels(num_ids, fontsize=font_size, fontweight='bold')
    ax[3].set_title(f"Difference (Combo - Linear Sum)", fontsize=12, fontweight='bold')
    ax[3].set_xlabel('Time (s)', fontsize=9, fontweight='bold')
    ax[3].spines['top'].set_visible(True), ax[3].spines['right'].set_visible(True)

    fig.colorbar(ddd, ax=ax[5], aspect=7)
    ax[5].spines['top'].set_visible(False), ax[5].spines['right'].set_visible(False)
    ax[5].spines['bottom'].set_visible(False), ax[5].spines['left'].set_visible(False)
    ax[5].set_yticks([]), ax[5].set_xticks([])


def resp_weight_scatter(weight_df, xcol='bg_FR', ycol='weightsB', threshold=0.03, quads=3):
    '''2022_09_06. Takes a dataframe and just scatters two of the columns from that dataframe.
    I was using it to check if weights are correlated with the concurrent sound's FR.'''
    quad, _ = ohel.quadrants_by_FR(weight_df, threshold=threshold, quad_return=quads)
    quad = quad.loc[quad.synth_kind == 'N'].copy()
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    sb.scatterplot(x=xcol, y=ycol, data=quad, ax=ax, s=3)
    X, Y = quad[xcol], quad[ycol]
    reg = stats.linregress(X, Y)
    x = np.asarray(ax.get_xlim())
    y = reg.slope*x + reg.intercept
    ax.plot(x, y, color='deepskyblue', label=f"slope: {reg.slope:.3f}\n"
                            f"coef: {reg.rvalue:.3f}\n"
                            f"p = {reg.pvalue:.3f}")
    ax.legend()


def resp_weight_multi_scatter(weight_df, ycol=['weightsA', 'weightsA', 'weightsB', 'weightsB'],
                              synth_kind='N', threshold=0.03, quads=3):
    '''Updated resp_weight_scatter to just plot all four combinations of FR/wt on one plot to avoid
    the silliness of four individual plots. Works the same, basically just give it a df.'''
    quad, _ = ohel.quadrants_by_FR(weight_df, threshold=threshold, quad_return=quads)
    quad = quad.loc[quad.synth_kind == synth_kind].copy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    xcol = ['bg_FR', 'fg_FR', 'bg_FR', 'fg_FR']
    # ycol = ['weightsA', 'weightsA', 'weightsB', 'weightsB']
    cols = ['deepskyblue', 'deepskyblue', 'yellowgreen', 'yellowgreen']

    for ax, xc, yc, col in zip(axes, xcol, ycol, cols):
        sb.scatterplot(x=xc, y=yc, data=quad, ax=ax, s=3, color=col)

        X, Y = quad[xc], quad[yc]
        reg = stats.linregress(X, Y)
        x = np.asarray(ax.get_xlim())
        y = reg.slope * x + reg.intercept
        ax.plot(x, y, color='deepskyblue', label=f"slope: {reg.slope:.3f}\n"
                                                 f"coef: {reg.rvalue:.3f}\n"
                                                 f"p = {reg.pvalue:.3f}")
        ax.legend()

    axes[0].set_ylabel(f"{ycol[0]}", fontsize=12, fontweight='bold')
    axes[2].set_ylabel(f"{ycol[2]}", fontsize=12, fontweight='bold')
    axes[2].set_xlabel(f"{xcol[0]}", fontsize=12, fontweight='bold')
    axes[3].set_xlabel(f"{xcol[1]}", fontsize=12, fontweight='bold')


def plot_single_relative_gain_hist(df, threshold=0.05, quad_return=3, synth_kind=None, r_cut=None):
    '''2022_09_06. Takes a DF (you filter it by types of sounds and area beforehand) and will plot
    a histogram showing the relative weights for a certain quadrant. It said distplot is deprecated
    so I'll have to figure something else out with histplot or displot, but coloring the histogram
    was a big task I couldn't figure out. Can kicked down the road.'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=quad_return)
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    if synth_kind:
        quad = quad.loc[quad.synth_kind == synth_kind]
    # Calculate relative gain and percent suppressed
    a1_rel_weight = (quad.weightsB - quad.weightsA) / (quad.weightsB + quad.weightsA)
    a1_supps = [cc for cc in a1_rel_weight if cc < 0]
    a1_percent_supp = np.around((len(a1_supps) / len(a1_rel_weight)) * 100, 1)
    # Filter dataframe to get rid of the couple with super weird, big or small weights
    rel = a1_rel_weight.loc[a1_rel_weight <= 2.5]
    rel = rel.loc[rel >= -2.5]
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(3,4))
    p = sb.distplot(rel, bins=50, color='black', norm_hist=True, kde=False)
    # p = sb.histplot(data=rel_df, x='rel', hue='kind', bins=50, color='black', element='step', kde=False)
    ymin, ymax = ax.get_ylim()
    ax.vlines(0, ymin, ymax, color='black', ls = '--', lw=1)
    # Change color of suppressed to red, enhanced to blue
    for rectangle in p.patches:
        if rectangle.get_x() < 0:
            rectangle.set_facecolor('tomato')
    for rectangle in p.patches:
        if rectangle.get_x() >= 0:
            rectangle.set_facecolor('dodgerblue')
    ax.set_xlabel('Relative Gain', fontweight='bold', fontsize=12)
    ax.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax.set_title(f'{synth_kind} - Percent\nSuppression:\n{a1_percent_supp}', fontsize=8)
    if r_cut:
        fig.suptitle(f"r >= {r_cut}")
    fig.tight_layout()


def sound_metric_scatter(df, x_metrics, y_metric, x_labels, area='A1', threshold=0.03,
                         jitter=[0.25,0.2,0.03],
                         quad_return=3, metric_filter=None, synth_kind='N', bin_kind='11',
                         title_text='', r_cut=None):
    '''Updated 2022_09_21 to add the ability to filter the dataframe by model fit accuracy.
    Makes a series of scatterplots that compare a stat of the sounds to some metric of data. In
    a usual situation it would be Tstationariness, bandwidth, and Fstationariness compared to relative
    gain. Can also be compared to weights.
    y_metric refers to the FIRST one it will input, for relative_gain this is not an issue. If you want
    to differentiate between weights the sound affects in others vs how that sound is weighted itself,
    input the one as it relates to BG, so 'weightsB' will be 'how that sound effects others' and will
    know to make the metric 'weightsA' for the FGs, for example.
    When inputting x_metric names, always make it a list. All entries should be found in the df being
    passed, but you should remove the BG_ or FG_ prefix.
    Made into a function from OLP_analysis_main on 2022_09_07'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=quad_return)
    quad = quad.loc[(quad.area==area) & (quad.synth_kind==synth_kind) & (quad.kind==bin_kind)]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    quad = quad.copy()

    # I use 2.5 for relative gain, I'm sure weights have one too...
    if metric_filter:
        quad = quad.loc[quad[y_metric] <= metric_filter]
        quad = quad.loc[quad[y_metric] >= -metric_filter]

    if y_metric=='BG_rel_gain':
        y_metric2, title, ylabel = 'FG_rel_gain', 'Relative Gain', 'Relative Gain'
    elif y_metric=='weightsB':
        y_metric2, title, ylabel = 'weightsA', 'How this sound effects a concurrent sound', 'Weight'
    elif y_metric=='weightsA':
        y_metric2, title, ylabel = 'weightsB', 'How this sound itself is weighted', 'Weight'
    else:
        y_metric2, title, ylabel = y_metric, y_metric, y_metric

    # fig, axes = plt.subplots(1, len(x_metrics), figsize=(len(x_metrics)*5, 6))
    fig, axes = plt.subplots(1, len(x_metrics), figsize=(12, 5))


    for cnt, (ax, met) in enumerate(zip(axes, x_metrics)):
        # Add a column that is the data for that metric, but jittered, for viewability
        quad[f'jitter_BG_{met}'] = quad[f'BG_{met}'] + np.random.normal(0, jitter[cnt], len(quad))
        quad[f'jitter_FG_{met}'] = quad[f'FG_{met}'] + np.random.normal(0, jitter[cnt], len(quad))
        # Do the plotting
        sb.scatterplot(x=f'jitter_BG_{met}', y=y_metric, data=quad, ax=ax, s=4, color='cornflowerblue')
        sb.scatterplot(x=f'jitter_FG_{met}', y=y_metric2, data=quad, ax=ax, s=4, color='olivedrab')
        ax.set_xlabel(x_labels[cnt], fontweight='bold', fontsize=10)
        if cnt==0:
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=10)
        else:
            ax.set_ylabel('')

        # Run a regression
        Y = np.concatenate((quad[y_metric].values, quad[y_metric2].values))
        X = np.concatenate((quad[f'BG_{met}'].values, quad[f'FG_{met}'].values))
        reg = stats.linregress(X, Y)
        x = np.asarray(ax.get_xlim())
        y = reg.slope * x + reg.intercept
        ax.plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                                                 f"coef: {reg.rvalue:.3f}\n"
                                                 f"p = {reg.pvalue:.3f}")
        ax.legend()


    fig.suptitle(f"{title} - {synth_kind} - {title_text} - r >= {r_cut}", fontweight='bold', fontsize=10)


def scatter_model_accuracy(df, stat='FG_rel_gain', bin_kind='11', synth_kind='N', threshold=0.03):
    '''2022_09_16. Takes a dataframe and filters it according to your inputs, then plots the three quadrants
    of positive firing rate as a scatter comparing model accuracy with relative_gain.'''
    df = df.dropna(axis=0, subset='r')
    df = df.loc[df.synth_kind==synth_kind]
    df = df.loc[df.kind==bin_kind]

    labels = ['BG and FG alone\nevoke response', 'Only BG alone\nevokes response', 'Only FG alone\nevokes response']
    quads = [3, 6, 2]
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    for cnt, (ax, qd) in enumerate(zip(axes, quads)):
        quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=qd)
        sb.scatterplot(data=quad, x=quad.r, y=quad[stat], s=3, ax=ax)

        X, Y = quad.r.values, quad[stat].values
        reg = stats.linregress(X, Y)
        x = np.asarray(ax.get_xlim())
        y = reg.slope * x + reg.intercept
        ax.plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                                              f"coef: {reg.rvalue:.3f}\n"
                                              f"p = {reg.pvalue:.3f}")
        ax.legend()
        ax.set_xlabel("Model Fit (r)", fontweight='bold', fontsize=8)
        ax.set_title(f"{labels[cnt]}", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"{stat}", fontweight='bold', fontsize=8)


def r_filtered_weight_histogram_summary(df, synth_kind='N', bin_kind='11', manual=None, cut=2, threshold=0.03):
    '''2022_09_16. Takes a dataframe, removes NaNs from model fit, then filters based on criteria,
    then splits into two dataframes for above and below the median r value. If you include manual,
    it'll add an extra plot for the manually definied lower r cutoff. It'll then pop up three
    separate plots.'''
    r_df = df.dropna(axis=0, subset='r')
    r_df = r_df.loc[r_df.synth_kind==synth_kind]
    r_df = r_df.loc[r_df.kind==bin_kind]
    r_df = r_df.loc[(r_df.weightsA > -cut) & (r_df.weightsA < cut) &
                    (r_df.weightsB > -cut) & (r_df.weightsB < cut)]
    mid_r = np.median(r_df.r)
    goods = r_df.loc[r_df.r >= mid_r]
    bads = r_df.loc[r_df.r <=mid_r]
    to_plot = [goods, bads]
    labels = [f'r >= {mid_r}, {synth_kind}, {bin_kind}', f'r < {mid_r}, {synth_kind}, {bin_kind}']
    if manual:
        extra = r_df.loc[r_df.r >= manual]
        to_plot.append(extra)
        labels.append(f'r >= {manual}, {synth_kind}, {bin_kind}')

    for dff, lbl in zip(to_plot, labels):
        oph.histogram_summary_plot(dff, 0.03, title_text=lbl)



def weights_supp_comp(weight_df, quads=3, thresh=0.03, r_cut=None):
    '''2022_09_30. Uses the usual stuff to calculate the old way of calculating suppression:
    ((rAB-sp) - (rA-sp) + (rB-sp)) / (rA-sp) + (rB-sp), and compares it with average weight:
    (wFG+wBG) / 2. Then scatters them.'''
    weight_df['avg_weight'] = (weight_df.weightsA + weight_df.weightsB) / 2
    weight_df['avg_supp'] = (-weight_df['supp']) / (weight_df['bg_FR'] + weight_df['fg_FR'])
    quad, _ = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.scatter(x=quad.avg_supp, y=quad.avg_weight, s=1)
    ax.set_xlabel('(LS - rAB) / rA+rB', fontweight='bold', fontsize=10)
    ax.set_ylabel('Mean Weights (wFG+wBG)/2', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.25), ax.set_xlim(-1.5, 1.5)
    ax.set_title(f'r >= {r_cut} - n={quad.shape[0]}', fontsize=10, fontweight='bold')
    fig.tight_layout()

def plot_all_weight_comparisons(df, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True):
    '''2022_11_08. Made for SFN/APAN poster panel 4, it displays the different fit epochs across a dataframe labeled
    with multiple different animals. FR and R I used for the poster was 0.03 and 0.6. Strict_r basically should always
    stay True at this point'''
    areas = list(df.area.unique())

    # This can be mushed into one liners using list comprehension and show_suffixes
    quad3 = df.loc[(df.bg_FR_start >= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                           & (df.bg_FR_end >= fr_thresh) & (df.fg_FR_end >= fr_thresh)]

    quad2 = df.loc[(np.abs(df.bg_FR_start) <= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                   & (np.abs(df.bg_FR_end) <= fr_thresh) & (df.fg_FR_end >= fr_thresh)]

    quad6 = df.loc[(df.bg_FR_start >= fr_thresh) & (np.abs(df.fg_FR_start) <= fr_thresh)
                   & (df.bg_FR_end >= fr_thresh) & (np.abs(df.fg_FR_end) <= fr_thresh)]

    fig, ax = plt.subplots(1, 4, figsize=(13, 6), sharey=True)
    ax = np.ravel(ax)

    colors = ['mediumorchid', 'darkorange', 'orangered', 'green']

    stat_list, filt_list = [], []
    for num, aa in enumerate(areas):
        area_df = quad3.loc[quad3.area==aa]
        if strict_r == True:
            filt = area_df.loc[(area_df['r_start'] >= r_thresh) & (area_df['r_end'] >= r_thresh)]
        else:
            filt = area_df.loc[area_df[f"r{ss}"] >= r_thresh]

        if summary == True:
            alph = 0.35
        else:
            alph = 1

        for ee, an in enumerate(list(filt.animal.unique())):
            animal_df = filt.loc[filt.animal == an]
            ax[num].scatter(x=['BG_start', 'BG_end'],
                            y=[np.nanmean(animal_df[f'weightsA_start']), np.nanmean(animal_df[f'weightsA_end'])],
                               label=f'{an} (n={len(animal_df)})', color=colors[ee], alpha=alph)#, marker=symbols[cnt])
            ax[num].scatter(x=['FG_start', 'FG_end'],
                            y=[np.nanmean(animal_df[f'weightsB_start']), np.nanmean(animal_df[f'weightsB_end'])],
                               color=colors[ee], alpha=alph) #, marker=symbols[cnt])
            ax[num].errorbar(x=['BG_start', 'BG_end'],
                            y=[np.nanmean(animal_df[f'weightsA_start']), np.nanmean(animal_df[f'weightsA_end'])],
                           yerr=[stats.sem(animal_df[f'weightsA_start']), stats.sem(animal_df[f'weightsA_end'])], xerr=None,
                           color=colors[ee], alpha=alph)
            ax[num].errorbar(x=['FG_start', 'FG_end'],
                            y=[np.nanmean(animal_df[f'weightsB_start']), np.nanmean(animal_df[f'weightsB_end'])],
                           yerr=[stats.sem(animal_df[f'weightsB_start']), stats.sem(animal_df[f'weightsB_end'])], xerr=None,
                           color=colors[ee], alpha=alph)

            ax[num].legend(fontsize=8, loc='upper right')

        BGsBGe = stats.ttest_ind(filt['weightsA_start'], filt['weightsA_end'])
        FGsFGe = stats.ttest_ind(filt['weightsB_start'], filt['weightsB_end'])
        BGsFGs = stats.ttest_ind(filt['weightsA_start'], filt['weightsB_start'])
        BGeFGe = stats.ttest_ind(filt['weightsA_end'], filt['weightsB_end'])

        tts = {f"BGsBGe_{aa}": BGsBGe.pvalue, f"FGsFGe_{aa}": FGsFGe.pvalue,
               f"BGsFGs_{aa}": BGsFGs.pvalue, f"BGeFGe_{aa}": BGeFGe.pvalue}
        print(tts)
        stat_list.append(tts), filt_list.append(filt)

        ax[0].set_ylabel('Mean Weight', fontsize=14, fontweight='bold')
        ax[num].set_title(f'{aa} - Respond to both\n BG and FG alone', fontsize=14, fontweight='bold')
        ax[num].tick_params(axis='both', which='major', labelsize=10)
        ax[num].set_xticklabels(['0-0.5s\nBG', '0.5-1s\nBG', '0-0.5s\nFG', '0.5-1s\nFG'], fontsize=12, fontweight='bold')

        if summary == True:
            ax[num].scatter(x=['BG_start', 'BG_end'],
                            y=[np.nanmean(filt[f'weightsA_start']), np.nanmean(filt[f'weightsA_end'])],
                            label=f'Total (n={len(filt)})', color='black')  # , marker=symbols[cnt])
            ax[num].scatter(x=['FG_start', 'FG_end'],
                            y=[np.nanmean(filt[f'weightsB_start']), np.nanmean(filt[f'weightsB_end'])],
                            color='black')  # , marker=symbols[cnt])
            ax[num].errorbar(x=['BG_start', 'BG_end'],
                             y=[np.nanmean(filt[f'weightsA_start']), np.nanmean(filt[f'weightsA_end'])],
                             yerr=[stats.sem(filt[f'weightsA_start']), stats.sem(filt[f'weightsA_end'])],
                             xerr=None, color='black')
            ax[num].errorbar(x=['FG_start', 'FG_end'],
                             y=[np.nanmean(filt[f'weightsB_start']), np.nanmean(filt[f'weightsB_end'])],
                             yerr=[stats.sem(filt[f'weightsB_start']), stats.sem(filt[f'weightsB_end'])],
                             xerr=None, color='black')

            ax[num].legend(fontsize=8, loc='upper right')

    for num, aa in enumerate(areas):
        area_FG, area_BG = quad2.loc[quad2.area == aa], quad6.loc[quad6.area == aa]

        # for cnt, ss in enumerate(show_suffixes):
        filt_BG = area_BG.loc[(area_BG['r_start'] >= r_thresh) & (area_BG['r_end'] >= r_thresh)]
        filt_FG = area_FG.loc[(area_FG['r_start'] >= r_thresh) & (area_FG['r_end'] >= r_thresh)]
        animal_BG, animal_FG = filt_BG.loc[filt_BG.animal == an], filt_FG.loc[filt_FG.animal == an]
        ax[num+len(areas)].scatter(x=['BG_start', 'BG_end'],
                        y=[np.nanmean(filt_BG[f'weightsA_start']), np.nanmean(filt_BG[f'weightsA_end'])],
                           label=f'BG+/FGo or\nBGo/FG+ (n={len(filt_BG)}, {len(filt_FG)})', color="dimgrey") #, marker=symbols[cnt])
        ax[num+len(areas)].scatter(x=['FG_start', 'FG_end'],
                        y=[np.nanmean(filt_FG[f'weightsB_start']), np.nanmean(filt_FG[f'weightsB_end'])],color='dimgrey')
                           # label=f'{an} (n={len(animal_BG)}, {len(animal_FG)})', color='dimgrey') #, marker=symbols[cnt])
        ax[num+len(areas)].errorbar(x=['BG_start', 'BG_end'],
                        y=[np.nanmean(filt_BG[f'weightsA_start']), np.nanmean(filt_BG[f'weightsA_end'])],
                       yerr=[stats.sem(filt_BG[f'weightsA_start']), stats.sem(filt_BG[f'weightsA_end'])], xerr=None,
                       color='dimgrey')
        ax[num+len(areas)].errorbar(x=['FG_start', 'FG_end'],
                        y=[np.nanmean(filt_FG[f'weightsB_start']), np.nanmean(filt_FG[f'weightsB_end'])],
                       yerr=[stats.sem(filt_FG[f'weightsB_start']), stats.sem(filt_FG[f'weightsB_end'])], xerr=None,
                       color='dimgrey')

        area_df = quad3.loc[quad3.area==aa]
        if strict_r == True:
            filt = area_df.loc[(area_df['r_start'] >= r_thresh) & (area_df['r_end'] >= r_thresh)]
        else:
            filt = area_df.loc[area_df[f"r{ss}"] >= r_thresh]

        ax[num + len(areas)].scatter(x=['BG_start', 'BG_end'],
                        y=[np.nanmean(filt[f'weightsA_start']), np.nanmean(filt[f'weightsA_end'])],
                        label=f'BG+/FG+ (n={len(filt)})', color='black')  # , marker=symbols[cnt])
        ax[num+len(areas)].scatter(x=['FG_start', 'FG_end'],
                        y=[np.nanmean(filt[f'weightsB_start']), np.nanmean(filt[f'weightsB_end'])],
                        color='black')  # , marker=symbols[cnt])
        ax[num+len(areas)].errorbar(x=['BG_start', 'BG_end'],
                         y=[np.nanmean(filt[f'weightsA_start']), np.nanmean(filt[f'weightsA_end'])],
                         yerr=[stats.sem(filt[f'weightsA_start']), stats.sem(filt[f'weightsA_end'])],
                         xerr=None,
                         color='black')
        ax[num+len(areas)].errorbar(x=['FG_start', 'FG_end'],
                         y=[np.nanmean(filt[f'weightsB_start']), np.nanmean(filt[f'weightsB_end'])],
                         yerr=[stats.sem(filt[f'weightsB_start']), stats.sem(filt[f'weightsB_end'])],
                         xerr=None,
                         color='black')

        BGsBGe = stats.ttest_ind(filt_BG['weightsA_start'], filt_BG['weightsA_end'])
        FGsFGe = stats.ttest_ind(filt_FG['weightsB_start'], filt_FG['weightsB_end'])

        ttt = {f'BGsBGe_null_{aa}': BGsBGe.pvalue, f'FGsFGe_null_{aa}': FGsFGe.pvalue}
        stat_list.append(ttt)

        ax[num+len(areas)].legend(fontsize=8, loc='upper right')
        ax[2].set_ylabel('Mean Weight', fontsize=14, fontweight='bold')
        ax[2].set_yticklabels([0.3,0.4,0.5,0.6,0.7,0.8])
        ax[num+len(areas)].set_xticklabels(['0-0.5s\nBG', '0.5-1s\nBG', '0-0.5s\nFG', '0.5-1s\nFG'], fontsize=12, fontweight='bold')
        ax[num+len(areas)].set_title(f'{aa} - Respond to only\none sound alone', fontsize=14, fontweight='bold')

    fig.suptitle(f"r >= {r_thresh}, FR >= {fr_thresh}, strict_r={strict_r}", fontweight='bold', fontsize=10)
    fig.tight_layout()

    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharey=True)
    ax = np.ravel(axes, 'F')

    edges = np.arange(-1, 2, .05)
    axn = 0
    for num, aaa in enumerate(areas):
        to_plot = filt_list[num]

        na, xa = np.histogram(to_plot['weightsA_start'], bins=edges)
        na = na / na.sum() * 100
        nb, xb = np.histogram(to_plot['weightsB_start'], bins=edges)
        nb = nb / nb.sum() * 100

        ax[axn].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
        ax[axn].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
        ax[axn].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=14)
        ax[axn].set_title(f"{aaa} - 0-0.5s", fontweight='bold', fontsize=14)
        ax[axn].tick_params(axis='both', which='major', labelsize=10)
        ax[axn].set_xlabel("Mean Weight", fontweight='bold', fontsize=14)

        axn += 1

        na, xa = np.histogram(to_plot['weightsA_end'], bins=edges)
        na = na / na.sum() * 100
        nb, xb = np.histogram(to_plot['weightsB_end'], bins=edges)
        nb = nb / nb.sum() * 100

        ax[axn].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
        ax[axn].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
        ax[axn].legend(('Background', 'Foreground'), fontsize=12)
        ax[0].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=14)
        ax[2].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=14)
        ax[axn].set_title(f"{aaa} - 0.5-1s", fontweight='bold', fontsize=14)
        ax[axn].tick_params(axis='both', which='major', labelsize=10)
        ax[axn].set_xlabel("Mean Weight", fontweight='bold', fontsize=14)

        axn += 1
    fig.tight_layout()

    return stat_list


def plot_partial_fit_bar(df, fr_thresh=0.03, r_thresh=0.6, suffixes=['_nopost', '_start', '_end'],
                         syn='A', bin='11', animal=None):
    '''2022_11_08. Takes your dataframe (could be single animal or multi animal, you specify, and plots the different fits
    based on what you input for suffixes. It'll put A1 on the top and PEG on bottom.'''
    areas = list(df.area.unique())
    if animal:
        df = df.loc[df.animal==animal]

    quad3 = df.loc[(df.bg_FR_nopost >= fr_thresh) & (df.fg_FR_nopost >= fr_thresh)
                           & (df.bg_FR_start >= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                           & (df.bg_FR_end >= fr_thresh) & (df.fg_FR_end >= fr_thresh)]
    plot_df = quad3.loc[quad3.kind==bin]
    plot_df = plot_df.loc[plot_df.synth_kind==syn]

    plot_df = plot_df.loc[(plot_df['r_nopost'] >= r_thresh) & (plot_df['r_start'] >= r_thresh) & (plot_df['r_end'] >= r_thresh)]

    fig, axes = plt.subplots(len(areas), len(suffixes), figsize=(3 * len(suffixes), 3 * len(areas)))
    ax = np.ravel(axes)

    dd = 0
    for aa in areas:
        area_df = plot_df.loc[plot_df.area==aa]
        ax[dd].set_ylabel(f"{aa}\n\nMean Weights", fontsize=8, fontweight='bold')
        for ss in suffixes:
            to_plot = area_df.loc[(area_df[f'weightsA{ss}'] < 2) & (area_df[f'weightsA{ss}'] > -1) &
                            (area_df[f'weightsB{ss}'] < 2) & (area_df[f'weightsB{ss}'] > -1)]

            BG, FG = np.nanmean(to_plot[f'weightsA{ss}']), np.nanmean(to_plot[f'weightsB{ss}'])
            BGsem, FGsem = stats.sem(to_plot[f'weightsA{ss}']), stats.sem(to_plot[f'weightsB{ss}'])
            ttest1 = stats.ttest_ind(to_plot[f'weightsA{ss}'], to_plot[f'weightsB{ss}'])
            ax[dd].bar("BG", BG, yerr=BGsem, color='deepskyblue')
            ax[dd].bar("FG", FG, yerr=FGsem, color='yellowgreen')
            if ttest1.pvalue < 0.001:
                title = f'{ss} - p<0.001'
            else:
                title = f"{ss} - {ttest1.pvalue:.3f}"
            ax[dd].set_title(f"{title}\nBG: {BG:.2f}, FG: {FG:.2f}\nn={len(to_plot)}", fontsize=8)
            dd += 1
    fig.suptitle(f"Bin: {bin} - r>={r_thresh} - FR>={fr_thresh} - synth: {syn}", fontsize=10, fontweight='bold')
    fig.tight_layout()


def plot_PSTH_example_progression(batch, cellid, bg, fg, bin_kind='11', synth_kind='A', sigma=None, error=False, specs=False):
    '''2022_11_28. Added option to toggle a second figure that makes spectrograms to paste on top.
    Added 2022_11_18 for WIP or talks. Takes any cell and outputs three identical psths, with the sounds in isolation,
    adding linear sum, then adding the combo response, for visuals on powerpoints.'''
    weight_df = ofit.OLP_fit_weights(batch=333, cells=[cellid])
    row = weight_df.loc[(weight_df.BG==bg) & (weight_df.FG==fg) &
                        (weight_df.kind==bin_kind) & (weight_df.synth_kind==synth_kind)]

    # Load response
    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    epo = row.epoch
    epochs = [f"STIM_{epo[0].split('_')[1]}_null", f"STIM_null_{epo[0].split('_')[2]}", epo[0]]

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100
    fs = 100
    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    r = norm_spont.extract_epochs(epochs)
    ls = np.squeeze(np.nanmean(r[epochs[0]] + r[epochs[1]],axis=0))

    # Some plotting calculations
    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs ) - prestim
    dur = manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['Duration']

    r['ls'] = ls
    ax_list = [[0,1,2], [0,1,2], [2], [1,2]]

    fig, axes = plt.subplots(figsize=(10, 10))
    iso = plt.subplot2grid((14, 10), (0, 0), rowspan=4, colspan=10)
    lin = plt.subplot2grid((14, 10), (5, 0), rowspan=4, colspan=10, sharey=iso)
    com = plt.subplot2grid((14, 10), (10, 0), rowspan=4, colspan=10, sharey=iso)
    ax = [iso, lin, com]

    tags = ['Background (BG)', 'Foreground (FG)', 'BG+FG Combo', 'Linear Sum']
    colors = ['deepskyblue','yellowgreen','dimgray', 'dimgray']
    styles = ['solid', 'solid', 'solid', '--']

    # Plot the three response lines
    for (cnt, kk) in enumerate(r.keys()):
        lst = ax_list[cnt]
        plot_resp = r[kk]
        mean_resp = np.squeeze(np.nanmean(plot_resp, axis=0))
        for axn in lst:
            if kk != 'ls':
                if sigma:
                    ax[axn].plot(time, sf.gaussian_filter1d(mean_resp, sigma) * rec['resp'].fs,
                                                       color=colors[cnt], label=f"{tags[cnt]}", ls=styles[cnt])
                if not sigma:
                    ax[axn].plot(time, mean_resp * rec['resp'].fs, color=colors[cnt], label=f"{tags[cnt]}", ls=styles[cnt])
                if error:
                    sem = np.squeeze(stats.sem(plot_resp, axis=0, nan_policy='omit'))
                    ax[axn].fill_between(time, sf.gaussian_filter1d((mean_resp - sem) * rec['resp'].fs, sigma),
                                    sf.gaussian_filter1d((mean_resp + sem) * rec['resp'].fs, sigma),
                                       alpha=0.4, color=colors[cnt])
            else:
            # Plot the linear sum line
                if sigma:
                    ax[axn].plot(time, sf.gaussian_filter1d(ls * rec['resp'].fs, sigma), color=colors[cnt],
                            ls=styles[cnt], label=tags[cnt], lw=1)
                if not sigma:
                    ax[axn].plot(time, ls * rec['resp'].fs, color=colors[cnt], ls=styles[cnt], label=tags[cnt], lw=1)

    for axn in range(len(ax)):
        ax[axn].set_xlim(-0.2, (dur + 0.3))  # arbitrary window I think is nice
        ymin, ymax = ax[axn].get_ylim()
        ax[axn].set_ylabel('spk/s', fontweight='bold', size=10)
        ax[axn].legend(loc='upper right', fontsize=6)
        ax[axn].vlines([0, dur], ymin, ymax, colors='black', linestyles=':', lw=1)
        ax[axn].set_ylim(ymin, ymax)
        ax[axn].set_xlabel('Time (s)', fontweight='bold', size=10)
        ax[axn].spines['top'].set_visible(True), ax[axn].spines['right'].set_visible(True)
        ax[axn].set_xticks([0.0, 0.5, 1.0])
    # fig.tight_layout()

    if specs == True:
        fig, axes = plt.subplots(figsize=(10, 10))
        specA = plt.subplot2grid((14, 10), (7, 0), rowspan=1, colspan=10)
        specB = plt.subplot2grid((14, 10), (8, 0), rowspan=1, colspan=10)
        com = plt.subplot2grid((14, 10), (10, 0), rowspan=4, colspan=10)
        ax = [com, specA, specB]
        r = norm_spont.extract_epochs(epochs)

        for (cnt, kk) in enumerate(r.keys()):

            plot_resp = r[kk]
            mean_resp = np.squeeze(np.nanmean(plot_resp, axis=0))
            if sigma:
                ax[0].plot(time, sf.gaussian_filter1d(mean_resp, sigma) * rec['resp'].fs,
                                                   color=colors[cnt], label=f"{tags[cnt]}", ls=styles[cnt])
            if not sigma:
                ax[0].plot(time, mean_resp * rec['resp'].fs, color=colors[cnt], label=f"{tags[cnt]}", ls=styles[cnt])

        ax[0].set_xlim(-0.2, (dur + 0.3))        # arbitrary window I think is nice
        ymin, ymax = ax[0].get_ylim()

        ax[0].set_ylabel('spk/s', fontweight='bold', size=12)
        ax[0].legend(loc='upper right', fontsize=18, prop=dict(weight='bold'), labelspacing=0.4)
        ax[0].vlines([0, dur], ymin, ymax, colors='black', linestyles=':')
        ax[0].set_ylim(ymin, ymax)
        ax[0].set_xlabel('Time (s)', fontweight='bold', size=12)
        ax[0].set_xticks([0.0, 0.5, 1.0])
        ax[0].set_xticklabels([0.0, 0.5, 1.0], fontsize=10)
        ax[0].spines['top'].set_visible(True), ax[0].spines['right'].set_visible(True)
        # ax[2].vlines(params['SilenceOnset'], ymax * .9, ymax, colors='black', linestyles='-', lw=0.25)

        xmin, xmax = ax[0].get_xlim()

        # specs
        folder_ids = [int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['BG_Folder'][-1]),
                int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['FG_Folder'][-1])]

        bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                            f'Background{folder_ids[0]}/*.wav'))
        fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                            f'Foreground{folder_ids[1]}/*.wav'))
        bg_path = [bb for bb in bg_dir if bg in bb][0]
        fg_path = [ff for ff in fg_dir if fg in ff][0]

        xf = 100
        low, high = xmin * xf, xmax * xf

        sfs, W = wavfile.read(bg_path)
        spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
        ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]],
                     cmap='gray_r')
        ax[1].set_xlim(low, high)
        ax[1].set_xticks([]), ax[1].set_yticks([])
        ax[1].set_xticklabels([]), ax[1].set_yticklabels([])
        ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)
        ax[1].set_ylabel(f"BG: {bg}", rotation=0, fontweight='bold', verticalalignment='center',
                         size=14, labelpad=-10)

        sfs, W = wavfile.read(fg_path)
        spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
        ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                     cmap='gray_r')
        ax[2].set_xlim(low, high)
        ax[2].set_xticks([]), ax[2].set_yticks([])
        ax[2].set_xticklabels([]), ax[2].set_yticklabels([])
        ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
        ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)
        ax[2].set_ylabel(f"FG: {fg}", rotation=0, fontweight='bold', verticalalignment='center',
                         size=14, labelpad=-10)

        # This just makes boxes around only the important part of the spec axis. So it all lines up.
        ymin, ymax = ax[2].get_ylim()
        ax[1].vlines([0, 1*rec['resp'].fs], ymin, ymax, color='black', lw=0.5)
        ax[1].hlines([ymin+2,ymax], 0, 1*rec['resp'].fs, color='black', lw=0.5)
        ax[2].vlines([0, 1*rec['resp'].fs], ymin, ymax, color='black', lw=0.5)
        ax[2].hlines([ymin+1,ymax], 0, 1*rec['resp'].fs, color='black', lw=0.5)


def psths_with_specs_partial_fit(df, cellid, bg, fg, batch=340, bin_kind='11', synth_kind='N',
                     sigma=None, error=True):
    '''Makes panel three of APAN 2021 poster and NGP 2022 poster, this is a better way than the way
    normalized_linear_error_figure in this file does it, which relies on an old way of loading and
    saving the data that doesn't use DFs and is stupid. The other way also only works with marmoset
    and maybe early ferret data, but definitely not binaural and synthetics. Use this, it's better.
    It does lack the linear error stat though, but that's not important anymore 2022_09_01.'''

    # Make figure bones. Could add another spectrogram below, there's space.
    f = plt.figure(figsize=(8, 6))
    psth = plt.subplot2grid((18, 3), (4, 0), rowspan=5, colspan=6)
    specA = plt.subplot2grid((18, 3), (0, 0), rowspan=2, colspan=6)
    specB = plt.subplot2grid((18, 3), (2, 0), rowspan=2, colspan=6)
    ax = [specA, specB, psth]

    tags = ['BG', 'FG', 'BG+FG']
    colors = ['deepskyblue','yellowgreen','dimgray']

    #Get this particular row out for some stuff down the road
    row = df.loc[(df.cellid==cellid) & (df.BG==bg) & (df.FG==fg) & (df.kind==bin_kind)
                 & (df.synth_kind==synth_kind)].squeeze()

    # Load response
    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    epo = row.epoch
    epochs = [f"STIM_{epo.split('_')[1]}_null", f"STIM_null_{epo.split('_')[2]}", epo]

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100
    fs = 100
    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    r = norm_spont.extract_epochs(epochs)
    ls = np.squeeze(np.nanmean(r[epochs[0]] + r[epochs[1]],axis=0))

    # Some plotting calculations
    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs ) - prestim
    dur = manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['Duration']

    # Plot the three response lines
    for (cnt, kk) in enumerate(r.keys()):
        plot_resp = r[kk]
        mean_resp = np.squeeze(np.nanmean(plot_resp, axis=0))
        if sigma:
            ax[2].plot(time, sf.gaussian_filter1d(mean_resp, sigma) * rec['resp'].fs,
                                               color=colors[cnt], label=f"{tags[cnt]}")
        if not sigma:
            ax[2].plot(time, mean_resp * rec['resp'].fs, color=colors[cnt], label=f"{tags[cnt]}")
        if error:
            sem = np.squeeze(stats.sem(plot_resp, axis=0, nan_policy='omit'))
            ax[2].fill_between(time, sf.gaussian_filter1d((mean_resp - sem) * rec['resp'].fs, sigma),
                            sf.gaussian_filter1d((mean_resp + sem) * rec['resp'].fs, sigma),
                               alpha=0.4, color=colors[cnt])
    # Plot the linear sum line
    if sigma:
        ax[2].plot(time, sf.gaussian_filter1d(ls * rec['resp'].fs, sigma), color='dimgray',
                ls='--', label='Linear Sum')
    if not sigma:
        ax[2].plot(time, ls * rec['resp'].fs, color='dimgray', ls='--', label='Linear Sum')
    ax[2].set_xlim(-0.2, (dur + 0.3))        # arbitrary window I think is nice
    ymin, ymax = ax[2].get_ylim()

    ax[2].set_ylabel('spk/s', fontweight='bold', size=12)
    ax[2].legend(loc='upper right', fontsize=18, prop=dict(weight='bold'), labelspacing=0.4)
    ax[2].vlines([0, dur], ymin, ymax, colors='black', linestyles=':')
    ax[2].set_ylim(ymin, ymax)
    ax[2].set_xlabel('Time (s)', fontweight='bold', size=12)
    ax[2].set_xticks([0.0, 0.5, 1.0])
    ax[2].set_xticklabels([0.0, 0.5, 1.0], fontsize=10)
    ax[2].spines['top'].set_visible(True), ax[2].spines['right'].set_visible(True)
    # ax[2].vlines(params['SilenceOnset'], ymax * .9, ymax, colors='black', linestyles='-', lw=0.25)

    ax[0].set_title(f"{cellid}\nwBG_full: {row.weightsA:.2f} - wFG_full: {row.weightsB:.2f} - r: {row.r:.2f}\nwBG_start: {row.weightsA_start:.2f} -"
                    f" wFG_start: {row.weightsB_start:.2f} - r: {row.r_start:.2f}\nwBG_end: {row.weightsA_end:.2f} - "
                    f"wFG_end: {row.weightsB_end:.2f} - r: {row.r_end:.2f}",
                    fontweight='bold', size=10)
    xmin, xmax = ax[2].get_xlim()

    # Spectrogram part
    folder_ids = [int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['BG_Folder'][-1]),
            int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['FG_Folder'][-1])]

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Foreground{folder_ids[1]}/*.wav'))
    bg_path = [bb for bb in bg_dir if epo.split('_')[1].split('-')[0][:2] in bb][0]
    fg_path = [ff for ff in fg_dir if epo.split('_')[2].split('-')[0][:2] in ff][0]

    xf = 100
    low, high = xmin * xf, xmax * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[0].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[0].set_xlim(low, high)
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xticklabels([]), ax[0].set_yticklabels([])
    ax[0].spines['top'].set_visible(False), ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False), ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel(f"BG: {row.BG}", rotation=0, fontweight='bold', verticalalignment='center',
                     size=14, labelpad=-10)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[1].set_xticklabels([]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)
    ax[1].set_ylabel(f"FG: {row.FG}", rotation=0, fontweight='bold', verticalalignment='center',
                     size=14, labelpad=-10)

    # This just makes boxes around only the important part of the spec axis. So it all lines up.
    ymin, ymax = ax[1].get_ylim()
    ax[0].vlines([0, 1*rec['resp'].fs], ymin, ymax, color='black', lw=0.5)
    ax[0].hlines([ymin+2,ymax], 0, 1*rec['resp'].fs, color='black', lw=0.5)
    ax[1].vlines([0, 1*rec['resp'].fs], ymin, ymax, color='black', lw=0.5)
    ax[1].hlines([ymin+1,ymax], 0, 1*rec['resp'].fs, color='black', lw=0.5)


def display_sound_envelopes(sound_df, type='FG', envs=True):
    '''2022_11_09. Give it a sound df generated from a cell and then choose whether you want to look at all of the FGs or
    BGs. Toggling envs will either plot the envelope of the sound, if True, or the spectrogram, if False'''
    if type == 'FG':
        fg_list = list(sound_df.loc[(sound_df.type == 'FG') & (sound_df.synth_kind == 'A')].name)
        fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                            f'Foreground3/*.wav'))
        paths = [aa for aa in fg_dir if aa.split('/')[-1].split('.')[0] in fg_list]
    elif type == 'BG':
        bg_list = list(sound_df.loc[(sound_df.type == 'BG') & (sound_df.synth_kind == 'A')].name)
        bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                            f'Background2/*.wav'))
        paths = [aa for aa in bg_dir if aa.split('/')[-1].split('.')[0] in bg_list]
    else:
        raise ValueErorr(f"You inputted {type} for type, it only takes 'BG' or 'FG'.")

    fig, axes = plt.subplots(4, 5, figsize=(18, 7))
    axes = np.ravel(axes)

    for cnt, (ax, pp) in enumerate(zip(axes, paths)):
        sfs, W = wavfile.read(pp)
        spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
        if envs == True:
            env = np.sum(spec, axis=0)
            ax.plot(env, color='black')
            half = int(np.floor(env.shape[0] / 2))
            _, ymax = ax.get_ylim()
            ax.vlines(half, 0, ymax, ls=':', colors='black')
        else:
            ax.imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                      cmap='gray_r')
            half = int(np.floor(spec.shape[1] / 2))
            ax.vlines(half, 0, spec.shape[0], ls=':', colors='black')

        ax.spines['top'].set_visible(True), ax.spines['right'].set_visible(True)
        ax.set_title(f"{pp.split('/')[-1].split('.')[0][2:]}", fontweight='bold')

    if len(axes) != len(paths):
        diff = len(axes) - len(paths)
        for ii in range(diff):
            axes[-(ii + 1)].set_xticks([]), axes[-(ii + 1)].set_yticks([])
            axes[-(ii + 1)].set_xticklabels([]), axes[-(ii + 1)].set_yticklabels([])
            axes[-(ii + 1)].spines['top'].set_visible(False), axes[-(ii + 1)].spines['bottom'].set_visible(False)
            axes[-(ii + 1)].spines['left'].set_visible(False), axes[-(ii + 1)].spines['right'].set_visible(False)


def plot_some_sound_stats(sound_df):
    stat = ['Fstationary', 'Tstationary', 'bandwidth']
    bgs, fgs = sound_df[sound_df['type'] == 'BG'], sound_df[sound_df['type'] == 'FG']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, ss in zip(axes, stat):
        print(ss)
        tt = stats.ttest_ind(bgs[ss], fgs[ss])
        print(f"BG: {np.nanmean(bgs[ss])} - FG: {np.nanmean(fgs[ss])}")
        print(f"BG: {stats.sem(bgs[ss])} - FG: {stats.sem(fgs[ss])}")
        ax.bar("BG", bgs[ss], yerr=stats.sem(bgs[ss]), color='deepskyblue')
        ax.bar("FG", fgs[ss], yerr=stats.sem(fgs[ss]), color='yellowgreen')
        if tt.pvalue < 0.001:
            ax.set_title(f"{ss}\np<0.001)", fontsize=8, fontweight='bold')
        else:
            ax.set_title(f"{ss}\n{tt.pvalue:.3f})", fontsize=8, fontweight='bold')
        print(tt)


def speaker_test_plot(weight_df_11, weight_df_22, weight_df_synth, threshs=[0.03, 0.02, 0.01]):
    '''2022_11_09. This is a half baked figure I used when I was trying to decide if the speaker setup was right on the
    second hemisphere of clathrus. It's not elegant, and you have to make the DFs from the binaural fit df and synth
    one on your own to put them in here.'''
    fig, axes = plt.subplots(1, len(threshs), figsize=(18, 12))
    axes = np.ravel(axes)

    for tt, ax in zip(threshs, axes):
        quad_11, _ = ohel.quadrants_by_FR(weight_df_11, threshold=tt, quad_return=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        quad_22, _ = ohel.quadrants_by_FR(weight_df_22, threshold=tt, quad_return=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        quad_synth, _ = ohel.quadrants_by_FR(weight_df_synth, threshold=tt, quad_return=[1, 2, 3, 4, 5, 6, 7, 8, 9])

        quads = [quad_synth, quad_11, quad_22]
        for qq in quads:
            for k, v in qq.items():
                qq[k] = v.shape[0]

        for qq in quads:
            total = sum(qq.values())
            for k, v in qq.items():
                qq[k] = np.around((v / total) * 100, 2)

        quad_df = pd.DataFrame(quads)
        quad_df = quad_df.rename(index={0: 'synth', 1: 'contra', 2: 'ipsi'})

        quad_df['+'] = quad_df[2] + quad_df[3] + quad_df[6]
        quad_df['-'] = quad_df[1] + quad_df[4] + quad_df[7] + quad_df[8] + quad_df[9] + quad_df[5]

        quad_array = quad_df.to_numpy()
        ax.plot(quad_array.T)
        ax.set_xticks(list(range(quad_array.shape[1])))
        ax.set_xticklabels(list(quad_df.columns))
        ax.set_title(f"threshold = {tt}")
    axes[0].set_ylabel("percent units")


def all_animal_scatter(df, fr_thresh=0.03, r_thresh=0.6):
    '''Uses a dataframe potentially with multiple areas and animals and plots the quadrants divided by your FR threshold
    as lines. Useful for talks and stuff. It returns a dictionary that gives you the % breakdown of the quads.'''
    colors = ['mediumorchid', 'darkorange', 'orangered', 'green']
    areas = list(df.area.unique())
    fig, ax = plt.subplots(1, len(areas), figsize=(len(areas) * 8, 8))

    animals = list(df.animal.unique())
    animals.reverse()
    counts = {}

    for num, aa in enumerate(areas):
        area_df = df.loc[df.area == aa]
        filt = area_df.loc[(area_df['r_start'] >= r_thresh) & (area_df['r_end'] >= r_thresh)]

        for ee, an in enumerate(animals):
            animal_df = filt.loc[filt.animal == an]
            ax[num].scatter(x=animal_df.bg_FR, y=animal_df.fg_FR,
                            label=f'{an} (n={len(animal_df)})', color=colors[ee], s=1)
            ax[num].legend(loc='upper left')

        xmin, xmax = ax[num].get_xlim()
        ymin, ymax = ax[num].get_ylim()
        ax[num].vlines([fr_thresh, -fr_thresh], ymin, ymax, color='black', lw=0.5)
        ax[num].hlines([fr_thresh, -fr_thresh], xmin, xmax, color='black', lw=0.5)
        ax[num].spines['top'].set_visible(True), ax[num].spines['right'].set_visible(True)

        ax[num].set_xlabel('BG Firing Rate', fontsize=12, fontweight='bold')
        ax[num].set_ylabel('FG Firing Rate', fontsize=12, fontweight='bold')
        ax[num].set_ylim([-0.2, 0.4]), ax[num].set_xlim([-0.2, 0.4])
        ax[num].set_title(f"{aa}", fontsize=12, fontweight='bold')

        quad, _ = ohel.quadrants_by_FR(filt, threshold=0.03, quad_return=[1,2,3,4,5,6,7,8,9])

        total = len(filt)
        for qq, val in quad.items():
            counts[f"{qq}_{aa}"] = int(np.around(((len(val) / total) * 100), 0))

    return counts


def spectrogram_stats_diagram(name, type, bg_fold=2, fg_fold=3, synth_kind='A'):
    '''2022_11_24. Takes any sound and corresponding type and will plot the spectrogram. From there it will take the
    most interesting of each time and frequency bin and collect the adjacent bins, which it will plot below for time
    and to the right for the spectral examples.'''
    if type == 'BG':
        kind_fold = f'Background{bg_fold}'
        colors = 'deepskyblue'
    elif type == 'FG':
        kind_fold = f'Foreground{fg_fold}'
        colors = 'yellowgreen'

    colors = ['black', 'dimgray', 'gray', 'darkgray', 'lightgray']
    colors.reverse()

    if synth_kind == 'A' or synth_kind == 'N':
        direct = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                            f'{kind_fold}/*.wav'))
    else:
        kind_dict = {'M': 'SpectrotemporalMod', 'U': 'Spectrotemporal', 'T': 'Temporal',
                     'S': 'Spectral', 'C': 'Cochlear'}
        direct = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                            f'{kind_fold}/{kind_dict[synth_kind]}/*.wav'))

    path = [bb for bb in direct if name in bb]
    if len(path) == 0:
        raise ValueError(f"Your sound name {name} isn't in there. Maybe add a space if it needs.")

    # define figure
    fig, axes = plt.subplots(figsize=(12, 6))
    spec = plt.subplot2grid((8, 12), (0, 0), rowspan=3, colspan=8)
    time = plt.subplot2grid((8, 12), (4, 0), rowspan=2, colspan=8)
    freq = plt.subplot2grid((8, 12), (0, 9), rowspan=3, colspan=2)
    ax = [spec, time, freq]

    sfs, W = wavfile.read(path[0])
    spec = gtgram(W, sfs, 0.02, 0.01, 96, 100, 16000)

    ax[0].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]],
             cmap='gray_r')
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xticklabels([]), ax[0].set_yticklabels([])
    ax[0].spines['top'].set_visible(True), ax[0].spines['bottom'].set_visible(True)
    ax[0].spines['left'].set_visible(True), ax[0].spines['right'].set_visible(True)
    ax[0].set_title(f"{type}: {name}", fontweight='bold', fontsize=18)
    ax[0].set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Time (s)', fontsize=14, fontweight='bold')

    # Now for the time stats
    time_mean = np.nanmean(spec, axis=0)
    freq_mean = np.nanmean(spec, axis=1)
    tmax, fmax = spec.shape[1], spec.shape[0]
    max = freq_mean.max()
    idx = list(freq_mean).index(max)
    if idx >= 4 and idx <= fmax - 4:
        idx_list = [idx - 4, idx - 2, idx, idx + 2, idx + 4]
    elif idx < 4:
        idx_list = [idx, idx + 2, idx + 4, idx + 6, idx + 8]
    elif idx > fmax-4:
        idx_list = [idx - 8, idx - 6, idx - 4, idx - 2, idx]
    offset = spec[idx_list, :].max() * 0.15
    start = offset * len(idx_list)
    cut_min = int(np.around(spec[idx_list, :].min(), 0))
    cut_max = int(np.around(spec[idx_list, :].max(), 0))

    i = start
    for cnt, ii in enumerate(idx_list):
        ax[1].plot(spec[ii, :] + i, color=colors[cnt], lw=2)

        i -= offset

    ymin, ymax = ax[1].get_ylim()
    ax[1].set_yticks([ymin, ymax])
    ax[1].set_yticklabels([cut_min, cut_max])

    ax[1].set_xticks([])
    ax[1].set_xticklabels([])
    ax[1].spines['top'].set_visible(True), ax[1].spines['bottom'].set_visible(True)
    ax[1].spines['left'].set_visible(True), ax[1].spines['right'].set_visible(True)
    ax[1].set_title(f"Average Standard Deviation: {np.around(np.std(spec, axis=1).mean(), 1)}", fontweight='bold', fontsize=12)

    # now Freq
    max = time_mean.max()
    idx = list(time_mean).index(max)
    if idx >= 4 and idx <= fmax - 4:
        idx_list = [idx - 4, idx - 2, idx, idx + 2, idx + 4]
    elif idx < 4:
        idx_list = [idx, idx + 2, idx + 4, idx + 6, idx + 8]
    elif idx > fmax-4:
        idx_list = [idx - 8, idx - 6, idx - 4, idx - 2, idx]
    offset = spec[:, idx_list].max() * 0.15
    start = offset * len(idx_list)
    cut_min = int(np.around(spec[:, idx_list].min(), 0))
    cut_max = int(np.around(spec[:, idx_list].max(), 0))

    i = start
    for cnt, ii in enumerate(idx_list):
        ax[2].plot((spec[:, ii] + i), range(96), color=colors[cnt], lw=2)
        i -= offset

    xmin, xmax = ax[2].get_xlim()
    ax[2].set_xticks([xmin, xmax])
    ax[2].set_xticklabels([cut_min, cut_max])

    ax[2].set_yticks([])
    ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(True), ax[2].spines['bottom'].set_visible(True)
    ax[2].spines['left'].set_visible(True), ax[2].spines['right'].set_visible(True)
    ax[2].set_title(f"Average Standard\nDeviation: {np.around(np.std(spec, axis=0).mean(), 1)}", fontweight='bold', fontsize=12)
