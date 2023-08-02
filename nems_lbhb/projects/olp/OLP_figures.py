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
import nems_lbhb.projects.olp.OLP_fit as ofit
import pandas as pd
from matplotlib import cm

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
                     sigma=None, error=True, title=None):
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

    if title:
        ax[0].set_title(f"{title}", fontweight='bold', size=16)
        # ax[0].set_title(f"wBG: {row.weightsA:.2f} wFG: {row.weightsB:.2f}\n{title}", fontweight='bold', size=16)
    else:
        ax[0].set_title(f"{cellid}", fontweight='bold', size=16)

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
                                 synth_kind='N', sigma=None, example=False, sort=True, lin_sum=True, positive_only=False):
    '''2023_05_10. Moved from OLP_home. It appears what I changed was the Lin sum options

    Takes out the BG, FG, combo, diff psth heatmaps from the interactive plot and makes it it's own
    figure. You provide the weight_df, site, and sounds, and then optionally you can throw a cellid
    or list of them in and it'll go ahead and only label those on the y axis so you can better see
    it. Turn example to True if you'd like it to be a more generically titled than using the actually
    epoch names, which are not good for posters. Added 2022_09_01

    Added sorting for the difference panel which will in turn sort all other panels 2022_09_07. Also,
    added mandatory normalization of responses by the max for each unit across the three epochs.

    2023_01_23. Added lin_sum option. This totally changes the figure and gets rid of the difference array plot
    and instead plots the heatmap of the linear sum and uses that for comparisons.

    response_heatmaps_comparison(weight_df, site='CLT008a', bg='Wind', fg='Geese', cellid='CLT008a-046-2',
                                     batch=340, bin_kind='11',
                                     synth_kind='A', sigma=2, sort=True, example=True, lin_sum=True)'''
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

    # # Gets the spont rate for each channel - maybe useful
    # resp = copy.copy(rec['resp'].rasterize())
    # rec['resp'].fs = 100
    # SR_list = []
    # for cc in resp.chans:
    #     inp = resp.extract_channels([cc])
    #     norm_spont, SR, STD = ohel.remove_spont_rate_std(inp)
    #     SR_list.append(SR)

    resp_plot = np.stack([np.nanmean(aa, axis=0) for aa in list(r.values())])

    # # Subtracts SR from each
    # for nn, sr in enumerate(SR_list):
    #     resp_plot[:, nn, :] = resp_plot[:, nn, :] - sr

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / rec['resp'].fs) - prestim
    dur = manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['Duration']

    # Gets rid of SR for the epochs we care about
    for ww in range(resp_plot.shape[1]):
        chan_prestim = resp_plot[:, ww, :int(prestim*fs)]
        SR = np.mean(chan_prestim)
        resp_plot[:, ww, :] = resp_plot[:, ww, :] - SR

    if lin_sum:
        ls = np.expand_dims(resp_plot[0, :, :] + resp_plot[1, :, :], axis=0)
        resp_plot = np.append(resp_plot, ls, axis=0)

    resps = resp_plot

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
        if lin_sum:
            sort_array = resp_plot[-2, :, int(prestim * fs):int((prestim + dur) * fs)]
        else:
            sort_array = diff_array[:,int(prestim*fs):int((prestim+dur)*fs)]
        means = list(np.nanmean(sort_array, axis=1))
        indexes = list(range(len(means)))
        sort_df = pd.DataFrame(list(zip(means, indexes)), columns=['mean', 'idx'])
        sort_df = sort_df.sort_values('mean', ascending=True)
        if positive_only:
            sort_list = [int(aa[1]['idx']) for aa in sort_df.iterrows() if aa[1]['mean'] > 0]
        else:
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
        num_ids[-1], num_ids[0] = '1', f'{len(num_ids)}'

    # Smooth if you have given it a sigma by which to smooth
    if sigma:
        resp_plot = sf.gaussian_filter1d(resps, sigma, axis=2)
        diff_array = sf.gaussian_filter1d(diff_array, sigma, axis=1)

    # Adding code to normalize each cell to the max of that cell to any of the epochs
    for nn in range(resp_plot.shape[1]):
        # max_val = np.max(np.abs(resp_plot[:,nn,int(prestim*fs):int((prestim+dur)*fs)]))
        max_val = np.max(np.abs(resp_plot[:, nn, :]))
        resp_plot[:, nn, :] = resp_plot[:, nn, :] / max_val

    # Get the min and max of the array, find the biggest magnitude and set max and min
    # to the abs and -abs of that so that the colormap is centered at zero
    cmax, cmin = np.max(resp_plot), np.min(resp_plot)
    biggest = np.maximum(np.abs(cmax),np.abs(cmin))
    cmax, cmin = np.abs(biggest), -np.abs(biggest)

    #if you don't want difference array and just the responses and the linear sum
    if lin_sum:
        if sort == True:
            resp_plot = resp_plot[:, sort_list, :]

        epochs.append('Linear Sum')

        fig, axes = plt.subplots(figsize=(9, 12))
        BGheat = plt.subplot2grid((11, 6), (0, 0), rowspan=2, colspan=5)
        FGheat = plt.subplot2grid((11, 6), (2, 0), rowspan=2, colspan=5)
        lsheat = plt.subplot2grid((11, 6), (4, 0), rowspan=2, colspan=5)
        combheat = plt.subplot2grid((11, 6), (6, 0), rowspan=2, colspan=5)
        cbar_main = plt.subplot2grid((11, 6), (2, 5), rowspan=4, colspan=1)
        ax = [BGheat, FGheat, lsheat, combheat, cbar_main]

        for (ww, qq) in enumerate(epochs):
            dd = ax[ww].imshow(resp_plot[ww, :, :], vmin=cmin, vmax=cmax,
                               cmap='bwr', aspect='auto', origin='lower',
                               extent=[time[0], time[-1], 0, len(all_cells)])
            ax[ww].vlines([int(prestim), int(prestim+dur)], ymin=0, ymax=len(all_cells),
                          color='black', lw=1, ls=':')
            # ax[qq].set_ylabel('Unit', fontweight='bold', fontsize=8)
            ax[ww].set_yticks([*range(0, len(sort_list))])
            ax[ww].set_yticklabels(num_ids, fontsize=font_size) #, fontweight='bold')
            ax[ww].set_xlim(-0.2, (dur + 0.3))  # arbitrary window I think is nice
            if example == True:
                titles = [f"BG - {bg}", f"FG - {fg}", f"Combo\nBG+FG", 'Linear Sum']
                ax[ww].set_ylabel(f"{titles[ww]}", fontsize=12, fontweight='bold', rotation=90,
                                  horizontalalignment='center') # , labelpad=40)
                ax[0].set_title(f'Site {all_cells.iloc[0].cellid[:7]} Responses', fontweight='bold', fontsize=12)
            else:
                ax[ww].set_title(f"{qq}", fontsize=8, fontweight='bold')
                ax[ww].set_ylabel('Unit', fontweight='bold', fontsize=8)
            ax[ww].spines['top'].set_visible(True), ax[ww].spines['right'].set_visible(True)

        # ax[2].set_xlabel('Time (s)', fontweight='bold', fontsize=9)
        ax[0].set_xticks([]), ax[1].set_xticks([]), ax[2].set_xticks([])
        # Add the colorbar to the axis to the right of these, the diff will get separate cbar
        fig.colorbar(dd, ax=ax[4], aspect=10)
        ax[4].spines['top'].set_visible(False), ax[4].spines['right'].set_visible(False)
        ax[4].spines['bottom'].set_visible(False), ax[4].spines['left'].set_visible(False)
        ax[4].set_yticks([]), ax[4].set_xticks([])
        ax[3].set_xlabel('Time (s)', fontsize=9, fontweight='bold')

    # Plot BG, FG, Combo and difference array
    else:
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


def resp_weight_multi_scatter(weight_df, ycol=['weightsA', 'weightsA', 'weightsB', 'weightsB'], fr_met='snr',
                              synth_kind='N', threshold=0.03, snr_threshold=0.12, quads=3, r_thresh=0.6, area='A1'):
    '''Updated resp_weight_scatter to just plot all four combinations of FR/wt on one plot to avoid
    the silliness of four individual plots. Works the same, basically just give it a df.'''
    if threshold:
        quad, _ = ohel.quadrants_by_FR(weight_df, threshold=threshold, quad_return=quads)
    if snr_threshold:
        weight_df = weight_df.loc[(weight_df.bg_snr >= snr_threshold) & (weight_df.fg_snr >= snr_threshold)]
    quad = weight_df
    if isinstance(synth_kind, list):
        quad = quad.loc[quad.synth_kind.isin(synth_kind)].copy()
    else:
        quad = quad.loc[quad.synth_kind == synth_kind].copy()

    if r_thresh:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_thresh]
    if area:
        quad = quad.loc[quad.area == area]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    xcol = [f'bg_{fr_met}', f'fg_{fr_met}', f'bg_{fr_met}', f'fg_{fr_met}']
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
    axes[0].set_ylim(-1.5,2.5)
    if threshold:
        fig.suptitle(f"FR Thresh: {threshold} - Area: {area} - r_thresh: {r_thresh}", fontweight='bold', fontsize=12)
    if snr_threshold:
        fig.suptitle(f"snr >= {snr_threshold} - Area: {area} - r_thresh: {r_thresh}", fontweight='bold', fontsize=12)

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


def sound_metric_scatter(df, x_metrics, y_metric, x_labels, suffix='', area='A1', threshold=0.03,
                         jitter=[0.25,0.2,0.03], snr_threshold=0.12,
                         quad_return=3, metric_filter=None, synth_kind='N', bin_kind='11',
                         title_text='', r_cut=None, mean=True):
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
    if threshold:
        df, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=quad_return)
    if snr_threshold:
        df = df.loc[(df.bg_snr >= snr_threshold) & (df.fg_snr >= snr_threshold)]

    quad = df.loc[(df.area==area) & (df.synth_kind==synth_kind) & (df.kind==bin_kind)]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    quad = quad.copy()
    # I use 2.5 for relative gain, I'm sure weights have one too...
    if metric_filter:
        quad = quad.loc[quad[y_metric] <= metric_filter]
        quad = quad.loc[quad[y_metric] >= -metric_filter]

    quad = ohel.get_sound_statistics_from_df(quad, percent_lims=[15, 85], append=True)

    if y_metric==f'BG_rel_gain{suffix}':
        y_metric2, title, ylabel = f'FG_rel_gain{suffix}', f'Relative Gain{suffix}', f'Relative Gain{suffix}'
    elif y_metric=='weightsB':
        y_metric2, title, ylabel = 'weightsA', 'How this sound effects a concurrent sound', 'Weight'
    elif y_metric=='weightsA':
        y_metric2, title, ylabel = 'weightsB', 'How this sound itself is weighted', 'Weight'
    else:
        y_metric2, title, ylabel = y_metric, y_metric, y_metric

    # fig, axes = plt.subplots(1, len(x_metrics), figsize=(len(x_metrics)*5, 6))
    fig, axes = plt.subplots(1, len(x_metrics), figsize=(10, 5))


    for cnt, (ax, met) in enumerate(zip(axes, x_metrics)):
        # Add a column that is the data for that metric, but jittered, for viewability
        if mean==False:
            quad[f'jitter_BG_{met}'] = quad[f'BG_{met}'] + np.random.normal(0, jitter[cnt], len(quad))
            quad[f'jitter_FG_{met}'] = quad[f'FG_{met}'] + np.random.normal(0, jitter[cnt], len(quad))
            # Do the plotting
            sb.scatterplot(x=f'jitter_BG_{met}', y=y_metric, data=quad, ax=ax, s=4, color='cornflowerblue')
            sb.scatterplot(x=f'jitter chro_FG_{met}', y=y_metric2, data=quad, ax=ax, s=4, color='olivedrab')
        else:
            to_plot_BG = quad[['BG', f'BG_{met}', y_metric]]
            mean_BG = to_plot_BG.groupby(by='BG').mean()
            sem_BG = to_plot_BG.groupby(by='BG').sem()

            to_plot_FG = quad[['FG', f'FG_{met}', y_metric2]]
            mean_FG = to_plot_FG.groupby(by='FG').mean()
            sem_FG = to_plot_FG.groupby(by='FG').sem()

            ax.errorbar(x=mean_BG[f'BG_{met}'], y=mean_BG[y_metric], yerr=sem_BG[y_metric], ls='none', color='black',
                        elinewidth=0.5)
            ax.scatter(x=mean_BG[f'BG_{met}'], y=mean_BG[y_metric], color='deepskyblue')

            ax.errorbar(x=mean_FG[f'FG_{met}'], y=mean_FG[y_metric2], yerr=sem_FG[y_metric2], ls='none', color='black',
                        elinewidth=0.5)
            ax.scatter(x=mean_FG[f'FG_{met}'], y=mean_FG[y_metric2], color='yellowgreen')

            # sb.scatterplot(x=f'BG_{met}', y=y_metric, data=to_plot_BG, ax=ax, s=30, color='cornflowerblue')
            # sb.scatterplot(x=f'FG_{met}', y=y_metric2, data=to_plot_FG, ax=ax, s=30, color='olivedrab')
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

    if threshold:
        fig.suptitle(f"{area} - {synth_kind} - r >= {r_cut} - FR_thresh >= {threshold} "
                     f"- {title_text}", fontweight='bold', fontsize=10)
    if snr_threshold:
        fig.suptitle(f"{area} - {synth_kind} - r >= {r_cut} - snr >= {snr_threshold} "
                     f"- {title_text}", fontweight='bold', fontsize=10)


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


def weights_supp_comp(weight_df, x='resp', area='A1', quads=3, thresh=None, snr_threshold=0.12, r_cut=None):
    '''2023_07_12. Updated to take an x parameter that will use rAB/(rA+rB) if you put in
    'resp' and will use the old suppresion metric if you pass 'supp.' Also, r_cut option.
    2022_09_30. Uses the usual stuff to calculate the old way of calculating suppression:
    ((rAB-sp) - (rA-sp) + (rB-sp)) / (rA-sp) + (rB-sp), and compares it with average weight:
    (wFG+wBG) / 2. Then scatters them.'''
    weight_df = weight_df.loc[weight_df.area==area]
    if area=='A1':
        col = 'indigo'
    elif area=='PEG':
        col = 'maroon'

    weight_df['avg_weight'] = (weight_df.weightsA + weight_df.weightsB) / 2
    if x=='supp':
        weight_df['avg_supp'] = (-weight_df['supp']) / (weight_df['bg_FR'] + weight_df['fg_FR'])
        xlabel = '(LS - rAB) / rA+rB'
    elif x=='resp':
        weight_df['avg_supp'] = (weight_df['combo_FR']) / (weight_df['bg_FR'] + weight_df['fg_FR'])
        # weight_df = weight_df.loc[(weight_df.combo_FR >= 0) & (weight_df.bg_FR >= 0) &
        #                           (weight_df.fg_FR >= 0)]
        xlabel = 'rAB / (rA + rB)'
        # weight_df = weight_df.loc[(weight_df.avg_supp >= 0) & (weight_df.avg_supp <= 2)]


    if thresh:
        weight_df, _ = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)
    if snr_threshold:
        weight_df = weight_df.loc[(weight_df.bg_snr >= snr_threshold) & (weight_df.fg_snr >= snr_threshold)]

    if r_cut:
        weight_df = weight_df.dropna(axis=0, subset='r')
        weight_df = weight_df.loc[weight_df.r >= r_cut]
    else:
        r_cut = 'None'
    area = weight_df.area.unique()[0]

    # from scipy.stats import pearsonr
    # corr, pp = pearsonr(quad.avg_supp, quad.avg_weight)
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.scatter(x=weight_df.avg_supp, y=weight_df.avg_weight, s=1, color=col)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=10)
    ax.set_ylabel('Mean Weights (wFG+wBG)/2', fontsize=10, fontweight='bold')
    # ax.plot([0,1], [0,1], color='black')
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    mini, maxi = np.min([ymin, xmin]), np.max([ymax, xmax])
    ax.set_ylim(-0.25, 1.25), ax.set_xlim(-0.25, 1.25)

    Y, X = weight_df.avg_weight, weight_df.avg_supp
    reg = stats.linregress(X, Y)
    x = np.asarray(ax.get_xlim())
    x = np.asarray([x[0]+0.05, x[1]-0.05])
    y = reg.slope * x + reg.intercept
    ax.plot(x, y, color='black', label=f"coef: {reg.rvalue:.3f}\n"
                                          f"p = {reg.pvalue:.3f}")
    # ax.plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
    #                                       f"coef: {reg.rvalue:.3f}\n"
    #                                       f"p = {reg.pvalue:.3f}")
    ax.legend()

    ax.set_title(f'{area} - snr >= {snr_threshold} -  fr >= {thresh} - r >= {r_cut}\nn={weight_df.shape[0]}\n', fontsize=10, fontweight='bold')
    fig.tight_layout()


def plot_all_weight_comparisons(df, fr_thresh=0.03, snr_threshold=0.12, r_thresh=0.6, strict_r=True,
                                weight_lim=[-0.5,2], summary=True, sep_hemi=False, sort_category=None):
    '''2022_11_08. Made for SFN/APAN poster panel 4, it displays the different fit epochs across a dataframe labeled
    with multiple different animals. FR and R I used for the poster was 0.03 and 0.6. Strict_r basically should always
    stay True at this point'''
    areas = list(df.area.unique())
    areas.sort()

    if sep_hemi == True:
        colors = ['mediumorchid', 'darkorange', 'orangered', 'green', 'yellow', 'blue']
    else:
        if sort_category:
            colors = ['red', 'navy', 'limegreen', 'magenta', 'teal', 'orange']
            df = df.copy()
            df['animal'] = df[sort_category]

            # voc_labels = {'Yes': 'Vocalization', 'No': 'Non-vocalization'}
            # df['animal'] = df['Vocalization'].map(voc_labels)
        else:
            df['animal'] = [cc.split('_')[0] for cc in df['animal']]
            colors = ['mediumorchid', 'darkorange', 'yellow', 'blue', 'green', 'pink']


    # This can be mushed into one liners using list comprehension and show_suffixes
    if fr_thresh:
        quad3 = df.loc[(df.bg_FR_start >= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                               & (df.bg_FR_end >= fr_thresh) & (df.fg_FR_end >= fr_thresh)]

        quad2 = df.loc[(np.abs(df.bg_FR_start) <= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                       & (np.abs(df.bg_FR_end) <= fr_thresh) & (df.fg_FR_end >= fr_thresh)]

        quad6 = df.loc[(df.bg_FR_start >= fr_thresh) & (np.abs(df.fg_FR_start) <= fr_thresh)
                       & (df.bg_FR_end >= fr_thresh) & (np.abs(df.fg_FR_end) <= fr_thresh)]
        print("FR thresh did happen")

    elif snr_threshold:
        print("FR thresh didn't happen")
        quad3 = df.loc[(df.bg_snr_start >= snr_threshold) & (df.fg_snr_start >= snr_threshold)
                       & (df.bg_snr_end >= snr_threshold) & (df.fg_snr_end >= snr_threshold)]

        quad2 = df.loc[(df.bg_snr_start < snr_threshold) & (df.fg_snr_start > snr_threshold)
                       & (df.bg_snr_end < snr_threshold) & (df.fg_snr_end > snr_threshold)]

        quad6 = df.loc[(df.bg_snr_start > snr_threshold) & (df.fg_snr_start < snr_threshold)
                       & (df.bg_snr_end > snr_threshold) & (df.fg_snr_end < snr_threshold)]
    else:
        raise ValueError('Need a threshold for this one, either FR or snr.')

    if weight_lim:
        quad3 = quad3.loc[((quad3[f'weightsA_start'] >= weight_lim[0]) & (quad3[f'weightsA_start'] <= weight_lim[1])) &
                          ((quad3[f'weightsB_start'] >= weight_lim[0]) & (quad3[f'weightsB_start'] <= weight_lim[1])) &
                          ((quad3[f'weightsA_end'] >= weight_lim[0]) & (quad3[f'weightsA_end'] <= weight_lim[1])) &
                          ((quad3[f'weightsB_end'] >= weight_lim[0]) & (quad3[f'weightsB_end'] <= weight_lim[1]))]

        quad2 = quad2.loc[((quad2[f'weightsA_start'] >= weight_lim[0]) & (quad2[f'weightsA_start'] <= weight_lim[1])) &
                          ((quad2[f'weightsB_start'] >= weight_lim[0]) & (quad2[f'weightsB_start'] <= weight_lim[1])) &
                          ((quad2[f'weightsA_end'] >= weight_lim[0]) & (quad2[f'weightsA_end'] <= weight_lim[1])) &
                          ((quad2[f'weightsB_end'] >= weight_lim[0]) & (quad2[f'weightsB_end'] <= weight_lim[1]))]

        quad6 = quad6.loc[((quad6[f'weightsA_start'] >= weight_lim[0]) & (quad6[f'weightsA_start'] <= weight_lim[1])) &
                          ((quad6[f'weightsB_start'] >= weight_lim[0]) & (quad6[f'weightsB_start'] <= weight_lim[1])) &
                          ((quad6[f'weightsA_end'] >= weight_lim[0]) & (quad6[f'weightsA_end'] <= weight_lim[1])) &
                          ((quad6[f'weightsB_end'] >= weight_lim[0]) & (quad6[f'weightsB_end'] <= weight_lim[1]))]

    fig, ax = plt.subplots(1, 4, figsize=(13, 6), sharey=True)
    ax = np.ravel(ax)


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
                            y=[np.median(animal_df[f'weightsA_start']), np.median(animal_df[f'weightsA_end'])],
                               label=f'{an} (n={len(animal_df)})', color=colors[ee], alpha=alph)#, marker=symbols[cnt])
            ax[num].scatter(x=['FG_start', 'FG_end'],
                            y=[np.median(animal_df[f'weightsB_start']), np.median(animal_df[f'weightsB_end'])],
                               color=colors[ee], alpha=alph) #, marker=symbols[cnt])
            ax[num].errorbar(x=['BG_start', 'BG_end'],
                            y=[np.median(animal_df[f'weightsA_start']), np.median(animal_df[f'weightsA_end'])],
                           yerr=[stats.sem(animal_df[f'weightsA_start']), stats.sem(animal_df[f'weightsA_end'])], xerr=None,
                           color=colors[ee], alpha=alph)
            ax[num].errorbar(x=['FG_start', 'FG_end'],
                            y=[np.median(animal_df[f'weightsB_start']), np.median(animal_df[f'weightsB_end'])],
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
                            y=[np.median(filt[f'weightsA_start']), np.median(filt[f'weightsA_end'])],
                            label=f'Total (n={len(filt)})', color='black')  # , marker=symbols[cnt])
            ax[num].scatter(x=['FG_start', 'FG_end'],
                            y=[np.median(filt[f'weightsB_start']), np.median(filt[f'weightsB_end'])],
                            color='black')  # , marker=symbols[cnt])
            ax[num].errorbar(x=['BG_start', 'BG_end'],
                             y=[np.median(filt[f'weightsA_start']), np.median(filt[f'weightsA_end'])],
                             yerr=[stats.sem(filt[f'weightsA_start']), stats.sem(filt[f'weightsA_end'])],
                             xerr=None, color='black')
            ax[num].errorbar(x=['FG_start', 'FG_end'],
                             y=[np.median(filt[f'weightsB_start']), np.median(filt[f'weightsB_end'])],
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
        # ax[2].set_yticklabels([0.3,0.4,0.5,0.6,0.7,0.8])
        ax[num+len(areas)].set_xticklabels(['0-0.5s\nBG', '0.5-1s\nBG', '0-0.5s\nFG', '0.5-1s\nFG'], fontsize=12, fontweight='bold')
        ax[num+len(areas)].set_title(f'{aa} - Respond to only\none sound alone', fontsize=14, fontweight='bold')

    fig.suptitle(f"r >= {r_thresh}, FR >= {fr_thresh}, snr >= {snr_threshold}, "
                 f"strict_r={strict_r}", fontweight='bold', fontsize=10)
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

        ax[axn].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue', linewidth=2)
        ax[axn].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen', linewidth=2)
        # ax[axn].set_ylabel('Percentage of cells', fontweight='bold', fontsize=12)
        ax[axn].set_title(f"{aaa} - 0-0.5s", fontweight='bold', fontsize=12)
        ax[axn].tick_params(axis='both', which='major', labelsize=10)
        ax[1].set_xlabel("Mean Weight", fontweight='bold', fontsize=12)
        ax[3].set_xlabel("Mean Weight", fontweight='bold', fontsize=12)

        axn += 1

        na, xa = np.histogram(to_plot['weightsA_end'], bins=edges)
        na = na / na.sum() * 100
        nb, xb = np.histogram(to_plot['weightsB_end'], bins=edges)
        nb = nb / nb.sum() * 100

        ax[axn].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue', linewidth=2)
        ax[axn].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen', linewidth=2)
        ax[axn].legend(('Background', 'Foreground'), fontsize=12)
        ax[0].set_ylabel('Percentage of cells', fontweight='bold', fontsize=12)
        ax[1].set_ylabel('Percentage of cells', fontweight='bold', fontsize=12)
        ax[axn].set_title(f"{aaa} - 0.5-1s", fontweight='bold', fontsize=12)
        ax[axn].tick_params(axis='both', which='major', labelsize=10)
        # ax[axn].set_xlabel("Mean Weight", fontweight='bold', fontsize=14)

        axn += 1
    fig.tight_layout()

    return stat_list


def plot_weight_prediction_comparisons(df, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True, pred=False, weight_lim=[-1,2]):
    '''2023_05_10. Moved from OLP_home. I believe this is the same as plot_all_weight_comparisons above but
    can handle when there is a prediction signal. Doesn't work if not.'''
    areas = list(df.area.unique())

    # This can be mushed into one liners using list comprehension and show_suffixes
    quad3 = df.loc[(df.bg_FR_start >= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                           & (df.bg_FR_end >= fr_thresh) & (df.fg_FR_end >= fr_thresh)]

    fig, ax = plt.subplots(1, 2, figsize=(7, 6), sharey=True)
    ax = np.ravel(ax)

    stat_list, filt_list = [], []
    for num, aa in enumerate(areas):
        area_df = quad3.loc[quad3.area == aa]
        if strict_r == False:
            filt = area_df.loc[(area_df[f'r_start'] >= r_thresh) & (area_df[f'r_end'] >= r_thresh)]
        else:
            filt = area_df.loc[(area_df[f'r_start'] >= r_thresh) & (area_df[f'r_end'] >= r_thresh) &
                               (area_df[f'r_start_pred'] >= r_thresh) & (area_df[f'r_end_pred'] >= r_thresh)]

        if weight_lim:
            # suffs = ['', '_start', '_end', '_pred', '_start_pred', '_end_pred']
            suffs = [aa[8:] for aa in filt.columns.to_list() if "weightsA" in aa and not 'nopost' in aa]
            for ss in suffs:
                filt = filt.loc[((filt[f'weightsA{ss}'] >= weight_lim[0]) & (filt[f'weightsA{ss}'] <= weight_lim[1])) &
                                ((filt[f'weightsB{ss}'] >= weight_lim[0]) & (filt[f'weightsB{ss}'] <= weight_lim[1]))]

        pred_suff, colors = ['', '_pred'], ['black', 'red']
        for nn, pr in enumerate(pred_suff):
            ax[num].scatter(x=['BG_start', 'BG_end'],
                                 y=[np.nanmean(filt[f'weightsA_start{pr}']), np.nanmean(filt[f'weightsA_end{pr}'])],
                                 label=f'Total{pr} (n={len(filt)})', color=colors[nn])  # , marker=symbols[cnt])
            ax[num].scatter(x=['FG_start', 'FG_end'],
                                 y=[np.nanmean(filt[f'weightsB_start{pr}']), np.nanmean(filt[f'weightsB_end{pr}'])],
                                 color=colors[nn])  # , marker=symbols[cnt])
            ax[num].errorbar(x=['BG_start', 'BG_end'],
                                  y=[np.nanmean(filt[f'weightsA_start{pr}']), np.nanmean(filt[f'weightsA_end{pr}'])],
                                  yerr=[stats.sem(filt[f'weightsA_start{pr}']), stats.sem(filt[f'weightsA_end{pr}'])],
                                  xerr=None, color=colors[nn])
            ax[num].errorbar(x=['FG_start', 'FG_end'],
                                  y=[np.nanmean(filt[f'weightsB_start{pr}']), np.nanmean(filt[f'weightsB_end{pr}'])],
                                  yerr=[stats.sem(filt[f'weightsB_start{pr}']), stats.sem(filt[f'weightsB_end{pr}'])],
                                  xerr=None, color=colors[nn])

            ax[num].legend(fontsize=8, loc='upper right')

            BGsBGe = stats.ttest_ind(filt[f'weightsA_start{pr}'], filt[f'weightsA_end{pr}'])
            FGsFGe = stats.ttest_ind(filt[f'weightsB_start{pr}'], filt[f'weightsB_end{pr}'])
            BGsFGs = stats.ttest_ind(filt[f'weightsA_start{pr}'], filt[f'weightsB_start{pr}'])
            BGeFGe = stats.ttest_ind(filt[f'weightsA_end{pr}'], filt[f'weightsB_end{pr}'])

            tts = {f"BGsBGe_{aa}{pr}": BGsBGe.pvalue, f"FGsFGe_{aa}{pr}": FGsFGe.pvalue,
                   f"BGsFGs_{aa}{pr}": BGsFGs.pvalue, f"BGeFGe_{aa}{pr}": BGeFGe.pvalue}
            print(tts)
            stat_list.append(tts), filt_list.append(filt)

        ax[0].set_ylabel(f'Mean Weight', fontsize=14, fontweight='bold')
        ax[num].set_title(f'{aa} - Respond to both\n BG and FG alone', fontsize=14, fontweight='bold')
        ax[num].tick_params(axis='both', which='major', labelsize=10)
        ax[num].set_xticklabels(['0-0.5s\nBG', '0.5-1s\nBG', '0-0.5s\nFG', '0.5-1s\nFG'], fontsize=12,
                                     fontweight='bold')

    fig.suptitle(f"r >= {r_thresh}, FR >= {fr_thresh}, strict_r={strict_r}, synth={df.synth_kind.unique()}", fontweight='bold', fontsize=10)
    fig.tight_layout()


def plot_partial_fit_bar(df, fr_thresh=0.03, r_thresh=0.6, suffixes=['', '_start', '_end'],
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


def sound_stats_half_compare(sound_df, suffixes=['_start', '_end'], metric='Tstationary', show='N'):
    '''2022_12_06. Either going to be very useful for many things or not at all. Will take three different mentrics
    from the sound_df dataframe and plot the connected points and then throw a bar in the background to summarize.'''
    fig, ax = plt.subplots(1, 1, figsize=(5, 7))

    to_plot = sound_df.loc[sound_df.synth_kind==show]
    to_plot_bg, to_plot_fg = to_plot.loc[to_plot.type=='BG'], to_plot.loc[to_plot.type=='FG']

    n = 2
    xv = np.arange(n)
    nc = len(suffixes)
    xvv = np.repeat(xv, nc)
    if len(suffixes)==3:
        offsets, width = np.asarray([-0.3, 0, 0.3]), 0.2
    elif len(suffixes) == 2:
        offsets, width = np.asarray([-0.2, 0.2]), 0.3
    else:
        offsets, width = np.asarray([0]), 0.5
    offsetsvv = np.tile(offsets, n)
    X = list(xvv + offsetsvv)

    colors = ['deepskyblue'] * len(suffixes) + ['yellowgreen'] * len(suffixes)
    plot_list_bg = [np.nanmean(to_plot_bg[f'{metric}{ss}']) for ss in suffixes]
    plot_list_fg = [np.nanmean(to_plot_fg[f'{metric}{ss}']) for ss in suffixes]

    ax.bar(x=X[:len(suffixes)], height=plot_list_bg, color='deepskyblue', width=width)
    ax.bar(x=X[len(suffixes):], height=plot_list_fg, color='yellowgreen', width=width)

    plot_bg = [to_plot_bg[f'{metric}{ss}'] for ss in suffixes]
    plot_fg = [to_plot_fg[f'{metric}{ss}'] for ss in suffixes]

    for cnt in range(to_plot_bg.shape[0]):
        ax.plot(X[:len(suffixes)], plot_bg, 'ko-', ms=3)
    for cnt in range(to_plot_fg.shape[0]):
        ax.plot(X[len(suffixes):], plot_fg, 'ko-', ms=3)

    ax.set_xticks(X)
    bg_labels, fg_labels = [f'BG\n{ss}' for ss in suffixes], [f'FG\n{ss}' for ss in suffixes]
    ax.set_xticklabels(bg_labels+fg_labels, fontsize=10, fontweight='bold', rotation=0)
    ax.set_ylabel(f"{metric}: {show}", fontsize=10, fontweight='bold')

    fig.tight_layout()


def plot_dynamic_errors(full_df, dyn_kind='all', snr_threshold=0.12, thresh=None, r_cut=None):
    '''2023_07_05. Rework of a function I made a few months back. Takes a dataframe that has had the
    dynamic calculations done to it using script_dynamic.py and enqueue_dynamic.py which call
    ofit.calc_dyn_metrics(). The dataframe is created and pooled with your existing, main df using
    ohel.merge_dynamic_error().'''
    areas = full_df.area.unique().tolist()
    areas.sort()

    if dyn_kind == 'fh' or dyn_kind == 'hf':
        fig, axes = plt.subplots(2, 1, figsize=(10,6))
        dyn_plot = [dyn_kind]
    elif dyn_kind == 'all':
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharey=True)
        dyn_plot = ['fh', 'hf']

    cc = 0
    for dd in dyn_plot:
        for cnt, ar in enumerate(areas):
            dyn_df = full_df.loc[full_df.dyn_kind == dd]
            area_df = dyn_df.loc[dyn_df.area == ar]
            n_before_thresh = len(area_df)

            if dd == 'hf':
                if thresh:
                    quad3 = area_df.loc[area_df.full_bg_FR >= thresh]
                if snr_threshold:
                    quad3 = area_df.loc[area_df.bg_snr >= snr_threshold]
                alone_col = 'yellowgreen'

            elif dd == 'fh':
                if thresh:
                    quad3 = area_df.loc[area_df.full_fg_FR >= thresh]
                if snr_threshold:
                    quad3 = area_df.loc[area_df.fg_snr >= snr_threshold]
                alone_col = 'deepskyblue'

            if r_cut:
                quad3 = quad3.dropna(axis=0, subset='r')
                quad3 = quad3.loc[quad3.r >= 0.6]

            n_after_thresh = len(quad3)

            E_full = np.array(area_df.E_full.to_list())#[:, 50:-50]
            E_alone = np.array(quad3.E_alone.to_list())#[:, 50:-50]

            full = quad3.groupby(['E_full']).mean()


            full_av = np.nanmean(E_full, axis=0)
            alone_av = np.nanmean(E_alone, axis=0)

            baseline = np.nanmean(alone_av[:int(alone_av.shape[0]/2)])

            se_full = E_full.std(axis=0) / np.sqrt(E_full.shape[0])
            se_alone = E_alone.std(axis=0) / np.sqrt(E_alone.shape[0])

            if full_av.shape[0] == 200:
                time = (np.arange(0, full_av.shape[0]) / 100)
                time = time - 0.5
            else:
                time = (np.arange(0, full_av.shape[0]) / 100)

            axes[cc+cnt].plot(time, full_av, label='Full Error', color='black')
            axes[cc+cnt].plot(time, alone_av, label='Alone Error', color=alone_col)

            axes[cc+cnt].fill_between(time, (full_av - se_full*2), (full_av + se_full*2),
                                 alpha=0.4, color='black')
            axes[cc+cnt].fill_between(time, (alone_av - se_alone*2), (alone_av + se_alone*2),
                                 alpha=0.4, color=alone_col)

            axes[cc+cnt].legend()
            axes[cc+cnt].set_title(f"{ar} - {dd} - n={len(quad3)}", fontweight='bold', fontsize=10)
            axes[cc+cnt].set_xticks(np.arange(time[0],time[-1],0.5))
            ymin, ymax = axes[cc+cnt].get_ylim()
            axes[cc+cnt].vlines([0.5], ymin, ymax, colors='black', linestyles=':')
            if full_av.shape[0] == 200:
                axes[cc + cnt].vlines([0, 1], ymin, ymax, colors='black', linestyles='--')
            axes[cc+cnt].hlines([baseline], time[0], time[-1], colors='black', linestyles='--', lw=0.5)
        axes[-1].set_xlabel("Time (s)", fontweight='bold', fontsize=10)
        cc += len(dyn_plot)
    fig.tight_layout()


def plot_dynamic_site_errors(full_df, dyn_kind='fh', area='A1', thresh=0.03, r_cut=None):
    '''2023_07_05. Rework of a function I made a few months back. Takes a dataframe that has had the
    dynamic calculations done to it using script_dynamic.py and enqueue_dynamic.py which call
    ofit.calc_dyn_metrics(). The dataframe is created and pooled with your existing, main df using
    ohel.merge_dynamic_error().'''
    dyn_df = full_df.loc[full_df.dyn_kind == dyn_kind]
    area_df = dyn_df.loc[dyn_df.area == area]

    sites = [dd.split('-')[0] for dd in area_df.cellid]
    area_df['site'] = sites
    unique_sites = list(area_df.site.unique())

    dims = int(np.ceil(np.sqrt(len(area_df.site.unique()))))

    fig, axes = plt.subplots(dims, dims, figsize=(20, 15))#, sharey=True)
    axes = np.ravel(axes)

    for site, ax in zip(unique_sites, axes):
        site_df = area_df.loc[area_df['site']==site]
        if dyn_kind == 'hf':
            # quad3 = site_df.loc[site_df.full_bg_FR >= thresh]
            alone_col = 'yellowgreen'
        elif dyn_kind == 'fh':
            # quad3 = site_df.loc[site_df.full_fg_FR >= thresh]
            alone_col = 'deepskyblue'
        quad3 = site_df
        # if r_cut:
        #     quad3 = quad3.dropna(axis=0, subset='r')
        #     quad3 = quad3.loc[quad3.r >= 0.6]

        E_full = np.array(quad3.E_full.to_list())#[:, 50:-50]
        E_alone = np.array(quad3.E_alone.to_list())#[:, 50:-50]

        full_av = np.nanmean(E_full, axis=0)
        alone_av = np.nanmean(E_alone, axis=0)

        # baseline = np.nanmean(alone_av[:int(alone_av.shape[0]/2)])

        se_full = E_full.std(axis=0) / np.sqrt(E_full.shape[0])
        se_alone = E_alone.std(axis=0) / np.sqrt(E_alone.shape[0])

        if full_av.shape[0] == 200:
            time = (np.arange(0, full_av.shape[0]) / 100)
            time = time - 0.5
        else:
            time = (np.arange(0, full_av.shape[0]) / 100)

        ax.plot(time, full_av, label='Full Error', color='black')
        ax.plot(time, alone_av, label='Alone Error', color=alone_col)

        # ax.fill_between(time, (full_av - se_full*2), (full_av + se_full*2),
        #                      alpha=0.4, color='black')
        # ax.fill_between(time, (alone_av - se_alone*2), (alone_av + se_alone*2),
        #                      alpha=0.4, color=alone_col)

        ax.legend()
        ax.set_title(f"{site} - n={len(quad3)}", fontweight='bold', fontsize=8)
        ax.set_xticks(np.arange(time[0],time[-1],0.5))
        ymin, ymax = ax.get_ylim()
        # ax.vlines([0.5], ymin, ymax, colors='black', linestyles=':')
        # if full_av.shape[0] == 200:
        #     ax.vlines([0, 1], ymin, ymax, colors='black', linestyles=':')
        # ax.hlines([baseline], time[0], time[-1], colors='black', linestyles='--', lw=0.5)

    fig.suptitle(f"{area} - {dyn_kind}", fontsize=10, fontweight='bold')
    fig.tight_layout()


def weight_summary_histograms(filt, threshold=None, snr_threshold=0.12, r_cut=None, area='A1', bar=True):
    '''2023_07_14. Function to make figures 2B and 2C. Takes a prefiltered (just for
    olp_type, snr, layer, stuff like that), and makes a histogram of the weights,
    plots the means, and then plots the relative gain histogram.'''

    if threshold:
        filt, _ = ohel.quadrants_by_FR(filt, threshold=threshold, quad_return=3)
    if snr_threshold:
        filt = filt.loc[(filt.bg_snr >= snr_threshold) & (filt.fg_snr >= snr_threshold)]

    area_df = filt.loc[filt.area==area]
    if r_cut:
        area_df = area_df.dropna(axis=0, subset='r')
        area_df = area_df.loc[area_df.r >= r_cut]

    to_plot = area_df

    f = plt.figure(figsize=(12, 6))
    hist = plt.subplot2grid((10, 18), (0, 0), rowspan=5, colspan=8)
    mean = plt.subplot2grid((10, 18), (0, 9), rowspan=5, colspan=2)
    relhist = plt.subplot2grid((10, 18), (0, 12), rowspan=5, colspan=7)
    ax = [hist, mean, relhist]

    edges = np.arange(-0.3, 1.5, .05)
    na, xa = np.histogram(to_plot.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(to_plot.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[0].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue', linewidth=2)
    ax[0].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen', linewidth=2)
    ax[0].legend(('Background', 'Foreground'), fontsize=12, prop=dict(weight='bold'), labelspacing=0.25)

    ax[0].set_ylabel('Percentage of cells', fontweight='bold', fontsize=10)
    ax[0].set_title(f"{area}, BG+/FG+, n={len(to_plot)}", fontweight='bold', fontsize=10)
    ax[0].set_xlabel("Weight", fontweight='bold', fontsize=10)
    ax[0].tick_params(axis='both', which='major', labelsize=8)
    ymin, ymax = ax[0].get_ylim()

    BG1, FG1 = np.median(to_plot.weightsA), np.median(to_plot.weightsB)
    BG1sem, FG1sem = stats.sem(to_plot.weightsA), stats.sem(to_plot.weightsB)
    ttest1 = stats.ttest_ind(to_plot.weightsA, to_plot.weightsB)

    # ax[1].boxplot([to_plot.weightsA, to_plot.weightsB],
    #               positions=[1,2], patch_artist=True,
    #               boxprops=dict(facecolor='deepskyblue'), showmeans=True,
    #               showfliers=False)
    #
    # box_plot = ax[1].boxplot([to_plot.weightsA, to_plot.weightsB], notch=True,
    #                          boxprops=dict(facecolor='deepskyblue'))

    if bar:
        ax[1].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
        ax[1].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')

    else:
        ax[1].bar("BG", BG1, yerr=BG1sem, color='white')
        ax[1].bar("FG", FG1, yerr=FG1sem, color='white')

        ax[1].scatter(x=['BG', 'FG'], y=[BG1, FG1], color=['deepskyblue', 'yellowgreen'])
        ax[1].errorbar(x=['BG', 'FG'], y=[BG1, FG1], yerr=[BG1sem, FG1sem], ls='none')#, color=['deepskyblue', 'yellowgreen'])

    # ax[1].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
    # ax[1].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')

    ax[1].set_ylabel('Mean Weight', fontweight='bold', fontsize=10)
    ax[1].set_xticklabels(['BG','FG'], fontsize=8, fontweight='bold')
    ax[1].tick_params(axis='y', which='major', labelsize=8)
    if ttest1.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"{ttest1.pvalue:.3f}"
    ax[1].set_title(f"BG: {np.around(BG1,2)}, FG: {np.around(FG1,2)}\n{title}", fontsize=8)


    rel_weight = (to_plot.weightsB - to_plot.weightsA) / (to_plot.weightsB + to_plot.weightsA)
    supps = [cc for cc in rel_weight if cc < 0]
    percent_supp = np.around((len(supps) / len(rel_weight)) * 100, 1)
    # Filter dataframe to get rid of the couple with super weird, big or small weights
    rel = rel_weight.loc[rel_weight <= 2.5]
    rel = rel.loc[rel >= -2.5]

    sups = [cc for cc in rel if cc < 0]
    enhs = [cc for cc in rel if cc >= 0]

    sup_edges = np.arange(-2.4, 0.1, .1)
    enh_edges = np.arange(0, 2.5, .1)
    na, xa = np.histogram(sups, bins=sup_edges)
    nb, xb = np.histogram(enhs, bins=enh_edges)
    aa = na / (na.sum() + nb.sum()) * 100
    bb = nb / (na.sum() + nb.sum()) * 100

    ax[2].hist(xa[:-1], xa, weights=aa, histtype='step', color='tomato', fill=True)
    ax[2].hist(xb[:-1], xb, weights=bb, histtype='step', color='dodgerblue', fill=True)

    ax[2].legend(('FG Suppressed', 'FG Enhanced'), fontsize=12, prop=dict(weight='bold'), labelspacing=0.25)
    ax[2].set_ylabel('Percentage of cells', fontweight='bold', fontsize=10)
    ax[2].set_xlabel("Relative Gain (RG)", fontweight='bold', fontsize=10)
    if threshold:
        ax[2].set_title(f"r >= {r_cut}, FR_thresh >= {threshold}\n% suppressed: {percent_supp}", fontsize=8)
    elif snr_threshold:
        ax[2].set_title(f"r >= {r_cut}, snr_thresh >= {snr_threshold}\n% suppressed: {percent_supp}", fontsize=8)
    ax[2].set_xlim(-1.75,1.75)


def r_weight_comp_distribution(filt, increment=0.2, snr_threshold=0.12, threshold=0.03, area='A1'):
    '''2023_07_17. Supplemental figure to show what weight distributions across the various groups of r thresholds.
    Give it an increment that can divisble into 1 and it'll add a last panel for a percent stacked bar graph.'''
    if threshold:
        filt, _ = ohel.quadrants_by_FR(filt, threshold=threshold, quad_return=3)
        thresh = threshold
    if snr_threshold:
        filt = filt.loc[(filt.bg_snr >= snr_threshold) & (filt.fg_snr >= snr_threshold)]
        thresh = snr_threshold

    area_df = filt.loc[filt.area==area]
    incs = np.arange(0, 1, increment)
    plots = len(incs)

    fig, ax = plt.subplots(1, plots+1, figsize=(plots*2.5, 4))

    inc_lims = np.append(incs, 1)
    totals, total, maxs = {}, 0, []
    for cnt, inc in enumerate(list(incs)):
        r_df = area_df.dropna(axis=0, subset='r')
        r_df = r_df.loc[(r_df.r >= inc) & (r_df.r < inc_lims[cnt+1])]

        BG1, FG1 = np.mean(r_df.weightsA), np.mean(r_df.weightsB)
        BG1sem, FG1sem = stats.sem(r_df.weightsA), stats.sem(r_df.weightsB)
        ttest1 = stats.ttest_ind(r_df.weightsA, r_df.weightsB)

        # ax[1].bar("BG", BG1, yerr=BG1sem, color='white')
        # ax[1].bar("FG", FG1, yerr=FG1sem, color='white')
        # ax[1].scatter(x=['BG', 'FG'], y=[BG1, FG1], color=['deepskyblue', 'yellowgreen'])
        # ax[1].errorbar(x=['BG', 'FG'], y=[BG1, FG1], yerr=[BG1sem, FG1sem], ls='none')#, color=['deepskyblue', 'yellowgreen'])

        ax[cnt].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
        ax[cnt].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')

        ax[cnt].set_xticklabels(['BG','FG'], fontsize=8, fontweight='bold')
        # ax.tick_params(axis='y', which='major', labelsize=8)
        if cnt != 0:
            ax[cnt].set_yticklabels([])

        lil, big = ax[cnt].get_ylim()
        maxs.append(big)
        bin_name = f"r={np.around(inc, 1)}-{np.around(inc_lims[cnt + 1], 1)}"
        totals[bin_name] = len(r_df)
        total += len(r_df)

        if ttest1.pvalue < 0.001:
            title = 'p<0.001'
        else:
            title = f"p={ttest1.pvalue:.3f}"
        ax[cnt].set_title(f"{area} - {bin_name}\nBG: {np.around(BG1,2)}, FG: {np.around(FG1,2)}\n"
                          f"n={len(r_df)}\n{title}", fontsize=8)

    big_y = np.max(maxs)
    for aa in range(len(maxs)):
        ax[aa].set_ylim(0, big_y)

    ax[0].set_ylabel('Mean Weight', fontweight='bold', fontsize=10)

    from matplotlib import cm
    greys = cm.get_cmap('inferno', 12)
    cols = greys(np.linspace(0, 0.9, len(incs))).tolist()
    # cols.reverse()

    percents = [(pp/total)*100 for pp in totals.values()]
    names = [lbl for lbl in list(totals.keys())]

    for cc in range(len(names)):
        bottom = np.sum(percents[cc+1:])
        ax[-1].bar('total', height=percents[cc], bottom=bottom, color=cols[cc],
                     width=1, label=names[cc])#, edgecolor='white')

    ax[-1].legend(names, bbox_to_anchor=(0.8,1.025), loc="upper left")
    ax[-1].set_ylabel('Percent', fontweight='bold', fontsize=10)
    ax[-1].set_xticks([])
    ax[-1].set_xlim(-1,1)
    ax[-1].set_title(f'snr>={snr_threshold}\nn={total}', fontsize=7)

    fig.tight_layout()


def weight_summary_histograms_flanks(filt, snr_threshold=0.12, fr_thresh=0.03, r_cut=None, area='A1'):
    '''2023_07_14. Makes the complement to 2B and 2C, but this time uses the weights when units only are responsive
    to BG or FG alone.'''
    if fr_thresh:
        quad2 = filt.loc[(np.abs(filt.bg_FR_start) <= fr_thresh) & (filt.fg_FR_start >= fr_thresh)
                   & (np.abs(filt.bg_FR_end) <= fr_thresh) & (filt.fg_FR_end >= fr_thresh)]

        quad6 = filt.loc[(filt.bg_FR_start >= fr_thresh) & (np.abs(filt.fg_FR_start) <= fr_thresh)
                   & (filt.bg_FR_end >= fr_thresh) & (np.abs(filt.fg_FR_end) <= fr_thresh)]
        title_thresh = f'fr >= {fr_thresh}'

    elif snr_threshold:
        quad2 = filt.loc[(filt.bg_snr_start < snr_threshold) & (filt.fg_snr_start > snr_threshold)
                       & (filt.bg_snr_end < snr_threshold) & (filt.fg_snr_end > snr_threshold)]

        quad6 = filt.loc[(filt.bg_snr_start > snr_threshold) & (filt.fg_snr_start < snr_threshold)
                       & (filt.bg_snr_end > snr_threshold) & (filt.fg_snr_end < snr_threshold)]
        title_thresh = f'snr >= {snr_threshold}'

    area_df2, area_df6 = quad2.loc[quad2.area==area], quad6.loc[quad6.area==area]

    if r_cut:
        area_df2, area_df6 = area_df2.dropna(axis=0, subset='r'), area_df6.dropna(axis=0, subset='r')
        area_df2, area_df6 = area_df2.loc[area_df2.r >= r_cut], area_df6.loc[area_df6.r >= r_cut]

    to_plotFG, to_plotBG = area_df2, area_df6

    f = plt.figure(figsize=(12, 6))
    histBG = plt.subplot2grid((10, 18), (0, 0), rowspan=5, colspan=7)
    histFG = plt.subplot2grid((10, 18), (0, 8), rowspan=5, colspan=7, sharey=histBG)
    mean = plt.subplot2grid((10, 18), (0, 16), rowspan=5, colspan=2)
    ax = [histBG, histFG, mean]

    edges = np.arange(-0.3, 1.5, .05)
    na, xa = np.histogram(to_plotBG.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(to_plotFG.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[0].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue', linewidth=2)
    ax[1].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen', linewidth=2)
    ax[0].set_title(f"{area}: Respond to BG alone only\nn={len(to_plotBG.weightsA)}", fontweight='bold', fontsize=10)
    ax[1].set_title(f"{area}: Respond to FG alone only\nn={len(to_plotFG.weightsB)}", fontweight='bold', fontsize=10)
    ax[0].set_xlabel("Background Weight", fontsize=10, fontweight='bold')
    ax[1].set_xlabel("Foreground Weight", fontsize=10, fontweight='bold')
    # ax[0].legend('BG', fontsize=12, prop=dict(weight='bold'))
    # ax[1].legend('FG', fontsize=12, prop=dict(weight='bold'))
    ax[0].set_ylabel('Percentage of cells', fontweight='bold', fontsize=10)

    BG1, FG1 = np.mean(to_plotBG.weightsA), np.mean(to_plotFG.weightsB)
    BG1sem, FG1sem = stats.sem(to_plotBG.weightsA), stats.sem(to_plotFG.weightsB)
    ttest1 = stats.ttest_ind(to_plotBG.weightsA, to_plotFG.weightsB)

    ax[2].bar("BG", BG1, yerr=BG1sem, color='white')
    ax[2].bar("FG", FG1, yerr=FG1sem, color='white')

    ax[2].scatter(x=['BG', 'FG'], y=[BG1, FG1], color=['deepskyblue', 'yellowgreen'])
    ax[2].errorbar(x=['BG', 'FG'], y=[BG1, FG1], yerr=[BG1sem, FG1sem], ls='none')#, color=['deepskyblue', 'yellowgreen'])

    # ax[1].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
    # ax[1].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')

    ax[2].set_ylabel('Mean Weight', fontweight='bold', fontsize=10)
    ax[2].set_xticklabels(['BG','FG'], fontsize=8, fontweight='bold')
    ax[2].tick_params(axis='y', which='major', labelsize=8)
    if ttest1.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"{ttest1.pvalue:.3f}"
    ax[2].set_title(f"{title_thresh}\nBG: {np.around(BG1,2)}, FG: {np.around(FG1,2)}\n{title}", fontsize=8)


def snr_scatter(df, xcol='bg_snr', ycol='fg_snr', thresh=0.3, area='A1'):
    '''2022_09_06. Takes a dataframe and just scatters two of the columns from that dataframe.
    I was using it to check if weights are correlated with the concurrent sound's FR.'''
    df = df.loc[df.area==area]

    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    sb.scatterplot(x=xcol, y=ycol, data=df, ax=ax[0], s=3)
    ax[0].set_xlabel(xcol, fontsize=8, fontweight='bold')
    ax[0].set_ylabel(ycol, fontsize=8, fontweight='bold')
    xmin, xmax = ax[0].get_xlim()
    ymin, ymax = ax[1].get_ylim()
    ax[0].vlines([thresh], ymin, ymax, colors='black', linestyles=':', lw=1)
    ax[0].hlines([thresh], xmin, xmax, colors='black', linestyles=':', lw=1)
    size = len(df)
    snr3 = len(df.loc[(df.bg_snr >= thresh) & (df.fg_snr >= thresh)]) / size * 100
    snr6 = len(df.loc[(df.bg_snr < thresh) & (df.fg_snr < thresh)]) / size * 100
    ax[0].set_title(f'{df.area.unique()[0]}: thresh={thresh}\nAbove: {np.around(snr3,1)}%, Below: {np.around(snr6,1)}%', fontsize=10, fontweight='bold')

    edges = np.arange(0, 1, .05)
    na, xa = np.histogram(df[xcol], bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(df[ycol], bins=edges)
    nb = nb / nb.sum() * 100

    ax[1].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue', linewidth=2)
    ax[2].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen', linewidth=2)
    ax[1].set_xlabel(xcol, fontsize=8, fontweight='bold'), ax[2].set_xlabel(ycol, fontsize=8, fontweight='bold')
    ymin, ymax = ax[1].get_ylim()
    ymin1, ymax1 = ax[2].get_ylim()
    maxmax = np.max([ymax, ymax1])
    ax[1].vlines([thresh], ymin, maxmax, colors='black', linestyles=':', lw=1)
    ax[2].vlines([thresh], ymin, maxmax, colors='black', linestyles=':', lw=1)