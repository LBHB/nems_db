from nems.analysis.gammatone.gtgram import gtgram
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


def resp_weight_multi_scatter(weight_df, synth_kind='N', threshold=0.03, quads=3):
    '''Updated resp_weight_scatter to just plot all four combinations of FR/wt on one plot to avoid
    the silliness of four individual plots. Works the same, basically just give it a df.'''
    quad, _ = ohel.quadrants_by_FR(weight_df, threshold=threshold, quad_return=quads)
    quad = quad.loc[quad.synth_kind == synth_kind].copy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    xcol = ['bg_FR', 'fg_FR', 'bg_FR', 'fg_FR']
    ycol = ['weightsA', 'weightsA', 'weightsB', 'weightsB']
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

    axes[0].set_ylabel("BG Weight", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("FG Weight", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("BG FR", fontsize=12, fontweight='bold')
    axes[3].set_xlabel("FG FR", fontsize=12, fontweight='bold')


def plot_single_relative_gain_hist(df, threshold=0.05, quad_return=3, synth_kind=None):
    '''2022_09_06. Takes a DF (you filter it by types of sounds and area beforehand) and will plot
    a histogram showing the relative weights for a certain quadrant. It said distplot is deprecated
    so I'll have to figure something else out with histplot or displot, but coloring the histogram
    was a big task I couldn't figure out. Can kicked down the road.'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=quad_return)
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
    fig.tight_layout()


def sound_metric_scatter(df, x_metrics, y_metric, x_labels, area='A1', threshold=0.03,
                         jitter=[0.25,0.2,0.03],
                         quad_return=3, metric_filter=None, synth_kind='N', bin_kind='11',
                         title_text=''):
    '''Makes a series of scatterplots that compare a stat of the sounds to some metric of data. In
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


    fig.suptitle(f"{title} - {synth_kind} - {title_text}", fontweight='bold', fontsize=10)
