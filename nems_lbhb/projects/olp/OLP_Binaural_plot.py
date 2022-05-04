from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import scipy.ndimage.filters as sf
from nems.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
from pathlib import Path
import glob
import nems_lbhb.projects.olp.OLP_helpers as ohel
import copy


def get_cell_names(dataf, show=True):
    filtered = dataf[['cellid', 'area']]
    cells = filtered.drop_duplicates()
    if show == True:
        print(cells)

    return cells


def get_pair_names(cell, dataf, show=True):
    filtered = dataf.loc[dataf.cellid == cell]
    sounds = filtered[['BG', 'FG']]
    uniques = sounds.drop_duplicates()
    if show == True:
        print(uniques)

    return uniques


def plot_binaural_psths(df, cellid, bg, fg, batch, save=False, close=False):
    '''Takes input of a data fram from ohel.calc_psth_metrics along with a cellid and pair of
    sounds and will plot all the spatial psth combos with spectrogram of sound. Can save.'''
    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = ohel.get_load_options(batch)
    rec = manager.get_recording(**options)

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())

    expt_params = manager.get_baphy_exptparams()  # Using Charlie's manager
    if len(expt_params) == 1:
        ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
    if len(expt_params) > 1:
        ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    BG_folder, FG_folder = ref_handle['BG_Folder'], ref_handle['FG_Folder']

    ## I could make this not dependent on DF if I add some code that loads the epochs of the cell
    ## that you inputted and applies that type label function, that's really all I'm using from df

    df_filtered = df[(df.BG == bg) & (df.FG == fg) & (df.cellid == cellid)]
    if len(df_filtered) == 0:
        pairs = get_pair_names(cellid, df, show=False)
        raise ValueError(f"The inputted BG: {bg} and FG: {fg} are not in {cellid}.\n"
                         f"Maybe try one of these:\n{pairs}")

    epochs = []
    name = df_filtered.epoch.loc[df_filtered.kind == '11'].values[0]
    bb, ff = name.split('_')[1], name.split('_')[2]
    bb1, ff1 = name.replace(ff, 'null'), name.replace(bb, 'null')
    epochs.append(bb1), epochs.append(ff1)

    name = df_filtered.epoch.loc[df_filtered.kind == '22'].values[0]
    bb, ff = name.split('_')[1], name.split('_')[2]
    bb2, ff2 = name.replace(ff, 'null'), name.replace(bb, 'null')
    epochs.append(bb2), epochs.append(ff2)
    epochs.extend(df_filtered.epoch.values)

    r = resp.extract_epochs(epochs)
    SR = df_filtered['SR'].values[0]

    f = plt.figure(figsize=(18, 9))
    psth11 = plt.subplot2grid((9, 8), (0, 0), rowspan=3, colspan=3)
    psth12 = plt.subplot2grid((9, 8), (0, 3), rowspan=3, colspan=3, sharey=psth11)
    psth21 = plt.subplot2grid((9, 8), (3, 0), rowspan=3, colspan=3, sharey=psth11)
    psth22 = plt.subplot2grid((9, 8), (3, 3), rowspan=3, colspan=3, sharey=psth11)
    specA1 = plt.subplot2grid((9, 8), (7, 0), rowspan=1, colspan=3)
    specB1 = plt.subplot2grid((9, 8), (8, 0), rowspan=1, colspan=3)
    specA2 = plt.subplot2grid((9, 8), (7, 3), rowspan=1, colspan=3)
    specB2 = plt.subplot2grid((9, 8), (8, 3), rowspan=1, colspan=3)
    psthbb = plt.subplot2grid((9, 8), (0, 6), rowspan=3, colspan=2, sharey=psth11)
    psthff = plt.subplot2grid((9, 8), (3, 6), rowspan=3, colspan=2, sharey=psth11)
    ax = [psth11, psth12, psth21, psth22, specA1, specB1, specA2, specB2, psthbb, psthff]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / options['rasterfs']) - prestim

    # r_mean = {e: np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
    r_mean = {e: np.squeeze(np.nanmean(r[e], axis=0)) - SR for e in epochs}

    epochs.extend(['lin11', 'lin12', 'lin21', 'lin22'])
    bg1, fg1, bg2, fg2 = epochs[0], epochs[1], epochs[2], epochs[3]
    r_mean['lin11'], r_mean['lin12'] = r_mean[bg1]+r_mean[fg1], r_mean[bg1]+r_mean[fg2]
    r_mean['lin21'], r_mean['lin22'] = r_mean[bg2]+r_mean[fg1], r_mean[bg2]+r_mean[fg2]

    colors = ['deepskyblue'] *3 + ['yellowgreen'] *3 + ['violet'] *3 + ['darksalmon'] *3 \
             + ['dimgray'] *4 + ['black'] *4
    styles = ['-'] *16 + [':'] *4
    ax_num = [0, 1, 8, 0, 2, 9, 2, 3, 8, 1, 3, 9, 0, 1, 2, 3, 0, 1, 2, 3]
    ep_num = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    labels = ['BG1'] *3 + ['FG1'] *3 + ['BG2'] *3 + ['FG2'] *3 \
             + ['BG1+FG1'] + ['BG1+FG2'] + ['BG2+FG1'] + ['BG2+FG2'] + ['LS'] *4

    for e, a, c, s, l in zip(ep_num, ax_num, colors, styles, labels):
        ax[a].plot(time, sf.gaussian_filter1d(r_mean[epochs[e]], sigma=2)
                   * options['rasterfs'], color=c, linestyle=s, label=l)

    ymin, ymax = ax[0].get_ylim()
    AXS = [0, 1, 2, 3, 8, 9]
    for AX in AXS:
        ax[AX].legend()
        ax[AX].vlines([0, 1.0], ymin, ymax, color='black', lw=0.75, ls='--')
        ax[AX].vlines(0.5, ymax * 0.9, ymax, color='black', lw=0.75, ls=':')
        if AX !=8 and AX !=9:
            ax[AX].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))
        else:
            ax[AX].set_xlim((-prestim * 0.15), (1 + (prestim * 0.25)))

        if AX == 0 or AX == 1 or AX == 8:
            plt.setp(ax[AX].get_xticklabels(), visible=False)
        if AX == 1 or AX == 3 or AX == 8 or AX == 9:
            plt.setp(ax[AX].get_yticklabels(), visible=False)
        if AX == 2 or AX == 3 or AX == 9:
            ax[AX].set_xlabel('Time(s)', fontweight='bold', fontsize=10)
        if AX == 0 or AX == 2:
            ax[AX].set_ylabel('Spikes', fontweight='bold', fontsize=10)

    ax[0].set_title(f"{cellid} - BG: {bg} - FG: {fg}", fontweight='bold', fontsize=12)

    bbn, ffn = bb[:2], ff[:2]
    bg_path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'{BG_folder}/{bbn}*.wav'))[0]
    fg_path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'{FG_folder}/{ffn}*.wav'))[0]

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    for AX in range(4,8):
        if AX == 4 or AX == 6:
            sfs, W = wavfile.read(bg_path)
        elif AX == 5 or AX == 7:
            sfs, W = wavfile.read(fg_path)
        spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
        ax[AX].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
        ax[AX].set_xlim(low, high)
        ax[AX].set_xticks([]), ax[AX].set_yticks([])
        ax[AX].set_xticklabels([]), ax[AX].set_yticklabels([])
        ax[AX].spines['top'].set_visible(False), ax[AX].spines['bottom'].set_visible(False)
        ax[AX].spines['left'].set_visible(False), ax[AX].spines['right'].set_visible(False)
    ax[4].set_ylabel(f"{bb.split('-')[0]}", fontweight='bold')
    ax[5].set_ylabel(f"{ff.split('-')[0]}", fontweight='bold')

    if save:
        site, animal, area, unit = cellid.split('-')[0], cellid[:3], df.area.loc[df.cellid == cellid].unique()[0], cellid[8:]
        path = f"/home/hamersky/OLP Binaural/{animal}/{area}/{site}/{unit}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Saving to {path + f'{cellid}-{bg}-{fg}.png'}")
        plt.savefig(path + f"{cellid}-{bg}-{fg}.png")
        if close:
            plt.close()


#
# def plot_binaural_psths(df, cellid, bg, fg, batch, save=False, close=False):
#     '''Takes input of a data fram from ohel.calc_psth_metrics along with a cellid and pair of
#     sounds and will plot all the spatial psth combos with spectrogram of sound. Can save.'''
#     manager = BAPHYExperiment(cellid=cellid, batch=batch)
#     options = ohel.get_load_options(batch)
#     rec = manager.get_recording(**options)
#
#     rec['resp'] = rec['resp'].extract_channels([cellid])
#     resp = copy.copy(rec['resp'].rasterize())
#
#     expt_params = manager.get_baphy_exptparams()  # Using Charlie's manager
#     if len(expt_params) == 1:
#         ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
#     if len(expt_params) > 1:
#         ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
#     BG_folder, FG_folder = ref_handle['BG_Folder'], ref_handle['FG_Folder']
#
#     ## I could make this not dependent on DF if I add some code that loads the epochs of the cell
#     ## that you inputted and applies that type label function, that's really all I'm using from df
#
#     df_filtered = df[(df.BG == bg) & (df.FG == fg) & (df.cellid == cellid)]
#     if len(df_filtered) == 0:
#         pairs = get_pair_names(cellid, df, show=False)
#         raise ValueError(f"The inputted BG: {bg} and FG: {fg} are not in {cellid}.\n"
#                          f"Maybe try one of these:\n{pairs}")
#
#     epochs = []
#     name = df_filtered.epoch.loc[df_filtered.kind == '11'].values[0]
#     bb, ff = name.split('_')[1], name.split('_')[2]
#     bb1, ff1 = name.replace(ff, 'null'), name.replace(bb, 'null')
#     epochs.append(bb1), epochs.append(ff1)
#
#     name = df_filtered.epoch.loc[df_filtered.kind == '22'].values[0]
#     bb, ff = name.split('_')[1], name.split('_')[2]
#     bb2, ff2 = name.replace(ff, 'null'), name.replace(bb, 'null')
#     epochs.append(bb2), epochs.append(ff2)
#     epochs.extend(df_filtered.epoch.values)
#
#     r = resp.extract_epochs(epochs)
#     SR = df_filtered['SR'].values[0]
#
#     f = plt.figure(figsize=(15, 9))
#     psth11 = plt.subplot2grid((9, 6), (0, 0), rowspan=3, colspan=3)
#     psth12 = plt.subplot2grid((9, 6), (0, 3), rowspan=3, colspan=3, sharey=psth11)
#     psth21 = plt.subplot2grid((9, 6), (3, 0), rowspan=3, colspan=3, sharey=psth11)
#     psth22 = plt.subplot2grid((9, 6), (3, 3), rowspan=3, colspan=3, sharey=psth11)
#     specA1 = plt.subplot2grid((9, 6), (7, 0), rowspan=1, colspan=3)
#     specB1 = plt.subplot2grid((9, 6), (8, 0), rowspan=1, colspan=3)
#     specA2 = plt.subplot2grid((9, 6), (7, 3), rowspan=1, colspan=3)
#     specB2 = plt.subplot2grid((9, 6), (8, 3), rowspan=1, colspan=3)
#     ax = [psth11, psth12, psth21, psth22, specA1, specB1, specA2, specB2]
#
#     prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
#     time = (np.arange(0, r[epochs[0]].shape[-1]) / options['rasterfs']) - prestim
#
#     # r_mean = {e: np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
#     r_mean = {e: np.squeeze(np.nanmean(r[e], axis=0)) - SR for e in epochs}
#
#     epochs.extend(['lin11', 'lin12', 'lin21', 'lin22'])
#     bg1, fg1, bg2, fg2 = epochs[0], epochs[1], epochs[2], epochs[3]
#     r_mean['lin11'], r_mean['lin12'] = r_mean[bg1]+r_mean[fg1], r_mean[bg1]+r_mean[fg2]
#     r_mean['lin21'], r_mean['lin22'] = r_mean[bg2]+r_mean[fg1], r_mean[bg2]+r_mean[fg2]
#
#     colors = ['deepskyblue'] *2 + ['yellowgreen'] *2 + ['violet'] *2 + ['darksalmon'] *2 \
#              + ['dimgray'] *4 + ['black'] *4
#     styles = ['-'] *12 + [':'] *4
#     ax_num = [0, 1, 0, 2, 2, 3, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3]
#     ep_num = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#     labels = ['BG1'] *2 + ['FG1'] *2 + ['BG2'] *2 + ['FG2'] *2 \
#              + ['BG1+FG1'] + ['BG1+FG2'] + ['BG2+FG1'] + ['BG2+FG2'] + ['LS'] *4
#
#     for e, a, c, s, l in zip(ep_num, ax_num, colors, styles, labels):
#         ax[a].plot(time, sf.gaussian_filter1d(r_mean[epochs[e]], sigma=2)
#                    * options['rasterfs'], color=c, linestyle=s, label=l)
#
#     ymin, ymax = ax[0].get_ylim()
#     for AX in range(4):
#         ax[AX].legend()
#         ax[AX].vlines([0, 1.0], ymin, ymax, color='black', lw=0.75, ls='--')
#         ax[AX].vlines(0.5, ymax * 0.9, ymax, color='black', lw=0.75, ls=':')
#         ax[AX].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))
#         if AX == 0 or AX == 1:
#             plt.setp(ax[AX].get_xticklabels(), visible=False)
#         if AX == 1 or AX == 3:
#             plt.setp(ax[AX].get_yticklabels(), visible=False)
#         if AX == 2 or AX == 3:
#             ax[AX].set_xlabel('Time(s)', fontweight='bold', fontsize=10)
#         if AX == 0 or AX == 2:
#             ax[AX].set_ylabel('Spikes', fontweight='bold', fontsize=10)
#
#     ax[0].set_title(f"{cellid} - BG: {bg} - FG: {fg}", fontweight='bold', fontsize=12)
#
#     bbn, ffn = bb[:2], ff[:2]
#     bg_path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
#                         f'{BG_folder}/{bbn}*.wav'))[0]
#     fg_path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
#                         f'{FG_folder}/{ffn}*.wav'))[0]
#
#     xf = 100
#     low, high = ax[0].get_xlim()
#     low, high = low * xf, high * xf
#
#     for AX in range(4,8):
#         if AX == 4 or AX == 6:
#             sfs, W = wavfile.read(bg_path)
#         elif AX == 5 or AX == 7:
#             sfs, W = wavfile.read(fg_path)
#         spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
#         ax[AX].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
#         ax[AX].set_xlim(low, high)
#         ax[AX].set_xticks([]), ax[AX].set_yticks([])
#         ax[AX].set_xticklabels([]), ax[AX].set_yticklabels([])
#         ax[AX].spines['top'].set_visible(False), ax[AX].spines['bottom'].set_visible(False)
#         ax[AX].spines['left'].set_visible(False), ax[AX].spines['right'].set_visible(False)
#     ax[4].set_ylabel(f"{bb.split('-')[0]}", fontweight='bold')
#     ax[5].set_ylabel(f"{ff.split('-')[0]}", fontweight='bold')
#
#     if save:
#         site, animal, area, unit = cellid.split('-')[0], cellid[:3], df.area.loc[df.cellid == cellid].unique()[0], cellid[8:]
#         path = f"/home/hamersky/OLP Binaural/{animal}/{area}/{site}/{unit}/"
#         Path(path).mkdir(parents=True, exist_ok=True)
#         print(f"Saving to {path + f'{cellid}-{bg}-{fg}.png'}")
#         plt.savefig(path + f"{cellid}-{bg}-{fg}.png")
#         if close:
#             plt.close()
