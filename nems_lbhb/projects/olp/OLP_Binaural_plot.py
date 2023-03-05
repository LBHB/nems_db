from matplotlib import pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from projects.olp.OLP_analysis_main import sound_df
from scipy import stats
import scipy.ndimage.filters as sf
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
from pathlib import Path
import glob
import nems_lbhb.projects.olp.OLP_helpers as ohel
import copy
sound_df=[]

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

    expt_params = manager.get_baphy_exptparams()
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

    colors = ['deepskyblue'] *3 + ['violet'] *3 + ['yellowgreen'] *3 + ['darksalmon'] *3 \
             + ['dimgray'] *4 + ['black'] *4
    styles = ['-'] *16 + [':'] *4
    ax_num = [0, 1, 8, 2, 3, 8, 0, 2, 9, 1, 3, 9, 0, 1, 2, 3, 0, 1, 2, 3]
    ep_num = [0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    labels = ['BG1'] *3 + ['BG2'] *3 + ['FG1'] *3 + ['FG2'] *3 \
             + ['BG1+FG1'] + ['BG1+FG2'] + ['BG2+FG1'] + ['BG2+FG2'] + ['LS'] *4

    for e, a, c, s, l in zip(ep_num, ax_num, colors, styles, labels):
        ax[a].plot(time, sf.gaussian_filter1d(r_mean[epochs[e]], sigma=2)
                   * options['rasterfs'], color=c, linestyle=s, label=l)

    ymin, ymax = ax[0].get_ylim()
    AXS = [0, 1, 2, 3, 8, 9]
    for AX, tt, aab, bab, ali, bli, prf in zip(range(4), df_filtered.kind, df_filtered.AcorAB,
                                          df_filtered.BcorAB, df_filtered.AcorLin,
                                               df_filtered.BcorLin, df_filtered.pref):
        ax[AX].legend((f'BG{tt[0]}, corr={np.around(aab, 3)}',
                       f'FG{tt[1]}, corr={np.around(bab, 3)}',
                       f'BG{tt[0]}+FG{tt[1]}',
                       f'LS, Acorr={np.around(ali, 3)}\nBcorr={np.around(bli, 3)}\npref={np.around(prf, 3)}'))
    for AX in AXS:
        ax[AX].vlines([0, 1.0], ymin, ymax, color='black', lw=0.75, ls='--')
        ax[AX].vlines(0.5, ymax * 0.9, ymax, color='black', lw=0.75, ls=':')
        ax[AX].spines['right'].set_visible(True), ax[AX].spines['top'].set_visible(True)
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


def binaural_weight_hist(df, threshold=0.05, area='A1', stat='mean'):
    edges = np.arange(-1,2,.05)

    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=3)
    quad = quad.loc[quad.area == area]

    f = plt.figure(figsize=(15, 12))
    hist11 = plt.subplot2grid((13, 16), (0, 0), rowspan=5, colspan=3)
    mean11 = plt.subplot2grid((13, 16), (0, 4), rowspan=5, colspan=2)
    hist12 = plt.subplot2grid((13, 16), (0, 8), rowspan=5, colspan=3, sharey=hist11)
    mean12 = plt.subplot2grid((13, 16), (0, 12), rowspan=5, colspan=2, sharey=mean11)
    hist21 = plt.subplot2grid((13, 16), (7, 0), rowspan=5, colspan=3, sharey=hist11)
    mean21 = plt.subplot2grid((13, 16), (7, 4), rowspan=5, colspan=2, sharey=mean11)
    hist22 = plt.subplot2grid((13, 16), (7, 8), rowspan=5, colspan=3, sharey=hist11)
    mean22 = plt.subplot2grid((13, 16), (7, 12), rowspan=5, colspan=2, sharey=mean11)
    ax = [hist11, hist12, hist21, hist22, mean11, mean12, mean21, mean22]

    dfs = [quad.loc[quad.kind == '11'], quad.loc[quad.kind == '12'],
           quad.loc[quad.kind == '21'], quad.loc[quad.kind == '22']]
    titles = ['BG Contra/FG Contra', 'BG Contra/FG Ipsi', 'BG Ipsi/FG Contra', 'BG Ipsi/FG Ipsi']
    types = ['11', '12', '21', '22']

    ttests = {}
    for aa, (DF, tt) in enumerate(zip(dfs, titles)):
        na, xa = np.histogram(DF.weightsA, bins=edges)
        na = na / na.sum() * 100
        nb, xb = np.histogram(DF.weightsB, bins=edges)
        nb = nb / nb.sum() * 100

        ax[aa].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
        ax[aa].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
        ax[aa].legend(('Background', 'Foreground'), fontsize=6)
        ax[aa].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=10)
        ax[aa].set_title(f"{tt}", fontweight='bold', fontsize=12)
        ax[aa].set_xlabel("Weight", fontweight='bold', fontsize=10)
        ymin, ymax = ax[aa].get_ylim()

        if stat == 'mean':
            BG1, FG1 = np.mean(DF.weightsA), np.mean(DF.weightsB)
            BG1sem, FG1sem = stats.sem(DF.weightsA), stats.sem(DF.weightsB)
            ttest = stats.ttest_ind(DF.weightsA, DF.weightsB)
            ax[aa+4].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
            ax[aa+4].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')
            ax[aa+4].set_ylabel('Mean Weight', fontweight='bold', fontsize=10)
            ttests[f'{types[aa]}'] = ttest
            if ttest.pvalue < 0.001:
                title = 'p<0.001'
            else:
                title = f"{ttest.pvalue:.3f}"
            ax[aa + 4].set_title(title, fontsize=8)

        if stat == 'median':
            BGmed, FGmed = DF.weightsA.median(), DF.weightsB.median()
            ax[aa+4].bar("BG", BGmed, color='deepskyblue')
            ax[aa+4].bar("FG", FGmed, color='yellowgreen')
            ax[aa+4].set_ylabel('Median Weight', fontweight='bold', fontsize=10)

    f.suptitle(f"{area}", fontweight='bold', fontsize=12)

    return ttests



def histogram_summary_plot(df, threshold=0.05, area='A1'):
    '''Pretty niche plot that will plot BG+/FG+ histograms and compare BG and FG weights,
    then plot BG+/FG- histogram and BG-/FG+ histogram separate and then compare BG and FG
    again in a bar graph. I guess you could put any thresholded quadrants you want, but the
    default is the only that makes sense. Last figure on APAN/SFN poster.'''
    df = df.loc[df.area == area]

    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=[3, 2, 6])
    quad3, quad2, quad6 = quad.values()

    f = plt.figure(figsize=(15, 7))
    histA = plt.subplot2grid((7, 17), (0, 0), rowspan=5, colspan=3)
    meanA = plt.subplot2grid((7, 17), (0, 4), rowspan=5, colspan=2)
    histB = plt.subplot2grid((7, 17), (0, 8), rowspan=5, colspan=3, sharey=histA)
    histC = plt.subplot2grid((7, 17), (0, 11), rowspan=5, colspan=3, sharey=histA)
    meanB = plt.subplot2grid((7, 17), (0, 15), rowspan=5, colspan=2, sharey=meanA)
    ax = [histA, meanA, histB, histC, meanB]

    edges = np.arange(-1, 2, .05)
    na, xa = np.histogram(quad3.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(quad3.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[0].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
    ax[0].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
    ax[0].legend(('Background', 'Foreground'), fontsize=7)
    ax[0].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=12)
    ax[0].set_title(f"Respond to both\nBG and FG alone", fontweight='bold', fontsize=16)
    ax[0].set_xlabel("Weight", fontweight='bold', fontsize=12)
    ymin, ymax = ax[0].get_ylim()

    BG1, FG1 = np.mean(quad3.weightsA), np.mean(quad3.weightsB)
    BG1sem, FG1sem = stats.sem(quad3.weightsA), stats.sem(quad3.weightsB)
    ttest1 = stats.ttest_ind(quad3.weightsA, quad3.weightsB)
    ax[1].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
    ax[1].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')
    ax[1].set_ylabel('Weight', fontweight='bold', fontsize=12)
    ax[1].set_ylim(0, 0.79)
    if ttest1.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"p={ttest1.pvalue:.3f}"
    ax[1].set_title(title, fontsize=8)

    BG2, FG2 = np.mean(quad6.weightsA), np.mean(quad2.weightsB)
    BG2sem, FG2sem = stats.sem(quad6.weightsA), stats.sem(quad2.weightsB)
    ttest2 = stats.ttest_ind(quad6.weightsA, quad2.weightsB)
    ax[4].bar("BG", BG2, yerr=BG2sem, color='deepskyblue')
    ax[4].bar("FG", FG2, yerr=FG2sem, color='yellowgreen')
    ax[4].set_ylabel("Weight", fontweight='bold', fontsize=12)
    ax[4].set_ylim(0, 0.79)
    if ttest2.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"p={ttest2.pvalue:.3f}"
    ax[4].set_title(title, fontsize=8)

    na, xa = np.histogram(quad6.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(quad2.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[2].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
    ax[3].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
    ax[2].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=12)
    ax[2].set_title(f"Respond to BG\nalone only", fontweight='bold', fontsize=16)
    ax[3].set_title(f"Respond to FG\nalone only", fontweight='bold', fontsize=16)
    ax[2].set_xlabel("Weight", fontweight='bold', fontsize=12)
    ax[3].set_xlabel("Weight", fontweight='bold', fontsize=12)
    ax[2].set_ylim(ymin, ymax), ax[3].set_ylim(ymin, ymax)
    ax[3].set_yticks([])

    f.suptitle(f"{area}", fontweight='bold', fontsize=12)

    return ttest1, ttest2


