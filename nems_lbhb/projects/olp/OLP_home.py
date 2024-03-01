import nems0.db as nd
import re
import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
import nems_lbhb.projects.olp.OLP_Synthetic_plot as osyn
import nems_lbhb.projects.olp.OLP_Binaural_plot as obip
import nems_lbhb.projects.olp.OLP_plot_helpers as oph
import nems_lbhb.projects.olp.OLP_figures as ofig
import nems_lbhb.projects.olp.OLP_plot_helpers as opl
import nems_lbhb.projects.olp.OLP_poster as opo
import scipy.ndimage.filters as sf
from scipy import stats
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import glob
import nems0.epoch as ep
from nems0 import db
import re
import nems_lbhb.SPO_helpers as sp
from nems0.xform_helper import load_model_xform
from datetime import date
import joblib as jl



sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
fs = 100


# Load your different, updated dataframes
path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_quarter_segments.h5'  # All quarter segments in one df
path = '/auto/users/hamersky/olp_analysis/ARM_Dynamic_OLP_segment0-500.h5' # ARM hopefully
path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_segment0-500.h5' #Vinaural half models
path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_sound_stats.h5' #weight + corr
path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5' # Still needs the CLT053 4 units
path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_segment0-500.h5' # The half models, use this now
path = '/auto/users/hamersky/olp_analysis/a1_celldat1.h5'
path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_segment0-500_with_stats.h5' # The half models, use this now


weight_df = ofit.OLP_fit_weights(loadpath=path)
weight_df['batch'] = 340

# OR
#Batch 341 prediction fit big boy
weight_df = jl.load('/auto/users/hamersky/olp_analysis/2023-01-12_Batch341_0-500_FULL')

fr_thresh=0.03
filt = weight_df.loc[(weight_df.bg_FR_start >= fr_thresh) & (weight_df.fg_FR_start >= fr_thresh)
               & (weight_df.bg_FR_end >= fr_thresh) & (weight_df.fg_FR_end >= fr_thresh)]
filt = filt.loc[filt.synth_kind=='A']

weight_lim = [-1, 2]

suffs = ['', '_pred']
for ss in suffs:
    filt = filt.loc[((filt[f'weightsA{ss}'] >= weight_lim[0]) & (filt[f'weightsA{ss}'] <= weight_lim[1])) &
                            ((filt[f'weightsB{ss}'] >= weight_lim[0]) & (filt[f'weightsB{ss}'] <= weight_lim[1]))]

r_thresh = 0.6
filt = filt.loc[(filt[f'r_start'] >= r_thresh) & (filt[f'r_end'] >= r_thresh)]

a1, peg = filt.loc[filt.area=='A1'], filt.loc[filt.area=='PEG']
fig, ax = plt.subplots(2, 2, figsize=(12,12), sharey=True, sharex=True)
ax = np.ravel(ax)


X, Y = a1.weightsA, a1.weightsA_pred
reg = stats.linregress(X, Y)
ax[0].scatter(x=X, y=Y, s=3, label=f"n={len(a1.weightsA)}"
                                   f"slope: {reg.slope:.3f}\n"
                                         f"coef: {reg.rvalue:.3f}\n"
                                         f"p = {reg.pvalue:.3f}")
ax[0].set_ylabel('BG Weight Pred'), ax[0].set_xlabel('BG Weight')
ax[0].set_title('A1')
ax[0].legend()
ax[0].plot([0,1.2], [0,1.2], color='black')
X, Y = peg.weightsA, peg.weightsA_pred
reg = stats.linregress(X, Y)
ax[1].scatter(x=X, y=Y, s=3, label=f"n={len(a1.weightsA)}"
                                   f"slope: {reg.slope:.3f}\n"
                                         f"coef: {reg.rvalue:.3f}\n"
                                         f"p = {reg.pvalue:.3f}")
ax[1].set_ylabel('BG Weight Pred'), ax[1].set_xlabel('BG Weight')
ax[1].set_title('PEG')
ax[1].legend()
ax[1].plot([0,1.2], [0,1.2], color='black')
X, Y = a1.weightsB, a1.weightsB_pred
reg = stats.linregress(X, Y)
ax[2].scatter(x=X, y=Y, s=3, color='yellowgreen', label=f"n={len(a1.weightsA)}"
                                   f"slope: {reg.slope:.3f}\n"
                                         f"coef: {reg.rvalue:.3f}\n"
                                         f"p = {reg.pvalue:.3f}")
ax[2].set_ylabel('FG Weight Pred'), ax[2].set_xlabel('FG Weight')
ax[2].set_title('A1')
ax[2].legend()
ax[2].plot([0,1.2], [0,1.2], color='black')
X, Y = peg.weightsB, peg.weightsB_pred
reg = stats.linregress(X, Y)
ax[3].scatter(x=X, y=Y, s=3, color='yellowgreen', label=f"n={len(a1.weightsA)}"
                                   f"slope: {reg.slope:.3f}\n"
                                         f"coef: {reg.rvalue:.3f}\n"
                                         f"p = {reg.pvalue:.3f}")
ax[3].set_ylabel('FG Weight Pred'), ax[3].set_xlabel('FG Weight')
ax[3].set_title('PEG')
ax[3].legend()
ax[3].plot([0,1.2], [0,1.2], color='black')
fig.suptitle(f"not filtered by FR or r")


'CLT012a-052-1'

response_heatmaps_comparison(weight_df, site='CLT009a', bg='Bulldozer', fg='FightSqueak', cellid=None,
                                     batch=340, bin_kind='11', synth_kind='A', sigma=1, sort=True,
                             example=True, lin_sum=True, positive_only=False)

response_heatmaps_comparison(weight_df, site='CLT012a', bg='Bees', fg='Bugle', cellid='CLT012a-052-1',
                                     batch=340, bin_kind='11', synth_kind='A', sigma=1, sort=True,
                             example=True, lin_sum=True, positive_only=False)

response_heatmaps_comparison(weight_df, site='CLT008a', bg='Wind', fg='Geese', cellid='CLT008a-046-2',
                                     batch=340, bin_kind='11', synth_kind='A', sigma=1, sort=True,
                             example=True, lin_sum=True, positive_only=False)

def response_heatmaps_comparison(df, site, bg, fg, cellid=None, batch=340, bin_kind='11',
                                 synth_kind='N', sigma=None, example=False, sort=True, lin_sum=True, positive_only=False):
    '''Takes out the BG, FG, combo, diff psth heatmaps from the interactive plot and makes it it's own
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
















if 'FG_rel_gain_start' not in weight_df.columns:
    weight_df['FG_rel_gain_start'] = (weight_df.weightsB_start - weight_df.weightsA_start) / \
                                      (np.abs(weight_df.weightsB_start) + np.abs(weight_df.weightsA_start))
    weight_df['FG_rel_gain_end'] = (weight_df.weightsB_end - weight_df.weightsA_end) / \
                                    (np.abs(weight_df.weightsB_end) + np.abs(weight_df.weightsA_end))
    weight_df['BG_rel_gain_start'] = (weight_df.weightsA_start - weight_df.weightsB_start) / \
                                      (np.abs(weight_df.weightsA_start) + np.abs(weight_df.weightsB_start))
    weight_df['BG_rel_gain_end'] = (weight_df.weightsA_end - weight_df.weightsB_end) / \
                                    (np.abs(weight_df.weightsA_end) + np.abs(weight_df.weightsB_end))


weight_df = weight_df.loc[weight_df.synth_kind=='N']
ofig.plot_all_weight_comparisons(weight_df, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True)

def plot_weight_prediction_comparisons(df, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True, pred=False, weight_lim=[-1,2]):
    areas = list(df.area.unique())

    # This can be mushed into one liners using list comprehension and show_suffixes
    quad3 = df.loc[(df.bg_FR_start >= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                           & (df.bg_FR_end >= fr_thresh) & (df.fg_FR_end >= fr_thresh)]

    quad2 = df.loc[(np.abs(df.bg_FR_start) <= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                   & (np.abs(df.bg_FR_end) <= fr_thresh) & (df.fg_FR_end >= fr_thresh)]

    quad6 = df.loc[(df.bg_FR_start >= fr_thresh) & (np.abs(df.fg_FR_start) <= fr_thresh)
                   & (df.bg_FR_end >= fr_thresh) & (np.abs(df.fg_FR_end) <= fr_thresh)]

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

def plot_all_weight_comparisons(df, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True, pred=False):
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

    if pred == True:
        fig, ax = plt.subplots(2, 4, figsize=(13, 12), sharey=True)
        pred_suff = ['', '_pred']
    else:
        fig, ax = plt.subplots(1, 4, figsize=(13, 6), sharey=True)
        pred_suff = ['']
    ax = np.ravel(ax)

    colors = ['mediumorchid', 'darkorange', 'orangered', 'green']

    stat_list, filt_list = [], []
    for nm, pr in enumerate(pred_suff):
        nm = nm * 4
        for num, aa in enumerate(areas):
            area_df = quad3.loc[quad3.area==aa]
            if strict_r == True:
                filt = area_df.loc[(area_df[f'r_start{pr}'] >= r_thresh) & (area_df[f'r_end{pr}'] >= r_thresh)]
            else:
                filt = area_df.loc[area_df[f"r{ss}"] >= r_thresh]

            if summary == True:
                alph = 0.35
            else:
                alph = 1

            for ee, an in enumerate(list(filt.animal.unique())):
                animal_df = filt.loc[filt.animal == an]
                ax[num + nm].scatter(x=['BG_start', 'BG_end'],
                                y=[np.nanmean(animal_df[f'weightsA_start{pr}']), np.nanmean(animal_df[f'weightsA_end{pr}'])],
                                   label=f'{an} (n={len(animal_df)})', color=colors[ee], alpha=alph)#, marker=symbols[cnt])
                ax[num + nm].scatter(x=['FG_start', 'FG_end'],
                                y=[np.nanmean(animal_df[f'weightsB_start{pr}']), np.nanmean(animal_df[f'weightsB_end{pr}'])],
                                   color=colors[ee], alpha=alph) #, marker=symbols[cnt])
                ax[num + nm].errorbar(x=['BG_start', 'BG_end'],
                                y=[np.nanmean(animal_df[f'weightsA_start{pr}']), np.nanmean(animal_df[f'weightsA_end{pr}'])],
                               yerr=[stats.sem(animal_df[f'weightsA_start{pr}']), stats.sem(animal_df[f'weightsA_end{pr}'])], xerr=None,
                               color=colors[ee], alpha=alph)
                ax[num + nm].errorbar(x=['FG_start', 'FG_end'],
                                y=[np.nanmean(animal_df[f'weightsB_start{pr}']), np.nanmean(animal_df[f'weightsB_end{pr}'])],
                               yerr=[stats.sem(animal_df[f'weightsB_start{pr}']), stats.sem(animal_df[f'weightsB_end{pr}'])], xerr=None,
                               color=colors[ee], alpha=alph)

                ax[num + nm].legend(fontsize=8, loc='upper right')

            BGsBGe = stats.ttest_ind(filt[f'weightsA_start{pr}'], filt[f'weightsA_end{pr}'])
            FGsFGe = stats.ttest_ind(filt[f'weightsB_start{pr}'], filt[f'weightsB_end{pr}'])
            BGsFGs = stats.ttest_ind(filt[f'weightsA_start{pr}'], filt[f'weightsB_start{pr}'])
            BGeFGe = stats.ttest_ind(filt[f'weightsA_end{pr}'], filt[f'weightsB_end{pr}'])

            tts = {f"BGsBGe_{aa}{pr}": BGsBGe.pvalue, f"FGsFGe_{aa}{pr}": FGsFGe.pvalue,
                   f"BGsFGs_{aa}{pr}": BGsFGs.pvalue, f"BGeFGe_{aa}{pr}": BGeFGe.pvalue}
            print(tts)
            stat_list.append(tts), filt_list.append(filt)

            ax[0].set_ylabel(f'Mean Weight{pr}', fontsize=14, fontweight='bold')
            ax[num + nm].set_title(f'{aa} - Respond to both\n BG and FG alone', fontsize=14, fontweight='bold')
            ax[num + nm].tick_params(axis='both', which='major', labelsize=10)
            ax[num + nm].set_xticklabels(['0-0.5s\nBG', '0.5-1s\nBG', '0-0.5s\nFG', '0.5-1s\nFG'], fontsize=12, fontweight='bold')

            if summary == True:
                ax[num + nm].scatter(x=['BG_start', 'BG_end'],
                                y=[np.nanmean(filt[f'weightsA_start{pr}']), np.nanmean(filt[f'weightsA_end{pr}'])],
                                label=f'Total (n={len(filt)})', color='black')  # , marker=symbols[cnt])
                ax[num + nm].scatter(x=['FG_start', 'FG_end'],
                                y=[np.nanmean(filt[f'weightsB_start{pr}']), np.nanmean(filt[f'weightsB_end{pr}'])],
                                color='black')  # , marker=symbols[cnt])
                ax[num + nm].errorbar(x=['BG_start', 'BG_end'],
                                 y=[np.nanmean(filt[f'weightsA_start{pr}']), np.nanmean(filt[f'weightsA_end{pr}'])],
                                 yerr=[stats.sem(filt[f'weightsA_start{pr}']), stats.sem(filt[f'weightsA_end{pr}'])],
                                 xerr=None, color='black')
                ax[num + nm].errorbar(x=['FG_start', 'FG_end'],
                                 y=[np.nanmean(filt[f'weightsB_start{pr}']), np.nanmean(filt[f'weightsB_end{pr}'])],
                                 yerr=[stats.sem(filt[f'weightsB_start{pr}']), stats.sem(filt[f'weightsB_end{pr}'])],
                                 xerr=None, color='black')

                ax[num + nm].legend(fontsize=8, loc='upper right')

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

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
axes[0].scatter(weight_a1.weightsA, weight_a1.weightsB, s=5, color='black')
axes[0].set_title(f'A1 - (n={len(weight_a1)})', fontweight='bold', fontsize=10)
axes[0].set_ylabel('FG Weights', fontweight='bold', fontsize=10)
axes[0].set_xlabel('BG Weights', fontweight='bold', fontsize=10)

axes[1].scatter(weight_peg.weightsA, weight_peg.weightsB, s=5, color='black')
axes[1].set_title(f'PEG - (n={len(weight_peg)})', fontweight='bold', fontsize=10)
axes[1].set_ylabel('FG Weights', fontweight='bold', fontsize=10)
axes[1].set_xlabel('BG Weights', fontweight='bold', fontsize=10)


ofig.weights_supp_comp(weight_df, quads=3, thresh=0.03, r_cut=0.6)
ofig.resp_weight_multi_scatter(weight_df, synth_kind='A', threshold=0.03)





path = '/auto/users/hamersky/olp_analysis/ARM_Dynamic_Errors.h5'
path = '/auto/users/hamersky/olp_analysis/ARM_Dynamic_Errors_EpochError.h5'
df = ofit.OLP_fit_weights(loadpath=path)


filt = 'ARM'

cell_df = nd.get_batch_cells(batch)
cell_list = cell_df['cellid'].tolist()
if isinstance(filt, str):
    cell_list = [cc for cc in cell_list if filt in cc]

if len(cell_list) == 0:
    raise ValueError(f"You did something wrong with your filter, there are no cells left.")


# Gets some cell metrics
metrics = []
for cellid in cell_list:
    cell_metric = calc_psth_metrics(batch, cellid)
    cell_metric.insert(loc=0, column='cellid', value=cellid)
    print(f"Adding cellid {cellid}.")
    metrics.append(cell_metric)
df = pd.concat(metrics)
df.reset_index()



thresh = 0.03
dyn_kind = 'fh'
areas = df.area.unique().tolist()

fig, axes = plt.subplots(2, 1, figsize=(10,6))

for cnt, ar in enumerate(areas):
    dyn_df = df.loc[df.dyn_kind==dyn_kind]
    area_df = dyn_df.loc[dyn_df.area==ar]

    # quad3 = area_df.loc[(area_df.bg_FR>=thresh) & (area_df.fg_FR>=thresh)]
    quad3 = area_df.loc[area_df.fg_FR>=thresh]

    E_full = np.array(quad3.E_full.to_list())[:, 50:-50]
    E_alone = np.array(quad3.E_alone.to_list())[:, 50:-50]

    full_av = np.nanmean(E_full, axis=0)
    alone_av = np.nanmean(E_alone, axis=0)

    baseline = np.nanmean(alone_av[:int(alone_av.shape[0]/2)])

    se_full = E_full.std(axis=0) / np.sqrt(E_full.shape[0])
    se_alone = E_alone.std(axis=0) / np.sqrt(E_alone.shape[0])


    if dyn_kind == 'fh':
        alone_col = 'deepskyblue'
    elif dyn_kind == 'hf':
        alone_col = 'yellowgreen'

    time = (np.arange(0, full_av.shape[0]) / 100)
    axes[cnt].plot(time, full_av, label='Full Error', color='black')
    axes[cnt].plot(time, alone_av, label='Alone Error', color=alone_col)

    axes[cnt].fill_between(time, (full_av - se_full*2), (full_av + se_full*2),
                         alpha=0.4, color='black')
    axes[cnt].fill_between(time, (alone_av - se_alone*2), (alone_av + se_alone*2),
                         alpha=0.4, color=alone_col)

    axes[cnt].legend()
    axes[cnt].set_title(f"{ar} - {dyn_kind} - n={len(quad3)}", fontweight='bold', fontsize=10)
    axes[cnt].set_xticks(np.arange(0,1,0.1))
    ymin, ymax = axes[cnt].get_ylim()
    axes[cnt].vlines([0.5], ymin, ymax, colors='black', linestyles=':')
    axes[cnt].hlines([baseline], time[0], time[-1], colors='black', linestyles='--', lw=0.5)
axes[1].set_xlabel("Time (s)", fontweight='bold', fontsize=10)


def calc_psth_metrics(batch, cellid, parmfile=None, paths=None):

    if parmfile:
        manager = BAPHYExperiment(parmfile)
    else:
        manager = BAPHYExperiment(cellid=cellid, batch=batch)

    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    area = db.pd_query(f"SELECT DISTINCT area FROM sCellFile where cellid like '{manager.siteid}%%'").area.iloc[0]

    if rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT'].shape[0] >= 2:
        # rec = ohel.remove_olp_test(rec)
        rec = ohel.remove_olp_test(rec)

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100

    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    params = ohel.get_expt_params(resp, manager, cellid)

    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2 = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    params['prestim'], params['poststim'] = epcs.iloc[0]['end'], ep2['end'] - ep2['start']
    params['lenstim'] = ep2['end']

    stim_epochs = ep.epoch_names_matching(resp.epochs, 'STIM_')

    if paths and cellid[:3] == 'TBR':
        print(f"Deprecated, run on {cellid} though...")
        stim_epochs, rec, resp = ohel.path_tabor_get_epochs(stim_epochs, rec, resp, params)

    epoch_repetitions = [resp.count_epoch(cc) for cc in stim_epochs]
    full_resp = np.empty((max(epoch_repetitions), len(stim_epochs),
                          (int(params['lenstim']) * rec['resp'].fs)))
    full_resp[:] = np.nan
    for cnt, epo in enumerate(stim_epochs):
        resps_list = resp.extract_epoch(epo)
        full_resp[:resps_list.shape[0], cnt, :] = resps_list[:, 0, :]

    #Grab and label epochs that have two sounds in them (no null)
    presil, postsil = int(params['prestim'] * rec['resp'].fs), int(params['poststim'] * rec['resp'].fs)
    twostims = resp.epochs[resp.epochs['name'].str.count('-0-1') == 2].copy()
    halfstims = resp.epochs[resp.epochs['name'].str.count(f"-{params['SilenceOnset']}-1") == 1].copy()
    halfstims = halfstims.loc[~halfstims['name'].str.contains('null')]

    ep_twostim = twostims.name.unique().tolist()
    ep_twostim.sort()
    ep_halfstim = halfstims.name.unique().tolist()
    ep_halfstim.sort()

    ep_names = resp.epochs[resp.epochs['name'].str.contains('STIM_')].copy()
    ep_names = ep_names.name.unique().tolist()
    ep_types = list(map(ohel.label_ep_type, ep_names))
    ep_synth_type = list(map(ohel.label_synth_type, ep_names))
    ep_dyn_type = list(map(ohel.label_dynamic_ep_type, ep_names))

    ep_df = pd.DataFrame({'name': ep_names, 'type': ep_types, 'synth_type': ep_synth_type, 'dyn_type': ep_dyn_type})

    cell_df = []
    for cnt, stimmy in enumerate(ep_halfstim):
        kind = ohel.label_ep_type(stimmy)
        synth_kind = ohel.label_synth_type(stimmy)
        dyn_kind = ohel.label_dynamic_ep_type(stimmy)
        # seps = (stimmy.split('_')[1], stimmy.split('_')[2])
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", stimmy)[0])
        BG, FG = seps[0].split('-')[0][2:], seps[1].split('-')[0][2:]
        BG_ep, FG_ep = f"STIM_{seps[0]}_null", f"STIM_null_{seps[1]}"

        if dyn_kind == 'fh':
            suffix = '-' + '-'.join(seps[0].split('-')[1:])
            alone = f'STIM_{seps[0]}_null'
            full = f"STIM_{seps[0]}_{seps[1].split('-')[0]}{suffix}"
        elif dyn_kind == 'hf':
            suffix = '-' + '-'.join(seps[1].split('-')[1:])
            alone = f'STIM_null_{seps[1]}'
            full = f"STIM_{seps[0].split('-')[0]}{suffix}_{seps[1]}"

        rhalf = resp.extract_epoch(stimmy)
        ralone, rfull = resp.extract_epoch(alone), resp.extract_epoch(full)
        rA, rB = resp.extract_epoch(BG_ep), resp.extract_epoch(FG_ep)

        # fn = lambda x: (np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR))
        # ralone_sm = np.squeeze(np.apply_along_axis(fn, 2, ralone))
        # rhalf_sm = np.squeeze(np.apply_along_axis(fn, 2, rhalf))
        # rfull_sm = np.squeeze(np.apply_along_axis(fn, 2, rfull))
        # rAsm = np.squeeze(np.apply_along_axis(fn, 2, rA))
        # rBsm = np.squeeze(np.apply_along_axis(fn, 2, rB))

        rA_st, rB_st = np.squeeze(np.nanmean(rA[:, :, presil:-postsil], axis=0)) - SR, \
                       np.squeeze(np.nanmean(rB[:, :, presil:-postsil], axis=0)) - SR
        A_FR, B_FR = np.nanmean(rA_st), np.nanmean(rB_st)

        # Get the average of repetitions and cut out just the stimulus
        ralone_st = np.squeeze(np.nanmean(ralone[:, :, presil:-postsil], axis=0))
        rhalf_st = np.squeeze(np.nanmean(rhalf[:, :, presil:-postsil], axis=0))
        rfull_st = np.squeeze(np.nanmean(rfull[:, :, presil:-postsil], axis=0))

        # Get correlations
        alonecorhalf = np.corrcoef(ralone_st, rhalf_st)[0, 1]  # Corr between resp to A and resp to dual
        fullcorhalf = np.corrcoef(rfull_st, rhalf_st)[0, 1]  # Corr between resp to B and resp to dual

        # FR
        alone_FR, half_FR, full_FR = np.nanmean(ralone_st), np.nanmean(rhalf_st), np.nanmean(rfull_st)

        std = np.std(np.concatenate([ralone_st, rhalf_st, rfull_st], axis=0))

        E_full = (np.abs(rfull_st - rhalf_st) - SR) / std
        E_alone = (np.abs(ralone_st - rhalf_st) - SR) / std

        # time = (np.arange(0, ralone.shape[-1]) / fs) - 0.5
        #
        # fig, ax = plt.subplots(2, 1, figsize=(10,8))
        #
        # ax[0].plot(time[presil:-postsil], ralone_st - SR, label='Alone')
        # ax[0].plot(time[presil:-postsil], rhalf_st - SR, label='Half')
        # ax[0].plot(time[presil:-postsil], rfull_st - SR, label='Full')
        # ax[0].legend()
        #
        # ax[1].plot(time[presil:-postsil], E_full, label='Full')
        # ax[1].plot(time[presil:-postsil], E_alone, label='Alone')
        # ax[1].legend()

        cell_df.append({'epoch': stimmy,
                        'kind': kind,
                        'synth_kind': synth_kind,
                        'dyn_kind': dyn_kind,
                        'BG': BG,
                        'FG': FG,
                        'fullcorhalf': fullcorhalf,
                        'alonecorhalf': alonecorhalf,
                        'bg_FR': A_FR,
                        'fg_FR': B_FR,
                        'half_FR': half_FR,
                        'full_FR': full_FR,
                        'E_alone': E_alone,
                        'E_full': E_full})

    cell_df = pd.DataFrame(cell_df)
    cell_df['SR'], cell_df['STD'] = SR, STD
    # cell_df['corcoef'], cell_df['avg_resp'], cell_df['snr'] = corcoef, avg_resp, snr
    cell_df.insert(loc=0, column='area', value=area)

    return cell_df












# Get the dataframe of sound statistics
sound_df = ohel.get_sound_statistics_full(weight_df, cuts=[0,0.5])




weight_df = ohel.add_sound_stats(weight_df, sound_df)



ofig.sound_metric_scatter(weight_df0, ['Fstationary', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Frequency\nNon-stationariness', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                          area='PEG', threshold=0.03, synth_kind='N', r_cut=0.6)





# # Tells you what sounds are below certain thresholds. And plots the sounds with their metrics.
# bad_dict = ohel.plot_sound_stats(sound_df, ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power', 'rel_gain'],
#                                  labels=['Frequency Non-stationarity', 'Temporal Non-stationarity', 'Bandwidth (octaves)',
#                                          'Max Power', 'RMS Power', 'Relative Gain'],
#                                  lines={'RMS_power': 0.95, 'max_power': 0.3}, synth_kind='N')
# bads = list(bad_dict['RMS_power'])
# bads = ['Waves', 'CashRegister', 'Heels', 'Keys', 'Woodblock', 'Castinets', 'Dice']  # Max Power
# Just gets us around running that above function, this is the output.
bads = ['CashRegister', 'Heels', 'Castinets', 'Dice']  # RMS Power Woodblock
weight_df = weight_df.loc[weight_df['BG'].apply(lambda x: x not in bads)]
weight_df = weight_df.loc[weight_df['FG'].apply(lambda x: x not in bads)]


# A nice function I made that filters all the things I usually try to filter, at once.
weight_df0 = ohel.filter_weight_df(weight_df, suffixes=[''], fr_thresh=0.03, r_thresh=0.6, quad_return=3,
                                   bin_kind='11', synth_kind=None, area='A1', weight_lims=[-1, 2],
                                   bads=True, bad_filt={'RMS_power': 0.95, 'max_power': 0.3})

weight_df0 = ohel.filter_synth_df_by(weight_df, use='N', suffixes=[''], fr_thresh=0.03, \
                                r_thresh=0.6, quad_return=3, bin_kind='11', weight_lims=[-1.5, 2.5], area='PEG')

weight_df0 = ohel.filter_synth_df_by(weight_df, use='C', suffixes=['', '_start', '_end'], fr_thresh=0.03, \
                                r_thresh=0.6, quad_return=3, bin_kind='11', weight_lims=[-1.5, 2.5], area='A1')

stat = osyn.synthetic_summary_weight_multi_bar(weight_df0, suffixes=['', '_start', '_end'],
                                               show=['N','M','S','T','C'], figsize=(12, 4))



weight_df0 = ohel.filter_weight_df(weight_df, suffixes=[''], fr_thresh=0.03, r_thresh=0.6, quad_return=3,
                                   bin_kind='11', synth_kind=None, area='A1', weight_lims=[-1, 2],
                                   bads=True, bad_filt={'RMS_power': 0.95, 'max_power': 0.3})
# Plots summary figure of the FG relative gain changes with synthetic condition and gives stats
ttests = osyn.synthetic_summary_relative_gain_bar(weight_df0)

ttests_a1 = synthetic_summary_relative_gain_bar(weight_df0_A1)
ttests_peg = synthetic_summary_relative_gain_bar(weight_df0_PEG)
ttests_a1_start = synthetic_summary_relative_gain_bar(weight_df0_A1_start)
ttests_a1_end = synthetic_summary_relative_gain_bar(weight_df0_A1_end)


a1_df0 = weight_df0.loc[weight_df0.area=='A1']
peg_df0 = weight_df0.loc[weight_df0.area=='PEG']
a1t = osyn.synthetic_summary_relative_gain_multi_bar(a1_df0, suffixes=['', '_start', '_end'])
pegt = osyn.synthetic_summary_relative_gain_multi_bar(peg_df0, suffixes=['', '_start', '_end'])



ttest = osyn.synthetic_summary_relative_gain_multi_bar(weight_df0, suffixes=['', '_start', '_end'])



osyn.synthetic_relative_gain_comparisons(weight_df, thresh=0.03, quads=3, area='A1',
                                              synth_show=['N', 'M','S','T','C'],
                                         r_cut=0.6, rel_cut=2.5)
osyn.synthetic_relative_gain_comparisons_specs(weight_df, 'Jackhammer', 'Fight Squeak', thresh=0.03,
                                               synth_show=['M', 'S', 'T', 'C'],
                                               quads=3, r_cut=0.6, rel_cut=2.5, area='A1', figsize=(20,15))


# Takes a spectrogram and makes side panels describing some metrics you can get from it
ofig.spectrogram_stats_diagram('Jackhammer', 'BG')
ofig.spectrogram_stats_diagram('Fight Squeak', 'FG')

# Compares the sound stats of the first and second half of the sound
ofig.sound_stats_half_compare(sound_df, suffixes=['_start', '_end'], metric='Tstationary', show='N')
ofig.sound_stats_half_compare(sound_df, suffixes=['_start', '_end'], metric='Fstationary', show='N')
ofig.sound_stats_half_compare(sound_df, suffixes=['_start', '_end'], metric='bandwidth', show='N')






# ##To make composite figure of animals. Use this to add additional animals to all_df
# animal = 'ARM'
# columns = ['cellid', 'area', 'epoch', 'animal', 'synth_kind', 'BG', 'FG', 'bg_FR', 'fg_FR', 'combo_FR', 'weightsA', 'weightsB', 'r',
#            'bg_FR_start', 'fg_FR_start', 'combo_FR_start', 'weightsA_start', 'weightsB_start', 'r_start',
#            'bg_FR_end', 'fg_FR_end', 'combo_FR_end', 'weightsA_end', 'weightsB_end', 'r_end',
#            'bg_FR_nopost', 'fg_FR_nopost', 'combo_FR_nopost', 'weightsA_nopost', 'weightsB_nopost', 'r_nopost']
# weight_dfs = [ARM, CLT1, CLT2]
# weight_dfs_cols = [wdf.filter(columns) for wdf in weight_dfs]
#
# all_df = pd.concat(weight_dfs_cols, axis=0)
# all_df = all_df.loc[(all_df.synth_kind == 'N') | (all_df.synth_kind == 'A')]
# Uses df of multiple animal weight fits and plots for the differnt fit epochs, make sure you load first.
path = '/auto/users/hamersky/olp_analysis/all_animals_OLP_segment0-500.h5' # Combination of only certain columns of all animals
all_df = ofit.OLP_fit_weights(loadpath=path)
# Plots the connected scatters showing how the different epoch weights relate across different animals
ofig.plot_all_weight_comparisons(all_df, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True)
# Plots a simple bar graph for quick comparison of how different fits affect overall weights in an individual or across animals
ofig.plot_partial_fit_bar(all_df, fr_thresh=0.03, r_thresh=0.6, suffixes=['_nopost', '_start', '_end'],
                          syn='A', bin='11', animal=None)

oph.generate_interactive_plot(all_df, xcolumn='bg_FR', ycolumn='fg_FR', threshold=0.03)

counts = ofig.all_animal_scatter(all_df, fr_thresh=0.03, r_thresh=0.6)




### For the figure if only I could find a good example 2022_11_01
ofig.psths_with_specs_partial_fit(weight_df, 'CLT047c-012-1', 'Bees', 'Gobble', sigma=1, error=False)
ofig.psths_with_specs_partial_fit(weight_df, 'CLT040c-051-1', 'Tuning', 'ManA', sigma=1, error=False)
ofig.psths_with_specs_partial_fit(weight_df, 'CLT052d-035-2', 'Bees', 'Chickens', sigma=1, error=False)
ofig.psths_with_specs_partial_fit(weight_df, 'CLT052d-018-1', 'Wind', 'Geese', sigma=1, error=False)


## 2022_10_24 plotting FGs so I can try and decide the envelope thing
ofig.display_sound_envelopes(sound_df, type='FG', envs=True)
ofig.plot_some_sound_stats(sound_df)



#Example PSTH for presentations
ofig.plot_PSTH_example_progression(333, cellid='TBR012a-31-1', bg='Wind', fg='Chirp', bin_kind='11', synth_kind='A',
                                   sigma=1, error=False, specs=True)


##########################
# Viewing synthetic metrics and weights, to figure out stats stuff
names = osyn.checkout_mods(11, weight_df, thresh=0.03, quads=3, r_cut=0.75)
names = osyn.checkout_mods_tidier(9, weight_df, show=['N','M','U','S','T','C'], thresh=0.03, quads=3, r_cut=0.7, area='A1')
names = osyn.checkout_mods_cleaner(23, weight_df, r_cut=0.75, area='A1')

## Stuff with synthetics viewing.
osyn.rel_gain_synth_scatter(weight_df, show=['N','M','S','T','C'],
                            thresh=0.03, quads=3, r_cut=0.8, area='A1')
osyn.rel_gain_synth_scatter_single(weight_df, show=['N','M','S','T','C'], thresh=0.03,
                              quads=3, r_cut=0.8, area='A1')
osyn.synthetic_relative_gain_comparisons(weight_df, thresh=0.03, quads=3, area='A1',
                                              synth_show=['N','M','S','T','C'],
                                         r_cut=0.7, rel_cut=2.5)
osyn.synthetic_relative_gain_comparisons_specs(weight_df, 'Jackhammer', 'Fight Squeak', thresh=0.03,
                                               synth_show=['N', 'M', 'S', 'T', 'C'],
                                               quads=3, r_cut=0.6, rel_cut=2.5, area='A1')

# Plots synthetic metrics, good for viewing like I want to do above
osyn.synth_scatter_metrics(weight_df, first='temp_ps_std', second='freq_ps_std', metric='rel_gain',
                      show=['N', 'M', 'S', 'T', 'C'], thresh=0.03, quads=3, r_cut=0.6, ref='N', area='A1')

osyn.synth_scatter_metrics(weight_df, first='temp_ps_std', second='freq_ps_std', metric='rel_gain',
                      show=['N', 'M', 'S', 'T', 'C'], thresh=0.03, quads=3, r_cut=0.8, ref=None, area='A1')

# Plot simple comparison of sites and synethetics in bar plot
osyn.plot_synthetic_weights(weight_df, plotA='weightsA', plotB='weightsB', thresh=0.04, areas=None,
                            synth_show=None, r_cut=0.75, title='Title')

# Plots all of the synthetic spectrogram features against a common condition for the stats given
osyn.sound_stats_comp_scatter(sound_df, ['Fstationary', 'Tstationary', 'bandwidth'],
                              main='N', comp=['M', 'U', 'S', 'T', 'C'], label=False)

# Plots all of the synthetic combinations with their respective sound statistics - big boy
osyn.sound_metric_scatter_all_synth(weight_df, x_metrics=['Fstationary', 'Tstationary', 'bandwidth', 'power'],
                          y_metric='BG_rel_gain', x_labels=['Frequency\nNon-Stationarity', 'Temporal\nNon-Stationarity',
                                          'Bandwidth', 'Power'],
                                    jitter=[0.25, 0.2, 0.03, 0.01, 0.003],
                                    area='A1', threshold=0.03,
                                    synth_show=['A', 'N', 'M', 'U', 'S', 'T', 'C'],
                                    title_text='All', r_cut=0.9)
# Without power stuff
osyn.sound_metric_scatter_all_synth(weight_df, x_metrics=['Fstationary', 'Tstationary', 'bandwidth'],
                                    y_metric='BG_rel_gain', x_labels=['Frequency\nNon-Stationarity',
                                    'Temporal\nNon-Stationarity', 'Bandwidth'],
                                    jitter=[0.25, 0.2, 0.03], area='A1', threshold=0.03,
                                    synth_show=['A', 'N', 'M', 'U', 'S', 'T', 'C'],
                                    title_text='Minus RMS bads', r_cut=0.75)



# Number 2 on the list of things to do
ofig.weights_supp_comp(weight_df, quads=3, thresh=0.03, r_cut=0.8)

# I use this for most things
quad, _ = ohel.quadrants_by_FR(weight_df, threshold=0.03, quad_return=3)

# For scatter of sound features to rel gain
ofig.sound_metric_scatter(weight_df, ['Fstationary', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Frequency\nNon-stationariness', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                          area='A1', threshold=0.03, synth_kind='N')

poster5_sound_metric_scatter(weight_df, ['Fstationary', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Frequency\nNon-stationariness', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                          area='A1', threshold=0.03, synth_kind='N', jitter=[0.2,0.03,0.25],
                         quad_return=3, metric_filter=None, bin_kind='11')

#interactive plot
oph.generate_interactive_plot(weight_df)

# Get scatters of FRs/weights
ofig.resp_weight_multi_scatter(weight_df, synth_kind='N', threshold=0.03)
ofig.resp_weight_multi_scatter(weight_df, ycol=['BG_rel_gain', 'BG_rel_gain', 'FG_rel_gain', 'FG_rel_gain'],
                               synth_kind='N', threshold=0.03)

# Some model accuracy figures to confirm goodness of model doesn't get rid of FG suppression
ofig.scatter_model_accuracy(weight_df, stat='FG_rel_gain', synth_kind='N', threshold=0.03)
ofig.r_filtered_weight_histogram_summary(weight_df, synth_kind='C', manual=0.85)

# Adds max_power, must use A
ofig.sound_metric_scatter(weight_df, ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power'],
                          'BG_rel_gain', ['Frequency\nNon-Stationarity', 'Temporal\nNon-Stationarity',
                                          'Bandwidth', 'Max Power', 'RMS Power'], jitter=[0.25, 0.2, 0.03, 0.03, 0.003],
                          area='A1', threshold=0.03, synth_kind='N', title_text='Removed Low')

# Not a great one but uses mod spec stats
ofig.sound_metric_scatter(weight_df, ['t50', 'f50'],
                          'BG_rel_gain', ['wt (Hz)', 'wf (cycles/s)'], jitter=[0.075, 0.0075],
                          area='A1', threshold=0.03, synth_kind='N', title_text='')

# Plots a single, example relative gain histogram
ofig.plot_single_relative_gain_hist(weight_df, 0.03, synth_kind='N')

# Plots the example that I piece together to make the linear model example
weight_df = ofit.OLP_fit_weights(batch=333, cells=['TBR012a-31-1'])
ofig.plot_linear_model_pieces_helper(weight_df, cellid='TBR012a-31-1', bg='Wind', fg='Chirp')

# Adds max_power
ofig.sound_metric_scatter(weight_df, ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power'],
                          'BG_rel_gain', ['Frequency\nNon-Stationarity', 'Temporal\nNon-Stationarity',
                                          'Bandwidth', 'Max Power', 'RMS Power'],
                          jitter=[0.25, 0.2, 0.03, 0.03, 0.0003],
                          area='A1', threshold=0.03, synth_kind='N',
                          title_text='removed low max power FGs')

## Testing something to make sure everything played right to the ferret
## need to generate weight_dfs based on the names, binaural 11 and 22, synthetic A
ofig.speaker_test_plot(weight_df_11, weight_df_22, weight_df_synth, threshs=[0.03, 0.02, 0.01])


# Add enhancement for an interactive plot that looks at the change of enchancement from start to end fit
weight_df0['FG_enhancement_start'] = weight_df0['weightsB_start'] - weight_df0['weightsA_start']
weight_df0['FG_enhancement_end'] = weight_df0['weightsB_end'] - weight_df0['weightsA_end']
oph.generate_interactive_plot(weight_df0, xcolumn='FG_enhancement_start', ycolumn='FG_enhancement_end', threshold=0.03)
oph.generate_interactive_plot(weight_df0, xcolumn='bg_FR', ycolumn='fg_FR', threshold=0.03)



# batch = 328 #Ferret A1
# batch = 329 #Ferret PEG
# batch = 333 #Marmoset (HOD+TBR)
# batch = 340 #All ferret OLP

# # Add new filenames as you need to add things
# filename = '_'
# storepath = f'/auto/users/hamersky/olp_analysis/{filename}.h5'

# To fit whole batch and save it
# weight_df = ofit.OLP_fit_weights(batch, savepath=storepath, filter=None)
# To fit only a specific parmfile and save it
# weight_df = ofit.OLP_fit_weights(batch, parmfile=parmfile, savepath=storepath, filter=None)
# Alternate to parmfile loading is use keyword to get the number experiment you want
# weight_df = ofit.OLP_fit_weights(batch, savepath=storepath, filter='CLT022')
# To filter by CLT Synthetic only, use a long list of experiment names
# synths = [f'CLT0{cc}' for cc in range(27,54)]
# weight_df = ofit.OLP_fit_weights(batch, savepath=storepath, filter=synths)

# # This is how you update an old dataframe from before 2022_09 to have all the useful statistics of present
# if 'synth_kind' not in weight_df:
#     weight_df['synth_kind'] = 'A'
# if 'kind' not in weight_df:
#     weight_df['kind'] = '11'
# weight_df['BG_rel_gain'] = (weight_df.weightsA - weight_df.weightsB) / \
#                            (np.abs(weight_df.weightsB) + np.abs(weight_df.weightsA))
# weight_df['FG_rel_gain'] = (weight_df.weightsB - weight_df.weightsA) / \
#                            (np.abs(weight_df.weightsB) + np.abs(weight_df.weightsA))

## Update weight_df to include mod statistics
# sound_df = ohel.get_sound_statistics_full(weight_df)
weight_df = weight_df.drop(labels=['BG_Tstationary_y', 'BG_bandwidth_y', 'BG_Fstationary_y', \
                       'FG_Tstationary_y', 'FG_bandwidth_y', 'FG_Fstationary_y', 'BG_RMS_power_y',
                                   'BG_max_power_y', 'FG_RMS_power_y', 'FG_max_power_y', 'BG_f50_y',
                                   'BG_t50_y', 'FG_f50_y', 'FG_t50_y'], axis=1)
weight_df = weight_df.drop(labels=['BG_temp_ps_x', 'BG_temp_ps_std_x', 'BG_freq_ps_x',
       'BG_freq_ps_std_x', 'FG_temp_ps_x', 'FG_temp_ps_std_x', 'FG_freq_ps_x',
       'FG_freq_ps_std_x', 'FG_rel_gain_start', 'FG_rel_gain_end',
       'BG_temp_ps_y', 'BG_temp_ps_std_y', 'BG_freq_ps_y', 'BG_freq_ps_std_y',
       'FG_temp_ps_y', 'FG_temp_ps_std_y', 'FG_freq_ps_y', 'FG_freq_ps_std_y'], axis=1)
# weight_df = ohel.add_sound_stats(weight_df, sound_df)
#
# os.makedirs(os.path.dirname(savepath), exist_ok=True)
# store = pd.HDFStore(savepath)
# df_store = copy.deepcopy(weight_df)
# store['df'] = df_store.copy()
# store.close()











## Get to stuff


##############################
###### Clathrus Mapping ######
######                  ######
import pathlib as pl
from nems_lbhb.penetration_map import penetration_map

#%%

sites = ['CLT028a', 'CLT029a', 'CLT030d', 'CLT031c', 'CLT032c', 'CLT033c', 'CLT034c',
         'CLT035c', 'CLT036c', 'CLT037c', 'CLT038a', 'CLT039c', 'CLT040c', 'CLT041c',
         'CLT042a', 'CLT043b', 'CLT044d', 'CLT045c', 'CLT046c', 'CLT047c', 'CLT048c',
         'CLT049c', 'CLT050c', 'CLT051c', 'CLT052d', 'CLT053a']

# Original landmark measurements
# landmarks = {'viral0': [0.39, 5.29, 1.89, 0.70, 5.96, 1.12, 42, 0],
#              'viral1': [0.39, 5.29, 1.89, 0.67, 6.14, 1.15, 42, 0]}

# corrected to better align with corresponding penetrations
# landmarks = {'viral0': [0.39, 5.25, 1.37, 0.70, 5.96, 1.12, 42, 0],
#              'viral1': [0.39, 5.25, 1.37, 0.67, 6.14, 1.15, 42, 0]}

# fig, coords = penetration_map(sites, equal_aspect=True, flip_X=True, flatten=False, landmarks=landmarks)
fig, coords = penetration_map(sites, equal_aspect=True, flip_X=False,
                              flatten=True, flip_YZ=True,
                              # landmarks=landmarks
                              )
fig.axes[0].grid()
# saves the scatter
mappath = pl.Path('/auto/data/lbhb/photos/Craniotomies/Clatrus/CLT_RH_map.png')
fig.savefig(mappath, transparent=True)




# Regression stuff
def _get_suppression(response, params):
    supp_array = np.empty([len(params['good_units']), len(params['pairs'])])
    for nn, pp in enumerate(params['pairs']):
        _, _, _, _, supp, _, _ = get_scatter_resps(nn, response)
        supp_array[:, nn] = supp

    return supp_array

def site_regression(supp_array, params):
    site_results = pd.DataFrame()
    shuffles = [None, 'neuron', 'stimulus']
    for shuf in shuffles:
        reg_results = neur_stim_reg(supp_array, params, shuf)
        site_results = site_results.append(reg_results, ignore_index=True)

    return site_results

def neur_stim_reg(supp_array, params, shuffle=None):
    y = supp_array.reshape(1, -1)  # flatten
    stimulus = np.tile(np.arange(0, supp_array.shape[1]), supp_array.shape[0])
    neuron = np.concatenate([np.ones(supp_array.shape[1]) * i for i in
                             range(supp_array.shape[0])], axis=0)

    X = np.stack([neuron, stimulus])
    X = pd.DataFrame(data=X.T, columns=['neuron', 'stimulus'])
    X = sm.add_constant(X)
    X['suppression'] = y.T

    if not shuffle:
        results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=X).fit()

    if shuffle == 'neuron':
        Xshuff = X.copy()
        Xshuff['neuron'] = Xshuff['neuron'].iloc[np.random.choice(
            np.arange(X.shape[0]), X.shape[0], replace=False)].values
        results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    if shuffle == 'stimulus':
        Xshuff = X.copy()
        Xshuff['stimulus'] = Xshuff['stimulus'].iloc[np.random.choice(
            np.arange(X.shape[0]), X.shape[0], replace=False)].values
        results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    reg_results = _regression_results(results, shuffle, params)

    return reg_results

def _regression_results(results, shuffle, params):
    intercept = results.params.loc[results.params.index.str.contains('Intercept')].values
    int_err = results.bse.loc[results.bse.index.str.contains('Intercept')].values
    int_conf = results.conf_int().loc[results.conf_int().index.str.contains('Intercept')].values[0]
    neuron_coeffs = results.params.loc[results.params.index.str.contains('neuron')].values
    neuron_coeffs = np.concatenate(([0], neuron_coeffs))
    stim_coeffs = results.params.loc[results.params.index.str.contains('stimulus')].values
    stim_coeffs = np.concatenate(([0], stim_coeffs))
    neur_coeffs = neuron_coeffs + intercept + stim_coeffs.mean()
    stim_coeffs = stim_coeffs + intercept + neuron_coeffs.mean()
    coef_list = np.concatenate((neur_coeffs, stim_coeffs))

    neuron_err = results.bse.loc[results.bse.index.str.contains('neuron')].values
    stim_err = results.bse.loc[results.bse.index.str.contains('stimulus')].values
    neuron_err = np.concatenate((int_err, neuron_err))
    stim_err = np.concatenate((int_err, stim_err))
    err_list = np.concatenate((neuron_err, stim_err))

    neur_low_conf = results.conf_int()[0].loc[results.conf_int().index.str.contains('neuron')].values
    neur_low_conf = np.concatenate(([int_conf[0]], neur_low_conf))
    stim_low_conf = results.conf_int()[0].loc[results.conf_int().index.str.contains('stimulus')].values
    stim_low_conf = np.concatenate(([int_conf[0]], stim_low_conf))
    low_list = np.concatenate((neur_low_conf, stim_low_conf))

    neur_high_conf = results.conf_int()[1].loc[results.conf_int().index.str.contains('neuron')].values
    neur_high_conf = np.concatenate(([int_conf[1]], neur_high_conf))
    stim_high_conf = results.conf_int()[1].loc[results.conf_int().index.str.contains('stimulus')].values
    stim_high_conf = np.concatenate(([int_conf[1]], stim_high_conf))
    high_list = np.concatenate((neur_high_conf, stim_high_conf))

    neur_list = ['neuron'] * len(neur_coeffs)
    stim_list = ['stimulus'] * len(stim_coeffs)
    name_list = np.concatenate((neur_list, stim_list))

    if shuffle == None:
        shuffle = 'full'
    shuff_list = [f"{shuffle}"] * len(name_list)
    site_list = [f"{params['experiment']}"] * len(name_list)
    r_list = [f"{np.round(results.rsquared, 4)}"] * len(name_list)

    name_list_actual = list(params['good_units'])
    name_list_actual.extend(params['pairs'])

    reg_results = pd.DataFrame(
        {'name': name_list_actual,
         'id': name_list,
         'site': site_list,
         'shuffle': shuff_list,
         'coeff': coef_list,
         'error': err_list,
         'conf_low': low_list,
         'conf_high': high_list,
         'rsquare': r_list
         })

    return reg_results

def multisite_reg_results(parmfiles):
    regression_results = pd.DataFrame()
    for file in parmfiles:
        params = load_experiment_params(file, rasterfs=100, sub_spont=True)
        response = get_response(params, sub_spont=False)
        corcoefs = _base_reliability(response, rep_dim=2, protect_dim=3)
        avg_resp = _significant_resp(response, params, protect_dim=3, time_dim=-1)
        response = _find_good_units(response, params,
                                    corcoefs=corcoefs, corcoefs_threshold=0.1,
                                    avg_resp=avg_resp, avg_threshold=0.2)
        supp_array = _get_suppression(response, params)
        site_results = site_regression(supp_array, params)

        regression_results = regression_results.append(site_results, ignore_index=True)

    return regression_results


########
########
#######




## 2023_01_03. This goes after I run the job and have a df.

saved_paths = glob.glob(f"/auto/users/hamersky/cache/*")

weight_df0 = []
for path in saved_paths:
    df = jl.load(path)
    weight_df0.append(df)

weight_df0 = pd.concat(weight_df0)
ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df0.namesA, weight_df0.namesB)]
weight_df0 = weight_df0.drop(columns=['namesA', 'namesB'])
weight_df0['epoch'] = ep_names

from datetime import date
today = date.today()
OLP_partialweights_db_path = \
    f'/auto/users/hamersky/olp_analysis/{date.today()}_Batch341_{weight_df0.fit_segment.unique()[0]}_nometrics'  # weight + corr

jl.dump(weight_df0, OLP_partialweights_db_path)

# This as it is won't have enough memory, use enqueue, which is set up for this. But this output
# is the same output as the weight_df0 above.
weight_df0, cuts_info = ofit.OLP_fit_partial_weights(341, threshold=None, snip=[0, 0.5], pred=True,
                                                    fit_epos='syn', fs=100, filter_animal=None,
                                                    filter_experiment=None, note="Batch431_oldway")

# # Runs metrics on the cells present in the fit list.
# weight_df0 = jl.load('/auto/users/hamersky/olp_analysis/2023-01-03_Batch341_0-500_nometrics')
#
# cell_list = list(set(weight_df0.cellid))
#
# cuts_info = ohel.get_cut_info(weight_df0)
# batch = 341
# metrics = []
# for cellid in cell_list:
#     cell_metric = ofit.calc_psth_metrics_cuts(batch, cellid, cut_ids=cuts_info)
#     cell_metric.insert(loc=0, column='cellid', value=cellid)
#     print(f"Adding cellid {cellid}.")
#     metrics.append(cell_metric)
# df = pd.concat(metrics)
# df.reset_index()
#
# ## Run me Jereme! Saves Metrics
# from datetime import date
# today = date.today()
# OLP_partialweights_db_path = \
#     f'/auto/users/hamersky/olp_analysis/{date.today()}_Batch341_{weight_df0.fit_segment.unique()[0]}_metrics'  # weight + corr
#
# jl.dump(df, OLP_partialweights_db_path)

#
# # This loads the no metrics and metrics dataframes and merges them to save a new one
# weight_df0 = jl.load('/auto/users/hamersky/olp_analysis/2023-01-03_Batch341_0-500_nometrics')
# df = jl.load('/auto/users/hamersky/olp_analysis/2023-01-04_Batch341_0-500_metrics')
# weight_df = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
# from datetime import date
# today = date.today()
# OLP_partialweights_db_path = \
#     f'/auto/users/hamersky/olp_analysis/{date.today()}_Batch341_{weight_df0.fit_segment.unique()[0]}_FULL'
# jl.dump(weight_df, OLP_partialweights_db_path)

#This loads the big boy from all the above.
weight_df = jl.load('/auto/users/hamersky/olp_analysis/2023-01-12_Batch341_0-500_FULL')

## I THINK GARBAGE NOW 2023_01_12
# ### Adding to slap in a fix for the FR
#     # Gets some cell metrics
#     metrics = []
#     for cellid in cell_list:
#         cell_metric = ofit.calc_psth_metrics_cuts(batch, cellid, cut_ids=cuts_info)
#         cell_metric.insert(loc=0, column='cellid', value=cellid)
#         print(f"Adding cellid {cellid}.")
#         metrics.append(cell_metric)
#     df = pd.concat(metrics)
#     df.reset_index()
#
#     weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
#     weight_df0['fit_segment'] = f"{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}"
#
#     OLP_savepath = f'/auto/users/hamersky/olp_analysis/Batch341_test_{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}_metrics.h5'  # weight + corr
#     os.makedirs(os.path.dirname(OLP_savepath), exist_ok=True)
#     store = pd.HDFStore(OLP_savepath)
#     df_store = copy.deepcopy(weight_df0)
#     store['df'] = df_store.copy()
#     store.close()
#
#
#
# OLP_metrics_db_path = f'/auto/users/hamersky/olp_analysis/ARM_Dynamic_test{int(snip[0] * 1000)}-{int((snip[0]+snip[1]) * 1000)}metrics.h5'  # weight + corr
# os.makedirs(os.path.dirname(OLP_metrics_db_path), exist_ok=True)
# store = pd.HDFStore(OLP_metrics_db_path)
# df_store = copy.deepcopy(df)
# store['df'] = df_store.copy()
# store.close()
#
# weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
# weight_df0['fit_segment'] = f"{int(snip[0] * 1000)}-{int((snip[0]+snip[1]) * 1000)}"
#
#
#
# OLP_savepath = f'/auto/users/hamersky/olp_analysis/ARM_Dynamic_OLP_segment{int(snip[0] * 1000)}-{int((snip[0]+snip[1]) * 1000)}.h5'  # weight + corr
# os.makedirs(os.path.dirname(OLP_savepath), exist_ok=True)
# store = pd.HDFStore(OLP_savepath)
# df_store = copy.deepcopy(all_df)
# store['df'] = df_store.copy()
# store.close()
#
#
# # I think this is when you're combining other dfs you loaded with a new fit
# weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
# # weight_df0['threshold'] = str(int(threshold * 100))
# # if df.shape[0] != weights_df.shape[0] or weight_df.shape[0] != weights_df.shape[0]:
# #     raise ValueError("Resulting weights_df does not match length of parts, some epochs were dropped.")
#
#
# OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/test500-750metrics.h5'  # weight + corr
# os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)
# store = pd.HDFStore(OLP_partialweights_db_path)
# df_store = copy.deepcopy(weight_df0)
# store['df'] = df_store.copy()
# store.close()

##load here, 2022_10_24, these are from clathrus synthetic as I try to fit the partial model
OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_control_segment500-750_goodmetrics.h5'  # weight + corr

OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_full_partial_weights_nometrics.h5'  # weight + corr
OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_full_partial_weights_withmetrics.h5'  # weight + corr

part_weights = False
if part_weights == True:
    os.makedirs(os.path.dirname(OLP_partialweights_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_partialweights_db_path)
    df_store=copy.deepcopy(weight_df0)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_partialweights_db_path)
    df=store['df']
    store.close()

import re



def calc_psth_metrics_cuts(batch, cellid, parmfile=None, paths=None, cut_ids=None):
    start_win_offset = 0  # Time (in sec) to offset the start of the window used to calculate threshold, exitatory percentage, and inhibitory percentage
    if parmfile:
        manager = BAPHYExperiment(parmfile)
    else:
        manager = BAPHYExperiment(cellid=cellid, batch=batch)

    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    area_df = db.pd_query(f"SELECT DISTINCT area FROM sCellFile where cellid like '{manager.siteid}%%'")
    area = area_df.area.iloc[0]

    if rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT'].shape[0] >= 2:
        # rec = ohel.remove_olp_test(rec)
        rec = remove_olp_test(rec)

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100

    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    # params = ohel.get_expt_params(resp, manager, cellid)
    params = get_expt_params(resp, manager, cellid)


    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2 = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    params['prestim'], params['poststim'] = epcs.iloc[0]['end'], ep2['end'] - ep2['start']
    params['lenstim'] = ep2['end']

    stim_epochs = ep.epoch_names_matching(resp.epochs, 'STIM_')

    if paths and cellid[:3] == 'TBR':
        print(f"Deprecated, run on {cellid} though...")
        stim_epochs, rec, resp = ohel.path_tabor_get_epochs(stim_epochs, rec, resp, params)

    epoch_repetitions = [resp.count_epoch(cc) for cc in stim_epochs]
    full_resp = np.empty((max(epoch_repetitions), len(stim_epochs),
                          (int(params['lenstim']) * rec['resp'].fs)))
    full_resp[:] = np.nan
    for cnt, epo in enumerate(stim_epochs):
        resps_list = resp.extract_epoch(epo)
        full_resp[:resps_list.shape[0], cnt, :] = resps_list[:, 0, :]

    #Calculate a few metrics
    corcoef = ohel.calc_base_reliability(full_resp)
    avg_resp = ohel.calc_average_response(full_resp, params)
    snr = compute_snr(resp)

    #Grab and label epochs that have two sounds in them (no null)
    presil, postsil = int(params['prestim'] * rec['resp'].fs), int(params['poststim'] * rec['resp'].fs)
    twostims = resp.epochs[resp.epochs['name'].str.count('-1') == 2].copy()
    ep_twostim = twostims.name.unique().tolist()
    ep_twostim.sort()

    ep_names = resp.epochs[resp.epochs['name'].str.contains('STIM_')].copy()
    ep_names = ep_names.name.unique().tolist()
    ep_types = list(map(ohel.label_ep_type, ep_names))
    ep_synth_type = list(map(ohel.label_synth_type, ep_names))

    ep_df = pd.DataFrame({'name': ep_names, 'type': ep_types, 'synth_type': ep_synth_type})

    cell_dff = []
    for cnt, stimmy in enumerate(ep_twostim):
        kind = ohel.label_pair_type(stimmy)
        # synth_kind = ohel.label_synth_type(stimmy)
        synth_kind = label_synth_type(stimmy)
        dynamic_kind = label_dynamic_ep_type(stimmy)
        # seps = (stimmy.split('_')[1], stimmy.split('_')[2])
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", stimmy)[0])
        BG, FG = seps[0].split('-')[0][2:], seps[1].split('-')[0][2:]

        Aepo, Bepo = 'STIM_' + seps[0] + '_null', 'STIM_null_' + seps[1]

        rAB = resp.extract_epoch(stimmy)
        rA, rB = resp.extract_epoch(Aepo), resp.extract_epoch(Bepo)

        fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR)
        rAsm = np.squeeze(np.apply_along_axis(fn, 2, rA))
        rBsm = np.squeeze(np.apply_along_axis(fn, 2, rB))
        rABsm = np.squeeze(np.apply_along_axis(fn, 2, rAB))

        rA_st, rB_st = rAsm[:, presil:-postsil], rBsm[:, presil:-postsil]
        rAB_st = rABsm[:, presil:-postsil]

        rAm, rBm = np.nanmean(rAsm, axis=0), np.nanmean(rBsm, axis=0)
        rABm = np.nanmean(rABsm, axis=0)

        AcorAB = np.corrcoef(rAm, rABm)[0, 1]  # Corr between resp to A and resp to dual
        BcorAB = np.corrcoef(rBm, rABm)[0, 1]  # Corr between resp to B and resp to dual

        A_FR, B_FR, AB_FR = np.nanmean(rA_st), np.nanmean(rB_st), np.nanmean(rAB_st)

        min_rep = np.min((rA.shape[0], rB.shape[0]))  # only will do something if SoundRepeats==Yes
        lin_resp = np.nanmean(rAsm[:min_rep, :] + rBsm[:min_rep, :], axis=0)
        supp = np.nanmean(lin_resp - AB_FR)

        AcorLin = np.corrcoef(rAm, lin_resp)[0, 1]  # Corr between resp to A and resp to lin
        BcorLin = np.corrcoef(rBm, lin_resp)[0, 1]  # Corr between resp to B and resp to lin

        Apref, Bpref = AcorAB - AcorLin, BcorAB - BcorLin
        pref = Apref - Bpref

        # If there are no cuts provided, just make one that takes everything.
        if not cut_ids:
            cut_ids = {'': np.full((int(params['lenstim'] * params['fs']),), True)}

        # Start the dict that becomes the df with universal things regardless of if cuts or not
        cell_dict = {'epoch': stimmy,
                        'kind': kind,
                        'synth_kind': synth_kind,
                        'dynamic_kind': dynamic_kind,
                        'BG': BG,
                        'FG': FG,
                        'AcorAB': AcorAB,
                        'BcorAB': BcorAB,
                        'AcorLin': AcorLin,
                        'BcorLin': BcorLin,
                        'pref': pref,
                        'Apref': Apref,
                        'Bpref': Bpref
                        }

        for lb, cut in cut_ids.items():
            cut_st = cut[presil:-postsil]
            rA_st_cut, rB_st_cut, rAB_st_cut = rA_st[:, cut_st], rB_st[:, cut_st], rAB_st[:, cut_st]
            rAsm_cut, rBsm_cut, rABsm_cut = rAsm[:, cut], rBsm[:, cut], rABsm[:, cut]

            # AcorAB = np.corrcoef(rAm_cut, rABm_cut)[0, 1]  # Corr between resp to A and resp to dual
            # BcorAB = np.corrcoef(rBm_cut, rABm_cut)[0, 1]  # Corr between resp to B and resp to dual

            A_FR, B_FR, AB_FR = np.nanmean(rA_st_cut), np.nanmean(rB_st_cut), np.nanmean(rAB_st_cut)

            min_rep = np.min((rA.shape[0], rB.shape[0])) #only will do something if SoundRepeats==Yes
            lin_resp = np.nanmean(rAsm_cut[:min_rep, :] + rBsm_cut[:min_rep, :], axis=0)
            supp = np.nanmean(lin_resp - AB_FR)

            # AcorLin = np.corrcoef(rAm_cut, lin_resp)[0, 1]  # Corr between resp to A and resp to lin
            # BcorLin = np.corrcoef(rBm_cut, lin_resp)[0, 1]  # Corr between resp to B and resp to lin

            # Apref, Bpref = AcorAB - AcorLin, BcorAB - BcorLin
            # pref = Apref - Bpref

            cell_dict[f"bg_FR{lb}"], cell_dict[f"fg_FR{lb}"], cell_dict[f"combo_FR{lb}"] = A_FR, B_FR, AB_FR
            # cell_dict[f"AcorAB{lb}"], cell_dict[f"BcorAB{lb}"] = AcorAB, BcorAB
            # cell_dict[f"AcorLin{lb}"], cell_dict[f"B_corLin{lb}"] = AcorLin, BcorLin
            # cell_dict[f"pref{lb}"], cell_dict[f"Apref{lb}"], cell_dict[f"Bpref{lb}"] = pref, Apref, Bpref
            cell_dict[f"supp{lb}"] = supp

        cell_dff.append(cell_dict)

        # if params['Binaural'] == 'Yes':
        #     dA, dB = ohel.get_binaural_adjacent_epochs(stimmy)
        #
        #     rdA, rdB = resp.extract_epoch(dA), resp.extract_epoch(dB)
        #     rdAm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdA))[:, presil:-postsil], axis=0)
        #     rdBm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdB))[:, presil:-postsil], axis=0)
        #
        #     ABcordA = np.corrcoef(rABm, rdAm)[0, 1]  # Corr between resp to AB and resp to BG swap
        #     ABcordB = np.corrcoef(rABm, rdBm)[0, 1]  # Corr between resp to AB and resp to FG swap

    cell_df = pd.DataFrame(cell_dff)
    cell_df['SR'], cell_df['STD'] = SR, STD
    # cell_df['corcoef'], cell_df['avg_resp'], cell_df['snr'] = corcoef, avg_resp, snr
    cell_df.insert(loc=0, column='area', value=area)

    return cell_df


OLP_fit_partial_weights(batch, threshold=None, synth=False, snip=None, fs=100, labels=None):
weight_list = []

weight_list = []
batch = 340
fs = 100
lfreq, hfreq, bins = 100, 24000, 48
# threshold = 0.75
threshold = None
snip = [0, 0.5]
synth = True
cell_df = nd.get_batch_cells(batch)
cell_list = cell_df['cellid'].tolist()
cell_list = ohel.manual_fix_units(cell_list)  # So far only useful for two TBR cells

# Only CLT synth units
cell_list = [cell for cell in cell_list if (cell.split('-')[0][:3] == 'CLT') & (int(cell.split('-')[0][3:6]) < 26)]
fit_epochs = ['N', 'C', 'T', 'S', 'U', 'M', 'A']
fit_epochs = ['10', '01', '20', '02', '11', '12', '21', '22']

loader = 'env100'
modelspecs_dir = '/auto/users/luke/Code/nems/modelspecs'

for cellid in cell_list:
    loadkey = 'ns.fs100'
    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = {'rasterfs': 100,
               'stim': False,
               'resp': True}
    rec = manager.get_recording(**options)

    # GET sound envelopes and get the indices for chopping?
    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    FG_folder, fgidx = ref_handle['FG_Folder'], list(set(ref_handle['Foreground']))
    fgidx.sort(key=int)
    idxstr = [str(ff).zfill(2) for ff in fgidx]

    fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'{FG_folder}/{ff}*.wav'))[0] for ff in idxstr]
    fgname = [ff.split('/')[-1].split('.')[0].replace(' ', '') for ff in fg_paths]
    ep_fg = [f"STIM_null_{ff}" for ff in fgname]

    prebins = int(ref_handle['PreStimSilence'] * options['rasterfs'])
    postbins = int(ref_handle['PostStimSilence'] * options['rasterfs'])
    durbins = int(ref_handle['Duration'] * options['rasterfs'])
    trialbins = durbins + postbins

    if threshold:
        env_cuts = {}
        for nm, pth in zip(fgname, fg_paths):
            sfs, W = wavfile.read(pth)
            spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

            env = np.nanmean(spec, axis=0)
            cutoff = np.max(env) * threshold

            # aboves = np.squeeze(np.argwhere(env >= cutoff))
            # belows = np.squeeze(np.argwhere(env < cutoff))

            highs, lows, whole_thing = env >= cutoff, env < cutoff, env > 0
            prestimFalse = np.full((prebins,), False)
            poststimTrue = np.full((trialbins - len(env),), True)
            poststimFalse = np.full((trialbins - len(env),), False)  ## Something is wrong here with the lengths

            full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
            aboves = np.concatenate((prestimFalse, highs, poststimFalse))
            belows = np.concatenate((prestimFalse, lows, poststimFalse))
            belows_post = np.concatenate((prestimFalse, lows, poststimTrue))

            env_cuts[nm] = [full, aboves, belows, belows_post]

            f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
            ax[0].plot(env)
            ax[0].hlines(cutoff, 0, 100, ls=':')
            ax[0].set_title(f"{nm}")
            ax[1].plot(env[highs])
            ax[2].plot(env[lows])

            cut_labels = ['', '_h', '_l', '_lp']

    if snip:
        start, dur = int(snip[0] * fs), int(snip[1] * fs)
        prestimFalse = np.full((prebins,), False)
        # poststimTrue = np.full((trialbins - len(env),), True)
        poststimFalse = np.full((trialbins - durbins), False)
        # if start == dur:
        #
        # else:
        end = durbins - start - dur
        goods = [False] * start + [True] * dur + [False] * end
        bads = [not ll for ll in goods]

        full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
        goods = np.concatenate((prestimFalse, goods, poststimFalse))
        bads = np.concatenate((prestimFalse, bads, poststimFalse))
        cut_list = [full, goods, bads]
        cut_labels = ['', '_good', '_bad']

    rec['resp'].fs = fs
    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())

    _, SR, _ = ohel.remove_spont_rate_std(resp)

    stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_')

    val = rec.copy()
    val['resp'] = val['resp'].rasterize()
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    est_sub = None

    df0 = val['resp'].epochs.copy()
    df2 = val['resp'].epochs.copy()
    if synth == True:
        df0['name'] = df0['name'].apply(ohel.label_synth_type)
    else:
        df0['name'] = df0['name'].apply(ohel.label_ep_type)
    df0 = df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])

    val['resp'].epochs = df3
    val_sub = copy.deepcopy(val)
    val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)

    val = val_sub
    fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR / rec['resp'].fs)
    val['resp'] = val['resp'].transform(fn)

    print(f'calc weights {cellid}')

    # where twostims fit actually begins
    epcs = val.epochs[val.epochs['name'].str.count('-0-1') >= 1].copy()
    sepname = epcs['name'].apply(get_sep_stim_names)
    epcs['nameA'] = [x[0] for x in sepname.values]
    epcs['nameB'] = [x[1] for x in sepname.values]

    # epochs with two sounds in them
    epcs_twostim = epcs[epcs['name'].str.count('-0-1') == 2].copy()

    A, B, AB, sepnames = ([], [], [], [])  # re-defining sepname
    for i in range(len(epcs_twostim)):
        if any((epcs['nameA'] == epcs_twostim.iloc[i].nameA) & (epcs['nameB'] == 'null')) \
                and any((epcs['nameA'] == 'null') & (epcs['nameB'] == epcs_twostim.iloc[i].nameB)):
            A.append('STIM_' + epcs_twostim.iloc[i].nameA + '_null')
            B.append('STIM_null_' + epcs_twostim.iloc[i].nameB)
            AB.append(epcs_twostim['name'].iloc[i])
            sepnames.append(sepname.iloc[i])

    # Calculate weights
    if synth == True:
        subsets = len(cut_list)
    else:
        subsets = len(list(env_cuts.values())[0])
    weights = np.zeros((2, len(AB), subsets))
    Efit = np.zeros((5, len(AB), subsets))
    nMSE = np.zeros((len(AB), subsets))
    nf = np.zeros((len(AB), subsets))
    r = np.zeros((len(AB), subsets))
    cut_len = np.zeros((len(AB), subsets - 1))
    get_error = []

    if synth:
        for i in range(len(AB)):
            names = [[A[i]], [B[i]], [AB[i]]]
            for ss, cut in enumerate(cut_list):
                weights[:, i, ss], Efit[:, i, ss], nMSE[i, ss], nf[i, ss], _, r[i, ss] = \
                    ofit.calc_psth_weights_of_model_responses_list(val, names,
                                                                   signame='resp', cuts=cut)
                if ss != 0:
                    cut_len[i, ss - 1] = np.sum(cut)

    else:
        for i in range(len(AB)):
            names = [[A[i]], [B[i]], [AB[i]]]
            Fg = names[1][0].split('_')[2].split('-')[0]
            cut_list = env_cuts[Fg]

            for ss, cut in enumerate(cut_list):
                weights[:, i, ss], Efit[:, i, ss], nMSE[i, ss], nf[i, ss], _, r[i, ss] = \
                    ofit.calc_psth_weights_of_model_responses_list(val, names,
                                                                   signame='resp', cuts=cut)
                if ss != 0:
                    cut_len[i, ss - 1] = np.sum(cut)
                # get_error.append(ge)

    ### This was all before I more smarter and less lazier and coded the stuff below to be flexible about how you're cutting
    # if subsets == 4 & synth == False:
    #     weight_df = pd.DataFrame(
    #         [epcs_twostim['nameA'].values, epcs_twostim['nameB'].values,
    #          weights[0, :, 0], weights[1, :, 0], nMSE[:, 0], nf[:, 0], r[:, 0],
    #          weights[0, :, 1], weights[1, :, 1], nMSE[:, 1], nf[:, 1], r[:, 1], cut_len[:,0],
    #          weights[0, :, 2], weights[1, :, 2], nMSE[:, 2], nf[:, 2], r[:, 2], cut_len[:,1],
    #          weights[0, :, 3], weights[1, :, 3], nMSE[:, 3], nf[:, 3], r[:, 3], cut_len[:,2],])
    #     weight_df = weight_df.T
    #     weight_df.columns = ['namesA', 'namesB', 'weightsA', 'weightsB', 'nMSE', 'nf', 'r',
    #                          'weightsA_h', 'weightsB_h', 'nMSE_h', 'nf_h', 'r_h', 'h_idxs',
    #                          'weightsA_l', 'weightsB_l', 'nMSE_l', 'nf_l', 'r_l', 'l_idxs',
    #                          'weightsA_lp', 'weightsB_lp', 'nMSE_lp', 'nf_lp', 'r_lp', 'lp_idxs']
    #     cols = ['namesA', 'namesB', 'weightsA', 'weightsB', 'nMSE']
    #     print(weight_df[cols])
    #
    #     weight_df = weight_df.astype({'weightsA': float, 'weightsB': float,
    #                                   'weightsA_h': float, 'weightsB_h': float,
    #                                   'weightsA_l': float, 'weightsB_l': float,
    #                                   'weightsA_lp': float, 'weightsB_lp': float,
    #                                   'nMSE': float, 'nf': float, 'r': float,
    #                                   'nMSE_h': float, 'nf_h': float, 'r_h': float,
    #                                   'nMSE_l': float, 'nf_l': float, 'r_l': float,
    #                                   'nMSE_lp': float, 'nf_lp': float, 'r_lp': float,
    #                                   'h_idxs': float, 'l_idxs': float, 'lp_idxs': float})

    # If this part is working the above code is useless.
    # Makes a list of lists that iterates through the arrays you created, then flattens them in the next line
    big_list = [[weights[0, :, ee], weights[1, :, ee], nMSE[:, ee], nf[:, ee], r[:, ee]] for ee in range(len(cut_list))]
    flat_list = [item for sublist in big_list for item in sublist]
    small_list = [epcs_twostim['nameA'].values, epcs_twostim['nameB'].values]
    # Combines the lists into a format that is conducive to the dataframe format I want to make
    bigger_list = small_list + flat_list
    weight_df = pd.DataFrame(bigger_list)
    weight_df = weight_df.T

    # Automatically generates a list of column names based on the names of the subsets provided above
    column_labels1 = ['namesA', 'namesB']
    column_labels2 = [[f"weightsA{cl}", f"weightsB{cl}", f"nMSE{cl}", f"nf{cl}", f"r{cl}"] for cl in cut_labels]
    column_labels_flat = [item for sublist in column_labels2 for item in sublist]
    column_labels = column_labels1 + column_labels_flat
    # Renames the columns according to that list - should work for any scenario as long as you specific names above
    weight_df.columns = column_labels1 + column_labels_flat

    # Not sure why I need this, I guess some may not be floats, so just doing it
    col_dict = {ii: float for ii in column_labels_flat}
    weight_df = weight_df.astype(col_dict)

    weight_df.insert(loc=0, column='cellid', value=cellid)
    weight_list.append(weight_df)

weight_df0 = pd.concat(weight_list)

ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df0.namesA, weight_df0.namesB)]
weight_df0 = weight_df0.drop(columns=['namesA', 'namesB'])
weight_df0['epoch'] = ep_names

OLP_partialweights_db_path = f'/auto/users/hamersky/olp_analysis/Binaural_OLP_control_segment{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}_nometrics.h5'  # weight + corr
os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)
store = pd.HDFStore(OLP_partialweights_db_path)
df_store = copy.deepcopy(weight_df0)
store['df'] = df_store.copy()
store.close()

### Adding to slap in a fix for the FR
# Gets some cell metrics
cuts_info = {cut_labels[i]: cut_list[i] for i in range(len(cut_list))}
metrics = []
for cellid in cell_list:
    cell_metric = calc_psth_metrics_cuts(batch, cellid, cut_ids=cuts_info)
    cell_metric.insert(loc=0, column='cellid', value=cellid)
    print(f"Adding cellid {cellid}.")
    metrics.append(cell_metric)
df = pd.concat(metrics)
df.reset_index()

OLP_metrics_db_path = f'/auto/users/hamersky/olp_analysis/Binaural_test{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}metrics.h5'  # weight + corr
os.makedirs(os.path.dirname(OLP_metrics_db_path), exist_ok=True)
store = pd.HDFStore(OLP_metrics_db_path)
df_store = copy.deepcopy(df)
store['df'] = df_store.copy()
store.close()

weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
weight_df0['fit_segment'] = f"{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}"

OLP_savepath = f'/auto/users/hamersky/olp_analysis/Binaural_OLP_segment{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}.h5'  # weight + corr
os.makedirs(os.path.dirname(OLP_savepath), exist_ok=True)
store = pd.HDFStore(OLP_savepath)
df_store = copy.deepcopy(weight_df)
store['df'] = df_store.copy()
store.close()

# I think this is when you're combining other dfs you loaded with a new fit
weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
# weight_df0['threshold'] = str(int(threshold * 100))
# if df.shape[0] != weights_df.shape[0] or weight_df.shape[0] != weights_df.shape[0]:
#     raise ValueError("Resulting weights_df does not match length of parts, some epochs were dropped.")


OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/test500-750metrics.h5'  # weight + corr
os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)
store = pd.HDFStore(OLP_partialweights_db_path)
df_store = copy.deepcopy(weight_df0)
store['df'] = df_store.copy()
store.close()

##load here, 2022_10_24, these are from clathrus synthetic as I try to fit the partial model
OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_control_segment500-750_goodmetrics.h5'  # weight + corr

OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_full_partial_weights_nometrics.h5'  # weight + corr
OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_full_partial_weights_withmetrics.h5'  # weight + corr

part_weights = False
if part_weights == True:
    os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)
    store = pd.HDFStore(OLP_partialweights_db_path)
    df_store = copy.deepcopy(weight_df0)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_partialweights_db_path)
    df = store['df']
    store.close()