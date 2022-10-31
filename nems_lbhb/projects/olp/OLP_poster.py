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


# ###############################################################################
# ### NGP Poster Figures ########################################################
# ### Many of these are found elsewhere, just bigger fonts specifically #########
# ### for NGP retreat poster ####################################################
#
# # Interactive scatter stuff
# ## Trying to get a good site out of the binaural
# df = weight_df.loc[weight_df.area=='A1'].copy()
# df = weight_df.loc[df.synth_kind == 'N'].copy()
# df = weight_df.loc[df.kind == '11'].copy()
# df['expt_num'] = [int(aa[4:6]) for aa in df['cellid']]
# # Chimes, Chainsaw, Blender, RockTumble, Drill, Sander, Hairdryer, Keys, Heels, KitWhine
# df = df.loc[df.expt_num == 8]
# df = df.loc[df.BG == 'Wind']
# df = df.loc[df.FG == 'Geese']
#
# # Poster Panel Three
# path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_sound_stats.h5'
# weight_df = ofit.OLP_fit_weights(batch, loadpath=path)
# if 'synth_kind' not in weight_df:
#     weight_df['synth_kind'] = 'A'
# opo.poster3_psths_with_specs(weight_df, 'CLT008a-046-2', 'Wind', 'Geese', sigma=1, synth_kind='A')
# opo.poster3_psths_with_specs(weight_df, 'CLT008a-019-3', 'Wind', 'Geese', sigma=1, synth_kind='A') # BG supp
# opo.poster3_response_heatmaps_comparison(weight_df, 'CLT008', 'Wind', 'Geese',
#                                   cellid=['046-2', '019-3'], example=True, synth_kind='A', sigma=2)
# # Poster Panel Four
# path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5'
# weight_df = ofit.OLP_fit_weights(batch, loadpath=path)
# syn_df = weight_df.loc[weight_df.synth_kind=='N']
#
# tt1, tt2, lens = opo.poster4_histogram_summary_plot(syn_df, threshold=0.03)
# opo.poster4_rel_gain_hist(syn_df)
#
# # Poster Panel Five
# path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5'
# weight_df = ofit.OLP_fit_weights(batch, loadpath=path)
# opo.poster5_synthetic_relative_gain_comparisons_specs(weight_df, 'Jackhammer', 'Fight Squeak',
#                                                       synth_show=['N','U','S','T','C'])
# weight_df['BG_rel_gain'] = (weight_df.weightsA - weight_df.weightsB) / \
#                            (np.abs(weight_df.weightsB) + np.abs(weight_df.weightsA))
# weight_df['FG_rel_gain'] = (weight_df.weightsB - weight_df.weightsA) / \
#                            (np.abs(weight_df.weightsB) + np.abs(weight_df.weightsA))
# opo.poster5_sound_metric_scatter(weight_df, ['Fstationary','Tstationary','bandwidth'], 'BG_rel_gain',
#                                  ['Frequency\nNon-stationarity', 'Temporal\nNon-stationarity',
#                                   'Bandwidth'], jitter=[0.25,0.2,0.03], metric_filter=2.5)



def poster3_psths_with_specs(df, cellid, bg, fg, batch=340, bin_kind='11', synth_kind='N',
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



def poster3_response_heatmaps_comparison(df, site, bg, fg, cellid=None, batch=340, bin_kind='11',
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
        ax[qq].set_yticklabels(num_ids, fontsize=9, fontweight='bold')
        ax[qq].set_xlim(-0.2, (dur + 0.3))  # arbitrary window I think is nice
        if example == True:
            titles = [f"BG:\n{bg}", f"FG:\n{fg}", f"Combo\nBG+FG"]
            ax[qq].set_ylabel(f"{titles[ww]}", fontsize=15, fontweight='bold', rotation=90,
                              horizontalalignment='center') #, labelpad=40)
            ax[0].set_title(f'Site {all_cells.iloc[0].cellid[:7]} Responses', fontweight='bold', fontsize=15)
        else:
            ax[qq].set_title(f"{epochs[ww]}", fontsize=8, fontweight='bold')
            ax[qq].set_ylabel('Unit', fontweight='bold', fontsize=8)
        ax[qq].spines['top'].set_visible(True), ax[qq].spines['right'].set_visible(True)
    ax[2].set_xlabel('Time (s)', fontweight='bold', fontsize=12)
    ax[2].set_xticks([0.0, 0.5, 1.0])
    ax[2].set_xticklabels([0.0, 0.5, 1.0], fontsize=10)
    ax[0].set_xticks([]), ax[1].set_xticks([])
    # Add the colorbar to the axis to the right of these, the diff will get separate cbar
    cbar = fig.colorbar(dd, ax=ax[4], aspect=7)
    cbar.ax.tick_params(labelsize=10)
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
    ax[3].set_xticks([0.0, 0.5, 1.0])
    ax[3].set_xticklabels([0.0, 0.5, 1.0], fontsize=10)
    ax[3].set_yticks([*range(0, len(all_cells))])
    ax[3].set_ylabel('Unit', fontweight='bold', fontsize=12)
    ax[3].set_yticklabels(num_ids, fontsize=9, fontweight='bold')
    ax[3].set_title(f"Difference (Combo - Linear Sum)", fontsize=15, fontweight='bold')
    ax[3].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax[3].spines['top'].set_visible(True), ax[3].spines['right'].set_visible(True)

    dbar = fig.colorbar(ddd, ax=ax[5], aspect=7)
    dbar.ax.tick_params(labelsize=10)
    ax[5].spines['top'].set_visible(False), ax[5].spines['right'].set_visible(False)
    ax[5].spines['bottom'].set_visible(False), ax[5].spines['left'].set_visible(False)
    ax[5].set_yticks([]), ax[5].set_xticks([])


def poster4_histogram_summary_plot(weight_df, threshold=0.05):
    '''Pretty niche plot that will plot BG+/FG+ histograms and compare BG and FG weights,
    then plot BG+/FG- histogram and BG-/FG+ histogram separate and then compare BG and FG
    again in a bar graph. I guess you could put any thresholded quadrants you want, but the
    default is the only that makes sense. Last figure on APAN/SFN poster.'''
    quad, _ = ohel.quadrants_by_FR(weight_df, threshold=threshold, quad_return=[3, 2, 6])
    quad3, quad2, quad6 = quad.values()

    f = plt.figure(figsize=(15, 7)) # I made the height 5 for NGP retreat poster
    histA = plt.subplot2grid((7, 17), (0, 0), rowspan=5, colspan=3)
    meanA = plt.subplot2grid((7, 17), (0, 4), rowspan=5, colspan=2)
    histB = plt.subplot2grid((7, 17), (0, 8), rowspan=5, colspan=3)
    histC = plt.subplot2grid((7, 17), (0, 11), rowspan=5, colspan=3)
    meanB = plt.subplot2grid((7, 17), (0, 15), rowspan=5, colspan=2)
    ax = [histA, meanA, histB, histC, meanB]

    edges = np.arange(-1, 2, .05)
    na, xa = np.histogram(quad3.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(quad3.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[0].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
    ax[0].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
    ax[0].legend(('Background', 'Foreground'), fontsize=12, prop=dict(weight='bold'), labelspacing=0.25)

    ax[0].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=14)
    ax[0].set_title(f"Respond to both\nBG and FG alone", fontweight='bold', fontsize=16)
    ax[0].set_xlabel("Mean Weight", fontweight='bold', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=10)
    ymin, ymax = ax[0].get_ylim()

    BG1, FG1 = np.mean(quad3.weightsA), np.mean(quad3.weightsB)
    BG1sem, FG1sem = stats.sem(quad3.weightsA), stats.sem(quad3.weightsB)
    ttest1 = stats.ttest_ind(quad3.weightsA, quad3.weightsB)
    ax[1].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
    ax[1].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')
    ax[1].set_ylabel('Mean Weight', fontweight='bold', fontsize=14)
    ax[1].set_xticklabels(['BG','FG'], fontsize=12, fontweight='bold')
    ax[1].tick_params(axis='y', which='major', labelsize=10)
    # ax[1].set_ylim(0, 0.79)
    if ttest1.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"{ttest1.pvalue:.3f}"
    ax[1].set_title(title, fontsize=12)

    BG2, FG2 = np.mean(quad6.weightsA), np.mean(quad2.weightsB)
    BG2sem, FG2sem = stats.sem(quad6.weightsA), stats.sem(quad2.weightsB)
    ttest2 = stats.ttest_ind(quad6.weightsA, quad2.weightsB)
    ax[4].bar("BG", BG2, yerr=BG2sem, color='deepskyblue')
    ax[4].bar("FG", FG2, yerr=FG2sem, color='yellowgreen')
    ax[4].set_ylabel("Weight", fontweight='bold', fontsize=14)
    ax[4].tick_params(axis='y', which='major', labelsize=10)
    # ax[4].set_ylim(0, 0.79)
    if ttest2.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"{ttest2.pvalue:.3f}"
    ax[4].set_title(title, fontsize=12)
    mean_big = np.max([BG1, FG1, BG2, FG2])
    ax[1].set_ylim(0, mean_big+(mean_big*0.1))
    ax[4].set_ylim(0, mean_big+(mean_big*0.1))
    ax[4].set_xticklabels(['BG','FG'], fontsize=12, fontweight='bold')

    na, xa = np.histogram(quad6.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(quad2.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[2].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
    ax[3].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
    ax[2].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=14)
    ax[2].set_title(f"Respond to BG\nalone only", fontweight='bold', fontsize=16)
    ax[3].set_title(f"Respond to FG\nalone only", fontweight='bold', fontsize=16)
    ax[2].set_xlabel("Weight", fontweight='bold', fontsize=14)
    ax[2].tick_params(axis='both', which='major', labelsize=10)
    ax[3].set_xlabel("Weight", fontweight='bold', fontsize=14)
    biggest = np.max([na,nb])
    # ax[2].set_ylim(ymin, ymax), ax[3].set_ylim(ymin, ymax)
    ax[2].set_ylim(ymin, biggest), ax[3].set_ylim(ymin, biggest)
    ax[3].set_yticks([])
    ax[3].tick_params(axis='both', which='major', labelsize=10)

    return ttest1, ttest2, [quad3.shape[0], quad2.shape[0], quad6.shape[0]]


def poster4_rel_gain_hist(df, threshold=0.03, quad_return=3):
    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=quad_return)
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
    ax.set_xlabel('Relative Gain', fontweight='bold', fontsize=14)
    ax.set_ylabel('Density', fontweight='bold', fontsize=14)
    ax.set_yticks([0.0,1.0])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title(f'Percent\nSuppression:\n{a1_percent_supp}', fontsize=13)
    # This may bite me in the ass, but hard code this in for now so that the vertical line through 0
    # doesn't end up chopping a bar in half depending on the range of rel
    ax.set_xlim(-2.5,1.5)
    fig.tight_layout()




def poster5_synthetic_relative_gain_comparisons_specs(df, bg, fg, thresh=0.03, quads=3,
                                                      area='A1', batch=340, synth_show=None):
    '''Made 2022_09_08. Makes the big figure on panel 5 of my 2022 NGP Poster. It takes a dataframe
    and a list of the synthetic code letters and vertically arranges them as histograms of relative
    gain, all aligned on the same zero so you can ideally see the shift. You also can put in a BG
    and FG name (with spaces in the name if it has!) and it will show the corresponding spectrograms
    for the synthetic conditions nextdoor to the relative gain plot. It's a fun figure.'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    quad = quad.loc[quad.area==area]

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    kind_alias = {'A': 'Non-RMS Norm\nNatural', 'N': 'Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal\nBandwidth', 'S': 'Spectral\nBandwidth',
                  'T': 'Temporal\nBandwidth', 'C': 'Bandwidth'}
    kind_dict = {'M': 'SpectrotemporalMod', 'U': 'Spectrotemporal', 'T': 'Temporal',
                  'S': 'Spectral', 'C': 'Cochlear'}

    # Shortcut to typing in all the actual conditions minus the control of the control
    if synth_show == 'A-':
        synth_show = ['N', 'C', 'T', 'S', 'U', 'M']
    if isinstance(synth_show, str):
        synth_show = [synth_show]

    lens = len(synth_show)
    hists, bgs, fgs = [], [], []
    # fig, axes = plt.subplots(len(synth_show), 3, figsize=(8, len(synth_show)*2))
    fig, axes = plt.subplots(figsize=(8, lens*2))
    fig, axes = plt.subplots(figsize=(9, 18))
    for aa in range(lens):
        hist = plt.subplot2grid((lens*5, 15), (0+(aa*5), 11), rowspan=4, colspan=4)
        bgsp = plt.subplot2grid((lens*5, 15), (1+(aa*5), 0), rowspan=3, colspan=4)
        fgsp = plt.subplot2grid((lens*5, 15), (1+(aa*5), 5), rowspan=3, colspan=4)
        hists.append(hist), bgs.append(bgsp), fgs.append(fgsp)
    ax = hists + bgs + fgs

    ymins, ymaxs, xmins, xmaxs = [], [], [], []
    for qq in range(lens):
        to_plot = quad.loc[quad.synth_kind==synth_show[qq]].copy()
        # Calculate relative gain and percent suppressed
        rel_gain = (to_plot.weightsB - to_plot.weightsA) / \
                              (to_plot.weightsB + to_plot.weightsA)
        supps = [cc for cc in rel_gain if cc < 0]
        percent_supp = np.around((len(supps) / len(rel_gain)) * 100, 1)
        print(percent_supp)

        rel_gain = rel_gain.loc[rel_gain <= 2.5]
        rel_gain = rel_gain.loc[rel_gain >= -2.5]

        # Plot
        p = sb.distplot(rel_gain, bins=50, color='black', norm_hist=True, kde=True, ax=ax[qq],
                        kde_kws=dict(linewidth=0.5))

        ymin, ymax = ax[qq].get_ylim()
        ax[qq].vlines(0, ymin, ymax, color='black', ls = '--', lw=1)
        # Change color of suppressed to red, enhanced to blue
        for rectangle in p.patches:
            if rectangle.get_x() < 0:
                rectangle.set_facecolor('tomato')
        for rectangle in p.patches:
            if rectangle.get_x() >= 0:
                rectangle.set_facecolor('dodgerblue')
        # This might have to be change if I use %, which I want to, but density is between 0-1, cleaner
        ax[qq].set_yticks([0,1])
        ax[qq].set_yticklabels([0,1])
        # All axes match natural. Would need to be changed if natural cuts for some reason.
        if qq == 0:
            ax[qq].set_ylabel('Density', fontweight='bold', fontsize=12)
            xmin, xmax = ax[qq].get_xlim()
        else:
            ax[qq].set_xlim(xmin, xmax)
            ax[qq].set_ylabel('')

        if qq == (lens-1):
            ax[qq].set_xlabel('Relative Gain', fontweight='bold', fontsize=12)
        ax[qq].text((xmin+(np.abs(xmin)*0.1)), 0.75,
                    f"Percent\nSupp:\n{percent_supp}", fontsize=9)
        ax[qq].tick_params(axis='both', which='major', labelsize=10)

        # Spectrogram parts
        manager = BAPHYExperiment(cellid=quad.iloc[0].cellid, batch=batch)
        folder_ids = [int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['BG_Folder'][-1]),
                int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['FG_Folder'][-1])]

        if synth_show[qq]=='A' or synth_show[qq]=='N':
            bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Background{folder_ids[0]}/*.wav'))
            fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Foreground{folder_ids[1]}/*.wav'))
        else:
            bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Background{folder_ids[0]}/{kind_dict[synth_show[qq]]}/*.wav'))
            fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Foreground{folder_ids[1]}/{kind_dict[synth_show[qq]]}/*.wav'))

        bg_path = [bb for bb in bg_dir if bg in bb]
        fg_path = [ff for ff in fg_dir if fg in ff]

        if len(bg_path)==0 or len(fg_path)==0:
            raise ValueError(f"Your BG {bg} or FG {fg} aren't in there. Maybe add a space if it needs.")

        paths = [bg_path, fg_path]
        # 1 and 2 because that is how much will get added to do the different axes
        for ww in range(1,3):
            sfs, W = wavfile.read(paths[ww-1][0])
            spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 16000)
            qqq = qq + (ww * lens) # get you to correct axes
            ax[qqq].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]],
                     cmap='gray_r')
            ax[qqq].set_xticks([]), ax[qqq].set_yticks([])
            ax[qqq].set_xticklabels([]), ax[qqq].set_yticklabels([])
            ax[qqq].spines['top'].set_visible(True), ax[qqq].spines['bottom'].set_visible(True)
            ax[qqq].spines['left'].set_visible(True), ax[qqq].spines['right'].set_visible(True)
            if ww == 1:
                ax[qqq].set_ylabel(f"{kind_alias[synth_show[qq]]}", fontsize=12, fontweight='bold',
                                   horizontalalignment='center', rotation=0, labelpad=40,
                                   verticalalignment='center')
            if qq == 0 and ww == 1:
                ax[qqq].set_title(f"BG: {bg}", fontweight='bold', fontsize=10)
            elif qq == 0 and ww == 2:
                ax[qqq].set_title(f"FG: {fg}", fontweight='bold', fontsize=10)
            # if qq == (lens - 1):
            #     ax[qqq].set_xlabel('Time (s)', fontweight='bold', fontsize=10)



def poster5_sound_metric_scatter(df, x_metrics, y_metric, x_labels, area='A1', threshold=0.03,
                         jitter=[0.2,0.03,0.25],
                         quad_return=3, metric_filter=None, synth_kind='N', bin_kind='11'):
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
        ax.set_xlabel(x_labels[cnt], fontweight='bold', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        if cnt==0:
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        else:
            ax.set_ylabel('')

        # Run a regression
        Y = np.concatenate((quad[y_metric].values, quad[y_metric2].values))
        X = np.concatenate((quad[f'BG_{met}'].values, quad[f'FG_{met}'].values))
        reg = stats.linregress(X, Y)
        x = np.asarray(ax.get_xlim())
        y = reg.slope * x + reg.intercept
        ax.plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\np = {reg.pvalue:.3f}")
        ax.legend(loc='upper right', fontsize=11, labelspacing=0.4)

        ax.set_yticks([-1.0, 0.0, 1.0])
    fig.tight_layout()

    axes[0].set_xticks([20, 40, 60])
    axes[1].set_xticks([0, 20, 40])
    axes[2].set_xticks([1, 2, 3, 4])

    # fig.suptitle(f"{title}", fontweight='bold', fontsize=10)
