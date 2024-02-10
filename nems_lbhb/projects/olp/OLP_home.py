import nems0.db as nd
import re
import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
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
from nems_lbhb import baphy_io
plt.rcParams['svg.fonttype'] = 'none'

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
fs = 100


# Load your different, updated dataframes
# path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_quarter_segments.h5'  # All quarter segments in one df
# path = '/auto/users/hamersky/olp_analysis/ARM_Dynamic_OLP_segment0-500.h5' # ARM hopefully
# path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_segment0-500.h5' #Vinaural half models
# path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_sound_stats.h5' #weight + corr
# path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5' # Still needs the CLT053 4 units
# path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_segment0-500.h5' # The half models, use this now
# path = '/auto/users/hamersky/olp_analysis/a1_celldat1.h5'
# path = '/auto/users/hamersky/olp_analysis/Synthetic_OLP_segment0-500_with_stats.h5' # The half models, use this now
# weight_df = ofit.OLP_fit_weights(loadpath=path)
# weight_df['batch'] = 340
#
#
# # The thing I was working on in January with fit
# path = '/auto/users/hamersky/olp_analysis/2023-01-12_Batch341_0-500_FULL'
#
#
# #marms
# path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch328_0-500_marm'
# weight_df = jl.load(path)
#
#
# #spikes path
# filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
# weight_dff = ohel.add_spike_widths(filt, save_name='ferrets_with_spikes3', cutoff={'SLJ': 0.35, 'PRN': 0.35, 'other': 0.375})
# path = f'/auto/users/hamersky/olp_analysis/ferrets_with_spikes'
# path = f'/auto/users/hamersky/olp_analysis/ferrets_with_spikes2'
# path = f'/auto/users/hamersky/olp_analysis/ferrets_with_spikes3'
# weight_df = jl.load(path)

# 2023_05_02. Starting with Prince data too and new df structure
# path = '/auto/users/hamersky/olp_analysis/2023-05-10_batch344_0-500_metrics' # Full one with updated PRNB layers
# path = '/auto/users/hamersky/olp_analysis/2023-05-17_batch344_0-500_metrics' #full one with PRNB layers and paths
# path = '/auto/users/hamersky/olp_analysis/2023-07-20_batch344_0-500_metrics' # full with new FR snr metric
# path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch344_0-500_metric'
# path = '/auto/users/hamersky/olp_analysis/2023-09-15_batch344_0-500_final'
# path = '/auto/users/hamersky/olp_analysis/2023-09-21_batch344_0-500_final'
path = '/auto/users/hamersky/olp_analysis/2023-09-22_batch344_0-500_final'
weight_df = jl.load(path)

filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
kit_filt = filt.loc[(filt.FG=='KitWhine') | (filt.FG=='KitHigh')] #|
                    #(filt.FG=='Kit_Whine') | (filt.FG=='Kit_High') | (filt.FG=='Kit_Low')]
stat_dict = ofig.weight_summary_histograms_manuscript(kit_filt, bar=True, stat_plot='median')

filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=False)
kit_filt = filt.loc[(filt.FG=='KitWhine') | (filt.FG=='KitHigh') |
                    (filt.FG=='Kit_Whine') | (filt.FG=='Kit_High') | (filt.FG=='Kit_Low')]
ofig.all_filter_stats(kit_filt, xcol='bg_snr', ycol='fg_snr', snr_thresh=0.12, r_cut=0.4, increment=0.2,
                 fr_thresh=0.01, xx='resp')


#behavior stuff 2024_03_01
path = '/auto/users/hamersky/olp_analysis/2024-01-04_batch349_final_20pre' # isec fit, 0.2s prestim
path = '/auto/users/hamersky/olp_analysis/2024-01-04_batch349_final' # 1sec fit, 0.5s prestim
path = '/auto/users/hamersky/olp_analysis/2024-01-05_batch349_200msto1200ms_final'  # 1s fit, starting at 1.2s
path = '/auto/users/hamersky/olp_analysis/2024-02-08_batch349_20to120ms'
weight_df = jl.load(path)

# "naive" comparison
path = '/auto/users/hamersky/olp_analysis/2024-01-08_batch352_LEMON_OLP_standard'
path = '/auto/users/hamersky/olp_analysis/2024-02-08_batch352_LEMON_OLP_standard'
weight_df = jl.load(path)

#2024_02_08 fitting half weights
filt = weight_df.loc[(weight_df.dyn_kind=='fh') | (weight_df.dyn_kind=='hf')]
filt = filt.loc[(filt.area == 'A1') | (filt.area == 'PEG')]
# Rename some layers that are named funny because of the labelling GUI, not a meaningful distinction
filt.loc[filt.layer == '4', 'layer'] = '44'
filt.loc[filt.layer == '5', 'layer'] = '56'
filt.loc[filt.layer == 'BS', 'layer'] = '13'
# Save only certain layers that are cortical
filt = filt.loc[(filt.layer == 'NA') | (filt.layer == '5') | (filt.layer == '44') | (filt.layer == '13') |
                (filt.layer == '4') | (filt.layer == '56') | (filt.layer == '16') | (filt.layer == 'BS')]
aa = filt

snr_threshold, rel_cut, r_cut, weight_lim = 0.12, 2.5, 0.4, [-0.5, 2]
filt = filt.loc[(filt.bg_snr_end_stim >= snr_threshold) & (filt.fg_snr_end_stim >= snr_threshold)]
filt = filt.loc[(filt[f'FG_rel_gain_end'] <= rel_cut) & (filt[f'FG_rel_gain_end'] >= -rel_cut)]
filt = filt.loc[((filt[f'weightsA_end'] >= weight_lim[0]) & (filt[f'weightsA_end'] <= weight_lim[1])) &
                    ((filt[f'weightsB_end'] >= weight_lim[0]) & (filt[f'weightsB_end'] <= weight_lim[1]))]
filt = filt.dropna(axis=0, subset='r')
filt = filt.loc[filt.r_end >= r_cut]

filt_fh = filt.loc[filt.dyn_kind=='fh']
stat_dict = weight_summary_histograms_(filt_fh, bar=True, stat_plot='median', suffix='_start', big_title='fh')

filt_hf = filt.loc[filt.dyn_kind=='hf']
stat_dict = weight_summary_histograms_(filt_hf, bar=True, stat_plot='median', suffix='_start', big_title='hf')


def weight_summary_histograms_(filt, bar=True, stat_plot='median', secondary='PEG', suffix='', big_title=''):
    '''2023_09_22. Plots the same as this function without manuscript on it, but it plots just what I want to show
    for the figure, whcih is A1 hist, bar, rel gain, PEG bar, rel gain'''

    if secondary:
        f = plt.figure(figsize=(15, 6))
        hist = plt.subplot2grid((10, 34), (0, 0), rowspan=5, colspan=8)
        mean = plt.subplot2grid((10, 34), (0, 10), rowspan=5, colspan=2)
        relhist = plt.subplot2grid((10, 34), (0, 14), rowspan=5, colspan=7)
        meanpeg = plt.subplot2grid((10, 34), (0, 23), rowspan=5, colspan=2, sharey=mean)
        relhistpeg = plt.subplot2grid((10, 34), (0, 27), rowspan=5, colspan=7, sharey=relhist)
        ax = [hist, mean, relhist, meanpeg, relhistpeg]
        to_plot = filt.loc[filt.area=='A1']
        areas = ['A1', secondary]

    else:
        f = plt.figure(figsize=(15, 6))
        hist = plt.subplot2grid((10, 34), (0, 0), rowspan=5, colspan=8)
        mean = plt.subplot2grid((10, 34), (0, 10), rowspan=5, colspan=2)
        relhist = plt.subplot2grid((10, 34), (0, 14), rowspan=5, colspan=7)
        ax = [hist, mean, relhist]
        to_plot = filt.loc[filt.area=='AC']
        areas = ['AC']

    edges = np.arange(-0.3, 1.5, .05)
    na, xa = np.histogram(to_plot[f'weightsA{suffix}'], bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(to_plot[f'weightsB{suffix}'], bins=edges)
    nb = nb / nb.sum() * 100
    ax[0].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue', linewidth=2)
    ax[0].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen', linewidth=2)
    ax[0].legend(('BG', 'FG'), fontsize=14, prop=dict(weight='bold'), labelspacing=0.25)

    ax[0].set_ylabel('Percent of\nneuron/stimulus pairs', fontweight='bold', fontsize=12)
    ax[0].set_title(f"A1, BG+/FG+, n={len(to_plot)}", fontweight='bold', fontsize=12)
    ax[0].set_xlabel("Weight", fontweight='bold', fontsize=12)
    ax[0].tick_params(axis='both', which='major', labelsize=8)
    ymin, ymax = ax[0].get_ylim()

    stat_dict = {}
    for aaa, ar in enumerate(areas):
        if aaa==0:
            ee = 1
        else:
            ee = 3
        to_plot = filt.loc[filt.area==ar]

        if stat_plot=='mean':
            bg_m, bg_m = np.mean(to_plot[f'weightsA{suffix}']), np.mean(to_plot[f'weightsB{suffix}'])
            bg_se, fg_se = stats.sem(to_plot[f'weightsA{suffix}']), stats.sem(to_plot[f'weightsB{suffix}'])
            label = 'Mean'
        elif stat_plot=='median':
            bg_m, bg_se = ofig.jack_mean_err(to_plot[f'weightsA{suffix}'], do_median=True)
            fg_m, fg_se = ofig.jack_mean_err(to_plot[f'weightsB{suffix}'], do_median=True)
            label = 'Median'

        ttest1 = stats.ttest_ind(to_plot[f'weightsA{suffix}'], to_plot[f'weightsB{suffix}'])
        ttest2 = stats.wilcoxon(to_plot[f'weightsA{suffix}'], to_plot[f'weightsB{suffix}'])
        stat_dict[f'{ar}_bar'] = ttest2.pvalue
        print(ttest2.pvalue)

        if bar:
            ax[ee].bar("BG", bg_m, yerr=bg_se, color='deepskyblue')
            ax[ee].bar("FG", fg_m, yerr=fg_se, color='yellowgreen')

        else:
            ax[ee].bar("BG", bg_m, yerr=bg_se, color='white')
            ax[ee].bar("FG", fg_m, yerr=fg_se, color='white')

            ax[ee].scatter(x=['BG', 'FG'], y=[bg_m, fg_m], color=['deepskyblue', 'yellowgreen'])
            ax[ee].errorbar(x=['BG', 'FG'], y=[bg_m, fg_m], yerr=[bg_se, fg_se], ls='none')#, color=['deepskyblue', 'yellowgreen'])

        ax[ee].set_ylabel(f'{label} Weight', fontweight='bold', fontsize=12)
        ax[ee].set_xticklabels(['BG','FG'], fontsize=10, fontweight='bold')
        ax[ee].tick_params(axis='y', which='major', labelsize=8)
        if ttest2.pvalue < 0.001:
            title = 'p<0.001'
        else:
            title = f"{ttest2.pvalue:.3f}"
        ax[ee].set_title(f"BG: {np.around(bg_m,2)}, FG: {np.around(fg_m,2)}\n{title}", fontsize=10)


        rel_weight = (to_plot[f'weightsB{suffix}'] - to_plot[f'weightsA{suffix}']) / \
                     (to_plot[f'weightsB{suffix}'] + to_plot[f'weightsA{suffix}'])
        supps = [cc for cc in rel_weight if cc < 0]
        percent_supp = np.around((len(supps) / len(rel_weight)) * 100, 1)
        # Filter dataframe to get rid of the couple with super weird, big or small weights
        rel = rel_weight.loc[rel_weight <= 2.5]
        rel = rel.loc[rel >= -2.5]

        ttt = stats.ttest_1samp(rel, 0)
        stat_dict[f'{ar}_hist'] = ttt.pvalue
        print(f"RG is {ttt.pvalue} relative to 0.")

        sups = [cc for cc in rel if cc < 0]
        enhs = [cc for cc in rel if cc >= 0]

        sup_edges = np.arange(-2.4, 0.1, .1)
        enh_edges = np.arange(0, 2.5, .1)
        na, xa = np.histogram(sups, bins=sup_edges)
        nb, xb = np.histogram(enhs, bins=enh_edges)
        aa = na / (na.sum() + nb.sum()) * 100
        bb = nb / (na.sum() + nb.sum()) * 100

        ax[ee+1].hist(xa[:-1], xa, weights=aa, histtype='step', color='tomato', fill=True)
        ax[ee+1].hist(xb[:-1], xb, weights=bb, histtype='step', color='dodgerblue', fill=True)

        ax[ee+1].legend(('FG Suppressed', 'FG Enhanced'), fontsize=14, prop=dict(weight='bold'), labelspacing=0.25)
        ax[ee+1].set_ylabel('Percent of\nneuron/stimulus pairs', fontweight='bold', fontsize=12)
        ax[ee+1].set_xlabel("Relative Gain (RG)", fontweight='bold', fontsize=12)
        ax[ee+1].set_title(f"% suppressed: {percent_supp}", fontsize=10)
        ax[ee+1].set_xlim(-1.75,1.75)

    f.suptitle(big_title, fontweight='bold', fontsize=12)
    f.tight_layout()
    return stat_dict


weight_df['site'] = [dd[:6] for dd in weight_df['cellid']]

weight_df = weight_df.loc[(weight_df.area=='A1')] #| (weight_df.area=='PEG')]
weight_df = weight_df.loc[weight_df.snr==0]
weight_df = weight_df.loc[((weight_df.fc==1) & (weight_df.bc==1))]# | ((weight_df.fc==2) & (weight_df.bc==2))]
snr_thresh = 0.08
weight_dff = weight_df.loc[(weight_df.bg_snr_active>=snr_thresh) & (weight_df.fg_snr_active>=snr_thresh)
                           & (weight_df.bg_snr_passive>=snr_thresh) & (weight_df.fg_snr_passive>=snr_thresh)]

weight_dff = weight_df.loc[(weight_df.bg_snr_passive>=snr_thresh) & (weight_df.fg_snr_passive>=snr_thresh)]
weight_dff = weight_df.loc[(weight_df.bg_snr_active>=snr_thresh) & (weight_df.fg_snr_active>=snr_thresh)]


r_thresh = 0.4
weight_dff = weight_dff.loc[(weight_dff.r_active>=r_thresh) & (weight_dff.r_passive>=r_thresh)]

weight_dff = weight_dff.loc[weight_dff.r_passive>=r_thresh]
weight_dff = weight_dff.loc[weight_dff.r_active>=r_thresh]

cellepopairs = [(cid, epo) for cid, epo in zip(weight_dff['cellid'], weight_dff['fgbg'])]



stat_dict = ofig.weight_summary_histograms_manuscript(weight_dff, bar=True, stat_plot='median')

fig, ax = plt.subplots(1, 2, figsize=(10,4))
width = 0.2
# ax.bar(1-width, weight_dff.weightsA_passive, width=0.4, color='deepskyblue')
# ax.bar(1+width, weight_dff.weightsB_passive, width=0.4, color='yellowgreen')

suff = ['_passive', '_active']
tts = []
for cnt, ss in enumerate(suff):
    bg_m, bg_se = ofig.jack_mean_err(weight_dff[f'weightsA{ss}'], do_median=True)
    fg_m, fg_se = ofig.jack_mean_err(weight_dff[f'weightsB{ss}'], do_median=True)
    label = 'Median'
    ax[0].bar(cnt-width, bg_m, yerr=bg_se, width=width*2, color='deepskyblue')
    ax[0].bar(cnt+width, fg_m, yerr=fg_se, width=width*2, color='yellowgreen')
    ttest1 = np.around(stats.wilcoxon(weight_dff[f'weightsA{ss}'], weight_dff[f'weightsB{ss}']).pvalue, 5)
    tts.append(ttest1)
ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['Passive', f'Active\nn={len(weight_dff)}'])
ax[0].set_ylabel('Median Weight', fontweight='bold', fontsize=10)
ax[0].set_title(f'passive: p={tts[0]}, active: p={tts[1]}')

ax[1].barh(y=cnt+width, width=weight_dff.FG_rel_gain_passive.mean(), color='purple',
        linestyle='None', height=width*2, label=f'Passive, n={len(weight_dff)}')
ax[1].barh(y=cnt-width, width=weight_dff.FG_rel_gain_active.mean(), color='orange',
        linestyle='None', height=width*2, label=f'Active')
ax[1].errorbar(y=cnt+width, x=weight_dff.FG_rel_gain_passive.mean(), elinewidth=2, capsize=4,
            xerr=weight_dff.FG_rel_gain_passive.sem(), color='black', linestyle='None', yerr=None)
ax[1].errorbar(y=cnt-width, x=weight_dff.FG_rel_gain_active.mean(), elinewidth=2, capsize=4,
            xerr=weight_dff.FG_rel_gain_active.sem(), color='black', linestyle='None', yerr=None)
ttest2 = stats.wilcoxon(weight_dff.FG_rel_gain_passive, weight_dff.FG_rel_gain_active)

ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(True)
ax[1].set_yticks([cnt+width, cnt-width])
ax[1].set_yticklabels(['Passive', f"Active\nn={len(weight_dff)}"])
ax[1].set_xlabel('Relative Gain', fontweight='bold', fontsize=10)
ax[1].set_title(f'p={np.around(ttest2.pvalue,5)}')



ofig.all_filter_stats(weight_df, xcol='bg_snr_passive', ycol='fg_snr_passive', snr_thresh=0.12,
                 fr_thresh=0.0001, xx='resp', supp=False)
ofig.all_filter_stats(weight_df, xcol='bg_snr_active', ycol='fg_snr_active', snr_thresh=0.12,
                 fr_thresh=0.0001, xx='resp', supp=False)

big_stat_plot_active_passive(weight_df, snr_thresh=0.08, r_thresh=0.4, strict_lims=False)


def big_stat_plot_active_passive(dff, snr_thresh=0.12, r_thresh=0.4, increment=0.2,
                                 suffs=['passive', 'active'], strict_lims=True):
    '''2023_09_28. This is a summary of ofig.snr_scatter(), ofig.r_weight_comp_distribution(), and
    ofig.weights_supp_comp() for the manuscript.'''

    xcol, ycol = 'bg_snr', 'fg_snr'
    f = plt.figure(figsize=(16, 8))
    a1snr = plt.subplot2grid((10, 31), (0, 0), rowspan=5, colspan=5)
    a1r = plt.subplot2grid((10, 31), (0, 6), rowspan=5, colspan=6)
    a1rsum = plt.subplot2grid((10, 31), (0, 13), rowspan=5, colspan=2)

    pegsnr = plt.subplot2grid((10, 31), (6, 0), rowspan=5, colspan=5, sharex=a1snr, sharey=a1snr)
    pegr = plt.subplot2grid((10, 31), (6, 6), rowspan=5, colspan=6)#, sharex=a1r, sharey=a1r)
    pegrsum = plt.subplot2grid((10, 31), (6, 13), rowspan=5, colspan=2)

    ws = plt.subplot2grid((10, 31), (3, 18), rowspan=5, colspan=6)
    rg = plt.subplot2grid((10, 31), (3, 25), rowspan=5, colspan=6)#, sharex=a1r, sharey=a1r)

    ax = [a1snr, a1r, a1rsum, pegsnr, pegr, pegrsum, ws, rg]
    aa=0
    keeps, cross_good = [], []
    for AR in suffs:
        suffy = '_' + AR

        # to_scatter = dff.iloc[::3, :]
        to_scatter = dff

        ax[aa].scatter(x=to_scatter[f'{xcol}{suffy}'], y=to_scatter[f'{ycol}{suffy}'], color='dimgrey', s=1)

        ax[aa].set_xlabel('BG snr', fontsize=8, fontweight='bold')
        ax[aa].set_ylabel('FG snr', fontsize=8, fontweight='bold')
        xmin, xmax = ax[aa].get_xlim()
        ymin, ymax = ax[aa].get_ylim()
        ax[aa].vlines([snr_thresh], 0, ymax, colors='black', linestyles=':', lw=1)
        ax[aa].hlines([snr_thresh], 0, xmax, colors='black', linestyles=':', lw=1)
        ax[aa].set_xlim(0, xmax), ax[aa].set_ylim(0, ymax)
        size = len(dff)
        snr3 = len(dff.loc[(dff[xcol+suffy] >= snr_thresh) & (dff[ycol+suffy] >= snr_thresh)]) / size * 100
        snr5 = len(dff.loc[(dff[xcol+suffy] < snr_thresh) & (dff[ycol+suffy] < snr_thresh)]) / size * 100
        snr2 = len(dff.loc[(dff[xcol+suffy] < snr_thresh) & (dff[ycol+suffy] >= snr_thresh)]) / size * 100
        snr6 = len(dff.loc[(dff[xcol+suffy] >= snr_thresh) & (dff[ycol+suffy] < snr_thresh)]) / size * 100
        ax[aa].set_title(f'{dff.area.unique()[0]}: thresh={snr_thresh}, n={len(dff)}\n'
                         f'Above: {np.around(snr3,1)}%, Below: {np.around(snr5,1)}%\n'
                        f'FG Only: {np.around(snr2,1)}%, BG only: {np.around(snr6,2)}%', fontsize=8, fontweight='bold')
        ax[aa].set_aspect('equal')

        if strict_lims==True:
            suffies = ['_' + dd for dd in suffs]
            snr_filt = dff.loc[(dff[xcol+suffies[0]] >= snr_thresh) & (dff[ycol+suffies[0]] >= snr_thresh)
                                   & (dff[xcol+suffies[1]] >= snr_thresh) & (
                                               dff[ycol+suffies[1]] >= snr_thresh)]
            # ax[aa].scatter(x=snr_filt[f'{xcol}{suffy}'], y=snr_filt[f'{ycol}{suffy}'], color='black', s=1)
            keeps.append(snr_filt)
            ax[2].set_title("strict lims yup")

        else:
            snr_filt = dff.loc[(dff[xcol+suffy] >= snr_thresh) & (dff[ycol+suffy] >= snr_thresh)]
            snr_filt['cellepo'] = snr_filt['cellid'] + ',' + snr_filt['fgbg']
            keeps.append(snr_filt)
            ax[2].set_title("strict lims nope")
            cross_good.append(snr_filt)

        aa+=1

        incs = np.arange(0, 1, increment)
        plots = len(incs)

        inc_lims = np.append(incs, 1)
        off = 0.2
        totals, total, maxs, stat_dict = {}, 0, [], {}
        for cnt, inc in enumerate(list(incs)):
            r_df = snr_filt.dropna(axis=0, subset=f'r{suffy}')
            r_df = r_df.loc[(r_df[f'r{suffy}'] >= inc) & (r_df[f'r{suffy}'] < inc_lims[cnt+1])]

            BG1, FG1 = np.mean(r_df[f'weightsA{suffy}']), np.mean(r_df[f'weightsB{suffy}'])
            BG1sem, FG1sem = stats.sem(r_df[f'weightsA{suffy}']), stats.sem(r_df[f'weightsB{suffy}'])
            ttest1 = stats.ttest_ind(r_df[f'weightsA{suffy}'], r_df[f'weightsB{suffy}'])

            ax[aa].bar(cnt-off, BG1, yerr=BG1sem, color='deepskyblue', width=off*2)
            ax[aa].bar(cnt+off, FG1, yerr=FG1sem, color='yellowgreen', width=off*2)

            bin_name = f"r={np.around(inc, 1)}-{np.around(inc_lims[cnt + 1], 1)}"
            print(bin_name)
            totals[bin_name] = len(r_df)
            total += len(r_df)
            stat_dict[bin_name] = ttest1.pvalue

        ax[aa].set_xticks(np.arange(0,len(incs)))
        ax[aa].set_xticklabels([f'{dd}\nn={ee}\n%={np.around((ee/total)*100, 1)}\np={np.around(stat_dict[dd],4)}'
                               for (dd, ee) in totals.items()], fontsize=6)
        ax[aa].set_aspect('auto')
        ax[aa].set_ylabel('Mean Weight', fontweight='bold', fontsize=10)
        ax[aa].set_title(AR, fontweight='bold', fontsize=13)

        aa+=1

        from matplotlib import cm
        greys = cm.get_cmap('inferno', 12)
        cols = greys(np.linspace(0, 0.9, len(incs))).tolist()
        # cols.reverse()

        percents = [(pp/total)*100 for pp in totals.values()]
        names = [lbl for lbl in list(totals.keys())]

        for cc in range(len(names)):
            bottom = np.sum(percents[cc+1:])
            ax[aa].bar('total', height=percents[cc], bottom=bottom, color=cols[cc],
                         width=1, label=names[cc])#, edgecolor='white')

        ax[aa].legend(names, bbox_to_anchor=(0.8,1.025), loc="upper left")
        ax[aa].set_ylabel('Percent', fontweight='bold', fontsize=10)
        ax[aa].set_xticks([])
        ax[aa].set_xlim(-1,1)

        aa+=1

    if strict_lims==True:
        r_dfs = [ddff.loc[(ddff[f'r_{suffs[0]}'] >= r_thresh) & (ddff[f'r_{suffs[1]}'] >= r_thresh)] for ddff in keeps]
    else:
        r_dfs = [ddff.loc[ddff[f'r_{suffs[su]}'] >= r_thresh] for (ddff, su) in zip(keeps,range(len(suffs)))]
        ax[3].scatter(x=r_dfs[0][f'{xcol}_{suffs[0]}'], y=r_dfs[0][f'{ycol}_{suffs[1]}'], color='black', s=1)
        ax[0].scatter(x=r_dfs[1][f'{xcol}_{suffs[0]}'], y=r_dfs[1][f'{ycol}_{suffs[1]}'], color='black', s=1)

    width = 0.2

    suff = ['_' + se for se in suffs]
    tts = []
    for cntt, ss in enumerate(suff):
        to_plot = r_dfs[cntt]
        bg_m, bg_se = ofig.jack_mean_err(to_plot[f'weightsA{ss}'], do_median=True)
        fg_m, fg_se = ofig.jack_mean_err(to_plot[f'weightsB{ss}'], do_median=True)

        ax[aa].bar(cntt - width, bg_m, yerr=bg_se, width=width * 2, color='deepskyblue')
        ax[aa].bar(cntt + width, fg_m, yerr=fg_se, width=width * 2, color='yellowgreen')
        ttest1 = np.around(stats.wilcoxon(to_plot[f'weightsA{ss}'], to_plot[f'weightsB{ss}']).pvalue, 5)
        tts.append(ttest1)
    ax[aa].set_xticks([0, 1])
    ax[aa].set_xticklabels([f'{suffs[0]}\nn={len(r_dfs[0])}', f'{suffs[1]}\nn={len(r_dfs[1])}'])
    ax[aa].set_ylabel('Median Weight', fontweight='bold', fontsize=10)
    if strict_lims:
        ax[aa].set_title(f'r_lim>={r_thresh}\n{suffs[0]}: p={tts[0]}, {suffs[1]}: p={tts[1]}', fontweight='bold')
    else:
        list1, list2 = keeps[0]['cellepo'].to_list(), keeps[1]['cellepo'].to_list()
        overlap_snr = [element for element in list1 if element in list2]
        list1, list2 = r_dfs[0]['cellepo'].to_list(), r_dfs[1]['cellepo'].to_list()
        overlap_r = [element for element in list1 if element in list2]

        ax[aa].set_title(f'r_lim>={r_thresh}\n{suffs[0]}: p={tts[0]}, {suffs[1]}: p={tts[1]}\nCommon instances snr: n={len(overlap_snr)}\n'
                         f'Common instances r: n={len(overlap_r)}', fontweight='bold')

    aa+=1

    ax[aa].barh(y=cnt + width, width=r_dfs[0][f'FG_rel_gain_{suffs[0]}'].mean(), color='orange',
               linestyle='None', height=width * 2, label=f'{suffs[0]}, n={len(r_dfs[0])}')
    ax[aa].barh(y=cnt - width, width=r_dfs[1][f'FG_rel_gain_{suffs[1]}'].mean(), color='purple',
               linestyle='None', height=width * 2, label=f'{suffs[1]}, n={len(r_dfs[1])}')
    ax[aa].errorbar(y=cnt + width, x=r_dfs[0][f'FG_rel_gain_{suffs[0]}'].mean(), elinewidth=2, capsize=4,
                   xerr=r_dfs[0][f'FG_rel_gain_{suffs[0]}'].sem(), color='black', linestyle='None', yerr=None)
    ax[aa].errorbar(y=cnt - width, x=r_dfs[1][f'FG_rel_gain_{suffs[1]}'].mean(), elinewidth=2, capsize=4,
                   xerr=r_dfs[1][f'FG_rel_gain_{suffs[1]}'].sem(), color='black', linestyle='None', yerr=None)

    if strict_lims:
        ttest2 = stats.wilcoxon(r_dfs[0][f'FG_rel_gain_{suffs[0]}'], r_dfs[1][f'FG_rel_gain_{suffs[1]}'], nan_policy='omit')
    else:
        ttest2 = stats.mannwhitneyu(r_dfs[0][f'FG_rel_gain_{suffs[0]}'], r_dfs[1][f'FG_rel_gain_{suffs[1]}'], nan_policy='omit')

    ax[aa].yaxis.tick_right()
    ax[aa].yaxis.set_label_position("right")
    ax[aa].spines['left'].set_visible(False), ax[aa].spines['right'].set_visible(True)
    ax[aa].set_yticks([cnt + width, cnt - width])
    ax[aa].set_yticklabels([f'{suffs[0]}\nn={len(r_dfs[0])}', f"{suffs[1]}\nn={len(r_dfs[1])}"])
    ax[aa].set_xlabel('Relative Gain', fontweight='bold', fontsize=10)
    ax[aa].set_title(f'p={np.around(ttest2.pvalue, 5)}')

    f.tight_layout()



# This does what all those bs filters I clicked around and ran everytime I wanted to start something
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)

filt = ohel.get_olp_filter(weight_df, kind='sounds', metric=True)


# 2023 WIP bignat spectrogram
path = '/auto/data/sounds/BigNat/v2/seq0354.wav'
fig, ax = plt.subplots(1, 1, figsize=(12, 2))
specs = []
sfs, W = wavfile.read(path)
spec = gtgram(W, sfs, 0.02, 0.01, 48, 0, 12000)
ax.imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
              cmap='gray_r')
ax.spines['top'].set_visible(True), ax.spines['right'].set_visible(True)
xx = list(np.arange(0, spec.shape[1], 200))
xxs = [int(dd) for dd in list(np.arange(0, spec.shape[1], 200) / 100)]
ax.set_xticks(xx)
ax.set_xticklabels(xxs)
ax.set_xlabel('Time (s)', fontweight='bold', fontsize=10)
fig.tight_layout()
ax.set_yticks([])
ax.set_ylabel('Frequency (Hz)', fontweight='bold', fontsize=10)





## 2023_01_03. This goes after I run the job and have a df.
#2024_02_08 turn me into a function
saved_paths = glob.glob(f"/auto/users/hamersky/cache_full/*")
saved_paths = glob.glob(f"/auto/users/hamersky/cache_full_behavior/*")
saved_paths = glob.glob(f"/auto/users/hamersky/cache_behavior/*")
saved_paths = glob.glob(f"/auto/users/hamersky/cache_OLP_LDO/*")


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
    f'/auto/users/hamersky/olp_analysis/{date.today()}_batch' \
    f'{weight_df0.batch.unique()[0]}_LEMON_OLP_standard'

jl.dump(weight_df0, OLP_partialweights_db_path)






full_df = ohel.merge_dynamic_error(weight_df, dynamic_path='cache_dyn_final', SNR=0)
filt = full_df

filtt = filt.loc[(filt.bg_snr >= 0.12) & (filt.fg_snr >= 0.12)]

dyn, ar = 'hf', 'A1'

filtt = filtt.dropna(axis=0, subset='r')
filtt = filtt.loc[filtt.r >= 0.4]
dyn_df = filtt.loc[filtt.dyn_kind == dyn]
area_df = dyn_df.loc[dyn_df.area == ar]

area_df = area_df.loc[(area_df.bg_snr >= 0.3) & (area_df.fg_snr >= 0.3)]
area_df = area_df.loc[(area_df.bg_FR >= 0.3) & (area_df.fg_FR >= 0.3)]
ofig.plot_dynamic_row_psth(area_df, 3, dyn, smooth=True, sigma=1)






full_df = ohel.merge_dynamic_error(weight_df, dynamic_path='cache_dyn_no_spont', SNR=0)
ofig.example_dynamic_psth(full_df, 'PRN022a-211-2', 'Tuning', 'KitWhine', dyn='fh', smooth=True, sigma=1)
ofig.example_dynamic_psth(full_df, 'PRN015a-315-1', 'Waves', 'Gobble', dyn='fh', smooth=True, sigma=1)
full_df = ohel.merge_dynamic_error(weight_df, dynamic_path='cache_dyn_no_spont', SNR=0)
ofig.example_dynamic_psth(full_df, 'PRN017a-319-1', 'Stream', 'KitHigh', dyn='hf', smooth=True, sigma=1)
ofig.example_dynamic_psth(full_df, 'TNC056a-241-1', 'Blender', 'Dice', dyn='hf', smooth=True, sigma=1)
ofig.example_dynamic_psth(full_df, 'PRN017a-319-1', 'Stream', 'ManA', dyn='hf', smooth=True, sigma=1)









###This is where the figure stuff was for manuscript














weight_df = weight_df.loc[weight_df.synth_kind=='N']
ofig.plot_all_weight_comparisons(filt, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True)

plot_weight_prediction_comparisons(filt, fr_thresh=0.03, r_thresh=0.6, strict_r=True, summary=True)





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



ofig.sound_metric_scatter(weight_df0, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                          area='A1', threshold=0.03, synth_kind='N', r_cut=0.6, jitter=[0.005,0.2,0.03])


ofig.sound_metric_scatter(sound_df, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                          area='A1', threshold=0.03, synth_kind='N', r_cut=0.6)



# # Tells you what sounds are below certain thresholds. And plots the sounds with their metrics.
bad_dict = ohel.plot_sound_stats(sound_df, ['Fcorr', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power', 'rel_gain'],
                                 labels=['Frequency Non-stationarity', 'Temporal Non-stationarity', 'Bandwidth (octaves)',
                                         'Max Power', 'RMS Power', 'Relative Gain'],
                                 lines={'RMS_power': 0.95, 'max_power': 0.3}, synth_kind='A')
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
                                r_thresh=0.6, quad_return=3, bin_kind='11', weight_lims=[-1.5, 2.5], area='A1')

weight_df0 = ohel.filter_synth_df_by(weight_df, use='C', suffixes=['', '_start', '_end'], fr_thresh=0.03, \
                                r_thresh=0.6, quad_return=3, bin_kind='11', weight_lims=[-1.5, 2.5], area='A1')

stat = osyn.synthetic_summary_weight_multi_bar(weight_df0, suffixes=['', '_start', '_end'],
                                               show=['N','M','S','T','C'], figsize=(12, 4))



weight_df0 = ohel.filter_weight_df(weight_df, suffixes=[''], fr_thresh=0.03, r_thresh=0.6, quad_return=3,
                                   bin_kind='11', synth_kind=None, area='A1', weight_lims=[-1, 2],
                                   bads=True, bad_filt={'RMS_power': 0.95, 'max_power': 0.3})






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


ofig.psths_with_specs_partial_fit(filt_hf, 'ARM013b-13-1', 'Insect_Buzz', 'Gobble_High', sigma=1, error=False, synth_kind='A')


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
    if row.dyn_kind=='hf':
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
ofig.weights_supp_comp(weight_df, quads=3, thresh=0.03, r_cut=0.6)

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
ofig.plot_single_relative_gain_hist(filt, 0.03, synth_kind=None)

# Plots the example that I piece together to make the linear model example
weight_df0 = ofit.OLP_fit_weights(batch=333, cells=['TBR012a-31-1'], sound_stats=False)
ofig.plot_linear_model_pieces_helper(weight_df0, cellid='TBR012a-31-1', bg='Wind', fg='Chirp')

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

saved_paths = glob.glob(f"/auto/users/hamersky/cache_snr/*")

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
    f'/auto/users/hamersky/olp_analysis/{date.today()}_batch{weight_df0.batch.unique()[0]}_{weight_df0.fit_segment.unique()[0]}_metric'  # weight + corr

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