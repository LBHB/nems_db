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

# Load your different, updated dataframes -- as of 2024_02_08, I don't think these are ever used anymore
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

# The thing I was working on in January with fit
path = '/auto/users/hamersky/olp_analysis/2023-01-12_Batch341_0-500_FULL'

#marms
path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch328_0-500_marm'
weight_df = jl.load(path)

# 2023_05_02. Starting with Prince data too and new df structure
# path = '/auto/users/hamersky/olp_analysis/2023-05-10_batch344_0-500_metrics' # Full one with updated PRNB layers
# path = '/auto/users/hamersky/olp_analysis/2023-05-17_batch344_0-500_metrics' #full one with PRNB layers and paths
# path = '/auto/users/hamersky/olp_analysis/2023-07-20_batch344_0-500_metrics' # full with new FR snr metric
# path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch344_0-500_metric'
# path = '/auto/users/hamersky/olp_analysis/2023-09-15_batch344_0-500_final'
# path = '/auto/users/hamersky/olp_analysis/2023-09-21_batch344_0-500_final'

#This is, indeed, the final one. 2024_02_08
#Generated using the enqueue.py in this /projects/olp, for batch 344
path = '/auto/users/hamersky/olp_analysis/2023-09-22_batch344_0-500_final'
weight_df = jl.load(path)


#####################################
####### Figure 1 ####################
#### Introduction and example psths #

#B Example schematic spectrograms, rest in inkscape
ofig.intro_figure_spectrograms_colors()

# C. PSTHs
ofig.psths_with_specs(weight_df, 'CLT008a-046-2', 'Wind', 'Geese', batch=340, bin_kind='11', synth_kind='A',
                         sigma=1, error=False, title='A1 046-2')
ofig.psths_with_specs(weight_df, 'CLT012a-052-1', 'Bees', 'Bugle', batch=340, bin_kind='11', synth_kind='A',
                         sigma=1, error=False, title='A1 052-1')
ofig.psths_with_specs(weight_df, 'CLT052d-018-1', 'Wind', 'Geese', batch=340, bin_kind='11', synth_kind='A',
                         sigma=1, error=False, title='PEG 018-1')


# D. Heatmap example of suppression. Some good examples.

# These are the ones used below, others include:
# site, bg, fg, cellid = 'CLT009a', 'Bulldozer', 'FightSqueak', None
# site, bg, fg, cellid = 'CLT012a', 'Bees', 'Bugle', 'CLT012a-052-1'
# PEG supplemental in some versions
# site, bg, fg, cellid = 'CLT052d', 'Wind', 'Geese', 'CLT052d-018-1'

ofig.response_heatmaps_comparison(weight_df, site='CLT008a', bg='Wind', fg='Geese', cellid='CLT008a-046-2',
                                     batch=340, bin_kind='11', synth_kind='A', sigma=3, sort=True,
                             example=True, lin_sum=True, positive_only=False)


####################################################
######## Figure 2 ##################################
#### Intro to the model and summary of the weights #

#2A
# Model - something stopped working with this code, but it just makes pieces that manually get picture edited
# into a way that makes sense
path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch328_0-500_marm'
weight_df = jl.load(path)
ofig.plot_linear_model_pieces_helper(weight_df, cellid='TBR012a-31-1', bg='Wind', fg='Chirp')

#2C weights summary final version
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
stat_dict = ofig.weight_summary_histograms_manuscript(filt, bar=True, stat_plot='median')

#2B weights summary A1 and PEG
ofig.weight_summary_histograms(filt, threshold=None, snr_threshold=0.12, r_cut=0.4, area='A1', rel_cut=2.5,
                               bar=True, stat_plot='median')
ofig.weight_summary_histograms(filt, threshold=None, snr_threshold=0.12, r_cut=0.4, area='PEG', rel_cut=2.5,
                               bar=True, stat_plot='median')


##############################################
####### Figure 3 #############################
#### Dynamic experiments, intro and analysis #
#3B
full_df = ohel.merge_dynamic_error(weight_df, dynamic_path='cache_dyn_final', SNR=0)
ofig.example_dynamic_psth(full_df, 'PRN022a-211-2', 'Tuning', 'KitWhine', dyn='fh', smooth=True, sigma=1)
# ofig.example_dynamic_psth(full_df, 'PRN015a-315-1', 'Waves', 'Gobble', dyn='fh', smooth=True, sigma=1)

#3C
fh = ofig.plot_dynamic_errors(full_df, dyn_kind='fh', snr_threshold=0.12)

#3D
full_df = ohel.merge_dynamic_error(weight_df, dynamic_path='cache_dyn_no_spont', SNR=0)
ofig.example_dynamic_psth(full_df, 'PRN017a-319-1', 'Stream', 'KitHigh', dyn='hf', smooth=True, sigma=1)
# ofig.example_dynamic_psth(full_df, 'TNC056a-241-1', 'Blender', 'Dice', dyn='hf', smooth=True, sigma=1)
# ofig.example_dynamic_psth(full_df, 'PRN017a-319-1', 'Stream', 'ManA', dyn='hf', smooth=True, sigma=1)

#3E
hf = ofig.plot_dynamic_errors(full_df, dyn_kind='hf', snr_threshold=0.12)


###################################
####### Figure 4 ##################
#### Half fit histograms and plot #
#3A/B
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
stt = ofig.plot_all_weight_comparisons(filt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
                                 weight_lim=[-0.5, 2], summary=True, sep_hemi=False, sort_category=None,
                                 stat_plot='median', flanks=False, stat_kind='paired', uniform_animal=False)

stt = ofig.plot_all_weight_comparisons_RG(filt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
                                 weight_lim=[-0.5, 2], summary=True, sep_hemi=False, sort_category=None,
                                 uniform_animal=True)

# For the figure if only I could find a good example 2022_11_01 - Ultimately useful for presentations
# ofig.psths_with_specs_partial_fit(weight_df, 'CLT047c-012-1', 'Bees', 'Gobble', sigma=1, error=False)
# ofig.psths_with_specs_partial_fit(weight_df, 'CLT040c-051-1', 'Tuning', 'ManA', sigma=1, error=False)
# ofig.psths_with_specs_partial_fit(weight_df, 'CLT052d-035-2', 'Bees', 'Chickens', sigma=1, error=False)
ofig.psths_with_specs_partial_fit(weight_df, 'CLT052d-018-1', 'Wind', 'Geese', sigma=1, error=False)
ofig.psths_with_specs_partial_fit(weight_df, 'CLT012a-052-1', 'Bees', 'Bugle', sigma=1, error=False, synth_kind='A')


############################################
####### Figure 5 ###########################
#### Sound stats relating to relative gain #
# # Figure 5A sound stat
filt = ohel.get_olp_filter(weight_df, kind='sounds', metric=True)
sound_df = ohel.get_sound_statistics_from_df(filt, percent_lims=[15,85], append=False)
sound_df = ohel.label_vocalization(sound_df, species='sounds')
ohel.plot_sound_stats(sound_df, ['Fcorr', 'Tstationary', 'bandwidth'],
                 labels=['Frequency Correlation', 'Temporal Non-stationarity', 'Bandwidth (octaves)'],
                 synth_kind=None, sort=True)

# 5C sound stats with rel gain
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)

_, stt = ofig.sound_metric_scatter(filt, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'], suffix='',
                          area='A1', metric_filter=2.5, snr_threshold=0.12, threshold=None, synth_kind=None,
                          r_cut=0.4, jitter=None, mean=True, vocalization=True)
_, stt = ofig.sound_metric_scatter(filt, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'], suffix='',
                          area='PEG', metric_filter=2.5, snr_threshold=0.12, threshold=None, synth_kind=None,
                          r_cut=0.4, jitter=None, mean=True, vocalization=True)

# ofig.sound_metric_scatter(sound_df, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
#                           ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'], suffix='',
#                           area='A1', threshold=0.03, synth_kind='N', r_cut=0.6, jitter=[0.005,0.2,0.03])

# 5C+ Spectral overlap
ofig.plot_spectral_overlap_scatter(filt, area='A1')
ofig.plot_spectral_overlap_scatter(filt, area='PEG')

# Helpful for diagnosing spectral overlap stuff
# overlap_info = ohel.get_spectral_overlap_stats_and_paths(filt, area=None)
# row = ohel.plot_spectral_overlap_specs(overlap_info, BG='Bees', FG='FightSqueak')
# row = ohel.plot_spectral_overlap_specs(overlap_info, BG='Jackhammer', FG='WomanA')

#5D Vocalizations
voc_masks = {'No': 'Non-vocalization', 'Other': 'Other\nVocalization', 'Yes': 'Ferret\nVocalization'}
filtt = filt.copy()
filtt['Vocalization'] = filtt['Vocalization'].map(voc_masks)
# Didn't use first and second half breakdown
# ofig.plot_all_weight_comparisons(filtt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
#                                  weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category="Vocalization",
#                                  stat_plot='median', flanks=False)
#or USED THIS
ofig.metric_weight_bars(filtt, snr_threshold=0.12, r_cut=0.4, area='A1', category='Vocalization')
ofig.metric_weight_bars(filtt, snr_threshold=0.12, r_cut=0.4, area='PEG', category='Vocalization')
#or didn't use this
# ofig.summary_relative_gain_all_areas(filtt, kind_show=['Ferret\nVocalization', 'Other\nVocalization', 'Non-vocalization'],
#                                      category='Vocalization',
#                                      mult_comp=1, statistic='independent')

#5E Regression
a1, peg = ofig.plot_big_sound_stat_regression(filt, xvar=['Fcorr', 'Tstationary', 'bandwidth', 'snr', 'spectral_overlap'],
                                    cat='Vocalization', omit='C(Vocalization)[T.3]')

#Trying multiple regression with shuffling
# a1, voc_label = ohel.run_sound_stats_reg(filt, r_cut=0.4, snr_threshold=0.12, suffix='', synth=None,
#               xs=['Fcorr', 'Tstationary', 'bandwidth', 'snr', 'spectral_overlap'],
#               category='Vocalization', area='A1', shuffle=True)
# peg, voc_label = ohel.run_sound_stats_reg(filt, r_cut=0.4, snr_threshold=0.12, suffix='', synth=None,
#               xs=['Fcorr', 'Tstationary', 'bandwidth', 'snr', 'spectral_overlap'],
#               category='Vocalization', area='PEG', shuffle=True)
# for aa in a1.keys():
#     print(f'{aa}, r={np.around(a1[aa].rsquared, 3)}')

##Other stats stuff probably not useful but maybe
# # Takes a spectrogram and makes side panels describing some metrics you can get from it
# ofig.spectrogram_stats_diagram('Jackhammer', 'BG')
# ofig.spectrogram_stats_diagram('Fight Squeak', 'FG')
#
# # Compares the sound stats of the first and second half of the sound
# ofig.sound_stats_half_compare(sound_df, suffixes=['_start', '_end'], metric='Tstationary', show='N')
# ofig.sound_stats_half_compare(sound_df, suffixes=['_start', '_end'], metric='Fstationary', show='N')
# ofig.sound_stats_half_compare(sound_df, suffixes=['_start', '_end'], metric='bandwidth', show='N')
#
# # 2023_05_19. Testing spectral correlation stuff.
# sound_df.FG.unique()
# sound_df.BG.unique()
#
# sn = 'Gobble'
# kind = 'FG'
# osyn.plot_cc_cuts(sound_df, sn, kind, percent_lims=[10,90], sk='N')
#
# # 2023_05_19. The big spectral correlation viewer.
# osyn.plot_spec_cc(sound_df, 'BG', percent_lims=[10,90], sk='N')
# osyn.plot_spec_cc(sound_df, 'FG', percent_lims=[10,90], sk='N')


########################################
####### Figure 6 #######################
#### Synthetic explanation and summary #

# 6A shows relative gain histograms across synthetic conditions
filt = ohel.get_olp_filter(weight_df, kind='synthetic', metric=False)

# filters our categories using the conditions given
df = ohel.filter_across_condition(filt, synth_show=['N','M','S','T','C'], filt_kind='synth_kind',
                                  snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=[''])

# Example spectrograms for the degradations plus general histograms
osyn.synthetic_relative_gain_comparisons_specs(df, 'Waterfall', 'WomanA', thresh=None, snr_threshold=0.12,
                                               synth_show=['N', 'M', 'S', 'T', 'C'],
                                               quads=3, r_cut=0.4, rel_cut=2.5, area='A1', figsize=(20,15))

osyn.synthetic_relative_gain_comparisons(df, snr_threshold=0.12, thresh=None, quads=3, area='PEG',
                                              synth_show=['M','S','T','C'], r_cut=0.4, rel_cut=2.5)

#6B/C Summary of above RG histograms
stt = ofig.summary_relative_gain_all_areas(df, kind_show=['N', 'M','S','T','C'], category='synth_kind', mult_comp=1,
                                           statistic='paired', group_by_area=True)
# stat_dict = osyn.synthetic_summary_relative_gain_all_areas(df, synth_show=['N', 'M','S','T','C'], mult_comp=5,
#                                                            group_by_area=True)

# 6D
# df = ohel.filter_across_synths(filt, synth_show=['N','M','S','T','C'], snr_threshold=0.12, r_cut=0.4,
#                                rel_cut=2.5, suffix=[''])
osyn.synthetic_sound_metric_scatters(df, ['Fcorr', 'Tstationary', 'bandwidth'], synth_show=['N', 'M', 'S', 'T', 'C'],
                          x_labels=['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                              suffix='')

# import statsmodels.formula.api as smf
# df_reg = df[['area', 'synth_kind', 'cellid', 'FG_rel_gain']]
# df_reg['area'] = df_reg['area'].map({'A1': 0, 'PEG': 1})
# df_reg['synth_kind'] = df_reg['synth_kind'].map({'N': 0, 'M': 1, 'S': 2, 'T': 3, 'C': 4})
# cellid_dict = {cc: cnt for cnt, cc in enumerate(df_reg.cellid.unique().tolist())}
# df_reg['cellid'] = df_reg['cellid'].map(cellid_dict)
# mod = smf.mixedlm("FG_rel_gain ~ C(synth_kind) + C(area)", df_reg, groups=df_reg['cellid'])
# # mod = smf.ols(formula="FG_rel_gain ~ C(synth_kind) + C(area) + C(cellid)", data=df_reg)
# est = mod.fit()

# Makes the synth scatters with a single area, BGs on top, FGs on bottom -- Probably not needed
# osyn.sound_metric_scatter_bgfg_sep(sound_df, ['Fcorr', 'Tstationary', 'bandwidth'], fr_thresh=0.03, r_cut=0.6,
#                           x_labels=['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'])
#6C old
# weight_df0 = ohel.filter_synth_df_by(weight_df, use='N', suffixes=[''], fr_thresh=0.03, \
#                                 r_thresh=0.4, quad_return=3, bin_kind='11', weight_lims=[-1.5, 2.5], area='A1')
#
# weight_df0 = ohel.filter_synth_df_by(filt, use='C', suffixes=['', '_start', '_end'], fr_thresh=0.03, \
#                                 r_thresh=0.4, quad_return=3, bin_kind='11', weight_lims=[-1.5, 2.5], area='PEG')
# a1_df0 = df.loc[df.area=='A1']
# peg_df0 = df.loc[df.area=='PEG']
# a1t = osyn.synthetic_summary_relative_gain_multi_bar(a1_df0, suffixes=['', '_start', '_end'])
# pegt = osyn.synthetic_summary_relative_gain_multi_bar(peg_df0, suffixes=['', '_start', '_end'])

# Maybe a supplement?
# osyn.sound_metric_scatter_combined_flanks(filt, ['Fcorr', 'Tstationary', 'bandwidth'], fr_thresh=0.03, r_cut=0.6,
#                           x_labels=['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'],
#                               suffix='_end')


######################################
####### Figure 7 #####################
#### Everything and the kitchen sink #
# 7A Marmoset
path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch328_0-500_marm'
weight_dff = jl.load(path)
filt = ohel.label_vocalization(weight_dff, species='marmoset')

# filt.loc[(filt['area'] == 'A1orR') | (filt['area'] == 'R'), 'area'] = 'A1'
# filt = filt.loc[filt['area'].isin(['ML', 'AL', 'A1'])]
# filt.loc[(filt['area'] == 'AL') | (filt['area'] == 'ML'), 'area'] = '2nd'
# filt.loc[(filt['area'] == 'AL?') | (filt['area'] == 'ML?'), 'area'] = '2nd'
# filt = filt.loc[filt['area'].isin(['A1', '2nd'])]
filt['area'] = 'AC'

# Don't keep the A1?/AL? labels, they're kinda small anyway
filt = filt.loc[filt.dyn_kind=='ff']

# sound_df = ohel.get_sound_statistics_from_df(filt, percent_lims=[15,85], append=False)
# bad_dict = ohel.plot_sound_stats(sound_df, ['Fcorr', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power'],
#                                  labels=['Frequency Non-stationarity', 'Temporal Non-stationarity', 'Bandwidth (octaves)',
#                                          'Max Power', 'RMS Power'], lines={'RMS_power': 0.95, 'max_power': 0.3}, synth_kind='A')
# bads = list(bad_dict['RMS_power'])
bads = ['Branch', 'CashRegister', 'Heels', 'Woodblock', 'Castinets', 'Dice'] #rms
# Just gets us around running that above function, this is the output.
filt = filt.loc[filt['BG'].apply(lambda x: x not in bads)]
filt = filt.loc[filt['FG'].apply(lambda x: x not in bads)]

quiet = filt.loc[filt.noisy!='Yes']
filt = ohel.df_filters(filt, snr_threshold=0.12, rel_cut=2.5, r_cut=0.4, weight_lim=[-0.5, 2])
stt = ofig.weight_summary_histograms_manuscript(filt, bar=True, stat_plot='median', secondary=None)
ofig.metric_weight_bars(quiet, snr_threshold=0.12, r_cut=0.4)

# None of this seems to be necessary anymore :'(
# area_summary_bars(filt, snr_threshold=0.12, r_cut=0.4, category='fg_noise')
# filt = quiet
# snr_threshold = 0.12
# filt = filt.loc[(filt.bg_snr >= snr_threshold) & (filt.fg_snr >= snr_threshold)]
# r_cut = 0.4
# filt = filt.dropna(axis=0, subset='r')
# filt = filt.loc[filt.r >= r_cut]
# rel_cut = 2.5
# filt = filt.loc[(filt['FG_rel_gain'] <= rel_cut) & (filt[f'FG_rel_gain'] >= -rel_cut)]
# filt = filt.loc[(filt['BG_rel_gain'] <= rel_cut) & (filt[f'BG_rel_gain'] >= -rel_cut)]
#
# area = 'A1' # of '2nd'
# ofig.sound_metric_scatter(filt, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
#                           ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'], suffix='',
#                           area=area, metric_filter=2.5, snr_threshold=0.12, threshold=None, synth_kind='A',
#                           r_cut=None, jitter=None, mean=True)
# filt['marmoset'] = 'marm'
# stat_dict = ofig.summary_relative_gain_all_areas(filt, kind_show=['marm'], category='marmoset', mult_comp=None,
#                                      statistic='paired', secondary_area_name='2nd')

# 7B - layers in
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
stat_dict = ofig.summary_relative_gain_all_areas(filt, kind_show=['13', '44', '56'], category='layer', mult_comp=1,
                                     statistic='independent', group_by_area=True)

# 7C - Binaural in ferrets
# summary for stephen
# filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
# filt['species'] = 'ferret'
# ofig.summary_relative_gain_all_areas(filt, kind_show=['ferret'], category='species', mult_comp=3, statistic='independent')
#
# a1, peg = filt.loc[filt.area=='A1'], filt.loc[filt.area=='PEG']
# stt = stats.mannwhitneyu(a1['FG_rel_gain'], peg['FG_rel_gain']).pvalue

filt = ohel.get_olp_filter(weight_df, kind='binaural', metric=False)
bin_dff = ohel.filter_across_condition(filt, synth_show=['11','21','12','22'], filt_kind='kind',
                                 snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=[''])
ofig.summary_relative_gain_all_areas(bin_dff, kind_show=['11','21','12','22'], category='kind', mult_comp=1)

# stt = ofig.summary_relative_gain_combine_areas(bin_dff, kind_show=['11','21','12','22'], category='kind', mult_comp=3)
# bin_df = filter_across_condition(filt, synth_show=['11','12','21','22'], filt_kind='kind',
#                                  snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=['_start', '_end'])
#
# ofig.plot_all_weight_comparisons(bin_df, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
#                                  weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category='kind')

#7D - SNR in ferrets
# Filter by SNR
filt = ohel.get_olp_filter(weight_df, kind='SNR', metric=False)

snr_df = ohel.filter_across_condition(filt, synth_show=[0, 5, 10], filt_kind='SNR', weight_lim=[-0.5, 2],
                                 snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=[''])
ofig.summary_relative_gain_all_areas(snr_df, kind_show=[0, 5, 10], category='SNR', mult_comp=1, statistic='independent',
                                     group_by_area=True)
# stt = ofig.summary_relative_gain_combine_areas(snr_df, kind_show=[0, 5, 10], category='SNR', mult_comp=1, statistic='paired')
# ofig.plot_all_weight_comparisons(snr_df, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
#                                  weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category='SNR')

#7E - by site in ferrets
ofig.site_relative_gain_summary(filt, snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, weight_lim=[-0.5,2])

# 7? spike width
#spikes path
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
weight_dff = ohel.add_spike_widths(filt, save_name='ferrets_with_spikes3', cutoff={'SLJ': 0.35, 'PRN': 0.35, 'other': 0.375})
path = f'/auto/users/hamersky/olp_analysis/ferrets_with_spikes'
path = f'/auto/users/hamersky/olp_analysis/ferrets_with_spikes2'
path = f'/auto/users/hamersky/olp_analysis/ferrets_with_spikes3'
weight_df = jl.load(path)

# filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
filt = weight_df
filt = filt.loc[filt.animal!='SLJ']
ohel.plot_spike_width_distributions(filt, split_critter='PRN', line=[0.35, 0.375])
# ofig.plot_all_weight_comparisons(filt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
#                                  weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category='width', flanks=False)

ohel.plot_spike_width_distributions(filt, split_critter=None, line=[0.375])
stat_dict = ofig.summary_relative_gain_all_areas(filt, kind_show=['broad', 'narrow'], category='width', mult_comp=1,
                                     statistic='independent', group_by_area=True)


#### Supplementaries ####

#######################
####### S1 ############
#### Example heatmaps #
ofig.response_heatmaps_comparison(weight_df, site='CLT012a', bg='Bees', fg='Bugle', cellid='CLT012a-052-1',
                                  batch=344, bin_kind='11', synth_kind='A', sigma=3, sort=True, example=True,
                                  lin_sum=True, positive_only=False)

ofig.response_heatmaps_comparison(weight_df, site='CLT012a', bg='Bees', fg='Bugle', cellid='CLT012a-052-1',
                                  batch=344, bin_kind='11', synth_kind='A', sigma=3, sort=True, example=True,
                                  lin_sum=True, positive_only=False)


########################
####### S2 #############
#### Some metric stuff #
# ofig.snr_scatter(filt, thresh=0.12, area='A1')
# ofig.snr_scatter(filt, thresh=0.12, area='PEG')
# #
# ofig.r_weight_comp_distribution(filt, increment=0.2, snr_threshold=0.12, threshold=None, area='A1')
# ofig.r_weight_comp_distribution(filt, increment=0.2, snr_threshold =0.12, threshold=None, area='PEG')

filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=False)
ofig.all_filter_stats(filt, xcol='bg_snr', ycol='fg_snr', snr_thresh=0.08, r_cut=0.4, increment=0.2,
                 fr_thresh=0.01, xx='resp')

ofig.r_weight_comp_distribution(filt, increment=0.2, snr_threshold=0.12, threshold=None, area='A1')
ofig.r_weight_comp_distribution(filt, increment=0.2, snr_threshold=0.12, threshold=None, area='PEG')

filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
ofig.weights_supp_comp(filt, x='resp', area='A1', quads=3, thresh=0.01, snr_threshold=0.12, r_cut=0.4)
ofig.weights_supp_comp(filt, x='resp', area='PEG', quads=3, thresh=0.01, snr_threshold=0.12, r_cut=0.4)


##########################
####### S3 ###############
#### Controls and things #
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=False)
ofig.weight_summary_histograms_flanks(filt, snr_threshold=0.12, stat_plot='median', bar=True)

# ofig.resp_weight_multi_scatter(filt, synth_kind=['N', 'A'], threshold=None, snr_threshold=0.12,
#                                r_thresh=None, area='A1')
# ofig.resp_weight_multi_scatter(filt, synth_kind=['N', 'A'], threshold=None, snr_threshold=0.12,
#                                r_thresh=None, area='PEG')
# ofig.snr_weight_scatter(filt, ycol='weightsB-weightsA', fr_met='fg_snr-bg_snr', threshold=None, rel_cut=2.5,
#                         snr_threshold=0.12, quads=3, r_thresh=0.4, weight_lims=[-0.5,2], area='A1')
# ofig.snr_weight_scatter(filt, ycol='weightsB-weightsA', fr_met='fg_snr-bg_snr', threshold=None, rel_cut=2.5,
#                         snr_threshold=0.12, quads=3, r_thresh=0.4, weight_lims=[-0.5,2], area='PEG')
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
ofig.snr_weight_scatter_multi_area(filt, ycol='weightsB-weightsA', fr_met='fg_snr-bg_snr', threshold=None, rel_cut=2.5,
                              snr_threshold=0.12, quads=3, r_thresh=0.4, weight_lims=[-0.5,2])

ofig.snr_weight_scatter_all_areas(filt)

#Summary of animals
ofig.summary_relative_gain_all_areas_by_animal(weight_df)

######################################
####### S4 ###########################
#### The only BG or FG responds data #
ofig.weight_summary_histograms_flanks(filt, snr_threshold=0.12, fr_thresh=None, r_cut=0.4, area='A1')
ofig.weight_summary_histograms_flanks(filt, snr_threshold=0.12, fr_thresh=None, r_cut=0.4, area='PEG')

ofig.plot_all_weight_comparisons(filt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
                                 weight_lim=[-0.5,2], summary=True, sep_hemi=False, sort_category=None)

#### Didn't get used
ofig.sound_stat_violin(weight_df, mets=['Fcorr', 'Tstationary', 'bandwidth'],
                  met_labels=['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'])

####################
####################




