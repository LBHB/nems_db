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


# The thing I was working on in January with fit
path = '/auto/users/hamersky/olp_analysis/2023-01-12_Batch341_0-500_FULL'


#marms
path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch328_0-500_marm'
weight_df = jl.load(path)


#spikes path
weight_dff = ohel.add_spike_widths(filt, save_name='ferrets_with_spikes', cutoff={'PRN': 0.3, 'other': 0.375})
path = f'/auto/users/hamersky/olp_analysis/ferrets_with_spikes'
weight_df = jl.load(path)

# 2023_05_02. Starting with Prince data too and new df structure
# path = '/auto/users/hamersky/olp_analysis/2023-05-10_batch344_0-500_metrics' # Full one with updated PRNB layers
# path = '/auto/users/hamersky/olp_analysis/2023-05-17_batch344_0-500_metrics' #full one with PRNB layers and paths
# path = '/auto/users/hamersky/olp_analysis/2023-07-20_batch344_0-500_metrics' # full with new FR snr metric
path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch344_0-500_metric'
weight_df = jl.load(path)


# Using this to play with the statistics of the sounds
filt = weight_df
filt = ohel.label_vocalization(filt, species='ferret')


filt = filt.loc[(filt.area=='A1') | (filt.area=='PEG')]

filt.loc[filt.layer=='4', 'layer'] = '44'
filt.loc[filt.layer=='5', 'layer'] = '56'
filt.loc[filt.layer=='BS', 'layer'] = '13'
filt = filt.loc[(filt.layer=='NA') | (filt.layer=='5') | (filt.layer=='44') | (filt.layer=='13') |
                (filt.layer=='4') | (filt.layer=='56') | (filt.layer=='16') | (filt.layer=='BS')]

filt = filt.loc[filt.dyn_kind=='ff']
filt = filt.loc[filt.kind=='11']
filt = filt.loc[filt.SNR==0]
filt = filt.loc[filt.olp_type=='synthetic']
filt = filt.loc[filt.olp_type=='binaural']


# filt = filt.loc[((filt.synth_kind=='N') & (filt['animal']=='CLT') & (filt['olp_type']=='synthetic')) |
#                 ((filt.synth_kind=='A') & (filt['animal'].isin(['CLT', 'PRN'])) & (filt['olp_type']=='binaural')) |
#                 ((filt.synth_kind=='N') & (filt['animal']=='PRN') |
#                 ((filt.synth_kind=='A') & (filt['animal'].isin(['TNC','ARM']))))]
filt = filt.loc[((filt.synth_kind=='N') & (filt['animal']=='CLT') & (filt['olp_type']=='synthetic')) |
                ((filt.synth_kind=='A') & (filt['animal']=='CLT') & (filt['olp_type']!='synthetic')) |
                ((filt.synth_kind=='N') & (filt['animal']=='PRN')) |
                ((filt.synth_kind=='A') & (filt['animal'].isin(['TNC','ARM'])))]
#
# filt = filt.loc[((filt.synth_kind=='N') & (filt['animal']=='CLT') & (filt['olp_type']=='synthetic')) |
#                 ((filt.synth_kind=='N') & (filt['animal'].isin(['CLT', 'PRN']))) |
#                 ((filt.synth_kind=='A') & (filt['animal'].isin(['TNC','ARM'])))]

# filt = filt.loc[((filt.synth_kind=='N') & (filt['animal'].isin(['CLT_A','CLT_B','PRN_A','PRN_B']))) |
#                 ((filt.synth_kind=='A') & (filt['animal'].isin(['TNC','ARM'])))]
# filt = filt.loc[(filt.synth_kind=='N') | (filt.synth_kind=='A')]
# filt = filt.loc[(filt.synth_kind=='N')]

# sound_df = ohel.get_sound_statistics_from_df(filt, percent_lims=[15,85], append=False)
# bad_dict = ohel.plot_sound_stats(sound_df, ['max_power', 'RMS_power'], labels=['Max Power', 'RMS Power'],
#                                  lines={'RMS_power': 0.95, 'max_power': 0.3}, synth_kind='N')
# bads = ['CashRegister', 'Heels', 'Castinets', 'Dice']  # RMS Power Woodblock for 'N'
bads = ['Branch', 'CashRegister', 'Heels', 'Woodblock', 'Castinets', 'Dice'] #RMS power
filt = filt.loc[filt['BG'].apply(lambda x: x not in bads)]
filt = filt.loc[filt['FG'].apply(lambda x: x not in bads)]

filt = ohel.df_filters(filt, snr_threshold=0.12, rel_cut=2.5, r_cut=0.4, weight_lim=[-0.5,2])






from nems_lbhb.stats import jack_mean_err
m,se = jack_mean_err(x, do_median=True)

















#Trying multiple regression
a1, voc_label = ohel.run_sound_stats_reg(filt, r_cut=0.4, snr_threshold=0.12, suffix='', synth=None,
              xs=['Fcorr', 'Tstationary', 'bandwidth', 'snr', 'spectral_overlap'],
              category='Vocalization', area='A1', shuffle=True)

peg, voc_label = ohel.run_sound_stats_reg(filt, r_cut=0.4, snr_threshold=0.12, suffix='', synth=None,
              xs=['Fcorr', 'Tstationary', 'bandwidth', 'snr', 'spectral_overlap'],
              category='Vocalization', area='PEG', shuffle=True)

for aa in a1.keys():
    print(f'{aa}, r={np.around(a1[aa].rsquared, 2)}')

ofig.plot_big_sound_stat_regression(filt, xvar=['Fcorr', 'Tstationary', 'bandwidth', 'snr', 'spectral_overlap'],
                        cat='Vocalization', omit='C(Vocalization)[T.3]')
















## Figure 1 ##
# C. PSTHs
ofig.psths_with_specs(weight_df, 'CLT008a-046-2', 'Wind', 'Geese', batch=340, bin_kind='11', synth_kind='A',
                         sigma=1, error=False, title='A1 046-2')
ofig.psths_with_specs(weight_df, 'CLT012a-052-1', 'Bees', 'Bugle', batch=340, bin_kind='11', synth_kind='A',
                         sigma=1, error=False, title='A1 052-1')
ofig.psths_with_specs(weight_df, 'CLT052d-018-1', 'Wind', 'Geese', batch=340, bin_kind='11', synth_kind='A',
                         sigma=1, error=False, title='PEG 018-1')


# D. Heatmap example of suppression. Some good examples.
# ofig.response_heatmaps_comparison(weight_df, site='CLT009a', bg='Bulldozer', fg='FightSqueak', cellid=None,
#                                      batch=340, bin_kind='11', synth_kind='A', sigma=1, sort=True,
#                              example=True, lin_sum=True, positive_only=False)
#
# ofig.response_heatmaps_comparison(weight_df, site='CLT012a', bg='Bees', fg='Bugle', cellid='CLT012a-052-1',
#                                      batch=340, bin_kind='11', synth_kind='A', sigma=1, sort=True,
#                              example=True, lin_sum=True, positive_only=False)
ofig.response_heatmaps_comparison(weight_df, site='CLT008a', bg='Wind', fg='Geese', cellid='CLT008a-046-2',
                                     batch=340, bin_kind='11', synth_kind='A', sigma=3, sort=True,
                             example=True, lin_sum=True, positive_only=False)

ofig.response_heatmaps_comparison(weight_df, site='CLT012a', bg='Bees', fg='Bugle', cellid='CLT012a-052-1',
                                     batch=340, bin_kind='11', synth_kind='A', sigma=3, sort=True,
                             example=True, lin_sum=True, positive_only=False)

ofig.response_heatmaps_comparison(weight_df, site='CLT052d', bg='Wind', fg='Geese', cellid='CLT052d-018-1',
                                     batch=340, bin_kind='11', synth_kind='A', sigma=3, sort=True,
                             example=True, lin_sum=True, positive_only=False)

weight_df['site'] = [dd[:7] for dd in weight_df['cellid']]

ofig.response_heatmaps_comparison(weight_df, site='ARM029a', bg='Wind', fg='Geese', cellid='CLT008a-046-2',
                                     batch=340, bin_kind='11', synth_kind='A', sigma=3, sort=True,
                             example=True, lin_sum=True, positive_only=False)

## Figure 2 ##
#2A
# Model
#2B weights summary A1
ofig.weight_summary_histograms(filt, threshold=None, snr_threshold=0.12, r_cut=0.4, area='A1', rel_cut=2.5,
                               bar=True)
#2C weights summary PEG
ofig.weight_summary_histograms(filt, threshold=None, snr_threshold=0.12, r_cut=0.4, area='PEG', rel_cut=2.5,
                               bar=True)





## For the figure if only I could find a good example 2022_11_01
# ofig.psths_with_specs_partial_fit(weight_df, 'CLT047c-012-1', 'Bees', 'Gobble', sigma=1, error=False)
# ofig.psths_with_specs_partial_fit(weight_df, 'CLT040c-051-1', 'Tuning', 'ManA', sigma=1, error=False)
# ofig.psths_with_specs_partial_fit(weight_df, 'CLT052d-035-2', 'Bees', 'Chickens', sigma=1, error=False)
ofig.psths_with_specs_partial_fit(weight_df, 'CLT052d-018-1', 'Wind', 'Geese', sigma=1, error=False)
ofig.psths_with_specs_partial_fit(weight_df, 'CLT012a-052-1', 'Bees', 'Bugle', sigma=1, error=False, synth_kind='A')


## Figure 3 ##
#3A/B
ofig.plot_all_weight_comparisons(filt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
                                 weight_lim=[-0.5, 2], summary=True, sep_hemi=False, sort_category=None)



# 3A Relative gain intro
# ofig.plot_single_relative_gain_hist(filt, threshold=0.03, r_cut=0.06)

## Figure 4 ##
# 4B
# Figure dynamic - to gather and plot the dynamic data. All you need is to load main weight_df
# full_df = ohel.merge_dynamic_error(weight_df, dynamic_path='cache_dyn', SNR=0)
full_df = ohel.merge_dynamic_error(weight_df, dynamic_path='cache_dyn_no_spont', SNR=0)
filt = full_df
#filt for the layers, area, get rid of bad sounds
sites = [dd.split('-')[0] for dd in full_df.cellid]
full_df['site'] = sites

ofig.plot_dynamic_errors(full_df, dyn_kind='fh', snr_threshold=0.12, thresh=None)
ofig.plot_dynamic_errors(full_df, dyn_kind='fh', snr_threshold=None, thresh=0.03)

ofig.plot_dynamic_site_errors(full_df, dyn_kind='fh', thresh=0.03, snr_threshold=None)



## Figure 5 ##
# # Figure 5A sound stat
sound_df = ohel.get_sound_statistics_from_df(filt, percent_lims=[15,85], append=False)
# filt was filtered for snr and r, both areas left
ohel.plot_sound_stats(sound_df, ['Fcorr', 'Tstationary', 'bandwidth'],
                 labels=['Frequency Correlation', 'Temporal Non-stationarity', 'Bandwidth (octaves)'],
                 synth_kind='A')

# 5C sound stats with rel gain
ofig.sound_metric_scatter(filt, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'], suffix='',
                          area='A1', metric_filter=2.5, snr_threshold=0.12, threshold=None, synth_kind=None,
                          r_cut=0.4, jitter=None, mean=True, vocalization=True)
ofig.sound_metric_scatter(filt, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
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
ofig.plot_all_weight_comparisons(filtt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
                                 weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category="Vocalization",
                                 flanks=False)

#5E Regression
ofig.plot_big_sound_stat_regression(filt, xvar=['Fcorr', 'Tstationary', 'bandwidth', 'snr', 'spectral_overlap'],
                        cat='Vocalization', omit='C(Vocalization)[T.3]')



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

## Figure 6 ##

## Figure 6 ##
# 6A shows relative gain histograms across synthetic conditions
df = ohel.filter_across_condition(filt, synth_show=['M','S','T','C'], filt_kind='synth_kind',
                                  snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=[''])

osyn.synthetic_relative_gain_comparisons_specs(df, 'Jackhammer', 'Fight Squeak', thresh=None, snr_threshold=0.12,
                                               synth_show=['M', 'S', 'T', 'C'],
                                               quads=3, r_cut=0.4, rel_cut=2.5, area='A1', figsize=(20,15))

osyn.synthetic_relative_gain_comparisons(df, snr_threshold=0.12, thresh=None, quads=3, area='PEG',
                                              synth_show=['M','S','T','C'],
                                         r_cut=0.4, rel_cut=2.5)

#6B/C Summary of above RG histograms
stat_dict = osyn.synthetic_summary_relative_gain_all_areas(df, synth_show=['M','S','T','C'], mult_comp=5)

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

# 6D
df = ohel.filter_across_synths(filt, synth_show=['N','M','S','T','C'], snr_threshold=0.12, r_cut=0.4,
                               rel_cut=2.5, suffix=[''])
osyn.synthetic_sound_metric_scatters(df, ['Fcorr', 'Tstationary', 'bandwidth'], synth_show=['N', 'M', 'S', 'T', 'C'],
                          x_labels=['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                              suffix='')
# Maybe a supplement?
# osyn.sound_metric_scatter_combined_flanks(filt, ['Fcorr', 'Tstationary', 'bandwidth'], fr_thresh=0.03, r_cut=0.6,
#                           x_labels=['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'],
#                               suffix='_end')

## Figure 7 ##
# 7A Marmoset
path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch328_0-500_marm'
weight_df = jl.load(path)
filt = ohel.label_vocalization(weight_df, species='marmoset')

filt.loc[(filt['area'] == 'A1orR') | (filt['area'] == 'R'), 'area'] = 'A1'
filt = filt.loc[filt['area'].isin(['ML', 'AL', 'A1'])]
filt.loc[(filt['area'] == 'AL') | (filt['area'] == 'ML'), 'area'] = '2nd'
filt = filt.loc[filt['area'].isin(['A1', '2nd'])]

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


ofig.weight_summary_histograms(filt, threshold=None, snr_threshold=0.12, r_cut=0.4, area='A1', rel_cut=2.5,
                               bar=True)
ofig.weight_summary_histograms(filt, threshold=None, snr_threshold=0.12, r_cut=0.4, area='2nd', rel_cut=2.5,
                               bar=True)

ofig.metric_weight_bars(quiet, snr_threshold=0.12, r_cut=0.4)




# area_summary_bars(filt, snr_threshold=0.12, r_cut=0.4, category='fg_noise')

filt = quiet
snr_threshold = 0.12
filt = filt.loc[(filt.bg_snr >= snr_threshold) & (filt.fg_snr >= snr_threshold)]
r_cut = 0.4
filt = filt.dropna(axis=0, subset='r')
filt = filt.loc[filt.r >= r_cut]
rel_cut = 2.5
filt = filt.loc[(filt['FG_rel_gain'] <= rel_cut) & (filt[f'FG_rel_gain'] >= -rel_cut)]
filt = filt.loc[(filt['BG_rel_gain'] <= rel_cut) & (filt[f'BG_rel_gain'] >= -rel_cut)]

area = '2nd' # of '2nd'
ofig.sound_metric_scatter(filt, ['Fcorr', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'], suffix='',
                          area=area, metric_filter=2.5, snr_threshold=0.12, threshold=None, synth_kind='A',
                          r_cut=0.4, jitter=None, mean=True)
filt['marmoset'] = 'marm'
stat_dict = ofig.summary_relative_gain_all_areas(filt, kind_show=['marm'], category='marmoset', mult_comp=None,
                                     statistic='paired', secondary_area_name='2nd')


# 7B - layers in ferrets
stat_dict = ofig.summary_relative_gain_all_areas(filt, kind_show=['13', '44', '56'], category='layer', mult_comp=3,
                                     statistic='independent')

# 7C - Binaural in ferrets
bin_dff = ohel.filter_across_condition(filt, synth_show=['11','21','12','22'], filt_kind='kind',
                                 snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=[''])
ofig.summary_relative_gain_all_areas(bin_dff, kind_show=['11','21','12','22'], category='kind', mult_comp=3)

# bin_df = filter_across_condition(filt, synth_show=['11','12','21','22'], filt_kind='kind',
#                                  snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=['_start', '_end'])
#
# ofig.plot_all_weight_comparisons(bin_df, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
#                                  weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category='kind')

#7D - SNR in ferrets
# Filter by SNR
filt = filt.loc[(filt.olp_type!='binaural') & (filt.olp_type!='synthetic')]
filt['filt_name'] = filt['filt_name'] = filt['cellid'] + '-' + filt['BG'] + '-' + filt['FG']

snr10 = filt.loc[filt.SNR==10]
epoch_names = snr10.filt_name.tolist()

filt = filt.loc[filt.filt_name.isin(epoch_names)]
filt = filt.drop(labels=['filt_name'], axis=1)

snr_df = ohel.filter_across_condition(filt, synth_show=[0, 10], filt_kind='SNR', weight_lim=[-0.5, 2],
                                 snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, suffix=['_start', '_end'])
ofig.summary_relative_gain_all_areas(snr_df, kind_show=[0, 10], category='SNR', mult_comp=3, statistic='paired')

# ofig.plot_all_weight_comparisons(snr_df, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
#                                  weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category='SNR')



#7E - by site in ferrets
ofig.site_relative_gain_summary(filt, snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, weight_lim=[-0.5,2])



# 7? spike width
ohel.plot_spike_width_distributions(filt, split_critter='PRN', line=[0.3, 0.375])
ofig.plot_all_weight_comparisons(filt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
                                 weight_lim=[-0.5,2], summary=False, sep_hemi=False, sort_category='width', flanks=False)
stat_dict = ofig.summary_relative_gain_all_areas(filt, kind_show=['broad', 'narrow'], category='width', mult_comp=1,
                                     statistic='independent')


#S1
ofig.response_heatmaps_comparison(weight_df, site='CLT012a', bg='Bees', fg='Bugle', cellid='CLT012a-052-1',
                                  batch=344, bin_kind='11', synth_kind='A', sigma=3, sort=True, example=True,
                                  lin_sum=True, positive_only=False)

ofig.response_heatmaps_comparison(weight_df, site='CLT012a', bg='Bees', fg='Bugle', cellid='CLT012a-052-1',
                                  batch=344, bin_kind='11', synth_kind='A', sigma=3, sort=True, example=True,
                                  lin_sum=True, positive_only=False)

#S2
ofig.snr_scatter(filt, thresh=0.12, area='A1')
ofig.snr_scatter(filt, thresh=0.12, area='PEG')

ofig.r_weight_comp_distribution(filt, increment=0.2, snr_threshold=0.12, threshold=None, area='A1')
ofig.r_weight_comp_distribution(filt, increment=0.2, snr_threshold =0.12, threshold=None, area='PEG')

#S3
ofig.weights_supp_comp(filt, x='resp', area='A1', quads=3, thresh=0.03, snr_threshold=0.12, r_cut=None)
ofig.weights_supp_comp(filt, x='resp', area='PEG', quads=3, thresh=0.03, snr_threshold=0.12, r_cut=None)

#S4
# ofig.resp_weight_multi_scatter(filt, synth_kind=['N', 'A'], threshold=None, snr_threshold=0.12,
#                                r_thresh=None, area='A1')
# ofig.resp_weight_multi_scatter(filt, synth_kind=['N', 'A'], threshold=None, snr_threshold=0.12,
#                                r_thresh=None, area='PEG')
ofig.snr_weight_scatter(filt, ycol='weightsB-weightsA', fr_met='fg_snr-bg_snr', threshold=None, rel_cut=2.5,
                        snr_threshold=0.12, quads=3, r_thresh=None, weight_lims=[-0.5,2], area='A1')
ofig.snr_weight_scatter(filt, ycol='weightsB-weightsA', fr_met='fg_snr-bg_snr', threshold=None, rel_cut=2.5,
                        snr_threshold=0.12, quads=3, r_thresh=None, weight_lims=[-0.5,2], area='PEG')

ofig.snr_weight_scatter_all_areas(filt)


#S5
ofig.weight_summary_histograms_flanks(filt, snr_threshold=0.12, fr_thresh=None, r_cut=0.4, area='A1')
ofig.weight_summary_histograms_flanks(filt, snr_threshold=0.12, fr_thresh=None, r_cut=0.4, area='PEG')

ofig.plot_all_weight_comparisons(filt, fr_thresh=None, snr_threshold=0.12, r_thresh=0.4, strict_r=True,
                                 weight_lim=[-0.5,2], summary=True, sep_hemi=False, sort_category=None)

#S6

ohel.sound_stat_violin(weight_df, mets=['Fcorr', 'Tstationary', 'bandwidth'],
                  met_labels=['Spectral\nCorrelation', 'Temporal\nNon-Stationariness', 'Bandwidth'])
####################
####################





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