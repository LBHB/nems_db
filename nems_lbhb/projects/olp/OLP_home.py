import nems.db as nd
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
from nems.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import glob
import nems.epoch as ep

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
fs = 100

# Load your different, updated dataframes
path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_sound_stats.h5' #weight + corr
path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5' # Still needs the CLT053 4 units

weight_df = ofit.OLP_fit_weights(loadpath=path)


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
# sound_df = ohel.get_sound_statistics_full(weight_df)
# weight_df = weight_df.drop(labels=['BG_Tstationary', 'BG_bandwidth', 'BG_Fstationary', \
#                        'FG_Tstationary', 'FG_bandwidth', 'FG_Fstationary'], axis=1)
# weight_df = ohel.add_sound_stats(weight_df, sound_df)








# Get the dataframe of sound statistics
sound_df = ohel.get_sound_statistics_full(weight_df)

# Plot simple comparison of sites and synethetics
osyn.plot_synthetic_weights(weight_df, thresh=0.05, areas=None, synth_show=None)

# I use this for most things
quad, _ = ohel.quadrants_by_FR(weight_df, threshold=0.03, quad_return=3)

# For scatter of sound features to rel gain
ofig.sound_metric_scatter(weight_df, ['Fstationary', 'Tstationary', 'bandwidth'], 'BG_rel_gain',
                          ['Frequency\nNon-stationariness', 'Temporal\nNon-Stationariness', 'Bandwidth'],
                          area='A1', threshold=0.03, synth_kind='N')

#interactive plot
oph.generate_interactive_plot(weight_df)

# Get scatters of FRs/weights
ofig.resp_weight_multi_scatter(weight_df, synth_kind='N', threshold=0.03)


# For scatter of sound features to rel gain
sound_df = ohel.get_sound_statistics_full(weight_df)
# Adds max_power, must use A
ofig.sound_metric_scatter(weight_df, ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power'],
                          'BG_rel_gain', ['Frequency\nNon-Stationarity', 'Temporal\nNon-Stationarity',
                                          'Bandwidth', 'Max Power', 'RMS Power'], jitter=[0.25, 0.2, 0.03, 0.03, 0.003],
                          area='A1', threshold=0.03, synth_kind='N', title_text='Removed Low')

# Tells you what sounds are below certain thresholds. And plots the sounds with their metrics.
stats=['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power', 'rel_gain']
labels=['Frequency Non-stationarity', 'Temporal Non-stationarity', 'Bandwidth (octaves)', 'Max Power', 'RMS Power', 'Relative Gain']
lines = {'RMS_power': 0.95, 'max_power': 0.4}
bad_dict = ohel.plot_sound_stats(sound_df, stats, labels=labels, lines=lines, synth_kind='C')

bads = list(set([item for sublist in list(bad_dict.values()) for item in sublist]))
#bads = list(bad_dict['RMS_power'])

weight_df = weight_df.loc[weight_df['BG'].apply(lambda x: x not in bads)]
weight_df = weight_df.loc[weight_df['FG'].apply(lambda x: x not in bads)]
#OR weight_df = weight_df.loc[weight_df['FG'].apply(lambda x: x in bads)]
weight_df = weight_df.loc[weight_df.kind=='11']


ofig.plot_single_relative_gain_hist(weight_df, 0.03, synth_kind='N')
osyn.plot_synthetic_weights(filt_df, thresh=0.05, areas=None, synth_show=None)



# Adds max_power, must use A
ofig.sound_metric_scatter(filt_df, ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power'],
                          'BG_rel_gain', ['Frequency\nNon-Stationarity', 'Temporal\nNon-Stationarity',
                                          'Bandwidth', 'Max Power', 'RMS Power'],
                          jitter=[0.25, 0.2, 0.03, 0.03, 0.0003],
                          area='A1', threshold=0.03, synth_kind='N',
                          title_text='removed low max power FGs')



