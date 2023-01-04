import nems0.db as nd
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

weight_df = ofit.OLP_fit_weights(loadpath=path)
weight_df['batch'] = 340

# Get the dataframe of sound statistics
sound_df = ohel.get_sound_statistics_full(weight_df)

# # Tells you what sounds are below certain thresholds. And plots the sounds with their metrics.
# bad_dict = ohel.plot_sound_stats(sound_df, ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power', 'rel_gain'],
#                                  labels=['Frequency Non-stationarity', 'Temporal Non-stationarity', 'Bandwidth (octaves)',
#                                          'Max Power', 'RMS Power', 'Relative Gain'],
#                                  lines={'RMS_power': 0.95, 'max_power': 0.3}, synth_kind='N')
# bads = list(bad_dict['RMS_power'])
# bads = ['Waves', 'CashRegister', 'Heels', 'Keys', 'Woodblock', 'Castinets', 'Dice']  # Max Power
# Just gets us around running that above function, this is the output.
bads = ['CashRegister', 'Heels', 'Woodblock', 'Castinets', 'Dice']  # RMS Power
weight_df = weight_df.loc[weight_df['BG'].apply(lambda x: x not in bads)]
weight_df = weight_df.loc[weight_df['FG'].apply(lambda x: x not in bads)]


# A nice function I made that filters all the things I usually try to filter, at once.
weight_df0 = ohel.filter_weight_df(weight_df, suffixes=['_start', '_end'], fr_thresh=0.03, r_thresh=0.6, quad_return=3,
                                   bin_kind='11', synth_kind=None, area=None, weight_lims=[-1, 2],
                                   bads=True, bad_filt={'RMS_power': 0.95, 'max_power': 0.3})

# Add enhancement
weight_df0['FG_enhancement_start'] = weight_df0['weightsB_start'] - weight_df0['weightsA_start']
weight_df0['FG_enhancement_end'] = weight_df0['weightsB_end'] - weight_df0['weightsA_end']

oph.generate_interactive_plot(weight_df0, xcolumn='FG_enhancement_start', ycolumn='FG_enhancement_end', threshold=0.03)
oph.generate_interactive_plot(weight_df0, xcolumn='bg_FR', ycolumn='fg_FR', threshold=0.03)


# Use for presentation figures, it makes some progressive PSTHs
ofig.plot_PSTH_example_progression(333, cellid='TBR012a-31-1', bg='Wind', fg='Chirp', bin_kind='11', synth_kind='A',
                                   sigma=1, error=False)


weight_df0 = ohel.filter_weight_df(weight_df, suffixes=['', '_start', '_end'], fr_thresh=0.03, r_thresh=0.6, quad_return=3,
                                   bin_kind='11', synth_kind=None, area='PEG', weight_lims=[-1, 2],
                                   bads=True, bad_filt={'RMS_power': 0.95, 'max_power': 0.3})
weight_df0['FG_rel_gain_start'] = (weight_df0.weightsB_start - weight_df0.weightsA_start) / \
                                  (np.abs(weight_df0.weightsB_start) + np.abs(weight_df0.weightsA_start))
weight_df0['FG_rel_gain_end'] = (weight_df0.weightsB_end - weight_df0.weightsA_end) / \
                                (np.abs(weight_df0.weightsB_end) + np.abs(weight_df0.weightsA_end))
# Plots summary figure of the FG relative gain changes with synthetic condition and gives stats
ttests = osyn.synthetic_summary_relative_gain_bar(weight_df0)

ttests_a1 = synthetic_summary_relative_gain_bar(weight_df0_A1)
ttests_peg = synthetic_summary_relative_gain_bar(weight_df0_PEG)
ttests_a1_start = synthetic_summary_relative_gain_bar(weight_df0_A1_start)
ttests_a1_end = synthetic_summary_relative_gain_bar(weight_df0_A1_end)


ttest = synthetic_summary_relative_gain_multi_bar(weight_df0, suffixes=['', '_start', '_end'])



osyn.synthetic_relative_gain_comparisons(weight_df, thresh=0.03, quads=3, area='A1',
                                              synth_show=['M','S','T','C'],
                                         r_cut=0.6, rel_cut=2.5)
osyn.synthetic_relative_gain_comparisons_specs(weight_df, 'Jackhammer', 'Fight Squeak', thresh=0.03,
                                               synth_show=['N', 'M', 'S', 'T', 'C'],
                                               quads=3, r_cut=0.6, rel_cut=2.5, area='A1', figsize=(20,15))


# Takes a spectrogram and makes side panels describing some metrics you can get from it
ofig.spectrogram_stats_diagram('Jackhammer', 'BG')
ofig.spectrogram_stats_diagram('Fight Squeak', 'FG')



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
# weight_df = weight_df.drop(labels=['BG_Tstationary', 'BG_bandwidth', 'BG_Fstationary', \
#                        'FG_Tstationary', 'FG_bandwidth', 'FG_Fstationary', 'BG_RMS_power',
#                                    'BG_max_power', 'FG_RMS_power', 'FG_max_power', 'BG_f50',
#                                    'BG_t50', 'FG_f50', 'FG_t50'], axis=1)
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


from nems_lbhb.baphy_experiment import BAPHYExperiment
import copy
import nems0.epoch as ep
import nems0.preprocessing as preproc
import nems_lbhb.SPO_helpers as sp
import glob
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import re
import itertools
def get_sep_stim_names(stim_name):
    seps = [m.start() for m in re.finditer('_(\d|n)', stim_name)]
    if len(seps) < 2 or len(seps) > 2:
        return None
    else:
        return [stim_name[seps[0] + 1:seps[1]], stim_name[seps[1] + 1:]]

def OLP_fit_partial_weights(batch, threshold=None, synth=False, snip=None, fs=100, labels=None,
                            filter_animal=None, filter_experiment=None):
    weight_list = []

    if threshold:
        lfreq, hfreq, bins = 100, 24000, 48
        labels = ['', '_h', '_l', '_lp']

    # Snip refers to if you are going to divide the fit by time.
    if snip:


    cell_df = nd.get_batch_cells(batch)
    cell_list = cell_df['cellid'].tolist()
    cell_list = ohel.manual_fix_units(cell_list)  # So far only useful for two TBR cells

    #Only CLT synth units
    if filter_animal:
        cell_list = [cell for cell in cell_list if cell.split('-')[0][:3]==filter_animal]
    if filter_experiment:
        if filter_experiment[0] == '>':
            cell_list = [cell for cell in cell_list if cell_list.split('-')[0][:3] > filter_experiment[1]]


cell_list = [cell for cell in cell_list if (cell.split('-')[0][:3]=='CLT') & (int(cell.split('-')[0][3:6]) < 26)]

fit_epochs = ['10', '01', '20', '02', '11', '12', '21', '22']
fit_epochs = ['N', 'C', 'T', 'S', 'U', 'M', 'A']
fit_epochs = ['ff', 'fh', 'fn', 'hf', 'hh', 'hn', 'nf', 'nh']
loader = 'env100'
modelspecs_dir = '/auto/users/luke/Code/nems/modelspecs'

for cellid in cell_list:
    loadkey = 'ns.fs100'
    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = {'rasterfs': 100,
               'stim': False,
               'resp': True}
    rec = manager.get_recording(**options)

    #GET sound envelopes and get the indices for chopping?
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
            poststimFalse = np.full((trialbins - len(env),), False) ## Something is wrong here with the lengths

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
        start, dur = int(snip[0]*fs), int(snip[1]*fs)
        prestimFalse = np.full((prebins,), False)
        # poststimTrue = np.full((trialbins - len(env),), True)
        poststimFalse = np.full((trialbins - durbins), False)
        # if start == dur:
        #
        # else:
        end = durbins - start - dur
        goods = [False]*start + [True]*dur + [False]*end
        bads = [not ll for ll in goods]


        full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
        goods = np.concatenate((prestimFalse, goods, poststimFalse))
        bads = np.concatenate((prestimFalse, bads, poststimFalse))
        full_nopost = np.concatenate((prestimFalse, np.full((durbins,), True), poststimFalse))
        cut_list = [full, goods, bads, full_nopost]
        cut_labels = ['', '_start', '_end', '_nopost']
        #
        # cut_labels = ['', '_good', '_bad']


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
    #change this shit
    # if synth == True:
    #     df0['name'] = df0['name'].apply(ohel.label_synth_type)
    # else:
    # df0['name'] = df0['name'].apply(ohel.label_ep_type)
    df0['name'] = df0['name'].apply(label_dynamic_ep_type)




    df0 = df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])

    val['resp'].epochs = df3
    val_sub = copy.deepcopy(val)
    val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)

    val = val_sub
    fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR / rec['resp'].fs)
    val['resp'] = val['resp'].transform(fn)

    print(f'calc weights {cellid}')

    #where twostims fit actually begins
    ## This is where adapt to fitting only the half stimuli
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

    #Calculate weights
    if synth == True:
        subsets = len(cut_list)
    else:
        subsets = len(list(env_cuts.values())[0])
    weights = np.zeros((2, len(AB), subsets))
    Efit = np.zeros((5,len(AB), subsets))
    nMSE = np.zeros((len(AB), subsets))
    nf = np.zeros((len(AB), subsets))
    r = np.zeros((len(AB), subsets))
    cut_len = np.zeros((len(AB), subsets-1))
    get_error=[]

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
            names=[[A[i]],[B[i]],[AB[i]]]
            Fg = names[1][0].split('_')[2].split('-')[0]
            cut_list = env_cuts[Fg]

            for ss, cut in enumerate(cut_list):
                weights[:,i,ss], Efit[:,i,ss], nMSE[i,ss], nf[i,ss], _, r[i,ss] = \
                        ofit.calc_psth_weights_of_model_responses_list(val, names,
                                                                  signame='resp', cuts=cut)
                if ss != 0:
                    cut_len[i, ss-1] = np.sum(cut)
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
    #Combines the lists into a format that is conducive to the dataframe format I want to make
    bigger_list = small_list + flat_list
    weight_df = pd.DataFrame(bigger_list)
    weight_df = weight_df.T

    #Automatically generates a list of column names based on the names of the subsets provided above
    column_labels1 = ['namesA', 'namesB']
    column_labels2 = [[f"weightsA{cl}", f"weightsB{cl}", f"nMSE{cl}", f"nf{cl}", f"r{cl}"] for cl in cut_labels]
    column_labels_flat = [item for sublist in column_labels2 for item in sublist]
    column_labels = column_labels1 + column_labels_flat
    #Renames the columns according to that list - should work for any scenario as long as you specific names above
    weight_df.columns = column_labels1 + column_labels_flat

    #Not sure why I need this, I guess some may not be floats, so just doing it
    col_dict = {ii: float for ii in column_labels_flat}
    weight_df = weight_df.astype(col_dict)

    weight_df.insert(loc=0, column='cellid', value=cellid)
    weight_list.append(weight_df)

weight_df0 = pd.concat(weight_list)


ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df0.namesA, weight_df0.namesB)]
weight_df0 = weight_df0.drop(columns=['namesA', 'namesB'])
weight_df0['epoch'] = ep_names

OLP_partialweights_db_path = f'/auto/users/hamersky/olp_analysis/ARM_Dynamic_OLP_control_segment{int(snip[0] * 1000)}-{int((snip[0]+snip[1]) * 1000)}_nometrics.h5'  # weight + corr
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


OLP_metrics_db_path = f'/auto/users/hamersky/olp_analysis/ARM_Dynamic_test{int(snip[0] * 1000)}-{int((snip[0]+snip[1]) * 1000)}metrics.h5'  # weight + corr
os.makedirs(os.path.dirname(OLP_metrics_db_path), exist_ok=True)
store = pd.HDFStore(OLP_metrics_db_path)
df_store = copy.deepcopy(df)
store['df'] = df_store.copy()
store.close()

weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
weight_df0['fit_segment'] = f"{int(snip[0] * 1000)}-{int((snip[0]+snip[1]) * 1000)}"



OLP_savepath = f'/auto/users/hamersky/olp_analysis/ARM_Dynamic_OLP_segment{int(snip[0] * 1000)}-{int((snip[0]+snip[1]) * 1000)}.h5'  # weight + corr
os.makedirs(os.path.dirname(OLP_savepath), exist_ok=True)
store = pd.HDFStore(OLP_savepath)
df_store = copy.deepcopy(all_df)
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
def label_dynamic_ep_type(ep_name):
    '''Labels epochs that have one or two stimuli in it according to its duration (dynamic stimuli.
    First position refers to BG, second to FG. n means null, f means full length, h means half length'''
    if len(ep_name.split('_')) == 1 or ep_name[:5] != 'STIM_':
        stim_type = None
    elif len(list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])) == 2:
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])

        if len(seps[0].split('-')) >= 2 or len(seps[1].split('-')) >= 2:
            if seps[0] != 'null' and seps[1] != 'null':
                if seps[0].split('-')[1] == '0':
                    btype = 'f'
                else:
                    btype = 'h'
                if seps[1].split('-')[1] == '0':
                    ftype = 'f'
                else:
                    ftype = 'h'
                stim_type = btype + ftype
            else:
                if seps[0] == 'null':
                    if seps[1].split('-')[1] == '0':
                        ftype = 'f'
                    else:
                        ftype = 'h'
                    stim_type = 'n' + ftype
                elif seps[1] == 'null':
                    if seps[0].split('-')[1] == '0':
                        btype = 'f'
                    else:
                        btype = 'h'
                    stim_type = btype + 'n'
    else:
        stim_type = None
        print(f"None of your labeling things worked for {ep_name}, you should look into that.")

    return stim_type


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
df_store = copy.deepcopy(weight_df0)
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