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
path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_sound_stats.h5' #weight + corr
path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5' # Still needs the CLT053 4 units

weight_df = ofit.OLP_fit_weights(loadpath=path)

# Get the dataframe of sound statistics
sound_df = ohel.get_sound_statistics_full(weight_df)

# Tells you what sounds are below certain thresholds. And plots the sounds with their metrics.
stats = ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power', 'rel_gain']
labels = ['Frequency Non-stationarity', 'Temporal Non-stationarity', 'Bandwidth (octaves)', 'Max Power', 'RMS Power', 'Relative Gain']
lines = {'RMS_power': 0.95, 'max_power': 0.4}
bad_dict = ohel.plot_sound_stats(sound_df, stats, labels=labels, lines=lines, synth_kind='N')
# bads = list(set([item for sublist in list(bad_dict.values()) for item in sublist]))
bads = list(bad_dict['RMS_power'])
weight_df = weight_df.loc[weight_df['BG'].apply(lambda x: x not in bads)]
weight_df = weight_df.loc[weight_df['FG'].apply(lambda x: x not in bads)]
#OR weight_df = weight_df.loc[weight_df['FG'].apply(lambda x: x in bads)]



def checkout_mods(type, df, spec_ax, show=['N', 'M', 'S', 'T', 'C'], thresh=0.03, quads=3, r_cut=None, area=None):
    '''2022_09_28. Takes a number from the list of names (you have to run it once first with
    a random number to get the indexes printed out) and will plot some modulation specs along
    the degraded synthetic sounds. Nice for browsing.'''
    quad, threshold = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    if area:
        quad = quad.loc[quad.area == area]

    n_synths = quad.groupby(type).agg(count=('synth_kind', lambda x: pd.Series.nunique(x)))
    bad = n_synths.query(f'count < {len(show)}').index
    quad = quad.loc[~quad[type].isin(bad), :]

    bbs = list(set([bb.split('_')[1].split('-')[0] for bb in quad.epoch]))
    ffs = list(set([ff.split('_')[2].split('-')[0] for ff in quad.epoch]))
    bbs.sort(key=lambda x: x[:2]), ffs.sort(key=lambda x: x[:2])
    bbs, ffs = [bb[2:] for bb in bbs], [ff[2:] for ff in ffs]

    if type == 'FG':
        sounds, col = ffs, 'yellowgreen'
    elif type == 'BG':
        sounds, col = bbs[:9], 'deepskyblue'

    if spec_ax == 'T':
        title = 'Temporal PS'
    elif spec_ax == 'F':
        title = 'Spectral PS'

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    kind_alias = {'A': 'Non-RMS Norm\nNatural', 'N': 'RMS Norm\nNatural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}

    fig, axes = plt.subplots(len(show), len(sounds), figsize=(len(sounds), len(show)*2))
    ax = np.ravel(axes, 'F')
    dd = 0
    for cnt, ss in enumerate(sounds):
        name_df = quad.loc[quad[type]==ss]
        ax[dd].set_title(ss, fontsize=10, fontweight='bold')
        for syn in show:
            # This is getting the mean rel gain for each sound (FG rel gain for FGs, etc)
            synth_df = name_df.loc[name_df.synth_kind == syn].copy()

            gain_df = synth_df[[type, f'{type}_rel_gain']]
            gain_mean = np.around(gain_df[f'{type}_rel_gain'].mean(), 3)

            if spec_ax == 'T':
                ps = np.sum(synth_df[f'{type}_temp_ps'].iloc[0], axis=0)
            elif spec_ax == 'F':
                ps = np.sum(synth_df[f'{type}_freq_ps'].iloc[0], axis=1)

            ax[dd].plot(ps[1:], color=col)
            ymin,ymax = ax[dd].get_ylim()
            ax[dd].annotate(f"{gain_mean}", xy=(len(ps)/2,ymax), fontsize=8, fontweight='bold')
            ax[dd].set_yticks([])
            if cnt == 0:
                ax[dd].set_ylabel(f"{kind_alias[syn]}", fontsize=10, fontweight='bold')
            dd += 1
    fig.suptitle(title, fontsize=10, fontweight='bold')
    fig.tight_layout()








##
# Stuff with synthetics viewing.
names = osyn.checkout_mods(21, weight_df, thresh=0.03, quads=3, r_cut=0.8)
osyn.rel_gain_synth_scatter(weight_df, show=['N','M','S','T','C'],
                            thresh=0.03, quads=3, r_cut=0.8, area='A1')
osyn.rel_gain_synth_scatter_single(weight_df, show=['N','M','S','T','C'], thresh=0.03,
                              quads=3, r_cut=0.8, area='A1')
osyn.synthetic_relative_gain_comparisons(weight_df, thresh=0.06, quads=3, area='A1',
                                              synth_show=['N','M','S','T','C'],
                                         r_cut=0.8, rel_cut=2.5)
osyn.synthetic_relative_gain_comparisons_specs(weight_df, 'Rock Tumble', 'Typing', thresh=0.03,
                                               synth_show=['N', 'M', 'S', 'T', 'C'],
                                               quads=3, r_cut=0.8, rel_cut=2.5, area='A1')

# Plots synthetic metrics, good for viewing like I want to do above
osyn.synth_scatter_metrics(weight_df, first='temp_ps_std', second='freq_ps_std', metric='rel_gain',
                      show=['N', 'M', 'S', 'T', 'C'], thresh=0.03, quads=3, r_cut=0.8, ref='M', area='A1')

osyn.synth_scatter_metrics(weight_df, first='temp_ps_std', second='freq_ps_std', metric='rel_gain',
                      show=['N', 'M', 'S', 'T', 'C'], thresh=0.03, quads=3, r_cut=0.8, ref=None, area='A1')

# Plot simple comparison of sites and synethetics
osyn.plot_synthetic_weights(weight_df, thresh=0.05, areas=None, synth_show=None, r_cut=0.75)

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
ofig.sound_metric_scatter(filt_df, ['Fstationary', 'Tstationary', 'bandwidth', 'max_power', 'RMS_power'],
                          'BG_rel_gain', ['Frequency\nNon-Stationarity', 'Temporal\nNon-Stationarity',
                                          'Bandwidth', 'Max Power', 'RMS Power'],
                          jitter=[0.25, 0.2, 0.03, 0.03, 0.0003],
                          area='A1', threshold=0.03, synth_kind='N',
                          title_text='removed low max power FGs')






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


