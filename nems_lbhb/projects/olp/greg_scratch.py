import OLP_analysis as olp

parmfiles = ['/auto/data/daq/Armillaria/ARM013/ARM013b32_p_OLP',   #0
             '/auto/data/daq/Armillaria/ARM015/ARM015b15_p_OLP',   #1
             '/auto/data/daq/Armillaria/ARM029/ARM029a14_p_OLP',   #15
             '/auto/data/daq/Armillaria/ARM030/ARM030a12_p_OLP',   #16
             '/auto/data/daq/Armillaria/ARM031/ARM031a16_p_OLP',   #17
             '/auto/data/daq/Armillaria/ARM032/ARM032a18_p_OLP',   #18
             '/auto/data/daq/Armillaria/ARM033/ARM033a20_p_OLP']   #19

parmfiles = ['/auto/data/daq/Armillaria/ARM016/ARM016c15_p_OLP',  # 2
             '/auto/data/daq/Armillaria/ARM017/ARM017a10_p_OLP',  # 3
             '/auto/data/daq/Armillaria/ARM018/ARM018a05_p_OLP',  # 4
             '/auto/data/daq/Armillaria/ARM019/ARM019a07_p_OLP',  # 5
             '/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP',  # 6
             '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP',  # 7
             '/auto/data/daq/Armillaria/ARM022/ARM022b15_p_OLP',  # 8
             '/auto/data/daq/Armillaria/ARM023/ARM023a11_p_OLP',  # 9
             '/auto/data/daq/Armillaria/ARM024/ARM024a10_p_OLP',  # 10
             '/auto/data/daq/Armillaria/ARM025/ARM025a10_p_OLP',  # 11
             '/auto/data/daq/Armillaria/ARM026/ARM026b07_p_OLP',  # 12
             '/auto/data/daq/Armillaria/ARM027/ARM027a15_p_OLP',  # 13
             '/auto/data/daq/Armillaria/ARM028/ARM028b13_p_OLP']  # 14

parmfiles_A1 = parmfiles[0:2] + parmfiles[15:]
parmfiles_PEG = parmfiles[2:15]

response, params = olp._response_params(parmfile)

supp_array = olp._get_suppression(response, params)
site_results = olp.site_regression(supp_array, params)

import nems_lbhb.gcmodel.figures.snr as snr
snrs = snr.compute_snr(params['response'])
params = olp.load_experiment_params(parmfile, rasterfs=100, sub_spont=True)
response = olp.get_response(params, sub_spont=False)

regression_results = olp.multisite_reg_results(parmfiles)



####
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import scipy.ndimage.filters as sf
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import numpy as np
bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                    f'Background2/*.wav'))
fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                    f'Foreground3/*.wav'))
fgs= fg_dir[:5] #ferrets

fgs = fg_dir[5:15] # marms
names = [aa.split('/')[-1] for aa in fgs]

pl = 0

fig, axes = plt.subplots(5,1, sharey=True)
ax = axes.ravel()
for pl in range(len(fgs)):
    path = fgs[pl]
    sfs, W = wavfile.read(path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 32000)

    # ax[pl].plot(W)
    ax[pl].set_title(f"{names[pl]}")
    ax[pl].imshow(np.log(spec), aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])

fig.tight_layout()

pl = 0

fig, axes = plt.subplots(5,1, sharey=True)
ax = axes.ravel()

for pl in range(len(fgs)):
    path = fgs[pl]
    sfs, W = wavfile.read(path)

    # ax[pl].plot(W)
    ax[pl].set_title(f"{names[pl]}")
    ax[pl].plot(W)

fig.tight_layout()


###


olp.psth_fulls_allunits(1, response, params, 2)
unit = 8
olp.psth_allpairs([2,3], unit, response, params, sigma=2, sum=False)
olp.psth_allpairs([4,1,3], unit, response, params, sigma=2, sum=True)
olp.psth_allpairs([0,5,7], unit, response, params, sigma=2, sum=True)
olp.psth_allpairs([2,7], unit, response, params, sigma=2, sum=False)

olp.z_allpairs([2,3], unit, response, params, sigma=2, z_av=False)
olp.z_allpairs([0,1,2], unit, response, params, sigma=2, z_av=False)
olp.z_allpairs([0,5,7], unit, response, params, sigma=2, z_av=False)
olp.z_allpairs([2,7], unit, response, params, sigma=2, z_av=False)


unit = 1
pair = 1

olp.plot_rasters([0,1,2], 1, unit, response, params, 2)







fig, ax = plt.subplots()
x = np.linspace(0,neuron_coeffs.shape[0],neuron_coeffs.shape[0])
ax.errorbar(x, neuron_coeffs, yerr=neuron_err,
            color='black', label='Full Model', capsize=5)
ax.errorbar(x, neuron_coeffs_shuff, yerr= neuron_err_shuff,
            color='green', label='Stimulus Shuffled', capsize=5)
ax.legend()

fig, ax = plt.subplots()
x = np.linspace(0,stim_coeffs.shape[0],stim_coeffs.shape[0])
ax.errorbar(x, stim_coeffs, yerr=stim_err,
            color='black', label='Full Model', capsize=5)
ax.errorbar(x, stim_coeffs_shuff, yerr= stim_err_shuff,
            color='blue', label='Neuron Shuffled', capsize=5)
ax.legend()

by_neuron = np.stack((neuron_coeffs,neuron_coeffs_shuff), axis=1)
by_stimulus = np.stack((stim_coeffs,stim_coeffs_shuff), axis=1)



#Coefficients
fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,5))
x = np.linspace(0,neur_coeffs.shape[0],neur_coeffs.shape[0])
ax[0].errorbar(x, neur_coeffs, yerr=neuron_err, color='black')
ax[0].set_xlabel('Neuron', fontweight='bold', size=15)
ax[0].set_ylabel('Suppression Units', fontweight='bold', size=15)
ax[0].axhline(0, linestyle=':', color='black')

x = np.linspace(0,stim_coeffs.shape[0],stim_coeffs.shape[0])
ax[1].errorbar(x, stim_coeffs, yerr=stim_err, color='black')
ax[1].set_xlabel('Stimulus Pair', fontweight='bold', size=15)
ax[1].axhline(0, linestyle=':', color='black')

##########
##########
#FIGURE FOR POSTER PSTHS and stuff
import OLP_analysis as olp
parmfile = '/auto/data/daq/Tabor/TBR012/TBR012a14_p_OLP.m'
response, params = olp._response_params(parmfile)
olp.psth_comp_figure([0,1,2], 0, 12, response, params, 2, True)
olp.psth_comp_figure([0,1,2], 0, 20, response, params, 2, True)
olp.z_heatmaps_onepairs_figure([0,1,2], 0, response, params, tags=[12,20], sigma=2, arranged=True)

##Waveform plots for figure 1 of poster
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

#Fulls, probably don't use
path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set3/'
BG_path = path + 'cat312_rec1_wind_excerpt1.wav'
FG_path = path + '00cat669_rec2_marmoset_chirp_excerpt1.wav'

#partials actually used
path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
BG_path = path + 'Background2/10Wind.wav'
FG_path = path + 'Foreground3/07Chirp.wav'

paths = [BG_path, FG_path]
wavs = []
tags = ['Background\n(BG)', 'Foreground\n(FG)', 'BG+FG\nCombo']
tags = ['Sound Texture -\nBackground (BG)', 'Marmoset Vocalization -\nForeground (FG)', 'BG+FG Combination']
colors = ['deepskyblue', 'yellowgreen', 'dimgray']

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6,6))
for nn, pth in enumerate(paths):
    sfs, W = wavfile.read(pth)
    ax[nn].plot(W, color=colors[nn])

    ax[nn].spines['top'].set_visible(False), ax[nn].spines['bottom'].set_visible(False)
    ax[nn].spines['left'].set_visible(False), ax[nn].spines['right'].set_visible(False)
    ax[nn].set_yticks([]), ax[nn].set_xticks([])
    ax[nn].set_title(f"{tags[nn]}", fontweight='bold', fontsize=18)
    # ax[nn].set_ylabel(f"{tags[nn]}", fontweight='bold', fontsize=16, rotation=0, labelpad=25)

    wavs.append(W)

comb = wavs[0] + wavs[1]
ax[2].plot(comb, color=colors[2])
ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)
ax[2].set_yticks([]), ax[2].set_xticks([])
# ax[2].set_ylabel(f"{tags[2]}", fontweight='bold', fontsize=16, rotation=0, labelpad=25)
ax[2].set_title(f"{tags[2]}", fontweight='bold', fontsize=18)
fig.tight_layout()


#wasting my time - or intro figure, whatever you want to call it
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import matplotlib.pyplot as plt

path = '/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set3/'
piano_path = path + 'cat268_rec1_classicalsolo_haydn_piano-sonata-53_24sec_excerpt1.wav'
bass_path = path + 'cat20_rec1_acoustic_bass_gillespie_bass_solo_excerpt1.wav'
violin_path = path + 'cat394_rec1_violin_excerpt1.wav'
paths = [piano_path, violin_path, bass_path]

fig, ax = plt.subplots(5,1, figsize=(7.5,10))
specs = []
for nn, pth in enumerate(paths):
    sfs, W = wavfile.read(pth)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 0, 12000)
    ax[nn].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                  cmap='gray_r')
    # ax[nn].set_xticks([0, 100, 200, 300, 400]), ax[nn].set_yticks([])
    # ax[nn].set_xticklabels([0, 1.0, 2.0, 3.0, 4.0]), ax[nn].set_yticklabels([])
    ax[nn].set_yticks([]), ax[nn].set_xticks([])
    # ax[nn].spines['top'].set_visible(False), ax[nn].spines['bottom'].set_visible(False)
    # ax[nn].spines['left'].set_visible(False), ax[nn].spines['right'].set_visible(False)

    specs.append(spec)

comb = specs[0] + specs[1] + specs[2]
ax[4].imshow(comb, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
             cmap='gray_r')
ax[3].set_yticks([]), ax[3].set_xticks([])
ax[3].spines['top'].set_visible(False), ax[3].spines['bottom'].set_visible(False)
ax[3].spines['left'].set_visible(False), ax[3].spines['right'].set_visible(False)
ax[4].set_yticks([]), ax[4].set_xticks([])
# ax[4].spines['top'].set_visible(False), ax[4].spines['bottom'].set_visible(False)
# ax[4].spines['left'].set_visible(False), ax[4].spines['right'].set_visible(False)
ax[4].set_ylabel('Frequency (Hz)'), ax[4].set_xlabel('Time (s)')
ax[1].set_ylabel('Frequency (Hz)'), ax[2].set_xlabel('Time (s)')

##2/7/22 add to make the waveforms for WIP
fig, ax = plt.subplots(5,1, figsize=(7.5,10), sharey=True)
waves = []
for nn, pth in enumerate(paths):
    sfs, W = wavfile.read(pth)
    ax[nn].plot(W, color='dimgrey', lw=1)
    ax[nn].set_yticks([]), ax[nn].set_xticks([])
    waves.append(W)
comb = waves[0] + waves[1] + waves[2]
ax[4].plot(comb, color='dimgrey', lw=1)
ax[3].set_yticks([]), ax[3].set_xticks([])
ax[3].spines['top'].set_visible(False), ax[3].spines['bottom'].set_visible(False)
ax[3].spines['left'].set_visible(False), ax[3].spines['right'].set_visible(False)
ax[4].set_yticks([]), ax[4].set_xticks([])
ax[4].set_ylabel('Frequency (Hz)'), ax[4].set_xlabel('Time (s)')
ax[1].set_ylabel('Frequency (Hz)'), ax[2].set_xlabel('Time (s)')

####

import OLP_analysis as olp
import numpy as np
import scipy.ndimage.filters as sf
import sys
sys.path.extend(['/auto/users/hamersky/olp'])
import statsmodels.formula.api as smf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# from statsmodels.stats.anova import anova_lm
# print(anova_lm(results))
parmfiles_hood = ['/auto/data/daq/Hood/HOD005/HOD005b09_p_OLP.m',
                  '/auto/data/daq/Hood/HOD006/HOD006b11_p_OLP.m',
                  '/auto/data/daq/Hood/HOD007/HOD007a10_p_OLP.m',
                  '/auto/data/daq/Hood/HOD008/HOD008d11_p_OLP.m',
                  '/auto/data/daq/Hood/HOD009/HOD009a09_p_OLP.m']

parmfiles_tabor = ['/auto/data/daq/Tabor/TBR007/TBR007a10_p_OLP.m',
            '/auto/data/daq/Tabor/TBR008/TBR008a12_p_OLP.m',
            '/auto/data/daq/Tabor/TBR009/TBR009a10_p_OLP.m',
            '/auto/data/daq/Tabor/TBR010/TBR010a11_p_OLP.m',
            '/auto/data/daq/Tabor/TBR011/TBR011a17_p_OLP.m',
            '/auto/data/daq/Tabor/TBR012/TBR012a14_p_OLP.m',
            '/auto/data/daq/Tabor/TBR013/TBR013a15_p_OLP.m',
            '/auto/data/daq/Tabor/TBR017/TBR017a13_a_OLP.m',
            '/auto/data/daq/Tabor/TBR019/TBR019a16_p_OLP.m',
            '/auto/data/daq/Tabor/TBR020/TBR020a16_p_OLP.m',
            '/auto/data/daq/Tabor/TBR021/TBR021a11_p_OLP.m',
            '/auto/data/daq/Tabor/TBR022/TBR022a14_a_OLP.m',
            '/auto/data/daq/Tabor/TBR023/TBR023a14_p_OLP.m',
            '/auto/data/daq/Tabor/TBR025/TBR025a13_p_OLP.m',
            '/auto/data/daq/Tabor/TBR026/TBR026a16_p_OLP.m',
            '/auto/data/daq/Tabor/TBR027/TBR027a14_p_OLP.m',
            '/auto/data/daq/Tabor/TBR028/TBR028a08_p_OLP.m',
            '/auto/data/daq/Tabor/TBR030/TBR030a13_p_OLP.m',
            '/auto/data/daq/Tabor/TBR031/TBR031a13_p_OLP.m',
            '/auto/data/daq/Tabor/TBR034/TBR034a14_p_OLP.m',
            '/auto/data/daq/Tabor/TBR035/TBR035a15_p_OLP.m',
            '/auto/data/daq/Tabor/TBR036/TBR036a14_p_OLP.m']

parmfile = '/auto/data/daq/Tabor/TBR007/TBR007a10_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR008/TBR008a12_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR009/TBR009a10_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR010/TBR010a11_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR011/TBR011a17_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR012/TBR012a14_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR013/TBR013a15_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR017/TBR017a13_a_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR019/TBR019a16_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR020/TBR020a16_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR021/TBR021a11_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR022/TBR022a14_a_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR023/TBR023a14_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR025/TBR025a13_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR026/TBR026a16_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR027/TBR027a14_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR028/TBR028a08_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR030/TBR030a13_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR031/TBR031a13_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR034/TBR034a14_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR035/TBR035a15_p_OLP.m'
parmfile = '/auto/data/daq/Tabor/TBR036/TBR036a14_p_OLP.m'

regression_stuff(parmfile, plot=True, dataframe='none')
Pred = regression_stuff(parmfile, plot=False, dataframe='none')

neur_pred = np.mean(Pred, axis=1)
neur_sem = stats.sem(Pred, axis=1)
stim_pred = np.mean(Pred, axis=0)
stim_sem = stats.sem(Pred, axis=0)


fig, ax = plt.subplots(1,2, figsize=(7,3), sharey=True)
ax[0].errorbar(range(Pred.shape[0]), neur_pred*-1, yerr=neur_sem, marker='.', ls='none', color='black')
ax[1].errorbar(range(Pred.shape[1]), stim_pred*-1, yerr=stim_sem, marker='.', ls='none', color='black')
ax[0].set_xlabel('Neuron', fontweight='bold', size=15)
ax[0].set_ylabel('Mean Weight', fontweight='bold', size=15)
ax[0].axhline(0, linestyle=':', color='black')
ax[1].set_xlabel('Stimulus Pair', fontweight='bold', size=15)
ax[1].axhline(0, linestyle=':', color='black')

fig.tight_layout()



plt.figure()
plt.plot(mean_pred)

parmfiles = parmfiles_hood + parmfiles_tabor
#make plot of all r squared values across sites
plot_all_rsquares(parmfiles, emp=)


full_results = pd.DataFrame()
for pp in parmfiles:
    full_results = regression_stuff(pp, plot=False, rdf=None, full_results=full_results, dataframe='all')


stimmies = full_results.loc[full_results.id=='stimulus']
stimmies = stimmies.loc[stimmies.coeff<=2]
neuries = full_results.loc[full_results.id=='neuron']
neuries = neuries.loc[neuries.coeff<=2]


stcos = np.asarray(stimmies.coeff)
neucos = np.asarray(neuries.coeff)

fig, ax = plt.subplots(figsize=(5,5))
ax.hist([neucos,stcos], bins=100)
ax.hist(stcos, bins=100)


#Plots the R values atop one another
def plot_all_rsquares(parmfiles, units=10, emp=None):
    rdf = pd.DataFrame(columns=['site', 'units', 'full', 'neuron', 'stimulus'])
    for pp in parmfiles:
        rdf = regression_stuff(pp, plot=False, dataframe='r', rdf=rdf)

    rdf = rdf.loc[rdf['units'] >= units].copy()

    fig, ax = plt.subplots(figsize=(3, 4.5))
    fig.subplots_adjust(left=0.235)

    toplot = rdf.loc[:, ['stimulus', 'neuron', 'full']].values.T
    ax.plot(toplot, linestyle='-', color='lightgrey')

    if emp:
        ss = rdf.loc[rdf.site==emp]
        tpl = ss.loc[:, ['stimulus', 'neuron', 'full']].values.T
        ax.plot(tpl, ls='-', color='darkgrey')

    meanr = np.mean(toplot, axis=1)
    error = stats.sem(toplot, axis=1)
    ax.errorbar([0, 1, 2], meanr, yerr=error, lw=2, elinewidth=2, linestyle='-', color='black')

    ax.set_ylabel('R_squared', fontweight='bold', size=12)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Neuron\nShuffled', 'Stimulus\nShuffled', 'Full\nModel'],
                       fontweight='bold', size=8)
    ax.axhline(0, linestyle=':', color='black')
    ax.set_title(f"All Sites (n={toplot.shape[1]})", fontweight='bold', size=12)


def regression_stuff(parmfile, plot=True, rdf=None, full_results=None, dataframe='all'):
    import seaborn as sns

    response, params = olp._response_params(parmfile)
    pairs, units = response.shape[0], response.shape[3]
    supp_array = olp._get_suppression(response, params)
    y = supp_array.reshape(1, -1)  # flatten
    stimulus = np.tile(np.arange(0, supp_array.shape[1]), supp_array.shape[0])
    neuron = np.concatenate([np.ones(supp_array.shape[1]) * i for i in
                             range(supp_array.shape[0])], axis=0)

    X = np.stack([neuron, stimulus])
    X = pd.DataFrame(data=X.T, columns=['neuron', 'stimulus'])
    X = sm.add_constant(X)
    X['suppression'] = y.T

    results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=X).fit()

    Xshuff = X.copy()
    Xshuff['neuron'] = Xshuff['neuron'].iloc[np.random.choice(
        np.arange(X.shape[0]), X.shape[0], replace=False)].values
    results_neushuf = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    Xshuff = X.copy()
    Xshuff['stimulus'] = Xshuff['stimulus'].iloc[np.random.choice(
        np.arange(X.shape[0]), X.shape[0], replace=False)].values
    results_stimshuf = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    full_pred = results.predict(X)
    neur_pred = results_stimshuf.predict(X)
    stim_pred = results_neushuf.predict(X)


    full_pred = full_pred.values.reshape(units,pairs)
    neur_pred = neur_pred.values.reshape(units,pairs)
    stim_pred = stim_pred.values.reshape(units,pairs)

    vmax = np.max([np.max(supp_array),np.max(full_pred),np.max(neur_pred),np.max(stim_pred)])
    vmin = np.min([np.min(supp_array),np.min(full_pred),np.min(neur_pred),np.min(stim_pred)])
    lims = max(abs(vmax),abs(vmin))

    full_r = results.rsquared
    neur_r = results_stimshuf.rsquared
    stim_r = results_neushuf.rsquared

    if plot==True:
        fig, ax = plt.subplots(1,4, figsize=(15,3))
        ax[0].imshow(supp_array*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='coolwarm')
        ax[0].set_title('Actual Values', fontweight='bold', size=12)
        ax[0].set_ylabel('Neurons', fontweight='bold', fontsize=10)
        ax[1].imshow(full_pred*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='coolwarm')
        ax[1].set_title('Full Model', fontweight='bold', size=12)
        ax[2].imshow(neur_pred*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='coolwarm')
        ax[2].set_title('Stimulus Shuffle', fontweight='bold', size=12)
        ax[3].imshow(stim_pred*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='coolwarm')
        ax[3].set_title('Neuron Shuffle', fontweight='bold', size=12)
        plt.xlabel("Stimulus Pair", fontweight='bold', fontsize=10)
        fig.subplots_adjust(bottom=0.235)

        #RdBu_r bwr

        fig, ax = plt.subplots()
        ax.plot([stim_r, neur_r, full_r], linestyle='-', color='black')
        ax.set_ylabel('R_squared', fontweight='bold', size=12)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Neuron\nShuffled', 'Stimulus\nShuffled', 'Full\nModel'],
                                fontweight='bold', size=8)
        ax.axhline(0, linestyle=':', color='black')
        ax.set_title(f"{params['experiment']}", fontweight='bold', size=15)

        consta = results.params.const

        site_results = olp.site_regression(supp_array, params)
        results_full = site_results.loc[site_results.shuffle=='full']
        results_neur = site_results.loc[site_results.shuffle=='neuron']
        results_stim = site_results.loc[site_results.shuffle=='stimulus']

        neur_coeffs = results_full.loc[results_full.id=='neuron'].coeff
        stim_coeffs = results_full.loc[results_full.id=='stimulus'].coeff
        neuron_err = results_full.loc[results_full.id=='neuron'].error
        stim_err = results_full.loc[results_full.id=='stimulus'].error

        #Coefficients
        fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,5))
        x = np.linspace(0,neur_coeffs.shape[0],neur_coeffs.shape[0])
        ax[0].errorbar(x, neur_coeffs, yerr=neuron_err, ls='none', marker='.', color='black')
        ax[0].set_xlabel('Neuron', fontweight='bold', size=15)
        ax[0].set_ylabel('Suppression Units', fontweight='bold', size=15)
        ax[0].axhline(0, linestyle=':', color='black')

        x = np.linspace(0,stim_coeffs.shape[0],stim_coeffs.shape[0])
        ax[1].errorbar(x, stim_coeffs, yerr=stim_err, ls='none', marker='.', color='black')
        ax[1].set_xlabel('Stimulus Pair', fontweight='bold', size=15)
        ax[1].axhline(0, linestyle=':', color='black')

        # fig, ax = plt.subplots(figsize=(5,5))
        # ax.hist(neur_coeffs)

        fig,ax = plt.subplots()
        sns.swarmplot(x='id', y='coeff', hue='id', data=results_full)

    if dataframe=='r':
        rdf = rdf.append({'site': params['experiment'],
                          'units': len(params['good_units']),
                          'full': full_r, 'neuron': neur_r, 'stimulus': stim_r}, ignore_index=True)
        print(f"{rdf.iloc[-1]}")

        return rdf

    elif dataframe=='none':
        fig, ax = plt.subplots(1,2, figsize=(7.5,3))
        ax[0].imshow(supp_array*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='coolwarm')
        ax[0].set_title('Actual Values', fontweight='bold', size=12)
        ax[1].imshow(full_pred*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='coolwarm')
        ax[1].set_title('Full Model', fontweight='bold', size=12)

        return full_pred

    elif dataframe=='all' and len(params['good_units'])>=10:
        site_results = olp.site_regression(supp_array, params)
        results_full = site_results.loc[site_results.shuffle=='full']
        results_full['site'] = params['experiment']
        full_results = full_results.append(results_full)
        print(f"Adding site {params['experiment']} to dataframe.")

        return full_results

    else:
        print(f"Site {params['experiment']} only has {len(params['good_units'])} good"
              f"units, not adding to dataframe.")

        return full_results


def regression_stuff(parmfile, plot=True, rdf=None):
    response, params = olp._response_params(parmfile)
    pairs, units = response.shape[0], response.shape[3]
    supp_array = olp._get_suppression(response, params)
    y = supp_array.reshape(1, -1)  # flatten
    stimulus = np.tile(np.arange(0, supp_array.shape[1]), supp_array.shape[0])
    neuron = np.concatenate([np.ones(supp_array.shape[1]) * i for i in
                             range(supp_array.shape[0])], axis=0)

    X = np.stack([neuron, stimulus])
    X = pd.DataFrame(data=X.T, columns=['neuron', 'stimulus'])
    X = sm.add_constant(X)
    X['suppression'] = y.T

    results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=X).fit()

    Xshuff = X.copy()
    Xshuff['neuron'] = Xshuff['neuron'].iloc[np.random.choice(
        np.arange(X.shape[0]), X.shape[0], replace=False)].values
    results_neushuf = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    Xshuff = X.copy()
    Xshuff['stimulus'] = Xshuff['stimulus'].iloc[np.random.choice(
        np.arange(X.shape[0]), X.shape[0], replace=False)].values
    results_stimshuf = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    # results = olp._regression_results(results, None, params)
    # results_neushuf = olp._regression_results(results_neushuf, None, params)
    # results_stimshuf = olp._regression_results(results_stimshuf, None, params)

    full_pred = results.predict(X)
    neur_pred = results_stimshuf.predict(X)
    stim_pred = results_neushuf.predict(X)


    full_pred = full_pred.values.reshape(units,pairs)
    neur_pred = neur_pred.values.reshape(units,pairs)
    stim_pred = stim_pred.values.reshape(units,pairs)

    vmax = np.max([np.max(supp_array),np.max(full_pred),np.max(neur_pred),np.max(stim_pred)])
    vmin = np.min([np.min(supp_array),np.min(full_pred),np.min(neur_pred),np.min(stim_pred)])
    lims = max(abs(vmax),abs(vmin))

    full_r = results.rsquared
    neur_r = results_stimshuf.rsquared
    stim_r = results_neushuf.rsquared

    if plot==True:
        fig, ax = plt.subplots(1,4, figsize=(15,3))
        ax[0].imshow(supp_array*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='bwr')
        ax[0].set_title('Actual Values', fontweight='bold', size=12)
        ax[1].imshow(full_pred*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='bwr')
        ax[1].set_title('Full Model', fontweight='bold', size=12)
        ax[2].imshow(neur_pred*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='bwr')
        ax[2].set_title('Stimulus Shuffle', fontweight='bold', size=12)
        ax[3].imshow(stim_pred*-1,aspect='auto', vmin=-lims, vmax=lims, cmap='bwr')
        ax[3].set_title('Neuron Shuffle', fontweight='bold', size=12)
        #RdBu_r

        fig, ax = plt.subplots()
        ax.plot([stim_r, neur_r, full_r], linestyle='-', color='black')
        ax.set_ylabel('R_squared', fontweight='bold', size=12)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Neuron\nShuffled', 'Stimulus\nShuffled', 'Full\nModel'],
                                fontweight='bold', size=8)
        ax.axhline(0, linestyle=':', color='black')
        ax.set_title(f"{params['experiment']}", fontweight='bold', size=15)

        consta = results.params.const

        site_results = olp.site_regression(supp_array, params)
        results_full = site_results.loc[site_results.shuffle=='full']
        results_neur = site_results.loc[site_results.shuffle=='neuron']
        results_stim = site_results.loc[site_results.shuffle=='stimulus']

        neur_coeffs = results_full.loc[results_full.id=='neuron'].coeff
        stim_coeffs = results_full.loc[results_full.id=='stimulus'].coeff
        neuron_err = results_full.loc[results_full.id=='neuron'].error
        stim_err = results_full.loc[results_full.id=='stimulus'].error

        #Coefficients
        fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,5))
        x = np.linspace(0,neur_coeffs.shape[0],neur_coeffs.shape[0])
        ax[0].errorbar(x, neur_coeffs, yerr=neuron_err, ls='none', marker='.', color='black')
        ax[0].set_xlabel('Neuron', fontweight='bold', size=15)
        ax[0].set_ylabel('Suppression Units', fontweight='bold', size=15)
        ax[0].axhline(0, linestyle=':', color='black')

        x = np.linspace(0,stim_coeffs.shape[0],stim_coeffs.shape[0])
        ax[1].errorbar(x, stim_coeffs, yerr=stim_err, ls='none', marker='.', color='black')
        ax[1].set_xlabel('Stimulus Pair', fontweight='bold', size=15)
        ax[1].axhline(0, linestyle=':', color='black')

    else:
        rdf = rdf.append({'site': params['experiment'],
                          'units': len(params['good_units']),
                          'full': full_r, 'neuron': neur_r, 'stimulus': stim_r}, ignore_index=True)
        print(f"{rdf.iloc[-1]}")

        return rdf

####
####


fig, ax = plt.subplots(figsize=(3,5))
ax.plot([stim_r, neur_r, full_r], linestyle='-', color='black')
ax.set_ylabel('R_squared', fontweight='bold', size=12)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Neuron\nShuffled','Stimulus\nShuffled','Full Model'],
                   fontweight='bold', size=10)
ax.axhline(0, linestyle=':', color='black')
fig.tight_layout()

plt.figure()
plt.plot([stim_r,neur_r,full_r])
plt.axhline(0, linestyle=':')


def get_rsquared():
    fig, ax = plt.subplots(1,len(parmfiles), sharey=True, figsize=(15, 5))

    suppression = {}
    for site in parmfiles:
        params = olp.load_experiment_params(site, rasterfs=100, sub_spont=True)
        response = olp.get_response(params, sub_spont=False)
        corcoefs = olp._base_reliability(response, rep_dim=2, protect_dim=3)
        avg_resp = olp._significant_resp(response, params, protect_dim=3, time_dim=-1)
        response = olp._find_good_units(response, params,
                                        corcoefs=corcoefs, corcoefs_threshold=0.1,
                                        avg_resp=avg_resp, avg_threshold=0.2)

        supp_array = np.empty([len(params['good_units']), len(params['pairs'])])
        for nn, pp in enumerate(params['pairs']):
            _, _, _, _, supp, _, _ = olp.get_scatter_resps(nn, response)
            supp_array[:, nn] = supp

        suppression[f"{params['experiment']}"] = supp_array

    for cnt, (site, supp_array) in enumerate(suppression.items()):
        y = supp_array.reshape(1, -1)  # flatten
        stimulus = np.tile(np.arange(0, supp_array.shape[1]), supp_array.shape[0])
        neuron = np.concatenate([np.ones(supp_array.shape[1]) * i for i in
                                 range(supp_array.shape[0])], axis=0)

        X = np.stack([neuron, stimulus])
        X = pd.DataFrame(data=X.T, columns=['neuron', 'stimulus'])
        X = sm.add_constant(X)
        X['suppression'] = y.T

        results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=X).fit()


        Xshuff = X.copy()
        Xshuff['neuron'] = Xshuff['neuron'].iloc[np.random.choice(
            np.arange(X.shape[0]), X.shape[0], replace=False)].values
        results_neushuf = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

        Xshuff = X.copy()
        Xshuff['stimulus'] = Xshuff['stimulus'].iloc[np.random.choice(
            np.arange(X.shape[0]), X.shape[0], replace=False)].values
        results_stimshuf = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

        full_r = results.rsquared
        neur_r = results_stimshuf.rsquared
        stim_r = results_neushuf.rsquared

        ax[cnt].plot([stim_r, neur_r, full_r], linestyle='-', color='black')
        ax[0].set_ylabel('R_squared', fontweight='bold', size=12)
        ax[cnt].set_xticks([0, 1, 2])
        ax[cnt].set_xticklabels(['Neuron\nShuffled', 'Stimulus\nShuffled', 'Full\nModel'],
                           fontweight='bold', size=8)
        ax[cnt].axhline(0, linestyle=':', color='black')
        ax[cnt].set_title(f"{site}", fontweight='bold', size=15)
    fig.tight_layout()



##SUPPRESSION LINE PLOT
import matplotlib.pyplot as plt
import numpy as np
supp_array = np.empty([len(params['good_units']), len(params['pairs'])])
for nn, pp in enumerate(params['pairs']):
    _, _, _, _, supp, _, _ = olp.get_scatter_resps(nn, response)
    supp_array[:,nn] = supp

fig, ax = plt.subplots()
ax.plot(supp_array.T, marker='o', linestyle=':')
ax.set_xticks([*range(len(params['pairs']))])
ax.set_xticklabels(params['pairs'], ha='right', rotation=40)
ax.set_ylabel('Suppression\n(BG + FG) - BG/FG combo')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.suptitle(f"Experiment {params['experiment']}", fontweight='bold')
fig.tight_layout()


olp.psth_comp([0,1,2], pair, unit, response, params, sigma=2, z=True, sum=True)
pair = 1
#Scatter plots
#bgfg v suppression
olp.bgfg_suppression_scatter(pair, response, params)
#bg v fg
olp.bg_fg_scatter(pair, response, params)
#lin v combo
olp.lin_combo_scatter(pair, response, params)
#bg v fg and lin v combo
olp.bgfg_lincombo_scatter(pair, response, params)


#Big overview of data, plots all full BG and FG responses for all units
olp.psth_fulls_allunits(pair, response, params, 2)


#Plots a bunch of heat maps to get overview of data
# [2,3] (hBG,fFG) - [2,7] (fBG, hFG) - [0,1,2] fBG/fFG - [1,4,3] (hBG,fFG)
olp.z_heatmaps_allpairs([4,1,3], response, params, 2)
olp.z_heatmaps_allpairs([0,1,2], response, params, 2, arranged=True)
olp.z_heatmaps_allpairs([0,1,2], response, params, 2, arranged=False)
olp.z_heatmaps_allpairs([0,5,7], response, params, 2)


pair, unit = 1, 1
olp.plot_combos(pair, unit, response, params, 2)
olp.psth_allpairs([0,1,2], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([0,1,2], unit, response, params, sigma=2, z_av=False)

olp.psth_allpairs([4,1,3], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([4,1,3], unit, response, params, sigma=2, z_av=False)

olp.psth_allpairs([0,5,7], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([0,5,7], unit, response, params, sigma=2, z_av=False)

olp.psth_allunits([0,1,2], pair, response, params, sigma=2, sum=False)




z, zp, z2, zp2 = olp.z_bgfg_compare([0,2], [1,2], 2, response, params, 2)
olp.z_bgfg_compare2(z,zp,z2,zp2,unit,response,params,2)

import matplotlib.pyplot as plt
import numpy as np

pca_resp = np.nanmean(response[3,1,:,:,:], axis=0)

olp.plot_auc_mean([0,1,2], response, params)
olp.plot_auc_mean([0,5,7], response, params)
olp.plot_auc_mean([4,1,3], response, params)



#####################################
#This one does the same as bgfg_suppression_scatter but in subplots rather than on one
fig, ax = plt.subplots()
for pnt in range(supp.shape[0]):
    ax.plot(bg_fg[pnt,:], supp_supp[pnt, :], marker='o', linestyle=':',
            zorder=-1, label=params['good_units'][pnt])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()

fig, ax = plt.subplots(1, 2, sharey=True)
for pnt in range(supp.shape[0]):
    ax[0].plot(mean_bg[pnt], supp[pnt], marker='o', linestyle='None',
               label=params['good_units'][pnt])
    ax[1].plot(mean_fg[pnt], supp[pnt], marker='o', linestyle='None',
               label=params['good_units'][pnt])
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax[0].set_ylabel('Suppression\n(BG + FG) - BG/FG combo')
ax[0].set_xlabel('BG Response'), ax[1].set_xlabel('FG Response')
ax[0].set_box_aspect(1), ax[1].set_box_aspect(1)
fig.suptitle(f"Experiment {params['experiment']} - Pair {pair}\n"
             f"Background {params['pairs'][pair][0]} - Foreground {params['pairs'][pair][1]}",
             fontweight='bold')
fig.tight_layout()
#####################################



colors = ['deepskyblue', 'yellowgreen', 'dimgray']
pre_bin = int(params['PreStimSilence'] * params['fs'])
post_bin = int((params['stim length'] - params['PostStimSilence']) * params['fs'])
for cnt, cmb in enumerate(combo):
    resp_sub = np.nanmean(response[:, cmb, :, :, int(pre_bin):int(post_bin)], axis=0)
    mean_resp = np.nanmean(resp_sub, axis=0)
    x = np.linspace(0, resp_sub.shape[-1] / params['fs'], resp_sub.shape[-1]) \
        - params['PreStimSilence']

    auc = np.sum(mean_resp, axis=1)
    center = np.sum(mean_resp * x, axis=1) / np.sum(mean_resp, axis=1)
    plt.plot(auc, center, marker='o', linestyle='None', color=colors[cnt],
             label=params['combos'][cmb])
    plt.xlabel('Area Under Curve'), plt.ylabel('Center')
    plt.title(f"Experiment {params['experiment']} - Combos {combo}")
    plt.legend()


#function to see what pairs used

pair_dict = {}
for cnt, parmfile in enumerate(parmfiles):
    params = olp.load_experiment_params(parmfile, rasterfs=100, sub_spont=True)
    bgs = np.expand_dims(np.asarray(params['Background']), axis=1)
    fgs = np.expand_dims(np.asarray(params['Foreground']), axis=1)
    pairs = np.concatenate((bgs, fgs), axis=1)
    pair_dict[f"{params['experiment']}"] = pairs

fig, ax = plt.subplots()
# for pp in range(pair_dict['ARM020'].shape[0]):
for cnt, site in enumerate(pair_dict.keys()):
    ax.plot(pair_dict[site][:,0], pair_dict[site][:,1], marker='o', linestyle='None',
        label=site)


##For when I'm doing batch stuff.
parmfiles = ['/auto/data/daq/Armillaria/ARM013/ARM013b32_p_OLP',
             '/auto/data/daq/Armillaria/ARM015/ARM015b15_p_OLP',
             '/auto/data/daq/Armillaria/ARM016/ARM016c15_p_OLP',
             '/auto/data/daq/Armillaria/ARM017/ARM017a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM018/ARM018a05_p_OLP',
             '/auto/data/daq/Armillaria/ARM019/ARM019a07_p_OLP',
             '/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP',
             '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP',
             '/auto/data/daq/Armillaria/ARM022/ARM022b15_p_OLP',
             '/auto/data/daq/Armillaria/ARM023/ARM023a11_p_OLP',
             '/auto/data/daq/Armillaria/ARM024/ARM024a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM025/ARM025a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM026/ARM026b07_p_OLP',
             '/auto/data/daq/Armillaria/ARM027/ARM027a15_p_OLP',
             '/auto/data/daq/Armillaria/ARM028/ARM028b13_p_OLP',
             '/auto/data/daq/Armillaria/ARM029/ARM029a14_p_OLP',
             '/auto/data/daq/Armillaria/ARM030/ARM030a12_p_OLP',
             '/auto/data/daq/Armillaria/ARM031/ARM031a16_p_OLP',
             '/auto/data/daq/Armillaria/ARM032/ARM032a18_p_OLP',
             '/auto/data/daq/Armillaria/ARM033/ARM033a20_p_OLP']

parmfiles = ['/auto/data/daq/Armillaria/ARM015/ARM015b15_p_OLP',
             '/auto/data/daq/Armillaria/ARM017/ARM017a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP',
             '/auto/data/daq/Armillaria/ARM024/ARM024a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM025/ARM025a10_p_OLP']
responses, parameters = olp.load_parms(parmfiles)

olp.multi_exp_auccenter(0, responses, parameters)
olp.multi_exp_auccenter(1, responses, parameters)
olp.multi_exp_auccenter(2, responses, parameters)
olp.multi_exp_auccenter(3, responses, parameters)
olp.multi_exp_auccenter(7, responses, parameters)

import matplotlib.pyplot as plt
import numpy as np

combos = [0,1,2,3,4,5,6,7]

threshold = 3
markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd', '*']
all_sites_arrays = list()
all_units_ids = list()
colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray', 'olive']
all_aucs, all_centers = {}, {}
all_sites_centers = list()

fig, ax = plt.subplots()

for cnt, exp in enumerate(responses.keys()):
    auc_array = np.empty([5, responses[exp].shape[-2]])
    center_array = np.empty([5, responses[exp].shape[-2]])
    for cmbnum, cmb in enumerate(combos):
        pre_bin = int(parameters[exp]['PreStimSilence'] * parameters[exp]['fs'])
        post_bin = int(responses[exp].shape[-1] - (parameters[exp]['PostStimSilence'] * parameters[exp]['fs']))
        resp_sub = np.nanmean(responses[exp][:, cmb, :, :, :], axis=0)
        mean_resp = np.nanmean(resp_sub[..., int(pre_bin):int(post_bin)], axis=0)

        x = np.linspace(0, mean_resp.shape[-1] / parameters[exp]['fs'], mean_resp.shape[-1])
        center = np.sum(np.abs(mean_resp) * x, axis=1) / np.sum(np.abs(mean_resp), axis=1)

        auc = np.sum(mean_resp, axis=1)
        auc_array[cmbnum,:] = auc
        center_array[cmbnum,:] = center

    resp_mask = np.abs(auc_array) > threshold
    resp_mask = np.any(resp_mask, axis=0)
    auc_array = auc_array[:, resp_mask]
    center_array = center_array[:, resp_mask]

    all_units_ids.append(np.asarray(parameters[exp]['good_units'])[resp_mask])
    all_sites_arrays.append(auc_array)
    all_sites_centers.append(center_array)

    all_aucs[exp] = auc_array
    all_centers[exp] = center_array

    ax.plot(auc_array, marker=markers[cnt], linestyle='--', color=colors[cnt])
    ax.scatter([0],auc_array[0,0], color=colors[cnt], label=exp)

all_sites_arrays = np.concatenate(all_sites_arrays, axis=1)
all_units_ids = np.concatenate(all_units_ids, axis=0)
ax.set_ylabel('Area Under Curve')
ax.set_xticks([0,1.0,2.0,3.0,4.0])
ax.set_xticklabels(['Full BG','Full FG','Full BG/Full FG','Half BG/Full FG','Full BG/Half FG'])
ax.set_title(f'Threshold |AUC| > {threshold}')
_ = ax.legend()

plot_combos = [3,1,3]
label_combos = [combos[aa] for aa in plot_combos]
site = 'ARM020'
colors = ['deepskyblue', 'yellowgreen', 'dimgray']

fig, ax = plt.subplots()
for ct, cb in enumerate(plot_combos):
    ax.plot(all_aucs[site][ct], all_centers[site][ct], linestyle='None', marker='o',
            label=parameters[site]['combos'][label_combos[ct]], color=colors[ct])
    ax.legend()
ax.set_xlabel('Area Under Curve'), ax.set_ylabel('Center of Mass')
plt.title(f"Experiment {parameters[site]['experiment']} - Combos {label_combos}")


combo = 4
lab_comb = combos[combo]
fig, ax = plt.subplots()
for ct, site in enumerate(all_aucs.keys()):
    ax.plot(all_aucs[site][ct], all_centers[site][ct], linestyle='None', marker=markers[ct],
            label=site)
    ax.legend()
ax.set_title(f"Combo {lab_comb} {parameters[site]['combos'][lab_comb]}")

# all_sites_arrays = np.concatenate(all_sites_arrays, axis=1)
# plt.plot(all_sites_arrays, marker=markers[cnt], linestyle='--', label=exp)



def z_heatmaps_onepairs(resp_idx, pair, response, params, sigma=None, arranged=False):
    """Plots a two column figure of subplots, one for each sound pair, displaying a heat map
    of the zscore for all the units."""
    zscore, z_params = olp.get_z(resp_idx, response, params)

    if sigma is not None:
        zscore = sf.gaussian_filter1d(zscore, sigma, axis=2)
        zmin, zmax = np.min(np.min(zscore, axis=2)), np.max(np.max(zscore, axis=2))
        abs_max = max(abs(zmin),zmax)
    else:
        zmin, zmax = np.min(np.min(zscore, axis=1)), np.max(np.max(zscore, axis=1))
        abs_max = max(abs(zmin),zmax)

    if arranged:
        prebin = int(params['PreStimSilence'] * params['fs'])
        postbin = int((params['stim length'] - params['PostStimSilence']) * params['fs'])
        z_time_avg = np.nanmean(zscore[pair,:,prebin:postbin], axis=1)
        idx = np.argsort(z_time_avg)
        zscore = zscore[:, idx, :]

    fig, ax = plt.subplots()

    im = ax.imshow(zscore[pair, :, :], aspect='auto', cmap='RdBu_r',
              extent=[-0.5, (zscore[pair, :, :].shape[1] / params['fs']) -
                      0.5, zscore[pair, :, :].shape[0], 0], vmin=-abs_max, vmax=abs_max)
    ax.set_title(f"Pair {pair}: BG {params['pairs'][pair][0]} - FG {params['pairs'][pair][1]}",
                 fontweight='bold')
    ymin, ymax = ax.get_ylim()
    ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles='--', lw=1)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin + 0.3, xmax - 0.2)
    ax.set_xticks([0, 0.5, 1.0])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # fig.text(0.5, 0.03, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.05, 0.5, 'Neurons', ha='center', va='center', rotation='vertical', fontweight='bold')

    # fig.suptitle(f"Experiment {params['experiment']} - Combo Index {z_params['resp_idx']} - "
    #              f"{z_params['idx_names']} - Sigma {sigma}\n"
    #              f"{z_params['label']}", fontweight='bold')



#plot electrode shank
from nems_lbhb.plots import plot_weights_64D
plot_weights_64D(np.zeros(64),[f'AMT001a-{x}-1' for x in range(64)])