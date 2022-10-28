
####################################################
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import numpy as np
import seaborn as sb
import scipy.stats as sst
from nems_lbhb.gcmodel.figures.snr import compute_snr
from nems0.preprocessing import generate_psth_from_resp
import scipy.ndimage.filters as sf
from pathlib import Path
import os
import matplotlib.pyplot as plt
from nems_lbhb.baphy_experiment import BAPHYExperiment
import numpy as np
import sys
sys.path.insert(0,'/auto/users/luke/Code/Python/Utilities')
import logging

log = logging.getLogger(__name__)

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))

parmfiles = ['/auto/data/daq/Tabor/TBR007/TBR007a10_p_OLP.m',
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

# OLP_cell_metrics_db_path='/auto/users/luke/Projects/OLP/NEMS/celldat_A1_v1.h5'
#
# parm = '/auto/data/daq/Tabor/TBR011/TBR011a17_p_OLP.m'
#
parm = '/auto/data/daq/Tabor/TBR007/TBR007a10_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR008/TBR008a12_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR009/TBR009a10_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR010/TBR010a11_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR011/TBR011a17_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR012/TBR012a14_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR013/TBR013a15_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR017/TBR017a13_a_OLP.m'
parm = '/auto/data/daq/Tabor/TBR019/TBR019a16_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR020/TBR020a16_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR021/TBR021a11_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR022/TBR022a14_a_OLP.m'
parm = '/auto/data/daq/Tabor/TBR023/TBR023a14_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR025/TBR025a13_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR026/TBR026a16_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR027/TBR027a14_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR028/TBR028a08_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR030/TBR030a13_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR031/TBR031a13_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR034/TBR034a14_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR035/TBR035a15_p_OLP.m'
parm = '/auto/data/daq/Tabor/TBR036/TBR036a14_p_OLP.m'

def get_pairs(parm, rasterfs=100):
    expt = BAPHYExperiment(parm)
    rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
    resp = rec['resp'].rasterize()
    expt_params = expt.get_baphy_exptparams()  # Using Charlie's manager
    ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
    soundies = list(ref_handle['SoundPairs'].values())
    pairs = [tuple([j for j in (soundies[s]['bg_sound_name'].split('.')[0],
                                          soundies[s]['fg_sound_name'].split('.')[0])])
                       for s in range(len(soundies))]
    for c, p in enumerate(pairs):
        print(f"{c} - {p}")
    print(f"There are {len(resp.chans) - 1} units and {len(pairs)} sound pairs.")
    print("Returning one value less than channel and pair count.")

    return (len(pairs)-1), (len(resp.chans)-1)

for parm in parmfiles:
    rasterfs = 100
    pp, cc = get_pairs(parm)
    # pair_idx = 0
    # cell = 1

    for c1 in range(cc):
        for p1 in range(pp):
            psth_responses(parm, p1, c1, sigma=1.5, save=True)




def psth_responses(parm, pair_idx, cell, sigma=2, save=False, rasterfs=100):
    expt = BAPHYExperiment(parm)
    rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
    resp = rec['resp'].rasterize()

    site, unit = expt.siteid[:-1], resp.chans[cell]
    if len(resp.chans) <= cell:
        raise ValueError(f"Cell {cell} is out of range for site with {len(resp.chans) - 1} units")

    expt_params = expt.get_baphy_exptparams()  # Using Charlie's manager
    ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
    soundies = list(ref_handle['SoundPairs'].values())
    pairs = [tuple([j for j in (soundies[s]['bg_sound_name'].split('.')[0],
                                soundies[s]['fg_sound_name'].split('.')[0])])
             for s in range(len(soundies))]

    if len(pairs) <= pair_idx:
        raise ValueError(f"Pair_idx {pair_idx} is out of range for unit with {len(pairs) - 1} sound pairs")

    BG, FG = pairs[pair_idx]
    colors = ['deepskyblue', 'yellowgreen', 'grey', 'silver']
    bg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/Background2/{BG}.wav'
    fg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/Foreground3/{FG}.wav'

    #turn pairs into format for epochs, not file getting
    for c, t in enumerate(pairs):
        pairs[c] = tuple([ss.replace(' ', '') for ss in t])
    BG, FG = pairs[pair_idx]
    epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1',
              f'STIM_{BG}-0.5-1_{FG}-0-1']

    prestim = ref_handle['PreStimSilence']

    f = plt.figure(figsize=(15,9))
    psth = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=5)
    specBG = plt.subplot2grid((4, 5), (2, 0), rowspan=1, colspan=5)
    specFG = plt.subplot2grid((4, 5), (3, 0), rowspan=1, colspan=5)

    ax = [psth, specBG, specFG]

    r = resp.extract_epochs(epochs)

    time = (np.arange(0, r[epochs[0]].shape[-1]) / rasterfs ) - prestim

    for e, c in zip(epochs, colors):
        ax[0].plot(time, sf.gaussian_filter1d(np.nanmean(r[e][:,cell,:], axis=0), sigma=sigma)
             * rasterfs, color=c, label=e)
    ax[0].legend()
    ax[0].set_title(f"{resp.chans[cell]} - Pair {pair_idx} - BG: {BG} - FG: {FG} - sigma={sigma}", weight='bold')
    # ax[0].set_xlim([0-(prestim/2), time[-1]])
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymin, ymax, color='black', lw=0.75, ls=':')

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)
    ymin2, ymax2 = ax[1].get_ylim()
    ax[1].vlines((spec.shape[-1]+1)/2, ymin2, ymax2, color='white', lw=0.75, ls=':')

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([0, 20, 40, 60, 80]), ax[2].set_yticks([])
    ax[2].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)
    ymin3, ymax3 = ax[2].get_ylim()
    ax[2].vlines((spec.shape[-1]+1)/2, ymin3, ymax3, color='white', lw=0.75, ls=':')

    ax[2].set_xlabel('Seconds', weight='bold')
    ax[1].set_ylabel(f"Background:\n{BG}", weight='bold', labelpad=-80, rotation=0)
    ax[2].set_ylabel(f"Foreground:\n{FG}", weight='bold', labelpad=-80, rotation=0)

    if save:
        path = f"/home/hamersky/Tabor PSTHs/{site}/"
        # if os.path.isfile(path):
        Path(path).mkdir(parents=True, exist_ok=True)

        plt.savefig(path + f"{unit} - Pair {pair_idx} - {BG} - {FG} - sigma{sigma}.png")
        plt.close()



# parm = '/auto/data/daq/Tabor/TBR011/TBR011a17_p_OLP.m'
# # parm = '/auto/data/daq/Tabor/TBR007/TBR007a10_p_OLP.m'
# expt = BAPHYExperiment(parm)
# rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
# resp = rec['resp'].rasterize()
#
# expt_params = expt.get_baphy_exptparams()  # Using Charlie's manager
# ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
# soundies = list(ref_handle['SoundPairs'].values())
# pairs = [tuple([j for j in (soundies[s]['bg_sound_name'].split('.')[0],
#                             soundies[s]['fg_sound_name'].split('.')[0])])
#          for s in range(len(soundies))]
# #TBR011a-58-1 = -3
# cell = -3  #TBR011a-58-1
# cell = 2  #
# cell = 3 #large effect
#
# epochs = ['STIM_17Tuning-0-1_13Tsik-0-1', 'STIM_17Tuning-0-1_null', 'STIM_null_13Tsik-0-1',
#           'STIM_17Tuning-0.5-1_13Tsik-0-1', 'STIM_17Tuning-0-1_13Tsik-0.5-1']
# epochs = ['STIM_17Tuning-0-1_13Tsik-0-1', 'STIM_null_13Tsik-0-1',
#           'STIM_17Tuning-0.5-1_13Tsik-0-1']
# epochs = ['STIM_17Tuning-0-1_13Tsik-0-1', 'STIM_17Tuning-0-1_null',
#           'STIM_17Tuning-0-1_13Tsik-0.5-1']
#
# epochs = ['STIM_17Tuning-0-1_13Tsik-0-1', 'STIM_null_13Tsik-0-1',
#           'STIM_17Tuning-0.5-1_13Tsik-0-1', 'STIM_17Tuning-0-1_null']
# colors = ['black', 'yellowgreen', 'lightgrey', 'blue']
#
# #final colors for better plot
# epochs = ['STIM_17Tuning-0-1_null', 'STIM_null_13Tsik-0-1', 'STIM_17Tuning-0-1_13Tsik-0-1',
#           'STIM_17Tuning-0.5-1_13Tsik-0-1']
# colors = ['deepskyblue', 'yellowgreen', 'grey', 'rosybrown']
#
#
# epochs = [f'STIM_17Tuning-0-1_null', 'STIM_null_13Tsik-0-1', 'STIM_17Tuning-0-1_13Tsik-0-1',
#           'STIM_17Tuning-0.5-1_13Tsik-0-1']
# colors = ['deepskyblue', 'yellowgreen', 'grey', 'rosybrown']
#
# bg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/Background1/{AA}.wav'
# fg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/Foreground2/{BB}.wav'
#
# smooval = 6
#
# f = plt.figure(figsize=(15, 9))
# psth = plt.subplot2grid((4, 6), (0, 0), rowspan=2, colspan=6)
# specBG = plt.subplot2grid((4, 6), (2, 0), rowspan=1, colspan=6)
# specFG = plt.subplot2grid((4, 6), (3, 0), rowspan=1, colspan=6)
#
# ax = [psth, specBG, specFG]
#
# time = np.arange(0, r[epochs[0]].shape[-1] / rasterfs)
#
# for e, c in zip(epochs, colors):
#     mean = r[e][:,cell,:].mean(axis=0)
#     mean = smooth(mean, smooval)
#     ax[0].plot(mean, label=e, color=c)
# ax[0].legend()
# ax[0].set_title(f"{resp.chans[cell]}")
#
#
#
# r = resp.extract_epochs(epochs)
# fig, ax = plt.subplots()
# for e, c in zip(epochs,colors):
#     mean = r[e][:,cell,:].mean(axis=0)
#     mean = smooth(mean, 7)
#     ax.plot(mean, label=e, color=c)
# ax.legend()
# ax.set_title(f"{resp.chans[cell]}")
# ymin,ymax = ax.get_ylim()
# ax.vlines([50, 100, 150], ymin, ymax, color = 'black', ls=':')