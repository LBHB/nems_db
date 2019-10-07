# Example behavior analyses for rewardTargetLBHB behaviors:
#   two tone discrimnation / variable reward
#   variable SNR pure tone detect
#   single pure tone detect
# CRH - 10/06/2019


import nems_lbhb.behavior as beh
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.io as io
import os

def get_square_asp(ax):
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    return asp

# ============================== two tone discimination / variable reward =========================================

# Analyze two-tone task (one target rewarded, one target not rewarded)
p1 = '/auto/data/daq/Drechsler/training2019/Drechsler_2019_10_03_BVT_3.m'  
p2 = '/auto/data/daq/Drechsler/training2019/Drechsler_2019_10_03_BVT_4.m'
parmfiles = [p1, p2]
for parmfile in parmfiles:

    # load experiment parameters and events. By default, this function will also label
    # each trial in events as HIT_TRIAL, MISS_TRIAL, CORRECT_REJECT_TRIAL etc.
    # To suppress this and just load raw baphy events, use classify_trials=False
    exptparams, exptevents = beh.load_behavior(parmfile, classify_trials=True)

    # Compute behavioral metrics (HR / FAR / d'). Use options dictionary to specify 
    # which trials to include in the analysis. in the example below, valid_options
    # replicates the default options dictionary which excludes HITs after FA, CUE trials etc.
    # and all_options includes ALL trials
    valid_options = {'keep_early_trials': False, 'keep_cue_trials': False, 'keep_following_incorrect_trial': False}
    all_options = {'keep_early_trials': True, 'keep_cue_trials': True, 'keep_following_incorrect_trial': True}      

    valid_metrics = beh.compute_metrics(exptparams, exptevents, **valid_options)
    all_metrics = beh.compute_metrics(exptparams, exptevents, **all_options)

    # Get RT vectors for each target. Again, use the dictionary above to specify which trials to use
    valid_RTs = beh.compute_RTs(exptparams, exptevents, **valid_options)
    all_RTs = beh.compute_RTs(exptparams, exptevents, **all_options)

    # Plot RT histograms of performance

    # For plotting purposes in this demo, also use exptparams to figure out pump duration for each tone
    pump_dur = exptparams['BehaveObject'][1]['PumpDuration']
    pump_dur.append(0)

    bins = np.arange(0, 1, 0.05)
    f, ax = plt.subplots(1, 2)
    for i, k in enumerate(all_RTs.keys()): 
        counts, xvals = np.histogram(all_RTs[k], bins=bins)
        rwdstr = 'rewarded' if pump_dur[i] > 0 else 'non-rewarded'
        ax[0].step(xvals[:-1], np.cumsum(counts) / len(all_RTs[k]) * all_metrics['RR'][k], label='{0}, {1}'.format(k, rwdstr))

    ax[0].legend(fontsize=6)
    ax[0].set_xlabel('Reaction time', fontsize=8)
    ax[0].set_ylabel('HR', fontsize=8)
    ax[0].set_title("All trials", fontsize=8)
    ax[0].set_aspect(get_square_asp(ax[0]))

    for i, k in enumerate(valid_RTs.keys()): 
        counts, xvals = np.histogram(valid_RTs[k], bins=bins)
        rwdstr = 'rewarded' if pump_dur[i] > 0 else 'non-rewarded'
        ax[1].step(xvals[:-1], np.cumsum(counts) / len(valid_RTs[k]) * valid_metrics['RR'][k], label='{0}, {1}'.format(k, rwdstr))

    ax[1].legend(fontsize=6)
    ax[1].set_xlabel('Reaction time', fontsize=8)
    ax[1].set_ylabel('HR', fontsize=8)
    ax[1].set_title("Valid trials", fontsize=8)
    ax[1].set_aspect(get_square_asp(ax[1]))

    f.suptitle(parmfile.split(os.path.sep)[-1])

    f.tight_layout()

# ====================================== Variable SNR tone detect ====================================================

# Analyze variable SNR tone detection task and create psychometric curve.
parmfile = '/auto/data/daq/Babybell/training2017/Babybell_2017_12_21_PTD_5.m'
exptparams, exptevents = beh.load_behavior(parmfile, classify_trials=True)
valid_options = {'keep_early_trials': False, 'keep_cue_trials': False, 'keep_following_incorrect_trial': False}
all_options = {'keep_early_trials': True, 'keep_cue_trials': True, 'keep_following_incorrect_trial': True}      

valid_metrics = beh.compute_metrics(exptparams, exptevents, **valid_options)
all_metrics = beh.compute_metrics(exptparams, exptevents, **all_options)

# Get RT vectors for each target. Again, use the dictionary above to specify which trials to use
valid_RTs = beh.compute_RTs(exptparams, exptevents, **valid_options)
all_RTs = beh.compute_RTs(exptparams, exptevents, **all_options)

# create RT histograms and RT psychometric curves
f, ax = plt.subplots(2, 2)
resp_window = exptparams['BehaveObject'][1]['ResponseWindow'] 
SNR = exptparams['TrialObject'][1]['RelativeTarRefdB'] 

# plot RT histograms on top
bins = np.arange(0, resp_window, 0.05)
for i, k in enumerate(all_RTs.keys()): 
    counts, xvals = np.histogram(all_RTs[k], bins=bins)
    try:
        snr = SNR[i]
    except:
        snr = '0'
    ax[0, 0].step(xvals[:-1], np.cumsum(counts) / len(all_RTs[k]) * all_metrics['RR'][k], label='{0}, {1} dB'.format(k, snr))

ax[0, 0].legend(fontsize=6)
ax[0, 0].set_xlabel('Reaction time', fontsize=8)
ax[0, 0].set_ylabel('HR', fontsize=8)
ax[0, 0].set_title("All trials", fontsize=8)
ax[0, 0].set_aspect(get_square_asp(ax[0, 0]))

for i, k in enumerate(valid_RTs.keys()): 
    counts, xvals = np.histogram(valid_RTs[k], bins=bins)
    try:
        snr = SNR[i]
    except:
        snr = '0'
    ax[0, 1].step(xvals[:-1], np.cumsum(counts) / len(valid_RTs[k]) * valid_metrics['RR'][k], label='{0}, {1}dB'.format(k, snr))

ax[0, 1].legend(fontsize=6)
ax[0, 1].set_xlabel('Reaction time', fontsize=8)
ax[0, 1].set_ylabel('HR', fontsize=8)
ax[0, 1].set_title("Valid trials", fontsize=8)
ax[0, 1].set_aspect(get_square_asp(ax[0, 1]))

# plot RTs on bottom
all_rt = [rt.mean() for k, rt in all_RTs.items() if 'Ref' not in k]
ax[1, 0].plot(range(0, len(all_rt)), all_rt, marker='o', color='k')
ax[1, 0].set_xticks(np.arange(0, len(all_rt)))
ax[1, 0].set_xticklabels(SNR)
ax[1, 0].set_xlabel('Relative Ref Tar dB', fontsize=8)
ax[1, 0].set_ylabel("Reaction time", fontsize=8)
ax[1, 0].set_aspect(get_square_asp(ax[1, 0]))

valid_rt = [rt.mean() for k, rt in valid_RTs.items() if 'Ref' not in k]
ax[1, 1].plot(range(0, len(valid_rt)), valid_rt, marker='o', color='k')
ax[1, 1].set_xticks(np.arange(0, len(valid_rt)))
ax[1, 1].set_xticklabels(SNR)
ax[1, 1].set_xlabel('Relative Ref Tar dB', fontsize=8)
ax[1, 1].set_ylabel("Reaction time", fontsize=8)
ax[1, 1].set_aspect(get_square_asp(ax[1, 1]))

f.suptitle(parmfile.split(os.path.sep)[-1])

f.tight_layout()


# ============================= Pure tone detect ===============================================
parmfile = '/auto/data/daq/Leyva/ley072/ley072b04_a_PTD.m'
exptparams, exptevents = beh.load_behavior(parmfile, classify_trials=True)
valid_options = {'keep_early_trials': False, 'keep_cue_trials': False, 'keep_following_incorrect_trial': False}
all_options = {'keep_early_trials': True, 'keep_cue_trials': True, 'keep_following_incorrect_trial': True}      

valid_metrics = beh.compute_metrics(exptparams, exptevents, **valid_options)
all_metrics = beh.compute_metrics(exptparams, exptevents, **all_options)

# Get RT vectors for each target. Again, use the dictionary above to specify which trials to use
valid_RTs = beh.compute_RTs(exptparams, exptevents, **valid_options)
all_RTs = beh.compute_RTs(exptparams, exptevents, **all_options)

resp_window = exptparams['BehaveObject'][1]['ResponseWindow'] 
bins = np.arange(0, resp_window, 0.1)

f, ax = plt.subplots(1, 2)
for i, k in enumerate(all_RTs.keys()): 
    counts, xvals = np.histogram(all_RTs[k], bins=bins)
    ax[0].step(xvals[:-1], np.cumsum(counts) / len(all_RTs[k]) * all_metrics['RR'][k], label='{0}'.format(k))

ax[0].legend(fontsize=6)
ax[0].set_xlabel('Reaction time', fontsize=8)
ax[0].set_ylabel('HR', fontsize=8)
ax[0].set_title("All trials", fontsize=8)
ax[0].set_aspect(get_square_asp(ax[0]))

for i, k in enumerate(valid_RTs.keys()): 
    counts, xvals = np.histogram(valid_RTs[k], bins=bins)
    ax[1].step(xvals[:-1], np.cumsum(counts) / len(valid_RTs[k]) * valid_metrics['RR'][k], label='{0}'.format(k))

ax[1].legend(fontsize=6)
ax[1].set_xlabel('Reaction time', fontsize=8)
ax[1].set_ylabel('HR', fontsize=8)
ax[1].set_title("All trials", fontsize=8)
ax[1].set_aspect(get_square_asp(ax[1]))

f.suptitle(parmfile.split(os.path.sep)[-1])

f.tight_layout()

plt.show()
