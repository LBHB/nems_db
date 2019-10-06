import nems_lbhb.behavior as beh
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.io as io
import os

def get_square_asp(ax):
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    return asp

parmfile = '/auto/data/daq/Drechsler/training2019/Drechsler_2019_10_03_BVT_3.m'  
parmfile = '/auto/data/daq/Drechsler/training2019/Drechsler_2019_10_03_BVT_4.m'
parmfile = '/auto/data/daq/Drechsler/training2019/Drechsler_2019_10_04_BVT_1.m'
_, exptparams, _ = io.baphy_parm_read(parmfile)
pump_dur = exptparams['BehaveObject'][1]['PumpDuration']
pump_dur.append(0)
remove_trials = None #np.arange(140, 156)
options = {'corr_rej_reminder': True, 'unique_trials_only': False, 'force_remove': remove_trials,
                    'early_trials': False, 'cue_trials': True}
# plot an example RT histogram
RT = beh.get_RT(parmfile, **options)
HR = beh.get_HR_FAR(parmfile, **options)

bins = np.arange(0, 1, 0.05)
f, ax = plt.subplots(1, 2)
for i, (k, hr_key) in enumerate(zip(RT.keys(), HR.keys())): 
    counts, xvals = np.histogram(RT[k], bins=bins)
    rwdstr = 'rewarded' if pump_dur[i] > 0 else 'non-rewarded'
    ax[0].step(xvals[:-1], np.cumsum(counts) / len(RT[k]) * HR[hr_key], label='{0}, {1}'.format(k, rwdstr))

ax[0].legend()
ax[0].set_xlabel('Reaction time')
ax[0].set_title("All trials")
ax[0].set_aspect(get_square_asp(ax[0]))

options = {'corr_rej_reminder': False, 'unique_trials_only': True, 'force_remove': remove_trials,
                    'early_trials': False, 'cue_trials': False}
# plot an example RT histogram
RT = beh.get_RT(parmfile, **options)
HR = beh.get_HR_FAR(parmfile, **options)

bins = np.arange(0, 1, 0.05)
for i, (k, hr_key) in enumerate(zip(RT.keys(), HR.keys())): 
    counts, xvals = np.histogram(RT[k], bins=bins)
    rwdstr = 'rewarded' if pump_dur[i] > 0 else 'non-rewarded'
    ax[1].step(xvals[:-1], np.cumsum(counts) / len(RT[k]) * HR[hr_key], label='{0}, {1}'.format(k, rwdstr))

ax[1].legend()
ax[1].set_xlabel('Reaction time')
ax[1].set_title("Valid trials")

ax[1].set_aspect(get_square_asp(ax[1]))

f.suptitle(parmfile.split(os.path.sep)[-1])

f.tight_layout()

plt.show()
