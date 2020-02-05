import nems_lbhb.behavior as beh
import numpy as np
import matplotlib.pyplot as plt

parmfile = '/auto/data/daq/Leyva/ley072/ley072b04_a_PTD.m'
exptparams, exptevents = beh.load_behavior(parmfile, classify_trials=True)

# options to keep all trials
all_options = {'keep_early_trials': True, 'keep_cue_trials': True, 'keep_following_incorrect_trial': True}      

# mark invalid trial (sounds that fall in invalid slots for a target)
events = beh.mark_invalid_trials(exptparams, exptevents, **all_options)

# exclude NULL sound events
events = events[(events.soundTrial != 'NULL') | (events.name=='LICK')]
events = events[(events.invalidSoundTrial==False) | (events.soundTrial == 'NULL')]

# get RTs for REFs
trials = events.Trial.unique()
rts = []
for t in trials:
    df = events[(events.Trial==t) & (events.name.str.contains(', Reference') | \
                    events.name.str.contains('LICK'))]
    sound_onsets = df[events.name.str.contains(', Reference')].start.values
    if ((df.name=='LICK').sum() > 0) & (len(sound_onsets) > 0):
        fl = df[df.name=='LICK']['start'].values[0]
        rt = fl - sound_onsets
        rts.extend(rt)
    else:
        rts.extend([np.inf])
    
ref_rts = np.array(rts)

# get RTs for TARGETs
tar_df = events[events.name.str.contains(', Target')]
tar_trials = tar_df.Trial.unique()
for t in trials:
    df = events[(events.Trial==t) & (events.name.str.contains(', Target') | \
                    events.name.str.contains('LICK'))]
    sound_onsets = df[events.name.str.contains(', Target')].start.values
    if ((df.name=='LICK').sum() > 0) & (len(sound_onsets) > 0):
        fl = df[df.name=='LICK']['start'].values[0]
        rt = fl - sound_onsets
        rts.extend(rt)
    else:
        rts.extend([np.inf])

tar_rts = np.array(rts)

RTs = {'Target': tar_rts, 'Reference': ref_rts}

f, ax = plt.subplots(1, 1)

bins = np.arange(0, RTs['Reference'][np.isfinite(RTs['Reference'])].max(), 0.25)
ax.hist(RTs['Target'][np.isfinite(RTs['Target'])], edgecolor='k', bins=bins, label='Tar')
ax.hist(RTs['Reference'][np.isfinite(RTs['Reference'])], edgecolor='k', bins=bins, label='Ref')
ax.legend(fontsize=8, frameon=False)
ax.set_xlabel('RT')

plt.show()