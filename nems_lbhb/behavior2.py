import numpy as np
from scipy.stats import norm
import pandas as pd
import nems_lbhb.io as io
import logging

log = logging.getLogger(__name__)

def load_behavior_events(parmfile, classify_trials=True):
    """
    Load behavior events/params for the given parmfile using 
    baphy parmread. By default, add trial labels to the event
    dataframe using behavior.create_trial_labels
    To load raw event/params from baphy_parm_read, set classify=False
    """

    _, exptparams, exptevents = io.baphy_parm_read(parmfile)

    if classify_trials == False:
        pass    
    else:
        exptevents = create_trial_labels(exptevents)
    
    return exptparams, exptevents


def create_trial_labels(exptparams, exptevents):
    """
    Classify every trial as HIT_TRIAL, MISS_TRIAL etc.
    Also, add two columns to number and classify each individual sound as its own trial
        - For example, each REF can be CORRECT_REJECT or FALSE_ALARM etc.
        - For sounds that weren't played (for ex because of lick early on causing a FA),
            label them NULL and give them trial number 0.
    """

    all_trials = np.unique(exptevents['Trial'])
    early_win = exptparams['BehaveObject'][1]['EarlyWindow']
    resp_win = exptparams['BehaveObject'][1]['ResponseWindow']
    refCountDist = exptparams['TrialObject'][1]['ReferenceCountFreq']
    refPreStim = exptparams['TrialObject'][1]['ReferenceHandle'][1]['PreStimSilence']
    refDuration = exptparams['TrialObject'][1]['ReferenceHandle'][1]['Duration']
    refPostStim = exptparams['TrialObject'][1]['ReferenceHandle'][1]['PostStimSilence']
    tarPreStim = exptparams['TrialObject'][1]['TargetHandle'][1]['PreStimSilence']
    tar_names = exptparams['TrialObject'][1]['TargetHandle'][1]['Names']
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
    invalidRefSlots = np.append(0, np.argwhere(refCountDist==0)[:-1]+1)
    min_time = len(invalidRefSlots) * (refPreStim + refDuration + refPostStim) + refPreStim + early_win
    # for each trial, append a line to the event data frame specifying the trial
    # category. Also, create new column to label each sound w/in the trial
    trial_dfs = []
    for t in all_trials:
        ev = exptevents[exptevents['Trial']==t].copy()
        trial_outcome = None
        if sum(ev.name.str.contains('LICK'))>0:
            # lick(s) detected
            fl = ev[ev.name=='LICK']['start'].values[0]
            # decide what each sound is, then use this to classify the trial
            sID = []
            for _, r in ev.iterrows():
                name = r['name']
                if ('PreStim' in name) | ('PostStim' in name):
                    sID.append('NULL')
                
                elif 'Reference' in name:
                    ref_start = r['start']
                    if (fl < min_time) | ((fl > ref_start) & (fl < (ref_start + early_win))):
                        sID.append('EARLY_TRIAL')
                        trial_outcome = 'EARLY_TRIAL'
                    elif (fl < ref_start):
                        sID.append('NULL')
                    elif (fl > ref_start) & (fl < ref_start + resp_win):
                        sID.append('FALSE_ALARM_TRIAL')
                        trial_outcome = 'FALSE_ALARM_TRIAL'
                    elif ((fl > ref_start) & (fl > ref_start + resp_win)) | \
                            (fl < ref_start):
                        sID.append('CORRECT_REJECT_TRIAL')
                    else:
                        sID.append('UNKNOWN')

                elif 'Target' in name:
                    tar_start = r['start']
                    rewarded = (pump_dur[[True if t in name else False for t in tar_names]] > 0)[0]
                    if rewarded:
                        if fl < tar_start:
                            sID.append('NULL')
                        elif (fl > tar_start + early_win) & (fl < (tar_start + resp_win + early_win)):
                            sID.append('HIT_TRIAL')
                            trial_outcome = 'HIT_TRIAL'
                        elif ((fl > tar_start) & (fl > (tar_start + resp_win + early_win))):
                            sID.append('MISS_TRIAL')
                            trial_outcome = 'MISS_TRIAL'
                        elif (fl > tar_start) & (fl < (tar_start + early_win)):
                            sID.append('EARLY_TRIAL')
                            if fl < (tar_start - refPostStim - refDuration - tarPreStim + early_win + resp_win):
                                trial_outcome = 'FALSE_ALARM'
                            else:
                                trial_outcome = 'EARLY_TRIAL'
                        else:
                            sID.append('UNKNOWN')
                    else:
                        if fl < tar_start:
                            sID.append('NULL')
                        elif (fl > tar_start + early_win) & (fl < (tar_start + resp_win + early_win)):
                            sID.append('INCORRECT_HIT_TRIAL')
                            trial_outcome = 'INCORRECT_HIT_TRIAL'
                        elif ((fl > tar_start + early_win) & (fl > (tar_start + resp_win + early_win))):
                            sID.append('CORRECT_REJECT_TRIAL')
                            trial_outcome = 'CORRECT_REJECT_TRIAL'
                        elif (fl > tar_start) & (fl < (tar_start + early_win)):
                            sID.append('EARLY_TRIAL')
                            if fl < (tar_start - refPostStim - refDuration - tarPreStim + early_win + resp_win):
                                trial_outcome = 'FALSE_ALARM'
                            else:
                                trial_outcome = 'EARLY_TRIAL'
                        else:
                            sID.append('UNKNOWN')
                
                else:
                    sID.append('NUll')

            ev.insert(4, 'sound_category', sID)
            if trial_outcome is not None:  
                outcome = ev[ev.name.str.contains('OUTCOME') | ev.name.str.contains('BEHAVIOR')]['name'].values[0]
                ev.loc[ev.name==outcome, 'name'] = trial_outcome
            trial_dfs.append(ev)

        else:
            # no licks, MISS_TRIAL
            outcome = ev[ev.name.str.contains('OUTCOME') | ev.name.str.contains('BEHAVIOR')]['name'].values[0]
            ev.loc[ev.name==outcome, 'name'] = 'MISS_TRIAL'
            sID = []
            for name in ev.name:
                if ('PreStim' in name) | ('PostStim' in name):
                    sID.append('NULL')
                elif 'Reference' in name:
                    sID.append('CORRECT_REJECT_TRIAL')
                elif 'Target' in name:
                    rewarded = (pump_dur[[True if t in name else False for t in tar_names]] > 0)[0]
                    if rewarded:
                        sID.append('MISS_TRIAL')
                    else: 
                        sID.append('CORRECT_REJECT_TRIAL')
                        ev.loc[ev.name=='MISS_TRIAL', 'name'] = 'CORRECT_REJECT_TRIAL'
                else:
                    sID.append('NULL')
            ev.insert(4, 'sound_category', sID)

            trial_dfs.append(ev)

    new_events = pd.concat(trial_dfs)            
        
    return new_events