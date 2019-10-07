# Set of functions to perform basic behavioral analyses on RewardTargetLBHB baphy experiments
# CRH, 10/06/2019

import numpy as np
from scipy.stats import norm
import pandas as pd
import nems_lbhb.io as io
from itertools import combinations
import logging

log = logging.getLogger(__name__)

def load_behavior(parmfile, classify_trials=True):
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
        exptevents = create_trial_labels(exptparams, exptevents)
    
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

    if pump_dur.shape == ():
        pump_dur = np.tile(pump_dur, [len(tar_names)])

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
                    if (fl < min_time):
                        sID.append('EARLY_TRIAL')
                        trial_outcome = 'EARLY_TRIAL'
                    elif ((fl > ref_start) & (fl < (ref_start + early_win))):
                        sID.append('EARLY_TRIAL')
                        # if in window of prevrious ref, trial is FA, else it's Early
                        if fl < (ref_start - refPostStim - refDuration - refPreStim + early_win + resp_win):
                            trial_outcome = 'FALSE_ALARM_TRIAL'
                        else:
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

            ev.insert(4, 'soundTrial', sID)
            if trial_outcome is not None:  
                outcome = ev[ev.name.str.contains('OUTCOME') | ev.name.str.contains('BEHAVIOR')]['name'].values[0]
                ev.loc[ev.name==outcome, 'name'] = trial_outcome
            trial_dfs.append(ev)

        else:
            # no licks, MISS_TRIAL
            outcome = ev[ev.name.str.contains('OUTCOME') | ev.name.str.contains('BEHAVIOR')]['name'].values[0]
            sID = []
            for name in ev.name:
                if ('PreStim' in name) | ('PostStim' in name):
                    sID.append('NULL')
                elif 'Reference' in name:
                    sID.append('CORRECT_REJECT_TRIAL')
                    trial_outcome = 'MISS_TRIAL'
                elif 'Target' in name:
                    rewarded = (pump_dur[[True if t in name else False for t in tar_names]] > 0)[0]
                    if rewarded:
                        sID.append('MISS_TRIAL')
                        trial_outcome = 'MISS_TRIAL'
                    else: 
                        sID.append('CORRECT_REJECT_TRIAL')
                        trial_outcome = 'CORRECT_REJECT_TRIAL'
                else:
                    sID.append('NULL')

            ev.loc[ev.name==outcome, 'name'] = trial_outcome
            ev.insert(4, 'soundTrial', sID)

            trial_dfs.append(ev)

    new_events = pd.concat(trial_dfs)
        
    return new_events


def mark_invalid_trials(exptparams, exptevents, **options):
    """
    Add two boolean columns to the events dataframe specifying if each 
    sound / trial is valid.

    options dictionary:
        - keep_early_trials: default = False
                default, exclude early response trials. If True, treat as FAs.
        - keep_cue_trials: default = False
                default, exclude cue trials. If True, analyze cue trials with logic specified in remained options fields.
        - keep_follwing_incorrect_trial: default = False
                default, exclude trials that follow incorrect trials e.g. miss of rewarded target, false alarm, hit of unrewarded target.
                If True, treat each trial as normal.
    """
    
    events = exptevents.copy()

    # first, make sure that trials have been labeled using nems_lbhb.behavior.create_trial_labels()
    if 'soundTrial' not in events.columns:
        log.info('Trials have not been labeled. Labeling trials in order to compute metrics... ')
        events = create_trial_labels(exptparams, exptevents)

    # set default options
    keep_early_trials = options.get('keep_early_trials', False)
    keep_cue_trials = options.get('keep_cue_trials', False)
    keep_following_incorrect_trial = options.get('keep_following_incorrect_trial', False)

    iv_trials = False * np.ones(events.shape[0]).astype(np.bool)
    iv_sound_trials = False * np.ones(events.shape[0]).astype(np.bool)
    if keep_early_trials == False:
        # mark early trials as invalid per soundTrial
        iv = events['soundTrial']=='EARLY_TRIAL'
        iv_sound_trials = iv | iv_sound_trials
        
        # mark overall trials as early
        et_trials = events[events.name.str.contains('EARLY_TRIAL')]['Trial'].values
        iv = events.Trial.isin(et_trials)
        iv_trials = iv | iv_trials

    if keep_cue_trials == False:
        # mark cue trials as invalid
        # same for single sounds and overall trials
        nCueTrials = exptparams['TrialObject'][1]['CueTrialCount']
        cue_hit_trials = events[events.name.str.contains('HIT_TRIAL')]['Trial'][:nCueTrials+1].values
        iv = events.Trial.isin(np.arange(1, cue_hit_trials.max()))
        iv_sound_trials = iv | iv_sound_trials
        iv_trials = iv | iv_trials

    if keep_following_incorrect_trial == False:
        # mark trials following an incorrect trial as invalid
        # this happens only on a per baphy trial basis i.e. a REF after a FA is marked as NULL already
        # so don't both updating iv_sound_trials for this
        incorrect_trials = events[events.name.isin(['INCORRECT_HIT_TRIAL', 'MISS_TRIAL', 'FALSE_ALARM_TRIAL'])]['Trial'].values
        following_trials = incorrect_trials + 1
        iv = events.Trial.isin(following_trials)
        iv_trials = iv | iv_trials

    # Finally, make sure that soundTrials labeled as NULL are marked as invalidSoundTrials
    iv = events.soundTrial == 'NULL'
    iv_sound_trials = iv | iv_sound_trials

    events.insert(events.shape[-1], 'invalidSoundTrial', iv_sound_trials)
    events.insert(events.shape[-1], 'invalidTrial', iv_trials)    

    return events


def compute_metrics(exptparams, exptevents, **options):
    """
    Return the following metrics in a dictionary:
        - Response Rate (RR):
            - Per target correct hit rate (hit trials)
            - Per target incorrect hit rate (incorrect hits = total - correct rejects) -- NaN if all targets are rewarded
            - Reference false alarm rate
            (note, incorrect hit rate / correct hit rate are not distinguished in the resulting dictionary.
            Results are just labeled by target name, so use extparams to decide which are rewarded/unrewarded
            outside of this function)

        - Discrimination Index (DI):
            - DI per target
            - DI between pairs of targets (if multiple)

    options dictionary:
        ====== These defaults are set in nems_lbhb.behavior.mark_invalid_trials ========
        - keep_early_trials: default = False
                default, exclude early response trials. If True, treat as FAs.
        - keep_cue_trials: default = False
                default, exclude cue trials. If True, analyze cue trials normally.
        - keep_follwing_incorrect_trial: default = False
                default, exclude trials that follow incorrect trials e.g. miss of rewarded target, false alarm, hit of unrewarded target.
                If True, analyze these trials normally.
        =============== These defaults set inside this function ========================

    """

    events = exptevents.copy()
    
    # 1) mark invalid trials
    events = mark_invalid_trials(exptparams, events, **options)

    # 2) calculate metrics
    metrics = _compute_metrics(exptparams, events)

    return metrics


def _compute_metrics(exptparams, exptevents):
    """
    "private" function called by nems_lbhb.behavior.compute_metrics(). See latter function for docs.
    """

    targets = exptparams['TrialObject'][1]['TargetHandle'][1]['Names']
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])

    if pump_dur.shape == ():
        pump_dur = np.tile(pump_dur, [len(targets)])

    # for each target, decide if rewarded / unrewarded the get the 
    # hit rate / miss rate 
    R = {'RR': {}, 'dprime': {}, 'DI': {}}
    for pd, tar_key in zip(pump_dur, targets):
        rewarded = pd > 0
        tar = 'Stim , {} , Target'.format(tar_key)
        if rewarded:
            # looking for "HIT_TRIALS" and "MISS_TRIALS"
            allTarTrials = exptevents[(exptevents.name==tar)]['Trial']
            validTrialList = exptevents[exptevents.Trial.isin(allTarTrials) & \
                                        (exptevents.invalidTrial==False) & \
                                        (exptevents.name.isin(['HIT_TRIAL', 'MISS_TRIAL']))]['Trial']
            nTrials = len(np.unique(validTrialList))
            validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]
            nHits = (validTrialdf.name=='HIT_TRIAL').sum()
            R['RR'][tar_key] = nHits / nTrials
        else:
            # looking for "INCORRECT_HIT_TRIALS" and "CORRECT_REJECT_TRIALS"
            allTarTrials = exptevents[(exptevents.name==tar)]['Trial']
            validTrialList = exptevents[exptevents.Trial.isin(allTarTrials) & \
                                        (exptevents.invalidTrial==False) & \
                                        (exptevents.name.isin(['CORRECT_REJECT_TRIAL', 'INCORRECT_HIT_TRIAL']))]['Trial']
            nTrials = len(np.unique(validTrialList))
            validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]
            nHits = (validTrialdf.name=='INCORRECT_HIT_TRIAL').sum()
            R['RR'][tar_key] = nHits / nTrials

        # Compute the FAR (REF hit rate) (Note, we include early trials here just in case they have not be marked 
        # invalid in the options dictionary. We never want early target responses. So that's not an option
        # above)
        allRefTrials = np.unique(exptevents[(exptevents.name.str.contains('Reference'))]['Trial'].values)
        validTrialList = exptevents[exptevents.Trial.isin(allRefTrials) & \
                                        (exptevents.invalidSoundTrial==False) & \
                                        (exptevents.soundTrial.isin(['FALSE_ALARM_TRIAL', 'CORRECT_REJECT_TRIAL', 'EARLY_TRIAL']))]['Trial']
        validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]
        nTrials = sum((validTrialdf.invalidSoundTrial==False) & (validTrialdf.soundTrial.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL', 'CORRECT_REJECT_TRIAL'])))
        nFA = ((validTrialdf.soundTrial=='FALSE_ALARM_TRIAL') | (validTrialdf.soundTrial=='EARLY_TRIAL')).sum()
        R['RR']['Reference'] = nFA / nTrials

        # Use the HRs above to compute d' values
        # for each target
        tar_keys = [k for k in R['RR'].keys() if k != 'Reference']
        for tar in tar_keys:
            R['dprime'][tar] = _compute_dprime(R['RR'][tar], R['RR']['Reference'])
        
        # Calculate the RT vectors for each target and for References, then compute DI
        # - Yin, Fritz, & Shamma, 2010 JASA
        # DI is the area under the ROC curve defined by plotting cummulative HR against 
        # cummalative FAR
        tar_RTs = _get_target_RTs(exptparams, exptevents)
        ref_RTs = _get_reference_RTs(exptparams, exptevents)
        resp_window = exptparams['BehaveObject'][1]['ResponseWindow']
        for tar in tar_keys:
            R['DI'][tar] = _compute_DI(tar_RTs[tar], R['RR'][tar], 
                                       ref_RTs, R['RR']['Reference'], 
                                       resp_window)

    return R


def _compute_DI(tar_RTs, tarHR, ref_RTs, FAR, resp_window, dx=0.1):
    """
    Compute discrimination index (DI) between a given target and and reference -- Yin, Fritz, & Shamma, 2010 JASA

    This metric combines HR, FAR, and reaction time to produce a metric between 0 and 1 describing the 
    animal's behavioral performance. DI=1 corresponds to perfect performance, DI=0.5 corresponds to chance performance.
    Less than 0.5 would indicate a preference for the Reference over target sounds.
    """

    # create set of rt bins
    bins = np.arange(0, resp_window, dx)

    # compute response probability in each bin
    tar_counts, _ = np.histogram(tar_RTs, bins=bins)
    ref_counts, _ = np.histogram(ref_RTs, bins=bins)
    tar_prob = (np.cumsum(tar_counts) / sum(tar_counts)) * tarHR
    ref_prob = (np.cumsum(ref_counts) / sum(ref_counts)) * FAR    

    # force the area bounded by the ROC curve to end at (1, 1)
    tar_RT_prob = np.append(tar_prob, 1)
    ref_RT_prob = np.append(ref_prob, 1)

    # compute area under the curve using trapezoid approximation
    auc = np.trapz(tar_RT_prob, ref_RT_prob)

    return auc

def _compute_dprime(hr, far):
    """
    "private" function for computing d' given any two response rates.
    In general, hr will be a target hit rate and far will be a 
    reference false alarm rate.
    """
    if far == 1:
        far = 0.99
    elif far == 0:
        far = 0.01
    zFAR = norm.ppf(far)

    if hr == 1:
        hr = 0.99
    elif hr == 0:
        hr = 0.01

    zHR = norm.ppf(hr)
    dprime = zHR - zFAR

    return dprime


def compute_RTs(exptparams, exptevents, **options):
    """
    Compute RT vectors for each target and for REF on all valid trials

    options dictionary:
        ====== These defaults are set in nems_lbhb.behavior.mark_invalid_trials =======
        - keep_early_trials: default = False
                default, exclude early response trials. If True, treat as FAs.
        - keep_cue_trials: default = False
                default, exclude cue trials. If True, analyze cue trials with logic specified in remained options fields.
        - keep_follwing_incorrect_trial: default = False
                default, exclude trials that follow incorrect trials e.g. miss of rewarded target, false alarm, hit of unrewarded target.
                If True, treat each trial as normal.
    """

    events = exptevents.copy()

    # check to see if trials have been labeled as valid or not
    if 'invalidTrial' not in events.columns:
        events = mark_invalid_trials(exptparams, events, **options)

    tar_rts = _get_target_RTs(exptparams, events)
    ref_rts = _get_reference_RTs(exptparams, events)

    tar_rts['Reference'] = np.array(ref_rts)

    return tar_rts


def _get_target_RTs(exptparams, exptevents):
    """
    "private" function to get RTs for targets. Separate from refs because logic is slightly different
    """

    targets = exptparams['TrialObject'][1]['TargetHandle'][1]['Names']
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
    early_win = exptparams['BehaveObject'][1]['EarlyWindow']

    if pump_dur.shape == ():
        pump_dur = np.tile(pump_dur, [len(targets)])

    RTs = dict.fromkeys(targets)
    for pd, tar_key in zip(pump_dur, targets):
        rewarded = pd > 0
        tar = 'Stim , {} , Target'.format(tar_key)  
        if rewarded:
            # looking for "HIT_TRIALS" and "MISS_TRIALS"
            allTarTrials = exptevents[(exptevents.name==tar)]['Trial']
            validTrialList = exptevents[exptevents.Trial.isin(allTarTrials) & \
                                        (exptevents.invalidTrial==False) & \
                                        (exptevents.name=='HIT_TRIAL')]['Trial']
            validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]
            fl = []
            tar_start = []
            for t in validTrialList:
                tdf = validTrialdf[validTrialdf.Trial==t]
                fl.append(tdf[tdf.name=='LICK']['start'].values[0])
                tar_start.append(tdf[tdf.name==tar]['start'].values[0])
            
            RTs[tar_key] = np.array(fl) - np.array(tar_start) - early_win
        else:
            # looking for "INCORRECT_HIT_TRIALS" and "CORRECT_REJECT_TRIALS"
            allTarTrials = exptevents[(exptevents.name==tar)]['Trial']
            validTrialList = exptevents[exptevents.Trial.isin(allTarTrials) & \
                                        (exptevents.invalidTrial==False) & \
                                        (exptevents.name=='INCORRECT_HIT_TRIAL')]['Trial']
            validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]
            fl = []
            tar_start = []
            for t in validTrialList:
                tdf = validTrialdf[validTrialdf.Trial==t]
                fl.append(tdf[tdf.name=='LICK']['start'].values[0])
                tar_start.append(tdf[tdf.name==tar]['start'].values[0])
            
            RTs[tar_key] = np.array(fl) - (np.array(tar_start) + early_win)

    return RTs


def _get_reference_RTs(exptparams, exptevents):
    """
    "private" function to get RTs for references. Separate from targets because logic is slightly different
    """
    early_win = exptparams['BehaveObject'][1]['EarlyWindow']

    allRefTrials = np.unique(exptevents[(exptevents.name.str.contains('Reference'))]['Trial'].values)
    validTrialList = exptevents[exptevents.Trial.isin(allRefTrials) & \
                                    (exptevents.invalidSoundTrial==False) & \
                                    (exptevents.soundTrial.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL']))]['Trial'].unique()
    validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]

    # NOTE: for each trial, could have multiple RTs for the same lick (if lick is in window of more than one
    # ref)

    rts = []
    for t in validTrialList:
        tdf = validTrialdf[validTrialdf.Trial == t]
        # get only FALSE_ALARM_TRIALS or EARLY_TRIALS with invalidSoundTrial is also False
        sound_onsets = tdf[tdf.soundTrial.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL'])]['start'].values
        fl = tdf[tdf.name=='LICK']['start'].values[0]
        resp_window_start = sound_onsets + early_win

        for s in resp_window_start:
            rts.append(fl - s)

    return np.array(rts)



    