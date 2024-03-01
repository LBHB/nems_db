# Set of functions to perform basic behavioral analyses on RewardTargetLBHB baphy experiments
# CRH, 10/06/2019

import numpy as np
from scipy.stats import norm
import pandas as pd
from itertools import combinations, permutations
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)

def create_trial_labels(exptparams, exptevents):
    """
    Classify every trial as HIT_TRIAL, MISS_TRIAL etc.
    Also, add two columns to number and classify each individual sound as its own trial
        - For example, each REF can be CORRECT_REJECT or FALSE_ALARM etc.
        - For sounds that weren't played (for ex because of lick early on causing a FA),
            label them NULL and dont give them a trial number.
    TODO: make this function specific to the BehaviorObject (ie, RewardTargetLBHB,
          ClassicalConditioning)
    NOTE: baphy BehaviorControl has an annoying bug where it can sometimes miss very brief lick events.
            This creates a discrepancy between posthoc performance analysis and what was actually carried out
            by behavior control. For example, BehaviorControl might miss an early lick, but then catch a "correct lick" 
            and deliver a reward. Posthoc behavior analysis will call this a FA, even though a reward was delivered.
            In this code, we take this conservative approach. We throw out all data following early licks (regardless of
            if they were detected by BehaviorControl or not)
    NOTE: 2024.02.29 - Noted another weird rare edge case where baphy seems to have
            missed a lick. In these cases, my code was calling these misses, even though a target was
            never presented. Added an extra check for this so that now these should be called
            false alarms.
    """

    all_trials = np.unique(exptevents['Trial'])
    early_win = exptparams['BehaveObject'][1]['EarlyWindow']
    resp_win = exptparams['BehaveObject'][1]['ResponseWindow']
    tarPreStim = exptparams['TrialObject'][1]['TargetHandle'][1]['PreStimSilence']
    refPreStim = exptparams['TrialObject'][1]['ReferenceHandle'][1]['PreStimSilence']
    refDuration = exptparams['TrialObject'][1]['ReferenceHandle'][1]['Duration']
    refPostStim = exptparams['TrialObject'][1]['ReferenceHandle'][1]['PostStimSilence']
    tar_names = exptparams['TrialObject'][1]['TargetHandle'][1]['Names']
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])

    if pump_dur.shape == ():
        pump_dur = np.tile(pump_dur, [len(tar_names)])

    # for each trial, append a line to the event data frame specifying the baphy trial
    # outcome. Also, create new column to label each sound token w/in the trial
    trial_dfs = []
    for t in all_trials:
        ev = exptevents[exptevents['Trial']==t].copy()
        trial_outcome = None # gets updated on every iteration. Whatever the last sound in the
                             # trial gets labeled as is what the trial outcome will be
        catch = False        # make sure a hit on a REF/TAR following a CR of a Catch doesn't overwrite trial outcome
        target = False       # make sure that sounds following a target hit / miss don't cause the trial to be misclassified
        if sum(ev.name=='LICK')>0:
            # lick(s) detected
            fl = ev[ev.name=='LICK']['start'].values[0]
            # decide what each sound is, then use this to classify the trial
            sID = []
            # add reaction time for each sound (measured from sound onset). 
            # If no lick, set to np.inf.
            # if NULL sound trial, set to np.nan (prestim silence, sound never played etc.)
            rt = []
            for _, r in ev.iterrows():
                name = r['name']
                if ('PreStim' in name) | ('PostStim' in name):
                    sID.append('NULL')
                    rt.append(np.nan)

                elif 'Reference' in name:
                    ref_start = r['start']
                    if ((fl >= ref_start) & (fl < (ref_start + early_win))):
                        sID.append('EARLY_TRIAL')
                        rt.append(fl - ref_start)
                        # if in window of prevrious ref, trial is FA, else it's Early
                        if fl < (ref_start - refPostStim - refDuration - refPreStim + early_win + resp_win):
                            if (not catch) & (not target):
                                trial_outcome = 'FALSE_ALARM_TRIAL'
                        else:
                            if (not catch) & (not target) & (fl < (refDuration + refPostStim)):
                                # only make the trial outcome label an early trial if on the first sound.
                                # the sound token (sID) will get labeled correctly
                                trial_outcome = 'EARLY_TRIAL'
                            elif (not catch) & (not target):
                                trial_outcome = 'FALSE_ALARM_TRIAL'
                    elif (fl < ref_start):
                        # sound never played because of an early lick
                        sID.append('NULL')
                        rt.append(np.nan)
                        if (not catch) & (not target) & (fl < (refDuration + refPostStim)):
                            # catch did not precede this reference and this was the first ref in the trial
                            # label trial as early
                            trial_outcome = 'EARLY_TRIAL'
                        elif (not catch) & (not target):
                            trial_outcome = 'FALSE_ALARM_TRIAL'
                    elif (fl > ref_start) & (fl <= ref_start + early_win + resp_win):
                        sID.append('FALSE_ALARM_TRIAL')
                        rt.append(fl - ref_start)
                        if (not catch) & (not target):
                            trial_outcome = 'FALSE_ALARM_TRIAL'
                    elif ((fl > ref_start) & (fl > ref_start + early_win + resp_win)):
                        rt.append(fl - ref_start)
                        sID.append('CORRECT_REJECT_TRIAL')
                        if (not catch) & (not target):
                            trial_outcome = 'CORRECT_REJECT_TRIAL'
                    else:
                        rt.append(np.nan)
                        sID.append('UNKNOWN')

                elif 'Target' in name:
                    target = True
                    tar_start = r['start']
                    rewarded = (pump_dur[[True if t == name.split(',')[1].replace(' ', '') else False for t in tar_names]] > 0)[0]
                    if rewarded:
                        # "classic" rewarded target case
                        if fl <= tar_start:
                            sID.append('NULL')
                            rt.append(np.nan)
                        elif (fl > (tar_start + early_win)) & (fl <= (tar_start + resp_win + early_win)):
                            sID.append('HIT_TRIAL')
                            rt.append(fl - tar_start)
                            if not catch:
                                trial_outcome = 'HIT_TRIAL'
                        elif ((fl > tar_start) & (fl > (tar_start + resp_win + early_win))):
                            sID.append('MISS_TRIAL')
                            rt.append(fl - tar_start)
                            trial_outcome = 'MISS_TRIAL'
                            # don't have catch "catch" here bc if miss here, not really a correct reject. Probably just a miss
                        elif (fl > tar_start) & (fl <= (tar_start + early_win)):
                            sID.append('EARLY_TRIAL')
                            rt.append(fl - tar_start)

                            #if not catch:
                            #    if fl < (tar_start - refPostStim - refDuration - tarPreStim + early_win + resp_win):
                                    # is lick in the response window of the previous reference?
                            #        trial_outcome = 'FALSE_ALARM_TRIAL'
                            #    else:
                            #        trial_outcome = 'EARLY_TRIAL'
                            # CRH 06.24.2020 - always call this an early trial bc we don't want to classify
                            # this as a valid target if the lick came before the early resp window
                            if not catch:
                                trial_outcome = 'EARLY_TRIAL'
                        else:
                            rt.append(np.nan)
                            sID.append('UNKNOWN')
                    else:
                        # Non-rewarded "Target", treat like a "Catch"
                        if fl <= tar_start:
                            sID.append('NULL')
                            rt.append(np.nan)
                        elif (fl > tar_start + early_win) & (fl <= (tar_start + resp_win + early_win)):
                            sID.append('INCORRECT_HIT_TRIAL')
                            rt.append(fl - tar_start)
                            trial_outcome = 'INCORRECT_HIT_TRIAL'
                        elif ((fl > tar_start + early_win) & (fl > (tar_start + resp_win + early_win))):
                            sID.append('CORRECT_REJECT_TRIAL')
                            rt.append(fl - tar_start)
                            trial_outcome = 'CORRECT_REJECT_TRIAL'
                        elif (fl > tar_start) & (fl <= (tar_start + early_win)):
                            sID.append('EARLY_TRIAL')
                            rt.append(fl - tar_start)

                            #if not catch:
                            #    if fl < (tar_start - refPostStim - refDuration - tarPreStim + early_win + resp_win):
                            #        trial_outcome = 'FALSE_ALARM_TRIAL'
                            #    else:
                            #        trial_outcome = 'EARLY_TRIAL'
                            # CRH 06.24.2020 - always call this an early trial bc we don't want to classify
                            # this as a valid target if the lick came before the early resp window
                            if not catch:
                                trial_outcome = 'EARLY_TRIAL'
                        else:
                            rt.append(np.nan)
                            sID.append('UNKNOWN')

                elif 'Catch' in name:
                    catch = True
                    # NOTE that "Catch" are set up to play before targets. So, if a target plays 
                    # after the Catch (e.g. a CORRECT_REJECT_TRIAL), trial_outcome will get overwritten
                    # based on the target outcome. That's why we create "catch" bool here to look for that in a
                    # post catch target. (because we want it to be labeled CORRECT_REJECT_TRIAL)
                    catch_start = r['start']
                    rewarded = (pump_dur[[True if t == name.split(',')[1].replace(' ', '') else False for t in tar_names]] > 0)[0]
                    if rewarded:
                        # Found a rewarded Catch. Treat it like Target. e.g. if no lick, call it a MISS, not CORRECT_REJECT
                        if fl <= catch_start:
                            sID.append('NULL')
                            rt.append(np.nan)
                        elif (fl > (catch_start + early_win)) & (fl <= (catch_start + resp_win + early_win)):
                            sID.append('HIT_TRIAL')
                            rt.append(fl - catch_start)
                            trial_outcome = 'HIT_TRIAL'
                        elif ((fl > catch_start) & (fl > (catch_start + resp_win + early_win))):
                            sID.append('MISS_TRIAL')
                            rt.append(fl - catch_start)
                            trial_outcome = 'MISS_TRIAL'
                        elif (fl > catch_start) & (fl <= (catch_start + early_win)):
                            sID.append('EARLY_TRIAL')
                            rt.append(fl - catch_start)
                            
                            
                            #if fl < (tar_start - refPostStim - refDuration - tarPreStim + early_win + resp_win):
                                # is lick in the response window of the previous reference?
                            #    trial_outcome = 'FALSE_ALARM_TRIAL'
                            #else:
                            #    trial_outcome = 'EARLY_TRIAL'
                            # CRH 06.24.2020 - always call this an early trial bc we don't want to classify
                            # this as a valid target if the lick came before the early resp window
                            trial_outcome = 'EARLY_TRIAL'
                        else:
                            rt.append(np.nan)
                            sID.append('UNKNOWN')
                    else:
                        # "Classic" Catch stimulus. If no lick, call CORRECT_REJECT
                        if fl <= catch_start:
                            sID.append('NULL')
                            rt.append(np.nan)
                        elif (fl > catch_start + early_win) & (fl <= (catch_start + resp_win + early_win)):
                            sID.append('INCORRECT_HIT_TRIAL')
                            rt.append(fl - catch_start)
                            trial_outcome = 'INCORRECT_HIT_TRIAL'
                        elif ((fl > catch_start + early_win) & (fl > (catch_start + resp_win + early_win))):
                            sID.append('CORRECT_REJECT_TRIAL')
                            rt.append(fl - catch_start)
                            trial_outcome = 'CORRECT_REJECT_TRIAL'
                        elif (fl > catch_start) & (fl <= (catch_start + early_win)):
                            sID.append('EARLY_TRIAL')
                            rt.append(fl - catch_start)
                            
                            #if fl < (tar_start - refPostStim - refDuration - tarPreStim + early_win + resp_win):
                            #    trial_outcome = 'FALSE_ALARM_TRIAL'
                            #else:
                            #    trial_outcome = 'EARLY_TRIAL'
                            # CRH 06.24.2020 - always call this an early trial bc we don't want to classify
                            # this as a valid target if the lick came before the early resp window
                            trial_outcome = 'EARLY_TRIAL'
                        else:
                            rt.append(np.nan)
                            sID.append('UNKNOWN')
                
                else:
                    rt.append(np.nan)
                    sID.append('NULL')

            ev.insert(4, 'soundTrial', sID)
            ev.insert(5, 'RT', rt)

            # update trial outcome
            if trial_outcome is not None:  
                baphy_outcome = ev[ev.name.str.contains('OUTCOME') | ev.name.str.contains('BEHAVIOR')]['name'].values[0]
                ev.loc[ev.name==baphy_outcome, 'name'] = trial_outcome
                # and update the time to span whole trial
                ev.loc[ev.name==trial_outcome, 'start'] = ev.start.min()
                ev.loc[ev.name==trial_outcome, 'end'] = ev.end.max()
            else:
                # this happens when a lick occurs before a sound epoch (ref / catch / target). In this case,
                # call it an EARLY_TRIAL
                trial_outcome = 'EARLY_TRIAL'
                baphy_outcome = ev[ev.name.str.contains('OUTCOME') | ev.name.str.contains('BEHAVIOR')]['name'].values[0]
                ev.loc[ev.name==baphy_outcome, 'name'] = trial_outcome
                ev.loc[ev.name==trial_outcome, 'start'] = ev.start.min()
                ev.loc[ev.name==trial_outcome, 'end'] = ev.end.max()

            trial_dfs.append(ev)

        else:
            # no licks, MISS_TRIAL
            outcome = ev[ev.name.str.contains('OUTCOME') | ev.name.str.contains('BEHAVIOR')]['name'].values[0]
            sID = []
            rt = []
            trial_outcome = None
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
                elif 'Catch' in name:
                    # same logic as target, but really, a rewarded Catch is a bug...
                    rewarded = (pump_dur[[True if t in name else False for t in tar_names]] > 0)[0]
                    if rewarded:
                        sID.append('MISS_TRIAL')
                        trial_outcome = 'MISS_TRIAL'
                    else: 
                        sID.append('CORRECT_REJECT_TRIAL')
                        trial_outcome = 'CORRECT_REJECT_TRIAL'
                else:
                    sID.append('NULL')
            
                rt.append(np.inf)

            if (trial_outcome is None) & (ev.name.str.contains('.*Stim.*', regex=True).sum()==0):
                # early trial - sound shut off before sounds played
                trial_outcome = 'EARLY_TRIAL'
            
            if (trial_outcome == "MISS_TRIAL") & ("MISS_TRIAL" not in sID) & (outcome=="OUTCOME,FALSEALARM"):
                # this means that we're calling it a miss but never saw a target stimulus, somehow
                # baphy called it a false alarm. So, I guess a lick was detected online and trial aborted
                trial_outcome = "FALSE_ALARM_TRIAL"
                
            ev.loc[ev.name==outcome, 'name'] = trial_outcome
            # and update the time to span whole trial
            ev.loc[ev.name==trial_outcome, 'start'] = ev.start.min()
            ev.loc[ev.name==trial_outcome, 'end'] = ev.end.max()
            ev.insert(4, 'soundTrial', sID)
            ev.insert(5, 'RT', rt)

            trial_dfs.append(ev)

    new_events = pd.concat(trial_dfs)

    # Number the (non NULL) sound trials
    new_events['soundTrialidx'] = 0
    trialidx = np.arange(1, (new_events.soundTrial!='NULL').sum()+1)
    new_events.loc[new_events.soundTrial!='NULL', 'soundTrialidx'] = trialidx

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
        - trial_numbers: default = None
                if not None, keep only trial numbers contained in trial_numbers
                can be list, np.array, tuple...
        - sound_trial_numbers: default = None
                if not None, keep only sound trial numbers contained in trial_numbers
                can be list, np.array, tuple...
    """
    
    events = exptevents.copy()

    # first, make sure that trials have been labeled using nems_lbhb.behavior.create_trial_labels()
    if 'soundTrial' not in events.columns:
        log.info('Trials have not been labeled. Labeling trials in order to compute metrics... ')
        events = create_trial_labels(exptparams, exptevents)
    
    # delete invalid columns if they already exists as a safeguard against weird conflicts
    if 'invalidTrial' in events.columns:
        events = events.drop(columns=['invalidTrial', 'invalidSoundTrial'])
    
    # set default options
    keep_early_trials = options.get('keep_early_trials', False)
    keep_cue_trials = options.get('keep_cue_trials', False)
    keep_following_incorrect_trial = options.get('keep_following_incorrect_trial', False)
    trial_numbers = options.get('trial_numbers', None)
    sound_trial_numbers = options.get('sound_trial_numbers', None)

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
        # TODO add logic to use new structure in exptparams['TrialParams'][1]['CueSeg']
        # to determine cue trials for newer baphy files (after Luke's modifications)
        nCueTrials = exptparams['TrialObject'][1].get('CueTrialCount',0)
        if nCueTrials > 0:
            cue_hit_trials = events[events.name.str.contains('HIT_TRIAL')]['Trial'][:(nCueTrials+1)].values
            iv = events.Trial.isin(np.arange(1, cue_hit_trials.max()))
            iv_sound_trials = iv | iv_sound_trials
            iv_trials = iv | iv_trials

    if keep_following_incorrect_trial == False:
        # mark trials following an incorrect trial as invalid
        # this is a baphy centric operation. the idea is that following 
        # an incorrecte baphy trial (miss / FA / incorrect hit) the trial
        # will be repeated by baphy so we remove this from behavioral analysis
        incorrect_trials = events[events.name.isin(['INCORRECT_HIT_TRIAL', 'MISS_TRIAL', 'FALSE_ALARM_TRIAL'])]['Trial'].values
        following_trials = incorrect_trials + 1
        iv = events.Trial.isin(following_trials)
        iv_trials = iv | iv_trials
        # make sure all sound tokens in these trials are excluded as well
        iv_sound_trials = iv | iv_sound_trials
    
    if trial_numbers is not None:
        # mark all trial not in trial_numbers as invalid
        iv = (events.Trial.isin(trial_numbers) == False)
        iv_trials = iv | iv_trials
        iv_sound_trials = iv | iv_sound_trials
    
    if sound_trial_numbers is not None:
        # mark all trial not in trial_numbers as invalid
        iv = (events.soundTrialidx.isin(sound_trial_numbers) == False)
        iv_sound_trials = iv | iv_sound_trials

    # Mark any REFs that occur in an invalid target slot as invalid sounds
    # so that they won't be used for DI / RT calculations
    # find the earliest target presentation
    targetstart_min = events[events.name.str.contains('Target')].start.min()
    iv = (events.soundTrial != 'NULL') & (events.start < targetstart_min)
    iv_sound_trials = iv | iv_sound_trials

    # Finally, make sure that soundTrials labeled as NULL are marked as invalidSoundTrials
    iv = events.soundTrial == 'NULL'
    iv_sound_trials = iv | iv_sound_trials

    #events.insert(events.shape[-1], 'invalidSoundTrial', iv_sound_trials)
    #events.insert(events.shape[-1], 'invalidTrial', iv_trials)    
    events['invalidSoundTrial'] = iv_sound_trials
    events['invalidTrial'] = iv_trials

    # double check that any trials where all the sounds were invalid are
    # also marked as invalid
    iv_trials_based_on_sound = []
    for i in events.Trial.unique():
        evt = events[events.Trial==i]
        if evt['invalidSoundTrial'].sum() == evt.shape[0]:
            iv_trials_based_on_sound.append(i)

    iv = events.Trial.isin(iv_trials_based_on_sound)
    events['invalidTrial'] =  iv | events['invalidTrial']

    return events


def compute_metrics(exptparams, exptevents, **options):
    """
    Return the following metrics in a dictionary:
        - Response Rate (RR):
            - Per target correct hit rate (hit trials)
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
    metrics = _compute_metrics(exptparams, events, **options)

    return metrics


def _compute_metrics(exptparams, exptevents, **options):
    """
    "private" function called by nems_lbhb.behavior.compute_metrics(). See latter function for docs.
    """
    targets = exptparams['TrialObject'][1]['TargetHandle'][1]['Names']
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
    try:
        isCatch = [int(i) for i in exptparams['TrialObject'][1]['IsCatch']]
    except:
        isCatch = [0 for i in targets]
    if pump_dur.shape == ():
        pump_dur = np.tile(pump_dur, [len(targets)])

    # for each target, decide if rewarded / unrewarded the get the 
    # hit rate / miss rate 
    R = {'RR': {}, 'dprime': {}, 'DI': {}, 'nTrials': {}}
    for tar_key, catch in zip(targets, isCatch):
        if catch == 1:
            tar = 'Stim , {} , Catch'.format(tar_key)
        else:
            tar = 'Stim , {} , Target'.format(tar_key)
        # Doesn't actually matter for this if rewarded or not. If there was a lick, it will be labeled
        # either a HIT or INCORRECT hit. For the purposes of response rate, we don't care which
        allTarTrials = exptevents[(exptevents.name==tar) & (exptevents.soundTrial!='NULL')]['Trial']
        validTrialList = exptevents[exptevents.Trial.isin(allTarTrials) & \
                                    (exptevents.invalidSoundTrial==False) & \
                                    (exptevents.invalidTrial==False) & \
                                    (exptevents.soundTrial.isin(['HIT_TRIAL', 'MISS_TRIAL', 'CUE_TRIAL', \
                                            'INCORRECT_HIT_TRIAL', 'CORRECT_REJECT_TRIAL', 'EARLY_TRIAL']))]['Trial']
        nTrials = len(np.unique(validTrialList))
        validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]
        #nHits = (validTrialdf.name.isin(['HIT_TRIAL', 'INCORRECT_HIT_TRIAL', 'EARLY_TRIAL'])).sum()
        # 09.10.2020 -- Don't use baphy outcome for this, should use sound token classication. Otherwise HR is
        # messed up for two target trials (e.g. on a trial with a catch)
        nHits = ((validTrialdf.soundTrial.isin(['HIT_TRIAL', 'INCORRECT_HIT_TRIAL'])) & \
                               (validTrialdf.name==tar)).sum()
        if nTrials == 0:
            R['RR'][tar_key] = np.nan
        else:
            R['RR'][tar_key] = nHits / nTrials

        R['nTrials'][tar_key] = nTrials

        # Compute the FAR (REF hit rate) (Note, we include early trials here just in case they have not be marked 
        # invalid in the options dictionary. We never want early target responses. So that's not an option
        # above)
        allRefTrials = np.unique(exptevents[(exptevents.name.str.contains('Reference'))]['Trial'].values)
        validTrialList = exptevents[exptevents.Trial.isin(allRefTrials) & \
                                        (exptevents.invalidSoundTrial==False) & \
                                        (exptevents.invalidTrial==False) & \
                                        (exptevents.soundTrial.isin(['FALSE_ALARM_TRIAL', 'CORRECT_REJECT_TRIAL', 'EARLY_TRIAL']))]['Trial']
        validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]
        
        nTrials = sum((validTrialdf.invalidSoundTrial==False) & (validTrialdf.soundTrial.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL', 'CORRECT_REJECT_TRIAL'])))
        nFA = ((validTrialdf.invalidSoundTrial==False) & ((validTrialdf.soundTrial=='FALSE_ALARM_TRIAL') | (validTrialdf.soundTrial=='EARLY_TRIAL'))).sum()
        if nTrials == 0:
            R['RR']['Reference'] = np.nan
        else:
            R['RR']['Reference'] = nFA / nTrials

        R['nTrials']['Reference'] = nTrials

    # Use the HRs above to compute d' values
    # for each target
    tar_keys = [k for k in R['RR'].keys() if k != 'Reference']
    for tar in tar_keys:
        R['dprime'][tar] = _compute_dprime(R['RR'][tar], R['RR']['Reference'])
    
    # Calculate the RT vectors for each target and for References, then compute DI
    # - Yin, Fritz, & Shamma, 2010 JASA
    # DI is the area under the ROC curve defined by plotting cummulative HR against 
    # cummulative FAR

    # new procedure to replace the one below. Now RTs (all of them) are saved in
    # events dataframe, so don't need to compute them here.
    resp_window = exptparams['BehaveObject'][1]['ResponseWindow'] # TODO make user def. param.
    early_window = exptparams['BehaveObject'][1]['EarlyWindow'] # TODO make user def. param.
    R['DI'] = _compute_DI(exptparams, exptevents, resp_window, early_window) 
    
    # if multiple targets exist, compute the discriminability between targets too
    if len(R['dprime'].keys()) > 1:
        R['LI'] = _compute_LI(exptparams, exptevents, resp_window, early_window)
        # compute pairwise target dprimes
        tar_groups = list(permutations(R['dprime'].keys(), 2))
        for tg in tar_groups:
            kk = tg[0]+'_'+tg[1]
            R["dprime"][kk] = _compute_dprime(R["RR"][tg[0]], R["RR"][tg[1]])

    return R


def _compute_DI(exptparams, exptevents, resp_window, early_window, dx=0.1, **options):
    """
    Compute discrimination index (DI) between a given target and and reference -- Yin, Fritz, & Shamma, 2010 JASA

    This metric combines HR, FAR, and reaction time to produce a metric between 0 and 1 describing the 
    animal's behavioral performance. DI=1 corresponds to perfect performance, DI=0.5 corresponds to chance performance.
    Less than 0.5 would indicate a preference for the Reference over target sounds.
    """
    # create set of rt bins
    bins = np.arange(early_window, early_window + resp_window + dx, dx)
    RTs = get_reaction_times(exptparams, exptevents, **options)
    
    # compute response probability in each bin for REFs
    ref_counts, _ = np.histogram(RTs['Reference'], bins=bins)
    
    if sum(ref_counts) > 0:
        FAR = sum(ref_counts) / len(RTs['Reference'])
        ref_prob = (np.cumsum(ref_counts) / sum(ref_counts)) * FAR
    else:
        ref_prob = np.nan

    ref_RT_prob = np.append(ref_prob, 1)

    # now do the same for each target
    tar_RT_prob = {}
    auc = {}
    for t in RTs['Target'].keys():
        tar_counts, _ = np.histogram(RTs['Target'][t], bins=bins)
        if sum(tar_counts) > 0:
            HR = sum(tar_counts) / len(RTs['Target'][t])
            tar_prob = (np.cumsum(tar_counts) / sum(tar_counts)) * HR
        else:
            tar_prob = np.nan

        # force the area bounded by the ROC curve to end at (1, 1)
        tar_RT_prob[t] = np.append(tar_prob, 1)

        # finally compute area under the curve using trapezoid approximation (DI)
        # for each target
        auc[t] = np.trapz(tar_RT_prob[t], ref_RT_prob)

    return auc


def _compute_LI(exptparams, exptevents, resp_window, early_window, dx=0.1, **options):
    '''
    Exact same as DI, but this compute the discrimination between targets (if there are multiple
    targets). Also, if there is an unrewarded target, this will group all rewarded targets
    and compute a rew. vs. n. rew. LI. This is really a speciality function for BVT/rewardLearning data. 
    crh 2/28/2020
    '''
    # create set of rt bins
    bins = np.arange(early_window, early_window + resp_window + dx, dx)
    RTs = get_reaction_times(exptparams, exptevents, **options)

    if len(RTs['Target'].keys()) == 1:
        print("only one target. Can't compute LI between categories")

    # compute resp probs for each RT for each target
    # if there are rew. targets and not rew. targets, add a key that groups
    # those categories
    tar_names = exptparams['TrialObject'][1]['TargetHandle'][1]['Names']
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
    if pump_dur.size==1:
        pump_dur = np.tile(pump_dur, [len(tar_names)]).tolist()
    rew_tars = [t for i, t in enumerate(tar_names) if pump_dur[i]>0]
    nr_tars = [t for i, t in enumerate(tar_names) if pump_dur[i]==0]

    if (len(rew_tars) > 1):
        tar_names.append(','.join(rew_tars))
    elif (len(nr_tars) > 1):
        tar_names.append(','.join(nr_tars))

    tar_RT_prob = {}
    auc = {}
    for t in tar_names:
        if ',' not in t:
            # normal case, single target
            try:
                tar_counts, _ = np.histogram(RTs['Target'][t], bins=bins)
                if sum(tar_counts) > 0:
                    HR = sum(tar_counts) / len(RTs['Target'][t])
                    tar_prob = (np.cumsum(tar_counts) / sum(tar_counts)) * HR
                else:
                    tar_prob = np.nan
            except:
                # this should catch cases where the given target doesn't exist in the 
                # events you've masked for this computation. In this case, the target
                # is in exptparams, but there aren't any reaction times bc it was never 
                # presented
                tar_prob = np.nan
        else:
            # case for multiple targets being grouped
            ntrials = 0
            for i, tar in enumerate(t.split(',')):
                try:
                    tc, _ = np.histogram(RTs['Target'][tar], bins=bins)
                    if i == 0:
                        tar_counts = tc[:, np.newaxis]
                    else:
                        tar_counts = np.concatenate((tar_counts, tc[:, np.newaxis]), axis=1)
                    ntrials += len(RTs['Target'][tar])
                except: 
                    # see try/catch above for explanation
                    pass
            
            try:
                tar_counts = np.nansum(tar_counts, axis=-1)
                if sum(tar_counts) > 0:
                    HR = np.sum(tar_counts) / ntrials
                    tar_prob = (np.cumsum(tar_counts) / np.sum(tar_counts)) * HR
                else:
                    tar_prob = np.nan     
            except:
                tar_prob = np.nan        

        # force the area bounded by the ROC curve to end at (1, 1)
        tar_RT_prob[t] = np.append(tar_prob, 1)

    # for each pair of tar/ref comparisons
    # compute area under the curve using trapezoid approximation (DI)
    tar_groups = list(permutations(tar_RT_prob.keys(), 2))
    # strip permutations where targets w/in the group, are being compared to themselves
    tar_groups = [t for t in tar_groups if (t[0] not in t[1].split(',') and (t[1] not in t[0].split(',')))]
    tar_group_keys = [t[0]+'_'+t[1] for t in tar_groups]
    auc = {}
    for t, tk in zip(tar_groups, tar_group_keys):
        auc[tk] = np.trapz(tar_RT_prob[t[0]], tar_RT_prob[t[1]])

    return auc
    

def get_reaction_times(exptparams, exptevents, **options):
    events = exptevents.copy()
    if 'RT' not in events.columns:
        events = create_trial_labels(exptparams, events)
    if 'invalidSoundTrial' not in events.columns:
        events = mark_invalid_trials(exptparams, events, **options)
    
    tar_mask = (events.name.str.contains('Target') | events.name.str.contains('Catch')) & \
               ~events.name.str.contains('Silence') & \
               ~events.invalidSoundTrial & \
               ~events.invalidTrial
    ref_mask = events.name.str.contains('Reference') & \
               ~events.name.str.contains('Silence') & \
               ~events.invalidSoundTrial & \
               ~events.invalidTrial

    ref_RTs = events[ref_mask]['RT'].values

    # for each unique target, get RTs
    unique_targets = events[tar_mask].name.unique()
    targets = [t.split(' , ')[1] for t in unique_targets]
    #targets = [t.strip('Stim , ') for t in unique_targets]
    #targets = [t.strip(' , Target') for t in targets]
    #targets = [t.strip(' , Catch') for t in targets]
    tar_RTs = {}
    for tar, tar_key in zip(unique_targets, targets):
        mask = tar_mask & (events.name == tar)
        tar_RTs[tar_key] = events[mask]['RT'].values

    return {'Target': tar_RTs, 'Reference': ref_RTs} 


def _compute_DI_deprecated(tar_RTs, tarHR, ref_RTs, FAR, resp_window, dx=0.1):
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
    if sum(tar_counts) > 0:
        tar_prob = (np.cumsum(tar_counts) / sum(tar_counts)) * tarHR
    else:
        tar_prob = np.nan
    
    if sum(ref_counts) > 0:
        ref_prob = (np.cumsum(ref_counts) / sum(ref_counts)) * FAR    
    else:
        ref_prob = np.nan

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
    raise DeprecationWarning
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
    raise DeprecationWarning
    targets = exptparams['TrialObject'][1]['TargetHandle'][1]['Names']
    pump_dur = np.array(exptparams['BehaveObject'][1]['PumpDuration'])
    if pump_dur.size==1:
        pump_dur = np.tile(pump_dur, [len(tar_names)]).tolist()
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
            
            RTs[tar_key] = np.array(fl) - (np.array(tar_start)) - early_win

    return RTs


def _get_reference_RTs(exptparams, exptevents):
    """
    "private" function to get RTs for references. Separate from targets because logic is slightly different
    """
    raise DeprecationWarning
    early_win = exptparams['BehaveObject'][1]['EarlyWindow']
    resp_win_len = exptparams['BehaveObject'][1]['ResponseWindow'] + early_win
    allRefTrials = np.unique(exptevents[(exptevents.name.str.contains('Reference'))]['Trial'].values)
    validTrialList = exptevents[exptevents.Trial.isin(allRefTrials) & \
                                    (exptevents.soundTrial.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL']))]['Trial'].unique()
    validTrialdf = exptevents[exptevents.Trial.isin(validTrialList)]

    # NOTE: for each trial, could have multiple RTs for the same lick (if lick is in window of more than one
    # ref)
    rts = []
    for t in validTrialList:
        tdf = validTrialdf[validTrialdf.Trial == t]
        # get only FALSE_ALARM_TRIALS or EARLY_TRIALS where invalidSoundTrial is also False
        sound_onsets = tdf[(tdf.soundTrial.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL'])) & \
                            (tdf.invalidSoundTrial==False)]['start'].values
        fl = tdf[tdf.name=='LICK']['start'].values[0]

        for s in sound_onsets:
            rt = fl - s
            # exlucde rt if outside resp window for this stim,
            # or if negative (this would mean that baphy saved
            # sound events that never played. Happens on early trials)
            if (rt <= resp_win_len) & (rt >= 0):
                rts.append(rt)

    return np.array(rts)



# =================================== Plotting functions ==========================================

def plot_RT_hist(exptparams, exptevents, ax=None, bins=None, **trial_options):

    if ax is None:
        _, ax = plt.subplots(1, 1)
    
    # get behavior metrics (for the response rate)
    metrics = compute_metrics(exptparams, exptevents, **trial_options)

    # get reaction times
    RTs = compute_RTs(exptparams, exptevents, **trial_options)

    # set bins for histogram
    if bins is None:
        bins = np.arange(0, 1, 0.1)

    skeys = np.sort(list(RTs.keys()))

    # Now, for each soundID (reference, target1, target2 etc.),
    # create histogram
    for k in skeys:
        counts, xvals = np.histogram(RTs[k], bins=bins)
        try:
            DI = round(metrics['DI'][k], 2)
        except:
            DI = 'N/A'
        HR = round(metrics['RR'][k], 2)
        n = metrics['nTrials'][k]
        ax.step(xvals[:-1], 
                    np.cumsum(counts) / len(RTs[k]) * metrics['RR'][k], 
                    label='{0}, DI: {1}, HR: {2}, n: {3}'.format(k, DI, HR, n), lw=2)
    
    ax.legend(fontsize=6, frameon=False)
    ax.set_xlabel('Reaction time', fontsize=8)
    ax.set_ylabel('HR', fontsize=8)
    ax.set_title("RT histogram", fontsize=8)

    return ax


def plot_cummulative_HR(exptparams, exptevents, ax=None, annotate=False, **trial_options):
    '''
    Loop through events and plot the HR for each sound. Take steps of one (sound) trial. 
    A point gets added to each curve every time a new sound of that type occurs and
    the HR is based on what happened during all previous SOUND trials (not baphy trials)
    
    if annotate==True
        run _get_marker to find appropriate marker based on trial type

    will plot everything (all trials), but only update the HR value by considering VALID sounds
    '''

    # get range of sound trials
    soundTrials = np.sort(exptevents.soundTrialidx.unique())[1:]

    # this should always be done on a "per sound trial basis", 
    # so we will force trial_numbers (BAPHY trials) to None (meaning 
    # that the only trial data mask will be based on sound trials)
    trial_options.update({'trial_numbers': None})

    # compute metrics at first sound trial
    trial_options.update({'sound_trial_numbers': np.arange(1, 2)})
    metrics = compute_metrics(exptparams, exptevents, **trial_options)

    skeys = np.sort(list(metrics['RR'].keys()))

    HR = {k: [] for k in skeys}
    markers = {k: [] for k in skeys} 
    facecolor = {k: [] for k in skeys} 
    edgecolor = {k: [] for k in skeys}
    sound_trial_n = {k: [] for k in skeys}
    mpl_defaults = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sound = exptevents[exptevents.soundTrialidx==1]['name'].values[0]
    trial_type = exptevents[exptevents.soundTrialidx==1]['soundTrial'].values[0]

    log.info('Calculating HRs for trial 1 / {0}'.format(len(soundTrials)))
    for i, k in enumerate(HR.keys()):
        # check if key was present on this trial. If not, don't append.
        if (k in sound) & (np.isnan(metrics['RR'][k])==False):
            HR[k].append(metrics['RR'][k])
            m = _get_marker(trial_type)
            c = mpl_defaults[i]
            if (trial_type == 'MISS_TRIAL') | (trial_type == 'CORRECT_REJECT_TRIAL'):
                # empty marker for miss trials
                fc = 'none'
            else:
                fc = c
            markers[k].append(m)
            facecolor[k].append(fc)
            edgecolor[k].append(c)
            sound_trial_n[k].append(1)

    # Now that HR dict is initialized, loop over remaining sound trials 
    # and update
    for t in soundTrials[2:]:
        if t % 50 == 0:
            log.info('Calculating HRs for trial {0} / {1}'.format(t, len(soundTrials)))
        sound = exptevents[exptevents.soundTrialidx==t]['name'].values[0]
        trial_type = exptevents[exptevents.soundTrialidx==t]['soundTrial'].values[0]
        
        # compute cummulative metrics
        trial_options.update({'sound_trial_numbers': np.arange(1, t)})
        metrics = compute_metrics(exptparams, exptevents, **trial_options)

        # loop over keys to see which to update
        for i, k in enumerate(HR.keys()):
            # check if key was present on this trial and if there's valid data
            # If not, don't append.
            if (k in sound) & (np.isnan(metrics['RR'][k])==False):
                HR[k].append(metrics['RR'][k])
                m = _get_marker(trial_type)
                c = mpl_defaults[i]
                if (trial_type == 'MISS_TRIAL') | (trial_type == 'CORRECT_REJECT_TRIAL'):
                    # empty marker for miss trials
                    fc = 'none'
                else:
                    fc = c
                markers[k].append(m)
                facecolor[k].append(fc)
                edgecolor[k].append(c)
                sound_trial_n[k].append(t-1)
    
    # generate figure based on HR, markers, facecolor, and edgecolor dictionaries
    if ax is None:
        _, ax = plt.subplots(1, 1)
    
    for k in HR.keys():        
        if annotate:
            ax.plot(sound_trial_n[k], HR[k], alpha=0.5, label=k)
            mscatter(sound_trial_n[k], HR[k], m=markers[k], color=facecolor[k], edgecolor=edgecolor[k], s=25, ax=ax)
        else:
            ax.plot(sound_trial_n[k], HR[k], '.-', alpha=0.5, label=k)
    
    ax.set_xlabel('Trial', fontsize=8)
    ax.set_ylabel('Hit Rate', fontsize=8)
    ax.legend(frameon=False, fontsize=8)

    return ax


def _get_marker(soundTrial):
    '''
        open circle = miss
        closed circle = hit
        + = early response 
        * = False alarm
        open square = correct reject
        closed square = incorrect hit
    '''

    if soundTrial=='MISS_TRIAL':
        marker = 'o'
    elif soundTrial=='HIT_TRIAL':
        marker = 'o'
    elif soundTrial=='EARLY_TRIAL':
        marker = '+'
    elif soundTrial=='CORRECT_REJECT_TRIAL':
        marker = 's'
    elif soundTrial=='INCORRECT_HIT_TRIAL':
        marker = 's'
    elif soundTrial=='FALSE_ALARM_TRIAL':
        marker = '*'
    else:
        marker = 'x'

    return marker



def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc