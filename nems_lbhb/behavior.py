import numpy as np
import itertools
import sys
import nems_lbhb.io as io

def compute_di(parmfile, **options):
    """
    calculate behavioral performance on a per-token basis using "traditional" performance metrics.
    Copy of di_nolick.m from lbhb_tools: https://bitbucket.org/lbhb/lbhb-tools/src/master/

    CRH, 08-20-2019

    Inputs:
    ================================================================================
    parmfile - string with full path to .m filename
    options - dictionary
        
        force_use_original_target_window: if true, forse hit anf FA analysis to
            use target window used during data collection. Otherwise, if target
            window is longer than RefSegLen, sets target window to RefSegLen.
        
        trials: trials over which to calculate metrics
        
        stop_respwin_offset: offset to add to end of time window over which di is
            calculated (default 1). The window end is the end of the target window,
            plus this offset value.
    
    Outputs:
    ================================================================================
    metrics and metrics_newT are dictionaries containing the metrics
       metrics_newT has metrics calculated only from trials immediately
       following hits or misses (should be trials in which a new stimulus was
       played)
            DI2 = (1+HR-FAR)/2;  % area under the ROC
            Bias2 = (HR+FAR)/2
            DI = area under ROC curved based on RTs
    """

    # set defaults for options dictionary
    force_use_original_target_window = options.get('force_use_original_target_window', False)
    trials = options.get("trials", False)
    stop_respwin_offset = options.get('stop_respwin_offset', True)

    # load parmfile using baphy io
    globalparams, exptparams, exptevents = io.baphy_parm_read(parmfile)

    trialparms = exptparams['TrialObject'][1]
    
    # ???? What is this for ??????
    two_target = 'TargetDistSet' in trialparms.keys() and np.any(np.array(trialparms['TargetDistSet'])>1)

    # need to add one because we're translating from matlab which indexes at 1
    if 'rawfilecount' in globalparams.keys():
        trialcount = globalparams['rawfilecount']
    else:
        trialcount = len(exptparams['Performance'])
    
    wanted_trials = np.arange(1, trialcount+1)
    perf = dict((k, exptparams['Performance'][k]) for k in wanted_trials if k in exptparams['Performance'])

    behaviorparams = exptparams['BehaveObject'][1]

    TarWindowStart = behaviorparams['EarlyWindow']

    # "strict" - FA is response to any possible target slot preceeding the target
    TarPreStimSilence = trialparms['TargetHandle'][1]['PreStimSilence']
    if 'SingleRefSegmentLen' in trialparms.keys() and trialparms['SingleRefSegmentLen'] > 0:
        RefSegLen = trialparms['SingleRefSegmentLen']
        PossibleTarTimes = ((np.where(trialparms['ReferenceCountFreq'])[0]-0) + 1) * \
                trialparms['SingleRefSegmentLen'] + perf[1]['FirstRefTime']
        
        if two_target:
            PossibleTar2Offsets = (np.where(trialparms['Tar2SegCountFreq'])[0]) * \
                    trialparms['Tar2SegmentLen'] + perf[1]['FirstRefTime']
            PossibleTar2Times = np.tile(PossibleTarTimes, (1, len(PossibleTar2Offsets))) \
                    + np.tile(PossibleTar2Offsets[:, np.newaxis], (len(PossibleTarTimes), 1))
            PossibleTar2Times = np.unique(PossibleTar2Times)
    
    else:
        RefSegLen = trialparms['ReferenceHandle'][1]['PreStimSilence'] + \
                trialparms['ReferenceHandle'][1]['Duration'] + \
                trialparms['ReferenceHandle'][1]['PostStimSilence']
        PossibleTarTimes = np.unique([perf[k]['FirstTarTime'] for k in perf.keys()]).T

    if behaviorparams['ResponseWindow'] > RefSegLen and not force_use_original_target_window:
        TarWindowStop = TarWindowStart + RefSegLen
    else:
        TarWindowStop = TarWindowStart + behaviorparams['ResponseWindow']

    ReferenceHandle = trialparms['ReferenceHandle'][1]
    fields = [k for k in ReferenceHandle['UserDefinableFields']]

    # for cases where REF changes (think of this like changing the context. For example, coherent vs. incoherent REF streams)
    # FOR NOW, HARD CODING only to look for incoherent / coherent cases
    unique_tar_suffixes = np.unique([suf.split(":")[-1] for suf in trialparms['TargetHandle'][1]['Names']]) 
    trialref_type = -1 * np.ones((1, trialcount))
    if len(unique_tar_suffixes) > 0:
        for i in range(1, len(unique_tar_suffixes)+1):
            idx = np.array([id for id in range(1, len(perf)+1) if unique_tar_suffixes[i-1] in perf[id]['ThisTargetNote'][0]])
            trialref_type[:, idx-1] = i
            tar_suffixes = unique_tar_suffixes
    else: 
        tar_suffixes = []
        trialref_type = np.ones((1, trialcount))

    trialtargetid = np.zeros((1,trialcount))

    if 'UniqueTargets' in exptparams.keys() and len(exptparams['UniqueTargets']) > 1:
        UniqueCount = len(exptparams['UniqueTargets'])
        trialtargetid_all = np.zeros(trialcount).tolist()
        trialtargetid = np.zeros(trialcount)
        for tt in range(0, trialcount):
            if 'NullField' not in perf[1].keys() or perf[tt]['NullTrial']:
                if two_target:
                    trialtargetid_all[tt] = [np.where(x == np.array(exptparams['UniqueTargets']))[0][0] + 1 
                                                for x in perf[tt+1]['ThisTargetNote']]
                    try:
                        trialtargetid[tt] = trialtargetid_all[tt][0]
                    except:
                        trialtargetid[tt] = trialtargetid_all[tt]
                else:
                    trialtargetid[tt] = np.where(perf[tt+1]['ThisTargetNote'][0] ==
                                                    np.array(exptparams['UniqueTargets']))[0][0]
    else:
        UniqueCount = 1
        trialtargetid = np.ones(len(perf))

    if tar_suffixes != []:
        reftype_by_tarid = np.nan * np.ones(len(exptparams['UniqueTargets']))
        for i in range(0, len(tar_suffixes)):
            if two_target:
                idx = np.unique([np.array(trialtargetid_all)[(trialref_type==(i+1)).squeeze()]])
                idx = np.unique(list(itertools.chain.from_iterable(idx))) - 1
                reftype_by_tarid[idx] = i+1
            else:
                idx = np.unique([np.array(trialtargetid)[(trialref_type==(i+1)).squeeze()]])
                idx = np.unique(list(itertools.chain.from_iterable(idx))) - 1
                reftype_by_tarid[idx] = i+1
    else:
        reftype_by_tarid = np.ones(len(exptparams['UniqueTargets']))

    resptime = []
    resptimeperfect = []
    stimtype = []
    stimtime = []
    reftype = []
    tcounter = []
    trialnum = []

    # exclude misses at very beginning and end
    Misses = [perf[i+1]['Miss'] for i in range(0, len(perf))]
    if np.sum(Misses) == trialcount:
        # if all missed
        t1 = t2 = 1
    else:
        t1 = np.where(np.array(Misses)==0)[0][0]
        t2 = len(Misses) - np.where(np.array(Misses[::-1])==0)[0][0] 
    
    for tt in range(t1, t2):
        if 'FirstTarTime' not in perf[tt+1].keys() or perf[tt+1]['FirstTarTime'] == []:
            perf[tt+1]['FirstTarTime'] = perf[tt+1]['TarResponseWinStart'] - behaviorparams['EarlyWindow']
        if 'FirstLickTime' not in perf[tt+1].keys() or perf[tt+1]['FirstLickTime'] == []:
            if perf[tt+1]['FirstLickTime'] == []:
                perf[tt+1]['FirstLickTime'] = np.nan
            else:
                perf[tt+1]['FirstLickTime'] = np.min(exptevents[(exptevents['name']=='LICK') & 
                                                            (exptevents['Trial']==tt+1)]['start'])

        Ntar_per_reftype= len(trialparms['TargetIdxFreq'])

        if two_target:
            idx = ((np.array(trialtargetid_all[tt]) - 1)  % Ntar_per_reftype)
            Distlinds = np.array(trialparms['TargetDistSet'])[idx]  == 1

            if not np.any(Distlinds):
                sys.warning("Make sure this works if a trial doesn''t have a target from slot 1")
            
            if len(Distlinds) > 1:
                tar_time = np.array(perf[tt+1]['TarTimes'])[Distlinds]
            else:
                tar_time = perf[tt+1]['FirstTarTime']
                
        else:
            tar_time = perf[tt+1]['FirstTarTime']

        TarSlotCount = np.sum(PossibleTarTimes < tar_time)
        if TarSlotCount > 0:
            stimtime.extend(PossibleTarTimes[:TarSlotCount])
            stimtime.append(tar_time)
        else:
            stimtime.append(tar_time)

        resptime.extend((np.ones(TarSlotCount + 1) * perf[tt+1]['FirstLickTime']).tolist())
        # 0: ref, 1:tar1, 2: tar2
        stimtype.extend((np.zeros(TarSlotCount).tolist()))
        stimtype.extend([1])