import numpy as np
import itertools
import sys
import nems_lbhb.io as io

def compute_di_nolick(parmfile, **options):
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
            DI2 = (1+HR-FAR)/2  % area under the ROC
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
            PossibleTar2Offsets = (np.where(trialparms['Tar2SegCountFreq'])[0] + 1) * \
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
                perf[tt+1]['FirstLickTime'] = np.inf
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
            if type(tar_time) != float:
                stimtime.extend(tar_time)
            else:
                stimtime.append(tar_time)
        else:
            if type(tar_time) != float:
                stimtime.extend(tar_time)
            else:
                stimtime.append(tar_time)

        resptime.extend((np.ones(TarSlotCount + 1) * perf[tt+1]['FirstLickTime']).tolist())
        # 0: ref, 1:tar1, 2: tar2
        stimtype.extend((np.zeros(TarSlotCount).tolist()))
        stimtype.extend([1])

        reftype.extend(trialref_type[0, tt] * np.ones(TarSlotCount+1))

        if two_target:
            tcounter.extend(np.ones(TarSlotCount+1) * np.array(trialtargetid_all[tt])[Distlinds])
        else:
            tcounter.extend(np.ones(TarSlotCount+1) * trialtargetid[tt])

        trialnum.extend(np.ones(TarSlotCount+1) * (tt+1))

        if two_target:
            Dist2inds = np.array(trialparms['TargetDistSet'])[((np.array(trialtargetid_all[tt])-1) % Ntar_per_reftype)] == 2
            if np.sum(Dist2inds) == 1:
                tar2_time = np.expand_dims(perf[tt+1]['TarTimes'], 0)[np.expand_dims(Dist2inds, 0)]
                if 0:
                    Tar2SlotCount = np.sum(PossibleTar2Times < tar2_time)
                    PossibleTar2Times_this_trial = PossibleTar2Times[0:Tar2SlotCount]
                else:
                    PossibleTar2Times_this_trial = tar_time + PossibleTar2Offsets
                    PossibleTar2Times_this_trial = PossibleTar2Times_this_trial[~(PossibleTar2Times_this_trial >= tar2_time)]
                    Tar2SlotCount = len(PossibleTar2Times_this_trial)

                stimtime.extend(PossibleTar2Times_this_trial)
                stimtime.extend(tar2_time)
                resptime.extend(np.ones(Tar2SlotCount+1) * perf[tt+1]['FirstLickTime'])
                stimtype.extend(np.zeros(Tar2SlotCount))
                stimtype.append(1)
                reftype.extend(trialref_type[0, tt] * np.ones(Tar2SlotCount+1))
                tcounter.extend(np.ones(Tar2SlotCount+1) * np.expand_dims(trialtargetid_all[tt], 0)[np.expand_dims(Dist2inds, 0)])
                trialnum.extend(np.ones(Tar2SlotCount+1) * (tt+1))

            elif np.sum(Dist2inds) > 1:
                sys.warning('There should only be one target from TargetDistSet 2 per trial. There are more somehow...')

    resptime = np.array(resptime)
    resptime[resptime==0] = np.inf

    NoLick = resptime > (np.array(stimtime) + TarWindowStop)
    Lick = ((resptime >= (np.array(stimtime) + TarWindowStart)) & (resptime < (np.array(stimtime) + TarWindowStop)))
    ValidStim = resptime >= (np.array(stimtime) + TarWindowStart)

    stop_respwin = behaviorparams['EarlyWindow'] + behaviorparams['ResponseWindow'] + stop_respwin_offset
    early_window = behaviorparams['EarlyWindow']
    if trials == False:
        use = np.ones(len(trialnum)).astype(bool).tolist()
    else:

        use = [t for t in trialnum if t in trials]

    if two_target:
        repTarDistSet = np.tile(trialparms['TargetDistSet'] ,(1, len(tar_suffixes)))
    else:
        repTarDistSet = 1
    metrics = compute_metrics(Lick[use], NoLick[use], np.array(stimtype)[use], np.array(stimtime)[use],
                        np.array(resptime)[use], np.array(tcounter)[use],
                        stop_respwin, ValidStim[use], trialtargetid, np.array(trialnum)[use], np.array(reftype)[use],
                        reftype_by_tarid, early_window, repTarDistSet)
# ==================
    # metrics using only trials with new stimuli
    tf = []
    HorM_trials = np.where([True if exptparams['Performance'][k]['Hit'] or exptparams['Performance'][k]['Miss'] else False
                    for k in exptparams['Performance'].keys()][:-1])[0] + 1
    use_trials = HorM_trials+1
    if trials:
        use_trials(~ismember(use_trials,trials))=[]

    use = [True if x in np.append(1, use_trials) else False for x in trialnum]
    metrics_newT = compute_metrics(Lick[use], NoLick[use], np.array(stimtype)[use],
                            np.array(stimtime)[use], resptime[use], np.array(tcounter)[use],
                            stop_respwin, ValidStim[use], trialtargetid, np.array(trialnum)[use],
                            np.array(reftype)[use], reftype_by_tarid, early_window, repTarDistSet)

    # metrics using only trials with new stimuli, first half
    use_trials1 = use_trials[use_trials < (np.ones(len(use_trials)) * max(use_trials) / 2)]
    use = [True if x in use_trials1 else False for x in trialnum]
    metrics_newT['pt1'] = compute_metrics(Lick[use],NoLick[use], np.array(stimtype)[use],
                            np.array(stimtime)[use], resptime[use], np.array(tcounter)[use],
                            stop_respwin, ValidStim[use], trialtargetid,
                            np.array(trialnum)[use], np.array(reftype)[use],
                            reftype_by_tarid,early_window,repTarDistSet)
    use_trials2 = use_trials[use_trials > (np.ones(len(use_trials)) * max(use_trials) / 2)]
    use = [True if x in use_trials2 else False for x in trialnum]
    metrics_newT['pt2'] = compute_metrics(Lick[use],NoLick[use], np.array(stimtype)[use],
                            np.array(stimtime)[use], resptime[use], np.array(tcounter)[use],
                            stop_respwin, ValidStim[use], trialtargetid,
                            np.array(trialnum)[use], np.array(reftype)[use],
                            reftype_by_tarid,early_window,repTarDistSet)
                            
    return metrics, metrics_newT

def compute_metrics(Lick, NoLick, stimtype, stimtime, resptime, tcounter, stop_respwin, ValidStim, trialtargetid,
                    trialnum, reftype, reftype_by_tarid, early_window, repTarDistSet):
    m = dict()
    FA = Lick & ValidStim & (stimtype==np.zeros(len(stimtype)))
    CR = NoLick & ValidStim & (stimtype==np.zeros(len(stimtype)))
    Hit = Lick & ValidStim & (stimtype==np.ones(len(stimtype)))
    Miss = NoLick & ValidStim & (stimtype==np.ones(len(stimtype)))
    m['details'] = {'Hits':sum(Hit), 'Misses': sum(Miss), 'FAs': sum(FA), 'CRs': sum(CR)}
    m['HR'] = sum(Hit) / (sum(Hit)+sum(Miss))
    m['FAR'] = sum(FA) / (sum(FA)+sum(CR))
    # calculate DI using reaction time
    resptime[resptime==0] = np.inf
    di, hits, fas, tsteps = compute_di(stimtime[ValidStim], resptime[ValidStim], stimtype[ValidStim], stop_respwin)
    m['DI'] = di
    if np.all(~ValidStim):
        m['DI'] = np.nan

    m['DI2'] = (1+ m['HR'] - m['FAR']) / 2
    m['Bias2'] = (m['HR'] + m['FAR']) / 2

    NuniqueTars = len(reftype_by_tarid)
    uHit = np.zeros((1, NuniqueTars))
    uMiss = np.zeros((1, NuniqueTars))
    uFA = np.zeros((1, NuniqueTars))
    uET = np.zeros((1, NuniqueTars))
    uRT = np.zeros((1, NuniqueTars))
    sRT = np.zeros((1, NuniqueTars))
    medRT = np.zeros((1, NuniqueTars))
    qrRT = np.zeros((2, NuniqueTars))
    uN = np.zeros(NuniqueTars)
    uDI = np.zeros(NuniqueTars)
    uDI_hits = np.zeros((NuniqueTars, 50))
    uDI_fas = np.zeros((NuniqueTars, 50))
    for uu in range(0, NuniqueTars):
        uN[uu] = len(np.unique(trialnum[ValidStim & (tcounter == (np.ones(len(tcounter))) * (uu+1))]))

        hitI = Lick & ValidStim & (stimtype == np.ones(len(stimtype))) & (tcounter == ((uu+1) * np.ones(len(tcounter))))
        uHit[0, uu] = sum(hitI)
        missI = NoLick & ValidStim & (stimtype == np.ones(len(stimtype))) & (tcounter == ((uu+1) * np.ones(len(tcounter))))
        uMiss[0, uu] = sum(missI)
        uFA[0, uu] = uN[uu] - uHit[0, uu] - uMiss[0, uu]
        uET[0, uu] = sum((resptime < stimtime) & (stimtype == np.zeros(len(stimtype))) & (stimtime == np.ones(len(stimtime)) * min(stimtime)) \
                & (tcounter == ((uu+1) * np.ones(len(tcounter)))))
        RTs = resptime[hitI] - stimtime[hitI] - early_window
        if len(RTs) == 0:
            uRT[0, uu] = np.nan
            sRT[0, uu] = np.nan
            medRT[0, uu] = np.nan
            qrRT[:, uu] = np.ones(2) * np.nan

        else:
            uRT[0, uu] = np.mean(RTs)
            sRT[0, uu] = np.std(RTs, ddof=1) # to agree with matlab need to set degrees of freedom
            medRT[0, uu] = np.median(RTs)
            # this funct. behaves differently in python than matlab which is why
            # baphy's version of di_nolick returns slightly diff uDI values
            qrRT[:, uu] = np.percentile(RTs, [25, 75], interpolation='linear')


        FAI = Lick & ValidStim & (stimtype == np.zeros(len(stimtype))) & (tcounter == ((uu+1) * np.ones(len(tcounter))))
        RTs = resptime[FAI] - stimtime[FAI] - early_window

        inds = ValidStim & ((tcounter == ((uu+1) * np.ones(len(tcounter)))) | (stimtype == np.zeros(len(stimtype)))) \
                 & (reftype == (np.ones(len(reftype)) * reftype_by_tarid[uu]))
        if np.any(repTarDistSet > 1):
            tar_inds_using_this_set = np.where(repTarDistSet[0, uu] == repTarDistSet[0, :])[0] + 1
            idx_mask = [False if x in tar_inds_using_this_set else True for x in tcounter]
            inds[idx_mask] = False
        uDI[uu], uDI_hits[uu, :], uDI_fas[uu, :], tsteps = compute_di(stimtime[inds], resptime[inds], stimtype[inds], stop_respwin)
        if np.all(~inds):
            uDI[uu] = np.nan

    uDI[uN == 0] = np.nan
    uHR = []
    for i, v in enumerate((uHit + uMiss).squeeze()):
        if v != 0:
            uHR.append(uHit[0, i] / v)
        else:
            uHR.append(np.nan)
    uHR = np.array(uHR)
    uDI2 = (1 + uHR - m['FAR']) / 2

    m['details']['uHit'] = uHit
    m['details']['uMiss'] = uMiss
    m['details']['uFA'] = uFA
    m['details']['uET'] = uET
    m['details']['uHR'] = uHR
    m['details']['uRT'] = uRT
    m['details']['medRT'] = medRT
    m['details']['sRT'] = sRT
    m['details']['qrRT'] = qrRT
    m['details']['uDI'] = uDI
    m['details']['uDI2'] = uDI2
    m['details']['uN'] = uN
    m['details']['uDI_hits'] = uDI_hits
    m['details']['uDI_fas'] = uDI_fas
    m['details']['tsteps'] = tsteps

    return m


def compute_di(stimtime, resptime, stimtype, stop_respwin, stepcount=None):

    if stepcount is None:
       stepcount = 50
    tsteps = np.append(np.linspace(0, stop_respwin, stepcount-1) , np.inf)
    hits = np.zeros(stepcount)
    fas = np.zeros(stepcount)
    for tt in range(0, stepcount):
        hits[tt] = sum((stimtype == np.ones(len(stimtype))) & ((resptime-stimtime) <= tsteps[tt] * np.ones(len(stimtype))))
        fas[tt] = sum((stimtype == np.zeros(len(stimtype))) & ((resptime-stimtime) <= np.ones(len(stimtype)) * tsteps[tt]))
    # total number of targets presented, ie, one for each hit and miss trial
    targcount = sum(stimtype==1)
    # total number of references = total stim minus targcount
    refcount = sum(stimtype==0)

    hits = hits / (targcount + (targcount==0))
    fas= fas / (refcount + (refcount==0))
    hits[-1] = 1
    fas[-1] = 1

    w = (np.append(0, np.diff(fas)) + np.append(np.diff(fas), 0)) / 2
    di = sum(w * hits)
    w2 = (np.append(0, np.diff(hits)) + np.append(np.diff(hits), 0)) / 2
    di2 = 1 - sum(w2 * fas)

    di = (di+di2) / 2

    return di, hits, fas, tsteps


