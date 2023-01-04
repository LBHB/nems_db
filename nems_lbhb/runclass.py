"""
Probably temporary... not sure where this should exist...

Idea is that different runclasses in baphy may have special loading requirements.
Seems easiest to stick these "speciality" loading protocols all in one place, to avoid
cluttering the main loader.
"""
import logging
import numpy as np
import pandas as pd
import copy
from pathlib import Path

from scipy.signal import hilbert, resample
from scipy.io import wavfile
import matplotlib.pyplot as plt
from nems0.analysis.gammatone.gtgram import gtgram

log = logging.getLogger(__name__)

# ================================== TBP LOADING ================================
def TBP(exptevents, exptparams):
    """
    events is a dataframe of baphy events made by baphy_experiment
    params is exptparams from mfile

    return updated events and params
    """

    events = copy.deepcopy(exptevents)
    params = copy.deepcopy(exptparams)

    # deal with reminder targets
    # remove N1:X tags and just explicity assign "reminder" tags to the second target, "reminder" sounds
    event_targets = events[(events.name.str.contains('TAR_') | events.name.str.contains('CAT_') \
                | events.name.str.contains('Target') | events.name.str.contains('Catch')) & \
                    (~events.name.str.contains('PostStim') & ~events.name.str.contains('PreStim'))]['name'].unique()
    baphy_tar_strings = params['TrialObject'][1]['TargetHandle'][1]['Names']
    targetDistSet = np.array(params['TrialObject'][1]['TargetDistSet'])

    # update target name tags
    event_targets_new = event_targets.copy()
    for tidx, tar in enumerate(event_targets):
        # silly catch for if baphy tags are passed:
        if ' , ' in tar:
            prst, tar, post = tar.split(' , ')

            try:
                parmidx = np.argwhere(np.array(baphy_tar_strings)==tar.strip('TAR_'))[0][0]
                distSet = targetDistSet[parmidx]
                if distSet == 2:
                    event_targets_new[tidx] = ' , '.join([prst, (tar.split(':N')[0] + '+reminder').replace('TAR_', 'REM_'), post])
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0] + '+reminder'
                elif distSet == 1:
                    event_targets_new[tidx] = ' , '.join([prst, tar.split(':N')[0], post])
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0]
                else:
                    raise ValueError(f"Unknown case for TargetDistSet = {distSet}")

            except IndexError:
                parmidx = np.argwhere(np.array(baphy_tar_strings)==tar.strip('CAT_'))[0][0]
                distSet = targetDistSet[parmidx]
                if distSet == 2:
                    event_targets_new[tidx] = ' , '.join([prst, (tar.split(':N')[0] + '+reminder').replace('CAT_', 'REM_'), post])
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0] + '+reminder'
                elif distSet == 1:
                    event_targets_new[tidx] = ' , '.join([prst, tar.split(':N')[0], post])
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0]
                else:
                    raise ValueError(f"Unknown case for TargetDistSet = {distSet}")

        else:

            try:
                parmidx = np.argwhere(np.array(baphy_tar_strings)==tar.strip('TAR_'))[0][0]
                distSet = targetDistSet[parmidx]
                if distSet == 2:
                    event_targets_new[tidx] = (tar.split(':N')[0] + '+reminder').replace('TAR_', 'REM_')
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0] + '+reminder'
                elif distSet == 1:
                    event_targets_new[tidx] = tar.split(':N')[0]
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0]
                else:
                    raise ValueError(f"Unknown case for TargetDistSet = {distSet}")

            except IndexError:
                parmidx = np.argwhere(np.array(baphy_tar_strings)==tar.strip('CAT_'))[0][0]
                distSet = targetDistSet[parmidx]
                if distSet == 2:
                    event_targets_new[tidx] = (tar.split(':N')[0] + '+reminder').replace('CAT_', 'REM_')
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0] + '+reminder'
                elif distSet == 1:
                    event_targets_new[tidx] = tar.split(':N')[0]
                    baphy_tar_strings[parmidx] = baphy_tar_strings[parmidx].split(':N')[0]
                else:
                    raise ValueError(f"Unknown case for TargetDistSet = {distSet}")


    # update the events
    for idx, ev in enumerate(event_targets):
        events.loc[events.name==ev, 'name'] = event_targets_new[idx]
    new_events = events

    # updata params
    params['TrialObject'][1]['TargetHandle'][1]['Names'] = baphy_tar_strings
    new_params = params

    return new_events, new_params



def CPN (exptevents, exptparams):
    """
    adds an epoch defining AllPermutations or Triplets
    """

    # function copied over from MLE context_probe_analysis/src/data/epochs.py
    def _set_subepoch_pairs(epochs):
        '''
        Given epochs from a CPP or CPN experiments with names containing sequences of sub stimuli, creates new epochs
        specifying pairs of contiguous substimuli as context probe pairs, with adequate start and end times
        e.g. from 'STIM_sequence001: 1 , 3 , 2 , 4 , 4'  to 'C00_P01', 'C01_P03', 'C03_P02', 'C02_P04', 'C04_P04' and 'C04_P00'.

        :param epochs: pandas DataFrame. original CPP or CPN epochs
        :return: pandas DataFrame. Modified epochs included context probe pairs
        '''

        # selects the subset of eps corresponding to sound sequences
        seq_names = [ep_name for ep_name in epochs.name.unique() if ep_name[0:13] == 'STIM_sequence']
        if len(seq_names) == 0:
            raise ValueError("no eps starting with 'STIM'")

        ff_ep_name = epochs.name.isin(seq_names)
        relevant_eps = epochs.loc[ff_ep_name, :]

        # finds the duration of the prestim and poststim silences
        PreStimSilence = epochs.loc[epochs.name == 'PreStimSilence', ['start', 'end']].values
        PreStimSilence = PreStimSilence[0, 1] - PreStimSilence[0, 0]
        PostStimSilence = epochs.loc[epochs.name == 'PostStimSilence', ['start', 'end']].values
        PostStimSilence = PostStimSilence[0, 1] - PostStimSilence[0, 0]

        # organizes the subepochs in an array with shape E x S where E is the number of initial eps, and S is the number
        # of subepochs

        sub_epochs = relevant_eps.name.values
        # formats tags e.g. 'sequence001:1-2-3-4-5' into list of integers [1, 2, 3, 4, 5]
        sub_epochs = [[int(ss) for ss in ep_name.split(':')[1].split('-')] for ep_name in sub_epochs]
        sub_epochs = np.asarray(sub_epochs)

        # calculates the start and end of each subepochs based on the start and end of its mother epoch
        original_times = relevant_eps.loc[:, ['start', 'end']].values

        # initializes a matrix with shape SP x DF where SP is the number of subepochs including both singles and pairs
        # and DF is the DF columns to be: start and end

        total_subepochs = (sub_epochs.size * 2) + sub_epochs.shape[0]  # first terms includes bot signle and pair vocs
        # second term is for PostStimSilence as prb in pairs
        splited_times = np.zeros([total_subepochs, 2])
        new_names = np.empty([total_subepochs, 1], dtype='object')

        # determines the duration of an individual vocalization
        step = (original_times[0, 1] - original_times[0, 0] - PreStimSilence - PostStimSilence) / sub_epochs.shape[1]

        # changes the duration of silences when considered as contexts or probes
        if PreStimSilence != step or PostStimSilence != step:
            print('Pre or Post Stim different than sub stims, forcing to the same duration')
            PreSilStep = PostSilStep = step
        else:
            PreSilStep = PreStimSilence
            PostSilStep = PostStimSilence

        cc = 0
        # iterates over the original epochs
        for ee, (epoch, this_ep_sub_eps) in enumerate(zip(original_times, sub_epochs)):

            # iterates over single subepochs
            for ss, sub_ep in enumerate(this_ep_sub_eps):

                # first add as a single
                # start time
                start = epoch[0] + PreStimSilence + (step * ss)
                splited_times[cc, 0] = start
                # end time
                end = epoch[0] + PreStimSilence + (step * (ss + 1))
                splited_times[cc, 1] = end
                # name
                new_names[cc, 0] = f'voc_{sub_ep:02d}'

                # second add as a pair
                cc += 1
                # stim_num start time
                if ss == 0:  # special case for PreStimSilence as context
                    context = start - PreSilStep
                    name = f'C00_P{sub_ep:02d}'
                else:
                    context = start - step
                    name = f'C{this_ep_sub_eps[ss - 1]:02d}_P{sub_ep:02d}'

                splited_times[cc, 0] = context
                splited_times[cc, 1] = end
                new_names[cc, 0] = name
                cc += 1

            # finally add the PostStimSilences as prb in a pair

            context = start
            end = end + PostSilStep
            name = f'C{sub_ep:02d}_P00'

            splited_times[cc, 0] = context
            splited_times[cc, 1] = end
            new_names[cc, 0] = name

            cc += 1

        # Concatenate data array and names array and organizes in an epoch dataframe
        new_data = np.concatenate([splited_times, new_names], axis=1)
        sub_epochs = pd.DataFrame(data=new_data, columns=['start', 'end', 'name'])

        # adds the new eps to the old ones
        new_epochs = epochs.copy()
        new_epochs = new_epochs.append(sub_epochs)

        # formats by sorting, index and column order
        new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
        new_epochs.reset_index(drop=True, inplace=True)
        new_epochs = new_epochs.loc[:, ['start', 'end', 'name']]

        return new_epochs

    structure = exptparams['TrialObject'][1]['ReferenceHandle'][1]['SequenceStructure']
    line = pd.DataFrame({'name': structure.strip(), 'start': exptevents.iloc[0,1], 'end': exptevents.iloc[0,2]}, index=[0])

    new_events = pd.concat([line, exptevents], ignore_index=True)
    new_events = _set_subepoch_pairs(new_events)
    new_params = copy.deepcopy(exptparams)

    return new_events, new_params

def wav2env(w,fsin,fsout, axis=0, verbose=False):
    """
    extract and downsample waveform from envelope
    :param w:
    :param fsin:
    :param fsout:
    :param axis: int
        dimension along which to compute envelope
    :param verbose: bool
        if True, plot waveform and envelope

    :return:
    """

    duration = len(w)/fsin

    analytic_signal = hilbert(w)
    amplitude_envelope = np.abs(hilbert(w))

    new_samples = int(np.round(duration * fsout))

    env = resample(amplitude_envelope, new_samples)
    if verbose:
        tin = np.arange(len(w)) / fsin
        tout = np.arange(len(env)) / fsout
        plt.plot(tin, w, label='signal')
        plt.plot(tout, env, label='envelope')
        plt.legend()

    return env

def remove_clicks(w, max_threshold=15, verbose=False):

    w_clean = w

    # log compress everything > 67% of max
    crossover = 0.67 * max_threshold
    ii = (w>crossover)
    w_clean[ii] = crossover + np.log(w_clean[ii]-crossover+1);
    jj = (w<-crossover)
    w_clean[jj] = -crossover - np.log(-w_clean[jj]-crossover+1);

    if verbose:
       print(f'bins compressed down: {ii.sum()} up: {jj.sum()} max {np.abs(w).max():.2f}-->{np.abs(w_clean).max():.2f}')

    return w_clean

def NAT_stim(exptevents, exptparams, stimfmt='gtgram', separate_files_only=False, channels=18, rasterfs=100, f_min=200, f_max=20000,
             mono=False, binaural=False, binsplit=True,
             **options):
    """
    :param exptevents: from baphy
    :param exptparams: from baphy
    :param stimfmt: string, currently must be 'wav', 'gtgram', 'nenv'
    :param separate_files_only: boolean [=False]
        if True, just return each individual sound file rather than combinations
        (eg, as specified for binaural stim)
    :param channels: int
        number of gtgram channels
    :param rasterfs: int
        gtgram sampling rate
    :param f_min: float
        gtgram min frequency
    :param f_max: float
        gtgram max frequency
    :param mono: boolean [False]
        if True, collapse wavs to single channel (dumb control for binaural)
    :param binaural:
        if True, apply model to simulate sound at each ear. Currently, a very dumb HRTF
    :param options: dict
        extra stuff to pass through
    :return:
        stim, tags, stimparam
    """
    ReferenceClass = exptparams['TrialObject'][1]['ReferenceClass']
    ReferenceHandle = exptparams['TrialObject'][1]['ReferenceHandle'][1]
    OveralldB = exptparams['TrialObject'][1]['OveralldB']

    if (ReferenceClass=='BigNat') & \
            (exptparams['TrialObject'][1]['ReferenceHandle'][1].get('FitBinaural','None').strip() != 'None'):
        sound_root = Path(exptparams['TrialObject'][1]['ReferenceHandle'][1]['SoundPath'].replace("H:/", "/auto/data/"))

        #stim_epochs = exptevents.loc[exptevents.name.str.startswith("Stim"),'name'].tolist()
        #print(exptevents.loc[exptevents.name.str.startswith("Stim"),'name'].tolist()[:10])
        #wav1=[e.split(' , ')[1].split("+")[0].split(":")[0].replace("STIM_","") for e in stim_epochs]
        #wav2=[e.split(' , ')[1].split("+")[0].split(":")[0].replace("STIM_","") for e in stim_epochs]

        stim_epochs = exptparams['TrialObject'][1]['ReferenceHandle'][1]['Names']
        #print(exptparams['TrialObject'][1]['ReferenceHandle'][1]['Names'][:10])
        wav1=[e.split("+")[0].split(":")[0] for e in stim_epochs]
        chan1=[int(e.split("+")[0].split(":")[1])-1 for e in stim_epochs]
        wav2 = []
        chan2 = []
        for e in stim_epochs:
            if "+" in e:
                wav2.append(e.split("+")[1].split(":")[0])
                chan2.append(int(e.split("+")[1].split(":")[1])-1)
            else:
                wav2.append("NULL")
                chan2.append(0)

        type1 = [''] * len(wav1)
        type2 = [''] * len(wav2)
        #log.info(wav1[0],chan1[0],wav2[0],chan2[0])
        file_unique=wav1.copy()
        file_unique.extend(wav2)
        file_unique=list(set(file_unique))
        if 'NULL' in file_unique:
            file_unique.remove('NULL')

    elif (ReferenceClass == 'BigNat') | (ReferenceClass == 'NaturalSounds'):
        if ReferenceClass == 'BigNat':
            sound_root = exptparams['TrialObject'][1]['ReferenceHandle'][1]['SoundPath'].replace("\\", "/")
            sound_root = sound_root.replace("H:/", "/auto/data/")
            sound_root = Path(sound_root)
        else:
            subset = ReferenceHandle['Subsets']
            if subset == 1:
                sound_root=Path(f'/auto/users/svd/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds')
            else:
                sound_root=Path(f'/auto/users/svd/code/baphy/Config/lbhb/SoundObjects/@NaturalSounds/sounds_set{subset}')
        stim_epochs = ReferenceHandle['Names']
        file_unique=[f.replace('.wav','') for f in stim_epochs]

        wav1=file_unique.copy()
        chan1 = [0] * len(wav1)
        wav2=["NULL"] * len(wav1)
        chan2 = [0] * len(wav1)
        type1 = [''] * len(wav1)
        type2 = [''] * len(wav1)
    elif ReferenceClass == 'OverlappingPairs':
        if 'BG_Folder' in ReferenceHandle.keys():
            bg_folder = ReferenceHandle['BG_Folder']
            fg_folder = ReferenceHandle['FG_Folder']

            bg_root = Path(f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{bg_folder}')
            fg_root = Path(f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{fg_folder}')

            stim_epochs = ReferenceHandle['Names']
        else:
            bg_folder = 'Background1'
            fg_folder = ReferenceHandle['FG_Folder']
            bg_root = Path(f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{bg_folder}')
            fg_root = Path(f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{fg_folder}')

            stim_epochs=[]

        if len(stim_epochs)>0:
            #print(exptparams['TrialObject'][1]['ReferenceHandle'][1]['Names'][:10])
            wav1=[e.split("_")[0].split("-")[0] for e in stim_epochs]
            wav2=[e.split("_")[1].split("-")[0] for e in stim_epochs]

            # parse codes for synthetic sounds
            type1=[e.split("_")[0].split("-")[-1] for e in stim_epochs]
            type2=[e.split("_")[1].split("-")[-1] for e in stim_epochs]

            paths = {'C': 'Cochlear', 'T': 'Temporal', 'S': 'Spectral', 'U': 'Spectrotemporal', 'M': 'SpectrotemporalMod'}
            types = list(paths.keys())
            wav1 = [paths[t]+'/'+w+'_'+t if t in types else w for w,t in zip(wav1,type1)]
            wav2 = [paths[t]+'/'+w+'_'+t if t in types else w for w,t in zip(wav2,type2)]

            chan1 = [int(e.split("_")[0].split("-")[3]) - 1 if (e.split("_")[0] != 'null') and (len(e.split("_")[0].split("-"))>3) else 0 for e in stim_epochs]
            chan2 = [int(e.split("_")[1].split("-")[3]) - 1 if (e.split("_")[1] != 'null') and (len(e.split("_")[1].split("-"))>3) else 0 for e in stim_epochs]
        else:
            stim_epochs = exptevents.loc[exptevents.name.str.startswith("STIM_"),'name']
            stim_epochs = list(set([s.replace("STIM_","") for s in list(stim_epochs)]))

            wav1 = []
            wav2 = []
            type1 = []
            type2 = []
            for e in stim_epochs:
                e = e.replace("1_","1@").replace("null_","null@")
                wav1.append(e.split("@")[0].split("-")[0])
                wav2.append(e.split("@")[1].split("-")[0])
                if wav1[-1]!='null':
                    type1.append(float(e.split("@")[0].split("-")[1]))
                else:
                    type1.append(0.0)
                if wav2[-1]!='null':
                    type2.append(float(e.split("@")[1].split("-")[1]))
                else:
                    type2.append(0.0)

            chan1 = [0]*len(wav1)
            chan2 = [0]*len(wav2)

        #log.info(wav1[0],chan1[0],wav2[0],chan2[0])
        #wav1 = [wav for wav in wav1 if wav != 'null']
        #wav2 = [wav for wav in wav2 if wav != 'null']

        file_unique = wav1.copy()
        file_unique.extend(wav2)
        file_unique = list(set(file_unique))
        file_unique = [f for f in file_unique if f != 'null']

        log.info('NOTE: Stripping spaces from epoch names in OverlappingPairs files')
        stim_epochs = [s.replace(" ", "") for s in stim_epochs]
    else:
        raise ValueError(f"ReferenceClass {ReferenceClass} gtgram not supported.")

    max_chans = np.max(np.concatenate([np.array(chan1),np.array(chan2)]))+1
    max_chans_was = max_chans
    if mono:
        log.info("Forcing mono stimulus, averaging across space")
        chan1 = [0] * len(wav1)
        chan2 = [0] * len(wav2)
        max_chans = 1

    PreStimSilence = ReferenceHandle['PreStimSilence']
    Duration = ReferenceHandle['Duration']
    PostStimSilence = ReferenceHandle['PostStimSilence']
    log.info(f"Pre/Dur/Pos: {PreStimSilence}/{Duration}/{PostStimSilence}")

    wav_unique = {}
    fs0 = None
    for filename in file_unique:
        if ReferenceClass == "OverlappingPairs":
            try:
                fs, w = wavfile.read(Path(bg_root) / (filename + '.wav'))
            except:
                fs, w = wavfile.read(Path(fg_root) / (filename + '.wav'))
        else:
            fs, w = wavfile.read(sound_root / (filename+'.wav'))
        if w.dtype.name == 'int16':
            w = w / 32767
        elif w.dtype.name == 'int32':
            w = w / 2147483647

        if len(w)<fs*Duration:
            w = np.pad(w, [0, int(fs*Duration)-len(w)])

        if fs0 is None:
            fs0 = fs
        elif fs != fs0:
            this_samples = len(w)
            adjusted_samples = int(this_samples/fs*fs0)
            w = resample(w, adjusted_samples)
            fs=fs0
            print(f'Adjusting sampling rate for {filename} to {fs0}')
            #raise ValueError("fs mismatch between wav files. Need to implement resampling!")

        #print(f"{filename} fs={fs} len={w.shape}")
        duration_samples = int(np.floor(Duration * fs))

        # 10ms ramp at onset:
        w = w[:duration_samples].astype(float)
        ramp = np.hanning(.005 * fs * 2)
        ramp = ramp[:int(np.floor(len(ramp)/2))]
        w[:len(ramp)] *= ramp
        w[-len(ramp):] = w[-len(ramp):] * np.flipud(ramp)

        wav_unique[filename] = w[:, np.newaxis]

    if separate_files_only:
        # combine into pairs that were actually presented
        wav_all = wav_unique
    else:
        wav_all = {}
        fs_all = {}
        for (f1,c1,t1,f2,c2,t2,n) in zip(wav1,chan1,type1,wav2,chan2,type2,stim_epochs):
            #print(f1,f2)
            if f1.upper() != "NULL":
                w1 = wav_unique[f1].copy()
                if f2.upper() != "NULL":
                    w2 = wav_unique[f2].copy()
                else:
                    w2 = np.zeros(w1.shape)
            else:
                w2 = wav_unique[f2].copy()
                w1 = np.zeros(w2.shape)
            log.info(f"{f1}, {f2}, {w1.std()}, {w2.std()}")

            if ReferenceClass == "OverlappingPairs":
                if f1 == 'null':
                    pass
                elif type(t1) is float:
                    si = int(t1 * fs0)
                    w1[:si] = 0
                    # scale peak-to-peak amplitude to OveralldB
                    w1 = w1 / np.max(np.abs(w1)) * 5
                elif not (t1 in (types + ['N'])):
                    # scale peak-to-peak amplitude to OveralldB
                    w1 = w1 / np.max(np.abs(w1)) * 5
                else:
                    # scale by RMS amplitude
                    w1 = w1 / np.std(w1)
                if f2 == 'null':
                    pass
                elif type(t2) is float:
                    si = int(t2 * fs0)
                    w2[:si] = 0
                    # scale peak-to-peak amplitude to OveralldB
                    w2 = w2 / np.max(np.abs(w2)) * 5
                elif not(t2 in (types + ['N'])):
                    # scale peak-to-peak amplitude to OveralldB
                    w2 = w2 / np.max(np.abs(w2)) * 5
                else:
                    # scale by RMS amplitude
                    w2 = w2 / np.std(w2)

            elif ReferenceClass == "BigNat":
                w1 = remove_clicks(w1 * ReferenceHandle['FixedAmpScale'], 15)
                w2 = remove_clicks(w2 * ReferenceHandle['FixedAmpScale'], 15)

            elif ReferenceClass == "NaturalSounds":
                if f1.lower() != 'null':
                    w1 = w1 / np.max(np.abs(w1)) * 5
                if f2.lower() != 'null':
                    w2 = w2 / np.max(np.abs(w2)) * 5

            w = np.zeros((w1.shape[0], max_chans))
            if (binaural is None) | (binaural == False):
                #log.info(f'binaural model: None')
                w[:, [c1]] = w1
                w[:, [c2]] += w2
            else:
                #log.info(f'binaural model: {binaural}')
                #import pdb; pdb.set_trace()
                db_atten = 10
                factor = 10**(-db_atten/20)
                w[:, [c1]] = w1*1/(1+factor)+w2*factor/(1+factor)
                w[:, [c2]] += w2*1/(1+factor)+w1*factor/(1+factor)

            # scale to OveralldB level
            sf = 10 ** ((80 - OveralldB) / 20)
            w /= sf

            wav_all[n] = w

    # pad with zeros and convert to from wav to output format (or passthrough wav)
    sg_unique = {}
    stimparam = {'rasterfs': rasterfs}

    if stimfmt=='wav':
        for f,w in wav_all.items():
            sg_unique[f] = np.concatenate([np.zeros((int(np.floor(fs*PreStimSilence)),max_chans)),
                                         w,
                                         np.zeros((int(np.floor(fs*PostStimSilence)),max_chans))],axis=0).T

    elif stimfmt == 'nenv':

        for f, w in wav_all.items():
            sg_unique[f] = np.concatenate([np.zeros((int(np.floor(rasterfs * PreStimSilence)), max_chans)),
                                         wav2env(w, fs, rasterfs),
                                         np.zeros((int(np.floor(rasterfs * PostStimSilence)), max_chans))], axis=0).T

    elif stimfmt == 'lenv':
        def compress(x):
            y = np.log(x+1)
            y[y<0]=0
            return y
        for f, w in wav_all.items():
            sg_unique[f] = np.concatenate([np.zeros((int(np.floor(rasterfs * PreStimSilence)), max_chans)),
                                         compress(wav2env(w, fs, rasterfs)),
                                         np.zeros((int(np.floor(rasterfs * PostStimSilence)), max_chans))], axis=0).T

    elif stimfmt=='gtgram':
        window_time = 1 / rasterfs
        hop_time = 1 / rasterfs

        duration_bins = int(np.floor(rasterfs*Duration))
        sg_pre = np.zeros((channels,int(np.floor(rasterfs*PreStimSilence))))
        sg_null = np.zeros((channels,duration_bins))
        sg_post = np.zeros((channels,int(np.floor(rasterfs*PostStimSilence))))

        stimparam = {'f_min': f_min, 'f_max': f_max, 'rasterfs': rasterfs}
        padbins = int(np.ceil((window_time-hop_time)/2 * fs))

        for (f,w) in wav_all.items():
            if len(sg_unique)%100 == 99:
                log.info(f"i={len(sg_unique)+1} {f} {w.std(axis=0)}")
            sg = [gtgram(np.pad(w[:,i],[padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)
                  if w[:,i].var()>0 else sg_null
                  for i in range(w.shape[1])]

            sg = [np.concatenate([sg_pre, np.abs(s[:,:duration_bins])**0.5, sg_post], axis=1) for s in sg]

            if mono & (max_chans_was>1):
                sgshuff = np.random.permutation(sg[0].flatten())
                sgshuff = np.reshape(sgshuff, sg[0].shape)
                sg.append(sgshuff)
            sg_unique[f] = np.concatenate(sg, axis=0)

            if binsplit and (binaural!=False):
                sg_unique[f]=np.reshape(sg_unique[f],[channels,-1,sg_unique[f].shape[1]])
    return sg_unique, list(sg_unique.keys()), stimparam


