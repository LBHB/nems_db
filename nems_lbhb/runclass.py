"""
Probably temporary... not sure where this should exist...

Idea is that different runclasses in baphy may have special loading requirements.
Seems easiest to stick these "speciality" loading protocols all in one place, to avoid
cluttering the main loader.
"""
import numpy as np
import pandas as pd
import copy

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
        e.g. from 'STIM_sequence001: 1 , 3 , 2 , 4 , 4'  to 'C0_P1', 'C1_P3', 'C3_P2', 'C2_P4', 'C4_P4' and 'C4_P0'.

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
                new_names[cc, 0] = 'voc_{}'.format(sub_ep)

                # second add as a pair
                cc += 1
                # stim_num start time
                if ss == 0:  # special case for PreStimSilence as context
                    context = start - PreSilStep
                    name = 'C0_P{}'.format(sub_ep)
                else:
                    context = start - step
                    name = 'C{}_P{}'.format(this_ep_sub_eps[ss - 1], sub_ep)

                splited_times[cc, 0] = context
                splited_times[cc, 1] = end
                new_names[cc, 0] = name
                cc += 1

            # finally add the PostStimSilences as prb in a pair

            context = start
            end = end + PostSilStep
            name = 'C{}_P0'.format(sub_ep)

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