"""
Probably temporary... not sure where this should exist...

Idea is that different runclasses in baphy may have special loading requirements.
Seems easiest to stick these "speciality" loading protocols all in one place, to avoid
cluttering the main loader.
"""
import numpy as np
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