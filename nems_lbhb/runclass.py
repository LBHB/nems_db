"""
Probably temporary... not sure where this should exist...

Idea is that different runclasses in baphy may have special loading requirements.
Seems easiest to stick these "speciality" loading protocols all in one place, to avoid
cluttering the main loader.
"""


# ================================== TBP LOADING ================================
def TBP(events, params):
    """
    events is a dataframe of baphy events made by baphy_experiment
    parmas is exptparams from mfile
    """

    # deal with reminder targets
    # remove N1:X tags and just explicity assign "reminder" tags to the second target, "reminder" sounds
    targets = events[events.name.str.contains('TAR_')]['name'].unique()
    if sum([True for t in targets if ':N' in t])>0:
        pass
    else:
        # no N1 etc. tags. Means no repeated freq. SNR combos in this run. 
        # still need to label reminder trial, for consistency with above case
        pass


    return None