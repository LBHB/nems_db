import logging
import re

log = logging.getLogger(__name__)


# TODO: Delete after finished deprecating.
# Replaced with: load, splitcount, avgep, st, contrast

#def ozgf(loadkey, recording_uri):
#    recordings = [recording_uri]
#    pattern = re.compile(r'^ozgf\.fs(\d{1,}).ch(\d{1,})([a-zA-Z0-9\.]*)?')
#    parsed = re.match(pattern, loadkey)
#    # TODO: fs and chans useful for anything for the loader? They don't
#    #       seem to be used here, only in the baphy-specific stuff.
#    fs = parsed.group(1)
#    chans = parsed.group(2)
#    options = parsed.group(3)
#
#    # NOTE: These are dumb/greedy searches, so if many more options need
#    #       to be added later will need something more sofisticated.
#    #       The other loader keywords follow a similar pattern.
#    normalize = ('n' in options)
#    contrast = ('c' in options)
#    pupil = ('pup' in options)
#
#    if pupil:
#        xfspec = [['nems.xforms.load_recordings',
#                   {'recording_uri_list': recordings, 'normalize': normalize}],
#                  ['nems.xforms.make_state_signal',
#                   {'state_signals': ['pupil'], 'permute_signals': [],
#                    'new_signalname': 'state'}]]
#    else:
#        xfspec = [['nems.xforms.load_recordings',
#                   {'recording_uri_list': recordings, 'normalize': normalize}],
#                  ['nems.xforms.split_by_occurrence_counts',
#                   {'epoch_regex': '^STIM_'}],
#                  ['nems.xforms.average_away_stim_occurrences', {}]]
#
#    if contrast:
#        xfspec.insert(1, ['nems.xforms.add_contrast', {}])
#
#    return xfspec


# Replaced by: load, st, ref, splitep, avgep

#def env(loadkey, recording_uri):
#    pattern = re.compile(r'^env\.fs(\d{1,})([a-zA-Z0-9\.]*)$')
#    parsed = re.match(pattern, loadkey)
#    fs = parsed.group(1)
#    options = parsed.group(2)
#
#    normalize = ('n' in options)
#    pt = ('pt' in options)
#    mask = ('m' in options)
#
#    recordings = [recording_uri]
#    state_signals, permute_signals, _ = _state_model_loadkey_helper(loadkey)
#    use_state = (state_signals or permute_signals)
#
#    xfspec = [['nems.xforms.load_recordings',
#               {'recording_uri_list': recordings, 'normalize': normalize}]]
#    if use_state:
#        xfspec.append(['nems.xforms.make_state_signal',
#                       {'state_signals': state_signals,
#                        'permute_signals': permute_signals,
#                        'new_signalname': 'state'}])
#        if mask:
#            xfspec.append(['nems.xforms.remove_all_but_correct_references',
#                           {}])
#    elif pt:
#        xfspec.append(['nems.xforms.use_all_data_for_est_and_val', {}])
#    else:
#        xfspec.extend([['nems.xforms.split_by_occurrence_counts',
#                        {'epoch_regex': '^STIM_'}],
#                       ['nems.xforms.average_away_stim_occurrences', {}]])
#
#    return xfspec


# replaced by: load, .. evs, etc? SVD working on this.

#def psth(loadkey, recording_uri):
#    """
#    deprecated
#    m- replaced by masking
#    tar - to be replaced by evs load keyword
#    state signal generator - to be replaced by st load keyword
#    """
#    pattern = re.compile(r'^psth\.fs(\d{1,})([a-zA-Z0-9\.]*)$')
#    parsed = re.match(pattern, loadkey)
#    options = parsed.group(2)
#    fs = parsed.group(1)
#    smooth = ('s' in options)
#    mask = ('m' in options)
#    tar = ('tar' in options)
#
#    recordings = [recording_uri]
#    epoch_regex = '^STIM_'
##    state_signals, permute_signals, _ = _state_model_loadkey_helper(loadkey)
##
##    xfspec = [['nems.xforms.load_recordings',
##               {'recording_uri_list': recordings}],
##              ['nems.xforms.make_state_signal',
##               {'state_signals': state_signals,
##                'permute_signals': permute_signals,
##                'new_signalname': 'state'}]]
#    xfspec = [['nems.xforms.load_recordings',
#               {'recording_uri_list': recordings}]]
#    if mask:
#        xfspec.append(['nems.xforms.remove_all_but_correct_references', {}])
#    elif tar:
#        epoch_regex = '^TAR_'
#        xfspec.append(['nems.xforms.mask_all_but_targets', {}])
#    else:
#        xfspec.append(['nems.xforms.mask_all_but_correct_references', {}])
#
#    xfspec.append(['nems.xforms.generate_psth_from_resp',
#                   {'smooth_resp': smooth, 'epoch_regex': epoch_regex}])
#
#    return xfspec


# Replaced by: load, st, evs?

#def evt(loadkey, recording_uri):
#    pattern = re.compile(r'^evt\.fs(\d{0,})\.?(\w{0,})$')
#    parsed = re.match(pattern, loadkey)
#    fs = parsed.group(1)  # what is this?
#    state = parsed.group(2)  # handled by _state_model_loadkey_helper right now.
#    recordings = [recording_uri]
#
#    state_signals, permute_signals, epoch2_shuffle = \
#            _state_model_loadkey_helper(loadkey)
#
#    xfspec = [['nems.xforms.load_recordings',
#               {'recording_uri_list': recordings}],
#              ['nems.xforms.make_state_signal',
#               {'state_signals': state_signals,
#                'permute_signals': permute_signals,
#                'new_signalname': 'state'}],
#              ['nems.preprocessing.generate_stim_from_epochs',
#               {'new_signal_name': 'stim',
#                'epoch_regex': '^TAR_', 'epoch_shift': 5,
#                'epoch2_regex': 'LICK', 'epoch2_shift': -5,
#                'epoch2_shuffle': epoch2_shuffle, 'onsets_only': True},
#               ['rec'], ['rec']],
#              ['nems.xforms.mask_all_but_targets', {}]]
#
#    return xfspec


# TODO: delete after finished deprecating, no longer used in this module.
#      (may still need to move more of this over to lbhb_preprocessors)

def _state_model_loadkey_helper(loader):
    state_signals = []
    permute_signals = []
    epoch2_shuffle = False
    if (loader.startswith("evt")):

        if loader.endswith("pupbehtarlic"):
            state_signals = ['active', 'pupil']
            permute_signals = []
        elif loader.endswith("pup0behtarlic"):
            state_signals = ['active', 'pupil']
            permute_signals = ['pupil']
        elif loader.endswith("pupbeh0tarlic"):
            state_signals = ['active', 'pupil']
            permute_signals = ['active']
        elif loader.endswith("pup0beh0tarlic"):
            state_signals = ['active', 'pupil']
            permute_signals = ['active', 'pupil']
        elif loader.endswith("pupbehtarlic0"):
            state_signals = ['active', 'pupil']
            permute_signals = []
            epoch2_shuffle = True
        elif loader.endswith("pup0behtarlic0"):
            state_signals = ['active', 'pupil']
            permute_signals = ['pupil']
            epoch2_shuffle = True
        elif loader.endswith("pupbeh0tarlic0"):
            state_signals = ['active', 'pupil']
            permute_signals = ['active']
            epoch2_shuffle = True
        elif loader.endswith("pup0beh0tarlic0"):
            state_signals = ['active', 'pupil']
            permute_signals = ['active', 'pupil']
            epoch2_shuffle = True

        elif loader.endswith("lic"):
            pass

        elif loader.endswith("lic0"):
            epoch2_shuffle = True

        else:
            raise ValueError("unknown state_signals for evt loader")

    elif (loader.startswith("psth") or loader.startswith("nostim") or
          loader.startswith("env")):

        if loader.endswith("tarbehlic"):
            state_signals = ['active', 'lick']
            permute_signals = []
        elif loader.endswith("tarbeh0lic"):
            state_signals = ['active', 'lick']
            permute_signals = ['lick']
        elif loader.endswith("tarbehlic0"):
            state_signals = ['active', 'lick']
            permute_signals = ['lick']
        elif loader.endswith("tarbeh0lic0"):
            state_signals = ['active', 'lick']
            permute_signals = ['active', 'lick']
        elif loader.endswith("tarbehlicpup"):
            state_signals = ['active', 'lick', 'pup']
            permute_signals = []
        elif loader.endswith("pup0beh0"):
            state_signals = ['pupil', 'active']
            permute_signals = ['pupil', 'active']
        elif loader.endswith("pup0beh"):
            state_signals = ['pupil', 'active']
            permute_signals = ['pupil']
        elif loader.endswith("pupbeh0"):
            state_signals = ['pupil', 'active']
            permute_signals = ['active']
        elif loader.endswith("pupbeh"):
            state_signals = ['pupil', 'active']
            permute_signals = []
        elif loader.endswith("pupbehpxb0"):
            state_signals = ['pupil', 'active', 'p_x_a']
            permute_signals = ['p_x_a']
        elif loader.endswith("pupbehpxb"):
            state_signals = ['pupil', 'active', 'p_x_a']
            permute_signals = []

        elif loader.endswith("pup0pre0beh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = ['pupil', 'pre_passive']
        elif loader.endswith("puppre0beh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = ['pre_passive']
        elif loader.endswith("pup0prebeh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = ['pupil']
        elif loader.endswith("pupprebeh"):
            state_signals = ['pupil', 'pre_passive', 'active']
            permute_signals = []

        elif loader.endswith("pre0beh0"):
            state_signals = ['pre_passive', 'active']
            permute_signals = ['pre_passive', 'active']
        elif loader.endswith("pre0beh"):
            state_signals = ['pre_passive', 'active']
            permute_signals = ['pre_passive']
        elif loader.endswith("prebeh0"):
            state_signals = ['pre_passive', 'active']
            permute_signals = ['active']
        elif loader.endswith("prebeh"):
            state_signals = ['pre_passive', 'active']
            permute_signals = []

        elif loader.endswith("predif0beh"):
            state_signals = ['pre_passive', 'puretone_trials',
                             'hard_trials', 'active']
            permute_signals = ['puretone_trials', 'hard_trials']
        elif loader.endswith("predifbeh"):
            state_signals = ['pre_passive', 'puretone_trials',
                             'hard_trials', 'active']
            permute_signals = []
        elif loader.endswith("pbs0pev0beh0"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_bs', 'pupil_ev', 'active']
        elif loader.endswith("pbspev0beh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_ev']
        elif loader.endswith("pbs0pevbeh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_bs']
        elif loader.endswith("pbspevbeh0"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['pupil_bs', 'pupil_ev']
        elif loader.endswith("pbs0pev0beh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = ['active']
        elif loader.endswith("pbspevbeh"):
            state_signals = ['pupil_bs', 'pupil_ev', 'active']
            permute_signals = []

        elif loader.endswith("beh0"):
            state_signals = ['active']
            permute_signals = ['active']
        elif loader.endswith("beh"):
            state_signals = ['active']
            permute_signals = []

    return state_signals, permute_signals, epoch2_shuffle


def _aliased_loader(fn, loadkey):
    '''Forces the keyword fn to use the given loadkey. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <kw_head>.<option1>.<option2> paradigm.
    '''
    def ignorant_loader(ignored_key, recording_uri):
        return fn(loadkey, recording_uri)
    ignorant_loader.key = loadkey
    return ignorant_loader


# NOTE: Using the new keyword syntax is encouraged since it improves
#       readability; however, for exceptionally long keywords or ones
#       that get used very frequently, aliases can be implemented as below.
#       If your alias is causing errors, ask Jacob for help.


#ozgf1 = _aliased_loader(ozgf, 'ozgf.fs100.ch18')
