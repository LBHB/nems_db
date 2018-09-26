"""
preprocessor keywords specific to LBHB models
should occur after a loader keyword but before the modelspec keywords
several functions migrated out of old loader keywords
"""

import logging
import re

log = logging.getLogger(__name__)


def pas(loadkey, recording_uri):
    """
    pas = "passive only"
    mask out everything that doesn't fall in a "PASSIVE_EXPERIMENT" epoch
    """

    xfspec = [['nems.preprocessing.mask_keep_passive',
               {}, ['rec'], ['rec']]]

    return xfspec


def ref(kw):
    ops = kw.split('.')[1:]

    balance_rep_count = False
    for op in ops:
        if op.startswith('b'):
            balance_rep_count = True

    return [['nems.xforms.mask_all_but_correct_references',
             {'balance_rep_count': balance_rep_count}]]


def evs(loadkey):
    """
    evs = "event stimulus"
    currently this is specific to using target onset events and lick events
    as the "stimuli"

    broken out of evt loader keyword
    """
    pattern = re.compile(r'^evs\.([a-zA-Z0-9\.]*)$')
    parsed = re.match(pattern, loadkey)
    loader = parsed.group(1)

    # TODO: implement better parser for more flexibility
    loadset = loader.split(".")

    if loader == ("tar.lic"):
        epoch2_shuffle = False
    elif loader == ("tar.lic0"):
        epoch2_shuffle = True
    else:
        raise ValueError("unknown signals for alt-stimulus initializer")

    xfspec = [['nems.preprocessing.generate_stim_from_epochs',
               {'new_signal_name': 'stim',
                'epoch_regex': '^TAR_', 'epoch_shift': 5,
                'epoch2_regex': 'LICK', 'epoch2_shift': -5,
                'epoch2_shuffle': epoch2_shuffle, 'onsets_only': True},
               ['rec'], ['rec']],
              ['nems.xforms.mask_all_but_targets', {}]]

    return xfspec


def st(loadkey, recording_uri):
    """
    st = "state variable"
    generate a state signal

    broken out of evt/psth/etc loader keywords
    """
    pattern = re.compile(r'^st\.([a-zA-Z0-9\.]*)$')
    parsed = re.match(pattern, loadkey)
    loader = parsed.group(1)

    state_signals = []
    permute_signals = []

    loadset = loader.split(".")
    for l in loadset:
        if l.startswith("beh"):
            this_sig = ["active"]
        elif l.startswith('puppsd'):
            this_sig = ["pupil_psd"]
        elif l.startswith('pupcdxpup'):
            this_sig = ["pupil_cd_x_pupil"]
        elif l.startswith('pupcd'):
            this_sig = ["pupil_cd"]
        elif l.startswith('pupder'):
            this_sig = ['pupil_der']
        elif l.startswith('pxpd'):
            this_sig = ['p_x_pd']
        elif l.startswith("pup"):
            this_sig = ["pupil"]
        elif l.startswith("pxb"):
            this_sig = ["p_x_a"]
        elif l.startswith("pre"):
            this_sig = ["pre_passive"]
        elif l.startswith("dif"):
            this_sig = ["puretone_trials", "hard_trials"]
        elif l.startswith("pbs"):
            this_sig = ["pupil_bs"]
        elif l.startswith("pev"):
            this_sig = ["pupil_ev"]
        elif l.startswith("pas"):
            this_sig = ["each_passive"]
        elif l.startswith("r1"):
            this_sig = ["r1"]
        elif l.startswith("r2"):
            this_sig = ["r2"]
        elif l.startswith('ttp'):
            this_sig = ['hit_trials','miss_trials']
        elif l.startswith('far'):
            this_sig = ['far']
        elif l.startswith('hit'):
            this_sig = ['hit']
        else:
            raise ValueError("unknown signal code %s for state variable initializer", l)

        state_signals.extend(this_sig)
        if l.endswith("0"):
            permute_signals.extend(this_sig)

    xfspec = [['nems.xforms.make_state_signal',
               {'state_signals': state_signals,
                'permute_signals': permute_signals,
                'new_signalname': 'state'}]]
    return xfspec


def mod(loadkey, recording_uri):
    """
    Make a signal called "mod". Basically the residual resp (resp - psth) offset
    such that the min is 0 and the max is max(resp - psth + offset)
    """
    
    pattern = re.compile(r'^mod\.([a-zA-Z0-9\.]*)$')
    parsed = re.match(pattern, loadkey)
    op = parsed.group(1)
    
    if op == 'r':
        sig = 'resp'
    elif op == 'p':
        sig = 'pred'

    xfspec = [['nems.xforms.make_mod_signal',
               {'signal': sig}, ['rec'], ['rec']]]
    
    return xfspec

    
def contrast(loadkey):
    ops = loadkey.split('.')[1:]
    kwargs = {}
    for op in ops:
        if op.startswith('ms'):
            ms = op[2:].replace('d', '.')
            kwargs['ms'] = float(ms)
        elif op.startswith('pcnt'):
            percentile = int(op[4:])
            kwargs['percentile'] = percentile
        elif op == 'kz':
            # "keep zeros when calculating percentile cutoff"
            kwargs['ignore_zeros'] = False
        elif op == 'n':
            kwargs['normalize'] = True
        elif op == 'dlog':
            kwargs['dlog'] = True
        elif op == 'cont':
            kwargs['continuous'] = True

    return [['nems_lbhb.contrast_helpers.add_contrast', kwargs]]


def onoff(loadkey):
    return [['nems_lbhb.contrast_helpers.add_onoff', {}]]


def hrc(load_key, recording_uri):
    """
    Mask only data during stimuli that were repeated 10 or greater times.
    hrc = high rep count
    """
    # c_preprocessing is in Charlie's auto users directory
    xfspec = [['nems_lbhb.preprocessing.mask_high_repetion_stims',
               {'epoch_regex':'^STIM_'}, ['rec'], ['rec']]]

    return xfspec


def psthfr(load_key):
    """
    Generate psth from resp
    """
    options = load_key.split('.')[1:]
    smooth = ('s' in options)
    hilo = ('hilo' in options)
    jackknife = ('j' in options)
    epoch_regex = '^STIM_'
    if hilo:
        if jackknife:
             xfspec=[['nems_lbhb.preprocessing.hi_lo_psth_jack',
                     {'smooth_resp': smooth, 'epoch_regex': epoch_regex}]]
        else:
            xfspec=[['nems_lbhb.preprocessing.hi_lo_psth',
                     {'smooth_resp': smooth, 'epoch_regex': epoch_regex}]]
    else:
        if jackknife:
            xfspec=[['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold',
                     {'smooth_resp': smooth, 'epoch_regex': epoch_regex}]]
        else:
            xfspec=[['nems.xforms.generate_psth_from_resp',
                     {'smooth_resp': smooth, 'epoch_regex': epoch_regex}]]
    return xfspec


def rscsw(load_key, cellid, batch):
    """
    generate the signals for sliding window model. It's intended that these be
    added to the state signal later on. Will call the sliding window
    signal resp as if it's a normal nems encoding model. Little bit kludgy.
    CRH 2018-07-12
    """
    pattern = re.compile(r'^rscsw\.wl(\d{1,})\.sc(\d{1,})')
    parsed = re.match(pattern, load_key)
    win_length = parsed.group(1)
    state_correction = parsed.group(2)
    if state_correction == 0:
        state_correction = False
    else:
        state_correction = True

    xfspec = [['preprocessing_tools.make_rscsw_signals',
                   {'win_len': win_length,
                    'state_correction': state_correction,
                    'cellid': cellid,
                    'batch': batch},
                   ['rec'], ['rec']]]
    return xfspec
