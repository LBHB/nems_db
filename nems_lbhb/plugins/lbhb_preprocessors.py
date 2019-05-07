"""
preprocessor keywords specific to LBHB models
should occur after a loader keyword but before the modelspec keywords
several functions migrated out of old loader keywords
"""

import logging
import re

log = logging.getLogger(__name__)


def pas(loadkey):
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
    include_incorrect = False
    generate_evoked_mask = False
    for op in ops:
        if op.startswith('b'):
            balance_rep_count = True
        if op.startswith('a'):
            include_incorrect = True
        if op.startswith('e'):
            generate_evoked_mask = True

    return [['nems.xforms.mask_all_but_correct_references',
             {'balance_rep_count': balance_rep_count,
              'include_incorrect': include_incorrect,
              'generate_evoked_mask': generate_evoked_mask}]]


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


def st(loadkey):
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
        elif l.startswith("pup2"):
            this_sig = ["pupil2"]
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
        elif l.startswith("fil"):
            this_sig = ["each_file"]
        elif l.startswith("hlf"):
            this_sig = ["each_half"]
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
        elif l.startswith('rem'):
            this_sig = ['rem']
        elif l.startswith('eysp'):
            this_sig = ['pupil_eyespeed']
        elif l.startswith('prw'):
            this_sig = ['prw']
        elif l.startswith('pxprw'):
            this_sig = ['pup_x_prw']
        elif l.startswith("pop"):
            this_sig = ["population"]
        elif l.startswith("pxp"):
            this_sig = ["pupil_x_population"]
        elif l.startswith("bxp"):
            this_sig = ["active_x_population"]
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


def mod(loadkey):
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


def pca(loadkey):
    """
    computer pca (or some other state-space) on response
    """

    ops = loadkey.split(".")
    pc_source = "psth"
    overwrite_resp = True
    pc_count=None
    pc_idx=None
    for op in ops:
        if op == "psth":
            pc_source = "psth"
        elif op == "full":
            pc_source = "full"
        elif op == "noise":
            pc_source = "noise"
        elif op == "no":
            overwrite_resp = False
        elif op.startswith("cc"):
            pc_count=int(op[2:])
            pc_idx=list(range(pc_count))
        elif op.startswith("n"):
            pc_count=int(op[1:])+1
            pc_idx=[int(op[1:])]
    if pc_idx is not None:
        xfspec = [['nems.preprocessing.resp_to_pc',
                   {'pc_source': pc_source, 'overwrite_resp': overwrite_resp,
                    'pc_count': pc_count, 'pc_idx': pc_idx}]]
    else:
        xfspec = [['nems.preprocessing.resp_to_pc',
                   {'pc_source': pc_source, 'overwrite_resp': overwrite_resp,
                    'pc_count': pc_count}]]

    return xfspec

def popev(loadkey):
    return [['nems_lbhb.xform_wrappers.split_pop_rec_by_mask', {}]]



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
        elif op.startswith('b'):
            kwargs['bands'] = int(op[1:])

    return [['nems_lbhb.gcmodel.contrast.add_contrast', kwargs]]


def onoff(loadkey):
    return [['nems_lbhb.gcmodel.contrast.add_onoff', {}]]


def hrc(load_key):
    """
    Mask only data during stimuli that were repeated 10 or greater times.
    hrc = high rep count
    """
    xfspec = [['nems_lbhb.preprocessing.mask_high_repetion_stims',
               {'epoch_regex':'^STIM_'}, ['rec'], ['rec']]]

    return xfspec

def rc(load_key):
    """
    Mask only data from a specified runclass
    """
    runclass="NAT"
    options=load_key.split(".")
    if len(options)>1:
        runclass=options[1]
    xfspec = [['nems_lbhb.preprocessing.mask_runclass',
               {'runclass': runclass}, ['rec'], ['rec']]]

    return xfspec

def tor(load_ley):
    """
    Mask only TORC data
    """
    xfspec = [['nems_lbhb.preprocessing.mask_tor',
               {}, ['rec'], ['rec']]]

    return xfspec

def nat(load_ley):
    """
    Mask only NAT data
    """
    xfspec = [['nems_lbhb.preprocessing.mask_nat',
               {}, ['rec'], ['rec']]]

    return xfspec

def subset(load_key):
    """
    Create a mask so that model is fit only using a subset of the data.
    Subset is defined by epoh name.
    Used to mask stimuli of different categories (coherent, incoherent, or single)
    LAS
    """
    subsets = load_key.split('.')[1].split('+')
    xfspec = [['nems_lbhb.preprocessing.mask_subset_by_epoch',
               {'epoch_list':subsets}, ['rec'], ['rec']]]
    return xfspec

def psthfr(load_key):
    """
    Generate psth signal from resp psth.opt1.opt2 etc. By default, set model input_name to
    'psth' (unless ni option specified!).

    options:
    s : smooth
    hilo : call hi_lo_psth
    j : call generate_psth_from_est_for_both_est_and_val_nfold
    ni : don't set input_name to 'psth'.
    """
    options = load_key.split('.')[1:]
    smooth = ('s' in options)
    hilo = ('hilo' in options)
    jackknife = ('j' in options)
    use_as_input = ('ni' not in options)
    if 'stimtar' not in options:
        epoch_regex = '^STIM_'
    else:
        epoch_regex = ['^STIM_', '^TAR_']

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
                     {'smooth_resp': smooth, 'use_as_input': use_as_input, 'epoch_regex': epoch_regex}]]
    return xfspec

def sm(load_key):
    """
    Smooth a signal using preproc.smooth_epoch_segments
    options:
    pop : smooth population signal (default??)
    
    """
    options = load_key.split('.')[1:]

    if 'pop' in options:
        smooth_signal = 'population'
    else:
        smooth_signal = 'resp'
    if 'stimtar' not in options:
        epoch_regex = '^STIM_'
    else:
        epoch_regex = ['^STIM_', '^TAR_']
    
    xfspec=[['nems.preprocessing.smooth_signal_epochs',
            {'signal': smooth_signal, 'epoch_regex': epoch_regex}]]
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

def stSPO(load_key):
    #add SPO state signal
    return [['nems_lbhb.SPO_helpers.add_coherence_as_state',{}]]
