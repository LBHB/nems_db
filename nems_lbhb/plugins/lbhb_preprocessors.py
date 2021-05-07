"""
preprocessor keywords specific to LBHB models
should occur after a loader keyword but before the modelspec keywords
several functions migrated out of old loader keywords
"""

import logging
import re

import numpy as np

from nems.registry import xform, xmodule

log = logging.getLogger(__name__)

@xform()
def pas(loadkey):
    """
    pas = "passive only"
    mask out everything that doesn't fall in a "PASSIVE_EXPERIMENT" epoch
    """

    xfspec = [['nems.preprocessing.mask_keep_passive',
               {}, ['rec'], ['rec']]]

    return xfspec


@xform()
def ap1(loadkey):
    """
    ap1 = "first passive only"
    mask out everything that doesn't fall in an active for first passive FILE_ epoch
    """

    xfspec = [['nems.preprocessing.mask_late_passives',
               {}, ['rec'], ['rec']]]

    return xfspec


@xform()
def cor(kw):
    """
    create mask that removes incorrect trials
    :param kw:
    :return:
    """
    ops = kw.split('.')[1:]

    return [['nems.xforms.mask_incorrect', {}]]


@xform()
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

@xform()
def tar(kw):
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

    return [['nems.xforms.mask_all_but_targets',
             {'include_incorrect': include_incorrect}]]


@xform()
def reftar(kw):
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

    return [['nems_lbhb.preprocessing.mask_all_but_reference_target',
             {'include_incorrect': include_incorrect}]]


@xform()
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

    if loadset[0]=='tar':
        epoch_regex='^TAR_'
        epoch_shift = 5
    elif loadset[0]=='cct':
        epoch_regex = '^[A-Za-z]+_[0-9]+$'
        epoch_shift = 0
    else:
        raise ValueError('unknown stim spec')

    lick = False
    if len(loadset) >= 2:
        if loadset[1] == "lic":
            epoch2_shuffle = False
            lick = True
            epoch2_shift = -5
        elif loadset[1] == "lic0":
            epoch2_shuffle = True
            lick = True
            epoch2_shift = -5
        elif loadset[1] == "l20":
            epoch2_shuffle = False
            lick = True
            epoch2_shift = -20
        elif loadset[1] == "l20x0":
            epoch2_shuffle = True
            lick = True
            epoch2_shift = -20

        else:
            raise ValueError('evs option 2 not known')
    if lick:
        xfspec = [['nems.preprocessing.generate_stim_from_epochs',
                   {'new_signal_name': 'stim',
                    'epoch_regex': epoch_regex, 'epoch_shift': epoch_shift,
                    'epoch2_regex': 'LICK', 'epoch2_shift': epoch2_shift,
                    'epoch2_shuffle': epoch2_shuffle, 'onsets_only': True},
                   ['rec'], ['rec']]]
    else:
        xfspec = [['nems.preprocessing.generate_stim_from_epochs',
                   {'new_signal_name': 'stim',
                    'epoch_regex': epoch_regex, 'epoch_shift': epoch_shift,
                    'onsets_only': True},
                   ['rec'], ['rec']]]

    if loadset[0]=='tar':
        xfspec.append(['nems.xforms.mask_all_but_targets', {}])

    return xfspec


@xform()
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
        elif l.startswith("pvp"):
            this_sig = ["pupil_dup"]
        elif l.startswith("pwp"):
            this_sig = ["pupil_dup2"]
        elif l.startswith("pxb"):
            this_sig = ["p_x_a"]
        elif l.startswith("pxf"):
            this_sig = ["p_x_f"]
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
        elif l.startswith("afl"):
            this_sig = ["each_active"]
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


@xform()
def sml(kw):
    """
    set sm_win_len variable
    """
    ops = kw.split(".")[1:]
    sm_win_len = 180
    for op in ops:
        sm_win_len = float(op)
    xfspec = [['nems.xforms.init_context', {'sm_win_len': sm_win_len}]]

    return xfspec


@xform()
def rstate(kw):
    ops = kw.split(".")[1:]
    dopt = {}
    for op in ops:
        if op=='sh':
           dopt['shuffle_interactions'] = True
        elif op.startswith('s'):
           dopt['smooth_window'] = int(op[1:])

    return [['nems_lbhb.preprocessing.state_resp_outer', dopt]]


@xform()
def inp(loadkey):
    """
    inp = 'input signal'
    add the following signal as new channel(s) in the input signal
    """
    pattern = re.compile(r'^inp\.([a-zA-Z0-9\.]*)$')
    parsed = re.match(pattern, loadkey)
    loader = parsed.group(1)

    input_signals = []

    loadset = loader.split(".")
    for l in loadset:
        if l.startswith("pup"):
            this_sig = ["pupil"]
        elif l.startswith("pbs"):
            this_sig = ["pupil_bs"]
        input_signals.extend(this_sig)
    xfspec = [['nems.xforms.concatenate_input_channels',
               {'input_signals': input_signals}]]

    return xfspec


@xform()
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


@xform()
def pca(loadkey):
    """
    compute pca (or some other state-space) on response
    """

    ops = loadkey.split(".")[1:]
    pc_source = "psth"
    overwrite_resp = True
    pc_count=None
    pc_idx=None
    compute_power = 'no'
    whiten = True
    for op in ops:
        if op == "psth":
            pc_source = "psth"
        elif op == "all":
            pc_source = "all"
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
        elif op.startswith("p"):
            compute_power = "single_trial"

    if pc_idx is not None:
        xfspec = [['nems.preprocessing.resp_to_pc',
                   {'pc_source': pc_source, 'overwrite_resp': overwrite_resp,
                    'pc_count': pc_count, 'pc_idx': pc_idx, 'compute_power': compute_power, 'whiten': whiten}]]
    else:
        xfspec = [['nems.preprocessing.resp_to_pc',
                   {'pc_source': pc_source, 'overwrite_resp': overwrite_resp,
                    'pc_count': pc_count, 'compute_power': compute_power, 'whiten': whiten}]]

    return xfspec


@xform()
def popev(loadkey):
    return [['nems_lbhb.xform_wrappers.split_pop_rec_by_mask', {}]]


@xform()
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
        elif op.startswith('off'):
            kwargs['offset'] = int(op[3:])

    return [['nems_lbhb.gcmodel.contrast.add_contrast', kwargs]]


@xform()
def csum(loadkey):
    ops = loadkey.split('.')[1:]
    kwargs = {}
    for op in ops:
        if op.startswith('off'):
            offsets = int(op[3:])
            kwargs['offsets'] = np.array([[offsets]])

    return [['nems_lbhb.gcmodel.contrast.sum_contrast', kwargs]]


@xform()
def onoff(loadkey):
    return [['nems_lbhb.gcmodel.contrast.add_onoff', {}]]


@xform()
def hrc(load_key):
    """
    Mask only data during stimuli that were repeated 10 or greater times.
    hrc = high rep count
    """

    xfspec = [['nems_lbhb.preprocessing.mask_high_repetion_stims',
               {'epoch_regex':'^STIM_'},
                 ['rec'], ['rec']]]

    return xfspec


@xform()
def pbal(load_key):
    """
    Mask only epochs that are presented equally between large/small pupil conditions
    """
    xfspec = [['nems_lbhb.preprocessing.mask_pupil_balanced_epochs',
                {},
                ['rec'], ['rec']]]

    return xfspec

@xform()
def plgsm(load_key):
    """
    Create masks for large and small pupl
    """
    ops = load_key.split('.')[1:]
    kwargs = {}
    evoked_only = False
    ev_bins = 0
    add_per_stim = ('s' in ops)
    split_per_stim = ('sp' in ops)
    for op in ops:
        if op[:1] == 'e':
            evoked_only=True
            if len(op) > 1:
                ev_bins = int(op[1:])
    xfspec = [['nems_lbhb.preprocessing.pupil_large_small_masks', 
               {'evoked_only': evoked_only, 'ev_bins': ev_bins, 'add_per_stim': add_per_stim, 'split_per_stim': split_per_stim}]]

    return xfspec


@xform()
def ev(load_key):
    """
    Mask only evoked data
    """

    xfspec = [['nems_lbhb.preprocessing.mask_evoked', {}, ['rec'], ['rec']]]

    return xfspec


@xform()
def apm(load_key):
    """
    Add a mask signal ('p_mask') for pupil that can be used later on in fitting.
    Doesn't go in "true" mask signal.
    """

    xfspec = [['nems_lbhb.preprocessing.add_pupil_mask',
            {},
            ['rec'], ['rec']]]

    return xfspec


@xform()
def pm(load_key):
    """
    pm = pupil mask
    pm.b = mask only big pupil trials
    pm.s = mask only small pupil trials
    pm.s.bv = mask small pupil and balance big/small ref epochs for val set
            (bv is important for the nems.metrics that get calculated at the end)
    performs an AND mask (so will only create mask inside the existing current
        mask. If mask is None, creates mask with: rec = rec.create_mask(True))
    """
    raise DeprecationWarning("Is anyone using this??")
    options = load_key.split('.')
    if len(options)>1:
        if options[1] == 'b':
            condition='large'
        elif options[1] == 's':
            condition = 'small'
        else:
            log.info("unknown option passed to pupil mask...")

    balance = False
    if len(options)>2:
        if options[2] == 'bv':
            balance = True

    xfspec = [['nems_lbhb.preprocessing.pupil_mask',
            {'condition': condition, 'balance': balance},
            ['est', 'val'], ['est', 'val']]]

    return xfspec


@xform()
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


@xform()
def tor(load_ley):
    """
    Mask only TORC data
    """
    xfspec = [['nems_lbhb.preprocessing.mask_tor',
               {}, ['rec'], ['rec']]]

    return xfspec


@xform()
def nat(load_ley):
    """
    Mask only NAT data
    """
    xfspec = [['nems_lbhb.preprocessing.mask_nat',
               {}, ['rec'], ['rec']]]

    return xfspec


@xform()
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


@xform()
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
    channel_per_stim = ('sep' in options)
    if 'tar' in options:
        epoch_regex = '^(STIM_|TAR_|REF_|CAT_)'
        #epoch_regex='^TAR_'
    elif 'stimtar' not in options:
        epoch_regex = '^(STIM_|TAR_|REF_|CAT_)'
        #epoch_regex = '^STIM_'
    else:
        epoch_regex = '^(STIM_|TAR_|REF_|CAT_)'

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
                     {'smooth_resp': smooth, 'use_as_input': use_as_input,
                      'epoch_regex': epoch_regex, 'channel_per_stim': channel_per_stim}]]
    return xfspec


@xform()
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


@xform()
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


@xform()
def stSPO(load_key):
    #add SPO state signal
    permute=False
    options = load_key.split('.')[1:]
    permute = ('0' in options)
    baseline = ('nb' not in options)
    return [['nems_lbhb.SPO_helpers.add_coherence_as_state',{'permute':permute,'baseline':baseline}]]


@xform()
def stimenv(load_key):
    return [['nems_lbhb.preprocessing.transform_stim_envelope', {},
            ['rec'], ['rec']]]


@xform()
def residual(load_key):
    """
    Add residual signal to be used for pupil latent variable creation.
    Because LV creation happens dynamically during the fit,
    want to create this signal first so that shuffling
    (if specified) only happens one time on the outside.
    """
    options = load_key.split('.')

    shuffle = False
    cutoff = None
    signal = 'psth_sp'
    for op in options:
        if op.endswith('0'):
            shuffle = True
        elif op.startswith('hp'):
            cutoff = np.float(op[2:].replace(',','.'))
        elif op.startswith('pred'):
            signal = 'pred'

    xfspec = [['nems_lbhb.preprocessing.create_residual',
            {'shuffle': shuffle,
            'cutoff': cutoff,
            'signal': signal},
            ['rec'], ['rec']]]

    return xfspec


@xform()
def epsig(load_key):
    """
    Create epoch signal from epochs so that cost function has access to
    stim epoch times
    """

    xfspec = [['nems_lbhb.preprocessing.add_epoch_signal',
                {},
                ['rec'], ['rec']]]

    return xfspec


@xform()
def addmeta(load_key):
    """
    Add meta data to recording that can be used later on in the fit. For example,
    information about epochs could be useful.
    """

    xfspec = [['nems_lbhb.preprocessing.add_meta',
                {},
                ['rec'], ['rec']]]

    return xfspec


@xform()
def rz(load_key):
    """
    Transform resp into zscore. Add signal 'raw_resp' for original resp
    signal.
    """

    xfspec = [['nems_lbhb.preprocessing.zscore_resp',
                {}, ['rec'], ['rec']]]

    return xfspec

@xform()
def esth1(kw):
    ops = kw.split('.')[1:]
    if len(ops) > 0:
        seed_idx = int(ops[0])
    else:
        seed_idx = 0
    return [['nems_lbhb.gcmodel.initializers.est_halved', {'half': 1,
                                                           'seed_idx': seed_idx}]]

@xform()
def esth2(kw):
    ops = kw.split('.')[1:]
    if len(ops) > 0:
        seed_idx = int(ops[0])
    else:
        seed_idx = 0
    return [['nems_lbhb.gcmodel.initializers.est_halved', {'half': 2,
                                                           'seed_idx': seed_idx}]]
