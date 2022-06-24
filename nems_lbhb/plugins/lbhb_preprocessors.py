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
    parms = {}
    for op in ops:
        if op.startswith('i'):
            s = op[1:].replace('d', '.')
            parms['ITI_sec_to_include'] = float(s)

    return [['nems.xforms.mask_incorrect', parms]]


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

    parms = {'include_incorrect': False}

    ops = kw.split('.')[1:]
    for op in ops:
        if op.startswith('b'):
            parms['balance_rep_count'] = True
        if op.startswith('a'):
            parms['include_incorrect'] = True
        if op.startswith('e'):
            parms['generate_evoked_mask'] = True
        if op.startswith('i'):
            parms['include_ITI'] = True
            if len(op) > 1:
                s = op[1:].replace('d', '.')
                parms['ITI_sec_to_include'] = float(s)

    return [['nems_lbhb.preprocessing.mask_all_but_reference_target', parms]]


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
    pattern = re.compile(r'^st\.([a-zA-Z0-9\W+\.]*)$')
    parsed = re.match(pattern, loadkey)
    loader = parsed.group(1)

    state_signals = []
    permute_signals = []
    generate_signals = []

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
        elif l.startswith("ppp"):
            this_sig = ["pupiln"]
        elif l.startswith("dlp"):
            this_sig = ["dlc_pca"]
        elif l.startswith("pvp"):
            this_sig = ["pupil_dup"]
        elif l.startswith("pwp"):
            this_sig = ["pupil_dup2"]
        elif l.startswith("pxb"):
            this_sig = ["p_x_a"]
        elif l.startswith("pxf"):
            this_sig = ["p_x_f"]
        elif l.startswith("drf"):
            this_sig = ["drift"]
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
        elif l.startswith("pca"):
            this_sig = ["pca"]
        elif l.startswith("fpc"):
            this_sig = ["facepca"]
        else:
            raise ValueError("unknown signal code %s for state variable initializer", l)

        state_signals.extend(this_sig)

        # NEW -- check if we've specified to repeat this signal inside the state signal
        # crh, 14.12.2021
        if len(l.split("+"))>1:
            # repeat this signal r times
            if l.split("+")[1].startswith("r"):
                nReps = int(l.split("+")[1][1:])
                for idx, r in enumerate(range(nReps)):
                    state_signals.extend([ts+"_r"+str(idx+1) for ts in this_sig])
            elif l.split("+")[1].startswith("s"):
                permute_signals.extend(this_sig)
            elif l.split("+")[1].startswith("gp"):
                generate_signals.extend(this_sig)
            else:
                raise ValueError("Unexpected format for specifying state signals")

            # which signals to permute / generate randomly?
            if len(l.split("+"))==3:
                option = l.split("+")[2]
                if option.startswith("s"):
                    # permute the specified signals
                    if "," in option[1:]:
                        modchans = [int(k) for k in option[1:].split(",")]
                    else:
                        modchans = [int(option[1:])]
                    for mchan in modchans:
                        permute_signals.extend([ts+"_r"+str(mchan) if mchan!=0 else ts for ts in this_sig])

                elif option.startswith("gp"):
                    # gp generate the specified signals
                    if "," in option[2:]:
                        modchans = [int(k) for k in option[2:].split(",")]
                    else:
                        modchans = [int(option[2:])]
                    for mchan in modchans:
                        generate_signals.extend([ts+"_r"+str(mchan) if mchan!=0 else ts for ts in this_sig])
                else:
                    raise ValueError("Unexpected format for specifying state signal permutations")

        # old way
        else:
            if l.endswith("0"):
                permute_signals.extend(this_sig)
            if l.endswith("GP"):
                generate_signals.extend(this_sig)

    xfspec = [['nems.xforms.make_state_signal',
               {'state_signals': state_signals,
                'permute_signals': permute_signals,
                'generate_signals': generate_signals,
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
def popstim(kw):
    ops = kw.split(".")[1:]
    dopt = {'outsig': 'stim'}
    for op in ops:
        if op=='sh':
           dopt['shuffle_interactions'] = True
        elif op.startswith('s'):
           dopt['smooth_window'] = int(op[1:])

    return [['nems_lbhb.preprocessing.population_to_signal', dopt]]
    
@xform()
def popstate(kw):
    ops = kw.split(".")[1:]
    dopt = {'sigout': 'state'}
    for op in ops:
        if op=='sh':
           dopt['shuffle_interactions'] = True
        elif op=='x':
           dopt['cross_state'] = True
        elif op.startswith('s'):
           dopt['smooth_window'] = int(op[1:])

    return [['nems_lbhb.preprocessing.population_to_signal', dopt]]

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

    overwrite_resp = True
    pc_count=None
    pc_idx=None
    compute_power = 'no'
    whiten = True

    options = {}
    options['pc_source'] = "psth"
    for op in ops:
        if op == "psth":
            options['pc_source'] = "psth"
        elif op == "p":
            options['pc_source'] = "psth"
            options['resp_sig'] = "psth"
        elif op == "all":
            options['pc_source'] = "all"
        elif op == "noise":
            options['pc_source'] = "noise"
        elif op == "no":
            options['overwrite_resp'] = False
        elif op == "dlc":
            options['resp_sig'] = 'dlc'
            options['overwrite_resp'] = False
            options['pc_sig'] = 'dlc_pca'
        elif op.startswith("cc"):
            options['pc_count'] = int(op[2:])
            options['pc_idx']=list(range(options['pc_count']))
        elif op.startswith("n"):
            n = int(op[1:])
            options['pc_count'] = n+1
            options['pc_idx'] = [n]
        elif op.startswith("p"):
            options['compute_power'] = "single_trial"

    xfspec = [['nems.preprocessing.resp_to_pc', options]]

    return xfspec


@xform()
def popev(loadkey):
    ops = loadkey.split('.')[1:]
    keepfrac = 1.0
    for op in ops:
        if op.startswith("k"):
            keepfrac=int(op[1:]) / 100
    
    return [['nems_lbhb.xform_wrappers.split_pop_rec_by_mask', {'keepfrac': keepfrac}]]


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
def epcpn(load_key):
    """
    Fix epoch naming for cpn data
    """
    ops = load_key.split('.')[1:]
    sequence_only = ('seq' in ops)
    use_old = ('old' in ops) # use old (buggy) code
    xfspec = [['nems_lbhb.preprocessing.fix_cpn_epochs',
               {'sequence_only': sequence_only, 
               'use_old': use_old},
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
    Create masks for large and small pupil
    """
    ops = load_key.split('.')[1:]
    evoked_only = False
    custom_epochs = False
    respsort = False
    pupsort = False
    ev_bins = 0
    add_per_stim = ('s' in ops)
    split_per_stim = ('sp' in ops)
    reduce_mask = ('rm' in ops)
    pca_split = 0
    for op in ops:
        if op[:1] == 'p':
            if len(op) > 1:
                pca_split = int(op[1:])
            else:
                pca_split = 1

        if op[:1] == 'e':
            evoked_only = True
            if len(op) > 1:
                ev_bins = int(op[1:].strip('g').strip('r').strip('p'))
                if 'g' in op:
                    custom_epochs = True
                if 'r' in op:
                    respsort = True
                if 'p' in op:
                    pupsort = True
    xfspec = [['nems_lbhb.preprocessing.pupil_large_small_masks', 
               {'evoked_only': evoked_only, 'ev_bins': ev_bins,
                'add_per_stim': add_per_stim,
                'split_per_stim': split_per_stim,
                'custom_epochs': custom_epochs,
                'reduce_mask': reduce_mask,
                'respsort': respsort,
                'pupsort': pupsort,
               'pca_split': pca_split}]]

    return xfspec

@xform()
def tseg(load_key):
    """
    Create masks for large and small pupl
    """
    ops = load_key.split('.')[1:]
    segment_count = 8
    for op in ops:
        if op[:1] == 's':
            segment_count = int(op[1:])

    xfspec = [['nems_lbhb.preprocessing.mask_time_segments', 
               {'segment_count': segment_count}]]

    return xfspec

@xform()
def mvm(load_key):
    """
    Create masks for movement artifacts
    """
    ops = load_key.split('.')[1:]
    binsize = 1
    threshold = 0.25
    for op in ops:
        if op.startswith('t'):
            # threshold
            tkey = float(op[1:])
            if tkey == 1:
                threshold = tkey
            else:
                threshold = tkey / 100
        if op.startswith('w'):
            # window size in sec
            wkey = float(op[1:])
            if wkey > 10:
                binsize = wkey / 100
            else:
                binsize = wkey 
    
    xfspec = [['nems_lbhb.preprocessing.movement_mask', 
               {'threshold': threshold, 'binsize': binsize}]]

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
    mean_zero = ('z' in options)
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
                     {'smooth_resp': smooth, 'epoch_regex': epoch_regex, 'mean_zero': mean_zero}]]
        else:
            xfspec=[['nems.xforms.generate_psth_from_resp',
                     {'smooth_resp': smooth, 'use_as_input': use_as_input,
                      'epoch_regex': epoch_regex, 'channel_per_stim': channel_per_stim, 'mean_zero': mean_zero}]]
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
    options = load_key.split('.')
    use_mask = False
    for o in options:
        if o=='m':
            use_mask = True
    
    xfspec = [['nems_lbhb.preprocessing.zscore_resp',
                {'use_mask': use_mask}, 
                ['rec'], ['rec']]]

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

@xform('dline')
def dline(kw):
    """
    Stacks the state signal to be expressed as delayed lines, with a fixed delay, and duration,
    taking or not the mean for the duration windo.

    format: dline.{delay:int}.{duration:int}.{use_window_mean:bool}(optional:.i.{iput_signal:str}.o.{ouput_signal:str}).

    not: input and output signal are optional. The default being state->state

    e.g. dline.10.15.1:
    shifts the state signal 10 samples forward (effectively looking 10 samples into the past)
    considers a window of 15 samples looking back (after the shift)
    takes the mean of said window (instead of stacking the 15 samples as delayed lines)

    e.g. dline.10.15.1.i.resp.o.state:
    same as the previous example, exept takes in response data and outputs a recordign with a modified state signal.
    """

    # positional parameter parsing
    arguments = kw.split('.')
    delay, duration, use_window_mean = [parg for idx, parg in enumerate(arguments) if idx in [1,2,3]]

    # keyword arguments, state is default input and output if not specified
    input_matches = re.findall('\.i\.([A-Za-z]+)', kw)
    input_signal = input_matches[0]if input_matches else 'state'

    output_matches = re.findall('\.o\.([A-Za-z]+)', kw)
    output_signal = output_matches[0]if output_matches else 'state'

    return [['nems_lbhb.preprocessing.stack_signal_as_delayed_lines',
             {'signal': input_signal,
              'delay': int(delay),
              'duration': int(duration),
              'use_window_mean':int(use_window_mean),
              'output_signal': output_signal},
             ['rec'], ['rec']
             ]]


@xform('shfcat')
def shfcat(kw):
    """
    shuffle and concatenate signal into a single signal. The shufling is independent (even absetn) for differente signals

    format:

    e.g. shfcat.i.resp0.state.o.state
    meaning it takes stim and shuffles it (0), the state as is (no zero, no shuffle) and concatenate into a signal called
    state (overwriting the original on)
    """
    # keyword arguments, state is default input and output if not specified
    raw_signals = re.findall('\.i\.(.*)\.o\.', kw)[0].split('.')

    output_signal = re.findall('\.o\.(\w+)\Z', kw)[0]

    # parses the input signals. defining the action to do with each according to the code
    code_map = {'0': 'shuffle',
                '1': 'roll'}

    input_signals = list()
    to_shuffle = list()
    for isig in raw_signals:
        code_n = isig[-1]
        if code_n in code_map:
            to_shuffle.append(code_map[code_n])
            input_signals.append(isig[:-1])
        else:
            to_shuffle.append('pass')
            input_signals.append(isig)

    return [['nems_lbhb.preprocessing.shuffle_and_concat_signals',
             {'signals': input_signals,
              'to_shuffle': to_shuffle,
              'output_signal': output_signal},
             ['rec'], ['rec']
             ]]