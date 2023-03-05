import logging
import re

from nems0.registry import xform, xmodule
import nems0.db as nd

log = logging.getLogger(__name__)

# TODO: Delete after finished deprecating.
# Replaced with: load, splitcount, avgep, st, contrast

def _load_dict(loadkey, cellid=None, batch=None):
    d = {'loadkey': loadkey}
    if cellid is not None:
        d['cellid'] = cellid
    if batch is not None:
        d['batch'] = batch
    return d

def _parse_baphy_loadkey(loadkey, cellid=None, batch=None, siteid=None, **options):

    from nems_lbhb.xform_wrappers import generate_recording_uri
    import nems_lbhb.baphy as nb
    from nems_lbhb import baphy_io

    pc_idx = None

    if type(cellid) is str:
        cc = cellid.split("_")
        if (len(cc) > 1) and (cc[1][0] == "P"):
            pc_idx = [int(cc[1][1:])]
            cellid = cc[0]

        elif (len(cellid.split('+')) > 1):
            # list of cellids (specified in model queue by separating with '_')
            cellid = cellid.split('+')

    recording_uri = generate_recording_uri(cellid=cellid, batch=batch,
                                           loadkey=loadkey, siteid=siteid)
    if type(recording_uri) is list:
        recording_uri_list=recording_uri
    else:
        recording_uri_list=[recording_uri]

    # update the cellid in context so that we don't have to parse the cellid
    # again in xforms
    t_ops = {} # options.copy()
    t_ops['cellid'] = cellid
    t_ops['batch'] = batch
    if cellid in ['none', 'NAT3', 'NAT4', 'NAT4v2', 'NAT1', 'ALLCELLS']:
        cells_to_extract = cellid
    else:
        cells_to_extract, _ = baphy_io.parse_cellid(t_ops)

    context = {'recording_uri_list': recording_uri_list, 'cellid': cells_to_extract}

    if pc_idx is not None:
        context['pc_idx'] = pc_idx

    return [['nems0.xforms.init_context', context]]


@xform()
def env(loadkey, cellid=None, batch=None, siteid=None, **options):
    """
    envelope loader
       extra parameters handled by loadkey parser in baphy_load_wrapper
    """

    d = _load_dict(loadkey, cellid, batch)
    xfspec = [['nems_lbhb.xform_wrappers.baphy_load_wrapper', d]]
    return xfspec


@xform()
def psth(loadkey, cellid=None, batch=None, siteid=None, **options):
    """
    psth loader (no stim)
       extra parameters handled by loadkey parser in baphy_load_wrapper
    """
    return _parse_baphy_loadkey(loadkey, cellid=cellid, batch=batch, siteid=siteid, **options)

    #d = _load_dict(loadkey, cellid, batch)
    #xfspec = [['nems_lbhb.xform_wrappers.baphy_load_wrapper', d]]
    #return xfspec


@xform()
def gtgram(loadkey, cellid=None, batch=None, siteid=None, **options):
    """
    gammatone filter, nems built-in, reads from wav
       extra parameters handled by loadkey parser in baphy_load_wrapper
    """
    return _parse_baphy_loadkey(loadkey, cellid=cellid, batch=batch, siteid=siteid, **options)


@xform()
def ozgf(loadkey, cellid=None, batch=None, siteid=None, **options):
    """
    gammatone filter
       extra parameters handled by loadkey parser in baphy_load_wrapper
    """
    return _parse_baphy_loadkey(loadkey, cellid=cellid, batch=batch, siteid=siteid, **options)

    #d = _load_dict(loadkey, cellid, batch)
    #xfspec = [['nems_lbhb.xform_wrappers.baphy_load_wrapper', d]]
    #return xfspec


@xform()
def parm(loadkey, cellid=None, batch=None, siteid=None, **options):
    """
    parm spectrogram
       extra parameters handled by loadkey parser in baphy_load_wrapper
    """
    d = _load_dict(loadkey, cellid, batch)
    xfspec = [['nems_lbhb.xform_wrappers.baphy_load_wrapper', d]]
    return xfspec

@xform()
def ll(loadkey, cellid=None, batch=None, siteid=None, **options):
    """
    labeled line spectrogram
       extra parameters handled by loadkey parser in baphy_load_wrapper
    """
    d = _load_dict(loadkey, cellid, batch)
    xfspec = [['nems_lbhb.xform_wrappers.baphy_load_wrapper', d]]
    return xfspec


@xform()
def ns(loadkey, cellid=None, batch=None, siteid=None, **options):
    d = _load_dict(loadkey, cellid, batch)
    xfspec = [['nems_lbhb.xform_wrappers.baphy_load_wrapper', d]]
    return xfspec


@xform()
def SPOld(loadkey, recording_uri=None, cellid=None):
    import nems0.plugins.default_loaders
    xfspec = nems0.plugins.default_loaders.ld(loadkey, recording_uri=recording_uri,cellid=cellid)
    xfspec.append(['nems_lbhb.SPO_helpers.load',{}])
    return xfspec


@xform()
def SPOsev(kw):
    epoch_regex = '^STIM'
    xfspec = [['nems_lbhb.SPO_helpers.split_by_occurrence_counts_SPO',
               {'epoch_regex': epoch_regex}]]
    xfspec.append(['nems0.xforms.average_away_stim_occurrences',
     {'epoch_regex': epoch_regex}])
    return xfspec

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
#        xfspec = [['nems0.xforms.load_recordings',
#                   {'recording_uri_list': recordings, 'normalize': normalize}],
#                  ['nems0.xforms.make_state_signal',
#                   {'state_signals': ['pupil'], 'permute_signals': [],
#                    'new_signalname': 'state'}]]
#    else:
#        xfspec = [['nems0.xforms.load_recordings',
#                   {'recording_uri_list': recordings, 'normalize': normalize}],
#                  ['nems0.xforms.split_by_occurrence_counts',
#                   {'epoch_regex': '^STIM_'}],
#                  ['nems0.xforms.average_away_stim_occurrences', {}]]
#
#    if contrast:
#        xfspec.insert(1, ['nems0.xforms.add_contrast', {}])
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
#    xfspec = [['nems0.xforms.load_recordings',
#               {'recording_uri_list': recordings, 'normalize': normalize}]]
#    if use_state:
#        xfspec.append(['nems0.xforms.make_state_signal',
#                       {'state_signals': state_signals,
#                        'permute_signals': permute_signals,
#                        'new_signalname': 'state'}])
#        if mask:
#            xfspec.append(['nems0.xforms.remove_all_but_correct_references',
#                           {}])
#    elif pt:
#        xfspec.append(['nems0.xforms.use_all_data_for_est_and_val', {}])
#    else:
#        xfspec.extend([['nems0.xforms.split_by_occurrence_counts',
#                        {'epoch_regex': '^STIM_'}],
#                       ['nems0.xforms.average_away_stim_occurrences', {}]])
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
##    xfspec = [['nems0.xforms.load_recordings',
##               {'recording_uri_list': recordings}],
##              ['nems0.xforms.make_state_signal',
##               {'state_signals': state_signals,
##                'permute_signals': permute_signals,
##                'new_signalname': 'state'}]]
#    xfspec = [['nems0.xforms.load_recordings',
#               {'recording_uri_list': recordings}]]
#    if mask:
#        xfspec.append(['nems0.xforms.remove_all_but_correct_references', {}])
#    elif tar:
#        epoch_regex = '^TAR_'
#        xfspec.append(['nems0.xforms.mask_all_but_targets', {}])
#    else:
#        xfspec.append(['nems0.xforms.mask_all_but_correct_references', {}])
#
#    xfspec.append(['nems0.xforms.generate_psth_from_resp',
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
#    xfspec = [['nems0.xforms.load_recordings',
#               {'recording_uri_list': recordings}],
#              ['nems0.xforms.make_state_signal',
#               {'state_signals': state_signals,
#                'permute_signals': permute_signals,
#                'new_signalname': 'state'}],
#              ['nems0.preprocessing.generate_stim_from_epochs',
#               {'new_signal_name': 'stim',
#                'epoch_regex': '^TAR_', 'epoch_shift': 5,
#                'epoch2_regex': 'LICK', 'epoch2_shift': -5,
#                'epoch2_shuffle': epoch2_shuffle, 'onsets_only': True},
#               ['rec'], ['rec']],
#              ['nems0.xforms.mask_all_but_targets', {}]]
#
#    return xfspec


@xform()
def loadpop(loadkey):
    ops = loadkey.split('.')[1:]

    rand_match = False
    cell_count = 20
    best_cells = False
    holdout = None
    for op in ops:
        if op=='rnd':
            rand_match = True
        elif op=='bth':
            rand_match = 'both'
        elif op.startswith('cc'):
            cell_count = int(op[2:])
        elif op.startswith('bc'):
            cell_count = int(op[2:])
            best_cells=True
        elif op=='hs':
            holdout='site'
        elif op=='hm':
            holdout='matched'
        

    xfspec = [['nems_lbhb.xform_wrappers.pop_selector',
              {'loadkey': loadkey,
               'rand_match': rand_match, 'cell_count': cell_count,
               'best_cells': best_cells, 'holdout': holdout}]]

    return xfspec


@xform()
def cc(loadkey):
    options = loadkey.split('.')
    # First option is to pick # random cells, ex: cc.10
    # cell count of 0 loads all cells
    cell_count = int(options[1])
    seed_mod = 0
    for op in options[2:]:
        if op.startswith('sd'):
            seed_mod = int(op[2:])
        if op.startswith('xx'):
            try:
                # int for random number of cellids to exclude (all sites)
                exclusions = int(op[2:])
            except ValueError:
                # or specify a site to exclude. repeat option to exclude more than one.
                site = op[2:]
                if exclusions is None:
                    exclusions = [site]
                else:
                    exclusions += site

    xfspec = [['nems_lbhb.xform_wrappers.select_cell_count', {'cell_count': cell_count, 'seed_mod': seed_mod}]]

    return xfspec


@xform()
def mc(loadkey):
    seed_mod = 0
    options = loadkey.split('.')
    n_cells = -1

    for i, op in enumerate(options[1:]):
        if ':' in op and 'DRX' in options[i]:
            # special fix for DRX split siteids, they'll get split on the '.' but they shouldn't be
            op2 = options.pop(i+1)
            op1 = options.pop(i)
            options.append(op1 + '.' +  op2)

    for op in options[1:]:
        if op.startswith('sd'):
            seed_mod = int(op[2:])
        else:
            n_cells = int(op)

    if n_cells == -1:
        raise ValueError('an n_cells option must be specified for keyword: mc. ex:  mc.5')

    xfspec = [['nems_lbhb.xform_wrappers.max_cells', {'n_cells': n_cells, 'seed_mod': seed_mod}]]

    return xfspec


@xform()
def hc(loadkey):
    seed_mod = 0
    site = None
    options = loadkey.split('.')
    exclude_matched_cells = False

    for i, op in enumerate(options[1:]):
        if ':' in op and 'DRX' in options[i]:
            # special fix for DRX split siteids, they'll get split on the '.' but they shouldn't be
            op2 = options.pop(i+1)
            op1 = options.pop(i)
            options.append(op1 + '.' +  op2)

    for op in options[1:]:
        if op == 'ms':
            exclude_matched_cells = True
        else:
            site = op

    xfspec = [['nems_lbhb.xform_wrappers.holdout_cells', {'site': site, 'exclude_matched_cells': exclude_matched_cells}]]

    return xfspec


@xform()
def loadpred(loadkey):
    ops = loadkey.split('.')[1:]

    pc_count = 0
    pc_format = "psth"
    dmask = 'hrc'
    for op in ops:
        if op == 'rnd':
            rand_match = True
        if op[:2] == 'pc':
            pc_count = int(op[2:])
        if op == 'n':
            pc_format="noise"
        if op == 'a':
            pc_format="all"
        if op == 'cpn':
            dmask = 'epcpn-'+dmask
        if op.startswith('cpnmvm'):
            mvmmask = op[3:].replace(',', '.')
            dmask = f'epcpn-{mvmmask}-hrc'
        if op.startswith('cpnOldmvm'):
            mvmmask = op[6:].replace(',', '.')
            dmask = f'epcpn.old-{mvmmask}-hrc'
    if pc_count > 0:
        modelname_existing = f"psth.fs4.pup-ld-st.pup-{dmask}-pca.{pc_format}.cc{pc_count}-psthfr-aev_sdexp2.SxR_newtf.n.lr3e4.cont.et5.i50000"
    elif 'z' in ops:
        modelname_existing = f"psth.fs4.pup-ld-norm.r.ms-st.pup-{dmask}-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000"
    else:
        # modelname_existing = "psth.fs4.pup-ld-st.pup-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont"
        modelname_existing = f"psth.fs4.pup-ld-st.pup-{dmask}-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000"
            
    xfspec = [['nems_lbhb.xform_wrappers.load_existing_pred',
              {'modelname_existing': modelname_existing}]]

    return xfspec


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
