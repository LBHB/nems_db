import logging

from nems.fitters.api import coordinate_descent, scipy_minimize

log = logging.getLogger(__name__)


def generate_loader_xfspec(loader, recording_uri):

    recordings = [recording_uri]

    if loader in ["ozgf100ch18", "ozgf100ch18n"]:
        normalize = int(loader == "ozgf100ch18n")
        xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems_lbhb.old_xforms.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems_lbhb.old_xforms.xforms.average_away_stim_occurrences',{}]]

    elif loader in ["ozgf100ch18pup", "ozgf100ch18npup"]:
        normalize = int(loader == "ozgf100ch18npup")
        xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems_lbhb.old_xforms.xforms.make_state_signal',
                   {'state_signals': ['pupil'], 'permute_signals': [],
                    'new_signalname': 'state'}]]

    elif loader in ["env100","env100n"]:
        normalize = int(loader == "env100n")
        xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems_lbhb.old_xforms.xforms.split_by_occurrence_counts',
                   {'epoch_regex': '^STIM_'}],
                  ['nems_lbhb.old_xforms.xforms.average_away_stim_occurrences', {}]]

    elif loader in ["env100pt","env100ptn"]:
        normalize = int(loader == "env100ptn")
        xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                   {'recording_uri_list': recordings, 'normalize': normalize}],
                  ['nems_lbhb.old_xforms.xforms.use_all_data_for_est_and_val',
                   {}]]

    elif loader == "nostim10pup":
        # DEPRECATED?
        xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings', {'recording_uri_list': recordings}],
                  ['nems.preprocessing.make_state_signal', {'state_signals': ['pupil'], 'permute_signals': [], 'new_signalname': 'state'},['rec'],['rec']]]

    elif (loader.startswith("evt")):

        epoch2_shuffle = False
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
        else:
            raise ValueError("unknown state_signals for evt loader")

        if loader.startswith("evt20pup"):
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems.preprocessing.generate_stim_from_epochs',
                       {'new_signal_name': 'stim',
                        'epoch_regex': '^TAR_', 'epoch_shift': 5,
                        'epoch2_regex': 'LICK', 'epoch2_shift': -5,
                        'epoch2_shuffle': epoch2_shuffle, 'onsets_only': True},
                       ['rec'], ['rec']],
                      ['nems_lbhb.old_xforms.xforms.mask_all_but_targets', {}]]

        else:
            raise ValueError("unknown evt loader keyword")


    elif (loader.startswith("psth") or loader.startswith("nostim") or
          loader.startswith("env")):

        if loader.endswith("tarbehlic"):
            state_signals = ['active','lick']
            permute_signals = []
        elif loader.endswith("tarbeh0lic"):
            state_signals = ['active','lick']
            permute_signals = ['lick']
        elif loader.endswith("tarbehlic0"):
            state_signals = ['active','lick']
            permute_signals = ['lick']
        elif loader.endswith("tarbeh0lic0"):
            state_signals = ['active','lick']
            permute_signals = ['active','lick']
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

        else:
            raise ValueError("invalid loader string")

        if loader.startswith("psth20tar"):
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems_lbhb.old_xforms.xforms.mask_all_but_targets', {}],
                      ['nems_lbhb.old_xforms.xforms.generate_psth_from_resp',
                       {'epoch_regex': '^TAR_'}]]

        elif loader.startswith("psths"):
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems_lbhb.old_xforms.xforms.mask_all_but_correct_references', {}],
                      ['nems_lbhb.old_xforms.xforms.generate_psth_from_resp',
                       {'smooth_resp': True}]]

        elif loader.startswith("psthm"):
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems_lbhb.old_xforms.xforms.remove_all_but_correct_references', {}],
                      ['nems_lbhb.old_xforms.xforms.generate_psth_from_resp', {}]]

        elif loader.startswith("psth"):
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems_lbhb.old_xforms.xforms.mask_all_but_correct_references', {}],
                      ['nems_lbhb.old_xforms.xforms.generate_psth_from_resp', {}]]

        elif loader.startswith("envm"):
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems_lbhb.old_xforms.xforms.remove_all_but_correct_references', {}]]

        elif loader.startswith("env"):
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems_lbhb.old_xforms.xforms.mask_all_but_correct_references', {}]]

        else:
            xfspec = [['nems_lbhb.old_xforms.xforms.load_recordings',
                       {'recording_uri_list': recordings}],
                      ['nems_lbhb.old_xforms.xforms.make_state_signal',
                       {'state_signals': state_signals,
                        'permute_signals': permute_signals,
                        'new_signalname': 'state'}],
                      ['nems_lbhb.old_xforms.xforms.remove_all_but_correct_references', {}]]
        # end of psth / env loader processing
    else:
        raise ValueError('unknown loader string')

    return xfspec


def generate_fitter_xfspec(fitkey, fitkey_kwargs=None):

    xfspec = []
    pfolds = 10

    # parse the fit spec: Use gradient descent on whole data set(Fast)
    if fitkey in ["fit01", "basic"]:
        # prefit strf
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

    elif fitkey in ["fit01a", "basicqk"]:
        # prefit strf
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic',
                       {'max_iter': 1000, 'tolerance': 1e-5}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

    elif fitkey in ["fit01b", "basic-shr"]:
        # prefit strf
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic_shr_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic',
                       {'shrinkage': 1, 'tolerance': 1e-8}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey in ["fit01b", "basic-cd"]:
        # prefit strf
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic_cd', {'shrinkage': 0}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey in ["fit01b", "basic-cd-shr"]:
        # prefit strf
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic_cd',
                       {'shrinkage': 1, 'tolerance': 1e-8}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey == "fitjk01":

        log.info("n-fold fitting...")
        xfspec.append(['nems_lbhb.old_xforms.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

    elif fitkey == "state01-jkm":

        xfspec.append(['nems_lbhb.old_xforms.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_state_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey == "state01-jk":

        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_state_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold', {}])  # 'ftol': 1e-6
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey == "state01-jk-shr":

        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_state_init', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold_shrinkage', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif (fitkey == "basic-nf"):

        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey == "cd-nf":

        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems_lbhb.old_xforms.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_cd_nfold', {'ftol': 1e-6}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

    elif (fitkey == "fitpjk01") or (fitkey == "basic-nfm"):

        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems_lbhb.old_xforms.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif (fitkey == "basic-nftrial"):

        log.info("n-fold fitting...")
        tfolds = 5
        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': tfolds, 'epoch_name': 'TRIAL'}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey == "basic-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems_lbhb.old_xforms.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_nfold_shrinkage', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

    elif fitkey == "cd-nf-shr":

        log.info("n-fold fitting...")
        xfspec.append(['nems_lbhb.old_xforms.xforms.mask_for_jackknife',
                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
        # xfspec.append(['nems_lbhb.old_xforms.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_cd_nfold_shrinkage', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

#    elif fitkey == "iter-cd-nf-shr":
#
#        log.info("Iterative cd, n-fold, shrinkage fitting...")
#        xfspec.append(['nems_lbhb.old_xforms.xforms.split_for_jackknife',
#                       {'njacks': pfolds, 'epoch_name': 'REFERENCE'}])
#        #xfspec.append(['nems_lbhb.old_xforms.xforms.generate_psth_from_est_for_both_est_and_val_nfold', {}])
#        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_iter_cd_nfold_shrink', {}])
#        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

    elif fitkey == "fit02":
        # no pre-fit
        log.info("Performing full fit...")
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_basic', {}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict',    {}])

    elif fitkey == "fitsubs":
        '''fit_subsets with scipy_minimize'''
        kw_list = ['module_sets', 'tolerance', 'fitter']
        defaults = [None, 1e-4, coordinate_descent]
        module_sets, tolerance, my_fitter = \
            _get_my_kwargs(fitkey_kwargs, kw_list, defaults)
        xfspec.append([
                'nems_lbhb.old_xforms.xforms.fit_module_sets',
                {'module_sets': module_sets, 'fitter': scipy_minimize,
                 'tolerance': tolerance}
                ])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey.startswith("fitsubs"):
        xfspec.append(_parse_fitsubs(fitkey))
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey == "fititer":
        kw_list = ['module_sets', 'tolerances', 'tol_iter', 'fit_iter',
                   'fitter']
        defaults = [None, None, 100, 20, coordinate_descent]
        module_sets, tolerances, tol_iter, fit_iter, my_fitter = \
            _get_my_kwargs(fitkey_kwargs, kw_list, defaults)
        xfspec.append([
                'nems_lbhb.old_xforms.xforms.fit_iteratively',
                {'module_sets': module_sets, 'fitter': my_fitter,
                 'tolerances': tolerances, 'tol_iter': tol_iter,
                 'fit_iter': fit_iter}
                ])
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey.startswith("fititer") or fitkey.startswith("iter"):
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_iter_init', {}])
        xfspec.append(_parse_fititer(fitkey))
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    elif fitkey.startswith("state"):
        xfspec.append(['nems_lbhb.old_xforms.xforms.split_for_jackknife',
                       {'njacks': 5, 'epoch_name': 'REFERENCE'}])
        xfspec.append(['nems_lbhb.old_xforms.xforms.fit_state_init', {}])
        xfspec.append(_parse_fititer(fitkey))
        xfspec.append(['nems_lbhb.old_xforms.xforms.predict', {}])

    else:
        raise ValueError('unknown fitter string ' + fitkey)

    return xfspec


def _get_my_kwargs(kwargs, kw_list, defaults):
    '''Fetch value of kwarg if given, otherwise corresponding default'''
    my_kwargs = []
    for kw, default in zip(kw_list, defaults):
        if kwargs is None:
            a = default
        else:
            a = kwargs.pop(kw, default)
        my_kwargs.append(a)
    return my_kwargs


def _parse_fititer(fit_keyword):
    # ex: fititer01-T4-T6-S0x1-S0x1x2x3-ti50-fi20
    # fitter: scipy_minimize; tolerances: [1e-4, 1e-6]; s
    # subsets: [[0,1], [0,1,2,3]]; tol_iter: 50; fit_iter: 20;
    # Note that order does not matter except for starting with
    # 'fititer<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = 'scipy_minimize'
    elif fit.endswith('02'):
        fitter = 'coordinate_descent'
    else:
        fitter = 'coordinate_descent'
        log.warn("Unrecognized or unspecified fit algorithm for fititer: %s\n"
                 "Using default instead: %s", fit, fitter)

    tolerances = []
    module_sets = []
    fit_iter = None
    tol_iter = None

    for c in chunks[1:]:
        if c.startswith('ti'):
            tol_iter = int(c[2:])
        elif c.startswith('fi'):
            fit_iter = int(c[2:])
        elif c.startswith('T'):
            power = int(c[1:])*-1
            tol = 10**(power)
            tolerances.append(tol)
        elif c.startswith('S'):
            indices = [int(i) for i in c[1:].split('x')]
            module_sets.append(indices)
        else:
            log.warning(
                    "Unrecognized segment in fititer keyword: %s\n"
                    "Correct syntax is:\n"
                    "fititer<fitter>-S<i>x<j>...-T<tolpower>...ti<tol_iter>"
                    "-fi<fit_iter>", c
                    )

    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return ['nems_lbhb.old_xforms.xforms.fit_iteratively',
            {'module_sets': module_sets, 'fitter': fitter,
             'tolerances': tolerances, 'tol_iter': tol_iter,
             'fit_iter': fit_iter}]


def _parse_fitsubs(fit_keyword):
    # ex: fitsubs02-S0x1-S0x1x2x3-it1000-T6
    # fitter: scipy_minimize; subsets: [[0,1], [0,1,2,3]];
    # max_iter: 1000;
    # Note that order does not matter except for starting with
    # 'fitsubs<some number>' to specify the analysis and fit algorithm
    chunks = fit_keyword.split('-')

    fit = chunks[0]
    if fit.endswith('01'):
        fitter = scipy_minimize
    elif fit.endswith('02'):
        fitter = coordinate_descent
    else:
        fitter = coordinate_descent
        log.warn("Unrecognized or unspecified fit algorithm for fitsubs: %s\n"
                 "Using default instead: %s", fit[7:], fitter)

    module_sets = []
    max_iter = None
    tolerance = None

    for c in chunks[1:]:
        if c.startswith('it'):
            max_iter = int(c[2:])
        elif c.startswith('S'):
            indices = [int(i) for i in c[1:].split('x')]
            module_sets.append(indices)
        elif c.startswith('T'):
            power = int(c[1:])*-1
            tolerance = 10**(power)
        else:
            log.warning(
                    "Unrecognized segment in fitsubs keyword: %s\n"
                    "Correct syntax is:\n"
                    "fitsubs<fitter>-S<i>x<j>...-T<tolpower>-it<max_iter>", c
                    )

    if not module_sets:
        module_sets = None

    return ['nems_lbhb.old_xforms.xforms.fit_iteratively',
            {'module_sets': module_sets, 'fitter': fitter,
             'tolerance': tolerance, 'max_iter': max_iter}]
