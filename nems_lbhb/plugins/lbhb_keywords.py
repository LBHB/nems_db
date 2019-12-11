from nems.plugins.default_keywords import wc, lvl, fir, firexp
import re
import logging
import copy

import numpy as np

log = logging.getLogger(__name__)


def ctwc(kw):
    '''
    Same as nems.plugins.keywords.fir but renamed for contrast
    to avoid confusion in the modelname and allow different
    options to be supported if needed.
    '''
    m = wc(kw[2:])
    m['fn_kwargs'].update({'i': 'contrast', 'o': 'ctpred'})
    return m


def gcwc(kw):
    m = wc(kw[2:])
    m['fn_kwargs'].update({'ci': 'contrast', 'co': 'ctpred'})
    m['fn'] = 'nems_lbhb.gcmodel.modules.weight_channels'

    plot = 'nems_lbhb.gcmodel.guiplots.contrast_spectrogram'
    if m.get('plot_fns'):
        m['plot_fns'].append(plot)
    else:
        m['plot_fns'] = [plot]

    return m


def ctfir(kw):
    '''
    Same as nems.plugins.keywords.fir but renamed for contrast
    to avoid confusion in the modelname and allow different
    options to be supported if needed.
    '''
    # TODO: Support separate bank for each logsig parameter?
    #       Or just skip straight to the CD model?

    pattern = re.compile(r'^ctfir\.?(\d{1,})x(\d{1,})x?(\d{1,})?$')
    parsed = re.match(pattern, kw)
    n_outputs = int(parsed.group(1))
    n_coefs = int(parsed.group(2))
    n_banks = parsed.group(3)  # None if not given in keyword string
    if n_banks is None:
        n_banks = 1
    else:
        n_banks = int(n_banks)

    p_coefficients = {
        'mean': np.zeros((n_outputs * n_banks, n_coefs)),
        'sd': np.ones((n_outputs * n_banks, n_coefs)),
    }

    if n_coefs > 2:
        # p_coefficients['mean'][:, 1] = 1
        # p_coefficients['mean'][:, 2] = -0.5
        p_coefficients['mean'][:, 1] = 1
    else:
        p_coefficients['mean'][:, 0] = 1


    if n_banks > 1:
        template = {
                'fn': 'nems.modules.fir.filter_bank',
                'fn_kwargs': {'i': 'ctpred', 'o': 'ctpred',
                              'bank_count': n_banks},
                'prior': {
                    'coefficients': ('Normal', p_coefficients)},
                }
    else:
        template = {
                'fn': 'nems.modules.fir.basic',
                'fn_kwargs': {'i': 'ctpred', 'o': 'ctpred'},
                'prior': {
                    'coefficients': ('Normal', p_coefficients)},
                'plot_fns': ['nems.plots.api.mod_output',
                             'nems.plots.api.strf_heatmap',
                             'nems.plots.api.strf_timeseries'],
                'plot_fn_idx': 1,
                }

#    p_coefficients = {'beta': np.full((n_outputs * n_banks, n_coefs), 0.1)}
#    template = {
#            'fn': 'nems.modules.fir.filter_bank',
#            'fn_kwargs': {'i': 'ctpred', 'o': 'ctpred', 'bank_count': n_banks},
#            'prior': {'coefficients': ('Exponential', p_coefficients)},
#            }

    return template


def ctfirexp(kw):
    m = firexp(kw[2:])
    m['fn_kwargs'].update({'i': 'ctpred', 'o': 'ctpred'})
    return m


def gcfir(kw):
    m = fir(kw[2:])
    m['fn_kwargs'].update({'ci': 'ctpred', 'co': 'ctpred'})
    m['fn'] = 'nems_lbhb.gcmodel.modules.fir'
    plot = 'nems_lbhb.gcmodel.guiplots.contrast_kernel_output'
    if m.get('plot_fns'):
        m['plot_fns'].append(plot)
    else:
        m['plot_fns'] = [plot]

    return m


def OOfir(kw):
    kw = 'ct' + kw[2:]
    template = ctfir(kw)
    template['fn_kwargs']['i'] = 'contrast'
    return template


def ctlvl(kw):
    '''
    Same as nems.plugins.keywords.lvl but renamed for
    contrast.
    '''
    m = lvl(kw[2:])
    m['fn_kwargs'].update({'i': 'ctpred', 'o': 'ctpred'})
    return m


def gclvl(kw):
    m = lvl(kw[2:])
    m['fn'] = 'nems_lbhb.gcmodel.modules.levelshift'
    m['fn_kwargs'].update({'ci': 'ctpred', 'co': 'ctpred'})
    if 'noCT' in kw:
        m['fn_kwargs'].update({'block_contrast': True})

    plot = 'nems_lbhb.gcmodel.guiplots.contrast_kernel_output'
    if m.get('plot_fns'):
        m['plot_fns'].append(plot)
    else:
        m['plot_fns'] = [plot]
    m['plot_fn_idx'] = m['plot_fns'].index(plot)

    return m


def ctk(kw):
    ops = kw.split('.')[1:]
    offsets = None
    fixed = False
    for op in ops:
        if op.startswith('off'):
            if 'x' in op:
                channels, initial = op[3:].split('x')
                offset_amount = int(initial)
                n_chans = int(channels)
                offsets = np.full((n_chans, 1), offset_amount)
            else:
                log.warning("integer offset for ctk module should only be used "
                            "with 'fixed' option.")
                offsets = int(op[3:])
        elif op == 'f':
            fixed = True

    template = {
            'fn': 'nems_lbhb.gcmodel.modules.contrast_kernel',
            'fn_kwargs': {'i': 'contrast', 'o': 'ctpred',
                          'wc_coefficients': None, 'fir_coefficients': None,
                          'mean': None, 'sd': None, 'coefficients': None,
                          'use_phi': False, 'compute_contrast': False,
                          'offsets': offsets, 'fixed': fixed},
            'plot_fns': ['nems_lbhb.gcmodel.guiplots.contrast_kernel_heatmap'],
            'phi': {},
            'prior': {},
            }

    return template


def ctk2(kw):
    all_groups = kw.split('.')
    n_channels, n_coefs = [int(s) for s in all_groups[1].split('x')]
    #ops = all_groups[2:]

    tau = np.array([[1]])
    a = np.array([[1]])
    b = np.array([[0]])
    s = np.array([[0]])
    mean = np.array([0.5])
    sd = np.array([0.4])
    sd_one = np.array([[1]])

    prior = {
            'tau': ('Exponential', {'beta': tau}),
            'a': ('Exponential', {'beta': a}),
            'b': ('Normal', {'mean': b, 'sd': sd_one}),
            's': ('Normal', {'mean': s, 'sd': sd_one}),
            'mean': ('Normal', {'mean': mean, 'sd': sd}),
            'sd': ('HalfNormal', {'sd': sd})
            }

    template = {
            'fn': 'nems_lbhb.gcmodel.modules.contrast',
            'fn_kwargs': {'i': 'stim', 'o': 'ctpred', 'n_channels': n_channels,
                          'n_coefs': n_coefs, 'c': 'contrast'},
            'phi': {},
            'prior': prior,
            'plot_fns': ['nems_lbhb.gcmodel.guiplots.contrast_kernel_heatmap2'],
            'bounds': {'tau': (1e-15, None), 'a': (1e-15, None),
                       'sd': (1e-15, None)}
            }

    return template


def ctk3(kw):
    # default 20ms offset, will be converted to fittable parameter
    # during latter portion of fit
    offsets = np.array([[20]])
    template = {
            'fn': 'nems_lbhb.gcmodel.modules.summed_contrast_kernel',
            'fn_kwargs': {'i': 'contrast', 'o': 'ctpred',
                          'compute_contrast': False, 'offsets': offsets,
                          'fixed': False},
            'plot_fns': ['nems_lbhb.gcmodel.guiplots.summed_contrast'],
            'phi': {},
            'prior': {}
            }

    return template


def dsig(kw):
    '''
    Note: these priors will typically be overwritten during initialization
          based on the input signal.
    '''
    ops = kw.split('.')[1:]
    eq = 'logsig'
    amp = False
    base = False
    kappa = False
    shift = False
    c = 'ctpred'
    logsig_bounds = False
    relsat_bounds = False
    norm = False
    alternate = False

    for op in ops:
        if op in ['logsig', 'l']:
            eq = 'logsig'
            logsig_bounds = True
        elif op in ['relsat', 'rs', 'saturated_rectifier']:
            eq = 'relsat'
            relsat_bounds = True
        elif op in ['dexp', 'd']:
            eq = 'dexp'
        elif op == 'a':
            amp = True
        elif op == 'b':
            base = True
        elif op == 'k':
            kappa = True
        elif op == 's':
            shift = True
        elif op.startswith('C'):
            c = op[1:]
        elif op == 'n':
            norm = True
        elif op == 'alt':
            alternate = True

    # Use all by default. Use none not an option (would just be static version)
    if (not amp) and (not base) and (not kappa) and (not shift):
        amp = True; base = True; kappa = True; shift = True

    template = {
        'fn': 'nems_lbhb.gcmodel.modules.dynamic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'c': c,
                      'eq': eq,
                      'norm': norm,
                      'alternate': alternate},
        'plot_fns': ['nems.plots.api.mod_output',
                     'nems.plots.api.pred_resp',
                     'nems.plots.api.before_and_after',
                     'nems.plots.api.nl_scatter'],
        'plot_fn_idx': 1,
        'prior': {'base': ('Exponential', {'beta': [0.1]}),
                  'amplitude': ('Normal', {'mean': [2.0], 'sd': 1.0}),
                  'shift': ('Normal', {'mean': [0.0], 'sd': [1.0]}),
                  'kappa': ('Normal', {'mean': [0.3], 'sd': [1.0]})},
        'bounds': {
                'base': (1e-15, None), 'base_mod': (1e-15, None),
                }
        }

    if logsig_bounds:
        template['bounds'] = {
                'base': (1e-15, None), 'base_mod': (1e-15, None),
                'amplitude': (1e-15, None), 'amplitude_mod': (1e-15, None),
                'shift': (None, None), 'shift_mod': (None, None),
                'kappa': (1e-15, None), 'kappa_mod': (1e-15, None),
                }
    elif relsat_bounds:
        template['bounds'] = {
                'base': (1e-15, None), 'base_mod': (1e-15, None),
                'amplitude': (1e-15, None), 'amplitude_mod': (1e-15, None),
                'shift': (None, None), 'shift_mod': (None, None),
                'kappa': (1e-15, None), 'kappa_mod': (1e-15, None)
                }

    zero_norm = ('Normal', {'mean': [0.0], 'sd': [1.0]})

    if amp:
        if alternate:
            template['prior']['amplitude_mod'] = copy.deepcopy(zero_norm)
        else:
            template['prior']['amplitude_mod'] = copy.deepcopy(
                    template['prior']['amplitude']
                    )

    if base:
        if alternate:
            template['prior']['base_mod'] = copy.deepcopy(zero_norm)
        else:
            template['prior']['base_mod'] = copy.deepcopy(
                    template['prior']['base']
                    )

    if kappa:
        if alternate:
            template['prior']['kappa_mod'] = copy.deepcopy(zero_norm)
        else:
            template['prior']['kappa_mod'] = copy.deepcopy(
                    template['prior']['kappa']
                    )

    if shift:
        if alternate:
            template['prior']['shift_mod'] = copy.deepcopy(zero_norm)
        else:
            template['prior']['shift_mod'] = copy.deepcopy(
                    template['prior']['shift']
                    )

    return template


def _one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def slogsig(kw):
    '''
    Generate and register modelspec for linear state gain model with rectification
    CRH 12/6/2019
    '''
    options = kw.split('.')
    pattern = re.compile(r'^slogsig\.?(\d{1,})x(\d{1,})$')
    parsed = re.match(pattern, '.'.join(options[0:2]))
    try:
        n_vars = int(parsed.group(1))
        if len(parsed.groups())>1:
            n_chans = int(parsed.group(2))
        else:
            n_chans = 1
    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "sdexp.{n_state_variables} \n"
                         "keyword given: %s" % kw)
    
    if 'd' in options[2:]:
        zeros = np.zeros([n_chans, n_vars])
        ones = 0.01 * np.ones([n_chans, n_vars])
        baseline_u = np.zeros([n_chans, 1])
        baseline_sd = np.ones([n_chans, 1])
        amplitude = 2 * np.ones([n_chans, 2])
        amp_sd = 0.01 * np.ones([n_chans, 2])
        template = {
        'fn': 'nems_lbhb.modules.state.state_logsig_dcgain',
        'fn_kwargs': {'i': 'pred',
                    'o': 'pred',
                    's': 'state'},
        'prior': {'g': ('Normal', {'mean': zeros, 'sd': ones}),
                'd': ('Normal', {'mean': zeros, 'sd': ones}),
                'b': ('Normal', {'mean': baseline_u, 'sd': baseline_sd}),
                'a': ('Normal', {'mean': amplitude, 'sd': amp_sd})},
        'plot_fns': ['nems_lbhb.plots.state_logsig_plot'],
            'plot_fn_idx': 0,
        'bounds': {'g': (None, None)}
        }
    else:
        zeros = np.zeros([n_chans, n_vars])
        ones = 0.01 * np.ones([n_chans, n_vars])
        baseline_u = np.zeros([n_chans, 1])
        baseline_sd = np.ones([n_chans, 1])
        amplitude = 2 * np.ones([n_chans, 1])
        amp_sd = 0.01 * np.ones([n_chans, 1])
        template = {
        'fn': 'nems_lbhb.modules.state.state_logsig',
        'fn_kwargs': {'i': 'pred',
                    'o': 'pred',
                    's': 'state'},
        'prior': {'g': ('Normal', {'mean': zeros, 'sd': ones}),
                'b': ('Normal', {'mean': baseline_u, 'sd': baseline_sd}),
                'a': ('Normal', {'mean': amplitude, 'sd': amp_sd})},
        'plot_fns': ['nems_lbhb.plots.state_logsig_plot'],
            'plot_fn_idx': 0,
        'bounds': {'g': (None, None)}
        }

    return template


def sexp(kw):
    '''
    Generate and register modulespec for the state_exp
     - gain only state model based on Rabinowitz  2015
    CRH 12/3/2019
    '''
    pattern = re.compile(r'^sexp\.?(\d{1,})x(\d{1,})$')
    parsed = re.match(pattern, kw)
    try:
        n_vars = int(parsed.group(1))
        if len(parsed.groups())>1:
            n_chans = int(parsed.group(2))
        else:
            n_chans = 1
    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "sdexp.{n_state_variables} \n"
                         "keyword given: %s" % kw)

    zeros = np.zeros([n_chans, n_vars])
    ones = 0.01 * np.ones([n_chans, n_vars])

    template = {
    'fn': 'nems_lbhb.modules.state.state_exp',
    'fn_kwargs': {'i': 'pred',
                  'o': 'pred',
                  's': 'state'},
    'prior': {'g': ('Normal', {'mean': zeros, 'sd': ones})},
    'bounds': {'g': (None, 10)}
    }

    return template


def lvexp(kw):
    '''
    Generate and register modelspec for fitting gain parms
        for the latent variable signal. (need to create LV first)

    CRH 12/4/2019
    '''

    parsed = kw.split('.')[1]

    n_vars = int(parsed.split('x')[0])
    if len(parsed.split('x'))>1:
        n_chans = int(parsed.split('x')[1])
    else:
        n_chans = 1

    zeros = np.zeros([n_chans, n_vars])
    ones = 0.01 * np.ones([n_chans, n_vars])

    template = {
    'fn': 'nems_lbhb.modules.state.state_exp',
    'fn_kwargs': {'i': 'pred',
                  'o': 'pred',
                  's': 'lv'},
    'prior': {'g': ('Normal', {'mean': zeros, 'sd': ones})},
    'bounds': {'g': (None, 10)}
    }

    return template


def lvlogsig(kw):
    '''
    Generate and register modelspec for linear state gain model with rectification
        for latent variable
    CRH 12/6/2019
    '''
    parsed = kw.split('.')[1]

    n_vars = int(parsed.split('x')[0])
    if len(parsed.split('x'))>1:
        n_chans = int(parsed.split('x')[1])
    else:
        n_chans = 1

    options = kw.split('.')
    if 'd' in options[2:]:
        zeros = np.zeros([n_chans, n_vars])
        ones = 0.01 * np.ones([n_chans, n_vars])
        baseline_u = np.zeros([n_chans, 1])
        baseline_sd = np.ones([n_chans, 1])
        amplitude = 2 * np.ones([n_chans, 2])
        amp_sd = 0.01 * np.ones([n_chans, 2])
        template = {
        'fn': 'nems_lbhb.modules.state.state_logsig_dcgain',
        'fn_kwargs': {'i': 'pred',
                    'o': 'pred',
                    's': 'lv'},
        'prior': {'g': ('Normal', {'mean': zeros, 'sd': ones}),
                'd': ('Normal', {'mean': zeros, 'sd': ones}),
                'b': ('Normal', {'mean': baseline_u, 'sd': baseline_sd}),
                'a': ('Normal', {'mean': amplitude, 'sd': amp_sd})},
        'plot_fns': ['nems_lbhb.plots.lv_logsig_plot'],
            'plot_fn_idx': 0,
        'bounds': {'g': (None, None)}
        }
    else:
        zeros = np.zeros([n_chans, n_vars])
        ones = 0.01 * np.ones([n_chans, n_vars])
        baseline_u = np.zeros([n_chans, 1])
        baseline_sd = np.ones([n_chans, 1])
        amplitude = 2 * np.ones([n_chans, 1])
        amp_sd = 0.01 * np.ones([n_chans, 1])
        template = {
        'fn': 'nems_lbhb.modules.state.state_logsig',
        'fn_kwargs': {'i': 'pred',
                    'o': 'pred',
                    's': 'lv'},
        'prior': {'g': ('Normal', {'mean': zeros, 'sd': ones}),
                'b': ('Normal', {'mean': baseline_u, 'sd': baseline_sd}),
                'a': ('Normal', {'mean': amplitude, 'sd': amp_sd})},
        'plot_fns': ['nems_lbhb.plots.lv_logsig_plot'],
            'plot_fn_idx': 0,
        'bounds': {'g': (None, None)}
        }

    return template


def lv(kw):
    '''
    Generate and register modelspec for create_lv
        1) Find the encoding (projection) weights for a lv model
        2) Add this lv to the list of rec signals

    CRH 12/4/2019
    '''
    parsed = kw.split('.')[1]

    n_vars = int(parsed.split('x')[0])
    if len(parsed.split('x'))>1:
        n_chans = int(parsed.split('x')[1])
    else:
        n_chans = 1

    shuffle = False
    cutoff = None
    ops = kw.split('.')[2:]
    for o in ops:
        if o == 'shuf':
            shuffle = True
        if 'hp' in o:
            cutoff = np.float(o[2:].replace(',', '.'))

    mean = 0.01 * np.ones([n_chans, n_vars])
    sd = 0.01 * np.ones([n_chans, n_vars])

    template = {
    'fn': 'nems_lbhb.modules.state.add_lv',
    'fn_kwargs': {'i': 'pred',
                  'o': 'pred',
                  'shuffle': shuffle,
                  'cutoff': cutoff},
    'plot_fns': ['nems_lbhb.plots.lv_quickplot'],
        'plot_fn_idx': 0,
    'prior': {'e': ('Normal', {'mean': mean, 'sd': sd})},
    'bounds': {'e': (None, None)}
    }

    return template


def sdexp(kw):
    '''
    Generate and register modulespec for the state_dexp

    Parameters
    ----------
    kw : str
        Expected format: r'^sdexp\.?(\d{1,})x(\d{1,})$'
        e.g., "sdexp.SxR" or "sdexp.S":
            S : number of state channels (required)
            R : number of channels to modulate (default = 1)
            currently not supported. R=1
        TODO add support for R>1, copy from stategain
    Options
    -------
    None
    '''
    options = kw.split('.')
    #pattern = re.compile(r'^sdexp\.?(\d{1,})x(\d{1,})$')
    pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
    parsed = re.match(pattern, options[1])
    if parsed is None:
        # backward compatible parsing if R not specified
        pattern = re.compile(r'^sdexp\.?(\d{1,})$')
        parsed = re.match(pattern, options[1])
    try:
        n_vars = int(parsed.group(1))
        if len(parsed.groups())>1:
            n_chans = int(parsed.group(2))
        else:
            n_chans = 1

    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "sdexp.{n_state_variables} \n"
                         "keyword given: %s" % kw)
    if n_chans > 1:
        pass
        # why was this here?
        #raise ValueError("sdexp R>1 not supported")

    state = 'state'
    for o in options[2:]:
        if o == 'lv':
            state = 'lv'

    zeros = np.zeros([n_chans, n_vars])
    ones = np.ones([n_chans, n_vars])
    g_mean = zeros.copy()
    g_mean[:, 0] = 1
    g_sd = ones.copy()
    d_mean = zeros
    d_sd = ones

    n_dims = 2 # one for gain, one for dc
    base_mean = np.zeros([n_dims, n_chans])
    base_sd = np.ones([n_dims, n_chans])
    amp_mean = base_mean + 0.2
    amp_sd = base_mean + 0.1
    #shift_mean = base_mean
    #shift_sd = base_sd
    kappa_mean = base_mean
    kappa_sd = amp_sd


    template = {
        'fn': 'nems_lbhb.modules.state.state_dexp',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      's': state},
        'plot_fns': ['nems.plots.api.mod_output',
                     'nems.plots.api.before_and_after',
                     'nems.plots.api.pred_resp',
                     'nems.plots.api.state_vars_timeseries',
                     'nems.plots.api.state_vars_psth_all'],
        'plot_fn_idx': 3,
        'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                  'd': ('Normal', {'mean': d_mean, 'sd': d_sd}),
                  'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                  'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                  'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
        }

    return template


def stategainchan(kw):
    """
    Same as nems default keyword stategain, but allows you to only modulate
    a select channels of the input

    stategainchan.SxN.0,1 would do stategain operation with channels 0 and 1 of
        the input signal
    """
    options = kw.split('.')
    in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')

    try:
        parsed = re.match(in_out_pattern, options[1])
        if parsed is None:
            # backward compatible parsing if R not specified
            n_vars = int(options[1])
            n_chans = 1

        else:
            n_vars = int(parsed.group(1))
            if len(parsed.groups())>1:
                n_chans = int(parsed.group(2))
            else:
                n_chans = 1
    except TypeError:
        raise ValueError("Got TypeError when parsing stategain keyword.\n"
                         "Make sure keyword is of the form: \n"
                         "stategain.{n_variables} or stategain.{n_variables}x{n_chans} \n"
                         "keyword given: %s" % kw)

    n_mod_chans = [int(c) for c in options[2].split(',')]
    if len(n_mod_chans) == 0:
        n_mod_chans = None
    else:
        n_chans = n_chans - len(n_mod_chans)

    zeros = np.zeros([n_chans, n_vars])
    ones = np.ones([n_chans, n_vars])
    g_mean = zeros.copy()
    g_mean[:, 0] = 1
    g_sd = ones.copy()
    d_mean = zeros
    d_sd = ones

    plot_fns = ['nems.plots.api.mod_output_all',
                'nems.plots.api.mod_output',
                'nems.plots.api.before_and_after',
                'nems.plots.api.pred_resp',
                'nems.plots.api.state_vars_timeseries',
                'nems.plots.api.state_vars_psth_all']
    if 'g' in options:
        template = {
            'fn': 'nems.modules.state.state_gain',
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          's': 'state',
                          'c': n_mod_chans},
            'plot_fns': plot_fns,
            'plot_fn_idx': 4,
            'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd})}
            }
    else:
        template = {
            'fn': 'nems.modules.state.state_dc_gain',
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          's': 'state',
                          'c': n_mod_chans},
            'plot_fns': plot_fns,
            'plot_fn_idx': 4,
            'prior': {'g': ('Normal', {'mean': g_mean, 'sd': g_sd}),
                      'd': ('Normal', {'mean': d_mean, 'sd': d_sd})}
            }

    return template


def _aliased_keyword(fn, kw):
    '''Forces the keyword fn to use the given kw. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <kw_head>.<option1>.<option2> paradigm.
    '''
    def ignorant_keyword(ignored_key):
        return fn(kw)
    return ignorant_keyword


# Old keywords that are identical except for the period
# (e.x. dexp1 vs dexp.1 or wc15x2 vs wc.15x2)
# don't need to be aliased, but more complicated ones that had options
# picked apart (like wc.NxN.n.g.c) will need to be aliased.


# These aren't actually  needed anymore since we separated old models
# from new ones, but gives an example of how aliasing can be done.

#wc_combinations = {}
#wcc_combinations = {}
#
#for n_in in (15, 18, 40):
#    for n_out in (1, 2, 3, 4):
#        for op in ('', 'g', 'g.n'):
#            old_k = 'wc%s%dx%d' % (op.strip('.'), n_in, n_out)
#            new_k = 'wc.%dx%d.%s' % (n_in, n_out, op)
#            wc_combinations[old_k] = _aliased_keyword(wc, new_k)
#
#for n_in in (1, 2, 3):
#    for n_out in (1, 2, 3, 4):
#        for op in ('c', 'n'):
#            old_k = 'wc%s%dx%d' % (op, n_in, n_out)
#            new_k = 'wc.%dx%d.%s' % (n_in, n_out, op)
#
#stp2b = _aliased_keyword(stp, 'stp.2.b')
#stpz2 = _aliased_keyword(stp, 'stp.2.z')
#stpn1 = _aliased_keyword(stp, 'stp.1.n')
#stpn2 = _aliased_keyword(stp, 'stp.2.n')
#dlogz = _aliased_keyword(stp, 'dlog')
#dlogf = _aliased_keyword(dlog, 'dlog.f')
