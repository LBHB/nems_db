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
    template = {
            'fn': 'nems_lbhb.gcmodel.modules.contrast_kernel',
            'fn_kwargs': {'i': 'contrast', 'o': 'ctpred',
                          'wc_coefficients': None, 'fir_coefficients': None,
                          'mean': None, 'sd': None, 'coefficients': None,
                          'auto_copy': False, 'compute_contrast': False},
            'phi': {},
            'prior': {},
            'bounds': {}
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
    bounded = False
    norm = False

    for op in ops:
        if op in ['logsig', 'l']:
            eq = 'logsig'
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
        elif op == 'bnd':
            bounded = True
        elif op == 'n':
            norm = True

    # Use all by default. Use none not an option (would just be static version)
    if (not amp) and (not base) and (not kappa) and (not shift):
        amp = True; base = True; kappa = True; shift = True

    template = {
        'fn': 'nems_lbhb.gcmodel.modules.dynamic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred',
                      'c': c,
                      'eq': eq,
                      'norm': norm},
        'plot_fns': ['nems.plots.api.mod_output',
                     'nems.plots.api.pred_resp',
                     'nems.plots.api.before_and_after',
                     'nems.plots.api.nl_scatter'],
        'plot_fn_idx': 2,
        'prior': {'base': ('Exponential', {'beta': [0.1]}),
                  'amplitude': ('Exponential', {'beta': [2.0]}),
                  'shift': ('Normal', {'mean': [1.0], 'sd': [1.0]}),
                  'kappa': ('Exponential', {'beta': [0.1]})},
        }

    if bounded:
        template['bounds'] = {
                'base': (1e-15, None), 'base_mod': (1e-15, None),
                'amplitude': (1e-15, None), 'amplitude_mod': (1e-15, None),
                'shift': (None, None), 'shift_mod': (None, None),
                'kappa': (1e-15, None), 'kappa_mod': (1e-15, None),
                }

    #zero_norm = ('Normal', {'mean': [0.0], 'sd': [1.0]})

    if amp:
        template['prior']['amplitude_mod'] = copy.deepcopy(
                template['prior']['amplitude']
                )

    if base:
        template['prior']['base_mod'] = copy.deepcopy(
                template['prior']['base']
                )

    if kappa:
        template['prior']['kappa_mod'] = copy.deepcopy(
                template['prior']['kappa']
                )

    if shift:
        template['prior']['shift_mod'] = copy.deepcopy(
                template['prior']['shift']
                )

    return template


def _one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


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
    pattern = re.compile(r'^sdexp\.?(\d{1,})x(\d{1,})$')
    parsed = re.match(pattern, kw)
    if parsed is None:
        # backward compatible parsing if R not specified
        pattern = re.compile(r'^sdexp\.?(\d{1,})$')
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
    if n_chans > 1:
        raise ValueError("sdexp R>1 not supported")
    zeros = np.zeros(n_vars)
    ones = np.ones(n_vars)
    g_mean = _one_zz(n_vars-1)
    g_sd = ones
    d_mean = zeros
    d_sd = ones
    n_dims = 2 # one for gain, one for dc
    base_mean = np.zeros([n_dims, 1]) if n_dims > 1 else np.array([0])
    base_sd = np.ones([n_dims, 1]) if n_dims > 1 else np.array([1])
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
                      's': 'state'},
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
