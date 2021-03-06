import re
import numpy as np

from nems.plugins.default_fitters import basic, iter, _parse_basic, _parse_iter, _extract_options
from nems.utils import escaped_split
from nems.registry import xform, xmodule


@xform()
def lnp(fitkey):
    ops = fitkey.split('.')[1:]
    kwargs = {}
    for op in ops:
        if op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['tolerance'] = 10**tolpower
        elif op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['max_iter'] = int(re.match(pattern, op).group(1))

    return [['nems_lbhb.lnp_helpers.lnp_basic', kwargs]]


@xform()
def gc(fitkey):
    ops = fitkey.split('.')[1:]
    kwargs = {}
    for op in ops:
        if op.startswith('t'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['tolerance'] = 10**tolpower
        elif op.startswith('pt'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['prefit_tolerance'] = 10**tolpower
        elif op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('pmi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['prefit_max_iter'] = int(re.match(pattern, op).group(1))
        elif op == 'fx':
            kwargs['fixed_strf'] = True
        elif op.startswith('nl'):
            kwargs['nl_mode'] = int(op[2:])

    return [['nems_lbhb.gcmodel.fitters.fit_gc', kwargs]]


@xform()
def gc2(fitkey):
    ops = fitkey.split('.')[1:]
    kwargs = {}
    xfspec = []
    for op in ops:
        if op.startswith('t'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['tolerance'] = 10**tolpower
        elif op.startswith('pt'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[2:])*(-1)
            kwargs['prefit_tolerance'] = 10**tolpower
        elif op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('pmi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['prefit_max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('nl'):
            kwargs['nl_mode'] = int(op[2:])
        elif op == 'PF':
            kwargs['post_fit'] = True
        elif op == 'NC':
            kwargs['post_copy'] = False
        elif op.startswith('rgc'):
            nr = int(op[3:])
            kwargs['n_random'] = nr
        elif op.startswith('rp'):
            rc = int(op[1:])
            xfspec.append(['nems.initializers.rand_phi', {'rand_count': rc}])
        elif op == 'SC':
            kwargs['summed_contrast'] = True
            kwargs['post_copy'] = False
            kwargs['post_fit'] = True

    xfspec.append(['nems_lbhb.gcmodel.fitters.fit_gc2', kwargs])
    return xfspec


@xform()
def gc3(fitkey):
    ops = fitkey.split('.')[1:]
    kwargs = {}
    xfspec = []
    for op in ops:
        if op.startswith('t'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['tolerance'] = 10**tolpower
        elif op.startswith('pt'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['prefit_tolerance'] = 10**tolpower
        elif op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('pmi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['prefit_max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('rgc'):
            nr = int(op[3:])
            kwargs['n_random'] = nr
        elif op.startswith('rp'):
            rc = int(op[1:])
            xfspec.append(['nems.initializers.rand_phi', {'rand_count': rc}])

    xfspec.append(['nems_lbhb.gcmodel.fitters.fit_gc3', kwargs])
    return xfspec


@xform()
def gc4(fitkey):
    ops = fitkey.split('.')[1:]
    kwargs = {}
    xfspec = []
    for op in ops:
        if op.startswith('t'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['tolerance'] = 10**tolpower
        elif op.startswith('pt'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[2:])*(-1)
            kwargs['prefit_tolerance'] = 10**tolpower
        elif op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('pmi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['prefit_max_iter'] = int(re.match(pattern, op).group(1))

    xfspec.append(['nems_lbhb.gcmodel.fitters.fit_gc4', kwargs])
    return xfspec


@xform()
def testLN(fitkey):
    ops = fitkey.split('.')[1:]
    kwargs = {}
    xfspec = []
    for op in ops:
        if op.startswith('t'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['tolerance'] = 10**tolpower
        elif op.startswith('pt'):
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            kwargs['prefit_tolerance'] = 10**tolpower
        elif op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('pmi'):
            pattern = re.compile(r'^mi(\d{1,})')
            kwargs['prefit_max_iter'] = int(re.match(pattern, op).group(1))
        elif op.startswith('r'):
            rc = int(op[1:])
            xfspec.append(['nems.initializers.rand_phi', {'rand_count': rc}])
    xfspec.append(['nems_lbhb.gcmodel.fitters.test_LN', kwargs])
    return xfspec


@xform()
def strfc(fitkey):
    return [['nems_lbhb.contrast_helpers.strf_to_contrast', {}]]


def _aliased_fitter(fn, fitkey):
    '''Forces the keyword fn to use the given fitkey. Used for implementing
    backwards compatibility with old keywords that did not follow the
    <all-alpha kw_head><numbers> paradigm.
    '''
    def ignorant_fitter(ignored_key):
        return fn(fitkey)
    ignorant_fitter.key = fitkey
    return ignorant_fitter


# NOTE: Using the new keyword syntax is encouraged since it improves
#       readability; however, for exceptionally long keywords or ones
#       that get used very frequently, aliases can be implemented as below.
#       If your alias is causing errors, ask Jacob for help.


fitjk01 = _aliased_fitter(basic, 'basic.nf5.epREFERENCE')


@xform()
def popiter(fitkey):
    '''
    Perform a fit_iteratively analysis on a model.

    Parameters
    ----------
    fitkey : str
        Expected format: iter.<fitter>.<misc>
        Example: iter.T3,4,7.S0,1.S2,3.S0,1,2,3.ti50.fi20
        Example translation:
            Use fit_iteratively with scipy_minimize
            (since 'cd' option is not present), 50 per-tolerance-level
            iterations, and 20 per-fit iterations.
            Begin with a tolerance level of 10**-3, followed by
            10**-4 and 10**-7. Within each tolerance level,
            first fit modules 0 and 1, then 2 and 3,
            and finally 0, 1, 2, and 3 all together.

    Options
    -------
    cd : Use coordinate_descent for fitting (default is scipy_minimize)
    TN,N,... : Use tolerance levels 10**-N for each N given, where N is
               any positive integer.
    SN,N,... : Fit model indices N, N... for each N given,
               where N is any positive integer or zero. May be provided
               multiple times to iterate over several successive subsets.
    tiN : Perform N per-tolerance-level iterations, where N is any
          positive integer.
    fiN : Perform N per-fit iterations, where N is any positive integer.

    '''

    # TODO: Support nfold and state fits for fit_iteratively?
    #       And epoch to go with state.
    options = _extract_options(fitkey)
    tolerances, module_sets, fit_iter, tol_iter, fitter = _parse_iter(options)
    if 'pcf' in options:
        flip_pcs = True
    else:
        flip_pcs = False
    xfspec = [['nems.analysis.fit_pop_model.init_pop_pca', {'flip_pcs': flip_pcs}],
              ['nems.analysis.fit_pop_model.fit_population_iteratively',
               {'module_sets': module_sets, 'fitter': fitter,
                'tolerances': tolerances, 'tol_iter': tol_iter,
                'fit_iter': fit_iter}]]

    #            ['est', 'modelspecs', 'IsReload'],
    #            ['modelspecs']]]

    return xfspec


@xform()
def pupLVbasic(fitkey):
    """
    exact same as fit basic, but add constraint that the latent variable (LV)
    must change variance between big/small pupil

    CRH 12/4/2019
    """
    xfspec = []

    options = _extract_options(fitkey)
    max_iter, tolerance, fitter, choose_best, fast_eval, alpha, rand_count, pup_constraint, signed_correlation = _parse_pupLVbasic(options)
    xfspec = []

    if rand_count>1:
        xfspec.append(['nems.initializers.rand_phi', {'rand_count': rand_count}])

    metric = 'pup_nmse'
    if pup_constraint == 'lv_var':
        metric = 'pup_nmse'
        metric_fn = 'nems_lbhb.lv_helpers.pup_nmse'
    elif pup_constraint == 'nc':
        metric = 'pup_nc_nmse'
        metric_fn = 'nems_lbhb.lv_helpers.pup_nc_nmse'
    elif pup_constraint == 'lv_only':
        metric = 'pup_dep_LVs'
        metric_fn = 'nems_lbhb.lv_helpers.pup_dep_LVs'

    xfspec.append(['nems.xforms.fit_wrapper',
                  {'max_iter': max_iter,
                   'fitter': fitter, 'tolerance': tolerance,
                   'metric': metric,
                   'fit_function': 'nems_lbhb.lv_helpers.fit_pupil_lv',
                   'alpha': alpha,
                   'signed_correlation': signed_correlation}])

    if choose_best:
        xfspec.append(['nems.analysis.test_prediction.pick_best_phi', {'metric_fn': metric_fn,
                                                                       'alpha': alpha,
                                                                       'signed_correlation': signed_correlation
                                                                       }])

    return xfspec

def _parse_pupLVbasic(options):
    '''Options specific to basic.'''
    max_iter = 1000
    tolerance = 1e-7
    fitter = 'scipy_minimize'
    choose_best = False
    fast_eval = False
    pup_constraint = 'lv_var'
    alpha = 0
    fast_alpha = None
    slow_alpha = None
    rand_count = 1
    signed_correlation = False
    for op in options:
        if op.startswith('mi'):
            pattern = re.compile(r'^mi(\d{1,})')
            max_iter = int(re.match(pattern, op).group(1))
        elif op.startswith('t'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            num = op.replace('d', '.').replace('\\', '')
            tolpower = float(num[1:])*(-1)
            tolerance = 10**tolpower
        elif op == 'cd':
            fitter = 'coordinate_descent'
        elif op == 'b':
            choose_best = True
        elif op == 'f':
            fast_eval = True
        elif op.startswith('rb'):
            if len(op) == 2:
                rand_count = 10
            else:
                rand_count = int(op[2:])
            choose_best = True

        elif op.startswith('af'):
            fast_alpha = np.float('.'.join(op[2:].split(':')))

        elif op.startswith('as'):
            slow_alpha = np.float('.'.join(op[2:].split(':')))

        elif op.startswith('a'):
            alpha = np.float('.'.join(op[1:].split(':')))

        elif op.startswith('sc'):
            # force fast LV variance to be (-) corr. with pupil
            signed_correlation = True

        elif op.startswith('constr'):
            pc = op[6:]
            if pc == 'LV':
                pup_constraint = 'lv_var'
            elif pc == 'NC':
                pup_constraint = 'nc'
            elif pc == 'LVonly':
                pup_constraint = 'lv_only'

    if (fast_alpha is not None) | (slow_alpha is not None):
        if (fast_alpha is None):
            alpha = slow_alpha
        elif (slow_alpha is None):
            alpha = fast_alpha
        else:
            alpha = {'fast_alpha': fast_alpha,
                     'slow_alpha': slow_alpha}

    return max_iter, tolerance, fitter, choose_best, fast_eval, alpha, rand_count, pup_constraint, signed_correlation


