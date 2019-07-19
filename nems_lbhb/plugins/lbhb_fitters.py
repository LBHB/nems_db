import re

from nems.plugins.default_fitters import basic
from nems.plugins.default_fitters import iter
from nems.utils import escaped_split


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
            tolpower = float(num[1:])*(-1)
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
        elif op.startswith('r'):
            rc = int(op[1:])
            xfspec.append(['nems.initializers.rand_phi', {'rand_count': rc}])

    xfspec.append(['nems_lbhb.gcmodel.fitters.fit_gc2', kwargs])
    return xfspec


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
    xfspec = [['nems_lbhb.fit_wrappers.init_pop_pca', {'flip_pcs': flip_pcs}],
              ['nems_lbhb.fit_wrappers.fit_population_iteratively',
               {'module_sets': module_sets, 'fitter': fitter,
                'tolerances': tolerances, 'tol_iter': tol_iter,
                'fit_iter': fit_iter}]]

    #            ['est', 'modelspecs', 'IsReload'],
    #            ['modelspecs']]]

    return xfspec


def _extract_options(fitkey):
    if fitkey == 'basic' or fitkey == 'iter':
        # empty options (i.e. just use defualts)
        options = []
    else:
        chunks = escaped_split(fitkey, '.')
        options = chunks[1:]
    return options


def _parse_basic(options):
    '''Options specific to basic.'''
    max_iter = 1000
    tolerance = 1e-7
    fitter = 'scipy_minimize'
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

    return max_iter, tolerance, fitter


def _parse_iter(options):
    '''Options specific to iter.'''
    tolerances = []
    module_sets = []
    fit_iter = 10
    tol_iter = 50
    fitter = 'scipy_minimize'

    for op in options:
        if op.startswith('ti'):
            tol_iter = int(op[2:])
        elif op.startswith('fi'):
            fit_iter = int(op[2:])
        elif op.startswith('T'):
            # Should use \ to escape going forward, but keep d-sub in
            # for backwards compatibility.
            nums = op.replace('d', '.').replace('\\', '')
            powers = [float(i) for i in nums[1:].split(',')]
            tolerances.extend([10**(-1*p) for p in powers])
        elif op.startswith('S'):
            indices = [int(i) for i in op[1:].split(',')]
            module_sets.append(indices)
        elif op == 'cd':
            fitter = 'coordinate_descent'

    if not tolerances:
        tolerances = None
    if not module_sets:
        module_sets = None

    return tolerances, module_sets, fit_iter, tol_iter, fitter
