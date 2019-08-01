import logging
import copy

import numpy as np

import nems.epoch
import nems.modelspec as ms
from nems.utils import find_module
from nems.initializers import (prefit_to_target, prefit_mod_subset, init_dexp,
                               init_logsig, init_relsat)
from nems.analysis.api import fit_basic
import nems.fitters.api
import nems.metrics.api as metrics
from nems import priors

log = logging.getLogger(__name__)


def init_contrast_model(est, modelspec, IsReload=False,
                        tolerance=10**-5.5, max_iter=700, copy_strf=False,
                        fitter='scipy_minimize', metric='nmse', **context):
    '''
    Sets initial values for weight_channels, fir, levelshift, and their
    contrast-dependent counterparts, as well as dynamic_sigmoid. Also
    performs a rough fit for each of these modules.

    Parameters
    ----------
    est : NEMS recording
        The recording to use for prefitting and for determining initial values.
        Expects the estimation portion of the dataset by default.
    modelspec : ModelSpec
        NEMS ModelSpec object.
    IsReload : boolean
        For use with xforms, specifies whether the model is being fit for
        the first time or if it is being loaded from a previous fit.
        If true, this function does nothing.
    tolerance : float
        Tolerance value to be passed to the optimizer.
    max_iter : int
        Maximum iteration count to be passed to the optimizer.
    copy_strf : boolean
        If true, use the pre-fitted phi values from weight_channels,
        fir, and levelshift as the initial values for their contrast-based
        counterparts.
    fitter : str
        Name of the optimization function to use, e.g. scipy_minimize
        or coordinate_descent. It will be imported from nems.fitters.api
    metric : str
        Name of the metric to optimize, e.g. 'nmse'. It will be imported
        from nems.metrics.api
    context : dictionary
        For use with xforms, contents will be updated by the return value.

    Returns
    -------
    {'modelspec': modelspec}

    '''

    if IsReload:
        return {}

    modelspec = copy.deepcopy(modelspec)

    # If there's no dynamic_sigmoid module, try doing
    # the normal linear-nonlinear initialization instead.
    if not find_module('dynamic_sigmoid', modelspec):
        new_ms = nems.initializers.prefit_LN(est, modelspec, max_iter=max_iter,
                                             tolerance=tolerance)
        return {'modelspec': new_ms}

    # Set up kwargs for prefit function.
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}
    fitter_fn = getattr(nems.fitters.api, fitter)
    if metric is not None:
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')
    else:
        metric_fn = None

    # fit without STP module first (if there is one)
    modelspec = prefit_to_target(est, modelspec, fit_basic,
                                 target_module='levelshift',
                                 extra_exclude=['stp'],
                                 fitter=fitter_fn,
                                 metric=metric_fn,
                                 fit_kwargs=fit_kwargs)

    # then initialize the STP module (if there is one)
    for i, m in enumerate(modelspec.modules):
        if 'stp' in m['fn']:
            m = priors.set_mean_phi([m])[0]  # Init phi for module
            modelspec[i] = m
            break


    log.info("initializing priors and bounds for dsig ...\n")
    modelspec = init_dsig(est, modelspec)

    # prefit only the static nonlinearity parameters first
    modelspec = _prefit_dsig_only(
                    est, modelspec, fit_basic,
                    fitter=fitter_fn,
                    metric=metric_fn,
                    fit_kwargs=fit_kwargs
                    )

    # Now prefit all of the contrast modules together.
    # Before this step, result of initialization should be identical
    # to prefit_LN
    if copy_strf:
        # Will only behave as expected if dimensions of strf
        # and contrast strf match!
        modelspec = _strf_to_contrast(modelspec)
    modelspec = _prefit_contrast_modules(
                    est, modelspec, fit_basic,
                    fitter=fitter_fn,
                    metric=metric_fn,
                    fit_kwargs=fit_kwargs
                    )

    # after prefitting contrast modules, update priors to reflect the
    # prefit values so that random sample fits incorporate the prefit info.
    modelspec = dsig_phi_to_prior(modelspec)

    return {'modelspec': modelspec}


def init_dsig(rec, modelspec, nl_mode=2):
    '''
    Initialization of priors for logistic_sigmoid,
    based on process described in methods of Rabinowitz et al. 2014.
    '''

    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        log.warning("No dsig module was found, can't initialize.")
        return modelspec

    if modelspec[dsig_idx]['fn_kwargs'].get('eq', '') in \
            ['dexp', 'd', 'double_exponential']:
        modelspec = _init_double_exponential(rec, modelspec, dsig_idx,
                                             nl_mode=nl_mode)
    elif modelspec[dsig_idx]['fn_kwargs'].get('eq', '') in \
            ['relsat', 'rs', 'saturated_rectifier']:
        modelspec = init_relsat(rec, modelspec)
    else:
        modelspec = init_logsig(rec, modelspec)

    return modelspec


def _init_double_exponential(rec, modelspec, target_i, nl_mode=2):
    # Call existing init_dexp
    modelspec = init_dexp(rec, modelspec, nl_mode=2, override_target_i=target_i)

    # Package initiailzation results into priors so that we can use those
    # to sample for some reasonable random fits if desired.
    amp = modelspec.phi[-1]['amplitude']
    base = modelspec.phi[-1]['base']
    kappa = modelspec.phi[-1]['kappa']
    shift = modelspec.phi[-1]['shift']

    amp_prior = ('Normal', {'mean': amp, 'sd': amp*2})
    base_prior = ('Exponential', {'beta': base})
    kappa_prior = ('Normal', {'mean': kappa, 'sd': kappa*2})
    shift_prior = ('Normal', {'mean': shift, 'sd': shift*2})

    modelspec[target_i]['prior'].update({
            'amplitude': amp_prior, 'base': base_prior, 'kappa': kappa_prior,
            'shift': shift_prior,
            })

    return modelspec


def dsig_phi_to_prior(modelspec):
    '''
    Sets priors for dynamic_sigmoid equal to the current phi for the
    same module. Used for random-sample fits - all samples are initialized
    and pre-fit the same way, and then randomly sampled from the new priors.

    Operates on modelspec IN-PLACE.

    Parameters
    ----------
    modelspec : list of dictionaries
        A NEMS modelspec containing, at minimum, a dynamic_sigmoid module

    Returns
    -------
    modelspec : A copy of the input modelspec with priors updated.

    '''

    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    phi = modelspec[dsig_idx]['phi']
    b = phi['base']
    a = phi['amplitude']
    k = phi['kappa']
    s = phi['shift']
    b_m = 'base_mod' in phi
    a_m = 'amplitude_mod' in phi
    k_m = 'kappa_mod' in phi
    s_m = 'shift_mod' in phi

    amp_prior = ('Normal', {'mean': a, 'sd': np.abs(a*2)})
    base_prior = ('Exponential', {'beta': b})
    kappa_prior = ('Normal', {'mean': k, 'sd': np.abs(k*2)})
    shift_prior = ('Normal', {'mean': s, 'sd': np.abs(s*2)})

    priors = {'amplitude': amp_prior, 'base': base_prior,
              'kappa': kappa_prior, 'shift': shift_prior}
    if b_m:
        priors['base_mod'] = base_prior
    if a_m:
        priors['amplitude_mod'] = amp_prior
    if k_m:
        priors['kappa_mod'] = kappa_prior
    if s_m:
        priors['shift_mod'] = shift_prior

    modelspec[dsig_idx]['prior'] = priors

#    try:
#        # still set as tuple like ('Exponential', {'beta': [[0]]})
#        modelspec[dsig_idx]['prior']['base'][1]['beta'] = b
#        if b_m:
#            modelspec[dsig_idx]['prior']['base_mod'][1]['beta'] = b
#    except TypeError:
#        # has been converted to distribution object
#        modelspec[dsig_idx]['prior']['base']._beta = b
#        if b_m:
#            modelspec[dsig_idx]['prior']['base_mod']._beta = b
#    try:
#        modelspec[dsig_idx]['prior']['amplitude'][1]['beta'] = a
#        if a_m:
#            modelspec[dsig_idx]['prior']['amplitude_mod'][1]['beta'] = a
#    except TypeError:
#        modelspec[dsig_idx]['prior']['amplitude']._beta = a
#        if a_m:
#            modelspec[dsig_idx]['prior']['amplitude_mod']._beta = a
#    try:
#        modelspec[dsig_idx]['prior']['shift'][1]['mean'] = s
#        modelspec[dsig_idx]['prior']['shift'][1]['sd'] = s*2
#        if s_m:
#            modelspec[dsig_idx]['prior']['shift_mod'][1]['mean'] = s
#            modelspec[dsig_idx]['prior']['shift_mod'][1]['sd'] = s*2
#    except TypeError:
#        modelspec[dsig_idx]['prior']['shift']._mean = s
#        modelspec[dsig_idx]['prior']['shift']._sd = s*2
#        if s_m:
#            modelspec[dsig_idx]['prior']['shift_mod']._mean = s
#            modelspec[dsig_idx]['prior']['shift_mod']._sd = s*2
#    try:
#        modelspec[dsig_idx]['prior']['kappa'][1]['mean'] = k
#        modelspec[dsig_idx]['prior']['kappa'][1]['sd'] = k*2
#        if k_m:
#            modelspec[dsig_idx]['prior']['kappa_mod'][1]['mean'] = k
#            modelspec[dsig_idx]['prior']['kappa_mod'][1]['sd'] = k*2
#    except TypeError:
#        modelspec[dsig_idx]['prior']['kappa']._mean = k
#        modelspec[dsig_idx]['prior']['kappa']._sd = k
#        if k_m:
#            modelspec[dsig_idx]['prior']['kappa_mod']._mean = k
#            modelspec[dsig_idx]['prior']['kappa_mod']._sd = k


def _prefit_contrast_modules(est, modelspec, analysis_function,
                             fitter, metric=None, fit_kwargs={}):
    '''
    Perform a rough fit that only allows contrast STRF and dynamic_sigmoid
    parameters to vary.
    '''
    # preserve input modelspec
    modelspec = copy.deepcopy(modelspec)

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    fit_idx = []
    fit_set = ['ctwc', 'ctfir', 'ctlvl', 'dsig']
    for i, m in enumerate(modelspec.modules):
        for id in fit_set:
            if id in m['id']:
                fit_idx.append(i)
                log.info('Found module %d (%s) for subset prefit', i, id)

    tmodelspec = modelspec.copy()

    if len(fit_idx) == 0:
        log.info('No modules matching fit_set for subset prefit')
        return modelspec

    exclude_idx = np.setdiff1d(np.arange(0, len(modelspec)),
                               np.array(fit_idx))
    for i in exclude_idx:
        m = tmodelspec[i]
        if not m.get('phi'):
            log.info('Intializing phi for module %d (%s)', i, m['fn'])
            old = m.get('prior', {})
            m = priors.set_mean_phi([m])[0]  # Inits phi
            m['prior'] = old

        log.info('Freezing phi for module %d (%s)', i, m['fn'])

        m['fn_kwargs'].update(m['phi'])
        m['phi'] = {}
        tmodelspec[i] = m

    # fit the subset of modules
    if metric is None:
        tmodelspec = analysis_function(est, tmodelspec, fitter=fitter,
                                       fit_kwargs=fit_kwargs)
    else:
        tmodelspec = analysis_function(est, tmodelspec, fitter=fitter,
                                       metric=metric, fit_kwargs=fit_kwargs)

    # reassemble the full modelspec with updated phi values from tmodelspec
    for i in fit_idx:
        modelspec[i] = tmodelspec[i]

    return modelspec


def _prefit_dsig_only(est, modelspec, analysis_function,
                      fitter, metric=None, fit_kwargs={}):
    '''
    Perform a rough fit that only allows dynamic_sigmoid parameters to vary.
    '''

    dsig_idx = find_module('dynamic_sigmoid', modelspec)

    # freeze all non-static dynamic sigmoid parameters
    dynamic_phi = {'amplitude_mod': False, 'base_mod': False,
                   'kappa_mod': False, 'shift_mod': False}
    for p in dynamic_phi:
        v = modelspec[dsig_idx]['prior'].pop(p, False)
        if v:
            modelspec[dsig_idx]['fn_kwargs'][p] = np.nan
            dynamic_phi[p] = v

    # Remove ctwc, ctfir, and ctlvl if they exist
    temp = []
    for i, m in enumerate(modelspec.modules):
        if 'ct' in m['id']:
            pass
        else:
            temp.append(m)
    temp = ms.ModelSpec(raw=[temp])
    temp = prefit_mod_subset(est, temp, analysis_function,
                             fit_set=['dynamic_sigmoid'], fitter=fitter,
                             metric=metric, fit_kwargs=fit_kwargs)

    # Put ctwc, ctfir, and ctlvl back in where applicable
    j = 0
    for i, m in enumerate(modelspec.modules):
        if 'ct' in m['id']:
            pass
        else:
            modelspec[i] = temp[j]
            j += 1

    # reset dynamic sigmoid parameters if they were frozen
    for p, v in dynamic_phi.items():
        if v:
            prior = priors._tuples_to_distributions({p: v})[p]
            modelspec[dsig_idx]['fn_kwargs'].pop(p, None)
            modelspec[dsig_idx]['prior'][p] = v
            modelspec[dsig_idx]['phi'][p] = prior.mean()

    return modelspec


def strf_to_contrast(modelspec, IsReload=False, **context):
    modelspec = copy.deepcopy(modelspec)[0]
    modelspec = _strf_to_contrast(modelspec)
    return {'modelspec': modelspec}


def _strf_to_contrast(modelspec, absolute_value=True):
    '''
    Copy prefitted WC and FIR phi values to contrast-based counterparts.
    '''
    modelspec = copy.deepcopy(modelspec)
    wc_idx, ctwc_idx = find_module('weight_channels', modelspec,
                                   find_all_matches=True)
    fir_idx, ctfir_idx = find_module('fir', modelspec,
                                     find_all_matches=True)

    log.info("Updating contrast phi to match prefitted strf ...")

    modelspec[ctwc_idx]['phi'] = copy.deepcopy(modelspec[wc_idx]['phi'])
    modelspec[ctfir_idx]['phi'] = copy.deepcopy(modelspec[fir_idx]['phi'])

    if absolute_value:
        for k, v in modelspec[ctwc_idx]['phi'].items():
            p = np.abs(v)
            modelspec[ctwc_idx]['phi'][k] = p

        for k, v in modelspec[ctfir_idx]['phi'].items():
            p = np.abs(v)
            modelspec[ctfir_idx]['phi'][k] = p

    return modelspec
