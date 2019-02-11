import logging
import copy

import numpy as np

import nems.epoch
import nems.modelspec as ms
from nems.utils import find_module
from nems.initializers import prefit_to_target, prefit_mod_subset
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
    else:
        modelspec = _init_logistic_sigmoid(rec, modelspec, dsig_idx)

    return modelspec


def _init_logistic_sigmoid(rec, modelspec, dsig_idx):

    if dsig_idx == len(modelspec):
        fit_portion = modelspec.modules
    else:
        fit_portion = modelspec.modules[:dsig_idx]

    # generate prediction from module preceeding dexp

    # HACK to get phi for ctwc, ctfir, ctlvl which have not been prefit yet
    for i, m in enumerate(fit_portion):
        if not m.get('phi', None):
            if [k in m['id'] for k in ['ctwc', 'ctfir', 'ctlvl']]:
                old = m.get('prior', {})
                m = priors.set_mean_phi([m])[0]
                m['prior'] = old
                fit_portion[i] = m
            else:
                log.warning("unexpected module missing phi during init step\n:"
                            "%s, #%d", m['id'], i)

    ms.fit_mode_on(fit_portion)
    rec = ms.evaluate(rec, fit_portion)
    ms.fit_mode_off(fit_portion)

    pred = rec['pred'].as_continuous()
    resp = rec['resp'].as_continuous()

    mean_pred = np.nanmean(pred)
    min_pred = np.nanmean(pred) - np.nanstd(pred)*3
    max_pred = np.nanmean(pred) + np.nanstd(pred)*3
    if min_pred < 0:
        min_pred = 0
        mean_pred = (min_pred+max_pred)/2

    pred_range = max_pred - min_pred
    min_resp = max(np.nanmean(resp)-np.nanstd(resp)*3, 0)  # must be >= 0
    max_resp = np.nanmean(resp)+np.nanstd(resp)*3
    resp_range = max_resp - min_resp

    # Rather than setting a hard value for initial phi,
    # set the prior distributions and let the fitter/analysis
    # decide how to use it.
    base0 = min_resp + 0.05*(resp_range)
    amplitude0 = resp_range
    shift0 = mean_pred
    kappa0 = pred_range
    log.info("Initial   base,amplitude,shift,kappa=({}, {}, {}, {})"
             .format(base0, amplitude0, shift0, kappa0))

    base = ('Exponential', {'beta': base0})
    amplitude = ('Exponential', {'beta': amplitude0})
    shift = ('Normal', {'mean': shift0, 'sd': pred_range**2})
    kappa = ('Exponential', {'beta': kappa0})

    modelspec[dsig_idx]['prior'].update({
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa,
            'base_mod': base, 'amplitude_mod':amplitude, 'shift_mod':shift,
            'kappa_mod': kappa
            })

    for kw in modelspec[dsig_idx]['fn_kwargs']:
        if kw in ['base_mod', 'amplitude_mod', 'shift_mod', 'kappa_mod']:
            modelspec[dsig_idx]['prior'].pop(kw)

    modelspec[dsig_idx]['bounds'] = {
            'base': (1e-15, None),
            'amplitude': (1e-15, None),
            'shift': (None, None),
            'kappa': (1e-15, None),
            }

    return modelspec


def _init_double_exponential(rec, modelspec, target_i, nl_mode=2):

    if target_i == len(modelspec):
        fit_portion = modelspec.modules
    else:
        fit_portion = modelspec.modules[:target_i]

    # generate prediction from modules preceeding dsig

    # ensures all previous modules have their phi initialized
    # choose prior mean if not found
    for i, m in enumerate(fit_portion):
        if ('phi' not in m.keys()) and ('prior' in m.keys()):
            log.debug('Phi not found for module, using mean of prior: %s',
                      m)
            m = priors.set_mean_phi([m])[0]  # Inits phi for 1 module
            fit_portion[i] = m

    ms.fit_mode_on(fit_portion)
    rec = ms.evaluate(rec, fit_portion)
    ms.fit_mode_off(fit_portion)

    in_signal = modelspec[target_i]['fn_kwargs']['i']
    pchans = rec[in_signal].shape[0]
    amp = np.zeros([pchans, 1])
    base = np.zeros([pchans, 1])
    kappa = np.zeros([pchans, 1])
    shift = np.zeros([pchans, 1])

    for i in range(pchans):
        resp = rec['resp'].as_continuous()
        pred = rec[in_signal].as_continuous()[i:(i+1), :]
        if resp.shape[0] == pchans:
            resp = resp[i:(i+1), :]

        keepidx = np.isfinite(resp) * np.isfinite(pred)
        resp = resp[keepidx]
        pred = pred[keepidx]

        # choose phi s.t. dexp starts as almost a straight line
        # phi=[max_out min_out slope mean_in]
        # meanr = np.nanmean(resp)
        stdr = np.nanstd(resp)

        # base = np.max(np.array([meanr - stdr * 4, 0]))
        base[i, 0] = np.min(resp)
        # base = meanr - stdr * 3

        # amp = np.max(resp) - np.min(resp)
        if nl_mode == 1:
            amp[i, 0] = stdr * 3
            predrange = 2 / (np.max(pred) - np.min(pred) + 1)
        elif nl_mode == 2:
            mask=np.zeros_like(pred,dtype=bool)
            pct=91
            while sum(mask)<.01*pred.shape[0]:
                pct-=1
                mask=pred>np.percentile(pred,pct)
            if pct !=90:
                log.warning('Init dexp: Default for init mode 2 is to find mean '
                         'of responses for times where pred>pctile(pred,90). '
                         '\nNo times were found so this was lowered to '
                         'pred>pctile(pred,%d).', pct)
            amp[i, 0] = resp[mask].mean()
            predrange = 2 / (np.std(pred)*3)
        else:
            raise ValueError('nl mode = {} not valid'.format(nl_mode))

        shift[i, 0] = np.mean(pred)
        # shift = (np.max(pred) + np.min(pred)) / 2

        kappa[i, 0] = np.log(predrange)

    modelspec[target_i]['phi'].update({
            'base': base, 'amplitude': amp, 'shift': shift, 'kappa': kappa
            })

    amp_prior = ('Normal', {'mean': amp, 'sd': 1.0})
    base_prior = ('Normal', {'mean': base, 'sd': 1.0})
    kappa_prior = ('Normal', {'mean': kappa, 'sd': 1.0})
    shift_prior = ('Normal', {'mean': shift, 'sd': 1.0})

    modelspec[target_i]['prior'].update({
            'base': base_prior, 'amplitude': amp_prior, 'shift': shift_prior,
            'kappa': kappa_prior,
            })

    log.info("Init dexp: %s", modelspec[target_i]['phi'])

    return modelspec


def dsig_phi_to_prior(modelspec):
    '''
    Sets priors for dynamic_sigmoid equal to the current phi for the
    same module. Used for random-sample fits - all samples are initialized
    and pre-fit the same way, and then randomly sampled from the new priors.

    Parameters
    ----------
    modelspec : list of dictionaries
        A NEMS modelspec containing, at minimum, a dynamic_sigmoid module

    Returns
    -------
    modelspec : A copy of the input modelspec with priors updated.

    '''

    modelspec = copy.deepcopy(modelspec)
    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    dsig = modelspec[dsig_idx]

    phi = dsig['phi']
    b = phi['base']
    a = phi['amplitude']
    k = phi['kappa']
    s = phi['shift']

    p = dsig['prior']
    p['base'][1]['beta'] = b
    p['amplitude'][1]['beta'] = a
    p['shift'][1]['mean'] = s  # Do anything to scale sd?
    p['kappa'][1]['beta'] = k

    return modelspec


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
