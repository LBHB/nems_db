'''
Functions and xforms wrappers for doing a full fit of the gain control model.
'''
import copy
import logging

import nems
import nems.utils
import nems.metrics.api as metrics
from nems.analysis.api import fit_basic
from nems_lbhb.gcmodel.initializers import init_dsig
from nems.initializers import prefit_mod_subset

log = logging.getLogger(__name__)


def fit_gc(modelspec, est, max_iter=1000, prefit_max_iter=700, tolerance=1e-7,
           prefit_tolerance=10**-5.5, metric='nmse', fitter='scipy_minimize',
           cost_function=None,  fixed_strf=False, IsReload=False, **context):
    '''
    Xforms wrapper for fitting the locked STRF=CTSTRF version of the GC model.

    Expects the version of the contrast gain control model in which the
    coefficients of the contrast-based filters always exactly match the
    coefficients of the stimulus-based filters. This model utilizes the
    reimplemented weight_channels, fir, and levelshift modules defined
    in nems_lbhb.gcmodel.modules

    Steps:
        -Freeze the GC portion of the model by movinug phi to fn_kwargs
         (which is only the second set of dsig parameters, since the filters
          are fixed).
        -Prefit the LN portion of the model (might also include STP)
        -Finish fitting the LN portion of the model (might also include STP)
        -Fit the GC portion of the model


    Parameters:
    -----------
    modelspec : NEMS ModelSpec
        Stores information about model architecture and parameter values.
    est : NEMS Recording
        A container for a related set of NEMS signals. In short, the data.
    max_iter : int
        Max number of times that the fitter can be called during optimization.
    tolerance : float
        Error tolerance argument to be passed to the fitter. E.g. stop
        optimization if change in error is smaller than this number.
    prefit_tolerance : float
        As tolerance, but only used for the prefitting step.
    metric : string
        Name of the function to use for calculating error. Used by the
        cost_function when comparing evaluations of modelspec.
    fitter : string
        Name of the optimization function to use
    cost_function : function object or None
        Function that will be passed to the optimizer for determining error.
        If None, lets nems.analysis.fit_basic decide what to use.
    IsReload : boolean
        Indicates to xforms evaluation if the model is being fit for the first
        time or being loaded from a saved analysis.
            if True: Do the fit
            if False: Skip the fit and just return modelspec as-is
    **context : dict
        Running record of the return values of each step in the xforms spec
        that has been evaluated so far. See nems.xforms.

    Returns:
    {'modelspec': modelspec} : dict
        Updates context for xforms evaluation

    '''

    # Start with a new copy of modelspec, and figure out where
    # dynamic_sigmoid is.
    modelspec = copy.deepcopy(modelspec)
    wc_idx = nems.utils.find_module('weight_channels', modelspec)
    fir_idx = nems.utils.find_module('fir', modelspec)
    lvl_idx = nems.utils.find_module('levelshift', modelspec)
    dsig_idx = nems.utils.find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        raise ValueError("fit_gc should only be used with modelspecs"
                         "containing dynamic_sigmoid")

    # Set up kwargs, fitter_fn and metric_fn arguments for fitting functions
    prefit_kwargs = {'tolerance': prefit_tolerance, 'max_iter': prefit_max_iter}
    fit_kwargs = {'tolerance': prefit_tolerance, 'max_iter': max_iter}
    fitter_fn = getattr(nems.fitters.api, fitter)
    if metric is not None:
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')
    else:
        metric_fn = None


    #########################################
    # 1: Freeze the GC portion of the model #
    #########################################
    log.info('Freezing dynamic portion of dsig and STRF modules...\n')
    frozen_phi = {}
    frozen_priors = {}
    for k in ['amplitude_mod', 'base_mod', 'shift_mod', 'kappa_mod']:
        if k in modelspec[dsig_idx]['phi']:
            frozen_phi[k] = modelspec[dsig_idx]['phi'].pop(k)
        if k in modelspec[dsig_idx]['prior']:
            frozen_priors[k] = modelspec[dsig_idx]['prior'].pop(k)
    modelspec[wc_idx]['fn_kwargs']['compute_contrast'] = False
    modelspec[fir_idx]['fn_kwargs']['compute_contrast'] = False
    modelspec[lvl_idx]['fn_kwargs']['compute_contrast'] = False



    ##################################################################
    # 2: Prefit the LN portion of the model (might also include STP) #
    ##################################################################

    log.info('Initializing linear model and performing rough fit ...\n')
    # fit without STP module first (if there is one)
    modelspec = nems.initializers.prefit_to_target(
            est, modelspec, fit_basic, target_module='levelshift',
            extra_exclude=['stp'], fitter=fitter_fn, metric=metric_fn,
            fit_kwargs=prefit_kwargs)

    # then initialize the STP module (if there is one)
    for i, m in enumerate(modelspec.modules):
        if 'stp' in m['fn']:
            log.info('Initializing STP module ...')
            m = nems.priors.set_mean_phi([m])[0]  # Init phi for module
            modelspec[i] = m
            break

    # now prefit static dsig
    log.info("Initializing priors and bounds for dsig ...")
    modelspec = init_dsig(est, modelspec)
    log.info('Performing rough fit of static nonlinearity ...\n')
    modelspec = prefit_mod_subset(est, modelspec, fit_basic,
                                  fit_set=['dynamic_sigmoid'], fitter=fitter_fn,
                                  metric=metric_fn, fit_kwargs=prefit_kwargs)


    ##########################################################################
    # 3: Finish fitting the LN portion of the model (might also include STP) #
    ##########################################################################

    log.info('Finishing fit for full LN model ...\n')
    modelspec = fit_basic(est, modelspec, fitter_fn, cost_function,
                          metric=metric_fn, metaname='fit_gc',
                          fit_kwargs=fit_kwargs)

    # 3b (OPTIONAL): Freeze STRF parameters before continuing.
    log.info('Freezing STRF parameters ...\n')
    if fixed_strf:
        modelspec[wc_idx]['fn_kwargs']['mean'] = \
                modelspec[wc_idx]['phi'].pop('mean')
        modelspec[wc_idx]['fn_kwargs']['sd'] = \
                modelspec[wc_idx]['phi'].pop('sd')
        modelspec[fir_idx]['fn_kwargs']['coefficients'] = \
                modelspec[fir_idx]['phi'].pop('coefficients')


    ######################################
    # 4: Fit the GC portion of the model #
    ######################################

    log.info('Unfreezing dynamic portion of dsig ...\n')
    # make dynamic_sigmoid dynamic again
    for k, v in frozen_phi.items():
        # Initialize _mod values equal to their counterparts
        # e.g. amplitude_mod[:4] = amplitude
        modelspec[dsig_idx]['phi'][k] = \
            modelspec[dsig_idx]['phi'][k[:-4]].copy()
    for k, v in frozen_priors.items():
        modelspec[dsig_idx]['prior'][k] = v
    modelspec[wc_idx]['fn_kwargs']['compute_contrast'] = True
    modelspec[fir_idx]['fn_kwargs']['compute_contrast'] = True
    modelspec[lvl_idx]['fn_kwargs']['compute_contrast'] = True

    log.info('Finishing fit for full GC model ...\n')
    modelspec = fit_basic(est, modelspec, fitter_fn, cost_function,
                          metric=metric_fn, metaname='fit_gc',
                          fit_kwargs=fit_kwargs)

    # 4b (OPTIONAL): If STRF was frozen, unfreeze it.
    log.info('Unfreezing STRF parameters ...\n')
    if fixed_strf:
        modelspec[wc_idx]['phi']['mean'] = \
                modelspec[wc_idx]['fn_kwargs'].pop('mean')
        modelspec[wc_idx]['phi']['sd'] = \
                modelspec[wc_idx]['fn_kwargs'].pop('sd')
        modelspec[fir_idx]['phi']['coefficients'] = \
                modelspec[fir_idx]['fn_kwargs'].pop('coefficients')

    return {'modelspec': modelspec}
