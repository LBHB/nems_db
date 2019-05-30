'''
Functions and xforms wrappers for doing a full fit of the gain control model.
'''
import copy
import logging

import nems
import nems.utils
import nems.metrics.api as metrics
from nems.analysis.api import fit_basic, basic_with_copy, pick_best_phi
from nems_lbhb.gcmodel.initializers import init_dsig
from nems.initializers import prefit_mod_subset, rand_phi
from nems.plots.heatmap import _get_wc_coefficients, _get_fir_coefficients

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

    if IsReload:
        return {}

    # Start with a new copy of modelspec, and figure out where
    # dynamic_sigmoid is.
    modelspec = copy.deepcopy(modelspec)
    est = copy.deepcopy(est)
    wc_idx = nems.utils.find_module('weight_channels', modelspec)
    fir_idx = nems.utils.find_module('fir', modelspec)
    lvl_idx = nems.utils.find_module('levelshift', modelspec)
    dsig_idx = nems.utils.find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        raise ValueError("fit_gc should only be used with modelspecs"
                         "containing dynamic_sigmoid")

    # Set up kwargs, fitter_fn and metric_fn arguments for fitting functions
    prefit_kwargs = {'tolerance': prefit_tolerance, 'max_iter': prefit_max_iter}
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}
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
            if not m.get('phi'):
                log.info('Initializing STP module ...')
                m = nems.priors.set_mean_phi([m])[0]  # Init phi for module
                modelspec[i] = m
            break

    # now prefit static dsig
    log.info("Initializing priors and bounds for dsig ...")
    modelspec = init_dsig(est, modelspec)
    log.info('Performing rough fit of static nonlinearity ...\n')
    modelspec = prefit_mod_subset(est, modelspec, fit_basic,
                                  fit_set=['dynamic_sigmoid'],
                                  fitter=fitter_fn,
                                  metric=metric_fn,
                                  fit_kwargs=prefit_kwargs)


    ##########################################################################
    # 3: Finish fitting the LN portion of the model (might also include STP) #
    ##########################################################################

    log.info('Finishing fit for full LN model ...\n')
    # Can't use metric=None directly to fit_basic or it will have a fit,
    # so split up arguments here and only add metric if we gave one.
    fb_args = [est, modelspec, fitter_fn, cost_function]
    fb_kwargs = {'metaname': 'fit_gc', 'fit_kwargs': fit_kwargs}
    if metric_fn is not None:
        fb_kwargs['metric'] = metric_fn
    modelspec = fit_basic(*fb_args, **fb_kwargs)

    # 3b (OPTIONAL): Freeze STRF parameters before continuing.
    log.info('Freezing STRF parameters ...\n')
    if fixed_strf:
        modelspec[wc_idx]['fn_kwargs']['mean'] = \
                modelspec[wc_idx]['phi'].pop('mean')
        modelspec[wc_idx]['fn_kwargs']['sd'] = \
                modelspec[wc_idx]['phi'].pop('sd')
        modelspec[fir_idx]['fn_kwargs']['coefficients'] = \
                modelspec[fir_idx]['phi'].pop('coefficients')
        modelspec[lvl_idx]['fn_kwargs']['level'] = \
                modelspec[lvl_idx]['phi'].pop('level')


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
        modelspec[lvl_idx]['phi']['level'] = \
                modelspec[lvl_idx]['fn_kwargs'].pop('level')

    return {'modelspec': modelspec}


def fit_gc2(modelspec, est, max_iter=1000, prefit_max_iter=700, tolerance=1e-7,
            prefit_tolerance=10**-5.5, metric='nmse', fitter='scipy_minimize',
            cost_function=None, IsReload=False, post_fit=False,
            **context):
    '''
    Xforms wrapper for fitting the locked STRF=CTSTRF version of the GC model.

    Expects the version of the contrast gain control model in which the
    coefficients of the contrast-based filters always exactly match the
    coefficients of the stimulus-based filters. This model utilizes the
    contrast_kernel module defined in nems_lbhb.gcmodel.modules, which
    is compatible with D > 1 wc and fir keywords.

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
    if IsReload:
        return {}

    wc_idx = nems.utils.find_module('weight_channels', modelspec)
    fir_idx = nems.utils.find_module('fir', modelspec)
    lvl_idx = nems.utils.find_module('levelshift', modelspec)
    ctk_idx = nems.utils.find_module('contrast_kernel', modelspec)
    dsig_idx = nems.utils.find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        raise ValueError("fit_gc should only be used with modelspecs"
                         "containing dynamic_sigmoid")
    # TODO: Be able to handle this case somehow
    if (('coefficients' in modelspec[wc_idx]['phi'])
        and ('coefficients' in modelspec[fir_idx]['phi'])):
           raise ValueError("both wc and fir are using nonparametric "
                            "coefficients, will cause problems with ctkernel")

    # Set up kwargs, fitter_fn and metric_fn arguments for fitting functions
    prefit_kwargs = {'tolerance': prefit_tolerance, 'max_iter': prefit_max_iter}
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}
    fitter_fn = getattr(nems.fitters.api, fitter)
    if metric is not None:
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')
    else:
        metric_fn = None


    # Iterate through fit indices
    for ii in range(modelspec.fit_count):
        log.info("Fit idx:  %d", ii)
        modelspec.set_fit(ii)

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
        modelspec[ctk_idx]['fn_kwargs']['compute_contrast'] = False



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
                if not m.get('phi'):
                    log.info('Initializing STP module ...')
                    m = nems.priors.set_mean_phi([m])[0]  # Init phi for module
                    modelspec[i] = m
                break

        # now prefit static dsig
        log.info("Initializing priors and bounds for dsig ...")
        modelspec = init_dsig(est, modelspec)
        import pdb
        pdb.set_trace()
        log.info('Performing rough fit of static nonlinearity ...\n')
        modelspec = prefit_mod_subset(est, modelspec, fit_basic,
                                      fit_set=['dynamic_sigmoid'],
                                      fitter=fitter_fn,
                                      metric=metric_fn,
                                      fit_kwargs=prefit_kwargs)


        ##########################################################################
        # 3: Finish fitting the LN portion of the model (might also include STP) #
        ##########################################################################

        log.info('Finishing fit for full LN model ...\n')
        # Can't use metric=None directly to fit_basic or it will have a fit,
        # so split up arguments here and only add metric if we gave one.
        fb_args = [est, modelspec, fitter_fn, cost_function]
        fb_kwargs = {'metaname': 'fit_gc', 'fit_kwargs': fit_kwargs}
        if metric_fn is not None:
            fb_kwargs['metric'] = metric_fn
        modelspec = fit_basic(*fb_args, **fb_kwargs)

        # 3b: Freeze STRF parameters before continuing, after setting
        #     coefficients for contrast_kernel.
        log.info("Copying STRF parameters to contrast_kernel...")
        wc_c = _get_wc_coefficients(modelspec)
        fir_c = _get_fir_coefficients(modelspec)
        modelspec[ctk_idx]['fn_kwargs'].update({'wc_coefficients': wc_c,
                                                'fir_coefficients': fir_c})

        log.info('Freezing STRF parameters ...\n')
        modelspec[wc_idx]['fn_kwargs'].update(modelspec[wc_idx]['phi'])
        frozen_wc = modelspec[wc_idx]['phi'].keys()
        modelspec[wc_idx]['phi'] = {}

        modelspec[fir_idx]['fn_kwargs'].update(modelspec[fir_idx]['phi'])
        frozen_fir = modelspec[fir_idx]['phi'].keys()
        modelspec[fir_idx]['phi'] = {}

        modelspec[lvl_idx]['fn_kwargs'].update(modelspec[lvl_idx]['phi'])
        frozen_lvl = modelspec[lvl_idx]['phi'].keys()
        modelspec[lvl_idx]['phi'] = {}


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
        modelspec[ctk_idx]['fn_kwargs']['compute_contrast'] = True


        log.info('Finishing fit for full GC model ...\n')
        modelspec = fit_basic(est, modelspec, fitter_fn, cost_function,
                              metric=metric_fn, metaname='fit_gc',
                              fit_kwargs=fit_kwargs)

        # 4b: Unfreeze STRF parameters.
        log.info('Unfreezing STRF parameters ...\n')
        for k in frozen_wc:
            modelspec[wc_idx]['phi'][k] = modelspec[wc_idx]['fn_kwargs'].pop(k)
        for k in frozen_fir:
            modelspec[fir_idx]['phi'][k] = modelspec[fir_idx]['fn_kwargs'].pop(k)
        for k in frozen_lvl:
            modelspec[lvl_idx]['phi'][k] = modelspec[lvl_idx]['fn_kwargs'].pop(k)



        ######################################
        # 5: Fit all modules together        #
        ######################################
        if post_fit:

            def cost(*args, **kwargs):
                copy = [(wc_idx, ctk_idx), (fir_idx, ctk_idx)]
                return basic_with_copy(*args, **kwargs, copy_phi=copy)

            log.info('Fitting all modules together ...\n')
            modelspec[ctk_idx]['fn_kwargs']['auto_copy'] = True
            modelspec = fit_basic(est, modelspec, fitter_fn,
                                  cost_function=cost,
                                  metric=metric_fn, metaname='fit_gc',
                                  fit_kwargs=fit_kwargs)

    modelspec.set_fit(0)
    return pick_best_phi(modelspec, est=est, **context)
