import logging
import copy

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import nems.epoch
import nems.modelspec as ms
from nems.utils import find_module
from nems import signal
from nems.modules.weight_channels import gaussian_coefficients
from nems.modules.fir import per_channel
from nems.modules.nonlinearity import (_logistic_sigmoid, _double_exponential,
                                       _dlog)
from nems.initializers import prefit_to_target, prefit_mod_subset
from nems.analysis.api import fit_basic
import nems.fitters.api
import nems.metrics.api as metrics
from nems import priors

log = logging.getLogger(__name__)


def make_contrast_signal(rec, name='contrast', source_name='stim', ms=500,
                         bins=None, bands=1, dlog=False, continuous=False,
                         normalize=False, percentile=50, ignore_zeros=True):
    '''
    Creates a new signal whose values represent the degree of variability
    in each channel of the source signal. Each value is based on the
    previous values within a range specified by either <ms> or <bins>.

    Contrast is calculated as the coefficient of variation within a rolling
    window, using the formula: standard deviation / mean.

    If more than one spectral band is used in the calculation, the contrast for
    a number of channels equal to floor(bands/2) at the "top" and "bottom" of
    the array will be calculated separately. For example, if bands=3, then
    the contrast of the topmost and bottommost channels will be based on
    the top 2 and bottom 2 channels, respectively, since the 3rd channel in
    each case would fall outside the array.

    Similarly, for any number of temporal bins based on ms, the "left" and
    "right" most "columns" of the array will be replaced with zeros. For
    LBHB's dataset this is a safe assumption since those portions of the array
    will always be filled with silence anyway, but this might necessitate
    padding for other datasets.

    Only supports RasterizedSignal contained within a NEMS recording.
    To operate directly on a 2d Array, use contrast_calculation.

    Parameters
    ----------
    rec : NEMS recording
        Recording containing, at minimum, the signal specified by "source_name."
    name : str
        Name of the new signal to be created
    source_name : str
        Name of the signal within rec whose data the contrast calculation will
        be performed on.
    ms : int
        Number of milliseconds to use for the temporal axis of the convolution
        filter. In conjunction with the sampling frequency of the source
        signal, ms will be translated into a number of bins according to
        the formula: number of bins = int((ms/1000) x sampling frequency
    bins : int
        Serves the same purpose as ms, except the number of bins is
        specified directly.
    bands : int
        Number of bins to use for the spectral axis of the convolution filter.
    dlog : boolean
        If true, apply a log transformation to the source signal before
        calculating contrast.
    continuous : boolean
        If true, do not rectify the contrast result.
        If false, set result equal to 1 where value is above <percentile>,
        0 otherwise.
    normalize : boolean
        If continuous is true, normalizes the result to the range 0 to 1.
    percentile : int
        If continuous is false, specifies the percentile cutoff for
        contrast rectification.
    ignore_zeros : boolean
        If true, and continuous is false, "columns" containing zeros for all
        spectral channels (i.e no stimulus) will be ignored when determining
        the percentile-based cutoff value for contrast rectification.

    Returns
    -------
    rec : NEMS recording
        A new recording containing all signals in the original recording plus
        a new signal named <name>.

    Examples
    --------
    If ms=100, source_signal.fs=100, and bands=1, convolution filter shape
    will be: (1, 21). The shape of the second axis includes (100*100/1000 + 1)
    zeros, to force the behavior of the convolution to be causal rather than
    anti-causal.

    For ms=100, source_signal.fs=100, and bands=5, convolution filter shape
    will be: (5, 21), where each "row" contains the same number of zeros as the
    previous example.


    '''

    rec = rec.copy()

    source_signal = rec[source_name]
    if not isinstance(source_signal, signal.RasterizedSignal):
        try:
            source_signal = source_signal.rasterize()
        except AttributeError:
            raise TypeError("signal with key {} was not a RasterizedSignal"
                            " and could not be converted to one."
                            .format(source_name))

    if dlog:
        log.info("Applying dlog transformation to stimulus prior to "
                 "contrast calculation.")
        fn = lambda x: _dlog(x, -1)
        source_signal = source_signal.transform(fn)
        rec[source_name] = source_signal

    if ms is not None:
        history = int((ms/1000)*source_signal.fs)
    elif bins is not None:
        history = int(bins)
    else:
        raise ValueError("Either ms or bins parameter must be specified.")
    history = max(1,history)

    # SVD constrast is now std / mean in rolling window (duration ms),
    # confined to each frequency channel
    array = source_signal.as_continuous().copy()
    array[np.isnan(array)] = 0
    contrast = contrast_calculation(array, history, bands, 'same')

    # Cropped time binds need to be filled in, otherwise convolution for
    # missing spectral bands will end up with empty 'corners'
    # (and normalization is thrown off for any stimuli with nonzero values
    #  near edges)
    # Reasonable to fill in with zeros for natural sounds dataset since
    # the stimuli are always surrounded by pre-post silence anyway
    cropped_time = history
    contrast[:, :cropped_time] = 0
    contrast[:, -cropped_time:] = 0

    # number of spectral channels that get removed for mode='valid'
    # total is times 2 for 'top' and 'bottom'
    cropped_khz = int(np.floor(bands/2))
    i = 0
    while cropped_khz > 0:
        reduced_bands = bands-cropped_khz

        # Replace top
        top_replacement = contrast_calculation(array[:reduced_bands, :],
                                               history, reduced_bands,
                                               'valid')
        contrast[i][cropped_time:-cropped_time] = top_replacement

        # Replace bottom
        bottom_replacement = contrast_calculation(array[-reduced_bands:, :],
                                                  history, reduced_bands,
                                                  'valid')
        contrast[-(i+1)][cropped_time:-cropped_time] = bottom_replacement

        i += 1
        cropped_khz -= 1

    if continuous:
        if normalize:
            # Map raw values to range 0 - 1
            contrast /= np.max(np.abs(contrast))
        rectified = contrast

    else:
        # Binary high/low contrast based on percentile cutoff.
        # 50th percentile by default.
        if ignore_zeros:
            # When calculating cutoff, ignore time bins where signal is 0
            # for all spectral channels (i.e. no stimulus present)
            no_zeros = contrast[:, ~np.all(contrast == 0, axis=0)]
            cutoff = np.nanpercentile(no_zeros, percentile)
        else:
            cutoff = np.nanpercentile(contrast, percentile)
        rectified = np.where(contrast >= cutoff, 1, 0)

    contrast_sig = source_signal._modified_copy(rectified)
    rec[name] = contrast_sig

    return rec


def contrast_calculation(array, history, bands, mode):
    '''
    Parameters
    ----------
    array : 2d Ndarray
        The data to perform the contrast calculation on,
        contrast = standard deviation / mean
    history : int
        The number of nonzero bins for the convolution filter.
        history + 1 zeros will be padded onto the second dimension.
    bands : int
        The number of bins in the first dimension of the convolution filter.
    mode : str
        See scipy.signal.conolve2d
        Generally, 'valid' to drop rows/columns that would require padding,
        'same' to pad those rows/columns with nans.

    Returns
    -------
    contrast : 2d Ndarray


    '''
    array = copy.deepcopy(array)
    filt = np.concatenate((np.zeros([bands, history+1]),
                           np.ones([bands, history])), axis=1)/(bands*history)
    mn = convolve2d(array, filt, mode=mode, fillvalue=np.nan)

    var = convolve2d(array ** 2, filt, mode=mode, fillvalue=np.nan) - mn**2

    contrast = np.sqrt(var) / (mn*.99 + np.nanmax(mn)*0.01)

    return contrast


def add_contrast(rec, name='contrast', source_name='stim', ms=500, bins=None,
                 continuous=False, normalize=False, dlog=False, bands=1,
                 percentile=50, ignore_zeros=True, IsReload=False, **context):
    '''xforms wrapper for make_contrast_signal'''
    rec_with_contrast = make_contrast_signal(
            rec, name=name, source_name=source_name, ms=ms, bins=bins,
            percentile=percentile, normalize=normalize, dlog=dlog, bands=bands,
            ignore_zeros=ignore_zeros, continuous=continuous
            )
    return {'rec': rec_with_contrast}


def add_onoff(rec, name='contrast', source='stim', isReload=False, **context):
    # TODO: not really working yet...
    new_rec = copy.deepcopy(rec)
    s = new_rec[source]
    if not isinstance(s, signal.RasterizedSignal):
        try:
            s = s.rasterize()
        except AttributeError:
            raise TypeError("signal with key {} was not a RasterizedSignal"
                            " and could not be converted to one."
                            .format(source))

    st_eps = nems.epoch.epoch_names_matching(s.epochs, '^STIM_')
    pre_eps = nems.epoch.epoch_names_matching(s.epochs, 'PreStimSilence')
    post_eps = nems.epoch.epoch_names_matching(s.epochs, 'PostStimSilence')

    st_indices = [s.get_epoch_indices(ep) for ep in st_eps]
    pre_indices = [s.get_epoch_indices(ep) for ep in pre_eps]
    post_indices = [s.get_epoch_indices(ep) for ep in post_eps]

    # Could definitely make this more efficient
    data = np.zeros([1, s.ntimes])
    for a in st_indices:
        for i in a:
            lb, ub = i
            data[:, lb:ub] = 1.0
    for a in pre_indices:
        for i in a:
            lb, ub = i
            data[:, lb:ub] = 0.0
    for a in post_indices:
        for i in a:
            lb, ub = i
            data[:, lb:ub] = 0.0

    attributes = s._get_attributes()
    attributes['chans'] = ['StimOnOff']
    new_sig = signal.RasterizedSignal(data=data, safety_checks=False,
                                      **attributes)
    new_rec[name] = new_sig

    return {'rec': new_rec}


def dynamic_sigmoid(rec, i, o, c, base, amplitude, shift, kappa,
                    base_mod=0, amplitude_mod=0, shift_mod=0,
                    kappa_mod=0, eq='logsig'):

    if not rec[c]:
        # If there's no ctpred yet (like during initialization),
        base_mod = np.nan
        amplitude_mod = np.nan
        shift_mod = np.nan
        kappa_mod = np.nan
        contrast = np.zeros_like(rec['resp'].as_continuous())
    else:
        contrast = rec[c].as_continuous()

    if np.isnan(base_mod):
        base_mod = base
    b = base + (base_mod - base)*contrast

    if np.isnan(amplitude_mod):
        amplitude_mod = amplitude
    a = amplitude + (amplitude_mod - amplitude)*contrast

    if np.isnan(shift_mod):
        shift_mod = shift
    s = shift + (shift_mod - shift)*contrast

    if np.isnan(kappa_mod):
        kappa_mod = kappa
    k = kappa + (kappa_mod - kappa)*contrast

    if eq.lower() in ['logsig', 'logistic_sigmoid', 'l']:
        fn = lambda x: _logistic_sigmoid(x, b, a, s, k)
    elif eq.lower() == ['dexp', 'double_exponential', 'd']:
        fn = lambda x: _double_exponential(x, b, a, s, k)
    else:
        # Not a recognized equation, do logistic_sigmoid by default.
        fn = lambda x: _logistic_sigmoid(x, b, a, s, k)

    return [rec[i].transform(fn, o)]


def init_dsig(rec, modelspec):
    '''
    Initialization of priors for logistic_sigmoid,
    based on process described in methods of Rabinowitz et al. 2014.
    '''

    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    if dsig_idx is None:
        log.warning("No dsig module was found, can't initialize.")
        return modelspec

    modelspec = copy.deepcopy(modelspec)
    rec = copy.deepcopy(rec)

    if modelspec[dsig_idx]['fn_kwargs'].get('eq', '') in \
            ['dexp', 'd', 'double_exponential']:
        modelspec = _init_double_exponential(rec, modelspec, dsig_idx)
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


def _init_double_exponential(rec, modelspec, target_i):

    if target_i == len(modelspec):
        fit_portion = modelspec.modules
    else:
        fit_portion = modelspec.modules[:target_i]

    # generate prediction from modules preceeding dsig

    # HACK
    for i, m in enumerate(fit_portion):
        if not m.get('phi', None):
            old = m.get('prior', {})
            m = priors.set_mean_phi([m])[0]
            m['prior'] = old
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
        amp[i, 0] = stdr * 3

        shift[i, 0] = np.mean(pred)
        # shift = (np.max(pred) + np.min(pred)) / 2

        predrange = 2 / (np.max(pred) - np.min(pred) + 1)
        kappa[i, 0] = np.log(predrange)

    amp = ('Normal', {'mean': amp, 'sd': 1.0})
    base = ('Normal', {'mean': base, 'sd': 1.0})
    kappa = ('Normal', {'mean': kappa, 'sd': 1.0})
    shift = ('Normal', {'mean': shift, 'sd': 1.0})

    modelspec[target_i]['prior'].update({
            'base': base, 'amplitude': amp, 'shift': shift, 'kappa': kappa,
            })


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
    modelspec = copy.deepcopy(modelspecs)[0]
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


def weight_channels(rec, i, o, ci, co, n_chan_in, mean, sd, **kwargs):
    '''
    Parameters
    ----------
    rec : recording
        Recording to transform
    i : string
        Name of input signal
    o : string
        Name of output signal
    mean : array-like (between 0 and 1)
        Centers of Gaussian channel weights
    sd : array-like
        Standard deviation of Gaussian channel weights
    '''
    coefficients = gaussian_coefficients(mean, sd, n_chan_in)
    fn = lambda x: coefficients @ x
    gc_fn = lambda x: np.abs(coefficients) @ x
    return [rec[i].transform(fn, o), rec[ci].transform(gc_fn, co)]


def fir(rec, i, o, ci, co, coefficients=[]):
    """
    apply fir filters of the same size in parallel. convolve in time, then
    sum across channels

    coefficients : 2d array
        all coefficients matrix shape=channel X time lag, for which
        .shape[0] matched to the channel count of the input

    input :
        nems signal named in 'i'. must have dimensionality matched to size
        of coefficients matrix.
    output :
        nems signal in 'o' will be 1 x time singal (single channel)
    """
    fn = lambda x: per_channel(x, coefficients)
    gc_fn = lambda x: per_channel(x, np.abs(coefficients))
    return [rec[i].transform(fn, o), rec[ci].transform(gc_fn, co)]


def levelshift(rec, i, o, ci, co, level):
    '''
    Parameters
    ----------
    level : a scalar to add to every element of the input signal.
    '''
    fn = lambda x: x + level
    gc_fn = lambda x: x + np.abs(level)
    return [rec[i].transform(fn, o), rec[ci].transform(gc_fn, co)]


def sample_DRC(fs=100, segment_duration=3000, n_segments=120, high_hw=15.0,
               low_hw=5.0, mean=40.0, n_tones=23, f_low=0.5, f_high=22.6):
    '''
    Generate a random contrast dynamic random chord (RC-DRC) stimulus.
    Implementation based on:
    "Spectrotemporal Contrast Kernels for Neurons in Primary Auditory Cortex,"
    Rabinowitz et al., 2012. doi: 10.1523/JNEUROSCI.1715-12.2012

    See rec_from_DRC to convert the array into a NEMS recording.

    Parameters
    ----------
    fs : int
        Sampling frequency in hertz to imitate.
    segment_duration : int
        Duration of each tone sampling segment in milliseconds.
    n_segments : int
        Total number of segments to sample. Length of axis 1 of the resulting
        array is fs*n_segments*segment_duration/1000
    high_hw : float
        Half-width of uniform distribution for high-contrast segments, in db.
    low_hw : float
        Half-width of uniform distribution for low-contrast segments, in db.
    mean : float
        Mean level of uniform distribution for both high- and low-contrast
        segments, in db.
    n_tones : int
        Total number of frequencies to use. Also determines length of axis 0
        of the resulting array.
    f_low : float
        Lowest frequency to use in khz.
    f_high : float
        Highest frequency to use in khz.

    Returns
    -------
    (stim, contrast, frequencies) : 2d Array, 2d Array, 1d Array
        stim: Sampled segments concatenated along axis 1,
              shape will be (n_tones, fs*n_segments*segment_duration/1000)
        contrast: Values set to 1 for high-contrast segments, 0 for low.
                  Shape will be the same as stim.
        frequencies: kHz values associated with axis 0 of stim,
                     shape will be (n_tones, )

    '''

    # Notes from Rab et al 2012
    # TODO:
    # levels changed every 25ms with 5ms linear ramps between chords

    # Done:
    # Nf = 23 pure tones
    # f_low = 500hz and f_high = 22.6 khz at 1/4 octave intervals
    # frequencies log-spaced between
    # amplitude of each tone always non-zero
    # Within each segment, distribution of levels for each band drawn from
    # high-contrast (half-width = 15db, SD = 8.7 db) uniform level distribution,
    # or low-contrast (half-width = 5db, SD = 2.9 db)
    # both distributions mean level 40 db
    # 3s duration for each segment split into 120 chords
    # presented between 80 and 120 segments for recording

    # I guess the freqs aren't actually necessary yet since we aren't generating
    # the stimuli, but maybe useful later.
    frequencies = np.logspace(np.log(f_low), np.log(f_high), num=n_tones,
                              base=np.e)
    segment_length = round(fs*segment_duration/1000)
    segment_shape = (n_tones, segment_length)

    stim_segments = []
    contrast_segments = []
    # For each segment, pick random number of high-contrast frequencies
    for i in range(n_segments):
        stim_segment = np.zeros(segment_shape)
        contrast_segment = np.zeros(segment_shape)
        high_low = np.random.randint(0, 2, size=(n_tones))
        high_bands = np.where(high_low > 0)[0]
        low_bands = np.where(high_low < 1)[0]

        high_sample = np.random.uniform(low=(mean-high_hw), high=(mean+high_hw),
                                 size=(high_bands.shape[0], segment_length))
        low_sample = np.random.uniform(low=(mean-low_hw), high=(mean+low_hw),
                                size=(low_bands.shape[0], segment_length))

        stim_segment[high_bands] = high_sample
        stim_segment[low_bands] = low_sample
        contrast_segment[high_bands] = 1

        stim_segments.append(stim_segment)
        contrast_segments.append(contrast_segment)

    stim = np.concatenate(stim_segments, axis=1)
    contrast = np.concatenate(contrast_segments, axis=1)
    return (stim, contrast, frequencies)


def rec_from_DRC(fs=100, n_segments=120, rec_name='DRC Test',
                 sig_names=['stim', 'contrast']):
    '''
    Generate a NEMS recording that contains a synthetic RC-DRC stimulus
    signal and its associated contrast signal.

    Parameters
    ----------
    fs : int
        Sampling rate to mimic when generating the signals.
    n_segments : int
        Number of 3 second segments the signal will contain.
    rec_name : str
        Name that will be given to the returned recording
    sig_names : list of strings
        Names that will be given to the generated signals. First name
        is for the stimulus, the second is for the contrast.

    Returns
    -------
    drc_rec : NEMS recording

    '''
    s, c, f = sample_DRC(fs=fs, n_segments=n_segments)
    freq_names = ['%.1f kHz' % khz for khz in reversed(f)]
    drc_rec = nems.recording.load_recording_from_arrays(
            [s, c], rec_name, fs, sig_names,
            signal_kwargs=[{'chans': freq_names}, {'chans': freq_names}],
            )

    return drc_rec


def test_DRC():
    '''
    Plot a sample DRC stimulus.
    '''
    fig = plt.figure()
    x, _, _ = sample_DRC(n_segments=12)
    plt.imshow(x, aspect='auto', cmap=plt.get_cmap('jet'))

    return fig


def gc_magnitude(b, b_m, a, a_m, s, s_m, k, k_m):
    '''
    Compute the magnitude of the gain control response for a given set of
    dynamic_sigmoid parameters as the mean difference between the
    sigmoid generated for high-contrast conditions vs for low-contrast
    conditions.

    Parameters
    ----------
    (See dynamic_simgoid and nems.modules.nonlinearity._logistic_sigmoid)
    b : float
        base
    b_m : float
        base_mod
    a : float
        amplitude
    a_m : float
        amplitude_mod
    s : float
        shift
    s_m : float
        shift_mod
    k : float
        kappa
    k_m : float
        kappa_mod

    Returns
    -------
    mag : float

    '''
    x_low = np.linspace(s*-1, s*3, 1000)
    x_high = np.linspace(s_m*-1, s_m*3, 1000)

    y_low = _logistic_sigmoid(x_low, b, a, s, k)
    y_high = _logistic_sigmoid(x_high, b_m, a_m, s_m, k_m)

    mag =  np.mean(y_high - y_low)
    return mag


def gc_magnitude_with_ctpred(ctpred, b, b_m, a, a_m, s, s_m, k, k_m):
    b = b + (b_m - b)*ctpred
    a = a + (a_m - a)*ctpred
    s = s + (s_m - s)*ctpred
    k = k + (k_m - k)*ctpred

    x_low = np.linspace(s[0]*-1, s[0]*3, 1000)

    # Can just use the first bin since they always start with silence
    b_low = b[0]
    a_low = a[0]
    s_low = s[0]
    k_low = k[0]

    some_contrast = ctpred[np.abs(ctpred - ctpred[0])/np.abs(ctpred[0]) > 0.02]
    high_contrast = ctpred > np.percentile(some_contrast, 50)
    b_high = np.median(b[high_contrast])
    a_high = np.median(a[high_contrast])
    s_high = np.median(s[high_contrast])
    k_high = np.median(k[high_contrast])

    x_high = np.linspace(s_high*-1, s_high*3, 1000)

    y_low = _logistic_sigmoid(x_low, b_low, a_low, s_low, k_low)
    y_high = _logistic_sigmoid(x_high, b_high, a_high, s_high, k_high)

    return np.mean(y_high - y_low)


# Notes:

# Old contrast calculation implementation:
#filt = np.concatenate((np.zeros([1, max(2, history+1)]),
#                       np.ones([1, max(1, history)])), axis=1)
#contrast = convolve2d(array, filt, mode='same')
