'''
Functions for calculating contrast based on a stimulus.

Functions:
----------
make_contrast_signal: Calculate contrast and add it to a NEMS recording
contrast_calculation: Nuts and bolts of the contrast calculation
add_contrast: xforms wrapper for make_contrast_signal
add_onoff: xforms wrapper for dumbed down version of contrast

'''


import copy
import logging

import numpy as np
from scipy.signal import convolve2d

import nems
from nems import signal
from nems0.modules.nonlinearity import _dlog
from nems0.modules.fir import _offset_coefficients

log = logging.getLogger(__name__)


def make_contrast_signal(rec, name='contrast', source_name='stim', ms=500,
                         bins=None, bands=1, dlog=False, continuous=False,
                         normalize=False, percentile=50, ignore_zeros=True,
                         offset=1):
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
    offset : int
        Number of bins to offset the center of the kernel by.
        For default offset value of 1, the 'ones' portion of the kernel
        will be offset from the center by 1 bin.
        Ex: using a history=3, offset=1
                [0, 0, 0, 0, 1, 1, 1]
            versus history=3, offset=2
                [0, 0, 0, 0, 0, 0, 1, 1, 1]

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
    contrast = contrast_calculation(array, history, bands, offset, 'same')

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
                                               offset, 'valid')
        contrast[i][cropped_time:-cropped_time] = top_replacement

        # Replace bottom
        bottom_replacement = contrast_calculation(array[-reduced_bands:, :],
                                                  history, reduced_bands,
                                                  offset, 'valid')
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


def contrast_calculation(array, history, bands, offset, mode):
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
    offset : int
        The number of bins to offset the center of the kernel by
        (see docs for make_contrast_signal).
    mode : str
        See scipy.signal.conolve2d
        Generally, 'valid' to drop rows/columns that would require padding,
        'same' to pad those rows/columns with nans.

    filt : 2d ndarray
        Overrides the default filter construction.

    Returns
    -------
    contrast : 2d Ndarray


    '''
    array = copy.deepcopy(array)
    filt = np.concatenate((np.zeros([bands, history + (offset*2 - 1)]),
                           np.ones([bands, history])), axis=1)
    filt /= bands*history

    mn = convolve2d(array, filt, mode=mode, fillvalue=np.nan)

    var = convolve2d(array ** 2, filt, mode=mode, fillvalue=np.nan) - mn**2

    contrast = np.sqrt(var) / (mn*.99 + np.nanmax(mn)*0.01)

    return contrast


def sum_contrast(rec, name='ctpred', source_name='contrast', offsets=20,
                 IsReload=False, **context):
    if IsReload:
        return {}

    if type(offsets) is int:
        offsets = np.array([[offsets]])

    rec = rec.copy()
    fs = rec[source_name].fs
    def fn(x):
        summed = np.expand_dims(np.sum(x, axis=0), axis=0)
        if not np.all(offsets == 0):
            summed = _offset_coefficients(summed, offsets=offsets, fs=fs,
                                          pad_bins=False)
        summed /= np.nanmax(summed)

        return summed

    summed = rec[source_name].transform(fn, name)
    rec[name] = summed
    return {'rec': rec}


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
    '''
    Extremely simplified version of a contrast calculation.

    (WIP)
    Assumes:
        Stimulus on = high contrast, stimulus off = low contrast
        Binary contrast values - 1 for high, 0 for low

    '''
    new_rec = copy.deepcopy(rec)
    s = new_rec[source]
    if not isinstance(s, signal.RasterizedSignal):
        try:
            s = s.rasterize()
        except AttributeError:
            raise TypeError("signal with key {} was not a RasterizedSignal"
                            " and could not be converted to one."
                            .format(source))

    st_eps = nems0.epoch.epoch_names_matching(s.epochs, '^STIM_')
    pre_eps = nems0.epoch.epoch_names_matching(s.epochs, 'PreStimSilence')
    post_eps = nems0.epoch.epoch_names_matching(s.epochs, 'PostStimSilence')

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
