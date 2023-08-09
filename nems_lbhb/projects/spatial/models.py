import logging
import numpy as np
import matplotlib.pyplot as plt

from nems.models.base import Model
from nems.layers import (
    WeightChannels, WeightChannelsGaussian, FiniteImpulseResponse, WeightChannelsMulti,
    WeightChannelsMultiGaussian,
    RectifiedLinear, DoubleExponential
)
from nems.visualization.model import plot_nl

log = logging.getLogger(__name__)

class LN_Tiled_STRF(Model):
    """Tiled Linear-nonlinear Spectro-Temporal Receptive Field model.

    Tiled so that the same weight channels can be applied to multiple
    dimensions of a stimulus. Permits a binaural model with fixed shape
    but scaled amplitude tuning for each ear.

    Contains the following layers:
    1. WeightChannelsMulti (speccount X rank X 1(Tilecount))
    2. FiniteImpulseResponse
    1. WeightChannels (Tilecount X 1)
    3. DoubleExponential, RectifiedLinear, or another static nonlinearity.

    Expects a single sound spectrogram as input, with shape (T, N), and a
    single recorded neural response as a target, with shape (T, 1), where
    T and N are the number of time bins and spectral channels, respectively.

    Parameters
    ----------
    time_bins : int.
     Number of "taps" in FIR filter. We have found that a 150-250ms filter
     typically sufficient for modeling A1 responses, or 15-25 bins for
     a 100 Hz sampling rate.
    channels : int.
     Number of spectral channels in spectrogram.
    rank : int; optional.
     Number of spectral weightings used as input to a reduced-rank filter.
     For example, `rank=1` indicates a frequency-time separable STRF.
     If unspecified, a full-rank filter will be used.
    gaussian : bool; default=True.
     If True, use gaussian functions (1 per `rank`) to parameterize
     spectral weightings. Unused if `rank is None`.
    nonlinearity : str; default='DoubleExponential'.
     Specifies which static nonlinearity to apply after the STRF.
     Default is the double exponential nonlinearity used in the paper
     cited above.
    nl_kwargs : dict; optional.
     Additional keyword arguments for the nonlinearity Layer, like
     `no_shift` or `no_offset` for `RectifiedLinear`.
    model_init_kwargs : dict; optional.
     Additional keyword arguments for `Model.__init__`, like `dtype`
     or `meta`.

    """

    def __init__(self, time_bins, channels, tile_count=2, rank=None,
                 gaussian=False, nonlinearity='DoubleExponential',
                 nl_kwargs=None, **model_init_kwargs):

        super().__init__(**model_init_kwargs)

        # Add STRF
        if rank is None:
            # Full-rank finite impulse response
            fir = FiniteImpulseResponse(shape=(time_bins, channels))
            self.add_layers(fir)
        else:
            wc_class = WeightChannelsGaussian if gaussian else WeightChannelsMulti
            wc = wc_class(shape=(channels, rank, 1))
            fir = FiniteImpulseResponse(shape=(time_bins, rank))
            wc2 = WeightChannels(shape=(tile_count, 1))
            self.add_layers(wc, fir, wc2)

        # Add static nonlinearity
        if nonlinearity in ['DoubleExponential', 'dexp', 'DEXP']:
            nl_class = DoubleExponential
        elif nonlinearity in ['RectifiedLinear', 'relu', 'ReLU']:
            nl_class = RectifiedLinear
        else:
            raise ValueError(
                f'Unrecognized nonlinearity for LN model:  {nonlinearity}.')

        if nl_kwargs is None: nl_kwargs = {}
        nonlinearity = nl_class(shape=(1,), **nl_kwargs)
        self.add_layers(nonlinearity)
        self.out_range = [[-1], [3]]

    @classmethod
    def from_data(cls, input, filter_duration, sampling_rate=1000, **kwargs):
        channels = input.shape[-1]
        time_bins = int(filter_duration / 1000 * sampling_rate)
        # TODO: modify initial parameters based on stimulus statistics?
        return LN_reconstruction(time_bins, channels, **kwargs)

    def fit_LBHB(self, X, Y, cost_function='nmse', fitter='tf'):
        """2-stage fit with freezing/unfreezing NL
        :param Y:
        :param cost_function:
        :param fitter:
        :return:
        """

        fitter_options = {'cost_function': cost_function,  # 'nmse'
                          'early_stopping_tolerance': 1e-3,
                          'validation_split': 0,
                          'learning_rate': 1e-2, 'epochs': 3000
                          }
        fitter_options2 = {'cost_function': cost_function,
                           'early_stopping_tolerance': 1e-4,
                           'validation_split': 0,
                           'learning_rate': 1e-3, 'epochs': 8000
                           }

        strf = self.sample_from_priors()
        log.info('Fit stage 1: w/o static output nonlinearity')
        strf.layers[-1].skip_nonlinearity()
        strf = strf.fit(input=X, target=Y, backend=fitter,
                        fitter_options=fitter_options, batch_size=None)
        strf.layers[-1].unskip_nonlinearity()
        log.info('Fit stage 2: with static output nonlinearity')
        strf = strf.fit(input=X, target=Y, backend=fitter,
                        verbose=0, fitter_options=fitter_options2, batch_size=None)

        ymin, ymax = Y.min(axis=tuple(range(Y.ndim - 1))), Y.max(axis=tuple(range(Y.ndim - 1)))
        ydelta = (ymax - ymin) * 0.1
        strf.out_range = [ymin - ydelta, ymax + ydelta]

        return strf

    def get_strf(self, channels=None):
        wc = self.layers[0].coefficients[:, :, 0]
        fir = self.layers[1].coefficients
        wc2 = self.layers[2].coefficients
        strf_ = wc @ fir.T
        strf1 = np.concatenate((strf_ * wc2[0, 0], strf_ * wc2[1, 0]), axis=0)

        return strf1

    def plot_strf(self, labels=None):
        strf1 = self.get_strf()

        channels_out = strf1.shape[-1]
        f, ax = plt.subplots(1, 2, figsize=(4, 2))

        mm = np.nanmax(abs(strf1))
        if self.fs is not None:
            extent = [0, strf1.shape[1] / self.fs, 0, strf1.shape[0]]
        else:
            extent = [0, strf1.shape[1], 0, strf1.shape[0]]
        ax[0].imshow(strf1, aspect='auto', interpolation='none', origin='lower',
                     cmap='bwr', vmin=-mm, vmax=mm, extent=extent)

        if self.fs is not None:
            ax[0].set_xlabel('Time lag (s)')
        else:
            ax[0].set_xlabel('Time lag (bins)')
        ax[0].set_ylabel('Input channel')

        ymin, ymax = self.out_range[0][0], self.out_range[1][0]
        plot_nl(self.layers[-1], range=[ymin, ymax], ax=ax[1])
        plt.tight_layout()

    # TODO
    # @module('LNSTRF')
    def from_keyword(keyword):
        # Return a list of module instances matching this pre-built Model?
        # That way these models can be used with kw system as well, e.g.
        # model = Model.from_keywords('LNSTRF')
        #
        # But would need the .from_keywords method to check for list vs single
        # module returned.
        pass

