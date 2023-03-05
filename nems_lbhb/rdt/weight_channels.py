from nems0.modules import weight_channels as wc


def gaussian(rec, mean, sd, n_chan_in, **kwargs):
    coefficients = wc.gaussian_coefficients(mean, sd, n_chan_in)
    fn = lambda x: coefficients @ x
    return [
        rec['bg'].transform(fn, 'bg_pred'),
        rec['fg'].transform(fn, 'fg_pred'),
    ]
