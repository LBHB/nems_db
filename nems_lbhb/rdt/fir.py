from nems0.modules import fir


def basic(rec, coefficients):
    fn = lambda x: fir.per_channel(x, coefficients)
    return [
        rec['bg_pred'].transform(fn, 'bg_pred'),
        rec['fg_pred'].transform(fn, 'fg_pred'),
    ]
