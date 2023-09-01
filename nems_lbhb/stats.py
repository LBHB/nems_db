import numpy as np

def j_mean_err(x, do_median=False, n=20):
    """
    Jackknifed estimate and SE of mean or media
    :param x: data, assume a 1d vector
    :param do_median: If True, compute jackknife estimate of median
                      rather than mean (default False)
    :param n: number of jackknifes (default 20)
    :return: m, se: Tuple estimate of mean and se
    """

    if len(x) == 1:
       return x, 0
    if n > len(x):
       n = len(x)

    mi = np.zeros(n)
    for ii in range(n):
        jackrange = np.ones_like(x, dtype=bool)
        jackrange[np.arange(ii, len(x), n)]=False

        if do_median:
            mi[ii] = np.nanmedian(x[jackrange])
        else:
            mi[ii] = np.nanmean(x[jackrange])

    m = np.nanmean(mi)
    se = np.nanstd(mi) * np.sqrt(n-1)

    return m, se

