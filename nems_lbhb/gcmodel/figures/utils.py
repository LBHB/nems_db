import pandas as pd
import numpy as np

from nems_db.params import fitted_params_per_batch


# Copied from:
# https://stackoverflow.com/questions/7965743/
# how-can-i-set-the-aspect-ratio-in-matplotlib
def adjustFigAspect(fig, aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_valid_improvements(batch, model1, model2, threshold = 2.5):
    # TODO: threshold 2.5 works for removing outliers in correlation scatter
    #       and maximizes r, but need an unbiased way to pick this number.
    #       Otherwise basically just cherrypicked the cutoff to make
    #       correlation better.

    # NOTE: Also helps to do this for both gc and stp, then
    #       list(set(gc_cells) & set(stp_cells)) to get the intersection.

    df1 = fitted_params_per_batch(batch, model1, stats_keys=[])
    df2 = fitted_params_per_batch(batch, model2, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    nrows = len(df1.index.values.tolist())

    df1_cells = df1.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df2.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df1[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df2[c] = nan_series
            df2_nans += 1

    print("# missing cells: %d, %d" % (df1_nans, df2_nans))

    # Force same cellid order now that cols are filled in
    df1 = df1[cellids]; df2 = df2[cellids];
    ratio = df1.loc['meta--r_test'] / df2.loc['meta--r_test']

    valid_improvements = ratio.loc[ratio < threshold].loc[ratio > 1/threshold]

    return valid_improvements.index.values.tolist()
