import pandas as pd
import numpy as np

from nems_db.params import fitted_params_per_batch
import nems0.db as nd


def improved_cells_to_list(batch, gc, stp, LN, combined, se_filter=True,
                           LN_filter=False, good_ln=0.0, as_lists=True):
    '''
    Returns:
    --------
    either, neither, gc_cells, stp_cells, combined_cells : lists or pd series
        Respectively, cellids for which:
            1) There is a significant improvement over the LN model for
            at least one of the three nonlinear models.
            2) All cellids for which performance is above 2*SE for all models.
            3) The gain control model performs significantly better than
               the LN model.
            4) The STP model performs significantly better than the LN model.
            5) The combined model performs significantly better than
               the LN model.

    '''

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(batch, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          LN_filter)

    gc_test = df_r[gc][cellids]
    stp_test = df_r[stp][cellids]
    ln_test = df_r[LN][cellids]
    combined_test = df_r[combined][cellids]

    gc_vs_ln = gc_test.values - ln_test.values
    stp_vs_ln = stp_test.values - ln_test.values
    combined_vs_ln = combined_test.values - ln_test.values
    gc_vs_ln = gc_vs_ln.astype('float32')
    stp_vs_ln = stp_vs_ln.astype('float32')
    combined_vs_ln = combined_vs_ln.astype('float32')

    ff = (np.isfinite(gc_vs_ln) & np.isfinite(stp_vs_ln)
          & np.isfinite(ln_test.values) & np.isfinite(combined_test.values))
    gc_vs_ln = gc_vs_ln[ff]
    stp_vs_ln = stp_vs_ln[ff]
    ln_test = ln_test.values[ff]
    combined_test = combined_test.values[ff]

    ln_err = df_e[LN][cellids]
    gc_err = df_e[gc][cellids]
    stp_err = df_e[stp][cellids]
    combined_err = df_e[combined][cellids]

    ln_good = df_r[LN][cellids] > good_ln
    gc_imp = gc_vs_ln > (ln_err + gc_err)
    stp_imp = stp_vs_ln > (ln_err + stp_err)
    combined_imp = combined_vs_ln > (ln_err + combined_err)

#    # none of the nonlinear models helps
#    neither_better = (ln_good & np.logical_not(gc_imp) & np.logical_not(stp_imp)
#                      & np.logical_not(combined_imp))
    # all cellids
    all_good = ln_good

    # at least this model helps
    gc_better = ln_good & gc_imp
    stp_better = ln_good & stp_imp
    combined_better = ln_good & combined_imp

    # at least one model helps
    either_better = ln_good & (gc_imp | stp_imp | combined_imp)

    # to get exclusives: use e.g. gc_better - stp_better - combined_better

    either_cells = gc_test[either_better].index.values.tolist()
    #neither_cells = gc_test[neither_better].index.values.tolist()
    all_cells = gc_test[all_good].index.values.tolist()
    gc_cells = gc_test[gc_better].index.values.tolist()
    stp_cells = gc_test[stp_better].index.values.tolist()
    combined_cells = gc_test[combined_better].index.values.tolist()

    if as_lists:
        return either_cells, all_cells, gc_cells, stp_cells, combined_cells
    else:
        return (either_better, all_good, gc_better, stp_better,
                combined_better)


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


def get_dataframes(batch, gc, stp, LN, combined):
    df_r = nd.batch_comp(batch, [gc, stp, LN, combined],
                         stat='r_test')
    df_c = nd.batch_comp(batch, [gc, stp, LN, combined],
                         stat='r_ceiling')
    df_e = nd.batch_comp(batch, [gc, stp, LN, combined],
                         stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    # and sort indexes, double check that all are equal
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)
    df_c.dropna(axis=0, how='any', inplace=True)
    df_r.sort_index(inplace=True)
    df_e.sort_index(inplace=True)
    df_c.sort_index(inplace=True)
    if (not np.all(df_r.index.values == df_e.index.values)) \
            or (not np.all(df_r.index.values == df_c.index.values)):
        raise ValueError('index mismatch in dataframes')

    return df_r, df_c, df_e


def get_filtered_cellids(batch, gc, stp, LN, combined, se_filter=True,
                         LN_filter=False, as_lists=True):

    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    df_f = nd.batch_comp(batch, [gc, stp, LN, combined], stat='r_floor')
    df_f.dropna(axis=0, how='any', inplace=True)
    df_f.sort_index(inplace=True)
    if not np.all(df_f.index.values == df_r.index.values):
        raise ValueError('index mismatch in dataframes')

    cellids = df_r.index.values.tolist()
    gc_test, gc_se, gc_floor = [d[gc] for d in [df_r, df_e, df_f]]
    stp_test, stp_se, stp_floor = [d[stp] for d in [df_r, df_e, df_f]]
    ln_test, ln_se, ln_floor = [d[LN] for d in [df_r, df_e, df_f]]
    gc_stp_test, gc_stp_se, gc_stp_floor = [d[combined] for d in
                                            [df_r, df_e, df_f]]

    if se_filter:
        # Remove if performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (gc_test > gc_floor) &
                      (stp_test > stp_se*2) & (stp_test > stp_floor) &
                      (ln_test > ln_se*2) & (ln_test > ln_floor) &
                      (gc_stp_test > gc_stp_se*2) & (gc_stp_test > gc_stp_floor))
    else:
        # Set to series w/ all True, so none are skipped
        good_cells = (gc_test != np.nan)

    if LN_filter:
        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se) |
                     (gc_stp_test+gc_stp_se < ln_test-ln_se))
    else:
        # Set to series w/ all False, so none are skipped
        bad_cells = (gc_test == np.nan)

    keep = good_cells & ~bad_cells


    if as_lists:
        cellids = df_r[keep].index.values.tolist()
        under_chance = df_r[~good_cells].index.values.tolist()
        less_LN = df_r[bad_cells].index.values.tolist()
        return cellids, under_chance, less_LN
    else:
        return keep, ~good_cells, bad_cells



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


# copied from:
# https://stackoverflow.com/questions/11882393/ ...
#   matplotlib-disregard-outliers-when-plotting
def is_outlier(points, thresh=5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def drop_common_outliers(*arrays):
    common_outliers = np.logical_or.reduce([is_outlier(a) for a in arrays])
    return [a[~common_outliers] for a in arrays]
