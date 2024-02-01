
import logging
from os.path import basename, join

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, butter, sosfilt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nems0 import db, preprocessing
import nems0.epoch as ep
import nems0.preprocessing as preproc
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems.layers import WeightChannels, FIR, LevelShift, \
    DoubleExponential, RectifiedLinear, ConcatSignals, WeightChannelsGaussian
from nems import Model
from nems.tools import json
from nems.layers.base import Layer, Phi, Parameter
from nems0.recording import load_recording
import nems.visualization.model as nplt
from nems0.modules.nonlinearity import _dlog
from nems_lbhb.projects.freemoving.free_tools import stim_filt_hrtf, compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems0.epoch import epoch_names_matching
from nems0.metrics.api import r_floor
from nems0 import xforms
from nems.preprocessing import (indices_by_fraction, split_at_indices, JackknifeIterator)
import random
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib

log = logging.getLogger(__name__)
def load_free_data(siteid, cellid=None, batch=None, rasterfs=50, runclassid=132,
                   recache=False, dlc_chans=10, dlc_threshold=0.2, compute_position=False, remove_layer=False, layer_perms=None, **options):

    sitenum = int(siteid[3:6])
    if batch==348:
        loadkey = f'nostim.fs{rasterfs}.dlc.bin'
    else:
        loadkey = f'nostim.fs{rasterfs}.dlc'

    log.info(f"{siteid}/{batch}/{loadkey}")
    recording_uri = generate_recording_uri(batch=batch, loadkey=loadkey, cellid=siteid)
    rec = load_recording(recording_uri)

    try:
        df_siteinfo = get_depth_info(siteid=siteid)
        a1cellids = df_siteinfo.loc[(df_siteinfo['area']=='A1') | (df_siteinfo['area']=='BS') |
                                    (df_siteinfo['area']=='PEG')].index.to_list()
    except:
        df_siteinfo = db.pd_query(f"SELECT DISTINCT cellid,area FROM sCellFile WHERE cellid like '{siteid}%%'" +
                         " AND area in ('A1','PEG','AC','BS')")
        a1cellids = df_siteinfo['cellid'].to_list()

    if cellid is not None:
        a1cellids=[cellid]

    log.info('Imputing missing DLC values')
    rec = impute_multi(rec, sig='dlc', empty_values=np.nan, keep_dims=dlc_chans)['rec']
    #rec = dlc2dist(rec, smooth_win=2, keep_dims=dlc_chans)
    dlc_data = rec['dlc'][:, :]
    dlc_valid = np.sum(np.isfinite(dlc_data), axis=0, keepdims=True) > 0

    rec['dlc_valid'] = rec['dlc']._modified_copy(data=dlc_valid, chans=['dlc_valid'])
    log.info(f"DLC valid bins: {rec['dlc_valid'].as_continuous().sum()}/{rec['dlc_valid'].shape[1]}")

    if compute_position:
        rec2 = stim_filt_hrtf(rec, hrtf_format='az', smooth_win=2,
                              f_min=200, f_max=20000, channels=18)['rec']
        # get angle to each speaker and scale -1 to 1
        theta = rec2['disttheta'].as_continuous()[[1, 3], :] / 180
        chans = rec['dlc'].chans + ['th1','th2']
        d = np.concatenate((rec['dlc']._data, theta), axis=0)
        rec['dlc']=rec['dlc']._modified_copy(data=d, chans=chans)


    rec['resp'] = rec['resp'].rasterize()

    cid = [i for i, c in enumerate(rec['resp'].chans) if c in a1cellids]
    cellids = [c for i, c in enumerate(rec['resp'].chans) if c in a1cellids]

    rec['resp'] = rec['resp'].extract_channels(cellids)
    rec.meta['siteid'] = siteid
    rec.meta['cellids'] = cellids

    try:
        rec.meta['depth'] = np.array([df_siteinfo.loc[c, 'depth'] for c in cellids])
        rec.meta['layer'] = np.array([df_siteinfo.loc[c, 'layer'] for c in cellids])
        #rec.meta['sw'] = np.array([df_siteinfo.loc[c, 'sw'] for c in cellids])
        rec.meta['sw'] = np.ones(len(cellids)) * 1
    except:
        rec.meta['depth'] = np.array([float(c.split("-")[-2]) for c in cellids])
        rec.meta['sw'] = np.ones(len(cellids)) * 100

    # get dist signals from dlc data
    rec = dlc2dist(rec, ref_x0y0=None, smooth_win=5, norm=False, verbose=False)

    # compute PSTH for repeated stimuli
    epoch_regex = "^STIM_"
    rec['psth'] = preprocessing.generate_average_sig(rec['resp'], 'psth', epoch_regex=epoch_regex)

    # use diff to predict dist
    rec['diff'] = rec['resp']._modified_copy(data = rec['resp']._data-rec['psth']._data)

    recs = []
    if remove_layer:
        rec_rl = rec.copy()
        lyrcellids = [cell for ci, cell in enumerate(rec_rl['resp'].chans) if rec_rl.meta['layer'][ci] == remove_layer]
        nonlyrcellids = [cell for ci, cell in enumerate(rec_rl['resp'].chans) if rec_rl.meta['layer'][ci] != remove_layer]
        rec_rl['resp'] = rec_rl['resp'].extract_channels(nonlyrcellids)
        rec_rl['diff'] = rec_rl['diff'].extract_channels(nonlyrcellids)
        rec_rl.meta['rlyr'] = remove_layer
        recs.append(rec_rl)
        for i in range(layer_perms):
            rec_perm = rec.copy()
            random.seed()
            rnd_indexes = random.sample(range(len(rec_perm['resp'].chans)), len(nonlyrcellids))
            rnd_cellids = [cell for ci, cell in enumerate(rec_perm['resp'].chans) if ci in rnd_indexes]
            rec_perm['resp'] = rec_perm['resp'].extract_channels(rnd_cellids)
            rec_perm['diff'] = rec_perm['diff'].extract_channels(rnd_cellids)
            rec_perm.meta['rlyr'] = 'rnd'+str(i)
            recs.append(rec_perm)

    else:
        recs.append(rec)
        rec.meta['rlyr'] = remove_layer

    return recs

def spatial_tc_2d(rec, occ_threshold=0.5, bin_width=20):
    # assume rasterized signals
    # check and see if normalization is applied. Function currently only works for unnormalized rasterized signals
    norm = rec['resp'].normalization
    if norm == 'minmax':
        resp = (rec['resp']._data.copy()*rec['resp'].norm_gain) + rec['resp'].norm_baseline
    else:
        resp = rec['resp']._data.copy()
    # assume DLC sig is ['dlc'] and index 2,3 corresponds to headpost x/y
    x_ind = rec['dlc'].chans.index('front_x')
    y_ind = rec['dlc'].chans.index('front_y')
    x = rec['dlc'][x_ind]
    y = rec['dlc'][y_ind]

    # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
    # x_range = np.ptp(x[~np.isnan(x)])
    # y_range = np.ptp(y[~np.isnan(y)])
    # xbin_num = int(np.round(x_range/bin_width))
    # ybin_num = int(np.round(y_range/bin_width))
    # xbins = np.linspace(np.nanmin(x), np.nanmax(x), xbin_num + 1)
    # ybins = np.linspace(np.nanmin(y), np.nanmax(y), ybin_num + 1)
    # generate linearly spaced bins for 2d histogram
    xbins = np.arange(np.nanmin(x), np.nanmax(x), bin_width)
    ybins = np.arange(np.nanmin(y), np.nanmax(y), bin_width)

    # generate occupancy histogram - count of x/y occupancy samples - need to convert to time to get spike rate later
    occupancy, x_edges, y_edges = np.histogram2d(x, y, [xbins, ybins],)
    tc = {}
    for cellindex, cellid in enumerate(rec['resp'].chans):
        # nasty way to unrasterize spikes and return a list of x/y positions for each spike
        n_x_loc = [item for sublist in [[xpos]*int(spk_cnt) for xpos, spk_cnt in zip(x, resp[cellindex, :]) if spk_cnt !=0] for item in sublist]
        n_y_loc = [item for sublist in [[ypos]*int(spk_cnt) for ypos, spk_cnt in zip(y, resp[cellindex, :]) if spk_cnt !=0] for item in sublist]
        # create a 2d histogram of spike locations
        spk_hist, x_edges, y_edges = np.histogram2d(n_x_loc, n_y_loc, [xbins, ybins],)
        # divide the spike locations by the amount of time spent in each bin to get firing rate
        rate_bin = spk_hist/(occupancy/rec.meta['rasterfs'])
        # add the tc for each cell to the dictionary with key == cellid, also transpose xy to yx for visualization purposes
        tc[cellid] = rate_bin

    # threshold occupancy so only bins with at least half a second of data are included
    for cellid in rec['resp'].chans:
        tc[cellid][np.where(occupancy < occ_threshold*rec.meta['rasterfs'])] = np.nan

    # get center position of each bin to be used in assigning x/y location
    x_cent = xbins[0:-1] + np.diff(xbins)/2
    y_cent = ybins[0:-1] + np.diff(ybins)/2
    # make feature dictionary x/y with positions
    xy = {'x':x_cent, 'y':y_cent}
    xy_edges = {'x':x_edges, 'y':y_edges}

    return tc, xy, xy_edges

def state_tc_2d(resp=gain, signal=dlc, resp_chans=None, sig_chans=['front_x', 'front_y',], rasterfs=None, occ_threshold=0.5, bin_width =1/30):
    # assume rasterized signals
    # check and see if normalization is applied. Function currently only works for unnormalized rasterized signals

    # assume DLC sig is ['dlc'] and index 2,3 corresponds to headpost x/y
    x_ind = sig_chans.index('front_x')
    y_ind = sig_chans.index('front_y')
    x = signal[x_ind]
    y = signal[y_ind]

    xbins = np.arange(np.nanmin(x), np.nanmax(x), bin_width)
    ybins = np.arange(np.nanmin(y), np.nanmax(y), bin_width)

    # generate occupancy histogram - count of x/y occupancy samples - need to convert to time to get spike rate later
    occupancy, x_edges, y_edges = np.histogram2d(x, y, [xbins, ybins],)
    tc = {}
    if resp_chans == None:
        resp_chans = np.range(len(resp[:, 0]))
        
    for cellindex, cellid in enumerate(resp_chans):
        # nasty way to unrasterize spikes and return a list of x/y positions for each spike
        n_x_loc = [item for sublist in [[xpos]*int(spk_cnt) for xpos, spk_cnt in zip(x, resp[cellindex, :]) if spk_cnt !=0] for item in sublist]
        n_y_loc = [item for sublist in [[ypos]*int(spk_cnt) for ypos, spk_cnt in zip(y, resp[cellindex, :]) if spk_cnt !=0] for item in sublist]
        # create a 2d histogram of spike locations
        spk_hist, x_edges, y_edges = np.histogram2d(n_x_loc, n_y_loc, [xbins, ybins],)
        # divide the spike locations by the amount of time spent in each bin to get firing rate
        rate_bin = spk_hist/(occupancy/rasterfs)
        # add the tc for each cell to the dictionary with key == cellid, also transpose xy to yx for visualization purposes
        tc[cellid] = rate_bin

    # threshold occupancy so only bins with at least half a second of data are included
    for cellid in resp_chans:
        tc[cellid][np.where(occupancy < occ_threshold*rasterfs)] = np.nan

    # get center position of each bin to be used in assigning x/y location
    x_cent = xbins[0:-1] + np.diff(xbins)/2
    y_cent = ybins[0:-1] + np.diff(ybins)/2
    # make feature dictionary x/y with positions
    xy = {'x':x_cent, 'y':y_cent}
    xy_edges = {'x':x_edges, 'y':y_edges}

    return tc, xy, xy_edges

def spatial_tc_jackknifed(rec, jk_num=10, jk_type='epoch', occ_threshold=0.25, pix_bins=20, binnums=[30, 20], zscore=True, joint=False):
    """
    jackknife data and compute 2d tuning curves fore each cell.
    Returns a dictionary of cell tuning curves with key == cellid for each jackknife.

    :param rec:
    :param jk_num:
    :param jk_type:
    :param occ_threshold:
    :param pix_bins:
    :return:
    """
    # assume rasterized signals
    x = rec['dlc'][2]
    y = rec['dlc'][3]

    # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
    x_range = np.ptp(x[~np.isnan(x)])
    y_range = np.ptp(y[~np.isnan(y)])

    if binnums:
        # set bins if user specified
        xbin_num = binnums[0]
        ybin_num = binnums[1]

    else:
        # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
        x_range = np.ptp(x[~np.isnan(x)])
        y_range = np.ptp(y[~np.isnan(y)])
        xbin_num = int(np.round(x_range / pix_bins))
        ybin_num = int(np.round(y_range / pix_bins))

    # generate linearly spaced bins for 2d histogram
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), xbin_num + 1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), ybin_num + 1)

    # random data splits
    tcs = []
    for jack in range(jk_num):
        # ask about difference in mask by epoch vs mask by time
        if jk_type == 'time':
            jrec = rec.jackknife_mask_by_time(jk_num, jack, invert=True)
        elif jk_type == 'epoch':
            jrec = rec.jackknife_mask_by_epoch(jk_num, jack, 'TRIAL', invert=True)
        jrec = jrec.apply_mask()
        # assume DLC sig is ['dlc'] and index 2,3 corresponds to headpost x/y
        x = jrec['dlc'][2]
        y = jrec['dlc'][3]

        # generate occupancy histogram - count of x/y occupancy samples - need to convert to time to get spike rate later
        occupancy, x_edges, y_edges = np.histogram2d(x, y, [xbins, ybins],)
        tc = {}
        for cellindex, cellid in enumerate(rec['resp'].chans):
            # nasty way to unrasterize spikes and return a list of x/y positions for each spike
            n_x_loc = [item for sublist in [[xpos]*int(spk_cnt) for xpos, spk_cnt in zip(x, jrec['resp'][cellindex]) if spk_cnt !=0] for item in sublist]
            n_y_loc = [item for sublist in [[ypos]*int(spk_cnt) for ypos, spk_cnt in zip(y, jrec['resp'][cellindex]) if spk_cnt !=0] for item in sublist]
            # create a 2d histogram of spike locations
            spk_hist, x_edges, y_edges = np.histogram2d(n_x_loc, n_y_loc, [xbins, ybins],)
            # divide the spike locations by the amount of time spent in each bin to get firing rate
            rate_bin = spk_hist/(occupancy/rec.meta['rasterfs'])
            if zscore == True:
                cmean, cstd = np.nanmean(rec['resp'][cellindex]), np.nanstd(rec['resp'][cellindex], ddof=1)
                rate_bin = (rate_bin-cmean)/cstd
            # add the tc for each cell to the dictionary with key == cellid, also transpose xy to yx for visualization purposes
            tc[cellid] = rate_bin



        # threshold occupancy so only bins with at least half a second of data are included
        for cellid in rec['resp'].chans:
            tc[cellid][np.where(occupancy < occ_threshold*rec.meta['rasterfs'])] = np.nan
        tcs.append(tc)

    debug = True
    joint = False
    # determine consistency of each jackknifed tc
    for cellid in rec['resp'].chans:
        cell_tcs = np.concatenate([tc[cellid][:, :, np.newaxis] for tc in tcs], axis=2)

        if joint == True:
            joint_mask = np.isnan(np.sum(cell_tcs, axis=2))
            for i in range(cell_tcs.shape[2]):
                cell_tcs[:,:, i][joint_mask] = np.nan

        bin_std = np.nanstd(cell_tcs, axis=2)
        bin_mean = np.nanmean(cell_tcs, axis=2)

        if debug == True:
            f, ax = plt.subplot_mosaic([[str(i) for i in range(cell_tcs.shape[2])],['mean',]*int(cell_tcs.shape[2]/2)+['std',]*int(cell_tcs.shape[2]/2)])
            for i in range(cell_tcs.shape[2]):
                ax[str(i)].imshow(cell_tcs[:, :, i])
            ax['mean'].imshow(bin_mean)
            ax['std'].imshow(bin_std)


    # get center position of each bin to be used in assigning x/y location
    x_cent = xbins[0:-1] + np.diff(xbins)/2
    y_cent = ybins[0:-1] + np.diff(ybins)/2

    # make feature dictionary x/y with positions
    xy = {'x':x_cent, 'y':y_cent}
    xy_edges = {'x':x_edges, 'y':y_edges}

    return tcs, xy, xy_edges


def dlc_to_tcpos(rec, xy):
    # get dlc data
    x = rec['dlc'][2]
    y = rec['dlc'][3]

    # find nearest binned values
    xnew = np.array([xy['x'][(np.abs(xy['x']-i)).argmin()] if ~np.isnan(i) else np.nan for i in x])
    ynew = np.array([xy['y'][(np.abs(xy['y']-i)).argmin()] if ~np.isnan(i) else np.nan for i in y])

    return xnew, ynew

def tc_stability(rec, tc, xy, occ_threshold=0.5, pix_bins=20, shuffle_dlc=False):

    ### for each bin split occupancy in half, then generate a firing rate hist for each data split, then calculate correlation between both halves of data ###

    # step 1, get hist bin for each dlc data point
    # get dlc data
    if shuffle_dlc:
        np.random.seed()
        x = rec['dlc'][2].copy()
        x.setflags(write=1)
        np.random.shuffle(x)
        y = rec['dlc'][3].copy()
        y.setflags(write=1)
        np.random.shuffle(y)
    else:
        x = rec['dlc'][2]
        y = rec['dlc'][3]

    # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
    x_range = np.ptp(x[~np.isnan(x)])
    y_range = np.ptp(y[~np.isnan(y)])

    xbin_num = int(np.round(x_range/pix_bins))
    ybin_num = int(np.round(y_range/pix_bins))

    # find nearest binned values
    xbinind = [int((np.abs(xy['x']-i)).argmin()) if ~np.isnan(i) else np.nan for i in x]
    ybinind = [int((np.abs(xy['y']-i)).argmin()) if ~np.isnan(i) else np.nan for i in y]


    bin_occupancy = np.zeros((xbin_num, ybin_num))
    for xi,yi in zip(xbinind, ybinind):
        if ~np.isnan(xi) and ~np.isnan(yi):
            bin_occupancy[xi, yi] = bin_occupancy[xi, yi] + 1
        else:
            continue

    mask1 = {}
    mask2 = {}
    for key in set(zip(xbinind, ybinind)):
        if ~np.isnan(key[0]) and ~np.isnan(key[1]):
            mask1[key] = []
            mask2[key] = []
        else:
            continue

    # split data in half. create empty matrix. for each data point, add 1 to count in each bin of empty matrix and append index to xy dict for mask1.
    # once half_count for that bin is half of total occupancy, append index to second mask for that bin.
    half_count = np.zeros((xbin_num, ybin_num))
    for i, key in enumerate(zip(xbinind, ybinind)):
        if ~np.isnan(key[0]) and ~np.isnan(key[1]):
            if half_count[key[0], key[1]] < bin_occupancy[key[0], key[1]]/2:
                half_count[key[0], key[1]] = half_count[key[0], key[1]] + 1
                mask1[key].append(i)
            else:
                mask2[key].append(i)
        else:
            continue

    # threshold occupancy
    for key in set(zip(xbinind, ybinind)):
        # each bin must have at least 0.5 second of data
        if ~np.isnan(key[0]) and ~np.isnan(key[1]):
            if bin_occupancy[key[0], key[1]] < rec.meta['rasterfs']*occ_threshold:
                mask1[key] = []
                mask2[key] = []
        else:
            continue

    tc1 = {}
    tc2 = {}
    tc_12cc = {}
    for i, cellname in enumerate(rec['resp'].chans):
        tmptc1 = np.full((xbin_num, ybin_num), np.nan)
        tmptc2 = np.full((xbin_num, ybin_num), np.nan)
        for tmpbin in list(mask1.keys()):
            tmptc1[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][mask1[tmpbin]])/(len(mask1[tmpbin])/rec.meta['rasterfs'])
            tmptc2[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][mask2[tmpbin]])/(len(mask2[tmpbin])/rec.meta['rasterfs'])
        tc1[cellname] = tmptc1
        tc2[cellname] = tmptc2
        tc_12cc[cellname] = np.corrcoef(tmptc1[~np.isnan(tmptc1)].flatten(), tmptc2[~np.isnan(tmptc2)].flatten())[0, 1]

    return tc_12cc, tc1, tc2

def cell_spatial_info(rec, xy, pix_bins=20, shuffle_dlc=False):
    """
    Spatial information (SI) for each unit was calculated from the linearized location data and firing rate following (Skaggs et al. 1992):
    SI=∑i=1NPiXirlog2Xir, where Pi is the probability of finding the animal in bin i, Xi is the sum of the firing rates observed when the animal was found in bin i - ammended to be the mean firing rate in bin i,
    r is the mean spiking activity of the neuron, and N is the number of bins of the linearized trajectory (104)
    :return:
    """

    # step 1, get hist bin for each dlc data point
    # get dlc data
    if shuffle_dlc:
        np.random.seed()
        x = rec['dlc'][2].copy()
        x.setflags(write=1)
        np.random.shuffle(x)
        y = rec['dlc'][3].copy()
        y.setflags(write=1)
        np.random.shuffle(y)
    else:
        x = rec['dlc'][2]
        y = rec['dlc'][3]

    # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
    x_range = np.ptp(x[~np.isnan(x)])
    y_range = np.ptp(y[~np.isnan(y)])

    xbin_num = int(np.round(x_range/pix_bins))
    ybin_num = int(np.round(y_range/pix_bins))

    # find nearest binned values
    xbinind = [int((np.abs(xy['x']-i)).argmin()) if ~np.isnan(i) else np.nan for i in x]
    ybinind = [int((np.abs(xy['y']-i)).argmin()) if ~np.isnan(i) else np.nan for i in y]


    bin_occupancy = np.zeros((xbin_num, ybin_num))
    for xi,yi in zip(xbinind, ybinind):
        if ~np.isnan(xi) and ~np.isnan(yi):
            bin_occupancy[xi, yi] = bin_occupancy[xi, yi] + 1
        else:
            continue

    # get bin probability distribution
    bin_probability = bin_occupancy/np.nansum(bin_occupancy)
    # for each bin of histogram generate a mask of index values
    bin_indexes = {}
    for key in set(zip(xbinind, ybinind)):
        if ~np.isnan(key[0]) and ~np.isnan(key[1]):
            bin_indexes[key] = []
        else:
            continue

    for i, key in enumerate(zip(xbinind, ybinind)):
        if ~np.isnan(key[0]) and ~np.isnan(key[1]):
            bin_indexes[key].append(i)
        else:
            continue

    # generate a tuning curve for each cell
    tuning_curve = {}
    for i, cellname in enumerate(rec['resp'].chans):
        tc_temp = np.full((xbin_num, ybin_num), np.nan)
        for tmpbin in list(bin_indexes.keys()):
            tc_temp[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][bin_indexes[tmpbin]])/(len(bin_indexes[tmpbin])/rec.meta['rasterfs'])
        tuning_curve[cellname] = tc_temp

    # calculate spatial information
    spatial_information = {}
    for cellname in rec['resp'].chans:
        tc = tuning_curve[cellname]
        mean_firing_rate = np.nanmean(tc.flatten())
        spatial_information[cellname] = np.nansum(bin_probability*tc/mean_firing_rate*np.log2(tc/mean_firing_rate))

    return spatial_information

def decode2d(rec, tuning_curves, xy):
    # implement quick bayesian decoding as implemented in pynapple decode2d function - modified to work with baphy
    from scipy.ndimage import gaussian_filter1d
    if set(rec['resp'].chans).issubset(list(tuning_curves.keys())):
        cellids = rec['resp'].chans
    else:
        raise ValueError("cellids in rec are not all included in tuning curves")

    # calculate occupancy
    # create dict of x and y bin start positions given xy dict from spatial_tc_2d
    binsxy = {}
    for i in xy.keys():
        diff = np.diff(xy[i])
        bins = xy[i][:-1] - diff / 2
        bins = np.hstack((bins, [bins[-1] + diff[-1], bins[-1] + 2 * diff[-1]]))
        binsxy[i] = bins
    # same xy features as spatial_tc_2d
    x = rec['dlc'][2][:]
    y = rec['dlc'][3][:]
    occupancy, x_edges, y_edges = np.histogram2d(x, y, [binsxy['x'], binsxy['y']])

    # flatten occupancy
    occupancy = occupancy.flatten()

    # transform tc dictionary into pure numpy array, reshape, and transpose.
    tc = np.array([tuning_curves[i] for i in cellids])
    tc = tc.reshape(tc.shape[0], np.prod(tc.shape[1:]))
    tc = tc.T

    # memory intensive: SigKill9 if window is too large...sliding window over
    resp = gaussian_filter1d(rec['resp'][:, :], axis=1, sigma=3)
    winsize = 2000
    if len(resp[0, :]) > winsize:
        win_steps = np.arange(winsize, len(resp[0, :]), winsize)
        win_resp = np.split(resp[:, :], win_steps, axis=1)
    else:
        win_resp = [resp[:,:]]
    decoded_pos_list = []
    pdist_list = []

    for resp in win_resp:
        # get binned neural data - assumes pre-rasterized
        resp = resp.T

        # bin size
        bin_size = 1/rec['resp'].fs

        # some bayesian probability density stuff...need to go through this more

        # e^(-bin_size*sum of firing rate for all cells in each bin)
        p1 = np.exp(-bin_size * np.nansum(tc, 1))

        # probability that animal is in a given bin based on time spent in that bin by time spent in all bins
        p2 = occupancy / occupancy.sum()

        # create new neuron spike bins, by number of spatial bins, by number of neurons array - this could get huge?
        ct2 = np.tile(resp[:, np.newaxis, :], (1, tc.shape[0], 1))

        # unsure what exactly this is doing? each
        p3 = np.nanprod(tc**ct2, -1)

        p = p1 * p2 * p3
        # normalize probability to 1 for each time point
        p = p / p.sum(axis=1)[:, np.newaxis]

        # get the maximum probability
        idxmax = np.argmax(p, 1)
        p = p.reshape(p.shape[0], len(xy['x']), len(xy['y']))
        idxmax2d = np.unravel_index(idxmax, (len(xy['x']), len(xy['y'])))

        decoded_pos = np.vstack((xy['x'][idxmax2d[0]], xy['y'][idxmax2d[1]])).T

        decoded_pos_list.append(decoded_pos)
        pdist_list.append(p)

    decoded_pos_all = np.concatenate(decoded_pos_list)
    pdist_all = np.concatenate(pdist_list)

    return decoded_pos_all, pdist_all

def decode2d_forloop(rec, tuning_curves, xy):
    #TODO finish the function...doesn't work. Something with the math in tempProd = np.nansum(np.log(np.tile(tc[pos_bin, :], (tbins, 1))**resp), axis=1)) is not working as expected.

    # implement quick bayesian decoding as implemented in pynapple decode2d function - modified to work with baphy
    if set(rec['resp'].chans).issubset(list(tuning_curves.keys())):
        cellids = rec['resp'].chans
    else:
        raise ValueError("cellids in rec are not all included in tuning curves")

    # calculate occupancy
    # create dict of x and y bin start positions given xy dict from spatial_tc_2d
    binsxy = {}
    for i in xy.keys():
        diff = np.diff(xy[i])
        bins = xy[i][:-1] - diff / 2
        bins = np.hstack((bins, [bins[-1] + diff[-1], bins[-1] + 2 * diff[-1]]))
        binsxy[i] = bins
    # same xy features as spatial_tc_2d
    x = rec['dlc'][2][:]
    y = rec['dlc'][3][:]
    occupancy, x_edges, y_edges = np.histogram2d(x, y, [binsxy['x'], binsxy['y']])

    # flatten occupancy for some reason? tbd...
    occupancy = occupancy.flatten()

    # transform tc dictionary into pure numpy array, reshape, and transpose.
    tc = np.array([tuning_curves[i] for i in cellids])
    tc = tc.reshape(tc.shape[0], np.prod(tc.shape[1:]))
    tc = tc.T

    # get binned neural data - assumes pre-rasterized
    resp = rec['resp'][:, :2000].T

    # Try vandermeer implementation of 1-step bayes - at least it includes formula and logic
    # https://rcweb.dartmouth.edu/~mvdm/wiki/doku.php?id=analysis:nsb2016:week10

    # Assuming a poison distribution for spike rate the probability for a given number of spikes given location x is as follows
    # P(ni|x) = ((τfi(x))^ni/ni!)*e^−τfi(x)
    # where fi(x) is the average firing rate of neuron i over all spatial positions x. ni is the number of spikes in the current time bin.
    # τ is the size of the time window used.

    # if we take another simple assumption that the spike count probability for each neuron are independent the probabiliy for
    # the population can be taken by multiplying probabilities together ∏ is prod for i=1 to N(number of neurons)
    #P(n|x) = N∏i=1 ((τfi(x))^ni)/ni!*e−τfi(x)

    # this can be combined with Bayes theorum and rearranged to give prob x given array of neurons N
    # P(x|n) =C(τ,n)*P(x)*(N∏i=1 fi(x)^ni)*e(−τN∑i=1fi(x))

    # number of spatial bins
    nBins = len(occupancy)

    # uniform occupancy - scaling factor?
    occUniform = np.tile(1/nBins, [nBins, 1])

    # bin size
    bin_size = 1/rec['resp'].fs

    # decoder alg
    # num time bins
    tbins = len(resp[:, 0])

    #initialize mat of time length by number of spatial bins
    p = np.empty((tbins, nBins))
    p[:] = np.nan
    for pos_bin in range(nBins):
        # take mean firing rate of every neuron as given by tc, for every position (pos_bin) in the histogram, and then raise it to the power of firing rate at every given timepoint
        tempProd = np.nansum(np.log(np.tile(tc[pos_bin, :], (tbins, 1))**resp), axis=1)
        #
        tempSum = np.exp(-bin_size*np.nansum(tc[pos_bin, :]))
        p[:, pos_bin] = np.exp(tempProd)*tempSum*occUniform[pos_bin]
    p = p/np.sum(p, axis=1)[:, np.newaxis]

    maxindex = np.argmax(p, axis=0)

    # reshape p dist back into 2D
    p = p.reshape(len(p[:, 0], len(xy['x']), len(xy['y'])))
    return p

def dist_occ_hist(rec, epochs, signal='dist', feature=0, bins = 40):

    # extract epochs
    dist_eps = rec[signal].extract_epoch(epochs)
    # find min and max dist value
    min_dist = np.nanmin(dist_eps[:, feature, :])
    max_dist = np.nanmax(dist_eps[:, feature, :])

    # generate bin edges
    distbins = np.linspace(min_dist, max_dist, bins + 1)

    # flatten distance and remove nans
    all_dist = dist_eps[:, feature, :].flatten()[~np.isnan(dist_eps[:, feature, :].flatten())]

    # create occupancy histogram
    occ_hist, edges = np.histogram(all_dist, bins=distbins)

    # find bin centers
    distbin_cent = distbins[0:-1] + np.diff(distbins)/2

    return occ_hist, distbin_cent

# for each cell generate a dist feature/firing rate tuning curve
def dist_tc(rec, epochs, hs_bins = 40, signal='dist', feature=0, ds_bins = 20, low_trial_drop=10):
    # extract epochs from rec for both neurons and distance
    tarlickeps = rec['resp'].extract_epoch(epochs)
    dist_eps = rec[signal].extract_epoch(epochs)
    dist_eps = dist_eps[:, feature, :]

    # calculate bin centers
    # find min and max dist value for all trials
    min_dist = np.nanmin(dist_eps)
    max_dist = np.nanmax(dist_eps)
    # generate bin edges
    # distbins = np.linspace(min_dist, max_dist,  hs_bins + 1)
    distbins = np.linspace(max_dist, min_dist, hs_bins + 1)
    # find bin centers
    bin_centers = distbins[0:-1] + np.diff(distbins) / 2

    # create an spike rate tuning curve for each cell given occupancy histogram and bin centers for histogram
    trial_dist_tc = {}
    for spk, spk_name in enumerate(rec['resp'].chans):
        # create temporary array of trials by bins to be filled with firing rates in each bin on each trial for current neuron
        tmp_dist_tc = np.full((len(epochs), len(bin_centers)), np.nan)
        for trial in range(len(epochs)):
            # get all distance values for a single trial
            trial_dists = dist_eps[trial, :][~np.isnan(dist_eps[trial, :])]
            # for each distance value, find the bin center from the dist histogram that it is closest to
            trial_dist_index = []
            for dist in dist_eps[trial, :]:
                # np.argmin() of nan returns 0 index - to avoid, check if each dist value is nan
                if np.isnan(dist):
                    trial_dist_index.append(np.nan)
                else:
                    trial_dist_index.append(np.abs(bin_centers - dist).argmin())
            trial_dist_index = np.array(trial_dist_index)
            # for each bin, create a mask of indexes where the animal was in that bin
            temp_trial_tc = np.full(len(bin_centers), np.nan)
            for bin in range(len(bin_centers)):
                # create bin mask = should be true for all values where animal was in current bin
                bin_mask = trial_dist_index == bin
                # take sum of all spikes where animal was in bin using bin mask and then divide by time spent in that bin to get spks/sec
                temp_trial_tc[bin] = np.sum(tarlickeps[trial, spk, :][bin_mask])/(np.sum(bin_mask)/rec.meta['rasterfs'])
            tmp_dist_tc[trial, :] = temp_trial_tc
        trial_dist_tc[spk_name] = tmp_dist_tc

    # interpolate distance over 1d tuning curves and then downsample distance
    low_dist_tc = {}
    # create uniformly sampled lower distance bins
    # low_dist = np.linspace(min_dist, max_dist, ds_bins + 1)
    low_dist = np.linspace(max_dist, min_dist, ds_bins + 1)
    low_dist_bin_centers =  low_dist[0:-1] + np.diff(low_dist) / 2
    # for each spike name, grab high sampled dist tuning curve and then go through each trial and interpolate spk and dist rate
    # get values at new low sampled distances
    for spk, spk_name in enumerate(rec['resp'].chans):
        # get single cell all trial dist tuning curve at high rate
        all_trial_tc = trial_dist_tc[spk_name]
        # generate temp low sampling rate tc for each trial
        smooth_trial_tc = np.full((len(all_trial_tc[:, 0]), ds_bins), np.nan)
        for trial in range(len(all_trial_tc[:, 0])):
            # get single trial spk distance histogram
            trial_tc = all_trial_tc[trial, :]
            # using high sampling rate bins where animal was present (non-nan) get distances (bin_centers, and firing rate in that bin and interpolate)
            f = interp1d(bin_centers[~np.isnan(trial_tc)], trial_tc[~np.isnan(trial_tc)], kind='linear', bounds_error=False)
            # using interp function get interpolated firing rates at desired lower sampling rate bin centers.
            smooth_trial_tc[trial, :] = f(low_dist_bin_centers)
        # append the downsampled interpolated data for each unit to dictionary
        low_dist_tc[spk_name] = smooth_trial_tc

    if low_trial_drop:
        low_trial_mask = np.sum(~np.isnan(trial_dist_tc[rec['resp'].chans[0]]), axis=0) > len(
            trial_dist_tc[rec['resp'].chans[0]][:, 0]) * low_trial_drop
        low_dist_low_trial_mask = np.sum(~np.isnan(low_dist_tc[rec['resp'].chans[0]]), axis=0) > len(
            low_dist_tc[rec['resp'].chans[0]][:, 0]) * low_trial_drop
        bin_centers = bin_centers[low_trial_mask]
        low_dist_bin_centers = low_dist_bin_centers[low_dist_low_trial_mask]
        for spk_name in rec['resp'].chans:
            trial_dist_tc[spk_name] = trial_dist_tc[spk_name][:, low_trial_mask]
            low_dist_tc[spk_name] = low_dist_tc[spk_name][:, low_dist_low_trial_mask]

    # return both high sampling rate tuning curves and low sampling rate tuning curves and distance values
    return trial_dist_tc, bin_centers, low_dist_tc, low_dist_bin_centers

def points_within_radius(points_list, target, radius):
    import math
    target_x, target_y = target
    result = []

    for point in points_list:
        point_x, point_y = point

        distance = math.sqrt((target_x - point_x) ** 2 + (target_y - point_y) ** 2)

        if distance <= radius:
            result.append(point)

    return result

# make function to create new signals for dlc values
# def dlc_within_radius(rec, target='Trial', **kwargs):
#
#     # check for kwargs assign defaults
#     if 'ref' in kwargs.keys():
#         ref = kwargs['ref']
#     else:
#         ref = 'start'
#     if 'radius' in kwargs.keys():
#         radius = kwargs['radius']
#     else:
#         radius = 'auto'
#     if 'signal' in kwargs.keys():
#         signal = kwargs['signal']
#     else:
#         signal = 'dlc'
#     if 'chan' in kwargs.keys():
#         chans = kwargs['chan']
#         if len(chans) > 2:
#             raise ValueError("Functions expects only two points (x/y). len(Chan) > 2")
#     else:
#         chans = ['front_x', 'front_y']
#
#     dlc_chan = rec[f"{signal}"].extract_chans(chans)
#
#
#     # grab the location of the target based on epochs
#     if type(target) == str:
#         print('Using epochs for target location.')
#         e = rec[f"{signal}"].epochs
#         tareps = e.loc[f"{target}"]
#         for ep in tareps:
#             if ref == 'start':
#                 print("using start")
#                 ep = dlc_chan.extract_epoch([:, :, 0]
#             elif ref == 'stop':
#                 print("using stop")
#
#     return points_within_radius(points_list, target_location, radius)


def xy_plot_animation(x_data, y_data, fname, type='.mp4', **kwargs):
    """
    given a list of arrays for x and y data make a matplotlib animation (gif/mp4).

    :param x_data: list of arrays for x postion
    :param y_data: list of arrays for y postion
    :param x_label: xlabel
    :param y_label: ylabel
    :return: animation saved to specified filepath
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    import os
    from matplotlib.pyplot import get_cmap
    import datetime
    import getpass


    # check for some basic kwargs and set defaults
    if 'dpi' in kwargs.keys():
        dpi = kwargs['dpi']
    else:
        dpi = 100
    if 'fps' in kwargs.keys():
        fps = kwargs['fps']
    else:
        fps = 20
    if 'ffmpeg_path' in kwargs.keys():
        ffmpeg_path = kwargs['ffmpeg_path']
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    else:
        ffmpeg_path = '/auto/users/wingertj/miniconda3/envs/nems_cpu/bin/ffmpeg'
    if 'labels' in kwargs.keys():
        labels = kwargs['labels']
    else:
        labels = [f"input{i}" for i in range(len(x_data))]
    # initialize figure
    fig, axs = plt.subplots(4,1)
    gs = axs[2].get_gridspec()
    # remove the underlying axes
    for ax in axs[2:]:
        ax.remove()
    axbig = fig.add_subplot(gs[2:])
    axs[0].get_shared_x_axes().join(axs[0], axs[1])

    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
        c = get_cmap(cmap)
    else:
        c = get_cmap('viridis')
    cspace = np.linspace(0, 1, len(x_data)+1)
    x_artist_dict = {i: axs[0].plot([], [], markevery=[-1], marker='o',markerfacecolor='r', color=c(cspace[i]), label=labels[i])[0] for i in range(len(x_data))}
    y_artist_dict = {i: axs[1].plot([], [], markevery=[-1], marker='o',markerfacecolor='r', color=c(cspace[i]))[0] for i in range(len(y_data))}
    xy_artist_dict = {i: axbig.plot([], [], markevery=[-1], marker='o',markerfacecolor='r', color=c(cspace[i]))[0] for i in range(len(x_data))}

    # get limits of data
    x_max = max([np.nanmax(l) for l in x_data])
    y_max = max([np.nanmax(l) for l in y_data])
    x_min = min([np.nanmin(l) for l in x_data])
    y_min = min([np.nanmin(l) for l in y_data])
    axs[0].set_ylim(np.floor(x_min), np.ceil(x_max))
    axs[1].set_ylim(np.floor(y_min), np.ceil(y_max))
    axbig.set_xlim(np.floor(x_min), np.ceil(x_max))
    axbig.set_ylim(np.floor(y_min), np.ceil(y_max))

    # setup frame writer
    vidname = fname.split('.')[0]
    metadata = dict(title=vidname, artist=getpass.getuser(), date=datetime.datetime.now())
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    pltime = np.linspace(0, len(x_data[0])/fps, len(x_data[0]))
    # plot and write to file
    window_size = 100
    tick_num = 10
    with writer.saving(fig, fname, dpi=dpi):
        for i in range(len(x_data[0])):
            if i < 100:
                for j in range(len(x_data)):
                    x_artist_dict[j].set_data(pltime[:i], x_data[j][:i])
                    y_artist_dict[j].set_data(pltime[:i], y_data[j][:i])
                    axs[0].set_xlim(0, pltime[window_size])
                    axs[1].set_xlim(0, pltime[window_size])
                    x_ticks = np.linspace(0, pltime[window_size], tick_num)
                    axs[0].set_xticks(x_ticks)
                    axs[1].set_xticks(x_ticks)
                    xy_artist_dict[j].set_data(x_data[j][:i], y_data[j][:i])
            else:
                for j in range(len(x_data)):
                    x_artist_dict[j].set_data(pltime[i-100:i], x_data[j][i-100:i])
                    y_artist_dict[j].set_data(pltime[i-100:i], y_data[j][i-100:i])
                    axs[0].set_xlim(pltime[i-100], pltime[i])
                    axs[1].set_xlim(pltime[i-100], pltime[i])
                    x_ticks = np.linspace(pltime[i-100], pltime[i], tick_num)
                    axs[0].set_xticks(x_ticks)
                    axs[1].set_xticks(x_ticks)
                    xy_artist_dict[j].set_data(x_data[j][i-100:i], y_data[j][i-100:i])
            axs[0].legend(loc='upper right')
            axs[0].set_xticklabels([])
            # axs[0].set_xlabel("time (s)")
            axs[1].set_xlabel("time (s)")
            axs[0].set_ylabel("X Pos\n(mean dev)")
            axs[1].set_ylabel("Y Pos\n(mean dev)")
            axbig.set_ylabel("Y Pos\n(mean dev)")
            axbig.set_xlabel("X Pos\n(mean dev)")
            fig.tight_layout()
            writer.grab_frame()

# make fake prediction data with jitter
def jitter(coords: tuple):
    import random
    x, y = coords
    # Add random noise in range -1 to 1
    jittered_x, jittered_y = x + random.uniform(-1, 1), y + random.uniform(-1, 1)

    return (jittered_x, jittered_y)

def target_lickspout_epochs(rec, tartime=0.6):
    import pandas as pd
    # extract epochs between target onset -tar- and lickspout entry -lick- as well as catch onset and fa
    e = rec['dlc'].epochs
    tar = e['name'].str.startswith("TARGET")
    lick = e['name'].str.startswith("LICK , HIT")
    tareps = rec['dlc'].epochs.loc[tar]
    lickeps = rec['dlc'].epochs.loc[lick]
    catch = e['name'].str.startswith("CATCH")
    FA = e['name'].str.startswith("LICK , FA")
    catcheps = rec['dlc'].epochs.loc[catch]
    faeps = rec['dlc'].epochs.loc[FA]

    # get all snr targets
    alltargetbool = e['name'].str.startswith("TAR")
    alltargetepochs = rec['dlc'].epochs.loc[alltargetbool]

    # for each lickspout entry, grab the preceding target timestamp (nosepoke signal is 0.6 ms prior)
    tarlick = []
    for ind in lickeps.index:
        tarind = tareps.index[tareps.index < ind][-1]
        licktime = lickeps[lickeps.index == ind]['start'].values[0]
        targettime = tareps[tareps.index == tarind]['start'].values[0] - tartime
        targetsnr = alltargetepochs[alltargetepochs.index == tarind - 1]['name'].values[0]
        tarlick.append((targettime, licktime, targetsnr))
    tarlick = pd.DataFrame(tarlick, columns=['start', 'end', 'name'])

    # for each FA grab the precding catch timestamp (nosepoke signal is 0.6 ms prior - 0.1 hold plut 0.5 s onset delay)
    catchfa = []
    for ind in faeps.index:
        catchind = catcheps.index[catcheps.index < ind][-1]
        fatime = faeps[faeps.index == ind]['start'].values[0]
        catchtime = catcheps[catcheps.index == catchind]['start'].values[0] - tartime
        catchname = catcheps[catcheps.index == catchind]['name'].values[0]
        catchfa.append((catchtime, fatime, catchname))
    catchfa = pd.DataFrame(catchfa, columns=['start', 'end', 'name'])

    # append hit and fa trials together and sort by trial type
    hit_fa = [tarlick, catchfa]
    hit_fa = pd.concat(hit_fa)
    hit_fa_epochs = hit_fa.sort_values('start').reset_index()

    # get unique snrs for plotting
    tar_snrs = np.unique(hit_fa['name'].values)
    #
    # hit_fa_epochs = np.concatenate((hit_fa['start'].values[:, np.newaxis], hit_fa['end'].values[:, np.newaxis]), axis=1)

    f, ax = plt.subplots(1,1)
    route = False
    for i in range(len(hit_fa_epochs)):
        tx = rec['dlc'][0][int(hit_fa_epochs['start'][i] * rec.meta['rasterfs'])]
        ty = rec['dlc'][1][int(hit_fa_epochs['start'][i] * rec.meta['rasterfs'])]
        lx = rec['dlc'][0][int(hit_fa_epochs['end'][i] * rec.meta['rasterfs'])]
        ly = rec['dlc'][1][int(hit_fa_epochs['end'][i] * rec.meta['rasterfs'])]
        ax.scatter(tx, ty, color='red')
        ax.scatter(lx, ly, color='blue')
        if route == True:
            for i in range(len(hit_fa_epochs)):
                if i < len(hit_fa_epochs):
                    rpx = rec['dlc'][0][int(hit_fa_epochs['end'][i] * rec.meta['rasterfs']):int(
                        tarlick['start'][i + 1] * rec.meta['rasterfs'])]
                    rpy = rec['dlc'][1][int(hit_fa_epochs['end'][i] * rec.meta['rasterfs']):int(
                        tarlick['start'][i + 1] * rec.meta['rasterfs'])]
                    ax.plot(rpx, rpy, color='green')
    return hit_fa_epochs, tar_snrs

def trials_to_path(rec, tartime=0.6):
    import pandas as pd
    # extract epochs between target onset -tar- and lickspout entry -lick- as well as catch onset and fa
    e = rec['dlc'].epochs
    tar = e['name'].str.startswith("TARGET")
    lick = e['name'].str.startswith("LICK , HIT")
    tareps = rec['dlc'].epochs.loc[tar]
    lickeps = rec['dlc'].epochs.loc[lick]
    catch = e['name'].str.startswith("CATCH")
    FA = e['name'].str.startswith("LICK , FA")
    catcheps = rec['dlc'].epochs.loc[catch]
    faeps = rec['dlc'].epochs.loc[FA]

    # get all snr targets
    e = rec['dlc'].epochs
    alltargetbool = e['name'].str.startswith("TAR")
    alltargetepochs = rec['dlc'].epochs.loc[alltargetbool]

    # for each lickspout entry, grab the preceding target timestamp (nosepoke signal is 0.6 ms prior)
    tarlick = []
    for ind in lickeps.index:
        tarind = tareps.index[tareps.index < ind][-1]
        licktime = lickeps[lickeps.index == ind]['start'].values[0]
        targettime = tareps[tareps.index == tarind]['start'].values[0] - tartime
        targetsnr = alltargetepochs[alltargetepochs.index == tarind - 1]['name'].values[0]
        tarlick.append((targettime, licktime, targetsnr))
    tarlick = pd.DataFrame(tarlick, columns=['start', 'end', 'name'])

    # for each FA grab the precding catch timestamp (nosepoke signal is 0.6 ms prior - 0.1 hold plut 0.5 s onset delay)
    catchfa = []
    for ind in faeps.index:
        catchind = catcheps.index[catcheps.index < ind][-1]
        fatime = faeps[faeps.index == ind]['start'].values[0]
        catchtime = catcheps[catcheps.index == catchind]['start'].values[0] - tartime
        catchname = catcheps[catcheps.index == catchind]['name'].values[0]
        catchfa.append((catchtime, fatime, catchname))
    catchfa = pd.DataFrame(catchfa, columns=['start', 'end', 'name'])

    # append hit and fa trials together and sort by trial type
    hit_fa = [tarlick, catchfa]
    hit_fa = pd.concat(hit_fa)
    hit_fa = hit_fa.sort_values('name')

    # get unique snrs for plotting
    tar_snrs = np.unique(hit_fa['name'].values)

    hit_fa_epochs = np.concatenate((hit_fa['start'].values[:, np.newaxis], hit_fa['end'].values[:, np.newaxis]), axis=1)

    return hit_fa_epochs, tar_snrs

def trial_2d_tc(rec, tartime=0.6, pix_bins=20):
    # create target to lickspout epochs - trials
    hit_fa_epochs, tar_snrs = target_lickspout_epochs(rec=rec, tartime=0.6)

    # grab target onset and lick spout entry epochs - tarlick - and catch fa - catchfa - from resp, dlc, dist
    tarlickeps = rec['resp'].extract_epoch(hit_fa_epochs)
    dlc_eps = rec['dlc'].extract_epoch(hit_fa_epochs)

    # trial xy
    trialx = dlc_eps[:, 2, :]
    trialy = dlc_eps[:, 3, :]

    # get tc shape from whole arena tc for comparison

    # assume DLC sig is ['dlc'] and index 2,3 corresponds to headpost x/y
    x = rec['dlc'][2]
    y = rec['dlc'][3]
    # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
    x_range = np.ptp(x[~np.isnan(x)])
    y_range = np.ptp(y[~np.isnan(y)])

    xbin_num = int(np.round(x_range/pix_bins))
    ybin_num = int(np.round(y_range/pix_bins))

    # generate linearly spaced bins for 2d histogram
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), xbin_num + 1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), ybin_num + 1)

    # generate occupancy histogram
    occupancy, x_edges, y_edges = np.histogram2d(trialx.flatten(), trialy.flatten(), [xbins, ybins], )

    # get bin centers
    x_cent = xbins[0:-1] + np.diff(xbins)/2
    y_cent = ybins[0:-1] + np.diff(ybins)/2

    # find nearest binned values
    xbinind = [int((np.abs(x_cent-i)).argmin()) if ~np.isnan(i) else np.nan for i in trialx.flatten()]
    ybinind = [int((np.abs(y_cent-i)).argmin()) if ~np.isnan(i) else np.nan for i in trialy.flatten()]

    # for each bin of histogram generate a mask of index values
    bin_indexes = {}
    for key in set(zip(xbinind, ybinind)):
        if ~np.isnan(key[0]) and ~np.isnan(key[1]):
            bin_indexes[key] = []
        else:
            continue

    for i, key in enumerate(zip(xbinind, ybinind)):
        if ~np.isnan(key[0]) and ~np.isnan(key[1]):
            bin_indexes[key].append(i)
        else:
            continue

    # generate a tuning curve for each cell
    tuning_curve = {}
    for i, cellname in enumerate(rec['resp'].chans):
        tc_temp = np.full((xbin_num, ybin_num), np.nan)
        for tmpbin in list(bin_indexes.keys()):
            tc_temp[tmpbin[0], tmpbin[1]] = np.nansum(tarlickeps[:, i, :].flatten()[bin_indexes[tmpbin]])/(len(bin_indexes[tmpbin])/rec.meta['rasterfs'])
        tuning_curve[cellname] = tc_temp

    return tuning_curve


def all_trials_plot(rec, cells, tartime=0.6, error='sem', tc2d=False):
    import pandas as pd
    # extract epochs between target onset -tar- and lickspout entry -lick- as well as catch onset and fa
    e = rec['dlc'].epochs
    tar = e['name'].str.startswith("TARGET")
    lick = e['name'].str.startswith("LICK , HIT")
    tareps = rec['dlc'].epochs.loc[tar]
    lickeps = rec['dlc'].epochs.loc[lick]
    catch = e['name'].str.startswith("CATCH")
    FA = e['name'].str.startswith("LICK , FA")
    catcheps = rec['dlc'].epochs.loc[catch]
    faeps = rec['dlc'].epochs.loc[FA]

    # get all snr targets
    e = rec['dlc'].epochs
    alltargetbool = e['name'].str.startswith("TAR")
    alltargetepochs = rec['dlc'].epochs.loc[alltargetbool]

    # for each lickspout entry, grab the preceding target timestamp (nosepoke signal is 0.6 ms prior)
    tarlick = []
    for ind in lickeps.index:
        tarind = tareps.index[tareps.index < ind][-1]
        licktime = lickeps[lickeps.index == ind]['start'].values[0]
        targettime = tareps[tareps.index == tarind]['start'].values[0] - tartime
        targetsnr = alltargetepochs[alltargetepochs.index == tarind - 1]['name'].values[0]
        tarlick.append((targettime, licktime, targetsnr))
    tarlick = pd.DataFrame(tarlick, columns=['start', 'end', 'name'])

    # for each FA grab the precding catch timestamp (nosepoke signal is 0.6 ms prior - 0.1 hold plut 0.5 s onset delay)
    catchfa = []
    for ind in faeps.index:
        catchind = catcheps.index[catcheps.index < ind][-1]
        fatime = faeps[faeps.index == ind]['start'].values[0]
        catchtime = catcheps[catcheps.index == catchind]['start'].values[0] - tartime
        catchname = catcheps[catcheps.index == catchind]['name'].values[0]
        catchfa.append((catchtime, fatime, catchname))
    catchfa = pd.DataFrame(catchfa, columns=['start', 'end', 'name'])

    # append hit and fa trials together and sort by trial type
    hit_fa = [tarlick, catchfa]
    hit_fa = pd.concat(hit_fa)
    hit_fa = hit_fa.sort_values('name')

    # get unique snrs for plotting
    tar_snrs = np.unique(hit_fa['name'].values)

    hit_fa_epochs = np.concatenate((hit_fa['start'].values[:, np.newaxis], hit_fa['end'].values[:, np.newaxis]), axis=1)
    # grab target onset and lick spout entry epochs - tarlick - and catch fa - catchfa - from resp, dlc, dist
    tarlickeps = (rec['resp'].extract_epoch(hit_fa_epochs))
    dlc_eps = (rec['dlc'].extract_epoch(hit_fa_epochs))
    dist_eps = (rec['dist'].extract_epoch(hit_fa_epochs))

    # get 2d tuning curve
    if tc2d:
        # get tc and xy pos for each cell
        tc, xy, xy_edges = spatial_tc_2d(rec)
        trial_tc = trial_2d_tc(rec=rec, tartime=0.6, pix_bins=20)
    else:
        pass

    # get 1d tuning curves for target/licks
    high_dist_tc, high_dist_bins, low_dist_tc, low_dist_bins = dist_tc(rec=rec, epochs=hit_fa_epochs, hs_bins=40,
                                                                       signal='dist', feature=0, ds_bins=20, low_trial_drop=0.5)


    # determine rows based on number of cells
    rasterfs = rec.meta['rasterfs']
    cell_num = len(cells)
    cell_indexes = [np.where(np.array(rec['resp'].chans) == c)[0][0] for c in cells]
    # get spont rate for each cell
    cells_resp = rec['resp'].extract_channels(cells)
    cell_mean = np.mean(cells_resp[:,:], axis=1)*rasterfs
    f, ax = plt.subplots(cell_num, 6, layout='tight', figsize = (12, 8))
    for i, cell in enumerate(cells):
        arena_plot = ax[i, 0].imshow(trial_tc[cell].T, origin='lower', extent=[xy_edges['x'][0], xy_edges['x'][-1], xy_edges['y'][0], xy_edges['y'][-1]])
        for trial in range(len(dlc_eps[:, 0, 0])):
            # trialx = np.array([(np.abs(xy['x'] - i)).argmin() if ~np.isnan(i) else np.nan for i in dlc_eps[trial, 2, :]])
            # trialy = np.array([(np.abs(xy['y'] - i)).argmin() if ~np.isnan(i) else np.nan for i in dlc_eps[trial, 3, :]])
            ax[i, 0].scatter(dlc_eps[trial, 2, 0], dlc_eps[trial, 3, 0], marker='o', color='lime', s=9, label='nosepoke' if trial == 0 else "", alpha=0.2)
            ax[i, 0].scatter(dlc_eps[trial, 2, int(tartime*rasterfs)], dlc_eps[trial, 3, int(tartime*rasterfs)], marker='x', s=9, color='red', label='target' if trial == 0 else "", alpha=0.2)
            ax[i, 0].scatter(dlc_eps[trial, 2, -1], dlc_eps[trial, 3, -1], marker='v', s=9, color='aqua', label='lickspout' if trial == 0 else "", alpha=0.2)
        ax[i, 0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        # ax[i, 0].set_xticks(np.arange(0, len(xy['x']), 5))
        # ax[i, 0].set_yticks(np.arange(0, len(xy['y']), 5))
        # ax[i, 0].set_xticklabels([int(val) for val in xy['x'][np.arange(0, len(xy['x']), 5)]])
        # ax[i, 0].set_yticklabels([int(val) for val in xy['y'][np.arange(0, len(xy['y']), 5)]])
        ax[i, 0].set_xlabel("pixels")
        ax[i, 0].set_ylabel("pixels")
        high_dist_hist = high_dist_tc[cell]
        ax[i, 1].imshow(high_dist_hist, aspect='auto', interpolation="none", origin='lower')
        snr_color = matplotlib.colormaps.get_cmap('Reds')
        snr_grad = np.linspace(0, 1, len(tar_snrs)+1)
        ax[i, 1].set_ylabel("trials")
        ax[i, 1].set_xlabel("distance from lickspout \n (pixels)")
        ax[i, 1].set_xticks(np.arange(0, len(high_dist_bins), 5))
        ax[i, 1].set_xticklabels([int(val) for val in high_dist_bins[np.arange(0, len(high_dist_bins), 5)]])
        low_dist_hist = low_dist_tc[cell]
        ax[i, 2].imshow(low_dist_hist, aspect='auto', interpolation="none", origin='lower')
        ax[i, 2].set_ylabel("trials")
        ax[i, 2].set_xlabel("distance from lickspout \n (pixels)")
        ax[i, 2].set_xticks(np.arange(0, len(low_dist_bins), 5))
        ax[i, 2].set_xticklabels([int(val) for val in low_dist_bins[np.arange(0, len(low_dist_bins), 5)]])

        # standard deviation of trials
        std_dev = np.nanstd(low_dist_tc[cell], axis=0)
        # standard error of mean
        sem = np.nanstd(low_dist_tc[cell], axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(low_dist_tc[cell]), axis=0))

        if error == "sem":
            dist_error = sem
        elif error == "std":
            dist_error = std_dev
        # norm_fr = np.nanmean(low_dist_tc[cell_name], axis=0) / np.nanmax(np.nanmean(low_dist_tc[cell_name], axis=0))
        norm_fr = np.nanmean(low_dist_tc[cell], axis=0)
        ax[i, 3].plot(norm_fr)
        dist_max_ind = np.where(norm_fr == max(norm_fr))
        dist_max = low_dist_bins[dist_max_ind]
        ax[i, 3].fill_between(np.arange(0, len(low_dist_bins)), norm_fr - dist_error, norm_fr + dist_error, alpha=0.1)
        ax[i, 3].set_ylabel("avg firing rate (spikes/s)")
        ax[i, 3].set_xlabel("distance from lickspout \n (pixels)")
        ax[i, 3].set_xticks(np.arange(0, len(low_dist_bins), 5))
        ax[i, 3].set_xticklabels([int(val) for val in low_dist_bins[np.arange(0, len(low_dist_bins), 5)]])
        ax[i, 3].scatter(dist_max_ind, norm_fr[dist_max_ind], marker = 'o', color='orange')
        ax[i, 3].axhline(y=cell_mean[i], color='purple')
        ax[i, 3].text(x=0, y=cell_mean[i], s='spont rate')
        ax[i, 4].imshow(tarlickeps[:, cell_indexes[i], :], aspect='auto', origin='lower')
        ax[i, 4].set_ylabel("trials")
        ax[i, 4].set_xticks(np.linspace(0, len(tarlickeps[0, cell_indexes[i], :]), 5))
        ax[i, 4].set_xticklabels([np.round(sample/rasterfs, decimals = 1) for sample in np.linspace(0, len(tarlickeps[0, cell_indexes[i], :]), 5)])
        ax[i, 4].set_xlabel("time from nosepoke (s)")
        for trial in range(len(tarlickeps[:, cell_indexes[i], 0])):
            nosepoke_time = 1
            lickspout_dist = dist_eps[trial, 0, :][~np.isnan(dist_eps[trial, 0, :])][-1]
            lickspout_time = np.where(dist_eps[trial, 0, :] == lickspout_dist)[0][0]
            # find time of closest distance value to 400 pixels from lickspout
            samples = np.arange(0, len(dist_eps[trial, 0, :]))
            trial_dist = dist_eps[trial, 0, :]
            f = interp1d(trial_dist[~np.isnan(trial_dist)], samples[~np.isnan(trial_dist)], bounds_error=False)
            dist_max_sample = [int(f(dist_max)) if ~np.isnan(f(dist_max)) else np.nan]
            ax[i, 4].scatter(nosepoke_time, trial, marker='|', color='lime', label='nosepoke' if trial == 0 else "")
            ax[i, 4].scatter(tartime*rasterfs, trial, marker='|', color='red', label='target' if trial == 0 else "")
            ax[i, 4].scatter(lickspout_time, trial, marker='|', color='aqua', label='lickspout' if trial == 0 else "")
            ax[i, 4].scatter(dist_max_sample, trial, marker = '|', color='orange', label=f'{dist_max} pixels' if trial == 0 else "")
        ax[i, 4].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        # auditory psth truncated to shortest trial duration
        smoothed_trial_epochs = gaussian_filter1d(tarlickeps[:, cell_indexes[i], np.sum(~np.isnan(tarlickeps[:, cell_indexes[i], :]), axis=0) == len(
            tarlickeps[:, cell_indexes[i], 0])], axis=1, sigma=2)
        target_psth = np.nanmean(smoothed_trial_epochs*rasterfs, axis=0)
        #std dev/sem of auditory psth
        # standard deviation of trials
        std_dev_aud = np.nanstd(smoothed_trial_epochs*rasterfs, axis=0)
        # standard error of mean
        sem_aud = np.nanstd(smoothed_trial_epochs*rasterfs, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(smoothed_trial_epochs*rec.meta["rasterfs"]), axis=0))
        if error == "sem":
            aud_error = sem_aud
        elif error == "std":
            aud_error = std_dev_aud
        ax[i, 5].plot(target_psth)
        ax[i, 5].fill_between(np.arange(0, len(target_psth)), target_psth - aud_error, target_psth + aud_error, alpha=0.1)
        ax[i, 5].set_ylabel("avg spikes/s")
        ax[i, 5].set_xticks(np.linspace(0, len(target_psth), 5))
        ax[i, 5].set_xticklabels([np.round(sample/rasterfs, decimals = 1) for sample in np.linspace(0, len(target_psth), 5)])
        ax[i, 5].axvline(tartime*rasterfs, color='red', label='target' if trial == 0 else "")
        ax[i, 5].set_xlabel("time from nosepoke (s)")
        ax[i, 5].axhline(y=cell_mean[i], color='purple')
        ax[i, 5].text(x=0, y=cell_mean[i], s='spont rate')
        for cval, snr in enumerate(tar_snrs):
            snr_mask = hit_fa['name'].str.startswith(snr).values
            snr_indexes = np.where(snr_mask == True)[0]
            min_snr_ind = snr_indexes[0]
            max_snr_ind = snr_indexes[-1]
            ax[i, 1].axvline(len(high_dist_tc[cell][0, :]), min_snr_ind/len(high_dist_tc[cell][:, 0]), max_snr_ind/len(high_dist_tc[cell][:, 0]), color = 'black' if snr=='CATCH' else snr_color(snr_grad[cval]))
            ax[i, 1].text(x=len(high_dist_tc[cell][0, :])+0.5, y=min_snr_ind, s=f"{snr}"[-5:], size=6, rotation='vertical')
            ax[i, 2].axvline(len(low_dist_tc[cell][0, :]), min_snr_ind/len(high_dist_tc[cell][:, 0]), max_snr_ind/len(high_dist_tc[cell][:, 0]), color = 'black' if snr=='CATCH' else snr_color(snr_grad[cval]))
            ax[i, 2].text(x=len(low_dist_tc[cell][0, :])+0.5, y=min_snr_ind, s=f"{snr}"[-5:], size=6, rotation='vertical')
            ax[i, 4].axvline(len(tarlickeps[0, cell_indexes[i], :]), min_snr_ind/len(high_dist_tc[cell][:, 0]), max_snr_ind/len(high_dist_tc[cell][:, 0]), color = 'black' if snr=='CATCH' else snr_color(snr_grad[cval]))
            ax[i, 4].text(x=len(tarlickeps[0, cell_indexes[i], :])+0.5, y=min_snr_ind, s=f"{snr}"[-5:], size=6, rotation='vertical')

            # plot fr as a function of distance for individual snrs
            # standard deviation of trials
            std_dev_snr = np.nanstd(low_dist_tc[cell][snr_mask], axis=0)
            # standard error of mean
            sem_snr = np.nanstd(low_dist_tc[cell][snr_mask, :], axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(low_dist_tc[cell][snr_mask, :]), axis=0))
            if error == "sem":
                dist_error_snr = sem
            elif error == "std":
                dist_error_snr = std_dev

            norm_fr_snr = np.nanmean(low_dist_tc[cell][snr_mask, :], axis=0)
            ax[i, 3].plot(norm_fr_snr, color='black' if snr=='CATCH' else snr_color(snr_grad[cval]))
            dist_max_ind_snr = np.where(norm_fr == max(norm_fr))
            dist_max_snr = low_dist_bins[dist_max_ind]
            ax[i, 3].fill_between(np.arange(0, len(low_dist_bins)), norm_fr_snr - dist_error_snr, norm_fr_snr + dist_error_snr,
                                  alpha=0.1, color='black' if snr=='CATCH' else snr_color(snr_grad[cval]))
            ax[i, 3].scatter(dist_max_ind_snr, norm_fr_snr[dist_max_ind_snr], marker='o', color='orange')


             # plot psth for each trial type
            valid_time_bins = np.sum(~np.isnan(tarlickeps[snr_mask, cell_indexes[i], :]), axis=0) == len(
                    tarlickeps[snr_mask, cell_indexes[i], 0])
            smoothed_trial_snr = gaussian_filter1d(tarlickeps[snr_mask, :,:][:, cell_indexes[i], valid_time_bins], axis=1, sigma=3)
            target_psth = np.nanmean(
                smoothed_trial_snr * rec.meta['rasterfs'], axis=0)

            # standard error of mean
            std_dev_aud = np.nanstd(smoothed_trial_snr * rec.meta["rasterfs"], axis=0)

            sem_aud = np.nanstd(smoothed_trial_snr * rec.meta["rasterfs"], axis=0,
                                ddof=1) / np.sqrt(np.sum(~np.isnan(smoothed_trial_snr * rec.meta["rasterfs"]), axis=0))
            if error == "sem":
                aud_error = sem_aud
            elif error == "std":
                aud_error = std_dev_aud
            ax[i, 5].plot(target_psth, color = 'black' if snr=='CATCH' else snr_color(snr_grad[cval]))
            ax[i, 5].fill_between(np.arange(0, len(target_psth)), target_psth - aud_error, target_psth + aud_error,
                                  alpha=0.1, color = 'black' if snr=='CATCH' else snr_color(snr_grad[cval]))

        ax[i, 0].set_title(f"{cell}")
        ax[i, 1].set_title(f"{cell}")
        ax[i, 2].set_title(f"{cell}")
        ax[i, 3].set_title(f"{cell}")
        ax[i, 4].set_title(f"{cell}")
        ax[i, 5].set_title(f"{cell}")

def dvt_allcells_plot(rec, tartime=0.6, error='sem'):
    import pandas as pd
    # extract epochs between target onset -tar- and lickspout entry -lick- as well as catch onset and fa
    e = rec['dlc'].epochs
    tar = e['name'].str.startswith("TARGET")
    lick = e['name'].str.startswith("LICK , HIT")
    tareps = rec['dlc'].epochs.loc[tar]
    lickeps = rec['dlc'].epochs.loc[lick]
    catch = e['name'].str.startswith("CATCH")
    FA = e['name'].str.startswith("LICK , FA")
    catcheps = rec['dlc'].epochs.loc[catch]
    faeps = rec['dlc'].epochs.loc[FA]

    # get all snr targets
    e = rec['dlc'].epochs
    alltargetbool = e['name'].str.startswith("TAR")
    alltargetepochs = rec['dlc'].epochs.loc[alltargetbool]

    # for each lickspout entry, grab the preceding target timestamp (nosepoke signal is 0.6 ms prior)
    tarlick = []
    for ind in lickeps.index:
        tarind = tareps.index[tareps.index < ind][-1]
        licktime = lickeps[lickeps.index == ind]['start'].values[0]
        targettime = tareps[tareps.index == tarind]['start'].values[0] - tartime
        targetsnr = alltargetepochs[alltargetepochs.index == tarind - 1]['name'].values[0]
        tarlick.append((targettime, licktime, targetsnr))
    tarlick = pd.DataFrame(tarlick, columns=['start', 'end', 'name'])

    # for each FA grab the precding catch timestamp (nosepoke signal is 0.6 ms prior - 0.1 hold plut 0.5 s onset delay)
    catchfa = []
    for ind in faeps.index:
        catchind = catcheps.index[catcheps.index < ind][-1]
        fatime = faeps[faeps.index == ind]['start'].values[0]
        catchtime = catcheps[catcheps.index == catchind]['start'].values[0] - tartime
        catchname = catcheps[catcheps.index == catchind]['name'].values[0]
        catchfa.append((catchtime, fatime, catchname))
    catchfa = pd.DataFrame(catchfa, columns=['start', 'end', 'name'])

    # append hit and fa trials together and sort by trial type
    hit_fa = [tarlick, catchfa]
    hit_fa = pd.concat(hit_fa)
    hit_fa = hit_fa.sort_values('name')

    # get unique snrs for plotting
    tar_snrs = np.unique(hit_fa['name'].values)

    hit_fa_epochs = np.concatenate((hit_fa['start'].values[:, np.newaxis], hit_fa['end'].values[:, np.newaxis]), axis=1)
    # grab target onset and lick spout entry epochs - tarlick - and catch fa - catchfa - from resp, dlc, dist
    tarlickeps = (rec['resp'].extract_epoch(hit_fa_epochs))
    dlc_eps = (rec['dlc'].extract_epoch(hit_fa_epochs))
    dist_eps = (rec['dist'].extract_epoch(hit_fa_epochs))

    # get 1d tuning curves for target/licks
    high_dist_tc, high_dist_bins, low_dist_tc, low_dist_bins = dist_tc(rec=rec, epochs=hit_fa_epochs, hs_bins=40,
                                                                       signal='dist', feature=0, ds_bins=20, low_trial_drop=0.5)
    # get all cell avg firing rate
    rasterfs = rec.meta['rasterfs']
    avg_resp = np.nanmean(rec['resp'][:, :], axis=1)*rasterfs
    cell_num = len(rec['resp'].chans)
    rowcol = int(np.ceil(np.sqrt(cell_num)))
    fig, ax = plt.subplots(rowcol, rowcol, figsize=(10, 10), layout='tight')
    ax = ax.flatten()
    ax2 = []
    for i in range(len(ax)):
        ax2.append(fig.add_subplot(rowcol, rowcol, i+1, frame_on=False))
    ax2 = np.array(ax2)
    plot_lim = cell_num if cell_num <= rowcol**2 else rowcol

    for i in range(plot_lim):
        # auditory psth truncated to shortest trial duration
        smoothed_trial_epochs = gaussian_filter1d(tarlickeps[:, i, np.sum(~np.isnan(tarlickeps[:, i, :]), axis=0) == len(
            tarlickeps[:, i, 0])], axis=1, sigma=2)
        target_psth = np.nanmean(smoothed_trial_epochs*rasterfs, axis=0)
        #std dev/sem of auditory psth
        # standard deviation of trials
        aud_std = np.nanstd(smoothed_trial_epochs*rasterfs, axis=0)
        # standard error of mean
        aud_sem = np.nanstd(smoothed_trial_epochs*rasterfs, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(smoothed_trial_epochs*rec.meta["rasterfs"]), axis=0))

        #dist error
        dist_hist = np.nanmean(low_dist_tc[rec['resp'].chans[i]],axis=0)
        dist_std = np.nanstd(low_dist_tc[rec['resp'].chans[i]], axis=0)
        # standard error of mean
        dist_sem = np.nanstd(low_dist_tc[rec['resp'].chans[i]], axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(low_dist_tc[rec['resp'].chans[i]]), axis=0))

        if error == "sem":
            aud_error = aud_sem
            dist_error = dist_sem
        elif error == "std":
            aud_error = aud_std
            dist_error= dist_std

        # find max firing rate for cell to format y axis
        target_max = max(target_psth)
        dist_max = max(np.nanmean(low_dist_tc[rec['resp'].chans[i]],axis=0))
        ymax = np.ceil(target_max if target_max > dist_max else dist_max)
        y_ticks = np.linspace(0, ymax, 4)
        ax2[i].set_title(rec['resp'].chans[i])
        ax[i].plot(low_dist_bins, dist_hist, color='blue', label='pixels from lickspout' if i == 0 else '')
        ax[i].fill_between(low_dist_bins, dist_hist-dist_error, dist_hist+dist_error, color='blue', alpha=0.2)
        ax[i].axhline(y=avg_resp[i], color='black', label='avg firing rate' if i==0 else '')
        ax[i].set_xticks(low_dist_bins[np.arange(0, len(low_dist_bins), 5)])
        ax[i].set_xticklabels([int(val) for val in low_dist_bins[np.arange(0, len(low_dist_bins), 5)]], color='blue')
        ax[i].set_xlabel(xlabel = "pixels" if i >= rowcol*(rowcol-1) else '', color='blue')
        ax[i].xaxis.tick_bottom()
        ax[i].yaxis.tick_left()
        ax[i].xaxis.set_label_position('bottom')
        ax[i].yaxis.set_label_position('left')
        ax[i].set_yticks(y_ticks)
        ax[i].set_ylabel(ylabel = "spikes/s" if i%rowcol == 0 else '')
        ax[i].spines[['right', 'top']].set_visible(True)
        ax[i].tick_params(axis='x', color='blue')
        ax2[i].plot(target_psth, color='red', label='time from nosepoke' if i == 0 else '')
        ax2[i].fill_between(np.arange(0, len(target_psth)), target_psth- aud_error, target_psth + aud_error, color='red',
                           alpha=0.2)
        ax2[i].set_xticks(np.linspace(0, len(target_psth), 5))
        ax2[i].set_xticklabels(
            [np.round(sample / rasterfs, decimals=1) for sample in np.linspace(0, len(target_psth), 5)], color='red')
        ax2[i].xaxis.tick_top()
        ax2[i].yaxis.tick_right()
        ax2[i].xaxis.set_label_position('top')
        ax2[i].spines[['right', 'top']].set_visible(True)
        ax2[i].set_yticks(y_ticks)
        ax2[i].set_yticklabels([])
        ax2[i].set_xlabel(xlabel="time from nosepoke" if i >= rowcol*(rowcol-1) else '', color='red')
        ax2[i].tick_params(axis='x', color='red')

    # remove unused axes
    for i in range(plot_lim, rowcol**2):
        ax[i].set_axis_off()
        ax2[i].set_axis_off()