import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from nems0 import db, preprocessing
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.motor.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential, LevelShift, ReLU
from nems.layers.base import Layer, Phi, Parameter
from scipy.interpolate import interp1d

runclassid = 132
rasterfs = 100
# sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
#dparm = db.pd_query(sql)

sql = f"SELECT distinct left(cellid,7) as siteid,stimpath,stimfile from sCellFile where runclassid={runclassid}"
dallfiles = db.pd_query(sql)
siteids = dallfiles['siteid'].unique().tolist()

# hardcode siteid with lots of cells and movement
siteid = 'PRN050a'
sql = f"SELECT count(cellid) as cellcount,stimpath,stimfile from sCellFile where cellid like '{siteid}%%' AND runclassid={runclassid} AND area='A1' group by stimpath,stimfile"
dparminfo = db.pd_query(sql)

parmfile = [r.stimpath+r.stimfile for i,r in dparminfo.iterrows()]
cellids=None

# else:
#     parmfile = ["/auto/data/daq/Prince/PRN015/PRN015a01_a_NTD",
#                 "/auto/data/daq/Prince/PRN015/PRN015a02_a_NTD"]
#     cellids = None

## load the recording
try:
    ex = BAPHYExperiment(parmfile=parmfile, cellid=cellids)
    print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

    recache = False

    # load recording
    # rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
    #                        dlc=True, recache=recache, rasterfs=rasterfs,
    #                        dlc_threshold=0.2, fill_invalid='interpolate')
    rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
                           dlc=True, recache=recache, rasterfs=rasterfs,
                           dlc_threshold=0.2, fill_invalid='interpolate')
except:
    raise ValueError(f"Problem loading {siteid}")

# generate 'dist' signal from dlc signal from approximately lickspout
rec = dlc2dist(rec, ref_x0y0=[470, 90], smooth_win=5, norm=False, verbose=False)

rec['dlc'].epochs
# grab A1 units
try:
    depth_info = baphy_io.get_depth_info(siteid=siteid)
    A1_units = depth_info.loc[depth_info['area']== 'A1'].index.tolist()
    # grab unit names for all units in resp that have depth info in A1
    A1_in_rec = [chan for chan in rec['resp'].chans if chan in A1_units]
    # compute PSTH for repeated stimuli
    epoch_regex = "^STIM_"
    rec['resp'] = rec['resp'].extract_channels(A1_in_rec)
    rec['resp'] = rec['resp'].rasterize()
    rec['psth'] = preprocessing.generate_average_sig(rec['resp'], 'psth', epoch_regex=epoch_regex)
except:
    raise ValueError("A1 units don't match units in rec?")

#TODO make spatial_tc_2d and decode_2d take different batches of data to avoid overfitting

def spatial_tc_2d(rec):
    # assume rasterized signals

    # assume DLC sig is ['dlc'] and index 2,3 corresponds to headpost x/y
    x = rec['dlc'][2]
    y = rec['dlc'][3]
    # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
    xbin_num = 30
    ybin_num = 20
    # generate linearly spaced bins for 2d histogram
    xbins = np.linspace(np.nanmin(x), np.nanmax(x), xbin_num + 1)
    ybins = np.linspace(np.nanmin(y), np.nanmax(y), ybin_num + 1)
    # generate occupancy histogram - count of x/y occupancy samples - need to convert to time to get spike rate later
    occupancy, x_edges, y_edges = np.histogram2d(x, y, [xbins, ybins],)
    tc = {}
    for cellindex, cellid in enumerate(rec['resp'].chans):
        # nasty way to unrasterize spikes and return a list of x/y positions for each spike
        n_x_loc = [item for sublist in [[xpos]*int(spk_cnt) for xpos, spk_cnt in zip(x, rec['resp'][cellindex]) if spk_cnt !=0] for item in sublist]
        n_y_loc = [item for sublist in [[ypos]*int(spk_cnt) for ypos, spk_cnt in zip(y, rec['resp'][cellindex]) if spk_cnt !=0] for item in sublist]
        # create a 2d histogram of spike locations
        spk_hist, x_edges, y_edges = np.histogram2d(n_x_loc, n_y_loc, [xbins, ybins],)
        # divide the spike locations by the amount of time spent in each bin to get firing rate
        rate_bin = spk_hist/(occupancy*rasterfs)
        # add the tc for each cell to the dictionary with key == cellid
        tc[cellid] = rate_bin
    # get center position of each bin to be used in assigning x/y location
    x_cent = xbins[0:-1] + np.diff(xbins)/2
    y_cent = ybins[0:-1] + np.diff(ybins)/2
    # make feature dictionary x/y with positions
    xy = {'x':x_cent, 'y':y_cent}

    return tc, xy, xbins, ybins


# get tc and xy pos for each cell
tc, xy, xbins, ybins = spatial_tc_2d(rec)

# convert dlc sig to nearest tc bins
def dlc_to_tcpos(xy):
    # get dlc data
    x = rec['dlc'][2]
    y = rec['dlc'][3]

    # find nearest binned values
    xnew = np.array([xy['x'][(np.abs(xy['x']-i)).argmin()] for i in x])
    ynew = np.array([xy['y'][(np.abs(xy['y']-i)).argmin()] for i in y])

    return xnew, ynew

def tc_stability(tc, xy):

    ### for each bin split occupancy in half, then generate a firing rate hist for each data split, then calculate correlation between both halves of data ###

    # step 1, get hist bin for each dlc data point
    # get dlc data
    x = rec['dlc'][2]
    y = rec['dlc'][3]

    # find nearest binned values
    xbinind = np.array([(np.abs(xy['x']-i)).argmin() for i in x])
    ybinind = np.array([(np.abs(xy['y']-i)).argmin() for i in y])

    # get occupancy in each bin
    # xbin_num = 30
    # ybin_num = 20
    # # generate linearly spaced bins for 2d histogram
    # xbins = np.linspace(np.nanmin(x), np.nanmax(x), xbin_num + 1)
    # ybins = np.linspace(np.nanmin(y), np.nanmax(y), ybin_num + 1)
    # # generate occupancy histogram - count of x/y occupancy samples - need to convert to time to get spike rate later
    # occupancy, x_edges, y_edges = np.histogram2d(x, y, [xbins, ybins], )
    bin_occupancy = np.zeros((30, 20))
    for xi,yi in zip(xbinind, ybinind):
        bin_occupancy[xi, yi] = bin_occupancy[xi, yi] + 1

    mask1 = {}
    mask2 = {}
    for key in set(zip(xbinind, ybinind)):
        mask1[key] = []
        mask2[key] = []

    # split data in half. create empty matrix. for each data point, add 1 to count in each bin of empty matrix and append index to xy dict for mask1.
    # once half_count for that bin is half of total occupancy, append index to second mask for that bin.
    half_count = np.zeros((30, 20))
    for i, xy in enumerate(zip(xbinind, ybinind)):
        if half_count[xy[0], xy[1]] < bin_occupancy[xy[0], xy[1]]/2:
            half_count[xy[0], xy[1]] = half_count[xy[0], xy[1]] + 1
            mask1[xy].append(i)
        else:
            mask2[xy].append(i)

    # threshold occupancy
    for xy in list(mask1.keys()):
        # each bin must have at least 0.5 second of data
        if len(mask2[xy]) < rasterfs/2:
            mask1[xy] = []
            mask2[xy] = []

    tc1 = {}
    tc2 = {}
    tc_12cc = {}
    for i, cellname in enumerate(rec['resp'].chans):
        tmptc1 = np.full((30, 20), np.nan)
        tmptc2 = np.full((30, 20), np.nan)
        for tmpbin in list(mask1.keys()):
            tmptc1[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][mask1[tmpbin]])/(len(mask1[tmpbin])/rasterfs)
            tmptc2[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][mask2[tmpbin]])/(len(mask2[tmpbin])/rasterfs)
        tc1[cellname] = tmptc1
        tc2[cellname] = tmptc2
        tc_12cc[cellname] = np.corrcoef(tmptc1[~np.isnan(tmptc1)].flatten(), tmptc2[~np.isnan(tmptc2)].flatten())[0, 1]

    return tc_12cc, tc1, tc2

def cell_spatial_info(xy):
    """
    Spatial information (SI) for each unit was calculated from the linearized location data and firing rate following (Skaggs et al. 1992):
    SI=∑i=1NPiXirlog2Xir, where Pi is the probability of finding the animal in bin i, Xi is the sum of the firing rates observed when the animal was found in bin i - ammended to be the mean firing rate in bin i,
    r is the mean spiking activity of the neuron, and N is the number of bins of the linearized trajectory (104)
    :return:
    """

    # step 1, get hist bin for each dlc data point
    # get dlc data
    x = rec['dlc'][2]
    y = rec['dlc'][3]

    # find nearest binned values
    xbinind = np.array([(np.abs(xy['x']-i)).argmin() for i in x])
    ybinind = np.array([(np.abs(xy['y']-i)).argmin() for i in y])

    # create histogram
    bin_occupancy = np.zeros((30, 20))
    for xi,yi in zip(xbinind, ybinind):
        bin_occupancy[xi, yi] = bin_occupancy[xi, yi] + 1

    # get bin probability distribution
    bin_probability = bin_occupancy/np.nansum(bin_occupancy)
    # for each bin of histogram generate a mask of index values
    bin_indexes = {}
    for key in set(zip(xbinind, ybinind)):
        bin_indexes[key] = []
    for i, xy in enumerate(zip(xbinind, ybinind)):
            bin_indexes[xy].append(i)

    # generate a tuning curve for each cell
    tuning_curve = {}
    for i, cellname in enumerate(rec['resp'].chans):
        tc_temp = np.full((30, 20), np.nan)
        for tmpbin in list(bin_indexes.keys()):
            tc_temp[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][bin_indexes[tmpbin]])/(len(bin_indexes[tmpbin])/rasterfs)
        tuning_curve[cellname] = tc_temp

    # calculate spatial information
    spatial_information = {}
    for cellname in rec['resp'].chans:
        tc = tuning_curve[cellname]
        mean_firing_rate = np.nanmean(tc.flatten())
        spatial_information[cellname] = np.nansum(bin_probability*tc/mean_firing_rate*np.log2(tc/mean_firing_rate))

    return spatial_information

cell_si = cell_spatial_info(xy)

tc_12_cc, tc1, tc2 = tc_stability(tc, xy)

# plot 3 most stable and 3 least stable cells

sorted_tc_cc = sorted(tc_12_cc.items(), key=lambda x:x[1], reverse=True)
sorted_tc_cc = dict(sorted_tc_cc)

best_3_ss = list(sorted_tc_cc.keys())[:3]
worst_3_ss = list(sorted_tc_cc.keys())[-3:]
f, ax = plt.subplots(6,2, layout='tight', figsize=(7,10))
for i, cell in enumerate(best_3_ss+worst_3_ss):
    ax[i, 0].imshow(tc1[cell])
    ax[i, 0].set_title(f"1st half: {cell}\n scc:{sorted_tc_cc[cell]}")
    ax[i, 1].imshow(tc2[cell])
    ax[i, 1].set_title(f"2nd half: {cell}\n scc:{sorted_tc_cc[cell]}")

# plot 3 cells with highest spatial info and 3 with least
sorted_si = sorted(cell_si.items(), key=lambda x:x[1], reverse=True)
sorted_si = dict(sorted_si)

best_3_si = list(sorted_si.keys())[:3]
worst_3_si = list(sorted_si.keys())[-3:]
f, ax = plt.subplots(6,3, layout='tight', figsize=(7,10))
for i, cell in enumerate(best_3_si+worst_3_si):
    ax[i, 0].imshow((tc1[cell] + tc2[cell])/2)
    ax[i, 0].set_title(f"cell: {cell}\n si:{sorted_si[cell]}")
    ax[i, 1].imshow(tc1[cell])
    ax[i, 1].set_title(f"1st half: {cell}\n scc:{sorted_tc_cc[cell]}")
    ax[i, 2].imshow(tc2[cell])
    ax[i, 2].set_title(f"2nd half: {cell}\n scc:{sorted_tc_cc[cell]}")

xnew, ynew = dlc_to_tcpos(xy)
f, ax = plt.subplots(1,1)
ax.plot(xnew[:1000], ynew[:1000])
ax.plot(rec['dlc'][2][:1000], rec['dlc'][3][:1000])
ax.set_title("binned x/y pos vs actual x/y")

# f, ax = plt.subplots(1,1)
# ax.plot(xnew[:1000])
# ax.plot(rec['dlc'][2][:1000])
# ax.set_title("binned x vs actual x")

# plot bin centers over cell tc for checking
# f, ax = plt.subplots(1,1)
# ax.imshow(tc[list(tc.keys())[-5]], origin='lower', extent=[ybins[0], ybins[-1], xbins[0], xbins[-1]] )
# for xpos in xy['x']:
#     xpos_leny = np.zeros_like(xy['y'])
#     xpos_leny[:] = xpos
#     ax.scatter(xy['y'], xpos_leny, color='red', s=0.4)

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


# run decoding and check similarity metrics cc vs euclidian
decoded_pos, p = decode2d(rec, tc, xy)

# plot all decoded data...do we see running pattern
fig, ax = plt.subplots(1,1)
ax.plot(decoded_pos[:, 0], decoded_pos[:, 1])
ax.plot(xnew[:], ynew[:])
ax.set_title("decoded pos and actual pos")

# similarity metrics
ccx = np.corrcoef(decoded_pos[~np.isnan(xnew), 0], xnew[~np.isnan(xnew)])[0, 1]
ccy = np.corrcoef(decoded_pos[~np.isnan(ynew), 1], ynew[~np.isnan(ynew)])[0, 1]
xynew = np.array(list(zip(xnew[~np.isnan(xnew)], ynew[~np.isnan(ynew)])))
xydecoded = np.array(list(zip(decoded_pos[~np.isnan(xnew), 0], decoded_pos[~np.isnan(ynew), 1])))
euc = np.array([np.sqrt(np.sum(np.square(p1 - p2))) for (p1, p2) in zip(xynew, xydecoded)]).mean()
fig, ax = plt.subplots(3,1, layout='tight')
ax[0].plot(decoded_pos[1000:4000, 0], label='decoded x')
ax[0].plot(xnew[1000:4000], label='real x')
ax[0].set_title(f"decoded vs real x: cc {ccx}")
ax[0].legend()
ax[1].plot(decoded_pos[1000:4000, 1], label='decoded y')
ax[1].plot(ynew[1000:4000], label='real y')
ax[1].set_title(f"decoded vs real y: cc {ccy}")
ax[1].legend()
ax[2].plot(decoded_pos[1000:4000, 0], decoded_pos[1000:4000, 1], label='decoded')
ax[2].plot(xnew[1000:4000], ynew[1000:4000], label='real')
ax[2].set_title(f"decoded vs real pos: mean euc {euc}")
ax[2].legend()


#TODO --- make single "trial" distance tuning ---
# create a 1d occupancy histogram and return bin centers
def dist_occ_hist(dist_eps, feature=0, bins = 40):

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
def dist_tc():
    trial_dist_tc = {}
    for spk, spk_name in enumerate(rec['resp'].chans):
        tmp_dist_tc = np.full((len(nan_free_dist), xbin_num), np.nan)
        for trial in range(len(nan_free_dist)):
            trial_dist_index = np.array([np.abs(distbin_cent - i).argmin() for i in nan_free_dist[trial]])
            trial_occ_hist, edges = np.histogram(nan_free_dist[trial], bins=distbins)
            trial_occ_hist = np.where(trial_occ_hist == 0, np.nan, trial_occ_hist)
            temp_trial_tc = np.full(len(trial_occ_hist), np.nan)
            for bin in range(len(trial_occ_hist)):
                bin_mask = trial_dist_index == bin
                temp_trial_tc[bin] = np.sum(nan_free_spks[trial][:, spk][bin_mask])/trial_occ_hist[bin]
            tmp_dist_tc[trial, :] = temp_trial_tc
        trial_dist_tc[spk_name] = tmp_dist_tc

    # interpolate distance over 1d tuning curves and then downsample distance
    smooth_dist_tc = {}
    low_bins = 20
    low_dist = np.linspace(np.nanmin(all_dist), np.nanmax(all_dist), low_bins)
    for spk, spk_name in enumerate(rec['resp'].chans):
        all_trial_tc = trial_dist_tc[spk_name]
        smooth_trial_tc = np.full((len(all_trial_tc[:, 0]), low_bins), np.nan)
        for trial in range(len(all_trial_tc[:, 0])):
            trial_tc = all_trial_tc[trial, :]
            f = interp1d(distbin_cent[~np.isnan(trial_tc)], trial_tc[~np.isnan(trial_tc)], kind='linear', bounds_error=False)
            smooth_trial_tc[trial, :] = f(low_dist)

        smooth_dist_tc[spk_name] = smooth_trial_tc

# extract epochs between target onset -tar- and lickspout entry -lick-
e=rec['dlc'].epochs
tar = e['name'].str.startswith("TARGET")
lick = e['name'].str.startswith("LICK , HIT")
tareps = rec['dlc'].epochs.loc[tar]
lickeps = rec['dlc'].epochs.loc[lick]

# for each lickspout entry, grab the preceding target timestamp (nosepoke signal is 0.5ms prior)
tarlick = []
for ind in lickeps.index:
    tarind = tareps.index[[tareps.index < ind]][-1]
    licktime = lickeps[lickeps.index == ind]['start'].values[0]
    targettime = tareps[tareps.index == tarind]['start'].values[0] - 0.5
    tarlick.append((targettime, licktime))

# grab target onset and lick spout entry epochs - tarlick - from resp, dlc, dist
tarlickeps = (rec['resp'].extract_epoch(np.array(tarlick)))
dlc_eps = (rec['dlc'].extract_epoch(np.array(tarlick)))
dist_eps = (rec['dist'].extract_epoch(np.array(tarlick)))

# sanity plots - dist signals begin and end near same location?
# f, ax = plt.subplots(2,1)
# for i in range(len(dlc_eps[:, 0, 0])):
#     ax[0].plot(dlc_eps[i, 1, :], dlc_eps[i, 2, :])
# ax[0].set_title("xy pos per trial")
# for i in range(len(dist_eps[:, 0, 0])):
#     ax[1].plot(dist_eps[i, 0, :])
# ax[1].axvline(0.5*rasterfs)
# ax[1].set_title("dist from lickspout per trial")

# create 1d occupancy hist for distance from lickspout
occ_hist, distbin_cent = dist_occ_hist(dist_eps, feature=0, bins=40)

# for each cell generate a dist feature/firing rate tuning curve
def dist_tc():
    trial_dist_tc = {}
    for spk, spk_name in enumerate(rec['resp'].chans):
        tmp_dist_tc = np.full((len(nan_free_dist), xbin_num), np.nan)
        for trial in range(len(nan_free_dist)):
            trial_dist_index = np.array([np.abs(distbin_cent - i).argmin() for i in nan_free_dist[trial]])
            trial_occ_hist, edges = np.histogram(nan_free_dist[trial], bins=distbins)
            trial_occ_hist = np.where(trial_occ_hist == 0, np.nan, trial_occ_hist)
            temp_trial_tc = np.full(len(trial_occ_hist), np.nan)
            for bin in range(len(trial_occ_hist)):
                bin_mask = trial_dist_index == bin
                temp_trial_tc[bin] = np.sum(nan_free_spks[trial][:, spk][bin_mask])/trial_occ_hist[bin]
            tmp_dist_tc[trial, :] = temp_trial_tc
        trial_dist_tc[spk_name] = tmp_dist_tc

    # interpolate distance over 1d tuning curves and then downsample distance
    smooth_dist_tc = {}
    low_bins = 20
    low_dist = np.linspace(np.nanmin(all_dist), np.nanmax(all_dist), low_bins)
    for spk, spk_name in enumerate(rec['resp'].chans):
        all_trial_tc = trial_dist_tc[spk_name]
        smooth_trial_tc = np.full((len(all_trial_tc[:, 0]), low_bins), np.nan)
        for trial in range(len(all_trial_tc[:, 0])):
            trial_tc = all_trial_tc[trial, :]
            f = interp1d(distbin_cent[~np.isnan(trial_tc)], trial_tc[~np.isnan(trial_tc)], kind='linear', bounds_error=False)
            smooth_trial_tc[trial, :] = f(low_dist)

        smooth_dist_tc[spk_name] = smooth_trial_tc


# plot a few neurons over trials as both a function of time and distance
f, ax = plt.subplots(3,3, layout='tight', figsize=(7,10))
best_cell_indexes = [np.where(np.array(rec['resp'].chans) == bc)[0][0] for bc in best_3_ss]
cellid_n =  rec['resp'].chans[31]
for i, (cell_name, cell_index) in enumerate(zip(best_3_ss, best_cell_indexes)):
    ax[i, 0].imshow((tc1[cell_name] + tc2[cell_name]).T/2)
    ax[i, 1].imshow(smooth_dist_tc[cell_name], aspect='auto')
    ax[i, 2].imshow(tarlickeps[:, cell_index, :],aspect='auto')

# plot trial by trial distance tuning and error
col = 10
row = 8
f, ax = plt.subplots(row,col, figsize=(12, 8), sharey=True, sharex=True)
ax = ax.flatten()
cellnum = len(rec['resp'].chans)
for i in range(cellnum, col*row):
    ax[i].set_axis_off()
for i, cell_name in enumerate(rec['resp'].chans):
    std_dev = np.nanstd(smooth_dist_tc[cell_name], axis=0)
    norm_fr = np.nanmean(smooth_dist_tc[cell_name], axis=0) / np.nanmax(np.nanmean(smooth_dist_tc[cell_name], axis=0))
    ax[i].plot(norm_fr)
    ax[i].fill_between(np.arange(0, low_bins), norm_fr-std_dev, norm_fr+std_dev, alpha=0.5)
# add big plot to label common x and y axis
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("normalized firing rate")
plt.xlabel("dist from lickspout")

bp = []