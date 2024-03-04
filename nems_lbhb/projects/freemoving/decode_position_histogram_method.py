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
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential, LevelShift, ReLU
from nems.layers.base import Layer, Phi, Parameter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import nems_lbhb.projects.freemoving.decoder_tools as dec

runclassid = 132
rasterfs = 100
# sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
#dparm = db.pd_query(sql)

# sql = f"SELECT distinct left(cellid,7) as siteid,stimpath,stimfile from sCellFile where runclassid={runclassid}"
# dallfiles = db.pd_query(sql)
# siteids = dallfiles['siteid'].unique().tolist()

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
    # rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
    #                        dlc=True, recache=False, rasterfs=rasterfs,
    #                        dlc_threshold=0.2, fill_invalid='interpolate')
    rec = ex.get_recording(resp=True, stim=False,
                           dlc=True, recache=False, rasterfs=rasterfs,
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
    if len(A1_in_rec) == 0:
        A1_in_rec = [chan for chan in rec['resp'].chans if chan[:7]+chan[15:] in A1_units]

    # compute PSTH for repeated stimuli
    epoch_regex = "^STIM_"
    rec['resp'] = rec['resp'].extract_channels(A1_in_rec)
    rec['resp'] = rec['resp'].rasterize()
    rec['psth'] = preprocessing.generate_average_sig(rec['resp'], 'psth', epoch_regex=epoch_regex)
except:
    raise ValueError("A1 units don't match units in rec?")

#TODO make spatial_tc_2d and decode_2d take different batches of data to avoid overfitting

# def spatial_tc_2d(rec, occ_threshold=0.5, pix_bins=20):
#     # assume rasterized signals
#
#     # assume DLC sig is ['dlc'] and index 2,3 corresponds to headpost x/y
#     x = rec['dlc'][2]
#     y = rec['dlc'][3]
#     # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
#     x_range = np.ptp(x[~np.isnan(x)])
#     y_range = np.ptp(y[~np.isnan(y)])
#
#     xbin_num = int(np.round(x_range/pix_bins))
#     ybin_num = int(np.round(y_range/pix_bins))
#
#     # generate linearly spaced bins for 2d histogram
#     xbins = np.linspace(np.nanmin(x), np.nanmax(x), xbin_num + 1)
#     ybins = np.linspace(np.nanmin(y), np.nanmax(y), ybin_num + 1)
#
#     # generate occupancy histogram - count of x/y occupancy samples - need to convert to time to get spike rate later
#     occupancy, x_edges, y_edges = np.histogram2d(x, y, [xbins, ybins],)
#     tc = {}
#     for cellindex, cellid in enumerate(rec['resp'].chans):
#         # nasty way to unrasterize spikes and return a list of x/y positions for each spike
#         n_x_loc = [item for sublist in [[xpos]*int(spk_cnt) for xpos, spk_cnt in zip(x, rec['resp'][cellindex]) if spk_cnt !=0] for item in sublist]
#         n_y_loc = [item for sublist in [[ypos]*int(spk_cnt) for ypos, spk_cnt in zip(y, rec['resp'][cellindex]) if spk_cnt !=0] for item in sublist]
#         # create a 2d histogram of spike locations
#         spk_hist, x_edges, y_edges = np.histogram2d(n_x_loc, n_y_loc, [xbins, ybins],)
#         # divide the spike locations by the amount of time spent in each bin to get firing rate
#         rate_bin = spk_hist/(occupancy/rec.meta['rasterfs'])
#         # add the tc for each cell to the dictionary with key == cellid, also transpose xy to yx for visualization purposes
#         tc[cellid] = rate_bin
#
#     # threshold occupancy so only bins with at least half a second of data are included
#     for cellid in rec['resp'].chans:
#         tc[cellid][np.where(occupancy < occ_threshold*rec.meta['rasterfs'])] = np.nan
#
#     # get center position of each bin to be used in assigning x/y location
#     x_cent = xbins[0:-1] + np.diff(xbins)/2
#     y_cent = ybins[0:-1] + np.diff(ybins)/2
#     # make feature dictionary x/y with positions
#     xy = {'x':x_cent, 'y':y_cent}
#     xy_edges = {'x':x_edges, 'y':y_edges}
#
#     return tc, xy, xy_edges


# get tc and xy pos for each cell
tc, xy, xy_edges = dec.spatial_tc_2d(rec)

# convert dlc sig to nearest tc bins
# def dlc_to_tcpos(rec, xy):
#     # get dlc data
#     x = rec['dlc'][2]
#     y = rec['dlc'][3]
#
#     # find nearest binned values
#     xnew = np.array([xy['x'][(np.abs(xy['x']-i)).argmin()] if ~np.isnan(i) else np.nan for i in x])
#     ynew = np.array([xy['y'][(np.abs(xy['y']-i)).argmin()] if ~np.isnan(i) else np.nan for i in y])
#
#     return xnew, ynew

# def tc_stability(rec, tc, xy, occ_threshold=0.5, pix_bins=20):
#
#     ### for each bin split occupancy in half, then generate a firing rate hist for each data split, then calculate correlation between both halves of data ###
#
#     # step 1, get hist bin for each dlc data point
#     # get dlc data
#     x = rec['dlc'][2]
#     y = rec['dlc'][3]
#     # how many bins? - 30x by 20y leads to about 20 square pixels per bin.
#     x_range = np.ptp(x[~np.isnan(x)])
#     y_range = np.ptp(y[~np.isnan(y)])
#
#     xbin_num = int(np.round(x_range/pix_bins))
#     ybin_num = int(np.round(y_range/pix_bins))
#
#     # find nearest binned values
#     xbinind = [int((np.abs(xy['x']-i)).argmin()) if ~np.isnan(i) else np.nan for i in x]
#     ybinind = [int((np.abs(xy['y']-i)).argmin()) if ~np.isnan(i) else np.nan for i in y]
#
#
#     bin_occupancy = np.zeros((xbin_num, ybin_num))
#     for xi,yi in zip(xbinind, ybinind):
#         if ~np.isnan(xi) and ~np.isnan(yi):
#             bin_occupancy[xi, yi] = bin_occupancy[xi, yi] + 1
#         else:
#             continue
#
#     mask1 = {}
#     mask2 = {}
#     for key in set(zip(xbinind, ybinind)):
#         if ~np.isnan(key[0]) and ~np.isnan(key[1]):
#             mask1[key] = []
#             mask2[key] = []
#         else:
#             continue
#
#     # split data in half. create empty matrix. for each data point, add 1 to count in each bin of empty matrix and append index to xy dict for mask1.
#     # once half_count for that bin is half of total occupancy, append index to second mask for that bin.
#     half_count = np.zeros((xbin_num, ybin_num))
#     for i, key in enumerate(zip(xbinind, ybinind)):
#         if ~np.isnan(key[0]) and ~np.isnan(key[1]):
#             if half_count[key[0], key[1]] < bin_occupancy[key[0], key[1]]/2:
#                 half_count[key[0], key[1]] = half_count[key[0], key[1]] + 1
#                 mask1[key].append(i)
#             else:
#                 mask2[key].append(i)
#         else:
#             continue
#
#     # threshold occupancy
#     for key in set(zip(xbinind, ybinind)):
#         # each bin must have at least 0.5 second of data
#         if ~np.isnan(key[0]) and ~np.isnan(key[1]):
#             if bin_occupancy[key[0], key[1]] < rec.meta['rasterfs']*occ_threshold:
#                 mask1[key] = []
#                 mask2[key] = []
#         else:
#             continue
#
#     tc1 = {}
#     tc2 = {}
#     tc_12cc = {}
#     for i, cellname in enumerate(rec['resp'].chans):
#         tmptc1 = np.full((xbin_num, ybin_num), np.nan)
#         tmptc2 = np.full((xbin_num, ybin_num), np.nan)
#         for tmpbin in list(mask1.keys()):
#             tmptc1[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][mask1[tmpbin]])/(len(mask1[tmpbin])/rec.meta['rasterfs'])
#             tmptc2[tmpbin[0], tmpbin[1]] = np.nansum(rec['resp'][i][mask2[tmpbin]])/(len(mask2[tmpbin])/rec.meta['rasterfs'])
#         tc1[cellname] = tmptc1
#         tc2[cellname] = tmptc2
#         tc_12cc[cellname] = np.corrcoef(tmptc1[~np.isnan(tmptc1)].flatten(), tmptc2[~np.isnan(tmptc2)].flatten())[0, 1]
#
#     return tc_12cc, tc1, tc2

cell_si = dec.cell_spatial_info(rec, xy)

tc_12_cc, tc1, tc2 = dec.tc_stability(rec, tc, xy)

# plot 3 most stable and 3 least stable cells

sorted_tc_cc = sorted(tc_12_cc.items(), key=lambda x:x[1], reverse=True)
sorted_tc_cc = dict(sorted_tc_cc)

best_3_ss = list(sorted_tc_cc.keys())[:3]
worst_3_ss = list(sorted_tc_cc.keys())[-3:]
f, ax = plt.subplots(6,3, layout='tight', figsize=(7,10))
for i, cell in enumerate(best_3_ss+worst_3_ss):
    ax[i, 0].imshow(tc[cell])
    ax[i, 1].imshow(tc1[cell])
    ax[i, 1].set_title(f"1st half: {cell}\n scc:{sorted_tc_cc[cell]}")
    ax[i, 2].imshow(tc2[cell])
    ax[i, 2].set_title(f"2nd half: {cell}\n scc:{sorted_tc_cc[cell]}")

# plot 3 cells with highest spatial info and 3 with least
sorted_si = sorted(cell_si.items(), key=lambda x:x[1], reverse=True)
sorted_si = dict(sorted_si)

best_3_si = list(sorted_si.keys())[:3]
worst_3_si = list(sorted_si.keys())[-3:]
f, ax = plt.subplots(6,3, layout='tight', figsize=(7,10))
for i, cell in enumerate(best_3_si+worst_3_si):
    ax[i, 0].imshow(tc[cell])
    ax[i, 0].set_title(f"cell: {cell}\n si:{sorted_si[cell]}")
    ax[i, 1].imshow(tc1[cell])
    ax[i, 1].set_title(f"1st half: {cell}\n scc:{sorted_tc_cc[cell]}")
    ax[i, 2].imshow(tc2[cell])
    ax[i, 2].set_title(f"2nd half: {cell}\n scc:{sorted_tc_cc[cell]}")

xnew, ynew = dec.dlc_to_tcpos(rec, xy)
f, ax = plt.subplots(1,1)
ax.plot(xnew[:1000], ynew[:1000])
ax.plot(rec['dlc'][2][:1000], rec['dlc'][3][:1000])
ax.set_title("binned x/y pos vs actual x/y")

f, ax = plt.subplots(1,1)
ax.plot(xnew[:1000])
ax.plot(rec['dlc'][2][:1000])
ax.set_title("binned x vs actual x")

# run decoding and check similarity metrics cc vs euclidian
# decoded_pos, p = decode2d(rec, tc, xy)
#
# # plot all decoded data...do we see running pattern
# fig, ax = plt.subplots(1,1)
# ax.plot(decoded_pos[:, 0], decoded_pos[:, 1])
# ax.plot(xnew[:], ynew[:])
# ax.set_title("decoded pos and actual pos")

#TODO --- make single "trial" distance tuning ---
# create a 1d occupancy histogram and return bin centers
# def dist_occ_hist(rec, epochs, signal='dist', feature=0, bins = 40):

# for each cell generate a dist feature/firing rate tuning curve
# all_trials_plot(rec=rec, cells=best_3_ss, tartime=0.6, error='sem', tc2d=True)
# dvt_allcells_plot(rec=rec, tartime=0.6, error='sem')
#
# trial_tc = trial_2d_tc(rec=rec)
#
# f, ax = plt.subplots(6,4, layout='tight', figsize=(7,10))
# for i, cell in enumerate(best_3_ss+worst_3_ss):
#     ax[i, 0].imshow(tc[cell].T)
#     ax[i, 1].imshow(tc1[cell].T)
#     ax[i, 1].set_title(f"1st half: {cell}\n scc:{sorted_tc_cc[cell]}")
#     ax[i, 2].imshow(tc2[cell].T)
#     ax[i, 2].set_title(f"2nd half: {cell}\n scc:{sorted_tc_cc[cell]}")
#     ax[i, 3].imshow(trial_tc[cell].T)
#     ax[i, 3].set_title(f"trial data only {cell}")

def trial_dvst_design_matrix(rec, tbinwidth=0.2, tbin_num=10, dbin_num=10):
    """create a design matrix with binary values for time from target onset, distance, velocity and an offset"""
    # create target to lickspout epochs - trials
    tartime=0.6
    rasterfs = rec.meta['rasterfs']
    dec.dlc_within_radius(rec, target='Trial')
    hit_fa_epochs, tar_snrs = dec.trials_to_path(rec=rec, tartime=0.6)

    # grab target onset and lick spout entry epochs - tarlick - and catch fa - catchfa - from resp, dlc, dist
    tarlickeps = rec['resp'].extract_epoch([hit_fa_epochs['start'], hit_fa_epochs['end']])
    dlc_eps = (rec['dlc'].extract_epoch(hit_fa_epochs))
    dist_eps = (rec['dist'].extract_epoch(hit_fa_epochs))
    trial_time_bin_edges = np.concatenate((np.arange(-tartime*rasterfs, 0, int(tbinwidth*rasterfs)), np.arange(0, len(tarlickeps[0, 0, :]), int(tbinwidth*rasterfs))))
    if tbin_num:
        trial_time_bin_edges = trial_time_bin_edges[:10]
    dist_bin_edges = []
    design_mat = []

    return design_mat


design_mat = trial_dvst_design_matrix(rec=rec)


bp = []