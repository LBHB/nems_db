"""
functions for plotting rasters, STRF analysis of TORCs.

TODO port from baphy functions in matlab - strf_offline, cell_rasters

"""
import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.baphy as baphy
import nems.db as nd
import pandas as pd
from matplotlib import cm
from nems_lbhb.baphy import baphy_load_recording_file, baphy_load_recording
from nems_lbhb.io import baphy_parm_read
import nems.epoch as ep
import matplotlib

def raster_plot(mfilename, ax=None, epoch_regex="REFERENCE", signal="resp",
                cellid=None, fs=1000, **options):
    """
    :param mfilename:
    :param options: cellid or channel, unit
        rasterfs : int
        psth : boolean
        tag_masks : string or list of regexs that have to match epoch names,
        signal : "resp" by default
    :return:
    """

    rec = baphy_load_recording_file(mfilename=mfilename, cellid=cellid,
                                    fs=fs, stim=False)

    resp = rec[signal]

    fs = resp.fs
    epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)

    d = resp.get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d)) - 0.5/fs
    dur = resp.get_epoch_bounds(epochs_to_extract[0])
    FullDuration = np.mean(np.diff(dur)) - 0.5/fs
    d = resp.get_epoch_bounds('PostStimSilence')
    if d.size > 0:
        PostStimSilence = np.min(np.diff(d)) - 0.5/fs
        dd = np.diff(d)
        dd = dd[dd > 0]
    else:
        dd = np.array([])
    if dd.size > 0:
        PostStimSilence = np.min(dd) - 0.5/fs
    else:
        PostStimSilence = 0
    Duration = FullDuration - PostStimSilence - PreStimSilence

    r = resp.extract_epochs(epochs_to_extract)

    colorset = ((0,0,0),(0.6,0.6,0.6))
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    if cellid is None:
        cellid = list(r.keys())[0]

    d = np.zeros((0,2))
    t0=0
    for i,k in enumerate(list(r.keys())):
        print("{} {}".format(k, t0))
        _r = r[k][cellid].copy()
        _r[:, 0] += t0
        d = np.concatenate((d, _r), axis=0)
        t0 = np.max(d[:, 0])+1

        ax.plot(_r[:, 1],_r[:, 0],'.', color=colorset[i % 2],
                markersize=3)

    ylim = ax.get_ylim()
    ax.plot(np.array([1,1])*PreStimSilence, ylim, 'g--')
    ax.plot(np.array([1,1])*(PreStimSilence+Duration), ylim, 'g--')

    ax.set_title(cellid)

    return ax

'''
mfilename="/auto/data/daq/Electra/ele150/ele150g02_p_TOR.m"
cellid = "ele150g-e1"
epoch_regex = "^STIM_"
epoch_regex = "REFERENCE"
signal = "resp"

ax = raster_plot(mfilename, cellid=cellid, epoch_regex=epoch_regex)
'''

#rec = baphy_load_recording_file(mfilename=mfilename, cellid=cellid, stim=False)

#rec = baphy.baphy_load_recording_file(mfilename=mfilename, cellid=cellid, stim=False)

#globalparams, exptparams, exptevents = baphy_parm_read(mfilename)

#rec=baphy.baphy_load_recording(mfilename=mfilename, cellid=cellid, stim=False)

def plot_topo_map(pendata, vmin=None, vmax=None):
    """
    Given a dict of cellids and bfs, plot a topographical map of tuning.
    Depends on penetrations having been marked in celldb/with baphy
    """
    cellids = list(pendata.keys())
    bf = list(pendata.values())
    # make sure cellids / bf stay sorted the same
    asort = np.argsort(cellids)
    bf = np.array(bf)[asort]
    cellids = np.array(cellids)[asort]

    # get unique pennames
    pennames = [c[:6] for c in cellids]
    pennames = np.unique(pennames)
    for i, p in enumerate(pennames):
        cells = [c for c in cellids if p in c]
        if i == 0:
            xy = nd.get_pen_location(cells)
        else:
            _xy = nd.get_pen_location(cells)
            xy = pd.concat([xy, _xy], axis=0)

    if xy.shape[0] != len(bf):
        raise ValueError("Number of returned positions does not match len(bf)")

    xy['bf'] = bf

    cmap = cm.get_cmap('jet')
    # Force range between 0 Hz and 24 kHz
    if vmax is None:
        vmax=24000
    if vmin is None:
        vmin = 0.1
    f, ax = plt.subplots(1, 1)
    # only plot color for non-nan sites:
    bf = xy['bf'].values
    inds = np.argwhere(~np.isnan(bf)).squeeze()
    bf = bf[inds]
    x_val = xy['x'].values[inds]
    y_val = xy['y'].values[inds]
    im = ax.scatter(x_val, y_val, c=bf,
                     vmin=vmin, vmax=vmax, cmap=cmap, s=500, edgecolors='white',
                     norm=matplotlib.colors.LogNorm())
    for r in range(xy.shape[0]):
        ax.annotate(xy.index[r], (xy['x'].values[r], xy['y'].values[r]))
    ax.invert_yaxis()
    plt.colorbar(im)

    f.tight_layout()
