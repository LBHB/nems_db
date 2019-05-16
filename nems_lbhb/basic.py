"""
functions for plotting rasters, STRF analysis of TORCs.

TODO port from baphy functions in matlab - strf_offline, cell_rasters

"""

import nems_lbhb.baphy as baphy
import nems.db as nd
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

def raster_plot(mfilename, **options):

    pass

mfilename="/data/daq/Electra/ele150/ele150g02_p_TOR.m"
cellid = "ele150g-e1"

#rec = baphy.baphy_load_recording_file(mfilename=mfilename, cellid=cellid, stim=False)


#rec=baphy.baphy_load_recording(mfilename=mfilename, cellid=cellid, stim=False)


def plot_topo_map(pendata, vmax=None):
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
    f, ax = plt.subplots(1, 1)
    im = ax.scatter(xy['x'].values, xy['y'].values, c=xy['bf'].values,
                     vmin=0, vmax=vmax, cmap=cmap, s=500, edgecolors='white')
    for r in range(xy.shape[0]):
        ax.annotate(xy.index[r], (xy['x'].values[r], xy['y'].values[r]))
    ax.invert_yaxis()
    plt.colorbar(im)

    f.tight_layout()
