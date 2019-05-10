"""
functions for plotting rasters, STRF analysis of TORCs.

TODO port from baphy functions in matlab - strf_offline, cell_rasters

"""

import baphy

def raster_plot(mfilename, **options):

    pass

mfilename="/data/daq/Electra/ele150/ele150g02_p_TOR.m"
cellid = "ele150g-e1"

rec = baphy.baphy_load_recording_file(mfilename=mfilename, cellid=cellid, stim=False)


#rec=baphy.baphy_load_recording(mfilename=mfilename, cellid=cellid, stim=False)
