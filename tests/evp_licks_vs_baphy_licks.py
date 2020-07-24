"""
Compare lick detection using evp file vs. lick events saved by baphy
"""
import nems_lbhb.io as io
from pathlib import Path

mfile = Path('/auto/data/daq/Tartufo/TAR010/TAR010c09_a_PTD.m')

# evp method
_, _, evp_events = io.baphy_parm_read(mfile, evpread=True)
evp_licks = evp_events[evp_events.name=='LICK']

# baphy events
_, _, baphy_events = io.baphy_parm_read(mfile, evpread=False)
baphy_licks = baphy_events[baphy_events.name=='LICK']