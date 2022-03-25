
import numpy as np
import matplotlib.pyplot as plt

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.baphy_io as baphy_io

parmfile = '/auto/data/daq/Clathrus/training2022/Clathrus_2022_01_11_TBP_1.m'
#dlcfilepath = '/auto/data/daq/Clathrus/training2022/sorted/Clathrus_2022_01_11_TBP_1.lickDLC_resnet50_multividJan14shuffle1_1030000.h5'
parmfile = '/auto/data/daq/Clathrus/CLT011/CLT011a05_a_TBP.m'

experiment = BAPHYExperiment(parmfile=parmfile)
rec = experiment.get_recording(rasterfs=30, recache=True, dlc=True,
                               resp=True, stim=False, dlc_threshold=0.25)

# find lick events
dlc = rec['dlc']
show_sec = 500

lick_events = dlc.epochs.loc[(dlc.epochs.name=='LICK') & (dlc.epochs.start<show_sec)].copy()
lick_frames = (lick_events['start'].values * dlc.fs).astype(int)

chans = dlc.chans
plt.figure()
for i in range(0,len(chans),2):
    plt.plot(dlc[i,:],-dlc[i+1,:], label=chans[i].replace("_x",""))

# whisker (x,y) : (12,13)
# tounge (x,y) : (8,9)
# chin (x,y) : (6,7)
plt.plot(dlc[12, lick_frames], -dlc[13, lick_frames], '.', markersize=3, color='k')
plt.plot(dlc[10, lick_frames], -dlc[11, lick_frames], '.', markersize=3, color='k')
plt.plot(dlc[8, lick_frames], -dlc[9, lick_frames], '.', markersize=3, color='k')
plt.plot(dlc[6, lick_frames], -dlc[7, lick_frames], '.', markersize=3, color='k')
plt.plot(dlc[4, lick_frames], -dlc[5, lick_frames], '.', markersize=3, color='k')
plt.plot(dlc[2, lick_frames], -dlc[3, lick_frames], '.', markersize=3, color='k')
plt.plot(dlc[0, lick_frames], -dlc[1, lick_frames], '.', markersize=3, color='k')
plt.legend()

# get list of trial outcomes
dlc.epochs.loc[dlc.epochs['name'].str.endswith("_TRIAL")]


# plot lick times on top of raw dlc trace
show_frames=int(show_sec*dlc.fs)

# find lick events
lick_events = dlc.epochs.loc[(dlc.epochs.name=='LICK') & (dlc.epochs.start<show_sec)].copy()

plt.figure()
t=np.arange(show_frames)/dlc.fs
plt.plot(t,dlc[12,:int(show_sec*dlc.fs)])

lick_events['y']=200
plt.plot(lick_events['start'],lick_events['y'],'r.')

