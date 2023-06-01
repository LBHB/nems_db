#stardard imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
import zarr
from scipy.io import wavfile
from scipy.signal import resample

#for PCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb import baphy_io
import nems_lbhb.plots as nplt
from nems0.analysis.gammatone.gtgram import gtgram
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems0.utils import smooth

# stuff from nems_lbhb.runclass:
from scipy import interpolate
from scipy.signal import hilbert, resample
from scipy.io import wavfile
import matplotlib.pyplot as plt
from nems0.analysis.gammatone.gtgram import gtgram
from nems0.analysis.gammatone.filters import centre_freqs
from nems_lbhb.projects.freemoving.free_tools import load_hrtf
from nems_lbhb.projects.freemoving import free_model
from nems0.modules.nonlinearity import _dlog
from nems0 import db
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems0.recording import load_recording
from nems_lbhb.preprocessing import impute_multi

rasterfs = 50
f_min = 200
f_max = 20000
channels = 18
imopts = {'origin': 'lower', 'interpolation': 'none',
          'cmap': 'gray_r', 'aspect': 'auto'}

batch = 346
siteids, cellids = db.get_batch_sites(346)

#siteid = 'PRN047a'
siteid = 'PRN015a'

siteid = 'PRN048a' # some of everything.
siteid = 'PRN051a' # some of everything.

d=get_spike_info(siteid=siteid, save_to_db=True)

dlc_chans=8
rasterfs=50
batch=348
rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans)



recording_uri = generate_recording_uri(batch=batch, loadkey='psth.fs100.dlc', cellid=siteid)
rec = load_recording(recording_uri)

dlc_chans = 8
rec = impute_multi(rec, sig='dlc', empty_values=np.nan, keep_dims=dlc_chans)['rec']

df_siteinfo = get_spike_info(siteid=siteid, save_to_db=False)
a1cellids = df_siteinfo.loc[(df_siteinfo['area'] == 'A1') | (df_siteinfo['area'] == 'BS') |
                            (df_siteinfo['area'] == 'PEG')].index.to_list()

rec['resp']=rec['resp'].rasterize()

e='STIM_00seq5_hand.wav'
epochs= rec['resp'].epochs.copy()
epochs['dur'] = epochs['end']-epochs['start']
epochs=epochs.loc[epochs['name']==e]
sh = epochs['dur']<20

epochs.loc[sh,'start'] = epochs.loc[sh,'start']-1.0
epochs.loc[sh,'end'] = epochs.loc[sh,'end']+1.0
resp = rec['resp'].copy()
resp.epochs=epochs

r=resp.extract_epoch(e, fix_overlap=None)
pfree = np.nanmean(r[sh],axis=0)
pfixed = np.nanmean(r[~sh],axis=0)
f, ax = plt.subplots(2,1, sharex=True)
ax[0].imshow(pfree, **imopts)
ax[1].imshow(pfixed, **imopts)

cid=13
f, ax = plt.subplots(2,1, sharex=True)
ax[0].imshow(r[:,cid,:], **imopts)
ax[1].plot(np.nanmean(r[sh,cid,:], axis=0), label='free')
ax[1].plot(np.nanmean(r[~sh,cid,:], axis=0), label='fixed')
ax[1].legend()

dlc_data = rec['dlc'].as_continuous().copy()
dlc_valid = np.sum(np.isfinite(dlc_data), axis=0) > 0

speaker1_x0y0 = 1.0, -0.8
speaker2_x0y0 = 0.0, -0.8
smooth_win=2
d1, theta1, vel, rvel, d_fwd, d_lat = compute_d_theta(
    dlc_data, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker1_x0y0)
d2, theta2, vel, rvel, d_fwd, d_lat = compute_d_theta(
    dlc_data, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker2_x0y0)

dlc_cartoon = np.array([0.5, 0.2, ])


fwidx=np.where((np.abs(dlc_data[0,:]-dlc_data[2,:])<0.005) &
               (dlc_data[1, :] < dlc_data[3, :]-0.03) &
               (np.abs(dlc_data[6,:]-dlc_data[4,:]-0.08)<0.02) &
               (np.abs(dlc_data[5,:]-dlc_data[7,:])<0.02))[0]

# &
#               (np.abs(dlc_data[6,:]-dlc_data[4,:]-18)<5)
plt.close('all')

plt.figure()
plt.scatter(dlc_data[0,::100],dlc_data[1,::100], s=1, color='lightgray')
#plt.plot(dlc_data[0,:240],dlc_data[1,:240], color='red')
#plt.plot(dlc_data[0,800:1180],dlc_data[1,800:1180], color='red')
colors=['red','blue','gray','gray','gray','gray']
for t in fwidx[::100]:
    for j in range(2):
        plt.scatter(dlc_data[j*2,t], dlc_data[j*2+1,t], s=2, color=colors[j])
    plt.plot(dlc_data[[0,2],t], dlc_data[[1,3],t],color='black')
    plt.plot(dlc_data[[4,6],t], dlc_data[[5,7],t],color='gray')

d = dlc_data[:,fwidx]
d[::2,:] += -d[[0],:] + 0.5
d[1::2,:] += -d[[1],:] + 0.2
np.mean(d, axis=1)

plt.figure()

for t in fwidx[::50]:
    xadj = -dlc_data[0,t]+0.5
    yadj = -dlc_data[1,t]+0.2
    for j in range(2):
        plt.scatter(dlc_data[j*2,t] + xadj,
                    dlc_data[j*2+1,t] + yadj, s=2, color=colors[j])
    plt.plot(dlc_data[[0,2],t] + xadj, dlc_data[[1,3],t] + yadj,color='black')
    plt.plot(dlc_data[[4,6],t] + xadj, dlc_data[[5,7],t] + yadj,color='gray')

md=np.mean(d, axis=1)
for j in range(2):
    plt.scatter(md[j*2], md[j*2+1], s=2, color=colors[j])
plt.plot(md[[0,2]], md[[1,3]],color='red')
plt.plot(md[[4,6]], md[[5,7]],color='red')

np.mean(dlc_data[:,fwidx], axis=1)
plt.hist(dlc_data[6,fwidx]-dlc_data[4,fwidx],bins=100)

f,ax=plt.subplots(2,1)
ax[0].plot(d1[0,:1000])
ax[0].plot(d2[0,:1000])
ax[1].plot(theta1[0,:1000])
ax[1].plot(theta2[0,:1000])
