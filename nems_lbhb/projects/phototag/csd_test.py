from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from nems import db
from nems.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys
from nems_lbhb.plots import plot_waveforms_64D

USE_DB = False

if USE_DB:
    expt_name = "TNC020a11_p_BNB"
    expt_name = "TNC017a10_p_BNB"
    dparm = db.pd_query(f"SELECT * FROM gDataRaw where parmfile like '{expt_name}%'")
    parmfile = dparm.resppath[0] + dparm.parmfile[0]
else:
    #hard-code path to parmfile
    #parmfile = "/auto/data/daq/Teonancatl/TNC018/TNC018a16_p_BNB.m"
    #parmfile = "/auto/data/daq/Teonancatl/TNC020/TNC020a11_p_BNB.m"
    parmfile = "/auto/data/daq/Teonancatl/TNC017/TNC017a03_p_BNB.m"
    parmfile = "/auto/data/daq/Teonancatl/TNC017/TNC017a10_p_BNB.m"
    parmfile = "/auto/data/daq/Teonancatl/TNC016/TNC016a03_p_BNB.m"
    parmfile = "/auto/data/daq/Teonancatl/TNC018/TNC018a03_p_BNB.m"
    parmfile = "/auto/data/daq/Tartufo/TAR010/TAR010a03_p_BNB.m"


## load the recording
ex = BAPHYExperiment(parmfile=parmfile)
print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

rec = ex.get_recording(raw=True, resp=False, stim=False, recache=False, rawchans=None, rasterfs=400)

print('smoothing...')
raw_data=rec['raw']._data.copy()
for i in range(raw_data.shape[0]):
    raw_data[i, :] = smooth(raw_data[i,:], 5)

rec['smoothed'] = rec['raw']._modified_copy(data=raw_data)

## extract traces for each stimulus event
ep = rec['raw'].epochs
eps = ep.loc[ep['name'].str.startswith("STIM"),'name']
epoch = ep.loc[ep['name'].str.startswith("STIM"),'name'].values[0]

r = rec['smoothed'].extract_epoch(epoch)
pre_ep = ep.loc[ep.name.str.startswith("PreStim")]
pre_stim_silence = pre_ep.iloc[0]['end']-pre_ep.iloc[0]['start']
epoch, pre_stim_silence, rec['raw'].fs


## plot raw traces
s = rec['raw'].epochs.loc[rec['raw'].epochs['name']==epoch, ['start','end']]
diff = (s['end']-s['start']).values

tt = np.arange(r.shape[2]) / rec['raw'].fs-pre_stim_silence
r_mean = r.mean(axis=0)

f, ax = plt.subplots(figsize=(4, 8))
plot_waveforms_64D(r_mean, chans=rec['raw'].chans, norm=False, ax=ax)
ax.set_title(ex.parmfile[0].stem + " raw trace")


## quick & dirty CSD computation and plot

# Assume 64D configuration, need different geometry for 64M
left_ch_nums = np.arange(3,64,3)-1
right_ch_nums = np.arange(4,65,3)-1
center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)-1

#spatial_filter = np.array([[0.05], [0.15], [0.7], [0.15], [0.05]])
spatial_filter = np.hanning(3)[:, np.newaxis]
spatial_filter = spatial_filter / spatial_filter.sum()

csd = np.zeros_like(r_mean)

csd_ = r_mean[left_ch_nums]
csd_ = convolve2d(csd_, spatial_filter, mode='same', boundary='symm')
csd[left_ch_nums[1:-1]] = csd_[1:-1] - csd_[:-2]/2 - csd_[2:]/2

csd_ = r_mean[right_ch_nums]
csd_ = convolve2d(csd_, spatial_filter, mode='same', boundary='symm')
csd[right_ch_nums[1:-1]] = csd_[1:-1] - csd_[:-2]/2 - csd_[2:]/2

csd_ = r_mean[center_ch_nums]
csd_ = convolve2d(csd_, spatial_filter, mode='same', boundary='symm')
csd[center_ch_nums[1:-1]] = csd_[1:-1] - csd_[:-2]/2 - csd_[2:]/2

f,ax = plt.subplots(figsize=(4, 8), sharey=True)
plot_waveforms_64D(csd, chans=rec['raw'].chans, norm=False, ax=ax)
ax.set_title(ex.parmfile[0].stem + " CSD")
