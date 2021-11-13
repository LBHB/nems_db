from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np

from nems import db
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys
from nems_lbhb.plots import plot_waveforms_64D

parmfile = "/auto/data/daq/Teonancatl/TNC018/TNC018a16_p_BNB.m"
#parmfile = "/auto/data/daq/Teonancatl/TNC020/TNC020a11_p_BNB.m"
#parmfile = "/auto/data/daq/Teonancatl/TNC018/TNC018a03_p_BNB.m"


## load the recording
ex = BAPHYExperiment(parmfile=parmfile)
print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

rec = ex.get_recording(raw=True, resp=False, stim=False, recache=False, rawchans=None, rasterfs=400)


## extract traces for each stimulus event
ep = rec['raw'].epochs
eps = ep.loc[ep['name'].str.startswith("STIM"),'name']
epoch = ep.loc[ep['name'].str.startswith("STIM"),'name'].values[0]

r = rec['raw'].extract_epoch(epoch)
pre_ep = ep.loc[ep.name.str.startswith("PreStim")]
pre_stim_silence = pre_ep.iloc[0]['end']-pre_ep.iloc[0]['start']
epoch, pre_stim_silence, rec['raw'].fs


## plot raw traces
plt.figure()
s=rec['raw'].epochs.loc[rec['raw'].epochs['name']=='STIM_1265',['start','end']]
diff=(s['end']-s['start']).values

tt=np.arange(r.shape[2])/rec['raw'].fs-pre_stim_silence
plt.plot(tt, r.mean(axis=0).T) # - np.nanmean(np.nanmean(r,axis=1), axis=0,keepdims=True).T)
plt.xlabel('Time from stim onset (sec)')
plt.title(ex.parmfile[0].stem)


## quick & dirty CSD computation and plot

# Assume 64D configuration, need different geometry for 64M
left_ch_nums = np.arange(3,64,3)-1
right_ch_nums = np.arange(4,65,3)-1
center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)-1

r_mean = rec['raw'].extract_epoch(epoch).mean(axis=0)
csd = np.zeros_like(r_mean)
csd[left_ch_nums[1:-1]] = r_mean[left_ch_nums[1:-1]] - r_mean[left_ch_nums[:-2]]/2 - r_mean[left_ch_nums[2:]]/2
csd[right_ch_nums[1:-1]] = r_mean[right_ch_nums[1:-1]] - r_mean[right_ch_nums[:-2]]/2 - r_mean[right_ch_nums[2:]]/2
csd[center_ch_nums[1:-1]] = r_mean[center_ch_nums[1:-1]] - r_mean[center_ch_nums[:-2]]/2 - r_mean[center_ch_nums[2:]]/2

f,ax = plt.subplots(figsize=(4,8))
plot_waveforms_64D(csd, chans=rec['raw'].chans, norm=False, ax=ax)
ax.set_title(ex.parmfile[0].stem)
