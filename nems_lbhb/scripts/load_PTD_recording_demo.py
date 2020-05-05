# load_PTD_recording_demo
#
# example of how to load a recording from the pure tone detect task and plot basic
# response & lick data

import numpy as np
import matplotlib.pyplot as plt
import os

import nems_lbhb.xform_wrappers as nw
import nems.plots.api as nplt
import nems.db as nd
from nems.xform_helper import load_model_xform
import nems.gui.editors as gui
from nems.recording import load_recording
from nems.preprocessing import generate_stim_from_epochs

modelname="psth.fs20.pup-ld-st.pup.beh-evs.tar.lic_fir.Nx40-lvl.1-stategain.3_jk.nf10-init.st-basic"
batch=307
cellid="TAR010c-09-1"

# "psth" = don't load stimulus spectrogram
# fs20 = 20 Hz sampling rate
# pup = load pupil trace
loadkey = "psth.fs20.pup"

# figure out filename of recording (& generate from raw data if necessary)
_r = nw.baphy_load_wrapper(cellid, batch, loadkey=loadkey)
uri = _r['recording_uri_list'][0]

# load it
rec = load_recording(uri)

# convert response signal from point process to raster
rec['resp'] = rec['resp'].rasterize()

# regular expression to match all stimulus events, ie, epochs starting with "STIM_"
epoch_regex="^STIM_"

# very specialized function to generate stimulus + lick signal
rec = generate_stim_from_epochs(rec=rec, new_signal_name='stim',
                                epoch_regex=epoch_regex, epoch_shift = 0,
                                epoch2_regex='LICK', epoch2_shift=0,
                                onsets_only=True)


rec = rec.create_mask('HIT_TRIAL')  # implied ACTIVE_EXPERIMENT
resp_active = rec['resp'].extract_epoch('STIM_500', mask=rec['mask']) * rec['resp'].fs

stim_active = rec['stim'].extract_epoch('STIM_500', mask=rec['mask'])
# lick is last channel of stim signal
lick_active = stim_active[:,-1,:]

rec = rec.create_mask('PASSIVE_EXPERIMENT')
resp_passive = rec['resp'].extract_epoch('STIM_500', mask=rec['mask']) * rec['resp'].fs

resp_active_psth = resp_active.mean(axis=0)
resp_passive_psth = resp_passive.mean(axis=0)#These have the same format as rec since they are just masked versions of rec
#dim [0] is the neurons, dim [1] is time points, and dim [2] is response (in spikes/s?)
lick_avg = lick_active.mean(axis=0)

max_resp = np.max(np.concatenate((resp_active_psth,resp_passive_psth), axis=0))

n_neurons = rec['resp'].shape[0]
tar_duration = resp_active_psth.shape[1] / rec['resp'].fs



#Plotting code
f, ax = plt.subplots(3,1)
im = ax[0].imshow(resp_passive_psth, clim=[0,max_resp], extent=[0, tar_duration, 1, n_neurons],
             aspect='auto')
ax[0].set_title('Passive target resp')
ax[0].set_ylabel('Neuron');
f.colorbar(im, ax=ax[0])

im = ax[1].imshow(resp_active_psth, clim=[0,max_resp], extent=[0, tar_duration, 1, n_neurons],
             aspect='auto')
ax[1].set_title('Active target response')

ax[2].plot(np.arange(resp_active_psth.shape[1])/rec['resp'].fs, lick_avg)
ax[2].set_title('Average lick rate')
ax[2].set_xlabel('Time (s)')

path=f'/auto/users/culpa/NEMS/results/demo/{cellid}'
os.makedirs(path)
f.savefig(path+'/figure.0000.png')
