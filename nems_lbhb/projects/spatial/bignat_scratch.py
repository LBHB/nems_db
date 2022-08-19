import numpy as np
import os
import io
import logging
import time
import matplotlib.pyplot as plt
import sys, importlib

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join, smooth
from nems import get_setting
from nems.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import POP_MODELS, SIG_TEST_MODELS
from nems import db
import nems.plots.api as nplt
import nems.epoch as ep
from nems_lbhb.baphy_experiment import BAPHYExperiment


batch=338
cellid="CLT013a-031-3"
siteid="CLT013a"

## load the recording
ex = BAPHYExperiment(batch=338, cellid=cellid)
print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

rec=ex.get_recording(loadkey='gtgram.ch18.fs100')

epochs = rec.epochs
is_stim = epochs['name'].str.startswith('STIM')
stim_epoch_list = epochs[is_stim]['name'].tolist()
stim_epochs = list(set(stim_epoch_list))
stim_epochs.sort()

epoch = stim_epochs[3]
resp=rec['resp'].rasterize()
stim=rec['stim'].rasterize()

# here's how you get rasters aligned to each stimulus:
# define regex for stimulus epochs
epoch_regex = '^STIM_'
epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
folded_resp = resp.extract_epochs(epochs_to_extract)

print('Response to each stimulus extracted to folded_resp dictionary')
plt.close('all')
for i in [4,5,7,13]:
    epoch = epochs_to_extract[i]
    # or just plot the PSTH for an example stimulus
    raster = resp.extract_epoch(epoch)
    psth = np.mean(raster, axis=0)
    spec = stim.extract_epoch(epoch)[0,:,:]

    norm_psth = psth - np.mean(psth,axis=1,keepdims=True)
    norm_psth /= np.std(psth, axis=1,keepdims=True)

    f,ax = plt.subplots(3,1, sharex=True)
    PreStimSilence=0.25

    nplt.plot_spectrogram(spec, fs=resp.fs, ax=ax[0], time_offset=PreStimSilence, title=epoch, cmap='gray_r')

    nplt.plot_spectrogram(psth, fs=resp.fs, ax=ax[1], time_offset=PreStimSilence, cmap='gray_r')

    PreStimSilence=0.25
    tt=np.arange(psth.shape[1])/resp.fs-PreStimSilence
    rasterfs = resp.fs

    ax[2].plot(tt,psth[25,:]*rasterfs, color='black', lw=1, label=resp.chans[25])
    ax[2].plot(tt,psth[27,:]*rasterfs, color='gray', lw=1, label=resp.chans[27])
    ax[2].set_title('sc={:.3f}'.format(sc[25, 27]))
    ax[2].set_ylabel('Spikes/sec')
    ax[2].set_ylim([0,75])
    ax[2].set_xlabel('Time')
    plt.legend()
    #fig.savefig("/Users/svd/Documents/current/ohrc_data_club_2019/example_psth_high_sc.pdf")
