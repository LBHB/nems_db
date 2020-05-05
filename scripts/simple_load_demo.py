#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:19:24 2018

@author: svd
"""

from nems.recording import Recording
import nems.signal as signal
import nems.epoch as ep
import numpy as np
import matplotlib.pyplot as plt

uri="/auto/data/nems_db/recordings/271/ozgf18_fs100/BRT026c-17-3.tgz"
rec=Recording.load(uri)

epoch_regex='^STIM_'
epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
resp_dict=rec['resp'].extract_epochs(epochs_to_extract)
epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
stim_dict=rec['stim'].extract_epochs(epochs_to_extract)

per_stim_psth = dict()
per_stim_repcount = dict()
stim=dict()
for k in resp_dict.keys():
    per_stim_psth[k] = np.nanmean(resp_dict[k], axis=0)
    per_stim_repcount[k] = resp_dict[k].shape[0]
    stim[k] = stim_dict[k][0,:,:]
    
k='STIM_ASE-04_Frying_Eggs.wav'

plt.figure()
plt.subplot(3,1,1)
plt.imshow(stim[k],aspect='auto')
plt.subplot(3,1,2)
plt.imshow(resp_dict[k][:,0,:],aspect='auto')
plt.subplot(3,1,3)
plt.plot(per_stim_psth[k].T)
