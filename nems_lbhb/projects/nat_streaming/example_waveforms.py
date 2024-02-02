"""
example waveforms for BCP R01 proposal cartoon figures

"""
import os
import io
import logging
import time
import sys, importlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy


wav1 = '/auto/data/sounds/backgrounds/v1/cat31_rec1_blender_excerpt1.wav'
wav1 = '/auto/data/sounds/backgrounds/v1/cat312_rec1_wind_excerpt1.wav'
wav2 = '/auto/data/sounds/vocalizations/v1/ferret-b1-001R.wav'
wav2 = '/auto/data/sounds/vocalizations/v1/ferret-b3-006R.wav'
wav2 = '/auto/data/sounds/vocalizations/v1/ferret-b5-006R.wav'

outfs = 16000

fs1,w1 = scipy.io.wavfile.read(wav1)
w1 = w1.astype(float)
w1/=w1.std()
w1 = scipy.signal.resample(w1, outfs*4)
w1=w1[:outfs*3]

fs2,w2 = scipy.io.wavfile.read(wav2)
w2 = w2.astype(float)
w2 /= w2.std()
w2 = scipy.signal.resample(w2, outfs*3)

f,ax = plt.subplots(2,1,figsize=(1.5,3), sharey=True, sharex=True)

ax[0].plot(w1, lw=0.5, color="blue")
ax[1].plot(w2, lw=0.5, color="red")
ax[0].set_axis_off()
ax[1].set_axis_off()

f.savefig('/home/svd/Documents/onedrive/proposals/r01_bcp/figures/raw/example_waveforms.pdf')
