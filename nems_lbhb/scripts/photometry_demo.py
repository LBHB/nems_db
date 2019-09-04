import nems_lbhb.io as io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
import scipy.ndimage.filters as snf

fp = '/auto/data/daq/Amanita/AMT039/'
fn = 'AMT039c05_p_VOC'
pup_fn = fp + 'sorted/' + fn + '.pickle'
photo_fn = fp + fn + '.photo.avi'

pup = io.load_raw_pupil(pup_fn)
photo = io.load_raw_photometry(photo_fn)

# choose signal length and force both signals to this to "align"
n_samps = int(pup.shape[0] / 10)

pup = ss.resample(pup[10:-10], n_samps)
# remove low peaks of photo signal
photo = pd.Series(photo.squeeze()).rolling(5).max().dropna().values[10:-10]
photo = ss.resample(photo, n_samps)

# fit a exponential decay to regress out the bleaching shift
x = np.arange(0, len(photo))
def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

parms, pcov = curve_fit(exp, x, photo, np.array([np.max(photo), 0.001, 0]))

photo_pred = exp(x, parms[0], parms[1], parms[2])

sigma = 5
f, ax = plt.subplots(4, 1, sharex=True)

ax[0].plot(pup, color='blue', lw=2)
ax[0].set_ylabel("pupil size")

ax[1].set_title("baseline corrected, smoothed")
ax[1].plot(snf.gaussian_filter1d(photo-photo_pred, sigma), color='k', lw=2)
ax[1].set_ylabel("abs(fluorescence)")

ax[2].set_title("baseline corrected")
ax[2].plot(photo-photo_pred, color='limegreen', lw=2)
ax[2].plot(snf.gaussian_filter1d(photo-photo_pred, sigma), color='k', lw=2)
ax[2].set_ylabel("abs(fluorescence)")

ax[3].set_title("raw")
ax[3].plot(photo, color='limegreen', lw=2)
ax[3].plot(photo_pred, color='k', lw=2)
ax[3].legend(['raw f', 'baseline drift'])
ax[3].set_ylabel("abs(fluorescence)")

plt.show()
