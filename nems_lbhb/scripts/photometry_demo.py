import nems_lbhb.io as io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
import scipy.ndimage.filters as snf

fp = '/auto/data/daq/Amanita/AMT039/'
fn = 'AMT039e02_p_VOC'
pup_fn = fp + 'sorted/' + fn + '.pickle'
photo_fn = fp + fn + '.photo.avi'

pup = io.load_raw_pupil(pup_fn)
photo1, photo2 = io.load_raw_photometry(photo_fn)

# choose signal length and force both signals to this to "align"
n_samps = int(pup.shape[0] / 10)

pup = ss.resample(pup[100:-100], n_samps)
# remove low peaks of photo signal
photo1 = pd.Series(photo1.squeeze()).rolling(5).max().dropna().values[100:-100]
photo1 = ss.resample(photo1, n_samps)

photo2 = pd.Series(photo2.squeeze()).rolling(5).max().dropna().values[100:-100]
photo2 = ss.resample(photo2, n_samps)

# fit a exponential decay to regress out the bleaching shift
x = np.arange(0, len(photo1))
def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

parms, pcov = curve_fit(exp, x, photo1, np.array([np.max(photo1), 0.001, 0]))
photo1_pred = exp(x, parms[0], parms[1], parms[2])
parms, pcov = curve_fit(exp, x, photo2, np.array([np.max(photo2), 0.001, 0]))
photo2_pred = exp(x, parms[0], parms[1], parms[2])

sigma = 5
f, ax = plt.subplots(4, 1, sharex=True)

ax[0].plot(pup, color='blue', lw=2)
ax[0].set_ylabel("pupil size")

ax[1].set_title("baseline corrected, smoothed")
ax[1].plot(snf.gaussian_filter1d(photo1-photo1_pred, sigma), color='r', lw=2, label='left roi')
ax[1].plot(snf.gaussian_filter1d(photo2-photo2_pred, sigma), color='b', lw=2, label='right roi')
ax[1].set_ylabel("abs(fluorescence)")

ax[2].set_title("baseline corrected")
ax[2].plot(photo1-photo1_pred, color='grey', lw=2)
ax[2].plot(photo2-photo2_pred, color='grey', lw=2)
ax[2].plot(snf.gaussian_filter1d(photo1-photo1_pred, sigma), color='r', lw=2, label='left roi')
ax[2].plot(snf.gaussian_filter1d(photo2-photo2_pred, sigma), color='b', lw=2, label='right roi')
ax[2].set_ylabel("abs(fluorescence)")

ax[3].set_title("raw")
ax[3].plot(photo1, color='grey', lw=2)
ax[3].plot(photo1_pred, color='r', lw=2, label='left roi')
ax[3].plot(photo2, color='grey', lw=2)
ax[3].plot(photo2_pred, color='b', lw=2,  label='right roi')
ax[3].legend()
ax[3].set_ylabel("abs(fluorescence)")

plt.show()
