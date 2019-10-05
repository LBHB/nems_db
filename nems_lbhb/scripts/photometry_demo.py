import nems_lbhb.io as io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
import scipy.ndimage.filters as snf

fp = '/auto/data/daq/Amanita/AMT039/'
fn = 'AMT039b04_p_VOC'
fp = '/auto/data/daq/Leyva/ley077/'
fn = 'ley077b03_p_VOC'
pup_fn = fp + 'sorted/' + fn + '.pickle'
photo_fn = fp + fn + '.photo.avi'

pup = io.load_raw_pupil(pup_fn)
photo1, photo2 = io.load_raw_photometry(photo_fn, framen=500)

# choose signal length and force both signals to this to "align"
n_samps = int(pup.shape[0] / 10)

pup = ss.resample(pup[100:-100], n_samps)

# get rid of very small values
photo1 = photo1[photo1[:, 0]>50, :]

# get upper (470nm) signal and lower (iso/415nm) signal ny finding peaks/extrema
upper_peaks, _ = ss.find_peaks(photo1.squeeze()- photo1.mean(), height = 0, distance=10)
lower_peaks, _ = ss.find_peaks(-(photo1.squeeze()- photo1.mean()), height = 0, distance=10)

if lower_peaks.shape[0] != upper_peaks.shape[0]:
    if (upper_peaks[0] > lower_peaks[0]) & (lower_peaks.shape[0] > upper_peaks.shape[0]):
        lower_peaks = [lower_peaks[i] for i, u in enumerate(upper_peaks) if lower_peaks[i] < u]
    elif (upper_peaks[0] > lower_peaks[0]) & (lower_peaks.shape[0] < upper_peaks.shape[0]):
        upper_peaks = [upper_peaks[i] for i, l in enumerate(lower_peaks) if l < upper_peaks[i]]
    elif (upper_peaks[0] < lower_peaks[0]) & (lower_peaks.shape[0] > upper_peaks.shape[0]):
        lower_peaks = [lower_peaks[i] for i, u in enumerate(upper_peaks) if lower_peaks[i] > u]
    elif (upper_peaks[0] < lower_peaks[0]) & (lower_peaks.shape[0] < upper_peaks.shape[0]):
        upper_peaks = [upper_peaks[i] for i, l in enumerate(lower_peaks) if l > upper_peaks[i]]
    
# get peaks
snfr_signal = photo1[upper_peaks].squeeze()
iso_signal = photo1[lower_peaks].squeeze()


# subtract the iso from the snfr to normalize for bleaching autofluorescence/movement etc.
snfr_signal_iso_norm = snfr_signal - iso_signal

# fit a exponential decay to regress out the bleaching shift
x = np.arange(0, len(snfr_signal_iso_norm))
def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

parms, pcov = curve_fit(exp, x, snfr_signal_iso_norm, np.array([np.max(snfr_signal_iso_norm), 0.001, 0]))
pred = exp(x, parms[0], parms[1], parms[2])

snfr_norm = snfr_signal_iso_norm - pred

f, ax = plt.subplots(5, 1)

ax[0].plot(pup, color='k', label='pupil')
ax[0].legend()

ax[1].plot(iso_signal, color='red', label='iso')
ax[1].legend()

ax[2].plot(snfr_signal, color='green', label='snfr')
ax[2].legend()

ax[3].plot(snfr_signal_iso_norm, color='green', label='iso correction')
ax[3].plot(pred, 'k', label='bleach correction factor')
ax[3].legend()

ax[4].plot(snf.gaussian_filter1d(snfr_norm, 10), color='green', label='iso + bleach correction')
ax[4].legend()

plt.show()



'''

# fit a exponential decay to regress out the bleaching shift
x = np.arange(0, len(photo1))
def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

parms, pcov = curve_fit(exp, x, photo1, np.array([np.mean(photo1), 0.001, 0]))
photo1_pred = exp(x, parms[0], parms[1], parms[2])

# get top half
photo_norm = photo1.squeeze() - photo1_pred
photo_thresh = photo_norm[photo_norm>0]
photo_thresh = [p for i, p in enumerate(photo_thresh) if photo_thresh[i-1]]

sigma = 1
f, ax = plt.subplots(3, 1, sharex=True)

ax[0].plot(pup, color='blue', lw=2)
ax[0].set_ylabel("pupil size")

ax[1].set_title("baseline corrected, smoothed")
ax[1].plot(snf.gaussian_filter1d(photo1-photo1_pred, sigma), color='r', lw=2, label='SnFR')
ax[1].legend()
ax[1].set_ylabel("abs(fluorescence)")

ax[2].plot(snf.gaussian_filter1d(photo2-photo2_pred, sigma), color='b', lw=2, label='iso')
ax[2].set_ylabel("abs(fluorescence)")
ax[2].legend()

f.suptitle(fn)

f.tight_layout()

f, ax = plt.subplots(3, 1, sharex=True)
ax[0].set_title("baseline corrected")
ax[0].plot(photo1-photo1_pred, color='grey', lw=2)
ax[0].plot(photo2-photo2_pred, color='grey', lw=2)
ax[0].plot(snf.gaussian_filter1d(photo1-photo1_pred, sigma), color='r', lw=2, label='SnFR')
ax[0].plot(snf.gaussian_filter1d(photo2-photo2_pred, sigma), color='b', lw=2, label='iso')
ax[0].set_ylabel("abs(fluorescence)")

ax[1].set_title("raw")
ax[1].plot(photo1, color='grey', lw=2)
ax[1].plot(photo1_pred, color='r', lw=2, label='SnFR')
ax[1].legend()
ax[1].set_ylabel("abs(fluorescence)")


ax[2].plot(photo2, color='grey', lw=2)
ax[2].plot(photo2_pred, color='b', lw=2,  label='iso')
ax[2].legend()
ax[2].set_ylabel("abs(fluorescence)")

f.suptitle(fn)

f.tight_layout()

plt.show()
'''
