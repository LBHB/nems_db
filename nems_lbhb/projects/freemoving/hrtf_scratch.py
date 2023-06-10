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
from nems0.utils import smooth

# stuff from nems_lbhb.runclass:
from scipy import interpolate
from scipy.signal import hilbert, resample
from scipy.io import wavfile
import matplotlib.pyplot as plt
from nems0.analysis.gammatone.gtgram import gtgram
from nems0.analysis.gammatone.filters import centre_freqs
from nems_lbhb.projects.freemoving import free_model
from nems0.modules.nonlinearity import _dlog
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist, load_hrtf, stim_filt_hrtf


def stim_filt_hrtf_DELME(rec, hrtf_format='az', smooth_win=2,
                   f_min=200, f_max=20000, channels=None):

    # require (stacked) binaural stim
    if channels is None:
        channels = int(rec['stim'].shape[0]/2)
    rasterfs = rec['stim'].fs
    stimcount = int(rec['stim'].shape[0]/channels)
    print(f"HRTF: stim is {channels} x {stimcount}")

    L0, R0, c, A = load_hrtf(format=hrtf_format, fmin=f_min, fmax=f_max, num_freqs=channels)

    # assume dlc has already been imputed and normalized to (0,1)
    dlc_data_imp = rec['dlc'][:, :]
    speaker1_x0y0 = 1.0, -0.8
    speaker2_x0y0 = 0.0, -0.8

    d1, theta1, vel, rvel, d_fwd, d_lat = compute_d_theta(
        dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker1_x0y0)
    d2, theta2, vel, rvel, d_fwd, d_lat = compute_d_theta(
        dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker2_x0y0)

    # fall-off over distance
    # -- currently fudged 5 db diff from front to back
    dist_atten = 6
    print(f"Imposing distance attenuation={dist_atten} dB ")
    gaind1 = -(d1 - 1) * dist_atten
    gaind2 = -(d2 - 1) * dist_atten

    f = interpolate.interp1d(A, R0, axis=1, fill_value="extrapolate")
    gainr1 = f(theta1)[:, 0, :] + gaind1
    gainr2 = f(theta2)[:, 0, :] + gaind2
    g = interpolate.interp1d(A, L0, axis=1, fill_value="extrapolate")
    gainl1 = g(theta1)[:, 0, :] + gaind1
    gainl2 = g(theta2)[:, 0, :] + gaind2

    s1 = rec['stim'].as_continuous()[:channels, :]
    if stimcount>1:
        s2 = rec['stim'].as_continuous()[channels:, :]
    else:
        s2 = np.zeros_like(s1)

    # dB = 10*log10(P2/P1)
    # so, to scale the gtgrams:
    #   P2 = 10^(dB/10) * P1
    #r12 = s1 * 10 ** (gainr1 / 10) + s2 * 10 ** (gainr2 / 10)
    #l12 = s1 * 10 ** (gainl1 / 10) + s2 * 10 ** (gainl2 / 10)
    r12 = np.sqrt((s1**2) * 10 ** (gainr1 / 10) + (s2**2) * 10 ** (gainr2 / 10))
    l12 = np.sqrt((s1**2) * 10 ** (gainl1 / 10) + (s2**2) * 10 ** (gainl2 / 10))

    binaural_stim = np.concatenate([r12,l12], axis=0)
    newrec = rec.copy()
    newrec['stim'] = newrec['stim']._modified_copy(data=binaural_stim)
    newrec['disttheta'] = newrec['stim']._modified_copy(
        data=np.concatenate([d1,theta1,d2,theta2],axis=0),
        chans=['d1','theta1','d2','theta2'])
    return {'rec': newrec}


rasterfs = 50
f_min=200
f_max=20000
channels=18
imopts={'origin': 'lower',
    'interpolation': 'none',
    'cmap': 'inferno',
    'aspect': 'auto'}


dlc_chans=10
siteid = 'PRN050a'
siteid = 'PRN048a'

rec = free_model.load_free_data(siteid, batch=348, rasterfs=rasterfs, dlc_chans=dlc_chans)

rec = stim_filt_hrtf(rec, hrtf_format='az', smooth_win=2, f_min=f_min, f_max=f_max, channels=channels)['rec']

dlc = rec['dlc'].as_continuous()[:8,:].T
stimamp = rec['stim'].as_continuous().mean(axis=0)

if siteid=="PRN015a":
    d_indexes=[11135, 11500, 12250, 13200, 300000]  # PRN015 -- good samples
    out_channel=21  # PRN015
elif siteid=="PRN047a":
    d_indexes=[12400, 12475, 12700, 13155]  #
    out_channel=1
    out_channel=17
    out_channel=12  # PRN047
elif siteid=="PRN048a":
    d_indexes=[12250, 12340, 12700, 12950]  #
    out_channel=6  #
    out_channel=14  #
else:
    raise ValueError("site not set up")
f,ax=plt.subplots(1,1,figsize=(12,3))
t1,t2=11000,15000
ax.plot(np.arange(t1,t2), dlc[t1:t2,0:3])
#plt.plot(np.arange(t1,t2), stimamp[t1:t2])
for i,t in enumerate(d_indexes[:4]):
    ax.axvline(t)
ax.legend(('Front x','Front y','Back x','Back y'), fontsize=12)
ax.invert_yaxis()

plt.figure()
plt.scatter(dlc[::10,0], dlc[::10,1], s=2, color='lightgray')
for i,t in enumerate(d_indexes):
    plt.text(dlc[t,0], dlc[t,1], f"{t}")
    plt.plot(dlc[t,0], dlc[t,1], "o", color='red')
    plt.plot(dlc[(t-6):(t+1),0], dlc[(t-6):(t+1),1], color='darkblue', lw=0.5)
    #plt.plot(dlc[t,2], dlc[t,3], "o", color='blue')
plt.gca().invert_yaxis()
plt.title(siteid, fontsize=12)

disttheta=rec['disttheta'].as_continuous()
for i,t in enumerate(d_indexes[:4]):
    print(f"t={t}: d1={disttheta[0,t]:.3f} th1={disttheta[1,t]:.3f} d2={disttheta[2,t]:.3f} th2={disttheta[3,t]:.3f}")

# log compress and normalize stim
fn = lambda x: _dlog(x, -1)
rec['stim'] = rec['stim'].transform(fn, 'stim')
rec['stim'] = rec['stim'].normalize('minmax')
rec['resp'] = rec['resp'].normalize('minmax')

L0, R0, c, A = load_hrtf(format='az', fmin=f_min, fmax=f_max, num_freqs=channels)

f,ax=plt.subplots(1,2)
ax[0].imshow(L0, origin='lower', extent=[A[0],A[-1],c[0],c[-1]], aspect='auto')
im=ax[1].imshow(R0, origin='lower', extent=[A[0],A[-1],c[0],c[-1]], aspect='auto')
#plt.colorbar(im, ax=ax[1])

#rec = dlc2dist(rec, smooth_win=2, keep_dims=dlc_chans)
# speaker1
dlc_data_imp = rec['dlc'][:, :]
speaker1_x0y0 = 1.0, -0.8
speaker2_x0y0 = 0.0, -0.8
smooth_win = 2

plt.figure()
plt.scatter(dlc_data_imp[0,::4], dlc_data_imp[1,::4], s=1)
plt.scatter(dlc_data_imp[2,::4], dlc_data_imp[3,::4], s=1)
plt.gca().invert_yaxis()
plt.plot(speaker1_x0y0[0], speaker1_x0y0[1], 'o', color='red')
plt.plot(speaker2_x0y0[0], speaker2_x0y0[1], 'o', color='red')
plt.plot(dlc_data_imp[0,0], dlc_data_imp[1,0], 'o', color='red')
plt.plot(dlc_data_imp[2,0], dlc_data_imp[3,0], 'o', color='blue')
plt.plot(dlc_data_imp[0,3900], dlc_data_imp[1,3900], 'o', color='red')
plt.plot(dlc_data_imp[2,3900], dlc_data_imp[3,3900], 'o', color='blue')
plt.plot(dlc_data_imp[0,6000], dlc_data_imp[1,6000], 'o', color='red')
plt.plot(dlc_data_imp[2,6000], dlc_data_imp[3,6000], 'o', color='blue')

d1, theta1, vel, rvel, d_fwd, d_lat = compute_d_theta(
    dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker1_x0y0)
d2, theta2, vel, rvel, d_fwd, d_lat = compute_d_theta(
    dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker2_x0y0)

# fall-off over distance
gaind1 = -(d1-1)*6
gaind2 = -(d2-1)*6

f = interpolate.interp1d(A, R0, axis=1, fill_value="extrapolate")
gainr1 = f(theta1)[:,0,:] + gaind1
gainr2 = f(theta2)[:,0,:] + gaind2
g = interpolate.interp1d(A, L0, axis=1, fill_value="extrapolate")
gainl1 = g(theta1)[:,0,:] + gaind1
gainl2 = g(theta2)[:,0,:] + gaind2

t1, t2=13000,15000
f,ax = plt.subplots(8,1, sharex=True)
ax[0].plot(np.arange(t1,t2), dlc_data_imp[:2,t1:t2].T)
ax[1].plot(np.arange(t1,t2), d1[0,t1:t2])
ax[1].plot(np.arange(t1,t2), d2[0,t1:t2])
ax[2].plot(np.arange(t1,t2), theta1[0,t1:t2],label='spk1 theta')
ax[2].plot(np.arange(t1,t2), theta2[0,t1:t2],label='spk2 theta')
ax[2].legend()

vmin=-12
vmax=5

im=ax[3].imshow(gainr1[:,t1:t2], extent=[t1,t2,1,18], vmin=vmin, vmax=vmax, **imopts)
ax[3].set_ylabel('spk1 R')

im=ax[4].imshow(gainl1[:,t1:t2], extent=[t1,t2,1,18], vmin=vmin, vmax=vmax, **imopts)
ax[4].set_ylabel('spk1 L')

im=ax[5].imshow(gainr2[:,t1:t2], extent=[t1,t2,1,18], vmin=vmin, vmax=vmax, **imopts)
ax[5].set_ylabel('spk2 R')

im=ax[6].imshow(gainl2[:,t1:t2], extent=[t1,t2,1,18], vmin=vmin, vmax=vmax, **imopts)
ax[6].set_ylabel('spk2 L')

im=ax[7].imshow(np.arange(vmin,vmax)[:,np.newaxis], extent=[t1,t2,vmin,vmax], **imopts)
ax[7].set_ylabel('gain dB')

plt.tight_layout()

s1 = rec['stim'].as_continuous()[:18,:]
s2 = rec['stim'].as_continuous()[18:,:]

# dB = 10*log10(P2/P1)
# P2 = 10^(dB/10) * P1
r12 = s1 * 10**(gainr1/10) + s2 * 10**(gainr2/10)
l12 = s1 * 10**(gainl1/10) + s2 * 10**(gainl2/10)
#r12 = np.sqrt((s1**2) * 10 ** (gainr1 / 10) + (s2**2) * 10 ** (gainr2 / 10))
#l12 = np.sqrt((s1**2) * 10 ** (gainl1 / 10) + (s2**2) * 10 ** (gainl2 / 10))

f,ax = plt.subplots(4,1, sharex=True)
im=ax[0].imshow(s1[:,t1:t2], extent=[t1,t2,1,18], **imopts)
ax[0].set_ylabel('stim 1')

im=ax[1].imshow(s2[:,t1:t2], extent=[t1,t2,1,18], **imopts)
ax[1].set_ylabel('stim 2')

vmax=np.nanmax(r12)
vmax=1
im=ax[2].imshow(r12[:,t1:t2], extent=[t1,t2,1,18], vmax=vmax, **imopts)
ax[2].set_ylabel('R ear')

im=ax[3].imshow(l12[:,t1:t2], extent=[t1,t2,1,18], vmax=vmax, **imopts)
ax[3].set_ylabel('L ear')



"""
wavfile1 = '/auto/data/sounds/BigNat/v2/00seq5_hand.wav'
wavfile2 = '/auto/data/sounds/BigNat/v2/00seq6_hand.wav'

fs, w1 = wavfile.read(wavfile1)
fs2, w2 = wavfile.read(wavfile2)
w12 = w1+w2

#wavfile.write('/tmp/test2.wav', 44000, w[0, :])
window_time = 1 / rasterfs
hop_time = 1 / rasterfs
padbins = int(np.ceil((window_time - hop_time) / 2 * fs))

s1 = gtgram(np.pad(w1, [padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)
s2 = gtgram(np.pad(w2, [padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)
s12 = gtgram(np.pad(w12, [padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)

f,ax = plt.subplots(4,1)
ax[0].imshow(np.log(s1+1), **imopts)
ax[1].imshow(np.log(s2+1), **imopts)
ax[2].imshow(np.log(s12+1), **imopts)
ax[3].imshow(np.log(s1+s2+1), **imopts)
"""