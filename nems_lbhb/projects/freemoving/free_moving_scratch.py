from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

from nems0 import db
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.motor.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist

USE_DB = True

if USE_DB:
    siteid = "PRN034a"
    siteid = "PRN010a"
    siteid = "PRN009a"
    siteid = "PRN011a"
    siteid = "PRN022a"
    siteid = "PRN015a"
    siteid = "PRN047a"
    runclassid = 132

    sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
    dparm = db.pd_query(sql)
    parmfile = [r.stimpath+r.stimfile for i,r in dparm.iterrows()]
    cellids=None
else:
    parmfile = ["/auto/data/daq/Prince/PRN015/PRN015a01_a_NTD",
                "/auto/data/daq/Prince/PRN015/PRN015a02_a_NTD"]
    cellids = None

## load the recording

rasterfs = 50

ex = BAPHYExperiment(parmfile=parmfile, cellid=cellids)
print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

recache = False
rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
                       dlc=True, recache=recache, rasterfs=rasterfs,
                       dlc_threshold=0.2, fill_invalid='interpolate')

# generate 'dist' signal from dlc signal
rec = dlc2dist(rec, ref_x0y0=None, smooth_win=1,
               egocentric_velocity=False, verbose=False)

# pull out relevant signals
resp = rec['resp'].rasterize()
stim = rec['stim']
cellcount = resp.shape[0]

epochs = rec.epochs
is_stim = epochs['name'].str.startswith('STIM')
stim_epoch_list = epochs[is_stim]['name'].tolist()
stim_epochs = list(set(stim_epoch_list))
stim_epochs.sort()

# get big stimulus and response matrices. Note that "time" here is real
# experiment time, so repeats of the test stimulus show up as single-trial
# stimuli.
#X = stim.rasterize().as_continuous()
Y = resp.as_continuous()

print('Single-trial stim/resp data extracted to X/Y matrices!')

# here's how you get rasters aligned to each stimulus:
# define regex for stimulus epochs
#epoch_regex = '^TARGET'
epoch_regex = '^STIM_'

epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
folded_resp = resp.extract_epochs(epochs_to_extract)

print('Response to each stimulus extracted to folded_resp dictionary')

# example epoch
epoch = epochs_to_extract[4]

# or just plot the PSTH for an example stimulus
raster = resp.extract_epoch(epoch)
psth = np.nanmean(raster, axis=0)
dist = rec['dist'].extract_epoch(epoch)[:,0,:]
theta = rec['dist'].extract_epoch(epoch)[:,1,:]
vel = rec['dist'].extract_epoch(epoch)[:,2,:]
rvel = rec['dist'].extract_epoch(epoch)[:,3,:]

try:
    spec = stim._data[epoch]
except:
    stim = stim.rasterize()
    spec = stim.extract_epoch(epoch)[0,:,:]
norm_psth = psth - np.mean(psth,axis=1,keepdims=True)
norm_psth /= np.std(psth,axis=1,keepdims=True)
sc = norm_psth @ norm_psth.T / norm_psth.shape[1]

# get example raster
if cellcount>45:
    c1,c2,c3=42,43,44
else:
    c1,c2,c3 = cellcount-3, cellcount-2, cellcount-1
    #c1,c2,c3=5,10,16
r=raster[:,c1,:]
r[np.isnan(r)]=0.1
r[r>2]=2


# Plot example segment from stimulus spectrogram and population response
SHOW_BINS = 250
max_time = SHOW_BINS/resp.fs

imopts = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'none'}

f, ax = plt.subplots(6,1, figsize=(6, 8), sharex=True)
ax[0].imshow(spec[:,:SHOW_BINS], extent=[0,max_time,0,spec.shape[0]],
             cmap='gray_r', **imopts)
ax[0].set_title(f"Spectrogram ({epoch})")

ax[1].imshow(norm_psth[:,:SHOW_BINS], extent=[0,max_time,0,norm_psth.shape[0]],
             cmap='gray_r', **imopts)
ax[1].set_title(f"Population PSTH (averaged over {raster.shape[0]} reps)")

ax[2].imshow(r[:,:SHOW_BINS], extent=[0,max_time,0,r.shape[0]],
             cmap='gray_r', **imopts)
ax[2].set_title(f"Raster (cell {c1}, {raster.shape[0]} reps)")

tt=np.arange(psth.shape[1])/resp.fs
ax[3].plot(tt[:SHOW_BINS],psth[c1,:SHOW_BINS], color='black', lw=1)
ax[3].set_xlabel('repetition')

#dist[np.isnan(dist)]=2000
vel[np.isnan(vel)]=1
vel[vel<-5]=-5
vel[vel>5]=5
ax[4].imshow(dist[:,:SHOW_BINS], extent=[0,max_time,0,dist.shape[0]],
             cmap='gray_r', **imopts)
ax[4].set_title(f"Dist from speaker")
ax[4].set_ylabel('repetition')

rvel[rvel<-180]+=360
rvel[rvel>180]-=360
rvel[rvel<-10]=-10
rvel[rvel>10]=10
ax[5].imshow(theta[:,:SHOW_BINS], extent=[0,max_time,0,theta.shape[0]],
             cmap='bwr', **imopts)
ax[5].set_title(f"Angle (CW) from speaker")
ax[5].set_ylabel('repetition')
plt.tight_layout()

if True:
    # Validate spatio-temporal alignment with target hit events
    e=resp.get_epoch_indices('LICK , FA')
    e=resp.get_epoch_indices('TARGET')
    e=resp.get_epoch_indices('LICK , HIT')
    tr = np.zeros((400,len(e)))
    for i,(a,b) in enumerate(e):
        if a>100:
            tr[:,i] = rec['dist'][0,(a-100):(a+300)]
    plt.figure()
    t = np.arange(400)/rasterfs-(100/rasterfs)
    plt.plot(t,tr)
    plt.xlabel('time from Hit (s)')
    plt.ylabel('distance from speaker (pixels)')
    plt.title(basename(parmfile[0]))

# summarize spatial and velocity distributions with scatter
free_scatter_sum(rec)

## regression analysis do (d, theta, v_d, v_theta)
##
# predict deviation from PSTH response?

# plt.close('all')

# option to get waveform info too
#siteinfo = baphy_io.get_spike_info(siteid=siteid, save_to_db=True)
# just depth info, faster
df_sitedata = baphy_io.get_depth_info(siteid=siteid)
cellids = resp.chans
chans = [f"{int(c.split('-')[1]):03d}" for c in cellids]
units = [c.split('-')[2] for c in cellids]
cellids = [f"{siteid}-{ch}-{u}" for ch,u in zip(chans,units)]
df_sitedata = df_sitedata.loc[cellids]
df_sitedata['chanstr']=chans
df_sitedata = df_sitedata.sort_values(by='chanstr')

# extract relevant spatial/motor info from processed "dist" signal
ddict = rec['dist'].extract_epochs(epochs_to_extract)
reps = np.max([r_.shape[0] for k_, r_ in ddict.items() if r_.shape[0]>2])
ddict = {k_: np.pad(r_, ((0, reps - r_.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=np.nan) for k_, r_ in ddict.items() if r_.shape[0]>1}

d0 = np.concatenate([d_[:, 0, :] for k_, d_ in ddict.items()], axis=1)
t0 = np.concatenate([d_[:, 1, :] for k_, d_ in ddict.items()], axis=1)
pgoodidx = np.isfinite(d0) & (d0 > 550) & (d0 < 1000) & np.isfinite(t0) & (np.abs(t0) < 150)

dv = np.concatenate([d_[:, 2, :] for k_, d_ in ddict.items()], axis=1)
tv = np.concatenate([d_[:, 3, :] for k_, d_ in ddict.items()], axis=1)
vgoodidx = np.isfinite(dv) & (np.abs(dv) < 10*rasterfs) & np.isfinite(tv) & (np.abs(tv) < 10*rasterfs)

fv = np.concatenate([d_[:, 4, :] for k_, d_ in ddict.items()], axis=1)
lv = np.concatenate([d_[:, 5, :] for k_, d_ in ddict.items()], axis=1)
lgoodidx = np.isfinite(fv) & (np.abs(fv) < 8*rasterfs) & np.isfinite(lv) & (np.abs(lv) < 8*rasterfs)

d0 = d0[pgoodidx]
t0 = t0[pgoodidx]
dv = dv[vgoodidx]
tv = tv[vgoodidx]
fv = fv[lgoodidx]
lv = lv[lgoodidx]

dall = np.concatenate([d_[:, :, :] for k_, d_ in ddict.items()], axis=2)
dallgoodidx = (np.sum(np.isfinite(dall),axis=1) == 6) & pgoodidx & vgoodidx & lgoodidx
dall = np.transpose(dall,[1,0,2])[:,dallgoodidx]

rdict = resp.extract_epochs(epochs_to_extract)
reps = np.max([r_.shape[0] for k_, r_ in rdict.items() if r_.shape[0] > 2])
rdict = {k_: np.pad(r_, ((0, reps - r_.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=np.nan) for
         k_, r_ in rdict.items() if r_.shape[0] > 1}

for c in range(resp.shape[0]):
    cellid = cellids[c]

    print(f"Regressing spike rate error against pos,vel for {cellid} ")

    r = np.concatenate([r_[:, c, :] for k_, r_ in rdict.items()], axis=1)
    p = np.nanmean(r, axis=0, keepdims=True)
    e = r - p
    # goodidx = np.isfinite(e) & np.isfinite(d) & (d>550) & (d<1100) & np.isfinite(t) & (np.abs(t)<135)
    eall = e[dallgoodidx]

    df = pd.DataFrame({'resp': eall, 'dist': dall[0, :], 'angle': dall[1, :],
                       'dv': dall[2, :], 'dt': dall[3, :],
                       'const': 1})
    s = smf.ols('resp ~ dist + angle + dv + dt', data=df)
    res = s.fit()
    coefs = res.params[1:]
    pvalues = res.pvalues[1:]
    df_sitedata.loc[cellid, ['d', 't', 'dv', 'tv']] = coefs.values
    df_sitedata.loc[cellid, ['pd', 'pt', 'pdv', 'ptv']] = pvalues.values

# histogram position and velocity traces
ll0 = np.linspace(d0.min(),d0.max(),15)
tt0 = np.linspace(t0.min(),t0.max(),15)
nn0 = np.zeros((len(ll0)-1,len(tt0)-1)) * np.nan
for i_,l_ in enumerate(ll0[:-1]):
    for j_,t_ in enumerate(tt0[:-1]):
        v_ = (d0>=l_)&(d0<ll0[i_+1]) & (t0>=t_) &(t0<tt0[j_+1])
        nn0[i_,j_] = v_.sum()
nn0[nn0<20]=np.nan
nn0[nn0>500]=500

llv = np.linspace(dv.min(),dv.max(),15)
ttv = np.linspace(tv.min(),tv.max(),15)
nnv = np.zeros((len(llv)-1,len(ttv)-1)) * np.nan
for i_,l_ in enumerate(llv[:-1]):
    for j_,t_ in enumerate(ttv[:-1]):
        v_ = (dv>=l_)&(dv<llv[i_+1]) & (tv>=t_) &(tv<ttv[j_+1])
        nnv[i_,j_] = v_.sum()
nnv[nnv<20]=np.nan
nnv[nnv>500]=500

lll = np.linspace(fv.min(),fv.max(),15)
ttl = np.linspace(lv.min(),lv.max(),15)
nnl = np.zeros((len(lll)-1,len(ttl)-1)) * np.nan
for i_,l_ in enumerate(lll[:-1]):
    for j_,t_ in enumerate(ttl[:-1]):
        v_ = (fv>=l_)&(fv<lll[i_+1]) & (lv>=t_) &(lv<ttl[j_+1])
        nnl[i_,j_] = v_.sum()
nnl[nnl<20]=np.nan
nnl[nnl>500]=500

# plot a bunch of heatmaps of E as a function of Pos or Vel
# depending on the value of USE_VEL
a1idx = np.where(df_sitedata['area'] == 'A1')[0]
cells = a1idx[np.linspace(0,len(a1idx)-1,24).astype(int)]

USE_VEL = 'linear'

if USE_VEL == 'linear':
    ll, tt, nn = lll, ttl, nnl
    d, t = fv, lv
    goodidx = lgoodidx

elif USE_VEL == 'rotation':
    ll, tt, nn = llv, ttv, nnv
    d, t = dv, tv
    goodidx = vgoodidx
else:
    ll, tt, nn = ll0, tt0, nn0
    d, t = d0, t0
    goodidx = pgoodidx

rows = 5
cols = int(np.ceil((len(cells) + 1) / rows))

f, ax = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), sharex=True, sharey=True)
ax = ax.flatten()
for i, c in enumerate(cells):
    cellid = resp.chans[c]
    cstr=cellid.replace(siteid,'')
    try:
        area = df_sitedata.loc[cellid, 'area']
    except:
        area = '??'

    r = np.concatenate([r_[:, c, :] for k_, r_ in rdict.items()], axis=1)
    p = np.nanmean(r, axis=0, keepdims=True)

    e = r - p
    # goodidx = np.isfinite(e) & np.isfinite(d) & (d>550) & (d<1100) & np.isfinite(t) & (np.abs(t)<135)
    eall = e[dallgoodidx]
    e = e[goodidx]

    mm = np.zeros((len(ll) - 1, len(tt) - 1)) * np.nan
    for i_, l_ in enumerate(ll[:-1]):
        for j_, t_ in enumerate(tt[:-1]):
            v_ = (d >= l_) & (d < ll[i_ + 1]) & (t >= t_) & (t < tt[j_ + 1])
            if (v_.sum() > 0):
                if (np.nanstd(e[v_]) > 0):
                    mm[i_, j_] = np.nanmean(e[v_]) / np.nanstd(e[v_])
                else:
                    mm[i_, j_] = np.nanmean(e[v_])

    mm[np.isnan(nn)] = 0
    x = (ll[:-1] + ll[1:]) / 2
    y = (tt[:-1] + tt[1:]) / 2
    X, Y = np.meshgrid(x, y)  # 2D grid for interpolation

    valididx = np.isfinite(mm)
    interp = LinearNDInterpolator(list(zip(X[valididx], Y[valididx])),
                                  mm[valididx], fill_value=np.nanmean(mm))

    # Z = interp(X, Y)
    # Z = gaussian_filter(Z, [0.75, 0.75])
    Z = mm
    df = pd.DataFrame({'resp': eall, 'dist': dall[0, :], 'angle': dall[1, :],
                       'dv': dall[2, :], 'dt': dall[3, :],
                       'const': 1})
    s = smf.ols('resp ~ dist + angle + dv + dt', data=df)
    res = s.fit()
    coefs = res.params[1:]
    pvalues = res.pvalues[1:]
    df_sitedata.loc[cellid, ['d','t','dv','tv']] = coefs
    df_sitedata.loc[cellid, ['pd','pt','pdv','ptv']] = pvalues.values

    ax[i].imshow(Z, extent=[tt[0], tt[-1], ll[-1], ll[0]],
                 aspect='auto', cmap='bwr', vmin=-np.abs(Z).max(), vmax=np.abs(Z).max())

    # ax[i,0].plot(cc,mm,'k')
    # ax[i].set_title(f'{resp.chans[c]} (d={coefs[0]*600:.2f},t={coefs[1]*180:.2f}) p={pvalues[0]:.2e},{pvalues[1]:.2e}')
    ax[i].set_title(f'{cstr} {area} ({Z.min():.2f},{Z.max():.2f})')

ax[-1].imshow(nn, extent=[tt[0], tt[-1], ll[-1], ll[0]], aspect='auto')
ax[-1].set_title(f'n')
plt.suptitle(resp.chans[0].split("-")[0])


## summary of a bunch of features for a selection of interesting units
SHOW_BINS=400
if siteid=='PRN015aXXX':
    cells = [44, 43, 42, 36, 35, 28, 27, 26, 5, 1]# PRN015
else:
    a1idx = np.where(df_sitedata['area']=='A1')[0]
    stepsize=int(np.ceil(len(a1idx)/10))
    cells = a1idx[::stepsize]


rows = len(cells)+1
cols=4

#f,ax = plt.subplots(rows,cols,figsize=(2*cols,2*rows), sharex=True, sharey=True)
#ax=ax.flatten()
f = plt.figure(figsize=(1.5*cols,2*rows))

for i,c in enumerate(cells):
    cellid = resp.chans[c]
    cstr = cellid.replace(siteid, '')
    try:
        area = df_sitedata.loc[cellid, 'area']
        depth = df_sitedata.loc[cellid, 'depth']
    except:
        area = '??'
        depth = '??'

    rdict = resp.extract_epochs(epochs_to_extract)
    reps = np.max([r_.shape[0] for k_, r_ in rdict.items() if r_.shape[0] > 2])
    rdict = {k_: np.pad(r_, ((0, reps - r_.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=np.nan) for
             k_, r_ in rdict.items() if r_.shape[0] > 1}

    r = np.concatenate([r_[:, c, :] for k_,r_ in rdict.items()], axis=1)
    p = np.nanmean(r,axis=0, keepdims=True)

    e = r-p
    #goodidx = np.isfinite(e) & np.isfinite(d) & (d>550) & (d<1100) & np.isfinite(t) & (np.abs(t)<135)
    eall = e[dallgoodidx]
    ev = e[vgoodidx]
    el = e[lgoodidx]
    e = e[pgoodidx]

    mm = np.zeros((len(ll0)-1,len(tt0)-1)) * np.nan
    for i_,l_ in enumerate(ll0[:-1]):
        for j_,t_ in enumerate(tt0[:-1]):
            v_ = (d0>=l_)&(d0<ll0[i_+1]) & (t0>=t_) &(t0<tt0[j_+1])
            if (v_.sum()>0):
                if (np.nanstd(e[v_])>0):
                    mm[i_,j_] = np.nanmean(e[v_]) / np.nanstd(e[v_])
                else:
                    mm[i_,j_] = np.nanmean(e[v_])
    mm[np.isnan(nn0)]=0

    x = (ll[:-1]+ll[1:])/2
    y = (tt[:-1]+tt[1:])/2
    X, Y = np.meshgrid(x, y)  # 2D grid for interpolation

    valididx = np.isfinite(mm)
    interp = LinearNDInterpolator(list(zip(X[valididx], Y[valididx])),
                                  mm[valididx], fill_value=np.nanmean(mm))

    #Z = interp(X, Y)
    #Z = gaussian_filter(Z, [0.75, 0.75])
    Z = mm

    mmv = np.zeros((len(llv)-1,len(ttv)-1)) * np.nan
    for i_,l_ in enumerate(llv[:-1]):
        for j_,t_ in enumerate(ttv[:-1]):
            v_ = (dv>=l_)&(dv<llv[i_+1]) & (tv>=t_) &(tv<ttv[j_+1])
            if (v_.sum()>0):
                if (np.nanstd(ev[v_])>0):
                    mmv[i_,j_] = np.nanmean(ev[v_]) / np.nanstd(ev[v_])
                else:
                    mmv[i_,j_] = np.nanmean(ev[v_])
    mmv[np.isnan(nnv)]=0

    mml = np.zeros((len(lll)-1,len(ttl)-1)) * np.nan
    for i_,l_ in enumerate(lll[:-1]):
        for j_,t_ in enumerate(ttl[:-1]):
            v_ = (fv>=l_)&(fv<lll[i_+1]) & (lv>=t_) &(lv<ttl[j_+1])
            if (v_.sum()>0):
                if (np.nanstd(el[v_])>0):
                    mml[i_,j_] = np.nanmean(el[v_]) / np.nanstd(el[v_])
                else:
                    mml[i_,j_] = np.nanmean(el[v_])
    mml[np.isnan(nnl)]=0

    vmin,vmax = -np.abs(mm).max(),np.abs(mm).max()
    vmin,vmax = -0.75, 0.75

    ax = f.add_subplot(rows,4,4*(i+1)+1)
    tpsth=np.arange(SHOW_BINS)/resp.fs
    ax.plot(tpsth, smooth(psth[c,:SHOW_BINS], 3), 'k', lw=0.5)
    ax.set_xlim([tpsth[0], tpsth[-1]])
    ax.set_title(f"{cstr} {area} {depth}")
    if i<len(cells)-1:
        ax.set_xticklabels([])

    ax = f.add_subplot(rows,4,cols*(i+1)+2)
    ax.imshow(Z, extent=[tt[0],tt[-1],ll[-1],ll[0]],
                 aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)

    #ax[i,0].plot(cc,mm,'k')
    #ax[i].set_title(f'{resp.chans[c]} (d={coefs[0]*600:.2f},t={coefs[1]*180:.2f}) p={pvalues[0]:.2e},{pvalues[1]:.2e}')
    ax.set_title(f'{cstr} ({Z.min():.2f},{Z.max():.2f})')
    if i<len(cells)-1:
        ax.set_xticklabels([])

    ax = f.add_subplot(rows,4,cols*(i+1)+3)
    vmin,vmax = -np.abs(mmv).max(),np.abs(mmv).max()
    vmin,vmax = -0.75, 0.75
    ax.imshow(mmv, extent=[ttv[0],ttv[-1],llv[-1],llv[0]],
                 aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    if i<len(cells)-1:
        ax.set_xticklabels([])

    #ax[i,0].plot(cc,mm,'k')
    #ax[i].set_title(f'{resp.chans[c]} (d={coefs[0]*600:.2f},t={coefs[1]*180:.2f}) p={pvalues[0]:.2e},{pvalues[1]:.2e}')
    ax.set_title(f'{cstr} ({mmv.min():.2f},{mmv.max():.2f})')
    if i<len(cells)-1:
        ax.set_xticklabels([])

    ax = f.add_subplot(rows,4,cols*(i+1)+4)
    vmin,vmax = -np.abs(mml).max(),np.abs(mml).max()
    vmin,vmax = -0.75, 0.75
    ax.imshow(mml, extent=[ttl[0],ttl[-1],lll[-1],lll[0]],
                 aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    ax.set_title(f'{cstr} ({mml.min():.2f},{mml.max():.2f})')
    if i<len(cells)-1:
        ax.set_xticklabels([])


ax = f.add_subplot(rows,4,1)
nplt.plot_spectrogram(spec[:,:SHOW_BINS], fs=resp.fs, ax=ax, title=epoch, cmap='gray_r')
ax.set_xlabel('')
ax.set_xticklabels([])

# counts of position bins
ax = f.add_subplot(rows,4,2)
ax.imshow(nn0, extent=[tt0[0],tt0[-1],ll0[-1],ll0[0]], aspect='auto')
ax.set_title(f'n pos')
ax.set_xticklabels([])

# counts of dist/velocity bins
ax = f.add_subplot(rows,4,3)
ax.imshow(nnv, extent=[ttv[0],ttv[-1],llv[-1],llv[0]], aspect='auto')
ax.set_title(f'n vel')
ax.set_xticklabels([])

# counts of lateral velocity bins
ax = f.add_subplot(rows,4,4)
ax.imshow(nnl, extent=[ttl[0],ttl[-1],lll[-1],lll[0]], aspect='auto')
ax.set_title(f'n vel')
ax.set_xticklabels([])

plt.suptitle(resp.chans[0].split("-")[0])
plt.tight_layout()




