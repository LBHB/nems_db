from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, zoom
import importlib
import datetime
import os

from nems0 import db
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems_lbhb.projects.freemoving import free_model, free_vs_fixed_strfs
from nems.tools import json
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info

# code to support dumping figures
#dt = datetime.date.today().strftime("%Y-%m-%d")
#figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
#os.makedirs(figpath, exist_ok=True)

# tested sites
siteid="LMD014a"
batch=349
rasterfs = 50
PreStimSilence = 1.0

if 0:
    runclass = "'NFB'"

    dfiles = db.pd_query(f"SELECT * FROM gDataRaw WHERE bad=0 and training=0 and runclass in ({runclass}) and cellid='{siteid}'")
    parmfiles = [os.path.join(r['resppath'],r['parmfile']) for i,r in dfiles.iterrows()]

    manager = BAPHYExperiment(parmfile=parmfiles[:1])
else:
    manager = BAPHYExperiment(batch=349, cellid=siteid)
rec = manager.get_recording(**{'rasterfs': rasterfs, 'resp': True, 'stim': False},
                            recache=False)
rec=rec.create_mask('ACTIVE_EXPERIMENT', mask_name='mask_active')
rec=rec.create_mask('PASSIVE_EXPERIMENT', mask_name='mask_passive')

resp=rec['resp'].rasterize()
epochs=resp.epochs

stim_epochs=ep.epoch_names_matching(epochs, '^STIM_')

dstim = pd.DataFrame({'epoch': stim_epochs, 'fg': '', 'bg': '', 'fgc': 0, 'bgc': 0, 'snr': 0})
for i,r in dstim.iterrows():
    e=dstim.loc[i,'epoch']
    s = e.split('_')[1:]
    b = s[0].split("-")
    f = s[1].split("-")
    if len(f)<5:
        s_snr='0'
    else:
        s_snr=f[4][:-2]
    if s_snr[0]=='n':
        snr = -float(s_snr[1:])
    else:
        snr = float(s_snr)

    if (snr>=-50) & (f[0].upper()!='NULL'):
        dstim.loc[i,'fg']=f[0]
        dstim.loc[i,'fgc']=int(f[3])
    else:
        dstim.loc[i,'fg']='NULL'
        dstim.loc[i,'fgc']=1
    if (snr<50) & (b[0].upper()!='NULL'):
        dstim.loc[i,'bg']=b[0]
        dstim.loc[i,'bgc']=int(b[3])
        dstim.loc[i, 'snr'] = snr
    else:
        dstim.loc[i,'bg']='NULL'
        dstim.loc[i,'bgc']=1
        if snr>50:
            dstim.loc[i,'snr']=snr-100
        else:
            dstim.loc[i,'snr']=snr

fg_unique = dstim['fg'].unique().tolist()
bg_unique = dstim['bg'].unique().tolist()
snr_unique = dstim['snr'].unique().tolist()
fg_unique.remove('NULL')
bg_unique.remove('NULL')
if -100 in snr_unique:
    snr_unique.remove(-100)
fc = [1,2]
bc = [1,2]
conds = np.array(np.meshgrid(snr_unique, fc, bc)).T.reshape(-1,3)

triads = []
cc=0
for i,f in enumerate(fg_unique):
    for j,b in enumerate(bg_unique):
        for k,c in enumerate(conds):
            print(f"{f} {b} {c}")
            snr,fc,bc = c
            fgbg = dstim.loc[(dstim.snr==snr) & (dstim.fgc==fc) & (dstim.bgc==bc) &
                           (dstim.fg==f) & (dstim.bg==b),'epoch'].values[0]
            fg = dstim.loc[(dstim.snr==snr) & (dstim.fgc==fc) &
                           (dstim.fg==f) & (dstim.bg=='NULL'),'epoch'].values[0]
            bg = dstim.loc[(dstim.bgc==bc) &
                           (dstim.fg=='NULL') & (dstim.bg==b),'epoch'].values[0]
            triads.append(pd.DataFrame({'f': f,'b':b, 'snr': snr, 'fc': fc, 'bc': bc,
                                  'fg': fg, 'bg': bg, 'fgbg': fgbg  }, index=[cc]))
            cc+=1
d = pd.concat(triads)

DROP_NO_ACTIVE=True
REQUIRE_CC=True
if DROP_NO_ACTIVE:
    d['valid']=True
    for i,r in d.iterrows():
        for j,m in enumerate(['mask_passive','mask_active']):
            try:
                rfg = resp.extract_epoch(r['fg'], mask=rec[m])
            except:
                d.loc[i,'valid']=False
        if REQUIRE_CC:
            if (r.fc+r.bc)>2:
                d.loc[i,'valid']=False
    d=d.loc[d['valid']].reset_index(drop=True)
triadcount=len(d)

#plt.close('all')
smwin=3
T=int((PreStimSilence+2)*rasterfs)
T0=int(PreStimSilence*rasterfs)
lw=0.75
pstr={1: 'C', 2: 'I'}
colors = ['deepskyblue', 'yellowgreen', 'dimgray']

f,ax = plt.subplots(1,3)
snr = [[],[]]
spont = [[],[]]
rstd = [[],[]]
for j,m in enumerate(['mask_passive','mask_active']):
    rall = resp.extract_epoch('REFERENCE', mask=rec[m])[:, :, :T].mean(axis=0)
    rall = rall-np.mean(rall[:,:T0], axis=1, keepdims=True)
    ax[j].imshow(rall, aspect='auto', interpolation='none', origin='lower',
                 cmap='bwr', vmin=-rall.max(), vmax=rall.max())
    snr[j]=rall[:,T0:].std(axis=1)/rall[:,:T0].std(axis=1)
    spont[j]=rall[:,:T0].mean(axis=1, keepdims=True)
    rstd[j]=rall[:,T0:].std(axis=1, keepdims=True)
    ax[j].set_title(f"{siteid} {m} meanresp")
    ax[2].plot(snr[j],label=m)

snr_threshold = 1.5
ax[2].axhline(snr_threshold, ls='--', color='black', lw=lw, label='SNR thresh')
ax[2].legend(frameon=False)

goodcells = (snr[0]>snr_threshold) & (snr[1]>snr_threshold)
ax[2].set_title(f"good cells {goodcells.sum()}/{len(goodcells)}")

cols = 2
rows = int(np.ceil(triadcount/cols))
f,ax = plt.subplots(rows,cols*2, sharex=True, sharey=True)
ax = ax.flatten()
ncells = goodcells.sum()

for i,r in d.iterrows():
    for j,m in enumerate(['mask_passive','mask_active']):
        try:
            rfg = resp.extract_epoch(r['fg'], mask=rec[m])[:,goodcells,:T].mean(axis=0)
            rbg = resp.extract_epoch(r['bg'], mask=rec[m])[:,goodcells,:T].mean(axis=0)
            rfgbg = resp.extract_epoch(r['fgbg'], mask=rec[m])[:,goodcells,:T].mean(axis=0)
            tt = np.arange(len(rfg))/rasterfs - PreStimSilence
            prebins = (tt<0).sum()
            rr = np.concatenate([(rfg-spont[j][goodcells,:])/rstd[j][goodcells,:],
                                  (rbg-spont[j][goodcells,:])/rstd[j][goodcells,:],
                                   (rfgbg-spont[j][goodcells,:])/rstd[j][goodcells,:]],axis=0)
            vmax=rr.max()
            vmin=-vmax
            ax[i*2+j].imshow(rr, extent=[-1, 2, rr.shape[0],1], cmap='bwr', vmin=vmin, vmax=vmax)
            ax[i*2+j].axhline(y=np.sum(goodcells),ls='--',lw=lw,color='gray')
            ax[i*2+j].axhline(y=np.sum(goodcells)*2,ls='--',lw=lw,color='gray')
        except:
            ax[i*2+j].set_axis_off()
    ax[i*2].set_title(f"{r.f[6:]} {r.b[10:16]} snr={r.snr},f={pstr[r.fc]},b={pstr[r.bc]} PAS")
    ax[i * 2+1].set_title(f"ACT")
ax[0].text(0.1-PreStimSilence,ncells/2,'FG')
ax[0].text(0.1-PreStimSilence,ncells/2+ncells,'BG')
ax[0].text(0.1-PreStimSilence,ncells/2+2*ncells,'FG+BG')
f.suptitle(f"{siteid}")
plt.tight_layout()

raise ValueError('stopping')

# plt.close('all')

# interesting cids: LMD004a00_a_NFB - 59, 52, 68, 25
#cidlist=[2,3,6,7,8,9,13,15,18,19,20,21,22,25,28,31,32,35,41,42,48,49,50,52,55,59,60,61,63,65 , 68,73,74,82,89]
cidlist = np.arange(len(resp.chans))
# plt.close('all')
cellcount = len(cidlist)
rows=int(np.ceil(np.sqrt(cellcount)))
cols= int(np.ceil(cellcount/rows))
f,ax=plt.subplots(rows,cols, sharex=True, sharey=True)
ax=ax.flatten()
r = d.loc[4]
print(r.fg)
print(r.bg)
print(r.fgbg)
for i,a in enumerate(ax[:cellcount]):
    cid=cidlist[i]
    rfg = resp.extract_epoch(r['fg'])[:,cid,:T].mean(axis=0)
    rbg = resp.extract_epoch(r['bg'])[:,cid,:T].mean(axis=0)
    rfgbg = resp.extract_epoch(r['fgbg'])[:,cid,:T].mean(axis=0)
    tt = np.arange(len(rfg))/rasterfs - PreStimSilence
    mm = np.max(np.concatenate([rfg, rbg, rfgbg]))
    a.plot(tt,smooth(rfg/mm,smwin),lw=lw, label='fg', color=colors[1])
    a.plot(tt,smooth(rbg/mm,smwin),lw=lw,label='bg', color=colors[0])
    a.plot(tt,smooth(rfgbg/mm,smwin),lw=lw,label='fg+bg', color=colors[2])
    a.set_title(f"CH {cid}", fontsize=7)
a.legend(frameon=False)
plt.tight_layout()

cid=52
#cols=int(np.ceil(np.sqrt(triadcount)))
cols=4
rows= int(np.ceil(triadcount/cols))
f,ax=plt.subplots(rows,cols*2, sharex=True, sharey=True)
ax=ax.flatten()

for i,r in d.iterrows():
    for j,m in enumerate(['mask_passive','mask_active']):
        try:
            rfg = resp.extract_epoch(r['fg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            rbg = resp.extract_epoch(r['bg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            rfgbg = resp.extract_epoch(r['fgbg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            tt = np.arange(len(rfg))/rasterfs - PreStimSilence
            prebins = (tt<0).sum()
            if (i==0) & (j==0):
                spont = np.mean(rfg[:prebins]+rbg[:prebins]+rfgbg[:prebins])/3
            ax[i*2+j].plot(tt,smooth(rfg,smwin),lw=lw, label='fg', color=colors[1])
            ax[i*2+j].plot(tt,smooth(rbg,smwin),lw=lw,label='bg', color=colors[0])
            ax[i*2+j].plot(tt,smooth(rfgbg,smwin),lw=lw,label='fg+bg', color=colors[2])
            ax[i*2+j].axhline(y=spont,ls='--',lw=lw,color='gray')
        except:
            ax[i*2+j].set_axis_off()
    ax[i*2].set_title(f"{r.f[6:]} {r.b[10:16]} snr={r.snr},f={pstr[r.fc]},b={pstr[r.bc]}")
ax[i*2].legend()
f.suptitle(f"{cid} {resp.chans[cid]}")
plt.tight_layout()

cchisnr = (d.fc==1) & (d.bc==1) & (d.snr==0)
dcch = d.loc[cchisnr].reset_index()
f,ax=plt.subplots(2,len(dcch))
lateonset=int((PreStimSilence+0.2)*rasterfs)

for i,r in dcch.iterrows():
    efg=r['fg']
    ebg=r['bg']
    rfgp=resp.extract_epoch(efg, mask=rec['mask_passive']).mean(axis=0)
    rbgp=resp.extract_epoch(ebg, mask=rec['mask_passive']).mean(axis=0)
    rfga=resp.extract_epoch(efg, mask=rec['mask_active'], allow_empty=True).mean(axis=0)
    rbga=resp.extract_epoch(ebg, mask=rec['mask_active'], allow_empty=True).mean(axis=0)
    ax[0,i].plot(rfgp.mean(axis=0))
    ax[0,i].plot(rfga.mean(axis=0))
    ax[0,i].set_title(r['f'])
    ax[0,i].axvline(x=lateonset, ls='--')
    ax[1,i].plot(rbgp.mean(axis=0))
    ax[1,i].plot(rbga.mean(axis=0))
    ax[1,i].set_title(r['b'])
    ax[1,i].axvline(x=lateonset, ls='--')

i=0
ef=dcch.loc[0]