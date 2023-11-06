from pathlib import Path
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
    MODELGROUPS, POP_MODELGROUPS, HELDOUT, MATCHED, EQUIVALENCE_MODELS_SINGLE, EQUIVALENCE_MODELS_POP,
    POP_MODELS, ALL_FAMILY_POP,
    SIG_TEST_MODELS,
    get_significant_cells, snr_by_batch, NAT4_A1_SITES, NAT4_PEG_SITES, PLOT_STAT, DOT_COLORS, DOT_MARKERS, base_path,
    linux_user, ALL_FAMILY_MODELS, VERSION, count_fits, int_path, a1, peg
)
from nems0 import db
from nems0.xform_helper import fit_model_xform, load_model_xform
from nems.tools import dstrf as dtools
from nems0.utils import shrinkage

batch=341
modelname="gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.jk5-lite.tf.lr1e4"
modelname="gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x30-fir.15x1x30-relu.30.f-wc.30xR-dexp.R_lite.tf.init.lr1e3.t2.jk5-lite.tf.lr1e4.t3"
cellid='PRN015a'

xf, ctx = fit_model_xform(cellid, batch, modelname, returnModel=True)

rec = ctx['rec']
est = ctx['est']
val = ctx['val']
model = ctx['modelspec']
model_list = ctx['modelspec_list']
cellids = rec['resp'].chans

time_step=57
D=15

t_indexes = np.arange(time_step, val['stim'].shape[1], 3)
log.info(f"Computing dSTRF at {len(t_indexes)} timepoints,t_step={time_step}")

out_channels=np.arange(110,115)
dstrfs=[]
vdstrfs=[]
for mi, m in enumerate(model_list):
    stim = {'input': val['stim'].as_continuous().T}
    d = m.dstrf(stim, D=D, out_channels=out_channels, t_indexes=t_indexes, reset_backend=False)
    dstrfs.append(d['input'])

dstrf = np.stack(dstrfs, axis=1)
s = np.std(dstrf, axis=(2,3,4), keepdims=True)
dstrf /= s
dstrf /= np.max(np.abs(dstrf))*0.9

mdstrf = dstrf.mean(axis=1,keepdims=True)
sdstrf = dstrf.std(axis=1,keepdims=True)
sdstrf[sdstrf==0]=1
z = np.abs(mdstrf)/sdstrf
mzdstrf = mdstrf.copy()
mzdstrf[z<2]=0
mzdstrf = shrinkage(mdstrf, sdstrf, sigrat=0.75)

mdstrf/=np.max(np.abs(mdstrf))*0.9
mzdstrf/=np.max(np.abs(mzdstrf))*0.9
dstrf = np.concatenate([dstrf, mdstrf, mzdstrf], axis=1)
pc_count=7
dpc, dpc_mag = dtools.compute_dpcs(dstrf[oi], pc_count=pc_count)

oi=0
di=100
imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
          'interpolation': 'none'}
imoptsz = {'cmap': 'bwr', 'origin': 'lower',
          'interpolation': 'none'}

f,ax=plt.subplots(len(out_channels), 10, figsize=(8,len(out_channels)), sharex=True, sharey=True)
f2,ax2=plt.subplots(len(out_channels), 10, figsize=(8,len(out_channels)), sharex=True, sharey=True)
for oi, oc in enumerate(out_channels):
    for di, d in enumerate(np.arange(100,1100,100)):
        ax[oi,di].imshow(mdstrf[oi, 0, d], **imopts)
        ax2[oi,di].imshow(mzdstrf[oi, 0, d], **imopts)

# plt.close('all')
f,ax=plt.subplots(len(out_channels), pc_count, figsize=(pc_count,len(out_channels)), sharex=True, sharey=True)
f2,ax2=plt.subplots(len(out_channels), pc_count, figsize=(pc_count,len(out_channels)), sharex=True, sharey=True)
dpc, dpc_mag = dtools.compute_dpcs(mdstrf[:,0], pc_count=pc_count)
dpcz, dpc_magz = dtools.compute_dpcs(mzdstrf[:,0], pc_count=pc_count)

for oi, oc in enumerate(out_channels):
    for di in range(pc_count):
        d = dpc[oi,di]
        d = d / np.max(np.abs(d)) / dpc_mag[0,oi] * dpc_mag[di,oi]
        ax[oi,di].imshow(d, **imopts)
        d = dpcz[oi,di]
        d = d / np.max(np.abs(d)) / dpc_magz[0,oi] * dpc_magz[di,oi]
        ax2[oi,di].imshow(d, **imopts)

# pc, pc_mag = dtools.compute_dpcs(d[np.newaxis, :, :, :], pc_count=pc_count)
dpc_frac = dpc_mag/dpc_mag.sum(axis=0,keepdims=True)
f,ax=plt.subplots()
ax.plot(dpc_frac[:,1:-1])
ax.plot(dpc_frac[:,-2],'X-')
ax.plot(dpc_frac[:,-1],'o-')

plt.legend(['1','2','3','4','5','m','mz'])
