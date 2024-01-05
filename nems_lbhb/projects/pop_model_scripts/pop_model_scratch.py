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
from nems0.recording import load_recording
from nems_lbhb.xform_wrappers import split_pop_rec_by_mask
from nems0.xforms import normalize_sig
from nems.preprocessing.spectrogram import gammagram
from nems0.modules.nonlinearity import _dlog

batch=322
siteids, cellids = db.get_batch_sites(batch)

modelnames = [
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_conv2d.5x5x4.stack-relu.72.o.s-wc.72x1x90-fir.15x1x90-relu.90.o.s-wc.90x120-relu.120.o.s-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'gtgram.fs100.ch32.pop-loadpop-norm.l1-popev_wc.32x1x70.g-fir.15x1x70-relu.70.o.s-wc.70x1x90-fir.10x1x90-relu.90.o.s-wc.90x120-relu.120.o.s-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'gtgram.fs100.ch32.pop-loadpop-norm.l1-popev_wc.32x1x70.g-fir.15x1x70-relu.70-wc.70x1x90-fir.10x1x90-relu.90-wc.90x120-relu.120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'gtgram.fs100.ch32.pop-loadpop-norm.l1-popev_wc.32x70.g-fir.1x15x70-relu.70.f-wc.70x90-fir.1x10x90-relu.90.f-wc.90x120-relu.120.f-wc.120xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4',
]
modelnames = [
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70.o.s-wc.70x1x90-fir.10x1x90-relu.90.o.s-wc.90x120-relu.120.o.s-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70-wc.70x1x90-fir.10x1x90-relu.90-wc.90x120-relu.120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
]

cellid = 'NAT4v2'
batch = 322

xf = [None] * len(modelnames)
ctx = [None] * len(modelnames)

for i,m in enumerate(modelnames):
    xf[i],ctx[i] = fit_model_xform(cellid, batch, m, returnModel=True)

savepath = '/auto/users/svd/python/nems/nems/tools/demo_data/saved_models/'
from nems.tools import json
for c,m in zip(ctx,modelnames):
    outfile=savepath+m+'.json'
    print(outfile)
    try:
        json.save_model(c['modelspec'], outfile)
    except:
        print('nems0')

r_test = np.stack([c['modelspec'].meta['r_test'] for c in ctx],axis=1)
print(r_test.mean(axis=0))

"""
from nems.tools import mapping
import importlib
importlib.reload(mapping)

modelspec = mapping.load_mapping_model()

wav = '/auto/users/svd/projects/snh_stimuli/naturalsound-iEEG-sanitized-mixtures/stim138_duck_quack.wav'
wav = '/auto/users/svd/projects/snh_stimuli/naturalsound-iEEG-sanitized-mixtures/stim174_girl_speaking.wav'

mapping.project(modelspec, wav=wav, w=None, fs=None,
                raw_scale=250, OveralldB=65, verbose=True)

uri = '/auto/data/nems_db/recordings/322/NAT4v2_gtgram.fs100.ch18.tgz'

rec = load_recording(uri)
#rec = normalize_sig(rec=rec, sig='stim', norm_method='minmax', log_compress=1)['rec']
#rec = normalize_sig(rec=rec, sig='resp', norm_method='minmax', log_compress='None')['rec']
stim = rec['stim'].as_continuous()
lstim = _dlog(stim, -1)
smax = lstim.max(axis=1, keepdims=True)
smin = lstim.min(axis=1, keepdims=True)

d = split_pop_rec_by_mask(rec)
est=d['est']
val=d['val'].apply_mask()

projection = modelspec.predict(val['stim'].as_continuous().T)

plt.figure()
plt.imshow(projection.T)

from scipy.io import wavfile
fs,w = wavfile.read(wav)
w = w / np.iinfo(w.dtype).max
w *= 250
OveralldB = 65
sf = 10 ** ((80 - OveralldB) / 20)
w /= sf


channels=18
rasterfs=100
f_min=200
f_max=20000
window_time = 1 / rasterfs
hop_time = 1 / rasterfs
padbins = int(np.ceil((window_time - hop_time) / 2 * fs))
log_compress=1


s = gammagram(np.pad(w,[padbins, padbins]), fs, window_time, hop_time, channels, f_min, f_max)
s = _dlog(s, -log_compress)
s -= smin.T
s /= (smax-smin).T

projection = modelspec.predict(s)

# plt.close('all')
f=plt.figure()
ax = [f.add_subplot(4,1,1), f.add_subplot(4,1,2), f.add_subplot(2,1,2)]
t=np.arange(len(w))/fs
ax[0].plot(t,w)
ax[0].set_xlim([t[0],t[-1]])
ax[0].set_xticklabels([])
ts=s.shape[0]/rasterfs
im=ax[1].imshow(s.T, origin='lower', extent=[0, ts, -0.5, s.shape[1]+0.5])
ax[1].set_xticklabels([])
ax[2].imshow(projection.T, origin='lower', interpolation='none',
             extent=[0, ts, -0.5, projection.shape[1]+0.5])

"""
