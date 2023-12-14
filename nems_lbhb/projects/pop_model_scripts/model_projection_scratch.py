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

from nems.tools import mapping
import importlib

batch=322
siteids, cellids = db.get_batch_sites(batch)

modelnames = [
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70.o.s-wc.70x1x90-fir.10x1x90-relu.90.o.s-wc.90x120-relu.120.o.s-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4.json',
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70-wc.70x1x90-fir.10x1x90-relu.90-wc.90x120-relu.120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4.json',
    'gtgram.fs100.ch32.pop-loadpop-norm.l1-popev_wc.32x1x70.g-fir.15x1x70-relu.70.o.s-wc.70x1x90-fir.10x1x90-relu.90.o.s-wc.90x120-relu.120.o.s-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4.json',
    'gtgram.fs100.ch32.pop-loadpop-norm.l1-popev_wc.32x1x70.g-fir.15x1x70-relu.70-wc.70x1x90-fir.10x1x90-relu.90-wc.90x120-relu.120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4.json',
    ]

modelname=modelnames[1]
model='CNN-32'
modelspec = mapping.load_mapping_model(name=model)

wav = '/auto/data/sounds/BigNat/v1/cat97_rec1_crickets_excerpt1.wav'
wav = '/auto/users/svd/projects/snh_stimuli/naturalsound-iEEG-sanitized-mixtures/stim174_girl_speaking.wav'
wav = '/auto/users/svd/projects/snh_stimuli/naturalsound-iEEG-sanitized-mixtures/stim138_duck_quack.wav'
wav = '/auto/data/sounds/vocalizations/v2/ferretb3001R.wav'

projection, f = mapping.project(modelspec, wavfilename=wav, w=None, fs=None,
                                raw_scale=250, OveralldB=65, verbose=True)
outpath = '/home/svd/Documents/onedrive/presentations/zhejiang_2023/figs/'
f.savefig(outpath+'project_ferret.pdf')
"""
importlib.reload(mapping)

outpath = '/home/svd/Documents/onedrive/class/neus_627_sys_neuro/2023/figs/'
wav = '/auto/data/sounds/BigNat/v1/cat97_rec1_crickets_excerpt1.wav'
f=mapping.spectrogram(wav=wav, rasterfs=200, channels=64)
f.savefig(outpath+'crickets.pdf')

wav = '/auto/data/sounds/BigNat/v1/cat185_rec1_hairdryer_excerpt1.wav'
f=mapping.spectrogram(wav=wav, rasterfs=200, channels=64)
f.savefig(outpath+'hairdryer.pdf')

wav = '/auto/data/sounds/BigNat/v1/00cat414_rec1_woman_speaking_excerpt1.wav'
f=mapping.spectrogram(wav=wav, rasterfs=200, channels=64)
f.savefig(outpath+'woman.pdf')
"""

if 0:
    # plot projection for model test data
    if channels==18:
        uri = '/auto/data/nems_db/recordings/322/NAT4v2_gtgram.fs100.ch18.tgz'
    elif channels==32:
        uri = '/auto/data/nems_db/recordings/322/NAT4v2_gtgram.fs100.ch32.tgz'
    else:
        raise ValueError("channels value not supported")

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


