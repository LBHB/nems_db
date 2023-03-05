import os
import logging
import numpy as np
import json as jsonlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import det

import nems0.epoch as ep
from nems import xforms
from nems0.plots.api import ax_remove_box, spectrogram, fig2BytesIO
from nems0.uri import NumpyEncoder
from nems_lbhb.analysis.pop_models import subspace_overlap, compute_dstrf, dstrf_pca

log = logging.getLogger(__name__)


"""
imported from pop_models.py
def subspace_overlap(u, v):
def compute_dstrf(modelspec, rec, index_range=None, sample_count=100, out_channel=[0], memory=10,
                  norm_mean=True, method='jacobian', **kwargs):
def dstrf_pca(modelspec, rec, pc_count=3, out_channel=[0], memory=10,
              **kwargs):
def dstrf_movie(rec, dstrf, out_channel, index_range, preview=False, mult=False, out_path="/tmp", 
                out_base=None, **kwargs):
def make_movie(ctx, cellid=None, out_channel=0, memory=10, index_range=None, **kwargs):
"""

def spo_dstrf_per_stream_condition(n_pc = 2, memory = 12, recname='val', cellids=None, **ctx):

    rec = ctx[recname].apply_mask()
    modelspec = ctx['modelspec']
    print(ctx['modelspec'].meta['modelname'])
    print(ctx['modelspec'].meta['cellid'])

    if cellids is None:
        # analyze all output channels
        cellids = rec['resp'].chans
        siteids = [c.split("-")[0] for c in cellids]

    out_channel = list(np.arange(len(cellids)))
    channel_count=len(out_channel)

    # figure out epoch bounds
    e = rec['resp'].epochs
    enames = ep.epoch_names_matching(e, '^STIM_')

    set_names=['stream 1','stream 2','coh','inc']
    esets=[
           ['STIM_T+si464+null', 'STIM_T+si516+null'],
           ['STIM_T+null+si464', 'STIM_T+null+si516'],
           ['STIM_T+si464+si464', 'STIM_T+si516+si516'],
           ['STIM_T+si464+si516', 'STIM_T+si516+si464']]
    index_sets = []
    for _es in esets:
        this_index=np.array([], dtype=int)
        for e in _es:
            x = rec['resp'].get_epoch_indices(e)
            print(f'{e}: {x}')
            this_index = np.concatenate((this_index,np.arange(x[0, 0],x[0, 1], dtype=int)))
        index_sets.append(this_index)

    pcs = [''] * 4
    pc_mag = [''] * 4

    for s in range(4):
        index_range = index_sets[s]

        # skip silent bins
        stim_mag = rec['stim'].as_continuous().sum(axis=0)
        stim_big = stim_mag > np.max(stim_mag) / 1000
        index_range = index_range[(index_range > memory) & stim_big[index_range.astype(int)]]
        print(f'Calculating dstrf for {channel_count} channels, {len(index_range)} timepoints, memory={memory}')
    
        pcs[s], pc_mag[s] = dstrf_pca(modelspec, rec, pc_count=n_pc, out_channel=out_channel,
                                      index_range=index_range, memory=memory)

    f2,axs=plt.subplots(8, 10, figsize=(20,12))
    cmax = np.min([channel_count, 10])
    for c in range(cmax):
        cellid = cellids[c]
        for s in range(4):
            for i in range(2):
                mm=np.max(np.abs(pcs[s][i,:,:,c]))
                _p = pcs[s][i,:,:,c] * pc_mag[s][i,c] / pc_mag[s][0,c]
                _p *= np.sign(_p.sum())
                _row = s*2 + i
                _col = c
                axs[_row,_col].imshow(_p,aspect='auto',origin='lower', clim=[-mm, mm])
                if i+s == 0:
                    axs[_row,_col].set_title(f'{cellid} {pc_mag[s][i,c]:.3f}', fontsize=8)
                else:
                    axs[_row,_col].set_title(f'{pc_mag[s][i,c]:.3f}', fontsize=8)
                if _col<7:
                    axs[_row,_col].set_xticks([])
                if c>0:
                    axs[_row,_col].set_yticks([])
                if (c==0):
                    axs[_row,_col].set_ylabel(set_names[s])
                ax_remove_box(axs[_row,_col])
    return f2, pcs, pc_mag


