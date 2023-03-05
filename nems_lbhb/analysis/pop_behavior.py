import os
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

import nems
import nems0.initializers
import nems0.priors
import nems0.utils
import nems0.xforms as xforms
import nems0.db as nd
import nems0.recording as recording
from nems0.xform_helper import fit_model_xform, load_model_xform

log = logging.getLogger(__name__)

from nems_lbhb.preprocessing import pupil_mask
from nems_lbhb.analysis.pop_models import dstrf_pca, subspace_overlap
from nems0.plots.api import ax_remove_box



def split_ap(rec, modelspec, active_slot=0, verbose=False, **ctx):
    """
    Option 1: split active vs. passive, fix pupil to mean level
    """
    # old: just take all active or passive
    #rec_active = ctx['rec'].and_mask("ACTIVE_EXPERIMENT").apply_mask()
    #rec_passive = ctx['rec'].and_mask("PASSIVE_EXPERIMENT").apply_mask()

    # new: just take first active or passive file
    e=rec['resp'].epochs
    fs = rec['resp'].fs
    ea_start=int(e.loc[e.name=="ACTIVE_EXPERIMENT",'start'].values[active_slot]*fs)
    ea_end=int(e.loc[e.name=="ACTIVE_EXPERIMENT",'end'].values[active_slot]*fs)
    ep_start=int(e.loc[e.name=="PASSIVE_EXPERIMENT",'start'].values[0]*fs)
    ep_end=int(e.loc[e.name=="PASSIVE_EXPERIMENT",'end'].values[0]*fs)

    m = rec['mask'].as_continuous()
    ma=m.copy()
    ma[0,:ea_start]=0
    ma[0,ea_end:]=0
    rec_active = rec.copy()
    rec_active['mask']=rec_active['mask']._modified_copy(data=ma)
    mp=m.copy()
    mp[0,:ep_start]=0
    mp[0,ep_end:]=0
    rec_passive = rec.copy()
    rec_passive['mask']=rec_passive['mask']._modified_copy(data=mp)

    rec_active = rec_active.apply_mask()
    rec_passive = rec_passive.apply_mask()

    # fix pupil size
    _d = rec_active['state'].as_continuous().copy()
    _d[1,:]=0
    rec_active['state']=rec_active['state']._modified_copy(data=_d)
    _d = rec_passive['state'].as_continuous().copy()
    _d[1,:]=0
    rec_passive['state']=rec_passive['state']._modified_copy(data=_d)

    rec_active = modelspec.evaluate(rec=rec_active)
    rec_passive = modelspec.evaluate(rec=rec_passive)

    if verbose:
        f,ax=plt.subplots(1,3,figsize=(16,2))
        ax[0].plot(rec_active['mask'].as_continuous()[0,:])
        ax[0].plot(rec_passive['mask'].as_continuous()[0,:])
        ax[1].plot(rec_passive['state'].as_continuous()[1:,:].T);
        ax[2].plot(rec_active['state'].as_continuous()[1:,:].T);

    return {'rec_passive': rec_passive, 'rec_active': rec_active}

def split_ls(rec, modelspec, verbose=False, **ctx):
    """
    Option 2: split large vs. small pupil (borrow same variable names from active/passive), fix to passive state.
    """
    rec_active = pupil_mask(rec, ctx['val'], 'large', False)[0].apply_mask()
    rec_passive = pupil_mask(rec, ctx['val'], 'small', False)[0].apply_mask()

    # fix behavior channels
    _d = rec_active['state'].as_continuous().copy()
    _d[2:,:]=0
    rec_active['state']=rec_active['state']._modified_copy(data=_d)
    _d = rec_passive['state'].as_continuous().copy()
    _d[2:,:]=0
    rec_passive['state']=rec_passive['state']._modified_copy(data=_d)

    rec_active = modelspec.evaluate(rec=rec_active)
    rec_passive = modelspec.evaluate(rec=rec_passive)

    if verbose:
        f,ax=plt.subplots(1,2,figsize=(12,2))
        ax[0].plot(rec_passive['state'].as_continuous()[1:,:].T);
        ax[1].plot(rec_active['state'].as_continuous()[1:,:].T);

    return {'rec_passive': rec_passive, 'rec_active': rec_active}


def split_pca(rec, modelspec, split_by='behavior', n_pc=4, memory=8, max_bins=4000, out_channel=None, verbose=False, **ctx):
    
    if out_channel is None:
        # analyze all output channels
        cellids = rec['resp'].chans
        siteids = [c.split("-")[0] for c in cellids]
        out_channel = list(np.arange(len(cellids)))
    else:
        cellids = [rec['resp'].chans[i] for i in out_channel]

    if split_by.startswith('behavior'):
        d=split_by.split(":")
        if len(d)==1:
           active_slot=0
        else:
           active_slot=int(d[-1])
        d = split_ap(rec=rec, modelspec=modelspec, active_slot=active_slot, verbose=verbose)
    elif split_by=='pupil':
        d = split_ls(rec=rec, modelspec=modelspec, verbose=verbose, val=ctx['val'])
    else:
        raise ValueError("invalid split by " + split_by)
    log.info(f'Processing {siteids[0]} {split_by}')
    rec_active = d['rec_active']
    rec_passive = d['rec_passive']

    channel_count=len(out_channel)

    # skip silent bins
    stim_mag = rec_active['stim'].as_continuous().sum(axis=0)
    index_range = np.arange(0, np.min([max_bins, len(stim_mag)]))
    stim_big = stim_mag > np.max(stim_mag) / 1000
    index_range = index_range[(index_range > memory) & stim_big[index_range]]

    log.info('Calculating rec_active dstrf for %d channels, %d timepoints, memory=%d',
         channel_count, len(index_range), memory)
    pcs_a, pc_mag_a = dstrf_pca(modelspec, rec_active, pc_count=n_pc, out_channel=out_channel,
                            index_range=index_range, memory=memory)

    log.info('Calculating rec_passive dstrf for %d channels, %d timepoints, memory=%d',
         channel_count, len(index_range), memory)
    pcs_p, pc_mag_p = dstrf_pca(modelspec, rec_passive, pc_count=n_pc, out_channel=out_channel,
                             index_range=index_range, memory=memory)

    keepchans = np.array(out_channel)
    #r = rec['resp'].as_continuous()[out_channel,:]
    #p = rec['pred'].as_continuous()[out_channel,:]
    #spaces = np.reshape(pcs[:,:,:,:], [n_pc, pcs.shape[1] * pcs.shape[2], pcs.shape[3]])

    f,axs=plt.subplots(4,10, figsize=(18,7))
    for c in range(np.min([channel_count,10])):
        cellid=cellids[c]
        for i in range(2):
            mm=np.max(np.abs(pcs_a[i,:,:,c]))
            _p = pcs_a[i,:,:,c]
            _p *= np.sign(_p.sum())
            axs[i,c].imshow(_p,aspect='auto',origin='lower', clim=[-mm, mm])
            axs[i,c].set_title(f'{cellid} act {pc_mag_a[i,c]:.3f}', fontsize=8)
            ax_remove_box(axs[i,c])

            mm=np.max(np.abs(pcs_p[i,:,:,c]))
            _p = pcs_p[i,:,:,c]
            _p *= np.sign(_p.sum())
            axs[i+2,c].imshow(_p,aspect='auto',origin='lower', clim=[-mm, mm])
            axs[i+2,c].set_title(f'{cellid} pas {pc_mag_p[i,c]:.3f}', fontsize=8)
            ax_remove_box(axs[i+2,c])
        
            axs[i,c].set_xticks([])
            if i<1:
                axs[i+2,c].set_xticks([])
            if c>0:
                axs[i,c].set_yticks([])

    f,ax=plt.subplots(1,3,figsize=(12,4))

    ax[0].plot(pc_mag_p);
    ax[0].set_title(f"{cellids[0]} - {split_by}")
    ax[1].plot(pc_mag_a);

    ax[2].plot(np.median(pc_mag_p, axis=1),color="gray")
    ax[2].plot(np.median(pc_mag_a, axis=1),color="black");

    return pc_mag_a, pc_mag_p, pcs_a, pcs_p


def ap_space_comp(pcs_a, pcs_p, n_pc, rec, modelspec, split_by='behavior', verbose=False, **ctx):

    if split_by.startswith('behavior'):
        d=split_by.split(":")
        if len(d)==1:
           active_slot=0
        else:
           active_slot=int(d[-1])
        d = split_ap(rec=rec, modelspec=modelspec, active_slot=active_slot, verbose=verbose)
    elif split_by=='pupil':
        d = split_ls(rec=rec, modelspec=modelspec, verbose=verbose, val=ctx['val'])
    else:
        raise ValueError("invalid split by " + split_by)

    rec_active = d['rec_active']
    rec_passive = d['rec_passive']
    out_channel = np.arange(rec_passive['resp'].shape[0])
    ra = rec_passive['resp'].as_continuous()[out_channel,:]
    rp = rec_active['resp'].as_continuous()[out_channel,:]
    a = rec_passive['pred'].as_continuous()[out_channel,:]
    p = rec_active['pred'].as_continuous()[out_channel,:]
    spaces_a = np.reshape(pcs_a[:2,:,:,:], [2*pcs_a.shape[1] * pcs_a.shape[2], pcs_a.shape[3]])
    spaces_p = np.reshape(pcs_p[:2,:,:,:], [2*pcs_p.shape[1] * pcs_p.shape[2], pcs_p.shape[3]])
    spaces_a.shape, spaces_p.shape

    f,ax=plt.subplots(1,3,figsize=(16,4))

    ua,rsa,va = np.linalg.svd(ra)
    up,rsp,vp = np.linalg.svd(rp)
    ax[0].plot(rsa)
    ax[0].plot(rsp);
    ax[0].set_title('response space')

    ua,psa,va = np.linalg.svd(a)
    up,psp,vp = np.linalg.svd(p)
    ax[1].plot(psa)
    ax[1].plot(psp);
    ax[1].set_title('pred space')

    ua,dsa,va = np.linalg.svd(spaces_a)
    up,dsp,vp = np.linalg.svd(spaces_p)
    ax[2].plot(dsa)
    ax[2].plot(dsp);
    ax[2].set_title('dSTRF space')

    return rsa, rsp, psa, psp, dsa, dsp


