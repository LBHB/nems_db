import os
import logging

import json as jsonlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.linalg import det
from scipy.ndimage import zoom, gaussian_filter1d

from nems import xforms
import nems.plots.api as nplt
from nems.plots.api import ax_remove_box, spectrogram, fig2BytesIO
from nems.plots.heatmap import _get_wc_coefficients, _get_fir_coefficients
from nems.uri import NumpyEncoder
from nems.utils import get_setting, smooth
from nems.modules.fir import per_channel


log = logging.getLogger(__name__)


def subspace_overlap(u, v):
    """
    cross correlation-like measure of overlap between two vector spaces
    from Sharpee PLoS CB 2017 paper
    u,v: n X x matrices sampling n-dim subspace of x-dim space
    """
    n = u.shape[1]

    _u = u / np.sqrt(np.sum(u ** 2, axis=0, keepdims=True))
    _v = v / np.sqrt(np.sum(v ** 2, axis=0, keepdims=True))

    num = np.power(np.abs(det(_u.T @ _v)), (1.0 / n))
    den = np.power(np.abs(det(_u.T @ _u)) * np.abs(det(_v.T @ _v)), (0.5 / n))

    return num / den


def compute_dstrf(modelspec, rec, index_range=None, sample_count=100, out_channel=[0], memory=10,
                  norm_mean=True, method='jacobian', **kwargs):

    # remove static nonlinearities from end of modelspec chain
    modelspec = modelspec.copy()
    """
    if ('double_exponential' in modelspec[-1]['fn']):
        log.info('removing dexp from tail')
        modelspec.pop_module()
    if ('relu' in modelspec[-1]['fn']):
        log.info('removing relu from tail')
        modelspec.pop_module()
    if ('levelshift' in modelspec[-1]['fn']):
        log.info('removing lvl from tail')
        modelspec.pop_module()
    """
    modelspec.rec = rec
    stimchans = rec['stim'].shape[0]
    bincount = rec['pred'].shape[1]
    stim_mean = np.mean(rec['stim'].as_continuous(), axis=1, keepdims=True)
    if index_range is None:
        index_range = np.arange(bincount)
        if sample_count is not None:
            np.random.shuffle(index_range)
            index_range = index_range[:sample_count]
    sample_count = len(index_range)
    dstrf = np.zeros((sample_count, stimchans, memory, len(out_channel)))
    for i, index in enumerate(index_range):
        if i % 50 == 0:
            log.info(f"dSTRF: {i}/{len(index_range)} idx={index}")
        if len(out_channel)==1:
            dstrf[i,:,:,0] = modelspec.get_dstrf(rec, index, memory, out_channel=out_channel, method=method)
        else:
            dstrf[i,:,:,:] = modelspec.get_dstrf(rec, index, memory, out_channel=out_channel, method=method)

    if norm_mean:
        dstrf *= stim_mean[np.newaxis, ..., np.newaxis]

    return dstrf


def strf_mtx(modelspec, rows=3, smooth=[1,2]):
    
    fir_layers = [idx for idx,m in enumerate(modelspec) if ('fir' in m['fn'])]
    wc_layers = np.array([idx for idx,m in enumerate(modelspec) if ('weight' in m['fn'])])
    fi = 0
    wc_coefs = _get_wc_coefficients(modelspec, fi)
    fir_coefs = _get_fir_coefficients(modelspec, fi)

    bank_count=modelspec[fir_layers[fi]]['fn_kwargs']['bank_count']
    chan_count = wc_coefs.shape[0]
    bank_chans = int(chan_count / bank_count)

    #print(wc_coefs.shape, fir_coefs.shape, bank_count, bank_chans)
    strfs = [zoom(wc_coefs[(bank_chans*i):(bank_chans*(i+1)),:].T @
                      fir_coefs[(bank_chans*i):(bank_chans*(i+1)), :], smooth, mode='constant')
                      for i in range(bank_count)]
    strfs = [np.pad(s,((0,0),(1,0))) for s in strfs]
    r_step = strfs[0].shape[0]+2
    c_step = strfs[0].shape[1]+int(np.ceil(strfs[0].shape[1]/rows))
    strf_all=np.full([r_step*rows-1, int(c_step * np.ceil((len(strfs)+1)/rows)+c_step/rows) ], np.nan)
    #print(strf_all.shape)

    for i in range(bank_count):
        m = np.max(np.abs(strfs[i]))
        if m:
            strfs[i] = strfs[i] / m
        
        r = (i%rows)*r_step
        c = int(i/rows)*c_step + ((rows-i)%rows)*int(strfs[0].shape[1]/rows)
        #print (r,c)

        strf_all[r:(r+strfs[i].shape[0]),c:(c+strfs[i].shape[1])]=strfs[i]         
        
    #f, ax=plt.subplots(1,1, figsize=(18,2))
    #ax.imshow(strf_all, origin='lower', aspect='auto')

    return strf_all, strfs


def plot_layer_outputs(modelspec, rec, index_range=None, sample_count=100, 
                       smooth=[2,2], figsize=None, example_idx=0, cmap='bwr',
                       performance_metric='r_ceiling', modelspec_ref=None, **kwargs):

    fs = rec['resp'].fs
    nl_layers = [idx for idx,m in enumerate(modelspec) if (('nonlinearity' in m['fn']) and ('dlog' not in m['fn']))]
    
    if index_range is None:
        index_range = np.arange(1000)
    elif len(index_range)==2:
        index_range = np.arange(index_range[0], index_range[1])
    elif len(index_range)==3:
        index_range = np.arange(index_range[0], index_range[1], index_range[2])
        
    pred = []
    for l in nl_layers:
        layer_output = modelspec.evaluate(rec, start=0, stop=l+1)
        pred.append(layer_output['pred'].as_continuous()[:,index_range])

    max_width = np.max([p.shape[0] for p in pred])
        
    rows = len(nl_layers)+2
    if figsize is None:
        figsize=(12, rows*1.5)

    f = plt.figure(constrained_layout=True, figsize=figsize)
    gs = f.add_gridspec(rows, 4)
    ax0 = f.add_subplot(gs[0, 0:2])
    ax1 = f.add_subplot(gs[1:(rows-1), 0:2])
    ax = [f.add_subplot(gs[i, 2:4]) for i in range(rows)]
    ax_zoom = f.add_subplot(gs[rows-1, 0])
    ax_perf = f.add_subplot(gs[rows-1, 1])

    srows=3
    strf_all, strfs = strf_mtx(modelspec, rows=srows, smooth=smooth)
    #mask=np.isnan(strf_all)
    #strf_all[mask]=0
    #strf_all = zoom(strf_all,[2,2])
    #mask = zoom(mask,[2,2])
    #strf_all[mask]=np.nan
    mm = np.nanmax(np.abs(strf_all))
   
    if cmap=='bwr':
        cmap = mpl.cm.get_cmap(name=get_setting('WEIGHTS_CMAP'))
        cmap.set_bad('lightgray',1.)
    elif cmap=='jet':
        cmap = mpl.cm.get_cmap(name='jet')
        cmap.set_bad('white',1.)
    else:
        cmap = mpl.cm.get_cmap(name=cmap)
        cmap.set_bad('white',1.)
        #cmap = mpl.cm.get_cmap(name='RdYlBu_r')
    ax0.imshow(strf_all, aspect='auto', cmap=cmap, origin='lower', interpolation='none', clim=[-mm, mm])
    ax0.set_axis_off()
    
    s = strfs[example_idx]
    extent = [0.5/(fs*smooth[1])-1.5/(fs*smooth[1]), (s.shape[1]+0.5)/(fs*smooth[1])-1.5/(fs*smooth[1]), 0.5, s.shape[0]+0.5]
    cl = np.nanmax(np.abs(s))
    im = ax_zoom.imshow(s, interpolation='none', aspect='auto', cmap=cmap, origin='lower', 
                        clim=[-cl,cl], extent=extent)
    plt.colorbar(im, ax=ax_zoom)
    ax_zoom.set_ylabel(f"input channel")
    ax_zoom.set_title(f"example L1 filter ({example_idx})")

    if modelspec_ref is not None:
        r_test0 = modelspec_ref.meta[performance_metric]
        ax_perf.plot(r_test0, color='lightgray')
    r_test = modelspec.meta[performance_metric]
    ax_perf.plot(r_test, color='black')
    ax_perf.plot([0, len(r_test)],[0, 0], '--', color='gray')
    ax_perf.set_xlabel('unit #')
    ax_perf.set_ylim([-0.05, 1])
    ax_perf.set_yticks([0,0.5,1])
    ax_perf.set_title(performance_metric)

    spec = rec['stim'].as_continuous()[:,index_range]
    extent = [0.5/fs, (spec.shape[1]+0.5)/fs, 0.5, spec.shape[0]+0.5]

    if np.mean(spec==0)>0.5:
       from nems_lbhb.tin_helpers import make_tbp_colormaps
       BwG, gR = make_tbp_colormaps()
       xx,yy=np.where(spec.T)
       colors = [BwG(i) for i in range(0,256,int(256/spec.shape[0]))]
       colors[-1]=gR(256)
       dur=0.3
       for x,y in zip(xx/fs,yy):
           ax[0].plot([x,x+dur],[y,y],'-',linewidth=2,color=colors[y])
       ax[0].set_xlim((extent[0],extent[1]))
    else:
       #im=ax.imshow(spec, origin='lower', interpolation='none', aspect='auto', extent=extent)

       ax[0].imshow(spec,aspect='auto',interpolation='none',origin='lower', cmap='gray_r')
    ax[0].set_ylabel("input channel")

    resp = rec['resp'].as_continuous()[:,index_range]
    extent = [0.5/fs, (resp.shape[1]+0.5)/fs, 0.5, resp.shape[0]+0.5]
    ax[-1].imshow(resp,aspect='auto',interpolation='none',origin='lower', extent=extent, cmap='gist_yarg')
    ax[-1].set_ylabel('resp')

    cmap="gist_yarg"
    cmap="Reds"
    for i,l in enumerate(nl_layers):
        p = pred[i]-pred[i][:,:1]
        extent = [0.5/fs, (p.shape[1]+0.5)/fs, 0.5, p.shape[0]+0.5]
        if i<len(nl_layers)-1:
            im=ax[i+1].imshow(p>0.01,aspect='auto',interpolation='none',origin='lower', 
                              extent=extent, cmap=cmap)
            ax[i+1].set_ylabel(f"L{i+1} activation")
        else:
            mm = np.nanmax(p)
            im=ax[i+1].imshow(p,aspect='auto',interpolation='none',origin='lower', 
                              clim=[0, mm], extent=extent, cmap="gist_yarg")
            ax[i+1].set_ylabel(f"pred")

        n1 = pred[i].shape[0]
        x1 = np.linspace(-n1/2, n1/2, n1)
        if i > 0:
            midx = nl_layers[i-1]+1
            if 'weight_channels' in modelspec[midx]['fn']:
                pass
            elif 'weight_channels' in modelspec[midx+1]['fn']:
                midx+=1
            c = modelspec.phi[midx].get('coefficients',None)
            if c is not None:
                c = c/np.max(np.abs(c))
            n0 = pred[i-1].shape[0]
            x0 = np.linspace(-n0/2, n0/2, n0)
            
            for i0 in range(0,n0):
                for i1 in range(0,n1):
                    if c is not None:
                        col = c[i1,i0]
                        if np.abs(col)>0.2:
                            if col>0:
                                col = [(1-col), 1-0.5*col, 1-0.5*col]
                            else:
                                col = [1+0.5*col, 1+0.5*col, 1+col]
                            ax1.plot([x0[i0], x1[i1]],[rows - (i), rows - (i+1)], 
                                     color=col, linewidth=0.5)
                    else:
                        col = [.5, .5, .5]
                        ax1.plot([x0[i0], x1[i1]],[rows - (i), rows - (i+1)], 
                                 color=col, linewidth=0.5)

        else:
            n0 = max_width
            x0 = np.linspace(-n0/2, n0/2, n1)
            #print(-n0/2, n0/2)

            for i1 in range(srows-1,n1,srows):
                ax1.plot([x0[i1], x1[i1]],[rows - (i+0.5), rows - (i+1)], color=[.5, .5, .5], linewidth=0.5)
                #print(x0[i1], x1[i1])
    for i,l in enumerate(nl_layers):

        n1 = pred[i].shape[0]
        x1 = np.linspace(-n1/2, n1/2, n1)
        ax1.scatter(x1, np.zeros(n1) + rows - (i+1), edgecolors='black', facecolors='white')
        #print(x1[0], x1[-1])
    ax1.set_ylim([1.5, rows-0.5])
    ax1.set_axis_off()
    modelname = modelspec.meta['modelname'].split("_")[1] 
    f.suptitle(f"{modelspec.meta['cellid']} {modelname} {index_range[0]}-{index_range[-1]}")
    return f

def force_signal_silence(rec, signal='stim'):
    rec = rec.copy()
    for e in ["PreStimSilence", "PostStimSilence"]:
        s = rec[signal].extract_epoch(e)
        s[:,:,:]=0
        rec[signal]=rec[signal].replace_epoch(e, s)
    return rec

def dstrf_details(rec,cellid,rr,dindex, dstrf=None, dpcs=None, memory=20, stepbins=3, maxbins=1500, n_pc=3):
    cellids = rec['resp'].chans
    match=[c==cellid for c in cellids]
    c = np.where(match)[0][0]
        
    # analyze all output channels
    out_channel = [c]
    channel_count=len(out_channel)

    if dstrf is not None:
        stimmag = dstrf.shape[0]

    rec = force_signal_silence(rec)
    stim_mag = rec['stim'].as_continuous()[:,:maxbins].sum(axis=0)
    index_range = np.arange(0, len(stim_mag), 1)
    if dstrf is None:
        log.info('Calculating dstrf for %d channels, %d timepoints (%d steps), memory=%d',
                 channel_count, len(index_range), stepbins, memory)
        dstrf = compute_dstrf(modelspec, rec.copy(), out_channel=out_channel,
                              memory=memory, index_range=index_range)

    if dpcs is None:
        # don't skip silent bins

        stim_big = stim_mag > np.max(stim_mag) / 1000
        driven_index_range = index_range[(index_range > memory) & stim_big[index_range]]
        driven_index_range = driven_index_range[(driven_index_range >= memory)]

        # limit number of bins to speed up analysis
        driven_index_range = driven_index_range[:maxbins]
        #index_range = np.arange(525,725)
    
        pcs, pc_mag = compute_dpcs(dstrf[driven_index_range,:,:,:], pc_count=n_pc)
    
    c_=0

    ii = np.arange(rr[0],rr[1])
    rr_orig = ii
    
    print(pcs.shape, dstrf.shape)
    u = np.reshape(pcs[:,:,:,c_],[n_pc, -1])
    d = np.reshape(dstrf[ii,:,:,c_],[len(ii),-1])
    pc_proj = d @ u.T

    stim = rec['stim'].as_continuous()
    pred=np.zeros((pcs.shape[0],stim.shape[1]))
    for i in range(pcs.shape[0]):
        pred[i,:] = per_channel(stim, np.fliplr(pcs[i,:,:,0]))

    n_strf=len(dindex)

    f = plt.figure(constrained_layout=True, figsize=(18,8))
    gs = f.add_gridspec(5, n_strf)
    ax0 = f.add_subplot(gs[0, :])
    ax1 = f.add_subplot(gs[1, :])
    ax2 = f.add_subplot(gs[2, :])
    ax = np.array([f.add_subplot(gs[3, i]) for i in range(n_strf)])
    ax3 = np.array([f.add_subplot(gs[4, i]) for i in range(n_strf)])

    ax0.imshow(rec['stim'].as_continuous()[:, rr_orig], aspect='auto', origin='lower', cmap="gray_r")
    xl=ax0.get_xlim()

    ax1.plot(rec['resp'].as_continuous()[c, rr_orig], color='gray');
    ax1.plot(rec['pred'].as_continuous()[c, rr_orig], color='purple');
    ax1.legend(('actual','pred'), frameon=False)
    ax1.set_xlim(xl)
    yl1=ax1.get_ylim()

    #ax2.plot(pc_proj);
    ax2.plot(pred[:,rr_orig].T);
    ax2.set_xlim(xl)
    ax2.set_ylabel('pc projection')
    ax2.legend(('PC1','PC2','PC3'), frameon=False)
    yl2=ax2.get_ylim()

    dindex = np.array(dindex)
    mmd=np.max(np.abs(dstrf[np.array(dindex)+rr[0],:,:,c_]))
    stim = rec['stim'].as_continuous()[:,rr_orig] ** 2
    mms = np.max(stim)
    stim /= mms
    
    for i,d in enumerate(dindex):
        ax1.plot([d,d],yl1,'--', color='darkgray')
        ax2.plot([d,d],yl2,'--', color='darkgray')
        _dstrf = dstrf[d+rr[0],:,:,c_]
        if True:
            #_dstrf = np.concatenate((_dstrf,stim[:,(d-_dstrf.shape[1]):d]*mmd), axis=0)
            _dstrf = np.concatenate((_dstrf,_dstrf*stim[:,(d-_dstrf.shape[1]):d]), axis=0)

            #_dstrf *= stim[:,(d-_dstrf.shape[1]):d]
        ds = np.fliplr(_dstrf)
        ds=zoom(ds, [2,2])
        ax[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmd, mmd], cmap=get_setting('WEIGHTS_CMAP'))
        #plot_heatmap(ds, aspect='auto', ax=ax[i], interpolation=2, clim=[-mmd, mmd], show_cbar=False, xlabel=None, ylabel=None)

        ax[i].set_title(f"Frame {d}", fontsize=8)
        if i<n_pc:
            ds=np.fliplr(pcs[i,:,:,c_])
            ds=zoom(ds, [2,2])
            mmp = np.max(np.abs(ds))
            #ax3[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmp, mmp])
            ax3[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmp, mmp], cmap=get_setting('WEIGHTS_CMAP'))
        else:
            ax3[i].set_axis_off()

    ax[0].set_ylabel('example frames')
    ax3[0].set_ylabel('PCs')

    return f


def compute_dpcs(dstrf, pc_count=3):

    #from sklearn.decomposition import PCA

    channel_count=dstrf.shape[3]
    s = list(dstrf.shape)
    s[0]=pc_count
    pcs = np.zeros(s)
    pc_mag = np.zeros((pc_count,channel_count))
    #import pdb; pdb.set_trace()
    for c in range(channel_count):
        d = np.reshape(dstrf[:, :, :, c], (dstrf.shape[0], s[1]*s[2]))
        #d -= d.mean(axis=0, keepdims=0)

        _u, _s, _v = np.linalg.svd(d.T @ d)
        _s = np.sqrt(_s)
        _s /= np.sum(_s[:pc_count])
        pcs[:, :, :, c] = np.reshape(_v[:pc_count, :],[pc_count, s[1], s[2]])
        pc_mag[:, c] = _s[:pc_count]

    return pcs, pc_mag


def dstrf_pca(modelspec, rec, pc_count=3, out_channel=[0], memory=10, return_dstrf=False,
              pca_index_range=None, **kwargs):

    dstrf = compute_dstrf(modelspec, rec.copy(), out_channel=out_channel,
                          memory=memory, **kwargs)

    if pca_index_range is None:
        pcs, pc_mag = compute_dpcs(dstrf, pc_count)
    else:
        pcs, pc_mag = compute_dpcs(dstrf[pca_index_range, :, :], pc_count)

    if return_dstrf:
       return pcs, pc_mag, dstrf
    else:
       return pcs, pc_mag


def pop_space_summary(recname='est', modelspec=None, rec=None, figures=None, n_pc=3, memory=20, maxbins=1000, stepbins=3, IsReload=False, batching=True, **ctx):

    if IsReload and batching:
        return {}

    if figures is None:
        figs = []
    else:
        figs = figures.copy()

    rec = ctx[recname]
    
    cellids = rec['resp'].chans
    siteids = [c.split("-")[0] for c in cellids]
    modelname = modelspec.meta['modelname']
    if 'cc20.bth' in modelname:
       fmt='bth'   # both single site and random
    elif 'cc20.rnd' in modelname:
       fmt='rnd'
    else:
       fmt='sng'  # single site only

    # analyze all output channels
    out_channel = list(np.arange(len(cellids)))
    channel_count=len(out_channel)

    # skip silent bins
    stim_mag = rec['stim'].as_continuous().sum(axis=0)
    stim_big = stim_mag > np.max(stim_mag) / 1000
    index_range = np.arange(0, len(stim_mag), stepbins)
    index_range = index_range[(index_range > memory) & stim_big[index_range]]

    # limit number of bins to speed up analysis
    index_range = index_range[:maxbins]

    log.info('Calculating dstrf for %d channels, %d timepoints (%d steps), memory=%d',
             channel_count, len(index_range), stepbins, memory)
    log.info(rec.signals.keys())
    pcs, pc_mag = dstrf_pca(modelspec, rec, pc_count=n_pc, out_channel=out_channel,
                           index_range=index_range, memory=memory)

    spaces = np.reshape(pcs, [n_pc, pcs.shape[1] * pcs.shape[2], pcs.shape[3]])

    keepchans=np.array(out_channel)
    r = rec['resp'].as_continuous()[out_channel,:]
    p = rec['pred'].as_continuous()[out_channel,:]

    olapcount = channel_count
    overlap = np.zeros((olapcount, olapcount))
    olap_same_site = []
    olap_diff_site = []
    olap_part_site = []
    r_cc_same_site = []
    r_cc_diff_site = []
    r_cc_part_site = []
    p_cc_same_site = []
    p_cc_diff_site = []
    p_cc_part_site = []
    for i in range(olapcount):
        for j in range(i):
            overlap[i, j] = subspace_overlap(spaces[:, :, i].T, spaces[:, :, j].T)
            overlap[j, i] = np.corrcoef(p[i, :], p[j, :])[0, 1]
            r_cc = np.corrcoef(r[i, :], r[j, :])[0, 1]

            if (fmt=='sng') | ((fmt=='bth') & (siteids[i] == siteids[0]) & (siteids[j] == siteids[0])):
                olap_same_site.append(overlap[i, j])
                p_cc_same_site.append(overlap[j, i])
                r_cc_same_site.append(r_cc)
            elif (fmt=='bth') & (siteids[j] == siteids[0]):
                olap_diff_site.append(overlap[i, j])
                p_cc_diff_site.append(overlap[j, i])
                r_cc_diff_site.append(r_cc)
            else:
                olap_part_site.append(overlap[i, j])
                p_cc_part_site.append(overlap[j, i])
                r_cc_part_site.append(r_cc)
    log.info(
        f"PC space same {np.mean(olap_same_site):.4f} partial: {np.mean(olap_part_site):.4f} diff: {np.mean(olap_diff_site):.4f}")
    log.info(
        f"Pred CC same {np.mean(p_cc_same_site):.4f} partial: {np.mean(p_cc_part_site):.4f} diff: {np.mean(p_cc_diff_site):.4f}")
    log.info(
        f"Resp CC same {np.mean(r_cc_same_site):.4f} partial: {np.mean(r_cc_part_site):.4f} diff: {np.mean(r_cc_diff_site):.4f}")

    f = plt.figure(constrained_layout=True, figsize=(10,8))
    show_pcs=5
    gs = f.add_gridspec(4, show_pcs+1)
    ax0 = f.add_subplot(gs[:2, :3])
    ax1 = f.add_subplot(gs[:2, 3:])
    ax = np.array([[f.add_subplot(gs[2, i]) for i in range(6)],
                   [f.add_subplot(gs[3, i]) for i in range(6)]])
    modelname_list = "\n".join(modelspec.meta["modelname"].split("_"))
    spectrogram(rec, sig_name='pred', title=None, ax=ax0)
    ax1.imshow(overlap, clim=[-1, 1])
    for i, s in enumerate(siteids[:olapcount]):
        if i > 0 and (siteids[i] != siteids[i - 1]) and (siteids[i - 1] == siteids[0]):
            ax1.plot([0, olapcount - 1], [i - 0.5, i - 0.5], 'b', linewidth=0.5)
            ax1.plot([i - 0.5, i - 0.5], [0, olapcount - 1], 'b', linewidth=0.5)
    ax1.text(olapcount+1, 0, f'PC space same {np.mean(olap_same_site):.3f}\n' +\
                               f'partial: {np.mean(olap_part_site):.3f}\n' +\
                               f'Resp CC same {np.mean(r_cc_same_site):.3f}\n' +\
                               f'partial: {np.mean(r_cc_part_site):.3f}\n' +\
                               f'Pred CC same {np.mean(p_cc_same_site):.3f}\n' +\
                               f'partial: {np.mean(p_cc_part_site):.3f}\n',
               va='top', fontsize=8)
    ax1.set_ylabel('diff site -- same site')
    ax_remove_box(ax1)

    sspaces = spaces * np.expand_dims(pc_mag, axis=1)
    cellcount=spaces.shape[-1]
    if fmt=='bth':
       n = int(cellcount/2)
       bspace1_ = np.transpose(sspaces[:,:,:n],[0,2,1])
       bspace2_ = np.transpose(sspaces[:,:,n:],[0,2,1])
    else:
       n = cellcount
       bspace1_ = np.transpose(sspaces,[0,2,1])
       bspace2_ = np.transpose(sspaces,[0,2,1])

    s=bspace1_.shape
    print(s)
    bspace1 = np.reshape(bspace1_, (s[0]*s[1], s[2]))
    bspace2 = np.reshape(bspace2_, (s[0]*s[1], s[2]))
    print(bspace1.shape, bspace2.shape)
    u1, s1, v1 = np.linalg.svd(bspace1.T @ bspace1)
    u2, s2, v2 = np.linalg.svd(bspace2.T @ bspace2)

    ax[0,0].plot(s1[:10]/np.sum(s1[:10]))
    ax[0,0].plot(s2[:10]/np.sum(s2[:10]))
    ax[0,0].set_ylabel("PC mag")
    
    ax[1,0].set_axis_off()
    
    for i in range(show_pcs):
        p1=u1[:,i]
        p1=np.reshape(p1,[pcs.shape[1],memory])
        p1 /= np.max(np.abs(p1))
        ax[0,i+1].imshow(np.fliplr(p1),origin='lower', clim=[-1, 1])
        ax[0,i+1].set_title(f"PC {i}")

        p2=u2[:,i]
        p2=np.reshape(p2,[pcs.shape[1],memory])    
        p2 /= np.max(np.abs(p2))
        ax[1,i+1].imshow(np.fliplr(p2),origin='lower', clim=[-1, 1])

    f.suptitle(f'{siteids[0]} pred PSTH\n{modelname_list}');

    figs.append(fig2BytesIO(f))

    f2,axs=plt.subplots(8, 10, figsize=(16,12))
    for c in range(np.min([40,channel_count])):
        cellid = cellids[c]
        for i in range(2):
            mm=np.max(np.abs(pcs[i,:,:,c]))
            _p = pcs[i,:,:,c]
            _p *= np.sign(_p.sum())
            os = int(c/10)*2
            _c = c % 10
            axs[i+os,_c].imshow(np.fliplr(_p),aspect='auto',origin='lower', clim=[-mm, mm])
            if i==0:
               axs[i+os,_c].set_title(f'{cellid} {pc_mag[i,c]:.3f}', fontsize=8)
            else:
               axs[i+os,_c].set_title(f'{pc_mag[i,c]:.3f}', fontsize=8)
            if i+os<7:
                axs[i+os,_c].set_xticks([])
            if c>0:
                axs[i+os,_c].set_yticks([])
            ax_remove_box(axs[i+os, _c])

    figs.append(fig2BytesIO(f2))

    modelspec.meta['olap_same_site'] = olap_same_site
    modelspec.meta['olap_part_site'] = olap_part_site
    modelspec.meta['r_cc_same_site'] = r_cc_same_site
    modelspec.meta['r_cc_part_site'] = r_cc_part_site
    modelspec.meta['p_cc_same_site'] = p_cc_same_site
    modelspec.meta['p_cc_part_site'] = p_cc_part_site
    modelspec.meta['pc_mag_same_site'] = s1[:10]
    modelspec.meta['pc_mag_part_site'] = s2[:10]
    modelspec.meta['pc_mag'] = pc_mag

    if batching:
        return {'figures': figs, 'modelspec': modelspec}
    else:
        return {'figures': [f, f2], 'modelspec': modelspec, 'pc_mag': pc_mag, 'pcs': pcs}


def dstrf_movie(rec, dstrf, out_channel, index_range, static=False, preview=False, mult=False, out_path="/tmp", 
                out_base=None, fps=10, **kwargs):
    #plt.close('all')

    cellcount=len(out_channel)
    framecount=len(index_range)
    cellids = [rec['resp'].chans[o] for o in out_channel]
    fs=rec['resp'].fs
    i = 0
    index = index_range[i]
    memory = dstrf.shape[2]

    stim=rec['stim'].as_continuous()-0.05
    stim[stim<0]=0
    stim = np.exp(stim*2)-1
    
    stim_lim = np.max(stim[:, index_range])*0.5

    s = stim[:, (index - memory*3):index]
    print("stim_lim ", stim_lim)

    im_list = []
    l1_list = []
    l2_list = []

    if static:
        max_frames = np.min([10, framecount])
        f, axs = plt.subplots(cellcount+1, max_frames, figsize=(16, 8))
        stim0col=0
    else:
        max_frames = framecount
        f, axs = plt.subplots(cellcount+1, 2, figsize=(4, 8))
        stim0col=1
    f.show()

    im_list.append(axs[0, stim0col].imshow(s, clim=[0, stim_lim], interpolation='none', origin='lower', 
                                           aspect='auto', cmap='Greys'))
    axs[0, stim0col].set_ylabel('stim')
    axs[0, stim0col].set_yticks([])
    axs[0, stim0col].set_xticks([])
    _title1 = axs[0, stim0col].set_title(f"dSTRF frame {index / fs:.3f}")

    axs[0, 0].set_axis_off()
    for cellidx in range(cellcount):
        d = dstrf[i, :, :, cellidx].copy()

        pred_max = np.max(rec['pred'].as_continuous()[cellidx, :])
        strf_lim = np.max(np.abs(dstrf[:, :, -1, cellidx])) * 2.0

        if mult:
            strf_lim = strf_lim * stim_lim
            d = d * s
        d=zoom(d,[2,2])
        print(f"strf_lim[{cellidx}]: {strf_lim} shape: {d.shape}")
        im_list.append(axs[cellidx+1,0].imshow(d, clim=[-strf_lim, strf_lim],
                                               interpolation='none', origin='lower',
                                               aspect='auto', cmap='bwr'))
        axs[cellidx+1, 0].set_yticks([])
        axs[cellidx+1, 0].set_xticks([])
        axs[cellidx+1, 0].set_ylabel(f"{cellids[cellidx]}", fontsize=6)
        ax_remove_box(axs[cellidx+1,0])

        if cellidx<cellcount-1:
            axs[cellidx+1, 1].set_xticklabels([])

        if not static:
           l1_list.append(axs[cellidx+1, 1].plot(rec['pred'].as_continuous()[out_channel[cellidx], (index - memory*2):index], '-', color='purple')[0])
           l2_list.append(axs[cellidx+1, 1].plot(rec['resp'].as_continuous()[out_channel[cellidx], (index - memory*2):index], '-', color='gray')[0])
           axs[cellidx+1, 1].set_ylim((0, pred_max))
           axs[cellidx+1, 1].set_yticks([])

    def dstrf_frame(i):
        index = index_range[i]
        s = stim[:, (index - memory*2):index]
        im_list[0].set_data(s)

        for cellidx in range(cellcount):
            if mult:
                d = dstrf[i, :, :, cellidx] * s
            else:
                d = dstrf[i, :, :, cellidx]
            d=zoom(d,[2,2])
            im_list[cellidx+1].set_data(d)

            _t = np.arange(memory*2)
            _r = rec['resp'].as_continuous()[cellidx, (index - memory*2-1):(index+1)]
            smooth_window=0.75
            #import pdb; pdb.set_trace()
            _r = gaussian_filter1d(_r, smooth_window, axis=0)[1:-1]

            #print(f"{index}: {_t.shape} {_r.shape}")
            l1_list[cellidx].set_data((_t, rec['pred'].as_continuous()[cellidx, (index - memory*2):index]))
            l2_list[cellidx].set_data((_t, _r))

        _title1.set_text(f"dSTRF frame {index/fs:.3f}")

        return tuple(im_list + l1_list + l2_list + [_title1])

    if static:
        #stim = np.sqrt(np.sqrt(rec['stim'].as_continuous()))
        #stim_lim = np.max(stim[:, index_range])
        #print("stim_lim ", stim_lim)

        for i in range(1, max_frames):
           index = index_range[i]

           s = stim[:, (index - memory):index]
           axs[0, i].imshow(s, clim=[0, stim_lim], interpolation='none', origin='lower', aspect='auto', cmap='Greys')
           axs[0, i].set_title(f"dSTRF frame {index}")
           axs[0, i].set_yticks([])
           axs[0, i].set_xticks([])

           for cellidx in range(cellcount):
               d = dstrf[i, :, :, cellidx].copy()

               pred_max = np.max(rec['pred'].as_continuous()[cellidx, :])
               strf_lim = np.max(np.abs(dstrf[:, :, :, cellidx])) / 2
               if mult:
                   strf_lim = strf_lim * stim_lim
                   d = d * s
               d=zoom(d,[2,2])
               axs[cellidx+1, i].imshow(d, clim=[-strf_lim, strf_lim], interpolation='none', origin='lower', 
                                        aspect='auto', cmap='bwr')
               axs[cellidx+1, i].set_yticks([])
               axs[cellidx+1, i].set_xticks([])
               #axs[cellidx+1, i].set_ylabel(f"{cellids[cellidx]}", fontsize=6)
               ax_remove_box(axs[cellidx+1, i])
        if out_base is None:
            if mult:
                out_base = f'{cellid}_{index_range[0]}-{index_range[-1]}_masked.pdf'
            else:
                out_base = f'{cellid}_{index_range[0]}-{index_range[-1]}.pdf'
        out_file = os.path.join(out_path, out_base)
        print(f'saving to: {out_file}')
        f.savefig(out_file)

    elif preview:
        for i in range(framecount):
            dstrf_frame(i)
            plt.pause(0.01)
    else:
        if out_base is None:
            if mult:
                out_base = f'{cellid}_{index_range[0]}-{index_range[-1]}_masked.mp4'
            else:
                out_base = f'{cellid}_{index_range[0]}-{index_range[-1]}.mp4'
        out_file = os.path.join(out_path, out_base)
        print(f'saving to: {out_file}')

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        line_ani = animation.FuncAnimation(f, dstrf_frame, framecount, fargs=(),
                                           interval=1, blit=True)
        line_ani.save(out_file, writer=writer)


def make_movie(ctx, cellid=None, out_channel=0, memory=10, index_range=None, **kwargs):

    rec = ctx['val'].apply_mask()
    modelspec = ctx['modelspec']
    index_range = index_range[index_range>memory]

    if cellid is not None:
        out_channel = [int(np.where([cellid == c for c in rec['resp'].chans])[0][0])]  # 26  #16 # 30
    else:
        if type(out_channel) is not list:
            out_channel = [out_channel]

        cellid = rec['resp'].chans[out_channel[0]]

    batch = modelspec.meta['batch']
    modelspecname = modelspec.meta.get('modelspecname', modelspec.meta['modelname'])
    print(f'cell {cellid} is chan {out_channel[0]}. computing dstrf with memory={memory}')
    out_base = f'{cellid}_{batch}_{modelspecname}_{index_range[0]}-{index_range[-1]}.mp4'

    dstrf = compute_dstrf(modelspec, rec.copy(), out_channel=out_channel,
                          index_range=index_range, memory=memory, **kwargs)

    dstrf_movie(rec, dstrf, out_channel, index_range, out_base=out_base, **kwargs)


def pop_test(out_channel=[26,7,8], index_range=None):
    # out_channel, index_range=[26,7,8], np.arange(200,550)
    if index_range is None:
        index_range = np.arange(200,550)

    modelpath = "/Users/svd/python/nems/results/271/TAR010c/TAR010c.dlog_wc.18x3.g_fir.1x10x3_relu.3_wc.3x55_lvl.55.unknown_fitter.2020-06-20T153243"
    cellid = "TAR010c-36-1"
    memory = 10
    xfspec, ctx = xforms.load_analysis(modelpath)

    make_movie(ctx, out_channel=out_channel, memory=memory, index_range=index_range, preview=True)

def pop_pca():
    modelpath = "/Users/svd/python/nems/results/271/TAR010c/TAR010c.dlog_wc.18x3.g_fir.1x10x3_relu.3_wc.3x55_lvl.55.unknown_fitter.2020-06-20T153243"
    cellid = "TAR010c-36-1"
    xfspec, ctx = xforms.load_analysis(modelpath)

    rec = ctx['val'].apply_mask()
    modelspec = ctx['modelspec']
    stim_mag = rec['stim'].as_continuous().sum(axis=0)

    memory = 10
    out_channel = [2,5,11,17]
    index_range = np.arange(0, len(stim_mag))
    stim_big = stim_mag>np.max(stim_mag)/1000
    index_range = index_range[(index_range>memory) & stim_big[index_range]]
    cellid = rec['resp'].chans[out_channel[0]]

    batch = modelspec.meta['batch']
    modelspecname = modelspec.meta.get('modelspecname', modelspec.meta['modelname'])
    print(f'cell {cellid} is chan {out_channel[0]}. computing dstrf with memory={memory}')

    channel_count = len(out_channel)
    pc_count = 5
    pcs,pc_mag=dstrf_pca(modelspec, rec, pc_count=pc_count, out_channel=out_channel,
                 index_range=index_range, memory=memory)

    f,axs=plt.subplots(pc_count,channel_count)
    for c in range(channel_count):
        for i in range(pc_count):
            mm=np.max(np.abs(pcs[i,:,:,c]))
            _p = pcs[i,:,:,c]
            _p *= np.sign(_p.sum())
            axs[i,c].imshow(_p,aspect='auto',origin='lower', clim=[-mm, mm])
            axs[i,c].set_title(f'pc {i}: {pc_mag[i,c]:.3f}')


def stp_test():
    modelpath = "/Users/svd/python/nems/results/271/TAR010c-18-1/TAR010c.dlog_wc.18x1.g_fir.1x15_lvl.1_dexp.1.unknown_fitter.2020-06-25T204004"
    modelpath = "/Users/svd/python/nems/results/271/TAR010c-18-1/TAR010c.dlog_wc.18x1.g_stp.1.q.s_fir.1x15_lvl.1_dexp.1.unknown_fitter.2020-06-22T031852"
    cellid = "TAR010c-18-1"
    index_range = np.arange(200, 400)
    memory = 10
    xfspec, ctx = xforms.load_analysis(modelpath)

    make_movie(ctx, cellid=cellid, memory=memory, index_range=index_range, preview=True)

def stp_pca():
    modelpath = "/Users/svd/python/nems/results/271/TAR010c-18-1/TAR010c.dlog_wc.18x1.g_fir.1x15_lvl.1_dexp.1.unknown_fitter.2020-06-25T204004"
    modelpath = "/Users/svd/python/nems/results/271/TAR010c-18-1/TAR010c.dlog_wc.18x1.g_stp.1.q.s_fir.1x15_lvl.1_dexp.1.unknown_fitter.2020-06-22T031852"
    cellid = "TAR010c-18-1"
    index_range = np.arange(0, 1000)
    memory = 10
    xfspec, ctx = xforms.load_analysis(modelpath)

    rec = ctx['val'].apply_mask()
    modelspec = ctx['modelspec']
    index_range = index_range[index_range>memory]

    if cellid is not None:
        out_channel = [int(np.where([cellid == c for c in rec['resp'].chans])[0][0])]  # 26  #16 # 30
    else:
        if type(out_channel) is not list:
            out_channel = [out_channel]

        cellid = rec['resp'].chans[out_channel[0]]

    batch = modelspec.meta['batch']
    modelspecname = modelspec.meta.get('modelspecname', modelspec.meta['modelname'])
    print(f'cell {cellid} is chan {out_channel[0]}. computing dstrf with memory={memory}')

    channel_count = len(out_channel)
    pc_count = 3
    pcs,pc_mag=dstrf_pca(modelspec, rec, pc_count=pc_count, out_channel=out_channel,
                 index_range=index_range, memory=memory)

    c=0
    f,axs=plt.subplots(pc_count,channel_count)
    for i in range(pc_count):
        mm=np.max(np.abs(pcs[i,:,:,c]))
        axs[i].imshow(pcs[i,:,:,c],aspect='auto',origin='lower', clim=[-mm, mm])
        axs[i].set_title(f'pc {i}: {pc_mag[i,c]:.3f}')


def db_test(out_channel=[6,7,8]):
    #get_results_file(batch, modelnames=None, cellids=None)
    from nems.db import get_results_file
    batch = 289
    #modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2.q-fir.2x15-lvl.1-dexp.1_tfinit.n.lr1e3-newtf.n.lr1e4'
    modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-fir.4x15-lvl.1-dexp.1_tfinit.n.lr1e3-newtf.n.lr1e4'

    # model from Jake
    #modelname = "ozgf.fs50.ch18.pop-ld-norm-pca.no-popev_dlog-wc.18x16.g-fir.1x8x16-relu.16-wc.16x24.z-relu.24-wc.24xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.rb5-newtf.n.lr1e4"
    modelname = 'ozgf.fs50.ch18.pop-ld-norm-pca.no-popev_dlog-conv2d.8.3x3.rep5-wcn.8-wc.8x12.z-relu.12-wc.12xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.rb5-newtf.n.lr1e4'
    modelname = 'ozgf.fs50.ch18.pop-ld-norm-pca.no-popev_dlog-wc.18x32.g-fir.1x8x32-relu.32-wc.32x60.z-relu.60-wc.60xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.rb5-newtf.n.lr1e4'

    # old SVD dev model
    #modelname = "ozgf.fs50.ch18.pop-loadpop.cc20-norm-pca.no-popev_dlog-wc.18x12.g-fir.1x8x12-relu.12-stp.12.q.s-wc.12x12.z-relu.12-wc.12xR.z-lvl.R-dexp.R_init.tf.n.rb5-tf.n.fa.lr5e3"

    batch = 289
    cellid = 'TAR010c-18-2'
    cellid= "AMT003c-32-1"
    d = get_results_file(batch, [modelname], [cellid])
    modelpath = d.modelpath[0]

    index_range = np.arange(2250, 2750)
    #index_range = np.arange(100, 4000)
    memory = 8

    xf, ctx = load_model_xform(cellid, batch, modelname)
    make_movie(ctx, out_channel=out_channel, memory=memory, index_range=index_range, preview=False, mult=False,
               out_path='/auto/data/tmp/')


def db_load():
    from nems.xform_helper import load_model_xform
    batch = 289
    #modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2.q-fir.2x15-lvl.1-dexp.1_tfinit.n.lr1e3-newtf.n.lr1e4'
    modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-fir.4x15-lvl.1-dexp.1_tfinit.n.lr1e3-newtf.n.lr1e4'

    # model from Jake
    modelname = 'ozgf.fs50.ch18.pop-ld-norm-pca.no-popev_dlog-conv2d.8.3x3.rep5-wcn.8-wc.8x12.z-relu.12-wc.12xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.rb5-newtf.n.lr1e4'
    modelname = 'ozgf.fs50.ch18.pop-ld-norm-pca.no-popev_dlog-wc.18x32.g-fir.1x8x32-relu.32-wc.32x60.z-relu.60-wc.60xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.rb5-newtf.n.lr1e4'
    #modelname = "ozgf.fs50.ch18.pop-ld-norm-pca.no-popev_dlog-wc.18x16.g-fir.1x8x16-relu.16-wc.16x24.z-relu.24-wc.24xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.rb5-newtf.n.lr1e4"

    # old SVD dev model
    #modelname = "ozgf.fs50.ch18.pop-loadpop.cc20-norm-pca.no-popev_dlog-wc.18x12.g-fir.1x8x12-relu.12-stp.12.q.s-wc.12x12.z-relu.12-wc.12xR.z-lvl.R-dexp.R_init.tf.n.rb5-tf.n.fa.lr5e3"

    batch = 289
    cellid = 'TAR010c-18-2'
    cellid= "AMT003c-32-1"

    xf,ctx = load_model_xform(cellid, batch, modelname)
    
    return xf,ctx


def db_pca():
    xfspec, ctx = db_load()

    rec = ctx['val'].apply_mask()
    modelspec = ctx['modelspec']
    stim_mag = rec['stim'].as_continuous().sum(axis=0)

    memory = 10
    out_channel = [2,5,11,17]
    index_range = np.arange(0, len(stim_mag))
    stim_big = stim_mag>np.max(stim_mag)/1000
    index_range = index_range[(index_range>memory) & stim_big[index_range]]
    cellid = rec['resp'].chans[out_channel[0]]

    batch = modelspec.meta['batch']
    modelspecname = modelspec.meta.get('modelspecname', modelspec.meta['modelname'])
    print(f'cell {cellid} is chan {out_channel[0]}. computing dstrf with memory={memory}')

    channel_count = len(out_channel)
    pc_count = 5
    pcs,pc_mag=dstrf_pca(modelspec, rec, pc_count=pc_count, out_channel=out_channel,
                 index_range=index_range, memory=memory)

    f,axs=plt.subplots(pc_count,channel_count)
    for c in range(channel_count):
        for i in range(pc_count):
            mm=np.max(np.abs(pcs[i,:,:,c]))
            _p = pcs[i,:,:,c]
            _p *= np.sign(_p.sum())
            axs[i,c].imshow(_p,aspect='auto',origin='lower', clim=[-mm, mm])
            axs[i,c].set_title(f'pc {i}: {pc_mag[i,c]:.3f}')


