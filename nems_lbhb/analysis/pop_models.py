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


def compute_dstrf(modelspec, rec, index_range=None, sample_count=100, out_channel=None, memory=20,
                  norm_mean=True, method='jacobian', use_rand_seed=0, **kwargs):

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
    
    if out_channel is None:
        out_channel = [0]
    
    if index_range is None:
        stim_mag = rec['stim'].as_continuous().sum(axis=0)
        stim_big = stim_mag > np.max(stim_mag) / 1000
        index_range = np.arange(0, len(stim_mag))
        index_range = index_range[(index_range > memory) & stim_big[index_range]]
        print(f"big frames in index_range: {len(index_range)}")
        if (sample_count is not None) and (len(index_range)>sample_count):
            state = np.random.get_state()
            np.random.seed(use_rand_seed)
            np.random.shuffle(index_range)
            np.random.set_state(state)
            
            index_range = index_range[:sample_count]
            print(f"trimmed to {sample_count} random subset")

    sample_count = len(index_range)
    dstrf = np.zeros((sample_count, stimchans, memory, len(out_channel)))
    for i, index in enumerate(index_range):
        if ((i+1) % 100 == 0):
            log.info(f"dSTRF: {i+1}/{len(index_range)} idx={index}")
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
                        if np.abs(col)>0.3:
                            if col>0:
                                #col = [(1-col), 1-0.5*col, 1-0.5*col]
                                col = [1-0.1*col, 1-col, 1-col]
                            else:
                                #col = [1+0.5*col, 1+0.5*col, 1+col]
                                col = [1+col, 1+col, 1+0.1*col]

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
    try:
        modelname = modelspec.meta['modelname'].split("_")[1] 
    except:
        modelname = modelspec.meta['modelname']
    f.suptitle(f"{modelspec.meta['cellid']} {modelname} {index_range[0]}-{index_range[-1]}")
    return f

def force_signal_silence(rec, signal='stim'):
    rec = rec.copy()
    for e in ["PreStimSilence", "PostStimSilence"]:
        s = rec[signal].extract_epoch(e)
        s[:,:,:]=0
        rec[signal]=rec[signal].replace_epoch(e, s)
    return rec


def model_pred_sum(ctx, cellid, rr=None, respcolor='lightgray', predcolor='purple', labels='model'):

    if type(ctx) is list:
        mult_ctx=True
        rec=ctx[0]['val']
        modelspec=ctx[0]['modelspec']
    else:
        mult_ctx=False
        rec = ctx['val']
        modelspec=ctx['modelspec']

    cellids = rec['resp'].chans
    match=[c==cellid for c in cellids]
    c = np.where(match)[0][0]
    
    if rr is None:
        rr=np.arange(np.min((1000,rec['resp'].shape[1])))
        
    f,ax = plt.subplots(2,1, figsize=(12,4), sharex=True)
    rr_orig=rr
    tt=np.arange(len(rr))/rec['resp'].fs
    
    ax[0].imshow(rec['stim'].as_continuous()[:, rr_orig], aspect='auto', origin='lower', cmap="gray_r", 
                 extent=[tt[0],tt[-1],0,rec['stim'].shape[0]])
    ax[0].set_title(cellid + "/" + modelspec.meta['modelname'].split("_")[1])
    
    ax[1].plot(tt, rec['resp'].as_continuous()[c, rr_orig], color=respcolor, label='resp');
    if mult_ctx:
        for e in range(len(ctx)):
            print(f"{labels[e]} {predcolor[e]}")
            r=ctx[e]['modelspec'].meta['r_test'][c][0]
            ax[1].plot(tt, ctx[e]['val']['pred'].as_continuous()[c, rr_orig], color=predcolor[e], label=f"{labels[e]} r={r:.2f}");
    else: 
        ax[1].plot(tt, rec['pred'].as_continuous()[c, rr_orig], color=predcolor, label=f"r={modelspec.meta['r_test'][c]}");

    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel('Time (s)')
    
    return f
    
from nems.utils import get_setting
from scipy.ndimage import zoom


def dstrf_pca_plot(pcs, pc_mag, cellids, clist=None, rows=1):
    channel_count=pcs.shape[-1]
    ccmax=rows*10
    if clist is None:
        clist=np.arange(np.min([ccmax,channel_count]))
        
    if len(clist)>ccmax:
        clist=clist[:ccmax]
    if len(clist)<10:
        cols=len(clist)
    else:
        cols=10
    f2,axs=plt.subplots(cols, 3*rows, figsize=(3*rows, cols*0.85))
    for c,cadj in enumerate(clist):
        cellid = cellids[cadj]
        pc_rat = pc_mag[:,cadj]/pc_mag[:,cadj].sum()
        for i in range(3):
            mm=np.max(np.abs(pcs[i,:,:,cadj]))
            _p = pcs[i,:,:,cadj]
            #_p *= np.sign(_p.sum())
            os = int(c/10)*3
            _c = c % 10
            
            rat=(pc_rat[i] / pc_mag[0,cadj]) #  ** 2
            #print(rat, mm)

            strf = np.fliplr(_p)
            strf[np.abs(strf)<np.std(strf)]=0
            strf = zoom(strf, [2,2])*rat
            #if np.abs(strf.min())>strf.max():
            #    strf=-strf

            ax = axs[_c, i+os]
            ax.imshow(strf,cmap='bwr', aspect='auto',origin='lower', clim=[-mm, mm])
            if i==0:
                ax.set_ylabel(f'{cellid}', fontsize=6)
            if _c==0:
                ax.set_title(f'PC {i+1}', fontsize=6)
            ax.text(ax.get_xlim()[1],ax.get_ylim()[0],
                    f'{pc_rat[i]:.2f}', ha='right',va='bottom', fontsize=8)
            if i+os<7:
                ax.set_xticks([])
            if (c>0) or (i+os<7):
                ax.set_yticks([])
            #ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.show()
    return f2



def dstrf_details(modelspec, rec,cellid,rr,dindex, dstrf=None, pcs=None, memory=20, stepbins=3, maxbins=1500, n_pc=3):
    cellids = rec['resp'].chans
    match = [c==cellid for c in cellids]
    c = np.where(match)[0][0]
        
    # analyze all output channels
    out_channel = [c]
    channel_count = len(out_channel)

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

    if pcs is None:
        # don't skip silent bins

        stim_big = stim_mag > np.max(stim_mag) / 1000
        driven_index_range = index_range[(index_range > memory) & stim_big[index_range]]
        driven_index_range = driven_index_range[(driven_index_range >= memory)]

        # limit number of bins to speed up analysis
        driven_index_range = driven_index_range[:maxbins]
        #index_range = np.arange(525,725)
    
        pcs, pc_mag = compute_dpcs(dstrf[driven_index_range,:,:,:], pc_count=n_pc)
    
    c_=0

    ii = np.arange(rr[0],rr[-1])
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
        if False:
            # stack a copy scaled by the current stim
            #_dstrf = np.concatenate((_dstrf,stim[:,(d-_dstrf.shape[1]):d]*mmd), axis=0)
            _dstrf = np.concatenate((_dstrf,_dstrf*stim[:,(d-_dstrf.shape[1]):d]), axis=0)
            #_dstrf *= stim[:,(d-_dstrf.shape[1]):d]
        ds = np.fliplr(_dstrf)
        #ds=zoom(ds, [2,2])
        ax[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmd, mmd], cmap=get_setting('WEIGHTS_CMAP'), interpolation='none')
        #plot_heatmap(ds, aspect='auto', ax=ax[i], interpolation=2, clim=[-mmd, mmd], show_cbar=False, xlabel=None, ylabel=None)

        ax[i].set_title(f"Frame {d}", fontsize=8)
        ds = np.fliplr(pcs[0,:,:,c_])*pc_mag[0]
        #ds = zoom(ds, [2,2])
        mmp = np.max(np.abs(ds))
        if i < n_pc:
            ds = np.fliplr(pcs[i,:,:,c_]*pc_mag[i])
            #ds = zoom(ds, [2,2])
            #ax3[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmp, mmp])
            ax3[i].imshow(ds, aspect='auto', origin='lower', clim=[-mmp, mmp], cmap=get_setting('WEIGHTS_CMAP'))
            ax3[i].text(pcs.shape[2]-9, 1, f'PC{i}: {pc_mag[i,0]:0.3f}')
        else:
            ax3[i].set_axis_off()

    ax[0].set_ylabel('Example frames')
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
        #_s = np.sqrt(_s)
        _s /= np.sum(_s[:pc_count])
        
        pcs[:, :, :, c] = np.reshape(_v[:pc_count, :],[pc_count, s[1], s[2]])
        
        mdstrf=dstrf[:, :, :, c].mean(axis=0)
        if np.sum(mdstrf * pcs[0,:,:,c])<0:
            pcs[0,:,:,c] = -pcs[0,:,:,c]
            #print(f"{c} adjusted to {np.sum(mdstrf * pcs[0,:,:,c])}")
        pc_mag[:, c] = _s[:pc_count]

    return pcs, pc_mag


def dstrf_pca(modelspec, rec, chunksize=20, out_channel=None, pc_count=3, 
              return_dstrf=False, **dstrf_parms):

    if out_channel is None:
        cellcount = len(modelspec.meta['cellids'])
        out_channel = list(np.arange(cellcount))
    else:
        cellcount = len(out_channel)
    chunkcount = int(np.ceil(cellcount/chunksize))

    for chunk in range(chunkcount):
        channels = list(range(chunk*chunksize,np.min([cellcount,(chunk+1)*chunksize])))
        c1 = modelspec.meta['cellids'][channels[0]]
        c2 = modelspec.meta['cellids'][channels[-1]]

        print(f"Computing dSTRF(s) for chunk {chunk}: {channels[0]}-{channels[-1]} / cellid {c1} to {c2}")
        dstrf_ = compute_dstrf(modelspec, rec, out_channel=channels, **dstrf_parms)
        pcs_, pc_mag_ = compute_dpcs(dstrf_, pc_count=pc_count)
        
        if chunk==0:
            dstrf = dstrf_.copy()
            pcs = pcs_.copy()
            pc_mag = pc_mag_.copy()
        else:
            if return_dstrf:
                dstrf = np.concatenate((dstrf, dstrf_), axis=3)
            pcs = np.concatenate((pcs, pcs_), axis=3)
            pc_mag = np.concatenate((pc_mag, pc_mag_), axis=1)
            
    if return_dstrf:
       return pcs, pc_mag, dstrf
    else:
       return pcs, pc_mag


def dstrf_analysis(modelspec=None, est=None, figures=None, pc_count=5, memory=25,
                   index_range=None, sample_count=1000,
                   chunksize=25,
                   IsReload=False, **kwargs):
    """
    Specialized postprocessor to analyze dstrf
    """
    if IsReload:
        log.info("Reload, skipping dstrf_analysis")
        return {}

    pcs, pc_mag, dstrf = dstrf_pca(modelspec, est, pc_count=5, out_channel=None, memory=memory,
                            index_range=None, sample_count=sample_count,
                            return_dstrf=True, chunksize=chunksize)
    modelspec.meta['dstrf_pcs'] = pcs
    modelspec.meta['dstrf_pc_mag'] = pc_mag

    rr = np.arange(150,600)
    dindex = [50,100,200,250,350,400]
    f = dstrf_details(modelspec, est, modelspec.meta['cellids'][0], rr, dindex, dstrf=None, pcs=None, memory=25, stepbins=3, maxbins=1500, n_pc=5)
    f.show()
    if figures is None:
        figures = [nplt.fig2BytesIO(f)]
    else:
        figures = figures.copy()
        figures.append(nplt.fig2BytesIO(f))

    return {'modelspec': modelspec, 'figures': figures}


def pop_space_summary(recname='est', modelspec=None, rec=None, figures=None, n_pc=3, memory=20, 
                      maxbins=1000, stepbins=3, IsReload=False, batching=True, **ctx):

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


def dstrf_movie(rec, dstrf, out_channel, index_range, static=False, preview=False, mult=False, threshold=0, out_path="/tmp", 
                out_base=None, fps=10, **kwargs):
    #plt.close('all')

    cellcount=len(out_channel)
    framecount=len(index_range)
    cellids = [rec['resp'].chans[o] for o in out_channel]
    fs=rec['resp'].fs
    i = 0
    index = index_range[i]
    memory = dstrf.shape[2]

    stim=rec['stim'].as_continuous() + 0.0  # -0.05
    stim[stim<0]=0
    
    # compress
    stim=stim ** 0.5
    stim_lim = np.max(stim[:, index_range])
    # alt: no compression, threshold max
    #stim_lim = np.max(stim[:, index_range]) *0.5

    s = stim[:, (index - memory+1):(index+1)]
    #print("stim_lim ", stim_lim)

    im_list = []
    l1_list = []
    l2_list = []

    if static:
        max_frames = np.min([10, framecount])
        f, axs = plt.subplots(cellcount+1, max_frames, figsize=(max_frames*1, (cellcount+1)*0.75))
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
    _title1 = axs[0, stim0col].set_title(f"dSTRF t={index / fs:.2f}")

    #axs[0, 0].set_axis_off()
    for cellidx in range(cellcount):
        d = dstrf[i, :, :, cellidx].copy()

        pred_max = np.max(rec['pred'].as_continuous()[cellidx, :])
        strf_lim = np.max(np.abs(dstrf[:, :, :, cellidx])) * 1.0

        if mult:
            strf_lim = strf_lim * stim_lim
            d = d * s
        d[np.abs(d)<threshold*strf_lim]=0
        d = zoom(d, [2,2])
        #print(f"strf_lim[{cellidx}]: {strf_lim} shape: {dstrf[:, :, :, cellidx].shape} reshape max: {np.abs(d).max()}")
        im_list.append(axs[cellidx+1,0].imshow(d, clim=[-strf_lim, strf_lim],
                                               interpolation='none', origin='lower',
                                               aspect='auto', cmap='bwr'))
        axs[cellidx+1, 0].set_yticks([])
        axs[cellidx+1, 0].set_xticks([])
        axs[cellidx+1, 0].set_ylabel(f"{cellids[cellidx]}", fontsize=6)

        if cellidx<cellcount-1:
            axs[cellidx+1, 1].set_xticklabels([])

        if not static:
           l1_list.append(axs[cellidx+1, 1].plot(rec['pred'].as_continuous()[out_channel[cellidx], (index - memory*2):index],
                                                 '-', color='purple')[0])
           l2_list.append(axs[cellidx+1, 1].plot(rec['resp'].as_continuous()[out_channel[cellidx], (index - memory*2):index],
                                                 '-', color='gray')[0])
           axs[cellidx+1, 1].set_ylim((0, pred_max))
           axs[cellidx+1, 1].set_yticks([])

    def dstrf_frame(i):
        """
        plot a frame for saving the animation
        :param i:
        :return:
        """
        index = index_range[i]
        s = stim[:, (index - memory+1):(index+1)]
        im_list[0].set_data(s)

        for cellidx in range(cellcount):
            if mult:
                d = dstrf[i, :, :, cellidx] * s
            else:
                d = dstrf[i, :, :, cellidx]
            d[np.abs(d)<threshold*strf_lim]=0
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

        _title1.set_text(f"dSTRF t={index/fs:.2f}")

        return tuple(im_list + l1_list + l2_list + [_title1])

    if static:
        #stim = np.sqrt(np.sqrt(rec['stim'].as_continuous()))
        #stim_lim = np.max(stim[:, index_range])
        #print("stim_lim ", stim_lim)

        for i in range(1, max_frames):
           index = index_range[i]

           s = stim[:, (index - memory+2):(index+2)]
           axs[0, i].imshow(s, clim=[0, stim_lim], interpolation='none', origin='lower', aspect='auto', cmap='Greys')
           axs[0, i].set_title(f"dSTRF t={index/fs:.2f}")
           axs[0, i].set_yticks([])
           axs[0, i].set_xticks([])

           for cellidx in range(cellcount):
               d = dstrf[i, :, :, cellidx].copy()

               pred_max = np.max(rec['pred'].as_continuous()[cellidx, :])
               strf_lim = np.max(np.abs(dstrf[:, :, :, cellidx]))  # /2
               if mult:
                   strf_lim = strf_lim * stim_lim
                   d = d * s
               d[np.abs(d)<threshold*strf_lim]=0
               d = zoom(d, [2, 2])
               axs[cellidx+1, i].imshow(d, clim=[-strf_lim, strf_lim], interpolation='none', origin='lower', 
                                        aspect='auto', cmap='bwr')
               axs[cellidx+1, i].set_yticks([])
               axs[cellidx+1, i].set_xticks([])
               #axs[cellidx+1, i].set_ylabel(f"{cellids[cellidx]}", fontsize=6)
               #ax_remove_box(axs[cellidx+1, i])

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

def simulate_pop(n=10, modelstring=None):

    from nems_lbhb.baphy_experiment import BAPHYExperiment
    from nems.initializers import from_keywords, rand_phi
    from nems_lbhb.baphy_io import fill_default_options

    parmfile = '/auto/data/daq/Amanita/AMT004/AMT004b13_p_NAT.m'

    options = {'stimfmt': 'ozgf', 'rasterfs': 100,
               'chancount': 18, 'resp': False, 'pupil': False, 'stim': True
              }
    options = fill_default_options(options)

    e=BAPHYExperiment(parmfile=parmfile)
    rec=e.get_recording(**options)
    rec['stim']=rec['stim'].rasterize().normalize()

    if modelstring is None:
        #modelstring = "wc.18x3.g-fir.3x10-lvl.1-dexp.1"
        #modelstring = "wc.18x3.g-fir.3x10-lvl.1"
        #modelstring = "wc.18x3.g-stp.3.q.s-do.3x20-lvl.1"
        modelstring = "wc.18x3.g-stp.3.q.s-do.3x20-lvl.1-dexp.1"
        
    modelspec = from_keywords(modelstring)
    
    modelspec = rand_phi(modelspec, rand_count=n)['modelspec']

    preds = []
    for fitidx in range(modelspec.fit_count):
        modelspec.set_fit(fitidx)
        if 'gaussian' in modelspec[0]['fn']:
            modelspec.phi[0]['mean'] = np.random.rand(modelspec.phi[0]['mean'].shape[0])
            modelspec.phi[0]['sd'] /= 3
            print(fitidx, "wc.g", modelspec.phi[0])

        if 'stp' in modelspec[1]['fn']:
            tau = modelspec.phi[1]['tau']
            tau[tau<0.001]=0.001
            modelspec.phi[1]['tau'] = tau * 1
            modelspec.phi[1]['u'] *= 1
            print(fitidx, "stp", modelspec.phi[1])
            
        if 'relu' in modelspec[2]['fn']:
            modelspec.phi[2]['offset'] -= 0.05
            print(fitidx, "relu offset", modelspec.phi[2]['offset'].T)

        rec_ = modelspec.evaluate(rec=rec)
        rec_['pred'].chans = [f'SIM000a-{fitidx:02d}-1']
        d = rec_['pred']._data.copy()
        d -= d[:10].mean()
        d += np.random.randn(d.shape[0],d.shape[1])*d.std()/2 + d.std()/2
        d[d<0]=0
        preds.append(rec_['pred']._modified_copy(data=d))
        
    resp = preds[0].concatenate_channels(preds)
    resp.name='resp'

    if True:
        stim = rec['stim']
        m=stim._data.mean(axis=1,keepdims=True)
        s=stim._data.std(axis=1,keepdims=True)
        print(f'Stim mean/std={m.mean()}/{s.mean()}')
        noise = np.random.randn(stim.shape[0],stim.shape[1])*m + s
        nstim = stim._modified_copy(data=stim._data + noise)
        nstim.epochs = stim.epochs.copy()
        for i,e in nstim.epochs.iterrows():
            if e['name'].endswith('.wav'):
                k = e['name']
                nstim.epochs.loc[i,'name']=k + "_noise"
                #nstim._data[k + "_noise"] = nstim._data[k]

        stim = stim.concatenate_time((stim,nstim))
        resp = resp.concatenate_time((resp,resp))
        resp.epochs = stim.epochs.copy()

        rec.signals['stim']=stim
    
    rec.add_signal(resp)
    rec.name = 'SIM000a'

    plt.figure(figsize=(12, 3))
    plt.plot(resp.as_continuous()[:, :1500].T)
    for i in range(resp.shape[0]):
        plt.text(0,resp._data[i,0],f"{i}", ha='right')
        
    return {'rec': rec, 'modelspec': modelspec}
