import os
import logging
import numpy as np
import json as jsonlib

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import det

from nems import xforms
from nems.plots.api import ax_remove_box, spectrogram, fig2BytesIO
from nems.uri import NumpyEncoder

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
    if ('double_exponential' in modelspec[-1]['fn']):
        log.info('removing dexp from tail')
        modelspec.pop_module()
    if ('relu' in modelspec[-1]['fn']):
        log.info('removing relu from tail')
        modelspec.pop_module()
    if ('levelshift' in modelspec[-1]['fn']):
        log.info('removing lvl from tail')
        modelspec.pop_module()

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


def dstrf_pca(modelspec, rec, pc_count=3, out_channel=[0], memory=10,
              **kwargs):

    dstrf = compute_dstrf(modelspec, rec.copy(), out_channel=out_channel,
                          memory=memory, **kwargs)

    channel_count=dstrf.shape[3]
    s = list(dstrf.shape)
    s[0]=pc_count
    pcs = np.zeros(s)
    pc_mag = np.zeros((pc_count,channel_count))
    for c in range(channel_count):
        d = np.reshape(dstrf[:, :, :, c], (dstrf.shape[0], s[1]*s[2]))
        _u, _s, _v = np.linalg.svd(d)
        _s *= _s
        _s /= np.sum(_s)
        pcs[:, :, :, c] = np.reshape(_v[:pc_count, :],[pc_count, s[1], s[2]])
        pc_mag[:, c] = _s[:pc_count]

    return pcs, pc_mag

def pop_space_summary(val, modelspec, rec=None, figures=None, n_pc=2, memory=12, maxbins=1000, stepbins=3, IsReload=False, **ctx):

    if IsReload:
        return {}

    if figures is None:
        figs = []
    else:
        figs = figures.copy()

    rec = val.apply_mask()
    
    cellids = val['resp'].chans
    siteids = [c.split("-")[0] for c in cellids]
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

            if (siteids[i] == siteids[0]) & (siteids[j] == siteids[0]):
                olap_same_site.append(overlap[i, j])
                p_cc_same_site.append(overlap[j, i])
                r_cc_same_site.append(r_cc)
            elif (siteids[j] == siteids[0]):
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

    f, ax = plt.subplots(2, 1, figsize=(6,9))
    modelname_list = modelspec.meta["modelname"].split("_")
    spectrogram(rec, sig_name='pred', title=f'{siteids[0]}/{modelname_list} pred PSTH', ax=ax[0]) 
    ax[1].imshow(overlap, clim=[-1, 1])

    for i, s in enumerate(siteids[:olapcount]):
        if i > 0 and (siteids[i] != siteids[i - 1]) and (siteids[i - 1] == siteids[0]):
            ax[1].plot([0, olapcount - 1], [i - 0.5, i - 0.5], 'b', linewidth=0.5)
            ax[1].plot([i - 0.5, i - 0.5], [0, olapcount - 1], 'b', linewidth=0.5)
    ax[1].text(olapcount+1, 0, f'PC space same {np.mean(olap_same_site):.3f}\n' +\
                               f'partial: {np.mean(olap_part_site):.3f}\n' +\
                               f'diff: {np.mean(olap_diff_site):.3f}\n' +\
                               f'Resp CC same {np.mean(r_cc_same_site):.3f}\n' +\
                               f'partial: {np.mean(r_cc_part_site):.3f}\n' +\
                               f'diff: {np.mean(r_cc_diff_site):.3f}\n' +\
                               f'Pred CC same {np.mean(p_cc_same_site):.3f}\n' +\
                               f'partial: {np.mean(p_cc_part_site):.3f}\n' +\
                               f'diff: {np.mean(p_cc_diff_site):.3f}',
               va='top', fontsize=8)
    ax[1].set_ylabel('diff site -- same site')
    ax_remove_box(ax[1])
    figs.append(fig2BytesIO(f))

    f2,axs=plt.subplots(8, 10, figsize=(16,12))
    for c in range(channel_count):
        cellid = cellids[c]
        for i in range(2):
            mm=np.max(np.abs(pcs[i,:,:,c]))
            _p = pcs[i,:,:,c]
            _p *= np.sign(_p.sum())
            os = int(c/10)*2
            _c = c % 10
            axs[i+os,_c].imshow(_p,aspect='auto',origin='lower', clim=[-mm, mm])
            if i==0:
               axs[i+os,_c].set_title(f'{cellid} {pc_mag[i,c]:.3f}', fontsize=8)
            else:
               axs[i+os,_c].set_title(f'{pc_mag[i,c]:.3f}', fontsize=8)
            if i<n_pc-1:
                axs[i+os,_c].set_xticks([])
            if c>0:
                axs[i+os,_c].set_yticks([])
            ax_remove_box(axs[i+os, _c])

    figs.append(fig2BytesIO(f2))

    extra_results = {'dstrf_overlap': overlap,
            'olap_same_site': olap_same_site,
            'olap_part_site': olap_part_site,
            'r_cc_same_site': r_cc_same_site,
            'r_cc_part_site': r_cc_part_site,
            'p_cc_same_site': p_cc_same_site,
            'p_cc_part_site': p_cc_part_site}
    modelspec.meta['extra_results']=jsonlib.dumps(extra_results, cls=NumpyEncoder) 

    return {'figures': figs, 'modelspec': modelspec}


def dstrf_movie(rec, dstrf, out_channel, index_range, static=False, preview=False, mult=False, out_path="/tmp", 
                out_base=None, **kwargs):
    #plt.close('all')

    cellcount=len(out_channel)
    framecount=len(index_range)
    cellids = [rec['resp'].chans[o] for o in out_channel]

    i = 0
    index = index_range[i]
    memory = dstrf.shape[2]

    stim = np.sqrt(np.sqrt(rec['stim'].as_continuous()))
    stim_lim = np.max(stim[:, index_range])

    s = stim[:, (index - memory):index]
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
    _title1 = axs[0, stim0col].set_title(f"dSTRF frame {index}")

    for cellidx in range(cellcount):
        d = dstrf[i, :, :, cellidx].copy()

        pred_max = np.max(rec['pred'].as_continuous()[cellidx, :])
        strf_lim = np.max(np.abs(dstrf[:, :, -1, cellidx])) / 2

        if mult:
            strf_lim = strf_lim * stim_lim
            d = d * s
        print(f"strf_lim[{cellidx}]: ", strf_lim)

        im_list.append(axs[cellidx+1,0].imshow(d, clim=[-strf_lim, strf_lim], interpolation='none', origin='lower', 
                                               aspect='auto'))
        axs[cellidx+1, 0].set_yticks([])
        axs[cellidx+1, 0].set_xticks([])
        axs[cellidx+1, 0].set_ylabel(f"{cellids[cellidx]}", fontsize=6)
        ax_remove_box(axs[cellidx+1,0])

        if not static:
           l1_list.append(axs[cellidx+1, 1].plot(rec['pred'].as_continuous()[out_channel[cellidx], (index - memory):index], '-')[0])
           l2_list.append(axs[cellidx+1, 1].plot(rec['resp'].as_continuous()[out_channel[cellidx], (index - memory):index], '--')[0])
           axs[cellidx+1, 1].set_ylim((0, pred_max))
           axs[cellidx+1, 1].set_yticks([])

    def dstrf_frame(i):
        index = index_range[i]
        s = stim[:, (index - memory):index]
        im_list[0].set_data(s)

        for cellidx in range(cellcount):
            if mult:
                d = dstrf[i, :, :, cellidx] * s
            else:
                d = dstrf[i, :, :, cellidx]

            im_list[cellidx+1].set_data(d)

            _t = np.arange(memory)
            l1_list[cellidx].set_data((_t, rec['pred'].as_continuous()[cellidx, (index - memory):index]))
            l2_list[cellidx].set_data((_t, rec['resp'].as_continuous()[cellidx, (index - memory):index]))

        _title1.set_text(f"dSTRF frame {index}")

        return tuple(im_list + l1_list + l2_list + [_title1])

    if static:
        stim = np.sqrt(np.sqrt(rec['stim'].as_continuous()))
        stim_lim = np.max(stim[:, index_range])
        print("stim_lim ", stim_lim)

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
               axs[cellidx+1, i].imshow(d, clim=[-strf_lim, strf_lim], interpolation='none', origin='lower', aspect='auto')
               axs[cellidx+1, i].set_yticks([])
               axs[cellidx+1, i].set_xticks([])
               #axs[cellidx+1, i].set_ylabel(f"{cellids[cellidx]}", fontsize=6)
               ax_remove_box(axs[cellidx+1, i])

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
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
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


