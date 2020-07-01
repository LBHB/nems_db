import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nems import xforms
from nems.plots.api import ax_remove_box

def dstrf_pca(modelspec, rec, index_range=None, sample_count=100, out_channel=[0], memory=10, norm_mean=True, **kwargs):

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
        if len(out_channel)==1:
            dstrf[i,:,:,0] = modelspec.get_dstrf(rec, index, memory, out_channel=out_channel)
        else:
            dstrf[i,:,:,:] = modelspec.get_dstrf(rec, index, memory, out_channel=out_channel)

    if norm_mean:
        dstrf *= stim_mean[np.newaxis, ..., np.newaxis]

    return dstrf


def dstrf_movie(rec, dstrf, out_channel, index_range, preview=False, mult=False, out_path="/tmp", 
                out_base=None, **kwargs):
    #plt.close('all')

    cellcount=len(out_channel)
    framecount=len(index_range)
    cellids = [rec['resp'].chans[o] for o in out_channel]

    i = 0
    index = index_range[i]
    memory = dstrf.shape[2]

    f, axs = plt.subplots(cellcount+1, 2, figsize=(4, 8))
    f.show()

    stim = np.sqrt(np.sqrt(rec['stim'].as_continuous()))

    stim_lim = np.max(stim[:, index_range])
    s = stim[:, (index - memory):index]
    print("stim_lim ", stim_lim)

    im_list = []
    l1_list = []
    l2_list = []
    im_list.append(axs[0, 1].imshow(s, clim=[0, stim_lim], interpolation='none', origin='lower', aspect='auto'))
    axs[0, 1].set_ylabel('stim')
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xticks([])
    _title1 = axs[0, 1].set_title(f"dSTRF frame {index}")

    for cellidx in range(cellcount):
        d = dstrf[i, :, :, cellidx].copy()

        pred_max = np.max(rec['pred'].as_continuous()[cellidx, :])
        strf_lim = np.max(np.abs(dstrf[:, :, :, cellidx])) / 2
        if mult:
            strf_lim[-1] = strf_lim[-1] * stim_lim / 100
            d = d * s
        im_list.append(axs[cellidx+1,0].imshow(d, clim=[-strf_lim, strf_lim], interpolation='none', origin='lower', aspect='auto'))
        axs[cellidx+1, 0].set_yticks([])
        axs[cellidx+1, 0].set_xticks([])
        axs[cellidx+1, 0].set_ylabel(f"{cellids[cellidx]}", fontsize=6)

        l1_list.append(axs[cellidx+1, 1].plot(rec['pred'].as_continuous()[out_channel[cellidx], (index - memory):index], '-')[0])
        l2_list.append(axs[cellidx+1, 1].plot(rec['resp'].as_continuous()[out_channel[cellidx], (index - memory):index], '--')[0])
        axs[cellidx+1, 1].set_ylim((0, pred_max))
        axs[cellidx+1, 1].set_yticks([])
        ax_remove_box(axs[cellidx+1,0])

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

    if preview:
        for i in range(framecount):
            dstrf_frame(i)
            plt.pause(0.01)
    else:
        if out_base is None:
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

    dstrf = dstrf_pca(modelspec, rec.copy(), out_channel=out_channel,
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


def stp_test():
    modelpath = "/Users/svd/python/nems/results/271/TAR010c-18-1/TAR010c.dlog_wc.18x1.g_stp.1.q.s_fir.1x15_lvl.1_dexp.1.unknown_fitter.2020-06-22T031852"
    cellid = "TAR010c-18-1"
    index_range = np.arange(200, 400)
    memory = 10
    xfspec, ctx = xforms.load_analysis(modelpath)

    make_movie(ctx, cellid=cellid, memory=memory, index_range=index_range, preview=True)


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


