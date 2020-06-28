import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nems import xforms


def dstrf_pca(modelspec, rec, index_range=None, sample_count=100, out_channel=0, memory=10):

    modelspec.rec = rec
    stimchans = rec['stim'].shape[0]
    bincount = rec['pred'].shape[1]
    if index_range is None:
        index_range = np.arange(bincount)
        if sample_count is not None:
            np.random.shuffle(index_range)
            index_range = index_range[:sample_count]
    sample_count = len(index_range)
    dstrf = np.zeros((sample_count, stimchans, memory))
    for i, index in enumerate(index_range):
        dstrf[i,:,:] = modelspec.get_dstrf(rec, index, memory, out_channel=out_channel)

    return dstrf


def dstrf_movie(rec, dstrf, out_channel, index_range, preview=False, mult=False):
    #plt.close('all')
    f, axs = plt.subplots(3, 1, figsize=(4, 6))
    f.show()
    pred_max = np.max(rec['pred'].as_continuous()[out_channel, :])
    strf_lim = np.max(np.abs(dstrf)) / 2
    stim_lim = np.max(rec['stim'].as_continuous()[:, index_range])
    cellid = rec['resp'].chans[out_channel]
    i=0
    index = index_range[i]
    memory = dstrf.shape[2]

    s = rec['stim'].as_continuous()[:, (index - memory):index]
    d = dstrf[i, :, :].copy()

    if mult:
        strf_lim = strf_lim * stim_lim /100
        d = d*s
    print(d)
    print(strf_lim)
    print(stim_lim)

    _im0 = axs[0].imshow(s, clim=[0, stim_lim], origin='lower', aspect='auto')
    axs[0].set_ylabel('stim')

    _im1 = axs[1].imshow(d, clim=[-strf_lim, strf_lim], origin='lower', aspect='auto')
    axs[1].set_ylabel('dstrf')

    _line1 = axs[2].plot(rec['pred'].as_continuous()[out_channel, (index - memory):index], '-')[0]
    _line2 = axs[2].plot(rec['resp'].as_continuous()[out_channel, (index - memory):index], '--')[0]
    axs[2].set_ylim((0, pred_max))
    axs[2].set_ylabel('pred/resp')
    _title1 = axs[0].set_title(f"{cellid} frame {index}")

    def dstrf_frame(i):
        index = index_range[i]
        s = rec['stim'].as_continuous()[:, (index - memory):index]
        if mult:
            d = dstrf[i, :, :] * s
        else:
            d = dstrf[i, :, :]

        _im0.set_data(s)
        _im1.set_data(d)

        _t = np.arange(memory)
        _line1.set_data((_t, rec['pred'].as_continuous()[out_channel, (index - memory):index]))
        _line2.set_data((_t, rec['resp'].as_continuous()[out_channel, (index - memory):index]))

        _title1.set_text(f"{cellid} frame {index}")

        return _im0, _im1, _line1, _line2, _title1

    framecount=200
    if preview:
        for i in range(framecount):
            dstrf_frame(i)
            plt.pause(0.01)
    else:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        line_ani = animation.FuncAnimation(f, dstrf_frame, framecount, fargs=(),
                                           interval=1, blit=True)
        line_ani.save(f'/tmp/{cellid}_{index_range[0]}-{index_range[-1]}.mp4', writer=writer)


def make_movie(modelpath, cellid=None, out_channel=0, memory=10, index_range=None, **kwargs):
    xfspec, ctx = xforms.load_analysis(modelpath)
    rec = ctx['val']
    modelspec = ctx['modelspec']

    if cellid is not None:
        out_channel = np.where([cellid == c for c in rec['resp'].chans])[0][0]  # 26  #16 # 30
    else:
        cellid = rec['resp'].chans[out_channel]
    print(f'cell {cellid} is chan {out_channel}. computing dstrf with memory={memory}')

    dstrf = dstrf_pca(modelspec, rec.copy(), out_channel=out_channel,
                      index_range=index_range, memory=memory)

    dstrf_movie(rec, dstrf, out_channel, index_range, **kwargs)


def pop_test():
    modelpath = "/Users/svd/python/nems/results/271/TAR010c/TAR010c.dlog_wc.18x3.g_fir.1x10x3_relu.3_wc.3x55_lvl.55.unknown_fitter.2020-06-20T153243"
    cellid = "TAR010c-36-1"
    index_range = np.arange(200, 400)
    memory = 10

    make_movie(modelpath, cellid=cellid, memory=memory, index_range=index_range, preview=True)

def stp_test():
    modelpath = "/Users/svd/python/nems/results/271/TAR010c-18-1/TAR010c.dlog_wc.18x1.g_stp.1.q.s_fir.1x15_lvl.1_dexp.1.unknown_fitter.2020-06-22T031852"
    cellid = "TAR010c-18-1"
    index_range = np.arange(200, 400)
    memory = 10

    make_movie(modelpath, cellid=cellid, memory=memory, index_range=index_range, preview=True)

def db_test():
    #get_results_file(batch, modelnames=None, cellids=None)
    from nems.db import get_results_file
    cellid= "TAR010c-18-2"
    batch = 289
    #modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2.q-fir.2x15-lvl.1-dexp.1_tfinit.n.lr1e3-newtf.n.lr1e4'
    modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-fir.4x15-lvl.1-dexp.1_tfinit.n.lr1e3-newtf.n.lr1e4'

    # model from Jake
    modelname = 'ozgf.fs50.ch18.pop-ld-norm-pca.no-popev_dlog-conv2d.8.3x3.rep5-wcn.8-wc.8x12.z-relu.12-wc.12xR.z-lvl.R-dexp.R_tfinit.n.lr1e3.rb5-newtf.n.lr1e4'
    batch = 289
    cellid = 'AMT003c-19-1'
    d = get_results_file(batch, [modelname], [cellid])
    modelpath = d.modelpath[0]

    index_range = np.arange(200, 400)
    memory = 10

    make_movie(modelpath, cellid=cellid, memory=memory, index_range=index_range, preview=False, mult=False)


