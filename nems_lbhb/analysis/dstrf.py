import logging
import numpy as np
import matplotlib.pyplot as plt

import nems0.db as nd

from nems.tools import dstrf as dtools
from nems.layers import filter
from nems.metrics import correlation

from nems0.utils import shrinkage
from nems0.plots.file import fig2BytesIO
from nems0 import xforms
from nems0.initializers import init_nl_lite

log = logging.getLogger(__name__)

def dstrf_pca(est, modelspec, val=None, modelspec_list=None,
              D=15, timestep=3, pc_count=5, out_channels=None,
              figures=None, fit_ss_model=False, IsReload=False,
              **ctx):

    if IsReload:
        # load dstrf data saved in modelpath
        return {}

    r = est

    if modelspec_list is None:
        modelspec_list = [modelspec]
    cellids = r['resp'].chans

    t_indexes = np.arange(timestep, r['stim'].shape[1], timestep)
    t_indexes = t_indexes[t_indexes>D]
    if out_channels is None:
        out_channels = np.arange(len(cellids))

    stim = {'input': r['stim'].as_continuous().T}
    if 'dlc' in r.signals.keys():
        stim['dlc']=r['dlc'].as_continuous().T
    dstrfs = []
    for mi, m in enumerate(modelspec_list):
        log.info(f"Computing dSTRF {mi+1}/{len(modelspec_list)} at {len(t_indexes)} points (timestep={timestep})")

        d = m.dstrf(stim, D=D, out_channels=out_channels, t_indexes=t_indexes, reset_backend=False)
        dstrfs.append(d['input'])

    dstrf = np.stack(dstrfs, axis=1)
    s = np.std(dstrf, axis=(2, 3, 4), keepdims=True)
    dstrf /= s
    dstrf /= np.max(np.abs(dstrf)) * 0.9

    mdstrf = dstrf.mean(axis=1, keepdims=True)
    sdstrf = dstrf.std(axis=1, keepdims=True)
    sdstrf[sdstrf == 0] = 1
    mzdstrf = shrinkage(mdstrf, sdstrf, sigrat=0.75)

    mdstrf /= np.max(np.abs(mdstrf)) * 0.9
    mzdstrf /= np.max(np.abs(mzdstrf)) * 0.9

    d = dtools.compute_dpcs(mdstrf[:, 0], pc_count=pc_count, as_dict=True)
    dz = dtools.compute_dpcs(mzdstrf[:, 0], pc_count=pc_count, as_dict=True)

    dpc = d['input']['pcs']
    dpc_mag = d['input']['pc_mag']
    dpcz = dz['input']['pcs']
    dpc_magz = dz['input']['pc_mag']
    dproj = dz['input']['projection']
    log.info(f"dproj.shape={dproj.shape}")
    for oi, oc in enumerate(out_channels):
        for di in range(pc_count):
            if dpcz[oi, di].sum()<0:
                dpcz[oi, di] = -dpcz[oi, di]
                dproj[oi,:,di] = -dproj[oi,:,di]
            if dpc[oi, di].sum()<0:
                dpc[oi, di] = -dpc[oi, di]

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
              'interpolation': 'none'}
    imoptsz = {'cmap': 'bwr', 'origin': 'lower',
               'interpolation': 'none'}

    f, ax = plt.subplots(len(out_channels), pc_count+1, figsize=(pc_count, len(out_channels)), sharex='col', sharey='col')
    f.subplots_adjust(top=0.98, bottom=0.02)
    if len(out_channels)==1:
        ax = ax[np.newaxis, ...]
    for oi, oc in enumerate(out_channels):
        ax[oi, -1].plot(dpc_magz[:,oi]/dpc_mag[:,oi].sum())
        for di in range(pc_count):
            d = dpcz[oi, di]
            d = d / np.max(np.abs(d)) / dpc_magz[0, oi] * dpc_magz[di, oi]
            ax[oi, di].imshow(np.fliplr(d), **imopts)
            ax[oi, di+1].set_yticklabels([])
        ax[oi, -2].text(0,10,modelspec.meta['cellids'][oi], fontsize=6)

    if figures is None:
        figures=[]
    figures.append(fig2BytesIO(f))
    modelspec.meta['dpc']=dpcz
    modelspec.meta['dpc_mag']=dpc_magz
    if fit_ss_model:
        subspace_model_fit(est, val, modelspec,
                           pc_count=pc_count, out_channels=out_channels)

    return {'modelspec': modelspec, 'figures': figures}

def subspace_model_fit(est, val, modelspec,
              pc_count=5, out_channels=None,
              figures=None, fit_ss_model=False, IsReload=False, **ctx):

    if IsReload:
        # load dstrf data saved in modelpath
        return

    X_val, Y_val = xforms.lite_input_dict(modelspec, val, epoch_name="")
    X_est, Y_est = xforms.lite_input_dict(modelspec, est, epoch_name="")
    r = est

    cellids = r['resp'].chans
    if out_channels is None:
        out_channels = np.arange(len(cellids))

    ssmodels = []
    sspredxc = np.zeros(len(out_channels))
    ss0predxc = np.zeros(len(out_channels))
    for oi, o in enumerate(out_channels):
        keywordstring = f'wc.{pc_count}x15-relu.15.o.s-wc.15x15-relu.15.o.s-wc.15x1-relu.1.o.s'
        lmodel0 = xforms.init_nems_keywords(keywordstring, meta=modelspec.meta)['modelspec']
        lmodel0 = lmodel0.sample_from_priors()
        fitter_options = {'cost_function': 'nmse', 'early_stopping_delay': 100,
                          'early_stopping_patience': 150,
                          'early_stopping_tolerance': 1e-3,
                          'learning_rate': 1e-3, 'epochs': 10000,
                          }
        fit_opts2 = fitter_options.copy()
        fit_opts2['early_stopping_tolerance'] = 5e-4
        fit_opts2['learning_rate'] = 5e-4

        dpcz = modelspec.meta['dpc']
        dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
        fir = filter.FIR(shape=dpcz.shape)
        fir['coefficients'] = np.flip(dpcz, axis=0)
        X = fir.evaluate(X_est['input'])
        Y = Y_est[:, [o]]

        lmodel0.layers[-1].skip_nonlinearity()
        lmodel = lmodel0.fit(input=X, target=Y, backend='tf', fitter_options=fitter_options)
        lmodel = init_nl_lite(lmodel, X, Y)
        lmodel.layers[-1].unskip_nonlinearity()
        lmodel = lmodel.fit(input=X, target=Y, backend='tf', fitter_options=fit_opts2)

        Xv = fir.evaluate(X_val['input'])
        Yv = Y_val[:, [o]]
        p0 = lmodel0.predict(Xv)
        p = lmodel.predict(Xv)
        sspredxc[oi] = correlation(p, Yv)
        ss0predxc[oi] = correlation(p0, Yv)
        ssmodels.append(lmodel)
    modelspec.meta['sspredxc'] = sspredxc

    if 'r_test' in modelspec.meta.keys():
        log.info("Cellid        Orig  Subspace")
        for oi, o in enumerate(out_channels):
            log.info(f"{modelspec.meta['cellids'][o]}" + \
                     f" {modelspec.meta['r_test'][o, 0]:.3f}" + \
                     f" {modelspec.meta['sspredxc'][oi]:.3f}")

    return {'modelspec': modelspec}
