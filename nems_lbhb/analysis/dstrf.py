"""
nems_lbhb.analysis.dstrf - tools for generating/analyzing dstrfs

dstrf_pca -
subspace_model_fit -

"""

import logging
import numpy as np
import matplotlib.pyplot as plt

import nems0.db as nd

from nems.tools import dstrf as dtools
from nems.layers import filter
from nems.metrics import correlation

from nems0.utils import shrinkage, smooth
from nems0.plots.file import fig2BytesIO
from nems0 import xforms
from nems0.initializers import init_nl_lite
from nems_lbhb.plots import histscatter2d, histmean2d, scatter_comp
from nems.models import LN

log = logging.getLogger(__name__)

def compute_extract_dpc(modelspec_list, stim, D, out_channels, t_indexes, reset_backend):
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

    for oi, oc in enumerate(out_channels):
        for di in range(pc_count):
            if dz['input']['pcs'][oi, di].sum()<0:
                dz['input']['pcs'][oi, di] = -dz['input']['pcs'][oi, di]
                dz['input']['projection'][oi,:,di] = -dz['input']['projection'][oi,:,di]
            if d['input']['pcs'][oi, di].sum()<0:
                d['input']['pcs'][oi, di] = -d['input']['pcs'][oi, di]

    dpc = d['input']['pcs']
    dpc_mag = d['input']['pc_mag']
    dpcz = dz['input']['pcs']
    dpc_magz = dz['input']['pc_mag']
    dproj = dz['input']['projection']
    log.info(f"dproj.shape={dproj.shape}")
    
    
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def dstrf_pca(est=None, modelspec=None, val=None, sig='input', modelspec_list=None,
              D=15, timestep=3, pc_count=10, max_frames=4000,
              out_channels=None,
              figures=None, fit_ss_model=False, ss_pccount=5, ss_dpc_var=0.95,
              first_lin=True,
              IsReload=False, **ctx):
    """xforms function
    use modelspec or modelspec_list to compute dSTRFs from est recording (using nems.tools.dstrf)
    then perform PCA on the collection of dSTRFs, save the top pc_count
    if fit_ss_model, fit a DNN using the projection into the subspace for each neurons
    save results in modelspec.meta
    """

    if IsReload:
        # load dstrf data saved in modelpath.. or don't if not needed?
        return {}
    if (est is None) or (modelspec is None):
        raise ValueError("est and modelspec parameters required")
    r = est

    if modelspec_list is None:
        modelspec_list = [modelspec]
    cellids = r['resp'].chans

    t_indexes = np.arange(timestep, r['stim'].shape[1], timestep)
    t_indexes = t_indexes[t_indexes>D]
    if out_channels is None:
        out_channels = np.arange(len(cellids))
    cellcount=len(out_channels)
    stim = {'input': r['stim'].as_continuous().T}
    if 'dlc' in r.signals.keys():
        stim['dlc']=r['dlc'].as_continuous().T
    dstrfs = []
    if len(t_indexes)>max_frames:
        log.info(f'Reducing t_indexes length to {max_frames}')
        t_indexes=t_indexes[:max_frames]

    tcount=len(t_indexes)
    U = r['stim'].shape[0]
    for mi, m in enumerate(modelspec_list):
        log.info(f"Computing dSTRF {mi+1}/{len(modelspec_list)} at {len(t_indexes)} points (timestep={timestep})")

        d = m.dstrf(stim, D=D, out_channels=out_channels, t_indexes=t_indexes, reset_backend=False)
        dstrfs.append(d[sig].astype(np.float32))  # to save memory
        
        # delete backend to prevent (potential?) copy error
        m.dstrf_backend=None
    
    dstrf = np.stack(dstrfs, axis=1)
    del dstrfs
    s = np.std(dstrf, axis=(2, 3, 4), keepdims=True)
    dstrf /= s
    dstrf /= np.max(np.abs(dstrf)) * 0.9

    log.info("Averaging across jackknifes")
    mdstrf = dstrf.mean(axis=1, keepdims=True)
    mdstrf /= np.max(np.abs(mdstrf)) * 0.9
    d = dtools.compute_dpcs(mdstrf[:, 0], pc_count=pc_count, first_lin=first_lin, as_dict=True)

    if len(modelspec_list)>1:
        log.info("Shrinking across jackknifes")
        sdstrf = dstrf.std(axis=1, keepdims=True)
        sdstrf[sdstrf == 0] = 1

        log.info("Calling shrinkage()")
        mzdstrf = shrinkage(mdstrf, sdstrf, sigrat=0.75)
        mzdstrf /= np.max(np.abs(mzdstrf)) * 0.9
        mean_dstrf = mzdstrf
        del sdstrf

        log.info("Calling compute_dpcs()")
        dz = dtools.compute_dpcs(mzdstrf[:, 0], pc_count=pc_count, first_lin=first_lin, as_dict=True)

    else:
        dz = d
        mean_dstrf = mdstrf

    del dstrf

    log.info("Computing site-wide dPCs")
    # dpcs for all cells in site
    T = len(t_indexes)
    F = mean_dstrf.shape[3]
    U = mean_dstrf.shape[4]
    dstrf_all = np.reshape(mean_dstrf,[1,cellcount*T,F,U])
    dall = dtools.compute_dpcs(dstrf_all, pc_count=pc_count, snr_threshold=None,
                               first_lin=False, as_dict=True)

    N = 10
    log.info(f"Computing dPC noise floor for each unit (N={N})")
    # compute noise floor by measuring PCs with shuffled spectro-temporal parameters
    sh_mags = []
    for i in range(N):
        m_ = shuffle_along_axis(mean_dstrf, axis=2)
        d_ = dtools.compute_dpcs(m_[:, 0], pc_count=pc_count, first_lin=first_lin,
                                 snr_threshold=None, as_dict=True, flip_sign=False)
        sh_mags.append(d_[sig]['pc_mag'])
    sh_mags = np.stack(sh_mags, axis=2)
    msh = sh_mags.mean(axis=2)
    esh = sh_mags.std(axis=2)/(N**0.5)

    dpc = d[sig]['pcs']
    dpc_mag = d[sig]['pc_mag']
    dpcz = dz[sig]['pcs']
    dpc_magz = dz[sig]['pc_mag']
    dproj = dz[sig]['projection']
    log.info(f"dproj.shape={dproj.shape}")

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower','interpolation': 'none'}

    f, ax = plt.subplots(len(out_channels), pc_count+1, figsize=(pc_count, len(out_channels)*0.75), sharex='col', sharey='col')
    f.subplots_adjust(top=0.98, bottom=0.02)
    if len(out_channels) == 1:
        ax = ax[np.newaxis, ...]
    for oi, oc in enumerate(out_channels):
        ax[oi, -1].plot(msh[:,oi]/msh[:,oi].sum(), lw=0.5, color='gray')
        ax[oi, -1].plot(dpc_magz[:,oi]/dpc_mag[:,oi].sum())
        for di in range(pc_count):
            d = dpcz[oi, di]
            d = d / np.max(np.abs(d)) / np.max(dpc_magz[:, oi]) * dpc_magz[di, oi]
            ax[oi, di].imshow(np.fliplr(d), **imopts)
            ax[oi, di+1].set_yticklabels([])
        yl=ax[oi,-1].get_ylim()
        ax[oi, -1].text(0,yl[1],modelspec.meta['cellids'][oi], fontsize=6, va='top')

    if figures is None:
        figures = []
    figures.append(fig2BytesIO(f))
    modelspec.meta['dpc']=dpcz
    modelspec.meta['dpc_mag']=dpc_magz
    modelspec.meta['dpc_mag_sh']=msh
    modelspec.meta['dpc_mag_e']=esh
    modelspec.meta['dpc_all'] = dall['input']['pcs']
    modelspec.meta['dpc_mag_all'] = dall['input']['pc_mag']

    if fit_ss_model:
        d=subspace_model_fit(est, val, modelspec, out_channels=out_channels,
                           pc_count=ss_pccount, dpc_var=ss_dpc_var)
        modelspec=d['modelspec']

    log.info("removing backends from modelspec")
    modelspec.backend=None
    modelspec.dstrf_backend=None
    return {'modelspec': modelspec, 'figures': figures}


def subspace_model_fit(est, val, modelspec,
              pc_count=5, dpc_var=0.8, out_channels=None, use_dpc_all=False, single_fit=True,
              figures=None, IsReload=False, **ctx):

    if IsReload:
        # load dstrf data saved in modelpath
        return {'modelspec': modelspec}
    batch_size = None  # X_est.shape[0]  # or None or bigger?

    #try:
    #    X_est, Y_est = xforms.lite_input_dict(modelspec, est, epoch_name="REFERENCE")
    #    X_val, Y_val = xforms.lite_input_dict(modelspec, val, epoch_name="REFERENCE")
    #except:
    X_est, Y_est = xforms.lite_input_dict(modelspec, est, epoch_name="")
    X_val, Y_val = xforms.lite_input_dict(modelspec, val, epoch_name="")
    X_est['input'] = X_est['input'][np.newaxis]
    Y_est = Y_est[np.newaxis]
    X_val['input'] = X_val['input'][np.newaxis]
    Y_val = Y_val[np.newaxis]

    r = est
    cellids = r['resp'].chans
    if out_channels is None:
        out_channels = np.arange(len(cellids))

    ssmodels = []
    sspredxc = np.zeros(len(out_channels))
    sspc_count = np.zeros(len(out_channels))
    ss0predxc = np.zeros(len(out_channels))

    if use_dpc_all & single_fit:
        out_channels=[out_channels]
    else:
        single_fit=False
    for oi, o in enumerate(out_channels):
        if single_fit:
            R=len(o)
            log.info(f"** Fitting SS model for {R} cells:")
            y_select=o
            pcc=pc_count
            keywordstring = f'wc.{pc_count}x15-relu.15.o.s-wc.15x30-relu.30.o.s-wc.30x{R}-dexp.{R}'
        else:
            R=1
            log.info(f"** Fitting SS model for cell {val['resp'].chans[o]} ({oi+1}/{len(out_channels)}):")
            y_select=[o]
            if pc_count is None:
                if use_dpc_all:
                    dpc_mag = modelspec.meta['dpc_mag_all'][:, 0] ** 2
                else:
                    dpc_mag = modelspec.meta['dpc_mag'][:, o] ** 2
                dpc_mag = dpc_mag / dpc_mag.sum()
                dsum = np.cumsum(dpc_mag)
                pcc = int(np.min(np.where(dsum > dpc_var)[0]) + 1)
                log.info(f'dpc_var={dpc_var}: pc_count={pcc}')
            else:
                pcc=pc_count
            keywordstring = f'wc.{pcc}x15-relu.15.o.s-wc.15x15-relu.15.o.s-wc.15x1-dexp.1'

        lmodel0 = xforms.init_nems_keywords(keywordstring, meta=modelspec.meta)['modelspec']
        lmodel0 = lmodel0.sample_from_priors()
        fitter_options = {'cost_function': 'nmse', 'early_stopping_delay': 100,
                          'early_stopping_patience': 150,
                          'early_stopping_tolerance': 1e-3,
                          'learning_rate': 1e-3, 'epochs': 10000,
                          }
        fit_opts2 = fitter_options.copy()
        fit_opts2['early_stopping_tolerance'] = 1e-4
        fit_opts2['learning_rate'] = 5e-4
        if use_dpc_all:
            dpc = modelspec.meta['dpc_all'][0, :pcc]
            dpc = np.moveaxis(dpc, [0, 1, 2], [2, 1, 0])
        else:
            dpc = modelspec.meta['dpc'][o, :pcc]
            dpc = np.moveaxis(dpc, [0, 1, 2], [2, 1, 0])
        fir = filter.FIR(shape=dpc.shape)
        fir['coefficients'] = np.flip(dpc, axis=0)
        X = np.stack([fir.evaluate(x) for x in X_est['input']], axis=0)
        Y = Y_est[:, :, y_select]

        lmodel0.layers[-1].skip_nonlinearity()
        lmodel = lmodel0.fit(input=X, target=Y, backend='tf',
                             batch_size=batch_size, fitter_options=fitter_options)
        lmodel = init_nl_lite(lmodel, X, Y)
        lmodel.layers[-1].unskip_nonlinearity()
        lmodel = lmodel.fit(input=X, target=Y, backend='tf',
                            batch_size=batch_size, fitter_options=fit_opts2)

        Xv = np.concatenate([fir.evaluate(x) for x in X_val['input']], axis=0)
        Yv = np.reshape(Y_val[:, :, y_select], [-1, R])
        p0 = lmodel0.predict(Xv)
        p = lmodel.predict(Xv)
        if single_fit:
            for ii in range(R):
                sspredxc[ii] = correlation(p[:, ii], Yv[:, ii])
                ss0predxc[ii] = correlation(p0[:, ii], Yv[:, ii])
                sspc_count[ii] = pcc
        else:
            sspredxc[oi] = correlation(p, Yv)
            ss0predxc[oi] = correlation(p0, Yv)
            sspc_count[oi] = pcc
        ssmodels.append(lmodel)
        
    if single_fit:
        out_channels = out_channels[0]

    newmodelspec=modelspec.copy()
    newmodelspec.meta['sspredxc'] = sspredxc
    newmodelspec.meta['sspc_count'] = sspc_count
    if 'r_test' in newmodelspec.meta.keys():
        log.info("Cellid        Orig  Subspace")
        for oi, o in enumerate(out_channels):
            log.info(f"{newmodelspec.meta['cellids'][o]}" + \
                     f" {newmodelspec.meta['r_test'][o, 0]:.3f}" + \
                     f" {newmodelspec.meta['sspredxc'][oi]:.3f}")

    return {'modelspec': newmodelspec}

def plot_dpcs(modelspec=None, out_channels=None, **ctx):
    dpcz = modelspec.meta['dpc']
    dpc_magz = modelspec.meta['dpc_mag']
    msh=modelspec.meta.get('dpc_mag_sh', None)
    esh=modelspec.meta.get('dpc_mag_e', None)
    # modelspec.meta['dpc_all'] = dall['input']['pcs']
    # modelspec.meta['dpc_mag_all'] = dall['input']['pc_mag']
    if out_channels is None:
        out_channels=np.arange(len(modelspec.meta['cellids']))
    pc_count=dpcz.shape[1]

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower', 'interpolation': 'none'}

    f, ax = plt.subplots(len(out_channels), pc_count + 1, figsize=(pc_count, len(out_channels) * 0.75), sharex='col',
                         sharey='col')
    f.subplots_adjust(top=0.98, bottom=0.02)
    if len(out_channels) == 1:
        ax = ax[np.newaxis, ...]
    for oi, oc in enumerate(out_channels):
        if msh is not None:
            ax[oi, -1].plot(msh[:,oi]/msh[:,oi].sum(), lw=0.5, color='gray')
        ax[oi, -1].plot(dpc_magz[:, oc] / dpc_magz[:, oc].sum())
        for di in range(pc_count):
            d = dpcz[oc, di]
            d = d / np.max(np.abs(d)) / np.max(dpc_magz[:, oc]) * dpc_magz[di, oc]
            ax[oi, di].imshow(np.fliplr(d), **imopts)
            ax[oi, di + 1].set_yticklabels([])
        yl = ax[oi, -1].get_ylim()
        ax[oi, -1].text(0, yl[1], modelspec.meta['cellids'][oc], fontsize=6, va='top')
    return f


def project_to_subspace(modelspec=None, X=None, out_channels=None, rec=None, est=None, val=None,
                        input_name='stim', ss_name='subspace', verbose=True, **ctx):

    cellids = modelspec.meta['cellids']
    if out_channels is None:
        out_channels = np.arange(len(cellids))
    if X is None:
        recs = [(n,r) for n,r in zip(['rec', 'est','val'],[rec, est, val]) if r is not None]
    else:
        recs = [('raw', X)]
    if X is None and (len(recs)==0):
        raise ValueError("must provide either X input matrix or valid NEMS recording")
    if 'dpc' not in modelspec.meta:
        raise ValueError("modelspec missing dSTRF pcs, run nems_lbhb.analysis.dstrf.dstrf_pca first")
    log.info(f"{out_channels}")
    res = {}
    for name, rec in recs:
        if verbose:
            log.info(f"** Recording {name}:")

        if type(rec) is not np.ndarray:
            inp = rec[input_name].as_continuous().T
        else:
            inp = rec

        outs = []
        res[name]=rec.copy()
        outcells=[c for i,c in enumerate(modelspec.meta['cellids']) if i in out_channels]
        for oi, o in enumerate(out_channels):
            if verbose:
                log.info(f"   Computing SS projection for {cellids[o]}:")

            dpcz = modelspec.meta['dpc']
            dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, o]
            fir = filter.FIR(shape=dpcz.shape)
            fir['coefficients'] = np.flip(dpcz, axis=0)

            ss = fir.evaluate(inp)
            outs.append(ss.T)

        ssout = np.stack(outs, axis=0)

        if name == 'raw':
            return ssout

        sig = res[name][input_name]._modified_copy(data=ssout, name=ss_name, chans=outcells)
        res[name].signals[ss_name] = sig

    return res

def project_model_to_ss(modelspec, X=None, rec=None, input_name='stim',
                        cellid=None, invert=False):

    if X is None:
        inp = rec[input_name].as_continuous().T
    else:
        inp = X
    if cellid is None:
        oi = 0
    else:
        oi = [i for i,c in enumerate(modelspec.meta['cellids']) if c==cellid][0]
    dpcz = modelspec.meta['dpc']
    dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
    fir = filter.FIR(shape=dpcz.shape)
    fir['coefficients'] = np.flip(dpcz, axis=0)
    print(oi,cellid)
    ss = fir.evaluate(inp)

    return ss



def plot_dpc_space(modelspec=None, cell_list=None, val=None, est=None, modelspec2=None, show_preds=True, plot_stim=True,
                   use_val=False, print_figs=False, **ctx):
    if cell_list is None:
        cell_list = ctx['cellids']
    elif type(cell_list) is str:
        cell_list = [cell_list]
    if use_val:
        rec=val
    else:
        rec=est
    X_est, Y_est = xforms.lite_input_dict(modelspec, rec, epoch_name="")

    orange = [i for i,c in enumerate(rec['resp'].chans) if c in cell_list]
    spont = np.zeros(len(rec['resp'].chans))
    print(orange)
    if modelspec2 is not None:
        lnstrf=LN.LNpop_get_strf(modelspec2)
    else:
        lnstrf=None
    for oi, cellid in zip(orange,cell_list):
        print(oi, cellid)

        #Y = project_to_subspace(modelspec, X=None, out_channels=None, rec=None, est=None, val=None,
        #                    input_name='stim', ss_name='subspace', verbose=True, **ctx)
        X = rec['stim'].as_continuous().T
        Y = project_to_subspace(modelspec, X, out_channels=[oi])[0].T

        dpcz = modelspec.meta['dpc']
        pc_count = dpcz.shape[1]
        dpc_magz = modelspec.meta['dpc_mag']
        dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
        #fir = filter.FIR(shape=dpcz.shape)
        #fir['coefficients'] = np.flip(dpcz, axis=0)

        pred = rec['pred'].as_continuous().T[:, oi]
        r = rec['resp'].as_continuous().T[:, oi]
        #X = rec['stim'].as_continuous().T
        #Y = fir.evaluate(X)
        #for i in range(pc_count):
        #    cc=np.corrcoef(Y[:,i],r)[0,1]
        #    if cc<0:
        #        Y[:,i]=-Y[:,i]
        #        modelspec.meta['dpc'][oi, i] = -modelspec.meta['dpc'][oi, i]

        pcp = 3
        imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
                  'interpolation': 'none'}
        if show_preds:
            f, ax = plt.subplots(pcp, 4, figsize=(6, 4.5))
        else:
            f, ax = plt.subplots(pcp, 3, figsize=(3, 4.5))
        for i in range(pcp):
            d = modelspec.meta['dpc'][oi, i]
            d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]

            if i == 0:
                ax[i, 1].imshow(np.fliplr(d), **imopts)
                ax[i, 1].set_ylabel(f'Dim {i + 1}')
            else:
                ax[i, 0].imshow(np.fliplr(d), **imopts)
                ax[i, 0].set_ylabel(f'Dim {i + 1}')

        ax[0, 0].plot(np.arange(1, dpc_magz.shape[0] + 1), dpc_magz[:, oi] / dpc_magz[:, oi].sum(), 'o-', markersize=3)
        ax[0, 0].set_xlabel('PC dimension')
        ax[0, 0].set_ylabel('Var. explained')
        ax[0, 0].set_xticks(np.arange(1, dpc_magz.shape[0] + 1))
        for j in range(1, pcp):
            Zresp = None
            Zpred = None
            ac,bc,Zresp,N=histmean2d(Y[:,0],Y[:,j],r, bins=20, ax=ax[j,1], spont=spont[oi], ex_pct=0.025, Z=Zresp)
            #N = histscatter2d(Y[:, 0], Y[:, j], r, N=1000, ax=ax[j, 1], ex_pct=0.05)
            ax[j, 1].set_xlabel('Dim 1')
            ax[j, 1].set_ylabel(f"Dim {j + 1}")
            if show_preds:
                ac,bc,Zpred,N=histmean2d(Y[:,0],Y[:,j],pred, bins=20, ax=ax[j,2], spont=spont[oi], ex_pct=0.025, Z=Zpred)
                #N = histscatter2d(Y[:, 0], Y[:, j], pred, N=1000, ax=ax[j, 2], ex_pct=0.05)
                ax[j, 2].set_xlabel('Dim 1')
                ax[j, 2].set_ylabel(f"Dim {j + 1}")
        if show_preds & (lnstrf is not None):
            l = lnstrf[:,:,oi]
            l /= np.abs(l).max()
            ax[0, 2].imshow(lnstrf[:,:15,oi], **imopts)
            ax[0, 2].set_title(f"LN: {modelspec2.meta['r_test'][oi, 0]:.3f}")

        ax[0, 0].set_title(cellid)
        ax[0, 1].set_title(f"CNN: {modelspec.meta['r_test'][oi, 0]:.3f} SS: {modelspec.meta['sspredxc'][oi]:.3f}")

        ymin, ymax = 1,0
        for pci in range(3):
            y = Y[:, pci]
            b = np.linspace(y.min(), y.max(), 11)
            mb = (b[:-1] + b[1:]) / 2
            mr = [np.mean(r[(y >= b[i]) & (y < b[i + 1])]) for i in range(10)]
            me = [np.std(r[(y >= b[i]) & (y < b[i + 1])]) / np.sqrt(np.sum((y >= b[i]) & (y < b[i + 1]))) for i in range(10)]
            ax[pci,3].errorbar(mb, mr, me)
            ax[pci,3].set_xlabel(f'Dim {pci+1} proj')
            ax[pci,3].set_ylabel('Mean prediction')
            yl=ax[pci,3].get_ylim()
            ymin = np.min([yl[0],ymin])
            ymax = np.max([yl[1], ymax])
        for pci in range(3):
            ax[pci, 3].set_ylim((ymin,ymax))
        ax[0,2].set_axis_off()
        plt.tight_layout()
    return f

def plot_dpc_proj(modelspec=None, cell_list=None, val=None, est=None, modelspec2=None, show_preds=True, plot_stim=True,
                   use_val=False, print_figs=False, T1=50, T2=250, **ctx):
    if cell_list is None:
        cell_list = ctx['cellids']
    elif type(cell_list) is str:
        cell_list = [cell_list]
    if use_val:
        rec=val
    else:
        rec=est

    X_est, Y_est = xforms.lite_input_dict(modelspec, rec, epoch_name="")
    fs=rec['resp'].fs

    orange = [i for i,c in enumerate(rec['resp'].chans) if c in cell_list]
    spont = np.zeros(len(rec['resp'].chans))

    cnnPred = modelspec.predict(X_est)
    if modelspec2 is not None:
        lnstrf=LN.LNpop_get_strf(modelspec2)
        lnPred = modelspec2.predict(X_est)
    else:
        lnstrf=None
        lnPred=None

    for oi, cellid in zip(orange,cell_list):
        print(oi, cellid)
        pcp = 2
        rows = pcp+4
        f = plt.figure(figsize=(8, rows))
        gs = f.add_gridspec(rows, 10)
        ax = np.zeros((rows-1, 3), dtype='O')
        for r in range(rows-1):
            for c in range(3):
                if (c <= 1):
                    ax[r, c] = f.add_subplot(gs[r, c])
                elif (c > 1):
                    ax[r, c] = f.add_subplot(gs[r, c:])
        ax2 = np.zeros(10, dtype='O')
        for c in range(10):
            ax2[c] = f.add_subplot(gs[-1, c])

        dpcz = modelspec.meta['dpc']
        pc_count = dpcz.shape[1]
        dpc_magz = modelspec.meta['dpc_mag']
        dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
        fir = filter.FIR(shape=dpcz.shape)
        fir['coefficients'] = np.flip(dpcz, axis=0)

        pred = rec['pred'].as_continuous().T[:, oi]
        X = rec['stim'].as_continuous().T
        r = rec['resp'].as_continuous().T[:, oi]
        if use_val:
            pred2 = est['pred'].as_continuous().T[:, oi]
            pred = np.concatenate([pred,pred2])
            X2 = est['stim'].as_continuous().T
            X = np.concatenate([X,X2], axis=0)
            r2 = est['resp'].as_continuous().T[:, oi]
            r = np.concatenate([r, r2])

        Y = fir.evaluate(X)
        for i in range(pc_count):
            cc=np.corrcoef(Y[:,i],r)[0,1]
            if cc < 0:
                Y[:,i]=-Y[:,i]
                modelspec.meta['dpc'][oi, i] = -modelspec.meta['dpc'][oi, i]

        imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
                  'interpolation': 'none'}
        trange=np.arange(T1,T2)/fs

        ax[0, 0].plot(np.arange(1, dpc_magz.shape[0] + 1), dpc_magz[:, oi] / dpc_magz[:, oi].sum(), 'o-', markersize=3)
        ax[0, 0].set_xlabel('PC dimension')
        ax[0, 0].set_ylabel('Var. explained')
        ax[0, 0].set_xticks(np.arange(1, dpc_magz.shape[0] + 1))
        ax[0, 2].imshow(X[T1:T2, :].T, origin='lower', cmap='gray_r',
                        extent=[trange[0],trange[-1],0,X.shape[1]])
        for i in range(pcp):
            d = modelspec.meta['dpc'][oi, i]
            d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]

            ax[i+1, 0].imshow(np.fliplr(d), **imopts)
            ax[i+1, 0].set_ylabel(f'Dim {i + 1}')
            ax[i+1, 1].imshow(d, **imopts)

            ax[i+1, 2].plot(trange,Y[T1:T2,i])
            ax[i+1, 2].set_xlim(trange[0],trange[-1])

        mm = Y_est[T1:T2, oi].max()
        ax[-2, 2].plot(trange, smooth(Y_est[T1:T2, oi].T,7), 'k', lw=0.5)
        ax[-2, 2].plot(trange,cnnPred[T1:T2, oi].T,color='purple')
        ax[-2, 2].set_xlim(trange[0], trange[-1])
        ax[-2, 2].text(trange[-1], mm, f"CNN: {modelspec.meta['r_test'][oi, 0]:.3f}", ha='right')

        if (lnstrf is not None):
            l = lnstrf[:,:,oi]
            l /= np.abs(l).max()
            ax[-1, 0].imshow(lnstrf[:,:15,oi], **imopts)
            ax[-1, 1].imshow(np.fliplr(lnstrf[:,:15,oi]), **imopts)
            ax[-1, 0].set_ylabel(f'LN STRF')

            ax[-1, 2].plot(trange, smooth(Y_est[T1:T2, oi].T,7), 'k', lw=0.5)
            ax[-1, 2].plot(trange, lnPred[T1:T2, oi].T,color='orange')
            ax[-1, 2].text(trange[-1], mm, f"LN: {modelspec2.meta['r_test'][oi, 0]:.3f}", ha='right')
            ax[-1, 2].set_xlim(trange[0], trange[-1])

        ac,bc,Zresp,N=histmean2d(Y[:,0], Y[:,1], pred, bins=20, ax=ax[0, 1], spont=spont[oi], ex_pct=0.01)
        T_set = np.arange(T1+10, T1+120, 10)
        for c,a in enumerate(ax2):
            t1, t2 = T_set[c], T_set[c+1]
            ac, bc, Zresp, N = histmean2d(Y[:, 0], Y[:, 1], pred, bins=20, ax=a, spont=spont[oi], ex_pct=0.025, Z=Zresp)
            a.plot(Y[T1:t2,0], Y[T1:t2,1], color='gray', lw=0.5)
            a.plot(Y[t1:t2,0], Y[t1:t2,1], color='lightgray')
            a.set_xlabel(f"{t1/fs:.1f}-{t2/fs:.1f} s", fontsize=8)
        for c,a in enumerate(ax.flatten()):
            if c<len(ax.flatten())-1:
                a.set_xticklabels([])
            else:
                x=a.get_xticklabels()
                a.set_xticklabels(x, fontsize=8)

            a.set_yticklabels([])

        yl=ax2[0].get_ylim()
        xl=ax2[0].get_xlim()
        for a in ax2.flatten():
            a.set_xticklabels([])
            a.set_yticklabels([])
            #a.set_axis_off()
            a.set_ylim(yl)
            a.set_xlim(xl)
        plt.tight_layout()
    return f
