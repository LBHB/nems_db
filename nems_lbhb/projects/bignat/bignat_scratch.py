from os.path import basename, join
import logging
import os
import io
import importlib

import numpy as np
import matplotlib.pyplot as plt

from nems0.utils import escaped_split, escaped_join, get_setting
from nems0.registry import KeywordRegistry, xforms_lib
from nems0 import xform_helper, xforms, db

from nems.layers import filter
from nems import Model
from nems.metrics import correlation
from nems.preprocessing import split
from nems.models.dataset import DataSet
from nems0.initializers import init_nl_lite

from nems0.registry import xform, scan_for_kw_defs
from nems.layers.tools import require_shape, pop_shape
from nems_lbhb.projects.bignat.bnt_defaults import POP_MODELS,SIG_TEST_MODELS
import nems_lbhb.plots as nplt

log = logging.getLogger(__name__)

use_saved_model=False

batch=343

siteids, cellids = db.get_batch_sites(batch)

load_kw = 'gtgram.fs100.ch18-ld-norm.l1-sev'
fit_kw = 'lite.tf.init.lr1e3.t3.es20.jk3-lite.tf.lr1e4.t5e4-dstrf.d15.t43.p5.ss'
modelnames = [
    f'{load_kw}_wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80.l2:4-fir.10x1x80-relu.80.f-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
    f'{load_kw}_wc.18x1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_{fit_kw}',
    f'{load_kw}_wc.18x1x6.g-fir.25x1x6-relu.6.f-wc.6x1-dexp.1_{fit_kw}'
]
shortnames = ['CNN 1d','LN','CNN single']
modelname = modelnames[0]
siteid = "CLT028c"
siteid = "PRN021a"
siteid = "PRN007a"
cellid = siteid

for i,m in enumerate(modelnames):
    if m==modelname:
        print(f'* {i:2d} {shortnames[i]:12s}  {m}')
    else:
        print(f'  {i:2d} {shortnames[i]:12s}  {m}')

if use_saved_model:
    cellid = [c for c in cellids if c.startswith(siteid)][0]

    xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True)
else:

    autoPlot = True
    saveInDB = True
    browse_results = False
    saveFile = True

    log.info('Initializing modelspec(s) for cell/batch %s/%d...', cellid, int(batch))

    # Segment modelname for meta information
    kws = modelname.split("_")
    modelspecname = "-".join(kws[1:-1])
    loadkey = kws[0]
    fitkey = kws[-1]

    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loadkey}

    xforms_kwargs = {}
    xforms_init_context = {'cellid': cellid, 'batch': int(batch)}
    recording_uri = None
    kw_kwargs = {}

    # equivalent of xform_helper.generate_xforms_spec():

    # parse modelname and assemble xfspecs for loader and fitter
    load_keywords, model_keywords, fit_keywords = escaped_split(modelname, '_')

    # Generate the xfspec, which defines the sequence of events
    # to run through (like a packaged-up script)
    xfspec = []

    # 0) set up initial context
    if xforms_init_context is None:
        xforms_init_context = {}
    if kw_kwargs is not None:
         xforms_init_context['kw_kwargs'] = kw_kwargs
    xforms_init_context['keywordstring'] = model_keywords
    xforms_init_context['meta'] = meta
    xfspec.append(['nems0.xforms.init_context', xforms_init_context])
    xforms_lib.kwargs = xforms_init_context.copy()

    # 1) Load the data
    xfspec.extend(xform_helper._parse_kw_string(load_keywords, xforms_lib))

    log.info("NEMS lite fork")
    # nems-lite fork
    xfspec.append(['nems0.xforms.init_nems_keywords', {}])

    xfspec.extend(xform_helper._parse_kw_string(fit_keywords, xforms_lib))
    xfspec.append(['nems0.xforms.predict_lite', {}])
    xfspec.append(['nems0.xforms.add_summary_statistics', {}])
    xfspec.append(['nems0.xforms.plot_lite', {}])

    # equivalent of xforms.evaluate():

    # Create a log stream set to the debug level; add it as a root log handler
    log_stream = io.StringIO()
    ch = logging.StreamHandler(log_stream)
    ch.setLevel(logging.DEBUG)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)
    rootlogger = logging.getLogger()
    rootlogger.addHandler(ch)

    ctx = {}
    for xfa in xfspec:
        if not('postprocess' in xfa[0]):
            ctx = xforms.evaluate_step(xfa, ctx)

    # Close the log, remove the handler, and add the 'log' string to context
    log.info('Done (re-)evaluating xforms.')
    ch.close()
    rootlogger.removeFilter(ch)

    log_xf = log_stream.getvalue()

    # save some extra metadata
    modelspec = ctx['modelspec']

    if saveFile:
        # save results
        if get_setting('USE_NEMS_BAPHY_API'):
            prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
        else:
            prefix = get_setting('NEMS_RESULTS_DIR')

        if type(cellid) is list:
            cell_name = cellid[0].split("-")[0]
        else:
            cell_name = cellid

        if modelspec.meta.get('engine', 'nems0') == 'nems-lite':
            xforms.save_lite(xfspec=xfspec, log=log_xf, **ctx)
        else:
            destination = os.path.join(prefix, str(batch), cell_name, modelspec.get_longname())

            for cellidx in range(modelspec.cell_count):
                modelspec.set_cell(cellidx)
                modelspec.meta['modelpath'] = destination
                modelspec.meta['figurefile'] = os.path.join(destination, 'figure.0000.png')
            modelspec.set_cell(0)

            log.info('Saving modelspec(s) to {0} ...'.format(destination))
            if ctx.get('save_context', False):
                ctx['log']=log_xf
                save_data = xforms.save_context(destination,
                                                ctx=ctx,
                                                xfspec=xfspec)
            else:
                save_data = xforms.save_analysis(destination,
                                                 recording=ctx.get('rec'),
                                                 modelspec=modelspec,
                                                 xfspec=xfspec,
                                                 figures=ctx.get('figures'),
                                                 log=log_xf,
                                                 update_meta=False)

    if saveInDB:
        # save performance and some other metadata in database Results table
        modelspec.meta['extra_results']='test'
        db.update_results_table(modelspec)

    for xfa in xfspec:
        if 'postprocess' in xfa[0]:
            log.info(f'Running postprocessing kw: {xfa[0]}')
            ctx = xforms.evaluate_step(xfa, ctx)

    log.info('Test fit complete')

for cid,cellid in enumerate(ctx['modelspec'].meta['cellids']):
    print(f"{cid:2d} {cellid} {ctx['modelspec'].meta['r_test'][cid,0]:.3f}")
print(f"MN ------------- {ctx['modelspec'].meta['r_test'].mean():.3f}")

#raise ValueError('model fit complete')

X_val, Y_val = xforms.lite_input_dict(ctx['modelspec'], ctx['val'], epoch_name="")
X_est, Y_est = xforms.lite_input_dict(ctx['modelspec'], ctx['est'], epoch_name="")

modelspec=ctx['modelspec']
dpcz = modelspec.meta['dpc']
dpc_magz = modelspec.meta['dpc_mag']
out_channels = np.arange(Y_est.shape[1])
pc_count = dpcz.shape[1]

# imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
#           'interpolation': 'none'}
#
# f, ax = plt.subplots(len(out_channels), pc_count + 1, figsize=(pc_count, len(out_channels)), sharex='col', sharey='col')
# f.subplots_adjust(top=0.98, bottom=0.02)
# if len(out_channels) == 1:
#     ax = ax[np.newaxis, ...]
# for oi, o in enumerate(out_channels):
#     ax[oi, -1].plot(dpc_magz[:, oi] / dpc_magz[:, oi].sum())
#     for di in range(pc_count):
#         d = dpcz[oi, di]
#         d = d / np.max(np.abs(d)) / dpc_magz[0, oi] * dpc_magz[di, oi]
#         ax[oi, di].imshow(np.fliplr(d), **imopts)
#         ax[oi, di + 1].set_yticklabels([])
#     ax[oi, -2].text(0,10,modelspec.meta['cellids'][o], fontsize=6)

#plt.close('all')
"""
from nems_lbhb.analysis import dstrf
import importlib
importlib.reload(dstrf)

out_channels=[27,28,29,30]
res = postprocessing.dstrf_pca(D=10, timestep=123, pc_count=7, out_channels=out_channels, **ctx)

val=ctx['val'].apply_mask()
stim=val['stim'].as_continuous()
"""

def histmean2d(a,b,d, bins=10, ax=None):
    # av = np.percentile(a, np.linspace(0, 100, bins+1))
    # bv = np.percentile(b, np.linspace(0, 100, bins+1))
    ab = np.percentile(a, [0.1, 99.9])
    bb = np.percentile(b, [0.1, 99.9])
    av = np.linspace(ab[0], ab[1], bins + 1)
    bv = np.linspace(bb[0], bb[1], bins + 1)
    ac = (av[:-1]+av[1:])/2
    bc = (bv[:-1]+bv[1:])/2

    mmv = np.zeros((bins, bins)) * np.nan
    N = np.zeros((bins, bins))

    for i_, a_ in enumerate(av[:-1]):
        for j_, b_ in enumerate(bv[:-1]):
            v_ = (a >= a_) & (a < av[i_ + 1]) & (b >= b_) & (b < bv[j_ + 1]) & np.isfinite(d)
            if (v_.sum() > 0):
                mmv[j_, i_] = np.nanmean(d[v_])
                N[j_,i_] = v_.sum()


    # find the zero bin:
    zi = np.min(np.where(av>=0)[0])-1
    zj = np.min(np.where(bv>=0)[0])-1
    r0 = mmv[zj,zi]
    mmv -=r0

    mmv[np.isnan(mmv)] = 0

    if ax is None:
        f,ax = plt.subplots()

    # option to interpolate (not used)
    # x = (llv[:-1] + llv[1:]) / 2
    # y = (ttv[:-1] + ttv[1:]) / 2
    # X, Y = np.meshgrid(x, y)  # 2D grid for interpolation
    #
    # valididx = np.isfinite(mm)
    # interp = LinearNDInterpolator(list(zip(X[valididx], Y[valididx])),
    #                               mm[valididx], fill_value=np.nanmean(mm))

    # plot heatmaps
    Z = mmv
    # Z = interp(X, Y)
    # zsm = 0.5
    # Zz = (Z == 0)
    # Z = gaussian_filter(Z, [zsm, zsm])
    # Z[Zz] = 0

    # cmap = matplotlib.cm.get_cmap('bwr')
    # cmap_ = cmap(np.linspace(0, 1, 255))
    # cmap_[127] = np.array([0.75, 0.75, 0.75, 1])
    # cmap_ = matplotlib.colors.ListedColormap(cmap_)
    cmap='bwr'

    vmin,vmax = -np.max(np.abs(Z)), np.max(np.abs(Z))

    im = ax.imshow(Z, extent=[bv[0], bv[-1], av[-1], av[0]],
                   interpolation='none', aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.contour(bc, ac, N, [0.5], linewidths=0.5)
    ax.axhline(0, ls='--')
    ax.axvline(0, ls='--')
    return N

oi, o = 20,20
out_channels=np.arange(Y_est.shape[1])
plt.close('all')
for oi in range(15,20):
    o=out_channels[oi]
    dpcz = modelspec.meta['dpc']
    dpc_magz = modelspec.meta['dpc_mag']
    dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
    fir = filter.FIR(shape=dpcz.shape)
    fir['coefficients'] = np.flip(dpcz, axis=0)

    #X = X_val['input']
    #r = Y_val[:, o]
    X = X_est['input']
    r = Y_est[:, o]
    Y = fir.evaluate(X)

    pcp=3
    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
              'interpolation': 'none'}
    f,ax = plt.subplots(3,pcp+1)
    for i in range(pcp):
        d = modelspec.meta['dpc'][oi, i]
        d = d / np.max(np.abs(d)) / dpc_magz[0, oi] * dpc_magz[i, oi]
        if i==0:
            ax[0, i].imshow(np.fliplr(d), **imopts)
            ax[0, i].set_ylabel(f'Dim {i+1}')
        else:
            ax[1, i].imshow(np.fliplr(d), **imopts)
            ax[1, i].set_xlabel(f'Dim {i+1}')

        y = Y[:, i]
        #b = np.percentile(y, np.linspace(0, 100, 11))
        b = np.linspace(y.min(), y.max(), 11)

        # b=np.linspace(y.min(),y.max(),11)
        mb = (b[:-1] + b[1:]) / 2
        mr = [np.mean(r[(y >= b[i]) & (y < b[i + 1])]) for i in range(10)]
        me = [np.std(r[(y >= b[i]) & (y < b[i + 1])]) / np.sqrt(np.sum((y >= b[i]) & (y < b[i + 1]))) for i in range(10)]
        ax[2, i].errorbar(mb, mr, me)

    ax[0, -1].plot(dpc_magz[:, oi] / dpc_magz[:, oi].sum())

    for j in range(1,pcp):
        N=histmean2d(Y[:,0],Y[:,j],r, bins=20, ax=ax[0,j])
    ax[0,0].set_title(modelspec.meta['cellids'][o])
    ax[0,1].set_title(f" Orig : {ctx['modelspec'].meta['r_test'][o, 0]:.3f}")
    ax[0,2].set_title(f"  Subspace predxc: {modelspec.meta['sspredxc'][oi]:.3f}")
    plt.tight_layout()

log.info("Cellid        Orig  Subspace")
for oi, o in enumerate(out_channels):
    log.info(f"{modelspec.meta['cellids'][o]}" + \
             f" {modelspec.meta['r_test'][o, 0]:.3f}" + \
             f" {modelspec.meta['sspredxc'][oi]:.3f}")

f,ax = plt.subplots()
nplt.scatter_comp(modelspec.meta['r_test'][:,0],
                  modelspec.meta['sspredxc'],
                  n1='CNN',n2='Subspace',hist_range=[0,1], ax=ax)
ax.set_title(siteid)

raise ValueError('summary plots complete')

# f, ax = plt.subplots(len(out_channels), pc_count, sharey='row')
# for oi, o in enumerate(out_channels):
#     dpcz = modelspec.meta['dpc']
#     dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, oi]
#     fir = filter.FIR(shape=dpcz.shape)
#     fir['coefficients'] = np.flip(dpcz,axis=0)
#     X = X_val['input']
#     Y = fir.evaluate(X)
#
#     r = Y_val[:,[o]]
#
#     for i in range(Y.shape[1]):
#         y = Y[:, i]
#         b = np.percentile(y, np.linspace(0, 100, 11))
#
#         #b=np.linspace(y.min(),y.max(),11)
#         mb = (b[:-1]+b[1:])/2
#         mr = [np.mean(r[(y>=b[i]) & (y<b[i+1])]) for i in range(10) ]
#         me = [np.std(r[(y>=b[i]) & (y<b[i+1])])/np.sqrt(np.sum((y>=b[i]) & (y<b[i+1]))) for i in range(10) ]
#         ax[oi, i].errorbar(mb, mr, me)
#
#     ax[oi, 0].set_ylabel(modelspec.meta['cellids'][o], fontsize=6)
# plt.tight_layout()

from nems_lbhb.analysis import dstrf
import importlib
importlib.reload(dstrf)

res = dstrf.dstrf_pca(D=12, timestep=123, pc_count=5, **ctx)
modelspec=res['modelspec']
res = dstrf.subspace_model_fit(modelspec=modelspec, est=ctx['est'],
                               val=ctx['val'],
                               pc_count=pc_count, out_channels=out_channels)
modelspec = res['modelspec']
nplt.scatter_comp(modelspec.meta['r_test'][:,0],
                  modelspec.meta['sspredxc'],
                  n1='CNN',n2='Subspace',hist_range=[0,1])


