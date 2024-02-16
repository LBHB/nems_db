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

from nems_lbhb.plots import histscatter2d, histmean2d
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
from nems_lbhb.plots import histscatter2d, histmean2d, scatter_comp

log = logging.getLogger(__name__)

use_saved_model = True

batch = 343
siteids, cellids = db.get_batch_sites(batch)

load_kw = 'gtgram.fs100.ch18-ld-norm.l1-sev'
fit_kw = 'lite.tf.init.lr1e3.t3.es20.jk5.rb3-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p5.ss'
modelnames = [
    f'{load_kw}_wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80.l2:4-fir.10x1x80-relu.80.f-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
    f'{load_kw}_wc.18x1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_{fit_kw}',
]

shortnames = ['CNN 1d','LN']
modelname = modelnames[0]

siteid = "PRN007a"
siteid = "PRN018a"
siteid = "CLT028c"
cellid = siteid

dpred = db.batch_comp(batch=batch, modelnames=modelnames, shortnames=shortnames)
dpred['siteid']=dpred.index
dpred['siteid']=dpred['siteid'].apply(db.get_siteid)

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
    xforms_init_context['meta'] = metaxforms_lib
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
            prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":xforms_lib"+str(get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
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
    #raise ValueError('model fit complete')

for cid,cellid in enumerate(ctx['modelspec'].meta['cellids']):
    print(f"{cid:2d} {cellid} {ctx['modelspec'].meta['r_test'][cid,0]:.3f}")
print(f"MN ------------- {ctx['modelspec'].meta['r_test'].mean():.3f}")


X_val, Y_val = xforms.lite_input_dict(ctx['modelspec'], ctx['val'], epoch_name="")
X_est, Y_est = xforms.lite_input_dict(ctx['modelspec'], ctx['est'], epoch_name="")

modelspec=ctx['modelspec']
dpcz = modelspec.meta['dpc']
dpc_magz = modelspec.meta['dpc_mag']
out_channels = np.arange(Y_est.shape[1])
pc_count = dpcz.shape[1]

import importlib
from nems_lbhb.analysis import dstrf
importlib.reload(dstrf)
from nems.tools import dstrf as dtools

cell_list=ctx['modelspec'].meta['cellids'][:3]
dstrf.plot_dpc_space(cell_list=cell_list, **ctx)

dpcp={'D': 15, 'timestep': 243, 'pc_count': 5, 'fit_ss_model': False,
      'first_lin': True}
res = dstrf.dstrf_pca(ctx['est'], ctx['modelspec'], val=ctx['val'], **dpcp)

dpc_mag = res['modelspec'].meta['dpc_mag']
msh = res['modelspec'].meta['dpc_mag_sh']
esh = res['modelspec'].meta['dpc_mag_sh']
plt.figure()

cid=7
x=np.arange(pc_count)
plt.errorbar(x,msh[:,cid],esh[:,cid])
#plt.plot(sh_mags[:,cid,:],lw=0.5,color='lightgray')
plt.plot(x,dpc_mag[:,cid])

raise ValueError('done')

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

first_lin=False
d = dtools.compute_dpcs(mdstrf[:, 0], pc_count=5, first_lin=first_lin, as_dict=True)
dpc = d['input']['pcs']
dpc_mag = d['input']['pc_mag']

sh_mags=[]
N=10
for i in range(N):
    m_ = shuffle_along_axis(mdstrf,axis=2)
    d_ = dtools.compute_dpcs(m_[:, 0], pc_count=5, first_lin=first_lin, as_dict=True)
    sh_mags.append(d_['input']['pc_mag'])
sh_mags=np.stack(sh_mags,axis=2)
msh=sh_mags.mean(axis=2)
esh=sh_mags.std(axis=2)# /(N**0.5)






plt.close('all')
for oi in orange:
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
        #N=histmean2d(Y[:,0],Y[:,j],r, bins=20, ax=ax[0,j])
        N = histscatter2d(Y[:, 0], Y[:, j], r, N=1000, ax=ax[0, j])
    ax[0,0].set_title(modelspec.meta['cellids'][o])
    ax[0,1].set_title(f" Orig : {ctx['modelspec'].meta['r_test'][o, 0]:.3f}")
    ax[0,2].set_title(f"  Subspace predxc: {modelspec.meta['sspredxc'][oi]:.3f}")
    plt.tight_layout()

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


