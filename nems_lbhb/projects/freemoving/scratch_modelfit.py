from os.path import basename, join
import logging
import os
import io

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, butter, sosfilt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nems0 import db
import nems0.epoch as ep
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems.layers import WeightChannels, FIR, LevelShift, \
    DoubleExponential, RectifiedLinear, Sigmoid, ConcatSignals, \
    MultiplySignals, MultiplyByExp, WeightGaussianExpand
from nems import Model
from nems.layers.base import Layer, Phi, Parameter
import nems.visualization.model as nplt
#import nems0.plots.api as nplt
from nems_lbhb.projects.freemoving import free_model, free_tools
from nems0.modules.nonlinearity import _dlog
from nems0.epoch import epoch_names_matching
from nems0.xform_helper import fit_model_xform
from nems0.utils import escaped_split, escaped_join, get_setting
from nems0.registry import KeywordRegistry, xforms_lib
from nems0 import xform_helper, xforms, db

log = logging.getLogger(__name__)

from nems0.registry import xform, scan_for_kw_defs
from nems.layers.tools import require_shape, pop_shape

siteid='PRN048a'
cellid=siteid
rasterfs = 50
batch=348

dlc_count=10
dlc1 = 40
strf_channels=20

dlc_memory=4
acount=20
dcount=10
l2count=30
tcount=acount+dcount
input_count = 36

old_model = True
if old_model:
    sep_kw = f'wcst.Nx1x{acount}.i-wcdl.{dlc_count}x1x{dcount}.i-first.8x1x{acount}-firdl.{dlc_memory}x1x{dcount}-cat-relu.{tcount}.o.s'
    aud_kw = f'wc.{tcount}x1x{l2count}-fir.4x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}xR-relu.R.o.s'
    model_kw = sep_kw + '-' + aud_kw
else:
    hrtf_kw = f'wcdl.{dlc_count}x{dlc1}.i-relud.{dlc1}.o.s-wcdl.{dlc1}x10-relud.10.o.s-wcdl.10x5-relud.5.o.s-wcdl.5x{input_count}-sigd.{input_count}.s.g-mult'
    aud_kw = f'wch.Nx1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}.o.s-wc.{strf_channels}x1x{l2count}-fir.10x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}xR-relu.R.o.s'
    model_kw = hrtf_kw + '-' + aud_kw

load_kw = "free.fs100.ch18-norm.l1-fev-shuf.dlc"
fit_kw = "lite.tf.cont.init.lr1e3.t3-lite.tf.cont.lr1e4"

modelname = "_".join([load_kw,model_kw,fit_kw])


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

    if ctx['modelspec'].meta.get('engine', 'nems0') == 'nems-lite':
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

#raise ValueError('stopping before postprocessing step')
for xfa in xfspec:
    if 'postprocess' in xfa[0]:
        log.info(f'Running postprocessing kw: {xfa[0]}')
        ctx = xforms.evaluate_step(xfa, ctx)
    """
    from nems_lbhb import postprocessing
    import importlib
    importlib.reload(postprocessing)
    newctx = postprocessing.dstrf_pca(**ctx)
    """

log.info('test fit complete')
raise ValueError('stopping before old code')



savefile =fit_model_xform(siteid, batch, modelname, saveInDB=True)


#siteid='SLJ029a'

rec = free_model.load_free_data(siteid, cellid=None, batch=batch, rasterfs=rasterfs,
                                recache=False, dlc_chans=8, dlc_threshold=0.2, compute_position=True)['rec']

rec['dlcsh'] = rec['dlc'].shuffle_time(rand_seed=1000)

ctx = free_model.free_split_rec(rec, apply_hrtf=True)

rec = ctx['rec']
est = ctx['est'].apply_mask()
val = ctx['val'].apply_mask()

print(rec.signals.keys())

dlc_sig = 'dlc'   # 'dlcsh'
stim0 = est['stim'].as_continuous().T
stim = np.concatenate([(stim0[:,:18]+stim0[:,18:])/2, (stim0[:,:18]-stim0[:,18:])],axis=1)
input = {'input': stim, 'dlc': est[dlc_sig].as_continuous().T}

target = est['resp'].rasterize().as_continuous().T

T = target.shape[0]
cellcount=target.shape[1]
input_count=input['input'].shape[1]
dlc_count=input['dlc'].shape[1]
dlc1 = 40
strf_channels=20
print(T,cellcount,input_count,dlc_count)

dlc_memory=4
acount=20
dcount=10
l2count=30
tcount=acount+dcount

exclude_dlc=False
newkw=False
oldkw=True
if exclude_dlc:
    layers = [
        WeightChannels(shape=(input_count, 1, 20), input='input', output='prediction'),
        FIR(shape=(15, 1, 20), input='prediction', output='prediction'),
        RectifiedLinear(shape=(1, 20), input='prediction', output='prediction',
                        no_offset=False, no_shift=False),
        WeightChannels(shape=(20, cellcount), input='prediction', output='prediction'),
        DoubleExponential(shape=(1, cellcount), input='prediction', output='prediction'),
    ]
    model0 = Model(layers=layers)

elif newkw:
    hrtf_kw = f'wcdl.{dlc_count}x{dlc1}.i-relud.{dlc1}.o.s-wcdl.{dlc1}x10-relud.10.o.s-wcdl.10x5-relud.5.o.s-wcdl.5x{input_count}-sigd.{input_count}.s.g-mult'
    #aud_kw = f'wch.{input_count}x1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}.o.s-wc.{strf_channels}x1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}.o.s-wc.{strf_channels}x{cellcount}-dexp.{cellcount}'
    aud_kw = f'wch.{input_count}x1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}.o.s-wc.{strf_channels}x1x{l2count}-fir.10x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}x{cellcount}-relu.{cellcount}.o.s'

    model0 = Model.from_keywords(hrtf_kw + '-' + aud_kw)

elif oldkw:
    sep_kw = f'wcst.{input_count}x1x{acount}.i-wcdl.{dlc_count}x1x{dcount}.i-first.8x1x{acount}-firdl.{dlc_memory}x1x{dcount}-cat-relu.{tcount}.o.s'
    aud_kw = f'wc.{tcount}x1x{l2count}-fir.4x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}x{cellcount}-relu.{cellcount}.o.s'
    model0 = Model.from_keywords(sep_kw + '-' + aud_kw)

else:
    layers = [
        WeightChannels(shape=(dlc_count, dlc1), input='dlc', output='hrtf'),
        RectifiedLinear(shape=(1, dlc1), input='hrtf', output='hrtf',
                        no_offset=False, no_shift=False),
        WeightChannels(shape=(dlc1, 10), input='hrtf', output='hrtf'),
        RectifiedLinear(shape=(1, 10), input='hrtf', output='hrtf',
                        no_offset=False, no_shift=False),
        WeightChannels(shape=(10, 5), input='hrtf', output='hrtf'),
        RectifiedLinear(shape=(1, 5), input='hrtf', output='hrtf1',
                        no_offset=False, no_shift=False),
        #WeightGaussianExpand(shape=(5, input_count), input='hrtf1', output='hrtf'),
        WeightChannels(shape=(5, input_count), input='hrtf1', output='hrtf'),
        Sigmoid(shape=(1, input_count), input='hrtf', output='hrtf',
                        no_shift=False, no_gain=False),
        # DoubleExponential(shape=(1, input_count), input='hrtf', output='hrtf'),
        # RectifiedLinear(shape=(1, input_count), input='hrtf', output='hrtf'),
        MultiplySignals(input=['input','hrtf'], output='hstim'),
        #MultiplyByExp(input=['stim','hrtf'], output='hstim'),
        WeightChannels(shape=(input_count, 1, strf_channels), input='hstim', output='prediction'),
        FIR(shape=(10, 1, strf_channels), input='prediction', output='prediction'),
        RectifiedLinear(shape=(1, strf_channels), input='prediction', output='prediction',
                        no_offset=False, no_shift=False),
        WeightChannels(shape=(strf_channels, 1, strf_channels), input='prediction', output='prediction'),
        FIR(shape=(10, 1, strf_channels), input='prediction', output='prediction'),
        RectifiedLinear(shape=(1, strf_channels), input='prediction', output='prediction',
                        no_offset=False, no_shift=False),
        WeightChannels(shape=(strf_channels, cellcount), input='prediction', output='prediction'),
        DoubleExponential(shape=(1, cellcount), input='prediction', output='prediction'),
    ]

    model0 = Model(layers=layers)

model0 = model0.sample_from_priors()
model0 = model0.sample_from_priors()

fitter = 'tf'
cost_function='nmse'
if fitter == 'scipy':
    fitter_options = {'cost_function': 'nmse', 'options': {'ftol':  1e-4, 'gtol': 1e-4, 'maxiter': 100}}
else:
    fitter_options = {'cost_function': cost_function,  # 'nmse'
                      'early_stopping_tolerance': 1e-3,
                      'validation_split': 0,
                      'learning_rate': 1e-2, 'epochs': 3000
                      }
    fitter_options2 = {'cost_function': cost_function,
                      'early_stopping_tolerance': 5e-4,
                      'validation_split': 0,
                      'learning_rate': 1e-3, 'epochs': 8000
                      }

#model = model0.fit(input=input, target=target,
#                  backend=fitter, fitter_options=fitter_options)
print('Fit stage 1: without static output nonlinearity')
model0.layers[-1].skip_nonlinearity()
model = model0.fit(input=input, target=target, backend=fitter,
                  fitter_options=fitter_options)
model.layers[-1].unskip_nonlinearity()
print('Fit stage 2: with static output nonlinearity')
model = model.fit(input=input, target=target, backend=fitter,
                  verbose=0, fitter_options=fitter_options2)

prediction = model.predict(input)
if type(prediction) is dict:
    hrtf = prediction['hrtf']
    if 'hstim' in prediction.keys():
        hstim = prediction['hstim']
    else:
        hstim = input['input']
    prediction = prediction['output']

cellcount = target.shape[1]
cc = np.array([np.corrcoef(prediction[:, i], target[:, i])[0, 1] for i in range(cellcount)])

i = 6

plt.figure(figsize=(6, 6))
N1 = 1500
N2 = 2500
ax = plt.subplot(6, 1, 2)
stim1 = np.concatenate([input['input'][:, :18] * 2 + input['input'][:, 18:], input['input'][:, :18] * 2 - input['input'][:, 18:]], axis=1)
im = ax.imshow(stim1[N1:N2, :].T)
plt.colorbar(im)

ax = plt.subplot(6, 2, 1)
ax.plot(cc)
ax.set_title(f'mean cc={cc.mean():.2f}')

if len(model.layers) > 8:
    ax = plt.subplot(6, 2, 2)
    nplt.plot_strf(model.layers[-4], model.layers[-5], ax=ax, fs=rasterfs);

    ax = plt.subplot(6, 1, 3)
    im = ax.imshow(input['dlc'][N1:N2, :].T)
    plt.colorbar(im)

    ax = plt.subplot(6, 1, 4)
    im = ax.imshow(hrtf[N1:N2, :].T)
    plt.colorbar(im)

    ax = plt.subplot(6, 1, 5)
    hstim1=np.concatenate([hstim[:,:18]*2+hstim[:,18:], hstim[:,:18]*2-hstim[:,18:]], axis=1)
    im = ax.imshow(hstim1[N1:N2, :].T)
    plt.colorbar(im)

ax = plt.subplot(6, 1, 6)
ax.plot(smooth(prediction[N1:N2, i]))
ax.plot(smooth(target[N1:N2, i]))
ax.set_title(f"pred cc={cc[i]:.3f}")
plt.tight_layout()


