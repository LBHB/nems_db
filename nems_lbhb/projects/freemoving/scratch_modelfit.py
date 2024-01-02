from os.path import basename, join
import logging
import os
import io

from nems0.utils import escaped_split, escaped_join, get_setting
from nems0.registry import KeywordRegistry, xforms_lib
from nems0 import xform_helper, xforms, db

log = logging.getLogger(__name__)

from nems0.registry import xform, scan_for_kw_defs
from nems.layers.tools import require_shape, pop_shape

siteid = 'PRN048a'
#siteid='PRN009a'
siteid = 'SLJ033a'
cellid = siteid
rasterfs = 50
batch = 348

dlc_count=12
dlc1 = 40
strf_channels=20

dlc_memory=3
acount=25
dcount=12
l2count=30
tcount=acount+dcount
input_count = 36

# allow .o.s in intermediate relus
#sep_kw = f'wcst.Nx1x{acount}.i-wcdl.{dlc_count}x1x{dcount}.i-first.8x1x{acount}-firdl.{dlc_memory}x1x{dcount}-cat-relu.{tcount}.o.s'
#aud_kw = f'wc.{tcount}x1x{l2count}-fir.4x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}xR-relu.R.o.s'
ros="" # ros=".o.s" # ros=""
reg=".l2:4"
#reg=""
sep_kw = f'wcst.Nx1x{acount}.i{reg}-wcdl.{dlc_count}x1x{dcount}.i{reg}-first.8x1x{acount}-firdl.{dlc_memory}x1x{dcount}.nc1-cat-relu.{tcount}{ros}'
aud_kw = f'wc.{tcount}x1x{l2count}{reg}-fir.4x1x{l2count}-relu.{l2count}{ros}-wc.{l2count}xR{reg}-relu.R.o.s'
model_kw_old = sep_kw + '-' + aud_kw

# allow .o.s in intermediate relus
#hrtf_kw = f'wcdl.{dlc_count}x{dlc1}.i-relud.{dlc1}.o.s-wcdl.{dlc1}x10-relud.10.o.s-wcdl.10x5-relud.5.o.s-wcdl.5x{input_count}-sigd.{input_count}.s.g-mult'
#aud_kw = f'wch.Nx1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}.o.s-wc.{strf_channels}x1x{l2count}-fir.10x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}xR-relu.R.o.s'
hrtf_kw = f'wcdl.{dlc_count}x{dlc1}.i-relud.{dlc1}{ros}-wcdl.{dlc1}x10-relud.10.o.s-wcdl.10x5-relud.5.o.s-wcdl.5x{input_count}-sigd.{input_count}.s.g-mult'
aud_kw = f'wch.Nx1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}-wc.{strf_channels}x1x{l2count}-fir.10x1x{l2count}-relu.{l2count}-wc.{l2count}xR-relu.R.o.s'
model_kw_new = hrtf_kw + '-' + aud_kw

model_kw_ln = f'wc.Nx1x{l2count}-fir.10x1x{l2count}-wc.{l2count}xR-relu.R.o.s'

# dlc effects from stategaindl (normal wc keywords for stim-->pred, wcdl/-s words to handle the dlc-->state path)
sep_kw = f'wcdl.{dlc_count}x1x{dcount}.i.s.l2-firs.{dlc_memory}x1x{dcount}.nc1-relus.{dcount}{ros}-wcs.{dcount}x{dcount}.l2-relus.{dcount}{ros}'
aud_kw = f'wc.Nx1x{acount}.i.l2-fir.8x1x{acount}-relu.{acount}{ros}-wc.{acount}x1x{l2count}.l2-fir.4x1x{l2count}-relu.{l2count}{ros}-wc.{l2count}xR.l2-stategain.{dcount+1}xR-relu.R.o.s'
model_kw_sg = sep_kw + '-' + aud_kw


load_kw_shuff = f"free.fs{rasterfs}.ch18-norm.l1-fev-shuf.dlc"
load_kw = f"free.fs{rasterfs}.ch18-norm.l1-fev"
load_kw_hrtf = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf"
load_kw_hrtf_shuff = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf-shuf.dlc"

jkn=6
load_kw_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.jk{jkn}"
load_kw_hrtf_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf.jk{jkn}"
load_kw_hrtf_shuff_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf.jk{jkn}-shuf.dlc"

rbn=3
fit_kw = "lite.tf.cont.init.lr1e3.t3-lite.tf.cont.lr1e4"
fit_kw_jk = f"lite.tf.cont.init.lr1e3.t3.rb{rbn}-lite.tf.cont.lr1e4.t5e4"  # .t3"

modelnames=[
#    "_".join([load_kw,model_kw_new,fit_kw]),
#    "_".join([load_kw_shuff, model_kw_new, fit_kw]),
#    "_".join([load_kw_jk, model_kw_new, fit_kw_jk]),
    "_".join([load_kw_hrtf_jk, model_kw_ln, fit_kw_jk]),
#    "_".join([load_kw,model_kw_old,fit_kw]),
#    "_".join([load_kw_shuff, model_kw_old, fit_kw]),
    "_".join([load_kw_hrtf_jk, model_kw_old, fit_kw_jk]),
    "_".join([load_kw_hrtf_shuff_jk, model_kw_old, fit_kw_jk]),
    "_".join([load_kw_hrtf_jk, model_kw_sg, fit_kw_jk]),
    "_".join([load_kw_hrtf_shuff_jk, model_kw_sg, fit_kw_jk]),
]
shortnames = [
    #'HRTF+DLC new nojk',
    #'HRTF+DLC new',
    #'HRTF+Dsh new',
    'HRTF+LN',
    #'HRTF+DLC old nojk',
    #'HRTF+Dsh old nojk',
    'HRTF+DLC old',
    'HRTF+Dsh old',
    'HRTF+DLC sg',
    'HRTF+Dsh sg',
]

modelname = modelnames[2]
modelname2 = modelnames[2]
for i,m in enumerate(modelnames):
    if m==modelname2:
        print(f'**{i:2d} {shortnames[i]:12s}  {m}')
    elif m==modelname:
        print(f'* {i:2d} {shortnames[i]:12s}  {m}')
    else:
        print(f'  {i:2d} {shortnames[i]:12s}  {m}')

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

raise ValueError('stopping')

from nems.visualization import model
import importlib
importlib.reload(model)

X_val, Y_val = xforms.lite_input_dict(ctx['modelspec'], ctx['val'], epoch_name="")

fig = model.plot_model(
    ctx['modelspec'], X_val, target=Y_val, sampling_rate=ctx['val']['resp'].fs)


from nems0 import epoch as ep
import matplotlib.pyplot as plt
resp=ctx['rec']['resp']
epochs = ep.epoch_names_matching(resp.epochs, "^TAR")
i=resp.get_epoch_bounds('TARGET')
i[:,0]-=0.5
r=resp.extract_epoch(i)
plt.figure()
plt.imshow(r.mean(axis=0))


from nems_lbhb.projects.freemoving import free_model, free_vs_fixed_strfs
import importlib
import numpy as np

importlib.reload(free_vs_fixed_strfs)

rec=ctx['val'].apply_mask()
modelspec_list=ctx['modelspec_list']

model_list = ctx['modelspec_list']
time_step=85
D=15
pc_count=5
reset_backend=False

pc_mags = []
mdstrfs = []
for model in modelspec_list:
    _pc_mags = []
    _mdstrfs = []
    for out_channel in range(rec['resp'].shape[0]):
        mdstrf, pc1, pc2, pc_mag = \
        free_vs_fixed_strfs.dstrf_snapshots(rec, [model], D=D, time_step=time_step, 
                                            out_channel=out_channel, pc_count=pc_count)
        _pc_mags.append(pc_mag)  # unit x model x didx x pc
        _mdstrfs.append(mdstrf)  # unit x model x didx x frequency x lag
    pc_mags.append(_pc_mags)
    mdstrfs.append(_mdstrfs)

"""
t_indexes = np.arange(time_step, rec['stim'].shape[1], time_step)
dlc = rec['dlc'].as_continuous().T
log.info(f"Computing dSTRF at {len(t_indexes)} timepoints, {dlc.shape[1]} DLC channels, t_step={time_step}")
if rec.meta['batch'] in [346, 347]:
    dicount = didx.shape[0]
else:
    dicount = 4

dstrf = {}
mdstrf = np.zeros((len(model_list), dicount, rec['stim'].shape[0], D))
pc1 = np.zeros((len(model_list), dicount, rec['stim'].shape[0], D))
pc2 = np.zeros((len(model_list), dicount, rec['stim'].shape[0], D))
pc_mag_all = np.zeros((len(model_list), dicount, pc_count))

for di in range(dicount):
    dlc1 = dlc.copy()
    dcount = dlc1.shape[1]
    didx_ = free_vs_fixed_strfs.adjust_didx(dlc, free_vs_fixed_strfs.didx)
    # didx_ = didx

    for t in t_indexes:
        dlc1[(t - didx_.shape[1] + 1):(t + 1), :] = didx_[di, :, :dcount]
    log.info(f"DLC values: {np.round(free_vs_fixed_strfs.didx[di, -1, :dcount], 3)}")
    # log.info(f'di={di} Applying HRTF for frozen DLC coordinates')
    # rec2 = rec.copy()
    # rec2['dlc'] = rec2['dlc']._modified_copy(data=dlc1.T)
    # rec2 = free_tools.stim_filt_hrtf(rec2, hrtf_format='az', smooth_win=2,
    #                                 f_min=200, f_max=20000, channels=18)['rec']

    for mi, m in enumerate(model_list):
        stim = {'input': rec['stim'].as_continuous().T, 'dlc': dlc1}
        dstrf[di] = m.dstrf(stim, D=D, out_channels=[out_channel], t_indexes=t_indexes, reset_backend=reset_backend)

        d = dstrf[di]['input'][0, :, :, :]

        if snr_threshold is not None:
            d = np.reshape(d, (d.shape[0], d.shape[1] * d.shape[2]))
            md = d.mean(axis=0, keepdims=True)
            e = np.std(d - md, axis=1) / np.std(md)
            if (e > snr_threshold).sum() > 0:
                log.info(f"Removed {(e > snr_threshold).sum()}/{len(d)} noisy dSTRFs for PCA calculation")

            d = dstrf[di]['input'][0, (e <= snr_threshold), :, :]
            dstrf[di]['input'] = d[np.newaxis, :, :, :]
        mdstrf[mi, di, :, :] = d.mean(axis=0)

        # svd attempting to make compatible with new format of compute_pcs
        try:
            if (d.size > 0) and (d.std() > 0):
                # pc, pc_mag = dtools.compute_dpcs(d[np.newaxis, :, :, :], pc_count=pc_count)
                dpc = dtools.compute_dpcs(dstrf[di], pc_count=pc_count)
                pc = dpc['input']['pcs']
                pc_mag = dpc['input']['pc_mag']

                pc1[mi, di, :, :] = pc[0, 0, :, :] * pc_mag[0, 0]
                pc2[mi, di, :, :] = pc[0, 1, :, :] * pc_mag[1, 0]
                pc_mag_all[mi, di, :] = pc_mag[:, 0]
        except:
            log.info('FAILED TO COMPUTE PCS. SETTING TO ZERO.')


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


"""