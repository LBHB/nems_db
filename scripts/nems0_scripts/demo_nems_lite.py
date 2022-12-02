# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import logging
import pickle
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nems0.analysis.api
import nems0.initializers
import nems0.preprocessing as preproc
import nems0.uri
from nems0 import db
from nems0 import xforms
from nems0 import recording
from nems0.fitters.api import scipy_minimize
from nems0.signal import RasterizedSignal
from nems import Model, visualization

log = logging.getLogger(__name__)

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems0.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems0.NEMS_PATH) / 'modelspecs'


# LOAD AND FORMAT RECORDING DATA
# X (stimulus) is a Frequency X Time matrix, sampled at a rate fs
# Y (response) is a Neuron X Time matrix, also sampled at fs. In this demo,
#   we're analyzing a single neuron, so Y is 1 x T

# this section illustrates several alternative methods for loading,
# each loading from a different file format
load_method = 1  # Traditional TAR010c NAT
#load_method = 3  # binaural NAT

if load_method==0:
    # download demo data
    recording.get_demo_recordings(signals_dir)
    
    # method 0: load NEMS native recording file
    datafile = signals_dir / 'TAR010c.NAT.fs100.ch18.tgz'
    cellid='TAR010c-18-2'
    rec = recording.load_recording(datafile)
    rec['resp']=rec['resp'].extract_channels([cellid])
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex="^STIM_")
    est=preproc.average_away_epoch_occurrences(est, epoch_regex="^STIM_")
    val=preproc.average_away_epoch_occurrences(val, epoch_regex="^STIM_")
    modelspec = 'wc.18x1-fir.15x1-dexp.1'

elif load_method==1:
    # download demo data
    print(f"loadmethod={load_method} : TAR010c-18-1.pkl")
    recording.get_demo_recordings(signals_dir)

    # method 1: load from a pkl datafile that contains full stim+response data
    # along with metadata (fs, stimulus epoch list)
    datafile = signals_dir / 'TAR010c-18-1.pkl'

    with open(datafile, 'rb') as f:
            #cellid, recname, fs, X, Y, X_val, Y_val = pickle.load(f)
            cellid, recname, fs, X, Y, epochs = pickle.load(f)

    # create NEMS-format recording objects from the raw data
    resp = RasterizedSignal(fs, Y, 'resp', recname, chans=[cellid], epochs=epochs)
    stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs)

    # create the recording object from the signals
    signals = {'resp': resp, 'stim': stim}
    rec = recording.Recording(signals)
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex="^STIM_")
    est=preproc.average_away_epoch_occurrences(est, epoch_regex="^STIM_")
    val=preproc.average_away_epoch_occurrences(val, epoch_regex="^STIM_")
    X_est=est['stim'].as_continuous().copy().T
    Y_est=est['resp'].as_continuous().T
    X_val=val['stim'].as_continuous().copy().T
    Y_val=val['resp'].as_continuous().T

    if 1:
        # save to pre-processed pickle:

        np.savez('/tmp/TAR010c-18-2.npz', X_est=X_est, Y_est=Y_est,
                 X_val=X_val, Y_val=Y_val)
        d = np.load('/tmp/TAR010c-18-2.npz')

    X_norm = X_est.max()
    X_est /= X_norm
    X_val /= X_norm
    
    #model = Model.from_keywords('wc.18x1.g-fir.15x1-dexp.1')
    modelspec = 'wc.18x1-fir.15x1-stp.1x1-dexp.1'
    modelspec = 'wc.18x1x2.g-fir.15x1x2-wc.2x1-dexp.1'
    #modelspec = 'wc.18x2.g-fir.15x2-dexp.1'

    #modelspec = 'wc.18x1-fir.15x1.s2-dexp.1'
    #Y_est = Y_est[::2,:]
    #Y_val = Y_val[::2,:]
elif load_method==2:
    # download demo data
    recording.get_demo_recordings(signals_dir)

    # method 2: load from CSV files - one per response, stimulus, epochs
    # X is a frequency X time spectrgram, sampled at 100 Hz
    # Y is a neuron X time PSTH, aligned with X. Ie, same number of time bins
    # epochs is a list of STIM events with start and stop time of each event
    # in seconds
    # The data have already been averaged across repeats, and the first three
    # stimuli were repeated ~20 times. They will be broken out into the
    # validation recording, used to evaluate model performance. The remaining
    # 90 stimuli will be used for estimation.
    fs=100
    cellid='TAR010c-18-2'
    recname='TAR010c'
    stimfile = signals_dir / 'TAR010c-NAT-stim.csv.gz'
    respfile = signals_dir / 'TAR010c-NAT-resp.csv.gz'
    epochsfile = signals_dir / 'TAR010c-NAT-epochs.csv'

    X=np.loadtxt(gzip.open(stimfile, mode='rb'), delimiter=",", skiprows=0)
    Y=np.loadtxt(gzip.open(respfile, mode='rb'), delimiter=",", skiprows=0)
    # get list of stimuli with start and stop times (in sec)
    epochs = pd.read_csv(epochsfile)

    val_split = 550*3 # validation data are the first 3 5.5 sec stimuli
    resp_chan = 11  # 11th cell is TAR010c-18-2
    X_val = X[:, :val_split]
    X_est = X[:, val_split:]
    epochs_val = epochs.loc[:2]
    epochs_est = epochs.loc[3:]
    Y_val = Y[[resp_chan], :val_split]
    Y_est = Y[[resp_chan], val_split:]

    # create NEMS-format recording objects from the raw data
    resp = RasterizedSignal(fs, Y_est, 'resp', recname, chans=[cellid], epochs=epochs_est)
    stim = RasterizedSignal(fs, X_est, 'stim', recname, epochs=epochs_est)

    # create the recording object from the signals
    signals = {'resp': resp, 'stim': stim}
    est = recording.Recording(signals)

    val_signals = {
            'resp': RasterizedSignal(fs, Y_val, 'resp', recname, chans=[cellid], epochs=epochs_val),
            'stim': RasterizedSignal(fs, X_val, 'stim', recname, epochs=epochs_val)}
    val = recording.Recording(val_signals)
    modelspec = 'wc.18x2-fir.15x2-dexp.1'
elif load_method == 3:

    # testing binaural NAT with various model architectures.
    batch=338
    modelname="gtgram.fs100.ch18.mono-ld.pop-norm.l1-sev_wc.Nx60-fir.1x20x60-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
    recording_uri = '/auto/data/nems_db/recordings/338/CLT041c_c6cec16ea49389fffff3880d5742bfc3a3cf4924.tgz'
    rec = recording.load_recording(recording_uri)

    cellid="CLT041c-054-2"
    cellids = ['CLT041c-054-1', 'CLT041c-054-2']
    cellids = ['CLT041c-054-2']
    cellids = [rec['resp'].chans[19]]
    cellids = [rec['resp'].chans[6]]
    print("cellids: ", cellids)

    ctx = {'rec': rec}
    ctx.update(xforms.normalize_sig(sig='resp', norm_method='minmax', **ctx))
    ctx.update(xforms.normalize_sig(sig='stim', norm_method='minmax', **ctx))
    ctx.update(xforms.split_by_occurrence_counts(epoch_regex='^STIM', **ctx))
    ctx.update(xforms.average_away_stim_occurrences(epoch_regex='^STIM', **ctx))
    X_est = ctx['est']['stim'].as_continuous().T.copy()
    Y_est = ctx['est']['resp'].extract_channels(cellids).as_continuous().T
    X_val = ctx['val']['stim'].as_continuous().T.copy()
    Y_val = ctx['val']['resp'].extract_channels(cellids).as_continuous().T

    X_norm = X_est.max()
    X_est /= X_norm
    X_val /= X_norm

    if True:
        X_est=np.swapaxes(np.reshape(X_est, [-1, 2, 18]), 2, 1)
        X_val=np.swapaxes(np.reshape(X_val, [-1, 2, 18]), 2, 1)

        # shorten est data for speed
        X_est = X_est[:50000, :, :]
        Y_est = Y_est[:50000, :]

        modelspec = 'wcb.18x3x1-fir.15x3x1-wc.2x1-dexp.1'
        modelspec = 'wcb.18x2x2-fir.15x2x2-wc.2x1-relu.1.o.s'
    else:
        modelspec = 'wc.36x2-fir.15x2-dexp.1'

        # shorten est data for speed
        X_est = X_est[:50000, :]
        Y_est = Y_est[:50000, :]

model = Model.from_keywords(modelspec)
model.set_dtype(np.float64)

# Set initial values
# quick & dirty, but may not work as desired
model.sample_from_priors()
model.sample_from_priors()

# for wc.g models
#model.layers[0]['mean'] = [0.3, 0.7] # [0.5] #
#model.layers[0]['sd'] = [0.4, 0.4] # [0.5] #

# wc = model.layers[0]['coefficients'].values
# wc = np.random.randn(*wc.shape) / 10
# model.layers[0]['coefficients'] = wc
# fir = model.layers[1]['coefficients'].values
# fir[1,:]=0.5
# fir[2,:]=-0.25
# model.layers[1]['coefficients'] = fir
#
# if len(model)>3:
#     wc = model.layers[2]['coefficients'].values
#     wc = np.random.randn(*wc.shape) / 10
#     model.layers[2]['coefficients'] = wc
#
# model.layers[-1]['shift'] = [0.] * len(cellids)
# model.layers[-1]['kappa'] = [1.] * len(cellids)
# model.layers[-1]['amplitude'] = [1.] * len(cellids)
# model.layers[-1]['base'] = [0.] * len(cellids)
#print(model)

# By default, `scipy.optimize.minimize` will be used for optimization
# (which can also be specified using the `backend` parameter). This also tells
# the model to use each layer's standard `evaluate` method for transforming
# inputs (whereas `backend='tf'`, for example, would use `Layer.tf_layer`).
# See `nems.models.base.Model.fit` for additional fitting options.

backend='tf'
split_batches = True
if backend=='tf':
    fitter_options = {'cost_function': 'squared_error', 'early_stopping_delay': 50, 'early_stopping_patience': 10,
                  'early_stopping_tolerance': 1e-3, 'validation_split': 0.2,
                  'learning_rate': 5e-3, 'epochs': 2000}

    if split_batches:
        est_n = 90
        x_bins_per_trial = int(X_est.shape[0]/est_n)
        y_bins_per_trial = int(Y_est.shape[0]/est_n)
        y = np.reshape(Y_est, [est_n, y_bins_per_trial, -1])
        x = np.reshape(X_est, [est_n, x_bins_per_trial, -1])
    else:
        x = np.expand_dims(X_est, axis=0)
        y = np.expand_dims(Y_est, axis=0)

    # Trying a TF fit:
    model.layers[-1].skip_nonlinearity()
    model = model.fit(input=x, target=y, backend='tf',
              fitter_options=fitter_options, batch_size=None)

    model.layers[-1].unskip_nonlinearity()
    model = model.fit(input=x, target=y, backend='tf',
              fitter_options=fitter_options, batch_size=None)

else:
    print('Fitting without NL ...')
    model.layers[-1].skip_nonlinearity()
    from nems.layers import ShortTermPlasticity
    for i, l in enumerate(model.layers):
        if isinstance(l, ShortTermPlasticity):
            log.info(f'Freezing parameters for layer {i}: {l.name}')
            model.layers[i].freeze_parameters()
    tolerance = 1e-5
    model = model.fit(input=X_est.astype(np.float32), target=Y_est.astype(np.float32), backend='scipy',
              fitter_options={'cost_function': 'nmse', 'options': {'ftol': tolerance, 'maxiter': 100}})
    model = model.fit(input=X_est, target=Y_est, backend='scipy',
              fitter_options={'cost_function': 'nmse', 'options': {'ftol': tolerance, 'maxiter': 100}})

    print('Now fitting with NL ...')
    model.layers[-1].unskip_nonlinearity()
    for i, l in enumerate(model.layers):
        model.layers[i].unfreeze_parameters()
    tolerance = 1e-6

    model = model.fit(input=X_est, target=Y_est, backend='scipy',
              fitter_options={'cost_function': 'nmse', 'options': {'ftol': tolerance, 'maxiter': 100}})

visualization.model.plot_model_with_parameters(model, X_est, target=Y_est)
visualization.model.plot_model_with_parameters(model, X_val, target=Y_val)

# Predict the response to the stimulus spectrogram using the fitted model.
d1=model.evaluate(X_val, n=1)['_last_output']
d2=model.evaluate(X_val, n=2)['_last_output']
plt.figure()
plt.plot(d1)
plt.plot(d2)

prediction = model.predict(X_val)
cc = np.corrcoef(prediction[:, 0], Y_val[:, 0])[0, 1]
print("pred xc:", cc)

"""
wc = model.layers[0].coefficients
fir = model.layers[1].coefficients

if len(wc.shape)>2:
    strf=wc[:,:,0] @ fir[:,:,0].T
    if wc.shape[2]>1:
        strf2=wc[:,:,1] @ fir[:,:,1].T
    else:
        strf2=strf
elif len(fir.shape)>2:
    strf=wc @ fir[:,:,0].T
    if fir.shape[2]>1:
        strf2=wc @ fir[:,:,1].T
    else:
        strf2 = strf
else:
    strf=wc @ fir.T
    strf2 = strf
strf *= model.layers[2].coefficients[0]
strf2 *= model.layers[2].coefficients[1]
cmax=np.max([np.abs(strf).max(), np.abs(strf2).max()])

fig=plt.figure()
spec = fig.add_gridspec(3, 4)

ax0 = fig.add_subplot(spec[0, :])
ax1 = fig.add_subplot(spec[1, :])
ax20 = fig.add_subplot(spec[2, 0])
ax21 = fig.add_subplot(spec[2, 1])
ax22 = fig.add_subplot(spec[2, 2:])

bins=550
fs=100
dur=bins/fs
chans = X_val.shape[1]
t=np.linspace(0,dur,bins)
_x = np.reshape(np.transpose(X_val[:bins,:,:],[0,2,1]),(bins,-1)).T
ax0.imshow(_x, aspect='auto', extent=[0,dur,-0.5,chans+0.5])
ax0.set_title(f"cc={cc:.3f}")
ax1.plot(t,Y_val[:bins,0])
ax1.plot(t,prediction[:bins,0])
ax1.set_xlim(ax0.get_xlim())
ax20.imshow(strf, origin='lower', extent=[0,strf.shape[1],-0.5,chans+0.5], clim=[-cmax, cmax])
ax21.imshow(strf2, origin='lower', extent=[0,strf.shape[1],-0.5,chans+0.5], clim=[-cmax, cmax])

xin = model.evaluate(input=X_val, n=len(model)-2)['_last_output']
xrange = np.linspace(xin.min(),xin.max(),100)[:,np.newaxis]
xout = model.layers[-1].evaluate(xrange)
ax22.plot(xin[:,0],Y_val[:,0],'.', markersize=2, color='lightgray')
ax22.plot(xrange[:,0],xout[:,0], color='k')


x1 = model.evaluate(input=X_val, n=0)['_last_output']
x2 = model.evaluate(input=X_val, n=1)['_last_output']
x3 = model.evaluate(input=X_val, n=2)['_last_output']
x4 = model.evaluate(input=X_val, n=3)['output']

f,ax = plt.subplots(6,1, sharex=True)

ax[0].imshow(X_val[:1000,:,0].T, aspect='auto', interpolation='none')
ax[0].set_ylabel('stim 0')

ax[1].imshow(X_val[:1000,:,1].T, aspect='auto', interpolation='none')
ax[1].set_ylabel('stim 1')

ax[2].plot(x1[:1000,:,0])
ax[2].plot(x1[:1000,:,1], '--')
ax[2].set_ylabel('wc1 output')

ax[3].plot(x2[:1000,0])
ax[3].plot(x2[:1000,1], '--')
ax[3].set_ylabel('fir output')

ax[4].plot(x3[:1000,:])
ax[4].set_ylabel('wc1 output')

ax[5].plot(x4[:1000,:])
ax[5].plot(Y_val[:1000,0])
ax[5].set_ylabel('dexp output')

"""