from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

from nems0 import db, preprocessing
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.motor.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist


from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential, LevelShift, ReLU
from nems.layers.base import Layer, Phi, Parameter

USE_DB = True

if USE_DB:
    siteid = "PRN034a"
    siteid = "PRN010a"
    siteid = "PRN009a"
    siteid = "PRN011a"
    siteid = "PRN022a"
    siteid = "PRN047a"
    siteid = "PRN015a"
    runclassid = 132

    sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
    dparm = db.pd_query(sql)
    parmfile = [r.stimpath+r.stimfile for i,r in dparm.iterrows()]
    cellids=None
else:
    parmfile = ["/auto/data/daq/Prince/PRN015/PRN015a01_a_NTD",
                "/auto/data/daq/Prince/PRN015/PRN015a02_a_NTD"]
    cellids = None

## load the recording

rasterfs = 50

ex = BAPHYExperiment(parmfile=parmfile, cellid=cellids)
print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

recache = False

# load recording
rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
                       dlc=True, recache=recache, rasterfs=rasterfs,
                       dlc_threshold=0.2, fill_invalid='interpolate')

# generate 'dist' signal from dlc signal
rec = dlc2dist(rec, ref_x0y0=None, smooth_win=5, norm=False, verbose=False)

# compute PSTH for repeated stimuli
epoch_regex = "^STIM_"
rec['resp'] = rec['resp'].rasterize()
rec['psth'] = preprocessing.generate_average_sig(rec['resp'], 'psth', epoch_regex=epoch_regex)

# use diff to predict dist
rec['diff'] = rec['resp']._modified_copy(data = rec['resp']._data-rec['psth']._data)


epochs = rec.epochs
is_stim = epochs['name'].str.startswith('STIM')
stim_epoch_list = epochs[is_stim]['name'].tolist()
stim_epochs = list(set(stim_epoch_list))
stim_epochs.sort()

cellids = rec['resp'].chans

est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)

e = stim_epochs[0]
p = val['psth'].extract_epoch(e)
r = val['resp'].extract_epoch(e)
d = val['diff'].extract_epoch(e)

f,ax=plt.subplots(3,1)
cid=96
ax[0].imshow(p[:,cid,:], aspect='auto', interpolation='none')
ax[1].imshow(r[:,cid,:], aspect='auto', interpolation='none')
ax[2].imshow(d[:,cid,:], aspect='auto', interpolation='none')

ax[0].set_title(f'psth {cellids[cid]} - {e}')
ax[1].set_title(f'single trial {cellids[cid]} - {e}')
ax[2].set_title(f'diff {cellids[cid]} - {e}')
plt.tight_layout()


# For a model that uses multiple inputs, we package the input data into
# a dictionary. This way, layers can specify which inputs they need using the
# assigned keys.

val = val.apply_mask()

# change 'diff' to 'resp' for non-PSTH subbed data
input = val['diff']._data.T

# subset of cells:
input = input[:,50:]
targetchan=4

target = val['dist']._data.T

good_timebins = (np.isnan(target).sum(axis=1) == 0)
input = input[good_timebins,:]
target = target[good_timebins,targetchan][:,np.newaxis]

target = target-target.mean(axis=0, keepdims=True)
target = target/target.std(axis=0, keepdims=True)


cellcount = input.shape[1]
dimcount = target.shape[1]


layers = [
    WeightChannels(shape=(cellcount,3)),
    FIR(shape=(2,3)),
    ReLU(shape=(3,)),
    WeightChannels(shape=(3,1)),
    LevelShift(shape=(dimcount,))
]
#DoubleExponential(shape=(dimcount,))

model = Model(layers=layers)
model = model.sample_from_priors()

tolerance = 1e-5
max_iter = 200

use_tf = True
if use_tf:
    input = np.expand_dims(input, axis=0)
    target = np.expand_dims(target, axis=0)

    fitter_options = {'cost_function': 'nmse', 'early_stopping_delay': 10,
                      'early_stopping_patience': 5,
                      'early_stopping_tolerance': tolerance,
                      'learning_rate': 1e-2, 'epochs': max_iter,
                      }
    model = model.fit(input=input, target=target, backend='tf',
                      fitter_options=fitter_options, batch_size=None)

    prediction = model.predict(input, batch_size=None)[0,:,:]

    f,ax=plt.subplots(1,1)
    ax.scatter(prediction,target,s=1)
    cc = np.corrcoef(prediction[:,0],target[0,:,0])[0,1]
    ax.set_title(f'targetchan={rec["dist"].chans[targetchan]}, cc={cc:.3f}')

else:
    fitter_options = {'cost_function': 'nmse', 'options': {'ftol': tolerance, 'gtol': tolerance/10, 'maxiter': max_iter}}

    #model.layers[-1].skip_nonlinearity()
    #model=model.fit(input=input, target=target, fitter_options=fitter_options)

    #model.layers[-1].unskip_nonlinearity()

    model =model.fit(input=input, target=target, fitter_options=fitter_options)

    prediction = model.predict(input)

    f,ax=plt.subplots(1,1)
    ax.scatter(prediction,target,s=1)
    cc = np.corrcoef(prediction[:,0],target[:,0])[0,1]
    ax.set_title(f'targetchan={rec["dist"].chans[targetchan]}, cc={cc:.3f}')



