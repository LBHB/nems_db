from os.path import basename, join
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
    DoubleExponential, RectifiedLinear, ConcatSignals
from nems import Model
from nems.layers.base import Layer, Phi, Parameter
import nems.visualization.model as nplt
#import nems0.plots.api as nplt

cellids = ['PRN015a-317-1', 'PRN015a-318-1', 'PRN015a-348-1']
siteid = db.get_siteid(cellids[0])
runclassid = 132

sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
dparm = db.pd_query(sql)
parmfile = [r.stimpath+r.stimfile for i, r in dparm.iterrows()]

## load the recording
rasterfs = 50
ex = BAPHYExperiment(parmfile=parmfile)
print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

recache = False
rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
                       dlc=True, recache=recache, rasterfs=50,
                       dlc_threshold=0.2, fill_invalid='interpolate')

dlc_data = rec['dlc'][:,:]
rec = impute_multi(rec, sig='dlc', empty_values=None, keep_dims=8)['rec']

rec['dlcsh'] = rec['dlc'].shuffle_time(rand_seed=1000)
dlc_sig = 'dlc'
input = {'stim': rec['stim'].rasterize().as_continuous().T,
         'dlc': rec[dlc_sig].as_continuous().T}

cid = [i for i,c in enumerate(rec['resp'].chans) if c in cellids]
target = rec['resp'].rasterize().as_continuous()[cid, :].T

layers = [
    WeightChannels(shape=(18, 1, 3), input='stim', output='prediction'),
    FIR(shape=(15, 1, 3), input='prediction', output='prediction'),
    WeightChannels(shape=(8, 1, 2), input='dlc', output='space'),
    FIR(shape=(15, 1, 2), input='space', output='space'),
    ConcatSignals(input=['prediction','space'],output='prediction'),
    RectifiedLinear(shape=(1, 5), input='prediction', output='prediction',
                    no_offset=False, no_shift=False),
    WeightChannels(shape=(5, 3), input='prediction', output='prediction'),
    LevelShift(shape=(1, 3), input='prediction', output='prediction'),
]

model = Model(layers=layers)
model = model.sample_from_priors()


fitter = 'tf'
if fitter == 'scipy':
    fitter_options = {'cost_function': 'nmse', 'options': {'ftol':  1e-4, 'gtol': 1e-4, 'maxiter': 100}}
else:
    fitter_options = {'cost_function': 'nmse',
                      'early_stopping_delay': 5,
                      'early_stopping_patience': 10,
                      'early_stopping_tolerance': 1e-4,
                      'validation_split': 0,
                      'learning_rate': 1e-2, 'epochs': 1000
                      }

model = model.fit(input=input, target=target,
                  backend=fitter, fitter_options=fitter_options)

prediction = model.predict(input)

cellcount=len(cellids)
cc=np.zeros(cellcount)

cc = [np.corrcoef(prediction['prediction'][:, i], target[:, i])[0, 1] for i in range(cellcount)]


i=0

plt.figure()
ax=plt.subplot(2,1,1)
ax.plot(smooth(prediction['prediction'][:3000, i]))
ax.plot(smooth(target[:3000, i]))
ax.set_title(f"pred cc={cc[i]:.3f}")

ax=plt.subplot(2,2,3)
nplt.plot_strf(model.layers[1],model.layers[0], ax=ax)
ax=plt.subplot(2,2,4)
nplt.plot_strf(model.layers[3],model.layers[2], ax=ax)
