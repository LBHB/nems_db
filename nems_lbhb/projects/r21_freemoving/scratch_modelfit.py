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
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems.layers import WeightChannels, FIR, LevelShift, DoubleExponential
from nems import Model
from nems.layers.base import Layer, Phi, Parameter


siteid = "PRN015a"
runclassid = 132

sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
dparm = db.pd_query(sql)
parmfile = [r.stimpath+r.stimfile for i,r in dparm.iterrows()]
cellids=None

## load the recording
rasterfs = 50
cid = 34
ex = BAPHYExperiment(parmfile=parmfile, cellid=cellids)
print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

recache = False
rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
                       dlc=True, recache=recache, rasterfs=50,
                       dlc_threshold=0.2, fill_invalid='interpolate')

dlc_data = rec['dlc'][:,:]
rec = impute_multi(rec, sig='dlc', empty_values=None, keep_dims=8)['rec']

input = {'stim': rec['stim'].rasterize().as_continuous().T,
         'dlc': rec['dlc'].as_continuous().T}
target = rec['resp'].rasterize().as_continuous()[cid,:]

class Concat(Layer):
    def evaluate(self, *inputs):
        # All inputs are treated the same, no fittable parameters.

        return np.concatenate(inputs, axis=1)

layers = [
    WeightChannels(shape=(18, 1, 2), input='stim', output='prediction'),
    FIR(shape=(15, 1, 2), input='prediction', output='prediction'),
    WeightChannels(shape=(8, 1, 2), input='dlc', output='space'),
    FIR(shape=(15, 1, 2), input='space', output='space'),
    Concat(input=['prediction','space'],output='prediction'),
    WeightChannels(shape=(4, 1)),
    LevelShift(shape=(1, 1)),
]

model = Model(layers=layers, output_name='output')
model = model.sample_from_priors()

options = {'cost_function': 'nmse', 'options': {'ftol':  1e-4, 'gtol': 1e-4, 'maxiter': 100}}

model = model.fit(input=input, target=target,
                  backend='scipy', fitter_options=options)

prediction = model.predict(input)
