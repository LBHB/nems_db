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
import nems0.epoch as ep
from nems.tools import json
from nems.models import LN
from nems_lbhb.projects.spatial.models import LN_Tiled_STRF

from nems.visualization.model import plot_nl
from nems.metrics import correlation

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.projects.freemoving.free_tools import stim_filt_hrtf, compute_d_theta, \
    free_scatter_sum, dlc2dist

log = logging.getLogger(__name__)

# testing binaural NAT with various model architectures.
batch=338
siteids,cellids=db.get_batch_sites(batch)

siteid="PRN031a"
siteid="PRN014b"
siteid="CLT013a"
siteid="SLJ021a"
siteid="PRN021a"
siteid="CLT041c"

# uncomment one of these lines to choose a different stimulus representation
#loadkey = "gtgram.fs100.ch18.bin10"   # binaural + HRTF
loadkey = "gtgram.fs100.ch18.bin100"   # binaural allocentric
#loadkey = "gtgram.fs100.ch18.mono"    # monoaural + noise to balance estimation noise

# architecture can be "full" or "tiled"
#architecture = "tiled"
#architecture = "full"
architecture = "fullpop"

time_lags = 20
rank = 8

recording_uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
rec = recording.load_recording(recording_uri)


print(f"Site {siteid} has {rec['resp'].shape[0]} cells")

# impose HRTF on stimul
rec=stim_filt_hrtf(rec)['rec']

ctx = {'rec': rec}
ctx.update(xforms.normalize_sig(sig='resp', norm_method='minmax', **ctx))
ctx.update(xforms.normalize_sig(sig='stim', norm_method='minmax', log_compress=1, **ctx))
ctx.update(xforms.split_by_occurrence_counts(epoch_regex='^STIM', **ctx))
ctx.update(xforms.average_away_stim_occurrences(epoch_regex='^STIM', **ctx))

epochs = ctx['est']['resp'].epochs
stim_epochs = ep.epoch_names_matching(epochs, "^STIM_")
mono_epochs = [e for e in stim_epochs if e.startswith("STIM_NULL") | e.endswith("NULL:2")]
bin_epochs = [e for e in stim_epochs if (e.startswith("STIM_NULL") is False) & (e.endswith("NULL:2") is False)]
val_epochs = ep.epoch_names_matching(ctx['val']['resp'].epochs, "^STIM_")

print(f"N stimuli: bin/mono/total: {len(bin_epochs)}/{len(mono_epochs)}/{len(stim_epochs)}")

#
# Extract data for a single neuron
#

# pick a cell
if architecture.endswith("pop"):
    cid=np.arange(ctx['est']['resp'].shape[0])
    cellid = siteid
else:
    cid=[19]  # number 19 is "nice"
    cellid = ctx['est']['resp'].chans[cid[0]]


X_ = ctx['est']['stim'].extract_epochs(stim_epochs)
Y_ = ctx['est']['resp'].extract_epochs(stim_epochs)
# convert to matrix
X_est = np.stack([X_[k][0,:,:].T for k in X_.keys()], axis=0)
Y_est = np.stack([Y_[k][0,cid,:].T for k in X_.keys()], axis=0)

X_ = ctx['val']['stim'].extract_epochs(val_epochs)
Y_ = ctx['val']['resp'].extract_epochs(val_epochs)
# convert to matrix
X_val = np.stack([X_[k][0,:,:].T for k in X_.keys()], axis=0)
Y_val = np.stack([Y_[k][0,cid,:].T for k in X_.keys()], axis=0)

print(X_est.shape, Y_est.shape, X_val.shape, Y_val.shape)

# do the fit

input_channels = X_est.shape[2]
output_channels = Y_est.shape[2]

modelname = f"{cellid}_{loadkey}_{architecture}_rank{rank}"
if architecture == 'full':
    strf_base = LN.LN_STRF(time_lags, input_channels, rank=rank, gaussian=False,
                           fs=rec['resp'].fs, name=modelname)
elif architecture == 'fullpop':
    strf_base = LN.LN_pop(time_lags, input_channels, output_channels,
                          rank=rank, gaussian=False,
                          fs=rec['resp'].fs, name=modelname)
elif architecture == 'tiled':
    strf_base = LN_Tiled_STRF(time_lags, input_channels, rank=rank, gaussian=False,
                              fs=rec['resp'].fs, name=modelname)
else:
    raise ValueError(f"Unknown architecture {architecture}")
strf = strf_base.fit_LBHB(X_est, Y_est)

# use STRF to predict validation response and measure accuracy with correlation function
predict = strf.predict(X_val, batch_size=None)
predict.shape, Y_val.shape
r=correlation(predict, Y_val)
print(f"prediction correlation={np.round(r,2)}")


# display fit result
labels = [f"{c[8:]} {rr:.3f}" for c,rr in zip(ctx['est']['resp'].chans,r)]

f1=strf.plot_strf(layer=1, plot_nl=True)
f2=strf.plot_strf(labels=labels)

s = strf.get_strf()
hcontra = s[:18,:,:]
hipsi = s[18:,:,:]
hsum = (hcontra+hipsi).std(axis=(0,1)) #/ s.std(axis=(0,1))
hdiff = (hcontra-hipsi).std(axis=(0,1)) #/ s.std(axis=(0,1))

plt.figure()
plt.scatter(hsum,hdiff)
mm = np.max(np.concatenate((hsum,hdiff)))

plt.plot([0,mm],[0,mm],'--')
plt.xlabel('std(sum)')
plt.ylabel('std(diff)')
plt.title(strf.name)

# save
json.save_model(strf, f'/tmp/modelspec_{siteid}_popstrf.json')

# load
model = json.load_model(f'/tmp/modelspec_{siteid}_popstrf.json')



