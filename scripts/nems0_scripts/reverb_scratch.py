import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import logging
import glob

import nems.analysis.api
import nems.initializers
import nems.recording as recording
import nems.preprocessing as preproc
import nems.uri
from nems.fitters.api import scipy_minimize
from nems.signal import RasterizedSignal, PointProcess

log = logging.getLogger(__name__)

signals_dir = Path('/auto/users/svd/projects/delgutte_ic/recordings')

files = [name.split('/')[-1] for name in glob.glob(str(signals_dir/'OBZ*csv'))]
cellids = [file.split('_dry.csv')[0] for file in files]

stimfile = signals_dir / 'dry_30x7000.csv'
fs = 200

i=8
cellid = cellids[i]
recname = cellid.split("-")[0]

spike_data={}
for cellid in cellids:
    respfile = signals_dir / (cellid+'_dry.csv')

    X=np.loadtxt(stimfile, delimiter=",", skiprows=0)
    spike_times = np.loadtxt(respfile, delimiter=",", skiprows=0)
    
    spike_data['cellid']=spike_times/1000 # convert to seconds

    d = {'name': [f'STIM_{i:02d}' for i in range(12)] + [f'REFERENCE' for i in range(12)],
     'start': [i*3.0 for i in range(12)]*2,
     'end': [(i+1)*3.0 for i in range(12)]*2}
epochs = pd.DataFrame(data=d)

# create NEMS-format recording objects from the raw data
resp = PointProcess(fs, spike_data, 'resp', recname, chans=cellids, epochs=epochs)
stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs)

# create the recording object from the signals
signals = {'resp': resp, 'stim': stim}
rec = recording.Recording(signals)

val = rec.jackknife_masks_by_epoch

f,ax=plt.subplots(2,1,figsize=(3,2))

ax[0].imshow(X,aspect='auto')
ax[0].set_title(cellid)
ax[1].plot(Y[0,:])
ax[1].set_xlim([0,X.shape[1]])


# INITIALIZE MODELSPEC

log.info('Initializing modelspec...')

# Method #1: create from "shorthand" keyword string
#modelspec_name = 'fir.18x15-lvl.1'        # "canonical" linear STRF
#modelspec_name = 'wc.18x1-fir.1x15-lvl.1'        # rank 1 STRF
#modelspec_name = 'wc.18x2.g-fir.2x15-lvl.1'      # rank 2 STRF, Gaussian spectral tuning
modelspec_name = 'wc.30x2.g-fir.2x12-lvl.1-dexp.1'  # rank 2 Gaussian + sigmoid static NL

# record some meta data for display and saving
meta = {'cellid': cellid,
        'batch': 271,
        'modelname': modelspec_name,
        'recording': est.name
        }
modelspec = nems.initializers.from_keywords(modelspec_name, meta=meta)

# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

log.info('Fitting model ...')

if 'nonlinearity' in modelspec[-1]['fn']:
    # quick fit linear part first to avoid local minima
    modelspec = nems.initializers.prefit_LN(
            est, modelspec, tolerance=1e-4, max_iter=500)

# then fit full nonlinear model
modelspec = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# GENERATE SUMMARY STATISTICS
log.info('Generating summary statistics ...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspec)

# evaluate prediction accuracy
modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)

log.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspec.meta['r_fit'][0][0],
        modelspec.meta['r_test'][0][0]))

# SAVE YOUR RESULTS

# uncomment to save model to disk
# logging.info('Saving Results...')
# modelspec.save_modelspecs(modelspecs_dir, modelspecs)

# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

log.info('Generating summary plot ...')

# Generate a summary plot
fig = modelspec.quickplot(rec=val)
fig.show()

# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# uncomment to browse the validation data
#from nems.gui.editors import EditorWindow
#ex = EditorWindow(modelspec=modelspec, rec=val)

# TODO SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.
