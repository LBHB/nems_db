import nems.db as nd  # NEMS database functions -- NOT celldb
import nems_lbhb.baphy as nb   # baphy-specific functions
import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
import numpy as np
import matplotlib.pyplot as plt
import nems.recording as recording
from nems.recording import load_recording
from nems.gui.recording_browser import browse_recording, browse_context
import nems.epoch as ep
import nems.plots.api as nplt
from nems.preprocessing import average_away_epoch_occurrences
from nems.xform_helper import load_model_xform
import nems.gui.editors as gui


# If using database:
batch = 289  # NAT + pupil
cellid = 'BRT036b-45-2'
#cellid = 'BRT037b-63-1'
cellid = 'TAR010c-13-1'
cellid = 'TAR009d-42-1'
options = {'rasterfs': 100, 'stimfmt': 'ozgf',
           'chancount': 18, 'pupil': True, 'stim': True}

# get the name of the cached recording
uri = nb.baphy_load_recording_uri(cellid=cellid, batch=batch, **options)
rec = load_recording(uri)

# convert to rasterized signals from PointProcess and TiledSignal
rec['resp']=rec['resp'].rasterize()
rec['stim']=rec['stim'].rasterize()

rec['resp'] = rec['resp'].extract_channels([cellid])
rec.meta["cellid"] = cellid

#est, val = estimation, validation data sets
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex="^STIM_")
est = average_away_epoch_occurrences(est, epoch_regex="^STIM_")
val = average_away_epoch_occurrences(val, epoch_regex="^STIM_")

# get matrices for fitting:
X_est = est['stim'].apply_mask().as_continuous()  # frequency x time
Y_est = est['resp'].apply_mask().as_continuous()  # neuron x time

# get matrices for testing model predictions:
X_val = val['stim'].apply_mask().as_continuous()
Y_val = val['resp'].apply_mask().as_continuous()


# find a stimulus to display
epoch_regex = '^STIM_'
epochs_to_extract = ep.epoch_names_matching(val.epochs, epoch_regex)
epoch=epochs_to_extract[0]

plt.figure()
ax = plt.subplot(3, 1, 1)
nplt.spectrogram_from_epoch(val['stim'], epoch, ax=ax, time_offset=2)

ax = plt.subplot(3, 1, 2)
nplt.timeseries_from_epoch([val['resp']], epoch, ax=ax)

raster = rec['resp'].extract_epoch(epoch)
ax = plt.subplot(3, 1, 3)
plt.imshow(raster[:,0,:])

plt.tight_layout()

# see what a "traditional" NEMS model looks like
nems_modelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-do.2x15-lvl.1-dexp.1_init-basic"
xfspec, ctx = load_model_xform(cellid, batch=batch, modelname=nems_modelname)
nplt.quickplot(ctx)

ex = gui.browse_xform_fit(ctx, xfspec)
