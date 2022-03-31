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
from nems.xform_helper import load_model_xform, fit_model_xform
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


##
batch, cellid = 308, 'AMT018a-09-1'
modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-fir.2x15x2-relu.2-wc.2x1-lvl.1-dexp.1_init.tf.rb5-tf.n'
xfspec, ctx = fit_model_xform(cellid, batch=batch, modelname=modelname)
nplt.quickplot(ctx)
ex = gui.browse_xform_fit(ctx, xfspec)

###Plot complexity of model versus how effective it was
batch = 308
metric = 'r_test'
metric2 = 'n_parms'
metric3 = 'se_test'
query = "SELECT {0}, {1}, {2}, {3} FROM NarfResults WHERE batch = 308".format(metric, metric2,metric3, 'modelname')
results = nd.pd_query(sql=query)
group = results.groupby('modelname')
mean_rtest = group.mean()
mean_rtest.plot(x='n_parms',y='r_test',kind='scatter',yerr='se_test')

#find 'best' model
best = mean_rtest['r_test'].idxmax()
###

#get the values for all cellids for the 'best' model, kind of just messing to see if I could
modelname = best #'ozgf.fs100.ch18-ld-sev_dlog-wc.18x6.g-fir.2x15x3-relu.3-wc.3x1-lvl.1-dexp.1_tf.n.rb10'
batch = 308
metric = 'r_test'
metric2 = 'r_fit'
query = "SELECT {0}, {1}, {2} FROM NarfResults WHERE modelname = modelname AND batch = 308".format(metric,metric2,'cellid')
# params = modelname
results = nd.pd_query(sql=query).set_index('cellid')

df.index[df['BoolCol']].tolist()


site = "AMT003c"
modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x6.g-fir.2x15x3-relu.3-wc.3x1-lvl.1-dexp.1_tf.n.rb10'
batch = 308
metric = 'r_test'
# metric2 = 'r_fit'
query = "SELECT {0}, {1} FROM NarfResults WHERE modelname = %s AND cellid like %s".format(metric,'cellid')
params = (modelname, site+'%')
results = nd.pd_query(sql=query,params=params)


####
import nems.gui.editors as gui
import nems.xform_helper as xhelp

cellid="BRT033b-03-3"
cellid="AMT003c-33-1"

cellid='BRT026c-41-2'

cellid="AMT005c-20-1"
batch=308
modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x6.g-fir.2x15x3-relu.3-wc.3x1-lvl.1-dexp.1_tf.n.rb10"
modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-fir.2x15x2-relu.2-wc.2x1-lvl.1-dexp.1_tf.n.rb10"

modelname='ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-fir.1x15x3-relu.3-wc.3x1-lvl.1-dexp.1_tf.n.rb10'
modelname='ozgf.fs100.ch18-ld-sev_dlog-wc.18x6.g-fir.2x15x3-relu.3-wc.3x1-lvl.1-dexp.1_tf.n.rb10'

modelname='ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-relu.1-lvl.1-dexp.1_tf.n.rb10'
# modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-relu.1-lvl.1-dexp.1_tf.n.rb5"
cellid ='AMT018a-04-1'
cellid = 'AMT005c-20-1'
modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x8.g-fir.8x15x1-lvl.1-dexp.1_tf.n.rb10'
modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x8.g-fir.2x15x4-relu.4-wc.4x1-lvl.1-dexp.1_tf.n.rb10'

xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
ex = gui.browse_xform_fit(ctx, xfspec)