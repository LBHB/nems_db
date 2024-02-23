import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime
import pandas as pd
import decoder_tools
from nems_lbhb import plots
from laminar_tools.lfp import lfp
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems0 import xform_helper, xforms, db

batch=348

siteid='PRN015a'
siteid='PRN020a'
siteid='PRN034a'
siteid='PRN033a'
siteid='PRN043a'
siteid='PRN059a'
siteid='PRN050a'
siteid='PRN057a'
siteid='CLT047c'
siteid='TNC047a'
siteid='PRN064a'
siteid='PRN048a'

# for comparison with histogram firing rate
# hist_df = pd.read_pickle('/auto/users/wingertj/data/spatial_hist_df.pkl')
# hist_df.loc[hist_df['layer']=='5', 'layer'] = '56'
# hist_df.loc[hist_df['layer']=='3', 'layer'] = '13'
# hist_df.loc[hist_df['layer']=='BS', 'layer'] = '13'
# hist_df.drop(hist_df[hist_df['layer']=='16'].index, inplace=True)
# hist_df = hist_df.sort_values('layer')

siteids,cellids = db.get_batch_sites(batch)
cellid=[c for c in cellids if c.startswith(siteid)][0]
print(siteid,cellid)

modelnames=['free.fs50.ch18-norm.l1-fev.jk8_wc.Nx1x30-fir.10x1x30-wc.30xR-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.hrtf.jk8_wc.Nx1x30-fir.10x1x30-wc.30xR-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.jk8-shuf.dlc_wcst.Nx1x25.i.l2:4-wcdl.12x1x12.i.l2:4-first.8x1x25-firdl.3x1x12.nc1-cat-relu.37-wc.37x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.hrtf.jk8-shuf.dlc_wcst.Nx1x25.i.l2:4-wcdl.12x1x12.i.l2:4-first.8x1x25-firdl.3x1x12.nc1-cat-relu.37-wc.37x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.hrtfae.jk8-shuf.dlc_wcst.Nx1x25.i.l2:4-wcdl.12x1x12.i.l2:4-first.8x1x25-firdl.3x1x12.nc1-cat-relu.37-wc.37x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.jk8_wcst.Nx1x25.i.l2:4-wcdl.12x1x12.i.l2:4-first.8x1x25-firdl.3x1x12.nc1-cat-relu.37-wc.37x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.hrtf.jk8_wcst.Nx1x25.i.l2:4-wcdl.12x1x12.i.l2:4-first.8x1x25-firdl.3x1x12.nc1-cat-relu.37-wc.37x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.hrtfae.jk8_wcst.Nx1x25.i.l2:4-wcdl.12x1x12.i.l2:4-first.8x1x25-firdl.3x1x12.nc1-cat-relu.37-wc.37x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.hrtf.jk8-shuf.dlc_wcdl.12x1x12.i.s.l2:4-firs.3x1x12.nc1-relus.12-wcs.12x12.l2:4-relus.12-wc.Nx1x25.i.l2:4-fir.8x1x25-relu.25-wc.25x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-stategain.13xR-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.jk8_wcdl.12x1x12.i.s.l2:4-firs.3x1x12.nc1-relus.12-wcs.12x12.l2:4-relus.12-wc.Nx1x25.i.l2:4-fir.8x1x25-relu.25-wc.25x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-stategain.13xR-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.hrtf.jk8_wcdl.12x1x12.i.s.l2:4-firs.3x1x12.nc1-relus.12-wcs.12x12.l2:4-relus.12-wc.Nx1x25.i.l2:4-fir.8x1x25-relu.25-wc.25x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-stategain.13xR-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            'free.fs50.ch18-norm.l1-fev.jk8-shuf.stim_wcst.Nx1x25.i.l2:4-wcdl.12x1x12.i.l2:4-first.8x1x25-firdl.3x1x12.nc1-cat-relu.37-wc.37x1x30.l2:4-fir.4x1x30-relu.30-wc.30xR.l2:4-relu.R.o.s_lite.tf.cont.init.lr1e3.t3.rb5-lite.tf.cont.lr1e4.t5e4',
            ]
shortnames=['NoH+LN','HRTF+LN','NoH+Dsh','HRTF+Dsh','HRTFae+Dsh',
            'NoH+DLC','HRTF+DLC','HRTFae+DLC',
            'HRTF+Dsh sg','NoH+DLC sg','HRTF+DLC sg','NoA+DLC']
modelname=modelnames[10]
shortname=shortnames[10]

# find significant cells
df = db.batch_comp(batch=348, modelnames = modelnames, shortnames=shortnames)
df_floor = db.batch_comp(batch=348, modelnames = modelnames, shortnames=shortnames, stat='r_floor')
mask = df > df_floor
# compare models
plots.scatter_comp(df[shortnames[8]], df[shortnames[10]])

xf, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname)


resp = ctx['val']['resp']
modelspec=ctx['modelspec']
X_val, Y_val = xforms.lite_input_dict(modelspec, ctx['val'])
prediction = modelspec.predict(X_val)
dlc=X_val['dlc']

parms = modelspec.layers[12].get_parameter_values(as_dict=True)
gain_=parms['gain']
offset_=parms['offset']
state_ = prediction['state']
state = np.concatenate([np.ones((state_.shape[0],1)),state_], axis=1)

gain = state @ gain_
offset = state @ offset_
cid=0

tcs, xy, xy_edges = decoder_tools.spatial_tc_2d(rec=ctx['rec'], occ_threshold=0.5, bin_width =1/30)
# make a gain field tuning curve function that puts into same frame of reference as spatial firing rate map
gtcs, _, _ = decoder_tools.state_tc_2d(resp=gain, signal=dlc, sig_chans=ctx['rec']['dlc'].chans, occ_threshold=0.5, bin_width =1/30)

#plt.close('all')
# grab xy histograms for site
# site_df = hist_df[hist_df['siteid'] == cellid[:7]]
cellcount=len(modelspec.meta['cellids'])
for os in [0,9]:
    f, ax = plt.subplots(2,9, figsize=(16,3), sharey=True, sharex=True)
    f1, ax1 = plt.subplots(1, 9, figsize=(16, 3), sharey=True, sharex=True)
    S=10
    ax[0,0].invert_yaxis()
    for cid in range(9):
        cidos = cid+os
        if cidos<cellcount:
            cu = modelspec.meta['cellids'][cidos].split("-", 1)[1]
            cellid = modelspec.meta['cellids'][cidos]
            tc = tcs[cellid]
            ax[0, cid].scatter(dlc[::S,0],dlc[::S,1],s=1,c=offset[::S,cidos],vmin=-0.1,vmax=0.1)
            ax[1, cid].scatter(dlc[::S,0],dlc[::S,1],s=1,c=gain[::S,cidos],vmin=0.5,vmax=2.0)
            xbins = tc.shape[0]
            ybins = tc.shape[1]
            imopts = {'cmap': 'viridis', 'interpolation': 'none', 'extent': [0, xbins * 20, 0, ybins * 20]}
            ax1[cid].imshow(tc.T, **imopts)
            ax[0, cid].set_title(f"{cu} {modelspec.meta['r_test'][cidos,0]:.2f}")
            ax1[cid].set_title(f"{cu} {modelspec.meta['r_test'][cidos,0]:.2f}")
    ax[0,0].set_ylabel('offset')
    ax[1,0].set_ylabel('gain')

# performance for cells is based on shuffled model vs non-shuffled model


bp = []