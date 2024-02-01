import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime

from laminar_tools.lfp import lfp
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems0 import xform_helper, xforms, db
from nems_lbhb.projects.freemoving import free_tools
import nems_lbhb.plots as nplt

batch=348


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

testids = [2,10]
ms = [m for i,m in enumerate(modelnames) if i in testids]
ss = [m for i,m in enumerate(shortnames) if i in testids]

dpred = db.batch_comp(batch=batch, modelnames=ms, shortnames=ss)
dpred['siteid']=dpred.index
dpred['siteid']=dpred['siteid'].apply(db.get_siteid)

modelname0=modelnames[testids[0]]
modelname=modelnames[testids[1]]
shortname=shortnames[testids[1]]

siteid='SLJ012a'
siteid='PRN043a'
siteid='SLJ016a'
siteid='PRN048a'

siteids,cellids = db.get_batch_sites(batch)
for siteid in siteids:
    cellid=[c for c in cellids if c.startswith(siteid)][0]
    print(siteid,cellid)

    xf0, ctx0 = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelnames[testids[0]])
    xf, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname)

    rasterfs=50
    smooth_win=2
    dlc_data_imp = ctx['rec']['dlc'][:, :]
    speaker1_x0y0 = 1.0, -0.8
    speaker2_x0y0 = 0.0, -0.8
    d1, theta1, vel, rvel, d_fwd, d_lat = free_tools.compute_d_theta(
        dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker1_x0y0)
    d2, theta2, vel, rvel, d_fwd, d_lat = free_tools.compute_d_theta(
        dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker2_x0y0)

    dvalid = ctx['rec']['dlc_valid']._data[0,:]
    disttheta = np.concatenate([d1,theta1,d2,theta2,vel,rvel,d_fwd,d_lat],axis=0)
    disttheta=disttheta[:,dvalid].T

    resp = ctx['val']['resp']
    modelspec=ctx['modelspec']
    modelspec0=ctx0['modelspec']
    X_val, Y_val = xforms.lite_input_dict(modelspec0, ctx0['val'], extra_signals=['disttheta'])
    prediction0 = modelspec0.predict(X_val)
    X_val, Y_val = xforms.lite_input_dict(modelspec, ctx['val'], extra_signals=['disttheta'])
    prediction = modelspec.predict(X_val)
    dlc=X_val['dlc']

    parms = modelspec.layers[12].get_parameter_values(as_dict=True)
    gain_=parms['gain']
    offset_=parms['offset']
    state_ = prediction['state']
    state = np.concatenate([np.ones((state_.shape[0],1)),state_], axis=1)
    pred=prediction['output']
    pred0=prediction0['output']
    gain = state @ gain_
    offset = state @ offset_

    plt.close('all')

    goodcells = [i for i in range(len(modelspec.meta['cellids']))
                 if modelspec.meta['r_test'][i,0]>0.11]
    cellcount=len(goodcells)
    osteps = np.arange(0,cellcount,9)
    for plot_var in ['space','vel']:
        for os in osteps:
            f, ax = plt.subplots(4,9, figsize=(16,6), sharex='row', sharey='row')
            S=10
            ax[0,0].invert_yaxis()
            ax[1,0].invert_yaxis()
            #ax[2,0].invert_yaxis()
            #ax[3,0].invert_yaxis()
            for cid in range(9):
                if cid+os<cellcount:
                    cidos = goodcells[cid + os]
                    g = gain[:,cidos]
                    g=g-g.mean()+1
                    ofs = offset[:,cidos]
                    ofs=ofs-ofs.mean()
                    r=Y_val[:,cidos]-pred0[:,cidos]
                    p=pred[:,cidos]-pred0[:,cidos]

                    if plot_var=='space':
                        nplt.histmean2d(dlc[:,0],dlc[:,1],ofs, bins=30, ax=ax[0,cid], ex_pct=0.001, spont=0, vmin=-0.05,vmax=0.05, zerolines=False, minN=15)
                        nplt.histmean2d(dlc[:,0],dlc[:,1],  g, bins=30, ax=ax[1,cid], ex_pct=0.001, spont=0, vmin=0.4, vmax=2.5, zerolines=False, minN=15)
                        nplt.histmean2d(dlc[:,0],dlc[:,1],  r, bins=30, ax=ax[2,cid], ex_pct=0.001, spont=0, vmin=None,   vmax=None, zerolines=False, minN=15)
                        nplt.histmean2d(dlc[:,0],dlc[:,1],  p, bins=30, ax=ax[3,cid], ex_pct=0.001, spont=0, vmin=None,   vmax=None, zerolines=False, minN=15)
                    elif plot_var=='vel':
                        nplt.histmean2d(disttheta[:, 7], disttheta[:, 6],ofs, bins=20, ax=ax[0,cid], ex_pct=0.1, spont=0, vmin=-0.05,vmax=0.05, zerolines=False, minN=10)
                        nplt.histmean2d(disttheta[:, 7], disttheta[:, 6],  g, bins=20, ax=ax[1,cid], ex_pct=0.1, spont=0, vmin=0.4, vmax=2.5, zerolines=False, minN=10)
                        nplt.histmean2d(disttheta[:, 7], disttheta[:, 6],  r, bins=20, ax=ax[2,cid], ex_pct=0.1, spont=0, vmin=None,   vmax=None, zerolines=False, minN=10)
                        nplt.histmean2d(disttheta[:, 7], disttheta[:, 6],  p, bins=20, ax=ax[3,cid], ex_pct=0.1, spont=0, vmin=None,   vmax=None, zerolines=False, minN=10)
                    else:
                        print(f'invalid plot_var {plot_var}')
                    #nplt.histmean2d(disttheta[:, 7], disttheta[:, 6],ofs,bins=20, ax=ax[2,cid], ex_pct=0.1, spont=0, vmin=-0.1, vmax=0.1, zerolines=True, minN=5)
                    #nplt.histmean2d(disttheta[:, 7], disttheta[:, 6],g, bins=20, ax=ax[3,cid], ex_pct=0.1, spont=0, vmin=0.4, vmax=2.5, zerolines=True, minN=5)
                    #nplt.histscatter2d(dlc[:,0],dlc[:,1],ofs,N=10000, ax=ax[0,cid], ex_pct=0.001, vmin=-0.1, vmax=0.1, zerolines=False)
                    #nplt.histscatter2d(dlc[:,0],dlc[:,1],g, N=10000, ax=ax[1,cid], ex_pct=0.001, vmin=0.5, vmax=2.0, zerolines=False)
                    #nplt.histscatter2d(dlc[:,0],dlc[:,1],r, N=10000, ax=ax[1,cid], ex_pct=0.001, vmin=0.0, vmax=0.8, zerolines=False)
                    #nplt.histscatter2d(disttheta[:, 7], disttheta[:, 6], ofs,N=10000, ax=ax[2,cid], ex_pct=0.05, spont=0, vmin=-0.1, vmax=0.1, zerolines=False)
                    #nplt.histscatter2d(disttheta[:, 7], disttheta[:, 6], g, N=10000, ax=ax[3,cid], ex_pct=0.05, spont=0, vmin=0.5, vmax=2, zerolines=False)

                    cellid = modelspec.meta['cellids'][cidos]
                    if cid==0:
                        cu = cellid
                    else:
                        cu = cellid.split("-", 1)[1]
                    ts = f"{cu} {dpred.loc[cellid,shortnames[testids[0]]]:.2f} v {dpred.loc[cellid,shortnames[testids[1]]]:.2f}"
                    ax[0, cid].set_title(ts)

            ax[0,0].set_ylabel('pos/offset')
            ax[1,0].set_ylabel('pos/gain')
            ax[0,0].set_ylabel('vel/offset')
            ax[1,0].set_ylabel('vel/gain')

            yl = ax[0, 0].get_ylim()
            if (plot_var=='vel') & (yl[0]>yl[1]):
                ax[0, 0].set_ylim((yl[1], yl[0]))
                ax[1, 0].set_ylim((yl[1], yl[0]))
                ax[2, 0].set_ylim((yl[1], yl[0]))
                ax[3, 0].set_ylim((yl[1], yl[0]))
                pass

            outpath='/home/svd/Documents/onedrive/projects/free_moving/heatmaps/'
            outfile = f"{outpath}{siteid}-os{os}-{plot_var}.jpg"
            f.savefig(outfile)
            outpath='/home/svd/Documents/onedrive/projects/free_moving/pdf/'
            outfile = f"{outpath}{siteid}-os{os}-{plot_var}.pdf"
            f.savefig(outfile)

