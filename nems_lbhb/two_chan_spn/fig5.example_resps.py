import os
import sys
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.disabled = True

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems_db.params
import numpy as np

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
#import nems.xform_helper as xhelp
#import nems_lbhb.xform_wrappers as nw
import nems.db as nd
import nems.plots.api as nplt
from nems.utils import find_module
from nems.plots.heatmap import _get_fir_coefficients, _get_wc_coefficients
from nems.plots.utils import ax_remove_box

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 6,
          'axes.titlesize': 6,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

batch = 259

# shrinkage
modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-mt.shr-basic"

# regular
modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

# parametric temporal filter
modelname1 = "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-dexp.1_init.r10-basic.b"
modelname2 = "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b"

modelnames = [modelname1, modelname2]
df = nd.batch_comp(batch, modelnames)
df['diff'] = df[modelname2] - df[modelname1]
df.sort_values('cellid', inplace=True, ascending=True)
m = df.index.str.startswith('por07') & (df[modelname2] > 0.3)
df['cellid'] = df.index

for index, c in df[m].iterrows():
    print("{}  {:.3f} - {:.3f} = {:.3f}".format(
            index, c[modelname2], c[modelname1], c['diff']))

plt.close('all')
outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev/"

savefig = False

#for cellid, c in df[m].iterrows():
#    fh = lplt.compare_model_preds(cellid,batch,modelname1,modelname2);
#    #fh.savefig(outpath + "fig1_model_preds_" + cellid + ".pdf")

m[np.cumsum(m)>4]=False

cellcount = np.sum(m)
colcount = 3
rowcount = cellcount+1
#rowcount = np.ceil((cellcount+1)/colcount)

i = 0
fh = plt.figure(figsize=(10,(cellcount+1)*0.8))
for cellid, c in df[m].iterrows():
    i += 1
    if i==1:
        ax0 = plt.subplot(rowcount,colcount,1)
        ax = plt.subplot(rowcount,colcount,i*colcount+1)

        _, ctx1, ctx2 = lplt.quick_pred_comp(cellid,batch,modelname1,modelname2,
                                             ax=(ax0,ax))
        ax0.get_xaxis().set_visible(False)
    else:
        ax = plt.subplot(rowcount,colcount,i*colcount+1)

        _, ctx1, ctx2 = lplt.quick_pred_comp(cellid,batch,modelname1,modelname2,
                                             ax=ax);

    if i < cellcount + 1:
        ax.get_xaxis().set_visible(False)

    fir1 = find_module('fir', ctx1['modelspec'])
    wc2 = find_module('weight_channels', ctx2['modelspec'])
    stp2 = find_module('stp', ctx2['modelspec'])
    fir2 = find_module('fir', ctx2['modelspec'])

    ctx1['modelspec'][fir1]['plot_fns'] = ['nems.plots.api.strf_timeseries']
    ctx2['modelspec'][wc2]['plot_fns'] = ['nems.plots.api.weight_channels_heatmap']
    ctx2['modelspec'][stp2]['plot_fns'] = ['nems.plots.api.before_and_after_stp']
    ctx2['modelspec'][fir2]['plot_fns'] = ['nems.plots.api.strf_timeseries']

    t_wc1 = _get_wc_coefficients(ctx1['modelspec'])
    t_fir1 = _get_fir_coefficients(ctx1['modelspec'])
    t_wc2 = _get_wc_coefficients(ctx2['modelspec'])
    t_fir2 = _get_fir_coefficients(ctx2['modelspec'])

    t_strf1 = t_wc1.T @ t_fir1

    x = np.mean(t_fir2, axis=1)
    imax = np.argmax(x)
    imin = np.argmin(x)
    ix = np.argwhere((x>x[imin]) & (x<x[imax]))
    ix = np.concatenate((np.array([imax,imin]), ix[:,0]))

    t_fir2 = t_fir2[ix]
    #t_wc = ctx2['modelspec'].phi[wc2]['coefficients'][ix]
    #t_tau = ctx2['modelspec'].phi[2]['tau'][ix]
    #t_u = ctx2['modelspec'].phi[2]['u'][ix]

    ctx2['modelspec'].phi[wc2]['coefficients'] = ctx2['modelspec'].phi[wc2]['coefficients'][ix]
    ctx2['modelspec'].phi[stp2]['tau'] = ctx2['modelspec'].phi[stp2]['tau'][ix[:3]]
    ctx2['modelspec'].phi[stp2]['u'] = ctx2['modelspec'].phi[stp2]['u'][ix[:3]]

    stream_colors=[(248/255, 153/255, 29/255),
                   (65/255, 207/255, 221/255),
                   (129/255, 201/255, 224/255),
                   (128/255, 128/255, 128/255),
                   (32/255, 32/255, 32/255)]

    channel_colors = [[254/255, 15/255, 6/255],
                      [217/255, 217/255, 217/255],
                      [129/255, 201/255, 224/255],
                      [128/255, 128/255, 128/255],
                      [32/255, 32/255, 32/255]
                      ]

    ax = plt.subplot(rowcount,colcount*2,i*colcount*2+3)
    #ctx1['modelspec'].plot(mod_index=fir1, plot_fn_idx=0,
    #                       ax=ax, rec=ctx1['val'], colors=stream_colors)
    ax.plot(t_strf1[0], color=stream_colors[0])
    ax.plot(t_strf1[1], color=stream_colors[1])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax_remove_box(ax)

    ax = plt.subplot(rowcount,colcount*2,i*colcount*2+4)
    #ctx2['modelspec'].plot(mod_index=wc2, plot_fn_idx=0, ax=ax, rec=ctx2['val'])
    t_wc2 = ctx2['modelspec'].phi[wc2]['coefficients'][:3]
    t_wc2_lim = np.max(np.abs(t_wc2))
    ax.imshow(t_wc2, cmap='bwr', clim=[-t_wc2_lim, t_wc2_lim])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax_remove_box(ax)

    ax = plt.subplot(rowcount,colcount*2,i*colcount*2+5)
    ctx2['modelspec'].plot(mod_index=stp2, plot_fn_idx=0, ax=ax, rec=ctx2['val'])
    #ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    if i==1:
        ax.legend(('1','2','3','in'))
    else:
        ax.get_legend().remove()

    ax = plt.subplot(rowcount,colcount*2,i*colcount*2+6)
    #ctx2['modelspec'].plot(mod_index=fir2, plot_fn_idx=0, ax=ax, rec=ctx2['val'])
    ax.plot(t_fir2[2], color=channel_colors[1])
    ax.plot(t_fir2[1], color=channel_colors[2])
    ax.plot(t_fir2[0], color=channel_colors[0])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    if i==1:
        ax.legend(('1','2','3'))

    ax_remove_box(ax)

    fh.canvas.draw()

if savefig:
    fh.savefig(outpath + "fig2_example_psth_preds.pdf")


