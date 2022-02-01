import nems.db as nd
import nems_lbhb.plots as lp
import nems_lbhb.SPO_helpers as sp
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nems_web.utilities.pruffix as prx

## Create 4 figures that will be populated by images created when fitting the model (2 for each model)
height = 1500
width = 1000
dpi = 150
figsize = width / float(dpi), height / float(dpi)
fig = plt.figure(figsize=figsize)
imageax = fig.add_axes([0, 0, 1, 1])
imageax.axis('off')
fig2 = plt.figure(figsize=figsize)
imageax2 = fig2.add_axes([0, 0, 1, 1])
imageax2.axis('off')
fig3 = plt.figure(figsize=figsize)
imageax = [imageax, fig3.add_axes([0, 0, 1, 1])]
imageax[1].axis('off')
fig4 = plt.figure(figsize=figsize)
imageax2 = [imageax2, fig4.add_axes([0, 0, 1, 1])]
imageax2[1].axis('off')
imi = 13

## Define your batch, metric to compare, and two models to be compared
batch=306
metric = 'r_test'
mns=['env.fs100-ld-sev_wc.2x40-fir.1x12x40-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
     'env.fs100-SPOld-SPOsev_wc.2x40-fir.1x12x40-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU']
mns=['env.fs100-SPOld-SPOsev_wc.2x40-fir.1x12x40-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU',
     'env.fs100-SPOld-SPOsev_fir.2x12x40-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU']
mns=['env.fs100-SPOld-SPOsev_dlog-fir.2x15.z-lvl.1_SDB-init-basic.t7-SPOpf.NEOB',
     'env.fs100-SPOld-SPOsev_dlog-fir.2x15.z-lvl.1-dexp.1_SDB-init-basic.t7-SPOpf.NEOB']
mns=['env.fs200-SPOld-SPOsev_fir.2x12x40.z-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU',
     'env.fs200-SPOld-SPOsev_fir.2x12x40-relu.40.f-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU']
mns=['env.fs200-SPOld-SPOsev_fir.2x12x40-relu.40.f-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU',
     'env.fs200-SPOld-SPOsev_fir.2x12x40-relu.40.f-wc.40xR-lvl.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU']
mns=['env.fs200-SPOld-SPOsev_fir.2x12x40-relu.40.f-wc.40xR-lvl.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU',
     'env.fs200-SPOld-SPOsev_fir.2x12x40-relu.40.f-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2-newtf.n.lr1e4.L2-SPOpf.GPU']

mns=['env.fs200-SPOld-SPOsev_fir.2x12x40.z-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4-SPOpf.GPU',
     'env.fs200-SPOld-SPOsev_fir.2x12x40-relu.40.f-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2-newtf.n.lr1e4.L2-SPOpf.GPU']
mns=['env.fs200-SPOld-SPOsev_fir.2x12x40-relu.40.f-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2-newtf.n.lr1e4.L2-SPOpf.GPU',
     'env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2:4:FIRWC-newtf.n.lr1e4.L2:4:FIRWC-SPOpf.GPU']

mns=['env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2:3:FIRWC-newtf.n.lr1e4.L2:3:FIRWC-SPOpf.GPU',
    'env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2:4:FIRWC-newtf.n.lr1e4.L2:4:FIRWC-SPOpf.GPU']

mns=['env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4.L2:4:FIRWC-SPOpf.GPU',
     'env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2:4:FIRWC-newtf.n.lr1e4.L2:4:FIRWC-SPOpf.GPU']
mns=['ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x10.g-fir.1x25x10-wc.10xR-lvl.R-dexp.R_prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4-dstrf',
     'ozgf.fs100.ch18-ld-norm.l1-sev_conv2d.10.8x3.rep3-wcn.110-relu.110-wc.110xR-lvl.R-dexp.R_prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4-dstrf']

#d=nd.batch_comp(batch=306,modelnames=mns)
#lp.scatter_comp(d[mns[0]],d[mns[1]],mns[0],mns[1])

## Get df of modefits
cells = sp.get_significant_cells(batch,mns,as_list=True)
print(f'{len(cells)} cells')
df = nd.get_results_file(batch,mns,cells)


## Define fnargs, arguments tha will be passed to a function called when you click on a point
fnargs = [{'ax': imageax, 'ft': 5, 'data_series_dict': 'dsx'},
          {'ax': imageax2, 'ft': 5, 'data_series_dict': 'dsy'}]
##Create figure
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot([0,1],[0,1],'grey')
val_range=np.array((0.0,1.0))
problem_cells=[]

## Split datafram by model, sort by cellid
dfx = df[df['modelname'] == mns[0]].copy()
dfx.sort_values('cellid',inplace=True)
dfy = df[df['modelname'] == mns[1]].copy()
dfy.sort_values('cellid', inplace=True)

## Create scatterplot
ph = sp.scatterplot_print_df(dfx, dfy, [metric, metric],
                  fn=[sp.show_img, sp.show_img], fnargs=fnargs,
                  color=[.7, .7, .7], ax=ax)
ax.set_aspect('equal', adjustable='box')
st=mns[0].find('sev')+4
end = mns[0].find('_', mns[0].find('_') + 1)
abbr, pre, suf = prx.find_common(mns)
#ax.set_title(mns[0][st:end],fontsize=8)
#ax.set_xlabel(mns[0]); ax.set_ylabel(mns[1])
ax.set_title(pre + ' * ' + suf,fontsize=8)
ax.set_xlabel(abbr[0]); ax.set_ylabel(abbr[1])
x=dfx[metric].values
y=dfy[metric].values
val_range[0]=np.concatenate((x,y,val_range[:1])).min()
val_range[1] = np.concatenate((x, y, val_range[1:])).max()
ff=np.isfinite(x) & np.isfinite(y)
rs=scipy.stats.wilcoxon(x[ff],y[ff])
ax.text(0,1,'p={0:.4f}\nmed(y-x)={1:.5f}'.format(rs.pvalue,np.median(y[ff]-x[ff])),verticalalignment='top')