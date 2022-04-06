import nems.db as nd
import nems_lbhb.plots as lp
import nems_lbhb.SPO_helpers as sp
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nems_web.utilities.pruffix as prx
from nems_lbhb.SPO_helpers import get_subs

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

metric = 'r_test'

comparisons=((0,1),)
batch=306
mns=['env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2:3:FIRWC-newtf.n.lr1e4.L2:3:FIRWC-SPOpf.GPU',
    'env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2:4:FIRWC-newtf.n.lr1e4.L2:4:FIRWC-SPOpf.GPU']

batch=306
mns=['env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4.L2:4:FIRWC-SPOpf.GPU',
     'env.fs200-SPOld-SPOsev_fir.2x20x40.z-relu.40-wc.40xR-lvl.R_tfinit.n.lr1e3.et3.L2:4:FIRWC-newtf.n.lr1e4.L2:4:FIRWC-SPOpf.GPU']
mns=['ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x10.g-fir.1x25x10-wc.10xR-lvl.R-dexp.R_prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4-dstrf',
     'ozgf.fs100.ch18-ld-norm.l1-sev_conv2d.10.8x3.rep3-wcn.110-relu.110-wc.110xR-lvl.R-dexp.R_prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4-dstrf']

mns= ['env.fs200-SPOld-SPOsev_dlog-fir.2x30x40-relu.40.f-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rbp30-newtf.n.lr1e4-SPOpf.GPU',
      'env.fs200-SPOld-SPOsev_dlog-fir.2x30x40-relu.40.f-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rbp10-newtf.n.lr1e4-SPOpf.Exa']
# mns=['env.fs200-SPOld-stSPO.nb-SPOsev-shuf.st_dlog-stategain.2x2.g.o1.b0d001:5-fir.2x30.z-lvl.1-dexp.1_SDB-init.t5.rb5-basic.t6-SPOpf.Exa',
#      'env.fs200-SPOld-stSPO.nb-SPOsev_dlog-stategain.2x2.g.o1.b0d001:5-fir.2x30.z-lvl.1-dexp.1_SDB-init.t5.rb5-basic.t6-SPOpf.Exa',
#      'env.fs200-SPOld-stSPO.nb-SPOsev-shuf.st_dlog-stategain.2x2.g.o1.b0d001:5-wc.2x2.c-stp.2-fir.2x30.z-lvl.1-dexp.1_SDB-init.t5.rb5-basic.t6-SPOpf.Exa',
#      'env.fs200-SPOld-stSPO.nb-SPOsev_dlog-stategain.2x2.g.o1.b0d001:5-wc.2x2.c-stp.2-fir.2x30.z-lvl.1-dexp.1_SDB-init.t5.rb5-basic.t6-SPOpf.Exa'
#      ]
# comparisons=((0,1),(2,3),(0,2))

## Get df of modefits
#cells = sp.get_significant_cells(batch,mns,as_list=True) #Sig cells across all models
#cells = sp.get_significant_cells(batch,mns[:1],as_list=True) #Sig cells in the first model
dfc = nd.get_results_file(batch,mns[:1]); cells = list(dfc['cellid'].values) # All cells fit in the first model
#cells = [cell for cell in cells if 'fre' not in cell]; print('Keeping only fred cells')
print(f'{len(cells)} cells')
df = nd.get_results_file(batch,mns,cells)

cells_fit_in_all = set(df.loc[df['modelname']==mns[0],'cellid'])
for mn_ in mns[1:]:
    cells_fit_in_all = cells_fit_in_all.intersection(set(df.loc[df['modelname']==mn_,'cellid']))
df = df[df['cellid'].isin(cells_fit_in_all)]
print(f'Dropped not fit, down to {len(df)/len(mns)} cells per model')


## Define fnargs, arguments tha will be passed to a function called when you click on a point
fnargs = [{'ax': imageax, 'ft': 5, 'data_series_dict': 'dsx'},
          {'ax': imageax2, 'ft': 5, 'data_series_dict': 'dsy'}]
##Create figure
subs = get_subs(len(comparisons))
fig, ax = plt.subplots(nrows=subs[1], ncols=subs[0])
if len(comparisons)==1:
    axf = [ax]
else:
    axf = ax.flatten()
abbr, pre, suf = prx.find_common(mns)
val_range=np.array((0.0,1.0))
for i,ax_ in enumerate(axf[:len(comparisons)]):
    ax_.plot([0,1],[0,1],'grey')

    ix = comparisons[i][0]
    iy = comparisons[i][1]
    ## Split datafram by model, sort by cellid
    dfx = df[df['modelname'] == mns[ix]].copy()
    dfx.sort_values('cellid',inplace=True)
    dfy = df[df['modelname'] == mns[iy]].copy()
    dfy.sort_values('cellid', inplace=True)

    ## Create scatterplot
    ph = sp.scatterplot_print_df(dfx, dfy, [metric, metric],
                      fn=[sp.show_img, sp.show_img], fnargs=fnargs,
                      color=[.7, .7, .7], ax=ax_)
    ax_.set_aspect('equal', adjustable='box')
    ax_.set_title(pre + ' * ' + suf,fontsize=8)
    ax_.set_xlabel(abbr[ix]); ax_.set_ylabel(abbr[iy])
    x=dfx[metric].values
    y=dfy[metric].values
    val_range[0]=np.concatenate((x,y,val_range[:1])).min()
    val_range[1] = np.concatenate((x, y, val_range[1:])).max()
    ff=np.isfinite(x) & np.isfinite(y)
    rs=scipy.stats.wilcoxon(x[ff],y[ff])
    ax_.text(0,1,'p={0:.4f}\nmed(y-x)={1:.5f}'.format(rs.pvalue,np.median(y[ff]-x[ff])),verticalalignment='top')

plot_range = val_range + np.array((-1,1))*.02*(val_range[1]-val_range[0])
[ax_.set_xlim(plot_range) for ax_ in axf]
[ax_.set_ylim(plot_range) for ax_ in axf]