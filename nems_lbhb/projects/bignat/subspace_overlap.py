from os.path import basename, join
import logging
import os
import io
import importlib

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cross_decomposition import CCA

from nems0.utils import escaped_split, escaped_join, get_setting
from nems0.registry import KeywordRegistry, xforms_lib
from nems0 import xform_helper, xforms, db

from nems.layers import filter
from nems import Model
from nems.metrics import correlation
from nems.preprocessing import split
from nems.models.dataset import DataSet
from nems.models import LN
from nems0.initializers import init_nl_lite

from nems0.registry import xform, scan_for_kw_defs
from nems.layers.tools import require_shape, pop_shape
from nems_lbhb.projects.bignat.bnt_defaults import POP_MODELS,SIG_TEST_MODELS
import nems_lbhb.plots as nplt
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
from nems_lbhb.plots import histscatter2d, histmean2d, scatter_comp
from nems_lbhb.analysis import dstrf
from nems_lbhb import baphy_io
from nems_lbhb.projects.bignat import clustering_helpers
log = logging.getLogger(__name__)

figpath='/home/svd/Documents/onedrive/projects/subspace_models/'

batch=343
siteids, cellids = db.get_batch_sites(batch)

first_lin=False
ss=""
load_kw = 'gtgram.fs100.ch32-ld-norm.l1-sev'
if first_lin:
    fit_kw = 'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss95'
else:
    fit_kw = f'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4-dstrf.d20.t47.p15.ss{ss}.nl'
fit_kw_lin = 'lite.tf.init.lr1e3.t3.es20.jk8.rb5-lite.tf.lr1e4.t5e4'
modelnames = [
    f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.s-wc.70x1x80.l2:4-fir.10x1x80-relu.80.s-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
    # f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80.l2:4-fir.10x1x80-relu.80.f-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
    f'{load_kw}_wc.Nx1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_{fit_kw_lin}',
    # f'{load_kw}_wc.Nx1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80.l2:4-fir.10x1x80-relu.80.f-wc.80x100.l2:4-relu.100.f-wc.100xR.l2:4-dexp.R_{fit_kw}',
]
modelname=modelnames[0]
shortnames = [f'CNN1d32-j8p15s{ss}', 'LN32-j8r5']

siteid="LMD052a"
siteid="LMD033a"
siteid="PRN008a"
siteid="PRN064a"
siteid="CLT051c"
siteid="PRN033a"
siteid="PRN018a"
siteid="CLT033c"

cellid = [c for c in cellids if c.startswith(siteid)][0]
xfspec, ctx = xform_helper.load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True, verbose=False)

df = db.batch_comp(batch,modelnames,shortnames=shortnames)
df_cellinfo = baphy_io.get_spike_info(siteid=siteid)
df = df.merge(df_cellinfo[['siteid', 'probechannel', 'layer', 'depth', 'area', 'iso', 'sw']],
              how='inner', left_index=True, right_index=True)
df['narrow']=df['sw']<0.35
df.loc[df['layer']=='5','layer']='56'
df.loc[df['layer']=='3','layer']='13'
df.loc[df['layer']=='BS','layer']='13'
s_group=['13']  # ,'44']
df['superficial']=df['layer'].isin(s_group)
df['sorted_class']=-1
df.loc[df['narrow']&df['superficial'],'sorted_class']=0
df.loc[~df['narrow']&df['superficial'],'sorted_class']=1
df.loc[df['narrow']&~df['superficial'],'sorted_class']=2
df.loc[~df['narrow']&~df['superficial'],'sorted_class']=3
df['improve']=(df[shortnames[0]]-df[shortnames[1]]) / (df[shortnames].sum(axis=1))

types = ['NS','RS','ND','RD']

r_test_thr=0.15
pc_mag_thr=0.01
modelspec = ctx['modelspec']
out_channels=np.arange(len(modelspec.meta['cellids']))
goodcellids = [c for o,c in zip(out_channels,modelspec.meta['cellids']) \
               if (modelspec.meta['r_test'][o, 0]>r_test_thr)]
g = np.array([i for i,c in enumerate(modelspec.meta['cellids']) if c in goodcellids])
#dpc_mag = modelspec.meta['dpc_mag'][:,g]
#dpc_mag=dpc_mag / dpc_mag.sum(axis=0, keepdims=True)
#gg = dpc_mag[2,:]>=pc_mag_thr
#goodcells_cut2 = [goodcellids[i] for i in range(len(goodcellids)) if gg[i]]
print(f"{siteid} {len(goodcellids)}/{len(out_channels)} hi-pred units (r_test>{r_test_thr:.2f}) {g.sum()}:{len(goodcellids)} hi pc2")

cell_list=goodcellids
cellcount=len(cell_list)
out_channels=[i for i,c in enumerate(modelspec.meta['cellids']) if c in cell_list]

#D=10
#F=32
#X2=np.zeros((D*F+D,F))
#for f in range(F):
#    X2[f*D+D-1,f]=1
#res = dstrf.project_to_subspace(modelspec=modelspec, X=X2, out_channels=out_channels, verbose=False)
#
#res.shape

rec=ctx['est']
dpcz = modelspec.meta['dpc_all']
pc_count = dpcz.shape[1]
dpc_magz = modelspec.meta['dpc_mag_all']
dpcz = np.moveaxis(dpcz, [0, 1, 2, 3], [3, 2, 1, 0])[:, :, :, 0]
fir = filter.FIR(shape=dpcz.shape)
fir['coefficients'] = np.flip(dpcz, axis=0)
X = rec['stim'].as_continuous().T
Y = fir.evaluate(X)

dpc_all=modelspec.meta['dpc_all']
dpc_mag_all=modelspec.meta['dpc_mag_all']**2
dpc=modelspec.meta['dpc'][g]
dpc_mag=modelspec.meta['dpc_mag'][:,g]**2
pc_count=dpc_mag_all.shape[0]

dmag = dpc_mag/dpc_mag.sum(axis=0,keepdims=True)
dall = dpc_mag_all[:,0]/dpc_mag_all[:,0].sum()

dpc_var=0.9
dsum=np.cumsum(dmag,axis=0)
cellcount=dsum.shape[1]
dm = np.argmax(dsum>dpc_var,axis=0)

cell_count=dpc.shape[0]
pc_count=dpc_all.shape[1]
F = dpc_all.shape[2]
U = dpc_all.shape[3]

dmag = dpc_mag / np.sum(dpc_mag, axis=0, keepdims=True)

full_sim = np.zeros((cell_count,pc_count))
for pci in range(pc_count):
    for i in range(cellcount):
        ref_pc = dpc_all[0,[pci]].flatten()
        test_pc = dpc[i,:(dm[i]+1)]
        x = np.array([np.corrcoef(ref_pc, t.flatten())[0,1] for t in test_pc])
        full_sim[i,pci]=((x**2).sum())

#cell_list=ctx2['modelspec'].meta['cellids']
cl = np.zeros(len(cell_list),dtype=int)
for i,c in enumerate(cell_list):
    cl[i]=df.loc[c,'sorted_class']
ii = np.argsort(cl)
cls=cl[ii]
lmean=np.zeros((4,full_sim.shape[1]))
for i in np.unique(cl):
    lmean[i] = full_sim[cl==i].mean(axis=0)
    lmean[i] /= lmean[i].mean()
full_sim_sorted=full_sim[ii,:]

f,ax=plt.subplots(1,3, figsize=(7,2))
ax[0].plot(dm[ii],np.arange(cellcount),'.')
ax[0].invert_yaxis()
ax[1].imshow(full_sim_sorted,vmin=0,vmax=1)
os = 0
for i in range(3):
    os = os + (cls==i).sum()
    ax[1].axhline(os-0.5,ls='--',color='white')

ax[1].set_title('per-units sim to site-wide dPCs')
ax[1].set_xlabel('PC number')
ax[2].plot(lmean.T)
ax[2].legend(types)
ax[2].set_title('avg sim to site-wide dPCs')
ax[2].set_xlabel('PC number');




# RS, hi-pred cells (bigger N, so can be pickier)
rs_cellids=[c for o,c in enumerate(modelspec.meta['cellids']) if (df.loc[c,'narrow']==False) & (modelspec.meta['r_test'][o, 0]>0.4)]
# NS, ok-pred cells
#ns_cellids=[c for o,c in enumerate(modelspec.meta['cellids']) if (df.loc[c,'narrow']==True) & (modelspec.meta['r_test'][o, 0]>0.2)]
ns_cellids=[c for o,c in enumerate(modelspec.meta['cellids']) if (df.loc[c,'narrow']==True) & (o in g)]

# pick which cells to display

goodcellids, goodlabel = ns_cellids, 'NS'

# break RS into halves
#goodcellids, goodlabel = rs_cellids[:int(len(rs_cellids)/2)], 'RS0'
#goodcellids, goodlabel = rs_cellids[int(len(rs_cellids)/2):], 'RS1'
#goodcellids, goodlabel = rs_cellids, 'RS'

depths= df.loc[goodcellids,['depth']]
depths = depths.sort_values(by='depth')
goodcellids=depths.index.to_list()

len(goodcellids), goodcellids

use_val = False
show_preds = True
cell_list = goodcellids

X_est, Y_est = xforms.lite_input_dict(modelspec, rec, epoch_name="")

orange = [rec['resp'].chans.index(c) for c in cell_list]

maxrows = 5
rows = np.min([len(orange), maxrows])
f = plt.figure(figsize=(10, rows * 1))
gs = f.add_gridspec(rows + 1, 10)
ax = np.zeros((rows + 1, 8), dtype='O')
for r in range(rows + 1):
    for c in range(8):
        if (c < 7) & (r >= 0):
            ax[r, c] = f.add_subplot(gs[r, c])
        elif (c >= 7):
            ax[r, c] = f.add_subplot(gs[r, c:])

imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
          'interpolation': 'none'}

# top row, just plot stim in one panel
T1 = 50
T2 = 550
ss = ctx['val']['stim'].as_continuous()[:, T1:T2]
ax[0, -1].imshow(ss, aspect='auto', origin='lower', cmap='gray_r')
ax[0, -1].set_yticklabels([])
ax[0, -1].set_xticklabels([])

# spont = np.zeros(len(rec['resp'].chans))
# spont = modelspec.meta['spont_mean']
# spont = [None] * len(modelspec.meta['cellids'])
spont = rec['pred'].as_continuous().T[:5, :].mean(axis=0)

pcp = 6
for pci in range(pcp):
    d = dpcz[:, :, pci].T
    d = d / np.max(np.abs(d))  # / dpc_magz[0, oi] * dpc_magz[i, oi]
    ax[0, pci].imshow(np.fliplr(d), **imopts)
    ax[0, pci].set_yticklabels([])
    ax[0, pci].set_xticklabels([])

for j_, (oi, cellid) in enumerate(zip(orange[:rows], cell_list[:rows])):
    j = j_ + 1
    print(oi, cellid)

    pred = rec['pred'].as_continuous().T[:, oi]
    # r = rec['resp'].as_continuous().T[:, oi]
    r = rec['pred'].as_continuous().T[:, oi]

    # mwf = df.loc[cellid,'mwf']
    # if df.loc[cellid,'narrow']:
    #    ax[j, 0].plot(mwf,'r', lw=1)
    # else:
    #    ax[j, 0].plot(mwf,'gray',lw=0.5)
    # ax[j, 0].set_xticklabels([])
    # ax[j, 0].set_yticklabels([])

    dp = full_sim[g == oi, :][0]
    p1rat = dp[1] / dp[0]
    ax[j, 0].plot(np.arange(1, dpc_magz.shape[0] + 1), dp, 'o-', markersize=3)
    ax[j, 0].set_ylabel(f"{cellid}")
    ax[j, 0].set_xticks(np.arange(1, dpc_magz.shape[0] + 1))
    ax[j, 0].set_xticklabels([])
    ax[j, 0].set_yticklabels([])

    ymin, ymax = 1, 0
    for pci in range(pcp):
        y = Y[:, pci]
        b = np.linspace(y.min(), y.max(), 11)
        mb = (b[:-1] + b[1:]) / 2
        mr = [np.mean(r[(y >= b[i]) & (y < b[i + 1])]) for i in range(10)]
        me = [np.std(r[(y >= b[i]) & (y < b[i + 1])]) / np.sqrt(np.sum((y >= b[i]) & (y < b[i + 1]))) for i in
              range(10)]
        ax[j, pcp].errorbar(mb, mr, me)
        # ax[j, i*2+2].set_ylabel('Mean prediction')
        # yl=ax[j, pci+1].get_ylim()
        # ymin = np.min([yl[0], ymin])
        # ymax = np.max([yl[1], ymax])
        # ax[j, pci+1].set_ylim((ymin,ymax))
        ax[j, pcp].set_xticklabels([])
        ax[j, pcp].set_yticklabels([])

    # PC0 vs. PCi heatmap
    Z = [None] * (pcp - 1)
    for pci in range(1, pcp):
        ac, bc, Z[pci - 1], N = histmean2d(Y[:, 0], Y[:, pci], r, bins=20, minN=3, ax=ax[j, pci], spont=spont[oi],
                                           ex_pct=0.01, Z=Z[pci - 1])
    zz = np.stack(Z).flatten()
    zz = zz[np.isfinite(zz)]
    vmin, vmax = np.percentile(zz, [1, 99])
    # vmin=-vmax

    for pci in range(1, pcp):
        ac, bc, _, N = histmean2d(Y[:, 0], Y[:, pci], r, bins=20, ax=ax[j, pci], spont=spont[oi], ex_pct=0.01,
                                  Z=Z[pci - 1],
                                  cmap='summer', vmin=vmin, vmax=vmax)
        ax[j, pci].plot(Y[T1:(T1 + 100), 0], Y[T1:(T1 + 100), pci], lw=0.5, color='k')
        ax[j, pci].set_yticklabels([])
        ax[j, pci].set_xticklabels([])

    # snippet of resp/pred PSTH
    rr = ctx['val']['resp'].as_continuous()[oi, T1:T2]
    pp = ctx['val']['pred'].as_continuous()[oi, T1:T2]
    pp = pp - pp.mean()
    r0 = rr.mean()
    pp = pp / pp.std() * (rr - r0).std()
    pp = pp + r0
    ax[j, -1].plot(pp, color='gray', lw=0.5)
    ax[j, -1].plot(rr, color='black', lw=0.5)
    ax[j, -1].set_yticklabels([])
    ax[j, -1].set_xticklabels([])
    ax[j, -1].set_xlim([0, T2 - T1])
    yl = ax[j, -1].get_ylim()
    depth = df.loc[cellid, 'depth']
    sw = df.loc[cellid, 'sw']
    ax[j, -1].text(0, yl[1], f"r={modelspec.meta['r_test'][oi, 0]:.3f} depth={depth} sw={sw:.2f}")

ax[1, 0].set_title(f'PC dimension')
for pci in range(pcp):
    ax[0, pci].set_title(f'Dim {pci + 1}')

plt.tight_layout()
# f.savefig(f"{figpath}dpcs_pred_depth_{siteid}_{goodlabel}_{batch}_{shortnames[0]}.pdf")

a=Y[:,0]
b=Y[:,1]
c=Y[:,2]
d=pred
ex_pct=1
bins=16
keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
ab = np.percentile(a[keep], [ex_pct, 100 - ex_pct])
bb = np.percentile(b[keep], [ex_pct, 100 - ex_pct])
cb = np.percentile(c[keep], [ex_pct, 100 - ex_pct])
av = np.linspace(ab[0], ab[1], bins + 1)
bv = np.linspace(bb[0], bb[1], bins + 1)
cv = np.linspace(cb[0], cb[1], bins + 1)

x,y,z = np.mgrid[ab[0]:ab[1]:(ab[1]-ab[0])/(bins+1), 
                 bb[0]:bb[1]:(bb[1]-bb[0])/(bins+1), 
                 cb[0]:cb[1]:(cb[1]-cb[0])/(bins+1)]

cell_list=goodcellids
orange = [rec['resp'].chans.index(c) for c in cell_list]
plotcount=9
mmvlist = []
for i,o in enumerate(np.arange(plotcount)):
    oi = orange[o]
    cellid = cell_list[o]

    print(i+1, '/', plotcount, cellid)

    pred = rec['pred'].as_continuous().T[:, oi]
    r = rec['resp'].as_continuous().T[:, oi]

    d=pred

    mmv = np.zeros((bins,bins,bins))
    N=np.zeros_like(mmv, dtype=int)
    for i_, a_ in enumerate(av[:-1]):
        for j_, b_ in enumerate(bv[:-1]):
            for k_, c_ in enumerate(cv[:-1]):
                v_ = (a >= a_) & (a < av[i_ + 1]) & (b >= b_) & (b < bv[j_ + 1])  & (c >= c_) & (c < cv[j_ + 1]) & np.isfinite(d)
                if (v_.sum() > 0):
                    mmv[k_,j_, i_] = np.nanmean(d[v_])
                    N[k_,j_,i_] = v_.sum()
        #print(i_,N[:,:,i_])
        #print(i_)
    mmvlist.append(mmv)
#mmvlist=mmvlist+[N.astype(float)]

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import gaussian_filter

level=0.5

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes docstring).
f = plt.figure(figsize=(5,5))
ax1 = f.add_subplot(1, 1, 1, projection='3d')

for i in range(0,len(mmvlist)):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    ds = gaussian_filter(mmvlist[i], 0.75)
    if ds.max()>10000:
        # special case, this is the N matrix
        ds/=10
    else:
        ds=ds/ds.max()
    ds = np.pad(ds,1,constant_values=0)

    verts, faces, normals, values = measure.marching_cubes(ds, level)

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set(edgecolor=CB_color_cycle[i], facecolor=CB_color_cycle[i], linewidth=0.25, alpha=0.4)
    ax1.add_collection3d(mesh)

s = mmvlist[0].shape
ax1.set_xlim(0, s[0])  # a = 6 (times two for 2nd ellipsoid)
ax1.set_ylim(0, s[1])  # b = 10
ax1.set_zlim(0, s[2])  # c = 16
ax1.view_init(elev=20, azim=-60, roll=0)
ax1.set_xlabel('Dim 1')
ax1.set_ylabel('Dim 2')
plt.suptitle(f"Cell group: {goodlabel}")
