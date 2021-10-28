import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 12,
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nems.db as nd
from nems import xforms
from nems.recording import load_recording
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.xform_wrappers import split_pop_rec_by_mask
from nems.utils import smooth
from nems_lbhb import plots as nplt
from nems.xform_helper import load_model_xform
from nems_lbhb import baphy_experiment

# NEMS PSTH pup analysis:
#  batch 331= CPN

batch = 334
cellid = "DRX006b-128-2"
modelnames = [
    "ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
    ]
shortnames = ['ln_pop', 'conv1dx2']

pca_file = '/auto/users/svd/python/scripts/NAT_pop_models/dpc334.csv'


runclass = "NAT"
sql="select sCellFile.*,gSingleCell.siteid,gSingleCell.phototag from gSingleCell INNER JOIN sCellFile ON gSingleCell.id=sCellFile.singleid" +\
   " INNER JOIN gRunClass on gRunClass.id=sCellFile.runclassid" +\
   f" WHERE gRunClass.name='{runclass}' AND not(isnull(phototag))"
d=nd.pd_query(sql)
d['parmfile']=d['stimpath']+d['stimfile']
print(f'cell/file combos with phototag labels={len(d)}')

dtag = d[['cellid','phototag']].groupby('cellid').max()

dpred = nd.batch_comp(batch=batch, modelnames=modelnames, stat="r_ceiling")
dpred.columns = shortnames
dpred['siteid'] = dpred.index
dpred['siteid']=dpred['siteid'].apply(nd.get_siteid)
dpred['diff'] = dpred[shortnames[1]]-dpred[shortnames[0]]

dpred = dpred.merge(dtag, how='inner', left_index=True, right_index=True)

dpc = pd.read_csv(pca_file, index_col=0)
dpred = dpred.merge(dpc[['cellid','pc1','pc2','pc3','pc4','pc5']],
                    how='inner', left_index=True, right_on='cellid')

dm = dpred.groupby(['siteid','phototag']).mean()
#dmc = dpred.groupby(['siteid','phototag']).count()
#dm = dm[['diff']].merge(dmc['label'],how='inner',left_index=True, right_index=True)


recfile = '/auto/data/nems_db/recordings/334/NAT4_ozgf.fs100.ch18.tgz'
tctx = {'rec': load_recording(recfile)}
tctx.update(split_pop_rec_by_mask(**tctx))
a_chans = dpred.loc[dpred['phototag']=='a','cellid'].to_list()
s_chans = dpred.loc[dpred['phototag']=='s','cellid'].to_list()

aresp=tctx['val'].apply_mask()['resp'].extract_channels(a_chans)
sresp=tctx['val'].apply_mask()['resp'].extract_channels(s_chans)

mean_resp = tctx['val'].apply_mask()['resp']._data.mean(axis=1)
dr = pd.DataFrame({'cellid': tctx['val']['resp'].chans,
                   'mean_resp': mean_resp})
dpred = dpred.merge(dr, how='left', left_on='cellid', right_on='cellid')

cmax=mean_resp.max()
f,ax = plt.subplots(3,1)
ax[0].imshow(aresp._data, aspect='auto',interpolation='none', clim=[0,cmax])
ax[1].imshow(sresp._data, aspect='auto',interpolation='none', clim=[0,cmax])
ax[2].plot(aresp._data.mean(axis=0),label='a')
ax[2].plot(sresp._data.mean(axis=0),label='s')
ax[2].legend()


from seaborn import scatterplot, barplot, histplot

_d = dpred.loc[(dpred['phototag']=='s') | (dpred['phototag']=='a')].reset_index()
_d['hi_resp']=_d['mean_resp']>0.2

_dm = _d.groupby(['siteid','phototag']).mean()

f, ax = plt.subplots(2, 4, figsize=(12,6))
ax = ax.flatten()

ax[0].plot([0.0,1], [0.0,1],'k--')
scatterplot(data=_d, x=shortnames[0], y=shortnames[1], hue='phototag', ax=ax[0]);
plt.setp(ax[0].get_legend().get_texts(), fontsize='10')

histplot(data=_d, x=shortnames[0], hue='phototag', ax=ax[1], hue_order=['s','a'])
plt.setp(ax[1].get_xticklabels(), fontsize='10'); #, frameon=False)
ax[1].set_ylabel(f'r_test ({shortnames[0]})',fontsize=10)

histplot(data=_d, x=shortnames[1], hue='phototag', ax=ax[2], hue_order=['s','a'])
plt.setp(ax[2].get_xticklabels(), fontsize='10'); #, frameon=False)
ax[1].set_ylabel(f'r_test ({shortnames[1]})',fontsize=10)

histplot(data=_d, x='diff', hue='phototag', ax=ax[3], hue_order=['s','a'])
plt.setp(ax[3].get_xticklabels(), fontsize='10'); #, frameon=False)
ax[1].set_ylabel(f'r_test ({"diff"})', fontsize=10)

#ax[4].plot([0.0,1], [0.0,1],'k--')
scatterplot(data=_d, x='pc1', y='diff', hue='phototag', ax=ax[4]);
plt.setp(ax[4].get_legend().get_texts(), fontsize='10')

histplot(data=_d, x='pc1', hue='phototag', ax=ax[5], hue_order=['s','a'])
plt.setp(ax[5].get_xticklabels(), fontsize='10')

scatterplot(data=_d, x='mean_resp', y=shortnames[1], hue='phototag', ax=ax[6]);
plt.setp(ax[6].get_legend().get_texts(), fontsize='10')
scatterplot(data=_d, x='mean_resp', y='diff', hue='phototag', ax=ax[7])
plt.setp(ax[7].get_legend().get_texts(), fontsize='10')

#barplot(data=_dm.reset_index(), x='siteid', y='diff', hue='phototag', ax=ax[7], hue_order=['s','a'])
#plt.setp(ax[7].get_xticklabels(), fontsize='10'); #, frameon=False)
#ax[7].set_ylabel(f'r difference ({shortnames[1]}-{shortnames[0]})',fontsize=10)

f.suptitle(modelnames[1])



