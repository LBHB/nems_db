import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import nems.db as nd
from pop_model_utils import get_significant_cells, SIG_TEST_MODELS, PLOT_STAT


# tentative: use best conv1dx2+d
half_test_modelspec = "wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R"

# test condition: take advantage of larger model population fit (hs: heldout), then fit single cell with half a dataset
# fit last layer on half the data, using prefit with the current site held-out, run per cell
modelname_half_prefit=[  # heldout, 100% of est data, then second stage fit X% of est data on single cell
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k10_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k15_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k25_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k50_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev_{half_test_modelspec}_prefit.hm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
]

# then fit last layer on heldout cell with half the data (same est data as for modelname_half_prefit), run per cell
modelname_half_fullfit=[  #
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k10_{half_test_modelspec}_prefit.htm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k15_{half_test_modelspec}_prefit.hfm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k25_{half_test_modelspec}_prefit.hqm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev.k50_{half_test_modelspec}_prefit.hhm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    f"ozgf.fs100.ch18-ld-norm.l1-sev_{half_test_modelspec}_prefit.hm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
]

batch = 322
sig_cells=get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

d = []
pcts = ['10', '15', '25', '50', '100']
mdls = ['LN', 'dnns', 'std', 'prefit']

# don't used heldout model for final fit
_modelname_half_prefit = modelname_half_prefit
_modelname_half_prefit[-1] = modelname_half_fullfit[-1]

pre = nd.batch_comp(batch, _modelname_half_prefit, cellids=sig_cells, stat=PLOT_STAT)
full = nd.batch_comp(batch, modelname_half_fullfit, cellids=sig_cells, stat=PLOT_STAT)

for n, h, m in zip(pcts, modelname_half_prefit, modelname_half_fullfit):
    d_ = full.loc[:, [m]]
    d_.columns = [PLOT_STAT]
    d_['midx'] = n
    d_['fit'] = "std"
    d.append(d_)

    d_ = pre.loc[:, [h]]
    d_.columns = [PLOT_STAT]
    d_['midx'] = n
    d_['fit'] = "prefit"
    d.append(d_)

dpred = pd.concat(d)
f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].plot([0, 1], [0, 1], 'k--')
x1, x2 = '10', 'std'
y1, y2 = '10', 'prefit'
x = dpred.loc[(dpred.midx == x1) & (dpred.fit == x2), 'r_ceiling']
y = dpred.loc[(dpred.midx == y1) & (dpred.fit == y2), 'r_ceiling']
sns.scatterplot(x=x, y=y, ax=ax[0])

ax[0].set_xlabel(f"{x2} {x1}% {x.median():.3f}")
ax[0].set_ylabel(f"{y2} {y1}% {y.median():.3f}")
ax[0].set_title(f'batch {batch} {x2} vs {y2}')
ax[0].set_xlim([-0.05, 1.05])
ax[0].set_ylim([-0.05, 1.05])

dpm = dpred.groupby(['midx', 'fit']).mean().reset_index()
dpm.midx = dpm.midx.astype(int)
dpm = dpm.pivot(index='midx', columns='fit', values='r_ceiling')
dpm.plot(ax=ax[1])
print(dpm)
