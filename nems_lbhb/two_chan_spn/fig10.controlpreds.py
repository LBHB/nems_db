import os
import sys
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.disabled = True

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems.db as nd
import nems_db.params
import numpy as np
import pandas as pd
import scipy.stats as ss

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.modelspec as ms
import nems.xforms as xforms
#import nems_lbhb.xform_wrappers as nw
import nems.db as nd
import nems.plots.api as nplt
from nems.utils import find_module


outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev2/"
save_fig = True
if save_fig:
    plt.close('all')

batch = 259

# output nonlinearities

modelnames=[
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-relu.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-logsig.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x5.c-do.5x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-relu.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-logsig.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-dexp.1_init.r10-basic.b"
    ]
_pre = "env.fs100-ld-sev_"
_suf = "_init.r10.b-basic"
modelnames=[
    _pre + "dlog-wc.2x5.c-do.5x15-lvl.1" + _suf,
    _pre + "dlog-wc.2x5.c-do.5x15-lvl.1-relu.1.b" + _suf,
    _pre + "dlog-wc.2x5.c-do.5x15-lvl.1-logsig.1" + _suf,
    _pre + "dlog-wc.2x5.c-do.5x15-lvl.1-dexp.1" + _suf,
    _pre + "dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1" + _suf,
    _pre + "dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-relu.1.b" + _suf,
    _pre + "dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-logsig.1" + _suf,
    _pre + "dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-dexp.1" + _suf
    ]

#   "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-fir.3x15-lvl.1-dexp.1_init-basic",

modelnames_fitter=[
    "env.fs100-ld-sev_dlog-fir.2x15-lvl.1-dexp.1_init-basic",
    "env.fs100-ld-sev_dlog-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init-basic",
    "env.fs100-ld-sev_dlog-wc.2x5.c-do.5x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-fir.3x15-lvl.1-dexp.1_init-basic",
    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-fir.3x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init-basic",
    "env.fs100-ld-sev_dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-dexp.1_init.r10-basic.b"
]

xc_range = [-0.05, 1.1]

n1=modelnames[3]
n2=modelnames[7]
nlogsig=modelnames[6]

df = nd.batch_comp(batch, modelnames, stat='r_ceiling')
df_r = nd.batch_comp(batch, modelnames, stat='r_test')
df_e = nd.batch_comp(batch, modelnames, stat='se_test')
df_n = nd.batch_comp(batch, modelnames, stat='n_parms')
cellcount = len(df)
modelcount = len(modelnames)

beta1_test = df_r[n1].copy()
beta2_test = df_r[n2].copy()

df_r[df_r==0] = np.nan
df[df==0] = np.nan

betalogsig_test = df_r[nlogsig]
se1 = df_e[n1]
se2 = df_e[n2]

# test for significant prediction at all
goodcells = ((beta2_test > se2*3) | (beta1_test > se1*3))

fh, ax = plt.subplots(2, 2, figsize=(9, 9))

offset = 0.5
m = np.array(np.nanmean(df.loc[goodcells], axis=0))

ax[0, 0].bar(np.arange(modelcount), m-offset, color='black', bottom=offset)
ax[0, 0].plot(np.array([-1, modelcount]), np.array([offset, offset]), 'k--')
ax[0, 0].set_ylim((offset-0.05, 0.85))

ax[0, 0].set_title("batch {}, n={}/{} good cells".format(
        batch, np.sum(goodcells), len(goodcells)))
ax[0, 0].set_ylabel('Mean prediction correlation')
ax[0, 0].set_xlabel('Model architecture')
nplt.ax_remove_box(ax[0, 0])

for i in range(modelcount-1):
    if i < 3:
        d1 = np.array(df[modelnames[i]])
        d2 = np.array(df[n1])
        ii = np.isfinite(d1) & np.isfinite(d2)
        s, p = ss.wilcoxon(d1[ii], d2[ii])
        ax[0, 0].text(i, m[i] + 0.02, "{:.1e}".format(p), ha='center', fontsize=6)
    elif i > 3:
        d1 = np.array(df[modelnames[i]])
        d2 = np.array(df[n2])
        ii = np.isfinite(d1) & np.isfinite(d2)
        s, p = ss.wilcoxon(d1[ii], d2[ii])
        ax[0, 0].text(i, m[i] + 0.02, "{:.1e}".format(p), ha='center', fontsize=6)

ax[0,0].set_xticks(np.arange(len(m)))
ax[0,0].set_xticklabels(np.round(m, 3))

modelgroups = {}
modelgroups['LN'] = [m  for m in modelnames if 'stp' not in m]
modelgroups['STP'] = [m  for m in modelnames if 'stp' in m]
lplt.model_comp_pareto(modelnames, batch, modelgroups=modelgroups,
                       goodcells=goodcells, ax=ax[0, 1])

ax[0, 1].text(26,0.55, '\n'.join(modelnames), fontsize=5)


#fit/parameterization comparison
# inherit goodcells from above
# cellid alphabetized, so mapping to new dataframes should be fine.

df = nd.batch_comp(batch, modelnames_fitter, stat='r_ceiling')
df_n = nd.batch_comp(batch, modelnames_fitter, stat='n_parms')
cellcount = len(df)
modelcount = len(modelnames_fitter)

m = np.array((df.loc[goodcells]).mean()[modelnames_fitter])
ax[1,0].bar(np.arange(modelcount), m-offset, color='black', bottom=offset)
ax[1,0].plot(np.array([-1, modelcount]), np.array([offset, offset]), 'k--')
ax[1,0].set_ylim((offset-0.05, 0.85))
ax[1,0].set_ylabel('Mean prediction correlation')
ax[1,0].set_xlabel('Model architecture')
nplt.ax_remove_box(ax[1,0])

for i in range(modelcount-1):
    if i < 3:
        d1 = np.array(df[modelnames_fitter[i]])
        d2 = np.array(df[modelnames_fitter[3]])
        s, p = ss.wilcoxon(d1, d2)
        ax[1, 0].text(i, m[i] + 0.02, "{:.1e}".format(p), ha='center', fontsize=6)
    elif i > 3:
        d1 = np.array(df[modelnames_fitter[i]])
        d2 = np.array(df[modelnames_fitter[-1]])
        s, p = ss.wilcoxon(d1, d2)
        ax[1, 0].text(i, m[i] + 0.02, "{:.1e}".format(p), ha='center', fontsize=6)

ax[1,0].set_xticks(np.arange(len(m)))
ax[1,0].set_xticklabels(np.round(m, 3))

modelgroups = {}
modelgroups['LN'] = [m  for m in modelnames_fitter if 'stp' not in m]
modelgroups['STP'] = [m  for m in modelnames_fitter if 'stp' in m]

lplt.model_comp_pareto(modelnames_fitter, batch, modelgroups=modelgroups,
                       goodcells=goodcells, ax=ax[1,1])
plt.text(25,0.55, '\n'.join(modelnames_fitter), fontsize=5)


if save_fig:
    batchstr = str(batch)
    fh.savefig(outpath + "fig10.NL_controls_batch"+batchstr+".pdf")
