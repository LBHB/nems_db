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

save_fig = False
if save_fig:
    plt.close('all')

outpath = "/auto/users/svd/docs/current/two_band_spn/eps/"



batch = 259

# output nonlinearities

modelnames=[
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-relu.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-qsig.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-logsig.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-relu.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-qsig.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-logsig.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b"
    ]

#   "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-fir.3x15-lvl.1-dexp.1_init-basic",

modelnames_fitter=[
    "env.fs100-ld-sev_dlog-fir.2x15-lvl.1-dexp.1_init-basic",
    "env.fs100-ld-sev_dlog-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init-basic",
    "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-fir.3x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init-basic",
    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b"
]

xc_range = [-0.05, 1.1]
n1="env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1_init.r10-basic.b"
n2="env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1_init.r10-basic.b"

df = nd.batch_comp(batch, modelnames, stat='r_ceiling')
df_r = nd.batch_comp(batch, modelnames, stat='r_test')
df_e = nd.batch_comp(batch, modelnames, stat='se_test')
df_n = nd.batch_comp(batch, modelnames, stat='n_parms')
cellcount = len(df)
modelcount = len(modelnames)

beta1_test = df_r[n1]
beta2_test = df_r[n2]
se1 = df_e[n1]
se2 = df_e[n2]

# test for signficant prediction at all
goodcells = ((beta2_test > se2*3) | (beta1_test > se1*3))

fh, ax = plt.subplots(2, 2, figsize=(9, 9))
m = np.array((df.loc[goodcells]**2).mean()[modelnames])
ax[0,0].bar(np.arange(modelcount), m, color='black')
ax[0,0].plot(np.array([-1, modelcount]), np.array([0, 0]), 'k--')
ax[0,0].set_ylim((-.05, 0.9))
ax[0,0].set_title("batch {}, n={}/{} good cells".format(
        batch, np.sum(goodcells), len(goodcells)))
ax[0,0].set_ylabel('median pred corr')
ax[0,0].set_xlabel('model architecture')
nplt.ax_remove_box(ax[0,0])

for i in range(modelcount-1):
    if i < 4:
        d1 = np.array(df[modelnames[i]])
        d2 = np.array(df[modelnames[4]])
        s, p = ss.wilcoxon(d1, d2)
        ax[0,0].text(i, m[i + 1] + 0.03, "{:.1e}".format(p), ha='center', fontsize=6)
    elif i>4:
        d1 = np.array(df[modelnames[i]])
        d2 = np.array(df[modelnames[9]])
        s, p = ss.wilcoxon(d1, d2)
        ax[0,0].text(i, m[i + 1] + 0.03, "{:.1e}".format(p), ha='center', fontsize=6)

ax[0,0].set_xticks(np.arange(len(m)))
ax[0,0].set_xticklabels(np.round(m, 3))

modelgroups = np.zeros(modelcount)
modelgroups[['stp' in m for m in modelnames]] = 1
lplt.model_comp_pareto(modelnames, batch, modelgroups=modelgroups,
                       goodcells=goodcells, ax=ax[0,1])


#fit/parameterization comparison
# inherit goodcells from above
# cellid alphabetized, so mapping to new dataframes should be fine.

df = nd.batch_comp(batch, modelnames_fitter, stat='r_ceiling')
df_n = nd.batch_comp(batch, modelnames_fitter, stat='n_parms')
cellcount = len(df)
modelcount = len(modelnames_fitter)

m = np.array((df.loc[goodcells]**2).mean()[modelnames_fitter])
ax[1,0].bar(np.arange(modelcount), m, color='black')
ax[1,0].plot(np.array([-1, modelcount]), np.array([0, 0]), 'k--')
ax[1,0].set_ylim((-.05, 0.9))
ax[1,0].set_ylabel('median r ceiling')
ax[1,0].set_xlabel('model architecture')
nplt.ax_remove_box(ax[1,0])

for i in range(modelcount-1):
    if i < 3:
        d1 = np.array(df[modelnames_fitter[i]])
        d2 = np.array(df[modelnames_fitter[3]])
        s, p = ss.wilcoxon(d1, d2)
        ax[1,0].text(i, m[i + 1] + 0.03, "{:.1e}".format(p), ha='center', fontsize=6)
    elif i>3:
        d1 = np.array(df[modelnames_fitter[i]])
        d2 = np.array(df[modelnames_fitter[-1]])
        s, p = ss.wilcoxon(d1, d2)
        ax[1,0].text(i, m[i + 1] + 0.03, "{:.1e}".format(p), ha='center', fontsize=6)

ax[1,0].set_xticks(np.arange(len(m)))
ax[1,0].set_xticklabels(np.round(m, 3))

modelgroups = np.zeros(modelcount)
modelgroups[['stp' in m for m in modelnames_fitter]] = 1
lplt.model_comp_pareto(modelnames_fitter, batch, modelgroups=modelgroups,
                       goodcells=goodcells, ax=ax[1,1])



if save_fig:
    batchstr = str(batch)
    fh.savefig(outpath + "fig10.NL_controls_batch"+batchstr+".pdf")
