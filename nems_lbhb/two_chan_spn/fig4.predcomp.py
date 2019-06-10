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

if 1:
    batch = 259
    # this was used in the original submission
    modelnames=["env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-stp.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-wc.2x2.c.n-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-wc.2x4.c.n-stp.4-fir.4x15-lvl.1-dexp.1_init-basic"]
    # new do models
    modelnames=["env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x4.c-stp.1.x.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-dexp.1_init.r10-basic.b",
               ]
    # cleaner STP effects, predictions slightly worse
#    modelnames=["env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-stp.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog.f-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog.f-wc.2x2.c.n-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog.f-wc.2x4.c.n-stp.4-fir.4x15-lvl.1-dexp.1_init-basic"]
#    modelnames=["env.fs100-ld-sev_dlog-fir.2x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog-fir.2x15-lvl.1-stp.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3-fir.3x15-lvl.1-dexp.1_init-basic",
#                "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4-fir.4x15-lvl.1-dexp.1_init-basic"]
    fileprefix="fig5.SPN"
    n1=modelnames[1]
    n2=modelnames[-2]
    label1 = "Rank-4 LN"
    label2 = "Rank-4 STP"
elif 1:
    batch = 289
    modelnames = ["ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-fir.3x15-lvl.1-dexp.1_init-basic",
                  "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"]
    #modelnames = ["ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic",
    #              "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"]
    n1=modelnames[0]
    n2=modelnames[1]
    fileprefix="fig9.NAT"
    label1 = "Rank-2 LN"
    label2 = "Rank-3 STP"

xc_range = [-0.05, 1.1]

df = nd.batch_comp(batch,modelnames,stat='r_ceiling')
df_r = nd.batch_comp(batch,modelnames,stat='r_test')
df_e = nd.batch_comp(batch,modelnames,stat='se_test')
df_n = nd.batch_comp(batch,modelnames,stat='n_parms')

cellcount = len(df)

beta1 = df[n1]
beta2 = df[n2]
beta1_test = df_r[n1]
beta2_test = df_r[n2]
se1 = df_e[n1]
se2 = df_e[n2]

beta1[beta1>1]=1
beta2[beta2>1]=1

# test for significant improvement
improvedcells = (beta2_test-se2 > beta1_test+se1)

# test for signficant prediction at all
goodcells = ((beta2_test > se2*3) | (beta1_test > se1*3))

fh1 = stateplots.beta_comp(beta1[goodcells], beta2[goodcells],
                           n1=label1, n2=label2,
                           hist_range=xc_range,
                           highlight=improvedcells[goodcells])
#fh1 = stateplots.beta_comp(beta1, beta2,
#                           n1='LN STRF', n2='RW3 STP STRF',
#                           hist_range=xc_range,
#                           highlight=improvedcells)

fh2, ax = plt.subplots(1, 2, figsize=(7, 3))
m = np.array((df.loc[goodcells]**2).mean()[modelnames])
ax[0].bar(np.arange(len(modelnames)), m, color='black')
ax[0].plot(np.array([-1, len(modelnames)]), np.array([0, 0]), 'k--')
ax[0].set_ylim((-.05, 0.9))
ax[0].set_title("batch {}, n={}/{} good cells".format(
        batch, np.sum(goodcells), len(goodcells)))
ax[0].set_ylabel('median pred corr')
ax[0].set_xlabel('model architecture')
nplt.ax_remove_box(ax[0])


for i in range(len(modelnames)-1):

    d1 = np.array(df[modelnames[i]])
    d2 = np.array(df[modelnames[i+1]])
    s, p = ss.wilcoxon(d1, d2)
    ax[0].text(i+0.5, m[i+1]+0.03, "{:.1e}".format(p), ha='center', fontsize=6)

ax[0].set_xticks(np.arange(len(m)))
ax[0].set_xticklabels(np.round(m,3))

b_modelnames = [
    "env.fs100-ld-sev_dlog-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-pz.2x15-lvl.1-dexp.1_init.r10-basic.b",

    "env.fs100-ld-sev_dlog-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x1.c-do.1x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x2.c-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x5.c-do.5x15-lvl.1-dexp.1_init.r10-basic.b",

    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.1.x.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.1.x.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",

    "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2.s-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",

    "env.fs100-ld-sev_dlog-wc.2x1.c-stp.2.s-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2.s-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-dexp.1_init.r10-basic.b",

    "env.fs100-ld-sev_dlog-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.q.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2.q-fir.2x15-lvl.1-dexp.1_init.r10-basic.b"
]



b_stp = np.array([('stp' in m) and ('.x.' not in m) for m in b_modelnames])
b_stp1 = np.array([('stp' in m) and ('.x.' in m) for m in b_modelnames])
b_ln = ~(b_stp | b_stp1)

modelgroups=np.zeros(len(b_modelnames))
modelgroups[b_stp1] = 1
modelgroups[b_stp] = 2
lplt.model_comp_pareto(b_modelnames, batch, modelgroups=modelgroups,
                       goodcells=goodcells, ax=ax[1])





if save_fig:
    batchstr = str(batch)
    fh1.savefig(outpath + fileprefix + ".pred_scatter_batch"+batchstr+".pdf")
    fh2.savefig(outpath + fileprefix + ".pred_sum_bar_batch"+batchstr+".pdf")
