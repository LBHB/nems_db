import os
import sys
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.disabled = True

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems0.db as nd
import nems_db.params
import numpy as np
import pandas as pd
import scipy.stats as ss

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems0.recording as recording
import nems0.epoch as ep
import nems0.modelspec as ms
import nems0.xforms as xforms
#import nems_lbhb.xform_wrappers as nw
import nems0.db as nd
import nems0.plots.api as nplt
from nems0.utils import find_module

save_fig = False
if save_fig:
    plt.close('all')

outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev2/"

USE_SPN = True
if USE_SPN:
    batch = 259
    # this was used in the original submission
    modelnames=["env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-stp.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-wc.2x2.c.n-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic",
                "env.fs100-ld-sev_dlog.f-wc.2x4.c.n-stp.4-fir.4x15-lvl.1-dexp.1_init-basic"]
    # new DO models
    modelnames=["env.fs100-ld-sev_dlog-wc.2x5.c-do.5x15-lvl.1-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x4.c-relu.4-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-stp.1.s-dexp.1_init.r10-basic.b",
                "env.fs100-ld-sev_dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-dexp.1_init.r10-basic.b"
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
    fileprefix="fig4.SPN"
    n1=modelnames[0]
    n2=modelnames[-1]
    label1 = "Rank-4 LN"
    label2 = "Rank-5 STP"
elif 1:
    batch = 289
    modelnames = ["ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-fir.3x15-lvl.1-dexp.1_init-basic",
                  "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"]
    #modelnames = ["ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic",
    #              "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"]

    # DO
    modelnames = ["ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-do.4x15-lvl.1-dexp.1_init.r10.b-basic",
                "ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10.b-basic"]

    n1=modelnames[0]
    n2=modelnames[1]
    fileprefix="fig9.NAT"
    label1 = "Rank-4 LN"
    label2 = "Rank-4 STP"

xc_range = [-0.05, 1.1]

df = nd.batch_comp(batch, modelnames, stat='r_ceiling')
df_r = nd.batch_comp(batch, modelnames, stat='r_test')
df_e = nd.batch_comp(batch, modelnames, stat='se_test')
df_n = nd.batch_comp(batch, modelnames, stat='n_parms')

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

if USE_SPN:
    ng = modelnames[-2]
    betag_test = df_r[ng]
    seg = df_e[ng]
    improvedcellsg = (betag_test - seg > beta1_test + se1)

# test for significant prediction at all
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
if USE_SPN:
    offset = 0.5
    max = 0.85
else:
    offset = 0.4
    max = 0.65
m = np.array((df.loc[goodcells]).mean()[modelnames])
print('mean: ' + str(m))
print('median: ' + str(np.array((df.loc[goodcells]).median()[modelnames])))
ax[0].bar(np.arange(len(modelnames)), m-offset, color='black', bottom=offset)
ax[0].plot(np.array([-1, len(modelnames)]), np.array([offset, offset]), 'k--')
ax[0].set_ylim((offset-0.05, max))
ax[0].set_title("batch {}, n={}/{} good cells".format(
        batch, np.sum(goodcells), len(goodcells)))
ax[0].set_ylabel('Mean var explained (r2)')
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
    "env.fs100-ld-sev_dlog-pz.2x15-lvl.1-dexp.1_init.r10-basic.b",

    "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.q.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
    "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2.q-fir.2x15-lvl.1-dexp.1_init.r10-basic.b"
]

modelgroups={}
if USE_SPN:
    modelgroups['LN'] = [
        "env.fs100-ld-sev_dlog-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
        ]
    modelgroups['doLN'] = [
        "env.fs100-ld-sev_dlog-wc.2x1.c-do.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x2.c-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x5.c-do.5x15-lvl.1-dexp.1_init.r10-basic.b"
        ]
    modelgroups['relu-doLN'] = [
        "env.fs100-ld-sev_dlog-wc.2x1.c-relu.1-do.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x2.c-relu.2-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x3.c-relu.3-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x4.c-relu.4-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x5.c-relu.5-do.5x15-lvl.1-dexp.1_init.r10-basic.b"
        ]
    modelgroups['STPx-LN'] = [
        "env.fs100-ld-sev_dlog-wc.2x2.c-stp.1.x.s-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x3.c-stp.1.x.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x4.c-stp.1.x.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x5.c-stp.1.x.s-do.5x15-lvl.1-dexp.1_init.r10-basic.b"
        ]
    modelgroups['doL-STP-N'] = [
        "env.fs100-ld-sev_dlog-wc.2x1.c-do.1x15-lvl.1-stp.1.s-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x2.c-do.2x15-lvl.1-stp.1.s-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-stp.1.s-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-stp.1.s-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x5.c-do.5x15-lvl.1-stp.1.s-dexp.1_init.r10-basic.b"
        ]
    modelgroups['STP-LN'] = [
        "env.fs100-ld-sev_dlog-wc.2x1.c-stp.1.s-fir.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2.s-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-fir.3x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-fir.4x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x5.c-stp.5.s-fir.5x15-lvl.1-dexp.1_init.r10-basic.b"
        ]
    modelgroups['STP-doLN'] = [
        "env.fs100-ld-sev_dlog-wc.2x1.c-stp.1.s-do.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x2.c-stp.2.s-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
        "env.fs100-ld-sev_dlog-wc.2x5.c-stp.5.s-do.5x15-lvl.1-dexp.1_init.r10-basic.b",
        ]
else:
    modelgroups['LN'] = [
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init.r10-basic.b",
    ]
    modelgroups['do-LN'] = [
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-do.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
    ]
    modelgroups['STP'] = [
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-stp.1.s-fir.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2.s-fir.2x15-lvl.1-dexp.1_init.r10-basic.b",
    ]
    modelgroups['do-STP'] = [
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-stp.1.s-do.1x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2.s-do.2x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-stp.3.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b",
        "ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b",
    ]

_, b_ceiling = lplt.model_comp_pareto(batch=batch, modelgroups=modelgroups,
                       goodcells=goodcells, ax=ax[1])
ax[1].set_ylim(ax[0].get_ylim())

if save_fig:
    batchstr = str(batch)
    fh1.savefig(outpath + fileprefix + ".pred_scatter_batch"+batchstr+".pdf")
    fh2.savefig(outpath + fileprefix + ".pred_sum_bar_batch"+batchstr+".pdf")
