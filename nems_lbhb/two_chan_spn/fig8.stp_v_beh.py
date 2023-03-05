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
from nems0.utils import find_module, ax_remove_box

save_fig = False
#if save_fig:
plt.close('all')

outpath = "/auto/users/svd/docs/current/two_band_spn/eps/"

batch = 274
batch2 = 275
modelnames=["env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
            "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
            "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
            "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-dexp.1_jk.nf5-init.st-basic",
            "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-fir.1x15-lvl.1-rep.2-dexp.2-mrg_jk.nf5-init.st-basic",
            "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-stp.1-rep.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic",
            "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-rep.2-stp.2-fir.1x15x2-lvl.2-dexp.2-mrg_jk.nf5-init.st-basic"]
#modelnames=["env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-stp.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-wc.2x3.c-stp.3-fir.3x15-lvl.1-dexp.1_init-basic",
#            "env.fs100-ld-sev_dlog.f-wc.2x4.c-stp.4-fir.4x15-lvl.1-dexp.1_init-basic"]
fileprefix="fig8.stp_v_beh"
n1=modelnames[0]
n2=modelnames[-3]

xc_range = [-0.05, 0.6]

df1 = nd.batch_comp(batch,modelnames,stat='r_test').reset_index()
df1_e = nd.batch_comp(batch,modelnames,stat='se_test').reset_index()

df2 = nd.batch_comp(batch2,modelnames,stat='r_test').reset_index()
df2_e = nd.batch_comp(batch2,modelnames,stat='se_test').reset_index()

df = df1.append(df2)
df_e = df1_e.append(df2_e)

cellcount = len(df)

beta1 = df[n1]
beta2 = df[n2]
beta1_test = df[n1]
beta2_test = df[n2]
se1 = df_e[n1]
se2 = df_e[n2]

beta1[beta1>1]=1
beta2[beta2>1]=1

# test for significant improvement
improvedcells = (beta2_test-se2 > beta1_test+se1)

# test for signficant prediction at all
goodcells = ((beta2_test > se2*3) | (beta1_test > se1*3))

fh = plt.figure()
ax = plt.subplot(2,2,1)
stateplots.beta_comp(beta1[goodcells], beta2[goodcells],
                     n1='LN STRF', n2='STP+BEH LN STRF',
                     hist_range=xc_range, ax=ax,
                     highlight=improvedcells[goodcells])


# LN vs. STP:
beta1b = df[modelnames[3]]
beta1a = df[modelnames[0]]
beta1 = beta1b - beta1a
se1a = df_e[modelnames[3]]
se1b= df_e[modelnames[0]]

b1=1
b0=0
beta2b = df[modelnames[b1]]
beta2a = df[modelnames[b0]]
beta2 = beta2b - beta2a
se2a = df_e[modelnames[b1]]
se2b= df_e[modelnames[b0]]

stpgood = (beta1 > se1a+se1b)
behgood = (beta2 > se2a+se2b)
neither_good = np.logical_not(stpgood) & np.logical_not(behgood)
both_good = stpgood & behgood
stp_only_good = stpgood & np.logical_not(behgood)
beh_only_good = np.logical_not(stpgood) & behgood

xc_range = np.array([-0.05, 0.15])
beta1[beta1<xc_range[0]]=xc_range[0]
beta2[beta2<xc_range[0]]=xc_range[0]

zz = np.zeros(2)
ax=plt.subplot(2,2,2)
ax.plot(xc_range,zz,'k--',linewidth=0.5)
ax.plot(zz,xc_range,'k--',linewidth=0.5)
ax.plot(xc_range, xc_range, 'k--', linewidth=0.5)
l = ax.plot(beta1[neither_good], beta2[neither_good], '.', color='lightgray') +\
    ax.plot(beta1[beh_only_good], beta2[beh_only_good], '.', color='purple') +\
    ax.plot(beta1[stp_only_good], beta2[stp_only_good], '.', color='orange') +\
    ax.plot(beta1[both_good], beta2[both_good], '.', color='black')
ax_remove_box(ax)
ax.set_aspect('equal', 'box')
#plt.axis('equal')
ax.set_xlim(xc_range)
ax.set_ylim(xc_range)
ax.set_xlabel('delta(stp)')
ax.set_ylabel('delta(beh)')

olap=np.zeros(100)
a = stpgood.values.copy()
b = behgood.values.copy()
for i in range(100):
    np.random.shuffle(a)
    olap[i] = np.sum(a & b)

ll=[np.sum(neither_good), np.sum(beh_only_good),
    np.sum(stp_only_good), np.sum(both_good)]
ax.legend(l, ll)

ax=plt.subplot(2,2,3)
m = np.array(df.loc[goodcells].mean()[modelnames])
xc_range = [-0.02, 0.2]
plt.bar(np.arange(len(modelnames)), m, color='black')
plt.plot(np.array([-1, len(modelnames)]), np.array([0, 0]), 'k--',
         linewidth=0.5)
plt.ylim(xc_range)
plt.title("batch {}, n={}/{} good cells".format(
        batch, np.sum(goodcells), len(goodcells)))
plt.ylabel('median pred corr')
plt.xlabel('model architecture')
ax_remove_box(ax)

for i in range(len(modelnames)-1):

    d1 = np.array(df[modelnames[i]])
    d2 = np.array(df[modelnames[i+1]])
    s, p = ss.wilcoxon(d1, d2)
    plt.text(i+0.5, m[i+1], "{:.1e}".format(p), ha='center', fontsize=6)

plt.xticks(np.arange(len(m)),np.round(m,3))

if save_fig:
    batchstr = str(batch)
    fh1.savefig(outpath + fileprefix + ".pred_scatter_batch"+batchstr+".pdf")
    fh2.savefig(outpath + fileprefix + ".pred_sum_bar_batch"+batchstr+".pdf")
