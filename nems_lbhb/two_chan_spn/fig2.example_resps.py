import os
import sys
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.disabled = True

#sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))
import nems_db.params
import numpy as np

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
#import nems.xform_helper as xhelp
#import nems_lbhb.xform_wrappers as nw
import nems.db as nd
import nems.plots.api as nplt
from nems.utils import find_module

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 6,
          'axes.titlesize': 6,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

batch = 259

# shrinkage
modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-mt.shr-basic"
# regular
modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

modelnames = [modelname1, modelname2]
df = nd.batch_comp(batch, modelnames)
df['diff'] = df[modelname2] - df[modelname1]
df.sort_values('cellid', inplace=True, ascending=True)
m = df.index.str.startswith('por07') & (df[modelname2] > 0.3)
df['cellid'] = df.index

for index, c in df[m].iterrows():
    print("{}  {:.3f} - {:.3f} = {:.3f}".format(
            index, c[modelname2], c[modelname1], c['diff']))

plt.close('all')
outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev/"

savefig = False

if 1:
    #cellid="por077a-c1"
    cellid = "por074b-d2"
    cellid = "por020a-c1"
    fh, ctx2 = lplt.compare_model_preds(cellid, batch, modelname1, modelname2);
    #xf1, ctx1 = lplt.get_model_preds(cellid, batch, modelname1)
    #xf2, ctx2 = lplt.get_model_preds(cellid, batch, modelname2)
    #nplt.diagnostic(ctx2);
    if savefig:
        fh.savefig(outpath + "fig1_model_preds_" + cellid + ".pdf")


elif 0:
    for cellid, c in df[m].iterrows():
        fh = lplt.compare_model_preds(cellid,batch,modelname1,modelname2);
        #fh.savefig(outpath + "fig1_model_preds_" + cellid + ".pdf")

