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

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems.recording as recording
import nems.epoch as ep
import nems.xforms as xforms
#import nems_lbhb.xform_wrappers as nw
import nems.db as nd
import nems.plots.api as nplt
from nems.utils import find_module

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

batch = 259

#modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-mt.shr-basic"
modelname1 = "env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic"
modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"
#modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"
#modelname2 = "env.fs100-ld-sev_dlog.f-wc.2x2.c.n-stp.2-fir.2x15-lvl.1-dexp.1_init-basic"

# parametric temporal filter - 4 chan
modelname1 = "env.fs100-ld-sev_dlog-wc.2x4.c-do.4x15-lvl.1-dexp.1_init.r10-basic.b"
modelname2 = "env.fs100-ld-sev_dlog-wc.2x4.c-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10-basic.b"

# 3 chan
modelname1 = "env.fs100-ld-sev_dlog-wc.2x3.c-do.3x15-lvl.1-dexp.1_init.r10-basic.b"
modelname2 = "env.fs100-ld-sev_dlog-wc.2x3.c-stp.3.s-do.3x15-lvl.1-dexp.1_init.r10-basic.b"

save_figs = False
outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev2/"
#if save_figs:
plt.close('all')

#cellid="por077a-c1"
#cellid = "chn003c-a1"
cellid = "eno009d-a1"
cellid = "eno027d-c1"
cellid = "eno029d-c1"
cellid = "por074b-d2"
cellid = "por074b-c2"
cellid = "por020a-c1"

fh1, ctx1, ctx2 = lplt.compare_model_preds(cellid, batch, modelname1, modelname2)

# xf2, ctx2 = lplt.get_model_preds(cellid, batch, modelname2)
fh2 = nplt.diagnostic(ctx2, pre_dur=0.25, dur=1.5)

if save_figs:
    print("saving figures...")
    fh1.savefig(outpath + "fig1.detailed_response_" + cellid + ".pdf")
    fh2.savefig(outpath + "fig4.model_steps_" + cellid + ".pdf")
