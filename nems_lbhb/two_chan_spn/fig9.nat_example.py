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
import scipy.stats as ss

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems0.recording as recording
import nems0.epoch as ep
import nems0.xforms as xforms
from nems0.utils import find_module, ax_remove_box
from nems0.metrics.stp import stp_magnitude
from nems0.modules.weight_channels import gaussian_coefficients
from nems0.modules.fir import da_coefficients
from nems0.xform_helper import load_model_xform
import nems0.plots.api as nplt

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

# start main code
outpath = "/auto/users/svd/docs/current/two_band_spn/eps_rev2/"
fileprefix="fig9.NAT"

save_fig = True
if save_fig:
    plt.close('all')

batch=289
modelnames = ["ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-do.4x15-lvl.1-dexp.1_init.r10.b-basic",
              "ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-stp.4.s-do.4x15-lvl.1-dexp.1_init.r10.b-basic"]

cellid="AMT005c-02-2"
cellid="TAR017b-33-2"
cellid="TAR010c-24-2"
cellid="TAR010c-38-2"
cellid="AMT003c-33-1"
cellid="bbl104h-12-1"

xf1, ctx1 = load_model_xform(batch=batch, modelname=modelnames[0], cellid=cellid)
xf2, ctx2 = load_model_xform(batch=batch, modelname=modelnames[1], cellid=cellid)
fh, ctx1, ctx2 = lplt.compare_model_preds(cellid, batch, modelnames[0], modelnames[1],
                                          max_pre=0.5, max_dur=4, stim_ids=[0,1],
                                          ctx1=ctx1, ctx2=ctx2)

#xf,ctx1=load_model_xform(batch=batch, modelname=modelnames[0], cellid=cellid)
#xf,ctx2=load_model_xform(batch=batch, modelname=modelnames[1], cellid=cellid)

#fh1=nplt.quickplot(ctx1)
#fh2=nplt.quickplot(ctx2)

if save_fig:
    fh.savefig(outpath + fileprefix + "_example_"+cellid+"_"+"modelcomp_batch_"+str(batch)+".pdf")


