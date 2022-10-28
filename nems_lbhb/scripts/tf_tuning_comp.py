import os
import sys
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.disabled = True

import nems0.db as nd
import nems_db.params
import numpy as np
import scipy.stats as ss

import nems_lbhb.stateplots as stateplots
import nems_lbhb.plots as lplt
import nems0.recording as recording
import nems0.epoch as ep
import nems0.xforms as xforms
import nems_lbhb.xform_wrappers as nw
import nems0.db as nd
import nems0.plots.api as nplt
from nems0.utils import find_module, ax_remove_box
from nems0.metrics.stp import stp_magnitude
from nems0.modules.weight_channels import gaussian_coefficients

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

dotcolor = 'black'
dotcolor_ns = 'lightgray'
thinlinecolor = 'gray'
barcolors = [(235/255, 47/255, 40/255), (115/255, 200/255, 239/255)]
barwidth = 0.5


batch = 289
modelname = "ozgf.fs100.ch18-ld-norm-sev_wc.18x2.g-fir.2x15-relu.1_tf"
modelname0 = "ozgf.fs100.ch18-ld-norm-sev_wc.18x2.g-fir.2x15-relu.1_init-basic"
d = nems_db.params.fitted_params_per_batch(batch, modelname,
                                           stats_keys=[], multi='first')
d0 = nems_db.params.fitted_params_per_batch(batch, modelname0,
                                           stats_keys=[], multi='first')

g = d[d.index == '0--wc.18x2.g--mean'].T
g['mean0'] = d0[d0.index == '0--wc.18x2.g--mean'].T
g['std'] = d[d.index == '0--wc.18x2.g--sd'].T
g['std0'] = d0[d0.index == '0--wc.18x2.g--sd'].T
g.columns = ['mean', 'mean0', 'std', 'std0']
