import numpy as np
import os
import io
import logging

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join
import nems.db as nd
from nems import get_setting
from nems.xform_helper import _xform_exists
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems.gui.recording_browser import browse_recording, browse_context
import nems.gui.editors as gui

log = logging.getLogger(__name__)
# NAT A1 SINGLE NEURON + PUPIL
batch = 289
cellid = 'TAR009d-42-1'
modelname = "ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"

xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)

#aw = browse_context(ctx, rec='val', signals=['stim', 'pred', 'resp'])
#aw = browse_context(ctx, signals=['state', 'psth', 'pred', 'resp'])

ex = gui.browse_xform_fit(ctx, xfspec)
#ex.show()

#ctx['modelspec'].quickplot(range=(8500,9500))
