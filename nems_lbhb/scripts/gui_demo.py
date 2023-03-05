import numpy as np
import os
import io
import logging
import copy

import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join
import nems0.db as nd
from nems import get_setting
from nems0.xform_helper import _xform_exists
from nems0.registry import KeywordRegistry
from nems0.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems0.gui.recording_browser import browse_recording, browse_context
import nems0.gui.editors as gui

log = logging.getLogger(__name__)
# NAT A1 SINGLE NEURON + PUPIL
batch = 289
cellid = 'TAR009d-42-1'
modelname = "ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x3.g-stp.3-fir.3x15-lvl.1-dexp.1_init-basic"


batch, cellid = 308, 'AMT018a-09-1'
modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-fir.2x15x2-relu.2-wc.2x1-lvl.1-dexp.1_tf.n.rb10'
modelname2 = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x4.g-fir.2x15x2-relu.2-wc.2x1-lvl.1-dexp.1_tf.n.rb5'
modelname2 = None

GUI = True

xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)

if GUI:
    # interactive model browser (matplotlib embedded Qt)
    ex = gui.browse_xform_fit(ctx, xfspec)

    if modelname2 is not None:
        xfspec2, ctx2 = xhelp.load_model_xform(cellid, batch, modelname2)
        ex2 = gui.browse_xform_fit(ctx2, xfspec2, control_widget=ex.editor.global_controls)

    #aw = browse_context(ctx, rec='val', signals=['stim', 'pred', 'resp'])
    #aw = browse_context(ctx, signals=['state', 'psth', 'pred', 'resp'])

else:
    # static model summary plot (matplotlib native)
    #ctx['modelspec'][1]['plot_fn_idx']=2
    time_range = (0,5)

    fig=ctx['modelspec'].quickplot(time_range=time_range, sig_names=['fg_comp', 'bg_comp'],
                               modidx_set=[1, 2, 3, 5])

    #fig.savefig('/auto/users/svd/docs/current/RDT/nems/{}-{}-{}-T{}-{}.pdf'.format(
    #    cellid,batch,modelname, time_range[0], time_range[1]))

