import numpy as np
import os
import io
import logging
import matplotlib.pyplot as plt

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

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

## RDT stream dependent STRFs
#batch, cellid = 269, 'sti032b-b3'
#batch, cellid = 269, 'oys042c-d1'
#batch, cellid = 273, 'chn041d-b1'
#batch, cellid = 273, 'zee027b-c1'
batch, cellid = 269, 'sti016a-a1'
batch, cellid = 269, 'chn008c-b1'
batch, cellid = 269, 'oys025b-a1'
batch, cellid = 269, 'chn019a-a1'
batch, cellid = 269, 'chn008a-c2'

modelname='rdtld-rdtshf-rdtsev.j.10-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic'
modelname='rdtld-rdtshf.str-rdtsev.j.10-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x3.g-do.3x15-lvl.1_init-basic'
modelname='rdtld-rdtshf-rdtsev.j.10-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x3.g-do.3x15-lvl.1_init-basic'

modelname='rdtld-rdtshf-rdtsev.j.10.ns-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic'
modelname='rdtld-rdtshf.str-rdtsev.j.10.ns-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic'

modelname='rdtld-rdtshf-rdtsev.j.10.ns-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x3.g-do.3x15-lvl.1-dexp.1_init.rb5-basic'
modelname='rdtld-rdtshf.rep.str-rdtsev.j.10.ns-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x3.g-do.3x15-lvl.1-dexp.1_init.rb5-basic'
modelname='rdtld-rdtshf.str-rdtsev.j.10.ns-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x3.g-do.3x15-lvl.1-dexp.1_init.rb5-basic'

# TODO: develop GUI to pick and load (xfspec, ctx)
xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)

GUI = True

if GUI:
    # interactive model browser (matplotlib embedded Qt)
    ex = gui.browse_xform_fit(ctx, xfspec)


    #aw = browse_context(ctx, rec='val', signals=['stim', 'pred', 'resp'])
    #aw = browse_context(ctx, signals=['state', 'psth', 'pred', 'resp'])

else:
    # static model summary plot (matplotlib native)
    ctx['modelspec'][1]['plot_fn_idx']=2
    #time_range = (4.5, 9)
    #time_range = (23, 26)
    #time_range = (44.5, 49) # ok response
    #time_range = (69.25, 69.25+3.5)
    #time_range = (73, 77)
    #time_range = (77.25, 80.0)  # not great actual response
    time_range = (23, 26) # best so far

    fig=ctx['modelspec'].quickplot(time_range=time_range, sig_names=['fg_comp', 'bg_comp'],
                               modidx_set=[1, 2, 3, 5])

    fig.savefig('/auto/users/svd/docs/current/RDT/nems/{}-{}-{}-T{}-{}.pdf'.format(
        cellid,batch,modelname, time_range[0], time_range[1]))
