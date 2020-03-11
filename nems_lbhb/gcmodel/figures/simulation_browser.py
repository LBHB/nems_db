from nems.gui.editors import browse_xform_fit
import nems.xform_helper as xhelp
from nems.xforms import normalize_sig
from nems_lbhb.gcmodel.figures.simulation import (build_toy_stp_cell,
                                                  build_toy_gc_cell,
                                                  build_toy_LN_cell)

_DEFAULT_CELL = 'TAR009d-42-1'
_DEFAULT_BATCH = 289
_DEFAULT_MODEL = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk3-dsig.d_gc2.PF'


if __name__ == 'main':
    xfspec, ctx = xhelp.load_model_xform(_DEFAULT_CELL, _DEFAULT_BATCH,
                                         _DEFAULT_MODEL)
    val = ctx['val']
#    val = normalize_sig(val, 'stim', 'minmax')['rec']
#    val = normalize_sig(val, 'resp', 'minmax')['rec']
    #gc_sim = build_toy_gc_cell(0, 0, 0, -0.5) #base, amp, shift, kappa
    #gc_sim[-2]['fn_kwargs']['compute_contrast'] = True
    stp_sim = build_toy_stp_cell([0, 0.1], [0.08, 0.08]) #u, tau
    #LN_sim = build_toy_LN_cell()

    ctx['val'] = stp_sim.evaluate(val)
    ctx['modelspec'] = stp_sim
    browse_xform_fit(ctx, xfspec)
