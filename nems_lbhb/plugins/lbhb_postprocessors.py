"""
lbhb_postprocessors.py

NEMS keywords for processing after model has been fit. E.g., generate predictions in
a non-standard configuration or generate a non-standard plot.

"""
import logging
import re

from nems0.registry import xform, xmodule

log = logging.getLogger(__name__)

@xform()
def ebc(loadkey):
    """
    ebc = evaluate_by_condition
    ebc = evaluate_by_condition
       finds prediction correlations by condition
    """
    options = loadkey.split('.')[1:]
    use_mask=True
    for op in options[1:]:
        if op == 'rmM':
            use_mask=False
    xfspec = [
        ['nems0.xforms.add_summary_statistics', {'use_mask': use_mask}],
        ['nems0.xforms.add_summary_statistics_by_condition',{'use_mask': use_mask}]
    ]
    return xfspec

@xform()
def SPOpf(loadkey):
    xfspec = [['nems0.xforms.predict', {'use_mask': False}]]
    #xfspec.append(['nems_lbhb.SPO_helpers.mask_out_Squares', {}]) Not included in val anymore
    xfspec = xfspec + ebc('ebc.rmM')
    xfspec.append(['nems0.xforms.plot_summary', {'time_range':(5, 23.0)}])
    xfspec.append(['nems_lbhb.SPO_helpers.plot_all_vals_',{}])
    # xfspec.append(['nems_lbhb.SPO_helpers.plot_linear_and_weighted_psths_model', {}])

    return xfspec

@xform()
def popsum(loadkey):

    return [['nems0.xforms.predict', {}],
            ['nems0.xforms.add_summary_statistics', {}],
            ['nems0.xforms.plot_summary', {}],
            ['nems_lbhb.stateplots.quick_pop_state_plot', {}]]

@xform()
def popspc(loadkey):
    return [['nems0.xforms.predict', {}],
            ['nems0.xforms.add_summary_statistics', {}],
            ['nems0.xforms.plot_summary', {}],
            ['nems_lbhb.analysis.pop_models.pop_space_summary', {'n_pc': 5}]]


@xform()
def tfheld(loadkey):
    freeze_layers = None
    options = loadkey.split('.')
    use_matched_recording = False
    #use_matched_random = False
    use_same_recording = False
    for op in options[1:]:
        if op.startswith('FL'):
            if ':' in op:
                # ex: FL0:5  would be freeze_layers = [0,1,2,3,4]
                lower, upper = [int(i) for i in op[2:].split(':')]
                freeze_layers = list(range(lower, upper))
            else:
                # ex: TL2x6x9  would be trainable_layers = [2, 6, 9]
                freeze_layers = [int(i) for i in op[2:].split('x')]
        elif op == 'ms':
            use_matched_recording = True
        # elif op == 'rnd':
        #     use_matched_random = True
        elif op == 'same':
            use_same_recording = True

    xfspec = [['nems_lbhb.xform_wrappers.switch_to_heldout_data', {'freeze_layers': freeze_layers,
                                                                   'use_matched_recording': use_matched_recording,
                                                                   #'use_matched_random': use_matched_random,
                                                                   'use_same_recording': use_same_recording}]]
    return xfspec

@xform('rd')
def rd(loadkey):
    """
    Run Decoding Analysis using context.
    In the future, add options to specify options for the decoding analysis.
    For now, just using defaults set in nems_lbhb.projects.nat_pup_decoding.do_decoding
    """
    use_pred = True
    options = loadkey.split(".")
    for op in options:
        if op == "resp":
            use_pred = False
            
    xfspec = [['nems_lbhb.postprocessing.run_decoding', {'use_pred': use_pred}]]
    return xfspec

@xform()
def rda(loadkey):
    """
    Run Decoding Analysis
    """
    xfspec = [['nems_lbhb.postprocessing.run_decoding_analysis', {}]]
    return xfspec

@xform()
def dstrf(loadkey):
    """
    Run Decoding Analysis
    """
    parms = {}
    options = loadkey.split(".")
    for op in options[1:]:
        if op.startswith('d'):
            parms['D']=int(op[1:])
        elif op.startswith('t'):
            parms['timestep'] = int(op[1:])
        elif op.startswith('p'):
            parms['pc_count']=int(op[1:])
        elif op.startswith('ss'):
            parms['fit_ss_model']=True

    return [['nems_lbhb.analysis.dstrf.dstrf_pca', parms]]

    return xfspec



@xform('svpred')
def svpred(kw):
    """
    saves a recording containing only the prediction signal alongside the xfomrs model
    """
    return [['nems_lbhb.postprocessing.save_pred_signal', {}, ]]
