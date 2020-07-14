"""
lbhb_postprocessors.py

NEMS keywords for processing after model has been fit. E.g., generate predictions in
a non-standard configuration or generate a non-standard plot.

"""
import logging
import re

from nems.registry import xform, xmodule

log = logging.getLogger(__name__)

@xform()
def ebc(loadkey):
    """
    ebc = evaluate_by_condition
       finds prediction correlations by condition
    """
    ops = loadkey.split('.')[1:]
    if ops[0] == 'rmM':
        use_mask=False
    else:
        use_mask=True
    xfspec = [
        ['nems.xforms.add_summary_statistics', {'use_mask': use_mask}],
        ['nems_lbhb.postprocessing.add_summary_statistics_by_condition',{}]
    ]
    return xfspec

@xform()
def SPOpf(loadkey):
    xfspec = [['nems.xforms.predict', {}]]
    xfspec = xfspec + ebc('ebc.rmM')
    xfspec.append(['nems.xforms.plot_summary', {}])
    xfspec.append(['nems_lbhb.SPO_helpers.plot_all_vals_',{}])
    xfspec.append(['nems_lbhb.SPO_helpers.plot_linear_and_weighted_psths_model', {}])

    return xfspec

@xform()
def popsum(loadkey):

    return [['nems.xforms.predict', {}],
            ['nems.xforms.add_summary_statistics', {}],
            ['nems.xforms.plot_summary', {}],
            ['nems_lbhb.stateplots.quick_pop_state_plot', {}]]

@xform()
def popspc(loadkey):
    return [['nems.xforms.predict', {}],
            ['nems.xforms.add_summary_statistics', {}],
            ['nems.xforms.plot_summary', {}],
            ['nems_lbhb.analysis.pop_models.pop_space_summary', {}]]

