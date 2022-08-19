"""
initializer keywords specific to LBHB models
should occur intermingled with fitter keywords
"""
import logging
import re

from nems.plugins.default_fitters import init as nems_init
from nems.registry import xform, xmodule

log = logging.getLogger(__name__)

@xform()
def init(kw):
    '''
    Same as default nems init except adds 'c' option for contrast model.
    '''
    xfspec = nems_init(kw)
    ops = kw.split('.')[1:]
    if 'c' in ops:
        xfspec[0][0] = 'nems_lbhb.gcmodel.initializers.init_contrast_model'
        if 'strfc' in ops:
            xfspec[0][1]['copy_strf'] = True
    elif 'lnp' in ops:
        xfspec[0][0] = 'nems_lbhb.lnp_helpers.init_lnp_model'

    return xfspec


@xform()
def pclast(kw):
    return [['nems_lbhb.initializers.pca_proj_layer', {}]]

@xform()
def prefit(kw):
    ops = kw.split('.')[1:]
    kwargs = {}
    kwargs['prefit_type']='site'

    # use_full_model means population model (vs. single-cell fit used for dnn-single)
    for op in ops:
        if op.startswith("b"):
            kwargs['pre_batch']=int(op[1:])
        elif op=='nf':
            kwargs['freeze_early']=False
        elif op=='h':
            kwargs['use_heldout'] = True
        elif op=='m':
            kwargs['use_matched'] = True
        elif op=='s':
            kwargs['use_simulated'] = True
        elif op=='f':
            kwargs['use_full_model'] = True

    if 'hm' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'matched'
    elif 'hs' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'heldout'

    elif 'hhm' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'matched_half'
    elif 'hqm' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'matched_quarter'
    elif 'hfm' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'matched_fifteen'
    elif 'htm' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'matched_ten'

    elif 'hhs' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'heldout_half'
    elif 'hqs' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'heldout_quarter'
    elif 'hfs' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'heldout_fifteen'
    elif 'hts' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'heldout_ten'
    
    elif 'init' in ops:
        kwargs['prefit_type'] = 'init'
    elif 'm' in ops:
        kwargs['prefit_type'] = 'matched'
    elif 'h' in ops:
        kwargs['prefit_type'] = 'heldout'
    
    elif 'titan' in ops:
        kwargs['use_full_model'] = True
        kwargs['prefit_type'] = 'titan'

    return [['nems_lbhb.initializers.initialize_with_prefit', kwargs]]
