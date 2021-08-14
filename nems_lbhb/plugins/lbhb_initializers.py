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
    use_matched = 'm' in ops
    use_simulated = 's' in ops
    
    return [['nems_lbhb.initializers.initialize_with_prefit', 
             {'use_matched': use_matched, 'use_simulated': use_simulated}]]
