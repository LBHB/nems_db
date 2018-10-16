"""
initializer keywords specific to LBHB models
should occur intermingled with fitter keywords
"""
import logging
import re

from nems.plugins.default_initializers import init as nems_init

log = logging.getLogger(__name__)


def init(kw):
    '''
    Same as default nems init except adds 'c' option for contrast model.
    '''
    xfspec = nems_init(kw)
    ops = kw.split('.')[1:]
    if 'c' in ops:
        xfspec[0][0] = 'nems_lbhb.contrast_helpers.init_contrast_model'
        if 'strfc' in ops:
            xfspec[0][1]['copy_strf'] = True

    return xfspec
