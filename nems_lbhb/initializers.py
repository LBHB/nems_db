#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nems_lbhb.initializers

Created on Fri Aug 31 12:50:49 2018

@author: svd
"""
import logging
import re
import numpy as np
import copy
import nems.epoch as ep
import nems.signal as signal
import scipy.fftpack as fp
import scipy.signal as ss

from nems.utils import find_module
from nems.preprocessing import resp_to_pc

log = logging.getLogger(__name__)


def pca_proj_layer(rec, modelspec, **ctx):

    weight_chan_idx = find_module("weight_channels", modelspec, find_all_matches=True)
    w = weight_chan_idx[-1]
    coefficients = modelspec.phi[w]['coefficients'].copy()
    pcs_needed = int(np.ceil(coefficients.shape[1] / 2))

    try:
        v = rec.meta['pc_weights'].T[:, :pcs_needed]
        log.info('Found %d sets of PC weights', pcs_needed)
    except:
        rec = resp_to_pc(rec=rec, pc_count=pcs_needed, pc_source='all', overwrite_resp=False, **ctx)['rec']
        v = rec.meta['pc_weights'].T[:, :pcs_needed]

    v = np.concatenate((v, -v), axis=1)
    modelspec = modelspec.copy()
    pre = modelspec.phi[w]['coefficients'].std()
    modelspec.phi[w]['coefficients'] = v[:,:coefficients.shape[1]]
    post = modelspec.phi[w]['coefficients'].std()
    log.info('Pasted pc weights into N x R = %d x %d weight channels matrix %.3f -> %.3f', v.shape[0], v.shape[1], pre, post)

    return {'modelspec': modelspec, 'rec': rec}
