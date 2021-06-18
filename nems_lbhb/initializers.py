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

from nems.utils import find_module, adjust_uri_prefix
from nems.preprocessing import resp_to_pc
from nems.initializers import load_phi
import nems.db as nd

log = logging.getLogger(__name__)


def initialize_with_prefit(modelspec, meta, area="A1", **ctx):
    """
    replace early layers of model with fit parameters from a "standard" model ... for now that's model with the same architecture fit
    to the NAT4 dataset
    """
    xi = find_module("weight_channels", modelspec, find_all_matches=True)
    if len(xi) == 0:
        raise ValueError(f"modelspec has not weight_channels layer to align")

    copy_layers = xi[-1]
    batch = meta['batch']
    modelname_parts = meta['modelname'].split("_")
    pre_parts = modelname_parts[0].split("-")
    post_parts = modelname_parts[2].split("-")
    post_part = "tfinit.n.lr1e3.et3.rb5.es20-newtf.n.lr1e4"
    model_search = pre_parts[0] + ".pop%%" + modelname_parts[1] + "%%" + post_part

    # hard-coded to use an A1 model!!!!
    if area == "A1":
        pre_cellid = 'ARM029a-07-6'
        pre_batch=322
    else:
        raise ValueError(f"area {area} prefit not implemented")
    
    sql = f"SELECT * FROM Results WHERE batch={batch} and cellid='{pre_cellid}' and modelname like '{model_search}'"
    log.info(sql)
    d = nd.pd_query(sql)

    old_uri = adjust_uri_prefix(['modelpath'][0] + '/modelspec.0000.json')
    log.info(f"Importing parameters from {old_uri}")

    new_ctx = load_phi(modelspec, prefit_uri=old_uri, copy_layers=copy_layers)
    new_ctx['freeze_layers'] = list(np.arange(copy_layers))
    return new_ctx


def pca_proj_layer(rec, modelspec, **ctx):
    from nems.tf.cnnlink_new import fit_tf, fit_tf_init

    weight_chan_idx = find_module("weight_channels", modelspec, find_all_matches=True)
    w = weight_chan_idx[-1]
    coefficients = modelspec.phi[w]['coefficients'].copy()
    pcs_needed = int(np.ceil(coefficients.shape[1] / 2))
    if 'state' in modelspec[w-1]['fn']:
        w -= 1

    try:
        v = rec.meta['pc_weights'].T[:, :pcs_needed]
        pc_rec = rec.copy()
        log.info('Found %d sets of PC weights', pcs_needed)
    except:
        pc_rec = resp_to_pc(rec=rec, pc_count=pcs_needed, pc_source='all', overwrite_resp=False, **ctx)['rec']
        v = pc_rec.meta['pc_weights'].T[:, :pcs_needed]
    v = np.concatenate((v, -v), axis=1)

    pc_modelspec = modelspec.copy()
    for i in range(w,len(modelspec)):
        pc_modelspec.pop_module()

    d = pc_rec.signals['pca'].as_continuous()[:pcs_needed,:]
    d = np.concatenate((d,-d), axis=0)
    d = d[:coefficients.shape[1],:]
    pc_rec['resp'] = pc_rec['resp']._modified_copy(data=d)

    #_d = fit_tf_init(pc_modelspec, pc_rec, nl_init='skip', use_modelspec_init=True, epoch_name="")
    _d = fit_tf_init(pc_modelspec, pc_rec, use_modelspec_init=True, epoch_name="")
    pc_modelspec = _d['modelspec']
    modelspec = modelspec.copy()
    for i in range(w):
        for k in modelspec.phi[i].keys():
            modelspec.phi[i][k] = pc_modelspec.phi[i][k]

    log.info('modelspec len: %d  pc_modelspec len: %d', len(modelspec), len(pc_modelspec))
    
    #pre = modelspec.phi[w]['coefficients'].std()
    #modelspec.phi[w]['coefficients'] = v[:,:coefficients.shape[1]]
    #post = modelspec.phi[w]['coefficients'].std()
    #log.info('Pasted pc weights into N x R = %d x %d weight channels matrix %.3f -> %.3f', v.shape[0], v.shape[1], pre, post)

    return {'modelspec': modelspec, 'pc_modelspec': pc_modelspec}
