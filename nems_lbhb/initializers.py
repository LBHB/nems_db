#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nems_lbhb.initializers

Created on Fri Aug 31 12:50:49 2018

@author: svd
"""
import logging
import os
import re
import hashlib
import copy

import numpy as np
import scipy.fftpack as fp
import scipy.signal as ss
import pandas as pd

import nems.epoch as ep
import nems.signal as signal
from nems.utils import find_module, adjust_uri_prefix
from nems.preprocessing import resp_to_pc
from nems.initializers import load_phi
import nems.db as nd
from nems_lbhb.xform_wrappers import _matching_cells
from nems import xform_helper, xforms
from nems.uri import save_resource
from nems.utils import get_default_savepath

log = logging.getLogger(__name__)


def initialize_with_prefit(modelspec, meta, area="A1", cellid=None, siteid=None, batch=322, pre_batch=None,
                           use_matched=False, use_simulated=False, use_full_model=False, 
                           prefit_type=None, freeze_early=True, IsReload=False, **ctx):
    """
    replace early layers of model with fit parameters from a "standard" model ... for now that's model with the same architecture fit
    to the NAT4 dataset
    
    for dnn single:
    initial model:
    modelname = "ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x4.g-fir.1x25x4-relu.4.f-wc.4x1-lvl.1-dexp.1_tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4.es20"
    
    use initial as pre-fit:
    modelname = "ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x4.g-fir.1x25x4-relu.4.f-wc.4x1-lvl.1-dexp.1_prefit-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.es20"

    """
    if IsReload:
        return {}

    xi = find_module("weight_channels", modelspec, find_all_matches=True)
    if len(xi) == 0:
        raise ValueError(f"modelspec has not weight_channels layer to align")

    copy_layers = xi[-1]
    freeze_layer_count = xi[-1]
    batch = int(meta['batch'])
    modelname_parts = meta['modelname'].split("_")
    
    if use_simulated:
        guess = '.'.join(['SIM000a', modelname_parts[1]])

        # remove problematic characters
        guess = re.sub('[:]', '', guess)
        guess = re.sub('[,]', '', guess)
        if len(guess) > 100:
            # If modelname is too long, causes filesystem errors.
            guess = guess[:75] + '...' + str(hashlib.sha1(guess.encode('utf-8')).hexdigest()[:20])

        old_uri = f"/auto/data/nems_db/modelspecs/{guess}/modelspec.0000.json"
        log.info('loading saved modelspec from: ' + old_uri)

        new_ctx = load_phi(modelspec, prefit_uri=old_uri, copy_layers=copy_layers)
        
        return new_ctx

    elif prefit_type == 'init':
        # use full pop file - SVD work in progress. current best?
        load_string_pop = "ozgf.fs100.ch18.pop-loadpop-norm.l1-popev"
        fit_string_pop = "tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4"

        pre_part = load_string_pop
        if len(modelname_parts[2].split("-")) > 2:
            post_part = "-".join(modelname_parts[2].split("-")[1:-1])
        else:
            post_part = "-".join(modelname_parts[2].split("-")[1:])

        model_search = "_".join([pre_part, modelname_parts[1], post_part])

        if pre_batch is None:
            pre_batch = batch
        if pre_batch in [322, 334]:
            pre_cellid = 'ARM029a-07-6'
        elif pre_batch == 323:
            pre_cellid = 'ARM017a-01-9'
        else:
            raise ValueError(f"batch {pre_batch} prefit not implemented yet.")

        log.info(f"prefit cellid={pre_cellid}, skipping init_fit")
        copy_layers = len(modelspec)

    elif use_full_model:
        
        # use full pop file - SVD work in progress. current best?
        load_string_pop = "ozgf.fs100.ch18.pop-loadpop-norm.l1-popev"
        fit_string_pop = "tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4"

        if prefit_type == 'heldout':
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev"
        elif prefit_type == 'matched':
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev"
        elif prefit_type == 'matched_half':
            # 50% est data (matched cell excluded)
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k50"
        elif prefit_type == 'matched_quarter':
            # 25% est data
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k25"
        elif prefit_type == 'matched_fifteen':
            # 15% est data
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k15"
        elif prefit_type == 'matched_ten':
            # 10% est data
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k10"
        elif prefit_type == 'heldout_half':
            # 50% est data, cell excluded (is this a useful condition?)
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k50"
        elif prefit_type == 'heldout_quarter':
            # 25% est data
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k25"
        elif prefit_type == 'heldout_fifteen':
            # 15% est data
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k15"
        elif prefit_type == 'heldout_ten':
            # 10% est data
            pre_part = "ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k10"
        elif 'R.q.s' in modelname_parts[1]:
            pre_part = "ozgf.fs100.ch18-ld-norm.l1-sev"
        elif 'ch32' in modelname_parts[0]:
            pre_part = "ozgf.fs100.ch32.pop-loadpop-norm.l1-popev"
        elif 'ch64' in modelname_parts[0]:
            pre_part = "ozgf.fs100.ch64.pop-loadpop-norm.l1-popev"
        elif batch==333:
            # not pre-concatenated recording. different stim for each site, 
            # so fit each site separately (unless titan)
            pre_part = "ozgf.fs100.ch18-ld-norm.l1-sev"
        else:
            #load_string_pop = "ozgf.fs100.ch18.pop-loadpop-norm.l1-popev"
            pre_part = load_string_pop

        if prefit_type == 'titan':
            if batch==333:
                pre_part = load_string_pop
                post_part = "tfinit.n.mc50.lr1e3.et4.es20-newtf.n.mc100.lr1e4"
            else:
                post_part = "tfinit.n.mc25.lr1e3.es20-newtf.n.mc100.lr1e4.exa"
        else:
            #post_part = "tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4.es20"
            post_part = fit_string_pop

        if modelname_parts[2].endswith(".l2:5") or modelname_parts[2].endswith(".l2:5-dstrf") or modelname_parts[2].endswith("ver5"):
            post_part += ".l2:5"
        elif modelname_parts[2].endswith(".l2:4") or modelname_parts[2].endswith(".l2:4-dstrf") or modelname_parts[2].endswith("ver4"):
            post_part += ".l2:4"
        elif modelname_parts[2].endswith(".l2:4.ver2"):
            post_part += ".l2:4.ver2"
        elif modelname_parts[2].endswith("ver2"):
            post_part += ".ver2"
        elif modelname_parts[2].endswith("ver1"):
            post_part += ".ver1"

        model_search = "_".join([pre_part, modelname_parts[1], post_part])
        if pre_batch is None:
            pre_batch = batch

        # this is a single-cell fit
        if type(cellid) is list:
            cellid = cellid[0]
        siteid = cellid.split("-")[0]
        allsiteids, allcellids = nd.get_batch_sites(batch, modelname_filter=model_search)
        allsiteids = [s.split(".")[0] for s in allsiteids]

        if (batch==323) and (pre_batch==322):
            matchfile=os.path.dirname(__file__) + "/projects/pop_model_scripts/snr_subset_map.csv"
            df = pd.read_csv(matchfile, index_col=0)
            pre_cellid = df.loc[df.PEG_cellid==cellid, 'A1_cellid'].values[0]
        elif (batch==322) and (pre_batch==323):
            matchfile=os.path.dirname(__file__) + "/projects/pop_model_scripts/snr_subset_map.csv"
            df = pd.read_csv(matchfile, index_col=0)
            pre_cellid = df.loc[df.A1_cellid==cellid, 'PEG_cellid'].values[0]

        elif siteid in allsiteids:
            # don't need to generalize, load from actual fit
            pre_cellid = cellid
        elif batch in [322, 334]:
            pre_cellid = 'ARM029a-07-6'
        elif pre_batch == 323:
            pre_cellid = 'ARM017a-01-9'
        else:
            raise ValueError(f"batch {batch} prefit not implemented yet.")
            
        log.info(f"prefit cellid={pre_cellid} prefit batch={pre_batch}")

    elif prefit_type == 'site':
        # exact same model, just fit for site, now being fit for single cell
        pre_parts = modelname_parts[0].split("-")
        post_parts = modelname_parts[2].split("-")
        model_search = modelname_parts[0] + "%%" + modelname_parts[1] + "%%" + "-".join(post_parts[1:])

        pre_cellid = cellid[0]
        pre_batch = batch
    elif prefit_type is not None:
        # this is a single-cell fit
        if type(cellid) is list:
            cellid = cellid[0]
            
        if prefit_type=='heldout':
            if siteid is None:
                siteid=cellid.split("-")[0]
            cellids, this_perf, alt_cellid, alt_perf = _matching_cells(batch=batch, siteid=siteid)

            pre_cellid = [c_alt for c,c_alt in zip(cellids,alt_cellid) if c==cellid][0]
            log.info(f"heldout init for {cellid} is {pre_cellid}")
        else:
            pre_cellid = cellid
            log.info(f"matched cellid prefit for {cellid}")
        if pre_batch is None:
            pre_batch = batch

        post_part = "tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4"
        if modelname_parts[2].endswith(".l2:5") or modelname_parts[2].endswith(".l2:5-dstrf"):
            post_part += ".l2:5"
        elif modelname_parts[2].endswith(".l2:4") or modelname_parts[2].endswith(".l2:4-dstrf"):
            post_part += ".l2:4"
        elif modelname_parts[2].endswith(".l2:4.ver2"):
            post_part += ".l2:4.ver2"
        elif modelname_parts[2].endswith("ver2"):
            post_part += ".ver2"
        modelname_parts[2] = post_part
        model_search="_".join(modelname_parts)

    elif modelname_parts[1].endswith(".1"):
        raise ValueError("deprecated prefit initialization?")
        # this is a single-cell fit
        if type(cellid) is list:
            cellid = cellid[0]
        
        if use_matched:
            # determine matched cell for this heldout cell
            if siteid is None:
                siteid=cellid.split("-")[0]
            cellids, this_perf, alt_cellid, alt_perf = _matching_cells(batch=batch, siteid=siteid)

            pre_cellid = [c_alt for c,c_alt in zip(cellids,alt_cellid) if c==cellid][0]
            log.info(f"matched cell for {cellid} is {pre_cellid}")
        else:
            pre_cellid = cellid[0]
            log.info(f"cellid prefit for {cellid}")
        if pre_batch is None:
            pre_batch = batch
        #postparts = modelname_parts[2].split("-")
        #postparts = [s for s in postparts if not(s.startswith("prefit"))]
        #modelname_parts[2]="-".join(postparts)
        modelname_parts[2] = "tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4.es20"
        model_search="_".join(modelname_parts)

    else:
        pre_parts = modelname_parts[0].split("-")
        post_parts = modelname_parts[2].split("-")    
        post_part = "tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4.ver2"
        model_search = pre_parts[0] + ".pop%%" + modelname_parts[1] + "%%" + post_part

        #ozgf.fs100.ch18.pop-loadpop-norm.l1-popev
        #wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R
        #tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4.ver2


        # hard-coded to use an A1 model!!!!
        if pre_batch == 322:
            pre_cellid = 'ARM029a-07-6'
        elif area == "A1":
            pre_cellid = 'ARM029a-07-6'
            pre_batch = 322
        else:
            raise ValueError(f"area {area} prefit not implemented")

    log.info(f"model_search: {model_search}")

    sql = f"SELECT * FROM Results WHERE batch={pre_batch} and cellid='{pre_cellid}' and modelname like '{model_search}'"
    #log.info(sql)
    
    d = nd.pd_query(sql)
    #old_uri = adjust_uri_prefix(d['modelpath'][0] + '/modelspec.0000.json')
    old_uri = adjust_uri_prefix(d['modelpath'][0])
    log.info(f"Importing parameters from {old_uri}")

    mspaths = [f"{old_uri}/modelspec.{i:04d}.json" for i in range(modelspec.cell_count)]
    print(mspaths)
    prefit_ctx = xforms.load_modelspecs([], uris=mspaths, IsReload=False)

    #_, prefit_ctx = xform_helper.load_model_xform(
    #    cellid=pre_cellid, batch=pre_batch,
    #    modelname=d['modelname'][0], eval_model=False)
    new_ctx = load_phi(modelspec, prefit_modelspec=prefit_ctx['modelspec'], copy_layers=copy_layers)
    if freeze_early:
        new_ctx['freeze_layers'] = list(np.arange(freeze_layer_count))
    if prefit_type == 'init':
        new_ctx['skip_init'] = True
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
