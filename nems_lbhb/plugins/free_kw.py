import logging
"""
from os.path import basename, join

import matplotlib.pyplot as plt
import numpy as np

from nems0 import db
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems_lbhb.preprocessing import impute_multi
from nems.layers import WeightChannels, FIR, LevelShift, \
    DoubleExponential, RectifiedLinear, ConcatSignals, WeightChannelsGaussian
from nems import Model
from nems.tools import json
from nems.layers.base import Layer, Phi, Parameter
from nems0.recording import load_recording
import nems.visualization.model as nplt
from nems0.modules.nonlinearity import _dlog
from nems_lbhb.projects.freemoving.free_tools import stim_filt_hrtf, compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems0.epoch import epoch_names_matching
from nems0.metrics.api import r_floor
from nems0 import xforms
from nems.preprocessing import (indices_by_fraction, split_at_indices, JackknifeIterator)
"""
from nems0.registry import xform, scan_for_kw_defs
from nems_lbhb.plugins.lbhb_loaders import _load_dict
from nems.registry import layer, keyword_lib

log = logging.getLogger(__name__)

@layer('wcdl')
def wcdl(keyword):
    k = keyword.replace('wcdl','wc')
    wc = keyword_lib[k]
    options = keyword.split('.')
    if 'i' in options:
        wc.input = 'dlc'
    else:
        wc.input = 'hrtf'
    if 's' in options:
        wc.output = 'state'
    else:
        wc.output = 'hrtf'
    return wc
    
@layer('firdl')
def firdl(keyword):
    k = keyword.replace('firdl','fir')
    fir = keyword_lib[k]
    fir.input = 'hrtf'
    fir.output = 'hrtf'
    return fir

@layer('relud')
def relud(keyword):
    k = keyword.replace('relud','relu')
    relu = keyword_lib[k]
    relu.input = 'hrtf'
    relu.output = 'hrtf'
    return relu

@layer('sigd')
def sigd(keyword):
    k = keyword.replace('sigd','sig')
    sig = keyword_lib[k]
    sig.input = 'hrtf'
    sig.output = 'hrtf'
    return sig

@layer('wcs')
def wcs(keyword):
    k = keyword.replace('wcs','wc')
    wc = keyword_lib[k]
    options = keyword.split('.')
    wc.input = 'state'
    wc.output = 'state'
    return wc
    
@layer('firs')
def firs(keyword):
    k = keyword.replace('firs','fir')
    fir = keyword_lib[k]
    fir.input = 'state'
    fir.output = 'state'
    return fir

@layer('relus')
def relus(keyword):
    k = keyword.replace('relus','relu')
    relu = keyword_lib[k]
    relu.input = 'state'
    relu.output = 'state'
    return relu

#@layer('stategaindl')
#def from_keyword(keyword):
#    """wrapper for stategain keyword, but make state input name 'dlc'
#    """
#    k = keyword.replace('stategaindl','stategain')
#    sg = keyword_lib[k]
#    sg.state_arg = 'hrtf'
#    return sg

@layer('wcst')
def wcst(keyword):
    k = keyword.replace('wcst','wc')
    wc = keyword_lib[k]
    options = keyword.split('.')
    if 'i' in options:
        wc.input = 'input'
    else:
        wc.input = 'stim'
    wc.output = 'stim'
    return wc

@layer('first')
def first(keyword):
    k = keyword.replace('first','fir')
    fir = keyword_lib[k]
    fir.input = 'stim'
    fir.output = 'stim'
    return fir

@layer('wch')
def wch(keyword):
    k = keyword.replace('wch','wc')
    wc = keyword_lib[k]
    wc.input = 'hstim'
    return wc

@xform()
def free(loadkey, cellid=None, batch=None, siteid=None, **options):
    d = _load_dict(loadkey, cellid, batch)
    d['siteid']=cellid
    d['compute_position']=True
    del d['cellid']
    xfspec = [['nems_lbhb.projects.freemoving.free_model.load_free_data', d]]
    return xfspec

@xform()
def fev(keyword):
    ops = keyword.split('.')[1:]
    d={}
    for op in ops:
        if op=='hrtf':
            d['apply_hrtf']=True
        elif op == 'hrtfae':
            d['apply_hrtf'] = 'az_el'
        elif op.startswith('jk'):
            d['jackknife_count']=int(op[2:])

    xfspec = [['nems_lbhb.projects.freemoving.free_model.free_split_rec', d]]
    return xfspec

