"""
Utilities for super-mega model
"""
#from os.path import basename, join
import os
import datetime
import copy
import time
import io
import itertools
import logging
from joblib import Memory

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, butter, sosfilt
import pandas as pd
#import scipy
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_array
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import seaborn as sns

import tensorflow as tf
from sklearn.decomposition import FactorAnalysis, PCA

from nems0 import db, epoch, initializers, xforms
import nems0.preprocessing as preproc
from nems0.utils import smooth, get_setting, parse_kw_string
from nems_lbhb import xform_wrappers, baphy_io, baphy_experiment

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems.layers import WeightChannels, FIR, LevelShift, \
    DoubleExponential, RectifiedLinear, ConcatSignals
from nems import Model
from nems.layers.base import Layer, Phi, Parameter
import nems.visualization.model as nplt
#import nems0.plots.api as nplt
from nems_lbhb.projects.freemoving.free_model import load_free_data, free_fit
from nems0.modules.nonlinearity import _dlog
from nems0.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems0.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems_lbhb.projects.bignat.bnt_defaults import POP_MODELS, SIG_TEST_MODELS, shortnames, POP_MODELS_OLD
import nems0.plots.api as n0plt

log = logging.getLogger(__name__)

JOB_LIB_PATH = get_setting("JOB_LIB_PATH")
memory = Memory(JOB_LIB_PATH)

def to_sparse4d(A):
    """encode 4D array sparsely (nested lists of csr_array, since csr_array can only be 2D)"""
    S = []
    for i in range(A.shape[0]):
        S_ = []
        for j in range(A.shape[1]):
            S_.append(csr_array(A[i,j]))
        S.append(S_)
    return S


def to_dense4d(S):
    """convert sparse 4D array back to full"""
    D = np.empty((len(S), len(S[0]))+S[0][0].shape)
    #print(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D[i,j,:,:]=S[i][j].toarray()
    return D


def data_subset(data, site_set, get_fit_data=True, input_name='stim', output_name='resp'):
    """extract partial dataset from special data dictionary"""
    if get_fit_data:
        #X_est = np.moveaxis(data[s]['est'].apply_mask()[input_name].extract_epoch("REFERENCE"), -1, 1)
        #Y_est = np.moveaxis(data[s]['est'].apply_mask()[output_name].extract_epoch("REFERENCE"), -1, 1)
        stim = np.concatenate([np.moveaxis(data[s]['est'].apply_mask()[input_name].extract_epoch("REFERENCE")[:,:,:2000], -1, 1) for s in site_set], axis=0)
        #stim = np.concatenate([data[s]['est']['stim'].as_continuous().T for s in site_set], axis=0)
        resplist = [np.moveaxis(data[s]['est'].apply_mask()[output_name].extract_epoch("REFERENCE")[:,:,:2000], -1, 1) for s in site_set]
        #resplist = [data[s]['est']['resp'].as_continuous().T for s in site_set]
        respbatches=[r.shape[0] for r in resplist]
        respbatchlen=[r.shape[1] for r in resplist]
        chancounts=[r.shape[2] for r in resplist]
        resptotal = int(np.sum(respbatches))
        chancount = int(np.sum(chancounts))
        
        resp = np.full((resptotal, respbatchlen[0], chancount), np.nan)
        cc=0
        tt=0
        for i,r in enumerate(resplist):
            _c = cc+r.shape[2]
            _t = tt+r.shape[0]
            resp[tt:_t, :, cc:_c] = r
            print(r.shape, cc, _c, tt, _t)
            
            cc=_c
            tt=_t
    else:
        stim = np.moveaxis(data[site_set[0]]['val'].apply_mask()[input_name].extract_epoch("REFERENCE")[:,:,:2000], -1, 1)
        resp = np.concatenate([np.moveaxis(data[s]['val'].apply_mask()[output_name].extract_epoch("REFERENCE")[:,:,:2000], -1, 1) for s in site_set], axis=2)

    return stim, resp


"""
from joblib import Memory
memory = Memory("cachedir")
@memory.cache
def f(x):
    print('Running f(%s)' % x)
    return x
"""

@memory.cache
def load_bnt_recording(siteid, batch=343, loadkey="gtgram.fs100.ch18",
                       pc_count=10, epoch_len=2000, val_single=False,
                       projection_method="pca"):
    """
    to test if val_single loaded properly and if sparse format works:
    val_single = np.stack([val_['resp'].extract_epoch(e)[:,:,:epoch_len] for e in val_epochs], axis=0)
    S = to_sparse4d(val_single)
    D = to_dense4d(S)
    val_single.sum(), D.sum()
    
    """
    
    log.info(f"Loading BNT data for site: {siteid}")

    ex = baphy_experiment.BAPHYExperiment(batch=batch, cellid=siteid)
    rec = ex.get_recording(loadkey=loadkey)
    rec['stim'] = rec['stim'].rasterize()
    fn = lambda x: _dlog(x, -1)
    rec['stim'] = rec['stim'].transform(fn, 'stim')
    rec['stim'] = rec['stim'].normalize('minmax')
    rec['resp'] = rec['resp'].rasterize()
    rec['resp'] = rec['resp'].normalize('minmax')
    rec = xforms.concat_constant(rec=rec, sig='stim', use_mean_val=True)['rec']

    if projection_method=='pca':
        rec = preproc.resp_to_pc(rec, resp_sig='resp', pc_sig='pca', pc_count=pc_count,
                                 pc_source='all', overwrite_resp=False, whiten=False)['rec']
    elif projection_method=='fa':
        fa = FactorAnalysis(n_components=pc_count, random_state=0)
        f = fa.fit_transform(rec['resp'].as_continuous().T)
        new_chans = [f'FA{i}' for i in range(pc_count)]
        rec['pca']=rec['resp']._modified_copy(data=f.T, chans=new_chans)
    else:
        raise ValueError(f'unsupported projection method {projection_method}')
    epoch_regex = '^STIM_'
    est, val_ = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
    est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
    val = preproc.average_away_epoch_occurrences(val_, epoch_regex=epoch_regex)
    est['resp'] = est['resp'].rasterize(sparse=True)

    # a bit kludgy, but we know there's a special prefix for the validation stimuli
    est_regex = "^STIM_seq"
    val_regex = "^STIM_00"
    e = rec['resp'].epochs

    est_epochs = epoch.epoch_names_matching(e, est_regex)
    val_epochs = epoch.epoch_names_matching(e, val_regex)
    est_indices = np.array([int(s[8:12]) for s in est_epochs])
    val_repcount = (e['name'] == val_epochs[0]).sum()
    _cellids = rec['resp'].chans
    cell_count = len(_cellids)
    #epochs_all.extend(est_epochs)
    log.info(f"{siteid} {est['resp'].shape} est N={len(est_epochs)} val reps={val_repcount}")

    dataset = {}
    dataset['est'] = est
    dataset['val'] = val
    dataset['est_epochs'] = est_epochs
    dataset['val_epochs'] = val_epochs
    dataset['est_indices'] = est_indices
    dataset['val_repcount'] = val_repcount
    dataset['cell_count'] = cell_count
    dataset['cellids'] = _cellids
    if val_single:
        dataset['val_single'] = to_sparse4d(
            np.stack([val_['resp'].extract_epoch(e)[:, :, :epoch_len]
                      for e in val_epochs], axis=0))

    return dataset


def get_submodel(model_full, site_set):
    """MOVED to nems_lbhb.initializers"""
    cell_siteids = model_full.meta['cell_siteids']
    site_mask = np.array([(s in site_set) for s in cell_siteids ])
    
    keywordstring = model_full.meta['keywordstub'].format(site_mask.sum())
    model_sub = Model.from_keywords(keywordstring)
    model_sub.meta['site_mask']=site_mask
    for i,l in enumerate(model_full.layers):
        _d=l.get_parameter_values(as_dict=True)
        if i>=len(model_full.layers)-2:
            d = {}
            for k, v in _d.items():
                if v.ndim==2:
                    d[k] = v[:,site_mask]       
                else:
                    d[k] = v[site_mask]
            print(i, k, v.shape, d[k].shape)
        else:
            d=_d
        model_sub.layers[i].set_parameter_values(d, ignore_bounds=True)

    return model_sub


def save_submodel(model_full, model_sub):
    """MOVED to nems_lbhb.initializers"""
    """
    NB this function operates in-place!!
    
    if copy all layers to model full except last 2. if model_sub.meta['site_mask'] is defined, copy last
    two layers of model_sub to channel ids [site_mask] in model_full
    """

    site_mask = model_sub.meta.get('site_mask', None)
    full_layer_offset = len(model_full.layers)-len(model_sub.layers)
    
    for i,l in enumerate(model_sub.layers):
        _d=l.get_parameter_values(as_dict=True)
        if i>=len(model_sub.layers)-2:
            if site_mask is not None:
                #print(i, i+full_layer_offset)
                d = model_full.layers[i+full_layer_offset].get_parameter_values(as_dict=True)
                for k, v in _d.items():
                    if v.ndim==2:
                        d[k][:,site_mask] = v
                    else:
                        d[k][site_mask] = v
                model_full.layers[i+full_layer_offset].set_parameter_values(d, ignore_bounds=True)
        else:
            model_full.layers[i+full_layer_offset].set_parameter_values(_d, ignore_bounds=True)
    return model_full


class btn_generator(tf.keras.utils.Sequence):
    
    def __init__(self, data, siteids=None, batch_size=32, recording='est', shuffle=True, input_name='stim', output_name='pca',
                 epoch_len=2000, frac=1.0, dtype=np.float32):
        'Initialization'
        self.data = data
        if siteids is None:
            self.siteids = list(data.keys())
        else:
            self.siteids = siteids
        self.recording = recording
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_name = input_name
        self.output_name = output_name
        self.dtype = dtype
        self.epoch_len = epoch_len
        self.frac = frac

        stim = self.data[self.siteids[0]][self.recording].apply_mask()[self.input_name].extract_epoch("REFERENCE")[:,:,:self.epoch_len]
        self.stim_channels = stim.shape[1]
        self.resp_channels = [self.data[s][self.recording][self.output_name].shape[0] for s in self.siteids]

        for s in siteids:
            self.data[s]['processed_stim'] = np.moveaxis(self.data[s][self.recording][self.input_name].extract_epoch("REFERENCE")[:,:,:self.epoch_len], -1, 1)
            self.data[s]['processed_resp'] = np.moveaxis(self.data[s][self.recording][self.output_name].extract_epoch("REFERENCE")[:,:,:self.epoch_len], -1, 1)
 
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if self.frac<0:
            f = int(np.round((1+self.frac)*len(indexes)))
            indexes = indexes[slice(f,None)]
        elif (self.frac<1) & (self.frac>0):
            f = int(np.round(self.frac*len(indexes)))
            indexes = indexes[slice(0,f)]

        # Generate data
        return self.__data_generation(indexes)
    
    def get_indexes(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if self.frac<0:
            f = int(np.round((1+self.frac)*len(indexes)))
            indexes = indexes[slice(f,None)]
        elif (self.frac<1) & (self.frac>0):
            f = int(np.round(self.frac*len(indexes)))
            indexes = indexes[slice(0,f)]

        return indexes
        
    def copy(self):
        return copy.copy(self)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        stimidx=np.concatenate([np.arange(len(self.data[k]['est_indices'])) for k in self.siteids])
        dsidx=np.concatenate([np.ones(len(self.data[k]['est_indices']), dtype=int)*i for  i,k in enumerate(self.siteids)])
        self.indexes = [(a,b) for a,b in zip(dsidx,stimidx)]
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def stim_shape(self):
        stim = self.data[self.siteids[0]][self.recording].apply_mask()[self.input_name].extract_epoch("REFERENCE")[:,:,:2000]
        return stim.shape
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # stim.shape, resp.shape
        # (300, 2000, 19)), ((300, 2000, 30)
        X = np.full((len(indexes), self.epoch_len, self.stim_channels), np.nan, dtype=self.dtype)
        y = np.full((len(indexes), self.epoch_len, np.sum(self.resp_channels)), np.nan, dtype=self.dtype)
        
        this_idx = indexes.copy()
        this_idx.sort()
        this_idx = np.array(this_idx)
        dsidx=np.unique(this_idx[:,0])
        os=0
        for d,dos in enumerate(self.resp_channels):
            s = self.siteids[d]
            ii = this_idx[:,0]==d
            os2=os+dos
            if ii.sum()>0:
                jj = this_idx[ii,1]
                #st = self.data[s][self.recording][self.input_name].extract_epoch("REFERENCE")[jj,:,:self.epoch_len]
                #X[ii,:,:] = np.moveaxis(st, -1, 1)
                #r = self.data[s][self.recording][self.output_name].extract_epoch("REFERENCE")[jj,:,:self.epoch_len]
                #os2=os+r.shape[1]
                #y[ii,:,os:os2] = np.moveaxis(r, -1, 1)
                #os=os2
                X[ii,:,:] = self.data[s]['processed_stim'][jj,:,:]
                y[ii,:,os:os2] = self.data[s]['processed_resp'][jj,:,:]
            os=os2
            
        return X, y



def pc_fit_old(keywordstub, datasets, pc_count=10,
           fitkw='lite.tf.init.mi5000.lr1e3.t3.lfse-lite.tf.mi5000.lr1e4.t5e5.lfse'):

    siteids = list(datasets.keys())
    total_pcs = int(len(siteids) * pc_count)

    keywordpc = f"{keywordstub}x{total_pcs}-lvl.{total_pcs}"

    model_pc = Model.from_keywords(keywordpc)
    model_pc.name = f"{keywordpc}"
    model_pc = model_pc.sample_from_priors()

    print(keywordpc)

    fitter = 'tf'
    loss = 'squared_error'  # 'nmse'  #
    fitter_options = {'cost_function': loss,
                      'early_stopping_delay': 100,
                      'early_stopping_patience': 130,
                      'early_stopping_tolerance': 1e-3,
                      'validation_split': 0.2,
                      'learning_rate': 1e-3,
                      'epochs': 5000,
                      'shuffle': True
                      }
    # stage 2: slower learning, smaller tolerance
    fitter_options2 = fitter_options.copy()
    fitter_options2['early_stopping_tolerance'] = 5e-5
    fitter_options2['learning_rate'] = 1e-4

    g = btn_generator(datasets, siteids, batch_size=100)
    print(g[0][0].shape)

    model_pc = model_pc.fit(input=g, target=None, backend=fitter, fitter_options=fitter_options,
                            verbose=1, batch_size=None)
    model_pc = model_pc.fit(input=g, target=None, backend=fitter, fitter_options=fitter_options2,
                            verbose=0, batch_size=None)

    # check performance on validation data
    X, Y = data_subset(datasets, siteids, output_name='pca', get_fit_data=False)
    pred = model_pc.predict(X, batch_size=None)
    n = pred.shape[-1]
    ccpca = np.zeros(n)
    for i in range(n):
        ccpca[i] = np.corrcoef(Y[:, :, i].flatten(), pred[:, :, i].flatten())[0, 1]

    f, ax = plt.subplots()
    ax.plot(ccpca, label=f'cc={np.mean(ccpca):.3f}')
    ax.axhline(np.mean(ccpca), color='r', linestyle='--')
    ax.legend()
    ax.set_title(keywordstub)
    model_pc.meta['r_test'] = ccpca

    return model_pc, ccpca


def pc_fit(keywordstub, datasets, pc_count=10,
           fitkw='lite.tf.init.mi5000.lr1e3.t3.lfse-lite.tf.mi5000.lr1e4.t5e5.lfse'):
    siteids = list(datasets.keys())
    total_pcs = int(len(siteids) * pc_count)

    keywordpc = f"{keywordstub}x{total_pcs}-lvl.{total_pcs}"

    model_pc = Model.from_keywords(keywordpc)
    model_pc.name = f"{keywordpc}_{fitkw}"
    model_pc = model_pc.sample_from_priors()

    log.info(keywordpc)
    log.info(fitkw)
    g = btn_generator(datasets, siteids, batch_size=32)
    log.info(f'g stim shape: {g[0][0].shape}')
    xf_spec = parse_kw_string(fitkw, xforms_lib)

    for fi, xfa in enumerate(xf_spec):
        fitter_options = {}
        backend = xfa[1].get('backend', 'tf')
        initialize_nl = xfa[1].get('initialize_nl', False)
        fitter_options['cost_function'] = xfa[1].get('cost_function', 'squared_error')  # 'nmse'  #
        fitter_options['early_stopping_delay'] = xfa[1].get('early_stopping_delay', 100)
        fitter_options['early_stopping_patience'] = xfa[1].get('early_stopping_patience', 150)
        fitter_options['early_stopping_tolerance'] = xfa[1].get('tolerance', 1e-3)
        fitter_options['validation_split'] = xfa[1].get('validation_split', 0.0)
        fitter_options['learning_rate'] = xfa[1].get('learning_rate', 1e-3)
        fitter_options['epochs'] = xfa[1].get('max_iter', 5000)
        fitter_options['shuffle'] = xfa[1].get('shuffle', True)
        if fi == 0:
            verbose = 1
        else:
            verbose = 0
        if initialize_nl:
            try:
                model_pc.layers[-1].skip_nonlinearity()
            except:
                log.info('No NL to exclude from stage 1 fit')
                initialize_nl = False

        log.info(f'Iteration {fi}, fitter_options: {fitter_options}')
        model_pc = model_pc.fit(input=g, target=None, backend=backend, fitter_options=fitter_options,
                                verbose=verbose, batch_size=None)

        if initialize_nl:
            model_pc.layers[-1].unskip_nonlinearity()

    # check performance on validation data
    X, Y = data_subset(datasets, siteids, output_name='pca', get_fit_data=False)
    pred = model_pc.predict(X, batch_size=None)
    n = pred.shape[-1]
    ccpca = np.zeros(n)
    for i in range(n):
        ccpca[i] = np.corrcoef(Y[:, :, i].flatten(), pred[:, :, i].flatten())[0, 1]

    f, ax = plt.subplots()
    ax.plot(ccpca, label=f'cc={np.mean(ccpca):.3f}')
    ax.axhline(np.mean(ccpca), color='r', linestyle='--')
    ax.legend()
    ax.set_title(keywordstub)
    model_pc.meta['r_test'] = ccpca

    return model_pc, ccpca


def do_bnt_fit(sitecount, keywordstub, batch=343, pc_count=10,
               fitkw='lite.tf.init.mi5000.lr1e3.t3.lfse-lite.tf.mi5000.lr1e4.t5e5.lfse',
               save_results=True):

    startime = time.time()
    log_stream = io.StringIO()
    ch = logging.StreamHandler(log_stream)
    ch.setLevel(logging.DEBUG)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)
    rootlogger = logging.getLogger()
    rootlogger.addHandler(ch)

    exclude_sites = ['CLT027c', 'CLT050c']
    batch = 343

    siteids, cellids = db.get_batch_sites(343)
    siteids = [s for s in siteids if s not in exclude_sites]

    siteids = siteids[:sitecount]
    log.info(siteids)

    datasets = {}
    for i, siteid in enumerate(siteids):
        datasets[siteid] = load_bnt_recording(
            siteid, batch=batch, loadkey="gtgram.fs100.ch18",
            pc_count=pc_count, val_single=False)

    cellids = [d['cellids'] for k,d in datasets.items()]
    cellids = list(itertools.chain(*cellids))
    cell_siteids = [c.split("-")[0] for c in cellids]

    modelspec, ccpca = pc_fit(keywordstub, datasets, fitkw=fitkw, pc_count=pc_count)

    #if get_setting('USE_NEMS_BAPHY_API'):
    #    prefix = 'http://' + get_setting('NEMS_BAPHY_API_HOST') + ":" + str(
    #        get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
    #else:
    #    prefix = get_setting('NEMS_RESULTS_DIR')
    #basepath = os.path.join(prefix, str(batch), f"CLTPRN{sitecount}")

    # use nems-lite model path namer
    #filepath = json.generate_model_filepath(modelspec, basepath=basepath)
    #destination = os.path.dirname(filepath)
    #modelspec.meta['modelpath'] = destination
    #modelspec.meta['figurefile'] = os.path.join(destination, 'figure.0000.png')
    modelspec.meta['runtime'] = int(time.time() - startime)
    modelspec.meta['batch'] = batch
    modelspec.meta['cellid'] = f"CLTPRN{sitecount}"
    modelspec.meta['r_test'] = ccpca
    figures = [n0plt.fig2BytesIO(plt.gcf())]

    log.info('Done evaluating do_bnt_fit.')
    ch.close()
    rootlogger.removeFilter(ch)
    logstring = log_stream.getvalue()
    if save_results:
        savepath = xforms.save_lite(
            xfspec=[['nems_lbhb.projects.bignat.fit_bnt.do_bnt_fit']],
            modelspec=modelspec, log=logstring, figures=figures
        )
        return savepath

    else:
        return modelspec, datasets

