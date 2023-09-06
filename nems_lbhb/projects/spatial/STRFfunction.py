import logging
import pickle
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nems0.analysis.api
import nems0.initializers
import nems0.preprocessing as preproc
from nems0.uri import save_resource
from nems0 import db
from nems0 import xforms
from nems0 import recording
from nems0.fitters.api import scipy_minimize
from nems0.signal import RasterizedSignal
import nems0.epoch as ep
from nems import Model
from nems.models import LN
from nems_lbhb.projects.spatial.models import LN_Tiled_STRF
from nems0 import get_setting
import json

from nems.visualization.model import plot_nl
from nems.metrics import correlation
from nems.tools.json import save_model, load_model, nems_to_json

from nems_lbhb.xform_wrappers import generate_recording_uri
log = logging.getLogger(__name__)

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems0.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems0.NEMS_PATH) / 'modelspecs'



def load_data(site,stim_format,architecture="LN_STRF"):
    siteid=site
    batch=338
    siteids, cellids = db.get_batch_sites(batch)
    if stim_format in ['binaural_HRTF','gtgram.fs100.ch18.bin6']:
        loadkey = "gtgram.fs100.ch18.bin6"   
    elif stim_format in ['binaural_allocentric','gtgram.fs100.ch18.bin100']:
        loadkey = "gtgram.fs100.ch18.bin100"  
    elif stim_format in ['monaural','gtgram.fs100.ch18.mono']: 
        loadkey = "gtgram.fs100.ch18.mono"
    else:
        raise ValueError(f"Unknown stim_format {stim_format}")
        
    recording_uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey)
    rec = recording.load_recording(recording_uri)

    cellnum = rec['resp'].shape[0] 
    ctx = {'rec': rec}
    ctx.update(xforms.normalize_sig(sig='resp', norm_method='minmax', **ctx))
    ctx.update(xforms.normalize_sig(sig='stim', norm_method='minmax', **ctx))
    ctx.update(xforms.split_by_occurrence_counts(epoch_regex='^STIM', **ctx))
    ctx.update(xforms.average_away_stim_occurrences(epoch_regex='^STIM', **ctx))
    cid = 0
    
    return cellnum, rec, ctx, loadkey, siteid, siteids

def fitSTRF(site,stim_format,cellnum, ctx,loadkey,architecture="LN_STRF", cellid=None):
    batch = 338
    epochs = ctx['est']['resp'].epochs
    stim_epochs = ep.epoch_names_matching(epochs, "^STIM_")
    mono_epochs = [e for e in stim_epochs if e.startswith("STIM_NULL") | e.endswith("NULL:2")]
    bin_epochs = [e for e in stim_epochs if (e.startswith("STIM_NULL") is False) & (e.endswith("NULL:2") is False)]
    val_epochs = ep.epoch_names_matching(ctx['val']['resp'].epochs, "^STIM_")


    rlist = []
    strflist = []
    cell_list = []

    cellids = ctx['est']['resp'].chans
    if cellid is None:
        cellnumlist = range(len(chans))
    else:
        cellnumlist = [i for i,c in enumerate(cellids) if c==cellid]

    for cid in cellnumlist:
        cellid = ctx['est']['resp'].chans[cid]
        log.info(f"Fitting model for cell {cellid} ({loadkey}, {architecture})")
        X_ = ctx['est']['stim'].extract_epochs(stim_epochs)
        Y_ = ctx['est']['resp'].extract_epochs(stim_epochs)
        # convert to matrix
        X_est = np.stack([X_[k][0,:,:].T for k in X_.keys()], axis=0)
        Y_est = np.stack([Y_[k][0,[cid],:].T for k in X_.keys()], axis=0)

        X_ = ctx['val']['stim'].extract_epochs(val_epochs)
        Y_ = ctx['val']['resp'].extract_epochs(val_epochs)
        # convert to matrix
        X_val = np.stack([X_[k][0,:,:].T for k in X_.keys()], axis=0)
        Y_val = np.stack([Y_[k][0,[cid],:].T for k in X_.keys()], axis=0)

        #fit single sound STRF
        if architecture == "LN_Tiled_STRF":
            input_channels = int(X_est.shape[2]/2)
            time_lags = 16
            rank = 5
            strf_base = LN_Tiled_STRF(time_lags, input_channels, rank=rank, gaussian=False, fs=ctx['est']['resp'].fs)
        else:
            input_channels = X_est.shape[2]
            time_lags = 16
            rank = 5
            strf_base = LN.LN_STRF(time_lags, input_channels, rank=rank, gaussian=False, fs=ctx['est']['resp'].fs) 
        
        strf = strf_base.fit_LBHB(X_est, Y_est) 


        predict = strf.predict(X_val, batch_size=None)
        r=correlation(predict, Y_val)
        r= np.round(r,3)
        rlist.append(r)

        # save some metadata
        strf.meta['cellid'] = cellid
        strf.meta['loadkey'] = loadkey
        strf.meta['r_test'] = r
        strf.meta['architecture'] = architecture
        strf.meta['siteid'] = site
        strf.meta['batch']=batch

        """
        # save the model fit to a unqiue path/file for this stim_format, model architecture and cellid
        save_path = f"/auto/users/alexis/results/{loadkey}/{architecture}/{cellid}.json"
        print(f"Saving model fit to {save_path}")
        save_model(strf, save_path)
        """

        #replace above code with this to save to the right place:

        
        if get_setting('USE_NEMS_BAPHY_API'):
            prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
        else:
            prefix = get_setting('NEMS_RESULTS_DIR')
    
        filepath = f"{prefix}/{batch}/{loadkey}/{architecture}/{cellid}.json"
        # call nems-lite JSON encoder
        data = nems_to_json(strf)
        save_resource(filepath, data=data)

        
        
        strflist.append(strf)
        cell_list.append(cellid)
        
    return rlist, strflist, r, strf, ctx, cell_list


def plotSTRF(site,stim_format,strflist, rlist, cellnum, ctx, architecture="LN_STRF"):

    epochs = ctx['est']['resp'].epochs
    stim_epochs = ep.epoch_names_matching(epochs, "^STIM_")
    val_epochs = ep.epoch_names_matching(ctx['val']['resp'].epochs, "^STIM_")
    
    siteid=site
    cid = 0
    f = plt.figure(figsize=(9,9))
    
    for i in range(cellnum):
# pick a cell
        cid=i
        cellid = ctx['est']['resp'].chans[cid]
        X_ = ctx['est']['stim'].extract_epochs(stim_epochs)
        Y_ = ctx['est']['resp'].extract_epochs(stim_epochs)
# convert to matrix
        X_est = np.stack([X_[k][0,:,:].T for k in X_.keys()], axis=0)
        Y_est = np.stack([Y_[k][0,[cid],:].T for k in X_.keys()], axis=0)

        X_ = ctx['val']['stim'].extract_epochs(val_epochs)
        Y_ = ctx['val']['resp'].extract_epochs(val_epochs)
# convert to matrix
        X_val = np.stack([X_[k][0,:,:].T for k in X_.keys()], axis=0)
        Y_val = np.stack([Y_[k][0,[cid],:].T for k in X_.keys()], axis=0)
    
        strf=strflist[i]
        predict = strf.predict(X_val, batch_size=None)
        r=correlation(predict, Y_val)
 
        ax = f.add_subplot(7,5,i+1)

        strf1 = strf.get_strf()
        channels_out = strf1.shape[-1]
    
        mm = np.nanmax(abs(strf1))

        extent = [0, strf1.shape[1] / strf.fs, 0, strf1.shape[0]] #this changes the axes to be in seconds instead of overall duration

        ax.imshow(strf1, aspect='auto', interpolation='none', origin='lower',cmap='bwr', vmin=-mm, vmax=mm, extent=extent) # graphing the strf
        ax.axhline(y=17.5, ls='--', color='black', lw=0.5) # the horizontal line separating contralateral + ipsilateral ear on the strf
    
        if strf.fs is not None:
            ax.set_xlabel('Time lag (s)')
        else:
            ax.set_xlabel('Time lag (bins)')
    
        ax.set_ylabel('Input channel')
        ax.set_title(f"{cellid} r={r:.3f}") #cell title
        for ax in f.get_axes():
            ax.label_outer() 

    f.suptitle(f"{siteid}, model = {stim_format}") #figure title
    plt.tight_layout()
    f.savefig(f"/auto/users/alexis/results/STRF_figures/{siteid}/{stim_format}/[{siteid}]_strfs.jpg")

    return rlist, strflist


def get_strf(model, channels=None):
    if model.layers[0].coefficients.ndim == 2:
        wc = model.layers[0].coefficients
        fir = model.layers[1].coefficients
        strf1 = wc @ fir.T
    else: 
        wc = model.layers[0].coefficients[:, :, 0]
        fir = model.layers[1].coefficients
        wc2 = model.layers[2].coefficients
        strf_ = wc @ fir.T
        strf1 = np.concatenate((strf_ * wc2[0, 0], strf_ * wc2[1, 0]), axis=0)
        
    return strf1







