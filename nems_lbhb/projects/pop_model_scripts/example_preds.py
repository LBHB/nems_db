import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
import importlib

import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 10,
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)

import nems
import nems.db as nd
import nems.xform_helper as xhelp
import nems_lbhb.xform_wrappers as xwrap
import nems.epoch as ep
from nems.xforms import evaluate_step
import nems_lbhb.baphy_io as io
from nems_lbhb import baphy_experiment
from nems.xform_helper import load_model_xform
from nems import xforms
from nems_lbhb.plots import scatter_bin_lin
from nems_lbhb.analysis import pop_models

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import load_string_pop, fit_string_pop, load_string_single, fit_string_single,\
    POP_MODELS, SIG_TEST_MODELS, shortnames, shortnamesp, MODELGROUPS, ALL_FAMILY_MODELS, ALL_FAMILY_POP, \
    a1, peg, base_path, single_column_short, single_column_tall, column_and_half_short, column_and_half_tall


high_res_ctx = None

def load_high_res_stim():
    
    global high_res_ctx
    
    if high_res_ctx is None:
        # load hi-res spectrogram
        batch=322
        cellid="DRX006b-128-2"

        b = baphy_experiment.BAPHYExperiment(batch=batch, cellid=cellid)
        tctx = {'rec': b.get_recording(loadkey="ozgf.fs100.ch64")}
        tctx = xforms.evaluate_step(['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM', 'keepfrac': 1.0}], tctx)
        tctx = xforms.evaluate_step(['nems.xforms.average_away_stim_occurrences', {'epoch_regex': '^STIM'}], tctx)
        high_res_ctx = tctx
        
    return high_res_ctx


def single_cell_examples():
    
    global column_and_half_short
    
    # load hi-res spectrogram    
    batch=322
    cellid="DRX006b-128-2"

    tctx = load_high_res_stim()
    
    example_models=[ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[2], ALL_FAMILY_MODELS[0]+'.l2:4-dstrf']
    example_shortnames=['ln_pop','conv1dx2+d','conv2dx3']

    cellids = ["DRX006b-128-2", "ARM030a-40-2"]   # , "ARM030a-23-2"
    fig, ax = plt.subplots(len(cellids)+1, 1, figsize=column_and_half_short, sharex=True)
    for i, cellid in enumerate(cellids):
        # LN
        xf0,ctx0=load_model_xform(cellid=cellid,batch=batch,modelname=example_models[0])
        xf1,ctx1=load_model_xform(cellid=cellid,batch=batch,modelname=example_models[1],
                                  eval_model=False)
        ctx1['val'] = ctx1['modelspec'].evaluate(rec=ctx0['val'].copy())

        xf2,ctx2=load_model_xform(cellid=cellid,batch=batch,modelname=example_models[2],
                                  eval_model=False)
        ctx2['val'] = ctx2['modelspec'].evaluate(rec=ctx0['val'].copy())

        ctx0['val']['stim'] = tctx['val']['stim']
        pop_models.model_pred_sum(
            [ctx0, ctx1, ctx2], cellid=cellid, rr=np.arange(150, 600),
            predcolor=['orange', 'blue', 'darkgreen'], labels=example_shortnames, ax=[ax[0],ax[i+1]])
        if i<len(cellids)-1:
            ax[i+1].set_xlabel('')

    return fig

def pop_model_example(figsize=None):
    
    tctx = load_high_res_stim()

    batch=322
    cellid="DRX006b-128-2"
    modelname = ALL_FAMILY_POP[2]
    xf, ctx = load_model_xform(cellid, batch, modelname)
    
    modelspec = ctx['modelspec'].copy()
    val=ctx['val'].apply_mask()
    
    # extract a subset of channels, since 9xx is too many
    N=50
    rr=slice(N, N*2, 1)
    modelspec.phi[8]['coefficients']=modelspec.phi[8]['coefficients'][rr,:]
    modelspec.phi[9]['level']=modelspec.phi[9]['level'][rr]
    modelspec.phi[10]['base']=modelspec.phi[10]['base'][rr]
    modelspec.phi[10]['amplitude']=modelspec.phi[10]['amplitude'][rr]
    modelspec.phi[10]['shift']=modelspec.phi[10]['shift'][rr]
    modelspec.phi[10]['kappa']=modelspec.phi[10]['kappa'][rr]
    val['resp']=val['resp']._modified_copy(data=val['resp'][rr,:])
    val['pred']=val['pred']._modified_copy(data=val['pred'][rr,:])
    modelspec.meta['cellid']=modelspec.meta['cellids'][N]
    modelspec.meta['cellids']=modelspec.meta['cellids'][rr]
    modelspec.meta['r_ceiling']=modelspec.meta['r_test'][rr]*1.1
    print(val['stim'].shape, val['resp'].shape, tctx['val']['stim'].shape)
    
    f=pop_models.plot_layer_outputs(modelspec, val, index_range=np.arange(150,600), example_idx=15, figsize=figsize, altstim=tctx['val']['stim']);
    
    return f
    
if __name__ == '__main__':
    
    fig = single_cell_examples()
    filename=base_path / 'fig5_pred_example.pdf'
    fig.savefig(filename, format='pdf', dpi='figure')

    #f=pop_model_example(figsize=column_and_half_tall)
    #filename=base_path / 'fig2_example.pdf'
    #f.savefig(filename, format='pdf', dpi='figure')