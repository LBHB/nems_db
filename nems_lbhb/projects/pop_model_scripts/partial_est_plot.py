import numpy as np
import os
import io
import logging
import matplotlib.pyplot as plt
import json as jsonlib
from scipy.ndimage import zoom
import sys, importlib
import copy
import pandas as pd
import seaborn as sns

import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 12,
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)

import nems.modelspec as ms
import nems.xforms as xforms
from nems.uri import json_numpy_obj_hook
from nems.xform_helper import fit_model_xform, load_model_xform, _xform_exists
from nems.utils import escaped_split, escaped_join, get_setting, find_module
import nems.db as nd
from nems import get_setting
from nems.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems_lbhb.analysis import pop_models
import nems.db as nd
from nems.plots.heatmap import plot_heatmap
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

log = logging.getLogger(__name__)

savefigs=False



from nems_lbhb.projects.pop_model_scripts.pop_model_utils import load_string_pop, fit_string_pop, load_string_single, fit_string_single, \
    POP_MODELS, SIG_TEST_MODELS, shortnames, shortnamesp, get_significant_cells

#out_path = "/auto/users/svd/projects/pop_models/"
#outpath="/auto/users/svd/docs/current/conf/apan2020/dstrf"
out_path = "/auto/users/svd/docs/current/uo_seminar/eps/"

def partial_est_plot(batch=322, PLOT_STAT='r_ceiling', figsize=None):
    if figsize is None:
        figsize = (8,4)

    sig_cells=get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)

    # tentative: use best conv1dx2+d
    half_test_modelspec = "wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R"

    # test condition: take advantage of larger model population fit (hs: heldout), then fit single cell with half a dataset
    # fit last layer on half the data, using prefit with the current site held-out, run per cell
    modelname_half_prefit=[f"ozgf.fs100.ch18-ld-norm.l1-sev.k10_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                           f"ozgf.fs100.ch18-ld-norm.l1-sev.k15_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                           f"ozgf.fs100.ch18-ld-norm.l1-sev.k25_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                           f"ozgf.fs100.ch18-ld-norm.l1-sev.k50_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                           f"ozgf.fs100.ch18-ld-norm.l1-sev_{half_test_modelspec}_prefit.hm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"]

    # then fit last layer on heldout cell with half the data (same est data as for modelname_half_prefit), run per cell
    modelname_half_fullfit=[f"ozgf.fs100.ch18-ld-norm.l1-sev.k10_{half_test_modelspec}_prefit.htm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                            f"ozgf.fs100.ch18-ld-norm.l1-sev.k15_{half_test_modelspec}_prefit.hfm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                            f"ozgf.fs100.ch18-ld-norm.l1-sev.k25_{half_test_modelspec}_prefit.hqm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                            f"ozgf.fs100.ch18-ld-norm.l1-sev.k50_{half_test_modelspec}_prefit.hhm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                            f"ozgf.fs100.ch18-ld-norm.l1-sev_{half_test_modelspec}_prefit.hm-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"]


    modelname_half_heldoutfullfit=[f"ozgf.fs100.ch18-ld-norm.l1-sev.k10_{half_test_modelspec}_prefit.hts-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                                   f"ozgf.fs100.ch18-ld-norm.l1-sev.k15_{half_test_modelspec}_prefit.hfs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                                   f"ozgf.fs100.ch18-ld-norm.l1-sev.k25_{half_test_modelspec}_prefit.hqs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                                   f"ozgf.fs100.ch18-ld-norm.l1-sev.k50_{half_test_modelspec}_prefit.hhs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
                                   f"ozgf.fs100.ch18-ld-norm.l1-sev_{half_test_modelspec}_prefit.hs-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"]

    pre = nd.batch_comp(batch, modelname_half_prefit, cellids=sig_cells, stat=PLOT_STAT)
    full = nd.batch_comp(batch, modelname_half_fullfit, cellids=sig_cells, stat=PLOT_STAT)
    heldout = nd.batch_comp(batch, modelname_half_heldoutfullfit, cellids=sig_cells, stat=PLOT_STAT)


    d = []
    pcts=['10','15','25','50','100']
    mdls=['LN','dnns','std','prefit']
    for n,h,m,s in zip(pcts,modelname_half_prefit,modelname_half_fullfit,modelname_half_heldoutfullfit):
        d_ = full.loc[:,[m]]
        d_.columns=[PLOT_STAT]
        d_['midx']=n
        d_['fit']="std"
        d.append(d_)
        d_ = pre.loc[:,[h]]
        d_.columns=[PLOT_STAT]
        d_['midx']=n
        d_['fit']="prefit"
        d.append(d_)

        # for now, don't plot the held-out within partial. too restrictive
        # and no clear benefit
        #d_ = heldout.loc[:,[s]]
        #d_.columns=[PLOT_STAT]
        #d_['midx']=n
        #d_['fit']="heldout"
        #d.append(d_)

    dpred=pd.concat(d)

    f,ax=plt.subplots(1,2,figsize=figsize)

    ax[0].plot([0,1],[0,1],'k--')
    x1,x2='10','std'
    y1,y2='10','prefit'
    x=dpred.loc[(dpred.midx==x1) & (dpred.fit==x2),'r_ceiling']
    y=dpred.loc[(dpred.midx==y1) & (dpred.fit==y2),'r_ceiling']
    sns.scatterplot(x=x, y=y, ax=ax[0])

    ax[0].set_xlabel(f"{x2} {x1}% {x.median():.3f}")
    ax[0].set_ylabel(f"{y2} {y1}% {y.median():.3f}")
    ax[0].set_title(f'batch {batch} {x2} vs {y2}')
    ax[0].set_xlim([-0.05,1.05])
    ax[0].set_ylim([-0.05,1.05])

    dpm = dpred.groupby(['midx','fit']).median().reset_index()
    dpm.midx = dpm.midx.astype(int)
    dpm = dpm.pivot(index='midx',columns='fit', values='r_ceiling')
    dpm.plot(ax=ax[1], legend=False)
    ax[1].legend(frameon=False)

    print(dpm)

    return f