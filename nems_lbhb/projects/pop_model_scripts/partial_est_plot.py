import numpy as np
import os
import io
import logging
import json as jsonlib

from scipy.ndimage import zoom
from scipy.stats import wilcoxon
import sys, importlib
import copy
import pandas as pd

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

savefigs = False

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import load_string_pop, fit_string_pop, load_string_single, fit_string_single, \
    POP_MODELS, SIG_TEST_MODELS, shortnames, shortnamesp, get_significant_cells, PLOT_STAT, \
    modelname_half_prefit, modelname_half_pop, modelname_half_fullfit, \
    modelname_half_heldoutpop, modelname_half_heldoutfullfit, mplparams, single_column_shorter

import matplotlib as mpl
mpl.rcParams.update(mplparams)
import matplotlib.pyplot as plt
import seaborn as sns

#out_path = "/auto/users/svd/projects/pop_models/"
#outpath="/auto/users/svd/docs/current/conf/apan2020/dstrf"
out_path = "/auto/users/svd/docs/current/uo_seminar/eps/"

def partial_est_plot(batch=322, PLOT_STAT='r_ceiling', figsize=None):
    if figsize is None:
        figsize = (8,4)

    sig_cells=get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)
    print(f"len(sig_cells)={len(sig_cells)}")
    # tentative: use best conv1dx2+d
    half_test_modelspec = "wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R"

    # modelname_half_prefit: test condition: take advantage of larger model population fit (hs: heldout),
    # then fit single cell with half a dataset
    # fit last layer on half the data, using prefit with the current site held-out, run per cell

    # modelname_half_fullfit: then fit last layer on heldout cell with half the data (same est data as for
    # modelname_half_prefit), run per cell

    # modelname_half_heldoutfullfit

    pre = nd.batch_comp(batch, modelname_half_prefit, cellids=sig_cells, stat=PLOT_STAT)
    full = nd.batch_comp(batch, modelname_half_fullfit, cellids=sig_cells, stat=PLOT_STAT)
    heldout = nd.batch_comp(batch, modelname_half_heldoutfullfit, cellids=sig_cells, stat=PLOT_STAT)

    d = []
    pcts=['10','15','25','50','100']
    mdls=['LN','dnns','std','prefit']
    for n,h,m,s in zip(pcts,modelname_half_prefit,modelname_half_fullfit,modelname_half_heldoutfullfit):
        d_ = full.loc[:,[m]]
        #d_ = heldout.loc[:,[s]]
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

    ax[0].plot([0, 1], [0, 1], 'k--')
    x1,x2,xlabel='10', 'std', 'Standard'
    y1,y2,ylabel='10', 'prefit','Pre-trained'
    x=dpred.loc[(dpred.midx==x1) & (dpred.fit==x2), [PLOT_STAT]]
    y=dpred.loc[(dpred.midx==y1) & (dpred.fit==y2), [PLOT_STAT]]
    _d = x.merge(y, how='inner', left_index=True, right_index=True, suffixes=('_x','_y'))
    ax[0].scatter(x=_d[PLOT_STAT+'_x'], y=_d[PLOT_STAT+'_y'], s=1, c='k')

    ax[0].set_xlabel(f"{xlabel} (median r={_d[PLOT_STAT+'_x'].median():.3f})")
    ax[0].set_ylabel(f"{ylabel} (median r={_d[PLOT_STAT+'_y'].median():.3f})")
    #ax[0].set_title(f'Batch {batch} {x1}%')
    ax[0].set_xlim([-0.05,1.05])
    ax[0].set_ylim([-0.05,1.05])
    ax[0].set_aspect('equal')

    dpm = dpred.groupby(['midx','fit']).median().reset_index()
    dpm.midx = dpm.midx.astype(int)
    dpm = dpm.pivot(index='midx', columns='fit', values='r_ceiling')

    for midx in pcts:
        x=dpred.loc[(dpred.midx==midx) & (dpred.fit=='std'), [PLOT_STAT]]
        y=dpred.loc[(dpred.midx==midx) & (dpred.fit=='prefit'), [PLOT_STAT]]
        _d = x.merge(y, how='inner', left_index=True, right_index=True, suffixes=('_x','_y'))

        if midx=='100':
            p=0.5
        else:
            w, p = wilcoxon(_d[PLOT_STAT+'_x'], _d[PLOT_STAT+'_y'])

        dpm.loc[int(midx), 'p'] = p

    ax[1].plot(dpm.index, dpm['std'], '-s', color='gray', label=xlabel, markersize=4.5)
    ax[1].plot(dpm.index, dpm['prefit'], '-o', color='k', label=ylabel, markersize=4.5)
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Fraction estimation data')
    ax[1].set_ylim(0.5, 0.7)
    ax[1].set_box_aspect(1)

    return f, dpm


if __name__ == '__main__':

    a1 = 322
    peg = 323

    sf=1.5
    single_column_short = (3.5*sf, 2.5*sf)
    single_column_tall = (3.5*sf, 6*sf)
    column_and_half_short = (5*sf, 2.5*sf)
    column_and_half_tall = (5*sf, 5*sf)
    fig5 = partial_est_plot(batch=a1, PLOT_STAT=PLOT_STAT, figsize=single_column_shorter)
