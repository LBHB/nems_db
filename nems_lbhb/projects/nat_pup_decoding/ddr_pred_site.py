import numpy as np
import os
import io
import logging
import time
import sys, importlib

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join, smooth
from nems import get_setting
from nems.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import POP_MODELS, SIG_TEST_MODELS
from nems import db
from nems.recording import load_recording
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems import epoch as ep
from charlieTools.nat_sounds_ms.decoding import plot_stimulus_pair
import nems_lbhb.projects.nat_pup_decoding.decoding as decoding

log = logging.getLogger(__name__)

ALL_SITES = ['BOL005c', 'BOL006b', 'bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128',
            'CRD016d', 'CRD017c',
            'TNC008a', 'TNC009a', 'TNC010a', 'TNC012a', 'TNC013a', 'TNC014a', 'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a', 'TNC020a']

def ddr_pred_site_sim(site, batch=None, modelname_base=None, save_fig=False, skip_plot=False):
    if batch is None:
        batch=331
        cellid = [c for c in db.get_batch_cells(batch).cellid if site in c][0]

        if len(cellid)==0:
            batch=322

    cellid = [c for c in db.get_batch_cells(batch).cellid if site in c][0]
    if len(cellid)==0:
        raise ValueError(f"No match for site {site} batch {batch}")

    states = ['st.pca0.pup+r1+s0,1', 'st.pca.pup+r1+s0,1', 'st.pca.pup+r1+s1', 'st.pca.pup+r1']

    if batch == 331:
        if modelname_base is None:
            modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                             "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                             "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3"
            #modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
            #                 "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
            #                 "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss4"
        resp_modelname = f"psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{states[-1]}-plgsm.p2-aev-rd.resp" + \
                         "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                         "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
    else:
        #modelname_base = "psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
        #                 "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
        #                 "_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r4-aev-ccnorm.md.t5.f0.ss3"
        modelname_base = "psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                         "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                         "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3"
        resp_modelname = f"psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{states[-1]}-plgsm.p2-aev-rd.resp" + \
                         "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                         "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
    log.info(f"site {site} modelname_base: {modelname_base}")
    modelnames = [resp_modelname] + [modelname_base.format(s) for s in states]
    labels = ['actual'] + states

    mse = np.zeros((4, 3))
    cc = np.zeros((4, 3))

    f, ax = plt.subplots(4, 5, figsize=(6, 5), sharex='row', sharey='row');
    for i, m in enumerate(modelnames):
        modelpath = db.get_results_file(batch=batch, modelnames=[m], cellids=[cellid]).iloc[0]["modelpath"]

        loader = decoding.DecodingResults()
        raw_res = loader.load_results(os.path.join(modelpath, "decoding_TDR.pickle"))

        raw_df = raw_res.numeric_results
        raw_df = raw_df.loc[pd.IndexSlice[raw_res.evoked_stimulus_pairs, 2], :].copy()

        if i == 0:
            mmraw0 = raw_df[["sp_dp", "bp_dp"]].values.max()
        ax[0, i].plot([0, mmraw0], [0, mmraw0], 'k--', lw=0.5)
        ax[0, i].scatter(raw_df["sp_dp"], raw_df["bp_dp"], s=3)
        #a, b = 'delta_pred', 'delta_act'
        a, b='delta_pred_raw', 'delta_act_raw'

        if i == 0:
            #raw_df = raw_df.loc[(raw_df["bp_dp"]>10) & (raw_df["sp_dp"]>10)]
            #raw_df = raw_df.loc[(raw_df["bp_dp"]<60) & (raw_df["sp_dp"]<60)]
            raw_df.loc[:, 'delta_act_raw'] = (raw_df["bp_dp"] - raw_df["sp_dp"])
            raw_df.loc[:, 'delta_act'] = (raw_df["bp_dp"] - raw_df["sp_dp"]) / (raw_df["bp_dp"] + raw_df["sp_dp"])
            raw_df.loc[:, 'bp_dp_act'] = raw_df["bp_dp"]
            raw_df.loc[:, 'sp_dp_act'] = raw_df["sp_dp"]
            resp_df = raw_df[['sp_dp_act', 'bp_dp_act', 'delta_act_raw', 'delta_act']]
            ax[1, 0].set_axis_off()
            ax[2, 0].set_axis_off()
            ax[3, 0].set_axis_off()
            mmraw = np.max(np.abs(raw_df[['delta_act_raw']].values))
            mmnorm = np.max(np.abs(raw_df[['delta_act']].values))
            ax[0, i].set_title(f"{labels[i]} n={len(raw_df)}")
        else:

            raw_df.loc[:, 'delta_pred_raw'] = (raw_df["bp_dp"] - raw_df["sp_dp"])
            raw_df.loc[:, 'delta_pred'] = (raw_df["bp_dp"] - raw_df["sp_dp"]) / (raw_df["bp_dp"] + raw_df["sp_dp"])
            raw_df = raw_df.merge(resp_df, how='inner', left_index=True, right_index=True)

            cc[i-1, 0] = np.corrcoef(raw_df['bp_dp'], raw_df['bp_dp_act'])[0, 1]
            mse[i-1, 0] = np.std(raw_df['bp_dp']-raw_df['bp_dp_act'])
            ax[1, i].scatter(raw_df['bp_dp'], raw_df['bp_dp_act'], s=3, alpha=0.4)
            ax[1, i].set_title(f"{cc[i-1, 0]:.3f}")

            a, b='delta_pred_raw', 'delta_act_raw'
            ax[2, i].plot([-mmraw, mmraw], [-mmraw, mmraw], 'k--', lw=0.5)
            ax[2, i].scatter(raw_df[a], raw_df[b], s=3, alpha=0.4)
            cc[i-1, 1] = np.corrcoef(raw_df[a], raw_df[b])[0, 1]
            mse[i-1, 1] = np.std(raw_df[a]-raw_df[b])
            ax[2, i].set_title(f"e={mse[i-1,1]:.1f} cc={cc[i-1,1]:.3f}")

            a, b = 'delta_pred', 'delta_act'
            ax[3, i].plot([-mmnorm, mmnorm], [-mmnorm, mmnorm], 'k--', lw=0.5)
            ax[3, i].scatter(raw_df[a], raw_df[b], s=3, alpha=0.4)
            cc[i-1,2] = np.corrcoef(raw_df[a], raw_df[b])[0, 1]
            mse[i-1,2] = np.std(raw_df[a]-raw_df[b])
            ax[3, i].set_title(f"e={mse[i-1,2]:.3f} cc={cc[i-1,2]:.3f}")

            ax[0, i].set_title(f"{labels[i]}")

    ax[0, 0].set_ylabel('big pupil dp')
    ax[0, 0].set_xlabel('small pupil dp')
    ax[1, 1].set_ylabel('actual big dp')
    ax[1, 1].set_xlabel('pred big dp')
    ax[2, 1].set_xlabel('pred delta raw')
    ax[2, 1].set_ylabel('act delta raw')
    ax[3, 1].set_xlabel('pred delta norm')
    ax[3, 1].set_ylabel('act delta norm')
    pupil_range = raw_res.pupil_range['range'].mean()
    f.suptitle(f"{site} - {batch} - puprange {pupil_range:.3f}")
    plt.tight_layout()

    if save_fig:
        f.savefig(f'/auto/users/svd/projects/pop_state/ddr_pred_{site}_{batch}.jpg')
    if skip_plot:
        plt.close(f)

    return labels[1:], cc, mse, pupil_range


def ddr_sum_all(batch=331, modelname_base=None):
    siteids, cellids = db.get_batch_sites(batch=batch)

    res = []
    for site in siteids:
        try:
            labels, cc, mse, pupil_range = ddr_pred_site_sim(
                site, batch=batch, modelname_base=modelname_base,
                skip_plot=True)
            labels2=[l+'_raw' for l in labels]

            d = {'site': site, 'batch': batch,
                 'pupil_range': pupil_range, 'cc_base': cc[0,0]}
            for l,c,m in zip(labels+labels2, cc[:,1:].T.flatten(), mse[:,1:].T.flatten()):
                d[l+'_cc'] = c
                #d[l+'_mse'] = m

            res.append(pd.DataFrame(d, index=[0]))
        except:
            print(f"Skipping site {site}")
    df = pd.concat(res, ignore_index=True)

    labels2 = [l+'_raw' for l in labels]
    f, ax = plt.subplots(2, 2, figsize=(5,5))
    #s = (df['pupil_range'] - df['pupil_range'].min()) * 50 + 1
    s = (df['cc_base']-df['cc_base'].min())*50+1
    mmin = np.nanmin(df[[labels[1]+'_cc',labels[2]+'_cc',labels[3]+'_cc']].values)
    mmax = np.nanmax(df[[labels[1]+'_cc',labels[2]+'_cc',labels[3]+'_cc']].values)
    ax[0,0].plot([mmin,mmax],[mmin,mmax],'k--',lw=0.5)
    ax[0,0].scatter(df[labels[1]+'_cc'],df[labels[2]+'_cc'],s=s)
    ax[0,0].set_ylabel('first-order')
    ax[0,0].set_xlabel('shuff')
    ax[0,1].plot([mmin,mmax],[mmin,mmax],'k--',lw=0.5)
    ax[0,1].scatter(df[labels2[1]+'_cc'],df[labels2[2]+'_cc'],s=s)
    ax[0,1].set_title('raw cc')

    ax[1,0].plot([mmin,mmax],[mmin,mmax],'k--',lw=0.5)
    ax[1,0].scatter(df[labels[2]+'_cc'],df[labels[3]+'_cc'],s=s)
    ax[1,0].set_ylabel('pup LV')
    ax[1,0].set_xlabel('first-order')
    ax[1,1].plot([mmin,mmax],[mmin,mmax],'k--',lw=0.5)
    ax[1,1].scatter(df[labels2[2]+'_cc'],df[labels2[3]+'_cc'],s=s)
    f.suptitle(f'batch {batch}')
    plt.tight_layout()
    #f.savefig(f'/auto/users/svd/projects/pop_state/ddr_pred_{batch}_all.jpg')

    return df, f

if __name__ == '__main__':
    site, batch = "CRD016d", 322
    site, batch = "TNC014a", 331
    site, batch = "TAR010c", 322
    site, batch = "AMT026a", 331
    site, batch = "ARM031a", 331
    site, batch = "CRD004a", 331
    site, batch = "CRD005b", 331
    site, batch = "CRD018d", 331
    site, batch = "TNC008a", 331
    site, batch = "TNC009a", 331
    site, batch = "TNC011a", 331
    site, batch = "TAR010c", 322
    site, batch = "TAR017b", 322
    site, batch = "TNC006a", 331
    site, batch = "AMT020a", 331
    site, batch = "AMT021b", 331
    site, batch = "AMT026a", 331
    site, batch = "ARM005e", 331
    site, batch = "ARM029a", 331

    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"
    modelname_base=None

    labels, cc, mse, pupil_range = ddr_pred_site_sim(site, batch=batch, modelname_base=modelname_base)

