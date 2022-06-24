"""
demo code:

from nems_lbhb.projects.nat_pup_decoding import ddr_pred_site

site='AMT020a'
batch=331
modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                 "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                 "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
modelnames, states = ddr_pred_site.parse_modelname_base(modelname_base)

# plot single site
labels, cc, mse, pupil_range = ddr_pred_site.ddr_pred_site_sim(site, batch=batch, modelname_base=modelname_base, save_fig=False);

# compute and plot summary
df, labels = ddr_pred_site.ddr_sum_all(batch=331, modelname_base=modelname_base)
ddr_pred_site.ddr_plot_pred_sum(df, labels, modelnames)


"""



import numpy as np
import os
import io
import logging
import time
import sys, importlib

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

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

def parse_modelname_base(modelname_base, batch=None):

    if batch is None:
        if 'epcpn' in modelname_base:
            batch = 331
        else:
            batch = 322
    if 'plgsm.er5' in modelname_base:
        short_set=True
    else:
        short_set=False

    if short_set:
        resp_modelname = f"psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{'st.pca.pup+r1'}-plgsm.er5-aev-rd.resp" + \
                     "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"

    elif batch == 331:
        resp_modelname = f"psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{'st.pca.pup+r1'}-plgsm.p2-aev-rd.resp" + \
                     "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
    else:
        resp_modelname = f"psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{'st.pca.pup+r1'}-plgsm.p2-aev-rd.resp" + \
                         "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                         "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"

    if 'lvnorm.6' in str(modelname_base):
        states = ['st.pca0.pup+r3+s0,1,2,3', 'st.pca.pup+r3+s0,1,2,3',
                  'st.pca.pup+r3+s1,2,3', 'st.pca.pup+r3+s2,3', 'st.pca.pup+r3+s3', 'st.pca0.pup+r3+s3']
    elif 'lvnorm.5' in str(modelname_base):
        states = ['st.pca0.pup+r2+s0,1,2', 'st.pca.pup+r2+s0,1,2',
                  'st.pca.pup+r2+s1,2', 'st.pca.pup+r2+s2', 'st.pca.pup+r2', 'st.pca0.pup+r2']
        #states = ['st.pca0.pup+r2+s0,1,2', 'st.pca0.pup+r2+s0,1,2',
        #          'st.pca0.pup+r2+s1,2', 'st.pca0.pup+r2+s2', 'st.pca0.pup+r2', 'st.pca0.pup+r2']
    else:
        states = ['st.pca0.pup+r1+s0,1', 'st.pca.pup+r1+s0,1', 'st.pca.pup+r1+s1', 'st.pca.pup+r1']

    modelnames = [resp_modelname] + [modelname_base.format(s) for s in states]

    return modelnames, states


def ddr_pred_site_sim(site, batch=None, modelname_base=None, just_return_mean_dp=False, save_fig=False, skip_plot=False):
    if batch is None:
        batch=331
        cellid = [c for c in db.get_batch_cells(batch).cellid if site in c][0]

        if len(cellid)==0:
            batch=322

    cellid = [c for c in db.get_batch_cells(batch).cellid if site in c][0]
    if len(cellid)==0:
        raise ValueError(f"No match for site {site} batch {batch}")

    if batch == 331:
        if modelname_base is None:
            modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                             "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                             "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    else:
        modelname_base = "psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                         "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3" + \
                         "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3"
    log.info(f"site {site} modelname_base: {modelname_base}")

    modelnames, states = parse_modelname_base(modelname_base)
    labels = ['actual'] + states

    mse = np.zeros((len(modelnames)-1, 3))
    cc = np.zeros((len(modelnames)-1, 3))

    if just_return_mean_dp:
        dp = np.zeros((len(modelnames),2))
        for i, m in enumerate(modelnames):
            modelpath = db.get_results_file(batch=batch, modelnames=[m], cellids=[cellid]).iloc[0]["modelpath"]

            loader = decoding.DecodingResults()
            raw_res = loader.load_results(os.path.join(modelpath, "decoding_TDR.pickle"))

            raw_df = raw_res.numeric_results
            raw_df = raw_df.loc[pd.IndexSlice[raw_res.evoked_stimulus_pairs, 2], :].copy()

            if 'mask_bins' in raw_res.meta.keys():
                mask_bins = raw_res.meta['mask_bins']
                fit_combos = [k for k, v in raw_res.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                                    (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
                all_combos = raw_res.evoked_stimulus_pairs
                val_combos = [c for c in all_combos if c not in fit_combos]
            s=raw_df["sp_dp"]/(raw_df["sp_dp"]+raw_df["bp_dp"])
            l=raw_df["bp_dp"]/(raw_df["sp_dp"]+raw_df["bp_dp"])
            dp[i,:] = [s.mean(), l.mean]
        return labels,dp
            
    f, ax = plt.subplots(4, len(modelnames), figsize=(8, 6), sharex='row', sharey='row');
    for i, m in enumerate(modelnames):
        modelpath = db.get_results_file(batch=batch, modelnames=[m], cellids=[cellid]).iloc[0]["modelpath"]

        loader = decoding.DecodingResults()
        raw_res = loader.load_results(os.path.join(modelpath, "decoding_TDR.pickle"))

        raw_df = raw_res.numeric_results
        raw_df = raw_df.loc[pd.IndexSlice[raw_res.evoked_stimulus_pairs, 2], :].copy()

        if 'mask_bins' in raw_res.meta.keys():
            mask_bins = raw_res.meta['mask_bins']
            fit_combos = [k for k, v in raw_res.mapping.items() if (('_'.join(v[0].split('_')[:-1]), int(v[0].split('_')[-1])) in mask_bins) & \
                                                                (('_'.join(v[1].split('_')[:-1]), int(v[1].split('_')[-1])) in mask_bins)]
            all_combos = raw_res.evoked_stimulus_pairs
            val_combos = [c for c in all_combos if c not in fit_combos]
            import pbd; pdb.set_trace()
            
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

            x = np.concatenate((raw_df['bp_dp'], raw_df['sp_dp']))
            y = np.concatenate((raw_df['bp_dp_act'], raw_df['sp_dp_act']))
            cc[i-1, 0] = np.corrcoef(x, y)[0, 1]
            mse[i-1, 0] = np.std(x-y)

            ax[1, i].scatter(raw_df['bp_dp'], raw_df['bp_dp_act'], s=3, alpha=0.4)
            ax[1, i].set_title(f"{cc[i-1, 0]:.3f}")

            # normed dp
            a, b='delta_pred', 'delta_act'
            ax[2, i].plot([-mmnorm, mmnorm], [-mmnorm, mmnorm], 'k--', lw=0.5)
            ax[2, i].scatter(raw_df[a], raw_df[b], s=3, alpha=0.4)
            cc[i-1, 1] = np.corrcoef(raw_df[a], raw_df[b])[0, 1]
            mse[i-1, 1] = np.std(raw_df[a]-raw_df[b])
            ax[2, i].set_title(f"e={mse[i-1,1]:.1f} cc={cc[i-1,1]:.3f}")

            # raw dp
            a, b = 'delta_pred_raw', 'delta_act_raw'
            ax[3, i].plot([-mmraw, mmraw], [-mmraw, mmraw], 'k--', lw=0.5)
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
            labelsabs = [l + '_abs_cc' for l in labels]
            labelscc = [l + '_cc' for l in labels]
            labelsraw = [l + '_raw_cc' for l in labels]
            labelsmabs = [l + '_abs_mse' for l in labels]
            labelsmse = [l + '_mse' for l in labels]
            labelsmraw = [l + '_raw_mse' for l in labels]

            d = {'site': site, 'batch': batch,
                 'pupil_range': pupil_range, 'cc_base': cc[0,0]}
            for i in range(len(labelscc)):
                d[labelsabs[i]]=cc[i,0]
                d[labelscc[i]]=cc[i,1]
                d[labelsraw[i]]=cc[i,2]
                d[labelsmabs[i]]=mse[i,0]
                d[labelsmse[i]]=mse[i,1]
                d[labelsmraw[i]]=mse[i,2]

            res.append(pd.DataFrame(d, index=[0]))
        except:
            print(f"Skipping site {site}")
            plt.close()

    df = pd.concat(res, ignore_index=True)

    labels = [labelsabs,labelscc, labelsraw , labelsmabs,labelsmse, labelsmraw]

    return df, labels


def ddr_plot_pred_sum(df, labels, modelnames, thr=0.45, thr_name=None):
    labelsabs, labelscc, labelsraw, labelmraw, labelsmse, labelsmraw = labels
    fontsize=8
    if len(modelnames)==5:
        c1,c2,c3=[1,2,3]
    elif len(modelnames)==7:
        c1,c2,c3=[2,3,4]

    plt.close('all')
    f1, ax = plt.subplots(len(labels), 1, figsize=(4,10), sharex=True)
    cats = ['ccabs','ccnorm','ccraw','mseabs','msenorm','mseraw']
    shortlabels=[l.replace('_cc','').replace('st.','') for l in labelscc]
    for i in range(len(labels)):
        d = df[labels[i]]
        d.columns=shortlabels
        sns.stripplot(data=d, s=2, ax=ax[i])
        ax[i].errorbar(np.arange(len(labels[i])), d.median(),
                        d.std()/np.sqrt(len(d)))
        ax[i].set_ylabel(cats[i])
        #sns.pointplot(data=df[labels[i]], ci=95, color='black', lw=1, markers='', join=False, s=0, ax=ax[i])
    ax[-1].set_xticklabels(shortlabels, rotation = 45)
    plt.tight_layout()

    f2, ax = plt.subplots(len(labels),2, figsize=(4,10))
    if thr_name==None:
        thr_name='pupil_range'
    print(f'threshold col: {thr_name} val: {thr}')
    s=df[thr_name].copy()
    pidx=s>thr
    s = 30*(s-s.min())+1
    print(f"thr={thr}, valid={pidx.sum()}/{len(pidx)}")
    for i in range(len(labels)):
        l=labels[i]
        shortlabels=[l.replace('_cc','').replace('st.','') for l in labels[i]]

        d = df[labels[i]]
        mmin = np.nanmin(df[[l[c1],l[c2],l[c3]]].values)
        mmax = np.nanmax(df[[l[c1],l[c2],l[c3]]].values)
        ax[i,0].plot([mmin,mmax],[mmin,mmax],'k--',lw=0.5)
        a=df[l[c2]]
        b=df[l[c3]]

        W,p = stats.wilcoxon(a[pidx],b[pidx])
        ax[i,0].scatter(df[l[c2]],df[l[c3]],s=s)
        ax[i,0].set_xlabel(shortlabels[c2], fontsize=fontsize)
        ax[i,0].set_ylabel(shortlabels[c3], fontsize=fontsize)
        ax[i,0].set_title(f"{cats[i]} {p:.3f}", fontsize=fontsize)

        a=df[l[c1]][s>thr]
        b=df[l[c3]][s>thr]
        W,p = stats.wilcoxon(a[pidx],b[pidx])
        ax[i,1].plot([mmin,mmax],[mmin,mmax],'k--',lw=0.5)
        ax[i,1].scatter(df[l[c1]],df[l[c3]],s=s)
        ax[i,1].set_xlabel(shortlabels[c1], fontsize=fontsize)
        ax[i,1].set_ylabel(shortlabels[c3], fontsize=fontsize)
        ax[i,1].set_title(f"{cats[i]} {p:.3f}", fontsize=fontsize)
    plt.tight_layout()

    return f1, f2



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

    modelname_base = None

    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"
    labels, cc, mse, pupil_range = ddr_pred_site_sim(site, batch=batch, modelname_base=modelname_base)

