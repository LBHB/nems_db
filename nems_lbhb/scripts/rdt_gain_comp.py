import os

from nems_lbhb.stateplots import beta_comp
import matplotlib.pyplot as plt
from nems import xforms
import nems_lbhb.xform_wrappers as nw
from nems.gui.recording_browser import browse_recording, browse_context
import nems.db as nd
import nems.modelspec as ms
import nems.xform_helper as xhelp
from nems_db.params import fitted_params_per_batch, fitted_params_per_cell, get_batch_modelspecs
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_ind, pearsonr
from nems.plots.utils import ax_remove_box

outpath='/auto/users/svd/docs/current/RDT/nems/'

keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1'

# 'rdtld-rdtshf.rep-rdtsev.j.10-rdtfmt',

loaders = ['rdtld-rdtshf.rep.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf-rdtsev.j.10-rdtfmt']
label0 = ['{}_RS', '{}_S', '{}']   #, '{}_R'
sxticks = ['rep+str', 'rep', 'noshuff'] # 'str',
modelnames = [l + "_" + keywordstring + "_init-basic" for l in loaders]

batches = [269]
batstring = ['A1']
batches = [273]
batstring = ['PEG']
batches = [269, 273]
batstring = ['A1','PEG']

modelname = modelnames[-1]

#fitted_params_per_cell(cellids, batch, modelname, multi='mean', meta=['r_test', 'r_fit', 'se_test'])
mod_key = 'id'
multi = 'mean'
meta = ['r_test', 'r_fit', 'se_test']
stats_keys = ['mean', 'std', 'sem', 'max', 'min']

rg = {}
rgdf = {}

for batch, bs in zip(batches, batstring):
    modelspecs = get_batch_modelspecs(batch, modelname, multi=multi, limit=None)
    modelspecs_shf = get_batch_modelspecs(batch, modelnames[1], multi=multi, limit=None)
    modelspecs_SR = get_batch_modelspecs(batch, modelnames[0], multi=multi, limit=None)
    
    stats = ms.summary_stats(modelspecs, mod_key=mod_key,
                             meta_include=meta, stats_keys=stats_keys)
    index = list(stats.keys())
    columns = [m[0].get('meta').get('cellid') for m in modelspecs]

    midx = 0
    fields = ['bg_gain', 'fg_gain']
    b = np.array([])
    f = np.array([])
    tar_id = np.array([])
    cellids = []
    b_S = np.array([])
    f_S = np.array([])
    c = np.array([])
    cid = []
    r_test = np.array([])
    se_test = np.array([])
    r_test_S = np.array([])
    se_test_S = np.array([])
    r_test_SR = np.array([])

    for i, m in enumerate(modelspecs):
        cellid=m.meta['cellid']
        ishf=0
        iSR = 0
        while modelspecs_shf[ishf].meta['cellid'] != cellid:
            ishf += 1
        mshf = modelspecs_shf[ishf]
        while modelspecs_shf[iSR].meta['cellid'] != cellid:
            iSR += 1
        mSR = modelspecs_SR[iSR]

        r=m.meta['r_test'][0]
        se=m.meta['se_test'][0]
        if r > se*2:
            b = np.append(b, m.phi[midx]['bg_gain'][1:])
            f = np.append(f, m.phi[midx]['fg_gain'][1:])
            tar_id = np.append(tar_id, np.arange(1,len(m.phi[midx]['fg_gain'][1:])+1))
            for i in range(len(m.phi[midx]['fg_gain'][1:])):
                cellids.append(cellid)
            
            s = np.ones(m.phi[midx]['fg_gain'][1:].shape)

            c = np.append(c, s * i)
            cid.extend([m['meta']['cellid']]*len(s))
            r_test = np.append(r_test, s * r)
            se_test = np.append(se_test, s * se)

            # aggregate gain changes for shuffled model. mean zero?
            r_test_S = np.append(r_test_S, s * mshf.meta['r_test'][0])
            se_test_S = np.append(se_test_S, s * mshf.meta['se_test'][0])
            b_S = np.append(b_S, mshf.phi[midx]['bg_gain'][1:])
            f_S = np.append(f_S, mshf.phi[midx]['fg_gain'][1:])

            r_test_SR = np.append(r_test_SR, s * mSR.meta['r_test'][0])

    rdiff = r_test - r_test_S
    gdiff = f-b
    si = (rdiff > (se_test + se_test_S)) & (np.abs(gdiff)<1.2)
    nsi = (rdiff <= (se_test + se_test_S)) & (np.abs(gdiff)<1.2)

    def _rdt_info(i):
        print("{}: f={:.3} b={:.3}".format(cid[i],f[i],b[i]))
        cellid = cid[i]
        xfspec, ctx = nw.load_model_baphy_xform(cellid, batch=batch,
                                                modelname=modelname)
        ctx['modelspec'].quickplot(rec=ctx['val'])


    histbins = np.linspace(-0.5, 0.5, 21)

    fig = plt.figure(figsize=(8,5))

    ax=plt.subplot(2,3,1)
    bound = 1.2
    beta_comp(b, f, n1='bg', n2='fg', hist_range=[-bound, bound],
              ax=ax, click_fun=_rdt_info, highlight=si, title=bs)

    ax = plt.subplot(2,3,2)

    stat, p = wilcoxon(f,b)
    md = np.mean(f-b)
    rg[batch] = gdiff

    list_of_tuples = list(zip(cellids, tar_id, f, b, gdiff, r_test, r_test_S, r_test_SR)) 
    rgdf[batch] = pd.DataFrame(list_of_tuples, columns=['cellid','tar_id','fg','bg',
                                                        'gdiff','r','r_S','r_SR'])
    
    h0, x0 = np.histogram(gdiff[nsi], bins=histbins)
    h, x = np.histogram(gdiff[si], bins=histbins)
    d=(x0[1]-x0[0])/2
    plt.bar(x0[:-1]+d, h0, width=d*1.8)
    plt.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    plt.plot([0, 0], ylim, 'k--')
    tx = x0[0]+d
    ty = ylim[1]*0.95
    t = "n={}/{}\np={:.3e}\nmd={:.4f}".format(np.sum(si),si.shape[0], p, md)
    plt.text(tx,ty, t, va='top')
    plt.xlabel('FG-BG gain')

    ax = plt.subplot(2,3,3)
    ax.plot(rdiff[nsi], gdiff[nsi], '.', color='lightgray')
    ax.plot(rdiff[si], gdiff[si], '.', color='black')
    r,p = pearsonr(rdiff[nsi+si], gdiff[nsi+si])
    ax_remove_box(ax)
    ax.set_xlabel('deltaR')
    ax.set_ylabel('deltaG')
    ax.set_title('R={:.3f} p={:.3f}'.format(r,p))

    ax=plt.subplot(2,3,4)
    beta_comp(b_S, f_S, n1='bg_S', n2='fg_S', ax=ax, hist_range=[-bound, bound],
              highlight=si, title=bs+" (shf)")

    ax = plt.subplot(2,3,5)

    gdiff_S = f_S-b_S
    stat, p = wilcoxon(f_S,b_S)
    md = np.mean(f_S-b_S)
    h0, x0 = np.histogram(gdiff_S[~si], bins=histbins)
    h, x = np.histogram(gdiff_S[si], bins=histbins)
    d=(x0[1]-x0[0])/2
    plt.bar(x0[:-1]+d, h0, width=d*1.8)
    plt.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    plt.plot([0, 0], ylim, 'k--')
    tx = x0[0]+d
    ty = ylim[1]*0.95
    t = "n={}/{}\np={:.3e}\nmd={:.4f}".format(np.sum(si),si.shape[0], p, md)
    plt.text(tx,ty, t, va='top')
    plt.xlabel('FG-BG gain')

    fig.savefig(outpath+'gain_comp_'+keywordstring+'_'+bs+'.pdf')
    rgdf[batch].to_csv(outpath+'strf_rg_summary_'+bs+'.csv')
    
ttest_result = ttest_ind(rg[269], rg[273])

print("Mean A1={:.3f} PEG={:.3f} p<{:.4f}".format(
    np.mean(rg[269]), np.mean(rg[273]), ttest_result.pvalue))

