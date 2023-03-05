import os

from nems_lbhb.stateplots import beta_comp
import matplotlib.pyplot as plt
from nems import xforms
import nems_lbhb.xform_wrappers as nw
from nems0.gui.recording_browser import browse_recording, browse_context
import nems0.db as nd
import nems0.modelspec as ms
import nems0.xform_helper as xhelp
from nems_db.params import fitted_params_per_batch, fitted_params_per_cell, get_batch_modelspecs
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_ind, pearsonr
from nems0.plots.utils import ax_remove_box
import seaborn as sns

outpath='/auto/users/svd/docs/current/RDT/nems/'
#outpath='/auto/users/bburan/'
#outpath='/tmp/'

keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1'


keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1'
loaders = ['rdtld-rdtshf.rep.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf-rdtsev.j.10-rdtfmt']

# new, rep only
loaders = ['rdtld-rdtshf.rep.str-rdtsev.j.10.ns.rep-rdtfmt',
           'rdtld-rdtshf.str-rdtsev.j.10.ns.rep-rdtfmt',
           'rdtld-rdtshf-rdtsev.j.10.ns.rep-rdtfmt']
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1'

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
            cid.extend([m.meta['cellid']]*len(s))
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


    figure, axes = plt.subplots(2, 3, figsize=(8, 5))

    ax=axes[0, 0]

    bound = 1.2
    histbins = np.linspace(-0.5, 0.5, 21)

    beta_comp(b, f, n1='bg', n2='fg', hist_range=[-bound, bound],
              ax=ax, click_fun=_rdt_info, highlight=si, title=bs)

    ax = axes[0, 1]

    stat, p = wilcoxon(f,b)
    md = np.mean(f-b)
    rg[batch] = gdiff

    list_of_tuples = list(zip(cellids, tar_id, f, b, gdiff, r_test, r_test_S, r_test_SR))
    rgdf[batch] = pd.DataFrame(list_of_tuples, columns=['cellid','tar_id','fg','bg',
                                                        'gdiff','r','r_S','r_SR'])
    rgdf[batch]['trained'] = rgdf[batch]['cellid'].str.contains('oys')

    h0, x0 = np.histogram(gdiff[nsi], bins=histbins)
    h, x = np.histogram(gdiff[si], bins=histbins)
    d=(x0[1]-x0[0])/2
    ax.bar(x0[:-1]+d, h0, width=d*1.8)
    ax.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    ax.plot([0, 0], ylim, 'k--')
    tx = x0[0]+d
    ty = ylim[1]*0.95
    t = "n={}/{}\np={:.3e}\nmd={:.4f}".format(np.sum(si),si.shape[0], p, md)
    ax.text(tx,ty, t, va='top')
    ax.set_xlabel('FG-BG gain')

    ax = axes[0, 2]

    ax.plot(rdiff[nsi], gdiff[nsi], 'o', color='gray', mec='w', mew=1)
    ax.plot(rdiff[si], gdiff[si], 'o', color='#83428c', mec='w', mew=1)
    r,p = pearsonr(rdiff[nsi+si], gdiff[nsi+si])

    x=np.polyfit(rdiff,gdiff,1)
    x0 = np.array(ax.get_xlim())
    y0 = x0*x[0]+x[1]

    ax.plot(x0,y0,'k--')
    ax_remove_box(ax)
    ax.set_xlabel('deltaR')
    ax.set_ylabel('deltaG')
    ax.set_title('R={:.3f} p={:.4e}'.format(r, p))

    ax = axes[1, 0]
    beta_comp(b_S, f_S, n1='bg_S', n2='fg_S', ax=ax, hist_range=[-bound, bound],
              highlight=si, title=bs+" (shf)")

    ax = axes[1, 1]

    gdiff_S = f_S-b_S
    stat, p = wilcoxon(f_S,b_S)
    md = np.mean(f_S-b_S)
    h0, x0 = np.histogram(gdiff_S[~si], bins=histbins)
    h, x = np.histogram(gdiff_S[si], bins=histbins)
    d=(x0[1]-x0[0])/2
    ax.bar(x0[:-1]+d, h0, width=d*1.8)
    ax.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    ax.plot([0, 0], ylim, 'k--')
    tx = x0[0]+d
    ty = ylim[1]*0.95
    t = "n={}/{}\np={:.3e}\nmd={:.4f}".format(np.sum(si),si.shape[0], p, md)
    ax.text(tx,ty, t, va='top')
    ax.set_xlabel('FG-BG gain')

    ax = axes[1, 2]
    mn = rgdf[batch].groupby('trained').mean()['gdiff'].values
    se = rgdf[batch].groupby('trained').std()['gdiff'].values / np.sqrt(rgdf[batch].groupby('trained').count()['gdiff'].values)

    ax.bar(x=np.array([0, 1]), height=mn, yerr=se)
    ax.set_ylabel('mean+sem gdiff')
    ax.set_xticklabels(['naive','trained'])
    ax.set_title('{:.3}+{:.3} / {:.3}+{:.3}'.format(mn[0],se[0],mn[1],se[1]))
    sns.despine(figure, offset=10)

    figure.tight_layout()
    figure.savefig(outpath+'gain_comp_'+keywordstring+'_'+bs+'.pdf')
    rgdf[batch].to_csv(outpath+'strf_rg_summary_'+bs+'.csv')

    print("Mean rep gain batch {}: {:.3f}".format(batch,np.mean(np.concatenate((f_S,b_S)))))
    print("Std rep gain batch {}: {:.3f}".format(batch, np.std(np.concatenate((f_S, b_S)))/np.sqrt(len(f_S)*2)))

ttest_result = ttest_ind(rg[269], rg[273])


print("Mean A1={:.3f}+{:.4f} PEG={:.3f}+{:.4f} p<{:.4f}".format(
    np.mean(rg[269]), np.std(rg[269])/np.sqrt(len(rg[269])),
    np.mean(rg[273]), np.std(rg[273])/np.sqrt(len(rg[273])),ttest_result.pvalue))

