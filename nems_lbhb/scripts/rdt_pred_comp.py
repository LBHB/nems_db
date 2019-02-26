import os
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from nems import xforms
import nems_lbhb.xform_wrappers as nw
from nems.gui.recording_browser import browse_recording, browse_context
import nems.db as nd
import nems.modelspec as ms
from nems_db.params import fitted_params_per_batch, fitted_params_per_cell, get_batch_modelspecs
import pandas as pd
import numpy as np
from nems_lbhb.stateplots import beta_comp

font_size=8
params = {'legend.fontsize': font_size-2,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

outpath='/auto/users/svd/docs/current/RDT/nems/'

keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1'

# removing the rep-shuffled model, since it's weird and unhelpful to plot
loaders = ['rdtld-rdtshf.rep.str-rdtsev-rdtfmt',
           'rdtld-rdtshf.str-rdtsev-rdtfmt',
           'rdtld-rdtshf-rdtsev-rdtfmt']
loaders = ['rdtld-rdtshf.rep.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf-rdtsev.j.10-rdtfmt']
# 'rdtld-rdtshf.rep-rdtsev.j.10-rdtfmt',

label0 = ['{}_RS', '{}_S', '{}']   # '{}_R',

sxticks = ['rep+str', 'rep', 'str', 'noshuff']
modelnames = [l + "_" + keywordstring + "_init-basic" for l in loaders]

batches = [269, 273]
batstring = ['A1','PEG']

fig = plt.figure(figsize=(10,5))
ax_mean = plt.subplot(1,3,1)
slegend = []

for b, batch in enumerate(batches):
    d=nd.batch_comp(batch=batch, modelnames=modelnames, stat='r_test')
    d.columns = [l0.format('r_test') for l0 in label0]
    dse=nd.batch_comp(batch=batch, modelnames=modelnames, stat='se_test')
    dse.columns = [l0.format('se_test') for l0 in label0]

    r = pd.concat([d,dse], sort=True, axis=1)

    r['sig'] = (((r['r_test']) > (2 * r['se_test'])) | ((r['r_test_RS']) > (2 * r['se_test_RS']))
                & np.isfinite(r['r_test']) & np.isfinite(r['r_test_RS']))
    r['sigdiffS'] = (((r['r_test'] - r['r_test_S']) > (r['se_test'] + r['se_test_S'])) &
                     r['sig'])
    r['sigdiffR'] = (((r['r_test_S'] - r['r_test_RS']) > (r['se_test_S'] + r['se_test_RS'])) &
                     r['sig'])
    r['sigdiffSR'] = (((r['r_test'] - r['r_test_RS']) > (r['se_test'] + r['se_test_RS'])) &
                      r['sig'])
    r['nsR'] = ~r['sigdiffR'] & r['sig']
    r['nsS'] = ~r['sigdiffS'] & r['sig']
    r['ns'] = ~r['sigdiffSR'] & r['sig']

    ax_mean.plot(r.loc[r['sig'],d.columns].mean().values, label=batstring[b])
    ax_mean.plot(r.loc[r['sig'],d.columns].median().values,ls='--')

    slegend.append('{} (n={}/{})'.format(batstring[b], r['sig'].sum(), len(r['sig'])))
    print(slegend[-1])
    print(r[r['sig']].mean())

    histbins = np.linspace(-0.1, 0.1, 21)

    ax = plt.subplot(2,3,2+3*b)
    h0, x0 = np.histogram(r.loc[r['nsS'],'r_test'] - r.loc[r['nsS'], 'r_test_S'],
                          bins=histbins)
    h, x = np.histogram(r.loc[r['sigdiffS'], 'r_test'] - r.loc[r['sigdiffS'], 'r_test_S'],
                        bins=histbins)
    d=(x0[1]-x0[0])/2
    plt.bar(x0[:-1]+d, h0, width=d*1.8)
    plt.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    plt.plot([0, 0], ylim, 'k--')
    tx = x0[0]+d
    ty = ylim[1]*0.95
    aa = r.loc[r['sig'],'r_test']
    bb = r.loc[r['sig'],'r_test_S']
    stat, p = wilcoxon(aa,bb)
    md = np.mean(aa-bb)
    t = "n={}/{}\np={:.3e}\nmd={:.4f}".format(np.sum(r['sigdiffS']),r.shape[0], p, md)

    plt.text(tx,ty, t, va='top')
    plt.xlabel('FG/BG stream improvement')
    plt.ylabel('{} units'.format(batstring[b]))
    if b == 0:
        plt.title('{}'.format(keywordstring))

    ax = plt.subplot(2,3,3+3*b)
    h0, x0 = np.histogram(r.loc[r['nsR'],'r_test_S'] - r.loc[r['nsR'], 'r_test_RS'],
                          bins=histbins)
    h, x = np.histogram(r.loc[r['sigdiffR'],'r_test_S'] - r.loc[r['sigdiffR'], 'r_test_RS'],
                        bins=histbins)
    d=(x0[1]-x0[0])/2
    plt.bar(x0[:-1]+d, h0, width=d*1.8)
    plt.bar(x0[:-1]+d, h, bottom=h0, width=d*1.8)
    ylim = ax.get_ylim()
    plt.plot([0, 0], ylim, 'k--')
    tx = x0[0]+d
    ty = ylim[1]*0.95
    aa = r.loc[r['sig'],'r_test_S']
    bb = r.loc[r['sig'],'r_test_RS']
    stat, p = wilcoxon(aa,bb)
    md = np.mean(aa-bb)
    t = "n={}/{}\np={:.3e}\nmd={:.4f}".format(np.sum(r['sigdiffR']),r.shape[0], p, md)
    plt.text(tx,ty, t, va='top')
    plt.xlabel('Rep/no-rep improvement')

ax_mean.legend()
ax_mean.set_xticks(np.arange(0,len(sxticks)))
ax_mean.set_xticklabels(sxticks)
ax_mean.set_ylabel('mean pred corr.')
#plt.suptitle('{}'.format(keywordstring))
plt.tight_layout()

#fig.savefig(outpath+'pred_comp_'+keywordstring+'.png')
fig.savefig(outpath+'pred_comp_'+keywordstring+'.pdf')
