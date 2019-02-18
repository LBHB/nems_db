import os

from nems_lbhb.stateplots import beta_comp
import matplotlib.pyplot as plt
from nems import xforms
import nems_lbhb.xform_wrappers as nw
from nems.gui.recording_browser import browse_recording, browse_context
import nems.db as nd
import nems.modelspec as ms
from nems_db.params import fitted_params_per_batch, fitted_params_per_cell, get_batch_modelspecs
import pandas as pd
import numpy as np

outpath='/auto/users/svd/docs/current/RDT/nems/'

keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1'
keywordstring = 'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'

# 'rdtld-rdtshf.rep-rdtsev.j.10-rdtfmt',

loaders = ['rdtld-rdtshf.rep.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf.str-rdtsev.j.10-rdtfmt',
           'rdtld-rdtshf-rdtsev.j.10-rdtfmt']
label0 = ['{}_RS', '{}_S', '{}']   #, '{}_R'
sxticks = ['rep+str', 'rep', 'noshuff'] # 'str',
modelnames = [l + "_" + keywordstring + "_init-basic" for l in loaders]

batches = [269, 273]
batstring = ['A1','PEG']
batches = [273]
batstring = ['PEG']
batches = [269]
batstring = ['A1']

modelname = modelnames[-1]

#fitted_params_per_cell(cellids, batch, modelname, multi='mean', meta=['r_test', 'r_fit', 'se_test'])
mod_key = 'id'
multi = 'mean'
meta = ['r_test', 'r_fit', 'se_test']
stats_keys = ['mean', 'std', 'sem', 'max', 'min']

for batch, bs in zip(batches, batstring):
    modelspecs = get_batch_modelspecs(batch, modelname, multi=multi, limit=None)
    modelspecs_shf = get_batch_modelspecs(batch, modelnames[2], multi=multi, limit=None)
    stats = ms.summary_stats(modelspecs, mod_key=mod_key,
                             meta_include=meta, stats_keys=stats_keys)
    index = list(stats.keys())
    columns = [m[0].get('meta').get('cellid') for m in modelspecs]

    midx = 0
    fields = ['bg_gain', 'fg_gain']
    b = np.array([])
    f = np.array([])
    c = np.array([])
    cid = []
    r_test = np.array([])
    se_test = np.array([])
    r_test_S = np.array([])
    se_test_S = np.array([])
    for i, m in enumerate(modelspecs):
        r=m.meta['r_test'][0]
        se=m.meta['se_test'][0]
        if r > se*2:
            b = np.append(b, m.phi[midx]['bg_gain'][1:])
            f = np.append(f, m.phi[midx]['fg_gain'][1:])
            s = np.ones(m.phi[midx]['fg_gain'][1:].shape)
            c = np.append(c, s * i)
            cid.extend([m['meta']['cellid']]*len(s))
            r_test = np.append(r_test, s * r)
            se_test = np.append(se_test, s * se)
            r_test_S = np.append(r_test_S, s * modelspecs_shf[i].meta['r_test'][0])
            se_test_S = np.append(se_test_S, s * modelspecs_shf[i].meta['se_test'][0])

    si = (r_test-r_test_S) > (se_test + se_test_S)

    def _rdt_info(i):
        print("{}: f={:.3} b={:.3}".format(cid[i],f[i],b[i]))
        cellid = cid[i]
        xfspec, ctx = nw.load_model_baphy_xform(cellid, batch=batch,
                                                modelname=modelname)
        ctx['modelspec'].quickplot(rec=ctx['val'])

    #plt.figure()
    #ax=plt.subplot(1,1,1)
    fig = beta_comp(b, f, n1='bg', n2='fg', hist_range=[-0.75, 0.75], click_fun=_rdt_info,
                    highlight=si, title=bs)

    fig.savefig(outpath+'gain_comp_'+keywordstring+'_'+bs+'.png')

