import logging

import numpy as np
import matplotlib.pyplot as plt

from nems0 import db
from nems0.xform_helper import load_model_xform, fit_model_xform

from nems.models import LN

log = logging.getLogger(__name__)

# testing binaural NAT with various model architectures.
batch=338
siteids,cellids0=db.get_batch_sites(batch)

rank=20
#modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"
modelname=f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}.l2-fir.20x1x{rank}-wc.{rank}xR.l2-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"

# nb: code to fit a model and save results so they can be loaded by load_model_xform:
# outpath = fit_model_xform(cellid, batch, modelname, saveInDB=True)


for cellid in cellids0[:10]:
    try:
        xf,ctx = load_model_xform(cellid, batch, modelname, eval_model=False)

        r = ctx['modelspec'].meta['r_test'][:,0]
        cellids = ctx['modelspec'].meta['cellids']
        labels = [f"{c[8:]} {rr:.3f}" for c,rr in zip(cellids, r)]

        #f1=LN.LN_plot_strf(ctx['modelspec'],layer=1, plot_nl=True)
        f2=LN.LN_plot_strf(ctx['modelspec'],labels=labels)

        s = LN.LN_get_strf(ctx['modelspec'])

        flip_cells = [i for i,c in enumerate(cellids) if '-B-' in c]

        for f in flip_cells:
            a=s[:18,:,f]
            b=s[18:,:,f]
            s[:,:,f] = np.concatenate((b,a), axis=0)

        hcontra = s[:18,:,:]
        hipsi = s[18:,:,:]

        sim = np.sign((hcontra*hipsi).sum(axis=(0,1)))

        magc = hcontra.std(axis=(0,1))
        magi = hipsi.std(axis=(0,1))

        hsum = (hcontra+hipsi).std(axis=(0,1)) #/ s.std(axis=(0,1))
        hdiff = (hcontra-hipsi).std(axis=(0,1)) #/ s.std(axis=(0,1))

        f,ax=plt.subplots(1,2)
        ax[0].scatter(hsum,hdiff)
        mm = np.max(np.concatenate((hsum,hdiff)))

        ax[0].plot([0,mm],[0,mm],'--')
        ax[0].set_xlabel('std(sum)')
        ax[0].set_ylabel('std(diff)')

        ax[1].scatter(r,magi/magc)
        mm = np.max(r)
        ax[1].plot([0,mm],[0,0],'--')
        ax[1].plot([0,mm],[1,1],'--')
        ax[1].set_xlabel('r_test')
        ax[1].set_ylabel('signed std(ipsi)/std(contra)')

        f.suptitle(ctx['modelspec'].name[:35])
    except:
        print(f"model fit not found for {cellid}")



modelnames=[
    f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"
    for rank in [5,8,12,16, 20]
    ] + \
    [f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4.t4"
       for rank in [20] ] + \
    [f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}.l2-fir.20x1x{rank}-wc.{rank}xR.l2-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"
    for rank in [20] ]
d = db.batch_comp(batch, modelnames, shortnames=['R5','R8','R12','R16','R20','R20t4','R20l2'])
df = db.batch_comp(batch, modelnames, stat='r_floor', shortnames=['R5','R8','R12','R16','R20','R20t4','R20l2'])

good = np.isfinite(d['R20l2']) & ((d>df).sum(axis=1)>=4)

d['cellid']=d.index
d['siteid']=d['cellid'].apply(db.get_siteid)

print(d.loc[good].median())

import seaborn as sns

f,ax=plt.subplots()
sns.scatterplot(d.loc[good], x='R20', y='R20l2', ax=ax)

