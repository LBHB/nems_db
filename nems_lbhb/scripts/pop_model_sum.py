import nems.db as nd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import re
import scipy.signal as signal

from nems_lbhb.xform_wrappers import baphy_load_wrapper
from nems.recording import load_recording
from nems.metrics.stp import stp_magnitude
from nems.xform_helper import load_model_xform
import nems.plots.api as nplt
import nems.gui.editors as gui

from nems_db.params import fitted_params_per_batch

from nems.xform_helper import load_model_xform
import nems.xforms as xforms
from nems.gui.recording_browser import browse_context

import matplotlib.pyplot as plt
import numpy as np

batch=289
reference_model="ozgf.fs50.ch18.pop-loadpop.cc20.rnd-norm-pca.no-popev_dlog-wc.18x5.g-fir.1x10x5-relu.5-wc.5xR.z-lvl.R_tf.s30"
d_ref = nd.batch_comp(modelnames=[reference_model], batch=batch, stat='r_test')
cellids = list(d_ref.index)

modelmask = ['%tf.n.s30', 'ozgf.fs50%init-%']
#modelmask = ['%popev%']
#modelmask = ['%loadpop%relu%']
modelquery = ("SELECT modelname, count(*) as n, max(lastmod) as " +
              "last_mod FROM NarfResults WHERE batch=%s" +
              " AND cellid in ('"+"','".join(cellids)+"')" +
              " AND (")
for i, m in enumerate(modelmask):
    modelquery += 'modelname like %s OR '
modelquery = modelquery[:-3]  # drop the trailing OR
modelquery += ') GROUP BY modelname ORDER BY modelname'
d_models = nd.pd_query(modelquery, (batch, *modelmask))

cell_count = d_models['n'].max()

modelnames = list(d_models.loc[d_models['n']==cell_count, 'modelname'])

d = nd.batch_comp(modelnames=modelnames, batch=batch, stat='r_test', cellids=cellids)
cellid = cellids[0]  # d.index[0]
dn = nd.batch_comp(modelnames=modelnames, batch=batch, stat='n_parms', cellids=[cellid])

engine = nd.Engine()
conn = engine.connect()

for m in dn.columns[:]:
    if dn[m][0] == 0:
        xfspec, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=m, eval_model=False)
        phi_vector = ctx['modelspec'].phi_vector
        print('nparms for {} = {}'.format(m, len(phi_vector)))

        sql="UPDATE Results SET n_parms={} WHERE batch={} AND modelname='{}'".format(len(phi_vector), batch, m)
        conn.execute(sql)

dn.rename(index={cellid: 'n_parms'}, inplace=True)
d = d.append(dn)

groups = []
#gap = "dlog-wc[\\d\\.a-z]+-fir[\\d\\.a-z]-relu[\\d\\.a-z]-wc[\\d\\.a-z]-"
gap1 = '[a-z-]+wc[\d\.a-z]+-fir[\d\.a-z]+-relu[\d\.a-z]+-wc[\d\.a-z]+-'
gap2 = '[a-z-]+wc[\d\.a-z]+-fir[\d\.a-z]+-lvl[\d\.a-z]+-wc[\d\.a-z]+-'
gap3 = '[a-z-]+wc[\d\.a-z]+-fir[\d\.a-z]+-relu[\d\.a-z]+-fir[\d\.a-z]+-relu[\d\.a-z]+-wc[\d\.a-z]+-'
for m in modelnames:
    big_k = m.split("_")
    k=big_k[1].split('-')
    rmatch = [i for i,s in enumerate(k) if (s.startswith('relu') or s.startswith('lvl'))]
    if ("loadpop" not in big_k[0]) or (len(rmatch) == 0):
        groups.append(m)
    elif (".r2-" in big_k[1]):
        big_k[1] = gap3 + "-".join(k[(rmatch[0]+4):])
        groups.append("_".join(big_k) + "$")
    elif k[rmatch[0]].startswith('relu'):
        big_k[1] = gap1 + "-".join(k[(rmatch[0]+2):])
        groups.append("_".join(big_k) + "$")
    elif k[rmatch[0]].startswith('lvl'):
        big_k[1] = gap2 + "-".join(k[(rmatch[0]+2):])
        groups.append("_".join(big_k) + "$")

groups = list(set(groups))

#nplt.pareto(d, groups=groups)

f = plt.figure(figsize=(12,8))
ax = plt.subplot(1,1,1)

n_parms = d.loc['n_parms']

mean_score = d.drop(index='n_parms').mean()

for g in groups:
    r = re.compile(g)
    m = [i for i in mean_score.index if r.match(i)]
    xc = np.array([mean_score.loc[i] for i in m])
    n = np.array([n_parms[i] for i in m])
    xc = xc[np.argsort(n)]
    n = np.sort(n)

    l = ax.plot(n, xc, ls='-', marker=".")
    c = l[0].get_color()
    gs = g.split("_")
    if "loadpop" not in gs[0]:
        gs[0] = "sng"
    elif "rnd" in gs[0]:
        gs[0] = "rnd"
    else:
        gs[0] = "pop"

    gs[1]=gs[1].replace(gap1, 'convN-')
    gs[1]=gs[1].replace(gap2, 'convL-')
    gs[1]=gs[1].replace(gap3, 'convNx2-')
    gs[1]=gs[1].replace(".z", '')
    gs[2]=gs[2].replace("$", '')

    glabel = "_".join(gs)
    
    if len(n) > 0:
        ax.text(n[-1], xc[-1], glabel, color=c, fontsize=7)
        
    print("{} : {}".format(g, len(n)))

