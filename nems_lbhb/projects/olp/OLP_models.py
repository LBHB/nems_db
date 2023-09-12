from nems0 import db
from nems_lbhb.projects.olp import olp_cnn_pred

batch=341
modelnames = [
    "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
    "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
]

siteids, cellids = db.get_batch_sites(batch=batch)

# pick a site, any site:
cid=35
siteid = siteids[cid]
print(f"{siteid} selected")

# load models, predict OLP resposnes
cell_epoch_df, rec1, rec2 = olp_cnn_pred.compare_olp_preds(siteid, batch=batch, modelnames=modelnames, verbose=False)

fits = []
for siteid in siteids:

for cc in np.arange(44,48,1):
    siteid = siteids[cc]
    cell_epoch_df, rec1, rec2 = olp_cnn_pred.compare_olp_preds(siteid, batch=batch, modelnames=modelnames,
                                                               verbose=False)
    d = cell_epoch_df.loc[cell_epoch_df['area'].isin(['A1','PEG'])]
    d['siteid'] = siteid
    print(f'Adding site {siteid}.')
    fits.append(d)

dff = pd.concat(fits)

import joblib as jl
import os

OLP_partialweights_db_path = f'/auto/users/hamersky/OLP_models/today_model'  # weight + corr
os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)

jl.dump(dff, OLP_partialweights_db_path)

path = '/auto/users/hamersky/OLP_models/today_model'
ddd = jl.load(path)


#compare predicted vs. actual, excluding non-AC units
d = cell_epoch_df.loc[cell_epoch_df['area'].isin(['A1','PEG'])]
f=olp_cnn_pred.plot_olp_preds(d, minresp=0.01, mingain=0.01, maxgain=2.0)




















batch=345
modelnames = [
    "gtgram.fs100.ch18.bin6-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4",
    "gtgram.fs100.ch18.bin6-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
]

siteids, cellids = db.get_batch_sites(batch=batch)

# pick a site, any site:
cid=5
siteid, cellid = siteids[cid], cellids[cid]
print(f"{siteid} selected")

# load models, predict OLP resposnes
cell_epoch_df, rec1, rec2 = olp_cnn_pred.compare_olp_preds(siteid, batch=batch, modelnames=modelnames, verbose=False)

#compare predicted vs. actual, excluding non-AC units
d = cell_epoch_df.loc[cell_epoch_df['area'].isin(['A1','PEG'])]
f=olp_cnn_pred.plot_olp_preds(d, minresp=0.01, mingain=0.01, maxgain=2.0)


