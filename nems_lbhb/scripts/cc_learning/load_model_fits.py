
import os
import glob
import numpy as np

from nems0.xforms import load_analysis

"""
examples illustrating how to load CC models exported from the lab without
a database connect.

Note, models exported using this command:
from nems0.db import export_fits
export_fits(315, 
        ['psth.fs20.pup-ld-st.pup.fil-ref-psthfr.s_stategain.S_jk.nf20-basic',
         'psth.fs20.pup-ld-st.pup.fil0-ref-psthfr.s_stategain.S_jk.nf20-basic',
         'psth.fs20.pup-ld-st.pup0.fil-ref-psthfr.s_stategain.S_jk.nf20-basic',
         'psth.fs20.pup-ld-st.pup0.fil0-ref-psthfr.s_stategain.S_jk.nf20-basic'],
        dest='/auto/users/svd/projects/reward_training/nems_export/torcs/')

"""

# datapath should indicate where the NEMS models have been saved
datapath = '/auto/users/svd/projects/reward_training/nems_export/torcs/'

# find all models, knowing that the folder names should contain "2020"
d = glob.glob(datapath + '*2020*')

# get unique list of cells and modelnames
cellids = list(set([f.split("_")[0] for f in d]))
modelnames = list(set([f.split("_")[1].split('.2020')[0] for f in d]))

# load an example model
cellid = 'NMK020c-29-1'
modelinfo = 'st.pup.fil-'

i, = np.where([(cellid in f) and (modelinfo in f) for f in d])
i = i[0]   # is an array, just take first entry (should just be 1)

xf, ctx = load_analysis(d[i], eval_model=False)

modelspec=ctx['modelspec']
print('Loaded model {}'.format(os.path.basename(d[i])))
print('Cellid: {}\nModel name: {}'.format(modelspec.meta['cellid'],
                                          modelspec.meta['modelname']))
print('r_test: {:.3f}'.format(modelspec.meta['r_test'][0][0]))

state_channels=modelspec.meta['state_chans']
for i, s in enumerate(state_channels):
    # find name of current file
    print("{}: offset={:.3f} gain={:.3f} MI={:.3f}".format(
        s, modelspec.phi[0]['d'][0, i], modelspec.phi[0]['g'][0, i],
        modelspec.meta['state_mod'][i]))

