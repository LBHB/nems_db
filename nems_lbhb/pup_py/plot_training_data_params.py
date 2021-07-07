"""
plot ellipse param distributions over all training frames to get a sense of how to 
normalize parameters
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import nems_db
nems_db_path = nems_db.__file__.split('/nems_db/__init__.py')[0]
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import pupil_settings as ps
import nems_lbhb.pup_py.utils as ut


training_files = os.listdir(ps.TRAIN_DATA_PATH)

params = np.zeros((len(training_files), 5))
for i, f in enumerate(training_files):
    fp = open(os.path.join(ps.TRAIN_DATA_PATH+f), "rb")
    d = pickle.load(fp)
    sf, im = ut.resize(d['frame'], size=(224, 224))
    par = d['ellipse_zack']
    params[i, :] = [par['X0_in'] * sf[0], par['Y0_in'] * sf[1], par['b'] * sf[1], par['a'] * sf[0], par['phi']]
PARMS = params

f, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.hist(params[:, 0] / 224, histtype='step', label='X0', bins=100)
ax.hist(params[:, 1] / 224, histtype='step', label='Y0', bins=100)
ax.hist(params[:, 2] / 112, histtype='step', label='b', bins=100)
ax.hist(params[:, 3] / 112, histtype='step', label='a', bins=100)
ax.hist(params[:, 4] / np.pi, histtype='step', label='phi', bins=100)

ax.legend(frameon=False)

f.tight_layout()

plt.show()
