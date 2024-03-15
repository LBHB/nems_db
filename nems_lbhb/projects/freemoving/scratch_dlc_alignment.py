#stardard imports
import os
import numpy as np
import logging

import matplotlib.pyplot as plt
from nems0 import db
from nems_lbhb.baphy_experiment import load_training, BAPHYExperiment

log = logging.getLogger(__name__)

parmfile='/auto/data/daq/LemonDisco/training2024/LemonDisco_2024_03_06_NFB_1'
parmfile='/auto/data/daq/LemonDisco/LMD045/LMD045a03_a_NFB'

parmfile='/auto/data/daq/Prince/PRN051/PRN051a04_a_NTD'
parmfile='/auto/data/daq/Prince/PRN051/PRN051a05_a_NTD'
parmfile='/auto/data/daq/Prince/PRN051/PRN051a06_a_NTD'
parmfile='/auto/data/daq/Prince/PRN051/PRN051a07_a_NTD'
parmfile='/auto/data/daq/Prince/PRN051/PRN051a08_a_NTD'
parmfile='/auto/data/daq/Prince/PRN051/PRN051a13_a_NTD'
parmfile='/auto/data/daq/Prince/PRN051/PRN051a15_a_NTD'

parmfile=['/auto/data/daq/Prince/PRN048/PRN048a01_a_NTD',
          '/auto/data/daq/Prince/PRN048/PRN048a02_a_NTD']
siteid="PRN048a"

parmfile=['/auto/data/daq/Prince/PRN050/PRN050a02_a_NTD',
          '/auto/data/daq/Prince/PRN050/PRN050a07_a_NTD']
siteid="PRN050a"

batchid=349
batchid=348
siteids, cellids = db.get_batch_sites(batchid)

#rec = load_training(parmfile, dlc=True)

for siteid in ["PRN075a"]: #siteids[16:17]:
    parmfile = siteid
    ex = BAPHYExperiment(batch=batchid, cellid=siteid)
    rec = ex.get_recording(resp=True, stim=False, dlc=True)

    dlc = rec['dlc']
    epochs = dlc.epochs

    epochs.loc[epochs.name.str.startswith("LICK")]

    labels = ['TRIAL','EARLY','INCORR','CORRECT']
    vos=-0.5  # video offset?
    preos=0.05
    posos=0.05
    trial_starts = dlc.get_epoch_bounds('TRIAL')
    peri_trials = np.stack([trial_starts[:,0]-preos-vos, trial_starts[:,0]+posos-vos], axis=1)
    lick0 = dlc.get_epoch_bounds('LICK , 0')
    peri_lick0 = np.stack([lick0[:,0]-preos-vos, lick0[:,0]+posos-vos], axis=1)
    #lick1 = dlc.get_epoch_bounds('LICK , 1')
    lick1 = dlc.get_epoch_bounds('LICK , FA')
    peri_lick1 = np.stack([lick1[:,0]-preos-vos, lick1[:,0]+posos-vos], axis=1)
    #lick2 = dlc.get_epoch_bounds('LICK , 2')
    lick2 = dlc.get_epoch_bounds('LICK , HIT')
    peri_lick2 = np.stack([lick2[:,0]-preos-vos, lick2[:,0]+posos-vos], axis=1)

    data = [peri_trials, peri_lick0, peri_lick1, peri_lick2]
    colors=['b','r','y','g']

    f, ax = plt.subplots(figsize=(4,3))

    for d,c,l in zip(data,colors,labels):
        if np.any(d):
            r = dlc.extract_epoch(d)

            x=r[:,0,:].T
            y=r[:,1,:].T

            ax.plot(x[:,0],y[:,0], color=c, lw=0.5, label=l)
            ax.plot(x[:,1:],y[:,1:], color=c, lw=0.5)
            ax.plot(x[0,:],y[0,:],'.', color=c)

    ax.invert_yaxis()
    ax.legend(fontsize=8)
    if type(parmfile) is list:
        ax.set_title(os.path.basename(parmfile[0]),fontsize=8)
    else:
        ax.set_title(os.path.basename(parmfile),fontsize=8)

