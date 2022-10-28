import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nems import db
from nems0.recording import load_recording
import nems0.preprocessing as preproc
from nems0.plots.api import raster
from nems0.utils import smooth

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment

from os.path import basename, join

batch = 242
loadkey = "env.fs100"
d_cells = db.get_batch_cells(batch=batch)
d_cells['siteid']=d_cells['cellid'].apply(db.get_siteid)
#d_cells.tail(5)

cellid='por042a-16-1'
cellid='por044a-10-1'

r1, r2 = 16, 21  # por026c  - nice contra-only and contra+ipsi
#r1, r2 = 26, 34  # por028d  - kind of messy
#r1, r2 = 40, 48  # por044a  - not very stable
#r1, r2 = 48, 59  # por049a  - decent diversity
#r1, r2 = 63, 65  # por068b -  unstable, contra simple
#r1, r2 = 65, 69  # por069a - one good cell, totally ipsi, another weak contra
#r1, r2 = 69, 72  # por069b  a bit unstable but some ipsi
#r1, r2 = 72, 75  # por070a  -  unstable, contra simple
#r1, r2 = 75,79  # por070c  - cool sparse, but not interesting.
#r1, r2 = 79,84  # por071b -- crummy resp
#r1, r2 = 87,91  # por072a  - some ipsi suppression? **
#r1, r2 = 92,94  # por073a  - unstable
#r1, r2 = 94,96  # por073b - maybe some ipsi??  **
#r1, r2 = 96,101  # por074a - unstable
#r1, r2 = 101,106  # por074b  - not super stable
#r1, r2 = 106,111  # por075b  -- a couple nice responses.
#r1, r2 = 111,117  # por075c  -- not a lot of trials
#r1, r2 = 117,121  # por076a  -- all contra
#r1, r2 = 121,123  # por076b  - maybe a nice ipsi suppression?
#r1, r2 = 123,127  # por077a  - nice resps. heavily chan 1
#r1, r2 = 127,130  # por078a  -- maybe nice?
#r1, r2 = 130,136  # por079a  # all chan1
#r1, r2 = 136,137 # por085b - crummy resps
#r1, r2 = 140,142 # por087a - crummy resps
#r1, r2 = 142,144  # por087b  - one ok


for i in range(r1,r2):
    cellid = d_cells.loc[i, 'cellid']

    uri = generate_recording_uri(cellid=cellid, batch=str(batch), loadkey=loadkey)

    rec = load_recording(uri)
    epoch_regex="STIM_"
    keepfrac=1

    rec = preproc.normalize_epoch_lengths(rec, resp_sig='resp', epoch_regex=epoch_regex)

    est, val = rec.split_using_epoch_occurrence_counts(
        epoch_regex=epoch_regex, keepfrac=keepfrac, verbose=True)

    stim = val['stim'].extract_epochs(epoch_names=epoch_regex, mask=val['mask'])
    resp_sig = val['resp'].extract_channels([cellid])
    resp = resp_sig.extract_epochs(epoch_names=epoch_regex, mask=val['mask'])

    #e=est['resp'].extract_epochs(epoch_names=epoch_regex, mask=est['mask'])

    for epoch in list(resp.keys()):
        bincount= resp[epoch].shape[2]
        times = np.arange(bincount)/val['resp'].fs

        f, ax = plt.subplots(3, 1, sharex=True, figsize=(3, 3))

        ax[0].plot(times, stim[epoch][0,:,:].T)
        ax[0].legend(['chan 1','chan 2'])
        ax[0].set_title(f"{i}: {cellid} - {epoch}", fontsize=8)

        raster(times, resp[epoch][:,0,:], xlabel='Time', ylabel='Trial', ax=ax[1])
        ax[2].plot(times, smooth(resp[epoch][:,0,:].mean(axis=0)))

    epoch1 = list(resp.keys())[0]
    epoch2 = list(resp.keys())[1]
    s = np.concatenate((stim[epoch1][0,:,:].T,stim[epoch2][0,:,:].T), axis=0)
    r_raster = np.concatenate((resp[epoch1][:,0,:].T,
                               resp[epoch2][:,0,:].T), axis=0)
    r=r_raster.mean(axis=1,keepdims=True)

    np.savetxt(f"{cellid}_stim.csv", s, delimiter=",")
    np.savetxt(f"{cellid}_raster.csv", r_raster, delimiter=",")
    np.savetxt(f"{cellid}_psth.csv", r, delimiter=",")


#est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
#val = preproc.average_away_epoch_occurrences(val, epoch_regex=epoch_regex)
#
#est['resp'].epochs.loc[est['resp'].epochs.name.str.startswith("STIM_")]

