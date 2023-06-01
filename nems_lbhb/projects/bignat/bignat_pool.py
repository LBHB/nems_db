import matplotlib.pyplot as plt

from nems_lbhb import xform_wrappers, baphy_io, baphy_experiment
from nems0 import db, xforms, epoch

batch=343
siteids,cellids = db.get_batch_sites(343)

for siteid in siteids[1:3]:
    print(f"site: {siteid}")

    ex = baphy_experiment.BAPHYExperiment(batch=batch, cellid=siteid)
    rec = ex.get_recording(loadkey="gtgram.fs100.ch18")

    epoch_regex='^STIM_'
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)

    est['resp'].shape

    epochs = epoch.epoch_names_matching(rec['resp'].epochs, epoch_regex)

    resp=val.apply_mask()['resp'].extract_epoch(epochs[0])

    f,ax=plt.subplots(2,1)
    ax[0].imshow(resp.mean(axis=0), aspect='auto', origin='lower', interpolation='none')



