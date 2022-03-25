import nems.db as nd
import nems_lbhb.baphy as nb
from nems.recording import Recording
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

# script to inspect stability of units across trials for PTD data. Useful quick check on sorting quality.

animal = 'Amanita'
site = 'AMT020a'
cellids = ['AMT020a-60-1'] #None #['AMT020a-60-1']

def plot_collapsed_ref_tar(animal, site, cellids=None):
    site += "%"

    sql = "SELECT DISTINCT cellid, rawid, respfile FROM sCellFile WHERE cellid like %s AND runclassid=%s"
    d = nd.pd_query(sql, params=(site, 42,))

    mfile = []
    for f in np.unique(d.respfile):
        f_ = f.split('.')[0]
        mfile.append('/auto/data/daq/{0}/{1}/{2}'.format(animal, site[:-2], f_))

    if cellids is None:
        cellid = np.unique(d.cellid).tolist()
    else:
        cellid = cellids
    options = {"siteid": site[:-1], 'cellid': cellids,
                "mfilename": mfile, 'stim': False, 'pupil': True, 'rasterfs': 1000}

    uri = nb.baphy_load_recording_uri(**options)
    rec = Recording.load(uri)
    all_pupil = rec['pupil']._data
    ncols = len(mfile)
    if cellids is None:
        cellids = rec['resp'].chans

    for c in cellids:
        f, ax = plt.subplots(1, ncols, sharey=True, figsize=(12, 5))
        ref_base = 0
        for i, mf in enumerate(mfile):
            fn = mf.split('/')[-1]
            ep_mask = [ep for ep in rec.epochs.name if fn in ep]
            R = rec.copy()
            R['resp'] = R['resp'].rasterize(fs=20)
            R['resp'].fs = 20
            R = R.and_mask(ep_mask).apply_mask(reset_epochs=True)
            if '_a_' in fn:
                R = R.and_mask(['HIT_TRIAL']).apply_mask(reset_epochs=True)

            resp = R['resp'].extract_channels([c])

            tar_reps = resp.extract_epoch('TARGET').shape[0]
            tar_m = np.nanmean(resp.extract_epoch('TARGET'), 0).squeeze() * R['resp'].fs
            tar_sem = R['resp'].fs * np.nanstd(resp.extract_epoch('TARGET'), 0).squeeze() / np.sqrt(tar_reps)

            ref_reps = resp.extract_epoch('REFERENCE').shape[0]
            ref_m = np.nanmean(resp.extract_epoch('REFERENCE'), 0).squeeze() * R['resp'].fs
            ref_sem = R['resp'].fs * np.nanstd(resp.extract_epoch('REFERENCE'), 0).squeeze() / np.sqrt(ref_reps)

            # plot psth's
            time = np.linspace(0, len(tar_m) / R['resp'].fs, len(tar_m))
            ax[i].plot(time, tar_m, color='red', lw=2)
            ax[i].fill_between(time, tar_m + tar_sem, tar_m - tar_sem, color='coral')
            time = np.linspace(0, len(ref_m) / R['resp'].fs, len(ref_m))
            ax[i].plot(time, ref_m, color='blue', lw=2)
            ax[i].fill_between(time, ref_m + ref_sem, ref_m - ref_sem, color='lightblue')

            # set title
            ax[i].set_title(fn, fontsize=8)
            # set labels
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Spk / sec')

            # get raster plot baseline
            base = np.max(np.concatenate((tar_m + tar_sem, ref_m + ref_sem)))
            if base > ref_base:
                ref_base = base

        for i, mf in enumerate(mfile):
            # plot the rasters
            fn = mf.split('/')[-1]
            ep_mask = [ep for ep in rec.epochs.name if fn in ep]
            rast_rec = rec.and_mask(ep_mask).apply_mask(reset_epochs=True)
            if '_a_' in fn:
                rast_rec = rast_rec.and_mask(['HIT_TRIAL']).apply_mask(reset_epochs=True)
            rast = rast_rec['resp'].extract_channels([c])

            ref_times = np.where(rast.extract_epoch('REFERENCE').squeeze())
            base = ref_base
            ref_pupil = np.nanmean(rast_rec['pupil'].extract_epoch('REFERENCE'), -1)
            xoffset = rast.extract_epoch('TARGET').shape[-1] / rec['resp'].fs + 0.01
            ax[i].plot(ref_pupil / np.max(all_pupil) + xoffset,
                       np.linspace(base, int(base * 2), len(ref_pupil)), color='k')
            ax[i].axvline(xoffset + np.median(all_pupil / np.max(all_pupil)), linestyle='--', color='lightgray')
            #import pdb; pdb.set_trace()
            if ref_times[0].size != 0:
                max_rep = ref_pupil.shape[0] - 1
                ref_locs = ref_times[0] * (base / max_rep)
                ref_locs = ref_locs + base
                ax[i].plot(ref_times[1] / rec['resp'].fs, ref_locs, '|', color='blue', markersize=1)

            tar_times = np.where(rast.extract_epoch('TARGET').squeeze())
            tar_pupil = np.nanmean(rast_rec['pupil'].extract_epoch('TARGET'), -1)
            tar_base = np.max(ref_locs) + 1
            ax[i].plot(tar_pupil / np.max(all_pupil) + xoffset,
                       np.linspace(tar_base, tar_base + base, len(tar_pupil)), color='k')
            if tar_times[0].size != 0:
                max_rep = tar_pupil.shape[0] - 1
                tar_locs = tar_times[0] * (base / max_rep)
                tar_locs = tar_locs + tar_base
                ax[i].plot(tar_times[1] / rec['resp'].fs, tar_locs, '|', color='red', markersize=1)

            # set ylim
            #ax[i].set_ylim((0, top))

            # set plot aspect
            #asp = np.diff(ax[i].get_xlim())[0] / np.diff(ax[i].get_ylim())[0]
            #ax[i].set_aspect(asp / 2)

        f.suptitle(c, fontsize=8)
        f.tight_layout()


plot_collapsed_ref_tar(animal, site, cellids)
plt.show()
