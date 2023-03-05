from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import nems
import nems0.db as nd
import nems_lbhb.xform_wrappers as xwrap
import nems0.epoch as ep


batch = 323
plot_stat = 'r_test'
modelname = "ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x8R.g-fir.8x20xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4",

a1_results = nd.batch_comp(batch=batch, modelnames=[modelname], stat=plot_stat)
r = a1_results[modelname]
all_r_cells = r.index.values

def snr_by_batch(batch, loadkey, save_path=None, load_path=None, frac_total=True):

    cellids = nd.get_batch_cells(batch, as_list=True)
    siteids = list(set([c.split('-')[0] for c in cellids]))

    snrs = []
    cells = []
    if load_path is None:
        for site in siteids:
            rec_path = xwrap.generate_recording_uri(site, batch, loadkey=loadkey)
            rec = nems0.recording.load_recording(rec_path)
            est, val = rec.split_using_epoch_occurrence_counts('^STIM_')
            for cellid in rec['resp'].chans:
                if cellid in cellids:
                    resp = val.apply_mask()['resp'].extract_channels([cellid])
                    snr = compute_snr(resp, frac_total=frac_total)
                    snrs.append(snr)
                    cells.append(cellid)

        results = {'cellid': cells, 'snr': snrs}
        df = pd.DataFrame.from_dict(results)
        df.dropna(inplace=True)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)

    else:
        df = pd.read_pickle(load_path)

    return df

def compute_snr(resp, frac_total=True):
    epochs = resp.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    resp_dict = resp.extract_epochs(stim_epochs)

    per_stim_snrs = []
    for stim, resp in resp_dict.items():
        resp = resp.squeeze()
        products = np.dot(resp, resp.T)
        per_rep_snrs = []
        for i, _ in enumerate(resp):
            total_power = products[i,i]
            signal_powers = np.delete(products[i], i)
            if frac_total:
                rep_snr = np.nanmean(signal_powers)/total_power
            else:
                rep_snr = np.nanmean(signal_powers/(total_power-signal_powers))

            per_rep_snrs.append(rep_snr)
        per_stim_snrs.append(np.nanmean(per_rep_snrs))

    return np.nanmean(per_stim_snrs)

# load snr for all cells in batch

for batch in [322,323]:
    snr_path = Path('intermediate_results/') / str(batch) / 'snr.pkl'
    snr_df = snr_by_batch(batch, 'ozgf.fs100.ch18', save_path=snr_path)
    snrs = snr_df['snr']


def partition_index_by_percentiles(series, q1, q2):
    first_cutoff = np.percentile(series, q1)
    second_cutoff = np.percentile(series, q2)
    all = series.index.values
    low = series[series < first_cutoff].index.values
    high = series[series > second_cutoff].index.values
    middle = np.setdiff1d(all, np.concatenate([low, high]))

    return low, middle, high


# Identify high, middle, low snr cells
low_snr_cells, middle_snr_cells, high_snr_cells = partition_index_by_percentiles(snrs, 25, 75)

# Identify high, middle, low r cells
low_r_cells, middle_r_cells, high_r_cells = partition_index_by_percentiles(r, 25, 75)

high_snr_low_r_test = high_snr_cells[np.in1d(high_snr_cells, low_r_cells)]
high_snr_high_r_test = high_snr_cells[np.in1d(high_snr_cells, high_r_cells)]

# pick some random cells each from first quartile, middle half, fourth quartile
n_cells = 10
low = np.random.choice(low_snr_cells, n_cells, replace=False)
middle = np.random.choice(middle_snr_cells, n_cells, replace=False)
high = np.random.choice(high_snr_cells, n_cells, replace=False)

loadkey = 'ozgf.fs100.ch18'
# for each category:
#for category, name in zip([low, middle, high], ['low', 'middle', 'high']):
for category, name in zip([high_snr_low_r_test, high_snr_high_r_test], ['high_snr_low_r_test', 'high_snr_high_r_test']):
    pdf = matplotlib.backends.backend_pdf.PdfPages(
            Path(figures_base_path) / str(batch) / name
            )
    for cellid in category:
        # load recording ( don't need evaluated model )
        rec_path = xwrap.generate_recording_uri(cellid, batch, loadkey=loadkey)
        rec = nems0.recording.load_recording(rec_path)
        est, val = rec.split_using_epoch_occurrence_counts('^STIM_')
        resp = val['resp'].extract_channels([cellid])
        epochs = resp.epochs
        fs = resp.fs
        resp = resp.as_continuous().flatten()

        # reshape from 1, T to reps, T
        stims = ep.epoch_names_matching(epochs, 'STIM_')
        stim_arrays = []
        for s in stims:
            reps = []
            row = epochs[epochs.name == s]
            starts = (row['start'].values * fs).astype(np.int)
            ends = (row['end'].values * fs).astype(np.int)
            skip_stim = False
            for i, j in zip(starts, ends):
                this_rep = resp[i:j]
                if np.isnan(this_rep).all():
                    # this is a stim from estimation set
                    skip_stim = True
                    break
                else:
                    reps.append(resp[i:j])
            if skip_stim: continue;
            rep_shapes = [r.shape for r in reps]
            max_len = np.max(rep_shapes)
            for i, shape in enumerate(rep_shapes):
                bin_diff = max_len - shape[0]
                if bin_diff > 0:
                    print('warning: had to pad rep length for cell: %s    by  %d bins' % (cellid, bin_diff))
                    reps[i] = np.hstack([reps[i], np.zeros(bin_diff,)])
            stim_array = np.vstack(reps)

            # add some nans at the end to make make it clear that time is being broken up between stimuli.
            nreps = len(reps)
            nan_flag = np.ones((nreps,50))
            stim_array = np.hstack([stim_array, nan_flag])
            stim_arrays.append(stim_array)

        raster = np.hstack(stim_arrays)
        # Set all spike entries to 1, not interested in multiple spikes if any exist
        raster[raster >= 1] = 1
        # generate raster plots and save to category's pdf
        fig = plt.figure(figsize=(8,2))
        plt.imshow(raster, cmap='Greys', interpolation='none', aspect='auto')
        fig.suptitle('cellid: %s,    snr: %.4f,    %s: %.4f' % (cellid, snrs[cellid], plot_stat, a1_results[modelname][cellid]))
        fig.tight_layout()
        pdf.savefig(fig, dpi=fig.dpi)
        plt.close(fig)

    # Done with this category, start new pdf
    pdf.close()
    plt.close('all')  # just to make double sure that everything is closed







