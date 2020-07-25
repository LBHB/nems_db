import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import stats, linalg
import matplotlib.pyplot as plt

import nems
import nems.xform_helper as xhelp
import nems.xforms as xforms
import nems.db as nd
import nems.epoch as ep
from nems.utils import ax_remove_box
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             get_valid_improvements,
                                             adjustFigAspect,
                                             improved_cells_to_list,
                                             is_outlier, drop_common_outliers)

from nems_db.params import fitted_params_per_batch
import nems_lbhb.xform_wrappers as xwrap
from nems_lbhb.gcmodel.figures.definitions import *

plt.rcParams.update(params)  # loaded from definitions


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


def snr_by_batch(batch, gc, stp, LN, combined, save_path=None, load_path=None,
                 frac_total=True):

    _, cellids, _, _, _ = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           as_lists=True)
    siteids = list(set([c.split('-')[0] for c in cellids]))
    loadkey = gc.split('_')[0]

    snrs = []
    cells = []
    if load_path is None:
        for site in siteids:
            rec_path = xwrap.generate_recording_uri(site, batch, loadkey=loadkey)
            rec = nems.recording.load_recording(rec_path)
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


def snr_vs_equivalence(snr_path, stp_path, gc_path):
    stp_equiv_df = pd.read_pickle(stp_path)
    gc_equiv_df = pd.read_pickle(gc_path)
    snr_df = pd.read_pickle(snr_path)
    cellids = list(set(stp_equiv_df.index.values.tolist()) &
                   set(gc_equiv_df.index.values.tolist()))

    snr_df = snr_df.loc[cellids].reindex(cellids)
    stp_equiv_df = stp_equiv_df.loc[cellids].reindex(cellids)
    gc_equiv_df = gc_equiv_df.loc[cellids].reindex(cellids)

    stp_equivs = stp_equiv_df['equivalence'].values
    gc_equivs = gc_equiv_df['equivalence'].values
    snrs = snr_df['snr'].values

    r_stp, p_stp = st.pearsonr(stp_equivs, snrs)
    r_gc, p_gc = st.pearsonr(gc_equivs, snrs)

    # TODO: remove fig size and axis labels, just using while testing
    fig1 = plt.figure()
    plt.scatter(snrs, stp_equivs, c=model_colors['stp'], s=big_scatter)
    plt.tight_layout()
    ax = plt.gca()
    ax_remove_box(ax)

    fig2 = plt.figure()
    plt.scatter(snrs, gc_equivs, c=model_colors['gc'], s=big_scatter)
    plt.tight_layout()
    ax = plt.gca()
    ax_remove_box(ax)


    fig3 = plt.figure(figsize=text_fig)
    text = ("SNR vs equivalence\n"
            "x axis: signal power / total power\n"
            "y axis: equivalence (partial corr)\n"
            "r_stp: %.4E\n"
            "p_stp: %.4E\n"
            "r_gc: %.4E\n"
            "p_gc: %.4E\n"

            % (r_stp, p_stp, r_gc, p_gc))

    plt.text(0.1, 0.5, text)
    ax_remove_box(ax)


    return fig1, fig2, fig3