import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import stats, linalg
import matplotlib.pyplot as plt

import nems
import nems0.xform_helper as xhelp
import nems0.xforms as xforms
import nems0.db as nd
import nems0.epoch as ep
from nems0.utils import ax_remove_box
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

def compute_snr_multi(resp, frac_total=True):
    epochs = resp.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    resp_dict = resp.extract_epochs(stim_epochs)

    chan_count=resp.shape[0]
    snr = np.zeros(chan_count)
    for cidx in range(chan_count):
        per_stim_snrs = []
        for stim, r in resp_dict.items():
            repcount=r.shape[0]
            if repcount>2:
                for j in range(repcount):
                    _r = r[:,cidx,:]
                    products = np.dot(_r, _r.T)
                    per_rep_snrs = []
                    for i in range(repcount):
                        total_power = products[i,i]
                        signal_powers = np.delete(products[i], i)
                        if frac_total:
                            rep_snr = np.nanmean(signal_powers)/total_power
                        else:
                            rep_snr = np.nanmean(signal_powers/(total_power-signal_powers))

                        per_rep_snrs.append(rep_snr)
                    per_stim_snrs.append(np.nanmean(per_rep_snrs))
        snr[cidx] = np.nanmean(per_stim_snrs)
        #print(resp.chans[cidx], snr[cidx])
    return snr


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

    md_snr = np.nanmedian(snrs)
    low_snr_mask = snrs < md_snr
    high_snr_mask = snrs >= md_snr
    stp_low_equivs = stp_equivs[low_snr_mask]
    stp_high_equivs = stp_equivs[high_snr_mask]
    gc_low_equivs = gc_equivs[low_snr_mask]
    gc_high_equivs = gc_equivs[high_snr_mask]

    md_stp_low = np.median(stp_low_equivs)
    md_stp_high = np.median(stp_high_equivs)
    md_gc_low = np.median(gc_low_equivs)
    md_gc_high = np.median(gc_high_equivs)

    u_stp, p_stp = st.mannwhitneyu(stp_low_equivs, stp_high_equivs,
                                   alternative='two-sided')
    u_gc, p_gc = st.mannwhitneyu(gc_low_equivs, gc_high_equivs,
                                 alternative='two-sided')

    #r_stp, p_stp = st.pearsonr(stp_equivs, snrs)
    #r_gc, p_gc = st.pearsonr(gc_equivs, snrs)

    fig1 = plt.figure(figsize=small_fig)
    ax1 = plt.gca()
    plt.scatter(snrs, stp_equivs, c=model_colors['stp'], s=big_scatter)
    ax1.axes.axvline(md_snr, color='black', linewidth=1, linestyle='dashed',
                    dashes=dash_spacing)
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    ax_remove_box(ax1)


    fig2 = plt.figure(figsize=small_fig)
    ax2 = plt.gca()
    plt.scatter(snrs, gc_equivs, c=model_colors['gc'], s=big_scatter)
    ax2.axes.axvline(md_snr, color='black', linewidth=1, linestyle='dashed',
                    dashes=dash_spacing)
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    ax_remove_box(ax2)

    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))


    fig3 = plt.figure(figsize=text_fig)
    text = ("SNR vs equivalence\n"
            "x axis: signal power / total power\n"
            "y axis: equivalence (partial corr)\n"
            "mannwhitneyu two sided low vs high snr\n"
            "n_high: %d\n"
            "n_low: %d\n"
            "u_stp: %.4E\n"
            "p_stp: %.4E\n"
            "md_stp_low: %.4E\n"
            "md_stp_high: %.4E\n"
            "u_gc: %.4E\n"
            "p_gc: %.4E\n"
            "md_gc_low: %.4E\n"
            "md_gc_high: %.4E"
            % (stp_high_equivs.size, stp_low_equivs.size, u_stp, p_stp,
               md_stp_low, md_stp_high, u_gc, p_gc, md_gc_low, md_gc_high))

    plt.text(0.1, 0.5, text)


    return fig1, fig2, fig3
