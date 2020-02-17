import os
import logging
import json
import pickle

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from scipy.signal import convolve

import nems.xform_helper as xhelp
import nems.epoch as ep
from nems.utils import ax_remove_box
import nems.plots.api as nplt
from nems_lbhb.gcmodel.figures.utils import improved_cells_to_list

from nems_lbhb.gcmodel.figures.definitions import *
log = logging.getLogger(__name__)


def save_examples_from_list(cellids, batch, gc, stp, LN, combined, directory,
                            skip_combined=False, normalize=True):
    for cellid in cellids:
        gc_ctx, stp_ctx, LN_ctx, combined_ctx = \
                _get_plot_contexts(cellid, batch, gc, stp, LN, combined)

        gc_pred, stp_pred, LN_pred, combined_pred, resp, stim = \
            _get_plot_signals(gc_ctx, stp_ctx, LN_ctx, combined_ctx)

        gc_v, stp_v, LN_v, combined_v = _get_plot_vals(gc_ctx, stp_ctx, LN_ctx,
                                                       combined_ctx)

        check_epochs = [gc_v.epochs == stp_v.epochs,
                        stp_v.epochs == LN_v.epochs,
                        LN_v.epochs == combined_v.epochs]
        equal = True
        for df in check_epochs:
            if not np.all(df['name'].values):
                equal = False
        if not equal:
            log.warning("Epochs don't match across models for cell: %s, skip",
                        cellid)
            break

        # break up into separate stims
        epochs = gc_v.epochs
        stims = ep.epoch_names_matching(epochs, 'STIM_')
        fs = gc_v['resp'].fs
        pdf = matplotlib.backends.backend_pdf.PdfPages(
                os.path.join(directory, cellid) + '.pdf')
        for c in [gc_ctx, stp_ctx, LN_ctx, combined_ctx]:
            text_fig = plt.figure(figsize=(12,8))
            text_fig.clf()
            meta = c['modelspec'].meta
            for k, v in meta.items():
                if type(v) == np.ndarray:
                    meta[k] = str(v[0])
                if type(v) != str:
                    meta[k] = str(v)

            text = json.dumps(meta, indent=2)
            text_fig.text(0.05, 0.95, text, transform=text_fig.transFigure,
                          size=8, verticalalignment='top', wrap=True)
            pdf.savefig(text_fig)
            plt.close(text_fig)

        for s in stims:
            row = epochs[epochs.name == s]
            start = int(row['start'].values[0]*fs)
            end = int(row['end'].values[0]*fs)

            resp_plot = resp[start:end]
            LN_plot = LN_pred[start:end]
            gc_plot = gc_pred[start:end]
            stp_plot = stp_pred[start:end]
            combined_plot = combined_pred[start:end]
            stim_plot = stim[:, start:end]

            if normalize:
                max_all = np.nanmax(np.concatenate(
                        [resp_plot, LN_plot, gc_plot, stp_plot, combined_plot]
                        ))
                gc_plot = gc_plot / max_all
                stp_plot = stp_plot / max_all
                LN_plot = LN_plot / max_all
                combined_plot = combined_plot / max_all
                resp_plot = resp_plot / max_all

            fig = plt.figure(figsize=(12,8))
            xmin = 0
            xmax = end - start
            plt.imshow(stim_plot, aspect='auto', cmap='viridis',
                       origin='lower', extent=(xmin, xmax, 1.3, 2.6))
            plt.plot(resp_plot, color='gray', alpha=0.2)
            plt.plot(LN_plot, color='black', alpha=0.5)
            plt.plot(gc_plot, color='green', alpha=0.5)
            plt.plot(stp_plot, color='blue', alpha=0.5)
            signals = ['Response', 'LN', 'GC', 'STP']
            if not skip_combined:
                plt.plot(combined_plot, color='orange', alpha=0.5)
                signals.append('GC+STP')
            plt.legend(signals, bbox_to_anchor=(0, 1.02, 1, 0.2),
                       mode='expand', loc='lower left', ncol=5)

            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()


def example_clip(cellid, batch, gc, stp, LN, combined, skip_combined=False,
                 normalize=True, stim_idx=0, trim_start=None, trim_end=None,
                 smooth_response=False, kernel_length=3, strf_spec='LN',
                 load_path=None, save_path=None):
    if load_path is None:
        gc_ctx, stp_ctx, LN_ctx, combined_ctx = \
                _get_plot_contexts(cellid, batch, gc, stp, LN, combined)
        if save_path is not None:
            results = {'contexts': [gc_ctx, stp_ctx, LN_ctx, combined_ctx]}
            pickle.dump(results, open(save_path, 'wb'))
    else:
        results = pickle.load(open(load_path, 'rb'))
        gc_ctx, stp_ctx, LN_ctx, combined_ctx = results['contexts']

    gc_pred, stp_pred, LN_pred, combined_pred, resp, stim = \
        _get_plot_signals(gc_ctx, stp_ctx, LN_ctx, combined_ctx)

    gc_v, stp_v, LN_v, combined_v = _get_plot_vals(gc_ctx, stp_ctx, LN_ctx,
                                                   combined_ctx)

    # break up into separate stims
    epochs = gc_v.epochs
    stims = ep.epoch_names_matching(epochs, 'STIM_')
    s = stims[stim_idx]
    row = epochs[epochs.name == s]
    fs = gc_v['resp'].fs
    start = int(row['start'].values[0]*fs)
    end = int(row['end'].values[0]*fs)

    if trim_start is not None:
        start += trim_start
    if trim_end is not None:
        end = start + (trim_end - trim_start)

    resp_plot = resp[start:end]
    if smooth_response:
        # box filter, "simple average"
        kernel = np.ones((kernel_length,))*(1/kernel_length)
        resp_plot = convolve(resp_plot, kernel, mode='same')
    LN_plot = LN_pred[start:end]
    gc_plot = gc_pred[start:end]
    stp_plot = stp_pred[start:end]
    combined_plot = combined_pred[start:end]
    stim_plot = stim[:, start:end]
    if normalize:
        max_all = np.nanmax(np.concatenate(
                [resp_plot, LN_plot, gc_plot, stp_plot, combined_plot]
                ))
        gc_plot = gc_plot / max_all
        stp_plot = stp_plot / max_all
        LN_plot = LN_plot / max_all
        combined_plot = combined_plot / max_all
        resp_plot = resp_plot / max_all

    fig = plt.figure(figsize=wide_fig)
    xmin = 0
    xmax = end - start
    plt.imshow(stim_plot, aspect='auto', cmap='Greys',
               origin='lower', extent=(xmin, xmax, 1.1, 1.5))
    lw = 0.75
    plt.plot(resp_plot, color='gray', alpha=0.65, linewidth=lw)
    t = np.linspace(0, resp_plot.shape[-1]-1, resp_plot.shape[-1])
    plt.fill_between(t, resp_plot, color='gray', alpha=0.15)
    plt.plot(gc_plot, color=model_colors['gc'], linewidth=lw)
    plt.plot(stp_plot, color=model_colors['stp'], alpha=0.65,
             linewidth=lw*1.25)
    plt.plot(LN_plot, color='black', alpha=0.55, linewidth=lw)
    signals = ['Response', 'LN', 'GC', 'STP']
    if not skip_combined:
        plt.plot(combined_plot, color='orange', linewidth=lw)
        signals.append('GC+STP')
    plt.ylim(-0.1, 1.5)
    ax = plt.gca()
    ax_remove_box(ax)

    fig2 = plt.figure(figsize=text_fig)
    text = ("cellid: %s\n"
            "stp_r_test: %.4f\n"
            "gc_r_test: %.4f\n"
            "LN_r_test: %.4f\n"
            "comb_r_test: %.4f"
            % (cellid,
               stp_ctx['modelspec'].meta['r_test'],
               gc_ctx['modelspec'].meta['r_test'],
               LN_ctx['modelspec'].meta['r_test'],
               combined_ctx['modelspec'].meta['r_test']
               ))
    plt.text(0.1, 0.5, text)


    # TODO: probably need to just rip code out of strf_heatmap instead,
    #       setting the extent is not working. or alternatively just
    #       resize it manually to mach the spectrogram
    fig3 = plt.figure(figsize=wide_fig)
    ax2 = plt.gca()

    if strf_spec == 'LN':
        modelspec = LN_ctx['modelspec']
    elif strf_spec == 'stp':
        modelspec = stp_ctx['modelspec']
    elif strf_spec == 'gc':
        modelspec = gc_ctx['modelspec']
    else:
        modelspec = combined_ctx['modelspec']

    nplt.strf_heatmap(modelspec, ax=ax2, show_factorized=False,
                 show_cbar=False, manual_extent=(0, 1, 1.1, 1.5))
    ax2.set_ylim(-0.1, 1.5)
    ax_remove_box(ax2)

    return fig, fig2, fig3


def _get_plot_contexts(cellid, batch, gc, stp, LN, combined):
    xfspec1, gc_ctx = xhelp.load_model_xform(cellid, batch, gc,
                                             eval_model=True)
    xfspec2, stp_ctx = xhelp.load_model_xform(cellid, batch, stp,
                                              eval_model=True)
    xfspec3, LN_ctx = xhelp.load_model_xform(cellid, batch, LN,
                                             eval_model=True)
    xfspec4, combined_ctx = xhelp.load_model_xform(cellid, batch, combined,
                                                   eval_model=True)

    return gc_ctx, stp_ctx, LN_ctx, combined_ctx


def _get_plot_signals(gc_ctx, stp_ctx, LN_ctx, combined_ctx):
    gc_pred = gc_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    stp_pred = stp_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    LN_pred = LN_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    combined_pred = combined_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    resp = gc_ctx['val'].apply_mask()['resp'].as_continuous().T.flatten()
    stim = gc_ctx['val'].apply_mask()['stim'].as_continuous()

    return gc_pred, stp_pred, LN_pred, combined_pred, resp, stim


def _get_plot_vals(gc_ctx, stp_ctx, LN_ctx, combined_ctx):
    gc_v = gc_ctx['val'].apply_mask(reset_epochs=True)
    stp_v = stp_ctx['val'].apply_mask(reset_epochs=True)
    LN_v = LN_ctx['val'].apply_mask(reset_epochs=True)
    combined_v = combined_ctx['val'].apply_mask(reset_epochs=True)

    return gc_v, stp_v, LN_v, combined_v
