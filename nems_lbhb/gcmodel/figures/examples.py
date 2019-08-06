import os
import logging
import json

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np

import nems.xform_helper as xhelp
import nems.epoch as ep
from nems_lbhb.gcmodel.figures.utils import improved_cells_to_list

log = logging.getLogger(__name__)


# TODO:    Deprecated below until they are fixed to use the new return format
#          of improved_cells_to_list


















def save_improved_cells(gc_cells, stp_cells, both_cells, batch, gc, stp, LN,
                        combined, gc_dir, stp_dir, both_dir):

    save_examples_from_list(gc_cells, batch, gc, stp, LN, combined, gc_dir,
                            skip_combined=True)
    save_examples_from_list(stp_cells, batch, gc, stp, LN, combined, stp_dir,
                            skip_combined=True)
    save_examples_from_list(both_cells, batch, gc, stp, LN, combined, both_dir,
                            skip_combined=False)


def example_cell(cellid, batch, gc, stp, LN, combined):

    gc_ctx, stp_ctx, LN_ctx, combined_ctx = \
            _get_plot_contexts(cellid, batch, gc, stp, LN, combined)

    gc_pred, stp_pred, LN_pred, combined_pred, resp, stim = \
        _get_plot_signals(gc_ctx, stp_ctx, LN_ctx, combined_ctx)

    def plot_maker(start=0, stop=None, show_gc=True, show_stp=True,
                   show_LN=True, show_combined=True, show_resp=True,
                   show_stim=True, cmap='viridis', figsize=(8,5),
                   show_yaxis=True, show_xaxis=True, seconds=False,
                   linewidth=1):
        signals = []
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        xmin = 0
        length = stim.shape[1]
        if stop is not None:
            xmax = stop - start
        else:
            xmax = length

        max_all = np.nanmax(np.concatenate(
                [resp[start:stop], LN_pred[start:stop], gc_pred[start:stop],
                 stp_pred[start:stop], combined_pred[start:stop]]
                ))
        gc_plot = gc_pred / max_all
        stp_plot = stp_pred / max_all
        LN_plot = LN_pred / max_all
        combined_plot = combined_pred / max_all
        resp_plot = resp / max_all

        if show_stim:
            plt.imshow(stim[:, start:stop], aspect='auto', cmap=cmap,
                       origin='lower', extent=(xmin, xmax, 1.05, 1.5))

        if show_resp:
            plt.plot(resp_plot[start:stop], color='gray', alpha=0.2, linewidth=linewidth)
            signals.append('Response')

        if show_LN:
            plt.plot(LN_plot[start:stop], color='black', alpha=0.5, linewidth=linewidth)
            signals.append('LN')

        if show_gc:
            plt.plot(gc_plot[start:stop], color='green', alpha=0.5, linewidth=linewidth)
            signals.append('GC')

        if show_stp:
            plt.plot(stp_plot[start:stop], color='blue', alpha=0.5, linewidth=linewidth)
            signals.append('STP')

        if show_combined:
            plt.plot(combined_plot[start:stop], color='orange', alpha=0.5, linewidth=linewidth)
            signals.append('GC+STP')

        plt.legend(signals, bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand',
                   loc='lower left', ncol=len(signals))

        if not show_yaxis:
            ax.get_yaxis().set_visible(False)
        if not show_xaxis:
            ax.get_xaxis().set_visible(False)

        return fig

    plot_maker.contexts = {'gc': gc_ctx, 'stp': stp_ctx, 'LN': LN_ctx,
                           'combined': combined_ctx}

    return plot_maker


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
