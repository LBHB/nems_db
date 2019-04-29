import matplotlib.pyplot as plt
import nems.xform_helper as xhelp


def example_cell(cellid, batch, gc, stp, LN, combined):

    xfspec1, gc_ctx = xhelp.load_model_xform(cellid, batch, gc,
                                             eval_model=True)
    xfspec2, stp_ctx = xhelp.load_model_xform(cellid, batch, stp,
                                              eval_model=True)
    xfspec3, LN_ctx = xhelp.load_model_xform(cellid, batch, LN,
                                             eval_model=True)
    xfspec4, combined_ctx = xhelp.load_model_xform(cellid, batch, combined,
                                                   eval_model=True)

    gc_pred = gc_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    stp_pred = stp_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    LN_pred = LN_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    combined_pred = combined_ctx['val'].apply_mask()['pred'].as_continuous().T.flatten()
    resp = gc_ctx['val'].apply_mask()['resp'].as_continuous().T.flatten()
    stim = gc_ctx['val'].apply_mask()['stim'].as_continuous()

    def plot_maker(start=0, stop=None, show_gc=True, show_stp=True,
                   show_LN=True, show_combined=True, show_resp=True,
                   show_stim=True):
        signals = []
        fig = plt.figure(figsize=(8, 4))
        xmin = 0
        length = stim.shape[1]
        if stop is not None:
            xmax = stop - start
        else:
            xmax = length

        if show_stim:
            plt.imshow(stim[:, start:stop], aspect='auto', cmap='viridis',
                       origin='lower', extent=(xmin, xmax, 1.5, 3))

        if show_resp:
            plt.plot(resp[start:stop], color='gray', alpha=0.2)
            signals.append('Response')

        if show_LN:
            plt.plot(LN_pred[start:stop], color='black', alpha=0.5)
            signals.append('LN')

        if show_gc:
            plt.plot(gc_pred[start:stop], color='green', alpha=0.5)
            signals.append('GC')

        if show_stp:
            plt.plot(stp_pred[start:stop], color='blue', alpha=0.5)
            signals.append('STP')

        if show_combined:
            plt.plot(combined_pred[start:stop], color='orange', alpha=0.5)
            signals.append('GC+STP')

        #plt.legend(signals)
        fig.legend(signals)

        return fig

    return plot_maker
