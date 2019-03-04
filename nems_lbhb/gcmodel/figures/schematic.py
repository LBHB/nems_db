import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import nems.modelspec as ms
import nems.xform_helper as xhelp
from nems.utils import find_module
from nems.modules.nonlinearity import _logistic_sigmoid, _double_exponential
from nems.plots.heatmap import _get_fir_coefficients, _get_wc_coefficients
from nems.metrics.stp import stp_magnitude


def pred_resp_parameters(cellid, batch, gc_model):

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, gc_model,
                                         eval_model=True)
    gc_ms = ctx['modelspec']
    val = copy.deepcopy(ctx['val'])
    gc_ms.recording = val
    dsig_idx = find_module('dynamic_sigmoid', gc_ms)

    # Normal pred from full gc and ln models
    pred_gc = ms.evaluate(val, gc_ms)['pred'].as_continuous().T

    # Remove effect of each %_mod and save corresponding prediction
    pred_no_base = _reduced_param_pred(gc_ms, val, dsig_idx, 'base')
    pred_no_amp = _reduced_param_pred(gc_ms, val, dsig_idx, 'amplitude')
    pred_no_shift = _reduced_param_pred(gc_ms, val, dsig_idx, 'shift')
    pred_no_kappa = _reduced_param_pred(gc_ms, val, dsig_idx, 'kappa')

    fig, (ax2, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(12, 6))

    ax2.plot(pred_no_base, color='black')
    ax2.plot(pred_no_base - pred_gc, color='green')
    ax2.set_title('-base')

    ax3.plot(pred_no_amp, color='black')
    ax3.plot(pred_no_amp - pred_gc, color='green')
    ax3.set_title('-amplitude')

    ax4.plot(pred_no_shift, color='black')
    ax4.plot(pred_no_shift - pred_gc, color='green')
    ax4.set_title('-shift')

    ax5.plot(pred_no_kappa, color='black')
    ax5.plot(pred_no_kappa - pred_gc, color='green')
    ax5.set_title('-kappa')

    fig.tight_layout()


    phi = gc_ms.phi[-1]
    b = phi['base']
    b_m = phi['base_mod']
    a = phi['amplitude']
    a_m = phi['amplitude_mod']
    s = phi['shift']
    s_m = phi['shift_mod']
    k = phi['kappa']
    k_m = phi['kappa_mod']

    fig2, ((ax1b, ax2b, ax5b), (ax3b, ax4b, ax6b)) = plt.subplots(2, 3, figsize=(8, 8))
    scale = np.asscalar(max(s, s_m)*2)
    x = np.linspace(scale*-1, scale*3, 10000)
    y = _double_exponential(x, b, a, s, k).flatten()

    ax1b.plot(x, y, color='black')
    ax1b.plot(x, _double_exponential(x, b_m, a, s, k).flatten(), color='green')
    ax1b.set_title('+base')

    ax2b.plot(x, y, color='black')
    ax2b.plot(x, _double_exponential(x, b, a_m, s, k).flatten(), color='green')
    ax2b.set_title('+amplitude')

    ax3b.plot(x, y, color='black')
    ax3b.plot(x, _double_exponential(x, b, a, s_m, k).flatten(), color='green')
    ax3b.set_title('+shift')

    ax4b.plot(x, y, color='black')
    ax4b.plot(x, _double_exponential(x, b, a, s, k_m).flatten(), color='green')
    ax4b.set_title('+kappa')

    ax5b.plot(x, y, color='black')
    ax5b.plot(x, _double_exponential(x, b_m, a_m, s_m, k_m).flatten(),
              color='green')
    ax5b.set_title('+all')

    fig2.tight_layout()


def _reduced_param_pred(mspec, rec, idx, param):
    gc_ms_no_param = mspec.copy()
    gc_ms_no_param[idx]['phi']['%s_mod' % param] = \
            gc_ms_no_param[idx]['phi']['%s' % param].copy()
    pred_no_param = ms.evaluate(rec, gc_ms_no_param)['pred'].as_continuous().T

    return pred_no_param


def contrast_breakdown(cellid, batch, model1, model2, model3, sample_every=5):
    '''
    model1: GC
    model2: STP
    model3: LN

    '''

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, model1)
    val = copy.deepcopy(ctx['val'])
    fs = val['resp'].fs
    mspec = ctx['modelspec']
    dsig_idx = find_module('dynamic_sigmoid', mspec)

    before = ms.evaluate(val, mspec, start=None, stop=dsig_idx)
    pred_before = copy.deepcopy(before['pred']).as_continuous()[0, :].T

    after = ms.evaluate(before.copy(), mspec, start=dsig_idx, stop=dsig_idx+1)
    pred_after = after['pred'].as_continuous()[0, :].T

    ctpred = after['ctpred'].as_continuous()[0, :]
    resp = after['resp'].as_continuous()[0, :]

    phi = mspec[dsig_idx]['phi']
    kappa = phi['kappa']
    shift = phi['shift']
    kappa_mod = phi['kappa_mod']
    shift_mod = phi['shift_mod']
    base = phi['base']
    amplitude = phi['amplitude']
    base_mod = phi['base_mod']
    amplitude_mod = phi['amplitude_mod']

    k = (kappa + (kappa_mod - kappa)*ctpred).flatten()
    s = (shift + (shift_mod - shift)*ctpred).flatten()
    b = (base + (base_mod - base)*ctpred).flatten()
    a = (amplitude + (amplitude_mod - amplitude)*ctpred).flatten()

    xfspec2, ctx2 = xhelp.load_model_xform(cellid, batch, model3)
    val2 = copy.deepcopy(ctx2['val'])
    mspec2 = ctx2['modelspec']
    logsig_idx = find_module('logistic_sigmoid', mspec2)
    dexp_idx = find_module('double_exponential', mspec2)
    nl_idx = logsig_idx if logsig_idx is not None else dexp_idx
    before2 = ms.evaluate(val2, mspec2, start=None, stop=nl_idx)
    pred_before_LN = copy.deepcopy(before2['pred']).as_continuous()[0, :].T
    after2 = ms.evaluate(before2.copy(), mspec2, start=nl_idx, stop=nl_idx+1)
    pred_after_LN_only = after2['pred'].as_continuous()[0, :].T

    if logsig_idx:
        nonlin_fn = _logistic_sigmoid
    else:
        nonlin_fn = _double_exponential

    mspec2 = ctx2['modelspec']
    ln_phi = mspec2[nl_idx]['phi']
    ln_k = ln_phi['kappa']
    ln_s = ln_phi['shift']
    ln_b = ln_phi['base']
    ln_a = ln_phi['amplitude']

    xfspec3, ctx3 = xhelp.load_model_xform(cellid, batch, model2)
    val3 = copy.deepcopy(ctx3['val'])
    mspec3 = ctx3['modelspec']
    logsig_idx = find_module('logistic_sigmoid', mspec2)
    dexp_idx = find_module('double_exponential', mspec2)
    nl_idx = logsig_idx if logsig_idx is not None else dexp_idx
    before3 = ms.evaluate(val3, mspec3, start=None, stop=nl_idx)
    pred_before_stp = copy.deepcopy(before3['pred']).as_continuous()[0, :].T
    after3 = ms.evaluate(before3.copy(), mspec3, start=nl_idx, stop=nl_idx+1)
    pred_after_stp = after3['pred'].as_continuous()[0, :].T

    # Re-align data w/o any NaN predictions and convert to real-time
    ff = np.isfinite(pred_before) & np.isfinite(pred_before_LN) \
            & np.isfinite(pred_before_stp) & np.isfinite(pred_after) \
            & np.isfinite(pred_after_LN_only) & np.isfinite(pred_after_stp)
    pred_before = pred_before[ff]
    pred_before_LN = pred_before_LN[ff]
    pred_before_stp = pred_before_stp[ff]
    pred_after = pred_after[ff]
    pred_after_LN_only = pred_after_LN_only[ff]
    pred_after_stp = pred_after_stp[ff]
    ctpred = ctpred[ff]
    resp = resp[ff]

    k = k[ff]
    s = s[ff]
    b = b[ff]
    a = a[ff]

#    static_k = np.full_like(k, ln_k)
#    static_s = np.full_like(s, ln_s)
#
#    static_b = np.full_like(b, ln_b)
#    static_a = np.full_like(a, ln_a)

    t = np.arange(len(pred_before))/fs

    # Contrast variables figure
    fig2 = plt.figure(figsize=(7, 12))
    st2 = fig2.suptitle("Cellid: %s\nModelname: %s" % (cellid, model1))
    gs2 = gridspec.GridSpec(12, 3)

    plt.subplot(gs2[0:3, 0])
    val = ctx['val'].apply_mask()
    plt.imshow(val['stim'].as_continuous(), origin='lower', aspect='auto')
    plt.title('Stimulus')


    modelspec = ctx['modelspec']

    plt.subplot(gs2[3:6, 0])
    wcc = _get_wc_coefficients(modelspec, idx=0)
    firc = _get_fir_coefficients(modelspec, idx=0)
    wc_coefs = np.array(wcc).T
    fir_coefs = np.array(firc)
    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    plt.title('STRF')

#    plt.subplot(gs2[0:3, 1])
#    plt.plot(t, s, linewidth=1, color='red')
#    plt.plot(t, static_s, linewidth=1, linestyle='dashed', color='red')
#    plt.title('Shift w/ GC vs Shift w/ LN')
#    plt.legend(['GC', 'LN'])
#
#    plt.subplot(gs2[3:6, 1])
#    plt.plot(t, k, linewidth=1, color='blue')
#    plt.plot(t, static_k, linewidth=1, linestyle='dashed', color='blue')
#    plt.title('Kappa w/ GC vs Kappa w/ LN')
#    plt.legend(['GC', 'LN'])
#
#    plt.subplot(gs2[6:9, 1])
#    plt.plot(t, b, linewidth=1, color='gray')
#    plt.plot(t, static_b, linewidth=1, linestyle='dashed', color='gray')
#    plt.title('Base w/ GC vs Base w/ LN')
#    plt.legend(['GC', 'LN'])
#
#    plt.subplot(gs2[9:12, 1])
#    plt.plot(t, a, linewidth=1, color='orange')
#    plt.plot(t, static_a, linewidth=1, linestyle='dashed', color='orange')
#    plt.title('Amplitude w/ GC vs Amplitude w/ LN')
#    plt.legend(['GC', 'LN'])

    ax2 = plt.subplot(gs2[6:9, 0])

    plt.subplot(gs2[9:12, 0])
    plt.plot(t, pred_after, color='black')
    plt.title('Prediction')

    plt.subplot(gs2[0:3, 1])
    plt.imshow(val['contrast'].as_continuous(), origin='lower', aspect='auto')
    plt.title('Contrast')

    plt.subplot(gs2[3:6, 1])

    if 'gcwc' not in model1:
        wcc = _get_wc_coefficients(modelspec, idx=1)
        firc = _get_fir_coefficients(modelspec, idx=1)
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)
    else:
        wc_coefs = np.abs(wc_coefs)
        fir_coefs = np.abs(fir_coefs)

    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    plt.title('Contrast STRF')

    plt.subplot(gs2[6:9, 1])
    plt.plot(t, ctpred, linewidth=1, color='purple')
    plt.title("Output from Contrast STRF")

    plt.subplot(gs2[9:12, 1])
    plt.plot(t, resp, color='green')
    plt.title('Response')

    plt.subplot(gs2[0:6, 2])
    x = np.linspace(-1*ln_s, 3*ln_s, 1000)
    y = nonlin_fn(x, ln_b, ln_a, ln_s, ln_k)
    plt.plot(x, y, color='black')
    plt.title('Static Nonlinearity')

    ax1 = plt.subplot(gs2[6:12, 2])

    y_min = 0
    y_max = 0
    x = np.linspace(-1*s[0], 3*s[0], 1000)
    sample_every = max(1, sample_every)
    sample_every = min(len(a), sample_every)
    cmap = matplotlib.cm.get_cmap('copper')
    color_pred = ctpred/np.max(np.abs(ctpred))
    alpha = 1.1 - 2/max(2.222222, np.log(sample_every))  # range from 0.1 to 1
    for i in range(int(len(a)/sample_every)):
        try:
            this_b = b[i*sample_every]
            this_a = a[i*sample_every]
            this_s = s[i*sample_every]
            this_k = k[i*sample_every]
            this_x = np.linspace(-1*this_s, 3*this_s, 1000)
            this_y1 = nonlin_fn(x, this_b, this_a, this_s, this_k)
            this_y2 = nonlin_fn(this_x, this_b, this_a, this_s, this_k)
            y_min = min(y_min, this_y1.min(), this_y2.min())
            y_max = max(y_max, this_y2.max(), this_y2.max())
            color = cmap(color_pred[i*sample_every])
            ax1.plot(x, this_y1, color='gray', alpha=alpha)
            ax2.plot(this_x+i*sample_every, this_y2, color=color, alpha=alpha)
        except IndexError:
            # Will happen on last attempt if array wasn't evenly divisible
            # by sample_every
            pass

    y2 = nonlin_fn(x, b[0], a[0], s[0], k[0])
    # no-stim sigmoid for reference
    ax1.plot(x, y2, color='black')
    # highest-contrast sigmoid for reference
    max_idx = np.argmax(np.abs(ctpred - ctpred[0]))
    y3 = nonlin_fn(x, b[max_idx], a[max_idx], s[max_idx], k[max_idx])
    ax1.plot(x, y3, color='red')
    some_contrast = np.abs(ctpred - ctpred[0])/np.abs(ctpred[0]) > 0.02
    threshold = np.percentile(ctpred[some_contrast], 50)
    # Ctpred goes "up" for higher contrast
    if ctpred[max_idx] >= ctpred[0]:
        high_contrast = ctpred >= threshold
        low_contrast = np.logical_and(ctpred < threshold, some_contrast)
    # '' goes "down" for higher contrast
    else:
        high_contrast = ctpred <= threshold
        low_contrast = np.logical_and(ctpred > threshold, some_contrast)

    high_b = b[high_contrast]; low_b = b[low_contrast]
    high_a = a[high_contrast]; low_a = a[low_contrast]
    high_s = s[high_contrast]; low_s = s[low_contrast]
    high_k = k[high_contrast]; low_k = k[low_contrast]
    y4 = nonlin_fn(x, np.median(high_b), np.median(high_a),
                   np.median(high_s), np.median(high_k))
    y5 = nonlin_fn(x, np.median(low_b), np.median(low_a),
                   np.median(low_s), np.median(low_k))
    ax1.plot(x, y4, color='orange')
    ax1.plot(x, y5, color='blue')
#    strength = gc_magnitude(base, base_mod, amplitude, amplitude_mod, shift,
#                            shift_mod, kappa, kappa_mod)
#    if strength > 0:
#        ax1.text(0.95, 0.05, "GC Strength: %.2f" % strength,
#                 ha='right', va='bottom', transform=ax1.transAxes)
#    else:
#        ax1.text(0.05, 0.95, "GC Strength: %.2f" % strength,
#                 ha='left', va='top', transform=ax1.transAxes)

    ax1.set_ylim(y_min*1.25, y_max*1.25)
    ax1.set_title('Dynamic Nonlinearity')
    ax2.set_title('Dynamic Nonlinearity')

    ymin = 0
    ymax = 0
    for i, ax in enumerate(fig2.axes[:8]):
        if i not in [3, 7]:
            ax.axes.get_xaxis().set_visible(False)
        else:
            ybottom, ytop = ax.get_ylim()
            ymin = min(ymin, ybottom)
            ymax = max(ymax, ytop)
        #ax.axes.get_yaxis().set_visible(False)

    # Set pred and resp on same scale
    fig2.axes[3].set_ylim(ymin, ymax)
    fig2.axes[7].set_ylim(ymin, ymax)

    plt.tight_layout(h_pad=1, w_pad=-1)
    st2.set_y(0.95)
    fig2.subplots_adjust(top=0.85)

    return fig2






def contrast_vs_stp_comparison(cellid, batch, model1, model2, model3, model4):
    '''
    model1: GC
    model2: STP
    model3: LN
    model4: GC+STP

    '''

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, model1)
    val = copy.deepcopy(ctx['val'])
    fs = val['resp'].fs
    mspec = ctx['modelspec']
    gc_r_test = mspec[0]['meta']['r_test']
    dsig_idx = find_module('dynamic_sigmoid', mspec)
    before = ms.evaluate(val, mspec, start=None, stop=dsig_idx)
    pred_before = copy.deepcopy(before['pred']).as_continuous()[0, :].T
    after = ms.evaluate(before.copy(), mspec, start=dsig_idx, stop=dsig_idx+1)
    pred_after = after['pred'].as_continuous()[0, :].T
    ctpred = after['ctpred'].as_continuous()[0, :]
    resp = after['resp'].as_continuous()[0, :]

    phi = mspec[dsig_idx]['phi']
    kappa = phi['kappa']
    shift = phi['shift']
    kappa_mod = phi['kappa_mod']
    shift_mod = phi['shift_mod']
    base = phi['base']
    amplitude = phi['amplitude']
    base_mod = phi['base_mod']
    amplitude_mod = phi['amplitude_mod']
    k = (kappa + (kappa_mod - kappa)*ctpred).flatten()
    s = (shift + (shift_mod - shift)*ctpred).flatten()
    b = (base + (base_mod - base)*ctpred).flatten()
    a = (amplitude + (amplitude_mod - amplitude)*ctpred).flatten()

    xfspec2, ctx2 = xhelp.load_model_xform(cellid, batch, model3)
    val2 = copy.deepcopy(ctx2['val'])
    mspec2 = ctx2['modelspec']
    ln_r_test = mspec2[0]['meta']['r_test']
    logsig_idx = find_module('logistic_sigmoid', mspec2)
    dexp_idx = find_module('double_exponential', mspec2)
    nl_idx = logsig_idx if logsig_idx is not None else dexp_idx
    before2 = ms.evaluate(val2, mspec2, start=None, stop=nl_idx)
    pred_before_LN = copy.deepcopy(before2['pred']).as_continuous()[0, :].T
    after2 = ms.evaluate(before2.copy(), mspec2, start=nl_idx, stop=nl_idx+1)
    pred_after_LN_only = after2['pred'].as_continuous()[0, :].T

    if logsig_idx:
        nonlin_fn = _logistic_sigmoid
    else:
        nonlin_fn = _double_exponential

    mspec2 = ctx2['modelspec']
    ln_phi = mspec2[nl_idx]['phi']
    ln_k = ln_phi['kappa']
    ln_s = ln_phi['shift']
    ln_b = ln_phi['base']
    ln_a = ln_phi['amplitude']

    xfspec3, ctx3 = xhelp.load_model_xform(cellid, batch, model2)
    val3 = copy.deepcopy(ctx3['val'])
    mspec3 = ctx3['modelspec']
    stp_r_test = mspec3[0]['meta']['r_test']
    logsig_idx = find_module('logistic_sigmoid', mspec3)
    dexp_idx = find_module('double_exponential', mspec3)
    nl_idx = logsig_idx if logsig_idx is not None else dexp_idx
    before3 = ms.evaluate(val3, mspec3, start=None, stop=nl_idx)
    pred_before_stp = copy.deepcopy(before3['pred']).as_continuous()[0, :].T
    after3 = ms.evaluate(before3.copy(), mspec3, start=nl_idx, stop=nl_idx+1)
    pred_after_stp = after3['pred'].as_continuous()[0, :].T

    mspec3 = ctx3['modelspec']
    stp_phi = mspec3[nl_idx]['phi']
    stp_k = stp_phi['kappa']
    stp_s = stp_phi['shift']
    stp_b = stp_phi['base']
    stp_a = stp_phi['amplitude']

    xfspec4, ctx4 = xhelp.load_model_xform(cellid, batch, model4)
    val4 = copy.deepcopy(ctx4['val'])
    mspec4 = ctx4['modelspec']
    gc_stp_r_test = mspec4[0]['meta']['r_test']
    dsig_idx = find_module('dynamic_sigmoid', mspec4)
    before4 = ms.evaluate(val4, mspec4, start=None, stop=dsig_idx)
    pred_before_gc_stp = copy.deepcopy(before4['pred']).as_continuous()[0, :].T
    after4 = ms.evaluate(before4.copy(), mspec4, start=dsig_idx, stop=dsig_idx+1)
    pred_after_gc_stp = after4['pred'].as_continuous()[0, :].T
    gc_stp_ctpred = after4['ctpred'].as_continuous()[0, :]

    gs_phi = mspec4[dsig_idx]['phi']
    gs_kappa = gs_phi['kappa']
    gs_shift = gs_phi['shift']
    gs_kappa_mod = gs_phi['kappa_mod']
    gs_shift_mod = gs_phi['shift_mod']
    gs_base = gs_phi['base']
    gs_amplitude = gs_phi['amplitude']
    gs_base_mod = gs_phi['base_mod']
    gs_amplitude_mod = gs_phi['amplitude_mod']
    gs_k = (kappa + (kappa_mod - kappa)*gc_stp_ctpred).flatten()
    gs_s = (shift + (shift_mod - shift)*gc_stp_ctpred).flatten()
    gs_b = (base + (base_mod - base)*gc_stp_ctpred).flatten()
    gs_a = (amplitude + (amplitude_mod - amplitude)*gc_stp_ctpred).flatten()

    # Re-align data w/o any NaN predictions and convert to real-time
    ff = np.isfinite(pred_before) & np.isfinite(pred_before_LN) \
            & np.isfinite(pred_before_stp) & np.isfinite(pred_after) \
            & np.isfinite(pred_after_LN_only) & np.isfinite(pred_after_stp) \
            & np.isfinite(pred_before_gc_stp) & np.isfinite(pred_after_gc_stp)
    pred_before = pred_before[ff]
    pred_before_LN = pred_before_LN[ff]
    pred_before_stp = pred_before_stp[ff]
    pred_after = pred_after[ff]
    pred_after_LN_only = pred_after_LN_only[ff]
    pred_after_stp = pred_after_stp[ff]
    pred_before_gc_stp = pred_before_gc_stp[ff]
    pred_after_gc_stp = pred_after_gc_stp[ff]
    ctpred = ctpred[ff]
    gc_stp_ctpred = gc_stp_ctpred[ff]
    resp = resp[ff]

    k = k[ff]
    s = s[ff]
    b = b[ff]
    a = a[ff]
    gs_k = gs_k[ff]
    gs_s = gs_s[ff]
    gs_b = gs_b[ff]
    gs_a = gs_a[ff]

    t = np.arange(len(pred_before))/fs

    fig1 = plt.figure(figsize=(10, 10))
    st1 = fig1.suptitle("Cellid: %s\nModelname: %s" % (cellid, model1))
    gs = gridspec.GridSpec(10, 5)

    # Labels
    ax = plt.subplot(gs[0, 0])
    plt.text(1, 1, 'STP Output', ha='right', va='top', transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[1, 0])
    plt.text(1, 1, 'STRF', ha='right', va='top', transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[2, 0])
    plt.text(1, 1, 'Pred Before NL', ha='right', va='top',
             transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[3, 0])
    plt.text(1, 1, 'GC STRF', ha='right', va='top', transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[4, 0])
    plt.text(1, 1, 'GC Output', ha='right', va='top', transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[5:7, 0])
    plt.text(1, 1, 'Nonlinearity', ha='right', va='top',
             transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[7, 0])
    plt.text(1, 1, 'Pred After NL', ha='right', va='top',
             transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[8, 0])
    plt.text(1, 1, 'Change vs LN', ha='right', va='top',
             transform=ax.transAxes)
    plt.axis('off')
    ax = plt.subplot(gs[9, 0])
    plt.text(1, 1, 'Response', ha='right', va='top', transform=ax.transAxes)
    plt.axis('off')


    # LN
    plt.subplot(gs[0, 1])
    plt.axis('off')
    plt.title('LN, r_test: %.2f' % ln_r_test)

    plt.subplot(gs[1, 1])
    # STRF
    modelspec = mspec2
    wcc = _get_wc_coefficients(modelspec, idx=0)
    firc = _get_fir_coefficients(modelspec, idx=0)
    wc_coefs = np.array(wcc).T
    fir_coefs = np.array(firc)
    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    # End STRF

    plt.subplot(gs[2, 1])
    plt.plot(t, pred_before_LN, linewidth=1, color='black')

    plt.subplot(gs[3, 1])
    plt.axis('off')

    plt.subplot(gs[4, 1])
    plt.axis('off')

    plt.subplot(gs[5:7, 1])
    x = np.linspace(-1*ln_s, 3*ln_s, 1000)
    y = nonlin_fn(x, ln_b, ln_a, ln_s, ln_k)
    plt.plot(x, y, color='black')

    plt.subplot(gs[7, 1])
    plt.plot(t, pred_after_LN_only, linewidth=1, color='black')

    plt.subplot(gs[8, 1])
    plt.axis('off')

    plt.subplot(gs[9, 1])
    plt.plot(t, resp, linewidth=1, color='green')


    # GC
    plt.subplot(gs[0, 2])
    plt.axis('off')
    plt.title('GC, r_test: %.2f' % gc_r_test)

    plt.subplot(gs[1, 2])
    # STRF
    modelspec = mspec
    wcc = _get_wc_coefficients(modelspec, idx=0)
    firc = _get_fir_coefficients(modelspec, idx=0)
    wc_coefs = np.array(wcc).T
    fir_coefs = np.array(firc)
    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    # End STRF

    plt.subplot(gs[2, 2])
    plt.plot(t, pred_before, linewidth=1, color='black')

    plt.subplot(gs[3, 2])
    # GC STRF
    if 'gcwc' not in model1:
        wcc = _get_wc_coefficients(modelspec, idx=1)
        firc = _get_fir_coefficients(modelspec, idx=1)
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)
    else:
        wc_coefs = np.abs(wc_coefs)
        fir_coefs = np.abs(fir_coefs)

    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    # End GC STRF

    plt.subplot(gs[4, 2])
    plt.plot(t, ctpred, linewidth=1, color='purple')

    ax = plt.subplot(gs[5:7, 2])
    # Dynamic sigmoid plot
    y_min = 0
    y_max = 0
    x = np.linspace(-1*s[0], 3*s[0], 1000)
    sample_every = 10
    alpha = 1.1 - 2/max(2.222222, np.log(sample_every))
    for i in range(int(len(a)/sample_every)):
        try:
            this_b = b[i*sample_every]
            this_a = a[i*sample_every]
            this_s = s[i*sample_every]
            this_k = k[i*sample_every]
            this_y1 = nonlin_fn(x, this_b, this_a, this_s, this_k)
            y_min = min(y_min, this_y1.min())
            y_max = max(y_max, this_y1.max())
            plt.plot(x, this_y1, color='gray', alpha=alpha)
        except IndexError:
            # Will happen on last attempt if array wasn't evenly divisible
            # by sample_every
            pass

    y2 = nonlin_fn(x, b[0], a[0], s[0], k[0])
    # no-stim sigmoid for reference
    plt.plot(x, y2, color='black')
    # highest-contrast sigmoid for reference
    max_idx = np.argmax(np.abs(ctpred - ctpred[0]))
    y3 = nonlin_fn(x, b[max_idx], a[max_idx], s[max_idx], k[max_idx])
    plt.plot(x, y3, color='red')
    some_contrast = np.abs(ctpred - ctpred[0])/np.abs(ctpred[0]) > 0.02
    threshold = np.percentile(ctpred[some_contrast], 50)
    # Ctpred goes "up" for higher contrast
    if ctpred[max_idx] >= ctpred[0]:
        high_contrast = ctpred >= threshold
        low_contrast = np.logical_and(ctpred < threshold, some_contrast)
    # '' goes "down" for higher contrast
    else:
        high_contrast = ctpred <= threshold
        low_contrast = np.logical_and(ctpred > threshold, some_contrast)

    high_b = b[high_contrast]; low_b = b[low_contrast]
    high_a = a[high_contrast]; low_a = a[low_contrast]
    high_s = s[high_contrast]; low_s = s[low_contrast]
    high_k = k[high_contrast]; low_k = k[low_contrast]
    y4 = nonlin_fn(x, np.median(high_b), np.median(high_a),
                   np.median(high_s), np.median(high_k))
    y5 = nonlin_fn(x, np.median(low_b), np.median(low_a),
                   np.median(low_s), np.median(low_k))
    plt.plot(x, y4, color='orange')
    plt.plot(x, y5, color='blue')
    # Strength metric is still weird, leave out for now.
#    strength = gc_magnitude(base, base_mod, amplitude, amplitude_mod, shift,
#                            shift_mod, kappa, kappa_mod)
#    if strength > 0:
#        plt.text(0.95, 0.05, "GC Strength: %.2f" % strength,
#                 ha='right', va='bottom', transform=ax.transAxes)
#    else:
#        plt.text(0.05, 0.95, "GC Strength: %.2f" % strength,
#                 ha='left', va='top', transform=ax.transAxes)
    # End nonlinearity plot

    plt.subplot(gs[7, 2])
    plt.plot(t, pred_after, linewidth=1, color='black')

    plt.subplot(gs[8, 2])
    change = pred_after - pred_after_LN_only
    plt.plot(t, change, linewidth=1, color='blue')

    plt.subplot(gs[9, 2])
    plt.plot(t, resp, linewidth=1, color='green')


    # STP
    plt.subplot(gs[0, 3])
    # TODO: simplify this? just cut and pasted from existing STP plot
    for m in mspec3:
        if 'stp' in m['fn']:
            break

    stp_mag, pred, pred_out = stp_magnitude(m['phi']['tau'], m['phi']['u'], fs)
    c = len(m['phi']['tau'])
    pred.name = 'before'
    pred_out.name = 'after'
    signals = []
    channels = []
    for i in range(c):
        signals.append(pred_out)
        channels.append(i)
    signals.append(pred)
    channels.append(0)

    times = []
    values = []
    #legend = []
    for sig, c in zip(signals, channels):
        # Get values from specified channel
        value_vector = sig.as_continuous()[c]
        # Convert indices to absolute time based on sampling frequency
        time_vector = np.arange(0, len(value_vector)) / sig.fs
        times.append(time_vector)
        values.append(value_vector)
        #if sig.chans is not None:
            #legend.append(sig.name+' '+sig.chans[c])

    cc = 0
    for ts, vs in zip(times, values):
        plt.plot(ts, vs)
        cc += 1

    #plt.legend(legend)
    plt.title('STP, r_test: %.2f' % stp_r_test)
    # End STP plot

    plt.subplot(gs[1, 3])
    # STRF
    modelspec = mspec3
    wcc = _get_wc_coefficients(modelspec, idx=0)
    firc = _get_fir_coefficients(modelspec, idx=0)
    wc_coefs = np.array(wcc).T
    fir_coefs = np.array(firc)
    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    # End STRF

    plt.subplot(gs[2, 3])
    plt.plot(t, pred_before_stp, linewidth=1, color='black')

    plt.subplot(gs[3, 3])
    plt.axis('off')

    plt.subplot(gs[4, 3])
    plt.axis('off')

    plt.subplot(gs[5:7, 3])
    x = np.linspace(-1*stp_s, 3*stp_s, 1000)
    y = nonlin_fn(x, stp_b, stp_a, stp_s, stp_k)
    plt.plot(x, y, color='black')

    plt.subplot(gs[7, 3])
    plt.plot(t, pred_after_stp, linewidth=1, color='black')

    plt.subplot(gs[8, 3])
    change2 = pred_after_stp - pred_after_LN_only
    plt.plot(t, change2, linewidth=1, color='blue')

    plt.subplot(gs[9, 3])
    plt.plot(t, resp, linewidth=1, color='green')


    # GC + STP
    plt.subplot(gs[0, 4])
    # TODO: simplify this? just cut and pasted from existing STP plot
    for m in mspec4:
        if 'stp' in m['fn']:
            break

    stp_mag, pred, pred_out = stp_magnitude(m['phi']['tau'], m['phi']['u'], fs)
    c = len(m['phi']['tau'])
    pred.name = 'before'
    pred_out.name = 'after'
    signals = []
    channels = []
    for i in range(c):
        signals.append(pred_out)
        channels.append(i)
    signals.append(pred)
    channels.append(0)

    times = []
    values = []
    #legend = []
    for sig, c in zip(signals, channels):
        # Get values from specified channel
        value_vector = sig.as_continuous()[c]
        # Convert indices to absolute time based on sampling frequency
        time_vector = np.arange(0, len(value_vector)) / sig.fs
        times.append(time_vector)
        values.append(value_vector)
        #if sig.chans is not None:
            #legend.append(sig.name+' '+sig.chans[c])

    cc = 0
    for ts, vs in zip(times, values):
        plt.plot(ts, vs)
        cc += 1

    #plt.legend(legend)
    plt.title('GC + STP, r_test: %.2f' % gc_stp_r_test)
    # End STP plot

    plt.subplot(gs[1, 4])
    # STRF
    modelspec = mspec4
    wcc = _get_wc_coefficients(modelspec, idx=0)
    firc = _get_fir_coefficients(modelspec, idx=0)
    wc_coefs = np.array(wcc).T
    fir_coefs = np.array(firc)
    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    # End STRF

    plt.subplot(gs[2, 4])
    plt.plot(t, pred_before_gc_stp, linewidth=1, color='black')

    plt.subplot(gs[3, 4])
    # GC STRF
    if 'gcwc' not in model1:
        wcc = _get_wc_coefficients(modelspec, idx=1)
        firc = _get_fir_coefficients(modelspec, idx=1)
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)
    else:
        wc_coefs = np.abs(wc_coefs)
        fir_coefs = np.abs(fir_coefs)

    if wc_coefs.shape[1] == fir_coefs.shape[0]:
        strf = wc_coefs @ fir_coefs
        show_factorized = True
    else:
        strf = fir_coefs
        show_factorized = False

    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
    else:
        everything = strf

    array = everything

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=clim, extent=extent)
    # End GC STRF

    plt.subplot(gs[4, 4])
    plt.plot(t, gc_stp_ctpred, linewidth=1, color='purple')

    ax = plt.subplot(gs[5:7, 4])
    # Dynamic sigmoid plot
    y_min = 0
    y_max = 0
    x = np.linspace(-1*gs_s[0], 3*gs_s[0], 1000)
    sample_every = 10
    alpha = 1.1 - 2/max(2.222222, np.log(sample_every))
    for i in range(int(len(a)/sample_every)):
        try:
            this_b = gs_b[i*sample_every]
            this_a = gs_a[i*sample_every]
            this_s = gs_s[i*sample_every]
            this_k = gs_k[i*sample_every]
            this_y1 = nonlin_fn(x, this_b, this_a, this_s, this_k)
            y_min = min(y_min, this_y1.min())
            y_max = max(y_max, this_y1.max())
            plt.plot(x, this_y1, color='gray', alpha=alpha)
        except IndexError:
            # Will happen on last attempt if array wasn't evenly divisible
            # by sample_every
            pass

    y2 = nonlin_fn(x, gs_b[0], gs_a[0], gs_s[0], gs_k[0])
    # no-stim sigmoid for reference
    plt.plot(x, y2, color='black')
    # highest-contrast sigmoid for reference
    max_idx = np.argmax(np.abs(gc_stp_ctpred - gc_stp_ctpred[0]))
    y3 = nonlin_fn(x, gs_b[max_idx], gs_a[max_idx], gs_s[max_idx],
                   gs_k[max_idx])
    plt.plot(x, y3, color='red')
    some_contrast = np.abs(gc_stp_ctpred - gc_stp_ctpred[0])\
                           /np.abs(gc_stp_ctpred[0]) > 0.02
    threshold = np.percentile(gc_stp_ctpred[some_contrast], 50)
    # Ctpred goes "up" for higher contrast
    if gc_stp_ctpred[max_idx] >= gc_stp_ctpred[0]:
        high_contrast = gc_stp_ctpred >= threshold
        low_contrast = np.logical_and(gc_stp_ctpred < threshold, some_contrast)
    # '' goes "down" for higher contrast
    else:
        high_contrast = gc_stp_ctpred <= threshold
        low_contrast = np.logical_and(gc_stp_ctpred > threshold, some_contrast)

    high_b = gs_b[high_contrast]; low_b = gs_b[low_contrast]
    high_a = gs_a[high_contrast]; low_a = gs_a[low_contrast]
    high_s = gs_s[high_contrast]; low_s = gs_s[low_contrast]
    high_k = gs_k[high_contrast]; low_k = gs_k[low_contrast]
    y4 = nonlin_fn(x, np.median(high_b), np.median(high_a),
                   np.median(high_s), np.median(high_k))
    y5 = nonlin_fn(x, np.median(low_b), np.median(low_a),
                   np.median(low_s), np.median(low_k))
    plt.plot(x, y4, color='orange')
    plt.plot(x, y5, color='blue')
#    strength = gc_magnitude(gs_base, gs_base_mod, gs_amplitude,
#                            gs_amplitude_mod, gs_shift, gs_shift_mod, gs_kappa,
#                            gs_kappa_mod)
#    if strength > 0:
#        plt.text(0.95, 0.05, "GC Strength: %.2f" % strength,
#                 ha='right', va='bottom', transform=ax.transAxes)
#    else:
#        plt.text(0.05, 0.95, "GC Strength: %.2f" % strength,
#                 ha='left', va='top', transform=ax.transAxes)
    # End nonlinearity plot

    plt.subplot(gs[7, 4])
    plt.plot(t, pred_after_gc_stp, linewidth=1, color='black')

    plt.subplot(gs[8, 4])
    change3 = pred_after_gc_stp - pred_after_LN_only
    plt.plot(t, change3, linewidth=1, color='blue')

    plt.subplot(gs[9, 4])
    plt.plot(t, resp, linewidth=1, color='green')


    # Normalize y axis across rows where appropriate
    ymin = 0
    ymax = 0
    pred_befores = [11, 20, 29, 38]
    for i, ax in enumerate(fig1.axes):
        if i in pred_befores:
            ybottom, ytop = ax.get_ylim()
            ymin = min(ymin, ybottom)
            ymax = max(ymax, ytop)
    for i, ax in enumerate(fig1.axes):
        if i in pred_befores:
            ax.set_ylim(ymin, ymax)

    ymin = 0
    ymax = 0
    gc_outputs = [22, 40]
    for i, ax in enumerate(fig1.axes):
        if i in gc_outputs:
            ybottom, ytop = ax.get_ylim()
            ymin = min(ymin, ybottom)
            ymax = max(ymax, ytop)
    for i, ax in enumerate(fig1.axes):
        if i in gc_outputs:
            ax.set_ylim(ymin, ymax)

    ymin = 0
    ymax = 0
    nonlinearities = [14, 23, 32, 41]
    for i, ax in enumerate(fig1.axes):
        if i in nonlinearities:
            ybottom, ytop = ax.get_ylim()
            ymin = min(ymin, ybottom)
            ymax = max(ymax, ytop)
    for i, ax in enumerate(fig1.axes):
        if i in nonlinearities:
            ax.set_ylim(ymin, ymax)

    ymin = 0
    ymax = 0
    pred_afters = [15, 24, 33, 42]
    pred_diffs = [16, 25, 34, 43]
    resp = [17, 26, 35, 44]
    for i, ax in enumerate(fig1.axes):
        if i in pred_afters + pred_diffs + resp:
            ybottom, ytop = ax.get_ylim()
            ymin = min(ymin, ybottom)
            ymax = max(ymax, ytop)
    for i, ax in enumerate(fig1.axes):
        if i in pred_afters + pred_diffs + resp:
            ax.set_ylim(ymin, ymax)

    # Only show x_axis on bottom row
    # Only show y_axis on right column
    for i, ax in enumerate(fig1.axes):
        if i not in resp:
            ax.axes.get_xaxis().set_visible(False)

        if not i > resp[-2]:
            ax.axes.get_yaxis().set_visible(False)
        else:
            ax.axes.get_yaxis().tick_right()

    #plt.tight_layout()
    st1.set_y(0.95)
    fig1.subplots_adjust(top=0.85)
    # End pred comparison

    return fig1