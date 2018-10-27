import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bokeh.plotting import show

from nems_db.xform_wrappers import load_model_baphy_xform
from nems_db.db import get_batch_cells, Tables, Session
import nems.xforms as xf
from nems_db.plot_helpers import plot_filtered_batch
from nems.utils import find_module
import nems.modelspec as ms
from nems_db.params import fitted_params_per_batch

gc_model = ("ozgf.fs100.ch18-ld-contrast.ms250-sev_"
            "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
            "ctwc.18x2.g-ctfir.2x15-ctlvl.1-dsig.l.k.s_"
            "init.c-basic")

gc_model_full = ("ozgf.fs100.ch18-ld-contrast.ms250-sev_"
                 "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                 "ctwc.18x2.g-ctfir.2x15-ctlvl.1-dsig.l_"
                 "init.c-basic")

gc_cont_full = ("ozgf.fs100.ch18-ld-contrast.ms100.cont.n-sev_"
                "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
                "init.c-basic")

gc_cont_reduced = ("ozgf.fs100.ch18-ld-contrast.ms100.cont.n-sev_"
                   "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                   "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l.k.s_"
                   "init.c-basic")

gc_cont_merged = ('ozgf.fs100.ch18-ld-contrast.ms100.cont.n-sev_'
                  'dlog.f-gcwc.18x1-gcfir.1x15-gclvl.1-dsig.l_'
                  'init.c-basic')

stp_model = ("ozgf.fs100.ch18-ld-sev_"
             "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-logsig_"
             "init-basic")

ln_model = ("ozgf.fs100.ch18-ld-sev_"
            "dlog.f-wc.18x2.g-fir.2x15-lvl.1-logsig_"
            "init-basic")

batch = 289

# Example cells
good_cell = 'TAR010c-13-1'
bad_cell = 'bbl086b-02-1'
gc_win1 = 'TAR017b-33-3'
gc_win2 = 'TAR017b-27-2'
gc_win3 = 'TAR010c-40-1'
stp_win1 = 'TAR010c-58-2'
stp_win2 = 'BRT033b-12-4'
ln_win = 'TAR010c-15-4'

# TODO: make loaded params DFs global as well to save time.
#       Can also maybe do this with some of the loaded xfspec, ctx tuples
#       (but not for average_r since that needs the entire batch)

# TODO: better font (or maybe easier to just edit that stuff in illustrator?


def run_all():
    performance_correlation_scatter()
    performance_bar()
    example_pred_overlay()
    average_r()  # Note: This one is *very* slow right now, hour or more
    contrast_examples()
    contrast_variables_timeseries()


# Scatter comparisons of overall model performance (similar to web ui)
# For:
# LN versus GC
# LN versus STP
# GC versus STP
def performance_scatters(model1=gc_cont_full, model2=ln_model, display=True):
    p1 = plot_filtered_batch(batch, [model1, model2], 'r_test', 'Scatter')

    if display:
        show(p1.plot)


def performance_correlation_scatter(model1=gc_cont_full, model2=stp_model,
                                    model3=ln_model):
    df1 = fitted_params_per_batch(batch, model1, stats_keys=[])
    df2 = fitted_params_per_batch(batch, model2, stats_keys=[])
    df3 = fitted_params_per_batch(batch, model3, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    nrows = len(df1.index.values.tolist())

    df1_cells = df1.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df2.loc['meta--r_test'].index.values.tolist()[5:]
    df3_cells = df3.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0
    df3_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df1[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df2[c] = nan_series
            df2_nans += 1
        if c not in df3_cells:
            df3[c] = nan_series
            df3_nans += 1

    print("# missing cells: %d, %d, %d" % (df1_nans, df2_nans, df3_nans))

    # Force same cellid order now that cols are filled in
    df1 = df1[cellids]; df2 = df2[cellids]; df3 = df3[cellids];

    gc_vs_ln = df1.loc['meta--r_test'].values - df3.loc['meta--r_test'].values
    stp_vs_ln = df2.loc['meta--r_test'].values - df3.loc['meta--r_test'].values
    gc_vs_ln = gc_vs_ln.astype('float32')
    stp_vs_ln = stp_vs_ln.astype('float32')
    ff = np.isfinite(gc_vs_ln) & np.isfinite(stp_vs_ln)
    gc_vs_ln = gc_vs_ln[ff]
    stp_vs_ln = stp_vs_ln[ff]
    r = np.corrcoef(gc_vs_ln, stp_vs_ln)[0, 1]
    n = gc_vs_ln.size

    y_max = np.max(stp_vs_ln)
    y_min = np.min(stp_vs_ln)
    x_max = np.max(gc_vs_ln)
    x_min = np.min(gc_vs_ln)

    abs_max = max(np.abs(y_max), np.abs(x_max), np.abs(y_min), np.abs(x_min))
    abs_max *= 1.15

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(gc_vs_ln, stp_vs_ln)
    plt.xlabel("GC - LN model")
    plt.ylabel("STP - LN model")
    plt.title("r: %.02f, n: %d" % (r, n))
    gca = plt.gca()
    gca.axes.axhline(0, color='black', linewidth=1, linestyle='dashed')
    gca.axes.axvline(0, color='black', linewidth=1, linestyle='dashed')
    plt.ylim(ymin=(-1)*abs_max, ymax=abs_max)
    plt.xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)


def performance_bar(model1=gc_cont_full, model2=stp_model, model3=ln_model):
    df1 = fitted_params_per_batch(batch, model1)
    df2 = fitted_params_per_batch(batch, model2)
    df3 = fitted_params_per_batch(batch, model3)

    # fill in missing cellids w/ nan
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    nrows = len(df1.index.values.tolist())

    df1_cells = df1.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df2.loc['meta--r_test'].index.values.tolist()[5:]
    df3_cells = df3.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0
    df3_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df1[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df2[c] = nan_series
            df2_nans += 1
        if c not in df3_cells:
            df3[c] = nan_series
            df3_nans += 1

    print("nans for dfs: %d, %d, %d" % (df1_nans, df2_nans, df3_nans))

    # idx 0 = mean, 1 = std, 2 = sem, 3 = max, 4 = min
    gc = df1.loc['meta--r_test'][0]
    stp = df2.loc['meta--r_test'][0]
    ln = df3.loc['meta--r_test'][0]
    largest = max(gc, stp, ln)

    gc_sem = df1.loc['meta--r_test'][2]
    stp_sem = df2.loc['meta--r_test'][2]
    ln_sem = df3.loc['meta--r_test'][2]

    if len(df1_cells) == len(df2_cells) == len(df3_cells):
        n_cells = len(df1_cells)
    else:
        print("warning: n values for different models didn't match, "
              "taking minimum")
        n_cells = min(len(df1_cells), len(df2_cells), len(df3_cells))

    plt.figure()
    plt.bar([1, 2, 3], [gc, stp, ln], color=['purple', 'green', 'gray'])
    plt.xticks([1, 2, 3], ['GC', 'STP', 'LN'])
    plt.ylim(ymax=largest*1.4)
    plt.errorbar([1, 2, 3], [gc, stp, ln], yerr=[gc_sem, stp_sem, ln_sem],
                 fmt='none', ecolor='black')
    common_kwargs = {'color': 'white', 'horizontalalignment': 'center'}
    plt.text(1, 0.2, "%0.04f" % gc, **common_kwargs)
    plt.text(2, 0.2, "%0.04f" % stp, **common_kwargs)
    plt.text(3, 0.2, "%0.04f" % ln, **common_kwargs)
    plt.title("Average Performance for GC, STP and LN models,\n"
              "n: %d" % n_cells)

# TODO: Maybe need to convert this to matplotlib? Probably easier to adjust
#       coloring on overlaps etc, bokeh hasn't been very easy to work with
#       in that respect.

# Overlay of prediction from STP versus prediction from GC for sample cell(s)
def example_pred_overlay(cellid=good_cell, model1=gc_cont_full,
                         model2=stp_model):
    xfspec1, ctx1 = load_model_baphy_xform(cellid, batch, model1)
    xfspec2, ctx2 = load_model_baphy_xform(cellid, batch, model2)
    plt.figure()
    #xf.plot_timeseries(ctx1, 'resp', cutoff=500)
    xf.plot_timeseries(ctx1, 'pred', cutoff=(200, 500))
    xf.plot_timeseries(ctx2, 'pred', cutoff=(200, 500))
    plt.legend([#'resp',
                'gc',
                'stp'])

# Some other metric ("equivalence"?) for quantifying how similar the fits are

# Average correlation for full pop. of cells?
def average_r(model1=gc_cont_full, model2=stp_model):
    # 1. query all of the relevant cell/model combos to get everything needed
    #    up to just before actually loading the model
    # - referenced _get_modelspecs in nems_db.params
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()

    # 2. actually load the models two at a time (one from stp, one from gc)
    # TODO: Computing this takes a *very* long time since we have to load
    #       every model and evaluate it to get the prediction.
    rs = []
    for i, cellid in enumerate(cellids):
        print("\n\n Starting cell # %d (out of %d)" % (i, len(cellids)))
        xfspec1, ctx1 = load_model_baphy_xform(cellid, batch, model1)
        xfspec2, ctx2 = load_model_baphy_xform(cellid, batch, model2)

    # 3. Compute the correlation for that cell
        pred1 = ctx1['val'][0]['pred'].as_continuous()
        pred2 = ctx2['val'][0]['pred'].as_continuous()

        ff = np.isfinite(pred1) & np.isfinite(pred2)
        a = (np.sum(ff) == 0)
        b = (np.sum(pred1[ff]) == 0)
        c = np.sum(pred2[ff] == 0)
        if a or b or c:
            r = 0
        else:
            cc = np.corrcoef(pred1[ff], pred2[ff])
            r = cc[0, 1]

        rs.append(r)

    # 4. Compute average once all cells processed.
    avg_r = np.nanmean(np.array(rs))
    print("average correlation between gc and stp preds: %.06f" % avg_r)
    return avg_r


# Plot of a couple example spectrogram -> contrast transformations
def contrast_examples():
    xfspec1, ctx1 = load_model_baphy_xform(good_cell, batch, gc_model)
    xfspec2, ctx2 = load_model_baphy_xform(bad_cell, batch, gc_model)

    plt.figure()
    plt.subplot(221)
    xf.plot_heatmap(ctx1, 'stim')
    plt.subplot(222)
    xf.plot_heatmap(ctx2, 'stim')
    plt.subplot(223)
    xf.plot_heatmap(ctx1, 'contrast')
    plt.subplot(224)
    xf.plot_heatmap(ctx2, 'contrast')


# Timeseries showing values of kappa & shift over time, alongside ctpred
# and final pred before & after
def contrast_variables_timeseries(cellid=good_cell, modelname=gc_cont_full):

    xfspec, ctx = load_model_baphy_xform(cellid, batch, modelname)
    val = copy.deepcopy(ctx['val'][0])
    fs = val['resp'].fs
    mspec = ctx['modelspecs'][0]
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

    k = kappa + kappa_mod*ctpred
    s = shift + shift_mod*ctpred
    b = base + base_mod*ctpred
    a = amplitude + amplitude_mod*ctpred

    xfspec2, ctx2 = load_model_baphy_xform(cellid, batch, ln_model)
    val2 = copy.deepcopy(ctx2['val'][0])
    mspec2 = ctx2['modelspecs'][0]
    logsig_idx = find_module('logistic_sigmoid', mspec2)
    #dexp_idx = find_module('double_exponential', mspec2)
    before2 = ms.evaluate(val2, mspec2, start=None, stop=logsig_idx)
    pred_before_LN = copy.deepcopy(before2['pred']).as_continuous()[0, :].T
    after2 = ms.evaluate(before2.copy(), mspec2, start=logsig_idx, stop=logsig_idx+1)
    pred_after_LN_only = after2['pred'].as_continuous()[0, :].T

    xfspec3, ctx3 = load_model_baphy_xform(cellid, batch, stp_model)
    val3 = copy.deepcopy(ctx3['val'][0])
    mspec3 = ctx3['modelspecs'][0]
    logsig_idx = find_module('logistic_sigmoid', mspec3)
    #dexp_idx = find_module('double_exponential', mspec2)
    before3 = ms.evaluate(val3, mspec3, start=None, stop=logsig_idx)
    pred_before_stp = copy.deepcopy(before3['pred']).as_continuous()[0, :].T
    after3 = ms.evaluate(before3.copy(), mspec3, start=logsig_idx, stop=logsig_idx+1)
    pred_after_stp = after3['pred'].as_continuous()[0, :].T

    mspec2 = ctx2['modelspecs'][0]
    logsig_idx = find_module('logistic_sigmoid', mspec2)
    ln_phi = mspec2[logsig_idx]['phi']
    ln_k = ln_phi['kappa']
    ln_s = ln_phi['shift']
    ln_b = ln_phi['base']
    ln_a = ln_phi['amplitude']

    # Re-align data w/o any NaN predictions and convert to real-time
    ff = np.isfinite(pred_before) & np.isfinite(pred_before_LN) \
            & np.isfinite(pred_before_stp) & np.isfinite(pred_after) \
            & np.isfinite(pred_after_LN_only) & np.isfinite(pred_after_stp)
    pred_before = pred_before[ff] * fs
    pred_before_LN = pred_before_LN[ff] * fs
    pred_before_stp = pred_before_stp[ff] * fs
    pred_after = pred_after[ff] * fs
    pred_after_LN_only = pred_after_LN_only[ff] * fs
    pred_after_stp = pred_after_stp[ff] * fs
    ctpred = ctpred[ff] * fs
    resp = resp[ff] * fs

    k = k[ff] * fs
    s = s[ff] * fs
    b = b[ff] * fs
    a = a[ff] * fs

    kp = (k - ln_k)/ln_k * 100
    sp = (s - ln_s)/ln_s * 100
    bp = (b - ln_b)/ln_b * 100
    ap = (a - ln_a)/ln_a * 100

    phi = xf.get_module(ctx2, 'logistic_sigmoid', key='fn')['phi']
    static_k = np.full_like(k, ln_k)
    static_s = np.full_like(s, ln_s)

    static_b = np.full_like(b, ln_b)
    static_a = np.full_like(a, ln_a)

    t = np.arange(len(pred_before))/fs

    fig1 = plt.figure(figsize=(6, 12))
    st1 = fig1.suptitle("Cellid: %s\nModelname: %s" % (cellid, modelname))
    gs = gridspec.GridSpec(6, 1)

    plt.subplot(gs[0, 0])
    plt.plot(t, pred_before, linewidth=1, color='black')
    plt.plot(t, pred_before_LN, linewidth=1, color='gray')
    plt.title("Prediction before Nonlinearity")

    plt.subplot(gs[1, 0])
    plt.plot(t, ctpred, linewidth=1, color='purple')
    plt.title("Output from Contrast STRF")

    plt.subplot(gs[2, 0])
    plt.plot(t, pred_after_LN_only, linewidth=1, color='black')
    plt.title("Prediction after Nonlinearity (No GC)")

    plt.subplot(gs[3, 0])
    plt.plot(t, pred_after, linewidth=1, color='purple')
    plt.title("Prediction after Nonlinearity (W/ GC)")

    plt.subplot(gs[4, 0])
    change = pred_after - pred_after_LN_only
    plt.plot(t, change, linewidth=1, color='blue')
    plt.title("Change to Prediction w/ GC")

    plt.subplot(gs[5, 0])
    plt.plot(t, resp, linewidth=1, color='green')
    plt.title("Response")


    ymin = 0
    ymax = 0
    for ax in fig1.axes[2:]:
        ybottom, ytop = ax.get_ylim()
        ymin = min(ymin, ybottom)
        ymax = max(ymax, ytop)
    for ax in fig1.axes[2:]:
        ax.set_ylim(ymin, ymax)
    for ax in fig1.axes[:-1]:
        ax.axes.get_xaxis().set_visible(False)

    plt.tight_layout()
    st1.set_y(0.95)
    fig1.subplots_adjust(top=0.85)


    fig2 = plt.figure(figsize=(7, 12))
    st2 = fig2.suptitle("Cellid: %s\nModelname: %s" % (cellid, modelname))
    gs2 = gridspec.GridSpec(7, 1)

    plt.subplot(gs2[0, 0])
    plt.plot(t, ctpred, linewidth=1, color='purple')
    plt.title("Output from Contrast STRF")

    plt.subplot(gs2[1:3, 0])
    plt.plot(t, sp, linewidth=1, color='red')
    plt.plot(t, kp, linewidth=1, color='blue')
    plt.plot(t, bp, linewidth=1, color='gray')
    plt.plot(t, ap, linewidth=1, color='orange')
    plt.title("% Change Relative to LN Model")
    plt.legend(['S', 'K', 'B', 'A'])

    plt.subplot(gs2[3, 0])
    plt.plot(t, s, linewidth=1, color='red')
    plt.plot(t, static_s, linewidth=1, linestyle='dashed', color='red')
    plt.title('Shift w/ GC vs Shift w/ LN')
    plt.legend(['GC', 'LN'])

    plt.subplot(gs2[4, 0])
    plt.plot(t, k, linewidth=1, color='blue')
    plt.plot(t, static_k, linewidth=1, linestyle='dashed', color='blue')
    plt.title('Kappa w/ GC vs Kappa w/ LN')
    plt.legend(['GC', 'LN'])

    plt.subplot(gs2[5, 0])
    plt.plot(t, b, linewidth=1, color='gray')
    plt.plot(t, static_b, linewidth=1, linestyle='dashed', color='gray')
    plt.title('Base w/ GC vs Base w/ LN')
    plt.legend(['GC', 'LN'])

    plt.subplot(gs2[6, 0])
    plt.plot(t, a, linewidth=1, color='orange')
    plt.plot(t, static_a, linewidth=1, linestyle='dashed', color='orange')
    plt.title('Amplitude w/ GC vs Amplitude w/ LN')
    plt.legend(['GC', 'LN'])

    plt.tight_layout()
    st2.set_y(0.95)
    fig2.subplots_adjust(top=0.85)
    for ax in fig2.axes[:-1]:
        ax.axes.get_xaxis().set_visible(False)


    fig3 = plt.figure(figsize=(6, 12))
    st3 = fig3.suptitle("Cellid: %s\nModelname: %s" % (cellid, modelname))
    gs3 = gridspec.GridSpec(6, 1)

    plt.subplot(gs3[0, 0])
    plt.plot(t, pred_before_stp, linewidth=1, color='black')
    plt.plot(t, pred_before_LN, linewidth=1, color='gray')
    plt.title("Prediction before Nonlinearity")

    plt.subplot(gs3[1, 0])
    plt.plot(t, np.ones_like(t))
    plt.title("Skip for now")

    plt.subplot(gs3[2, 0])
    plt.plot(t, pred_after_LN_only, linewidth=1, color='black')
    plt.title("Prediction after Nonlinearity (No STP)")

    plt.subplot(gs3[3, 0])
    plt.plot(t, pred_after_stp, linewidth=1, color='purple')
    plt.title("Prediction after Nonlinearity (W/ STP)")

    plt.subplot(gs3[4, 0])
    change2 = pred_after_stp - pred_after_LN_only
    plt.plot(t, change2, linewidth=1, color='blue')
    plt.title("Change to Prediction w/ STP")

    plt.subplot(gs3[5, 0])
    plt.plot(t, resp, linewidth=1, color='green')
    plt.title("Response")

    ymin = 0
    ymax = 0
    for ax in fig3.axes[2:]:
        ybottom, ytop = ax.get_ylim()
        ymin = min(ymin, ybottom)
        ymax = max(ymax, ytop)
    for ax in fig3.axes[2:]:
        ax.set_ylim(ymin, ymax)
    for ax in fig3.axes[:-1]:
        ax.axes.get_xaxis().set_visible(False)

    plt.tight_layout()
    st3.set_y(0.95)
    fig3.subplots_adjust(top=0.85)


# Average values for fitted contrast parameters, to compare to paper
# -- use param extraction functions
# Scatter of full model versus ".k.s" model
def mean_contrast_variables(modelname):

    df1 = fitted_params_per_batch(batch, modelname, mod_key='fn')

    amplitude_mods = df1[df1.index.str.contains('amplitude_mod')]
    base_mods = df1[df1.index.str.contains('base_mod')]
    kappa_mods = df1[df1.index.str.contains('kappa_mod')]
    shift_mods = df1[df1.index.str.contains('shift_mod')]

    avg_amp = amplitude_mods['mean'][0]
    avg_base = base_mods['mean'][0]
    avg_kappa = kappa_mods['mean'][0]
    avg_shift = shift_mods['mean'][0]

    max_amp = amplitude_mods['max'][0]
    max_base = base_mods['max'][0]
    max_kappa = kappa_mods['max'][0]
    max_shift = shift_mods['max'][0]

#    raw_amp = amplitude_mods.values[0][5:]
#    raw_base = base_mods.values[0][5:]
#    raw_kappa = kappa_mods.values[0][5:]
#    raw_shift = shift_mods.values[0][5:]

    print("Mean amplitude_mod: %.06f\n"
          "Mean base_mod: %.06f\n"
          "Mean kappa_mod: %.06f\n"
          "Mean shift_mod: %.06f\n" % (
                  avg_amp, avg_base, avg_kappa, avg_shift
                  ))

    # Better way to tell which ones are being modulated?
    # Can't really tell just from the average.
    print("ratio of max: %.06f, %.06f, %.06f, %.06f" % (
            avg_amp/max_amp, avg_base/max_base,
            avg_kappa/max_kappa, avg_shift/max_shift))


def continuous_contrast_improvements():
    df_full = fitted_params_per_batch(batch, gc_model_full, stats_keys=[])
    df_cont = fitted_params_per_batch(batch, gc_model_cont, stats_keys=[])
    df_stp = fitted_params_per_batch(batch, stp_model, stats_keys=[])
    df_ln = fitted_params_per_batch(batch, ln_model, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    nrows = len(df_full.index.values.tolist())

    df1_cells = df_full.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df_cont.loc['meta--r_test'].index.values.tolist()[5:]
    df3_cells = df_ln.loc['meta--r_test'].index.values.tolist()[5:]
    df4_cells = df_stp.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0
    df3_nans = 0
    df4_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df_full[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df_cont[c] = nan_series
            df2_nans += 1
        if c not in df3_cells:
            df_ln[c] = nan_series
            df3_nans += 1
        if c not in df4_cells:
            df_stp[c] = nan_series
            df4_nans += 1

    print("# missing cells: %d, %d, %d, %d" % (df1_nans, df2_nans, df3_nans,
                                               df4_nans))

    # Force same cellid order now that cols are filled in
    df_full = df_full[cellids]
    df_cont = df_cont[cellids]
    df_ln = df_ln[cellids]
    df_stp = df_stp[cellids]

    # Only look at cells that did better than linear for binary model
    full_vs_ln = df_full.loc['meta--r_test'].values - \
            df_ln.loc['meta--r_test'].values
    cont_vs_ln = df_cont.loc['meta--r_test'].values - \
            df_ln.loc['meta--r_test'].values
    full_vs_ln = full_vs_ln.astype('float32')
    cont_vs_ln = cont_vs_ln.astype('float32')

    better = full_vs_ln > 0
    #full_vs_ln = full_vs_ln[better]
    #cont_vs_ln = cont_vs_ln[better]

    # which cells got further improvement by keeping contrast continuous?
    cont_improve = (cont_vs_ln - full_vs_ln) > 0
    cont_vs_full = cont_vs_ln[cont_improve]

    # Keep indices so can extract cellid names
    cont_improve = (cont_vs_ln - full_vs_ln) > 0
    cont_better = np.logical_and(better, cont_improve)

    cont_cells = celldata['cellid'][cont_better].tolist()
    full_cells = celldata['cellid'][np.logical_not(cont_better)].tolist()

    df_full = df_full[full_cells]
    df_cont = df_cont[cont_cells]
    df_stp_full = df_stp[full_cells]
    df_stp_cont = df_stp[cont_cells]
    df_ln_full = df_ln[full_cells]
    df_ln_cont = df_ln[cont_cells]

    df_full_r = (df_full.loc['meta--r_test'].values
                 - df_ln_full.loc['meta--r_test']).astype('float32')
    df_cont_r = (df_cont.loc['meta--r_test'].values
                 - df_ln_cont.loc['meta--r_test']).astype('float32')
    df_stp_full_r = (df_stp_full.loc['meta--r_test'].values
                     - df_ln_full.loc['meta--r_test']).astype('float32')
    df_stp_cont_r = (df_stp_cont.loc['meta--r_test'].values
                     - df_ln_cont.loc['meta--r_test']).astype('float32')

    ff = np.isfinite(df_full_r) & np.isfinite(df_stp_full_r)
    df_full_r = df_full_r[ff]
    df_stp_full_r = df_stp_full_r[ff]
    ff = np.isfinite(df_cont_r) & np.isfinite(df_stp_cont_r)
    df_cont_r = df_cont_r[ff]
    df_stp_cont_r = df_stp_cont_r[ff]

    r1 = np.corrcoef(df_full_r, df_stp_full_r)[0, 1]
    r2 = np.corrcoef(df_cont_r, df_stp_cont_r)[0, 1]

    fig = plt.figure()
    plt.scatter(df_full_r, df_stp_full_r)
    plt.title('full, r: %.04f'%r1)
    adjustFigAspect(fig, aspect=1)

    fig = plt.figure()
    plt.scatter(df_cont_r, df_stp_cont_r)
    plt.title('cont, r: %.04f'%r2)
    adjustFigAspect(fig, aspect=1)


def gd_ratio(cellid=good_cell, modelname=gc_cont_full):

    xfspec, ctx = load_model_baphy_xform(cellid, batch, modelname)
    mspec = ctx['modelspecs'][0]
    dsig_idx = find_module('dynamic_sigmoid', mspec)
    phi = mspec[dsig_idx]['phi']

    return phi['kappa_mod']/phi['kappa']


def gd_scatter(batch=289, model1=gc_cont_full, model2=ln_model):

    df1 = fitted_params_per_batch(batch, model1, stats_keys=[])
    df2 = fitted_params_per_batch(batch, model2, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    nrows = len(df1.index.values.tolist())

    df1_cells = df1.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df2.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df1[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df2[c] = nan_series
            df2_nans += 1

    print("# missing cells: %d, %d" % (df1_nans, df2_nans))

    # Force same cellid order now that cols are filled in
    df1 = df1[cellids]; df2 = df2[cellids];

    gc_vs_ln = df1.loc['meta--r_test'].values - df2.loc['meta--r_test'].values
    gc_vs_ln = gc_vs_ln.astype('float32')

    kappa_mod = df1[df1.index.str.contains('kappa_mod')]
    kappa = df1[df1.index.str.contains('kappa$')]
    gd_ratio = (kappa_mod.values / kappa.values).astype('float32').flatten()

    ff = np.isfinite(gc_vs_ln) & np.isfinite(gd_ratio)
    gc_vs_ln = gc_vs_ln[ff]
    gd_ratio = gd_ratio[ff]

    r = np.corrcoef(gc_vs_ln, gd_ratio)[0, 1]
    n = gc_vs_ln.size

    y_max = np.max(gd_ratio)
    y_min = np.min(gd_ratio)
    x_max = np.max(gc_vs_ln)
    x_min = np.min(gc_vs_ln)

    abs_max = max(np.abs(y_max), np.abs(x_max), np.abs(y_min), np.abs(x_min))
    abs_max *= 1.15

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(gc_vs_ln, gd_ratio)
    plt.xlabel("GC - LN model")
    plt.ylabel("Gd ratio")
    plt.title("r: %.02f, n: %d" % (r, n))
    gca = plt.gca()
    gca.axes.axhline(0, color='black', linewidth=1, linestyle='dashed')
    gca.axes.axvline(0, color='black', linewidth=1, linestyle='dashed')
    plt.ylim(ymin=(-1)*abs_max, ymax=abs_max)
    plt.xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)


def get_valid_improvements(batch=289, model1=gc_cont_full, model2=ln_model,
                           threshold = 1.75):

    df1 = fitted_params_per_batch(batch, model1, stats_keys=[])
    df2 = fitted_params_per_batch(batch, model2, stats_keys=[])

    # fill in missing cellids w/ nan
    celldata = get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    nrows = len(df1.index.values.tolist())

    df1_cells = df1.loc['meta--r_test'].index.values.tolist()[5:]
    df2_cells = df2.loc['meta--r_test'].index.values.tolist()[5:]

    nan_series = pd.Series(np.full((nrows), np.nan))

    df1_nans = 0
    df2_nans = 0

    for c in cellids:
        if c not in df1_cells:
            df1[c] = nan_series
            df1_nans += 1
        if c not in df2_cells:
            df2[c] = nan_series
            df2_nans += 1

    print("# missing cells: %d, %d" % (df1_nans, df2_nans))

    # Force same cellid order now that cols are filled in
    df1 = df1[cellids]; df2 = df2[cellids];
    ratio = df1.loc['meta--r_test'] / df2.loc['meta--r_test']

    valid_improvements = ratio.loc[ratio < threshold].loc[ratio > 1/threshold]

    return valid_improvements.index.values.tolist()


def make_batch_from_subset(cellids, source_batch=289, new_batch=311):
    raise NotImplementedError("WIP, have to do more than just add to"
                              "NarfBatches apparently.")
    session = Session()
    NarfBatches = Tables()['NarfBatches']

    full_batch = (
            session.query(NarfBatches)
            .filter(NarfBatches.batch == source_batch)
            .all()
            )

    subset = [row for row in full_batch if row.cellid in cellids]
    new = [NarfBatches(batch=new_batch, cellid=row.cellid,
                       est_reps=row.est_reps, est_set=row.est_set,
                       est_snr=row.est_snr, filecodes=row.filecodes,
                       id=None, lastmod=row.lastmod,
                       min_isolation=row.min_isolation,
                       min_snr_index=row.min_snr_index, val_reps=row.val_reps,
                       val_set=row.val_set, val_snr=row.val_snr)
           for row in subset]

    [session.add(row) for row in new]

    session.commit()
    session.close()

# TODO: have to add to narfdata also
    # and maybe need to do something with rawid?

# Paragraphs describing the model setup and equations:
# - equations for dlog and linear strf
# - equations for logsig and dexp
# - equation/description for dynamic sigmoid




# Explanation of preprocessing steps:
# - stim loader - need details from SVD about this.
# - how was contrast signal calculated? (different options used?)
# - est/val split and trial averaging
# - where do the spikes get translated into psth? was it in avg step?
# - initialization and fitting process
# - correlation computation


# Copied from:
# https://stackoverflow.com/questions/7965743/
# how-can-i-set-the-aspect-ratio-in-matplotlib
def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)
