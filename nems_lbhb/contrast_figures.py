import copy
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
params = {'pdf.fonttype': 42,
         'ps.fonttype': 42}
plt.rcParams.update(params)
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatch

import nems.db as nd
from nems.db import get_batch_cells, Tables, Session
import nems.xforms as xf
import nems.xform_helper as xhelp
from nems.utils import find_module
import nems.modelspec as ms
from nems_db.params import fitted_params_per_batch
from nems_lbhb.contrast_helpers import (make_contrast_signal, rec_from_DRC,
                                        gc_magnitude)
from nems.metrics.stp import stp_magnitude
from nems.modules.nonlinearity import _logistic_sigmoid, _double_exponential
from nems.plots.heatmap import _get_wc_coefficients, _get_fir_coefficients


gc_cont_full = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
                "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
                "init.c-basic")

gc_cont_reduced = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
                   "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                   "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l.k.s_"
                   "init.c-basic")

gc_cont_dexp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
                "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
                "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.d_"
                "init.c-basic")

gc_cont_b3 = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n.b3-sev_"
              "dlog.f-wc.18x2.g-fir.2x15-lvl.1-"
              "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
              "init.c-basic")

gc_stp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n.b3-sev_"
          "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-"
          "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_"
          "init.c-basic")

gc_stp_dexp = ("ozgf.fs100.ch18-ld-contrast.ms70.cont.n-sev_"
               "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-"
               "ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.d_"
               "init.c-basic")

gc_cont_merged = ('ozgf.fs100.ch18-ld-contrast.ms100.cont.n-sev_'
                  'dlog.f-gcwc.18x1-gcfir.1x15-gclvl.1-dsig.l_'
                  'init.c-basic')

stp_model = ("ozgf.fs100.ch18-ld-sev_"
             "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-logsig_"
             "init-basic")

stp_dexp =  ("ozgf.fs100.ch18-ld-sev_"
             "dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-dexp.1_"
             "init-basic")

ln_model = ("ozgf.fs100.ch18-ld-sev_"
            "dlog.f-wc.18x2.g-fir.2x15-lvl.1-logsig_"
            "init-basic")

ln_dexp = ("ozgf.fs100.ch18-ld-sev_"
           "dlog.f-wc.18x2.g-fir.2x15-lvl.1-dexp.1_"
           "init-basic")

dexp_kwargs = {'model1': gc_cont_dexp, 'model2': stp_dexp, 'model3': ln_dexp,
               'model4': gc_stp_dexp}

batch = 289

# Example cells
good_cell = 'TAR010c-13-1'
bad_cell = 'bbl086b-02-1'
gc_win1 = 'TAR017b-33-3'
gc_win2 = 'TAR017b-27-2'
gc_win3 = 'TAR010c-40-1'
gc_win4 = 'eno052b-b1'
gc_win5 = 'bbl104h-12-1'
stp_win1 = 'TAR010c-58-2'
stp_win2 = 'BRT033b-12-4'
gc_stp_both_win = 'TAR010c-21-4'
ln_win = 'TAR010c-15-4'
gc_sharp_onset = 'bbl104h-10-2'
gc_beat_stp = 'TAR009d-28-1'
weird = 'TAR009d-22-1'

# Interesting example cells to look at in more detail:

# STP does better, GC about equal to LN
#cellid = 'bbl104h-33-1'

# Same as previous but much bigger difference in performance
#cellid = 'BRT026c-16-2'

# Reverse again: GC better, STP about equal to LN
# GC + STP also looks like it tracks resp better, even though R is slightly lower
#cellid = 'TAR009d-22-1'

# Seems like GC and STP are improving in a similar(?) way, but
# GC + STP does even better.
#cellid = 'TAR010c-13-1'

# Similar example to previous? STRFS very different
#cellid = 'TAR010c-20-1'

# Weird failure that responds to offsets between stims
#cellid = 'TAR010c-58-2'

# Long depression, STP does well but GC and GC+STP do worse
#cellid = 'TAR017b-04-1'

# Bit noisy but similar performance boosts for all 3
#cellid = 'TAR017b-22-1'

# Another case with facilitation where STP doesn't help but
# GC and GC+STP do.
#cellid = 'gus018b-a2'

# Another case where GC and STP each help a little, but GC+STP helps a lot
# Maybe implies a case where the two individual models are capturing
# mostly independent rather than shared info?
#cellid = 'gus019c-a2'

# Both improving in a somehwat similar way?
#cellid = 'TAR009d-15-1'


gc_color = '#69657C'
stp_color = '#394B5E'
ln_color = '#62838C'
gc_stp_color = '#215454'


# TODO: make loaded params DFs global as well to save time.
#       Can also maybe do this with some of the loaded xfspec, ctx tuples
#       (but not for average_r since that needs the entire batch)

# TODO: better font (or maybe easier to just edit that stuff in illustrator?


def run_all(model1=gc_cont_full, model2=stp_model, model3=ln_model,
            model4=gc_stp, cellid=gc_beat_stp, se_filter=True, sample_every=5,
            save=False):
    save_directory = ("/auto/users/jacob/notes/gc_figures/matplot_figs/"
                      "dexp_pdfs/")
    f1 = performance_scatters(model1=model1, model2=model2, model3=model3,
                              se_filter=se_filter)
    f2 = performance_correlation_scatter(model1=model1, model2=model2,
                                         model3=model3, se_filter=se_filter)

    f3 = performance_bar(model1=model1, model2=model2, model3=model3,
                         model4=model4, se_filter=se_filter)
    f4 = significance(model1=model1, model2=model2, model3=model3, model4=model4,
                      se_filter=se_filter)
    #example_pred_overlay()
    #average_r()  # Note: This one is *very* slow right now, hour or more
    #contrast_examples()
    f5 = contrast_breakdown(model1=model1, model2=model2, model3=model3,
                            cellid=cellid, sample_every=sample_every)
    f6 = contrast_vs_stp_comparison(model1=model1, model2=model2, model3=model3,
                                    model4=model4, cellid=cellid)

    if save:
        f1.savefig(save_directory + 'performance_scatters' + '.pdf')
        f2.savefig(save_directory + 'correlation_scatter' + '.pdf')
        f3.savefig(save_directory + 'summary_bar' + '.pdf')
        f4.savefig(save_directory + 'significance' + '.pdf')
        f5.savefig(save_directory + 'gc_schematic' + '.pdf')
        f6.savefig(save_directory + 'gc_vs_stp_comparison' + '.pdf')
        plt.close('all')

def example_cells(model1=gc_cont_full, model2=stp_model, model3=ln_model,
                  model4=gc_stp, run_id=0):
    example_cells = ['bbl104h-33-1', 'BRT026c-16-2', 'TAR009d-22-1',
                     'TAR010c-13-1', 'TAR010c-20-1', 'TAR010c-58-2',
                     'TAR017b-04-1', 'TAR017b-22-1', 'gus018b-a2',
                     'gus019c-a2', 'TAR009d-15-1']
    save_directory = ("/auto/users/jacob/notes/gc_figures/matplot_figs/"
                      "example_cells/run%d/" % run_id)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for c in example_cells:
        f = contrast_vs_stp_comparison(cellid=c, model1=model1, model2=model2,
                                       model3=model3, model4=model4)
        f.savefig(save_directory + c + '.pdf')
        f.savefig(save_directory + c + '.png')

        plt.close('all')


# Scatter comparisons of overall model performance (similar to web ui)
# For:
# LN versus GC
# LN versus STP
# GC versus STP
def performance_scatters(model1=gc_cont_full, model2=stp_model, model3=ln_model,
                         model4=gc_stp,  se_filter=False, ratio_filter=False,
                         threshold=2.5, manual_cellids=None):

    df_r = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    if se_filter:
        gc_test = df_r[model1]
        gc_se = df_e[model1]
        stp_test = df_r[model2]
        stp_se = df_e[model2]
        ln_test = df_r[model3]
        ln_se = df_e[model3]
        gc_stp_test = df_r[model4]
        gc_stp_se = df_e[model4]

        # Also remove if performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2) & (gc_stp_test > gc_stp_se*2))

        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se) |
                     (gc_stp_test+gc_stp_se < ln_test-ln_se))

        keep = good_cells & ~bad_cells

        cellids = df_r[keep].index.values.tolist()
        under_chance = df_r[~good_cells].index.values.tolist()
        less_LN = df_r[bad_cells].index.values.tolist()

    if ratio_filter:
        # Ex: for threshold = 2.5
        # Only use cellids where performance for gc/stp was within 2.5x
        # of LN performance (or where LN within 2.5x of gc/stp) to filter
        # outliers.
        c1 = get_valid_improvements(model1=model1, threshold=threshold)
        c2 = get_valid_improvements(model1=model2, threshold=threshold)
        cellids = list(set(c1) & set(c2) & set(cellids))

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids

    if not se_filter:
        under_chance = np.array([True]*len(df_r[model1]))
        less_LN = copy.deepcopy(under_chance)

    n_cells = len(cellids)
    n_under_chance = len(under_chance) if under_chance != cellids else 0
    n_less_LN = len(less_LN) if less_LN != cellids else 0

    gc_test = df_r[model1][cellids]
    gc_test_under_chance = df_r[model1][under_chance]
    gc_test_less_LN = df_r[model1][less_LN]

    stp_test = df_r[model2][cellids]
    stp_test_under_chance = df_r[model2][under_chance]
    stp_test_less_LN = df_r[model2][less_LN]

    ln_test = df_r[model3][cellids]
    ln_test_under_chance = df_r[model3][under_chance]
    ln_test_less_LN = df_r[model3][less_LN]

    gc_stp_test = df_r[model4][cellids]
    gc_stp_test_under_chance = df_r[model4][under_chance]
    gc_stp_test_less_LN = df_r[model4][less_LN]

    fig, axes = plt.subplots(2, 3)

    # Row 1 (vs LN)
    ax = axes[0][0]
    ax.scatter(gc_test, ln_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_test_under_chance, ln_test_under_chance, c='red', s=1)
    ax.scatter(gc_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('GC vs LN')
    ax.set_xlabel('GC')
    ax.set_ylabel('LN')
    ax.text(0.90, 0.00, 'n = %d' % n_cells, ha='right', va='bottom')
    ax.text(0.90, 0.10, 'uc = %d' % n_under_chance, ha='right', va='bottom',
            color='red')
    ax.text(0.90, 0.20, '<ln = %d' % n_less_LN, ha='right', va='bottom',
            color='blue')

    ax = axes[0][1]
    ax.scatter(stp_test, ln_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(stp_test_under_chance, ln_test_under_chance, c='red', s=1)
    ax.scatter(stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('STP vs LN')
    ax.set_xlabel('STP')
    ax.set_ylabel('LN')

    ax = axes[0][2]
    ax.scatter(gc_stp_test, ln_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_stp_test_under_chance, ln_test_under_chance, c='red', s=1)
    ax.scatter(gc_stp_test_less_LN, ln_test_less_LN, c='blue', s=1)
    ax.set_title('GC + STP vs LN')
    ax.set_xlabel('GC + STP')
    ax.set_ylabel('LN')

    # Row 2 (head-to-head)
    ax = axes[1][0]
    ax.scatter(gc_test, stp_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_test_under_chance, stp_test_under_chance, c='red', s=1)
    ax.scatter(gc_test_less_LN, stp_test_less_LN, c='blue', s=1)
    ax.set_title('GC vs STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('STP')

    ax = axes[1][1]
    ax.scatter(gc_test, gc_stp_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(gc_test_under_chance, gc_stp_test_under_chance, c='red', s=1)
    ax.scatter(gc_test_less_LN, gc_stp_test_less_LN, c='blue', s=1)
    ax.set_title('GC vs GC + STP')
    ax.set_xlabel('GC')
    ax.set_ylabel('GC + STP')

    ax = axes[1][2]
    ax.scatter(stp_test, gc_stp_test, c='black', s=1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', linewidth=0.5)
    ax.scatter(stp_test_under_chance, gc_stp_test_under_chance, c='red', s=1)
    ax.scatter(stp_test_less_LN, gc_stp_test_less_LN, c='blue', s=1)
    ax.set_title('STP vs GC + STP')
    ax.set_xlabel('STP')
    ax.set_ylabel('GC + STP')

    plt.tight_layout()

    return fig


def performance_correlation_scatter(model1=gc_cont_full, model2=stp_model,
                                    model3=ln_model, se_filter=False,
                                    ratio_filter=False, threshold=2.5,
                                    manual_cellids=None):

    df_r = nd.batch_comp(batch, [model1, model2, model3], stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3], stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    if se_filter:
        gc_test = df_r[model1]
        gc_se = df_e[model1]
        stp_test = df_r[model2]
        stp_se = df_e[model2]
        ln_test = df_r[model3]
        ln_se = df_e[model3]

        # Also remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                      (ln_test > ln_se*2))

        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se))

        keep = good_cells & ~bad_cells
        cellids = df_r[keep].index.values.tolist()

    if ratio_filter:
        # Ex: for threshold = 2.5
        # Only use cellids where performance for gc/stp was within 2.5x
        # of LN performance (or where LN within 2.5x of gc/stp) to filter
        # outliers.
        c1 = get_valid_improvements(model1=model1, threshold=threshold)
        c2 = get_valid_improvements(model1=model2, threshold=threshold)
        cellids = list(set(c1) & set(c2) & set(cellids))

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids

    gc_test = df_r[model1][cellids]
    stp_test = df_r[model2][cellids]
    ln_test = df_r[model3][cellids]

    gc_vs_ln = gc_test.values - ln_test.values
    stp_vs_ln = stp_test.values - ln_test.values
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
    plt.scatter(gc_vs_ln, stp_vs_ln, c='black', s=1)
    plt.xlabel("GC - LN model")
    plt.ylabel("STP - LN model")
    plt.title("Performance Improvements over LN\nr: %.02f, n: %d" % (r, n))
    gca = plt.gca()
    gca.axes.axhline(0, color='black', linewidth=1, linestyle='dashed')
    gca.axes.axvline(0, color='black', linewidth=1, linestyle='dashed')
    plt.ylim(ymin=(-1)*abs_max, ymax=abs_max)
    plt.xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)

    return fig


def equivalence_histogram(batch=289, model1=gc_cont_full, model2=stp_model,
                          model3=ln_model, se_filter=False, test_limit=None):

    df_r = nd.batch_comp(batch, [model1, model2, model3], stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3], stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    if se_filter:
        gc_test = df_r[model1]
        gc_se = df_e[model1]
        stp_test = df_r[model2]
        stp_se = df_e[model2]
        ln_test = df_r[model3]
        ln_se = df_e[model3]

        # Also remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2))

        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se))

        keep = good_cells & ~bad_cells

        cellids = df_r[keep].index.values.tolist()

    rs = []
    for c in cellids[:test_limit]:
        xf1, ctx1 = xhelp.load_model_xform(c, batch, model1)
        xf2, ctx2 = xhelp.load_model_xform(c, batch, model2)
        xf3, ctx3 = xhelp.load_model_xform(c, batch, model3)

        gc = ctx1['val'].apply_mask()['pred'].as_continuous()
        stp = ctx2['val'].apply_mask()['pred'].as_continuous()
        ln = ctx3['val'].apply_mask()['pred'].as_continuous()

        ff = np.isfinite(gc) & np.isfinite(stp) & np.isfinite(ln)
        rs.append(np.corrcoef(gc[ff]-ln[ff], stp[ff]-ln[ff])[0, 1])

    rs = np.array(rs)
    md = np.nanmedian(rs)

#    onetwo = np.percentile(rs, 20)
#    twothree = np.percentile(rs, 40)
#    threefour = np.percentile(rs, 60)
#    fourfive = np.percentile(rs, 80)
#    first = rs <= onetwo
#    second = rs <= twothree
#    third = rs <= threefour
#    fourth = rs <= fourfive
#    fifth = rs > fourfive

    n_cells = len(cellids)
    fig = plt.figure(figsize=(6, 6))
    plt.hist(rs, bins=30, range=[-0.5, 1], histtype='bar', color=['gray'])

#    plt.hist(rs[first], bins=30, range=[-0.5, 1], histtype='bar',
#             color=['#2A4738'])
#    plt.hist(rs[second], bins=30, range=[-0.5, 1], histtype='bar',
#             color=['#345945'])
#    plt.hist(rs[third], bins=30, range=[-0.5, 1], histtype='bar',
#             color=['#4A7F65'])
#    plt.hist(rs[fourth], bins=30, range=[-0.5, 1], histtype='bar',
#             color=['#26615C'])
#    plt.hist(rs[fifth], bins=30, range=[-0.5, 1], histtype='bar',
#             color=['#113842'])

    plt.plot(np.array([0,0]), np.array(fig.axes[0].get_ylim()), 'k--')
#    plt.plot(np.array([onetwo, onetwo]), np.array(fig.axes[0].get_ylim()),
#             'g--')
#    plt.plot(np.array([fourfive, fourfive]), np.array(fig.axes[0].get_ylim()),
#             'g--')
    plt.text(0.05, 0.95, 'n = %d\nmd = %.2f' % (n_cells, md),
             ha='left', va='top', transform=fig.axes[0].transAxes)
    plt.xlabel('CC, GC-LN vs STP-LN')
    plt.title('Equivalence of Change in Prediction Relative to LN Model')

    return fig


def performance_bar(model1=gc_cont_full, model2=stp_model, model3=ln_model,
                    model4=gc_stp,  se_filter=False, ratio_filter=False,
                    threshold=2.5, manual_cellids=None):

    df_r = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    if se_filter:
        gc_test = df_r[model1]
        gc_se = df_e[model1]
        stp_test = df_r[model2]
        stp_se = df_e[model2]
        ln_test = df_r[model3]
        ln_se = df_e[model3]
        gc_stp_test = df_r[model4]
        gc_stp_se = df_e[model4]

        # Also remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2) & (gc_stp_test > gc_stp_se*2))

        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se) |
                     (gc_stp_test+gc_stp_se < ln_test-ln_se))

        keep = good_cells & ~bad_cells

        cellids = df_r[keep].index.values.tolist()

    if ratio_filter:
        # Ex: for threshold = 2.5
        # Only use cellids where performance for gc/stp was within 2.5x
        # of LN performance (or where LN within 2.5x of gc/stp) to filter
        # outliers.
        c1 = get_valid_improvements(model1=model1, threshold=threshold)
        c2 = get_valid_improvements(model1=model2, threshold=threshold)
        cellids = list(set(c1) & set(c2) & set(cellids))

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids

    n_cells = len(cellids)
    gc_test = df_r[model1][cellids]
    gc_se = df_e[model1][cellids]
    stp_test = df_r[model2][cellids]
    stp_se = df_e[model2][cellids]
    ln_test = df_r[model3][cellids]
    ln_se = df_e[model3][cellids]
    gc_stp_test = df_r[model4][cellids]
    gc_stp_se = df_e[model4][cellids]

    gc = np.median(gc_test.values)
    stp = np.median(stp_test.values)
    ln = np.median(ln_test.values)
    gc_stp = np.median(gc_stp_test.values)
    largest = max(gc, stp, ln, gc_stp)

    # TODO: double check that this is valid, to just take mean of errors
    gc_sem = np.median(gc_se.values)
    stp_sem = np.median(stp_se.values)
    ln_sem = np.median(ln_se.values)
    gc_stp_sem = np.median(gc_stp_se.values)

    fig = plt.figure()
    plt.bar([1, 2, 3, 4], [gc, stp, ln, gc_stp],
            #color=['purple', 'green', 'gray', 'blue'])
            color=[gc_color, stp_color, ln_color, gc_stp_color])
    plt.xticks([1, 2, 3, 4], ['GC', 'STP', 'LN', 'GC + STP'])
    plt.ylim(ymax=largest*1.4)
    plt.errorbar([1, 2, 3, 4], [gc, stp, ln, gc_stp], yerr=[gc_sem, stp_sem,
                 ln_sem, gc_stp_sem], fmt='none', ecolor='black')
    common_kwargs = {'color': 'white', 'horizontalalignment': 'center'}
    plt.text(1, 0.2, "%0.04f" % gc, **common_kwargs)
    plt.text(2, 0.2, "%0.04f" % stp, **common_kwargs)
    plt.text(3, 0.2, "%0.04f" % ln, **common_kwargs)
    plt.text(4, 0.2, "%0.04f" % gc_stp, **common_kwargs)
    plt.title("Median Performance for GC, STP, LN, and GC + STP models,\n"
              "n: %d" % n_cells)

    return fig


def significance(model1=gc_cont_full, model2=stp_model, model3=ln_model,
                 model4=gc_stp,  se_filter=False, ratio_filter=False,
                 threshold=2.5, manual_cellids=None):

    df_r = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='r_ceiling')
    df_e = nd.batch_comp(batch, [model1, model2, model3, model4],
                         stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    if se_filter:
        gc_test = df_r[model1]
        gc_se = df_e[model1]
        stp_test = df_r[model2]
        stp_se = df_e[model2]
        ln_test = df_r[model3]
        ln_se = df_e[model3]
        gc_stp_test = df_r[model4]
        gc_stp_se = df_e[model4]

        # Also remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2) & (gc_stp_test > gc_stp_se*2))

        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se) |
                     (gc_stp_test+gc_stp_se < ln_test-ln_se))

        keep = good_cells & ~bad_cells

        cellids = df_r[keep].index.values.tolist()

    if ratio_filter:
        # Ex: for threshold = 2.5
        # Only use cellids where performance for gc/stp was within 2.5x
        # of LN performance (or where LN within 2.5x of gc/stp) to filter
        # outliers.
        c1 = get_valid_improvements(model1=model1, threshold=threshold)
        c2 = get_valid_improvements(model1=model2, threshold=threshold)
        cellids = list(set(c1) & set(c2) & set(cellids))

    if manual_cellids is not None:
        # WARNING: Will override se and ratio filters even if they are set
        cellids = manual_cellids

    gc_test = df_r[model1][cellids]
    stp_test = df_r[model2][cellids]
    ln_test = df_r[model3][cellids]
    gc_stp_test = df_r[model4][cellids]

    modelnames = ['GC', 'STP', 'LN', 'GC + STP']
    models = {'GC': gc_test, 'STP': stp_test, 'LN': ln_test,
              'GC + STP': gc_stp_test}
    array = np.ndarray(shape=(len(modelnames), len(modelnames)), dtype=float)

    for i, m_one in enumerate(modelnames):
        for j, m_two in enumerate(modelnames):
            # get series of values corresponding to selected measure
            # for each model
            series_one = models[m_one]
            series_two = models[m_two]
            if j == i:
                # if indices equal, on diagonal so no comparison
                array[i][j] = 0.00
            elif j > i:
                # if j is larger, below diagonal so get mean difference
                mean_one = np.mean(series_one)
                mean_two = np.mean(series_two)
                array[i][j] = abs(mean_one - mean_two)
            else:
                # if j is smaller, above diagonal so run t-test and
                # get p-value
                first = series_one.tolist()
                second = series_two.tolist()
                array[i][j] = st.wilcoxon(first, second)[1]

    xticks = range(len(modelnames))
    yticks = xticks
    minor_xticks = np.arange(-0.5, len(modelnames), 1)
    minor_yticks = np.arange(-0.5, len(modelnames), 1)

    fig = plt.figure(figsize=(len(modelnames),len(modelnames)))
    ax = plt.gca()

    # ripped from stackoverflow. adds text labels to the grid
    # at positions i,j (model x model)  with text z (value of array at i, j)
    for (i, j), z in np.ndenumerate(array):
        if j == i:
            color="#EBEBEB"
        elif j > i:
            color="#368DFF"
        else:
            if array[i][j] < 0.001:
                color="#74E572"
            elif array[i][j] < 0.01:
                color="#59AF57"
            elif array[i][j] < 0.05:
                color="#397038"
            else:
                color="#ABABAB"

        ax.add_patch(mpatch.Rectangle(
                xy=(j-0.5, i-0.5), width=1.0, height=1.0, angle=0.0,
                facecolor=color, edgecolor='black',
                ))
        if j == i:
            # don't draw text for diagonal
            continue
        formatting = '{:.04f}'
        if z <= 0.0001:
            formatting = '{:.2E}'
        ax.text(
                j, i, formatting.format(z), ha='center', va='center',
                )

    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticks(yticks)
    ax.set_yticklabels(modelnames, fontsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(modelnames, fontsize=10, rotation="vertical")
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(minor_xticks, minor=True)
    ax.grid(b=False)
    ax.grid(which='minor', color='b', linestyle='-', linewidth=0.75)
    ax.set_title("Wilcoxon Signed Test", ha='center', fontsize = 14)

    blue_patch = mpatch.Patch(
            color='#368DFF', label='Mean Difference', edgecolor='black'
            )
    p001_patch = mpatch.Patch(
            color='#74E572', label='P < 0.001', edgecolor='black'
            )
    p01_patch = mpatch.Patch(
            color='#59AF57', label='P < 0.01', edgecolor='black'
            )
    p05_patch = mpatch.Patch(
            color='#397038', label='P < 0.05', edgecolor='black'
            )
    nonsig_patch = mpatch.Patch(
            color='#ABABAB', label='Not Significant', edgecolor='black',
            )

    plt.legend(
            #bbox_to_anchor=(0., 1.02, 1., .102), ncol=2,
            bbox_to_anchor=(1.05, 1), ncol=1,
            loc=2, handles=[
                    p05_patch, p01_patch, p001_patch,
                    nonsig_patch, blue_patch,
                    ]
            )
    plt.tight_layout()

    return fig


def contrast_breakdown(cellid=gc_beat_stp, model1=gc_cont_full,
                       model2=stp_model, model3=ln_model, sample_every=5):

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

    wcc = _get_wc_coefficients(modelspec, idx=1)
    firc = _get_fir_coefficients(modelspec, idx=1)
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


def contrast_vs_stp_comparison(cellid=good_cell, model1=gc_cont_full,
                               model2=stp_model, model3=ln_model,
                               model4=gc_stp):

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
    wcc = _get_wc_coefficients(modelspec, idx=1)
    firc = _get_fir_coefficients(modelspec, idx=1)
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
    wcc = _get_wc_coefficients(modelspec, idx=1)
    firc = _get_fir_coefficients(modelspec, idx=1)
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

def test_DRC_with_contrast(ms=200, normalize=True, fs=100, bands=1,
                           percentile=50, n_segments=12):
    '''
    Plot a sample DRC stimulus next to assigned contrast
    and calculated contrast.
    '''
    drc = rec_from_DRC(fs=fs, n_segments=n_segments)
    rec = make_contrast_signal(drc, name='binary', continuous=False, ms=ms,
                                percentile=percentile, bands=bands)
    rec = make_contrast_signal(rec, name='continuous', continuous=True, ms=ms,
                                bands=bands, normalize=normalize)
    s = rec['stim'].as_continuous()
    c1 = rec['contrast'].as_continuous()
    c2 = rec['binary'].as_continuous()
    c3 = rec['continuous'].as_continuous()

    fig, axes = plt.subplots(4, 1)

    plt.sca(axes[0])
    plt.title('DRC stim')
    plt.imshow(s, aspect='auto', cmap=plt.get_cmap('jet'))

    plt.sca(axes[1])
    plt.title('Assigned Contrast')
    plt.imshow(c1, aspect='auto')

    plt.sca(axes[2])
    plt.title('Binary Calculated Contrast')
    plt.imshow(c2, aspect='auto')

    plt.sca(axes[3])
    plt.title('Continuous Calculated Contrast')
    plt.imshow(c3, aspect='auto')

    plt.tight_layout(h_pad=0.15)
    return fig


def gc_vs_stp_strengths(batch=289, model1=gc_cont_full, model2=stp_model,
                        model3=ln_model, se_filter=False):

    df_r = nd.batch_comp(batch, [model1, model2, model3], stat='r_test')
    df_e = nd.batch_comp(batch, [model1, model2, model3], stat='se_test')
    # Remove any cellids that have NaN for 1 or more models
    df_r.dropna(axis=0, how='any', inplace=True)
    df_e.dropna(axis=0, how='any', inplace=True)

    cellids = df_r.index.values.tolist()

    if se_filter:
        gc_test = df_r[model1]
        gc_se = df_e[model1]
        stp_test = df_r[model2]
        stp_se = df_e[model2]
        ln_test = df_r[model3]
        ln_se = df_e[model3]

        # Also remove is performance not significant at all
        good_cells = ((gc_test > gc_se*2) & (stp_test > stp_se*2) &
                     (ln_test > ln_se*2))

        # Remove if performance significantly worse than LN
        bad_cells = ((gc_test+gc_se < ln_test-ln_se) |
                     (stp_test+stp_se < ln_test-ln_se))

        keep = good_cells & ~bad_cells

        cellids = df_r[keep].index.values.tolist()

    gc_test = gc_test[cellids]
    stp_test = stp_test[cellids]
    ln_test = ln_test[cellids]

    gcs = []
    stps = []
    for c in cellids:
        xfspec1, ctx1 = xhelp.load_model_xform(c, batch, model1,
                                               eval_model=False)
        mspec1 = ctx1['modelspec']
        dsig_idx = find_module('dynamic_sigmoid', mspec1)
        phi1 = mspec1[dsig_idx]['phi']
        k = phi1['kappa']
        s = phi1['shift']
        k_m = phi1['kappa_mod']
        s_m = phi1['shift_mod']
        b = phi1['base']
        a = phi1['amplitude']
        b_m = phi1['base_mod']
        a_m = phi1['amplitude_mod']

        gc = gc_magnitude(b, b_m, a, a_m, s, s_m, k, k_m)
        gcs.append(gc)

        xfspec2, ctx2 = xhelp.load_model_xform(c, batch, model2,
                                               eval_model=False)
        mspec2 = ctx2['modelspec']
        stp_idx = find_module('stp', mspec2)
        phi2 = mspec2[stp_idx]['phi']
        tau = phi2['tau']
        u = phi2['u']

        stp = stp_magnitude(tau, u)[0]
        stps.append(stp)

    stps_arr = np.mean(np.array(stps), axis=1)
    gcs_arr = np.array(gcs)
#    stps_arr /= np.abs(stps_arr.max())
#    gcs_arr /= np.abs(gcs_arr.max())
    r_diff = np.abs(gc_test - stp_test)

    fig, axes = plt.subplots(3, 1)
    axes[0].scatter(r_diff, stps_arr, c='green', s=1)
    axes[0].scatter(r_diff, gcs_arr, c='black', s=1)
    axes[0].set_xlabel('Difference in Performance')
    axes[0].set_ylabel('GC, STP Magnitudes')

    axes[1].scatter(gcs_arr, stps_arr, c='black', s=1)
    axes[1].set_xlabel('GC Magnitude')
    axes[1].set_ylabel('STP Magnitude')

    axes[2].scatter(gcs_arr[gcs_arr < 0]*-1, stps_arr[gcs_arr < 0], c='black', s=1)
    axes[2].set_xlabel('|GC Magnitude|, Negatives Only')
    axes[2].set_ylabel('STP Magnitude')

    fig.tight_layout()


###############################################################################
#####################      UNUSED AT THE MOMENT    ############################
###############################################################################



# Overlay of prediction from STP versus prediction from GC for sample cell(s)
def example_pred_overlay(cellid=good_cell, model1=gc_cont_full,
                         model2=stp_model):
    xfspec1, ctx1 = xhelp.load_model_xform(cellid, batch, model1)
    xfspec2, ctx2 = xhelp.load_model_xform(cellid, batch, model2)
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
        xfspec1, ctx1 = xhelp.load_model_xform(cellid, batch, model1)
        xfspec2, ctx2 = xhelp.load_model_xform(cellid, batch, model2)

    # 3. Compute the correlation for that cell
        pred1 = ctx1['val']['pred'].as_continuous()
        pred2 = ctx2['val']['pred'].as_continuous()

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
    xfspec1, ctx1 = xhelp.load_model_xform(good_cell, batch, gc_model)
    xfspec2, ctx2 = xhelp.load_model_xform(bad_cell, batch, gc_model)

    plt.figure()
    plt.subplot(221)
    xf.plot_heatmap(ctx1, 'stim')
    plt.subplot(222)
    xf.plot_heatmap(ctx2, 'stim')
    plt.subplot(223)
    xf.plot_heatmap(ctx1, 'contrast')
    plt.subplot(224)
    xf.plot_heatmap(ctx2, 'contrast')


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

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    mspec = ctx['modelspec']
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

    # Force same cellid order now that missing cols are filled in
    df1 = df1[cellids]; df2 = df2[cellids];

    gc_vs_ln = df1.loc['meta--r_test'].values - df2.loc['meta--r_test'].values
    gc_vs_ln = gc_vs_ln.astype('float32')

    kappa_mod = df1[df1.index.str.contains('kappa_mod')]
    kappa = df1[df1.index.str.contains('kappa$')]
    gd_ratio = (kappa_mod.values / kappa.values).astype('float32').flatten()


    # For testing: Some times kappa is so small that the ratio ends up
    # throwing the scale off so far that the plot is worthless.
    # But majority of the time the ratio is less than 5ish, so try rectifying:
    gd_ratio[gd_ratio > 5] = 5
    gd_ratio[gd_ratio < -5] = -5
    # Then normalize to -1 to 1 scale for easier comparison to r value
    gd_ratio /= 5


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
                           threshold = 2.5):
    # TODO: threshold 2.5 works for removing outliers in correlation scatter
    #       and maximizes r, but need an unbiased way to pick this number.
    #       Otherwise basically just cherrypicked the cutoff to make
    #       correlation better.

    # NOTE: Also helps to do this for both gc and stp, then
    #       list(set(gc_cells) & set(stp_cells)) to get the intersection.

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

