import os
import datetime
import logging
from multiprocessing import Process

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import nems0.xform_helper as xhelp
from nems_lbhb.gcmodel.figures.autocorrelation import (load_batch_results,
                                                       tau_vs_model_performance)
from nems_lbhb.gcmodel.figures.drc import test_DRC_with_contrast
from nems_lbhb.gcmodel.figures.equivalence import (equivalence_scatter,
                                                   equivalence_histogram,
                                                   equivalence_effect_size)
from nems_lbhb.gcmodel.figures.parameters import (stp_distributions,
                                                  gc_distributions)
from nems_lbhb.gcmodel.figures.respstats import rate_histogram
from nems_lbhb.gcmodel.figures.soundstats import mean_sd_per_stim_by_batch
from nems_lbhb.gcmodel.figures.summary import (performance_scatters,
                                               performance_bar, significance,
                                               combined_vs_max,
                                               performance_table,
                                               single_scatter)
from nems_lbhb.gcmodel.figures.correlation import per_cell_group
from nems_lbhb.gcmodel.figures.examples import example_clip
from nems_lbhb.gcmodel.figures.simulation import compare_sims, compare_sim_fits
from nems_lbhb.gcmodel.figures.snr import snr_vs_equivalence
# Put all aliased modelnames and cellids in memory,
# e.g. test = "ozgf.fs100.ch18-ld-sev_wc.18x1.g-fir.1x15-lvl.1_init-basic"
from nems_lbhb.gcmodel.figures.definitions import *

log = logging.getLogger(__name__)


# Set plot appearance settings, imported from definitions
plt.rcParams.update(params)


# Plot parameters
figures_base_path = '/auto/users/jacob/notes/gc_rank3/figures/'
plot_stat = 'r_ceiling'
fig_format = 'pdf'
date = str(datetime.datetime.now()).split(' ')[0]
batch, gc, stp, LN, combined = default_args
batch2 = 263
gc_version='summed'


def just_final_figures():
    # Figure 0 -- model schematic (not in python)

    # Figure 1 -- stimulus examples with contrast and summed contrast computed,
    #             distributions of sound statistics for both batches
    # DRC contrast comparisons
    figures_to_save = []
    log.info('Contrast and DRC comparisons ...\n')
    fig1a, fig1aa = test_DRC_with_contrast(ms=30, normalize=True, fs=100, bands=1,
                                           percentile=70, n_segments=8)
    figures_to_save.append((fig1a, 'contrast_comparison'))
    figures_to_save.append((fig1aa, 'contrast_comparison_text'))

    # sound stats
    path_1 = load_paths['sound_stats'][str(batch)]
    path_2 = load_paths['sound_stats'][str(batch2)]
    fig1b, fig1bb = mean_sd_per_stim_by_batch(batch, load_path=path_1,
                                              manual_lims=True)
    fig1c, fig1cc = mean_sd_per_stim_by_batch(batch2, load_path=path_2,
                                              manual_lims=True)
    figures_to_save.append((fig1b, 'sound_stats_b%d' % batch))
    figures_to_save.append((fig1bb, 'sound_stats_b%d_text' % batch))
    figures_to_save.append((fig1c, 'sound_stats_b%d' % batch2))
    figures_to_save.append((fig1cc, 'sound_stats_b%d_text' % batch2))


    # Figure 2 -- summaries of prediction accuracy, for both only_improvements
    #             and for all cells.
    fig2a, fig2at = performance_bar(batch, gc, stp, LN, combined, plot_stat=plot_stat,
                                    abbr_yaxis=True, manual_y=(0.5, 0.65))
    fig2b, fig2bt = performance_bar(batch2, gc, stp, LN, combined, plot_stat=plot_stat,
                                    abbr_yaxis=True, manual_y=(0.5, 0.65))
    fig2aa = significance(batch, gc, stp, LN, combined, include_legend=False,
                          plot_stat=plot_stat)
    fig2bb = significance(batch2, gc, stp, LN, combined, include_legend=False,
                          plot_stat=plot_stat)
    fig2c, fig2cc = single_scatter(batch, gc, stp, LN, combined, compare=(2,3),
                           plot_stat=plot_stat)
    fig2d, fig2dd = combined_vs_max(batch, gc, stp, LN, combined,
                                    plot_stat=plot_stat,
                                    improved_only=True,
#                                    snr_path=snr_path,
#                                    exclude_low_snr=True
                                    )
    fig2e, fig2ee = single_scatter(batch2, gc, stp, LN, combined, compare=(2,3),
                                   plot_stat=plot_stat)

    figures_to_save.append((fig2a, 'performance_bar_289'))
    figures_to_save.append((fig2at, 'performance_bar_289_text'))
    figures_to_save.append((fig2aa, 'significance_tests_289'))
    figures_to_save.append((fig2b, 'performance_bar_263'))
    figures_to_save.append((fig2bt, 'performance_bar_263_text'))
    figures_to_save.append((fig2bb, 'significance_tests_263'))
    figures_to_save.append((fig2c, 'ln_vs_combined_289'))
    figures_to_save.append((fig2cc, 'ln_vs_combined_289_text'))
    figures_to_save.append((fig2e, 'ln_vs_combined_263'))
    figures_to_save.append((fig2ee, 'ln_vs_combined_263_text'))
#    for sf, sn in zip(scatter_figs, scatter_names):
#        figures_to_save.append((sf, sn))
    figures_to_save.append((fig2d, 'combined_vs_max_289'))
    figures_to_save.append((fig2dd, 'combined_vs_max_289_text'))



    # Figure 3:
    # Equivalence
    log.info('Equivalence analyses ...\n')
    snr_path = load_paths['snrs']
    fig3a, fig3aa = equivalence_scatter(batch, gc, stp, LN, combined,
                                        plot_stat=plot_stat, drop_outliers=True,
                                        color_improvements=True,
                                        self_equiv=True, self_eq_models=eq_both,
                                        exclude_low_snr=True,
                                        snr_path=snr_path,
                                        #enable_hover=True,
                                        )

    _, fig3aaa = equivalence_scatter(batch, gc, stp, LN, combined,
                                     plot_stat=plot_stat, drop_outliers=True,
                                     color_improvements=True,
                                     self_equiv=True, self_eq_models=cross_all)
    fig3b, fig3bb = equivalence_scatter(batch2, gc, stp, LN, combined,
                                        plot_stat=plot_stat, drop_outliers=True,
                                        color_improvements=True)
                                        #self_equiv=True, self_eq_models=eq_both)#,
                                        #manual_lims=(-0.3, 0.3))
#    fig3c = equivalence_scatter(batch, combined, stp, LN, gc,
#                                plot_stat=plot_stat, drop_outliers=True,
#                                color_improvements=True,
#                                xmodel='GC+STP', ymodel='STP')
#    fig3d = equivalence_scatter(batch, combined, gc, LN, stp,
#                                plot_stat=plot_stat, drop_outliers=True,
#                                color_improvements=True,
#                                xmodel='GC+STP', ymodel='GC')
    hist_path = load_paths['equivalence_effect_size'][gc_version]
    hist_path_263 = load_paths['equivalence_effect_size']['263']
    snr_path = load_paths['snrs']
    fig3e, fig3ee = equivalence_effect_size(batch, gc, stp, LN, combined,
                                            load_path=hist_path,
                                            only_improvements=True)
#    fig3f, fig3fff = equivalence_histogram(batch, gc, stp, LN, combined,
#                                                   load_path=hist_path,
#                                                   self_equiv=True,
#                                                   self_kwargs=eq_kwargs,
#                                                   eq_models=eq_both)
    fig3f, fig3ff = equivalence_histogram(batch, gc, stp, LN, combined,
                                      load_path=hist_path, self_equiv=True,
                                      self_kwargs=eq_kwargs,
                                      eq_models=eq_both,
                                      cross_kwargs=cross_kwargs,
                                      cross_models=cross_all,
                                      adjust_scores=True,
                                      use_median=False,
                                      #use_log_ratios=True,
                                      )
    fig3k, fig3kk = equivalence_histogram(batch, gc, stp, LN, combined,
                                      load_path=hist_path, self_equiv=True,
                                      self_kwargs=eq_kwargs,
                                      eq_models=eq_both,
                                      cross_kwargs=cross_kwargs,
                                      cross_models=cross_all,
                                      adjust_scores=False,
                                      exclude_low_snr=True,
                                      snr_path=snr_path
                                      )
    fig3g, fig3gg = equivalence_effect_size(batch2, gc, stp, LN, combined,
                                            load_path=hist_path_263,
                                            only_improvements=True)
    fig3h, fig3hhh = equivalence_histogram(batch2, gc, stp, LN, combined,
                                           load_path=hist_path_263)
    figures_to_save.append((fig3a, 'equivalence_scatter'))
    figures_to_save.append((fig3aa, 'equivalence_scatter_text'))
    figures_to_save.append((fig3aaa, 'equivalence_scatter_text_cross'))
    figures_to_save.append((fig3b, 'equivalence_scatter_263'))
    figures_to_save.append((fig3bb, 'equivalence_scatter_263_text'))
#    figures_to_save.append((fig3c, 'equivalence_scatter_comb_vs_stp'))
#    figures_to_save.append((fig3d, 'equivalence_scatter_comb_vs_gc'))
    figures_to_save.append((fig3e, 'equivalence_vs_effect'))
    figures_to_save.append((fig3f, 'equivalence_histogram'))
    figures_to_save.append((fig3ee, 'equivalence_effect_text'))
    #figures_to_save.append((fig3fff, 'equivalence_hist_text'))
    figures_to_save.append((fig3ff, 'equivalence_hist_text_crossed'))
    figures_to_save.append((fig3g, 'equivalence_vs_effect_263'))
    figures_to_save.append((fig3h, 'equivalence_histogram_263'))
    figures_to_save.append((fig3gg, 'equivalence_effect_text_263'))
    figures_to_save.append((fig3hhh, 'equivalence_hist_text_263'))
    figures_to_save.append((fig3k, 'equivalence_hist_snr'))
    figures_to_save.append((fig3kk, 'equivalence_hist_text_snr'))


    # figure 4:
    # Examples
    # stp helps, gc not
    s1 = '/auto/users/jacob/notes/gc_rank3/figures/examples/use_for_paper/stp.pickle'
    fig4a, fig4aa, fig4aaa = example_clip('AMT005c-20-1', *default_args,
                                          stim_idx=0,
                                          trim_start=40, trim_end=180,
                                          skip_combined=False,
                                          save_path=None,
                                          load_path=s1)

    # TODO: try to find a better example for including combined PSTH
    # gc helps, stp not (but just barely)
    s2 = '/auto/users/jacob/notes/gc_rank3/figures/examples/use_for_paper/gc.pickle'
    fig4b, fig4bb, fig4bbb = example_clip('TAR009d-22-1', *default_args,
                                          stim_idx=1,
                                          trim_start=190, trim_end=330,
                                          skip_combined=False,
                                          save_path=None,
                                          load_path=s2)

    # LN cell
    s3 = '/auto/users/jacob/notes/gc_rank3/figures/examples/use_for_paper/LN.pickle'
    fig4c, fig4cc, fig4ccc = example_clip('TAR010c-40-1', *default_args,
                                          stim_idx=0,
                                          trim_start=190, trim_end=330,
                                          skip_combined=False,
                                          save_path=None,
                                          load_path=s3)

    # difference cell
#    s4 = '/auto/users/jacob/notes/gc_rank3/figures/examples/use_for_paper/difference.pickle'
#    fig4d, fig4dd, fig4ddd = example_clip('AMT004b-26-2', *default_args,
#                                          stim_idx=13,
#                                          trim_start=40, trim_end=180,
#                                          skip_combined=False,
#                                          save_path=s4,
#                                          load_path=None)
    figures_to_save.append((fig4a, 'ex_stp_cell'))
    figures_to_save.append((fig4aa, 'ex_stp_cell_text'))
    figures_to_save.append((fig4aaa, 'ex_stp_cell_strf'))
    figures_to_save.append((fig4b, 'ex_gc_cell'))
    figures_to_save.append((fig4bb, 'ex_gc_cell_text'))
    figures_to_save.append((fig4bbb, 'ex_gc_cell_strf'))
    figures_to_save.append((fig4c, 'ex_LN_cell'))
    figures_to_save.append((fig4cc, 'ex_LN_cell_text'))
    figures_to_save.append((fig4ccc, 'ex_LN_cell_strf'))
    figures_to_save.append((fig4d, 'ex_difference_cell'))
    figures_to_save.append((fig4dd, 'ex_difference_cell_text'))
    figures_to_save.append((fig4ddd, 'ex_difference_cell_strf'))


    # Parameters
    log.info('Parameter analyses ...\n')
    fig5a1, fig5a2, fig5a3, fig5a4, fig5a5 = stp_distributions(
            batch, gc, stp, LN, combined, use_combined=False)
    gc_dist_figs = gc_distributions(batch, gc, stp, LN, combined,
                                    use_combined=False)
    labels = ['base_dist', 'base_text', 'amp_dist', 'amp_text',
              'shift_dist', 'shift_text', 'kappa_dist', 'kappa_text',
              'md_nonimp_effects', 'md_nonimp_text', 'md_imp_effects',
              'md_imp_text']

    fig5b1, fig5b2, fig5b3, fig5b4, fig5b5 = stp_distributions(
            batch2, gc, stp, LN, combined)
    gc_dist_figs2 = gc_distributions(batch2, gc, stp, LN, combined)
    labels2 = ['base_dist', 'base_text', 'amp_dist', 'amp_text',
              'shift_dist', 'shift_text', 'kappa_dist', 'kappa_text',
              'md_nonimp_effects', 'md_nonimp_text', 'md_imp_effects',
              'md_imp_text']
    labels2 = [s + '_263' for s in labels2]

    figures_to_save.append((fig5a1, 'tau_distributions'))
    figures_to_save.append((fig5a2, 'tau_dist_text'))
    figures_to_save.append((fig5a3, 'u_distributions'))
    figures_to_save.append((fig5a4, 'u_dist_text'))
    figures_to_save.append((fig5a5, 'med_stp_effects'))
    for f, n in zip(gc_dist_figs, labels):
        figures_to_save.append((f, n))

    figures_to_save.append((fig5b1, 'tau_distributions_263'))
    figures_to_save.append((fig5b2, 'tau_dist_text_263'))
    figures_to_save.append((fig5b3, 'u_distributions_263'))
    figures_to_save.append((fig5b4, 'u_dist_text_263'))
    figures_to_save.append((fig5b5, 'med_stp_effects_263'))
    for f, n in zip(gc_dist_figs2, labels2):
        figures_to_save.append((f, n))


    # simulations
    #fig6a = compare_sims(190, 270)
    #fig6b = compare_sims(1550,1610)

    # Simulation specs for re-generating simulations
    # TODO: STP isn't matching the previous simulation... not even
    # the response is the same. Did the cellid get overwritten or something?
#    xfspec1, ctx1 = xhelp.load_model_xform('AMT005c-20-1', batch, stp)
#    stp_spec = ctx1['modelspec']
#    xfspec2, ctx2 = xhelp.load_model_xform('TAR009d-22-1', batch, gc)
#    gc_spec = ctx2['modelspec']
#    xfspec3, ctx3 = xhelp.load_model_xform('TAR010c-40-1', batch, LN)
#    LN_spec = ctx3['modelspec']

    sim_start=740
    sim_end=880

    fig6c, fig6cc = compare_sim_fits(#simulation_spec=stp_spec,
            *default_args, load_path=load_paths['simulations']['stp_cell']['stp'],
            start=sim_start, end=sim_end, tag='stp_cell_stp_sim',
            skip_combined=False
            )
    fig6d, fig6dd = compare_sim_fits(#simulation_spec=gc_spec,
            *default_args, load_path=load_paths['simulations']['gc_cell']['gc'],
            start=sim_start, end=sim_end, tag='gc_cell_gc_sim',
            skip_combined=False
            )
#    fig6e, fig6ee = compare_sim_fits(
#            load_path=load_paths['simulations']['gc_cell']['stp'],
#            start=sim_start, end=sim_end, tag='gc_cell_stp_sim'
#            )
#    fig6f, fig6ff = compare_sim_fits(
#            load_path=load_paths['simulations']['stp_cell']['gc'],
#            start=sim_start, end=sim_end, tag='stp_cell_gc_sim'
#            )19s-
    fig6g, fig6gg = compare_sim_fits(#simulation_spec=LN_spec,
            *default_args, load_path=load_paths['simulations']['LN_cell']['LN'],
            start=sim_start, end=sim_end, tag='LN_cell_LN_sim',
            ext_start=2.3, skip_combined=False
            )
    #figures_to_save.append((fig6a, 'parameter_effects_stp_onset'))
    #figures_to_save.append((fig6b, 'parameter_effects_gc_up_or_down'))
    figures_to_save.append((fig6c, 'stp_cell_stp_sim'))
    figures_to_save.append((fig6cc, 'stp_cell_stp_sim_text'))
    figures_to_save.append((fig6d, 'gc_cell_gc_sim'))
    figures_to_save.append((fig6dd, 'gc_cell_gc_sim_text'))
#    figures_to_save.append((fig6e, 'gc_cell_stp_sim'))
#    figures_to_save.append((fig6ee, 'gc_cell_stp_sim_text'))
#    figures_to_save.append((fig6f, 'stp_cell_gc_sim'))
#    figures_to_save.append((fig6ff, 'stp_cell_gc_sim_text'))
    figures_to_save.append((fig6g, 'LN_cell_LN_sim'))
    figures_to_save.append((fig6gg, 'LN_cell_LN_sim_text'))



    # Response stats comparison
    fig7a, fig7aa = rate_histogram(batch, gc, stp, LN, combined, spont_289,
                                   rate_type='spont', allow_overlap=False)
    fig7b, fig7bb = rate_histogram(batch, gc, stp, LN, combined, mean_289,
                                   rate_type='mean', allow_overlap=False)
    # Do both for now, in some ways overlap makes sense
    # but in same ways it doesn't.
    fig7c, fig7cc = rate_histogram(batch, gc, stp, LN, combined, spont_289,
                                   rate_type='spont', allow_overlap=True)
    fig7d, fig7dd = rate_histogram(batch, gc, stp, LN, combined, mean_289,
                                   rate_type='mean', allow_overlap=True)

    figures_to_save.append((fig7a, 'spont_histogram'))
    figures_to_save.append((fig7aa, 'spont_text'))
    figures_to_save.append((fig7b, 'mean_histogram'))
    figures_to_save.append((fig7bb, 'mean_text'))

    figures_to_save.append((fig7c, 'spont_histogram_overlap'))
    figures_to_save.append((fig7cc, 'spont_text_overlap'))
    figures_to_save.append((fig7d, 'mean_histogram_overlap'))
    figures_to_save.append((fig7dd, 'mean_text_overlap'))


    # SNR analysis
    snr_path = load_paths['snrs']
    stp_path = load_paths['self_equivalence']['stp']
    gc_path = load_paths['self_equivalence']['gc']
    fig8a, fig8b, fig8c = snr_vs_equivalence(snr_path, stp_path, gc_path)

    figures_to_save.append((fig8a, 'snr_vs_equiv_stp'))
    figures_to_save.append((fig8b, 'snr_vs_equiv_gc'))
    figures_to_save.append((fig8c, 'snr_vs_equiv_text'))


    # Save everything
    base_path = os.path.join(figures_base_path, date, fig_format, plot_stat)
    all_figs_name = 'all_figures_%s_%s' % (gc_version, plot_stat) + '.pdf'
    log.info('Saving figures to base_path: \n%s', base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    pdf = matplotlib.backends.backend_pdf.PdfPages(
            os.path.join(base_path, all_figs_name)
            )
    for fig, name in figures_to_save:
        full_path = os.path.join(base_path, name) + ('.%s' % fig_format)
        fig.savefig(full_path, format=fig_format, dpi=fig.dpi)
        pdf.savefig(fig, dpi=fig.dpi)
        plt.close(fig)
    pdf.close()
    plt.close('all')  # just to make double sure that everything is closed


def rerun_saved_analyses(*functions_to_skip):
    # TODO: separate script for re-generating data for equivalence histogram,
    # autocorrelation, etc.
    functions_to_run = []

    # for each analysis:
    # if 'name' not in functions_to_skip:
    #     fn = partial(analysis_fn, arg1, arg2, save_path=save_path)
    #     functions_to_run.append(fn)


    # autocorrelation
    if 'autocorrelation' not in functions_to_skip:
        pass


    run_in_parallel(functions_to_run)


# copied from https://stackoverflow.com/questions/7207309/  ...
#             python-how-can-i-run-python-functions-in-parallel
def run_in_parallel(*functions):
    processes = []
    for fn in functions:
        p = Process(target=fn)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
