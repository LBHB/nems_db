import os
import datetime
import logging
from multiprocessing import Process

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from nems_lbhb.gcmodel.figures.autocorrelation import (load_batch_results,
                                                       tau_vs_model_performance)
from nems_lbhb.gcmodel.figures.drc import test_DRC_with_contrast
from nems_lbhb.gcmodel.figures.equivalence import (equivalence_scatter,
                                                   equivalence_histogram,
                                                   equivalence_effect_size)
from nems_lbhb.gcmodel.figures.parameters import (stp_distributions,
                                                  gc_distributions)
from nems_lbhb.gcmodel.figures.respstats import rate_vs_performance
from nems_lbhb.gcmodel.figures.soundstats import mean_sd_per_stim_by_batch
from nems_lbhb.gcmodel.figures.summary import (performance_scatters,
                                               performance_bar, significance,
                                               combined_vs_max)
# Put all aliased modelnames and cellids in memory,
# e.g. test = "ozgf.fs100.ch18-ld-sev_wc.18x1.g-fir.1x15-lvl.1_init-basic"
from nems_lbhb.gcmodel.figures.definitions import *

log = logging.getLogger(__name__)


# Set plot appearance settings, imported from definitions
plt.rcParams.update(params)


figures_base_path = '/auto/users/jacob/notes/gc_rank3/figures/'
# run once for gc_version='summed', again for ='kernel'
def run_all(batch, gc, stp, LN, combined, batch2=263, good_ln=0.0,
            plot_stat='r_ceiling', gc_version='summed', fig_format='pdf'):
    date = str(datetime.datetime.now()).split(' ')[0]
    # Assumes all batch analyses have already been run and saved.
    # If needed, use rerun_saved_analyses first and
    # update load_paths in definitions.
    figures_to_save = []   # append (fig, name)


    # Autocorrelation analyses
    log.info('Autocorrelation ...\n')
    df = load_batch_results(load_paths['AC'])
    # TOOD: figure out why log tau is giving a weird result
    fig1 = tau_vs_model_performance(df, batch, gc, stp, LN, combined,
                                    good_ln=good_ln, log_tau=False)
    figures_to_save.append((fig1, 'tau_vs_performance'))


    # DRC contrast comparisons
    log.info('Contrast and DRC comparisons ...\n')
    fig2 = test_DRC_with_contrast(ms=30, normalize=True, fs=100, bands=1,
                                  percentile=70, n_segments=8)
    figures_to_save.append((fig2, 'contrast_comparison'))


    # Equivalence
    log.info('Equivalence analyses ...\n')
    fig3 = equivalence_scatter(batch, gc, stp, LN, combined,
                               plot_stat=plot_stat)
    hist_path = load_paths['equivalence_histogram'][gc_version]
    fig4 = equivalence_histogram(batch, gc, stp, LN, combined,
                                 load_path=hist_path)
    figures_to_save.append((fig4, 'equivalence_histogram'))
#    hist_path = load_paths['equivalence_effect_size'][gc_version]
#    fig4a, fig4b = equivalence_effect_size(batch, gc, stp, LN, combined,
#                                           load_path=hist_path)
    figures_to_save.append((fig3, 'equivalence_scatter'))
#    figures_to_save.append((fig4a, 'equivalence_vs_effect'))
#    figures_to_save.append((fig4b, 'equivalence_histogram'))


    # Parameters
    log.info('Parameter analyses ...\n')
    fig5 = stp_distributions(batch, gc, stp, LN, combined, good_ln=good_ln)
    fig6 = gc_distributions(batch, gc, stp, LN, combined, good_ln=good_ln)
    # TOOD: something for the sigmoid parameters?
    # o     or just use some examples from the single cell plots?
    #fig7 = ...

    figures_to_save.append((fig5, 'stp_distributions'))
    figures_to_save.append((fig6, 'gc_distributions'))


    # response stats
    log.info('Response statistics ...\n')
    spont_path = load_paths['spont']
    max_path = load_paths['max']
    # 6 figures: compare gc, stp, and combined models for max and spont
    fig7a = rate_vs_performance(batch, gc, stp, LN, combined, compare='gc',
                                plot_stat=plot_stat, rate_stat='spont',
                                relative_performance=True,
                                load_path=spont_path)
    fig7b = rate_vs_performance(batch, gc, stp, LN, combined, compare='stp',
                                plot_stat=plot_stat, rate_stat='spont',
                                relative_performance=True,
                                load_path=spont_path)
    fig7c = rate_vs_performance(batch, gc, stp, LN, combined, compare='combined',
                                plot_stat=plot_stat, rate_stat='spont',
                                relative_performance=True,
                                load_path=spont_path)
    fig8a = rate_vs_performance(batch, gc, stp, LN, combined, compare='gc',
                                plot_stat=plot_stat, rate_stat='max',
                                relative_performance=True,
                                load_path=max_path)
    fig8b = rate_vs_performance(batch, gc, stp, LN, combined, compare='stp',
                                plot_stat=plot_stat, rate_stat='max',
                                relative_performance=True,
                                load_path=max_path)
    fig8c = rate_vs_performance(batch, gc, stp, LN, combined, compare='combined',
                                plot_stat=plot_stat, rate_stat='max',
                                relative_performance=True,
                                load_path=max_path)
    figures_to_save.append((fig7a, 'spont_vs_gc'))
    figures_to_save.append((fig7b, 'spont_vs_stp'))
    figures_to_save.append((fig7c, 'spont_vs_combined'))
    figures_to_save.append((fig8a, 'max_vs_gc'))
    figures_to_save.append((fig8b, 'max_vs_stp'))
    figures_to_save.append((fig8c, 'max_vs_combined'))


    # sound stats
    path_1 = load_paths['sound_stats'][str(batch)]
    path_2 = load_paths['sound_stats'][str(batch2)]
    fig9 = mean_sd_per_stim_by_batch(batch, load_path=path_1)
    fig10 = mean_sd_per_stim_by_batch(batch2, load_path=path_2)
    figures_to_save.append((fig9, 'sound_stats_b%d' % batch))
    figures_to_save.append((fig10, 'sound_stats_b%d' % batch2))


    # summary figures
    fig11 = performance_bar(batch, gc, stp, LN, combined, plot_stat=plot_stat)
    fig12 = performance_bar(batch, gc, stp, LN, combined, plot_stat=plot_stat,
                            only_improvements=True)
    fig13 = significance(batch, gc, stp, LN, combined, include_legend=False,
                         plot_stat=plot_stat)
    scatter_figs = performance_scatters(batch, gc, stp, LN, combined,
                                        plot_stat=plot_stat)
    scatter_names = ['ln_vs_gc', 'ln_vs_stp', 'ln_vs_combined', 'gc_vs_stp',
                     'gc_vs_combined', 'stp_vs_combined']
    fig14a, fig14b = combined_vs_max(batch, gc, stp, LN, combined,
                                     plot_stat=plot_stat)

    figures_to_save.append((fig11, 'performance_bar_all'))
    figures_to_save.append((fig12, 'performance_bar_improvements'))
    figures_to_save.append((fig13, 'significance_tests'))
    for sf, sn in zip(scatter_figs, scatter_names):
        figures_to_save.append((sf, sn))
    figures_to_save.append((fig14a, 'combined_vs_max_absolute'))
    figures_to_save.append((fig14b, 'combined_vs_max_relative'))


    # Examples
    # TODO


    # Save everything
    base_path = os.path.join(figures_base_path, date, fig_format, gc_version,
                             plot_stat)
    log.info('Saving figures to base_path: \n%s', base_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    pdf = matplotlib.backends.backend_pdf.PdfPages(
            os.path.join(base_path, 'all_figures') + '.pdf'
            )
    for fig, name in figures_to_save:
        full_path = os.path.join(base_path, name) + ('.%s' % fig_format)
        fig.savefig(full_path, format=fig_format, dpi=fig.dpi)
        pdf.savefig(fig, dpi=fig.dpi)
        plt.close(fig)
    pdf.close()



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
