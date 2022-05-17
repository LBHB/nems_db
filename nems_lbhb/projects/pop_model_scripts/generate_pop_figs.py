from pathlib import Path
import datetime
import os

import numpy as np

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
    mplparams, MODELGROUPS, POP_MODELGROUPS, HELDOUT, MATCHED, EQUIVALENCE_MODELS_SINGLE, EQUIVALENCE_MODELS_POP,
    POP_MODELS, ALL_FAMILY_POP, shortnames,
    SIG_TEST_MODELS,
    get_significant_cells, snr_by_batch, NAT4_A1_SITES, NAT4_PEG_SITES, PLOT_STAT, DOT_COLORS, DOT_MARKERS, base_path,
    linux_user, ALL_FAMILY_MODELS, VERSION, count_fits, int_path, a1, peg, single_column_short, single_column_tall,
    column_and_half_short, column_and_half_tall, single_column_shorter, double_column_short, double_column_shorter
)
import nems.db as nd

import matplotlib as mpl
mpl.rcParams.update(mplparams)  # import from pop_model_utils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from nems_lbhb.projects.pop_model_scripts.pareto_pop_plot import model_comp_pareto
from nems_lbhb.projects.pop_model_scripts.summary_plots import scatter_bar
from nems_lbhb.projects.pop_model_scripts.pop_correlation import correlation_histogram
from nems_lbhb.projects.pop_model_scripts.heldout_plots import generate_heldout_plots
from nems_lbhb.projects.pop_model_scripts.matched_snr_plots import plot_matched_snr, plot_heldout_a1_vs_peg
from nems_lbhb.projects.pop_model_scripts.partial_est_plot import partial_est_plot
from nems_lbhb.projects.pop_model_scripts.pred_scatter import plot_pred_scatter, bar_mean
from nems_lbhb.projects.pop_model_scripts.snr_batch import sparseness, sparseness_by_batch, sparseness_plot, sparseness_example, sparseness_figs


a1 = 322
peg = 323
stats_tests = []

########################################################################################################################
#################################   SCATTER/BAR PRED   #################################################################
########################################################################################################################

fig3a, axa = plt.subplots(1, 2, figsize=single_column_shorter, sharey=True)
fig3b, axb = plt.subplots(1, 2, figsize=single_column_shorter, sharey=True)
fig3a, n_sig_1d_LN, n_nonsig_1d_LN = plot_pred_scatter(a1, [ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[2]],
                                                      labels=['pop LN', '1D CNNx2'], ax=axa[0])
axa[0].legend()
fig3a, n_sig_1d_2d, n_nonsig_1d_2d = plot_pred_scatter(a1, [ALL_FAMILY_MODELS[0], ALL_FAMILY_MODELS[2]],
                                                      labels=['2D CNN', '1D CNNx2'], ax=axa[1])
ax3a, a1_medians, a1_bar_stats = bar_mean(a1, ALL_FAMILY_MODELS, stest=SIG_TEST_MODELS, ax=axb[0])
ax3a.set_box_aspect(1)
ax3b, peg_medians, peg_bar_stats = bar_mean(peg, ALL_FAMILY_MODELS, stest=SIG_TEST_MODELS, ax=axb[1])
ax3b.set_box_aspect(1)
fig3a.tight_layout()
fig3b.tight_layout()

# Get general cell and site count info
a1_r = nd.batch_comp(a1, ALL_FAMILY_MODELS, stat='r_test')
peg_r = nd.batch_comp(peg, ALL_FAMILY_MODELS, stat='r_test')
a1_cell_count = len(a1_r)
a1_site_count = len(set([s.split('-')[0] for s in a1_r.index.values]))
peg_cell_count = len(peg_r)
peg_site_count = len(set([s.split('-')[0] for s in peg_r.index.values]))

stats_tests.append('scatter / bar summary figure')
stats_tests.append('site and cell counts:')
stats_tests.append(f'A1: {a1_site_count} sites, {a1_cell_count} cells')
stats_tests.append(f'PEG: {peg_site_count} sites, {peg_cell_count} cells')

stats_tests.append('scatter plots')
stats_tests.append(f'1D vs LN, {n_sig_1d_LN} sig. cells, {n_nonsig_1d_LN} non-sig. cells')
stats_tests.append(f'1D vs 2D, {n_sig_1d_2d} sig. cells, {n_nonsig_1d_2d} non-sig. cells')

stats_tests.append('\nBar plots')
stats_tests.append('A1 median r_ceiling:')
stats_tests.append(f'{a1_medians}')
stats_tests.append('\nPEG median r_ceiling:')
stats_tests.append(f'{peg_medians}')

stats_tests.append('\nsig. testing between all models:')
stats_tests.append('A1:')
stats_tests.extend([f'{k}:  {v}' for k, v in a1_bar_stats.items()])
stats_tests.append('\nPEG')
stats_tests.extend([f'{k}:  {v}' for k, v in peg_bar_stats.items()])

relative_change_by_batch = '\n'.join([f'{n}:  {m1/m2}' for n, m1, m2 in zip(shortnames, a1_medians, peg_medians)])
stats_tests.append('\nRelative change  a1/peg:')
stats_tests.append(relative_change_by_batch)

cnn1d_vs_LN_A1 = a1_medians[2] / a1_medians[3]
cnn1d_vs_LN_PEG = peg_medians[2] / peg_medians[3]
cnn2d_vs_LN_A1 = a1_medians[0] / a1_medians[3]
cnn2d_vs_LN_PEG = peg_medians[0] / peg_medians[3]
stats_tests.append('\nRelative change CNN / LN:')
stats_tests.append(f'A1 CNN 1Dx2 / pop LN:  {cnn1d_vs_LN_A1}')
stats_tests.append(f'PEG CNN 1Dx2 / pop LN:  {cnn1d_vs_LN_PEG}')
stats_tests.append(f'A1 CNN 2D / pop LN:  {cnn2d_vs_LN_A1}')
stats_tests.append(f'PEG CNN 2D / pop LN:  {cnn2d_vs_LN_PEG}')


########################################################################################################################
#################################   PARETO  ############################################################################
########################################################################################################################


means = []
fig2, axes1 = plt.subplots(1, 2, figsize=column_and_half_short, sharex=True, sharey=True)
xlims = []
ylims = []

for i, batch in enumerate([a1, peg]):
    show_legend = (i==0)

    sig_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)
    _, b_ceiling, model_mean, labels = model_comp_pareto(batch, MODELGROUPS, axes1[i], sig_cells,
                                                 nparms_modelgroups=POP_MODELGROUPS,
                                                 dot_colors=DOT_COLORS, dot_markers=DOT_MARKERS,
                                                 plot_stat=PLOT_STAT, plot_medians=True,
                                                 labeled_models=ALL_FAMILY_MODELS,
                                                 show_legend=show_legend)
    means.append(model_mean)
    batch_name = 'A1' if batch == a1 else 'PEG'
    # axes1[i].set_title(f'{batch_name}')
    xlims.extend(axes1[i].get_xlim())
    ylims.extend(axes1[i].get_ylim())

for a in axes1:
    a.set_xlim(np.min(xlims), np.max(xlims))
    a.set_ylim(np.min(ylims), np.max(ylims))

relative_changes_per_model = [means[0][i] / means[1][i] for i, _ in enumerate(labels)]
relative_change_pareto = '\n'.join([f'{n}: {changes.mean()}' for n, changes in zip(labels, relative_changes_per_model)])
stats_tests.append('\n\nPareto plot, relative change a1/peg:')
stats_tests.append(relative_change_pareto)


########################################################################################################################
#################################   HELDOUT  ###########################################################################
########################################################################################################################
fig5, axes3 = plt.subplots(1, 2, figsize=column_and_half_short, sharex=True, sharey=True)
tests1, sig1, r1, m1, mds1 = generate_heldout_plots(a1, 'A1', ax=axes3[0])
tests2, sig2, r2, m2, mds2 = generate_heldout_plots(peg, 'PEG', ax=axes3[1])
axes3[0].set_ylabel('Prediction correlation')
axes3[1].set_ylabel('')

short_names = ['1D CNNx2', 'pop LN', 'CNN single']
print('Make sure short_names matches actual modelnames used!')
print('short_names: %s' % short_names)
print('HELDOUT: %s' % HELDOUT)

stats_tests.append("\n\nheldout vs matched, Sig. tests (U-statistic, p-value) for batch %d:" % a1)
stats_tests.append(''.join([f'{s}:   {t}|\n' for s, t in zip(short_names, tests1)]))
stats_tests.append("median diffs:")
stats_tests.append(str(mds1))
stats_tests.append("\n")
stats_tests.append("heldout vs matched, Sig. tests (U-statistic, p-value) for batch %d:" % peg)
stats_tests.append(''.join([f'{s}:   {t}|\n' for s, t in zip(short_names, tests2)]))
stats_tests.append("median diffs:")
stats_tests.append(str(mds2))


########################################################################################################################
#################################   DATA SUBSETS  ######################################################################
########################################################################################################################


fig6 = partial_est_plot(batch=a1, PLOT_STAT='r_ceiling', figsize=single_column_shorter)
fig6.tight_layout()


########################################################################################################################
#################################   EQUIVALENCE  #######################################################################
########################################################################################################################

a1_corr_path = int_path  / str(a1) / 'corr_nat4.pkl'
a1_corr_path_pop = int_path / str(a1) / 'corr_nat4_pop.pkl'
peg_corr_path = int_path / str(peg) / 'corr_nat4.pkl'
peg_corr_path_pop = int_path / str(peg) / 'corr_nat4_pop.pkl'

if 0:
    from pop_correlation import generate_psth_correlations_pop
    batch=322
    generate_psth_correlations_pop(batch, EQUIVALENCE_MODELS_POP, save_path=a1_corr_path)
    batch=323
    generate_psth_correlations_pop(batch, EQUIVALENCE_MODELS_POP, save_path=peg_corr_path)

fig7, axes2 = plt.subplots(1, 2, figsize=single_column_shorter)
axes2[0].set_box_aspect(1)
axes2[1].set_box_aspect(1)
a1_corr, a1_stats7 = correlation_histogram(
    a1, 'A1', load_path=a1_corr_path, force_rerun=False, use_pop_models=True, ax=axes2[0])
peg_corr, peg_stats7 = correlation_histogram(
    peg, 'PEG', load_path=peg_corr_path, force_rerun=False, use_pop_models=True, ax=axes2[1])

fig7.tight_layout()
stats_tests.append("\n\ncorrelation histograms, A1 sig tests: %s" % a1_stats7)
stats_tests.append("correlation histograms, PEG sig tests: %s" % peg_stats7)


########################################################################################################################
##################################   SPARSENESS   ######################################################################
########################################################################################################################

fig8, tests8 = sparseness_figs()
stats_tests.append("\n\nsparseness figs stats results:")
stats_tests.append(''.join([f'{t}\n' for t in tests8]))
fig8.tight_layout()

########################################################################################################################
#################################   SNR  ###############################################################################
########################################################################################################################

a1_snr_path = int_path / str(a1) / 'snr_nat4.csv'
peg_snr_path = int_path / str(peg) / 'snr_nat4.csv'

fig9, ax4 = plt.subplots(1, 3, figsize=column_and_half_short)
test_c1, test_LN, test_dnn, test_snr, a1_md, a1_md_match, peg_md, peg_md_match, \
    a1_md_snr, a1_md_snr_match, peg_md_snr, peg_md_snr_match = plot_matched_snr(
        a1, peg, a1_snr_path, peg_snr_path, plot_sanity_check=False, ax=ax4[0], inset_ax=ax4[1]
    )
ax4[0].set_ylabel('Prediction correlation')
ax4[1].set_xlabel('')
ax4[0].set_box_aspect(1)
ax4[1].set_box_aspect(1)

a1_heldout_r, peg_heldout_r, wilcoxon = plot_heldout_a1_vs_peg(a1_snr_path, peg_snr_path, ax=ax4[2])
ax4[2].set_box_aspect(1)
ax4[2].set_xlabel('')
ax4[2].set_ylabel('Prediction correlation')

tests9 = [('conv1D', test_c1), ('LN_pop', test_LN), ('dnn1_single', test_dnn)]
stats_tests.append('matched snr')
stats_tests.append('sig. tests:')
stats_tests.append(''.join([f'{s}:   {t}\n' for s, t in tests9]))

stats_tests.append('\nmedian prediction corr info:')
stats_tests.append('a1 full medians:')
stats_tests.append(str(a1_md))
stats_tests.append('a1 mathced medians:')
stats_tests.append(str(a1_md_match))
stats_tests.append('peg full medians:')
stats_tests.append(str(peg_md))
stats_tests.append('peg mathced medians:')
stats_tests.append(str(peg_md_match))
stats_tests.append('a1 - peg medians full:')
stats_tests.append(str(a1_md.values - peg_md.values))
stats_tests.append('a1 - peg matched:')
stats_tests.append(str(a1_md_match.values - peg_md_match.values))

stats_tests.append('\n median snr info:')
stats_tests.append(f'a1 median snr, full: {a1_md_snr}, matched: {a1_md_snr_match}')
stats_tests.append(f'peg median snr, full: {peg_md_snr}, matched: {peg_md_snr_match}')
stats_tests.append(f'test significance for full data: {test_snr}')

stats_tests.append('\ncross-batch heldout:')
stats_tests.append(f'A1-second median r: {np.median(a1_heldout_r)}')
stats_tests.append(f'PEG-second median r: {np.median(peg_heldout_r)}')
stats_tests.append(f'{wilcoxon}')

########################################################################################################################
#################################   SAVE PDFS  #########################################################################
########################################################################################################################

DO_SAVE=True
if DO_SAVE:
    figures_to_save = [
        (fig2, 'fig3_pareto'),
        (fig3a, 'fig4_scatters'),
        (fig3b, 'fig4_bars'),
        (fig5, 'fig6_heldout'),
        (fig6, 'fig7_data_subsets'),
        (fig7, 'fig8_equivalence'),
        (fig8, 'fig9_sparseness'),
        (fig9, 'fig10_snr'),
    ]

    pdf = PdfPages(base_path / 'all_figs.pdf')
    for fig, name in figures_to_save:
        full_path = (base_path / name).with_suffix('.pdf')
        fig.savefig(full_path, format='pdf', dpi=300)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
    pdf.close()

    with open(base_path / 'stats_file.txt', 'w') as stats_file:
        stats_file.write('\n'.join(stats_tests))
    #plt.close('all')  # just to make double sure that everything is closed
