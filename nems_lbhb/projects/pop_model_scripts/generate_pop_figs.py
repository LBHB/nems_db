from pathlib import Path
import datetime
import os

import numpy as np

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
    MODELGROUPS, POP_MODELGROUPS, HELDOUT, MATCHED, EQUIVALENCE_MODELS_SINGLE, EQUIVALENCE_MODELS_POP,
    POP_MODELS, ALL_FAMILY_POP,
    SIG_TEST_MODELS,
    get_significant_cells, snr_by_batch, NAT4_A1_SITES, NAT4_PEG_SITES, PLOT_STAT, DOT_COLORS, DOT_MARKERS, base_path,
    linux_user, ALL_FAMILY_MODELS, VERSION, count_fits, int_path, a1, peg, single_column_short, single_column_tall,
    column_and_half_short, column_and_half_tall
)

import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 10,
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from nems_lbhb.projects.pop_model_scripts.pareto_pop_plot import model_comp_pareto
from nems_lbhb.projects.pop_model_scripts.summary_plots import scatter_bar
from nems_lbhb.projects.pop_model_scripts.pop_correlation import correlation_histogram
from nems_lbhb.projects.pop_model_scripts.heldout_plots import generate_heldout_plots
from nems_lbhb.projects.pop_model_scripts.matched_snr_plots import plot_matched_snr
from nems_lbhb.projects.pop_model_scripts.partial_est_plot import partial_est_plot
from nems_lbhb.projects.pop_model_scripts.pred_scatter import plot_pred_scatter, bar_mean
from nems_lbhb.projects.pop_model_scripts.snr_batch import sparseness, sparseness_by_batch, sparseness_plot, sparseness_example, sparseness_figs


a1 = 322
peg = 323



########################################################################################################################
#################################   SCATTER/BAR PRED   #################################################################
########################################################################################################################

fig3, ax = plt.subplots(2, 2, figsize=column_and_half_tall)
plot_pred_scatter(a1, [ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[2]], labels=['1D CNN','pop LN'], ax=ax[0,0])
plot_pred_scatter(a1, [ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[0]], labels=['1D CNN','2D CNN'], ax=ax[0,1])
bar_mean(a1, ALL_FAMILY_MODELS, stest=SIG_TEST_MODELS, ax=ax[1,0])
bar_mean(peg, ALL_FAMILY_MODELS, stest=SIG_TEST_MODELS, ax=ax[1,1])
fig3.tight_layout()

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
    _, b_ceiling, model_mean = model_comp_pareto(batch, MODELGROUPS, axes1[i], sig_cells,
                                                 nparms_modelgroups=POP_MODELGROUPS,
                                                 dot_colors=DOT_COLORS, dot_markers=DOT_MARKERS,
                                                 plot_stat=PLOT_STAT, plot_medians=True,
                                                 labeled_models=ALL_FAMILY_MODELS,
                                                 show_legend=show_legend)
    means.append(model_mean)
    batch_name = 'A1' if batch == a1 else 'PEG'
    axes1[i].set_title(f'{batch_name}')
    xlims.extend(axes1[i].get_xlim())
    ylims.extend(axes1[i].get_ylim())

for a in axes1:
    a.set_xlim(np.min(xlims), np.max(xlims))
    a.set_ylim(np.min(ylims), np.max(ylims))




########################################################################################################################
#################################   HELDOUT  ###########################################################################
########################################################################################################################
fig5, axes3 = plt.subplots(1, 2, figsize=column_and_half_short, sharex=True, sharey=True)
tests1, sig1, r1, m1, mds1 = generate_heldout_plots(a1, 'A1', ax=axes3[0])
tests2, sig2, r2, m2, mds2 = generate_heldout_plots(peg, 'PEG', ax=axes3[1])
axes3[1].set_ylabel('')

print("\n\nheldout vs matched, Sig. tests for batch %d:" % a1)
print(tests1)
print("median diffs:")
print(mds1)
print("\n")
print("heldout vs matched, Sig. tests for batch %d:" % peg)
print(tests2)
print("median diffs:")
print(mds2)



########################################################################################################################
#################################   DATA SUBSETS  ######################################################################
########################################################################################################################


fig6 = partial_est_plot(batch=a1, PLOT_STAT='r_ceiling', figsize=column_and_half_short)


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

fig7, axes2 = plt.subplots(1, 2, figsize=column_and_half_short)
a1_corr, a1_p, a1_t = correlation_histogram(
    a1, 'A1', load_path=a1_corr_path, force_rerun=False, use_pop_models=True, ax=axes2[0])
peg_corr, peg_p, peg_t = correlation_histogram(
    peg, 'PEG', load_path=peg_corr_path, force_rerun=False, use_pop_models=True, ax=axes2[1])

fig7.tight_layout()
print("\n\ncorrelation histograms, A1 sig tests: p=%s,  t=%s" % (a1_p, a1_t))
print("correlation histograms, PEG sig tests: p=%s,  t=%s" % (peg_p, peg_t))


########################################################################################################################
##################################   SPARSENESS   ######################################################################
########################################################################################################################

fig8 = sparseness_figs()


########################################################################################################################
#################################   SNR  ###############################################################################
########################################################################################################################

a1_snr_path = int_path / str(a1) / 'snr_nat4.csv'
peg_snr_path = int_path / str(peg) / 'snr_nat4.csv'

fig9a, ax4a = plt.subplots(figsize=single_column_short)
fig9b, ax4b = plt.subplots(figsize=single_column_short)  # but actually resize manually in illustrator, as needed.
u_c1, p_c1, u_LN, p_LN, u_dnn, p_dnn = plot_matched_snr(a1, peg, a1_snr_path, peg_snr_path, plot_sanity_check=False,
                                                        ax=ax4a, inset_ax=ax4b)
print("matched snr Sig. tests:\n"
      "conv1Dx2: T: %.4e, p: %.4e\n"
      "LN: T: %.4e, p: %.4e\n"
      "dnn1_single: T: %.4e, p: %.4e\n" % (u_c1, p_c1, u_LN, p_LN, u_dnn, p_dnn))

########################################################################################################################
#################################   SAVE PDFS  #########################################################################
########################################################################################################################

DO_SAVE=True
if DO_SAVE:
    figures_to_save = [
        (fig2, 'fig2_pareto'),
        (fig3, 'fig3_scatter_bar'),
        (fig5, 'fig5_heldout'),
        (fig6, 'fig6_data_subsets'),
        (fig7, 'fig7_equivalence'),
        (fig8, 'fig8_sparseness'),
        (fig9a, 'fig9_snr'),
        (fig9b, 'fig9_snr_inset'),
    ]

    pdf = PdfPages(base_path / 'all_figs.pdf')
    for fig, name in figures_to_save:
        full_path = (base_path / name).with_suffix('.pdf')
        fig.savefig(full_path, format='pdf', dpi='figure')
        pdf.savefig(fig, dpi='figure')
        plt.close(fig)
    pdf.close()

    #plt.close('all')  # just to make double sure that everything is closed
