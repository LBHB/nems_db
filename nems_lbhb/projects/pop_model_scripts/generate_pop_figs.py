from pathlib import Path
import datetime
import os

import numpy as np
import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 12,
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from nems.utils import ax_remove_box

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (MODELGROUPS, POP_MODELGROUPS, HELDOUT, MATCHED, EQUIVALENCE_MODELS_SINGLE,
                             EQUIVALENCE_MODELS_POP, SIG_TEST_MODELS, get_significant_cells, snr_by_batch,
                             NAT4_A1_SITES, NAT4_PEG_SITES, PLOT_STAT, DOT_COLORS, DOT_MARKERS)

from nems_lbhb.projects.pop_model_scripts.pareto_pop_plot import model_comp_pareto
from nems_lbhb.projects.pop_model_scripts.pop_correlation import correlation_histogram
from nems_lbhb.projects.pop_model_scripts.heldout_plots import generate_heldout_plots
from nems_lbhb.projects.pop_model_scripts.matched_snr_plots import plot_matched_snr

a1 = 322
peg = 323
# TODO: adjust figure sizes
single_column_short = (3.5, 3)
single_column_tall = (3.5, 6)
column_and_half_short = (5, 3)
column_and_half_tall = (5, 6)
#inset = (1, 1)  # easier to just resize manually, making it this smaller makes things behave weirdly

########################################################################################################################
#################################   PARETO  ############################################################################
########################################################################################################################
means = []
fig1, axes1 = plt.subplots(1, 2, figsize=column_and_half_short)
xlims = []
ylims = []
for i, batch in enumerate([a1, peg]):
    sig_cells = get_significant_cells(batch, SIG_TEST_MODELS)
    show_legend = (i==0)
    _, b_ceiling, model_mean = model_comp_pareto(batch, MODELGROUPS, axes1[i], sig_cells,
                                                 nparms_modelgroups=POP_MODELGROUPS,
                                                 dot_colors=DOT_COLORS, dot_markers=DOT_MARKERS,
                                                 plot_stat=PLOT_STAT, plot_medians=True,
                                                 labeled_models=SIG_TEST_MODELS,
                                                 show_legend=show_legend)
    means.append(model_mean)
    batch_name = 'A1' if batch == a1 else 'PEG'
    axes1[i].set_title(f'{batch_name}')
    xlims.extend(axes1[i].get_xlim())
    ylims.extend(axes1[i].get_ylim())

xlims = np.array(xlims)
ylims = np.array(ylims)
min_x = xlims.min()
max_x = xlims.max()
min_y = ylims.min()
max_y = ylims.max()
for a in axes1:
    a.set_xlim(min_x, max_x)
    a.set_ylim(min_y, max_y)



########################################################################################################################
#################################   EQUIVALENCE  #######################################################################
########################################################################################################################
a1_corr_path = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(a1) / 'corr_nat4.pkl'
a1_corr_path_pop = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(a1) / 'corr_nat4_pop.pkl'
peg_corr_path = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(peg) / 'corr_nat4.pkl'
peg_corr_path_pop = Path('/auto/users/jacob/notes/new_equivalence_results/')  / str(peg) / 'corr_nat4_pop.pkl'

fig2, axes2 = plt.subplots(2, 1, figsize=single_column_tall)
a1_corr, a1_p, a1_t = correlation_histogram(a1, 'A1', save_path=a1_corr_path, load_path=a1_corr_path, force_rerun=False,
                                            ax=axes2[0])
peg_corr, peg_p, peg_t = correlation_histogram(peg, 'PEG', save_path=a1_corr_path, load_path=peg_corr_path,
                                               force_rerun=False, ax=axes2[1])
print("\n\ncorrelation histograms, A1 sig tests: p=%s,  t=%s" % (a1_p, a1_t))
print("correlation histograms, PEG sig tests: p=%s,  t=%s" % (peg_p, peg_t))



########################################################################################################################
#################################   HELDOUT  ###########################################################################
########################################################################################################################
fig3, axes3 = plt.subplots(2, 1, figsize=single_column_tall)
tests1, sig1, r1, m1, mds1 = generate_heldout_plots(a1, 'A1', ax=axes3[0], hide_xaxis=True)
tests2, sig2, r2, m2, mds2 = generate_heldout_plots(peg, 'PEG', ax=axes3[1])

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
#################################   SNR  ###############################################################################
########################################################################################################################
a1_snr_path = Path('/auto/users/jacob/notes/new_equivalence_results/') / str(a1) / 'snr_nat4.pkl'
peg_snr_path = Path('/auto/users/jacob/notes/new_equivalence_results/') / str(peg) / 'snr_nat4.pkl'

fig4a, ax4a = plt.subplots(figsize=single_column_short)
fig4b, ax4b = plt.subplots(figsize=single_column_short)  # but actually resize manually in illustrator, as needed.
u_c1, p_c1, u_LN, p_LN, u_dnn, p_dnn = plot_matched_snr(a1, peg, a1_snr_path, peg_snr_path, plot_sanity_check=False,
                                                                 ax=ax4a, inset_ax=ax4b)
print("matched snr Sig. tests:\n"
      "conv1Dx2: T: %.4e, p: %.4e\n"
      "LN: T: %.4e, p: %.4e\n"
      "dnn1_single: T: %.4e, p: %.4e\n" % (u_c1, p_c1, u_LN, p_LN, u_dnn, p_dnn))



########################################################################################################################
#################################   DATA SUBSETS  ######################################################################
########################################################################################################################

# TODO  (SVD running new models, then will send a script for generating the plots



########################################################################################################################
#################################   SAVE PDFS  #########################################################################
########################################################################################################################
figures_base_path = Path('/auto/users/jacob/notes/pop_model_figs/')
date = str(datetime.datetime.now()).split(' ')[0]
base_path = figures_base_path / date
figures_to_save = [
    (fig1, 'pareto'),
    (fig2, 'equivalence'),
    (fig3, 'heldout'),
    (fig4a, 'snr'),
    (fig4b, 'snr_inset')
    #(fig5, 'data_subsets') # TODO
]

if not base_path.is_dir():
    base_path.mkdir(parents=True, exist_ok=True)
pdf = PdfPages(base_path / 'all_figs.pdf')
for fig, name in figures_to_save:
    full_path = (base_path / name).with_suffix('.pdf')
    fig.savefig(full_path, format='pdf', dpi='figure')
    pdf.savefig(fig, dpi='figure')
    plt.close(fig)
pdf.close()
plt.close('all')  # just to make double sure that everything is closed
