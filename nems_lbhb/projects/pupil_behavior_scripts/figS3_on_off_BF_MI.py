import os
import helpers as helper
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import nems0.db as nd
from nems import get_setting
import nems0.plots.api as nplt
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob

dump_path = get_setting('NEMS_RESULTS_DIR')

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False

r0_threshold = 0.5
octave_cutoff = 0.5
yaxis_task = 'MI_task_unique'
AFL = True
if AFL:
    dump_results = 'd_pup_afl_sdexp.csv'
    #dump_results = 'd_pup_afl_sdexp_ap1.csv'
    model_string = 'st.pup.afl'
    p0_model = 'st.pup0.afl'
    b0_model = 'st.pup.afl0'
    shuf_model = 'st.pup0.afl0'
else:
    dump_results = 'd_pup_fil_sdexp.csv'
    model_string = 'st.pup.fil'
    p0_model = 'st.pup0.fil'
    b0_model = 'st.pup.fil0'
    shuf_model = 'st.pup0.fil0'

# NOTE: Decided not to show ON/OFF MI for pupil because effects are tiny. Instead, just show
# that pxf performance = pxf0 so there is no interaction between task and pupil effects

A1 = helper.preprocess_sdexp_dump(dump_results,
                                  batch=307,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
A1 = A1[A1.sig_psth]

IC = helper.preprocess_sdexp_dump(dump_results,
                                  batch=309,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
IC = IC[IC.sig_psth]


f, ax = plt.subplots(1, 2, figsize=(5,3), sharey='row')

sns.stripplot(x='sig_utask', y=yaxis_task, data=A1, hue='ON_BF', dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=5, ax=ax[0])
ax[0].axhline(0, linestyle='--', lw=2, color='grey')

pval = round(ss.ranksums(A1[A1.ON_BF & A1.sig_utask][yaxis_task], A1[~A1.ON_BF & A1.sig_utask][yaxis_task]).pvalue, 3)
off_med = round(A1[~A1.ON_BF & A1.sig_utask][yaxis_task].median(), 3)
on_med = round(A1[A1.ON_BF & A1.sig_utask][yaxis_task].median(), 3)
# get bootstrapped pval
A1['site'] = [c[:7] for c in A1.index]
sig_a1_on = get_bootstrapped_sample({s: A1[A1.ON_BF & A1.sig_utask & (A1.site==s)][yaxis_task] for s in A1.site.unique()}, nboot=1000)
sig_a1_off = get_bootstrapped_sample({s: A1[~A1.ON_BF & A1.sig_utask & (A1.site==s)][yaxis_task] for s in A1.site.unique()}, nboot=1000)
pboot, jm = get_direct_prob(sig_a1_on, sig_a1_off)

pval_ns = round(ss.ranksums(A1[A1.ON_BF & ~A1.sig_utask][yaxis_task], A1[~A1.ON_BF & ~A1.sig_utask][yaxis_task]).pvalue, 3)
off_med_ns = round(A1[~A1.ON_BF & ~A1.sig_utask][yaxis_task].median(), 3)
on_med_ns = round(A1[A1.ON_BF & ~A1.sig_utask][yaxis_task].median(), 3)
ns_a1_on = get_bootstrapped_sample({s: A1[A1.ON_BF & ~A1.sig_utask & (A1.site==s)][yaxis_task] for s in A1.site.unique()}, nboot=1000)
ns_a1_off = get_bootstrapped_sample({s: A1[~A1.ON_BF & ~A1.sig_utask & (A1.site==s)][yaxis_task] for s in A1.site.unique()}, nboot=1000)
pboot_ns, jm = get_direct_prob(ns_a1_on, ns_a1_off)

ax[0].set_title('A1 \n sig_cells: ON: {0}, OFF: {1}, pval: {2}, pboot: {6:.4f} \n'
                    'ns cells: ON: {3}, OFF: {4}, pval: {5}, pboot: {7:.4f}'.format(on_med, off_med, pval, on_med_ns, off_med_ns, pval_ns, pboot, pboot_ns))
nplt.ax_remove_box(ax[0])


sns.stripplot(x='sig_utask', y=yaxis_task, data=IC, hue='ON_BF', dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=5, ax=ax[1])
ax[1].axhline(0, linestyle='--', lw=2, color='grey')

pval = round(ss.ranksums(IC[IC.ON_BF & IC.sig_utask][yaxis_task], IC[~IC.ON_BF & IC.sig_utask][yaxis_task]).pvalue, 3)
off_med = round(IC[~IC.ON_BF & IC.sig_utask][yaxis_task].median(), 3)
on_med = round(IC[IC.ON_BF & IC.sig_utask][yaxis_task].median(), 3)
IC['site'] = [c[:7] for c in IC.index]
sig_ic_on = get_bootstrapped_sample({s: IC[IC.ON_BF & IC.sig_utask & (IC.site==s)][yaxis_task] for s in IC.site.unique()}, nboot=1000)
sig_ic_off = get_bootstrapped_sample({s: IC[~IC.ON_BF & IC.sig_utask & (IC.site==s)][yaxis_task] for s in IC.site.unique()}, nboot=1000)
pboot, jm = get_direct_prob(sig_ic_on, sig_ic_off)

pval_ns = round(ss.ranksums(IC[IC.ON_BF & ~IC.sig_utask][yaxis_task], IC[~IC.ON_BF & ~IC.sig_utask][yaxis_task]).pvalue, 3)
off_med_ns = round(IC[~IC.ON_BF & ~IC.sig_utask][yaxis_task].median(), 3)
on_med_ns = round(IC[IC.ON_BF & ~IC.sig_utask][yaxis_task].median(), 3)
ns_ic_on = get_bootstrapped_sample({s: IC[IC.ON_BF & ~IC.sig_utask & (IC.site==s)][yaxis_task] for s in IC.site.unique()}, nboot=1000)
ns_ic_off = get_bootstrapped_sample({s: IC[~IC.ON_BF & ~IC.sig_utask & (IC.site==s)][yaxis_task] for s in IC.site.unique()}, nboot=1000)
pboot_ns, jm = get_direct_prob(ns_ic_on, ns_ic_off)

ax[1].set_title('IC \n sig_cells: ON: {0}, OFF: {1}, pval: {2}, pboot: {6:.4f} \n'
                    'ns cells: ON: {3}, OFF: {4}, pval: {5}, pboot: {7:.4f}'.format(on_med, off_med, pval, on_med_ns, off_med_ns, pval_ns, pboot, pboot_ns))
nplt.ax_remove_box(ax[1])
f.tight_layout()

if 0:
    # plot individual model results for gain, MI, dc pupil and task
    f, ax = helper.stripplot_df(A1, hue='ON_BF', group_files=group_files)
    f.canvas.set_window_title('A1')
    f, ax = helper.stripplot_df(IC, hue='ON_BF', group_files=group_files)
    f.canvas.set_window_title('IC')

plt.show()

if save_fig:
    f.savefig(os.path.join(save_path,'figS3_on_off_BF.pdf'))
