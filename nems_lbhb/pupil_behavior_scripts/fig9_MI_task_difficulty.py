import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

import helpers as helper

from nems import get_setting
import nems.plots.api as nplt

dump_path = get_setting('NEMS_RESULTS_DIR')

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False

r0_threshold = 0.5
octave_cutoff = 0.5
yaxis = 'MI_task_unique'  # MI_task_unique, MI_task (task only)
sig_col = 'sig_utask'     # sig_utask (sig unique task effect), sig_task (sig task only), sig_state (sig state effect)
easy = [0,1]             # pure-tone = 0, low SNR = 1, high SNR = 3
hard = [3]

AFL = True
if AFL:
    dump_results = 'd_pup_afl_sdexp.csv'
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

# convert difficulty to True (for hard) and False (for easy)
#A1['difficulty'] = [True if x in hard else False for x in A1.difficulty]
#IC['difficulty'] = [True if x in hard else False for x in IC.difficulty]

# stripplot of MI split by difficulty and task significance
f, ax = plt.subplots(1, 2, figsize=(5, 3), sharey='row')

sns.stripplot(x=sig_col, y=yaxis, data=A1, hue='difficulty', dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=5, ax=ax[0])
ax[0].axhline(0, linestyle='--', lw=2, color='grey')

pval = round(ss.ranksums(A1[A1.difficulty & A1[sig_col]][yaxis], A1[~A1.difficulty & A1[sig_col]][yaxis]).pvalue, 3)
off_med = round(A1[~A1.difficulty & A1[sig_col]][yaxis].median(), 3)
on_med = round(A1[A1.difficulty & A1[sig_col]][yaxis].median(), 3)

pval_ns = round(ss.ranksums(A1[A1.difficulty & ~A1[sig_col]][yaxis], A1[~A1.difficulty & ~A1[sig_col]][yaxis]).pvalue, 3)
off_med_ns = round(A1[~A1.difficulty & ~A1[sig_col]][yaxis].median(), 3)
on_med_ns = round(A1[A1.difficulty & ~A1[sig_col]][yaxis].median(), 3)

ax[0].set_title('A1 \nsig_cells: HARD: {0}, EASY: {1}, p: {2}\n'
                'ns cells: HARD: {3}, EASY: {4}, p: {5}'.format(on_med, off_med, pval, on_med_ns, off_med_ns, pval_ns))
nplt.ax_remove_box(ax[0])

sns.stripplot(x=sig_col, y=yaxis, data=IC, hue='difficulty', dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=5, ax=ax[1])
ax[1].axhline(0, linestyle='--', lw=2, color='grey')

pval = round(ss.ranksums(IC[IC.difficulty & IC[sig_col]][yaxis], IC[~IC.difficulty & IC[sig_col]][yaxis]).pvalue, 3)
off_med = round(IC[~IC.difficulty & IC[sig_col]][yaxis].median(), 3)
on_med = round(IC[IC.difficulty & IC[sig_col]][yaxis].median(), 3)

pval_ns = round(ss.ranksums(IC[IC.difficulty & ~IC[sig_col]][yaxis], IC[~IC.difficulty & ~IC[sig_col]][yaxis]).pvalue, 3)
off_med_ns = round(IC[~IC.difficulty & ~IC[sig_col]][yaxis].median(), 3)
on_med_ns = round(IC[IC.difficulty & ~IC[sig_col]][yaxis].median(), 3)

ax[1].set_title('IC \nsig_cells: HARD: {0}, EASY: {1}, p: {2}\n'
                'ns cells: HARD: {3}, EASY: {4}, p: {5}'.format(on_med, off_med, pval, on_med_ns, off_med_ns, pval_ns))
nplt.ax_remove_box(ax[1])
f.tight_layout()


if save_fig:
    f.savefig(os.path.join(save_path, 'fig9_difficulty_percell.pdf'))

# only look at cells that were recorded in both conditions
f, ax = plt.subplots(2, 2, figsize=(6, 8))

# significant cells only
A1g = A1[A1[sig_col]].groupby(by=['cellid', 'difficulty']).mean()
mi = A1g[[yaxis]]
cells = []
for c in mi.index.get_level_values('cellid').unique():
    y = mi.loc[pd.IndexSlice[c, :], :].values
    if len(y) > 1:
        ax[0, 0].plot([0, 1], y, 'k', alpha=0.3)
        cells.append(c)
ax[0, 0].plot([0, 1], mi.loc[cells].groupby(by='difficulty').mean(), 'o-', lw=2, color='k')
ax[0, 0].set_xticks([0, 1])
ax[0, 0].set_xticklabels(['easy', 'hard'])
ax[0, 0].set_xlabel('Task block condition')
ax[0, 0].set_ylabel(yaxis)
ax[0, 0].axhline(0, linestyle='--', color='k')

ax[0, 1].errorbar([0, 1], mi.loc[cells].groupby(by='difficulty').mean().values.squeeze(), \
                            yerr=mi.loc[cells].groupby(by='difficulty').sem().values.squeeze(), lw=2, color='k')
ax[0, 1].set_xticks([0, 1])
ax[0, 1].set_xticklabels(['easy', 'hard'])
ax[0, 1].set_xlabel('Task block condition')
ax[0, 1].set_ylabel(yaxis)
ax[0, 1].axhline(0, linestyle='--', color='k')


# all cells
A1g = A1.groupby(by=['cellid', 'difficulty']).mean()
mi = A1g[[yaxis]]
cells = []
for c in mi.index.get_level_values('cellid').unique():
    y = mi.loc[pd.IndexSlice[c, :], :].values
    if len(y) > 1:
        ax[1, 0].plot([0, 1], y, 'k', alpha=0.3)
        cells.append(c)
ax[1, 0].plot([0, 1], mi.loc[cells].groupby(by='difficulty').mean(), 'o-', lw=2, color='k')
ax[1, 0].set_xticks([0, 1])
ax[1, 0].set_xticklabels(['easy', 'hard'])
ax[1, 0].set_xlabel('Task block condition')
ax[1, 0].set_ylabel(yaxis)
ax[1, 0].axhline(0, linestyle='--', color='k')

ax[1, 1].errorbar([0, 1], mi.loc[cells].groupby(by='difficulty').mean().values.squeeze(), \
                            yerr=mi.loc[cells].groupby(by='difficulty').sem().values.squeeze(), lw=2, color='k')
ax[1, 1].set_xticks([0, 1])
ax[1, 1].set_xticklabels(['easy', 'hard'])
ax[1, 1].set_xlabel('Task block condition')
ax[1, 1].set_ylabel(yaxis)
ax[1, 1].axhline(0, linestyle='--', color='k')


f.tight_layout()

plt.show()

if save_fig:
    f.savefig(os.path.join(save_path, 'fig9_difficulty_sum.pdf'))

