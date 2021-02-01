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
easy = [0]             # pure-tone = 0, low SNR = 1, high SNR = 3
medium = [1]
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

# medians
easy_med = round(A1[A1.difficulty.isin(easy) & A1[sig_col]][yaxis].median(), 3)
medium_med = round(A1[A1.difficulty.isin(medium) & A1[sig_col]][yaxis].median(), 3)
hard_med = round(A1[A1.difficulty.isin(hard) & A1[sig_col]][yaxis].median(), 3)

# pvals 
easy_v_medium = round(ss.ranksums(A1[A1.difficulty.isin(easy) & A1[sig_col]][yaxis], A1[A1.difficulty.isin(medium) & A1[sig_col]][yaxis]).pvalue, 3)
medium_v_hard = round(ss.ranksums(A1[A1.difficulty.isin(medium) & A1[sig_col]][yaxis], A1[A1.difficulty.isin(hard) & A1[sig_col]][yaxis]).pvalue, 3)
easy_v_hard = round(ss.ranksums(A1[A1.difficulty.isin(easy) & A1[sig_col]][yaxis], A1[A1.difficulty.isin(hard) & A1[sig_col]][yaxis]).pvalue, 3)

ax[0].set_title('A1 \n HARD: {0}, MEDIUM: {1}, EASY: {2}, \n'
                'p_em = {3}, p_mh = {4}, p_eh = {5}'.format(easy_med,
                                                            medium_med,
                                                            hard_med,
                                                            easy_v_medium,
                                                            medium_v_hard,
                                                            easy_v_hard))
nplt.ax_remove_box(ax[0])

_a=sns.stripplot(x=sig_col, y=yaxis, data=IC, hue='difficulty', dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=5, ax=ax[1])
ax[1].axhline(0, linestyle='--', lw=2, color='grey')
ax[1].legend().remove()

# medians
easy_med = round(IC[IC.difficulty.isin(easy) & IC[sig_col]][yaxis].median(), 3)
medium_med = round(IC[IC.difficulty.isin(medium) & IC[sig_col]][yaxis].median(), 3)
hard_med = round(IC[IC.difficulty.isin(hard) & IC[sig_col]][yaxis].median(), 3)

# pvals 
easy_v_medium = round(ss.ranksums(IC[IC.difficulty.isin(easy) & IC[sig_col]][yaxis], IC[IC.difficulty.isin(medium) & IC[sig_col]][yaxis]).pvalue, 3)
medium_v_hard = round(ss.ranksums(IC[IC.difficulty.isin(medium) & IC[sig_col]][yaxis], IC[IC.difficulty.isin(hard) & IC[sig_col]][yaxis]).pvalue, 3)
easy_v_hard = round(ss.ranksums(IC[IC.difficulty.isin(easy) & IC[sig_col]][yaxis], IC[IC.difficulty.isin(hard) & IC[sig_col]][yaxis]).pvalue, 3)

ax[1].set_title('IC \n HARD: {0}, MEDIUM: {1}, EASY: {2}, \n'
                'p_em = {3}, p_mh = {4}, p_eh = {5}'.format(easy_med,
                                                            medium_med,
                                                            hard_med,
                                                            easy_v_medium,
                                                            medium_v_hard,
                                                            easy_v_hard))
nplt.ax_remove_box(ax[1])
f.tight_layout()


if save_fig:
    f.savefig(os.path.join(save_path, 'figS4_difficulty_percell.pdf'))

# only look at cells that were recorded in all conditions
f, ax = plt.subplots(2, 2, figsize=(6, 8))

# significant cells only
A1g = A1[A1[sig_col]].groupby(by=['cellid', 'difficulty']).mean()
mi = A1g[[yaxis]]
cells = []
for c in mi.index.get_level_values('cellid').unique():
    y = mi.loc[pd.IndexSlice[c, :], :].values
    if len(y) == 3:
        ax[0, 0].plot([0, 1, 2], y, 'k', alpha=0.3)
        cells.append(c)
ax[0, 0].plot([0, 1, 2], mi.loc[cells].groupby(by='difficulty').mean(), 'o-', lw=2, color='k')
ax[0, 0].set_xticks([0, 1, 2])
ax[0, 0].set_xticklabels(['easy', 'medium', 'hard'])
ax[0, 0].set_xlabel('Task block condition')
ax[0, 0].set_ylabel(yaxis)
ax[0, 0].axhline(0, linestyle='--', color='k')

ax[0, 1].errorbar([0, 1, 2], mi.loc[cells].groupby(by='difficulty').mean().values.squeeze(), \
                            yerr=mi.loc[cells].groupby(by='difficulty').sem().values.squeeze(), lw=2, color='k')
ax[0, 1].set_xticks([0, 1, 2])
ax[0, 1].set_xticklabels(['easy', 'medium', 'hard'])
ax[0, 1].set_xlabel('Task block condition')
ax[0, 1].set_ylabel(yaxis)
ax[0, 1].axhline(0, linestyle='--', color='k')


# all cells
A1g = A1.groupby(by=['cellid', 'difficulty']).mean()
mi = A1g[[yaxis]]
cells = []
for c in mi.index.get_level_values('cellid').unique():
    y = mi.loc[pd.IndexSlice[c, :], :].values
    if len(y) == 3:
        ax[1, 0].plot([0, 1, 2], y, 'k', alpha=0.3)
        cells.append(c)
ax[1, 0].plot([0, 1, 2], mi.loc[cells].groupby(by='difficulty').mean(), 'o-', lw=2, color='k')
ax[1, 0].set_xticks([0, 1, 2])
ax[1, 0].set_xticklabels(['easy', 'medium', 'hard'])
ax[1, 0].set_xlabel('Task block condition')
ax[1, 0].set_ylabel(yaxis)
ax[1, 0].axhline(0, linestyle='--', color='k')

ax[1, 1].errorbar([0, 1, 2], mi.loc[cells].groupby(by='difficulty').mean().values.squeeze(), \
                            yerr=mi.loc[cells].groupby(by='difficulty').sem().values.squeeze(), lw=2, color='k')
ax[1, 1].set_xticks([0, 1, 2])
ax[1, 1].set_xticklabels(['easy', 'medium', 'hard'])
ax[1, 1].set_xlabel('Task block condition')
ax[1, 1].set_ylabel(yaxis)
ax[1, 1].axhline(0, linestyle='--', color='k')


f.tight_layout()

plt.show()

if save_fig:
    f.savefig(os.path.join(save_path, 'figS4_difficulty_sum.pdf'))

