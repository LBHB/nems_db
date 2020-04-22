import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

import helpers as helper

from nems import get_setting

dump_path = get_setting('NEMS_RESULTS_DIR')

r0_threshold = 0
octave_cutoff = 0.5
yaxis = 'MI_task_unique'
sig_task_only = False
sig_pupil_only = False

dump_results = 'd_pup_afl_sdexp.csv'
model_string = 'st.pup.afl'
p0_model = 'st.pup0.afl'
b0_model = 'st.pup.afl0'
shuf_model = 'st.pup0.afl0'

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
if sig_task_only:
    A1 = A1[A1.sig_utask]

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
if sig_task_only:
    IC = IC[IC.sig_utask]

# group cells according to the sign of mean mi (across all conditions)
A1g = A1.groupby(by=['difficulty', 'cellid'])[[yaxis]].mean()
sign = A1g.groupby(by='cellid').mean().apply(np.sign)
sign = sign.rename(columns={yaxis: 'sign'})
A1g = A1g.merge(sign, left_on='cellid', right_on='cellid', right_index=True).groupby(by=['difficulty', 'cellid']).mean()

A1_mean = A1g.groupby(by=['difficulty', 'sign']).mean()
A1_sem = A1g.groupby(by=['difficulty', 'sign']).sem()

ICg = IC.groupby(by=['difficulty', 'cellid'])[[yaxis]].mean()
sign = ICg.groupby(by='cellid').mean().apply(np.sign)
sign = sign.rename(columns={yaxis: 'sign'})
ICg = ICg.merge(sign, left_on='cellid', right_on='cellid', right_index=True).groupby(by=['difficulty', 'cellid']).mean()

IC_mean = ICg.groupby(by=['difficulty', 'sign']).mean()
IC_sem = ICg.groupby(by=['difficulty', 'sign']).sem()

f, ax = plt.subplots(2, 1, figsize=(6, 6))

# A1
difficulty = np.arange(0, len(set(A1_mean.index.get_level_values('difficulty').values)))
ax[0].bar(difficulty, A1_mean.loc[pd.IndexSlice[:, 1], yaxis].values, yerr=A1_sem.loc[pd.IndexSlice[:, 1], yaxis].values,
             color='lightgrey', edgecolor='k', lw=2, width=0.4)
ax[0].bar(difficulty, A1_mean.loc[pd.IndexSlice[:, -1], yaxis].values, yerr=A1_sem.loc[pd.IndexSlice[:, -1], yaxis].values,
             color='coral', edgecolor='k', lw=2, width=0.4)
ax[0].set_xticks(difficulty)
ax[0].set_xticklabels(['n={0}, {1}'.format(n, n2) for n, n2 in 
                            zip(A1g[A1g.sign==1].groupby(by='difficulty').count()[yaxis], 
                                A1g[A1g.sign==-1].groupby(by='difficulty').count()[yaxis])])
ax[0].set_ylabel(yaxis)
ax[0].set_xlabel('Task difficulty')
ax[0].set_title("A1")

# IC
difficulty = np.arange(0, len(set(IC_mean.index.get_level_values('difficulty').values)))
ax[1].bar(difficulty, IC_mean.loc[pd.IndexSlice[:, 1], yaxis].values, yerr=IC_sem.loc[pd.IndexSlice[:, 1], yaxis].values,
             color='lightgrey', edgecolor='k', lw=2, width=0.4)
ax[1].bar(difficulty, IC_mean.loc[pd.IndexSlice[:, -1], yaxis].values, yerr=IC_sem.loc[pd.IndexSlice[:, -1], yaxis].values,
             color='coral', edgecolor='k', lw=2, width=0.4)
ax[1].set_xticks(difficulty)
ax[1].set_xticklabels(['n={0}, {1}'.format(n, n2) for n, n2 in 
                            zip(ICg[ICg.sign==1].groupby(by='difficulty').count()[yaxis], 
                                ICg[ICg.sign==-1].groupby(by='difficulty').count()[yaxis])])
ax[1].set_ylabel(yaxis)
ax[1].set_xlabel('Task difficulty')
ax[1].set_title("IC")

f.tight_layout()


# only look at cells that were recorded in multiple conditions
f, ax = plt.subplots(1, 1, figsize=(5, 6))
A1g = A1.groupby(by=['cellid', 'difficulty']).mean()
mi = A1g[[yaxis]]
pt = []
easy = []
hard = []
ptn = []
easyn = []
hardn = []
cells = []
for c in mi.index.get_level_values('cellid').unique():
    x = mi.loc[pd.IndexSlice[c, :], :].index.get_level_values('difficulty').values
    x[x==3] = 2
    y = mi.loc[pd.IndexSlice[c, :], :].values
    if len(y) > 2:
        if np.sign(y.mean()) == 1:
            color = 'k'
            pt.append(y[0])
            easy.append(y[1])
            hard.append(y[2])
        else:
            color = 'r'
            ptn.append(y[0])
            easyn.append(y[1])
            hardn.append(y[2])
        ax.plot(x, y, color, alpha=0.3)
        cells.append(c)
ax.plot([0, 1, 2], [np.mean(pt), np.mean(easy), np.mean(hard)], 'o-', lw=2, color='k')
ax.plot([0, 1, 2], [np.mean(ptn), np.mean(easyn), np.mean(hardn)], 'o-', lw=2, color='r')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Pure-tone', 'easy', 'hard'])
ax.set_xlabel('Task block condition')
ax.set_ylabel(yaxis)
ax.axhline(0, linestyle='--', color='k')

f.tight_layout()

plt.show()