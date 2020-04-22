import helpers as helper
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from nems import get_setting
dump_path = get_setting('NEMS_RESULTS_DIR')

r0_threshold = 0
octave_cutoff = 0.5
yaxis = 'MI_task_unique'
sig_task_only = True
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
if sig_pupil_only:
    A1 = A1[A1.sig_upupil]
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
if sig_pupil_only:
    IC = IC[IC.sig_upupil]

f, ax = plt.subplots(2, 1, figsize=(6, 6))

# A1
difficulty = A1.difficulty.unique()
A1g = A1.groupby(by='difficulty')
ax[0].bar(difficulty, A1g[yaxis].mean(), yerr=A1g[yaxis].sem(), color='lightgrey', edgecolor='k', lw=2)
ax[0].set_xticks(difficulty)
ax[0].set_xticklabels(['n={}'.format(n) for n in A1g[yaxis].count()])
ax[0].set_ylabel(yaxis)
ax[0].set_xlabel('Task difficulty')
ax[0].set_title("A1")

# IC
difficulty = IC.difficulty.unique()
ICg = IC.groupby(by='difficulty')
ax[1].bar(difficulty, ICg[yaxis].mean(), yerr=ICg[yaxis].sem(), color='lightgrey', edgecolor='k', lw=2)
ax[1].set_xticks(difficulty)
ax[1].set_xticklabels(['n={}'.format(n) for n in ICg[yaxis].count()])
ax[1].set_ylabel(yaxis)
ax[1].set_xlabel('Task difficulty')
ax[1].set_title("IC")

f.tight_layout()


# only look at cells that were recorded in multiple conditions
f, ax = plt.subplots(1, 1, figsize=(4, 8))
A1g = A1.groupby(by=['cellid', 'difficulty']).mean()
mi = A1g[[yaxis]]
pt = []
easy = []
hard = []
for c in mi.index.get_level_values('cellid').unique():
    x = mi.loc[pd.IndexSlice[c, :], :].index.get_level_values('difficulty').values
    x[x==3] = 2
    y = mi.loc[pd.IndexSlice[c, :], :].values
    if len(y) > 2:
        y *= np.sign(y[2])
        pt.append(y[0])
        easy.append(y[1])
        hard.append(y[2])
        ax.plot(x, y, 'red', alpha=0.3)
ax.plot([0, 1, 2], [np.mean(pt), np.mean(easy), np.mean(hard)], 'o-', lw=2, color='r')
ax.axhline(0, linestyle='--', color='k')

