import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from numpy.random import normal

import helpers as helper

from nems import get_setting
import nems.plots.api as nplt

dump_path = get_setting('NEMS_RESULTS_DIR')

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = True

r0_threshold = 0.5
octave_cutoff = 0.5
#perf_stat='DI'
perf_stat='dp'

# important options
yaxis = 'r_pupil_unique'
#yaxis = 'r_task_unique'

#yaxis = 'MI_task_unique'
#yaxis = 'MI_task_unique_abs'
#yaxis = 'r_task'

sig_col = 'area'     # sig_utask (sig unique task effect), sig_task (sig task only), sig_state (sig state effect)
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

A1['animal']=A1.index.copy().str[:3]
IC['animal']=IC.index.copy().str[:3]
A1['MI_task_unique_abs'] = np.abs(A1['MI_task_unique'])
IC['MI_task_unique_abs'] = np.abs(IC['MI_task_unique'])
A1['area']='A1'
IC['area']='IC'
# compress DI slightly to avoid explosions
#A1['dp']=-np.log(100/A1.DI-1)
A1['dp']=-np.log(100/(A1.DI*0.92+6)-1)
A1.loc[A1.dp>3.5,'dp']=3.5
#IC['dp']=-np.log(100/IC.DI-1)
IC['dp']=-np.log(100/(IC.DI*0.92+6)-1)
IC.loc[IC.dp>3.5,'dp']=3.5

d = pd.concat((A1,IC))

split=82
d['di_class']='good'
d.loc[d.DI<split, 'di_class']='bad'

jitter_mag = 0.015
ms = 4

# Good vs. bad behavior
f1, ax = plt.subplots(1, 3, figsize=(8, 3), sharey='row')

sns.stripplot(x=sig_col, y=yaxis, data=d, hue='di_class', dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=ms, ax=ax[0])
ax[0].axhline(0, linestyle='--', lw=2, color='grey')

# medians
bad_med_a1 = round(d[(d.di_class=='bad') & (d.area=='A1')][yaxis].median(), 3)
good_med_a1 = round(d[(d.di_class=='good')  & (d.area=='A1')][yaxis].median(), 3)
bad_med_ic = round(d[(d.di_class=='bad') & (d.area=='IC')][yaxis].median(), 3)
good_med_ic = round(d[(d.di_class=='good')  & (d.area=='IC')][yaxis].median(), 3)

# pvals
bad_v_good_a1 = round(ss.ranksums(d[(d.di_class=='bad') & (d.area=='A1')][yaxis],
                                  d[(d.di_class=='good') & (d.area=='A1')][yaxis]).pvalue, 3)
bad_v_good_ic = round(ss.ranksums(d[(d.di_class=='bad') & (d.area=='IC')][yaxis],
                                  d[(d.di_class=='good') & (d.area=='IC')][yaxis]).pvalue, 3)

ax[0].set_title('A1: bad: {0}, good {1} p: {2:.3f}\nIC: bad: {3}, good {4} p: {5:.3f}'.format(
    bad_med_a1, good_med_a1, bad_v_good_a1,
    bad_med_ic, good_med_ic, bad_v_good_ic))
ax[0].set_ylim([-0.02, 0.22])
nplt.ax_remove_box(ax[0])

animal_colors = {'TAR': [244/255, 165/255, 130/255],
                 'bbl': [147/255, 197/255, 222/255],
                 'ley': [5/255, 113/255, 176/255],
                 'BRT': [202/255, 32/255, 39/255]}

for a in A1.animal.unique():
    _k = (d.area=='A1') & (d.animal==a)
    ax[1].plot(d.loc[_k, perf_stat]+normal(0,jitter_mag,np.sum(_k)),
               d.loc[_k, yaxis], '.', markersize=ms, color=animal_colors[a])
r, p = ss.pearsonr(d[(d.area=='A1')][perf_stat], d[(d.area=='A1')][yaxis])
beta = np.polyfit(d[(d.area=='A1')][perf_stat], d[(d.area=='A1')][yaxis],1)

ax[1].set_title(f'A1 r={r:.3f}, p={p:.4f}')
ax[1].set_xlabel(perf_stat)
ax[1].set_ylabel(yaxis)
if perf_stat=='DI':
    x0=np.array([58,102])
else:
    x0=np.array([0.5,3.5])
ax[1].plot(x0,x0*beta[0]+beta[1],'k--')
ax[1].set_xlim(x0)
ax[1].legend(A1.animal.unique(), frameon=False)
nplt.ax_remove_box(ax[1])

for a in IC.animal.unique():
    _k = (d.area=='IC') & (d.animal==a)
    ax[2].plot(d.loc[_k, perf_stat]+normal(0,jitter_mag,np.sum(_k)),
               d.loc[_k, yaxis], '.', markersize=ms, color=animal_colors[a])
r, p = ss.pearsonr(d[(d.area=='IC')][perf_stat], d[(d.area=='IC')][yaxis])
beta = np.polyfit(d[(d.area=='IC')][perf_stat], d[(d.area=='IC')][yaxis],1)

ax[2].set_title(f'IC r={r:.3f}, p={p:.4f}')
ax[2].set_xlabel(perf_stat)
ax[2].set_ylabel(yaxis)
if perf_stat=='DI':
    x0=np.array([58,102])
else:
    x0=np.array([0.5,3.5])
ax[2].plot(x0,x0*beta[0]+beta[1],'k--')
ax[2].set_xlim(x0)
ax[2].legend(IC.animal.unique(), frameon=False)
nplt.ax_remove_box(ax[2])

f1.tight_layout()



f2,ax = plt.subplots(2,2, figsize=(5, 5), sharey='row')
sns.stripplot(x='animal', y=yaxis, data=A1, dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=ms, ax=ax[0,0])
ax[0,0].axhline(0, linestyle='--', lw=2, color='grey')
sets_a1 = [A1.loc[A1['animal']==a, yaxis] for a in A1.animal.unique()]
means_a1 = [f'{a}={A1.loc[A1["animal"]==a, yaxis].mean():.3f}' for a in A1.animal.unique()]
F,p = ss.f_oneway(*sets_a1)
ax[0,0].set_title(f'A1 F={F:.3f} p={p:.3e}\n{",".join(means_a1)}')
nplt.ax_remove_box(ax[0,0])

sns.stripplot(x='animal', y=yaxis, data=IC, dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=5, ax=ax[0,1])
ax[0,1].axhline(0, linestyle='--', lw=2, color='grey')
sets_ic = [IC.loc[IC['animal']==a, yaxis] for a in IC.animal.unique()]
means_ic = [f'{a}={IC.loc[IC["animal"]==a, yaxis].mean():.3f}' for a in IC.animal.unique()]
F,p = ss.f_oneway(*sets_ic)
ax[0,1].set_title(f'IC F={F:.3f} p={p:.3e}\n{",".join(means_ic)}')
ax[0,1].set_ylim([-0.02, 0.22])
nplt.ax_remove_box(ax[0,1])

sns.stripplot(x='animal', y=perf_stat, data=A1, dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=ms, ax=ax[1,0])
if perf_stat=='DI':
    ax[1,0].axhline(50, linestyle='--', lw=2, color='grey')
else:
    ax[1,0].axhline(0, linestyle='--', lw=2, color='grey')
sets_a1 = [A1.loc[A1['animal']==a, perf_stat] for a in A1.animal.unique()]
means_a1 = [f'{a}={A1.loc[A1["animal"]==a, perf_stat].mean():.3f}' for a in A1.animal.unique()]
F,p = ss.f_oneway(*sets_a1)
ax[1,0].set_title(f'A1 F={F:.3f} p={p:.3e}\n{",".join(means_a1)}')
nplt.ax_remove_box(ax[1,0])

sns.stripplot(x='animal', y=perf_stat, data=IC, dodge=True, edgecolor='white', linewidth=0.5,
                        marker='o', size=5, ax=ax[1,1])
if perf_stat=='DI':
    ax[1,1].axhline(50, linestyle='--', lw=2, color='grey')
else:
    ax[1,1].axhline(0, linestyle='--', lw=2, color='grey')
sets_ic = [IC.loc[IC['animal']==a, perf_stat] for a in IC.animal.unique()]
means_ic = [f'{a}={IC.loc[IC["animal"]==a, perf_stat].mean():.3f}' for a in IC.animal.unique()]
F,p = ss.f_oneway(*sets_ic)
ax[1,1].set_title(f'IC F={F:.3f} p={p:.3e}\n{",".join(means_ic)}')

nplt.ax_remove_box(ax[1,1])

f2.tight_layout()

plt.show()
from statsmodels.graphics.api import interaction_plot, abline_plot

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

formula = f'{yaxis} ~ C(animal) + ' + perf_stat
lm = ols(formula, A1).fit()
table1 = anova_lm(lm)
print(table1)
lm = ols(formula, IC).fit()
table1 = anova_lm(lm)
print(table1)

# same stats, include pupil variance as an explanatory variable
path = os.path.dirname(helper.__file__)
pup_var = path + '/pupilVariance.csv'
pvar = pd.read_csv(pup_var, index_col=0)
A1 = A1.merge(pvar, on='cellid')
IC = IC.merge(pvar, on='cellid')
formula = f'{yaxis} ~ C(animal) + pnorm_var + ' + perf_stat
lm = ols(formula, A1).fit()
table1 = anova_lm(lm)
print(table1)
lm = ols(formula, IC).fit()
table1 = anova_lm(lm)
print(table1)

# SU vs. MU
f3, ax = plt.subplots(1, 2, figsize=(6, 3), sharey='row')

for _i, _yax in enumerate(['r_task_unique', 'r_pupil_unique']):
    sns.stripplot(x=sig_col, y=_yax, data=d, hue='SU', dodge=True, edgecolor='white', linewidth=0.5,
                            marker='o', size=ms, ax=ax[_i])
    ax[_i].axhline(0, linestyle='--', lw=2, color='grey')

    # medians
    bad_med_a1 = round(d[(d.SU) & (d.area=='A1')][_yax].median(), 3)
    good_med_a1 = round(d[(~d.SU)  & (d.area=='A1')][_yax].median(), 3)
    bad_med_ic = round(d[(d.SU) & (d.area=='IC')][_yax].median(), 3)
    good_med_ic = round(d[(~d.SU)  & (d.area=='IC')][_yax].median(), 3)

    # pvals
    bad_v_good_a1 = round(ss.ranksums(d[d.SU & (d.area=='A1')][_yax],
                                      d[~d.SU & (d.area=='A1')][_yax]).pvalue, 3)
    bad_v_good_ic = round(ss.ranksums(d[d.SU & (d.area=='IC')][_yax],
                                      d[~d.SU & (d.area=='IC')][_yax]).pvalue, 3)

    ax[_i].set_title('{6}, A1: SU: {0}, MU {1} p: {2:.3f}\nIC: bad: {3}, good {4} p: {5:.3f}'.format(
        bad_med_a1, good_med_a1, bad_v_good_a1,
        bad_med_ic, good_med_ic, bad_v_good_ic, _yax))
    nplt.ax_remove_box(ax[_i])

if save_fig:
    f1.savefig(os.path.join(save_path, f'figSn_performance_sum_{yaxis}.pdf'))
    f2.savefig(os.path.join(save_path, f'figSn_animal_sum_{yaxis}.pdf'))
    f3.savefig(os.path.join(save_path, f'figSn_su_mu_comp_{yaxis}.pdf'))

