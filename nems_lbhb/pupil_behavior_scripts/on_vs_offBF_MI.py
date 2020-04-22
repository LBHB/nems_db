import helpers as helper
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import nems.db as nd
from nems import get_setting
dump_path = get_setting('NEMS_RESULTS_DIR')

r0_threshold = 0
octave_cutoff = 0.3
yaxis_task = 'MI_task_unique'

dump_results = 'd_pup_afl_sdexp.csv'
model_string = 'st.pup.afl'
p0_model = 'st.pup0.afl'
b0_model = 'st.pup.afl0'
shuf_model = 'st.pup0.afl0'
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


f, ax = plt.subplots(2, 2, figsize=(8, 8), sharey='row')

# plot rtest for pxf and pxf0 models
m = 'psth.fs20.pup-ld-st.pup.afl.pxf-ref-psthfr.s_sdexp.S_jk.nf20-basic'
sql = "SELECT cellid, r_test, se_test FROM Results WHERE batch=307 and modelname='{}'".format(m)
r = nd.pd_query(sql)
r = r[~r.cellid.str.contains('AMT')]

m0 = 'psth.fs20.pup-ld-st.pup.afl.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'
sql = "SELECT cellid, r_test, se_test FROM Results WHERE batch=307 and modelname='{}'".format(m0)
r0 = nd.pd_query(sql)
r0 = r0[~r0.cellid.str.contains('AMT')]

r = r.merge(r0, on=['cellid'])

sig_cells = r[(r['r_test_x'] - r['r_test_y']) > (r['se_test_x'] + r['se_test_y'])].cellid.unique()

ax[0, 0].scatter(r['r_test_y'], r['r_test_x'], color='grey', edgecolor='white', s=50)
ax[0, 0].scatter(r[r.cellid.isin(sig_cells)]['r_test_y'], 
                r[r.cellid.isin(sig_cells)]['r_test_x'], color='k', edgecolor='white', s=50)
ax[0, 0].plot([0, 1], [0, 1], 'k--')
ax[0, 0].set_xlabel('pxf0, median: {}'.format(round(r['r_test_y'].median(), 3)))
ax[0, 0].set_ylabel('pxf, median: {}'.format(round(r['r_test_x'].median(), 3)))
pval = round(ss.wilcoxon(r['r_test_y'], r['r_test_x']).pvalue, 3)
ax[0, 0].set_title('A1, pval: {}'.format(pval))

m = 'psth.fs20.pup-ld-st.pup.afl.pxf-ref-psthfr.s_sdexp.S_jk.nf20-basic'
sql = "SELECT cellid, r_test, se_test FROM Results WHERE batch=309 and modelname='{}'".format(m)
r = nd.pd_query(sql)
r = r[~r.cellid.str.contains('AMT')]

m0 = 'psth.fs20.pup-ld-st.pup.afl.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'
sql = "SELECT cellid, r_test, se_test FROM Results WHERE batch=309 and modelname='{}'".format(m0)
r0 = nd.pd_query(sql)
r0 = r0[~r0.cellid.str.contains('AMT')]

r = r.merge(r0, on=['cellid'])

sig_cells = r[(r['r_test_x'] - r['r_test_y']) > (r['se_test_x'] + r['se_test_y'])].cellid.unique()

ax[0, 1].scatter(r['r_test_y'], r['r_test_x'], color='grey', edgecolor='white', s=50)
ax[0, 1].scatter(r[r.cellid.isin(sig_cells)]['r_test_y'], 
                r[r.cellid.isin(sig_cells)]['r_test_x'], color='k', edgecolor='white', s=50)
ax[0, 1].plot([0, 1], [0, 1], 'k--')
ax[0, 1].set_xlabel('pxf0, median: {}'.format(round(r['r_test_y'].median(), 3)))
ax[0, 1].set_ylabel('pxf, median: {}'.format(round(r['r_test_x'].median(), 3)))
pval = round(ss.wilcoxon(r['r_test_y'], r['r_test_x']).pvalue, 3)
ax[0, 1].set_title('IC, pval: {}'.format(pval))


sns.stripplot(x='sig_utask', y=yaxis_task, data=A1, hue='ON_BF', dodge=True, edgecolor='white', marker='o', ax=ax[1, 0])
ax[1, 0].axhline(0, linestyle='--', lw=2, color='grey')
pval = round(ss.ranksums(A1[A1.ON_BF & A1.sig_utask][yaxis_task], A1[~A1.ON_BF & A1.sig_utask][yaxis_task]).pvalue, 3)
off_med = round(A1[~A1.ON_BF & A1.sig_utask][yaxis_task].median(), 3)
on_med = round(A1[A1.ON_BF & A1.sig_utask][yaxis_task].median(), 3)
ax[1, 0].set_title('A1 \n ON: {0}, OFF: {1}, pval: {2}'.format(on_med, off_med, pval))


sns.stripplot(x='sig_utask', y=yaxis_task, data=IC, hue='ON_BF', dodge=True, edgecolor='white', marker='o', ax=ax[1, 1])
ax[1, 1].axhline(0, linestyle='--', lw=2, color='grey')
pval = round(ss.ranksums(IC[IC.ON_BF & IC.sig_utask][yaxis_task], IC[~IC.ON_BF & IC.sig_utask][yaxis_task]).pvalue, 3)
off_med = round(IC[~IC.ON_BF & IC.sig_utask][yaxis_task].median(), 3)
on_med = round(IC[IC.ON_BF & IC.sig_utask][yaxis_task].median(), 3)
ax[1, 1].set_title('IC \n ON: {0}, OFF: {1}, pval: {2}'.format(on_med, off_med, pval))


f.tight_layout()

f.canvas.set_window_title('A1')





if 0:
    # plot individual model results for gain, MI, dc pupil and task
    f, ax = helper.stripplot_df(A1, hue='ON_BF', group_files=group_files)
    f.canvas.set_window_title('A1')
    f, ax = helper.stripplot_df(IC, hue='ON_BF', group_files=group_files)
    f.canvas.set_window_title('IC')

plt.show()