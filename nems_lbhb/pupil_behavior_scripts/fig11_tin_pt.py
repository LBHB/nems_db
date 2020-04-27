"""
Compare ON/OFF MI effects for tone-in-noise blocks vs. pure tone detection blocks

Only look at IC for now.
"""

import os
import helpers as helper
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import nems.db as nd
from nems import get_setting
dump_path = get_setting('NEMS_RESULTS_DIR')

batches = [313]
r0_threshold = 0.5
octave_cutoff = 0.5
yaxis = 'MI_task'
sig_col = 'sig_task'
AFL = True
if AFL:
    dump_results = 'd_afl_sdexp.csv'
    model_string = 'st.afl'
    p0_model = None
    b0_model = 'st.afl0'
    shuf_model = 'st.afl0'
else:
    dump_results = 'd_fil_sdexp.csv'
    model_string = 'st.fil'
    p0_model = None
    b0_model = 'st.fil0'
    shuf_model = 'st.fil0'

IC = []
for batch in batches:
    _IC = helper.preprocess_sdexp_dump(dump_results,
                                    batch=batch,
                                    full_model=model_string,
                                    p0=p0_model,
                                    b0=b0_model,
                                    shuf_model=shuf_model,
                                    r0_threshold=r0_threshold,
                                    octave_cutoff=octave_cutoff,
                                    path=dump_path)
    _IC = _IC[_IC.sig_psth]
    IC.append(_IC)

IC = pd.concat(IC)

# group diff = 1 /3 together (because both tone in noise)
IC['difficulty'] = [True if x in [1, 3] else False for x in IC['difficulty']]

# make a plot for pure-tone only cells and another plot for tone in noise
IC_pt = IC[IC.difficulty==False]
IC_tin = IC[IC.difficulty==True]

f, ax = plt.subplots(1, 2, figsize=(8, 4), sharey='row')

sns.stripplot(x=sig_col, y=yaxis, data=IC_pt, hue='ON_BF', dodge=True, edgecolor='white', linewidth=1,
                        marker='o', size=8, ax=ax[0])
ax[0].axhline(0, linestyle='--', lw=2, color='grey')

pval = round(ss.ranksums(IC_pt[IC_pt.ON_BF & IC_pt.sig_utask][yaxis_task], IC_pt[~IC_pt.ON_BF & IC_pt.sig_utask][yaxis_task]).pvalue, 3)
off_med = round(IC_pt[~IC_pt.ON_BF & IC_pt.sig_utask][yaxis_task].median(), 3)
on_med = round(IC_pt[IC_pt.ON_BF & IC_pt.sig_utask][yaxis_task].median(), 3)

pval_ns = round(ss.ranksums(IC_pt[IC_pt.ON_BF & ~IC_pt.sig_utask][yaxis_task], IC_pt[~IC_pt.ON_BF & ~IC_pt.sig_utask][yaxis_task]).pvalue, 3)
off_med_ns = round(IC_pt[~IC_pt.ON_BF & ~IC_pt.sig_utask][yaxis_task].median(), 3)
on_med_ns = round(IC_pt[IC_pt.ON_BF & ~IC_pt.sig_utask][yaxis_task].median(), 3)

ax[0].set_title('Pure-tone \n sig_cells: ON: {0}, OFF: {1}, pval: {2} \n'
                    'ns cells: ON: {3}, OFF: {4}, pval: {5}'.format(on_med, off_med, pval, on_med_ns, off_med_ns, pval_ns))

sns.stripplot(x='sig_utask', y=yaxis, data=IC_tin, hue='ON_BF', dodge=True, edgecolor='white', linewidth=1,
                        marker='o', size=8, ax=ax[1])
ax[1].axhline(0, linestyle='--', lw=2, color='grey')

pval = round(ss.ranksums(IC_tin[IC_tin.ON_BF & IC_tin.sig_utask][yaxis_task], IC_tin[~IC_tin.ON_BF & IC_tin.sig_utask][yaxis_task]).pvalue, 3)
off_med = round(IC_tin[~IC_tin.ON_BF & IC_tin.sig_utask][yaxis_task].median(), 3)
on_med = round(IC_tin[IC_tin.ON_BF & IC_tin.sig_utask][yaxis_task].median(), 3)

pval_ns = round(ss.ranksums(IC_tin[IC_tin.ON_BF & ~IC_tin.sig_utask][yaxis_task], IC_tin[~IC_tin.ON_BF & ~IC_tin.sig_utask][yaxis_task]).pvalue, 3)
off_med_ns = round(IC_tin[~IC_tin.ON_BF & ~IC_tin.sig_utask][yaxis_task].median(), 3)
on_med_ns = round(IC_tin[IC_tin.ON_BF & ~IC_tin.sig_utask][yaxis_task].median(), 3)

ax[1].set_title('Tone in noise \n sig_cells: ON: {0}, OFF: {1}, pval: {2} \n'
                    'ns cells: ON: {3}, OFF: {4}, pval: {5}'.format(on_med, off_med, pval, on_med_ns, off_med_ns, pval_ns))
f.tight_layout()

plt.show()