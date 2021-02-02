import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import statsmodels.formula.api as smf
import matplotlib.collections as clt
import re
import pylab as pl

from nems_lbhb.pupil_behavior_scripts.mod_per_state import get_model_results_per_state_model
from nems_lbhb.pupil_behavior_scripts.mod_per_state import aud_vs_state
from nems_lbhb.pupil_behavior_scripts.mod_per_state import hlf_analysis
from nems_lbhb.pupil_behavior_scripts.mod_per_state import beh_only_plot
from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
from nems import get_setting
import nems.plots.api as nplt
import nems_lbhb.pupil_behavior_scripts.common as common
import nems_lbhb.pupil_behavior_scripts.helpers as helper
import nems.xform_helper as xhelp
import nems.db as db
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob


# recache results of pupil variance?
recache = False
path = os.path.dirname(helper.__file__)
pup_results = path + '/pupilVariance.csv'


# set path to dump file
dump_path = get_setting('NEMS_RESULTS_DIR')

helper_path = os.path.dirname(helper.__file__)

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False

# SPECIFY models
USE_AFL=True
if USE_AFL:
    dump_results = 'd_pup_afl_sdexp.csv'
    model_string = 'st.pup.afl'
    p0_model = 'st.pup0.afl'
    b0_model = 'st.pup.afl0'
    shuf_model = 'st.pup0.afl0'
else:
    dump_results = 'd_pup_beh_sdexp.csv'
    model_string = 'st.pup.beh'
    p0_model = 'st.pup0.beh'
    b0_model = 'st.pup.beh0'
    shuf_model = 'st.pup0.beh0'

# set params for BF characterization and sig. sensory response threshold
octave_cutoff = 0.5
r0_threshold = 0
group_files = True

# import / preprocess model results
A1 = helper.preprocess_sdexp_dump(dump_results,
                                  batch=307,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
A1['area'] = 'A1'
IC = helper.preprocess_sdexp_dump(dump_results,
                                  batch=309,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
d_IC_area = pd.read_csv(os.path.join(helper_path,'IC_cells_area.csv'), index_col=0)
IC = IC.merge(d_IC_area, on=['cellid'])

df = pd.concat([A1, IC])

#if group_files & ('beh' not in model_string):
#    area = df['area']
#    df = df.groupby(by=['cellid', 'ON_BF']).mean()
#    df['area'] = [area.loc[c] if type(area.loc[c]) is str else area.loc[c][0] for c in df.index.get_level_values('cellid')]


if recache | (os.path.isfile(pup_results)==False):
    # get all sites
    sites = np.unique([s[:7] for s in df.index.get_level_values(0)])

    # set column for siteid / batch
    df['siteid'] = [s[:7] for s in df.index.get_level_values(0)]
    df['batch'] = [307 if df['area'].iloc[i]=='A1' else 309 for i in range(0, df.shape[0])]
    # for each site, compute the mean / se of r_pup_unique and r_task_unique
    dfg = df[['r_task_unique', 'r_pupil_unique', 'siteid', 'batch']].groupby(by='siteid').mean()

    # for each site, load recording and get pupil variance across ref stimuli (use nems mask from fitting)
    modelname = 'psth.fs20.pup-ld-st.pup.beh-ref-psthfr_sdexp.S_jk.nf20-basic'
    tot = len(dfg.index)
    pdf = pd.DataFrame(index=df.index.get_level_values(0).unique(), columns=['variance', 'pmax', 'pnorm_var', 'var_norm'])
    for idx, (site, batch) in enumerate(zip(dfg.index, dfg['batch'])):
        # for this site, check to see if need to run multiple analyses
        # (for different cells w/ different number of parmfiles)
        d = db.get_batch_cell_data(batch=batch, cellid=site)
        parmCount = d.groupby(by='cellid').count()
        if len(parmCount.parm.unique()) > 1:
            parmCount['cellid'] = parmCount.index
            cgroups = parmCount.groupby(by='parm').agg(lambda column: ", ".join(column)).values
        elif (len(parmCount.parm.unique()) == 1) & (parmCount.shape[0] > 1):
            cgroups = [[', '.join(d.index.get_level_values(0).unique().values.tolist())]]
        else:
            cgroups = [[d.index.get_level_values(0)[0]]]
        print(f"\n \n site n={idx}/{tot} \n \n")
        for cg in cgroups:
            cid = cg[0].split(', ')[0]
            print(f"cellid: {cid} \n")
            xf, ctx = xhelp.load_model_xform(cid, batch, modelname, eval_model=True)

            # compute pupil stats - variacne, mean, var normed by mean
            pupil = ctx['val'].apply_mask()['pupil_raw']._data.squeeze().copy()
            var = np.nanvar(pupil)
            pm = np.nanmax(pupil)
            varn = var / pm
            pupil /= pm
            pnvar = np.nanvar(pupil)
            

            # save pupil stats per cellid
            for c in cg[0].split(', '):
                pdf.loc[c, :] = [var, pm, pnvar, varn]
    
    pdf.to_csv(pup_results)

else:
    pdf = pd.read_csv(pup_results, index_col=0)

np.random.seed(123)
A1 = A1.merge(pdf, on='cellid')
IC = IC.merge(pdf, on='cellid')
# plot r_pupil_unique vs. pupil variance for each cell, in each area
f, ax = plt.subplots(1, 2, figsize=(6, 3))

ax[0].scatter(A1['pnorm_var'], A1['r_pupil_unique'], edgecolor='white', s=25)
ax[0].set_xlabel('Pupil Variance')
ax[0].set_ylabel(r'$r_{pupil unique}$', fontsize=10)
r, p = ss.pearsonr(A1['pnorm_var'], A1['r_pupil_unique'])
A1['siteid'] = [c[:7] for c in A1.index]
pvar = {s: A1[A1.siteid==s]['pnorm_var'].values for s in A1.siteid.unique()}
punique = {s: A1[A1.siteid==s]['r_pupil_unique'].values for s in A1.siteid.unique()}
cc = get_bootstrapped_sample(pvar, punique, metric='corrcoef', nboot=100)
pboot = get_direct_prob(cc, np.zeros(cc.shape[0]))[0]
ax[0].set_title(f'A1, r={r:.3f}, p={p:.6f}, pboot={p:.3f}')

ax[1].scatter(IC['pnorm_var'], IC['r_pupil_unique'], edgecolor='white', s=25)
ax[1].set_xlabel('Pupil Variance')
ax[1].set_ylabel(r'$r_{pupil unique}$', fontsize=10)
r, p = ss.pearsonr(IC['pnorm_var'], IC['r_pupil_unique'])
IC['siteid'] = [c[:7] for c in IC.index]
pvar = {s: IC[IC.siteid==s]['pnorm_var'].values for s in IC.siteid.unique()}
punique = {s: IC[IC.siteid==s]['r_pupil_unique'].values for s in IC.siteid.unique()}
cc = get_bootstrapped_sample(pvar, punique, metric='corrcoef', nboot=100)
pboot = get_direct_prob(cc, np.zeros(cc.shape[0]))[0]
ax[1].set_title(f'IC, r={r}, p={p:.6f}, pboot={pboot:.3f}')

f.tight_layout()

plt.show()
