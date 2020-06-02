import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sci
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
import nems_lbhb.pupil_behavior_scripts.common as common
import nems_lbhb.pupil_behavior_scripts.helpers as helper

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False


PAS_ONLY = True
sig_cells_only = True # if true, for the line plot, only average over cells with sig. state effects
dump_path = get_setting('NEMS_RESULTS_DIR')
basemodel = "-ref-psthfr.s_sdexp.S"
state_list = ['st.pup0.fil0', 'st.pup0.fil', 'st.pup.fil0', 'st.pup.fil']
states = ['PASSIVE_0',  'ACTIVE_1', 'PASSIVE_1',
        'ACTIVE_2', 'PASSIVE_2']

batch=307
A1 = pd.read_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_sdexp.csv'))
# convert r values to numeric
try:
        A1['r'] = [np.float(r.strip('[]')) for r in A1['r'].values]
        A1['r_se'] = [np.float(r.strip('[]')) for r in A1['r_se'].values]
except:
        pass
A1 = A1[~A1.cellid.str.contains('AMT')]
batch=309
IC = pd.read_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_sdexp.csv'))
# convert r values to numeric
try:
        IC['r'] = [np.float(r.strip('[]')) for r in IC['r'].values]
        IC['r_se'] = [np.float(r.strip('[]')) for r in IC['r_se'].values]
except:
        pass
IC = IC[~IC.cellid.str.contains('AMT')]

# load afl results to get list of sig cells
dump_results = 'd_pup_afl_sdexp.csv'
model_string = 'st.pup.afl'
p0_model = 'st.pup0.afl'
b0_model = 'st.pup.afl0'
shuf_model = 'st.pup0.afl0'
octave_cutoff = 0.5
r0_threshold = 0
group_files = True
A1_afl = helper.preprocess_sdexp_dump(dump_results,
                                  batch=307,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)

task_only = A1_afl[A1_afl['sig_utask'] & ~A1_afl['sig_upupil']].index.unique().tolist()
pupil_only = A1_afl[~A1_afl['sig_utask'] & A1_afl['sig_upupil']].index.unique().tolist()
both = A1_afl[A1_afl['sig_utask'] & A1_afl['sig_upupil']].index.unique().tolist()
task_or_pupil = [c for c in A1_afl[A1_afl['sig_state']].index.unique() if \
                        (c not in task_only) & (c not in pupil_only) & (c not in both)] 
A1_sig = {'task_or_pupil': task_or_pupil, 'both': both, 'pupil_only': pupil_only, 'task_only': task_only}

IC_afl = helper.preprocess_sdexp_dump(dump_results,
                                  batch=309,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
task_only = IC_afl[IC_afl['sig_utask'] & ~IC_afl['sig_upupil']].index.unique().tolist()
pupil_only = IC_afl[~IC_afl['sig_utask'] & IC_afl['sig_upupil']].index.unique().tolist()
both = IC_afl[IC_afl['sig_utask'] & IC_afl['sig_upupil']].index.unique().tolist()
task_or_pupil = [c for c in IC_afl[IC_afl['sig_state']].index.unique() if \
                        (c not in task_only) & (c not in pupil_only) & (c not in both)] 
IC_sig = {'task_or_pupil': task_or_pupil, 'both': both, 'pupil_only': pupil_only, 'task_only': task_only}

if not PAS_ONLY:
        fA1, _1, _2 = helper.hlf_analysis(A1, state_list, norm_sign=True, sig_cells_only=sig_cells_only, states=states, scatter_sig_cells=A1_sig)

        fIC, _1, _2 = helper.hlf_analysis(IC, state_list, norm_sign=True, sig_cells_only=sig_cells_only, states=states, scatter_sig_cells=IC_sig)

else:
        # load pas only models
        batch=307
        A1_pas = pd.read_csv(os.path.join(dump_path, str(batch), 'd_pup_pas_sdexp.csv'))
        # convert r values to numeric
        try:
                A1_pas['r'] = [np.float(r.strip('[]')) for r in A1_pas['r'].values]
                A1_pas['r_se'] = [np.float(r.strip('[]')) for r in A1_pas['r_se'].values]
        except:
                pass
        A1_pas = A1_pas[~A1_pas.cellid.str.contains('AMT')]
        batch=309
        IC_pas = pd.read_csv(os.path.join(dump_path, str(batch), 'd_pup_pas_sdexp.csv'))
        # convert r values to numeric
        try:
                IC_pas['r'] = [np.float(r.strip('[]')) for r in IC_pas['r'].values]
                IC_pas['r_se'] = [np.float(r.strip('[]')) for r in IC_pas['r_se'].values]
        except:
                pass
        IC_pas = IC_pas[~IC_pas.cellid.str.contains('AMT')]

        # change task significance to be determined with the PAS model
        dump_results = 'd_pup_pas_sdexp.csv'
        model_string = 'st.pup.pas'
        p0_model = 'st.pup0.pas'
        b0_model = 'st.pup.pas0'
        shuf_model = 'st.pup0.pas0'
        octave_cutoff = 0.5
        r0_threshold = 0
        A1_afl = helper.preprocess_sdexp_dump(dump_results,
                                        batch=307,
                                        full_model=model_string,
                                        p0=p0_model,
                                        b0=b0_model,
                                        shuf_model=shuf_model,
                                        r0_threshold=r0_threshold,
                                        octave_cutoff=octave_cutoff,
                                        pas_model=True,
                                        path=dump_path)

        task_only = A1_afl[A1_afl['sig_utask'] & ~A1_afl['sig_upupil']].index.unique().tolist()
        pupil_only = A1_afl[~A1_afl['sig_utask'] & A1_afl['sig_upupil']].index.unique().tolist()
        both = A1_afl[A1_afl['sig_utask'] & A1_afl['sig_upupil']].index.unique().tolist()
        task_or_pupil = [c for c in A1_afl[A1_afl['sig_state']].index.unique() if \
                                (c not in task_only) & (c not in pupil_only) & (c not in both)] 
        A1_sig = {'task_or_pupil': task_or_pupil, 'both': both, 'pupil_only': pupil_only, 'task_only': task_only}

        IC_afl = helper.preprocess_sdexp_dump(dump_results,
                                        batch=309,
                                        full_model=model_string,
                                        p0=p0_model,
                                        b0=b0_model,
                                        shuf_model=shuf_model,
                                        r0_threshold=r0_threshold,
                                        octave_cutoff=octave_cutoff,
                                        pas_model=True,
                                        path=dump_path)
        task_only = IC_afl[IC_afl['sig_utask'] & ~IC_afl['sig_upupil']].index.unique().tolist()
        pupil_only = IC_afl[~IC_afl['sig_utask'] & IC_afl['sig_upupil']].index.unique().tolist()
        both = IC_afl[IC_afl['sig_utask'] & IC_afl['sig_upupil']].index.unique().tolist()
        task_or_pupil = [c for c in IC_afl[IC_afl['sig_state']].index.unique() if \
                                (c not in task_only) & (c not in pupil_only) & (c not in both)] 
        IC_sig = {'task_or_pupil': task_or_pupil, 'both': both, 'pupil_only': pupil_only, 'task_only': task_only}

        fA1, dMIu_A1, dMI_A1 = helper.hlf_analysis(A1, state_list, pas_df=A1_pas, norm_sign=True, sig_cells_only=sig_cells_only, states=states, scatter_sig_cells=A1_sig)
        fIC, dMIu_IC, dMI_IC = helper.hlf_analysis(IC, state_list, pas_df=IC_pas, norm_sign=True, sig_cells_only=sig_cells_only, states=states, scatter_sig_cells=IC_sig)

#stat, p = sci.wilcoxon(dMIu_A1, dMI_A1)


if save_fig:
    fA1.savefig(os.path.join(save_path,'fig7_pre_post_A1.pdf'))
    fIC.savefig(os.path.join(save_path,'fig7_pre_post_IC.pdf'))
