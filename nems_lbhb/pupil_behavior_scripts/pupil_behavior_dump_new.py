"""
CRH 04/08/2020

Meant to eventually replace pupil_behavior_dump.py which was saving results into the local
git repo. Better to have results saved to nems results directory on elephant.
"""

import pandas as pd
import os

from nems import get_setting
from nems_lbhb.pupil_behavior_scripts.mod_per_state import *

import logging
log = logging.getLogger(__name__)

# save results dataframes to nems results directory + /batch/
dump_path = get_setting('NEMS_RESULTS_DIR')


# ========================== stategain models ===============================

# beh only models
log.info('Saving stategain beh only models... ')
state_list = ['st.beh0','st.beh']
basemodel2 = "-ref-psthfr.s_stategain.S.new"
loader = "psth.fs20.pup-ld-"
batches = [295, 307, 309, 311, 312, 313]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_beh_stategain.csv'))

# afl only models
log.info('Saving stategain afl only models... ')
state_list = ['st.afl0','st.afl']
basemodel2 = "-ref-psthfr.s_stategain.S.new"
loader = "psth.fs20.pup-ld-"
batches = [295, 307, 309, 311, 312, 313]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_afl_stategain.csv'))

# pup + fil models
log.info('Saving stategain pup fil models... ')
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_stategain.csv'))


# pup + afl models
log.info('Saving stategain pup afl models... ')
state_list = ['st.pup0.afl0','st.pup0.afl','st.pup.afl0','st.pup.afl']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_stategain.csv'))


# pup + afl + pxf models
log.info('Saving stategain pup+afl+pxf models... ')
state_list = ['st.pup0.afl0.pxf0','st.pup0.afl.pxf0','st.pup.afl0.pxf0','st.pup.afl.pxf']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_pxf_stategain.csv'))


# ======================= stategain with ap1 only ===========================
# pup + fil models
log.info('Saving stategain pup + fil models with ap1... ')
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20.pup-ld-ap1-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_stategain_ap1.csv'))


# pup + afl models
log.info('Saving stategain pup + afl models with ap1... ')
state_list = ['st.pup0.afl0','st.pup0.afl','st.pup.afl0','st.pup.afl']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20.pup-ld-ap1-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_stategain_ap1.csv'))


# pup + afl + pxf models
log.info('Saving stategain pup+afl+pxf models with ap1... ')
state_list = ['st.pup0.afl0.pxf0','st.pup0.afl.pxf0','st.pup.afl0.pxf0','st.pup.afl.pxf']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20.pup-ld-ap1-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_pxf_stategain_ap1.csv'))

# ============================ sdexp models =================================

# beh only models
log.info('Saving sdexp beh only models... ')
state_list = ['st.beh0','st.beh']
basemodel2 = "-ref-psthfr.s_sdexp.S.new"
loader = "psth.fs20.pup-ld-"
batches = [295, 307, 309, 311, 312, 313]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_beh_sdexp.csv'))

# afl only models
log.info('Saving sdexp afl only models... ')
state_list = ['st.afl0','st.afl']
basemodel2 = "-ref-psthfr.s_sdexp.S.new"
loader = "psth.fs20.pup-ld-"
batches = [295, 307, 309, 311, 312, 313]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_afl_sdexp.csv'))

# pup + beh models
log.info('Saving sdexp pup + fil models... ')
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel2 = "-ref-psthfr.s_sdexp.S.new"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_beh_sdexp.csv'))

# pup + fil models
log.info('Saving sdexp pup + fil models... ')
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel2 = "-ref-psthfr.s_sdexp.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_sdexp.csv'))

# afl models
log.info('Saving sdexp pup + afl models... ')
state_list = ['st.pup0.afl0','st.pup0.afl','st.pup.afl0','st.pup.afl']
basemodel2 = "-ref-psthfr.s_sdexp.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_sdexp.csv'))

# afl + pxf models
log.info('Saving sdexp pup+afl+pxf models... ')
state_list = ['st.pup0.afl0.pxf0','st.pup0.afl.pxf0','st.pup.afl0.pxf0','st.pup.afl.pxf']
basemodel2 = "-ref-psthfr.s_sdexp.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_pxf_sdexp.csv'))


# ======================== sdexp models with ap1 =============================

# fil models
log.info('Saving sdexp pup + fil models with ap1... ')
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel2 = "-ref-psthfr.s_sdexp.S"
loader = "psth.fs20.pup-ld-ap1-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_sdexp_ap1.csv'))

# afl models
log.info('Saving sdexp pup + afl models with ap1... ')
state_list = ['st.pup0.afl0','st.pup0.afl','st.pup.afl0','st.pup.afl']
basemodel2 = "-ref-psthfr.s_sdexp.S"
loader = "psth.fs20.pup-ld-ap1-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_sdexp_ap1.csv'))

# afl + pxf models
log.info('Saving sdexp pup+afl+pxf models with ap1... ')
state_list = ['st.pup0.afl0.pxf0','st.pup0.afl.pxf0','st.pup.afl0.pxf0','st.pup.afl.pxf']
basemodel2 = "-ref-psthfr.s_sdexp.S"
loader = "psth.fs20.pup-ld-ap1-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv(os.path.join(dump_path, str(batch), 'd_pup_afl_pxf_sdexp_ap1.csv'))


