import matplotlib.pyplot as plt
import numpy as np
import os
import io
import pandas as pd

#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils
import nems.db as nd
import nems.recording as recording
import nems.epoch as ep

#import nems.baphy as nb
import nems_lbhb.xform_wrappers as nw
import nems_lbhb.stateplots as sp
from nems_lbhb.pupil_behavior_scripts import common

from nems_lbhb.pupil_behavior_scripts.mod_per_state import *

batch = 309  # IC SUA and MUA
batch = 295  # old (Slee) IC data
batch = 311  # A1 old (SVD) data -- on BF
batch = 312  # A1 old (SVD) data -- off BF
batch = 307  # A1 SUA and MUA

# pup vs. active/passive
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
#basemodel = "-ref-psthfr.s_stategain.S"
basemodel = "-ref-psthfr.s_sdexp.S"

#batch = 307  # A1 SUA and MUA
#batch = 309  # IC SUA and MUA
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)
    d.to_csv('d_'+str(batch)+'_pb.csv')

# fil only
state_list = ['st.fil0','st.fil']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv('d_'+str(batch)+'_fil.csv')

# pup+fil only, sdexp
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel2 = "-ref-psthfr.s_sdexp.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv('d_'+str(batch)+'_pup_fil.csv')

# pup+fil only stategain
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20.pup-ld-"
batches = [307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv('d_'+str(batch)+'_pup_fil_stategain.csv') 

# pup+fil only stategain (with independent NL for each state chan)
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel2 = "-ref-psthfr.s_sdexp.S.snl"
loader = "psth.fs20.pup-ld-"
fitter = "_jk.nf20-basic.t7"
batches = [309] #[307, 309]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv('d_'+str(batch)+'_pup_fil_sdexp_snl.csv') 

# batch 295 behavior only
state_list = ['st.fil','st.fil0']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20-ld-"
batches = [295]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv('d_'+str(batch)+'_fil_stategain.csv')                                  

# beh only
state_list = ['st.beh0','st.beh']
basemodel2 = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20-ld-"
fitter = "_jk.nf20-basic"
#batch = 307  # DS A1+MU
#batch = 311  # A1 old (SVD) data -- on BF
#batch = 313  # IC PTD data SU + MU
batches = [295, 307, 311, 312, 313]
for batch in batches:
    d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel2, loader=loader)
    d.to_csv('d_'+str(batch)+'_beh.csv')

### do a bunch of grouping/preprocessing

# SPECIFY pup+beh models
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

# RUN IF NOT CONNECTED TO SERVER
# A1 SUA+MUA: pup vs. beh 307 per state
d_pb307 = pd.read_csv('d_307_pb.csv')
d_pb309 = pd.read_csv('d_309_pb.csv')

d_pb307 = d_pb307.drop(['Unnamed: 0'], axis=1)

# Add extra metadata columns
d_pb307['r'] = d_pb307['r'].str.strip(to_strip="[]").astype(float)
d_pb307['R2'] = d_pb307['r']**2 * np.sign(d_pb307['r'])
d_pb307['area'] = 'A1'
d_pb307['sign'] = 'TBD'
d_pb307['experimenter'] = 'DS'
d_pb307['onBF'] = 'TBD'
d_pb307['SU'] = False
d_pb307.loc[d_pb307['isolation']>=90.0, 'SU'] = True
d_pb307['animal'] = d_pb307['cellid'].map(lambda x: x[:3])
d_pb307['task'] = 'TIN'

d_pb309 = d_pb309.drop(['Unnamed: 0'], axis=1)

# Add extra metadata columns. Some with "TBD" will be filled later
d_pb309['R2'] = d_pb309['r']**2 * np.sign(d_pb309['r'])
d_pb309['onBF'] = 'TBD'
d_pb309['experimenter'] = 'DS'
d_pb309['sign'] = 'TBD'
d_pb309['animal'] = d_pb309['cellid'].map(lambda x: x[:3])
d_pb309['task'] = 'TIN'
d_pb309.loc[d_pb309['animal']=='ley', 'task'] = 'TvN'
d_pb309['SU'] = False
d_pb309.loc[d_pb309['isolation']>=90.0, 'SU'] = True

d_IC_area = pd.read_csv('IC_cells_area.csv')
d_pb309 = pd.merge(left=d_pb309, right=d_IC_area, how='outer', on='cellid')

nan_rows = d_pb309[d_pb309['area'].isnull()]

d_307_strf = pd.read_csv('tuning_info_batch_307.csv')
d_307_strf = d_307_strf.drop(['Unnamed: 43'], axis=1)

d_309_strf = pd.read_csv('tuning_info_batch_309.csv')
d_309_strf = d_309_strf.drop(['Unnamed: 43'], axis=1)

d_307_309 = pd.concat([d_pb307, d_pb309], sort=False)
d_307_309_strf = pd.concat([d_307_strf, d_309_strf], sort=False)
df = pd.merge(left=d_307_309, right=d_307_309_strf, how='outer', on='cellid')

df = common.fix_TBD_onBF(df)

# creating subdf with only rows that match conditions
is_active = (df['state_chan'] == 'active')
is_pupil = (df['state_chan'] == 'pupil')
full_model = (df['state_sig'] == 'st.pup.beh')
null_model = (df['state_sig'] == 'st.pup0.beh0')
part_beh_model = (df['state_sig'] == 'st.pup0.beh')
part_pup_model = (df['state_sig'] == 'st.pup.beh0')

# adding new colums to df with differences of R2 full-null, MI full-partial pup,
# and MI partial beh-null
for cellid in df['cellid'].unique():
    mask_for_cellid = df['cellid'] == cellid
    active_full = df[is_active & full_model & mask_for_cellid]
    active_null = df[is_active & null_model & mask_for_cellid]
    active_part_beh = df[is_active & part_beh_model & mask_for_cellid]
    active_part_pup = df[is_active & part_pup_model & mask_for_cellid]

    pupil_full = df[is_pupil & full_model & mask_for_cellid]
    pupil_null = df[is_pupil & null_model & mask_for_cellid]
    pupil_part_beh = df[is_pupil & part_beh_model & mask_for_cellid]
    pupil_part_pup = df[is_pupil & part_pup_model & mask_for_cellid]

    if len(active_full) != 1:
        print(f'WARNING: active full is not one for {cellid}')
        continue
    if len(active_null) != 1:
        print(f'WARNING: active null is not one for {cellid}')
        continue
    if len(active_part_beh) != 1:
        print(f'WARNING: active part beh is not one for {cellid}')
        continue
    if len(active_part_pup) != 1:
        print(f'WARNING: active part pup is not one for {cellid}')
        continue
    df.loc[mask_for_cellid, 'R2_diff'] = active_full.iloc[0]['R2'] - active_null.iloc[0]['R2']
    df.loc[mask_for_cellid, 'R2beh_unique'] = active_full.iloc[0]['R2'] - active_part_pup.iloc[0]['R2']
    df.loc[mask_for_cellid, 'R2pup_unique'] = active_full.iloc[0]['R2'] - active_part_beh.iloc[0]['R2']

    df.loc[mask_for_cellid, 'MIbeh_only'] = active_part_beh.iloc[0]['MI'] - active_null.iloc[0]['MI']

    df.loc[mask_for_cellid, 'MIbeh_unique'] = active_full.iloc[0]['MI'] - active_part_pup.iloc[0]['MI']
    df.loc[mask_for_cellid, 'MIpup_unique'] = pupil_full.iloc[0]['MI'] - pupil_part_beh.iloc[0]['MI']

    # adding new colums to df with significance
    r_pup_beh = active_full.iloc[0]['r']
    r_pup_beh0 = active_part_pup.iloc[0]['r']
    r_pup0_beh = active_part_beh.iloc[0]['r']
    rse_pup_beh = active_full['r_se'].str.strip(to_strip='[]').astype(float).values[0]
    rse_pup_beh0 = active_part_pup['r_se'].str.strip(to_strip='[]').astype(float).values[0]
    rse_pup0_beh = active_part_beh['r_se'].str.strip(to_strip='[]').astype(float).values[0]
    r_pup0_beh0 = active_null.iloc[0]['r']
    rse_pup0_beh0 = active_null['r_se'].str.strip(to_strip='[]').astype(float).values[0]

    # units that had significant unique behavior
    df.loc[mask_for_cellid, 'sig_ubeh'] = (r_pup_beh - r_pup_beh0) > (rse_pup_beh + rse_pup_beh0)
    # units that had significant unique pupil
    df.loc[mask_for_cellid, 'sig_upup'] = (r_pup_beh - r_pup0_beh) > (rse_pup_beh + rse_pup0_beh)
    # units that had significant unique behavior or pupil
    df.loc[mask_for_cellid, 'sig_state'] = (r_pup_beh - r_pup0_beh0) > (rse_pup_beh + rse_pup0_beh0)
    # units that had significant behavior in behavior partial model
    df.loc[mask_for_cellid, 'sig_obeh'] = (r_pup0_beh - r_pup0_beh0) > (rse_pup0_beh + rse_pup0_beh0)

    df.loc[mask_for_cellid, 'sig_any'] = ((r_pup_beh - rse_pup_beh*3) > 0) | \
        ((r_pup0_beh0 - rse_pup0_beh0*3) > 0)

#df.to_csv('pup_beh_processed.csv')
df.to_csv('pup_beh_processed'+basemodel+'.csv')

d_b295 = pd.read_csv('d_295_beh.csv')
d_b307 = pd.read_csv('d_307_beh.csv')
d_b311 = pd.read_csv('d_311_beh.csv')
d_b312 = pd.read_csv('d_312_beh.csv')
d_b313 = pd.read_csv('d_313_beh.csv')

d_b295['R2'] = d_b295['r']**2 * np.sign(d_b295['r'])
d_b295['area'] = 'IC'
d_b295['sign'] = 'TBD'
d_b295['experimenter'] = 'SS'
d_b295['onBF'] = 'TBD'
d_b295['SU'] = False
d_b295.loc[d_b295['isolation']>=90.0, 'SU'] = True
d_b295['animal'] = d_b295['cellid'].map(lambda x: x[:3])
d_b295['task'] = 'PTD'

d_b307['R2'] = d_b307['r']**2 * np.sign(d_b307['r'])
d_b307['area'] = 'A1'
d_b307['sign'] = 'TBD'
d_b307['experimenter'] = 'DS'
d_b307['onBF'] = 'TBD'
d_b307['SU'] = False
d_b307.loc[d_b307['isolation']>=90.0, 'SU'] = True
d_b307['animal'] = d_b307['cellid'].map(lambda x: x[:3])
d_b307['task'] = 'TIN'

d_b311['R2'] = d_b311['r']**2 * np.sign(d_b311['r'])
d_b311['area'] = 'A1'
d_b311['sign'] = 'TBD'
d_b311['experimenter'] = 'SD'
d_b311['onBF'] = True
d_b311['SU'] = False
d_b311.loc[d_b307['isolation']>=90.0, 'SU'] = True
d_b311['animal'] = d_b311['cellid'].map(lambda x: x[:3])
d_b311['task'] = 'TIN'

d_b312['R2'] = d_b312['r']**2 * np.sign(d_b312['r'])
d_b312['area'] = 'A1'
d_b312['sign'] = 'TBD'
d_b312['experimenter'] = 'SD'
d_b312['onBF'] = False
d_b312['SU'] = False
d_b312.loc[d_b307['isolation']>=90.0, 'SU'] = True
d_b312['animal'] = d_b312['cellid'].map(lambda x: x[:3])
d_b312['task'] = 'PTD'

d_b313 = d_b313.drop(['Unnamed: 0'], axis=1)

# Add extra metadata columns. Some with "TBD" will be filled later
d_b313['R2'] = d_b313['r']**2 * np.sign(d_b313['r'])
d_b313['area'] = 'IC'
d_b313['onBF'] = 'TBD'
d_b313['experimenter'] = 'DS'
d_b313['sign'] = 'TBD'
d_b313['animal'] = d_b313['cellid'].map(lambda x: x[:3])
d_b313['task'] = 'PTD'
d_b313.loc[d_b313['animal']=='ley', 'task'] = 'TIN'
d_b313['SU'] = False
d_b313.loc[d_b313['isolation']>=90.0, 'SU'] = True

dfb = pd.concat([d_b295, d_b307, d_b311, d_b312, d_b313], sort=False)  # d_b312,
dfb.sort_values(by=['area','cellid','state_chan','state_sig'], inplace=True)

# creating subdf with only rows that match conditions
is_active = (dfb['state_chan'] == 'active')
full_model = (dfb['state_sig'] == 'st.beh')
null_model = (dfb['state_sig'] == 'st.beh0')

# adding new colums to df with differences of R2 full-null, MI full-partial pup,
# and MI partial beh-null
for cellid in dfb['cellid'].unique():
    mask_for_cellid = (dfb['cellid'] == cellid)
    active_full = dfb[is_active & full_model & mask_for_cellid]
    active_null = dfb[is_active & null_model & mask_for_cellid]

    if len(active_full) != 1:
        print(f'WARNING: active full is not one for {cellid}')
        continue
    if len(active_null) != 1:
        print(f'WARNING: active null is not one for {cellid}')
        continue

    dfb.loc[mask_for_cellid, 'R2_diff'] = active_full.iloc[0]['R2'] - active_null.iloc[0]['R2']

    dfb.loc[mask_for_cellid, 'MIbeh_only'] = active_full.iloc[0]['MI'] - active_null.iloc[0]['MI']

    # adding new colums to df with significance
    r_beh = active_full.iloc[0]['r']
    r_beh0 = active_null.iloc[0]['r']
    rse_beh = active_full.iloc[0]['r_se']
    rse_beh0 = active_null.iloc[0]['r_se']

    # units that had significant state effect
    dfb.loc[mask_for_cellid, 'sig_state'] = (r_beh - r_beh0) > (rse_beh + rse_beh0)
    dfb.loc[mask_for_cellid, 'sig_any'] = ((r_beh - rse_beh*2) > 0) | ((r_beh0 - rse_beh0*2) > 0)

    # False for extra conditions
    dfb.loc[mask_for_cellid, 'sig_ubeh'] = False
    dfb.loc[mask_for_cellid, 'sig_upup'] = False
    dfb.loc[mask_for_cellid, 'sig_obeh'] = False

dfb.to_csv('beh_only_processed'+basemodel2+'.csv')
#dfb.to_csv('beh_only_processed.csv')
