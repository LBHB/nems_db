#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:13:33 2018

@author: svd
"""

# scripts for showing changes in response across behabior blocks

from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
#from nems_lbhb.behavior_pupil_scripts.mod_per_state import get_model_results_per_state_model

do_single_cell = True  # if False, do pop summary

if do_single_cell:
    # SINGLE CELL EXAMPLES

    # DEFAULTS
    #loader= "psth.fs20.pup-ld-"
    #fitter = "_jk.nf20-basic"
    basemodel = "-ref-psthfr_stategain.S"

    # alternative state / file breakdowns
    #state_list = ['st.pup0.hlf0','st.pup.hlf0','st.pup.hlf']
    state_list = ['st.pup0.beh0','st.pup.beh0','st.pup0.beh','st.pup.beh']
    #state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']

    # individual cells - by default

    cellid="bbl081d-a1"

    # problem IC cells:
    #cellid="bbl081d-a2"  # fixed
    cellid="BRT016f-a1"  # fixed

    batch=309

    cellid="BRT026c-05-2"
    cellid="TAR010c-19-1"  #
    cellid="bbl102d-01-1"  # maybe good hybrid cell?
    cellid="TAR010c-33-1"
    cellid="BRT026c-20-1"
    cellid="TAR010c-06-1"  # pupil cell
    cellid="TAR010c-27-2"  # behavior

    # problem A1 cells:
    #cellid="BRT037b-24-1" # fixed
    #cellid="BRT037b-13-1" # fixed
    #cellid="BRT037b-36-1"# fixed
    batch=307

    model_per_time_wrapper(cellid, batch, basemodel=basemodel,
                           state_list=state_list)

elif 0:
    exec(open("/auto/users/svd/python/nems_db/nems_lbhb/pupil_behavior_scripts/mod_per_state.py").read())

    # POPULATION SUMMARY CELL EXAMPLES
    # use basemodel = "-ref-psthfr.s_sdexp.S" for better accuracy and
    # statistical power

    #state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
    #states = ['PASSIVE_0',  'ACTIVE_1','PASSIVE_1',  'ACTIVE_2','PASSIVE_2']

    state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
    #state_list = ['st.pup0.far0.hit0.hlf0','st.pup0.far0.hit0.hlf',
    #              'st.pup.far.hit.hlf0','st.pup.far.hit.hlf']
    states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
              'PASSIVE_1_A','PASSIVE_1_B', 'ACTIVE_2_A','ACTIVE_2_B',
              'PASSIVE_2_A','PASSIVE_2_B']
    #states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
    #          'PASSIVE_1_A','PASSIVE_1_B']

    basemodel = "-ref-psthfr.s_sdexp.S"
    #basemodel = "-ref.a-psthfr.s_sdexp.S"
    batch = 307
    #batch = 309

    df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

    dMI, dMI0 = hlf_analysis(df, state_list, states=states)

else:
    exec(open("/auto/users/svd/python/nems_db/nems_lbhb/pupil_behavior_scripts/mod_per_state.py").read())
    basemodel = "-ref-psthfr.s_sdexp.S"
    #basemodel = "-ref.a-psthfr.s_sdexp.S"
    batch = 309
    #b_unique  mean sig      0.030586
    #p_unique  mean sig      0.059588
    #batch = 307
    state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

    # figure out what cells show significant state ef
    da = df[df['state_chan']=='pupil']
    dp = pd.pivot_table(da, index='cellid',columns='state_sig',values=['r','r_se'])
    dr = dp['r'].copy()
    dr['b_unique'] = dr[state_list[3]]**2 - dr[state_list[2]]**2
    dr['p_unique'] = dr[state_list[3]]**2 - dr[state_list[1]]**2
    dr['bp_common'] = dr[state_list[3]]**2 - dr[state_list[0]]**2 - dr['b_unique'] - dr['p_unique']
    dr['bp_full'] = dr['b_unique']+dr['p_unique']+dr['bp_common']
    dr['null']=dr[state_list[0]]**2 * np.sign(dr[state_list[0]])
    dr['full']=dr[state_list[3]]**2 * np.sign(dr[state_list[3]])

    dr['sig']=((dp['r'][state_list[3]]-dp['r'][state_list[0]]) > \
         (dp['r_se'][state_list[3]]+dp['r_se'][state_list[0]]))
    plt.close('all')
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    beta_comp(dr['p_unique'], dr['b_unique'], n1='Pupil', n2='Behavior',
              highlight=dr['sig'], ax=ax, hist_range=[-0.05, 0.15]);