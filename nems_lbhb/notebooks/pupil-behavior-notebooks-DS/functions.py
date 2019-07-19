# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:00:27 2019

@author: daniela

Functions for pupil-behavior paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

color_b = '#C768D8'
color_p = '#47BF55'
color_both = 'black'
color_either = '#595959'
color_ns = 'lightgrey'


def find_sig_cellids(df, state_chan_val='active', condition='pb', sign_type = 'beh'):
    '''it takes a Pandas df as an argument and returns a Pandas series with cellids that 
    are significantly modulated by behavior if sign_type 'beh' according to P
    If sign_type = 'state', then it pulls out untis significantly modulated by either state, 
    pupil or behavior'''
    
    if condition=='pb' and sign_type == 'beh':
        df_state = df[df['state_chan']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')

        df_r['r_diff'] = df_r['st.pup0.beh']-df_r['st.pup0.beh0']
        df_rse['r_sum'] = df_rse['st.pup0.beh']+df_rse['st.pup0.beh0']
        
    elif condition=='pb' and sign_type == 'ubeh':
        df_state = df[df['state_chan']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')

        df_r['r_diff'] = df_r['st.pup.beh']-df_r['st.pup.beh0']
        df_rse['r_sum'] = df_rse['st.pup.beh']+df_rse['st.pup.beh0']
        
    elif condition=='pb' and sign_type == 'upup':
        df_state = df[df['state_chan']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')

        df_r['r_diff'] = df_r['st.pup.beh']-df_r['st.pup0.beh']
        df_rse['r_sum'] = df_rse['st.pup.beh']+df_rse['st.pup0.beh']
        
    elif condition=='pb' and sign_type == 'state':
        df_state = df[df['state_chan']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')

        df_r['r_diff'] = df_r['st.pup.beh']-df_r['st.pup0.beh0']
        df_rse['r_sum'] = df_rse['st.pup.beh']+df_rse['st.pup0.beh0']
        
    elif condition=='pp' and sign_type == 'beh':
        df_state = df[df['state_chan_alt']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')
        
        df_r['r_diff'] = df_r['st.pup0.pas']-df_r['st.pup0.pas0']
        df_rse['r_sum'] = df_rse['st.pup0.pas']+df_rse['st.pup0.pas0']
        
        
    elif condition=='pp' and sign_type == 'state':
        df_state = df[df['state_chan_alt']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')
        
        df_r['r_diff'] = df_r['st.pup0.pas']-df_r['st.pup0.pas0']
        df_rse['r_sum'] = df_rse['st.pup0.pas']+df_rse['st.pup0.pas0']
        
        
    elif condition=='pf' and sign_type == 'beh':
        df_state = df[df['state_chan_alt']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')
        
        df_r['r_diff'] = df_r['st.pup0.fil']-df_r['st.pup0.fil0']
        df_rse['r_sum'] = df_rse['st.pup0.fil']+df_rse['st.pup0.fil0']
        
        
    elif condition=='fil' and sign_type == 'state':
        df_state = df[df['state_chan']==state_chan_val]
        
        # Pivot the new df such that you only have r or r_se as data and state_sig as columns
        df_r = df_state.pivot(index='cellid', columns='state_sig', values='r')
        df_rse = df_state.pivot(index='cellid', columns='state_sig', values='r_se')

        df_r['r_diff'] = df_r['st.fil']-df_r['st.fil0']
        df_rse['r_sum'] = df_rse['st.fil']+df_rse['st.fil0']
        
        
    cellid_sig = df_r[df_r['r_diff']>df_rse['r_sum']].index
    
    return cellid_sig
    
    
    
def one_state(df, col_idx='MI', state='only', state_chan_val='active', state_sig1='st.pup0.beh', 
              state_sig2='st.pup0.beh0', state_var='task', condition='pb', absolute=None, 
              columns_to_keep=None):
    '''it takes a dataframe and a column_index and it returns two dfs with the difference 
    between the values of the column index one for all cells one for significant cells. 
    e.g. for MI, MIbeh = MIpup0beh-MIpup0beh0.
    if absolute is set to 1, then there will be another column with col_idx absolute value'''
    
    if condition=='pb':
        df_state = df[df['state_chan']==state_chan_val]
    
        df_ss1 = df_state[df_state['state_sig']==state_sig1]
        df_ss2 = df_state[df_state['state_sig']==state_sig2]
    
        # pivot model 1 and 2 to have the column_index for each state_chan
        df_ss1_col_idx = df_ss1.pivot(index='cellid', columns='state_chan', values=col_idx)
        df_ss2_col_idx = df_ss2.pivot(index='cellid', columns='state_chan', values=col_idx)
    
    elif condition=='pp' or condition=='pf' or condition=='fil':
        df_state = df[df['state_chan_alt']==state_chan_val]
    
        df_ss1 = df_state[df_state['state_sig']==state_sig1]
        df_ss2 = df_state[df_state['state_sig']==state_sig2]
    
        # pivot model 1 and 2 to have the column_index for each state_chan
        df_ss1_col_idx = df_ss1.pivot(index='cellid', columns='state_chan_alt', values=col_idx)
        df_ss2_col_idx = df_ss2.pivot(index='cellid', columns='state_chan_alt', values=col_idx)
        
    
    #change name of column to prepare for merging
    df_ss2_col_idx = df_ss2_col_idx.rename(index=str, columns={state_chan_val:state_chan_val+'0'})
    
    #reset index to get rid of multindexing in model 1 and 2
    df_ss1_col_idx = df_ss1_col_idx.reset_index()
    df_ss2_col_idx = df_ss2_col_idx.reset_index()
    
    # join dataframes
    df_col_idx_state = pd.merge(df_ss1_col_idx, df_ss2_col_idx, how='left', on='cellid')
    
    # add column with difference between state_chan and state_chan0
    df_col_idx_state[col_idx+'_'+state_var+'_'+state] = df_col_idx_state[state_chan_val]-df_col_idx_state[state_chan_val+'0']
    
    # set the index back to cellid to apply the loc method and get the significant cells
    df_col_idx_state = df_col_idx_state.set_index('cellid')
    
    # if absolute is set to 1, then there will be another column with col_idx absolute value
    if absolute==1:
        df_col_idx_state[col_idx+'_'+state_var+'_'+state+'_abs'] = abs(df_col_idx_state[col_idx+'_'+state_chan_val+'_'+state])
    
    #df_col_idx_state = df_col_0idx_state[col_idx+'_'+state_chan_val+'_'+state]
    
    if columns_to_keep:
        # Initialize empty columns to keep all to false
        for col in columns_to_keep:
            df_col_idx_state[col] = None
        # Now fill those columns with the correct values for those cells
        for cellid in df_col_idx_state.index.values.tolist():
            matching_rows =  df[df['cellid'] == cellid]
            # print(matching_rows)
            # print(matching_rows.iloc[0])
            #print(matching_rows.iloc[0][col])
            for col in columns_to_keep:
                df_col_idx_state.at[cellid, col] = matching_rows.iloc[0][col]
                
    return df_col_idx_state
    
    
def scatter_states(df1, df2, column1, column2, unit_set, xlim, ylim, dot_size1=2, dot_size2=4, 
                   colors = [color_ns, color_either, color_b, color_p, color_both],
                   title='scatter_states',
                   margin=True, area=None, bins=None, cellid=None):
    '''The basics of this function takes two data frames and specific columns for each data frame
    (eg 'MI_active_only' or 'R2_active_unique') and plots the scatter plot between those and the significant units.
    
    If margin=True, it will plot the marginal distributions'''
    
    sns.set(style="white")

    # Plot non-significant cells first
    subset_df1 = df1[(df1['sig_ubeh']==False) & (df1['sig_upup']==False) & (df1['sig_state']==False)]
    subset_df2 = df2[(df2['sig_ubeh']==False) & (df2['sig_upup']==False) & (df1['sig_state']==False)]
    print('not sig n = {}'.format(len(subset_df1)))
    color = colors[0]
    if area=='A1' or area=='ICX':
        scatplot = sns.JointGrid(x=subset_df1[column1], y=subset_df2[column2], xlim=xlim, ylim=ylim)
        scatplot = scatplot.plot_joint(plt.scatter, color=color, edgecolor="white", s=dot_size1, linewidth=0.3)
        
    else:
        scatplot = sns.JointGrid(x=subset_df1[column1], y=subset_df2[column2], xlim=xlim, ylim=ylim)
        scatplot = scatplot.plot_joint(plt.scatter, color=color, edgecolor="white", s=dot_size1, linewidth=0.3, marker = '^')
    

    # Plot the significant ubeh or upup cells
    subset_df1 = df1[(df1['sig_ubeh']==False) & (df1['sig_upup']==False) & (df1['sig_state']==True)]
    subset_df2 = df2[(df2['sig_ubeh']==False) & (df2['sig_upup']==False) & (df1['sig_state']==True)]
    print('ubeh or upup n = {}'.format(len(subset_df1)))
    color = colors[1]
    if area=='A1' or area=='ICX':
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, s=dot_size2, 
                                  edgecolor='white', linewidth=0.8)
    else:
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, s=dot_size2, 
                                  edgecolor='white', linewidth=0.8, marker='^')
    
    # Plot sig_ubeh cells
    subset_df1 = df1[(df1['sig_ubeh']==True) & (df1['sig_upup']==False)]
    subset_df2 = df2[(df2['sig_ubeh']==True) & (df2['sig_upup']==False)]
    print('ubeh n = {}'.format(len(subset_df1)))
    color = colors[2]
    if area=='A1' or area=='ICX':
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, 
                                  s=dot_size2, edgecolor='white', linewidth=0.8)
    else:
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, 
                                  s=dot_size2, edgecolor='white', linewidth=0.8, marker='^')
    
    # Plot sig_upup cells 
    subset_df1 = df1[(df1['sig_ubeh']==False) & (df1['sig_upup']==True)]
    subset_df2 = df2[(df2['sig_ubeh']==False) & (df2['sig_upup']==True)]
    print('upup n = {}'.format(len(subset_df1)))
    color = colors[3]
    if area=='A1' or area=='ICX':
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, 
                                  s=dot_size2, edgecolor='white', linewidth=0.8)
    else:
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, 
                                  s=dot_size2, edgecolor='white', linewidth=0.8, marker='^')
    
    # Plot the significant both pup and beh
    subset_df1 = df1[(df1['sig_ubeh']==True) & (df1['sig_upup']==True)]
    subset_df2 = df2[(df2['sig_ubeh']==True) & (df2['sig_upup']==True)]
    print('ubeh and upup = {}'.format(len(subset_df1)))
    color = colors[4]
    if area=='A1' or area=='ICX':
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, 
                                  s=dot_size2, edgecolor='white', linewidth=0.8)
    else:
        scatplot.ax_joint.scatter(subset_df1[column1], subset_df2[column2], color=color, 
                                  s=dot_size2, edgecolor='white', linewidth=0.8, marker='^')

    
    plt.axvline(0, linestyle='--', linewidth=0.5, color='k')
    plt.axhline(0, linestyle='--', linewidth=0.5, color='k')

    # need a slope and c to fix the position of line
    slope = 1
    c = xlim[0]

    x_min = xlim[0]
    x_max = xlim[1]
    y_min, y_max = c, c + slope*(x_max-x_min)
    plt.plot([x_min, x_max], [y_min, y_max], linewidth=0.5, linestyle='--', color='k')
    
    plt.xlabel(unit_set+'_'+column1)
    plt.ylabel(unit_set+'_'+column2) 
    
    #plt.savefig(unit_set+'_'+column1+column2+title+area+'.pdf')