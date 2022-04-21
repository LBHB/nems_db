#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:59:40 2018

@author: hellerc
"""

'''
Demo plotting pupil-behavior results from fit results folder
click event on scatter plot will load png for that cell
'''


import nems_lbhb.stateplots as sp
import pandas as pd


#results_path = '/auto/users/svd/projects/pupil-behavior/ppas_AC_301_psth.fs20_psthfr_sdexp.S_jk.nf10-init.st-basic'
results_path = '/auto/users/svd/projects/pupil-behavior/pb_AC_301_psth.fs20.pup_psthfr_sdexp.S_jk.nf10-init.st-basic'
#results_path = '/auto/users/svd/projects/pupil-behavior/ppas_IC_303_psth.fs20.pup_psthfr_sdexp.S_jk.nf10-init.st-basic'
#results_path = '/auto/users/svd/projects/pupil-behavior/pb_IC_303_psth.fs20.pup_psthfr_sdexp.S_jk.nf10-init.st-basic'

df = pd.read_csv(results_path+'/results.csv')

#beta1 = 'r_pup'
#beta2 = 'r_beh'
beta1 = 'r_beh_pup0'
beta2 = 'r_beh'

n1 = 'Task (pupil shuffled)'
n2 = 'Task'
hist_range = [-0.05, .1]
title = 'Explained Variarnce'


# if looking at AC cells
# cellids = ['TAR010c-06-1', 'TAR010c-27-2', 'BRT026c-29-1', 'BRT026c-17-1']

# if looking at IC cells
cellids = ['TAR010c-06-1', 'BRT026c-29-1']
highlight = []
for c in df['cellid']:
    if c in cellids:
        highlight.append(1)
    else:
        highlight.append(0)
        
#highlight = None # to plot all points and be able to click on them

# If highlight is NOT none, right-click to load figure for non-highlighted cells.
# left-click to load highlighted cells.
# If highlight is NONE, use left click for any cells

fh = sp.beta_comp_from_folder(beta1=beta1, beta2=beta2, n1=n1, n2=n2, title=title,
                              hist_range=hist_range, folder=results_path, highlight=highlight)

