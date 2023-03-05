#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:59:33 2018

@author: daniela
"""
#import nems_lbhb.xform_wrappers as nw
import nems0.xform_helper as xhelp

cellid="BRT036b-06-1"
modelname="psth.fs20.pup-ld-st.pup.beh-evs.tar.lic0_fir.Nx40-lvl.1-stategain.3_jk.nf10-init.st-basic"
batch=301
xf, ctx = xhelp.load_model_xform(cellid, batch, modelname)
