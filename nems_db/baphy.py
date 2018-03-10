#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017
@author: svd, changes added by njs
"""

import logging
log = logging.getLogger(__name__)

import re
import os
import os.path
import scipy.io
import scipy.ndimage.filters
import scipy.signal
import numpy as np
import sys
import io

import pandas as pd
import matplotlib.pyplot as plt
import nems.signal
import nems.recording

try:
    import nems_db.db as db
except Exception as e:
    log.info(e)
    log.info('Running without db')
    db = None
'''
try:
    import nems_config.Storage_Config as sc
except Exception as e:
    log.info(e)
    from nems_config.defaults import STORAGE_DEFAULTS
    sc = STORAGE_DEFAULTS
'''
# paths to baphy data -- standard locations on elephant
stim_cache_dir='/auto/data/tmp/tstim/'  # location of cached stimuli
spk_subdir='sorted/'   # location of spk.mat files relative to parmfiles


""" TODO : DELETE OR PRUNE EVERYTHING DOWN TO THE NATIVE BAPHY FUNCTIONS AT END """

def load_baphy_file(filepath, level=0):
    """
    This function loads data from a BAPHY .mat file located at 'filepath'.
    It returns two dictionaries contining the file data and metadata,
    'data' and 'meta', respectively. 'data' contains:
        {'stim':stimulus,'resp':response raster,'pupil':pupil diameters}
    Note that if there is no pupil data in the BAPHY file, 'pup' will return
    None. 'meta' contains:
        {'stimf':stimulus frequency,'respf':response frequency,'iso':isolation,
             'tags':stimulus tags,'tagidx':tag idx, 'ff':frequency channel bins,
             'prestim':prestim silence,'duration':stimulus duration,'poststim':
             poststim silence}
    """
    data = dict.fromkeys(['stim', 'resp', 'pupil'])
    matdata = scipy.io.loadmat(filepath, chars_as_strings=True)
    s = matdata['data'][0][level]
    print(s['fn_spike'])
    try:
        data = {}
        data['resp'] = s['resp_raster']
        data['stim'] = s['stim']
        data['respFs'] = s['respfs'][0][0]
        data['stimFs'] = s['stimfs'][0][0]
        data['stimparam'] = [str(''.join(letter)) for letter in s['fn_param']]
        data['isolation'] = s['isolation']
        data['prestim'] = s['tags'][0]['PreStimSilence'][0][0][0]
        data['poststim'] = s['tags'][0]['PostStimSilence'][0][0][0]
        data['duration'] = s['tags'][0]['Duration'][0][0][0]

        data['cellids']=s['cellids'][0]
        data['resp_fn']=s['fn_spike']
    except BaseException:
        data['raw_stim'] = s['stim'].copy()
        data['raw_resp'] = s['resp'].copy()
    try:
        data['pupil'] = s['pupil']
    except BaseException:
        data['pupil'] = None
    try:
        if s['estfile']:
            data['est'] = True
        else:
            data['est'] = False
    except ValueError:
        log.info("Est/val conditions not flagged in datafile")
    return(data)


def get_celldb_file(batch, cellid, fs=200, stimfmt='ozgf',
                    chancount=18, pertrial=False):
    """
    Given a stim/resp preprocessing parameters, figure out relevant cache filename.
    TODO: if cache file doesn't exist, have Matlab generate it
    @author: svd
    """

    rootpath = os.path.join(sc.DIRECTORY_ROOT, "nems_in_cache")
    if pertrial or batch in [269, 273, 284, 285]:
        ptstring = "_pertrial"
    else:
        ptstring = ""

    if stimfmt in ['none', 'parm', 'envelope']:
        fn = "{0}/batch{1}/{2}_b{1}{6}_{3}_fs{5}.mat".format(
            rootpath, batch, cellid, stimfmt, chancount, fs, ptstring)
    else:
        fn = "{0}/batch{1}/{2}_b{1}{6}_{3}_c{4}_fs{5}.mat".format(
            rootpath, batch, cellid, stimfmt, chancount, fs, ptstring)

    # placeholder. Need to check if file exists in nems_in_cache.
    # If not, call baphy function in Matlab to regenerate it:
    # fn=export_cellfile(batchid,cellid,fs,stimfmt,chancount)

    return fn


def load_baphy_ssa(filepath):
    """
    Load SSAs from matlab, cherrypiking the most convenient values
    of the mat file and parsing them into a dictionary. The mat file contains
    a struct wich can be a vector (multiple related recording of the same cell)
    therefore each poin in this vector is parced into a dictionary within a list.
    """
    matdata = scipy.io.loadmat(filepath, chars_as_strings=True)
    d = matdata['data']
    datalist = list()
    for i in range(d.size):
        m = d[0][i]
        data = dict.fromkeys(['stim', 'resp', 'tags'])
        try:
            params = m['stimparam'][0][0]
            data['PipDuration'] = round(params['Ref_PipDuration'][0][0], 4)
            data['PipInterval'] = round(params['Ref_PipInterval'][0][0], 4)
            Freq = params['Ref_Frequencies'].squeeze()
            data['Frequencies'] = Freq.tolist()
            Rates = params['Ref_F1Rates'].squeeze()
            data['Rates'] = Rates.tolist()
            data['Jitter'] = params['Ref_Jitter'][0][:]
            data['MinInterval'] = round(params['Ref_MinInterval'][0][0], 4)
        except BaseException:
            pass

        data['stimfmt'] = m['stimfmt'][0]

        if m['stimfmt'][0] == 'envelope':
            resp = np.swapaxes(m['resp_raster'], 0, 2)
            stim = np.squeeze(m['stim'])  # stim envelope seems to not be binay
            stim = stim / stim.max()
            stim = np.where(stim < 0.5, 0, 1)  # trasnforms stim to binary
            stim = np.swapaxes(stim, 1, 2)
            stim = stim[:, :, 0:resp.shape[2]]

            data['stim'] = stim

        elif m['stimfmt'][0] == 'none':
            data['stim'] = []
            resp = np.swapaxes(m['resp_raster'], 0, 1)

        data['resp'] = resp

        data['stimf'] = m['stimfs'][0][0]
        respf = m['respfs'][0][0]
        data['respf'] = respf
        data['isolation'] = round(m['isolation'][0][0], 4)
        try:
            data['tags'] = np.concatenate(m['tags'][0]['tags'][0][0], axis=0)
        except BaseException:
            pass
        try:
            data['tagidx'] = m['tags'][0]['tagidx'][0][0]
            data['ff'] = m['tags'][0]['ff'][0][0]
        except BaseException:
            pass
        prestim = m['tags'][0]['PreStimSilence'][0][0][0]
        data['prestim'] = prestim
        duration = m['tags'][0]['Duration'][0][0][0]
        data['duration'] = duration
        poststim = resp.shape[2] - (int(prestim) + int(duration)) * int(respf)
        data['poststim'] = poststim / respf

        try:
            data['pup'] = m['pupil']
        except BaseException:
            data['pup'] = None
        datalist.append(data)

    return (datalist)

def load_spike_raster(spkfile, options, nargout=None):
    '''
    # CRH added 1-5-2018, work in progress - meant to mirror the output of
    baphy's loadspikeraster

    inputs:
        spkfile - name of .spk.mat file generated using meska
        options - structure can contain the following fields:
            channel - electrode channel (default 1)
            unit - unit number (default 1)
            rasterfs in Hz (default 1000)
            includeprestim - raster includes silent period before stimulus onset
            tag_masks - cell array of strings to filter tags, eg,
                {'torc','reference'} or {'target'}.  AND logic.  default ={}
            psthonly - shape of r (default -1, see below)
            sorter - preferentially load spikes sorted by sorter.  if not
                sorted by sorter, just take primary sorting
            lfpclean - [0] if 1, remove MSE prediction of spikes (single
                trial) predicted by LFP
            includeincorrect - if 1, load all trials, not just correct (default 0)
            runclass - if set, load only data for runclass when multiple runclasses
                in file

        nargout: number of arguments to return

     outputs:
        r: spike raster
        tags:
        trialset:
        exptevents:
        sortextras: Not possible right now - not cached
        options: Not possibel right now - not cached

    '''




    # ========== see if cache file exists =====================
    need_matlab=0 # if set to 1, use matlab engine to generate the cache file

    # get path to spkfile
    if(len(spkfile.split('/'))==1):
        path_to_spkfile = os.getcwd()
    else:
        path_to_spkfile = os.path.dirname(spkfile)


    # define the cache file name using fucntion written below
    cache_fn= spike_cache_filename(spkfile,options)

    # make cache directory if it doesn't already exist
    path_to_cacheFile = os.path.join(path_to_spkfile,'cache')
    cache_file = os.path.join(path_to_cacheFile,cache_fn)
    print('loading from cache:')
    print(cache_file)
    if(os.path.isdir(path_to_cacheFile) and os.path.exists(cache_file)):
        out = scipy.io.loadmat(cache_file)
        r = out['r']
        tags = out['tags']
        trialset=out['trialset']
        exptevents=out['exptevents']

    elif(os.path.isdir(path_to_cacheFile)):
        need_matlab=1
    else:
        need_matlab = 1
        os.mkdir(path_to_cacheFile)


    # Generate the cache file so that it can be loaded by python
    if need_matlab:
        # only start the matlab engine if the cached file doesn't exist
        import matlab.engine
        eng = matlab.engine.start_matlab()
        baphy_util_path = '/auto/users/hellerc/baphy/Utilities'
        eng.addpath(baphy_util_path,nargout=0)
        # call matlab function to make the requested array
        eng.loadspikeraster(spkfile, options, nargout=0)    #TODO figure out a way to specify nargout without returning anything

        out = scipy.io.loadmat(cache_file)
        r = out['r']
        tags = out['tags']
        trialset=out['trialset']
        exptevents=out['exptevents']

    if nargout==None or nargout==1:
        return r
    elif nargout==2:
        return r, tags
    elif nargout==3:
        return r, tags, trialset
    else:
        return r, tags, trialset, exptevents

def load_pupil_raster(pupfile, options):
    '''
    # CRH added 1-12-2018, work in progress - meant to mirror the output of
    baphy's loadevpraster for pupil=1

    inputs:
        spkfile - name of .spk.mat file generated using meska
        options - structure can contain the following fields:
            pupil: must be = 1 or will not load pupil
            rasterfs in Hz (default 1000)
            includeprestim - raster includes silent period before stimulus onset
            tag_masks - cell array of strings to filter tags, eg,
                {'torc','reference'} or {'target'}.  AND logic.  default ={}
            psthonly - shape of r (default -1, see below)
            sorter - preferentially load spikes sorted by sorter.  if not
                sorted by sorter, just take primary sorting
            lfpclean - [0] if 1, remove MSE prediction of spikes (single
                trial) predicted by LFP
            includeincorrect - if 1, load all trials, not just correct (default 0)
            runclass - if set, load only data for runclass when multiple runclasses
                in file

            pupil_offset   see baphy documentation on loadevpraster
            pupil_median

     outputs:
        p: pupil raster in same shape as spike raster for same params
    '''

    # ========== see if cache file exists =====================
    need_matlab=0 # if set to 1, use matlab engine to generate the cache file

    # get path to spkfile
    if(len(pupfile.split('/'))==1):
        path_to_pupfile = os.getcwd()
    else:
        path_to_pupfile = os.path.dirname(pupfile)


    # define the cache file name using fucntion written below
    cache_fn= pupil_cache_filename(pupfile,options)

    # make cache directory if it doesn't already exist
    path_to_cacheFile = path_to_pupfile+'/tmp/'
    cache_file = os.path.join(path_to_cacheFile,cache_fn)
    print('loading from cache:')
    print(cache_file)
    if(os.path.isdir(path_to_cacheFile) and os.path.exists(cache_file)):
        out = scipy.io.loadmat(cache_file)
        p = out['r']           # it's called r in loadevpraster (where it's generated)
    else:
        need_matlab = 1



    # Generate the cache file so that it can be loaded by python
    if need_matlab:
        # only start the matlab engine if the cached file doesn't exist
        import matlab.engine
        eng = matlab.engine.start_matlab()
        baphy_util_path = '/auto/users/hellerc/baphy/Utilities'
        eng.addpath(baphy_util_path,nargout=0)
        # call matlab function to make the requested array
        eng.loadevpraster(pupfile, options, nargout=0)    # Don't want to pass stuff back. evpraster will cache the file
        out = scipy.io.loadmat(cache_file)
        p = out['r']

    return p


def spike_cache_filename(spkfile,options):
    '''
    Given the spkfile and options passed to load_spike_raster, generate the filename that
    will identify the unique cache file for that cell
    '''

    # parse the input in options
    try: channel=options['channel']
    except: channel=1

    try: unit=options['unit']
    except: unit=1

    try: rasterfs=options['rasterfs']
    except: rasterfs=1000.   # must be float for matlab if matlab engine is called

    try: tag_masks=options['tag_masks']; tag_name='tags-'+''.join(tag_masks);
    except: tag_masks=[]; tag_name='tags-Reference';

    try: runclass=options['runclass']; run='run-'+runclass;
    except: run='run-all';

    if 'includeprestim' in options and type(options['includeprestim'])==int:
        prestim='prestim-1';
    elif 'includeprestim' in options:
        prestim=str(options['includeprestim'])
        while ', ' in prestim:
            prestim=prestim.replace('[','').replace(']','').replace(', ','-')
        prestim='prestim-'+prestim;
    else: prestim='prestim-none';

    try: ic=options['includeincorrect']; ic='allTrials';
    except: ic='correctTrials';

    try: psthonly=options['psthonly']; psthonly=options['psthonly'];
    except: psthonly=-1;

    if len(str(channel))==1:
        ch_str='0'+str(channel)
    else:
        ch_str=str(channel)

    # define the cache file name
    spkfile_root_name=os.path.basename(spkfile).split('.')[0];
    cache_fn=spkfile_root_name+'_ch'+ch_str+'-'+str(unit)+'_fs'+str(int(rasterfs))+'_'+tag_name+'_'+run+'_'+prestim+'_'+ic+'_psth'+str(psthonly)+'.mat'

    return cache_fn

def pupil_cache_filename(pupfile, options):
    # parse the input in options
    try: pupil=options['pupil']; pupil_str='_pup';
    except: sys.exit('options does not set pupil=1')

    if 'pupil_offset' in options:
        offset = options['pupil_offset']
        if offset==0.75: #matlab default in evpraster
            offset_str='';
        else:
            offset_str='_offset-'+str(offset)
    else:
        offset_str=''

    if 'pupil_median' in options:
        med = options['pupil_median']
        if med==0: #matlab default in evpraster
            med_str='';
        else:
            med_str='_med-'+str(med)
    else:
       med_str=''

    try: rasterfs=options['rasterfs']
    except: rasterfs=1000.   # must be float for matlab if matlab engine is called

    try: tag_masks=options['tag_masks']; tag_name='tags-'+''.join(tag_masks);
    except: tag_masks=[]; tag_name='tags-Reference';

    try: runclass=options['runclass']; run='run-'+runclass;
    except: run='run-all';

    if 'includeprestim' in options and type(options['includeprestim'])==int:
        prestim='prestim-1';
    elif 'includeprestim' in options:
        prestim=str(options['includeprestim'])
        while ', ' in prestim:
            prestim=prestim.replace('[','').replace(']','').replace(', ','-')
        prestim='prestim-'+prestim;
    else: prestim='prestim-none';

    try: ic=options['includeincorrect']; ic='allTrials';
    except: ic='correctTrials';

    try: psthonly=options['psthonly']; psthonly=options['psthonly'];
    except: psthonly=-1;



    # define the cache file name
    pupfile_root_name=os.path.basename(pupfile).split('.')[0];


    cache_fn=pupfile_root_name+'_fs'+str(int(rasterfs))+'_'+tag_name+'_'+run+'_'+prestim+'_'+ic+'_psth'+str(psthonly)+offset_str+med_str+pupil_str+'.mat'

    return cache_fn


def spike_cache_filename2(spkfilestub,options):
    '''
    Given the stub for spike cache file and options typically passed to
    load_spike_raster, generate the unique filename for for that cell/format
    '''

    # parse the input in options
    try:
        rasterfs='_fs'+str(options['rasterfs'])
    except:
        rasterfs='_fs1000'

    try:
        tag_name='_tags-'+''.join(options['tag_masks'])
    except:
        tag_name='_tags-default'

    try:
        run='_run-'+options['runclass'];
    except:
        run='_run-all';

    if 'includeprestim' in options and type(options['includeprestim'])==int:
        prestim='_prestim-1';
    elif 'includeprestim' in options:
        prestim=str(options['includeprestim'])
        while ', ' in prestim:
            prestim=prestim.replace('[','').replace(']','').replace(', ','-')
        prestim='_prestim-'+prestim
    else:
        prestim='_prestim-none'

    try:
        if options['includeincorrect']:
            ic='_allTrials'
        else:
            ic='_correctTrials'
    except:
        ic='_correctTrials';

    try:
        psthonly='_psth'+str(options['psthonly'])
    except:
        psthonly='_psth-1'

    # define the cache file name
    cache_fn=spkfilestub+rasterfs+tag_name+run+prestim+ic+psthonly+'.mat'

    return cache_fn

def pupil_cache_filename2(pupfilestub,options):
    '''
    Given the stub for spike cache file and options typically passed to
    load_spike_raster, generate the unique filename for for that cell/format
    '''

    # parse the input in options
    try: pupil=options['pupil']; pupil_str='_pup';
    except: sys.exit('options does not set pupil=1')

    if 'pupil_offset' in options:
        offset = options['pupil_offset']
        if offset==0.75: #matlab default in evpraster
            offset_str='';
        else:
            offset_str='_offset-'+str(offset)
    else:
        offset_str=''

    if 'pupil_median' in options:
        med = options['pupil_median']
        if med==0: #matlab default in evpraster
            med_str='';
        else:
            med_str='_med-'+str(med)
    else:
       med_str=''

    try:
        rasterfs='_fs'+str(options['rasterfs'])
    except:
        rasterfs='_fs1000'

    try:
        tag_name='_tags-'+''.join(options['tag_masks'])
    except:
        tag_name='_tags-default'

    try:
        run='_run-'+options['runclass'];
    except:
        run='_run-all';

    if 'includeprestim' in options and type(options['includeprestim'])==int:
        prestim='_prestim-1';
    elif 'includeprestim' in options:
        prestim=str(options['includeprestim'])
        while ', ' in prestim:
            prestim=prestim.replace('[','').replace(']','').replace(', ','-')
        prestim='_prestim-'+prestim
    else:
        prestim='_prestim-none'

    try:
        if options['includeincorrect']:
            ic='_allTrials'
        else:
            ic='_correctTrials'
    except:
        ic='_correctTrials';

    try:
        psthonly='_psth'+str(options['psthonly'])
    except:
        psthonly='_psth-1'

    filt_str='';
    try:
        if options['pupil_highpass']>0:
            filt_str="{0}_hp{1:.2f}".format(filt_str,options['pupil_highpass'])
    except:
        pass
    try:
        if options['pupil_lowpass']>0:
           filt_str="{0}_lp{1:.2f}".format(filt_str,options['pupil_lowpass'])
    except:
        pass
    try:
        if options['pupil_derivative']!='':
           filt_str="{0}_D{1}".format(filt_str,options['pupil_derivative'])
    except:
        pass

    # define the cache file name
    #for i, pf in enumerate(pupfilestub):
    #    l = pf.split('.')[0].split('/')
    #   pupfilestub.iloc[i] = '/'.join(l[:-1])+'/tmp/'+l[-1]

    cache_fn=pupfilestub+rasterfs+tag_name+run+prestim+ic+psthonly+offset_str+med_str+filt_str+pupil_str+'.mat'

    return cache_fn


def stim_cache_filename(stimfile, options={}):
    """
    mimic cache file naming scheme from loadstimfrombaphy.m
    mfile syntax:
    % function [stim,stimparam]=loadstimfrombaphy(parmfile,startbin,stopbin,
    %                   filtfmt,fsout[=1000],chancount[=30],forceregen[=0],includeprestim[=0],SoundHandle[='ReferenceHandle'],repcount[=1]);
    SVD 2018-01-15
    """

    try:
        filtfmt=options['filtfmt']
    except:
        filtfmt='parm'

    try:
        fs='-fs'+str(options['fsout'])
    except:
        fs='-fs100'

    try:
        ch='-ch'+str(options['chancount'])
    except:
        ch='-ch0'

    try:
        if options['includeprestim']:
            incps='-incps1'
        else:
            incps=''
    except:
        incps='-incps1'

    #ppdir=['/auto/data/tmp/tstim/'];
    cache_fn=stimfile + '-' + filtfmt + fs + ch + incps + '.mat'

    return cache_fn

def load_site_raster(batch, site, options, runclass=None, rawid=None):
    '''
    Load a population raster given batch id, runclass, recording site, and options

    Input:
    -------------------------------------------------------------------------
    batch ex: batch=301
    runclass ex: runclass='PTD'
    recording site ex: site='TAR010c'

    options: dict
        min_isolation (float), default - load all cells above 70% isolation
        rasterfs (float), default: 100
        includeprestim (boolean), default: 1
        tag_masks (list of strings), default [], ex: ['Reference']
        active_passive (string), default: 'both' ex: 'p' or 'a'

        **** other options to be implemented ****

    Output:
    -------------------------------------------------------------------------
    r (numpy array), response matrix (bin x reps x stim x cells)
    meta (pandas data frame), df with all information from data base about the cells you've loaded (sorted in same order as last dim of r)

    '''

    # Parse inputs
    try: iso=options['min_isolation'];
    except: iso=70;

    try: options['rasterfs'];
    except: options['rasterfs']=100;

    try: options['includeprestim'];
    except: options['includeprestim']=1;

    try: options['tag_masks'];
    except: pass

    try: active_passive = options['active_passive'];
    except: active_passive='both';

    # Define parms dict to be passed to loading function
    parms=options;
    if 'min_isolation' in parms:
        del parms['min_isolation']   # not need to find cache file
    if 'active_passive' in parms:
        del parms['active_passive']

    cfd=db.get_batch_cells(batch=batch,cellid=site,rawid=rawid)


    cfd=cfd.sort_values('cellid') # sort the data frame by cellid so it agrees with the r matrix output
    cfd=cfd[cfd['min_isolation']>iso]
    cellids=np.sort(np.unique(cfd[cfd['min_isolation']>iso]['cellid'])) # only need list of unique id's

    # load data for all identified respfiles corresponding to cellids

    cellcount=len(cellids)

    a_p=[] # classify as active or passive based on respfile name

    for i, cid in enumerate(cellids):
        d=db.get_batch_cell_data(batch,cid,rawid=rawid)
        respfile=nu.baphy.spike_cache_filename2(d['raster'],parms)
        for j, rf in enumerate(respfile):
            rts=nu.io.load_matlab_matrix(rf,key="r")

            if i == 0:
                if '_a_' in rf:
                    a_p = a_p+[1]*rts.shape[1]
                else:
                    a_p = a_p+[0]*rts.shape[1]

            if j == 0:
                rt = rts;
            else:
                rt = np.concatenate((rt,rts),axis=1)
        if i == 0:
            r = np.empty((rt.shape[0],rt.shape[1],rt.shape[2],cellcount))
            r[:,:,:,0]=rt;
        else:
            r[:,:,:,i]=rt;

    if active_passive is 'a':
        r = r[:,np.array(a_p)==1,:,:]
    elif active_passive is 'p':
        r = r[:,np.array(a_p)==0,:,:];

    return r, cfd

def load_pup_raster(batch, site, options,runclass=None,rawid=None):
    '''
    Load a pupil raster given batch id, runclass, recording site, and options

    Input:
    -------------------------------------------------------------------------
    batch ex: batch=301
    runclass ex: runclass='PTD'
    recording site ex: site='TAR010c'

    options: dict
        rasterfs (float), default: 100
        includeprestim (boolean), default: 1
        tag_masks (list of strings), default [], ex: ['Reference']
        active_passive (string), default: 'both' ex: 'p' or 'a'

        **** other pupil options to be implemented ****

    Output:
    -------------------------------------------------------------------------
    p (numpy array), response matrix (bin x reps x stim)

    '''

    try: options['rasterfs'];
    except: options['rasterfs']=100;

    try: options['includeprestim'];
    except: options['includeprestim']=1;

    try: options['tag_masks'];
    except: pass

    try: active_passive = options['active_passive'];
    except: active_passive='both';

    try: derivative=options['derivative'];
    except: derivative=False;


    options['pupil']=1;

    d=db.get_batch_cell_data(batch,rawid=rawid)
    files = []
    for f in d['pupil'].unique():
        f = f.split('.')[0]
        if f is None:
            pass
        elif runclass is not None:
            if runclass in f and site in f:
                files.append(f)
        else:
            if site in f:
                files.append(f)

    files = pd.Series(files)


    pupfile=nu.baphy.pupil_cache_filename2(files,options)
    a_p = []
    for j, rf in enumerate(pupfile):

        pts=nu.io.load_matlab_matrix(pupfile.iloc[j],key='r')

        if '_a_' in rf:
            a_p = a_p+[1]*pts.shape[1]
        else:
            a_p = a_p+[0]*pts.shape[1]

        if j == 0:
            p = pts;
        else:
            p = np.concatenate((p,pts),axis=1)

    if active_passive is 'a':
        p = p[:,np.array(a_p)==1,:]
    elif active_passive is 'p':
        p = p[:,np.array(a_p)==0,:]

    return p


""" FROM HERE DOWN IS NEW SUPPORT FOR LOADING DIRECTLY FROM NATIVE
    BAPHY FILES
    """

def baphy_mat2py(s):

    s3=re.sub(r';', r'', s.rstrip())
    s3=re.sub(r'%',r'#',s3)
    s3=re.sub(r'\\',r'/',s3)
    s3=re.sub(r"\.([a-zA-Z0-9]+)'",r"XX\g<1>'",s3)
    s3=re.sub(r"\.([a-zA-Z0-9]+) ,",r"XX\g<1> ,",s3)
    s3=re.sub(r'globalparams\(1\)',r'globalparams',s3)
    s3=re.sub(r'exptparams\(1\)',r'exptparams',s3)

    s4=re.sub(r'\(([0-9]*)\)', r'[\g<1>]', s3)

    s5=re.sub(r'\.([A-Za-z][A-Za-z1-9_]+)', r"['\g<1>']", s4)

    s6=re.sub(r'([0-9]+) ', r"\g<0>,", s5)
    s6=re.sub(r'NaN ', r"np.nan,", s6)
    s6=re.sub(r'Inf ', r"np.inf,", s6)

    s7=re.sub(r"XX([a-zA-Z0-9]+)'",r".\g<1>'",s6)
    s7=re.sub(r"XX([a-zA-Z0-9]+) ,",r".\g<1> ,",s7)
    s7=re.sub(r',,',r',',s7)
    s7=re.sub(r',Hz',r'Hz',s7)
    s7=re.sub(r'NaN',r'np.nan',s7)
    s7=re.sub(r'zeros\(([0-9,]+)\)',r'np.zeros([\g<1>])',s7)
    s7=re.sub(r'{(.*)}',r'[\g<1>]',s7)

    return s7

def baphy_parm_read(filepath):
    print("Loading {0}".format(filepath))

    f = io.open(filepath, "r")
    s=f.readlines(-1)

    globalparams={}
    exptparams={}
    exptevents={}

    for ts in s:
        sout=baphy_mat2py(ts)
        #print(sout)
        try:
            exec(sout)
        except KeyError:
            ts1=sout.split('= [')
            ts1=ts1[0].split(',[')

            s1=ts1[0].split('[')
            sout1="[".join(s1[:-1]) + ' = {}'
            try:
                exec(sout1)
            except :
                s2=sout1.split('[')
                sout2="[".join(s2[:-1]) + ' = {}'
                try:
                    exec(sout2)
                except:
                    s3=sout2.split('[')
                    sout3="[".join(s3[:-1]) + ' = {}'
                    exec(sout3)
                    exec(sout2)

                exec(sout1)
            exec(sout)
        except NameError:
            print("NameError on: {0}".format(sout))
        except:
            print("Other error on: {0} to {1}".format(ts,sout))

    # special conversions

    # convert exptevents to a DataFrame:
    t=[exptevents[k] for k in exptevents]
    d=pd.DataFrame(t)
    exptevents=d.drop(['Rove'],axis=1)

    # rename columns to NEMS standard epoch names
    exptevents.columns=['name','start','end','Trial']
    for i in range(0,len(exptevents)):
        if exptevents.loc[i,'end'] == []:
            exptevents.loc[i,'end']=exptevents.loc[i,'start']

    return globalparams, exptparams, exptevents

def baphy_load_specgram(stimfilepath):

    matdata = scipy.io.loadmat(stimfilepath, chars_as_strings=True)

    stim=matdata['stim']

    stimparam=matdata['stimparam'][0][0]

    try:
        # case 1: loadstimfrombaphy format
        # remove redundant tags from tag list and stimulus array
        d=matdata['stimparam'][0][0][0][0]
        d=[x[0] for x in d]
        tags,tagids=np.unique(d, return_index=True)

        stim=stim[:,:,tagids]
    except:
        # loadstimbytrial format. don't want to filter by unique tags.
        # field names within stimparam don't seem to be preserved in this load format??
        d=matdata['stimparam'][0][0][2][0]
        tags=[x[0] for x in d]

    return stim,tags,stimparam


def baphy_stim_cachefile(exptparams,options,parmfilepath=None):
    """
    generate cache filename generated by loadstimfrombaphy

    code adapted from loadstimfrombaphy.m
    """

    if 'truncatetargets' not in options:
        options['truncatetargets']=1
    if 'pertrial' not in options:
        options['pertrial']=False
    if options['pertrial']:
        # loadstimbytrial cache filename format
        pp,bb=os.path.split(parmfilepath)
        bb=bb.split(".")[0]
        dstr="loadstimbytrial_{0}_ff{1}_fs{2}_cc{3}_trunc{4}.mat".format(
             bb,options['stimfmt'],options['rasterfs'],
             options['chancount'],options['truncatetargets'])
        return stim_cache_dir + dstr

    # otherwise use standard load stim from baphy format
    if options['runclass'] is None:
        RefObject=exptparams['TrialObject'][1]['ReferenceHandle'][1]
    elif 'runclass' in exptparams.keys():
        runclass=exptparams['runclass'].split("_")
        if len(runclass)>1 and runclass[1]==options["runclass"]:
            RefObject=exptparams['TrialObject'][1]['TargetHandle'][1]
        else:
            RefObject=exptparams['TrialObject'][1]['ReferenceHandle'][1]
    else:
        RefObject=exptparams['TrialObject'][1]['ReferenceHandle'][1]

    dstr=RefObject['descriptor']
    if dstr=='Torc':
        if 'RunClass' in exptparams['TrialObject'][1].keys():
            dstr=dstr+'-'+exptparams['TrialObject'][1]['RunClass'];
        else:
            dstr=dstr+'-TOR'

    # include all parameter values, even defaults, in filename
    fields=RefObject['UserDefinableFields']
    for cnt1 in range(0,len(fields),3):
        dstr="{0}-{1}".format(dstr,RefObject[fields[cnt1]])

    dstr=re.sub(r":",r"",dstr)

    if 'OveralldB' in exptparams['TrialObject'][1]:
        OveralldB=exptparams['TrialObject'][1]['OveralldB']
        dstr=dstr + "-{0}dB".format(OveralldB)
    else:
        OveralldB=0

    dstr=dstr+"-{0}-fs{1}-ch{2}".format(options['stimfmt'],options['rasterfs'],options['chancount'])

    if options['includeprestim']:
        dstr=dstr+'-incps1'

    dstr=re.sub(r"[ ,]",r"_",dstr)
    dstr=re.sub(r"[\[\]]",r"",dstr)

    return stim_cache_dir + dstr + '.mat'


def baphy_load_spike_data_raw(spkfilepath,channel=None,unit=None):

    matdata = scipy.io.loadmat(spkfilepath)#, chars_as_strings=True)

    sortinfo=matdata['sortinfo']
    if sortinfo.shape[0]>1:
        sortinfo=sortinfo.T
    sortinfo=sortinfo[0]

    # figure out sampling rate, used to convert spike times into seconds
    spikefs=matdata['rate'][0][0]

    return sortinfo,spikefs


def baphy_align_time(exptevents,sortinfo,spikefs,finalfs=0):

    # number of channels in recording (not all necessarily contain spikes)
    chancount=len(sortinfo)

    # figure out how long each trial is by the time of the last spike count.
    # this method is a hack! but since recordings are longer than the "official"
    # trial end time reported by baphy, this method preserves extra spikes
    TrialCount=np.max(exptevents['Trial'])

    TrialLen_sec=np.array(exptevents.loc[exptevents['name']=="TRIALSTOP"]['start'])
    TrialLen_spikefs=np.concatenate((np.zeros([1,1]),TrialLen_sec[:,np.newaxis]*spikefs),axis=0)

    for c in range(0,chancount):
        if len(sortinfo[c]) and sortinfo[c][0].size:
            s=sortinfo[c][0][0]['unitSpikes']
            s=np.reshape(s,(-1,1))
            unitcount=s.shape[0]
            for u in range(0,unitcount):
                st=s[u,0]

                #print('chan {0} unit {1}: {2} spikes'.format(c,u,st.shape[1]))
                for trialidx in range(1,TrialCount+1):
                    ff=(st[0,:]==trialidx)
                    if np.sum(ff):
                        utrial_spikefs=np.max(st[1,ff])
                        TrialLen_spikefs[trialidx,0]=np.max([utrial_spikefs,TrialLen_spikefs[trialidx,0]])

    # using the trial lengths, figure out adjustments to trial event times.
    if finalfs:
        print('rounding Trial offset spike times to even number of rasterfs bins')
        #print(TrialLen_spikefs)
        TrialLen_spikefs=np.ceil(TrialLen_spikefs/spikefs*finalfs)/finalfs*spikefs
        #print(TrialLen_spikefs)

    Offset_spikefs=np.cumsum(TrialLen_spikefs)
    Offset_sec=Offset_spikefs / spikefs  # how much to offset each trial

    # adjust times in exptevents to approximate time since experiment started
    # rather than time since trial started (native format)
    for Trialidx in range(1,TrialCount+1):
        #print("Adjusting trial {0} by {1} sec".format(Trialidx,Offset_sec[Trialidx-1]))
        ff= (exptevents['Trial'] == Trialidx)
        exptevents.loc[ff,['start','end']]=exptevents.loc[ff,['start','end']]+Offset_sec[Trialidx-1]

        #ff= (exptevents['Trial'] == Trialidx) & (exptevents['end'] > Offset_sec[Trialidx])
        #badevents,=np.where(ff)
        #print("{0} events past end of trial?".format(len(badevents)))
        #exptevents.drop(badevents)

    print("{0} trials totaling {1:.2f} sec".format(TrialCount,Offset_sec[-1]))

    # convert spike times from samples since trial started to
    # (approximate) seconds since experiment started (matched to exptevents)
    totalunits=0
    spiketimes=[]  # list of spike event times for each unit in this recording
    unit_names=[]  # string suffix for each unit (CC-U)
    chan_names=['a','b','c','d','e','f','g','h']
    for c in range(0,chancount):
        if len(sortinfo[c]) and sortinfo[c][0].size:
            s=sortinfo[c][0][0]['unitSpikes']
            comment=sortinfo[c][0][0][0][0][2][0]
            s=np.reshape(s,(-1,1))
            unitcount=s.shape[0]
            for u in range(0,unitcount):
                st=s[u,0]
                uniquetrials=np.unique(st[0,:])
                #print('chan {0} unit {1}: {2} spikes {3} trials'.format(c,u,st.shape[1],len(uniquetrials)))

                unit_spike_events=np.array([])
                for trialidx in uniquetrials:
                    ff=(st[0,:]==trialidx)
                    this_spike_events=st[1,ff]+Offset_spikefs[np.int(trialidx-1)]
                    if comment=='PC-cluster sorted by mespca.m':
                        # remove last spike, which is stray
                        this_spike_events=this_spike_events[:-1]
                    unit_spike_events=np.concatenate((unit_spike_events,this_spike_events),axis=0)
                    #print("   trial {0} first spike bin {1}".format(trialidx,st[1,ff]))

                totalunits+=1
                if chancount<=8:
                    unit_names.append("{0}{1}".format(chan_names[c],u+1))
                else:
                    unit_names.append("{0:02d}-{1}".format(c+1,u+1))
                spiketimes.append(unit_spike_events / spikefs)

    return exptevents,spiketimes,unit_names


def baphy_load_pupil_trace(pupilfilepath,exptevents,options={}):
    """ returns big_rs which is pupil trace resampled to options['rasterfs']
        and strialidx, which is the index into big_rs for the start of each
        trial. need to make sure the big_rs vector aligns with the other signals
    """

    rasterfs = options.get('rasterfs', 1000)
    pupil_offset = options.get('pupil_offset', 0.75)
    pupil_deblink = options.get('pupil_deblink',True)
    pupil_median = options.get('pupil_median',0)
    pupil_smooth = options.get('pupil_smooth',0)
    pupil_highpass = options.get('pupil_highpass',0)
    pupil_lowpass = options.get('pupil_lowpass',0)
    pupil_bandpass = options.get('pupil_bandpass',0)
    pupil_derivative = options.get('pupil_derivative','')
    pupil_mm = options.get('pupil_mm',False)
    verbose = options.get('verbose', False)

    if pupil_smooth:
        raise ValueError('pupil_smooth not implemented. try pupil_median?')
    if pupil_highpass:
        raise ValueError('pupil_highpass not implemented.')
    if pupil_lowpass:
        raise ValueError('pupil_lowpass not implemented.')
    if pupil_bandpass:
        raise ValueError('pupil_bandpass not implemented.')
    if pupil_derivative:
        raise ValueError('pupil_derivative not implemented.')
    if pupil_mm:
        raise ValueError('pupil_mm not implemented.')

    matdata = scipy.io.loadmat(pupilfilepath)

    p=matdata['pupil_data']
    params=p['params']
    if 'pupil_variable_name' not in options:
        options['pupil_variable_name']=params[0][0]['default_var'][0][0][0]
    if 'pupil_algorithm' not in options:
        options['pupil_algorithm']=params[0][0]['default'][0][0][0]

    results=p['results'][0][0][-1][options['pupil_algorithm']]
    pupil_diameter=np.array(results[0][options['pupil_variable_name']][0][0])

    fs_approximate = 10  # approx video framerate
    if pupil_deblink:
        dp = np.abs(np.diff(pupil_diameter,axis=0))
        blink = np.zeros(dp.shape)
        blink[dp > np.mean(dp) + 6*np.std(dp)]= 1
        box=np.ones([fs_approximate])/(fs_approximate)
        blink=np.convolve(blink[:,0],box,mode='same')
        blink[blink>0]=1
        blink[blink<=0]=0
        onidx,=np.where(np.diff(blink) > 0)
        offidx,=np.where(np.diff(blink) < 0)
        if onidx[0]>offidx[0]:
            onidx=np.concatenate((np.array([0]),onidx))
        if len(onidx)>len(offidx):
            offidx=np.concatenate((offidx,np.array([len(blink)])))
        deblinked = pupil_diameter.copy()
        for i,x1 in enumerate(onidx):
            x2 = offidx[i]
            if x2<x1:
                print([i,x1,x2])
                print("WHAT'S UP??")
            else:
                #print([i,x1,x2])
                deblinked[x1:x2,0] = np.linspace(deblinked[x1], deblinked[x2-1], x2-x1)

        if verbose:
            plt.figure()
            plt.plot(pupil_diameter)
            plt.plot(deblinked)
            plt.xlabel('Frame')
            plt.ylabel('Pupil')
            plt.legend('Raw', 'Deblinked')
            plt.title("Artifacts detected: {}".format(len(onidx)))
        pupil_diameter=deblinked

    #
    # resample and remove dropped frames

    #find and parse pupil events
    pp = ['PUPIL,' in x['name'] for i,x in exptevents.iterrows()]
    trials=list(exptevents.loc[pp,'Trial'])
    ntrials=len(trials)
    timestamp=np.zeros([ntrials+1])
    firstframe=np.zeros([ntrials+1])
    for i,x in exptevents.loc[pp].iterrows():
        t=x['Trial']-1
        s=x['name'].split(",[")
        p=eval("["+s[1])
        #print("{0} p=[{1}".format(i,s[1]))
        timestamp[t]=p[0]
        firstframe[t]=int(p[1])
    pp = ['PUPILSTOP' in x['name'] for i,x in exptevents.iterrows()]
    lastidx=np.argwhere(pp)[-1]

    s=exptevents.iloc[lastidx[0]]['name'].split(",[")
    p=eval("["+s[1])
    timestamp[-1]=p[0]
    firstframe[-1]=int(p[1])


    # align pupil with other events, probably by removing extra bins from between trials
    ff= (exptevents['name'] == 'TRIALSTART')
    start_events=exptevents.loc[ff,['start']].reset_index()
    start_events['StartBin']=(np.round(start_events['start']*options['rasterfs'])).astype(int)
    start_e=list(start_events['StartBin'])
    ff= (exptevents['name'] == 'TRIALSTOP')
    stop_events=exptevents.loc[ff,['start']].reset_index()
    stop_events['StopBin']=(np.round(stop_events['start']*options['rasterfs'])).astype(int)
    stop_e=list(stop_events['StopBin'])


    #calculate frame count and duration of each trial
    duration = np.diff(timestamp) * 24*60*60
    frame_count = np.diff(firstframe)

    # warp/resample each trial to compensate for dropped frames
    strialidx=np.zeros([ntrials+1])
    big_rs=np.array([])

    for ii in range(0,ntrials):
        d=pupil_diameter[int(firstframe[ii]):int(firstframe[ii]+frame_count[ii]),0]
        fs = frame_count[ii]/duration[ii]
        t = np.arange(0,len(d))/fs
        ti = np.arange((1/rasterfs)/2, duration[ii]+(1/rasterfs)/2, 1/rasterfs)
        #print("{0} len(d)={1} len(ti)={2} fs={3}".format(ii,len(d),len(ti),fs))
        di=np.interp(ti, t, d)
        big_rs=np.concatenate((big_rs,di),axis=0)
        if ii<ntrials-1 and len(big_rs)>start_e[ii+1]:
            big_rs=big_rs[:start_e[ii+1]]
        elif ii==ntrials-1:
            big_rs=big_rs[:stop_e[ii]]
        strialidx[ii+1]=len(big_rs)

    if pupil_median:
        kernel_size=int(round(pupil_median*rasterfs/2)*2+1)
        big_rs=scipy.signal.medfilt(big_rs, kernel_size=kernel_size)

    # shift pupil trace by offset, usually 0.75 sec
    offset_frames=int(pupil_offset*rasterfs)
    big_rs=np.roll(big_rs,-offset_frames)
    big_rs[-offset_frames:]=np.nan

    # shape to 1 x T to match NEMS signal specs
    big_rs = big_rs[np.newaxis,:]

    return big_rs,strialidx


def baphy_load_data(parmfilepath,options={}):

    """
    this feeds into baphy_load_recording and baphy_load_recording_RDT (see
        below)
    input:
        parmfilepath: baphy parameter file
        options: dictionary of loading options
            runclass: matches Reference1 or Reference2 events, depending

    current outputs:
        exptevents: pandas dataframe with one row per event. times in sec
              since experiment began
        spiketimes: list of lists. outer list indicates unit, inner list is
              the set of spike times (secs since expt started) for that unit
        unit_names: list of strings uniquely identifier each units by
              channel-unitnum (CC-U). can append to siteid- to get cellid
        stim: [channel X time X event] stimulus (spectrogram) matrix
        tags: list of string identifiers associate with each stim event
              (can be used to find events in exptevents)

    other things that could be returned:
        globalparams, exptparams: dictionaries with expt metadata from baphy

    """
    #default_options={'rasterfs':100, 'includeprestim':True,
    #                 'stimfmt':'ozgf', 'chancount':18,
    #                 'cellid': 'all'}
    #options=options.update(default_options)


    # add .m extension if missing
    if parmfilepath[-2:]!=".m":
        parmfilepath=parmfilepath+".m"
    # load parameter file
    globalparams, exptparams, exptevents = baphy_parm_read(parmfilepath)

    # TODO: use paths that match LBHB filesystem? new s3 filesystem?
    #       or make s3 match LBHB?

    # figure out stimulus cachefile to load
    if options['stim']:
        stimfilepath=baphy_stim_cachefile(exptparams,options,parmfilepath)
        print("Cached stim: {0}".format(stimfilepath))
        # load stimulus spectrogram
        stim,tags,stimparam = baphy_load_specgram(stimfilepath)
    else:
        stim=np.array([])

        if options['runclass'] is None:
            stim_object='ReferenceHandle'
        elif 'runclass' in exptparams.keys():
            runclass=exptparams['runclass'].split("_")
            if len(runclass)>1 and runclass[1]==options["runclass"]:
                stim_object='TargetHandle'
            else:
                stim_object='ReferenceHandle'
        else:
            stim_object='ReferenceHandle'

        tags=exptparams['TrialObject'][1][stim_object][1]['Names']
        stimparam=[]

    # figure out spike file to load
    pp,bb=os.path.split(parmfilepath)
    spkfilepath=pp + '/' + spk_subdir + re.sub(r"\.m$",".spk.mat",bb)
    print("Spike file: {0}".format(spkfilepath))

    # load spike times
    sortinfo,spikefs=baphy_load_spike_data_raw(spkfilepath)

    # adjust spike and event times to be in seconds since experiment started
    exptevents,spiketimes,unit_names = baphy_align_time(exptevents,sortinfo,spikefs,options['rasterfs'])

    # assign cellids to each unit
    siteid=globalparams['SiteID']
    unit_names=[siteid+"-"+x for x in unit_names]
    #print(unit_names)
    # pull out a single cell if 'all' not specified
    spike_dict={}
    for i,x in enumerate(unit_names):
        if x==options['cellid'] or options['cellid']=='all':
            spike_dict[x]=spiketimes[i]

    state_dict={}
    if options['pupil']:
        try:
            pupilfilepath=re.sub(r"\.m$",".pup.mat",parmfilepath)
            pupiltrace,ptrialidx=baphy_load_pupil_trace(pupilfilepath,exptevents,options)
            state_dict['pupiltrace']=pupiltrace

        except:
            print("No pupil data")

    return exptevents, stim, spike_dict, state_dict, tags, stimparam, exptparams


def baphy_load_dataset(parmfilepath,options={}):

    """
    this can be used to generate a recording object

    input:
        parmfilepath: baphy parameter file
        options: dictionary of loading options

    current outputs:
        event_times: pandas dataframe with one row per event. times in sec
              since experiment began
        spike_dict: dictionary of lists. spike_dict[cellid] is the set of
              spike times (secs since expt started) for that unit
        stim_dict: stim_dict[name] is [channel X time] stimulus
              (spectrogram) matrix, the times that the stimuli were played
              are rows in the event_times dataframe

    TODO: support for pupil and behavior. branch out different functions for
        different batches of analysis. (see RDT special case below)
    other things that could be returned:
        globalparams, exptparams: dictionaries with expt metadata from baphy

    """
    # get the relatively un-pre-processed data

    exptevents, stim, spike_dict, state_dict, tags, stimparam, exptparams = baphy_load_data(parmfilepath,options)

    # pre-process event list (event_times) to only contain useful events

    # extract each trial
    print('Creating trial events')
    tag_mask_start="TRIALSTART"
    tag_mask_stop="TRIALSTOP"
    ffstart=(exptevents['name'] == tag_mask_start)
    ffstop=(exptevents['name'] == tag_mask_stop)
    TrialCount=np.max(exptevents.loc[ffstart,'Trial'])
    event_times=pd.concat([exptevents.loc[ffstart,['start']].reset_index(),
                          exptevents.loc[ffstop,['end']].reset_index()], axis=1)
    event_times['name']="TRIAL"
    event_times=event_times.drop(columns=['index'])

    # figure out length of entire experiment
    file_start_time=np.min(event_times['start'])
    file_stop_time=np.max(event_times['end'])

    # add event characterizing outcome of each behavioral
    # trial (if behavior)
    print('Creating trial outcome events')
    note_map={'OUTCOME,FALSEALARM': 'FA_TRIAL',
              'OUTCOME,MISS': 'MISS_TRIAL',
              'BEHAVIOR,PUMPON,Pump': 'HIT_TRIAL'}
    this_event_times=event_times.copy()
    any_behavior=False
    for trialidx in range(1,TrialCount+1):
        ff=((exptevents['name']=='OUTCOME,FALSEALARM') | \
            (exptevents['name']=='OUTCOME,MISS') | \
            (exptevents['name']=='BEHAVIOR,PUMPON,Pump')) & \
           (exptevents['Trial']==trialidx)

        for i,d in exptevents.loc[ff].iterrows():
            #print("{0}: {1} - {2} - {3}".format(i,
            #      d['Trial'],d['name'],d['end']))

            this_event_times.loc[trialidx-1,'name']= \
               note_map[d['name']]
            any_behavior=True

    te=pd.DataFrame(index=[0],columns=(event_times.columns))
    if any_behavior:
        # only concatenate newly labeled trials if events occured that reflect
        # behavior. There's probably a less kludgy way of checking for this
        # before actually running through the above loop
        event_times=pd.concat([event_times, this_event_times])
        te.loc[0]=[file_start_time,file_stop_time,'ACTIVE_EXPERIMENT']
    else:
        te.loc[0]=[file_start_time,file_stop_time,'PASSIVE_EXPERIMENT']
    event_times=event_times.append(te)

    # remove events DURING or AFTER LICK
    print('Removing post-response stimuli')
    ff=(exptevents['name']=='LICK')
    keepevents=np.ones(len(exptevents))==1
    for i,d in exptevents.loc[ff].iterrows():
        trialidx=d['Trial']
        start=d['start']
        fflate=(exptevents['end'] > start) & \
          (exptevents['Trial']==trialidx) & \
          (exptevents['name'].str.contains('Stim , '))
        for i,d in exptevents.loc[fflate].iterrows():
            #print("{0}: {1} - {2} - {3}>{4}".format(i,
            #      d['Trial'],d['name'],d['end'],start))
            # remove Pre- and PostStimSilence as well
            keepevents[(i-1):(i+2)]=False

    print("Keeping {0}/{1} events that precede responses".format(
            np.sum(keepevents),len(keepevents)))
    exptevents=exptevents[keepevents].reset_index()

    #ff=(exptevents['Trial']==3)
    #exptevents.loc[ff]

    stim_dict={}

    if 'pertrial' in options and options['pertrial']:
        # NOT COMPLETE!

        # make stimulus events unique to each trial
        this_event_times=event_times.copy()
        for eventidx in range(0,TrialCount):
            event_name="TRIAL{0}".format(eventidx)
            this_event_times.loc[eventidx,'name']=event_name
            if options['stim']:
                stim_dict[event_name]=stim[:,:,eventidx]
        event_times=pd.concat([event_times, this_event_times])

    else:
        # generate stimulus events unique to each distinct stimulus
        ff_tar_events=exptevents['name'].str.contains('Target')
        ff_pre_all=exptevents['name']==""
        ff_post_all=ff_pre_all.copy()

        for eventidx in range(0,len(tags)):

            if options['stim']:
                # save stimulus for this event as separate dictionary entry
                stim_dict["STIM_"+tags[eventidx]]=stim[:,:,eventidx]
            else:
                stim_dict["STIM_"+tags[eventidx]]=np.array([[]])
            # complicated experiment-specific part
            tag_mask_start="PreStimSilence , "+tags[eventidx]+" , Reference"
            tag_mask_stop="PostStimSilence , "+tags[eventidx]+" , Reference"

            ffstart=(exptevents['name']==tag_mask_start)
            if np.sum(ffstart)>0:
                ffstop=(exptevents['name']==tag_mask_stop)
            else:
                ffstart=(exptevents['name'].str.contains(tag_mask_start))
                ffstop=(exptevents['name'].str.contains(tag_mask_stop))

            # create intial list of stimulus events
            this_event_times=pd.concat([exptevents.loc[ffstart,['start']].reset_index(),
                                  exptevents.loc[ffstop,['end']].reset_index()], axis=1)
            this_event_times=this_event_times.drop(columns=['index'])
            this_event_times['name']="STIM_"+tags[eventidx]

            # screen for conflicts with target events
            keepevents=np.ones(len(this_event_times))==1
            for i,d in this_event_times.iterrows():
                f=ff_tar_events & (exptevents['start']<d['end']-0.001) & \
                  (exptevents['end']>d['start']+0.001)

                if np.sum(f):
                    #print("Stim (event {0}: {1:.2f}-{2:.2f} {3}".format(eventidx,d['start'],
                    #      d['end'],d['name']))
                    #print("??? But did it happen?  ? Conflicting target: {0}-{1} {2}".format(exptevents['start'][j],
                    #      exptevents['end'][j],exptevents['name'][j]))
                    keepevents[i]=False

            if np.sum(keepevents==False):
                print("Removed {0}/{1} events that overlap with target".format(
                        np.sum(keepevents==False),len(keepevents)))
            # create final list of these stimulus events
            this_event_times=this_event_times[keepevents]
            tff,=np.where(ffstart)
            ffstart[tff[keepevents==False]]=False
            tff,=np.where(ffstop)
            ffstop[tff[keepevents==False]]=False

            event_times=event_times.append(this_event_times, ignore_index=True)
            this_event_times['name']="REFERENCE"
            event_times=event_times.append(this_event_times, ignore_index=True)
            #event_times=pd.concat([event_times, this_event_times])

            ff_pre_all=ff_pre_all | ffstart
            ff_post_all=ff_post_all | ffstop

        # generate list of corresponding pre/post events
        this_event_times2=pd.concat([exptevents.loc[ff_pre_all,['start']],
                              exptevents.loc[ff_pre_all,['end']]], axis=1)
        this_event_times2['name']='PreStimSilence'

        this_event_times3=pd.concat([exptevents.loc[ff_post_all,['start']],
                              exptevents.loc[ff_post_all,['end']]], axis=1)
        this_event_times3['name']='PostStimSilence'

        event_times=event_times.append(this_event_times2, ignore_index=True)
        event_times=event_times.append(this_event_times3, ignore_index=True)
        #event_times=pd.concat([event_times, this_event_times2, this_event_times3])

    # add behavior events
    if exptparams['runclass']=='PTD' and any_behavior:
        # special events for tone in noise task
        tar_idx_freq=exptparams['TrialObject'][1]['TargetIdxFreq']
        tar_snr=exptparams['TrialObject'][1]['RelativeTarRefdB']
        common_tar_idx,=np.where(tar_idx_freq==np.max(tar_idx_freq))

        if isinstance(tar_idx_freq, (int)) or len(tar_idx_freq)==1 or np.isinf(tar_snr[0]):
            diff_event='PURETONE_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (np.max(common_tar_idx)<2):
            diff_event='EASY_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (2 in common_tar_idx):
            diff_event='MEDIUM_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (np.min(common_tar_idx)>2):
            diff_event='HARD_BEHAVIOR'
        else:
            diff_event='PURETONE_BEHAVIOR'
        te=pd.DataFrame(index=[0],columns=(event_times.columns))
        te.loc[0]=[file_start_time,file_stop_time,diff_event]
        event_times=event_times.append(te, ignore_index=True)
        #event_times=pd.concat([event_times, te])

    # sort by when the event occured in experiment time
    event_times=event_times.sort_values(by=['start','end'], ascending=[1, 0]).reset_index()
    event_times=event_times.drop(columns=['index'])

    return event_times, spike_dict, stim_dict, state_dict


def baphy_load_dataset_RDT(parmfilepath,options={}):
    """
    this can be used to generate a recording object for an RDT experiment
        based largely on baphy_load_recording but with several additional
        specialized outputs

    input:
        parmfilepath: baphy parameter file
        options: dictionary of loading options

    current outputs:
        event_times: pandas dataframe with one row per event. times in sec
              since experiment began
        spike_dict: dictionary of lists. spike_dict[cellid] is the set of
              spike times (secs since expt started) for that unit
        stim_dict: stim_dict[name] is [channel X time] stimulus
              (spectrogram) matrix, the times that the stimuli were played
              are rows in the event_times dataframe
        stim1_dict: same thing but for foreground stream only
        stim2_dict: background stream
        state_dict: dictionary of continuous Tx1 signals indicating
           state_dict['repeating_phase']=when in repeating phase
           state_dict['single_stream']=when trial is single stream
           state_dict['targetid']=target id on the current trial

    TODO : merge back into general loading function ? Or keep separate?
    """

    options['pupil'] = options.get('pupil',False)
    options['stim'] = options.get('stim',True)
    options['runclass'] = options.get('runclass',None)

    # get the relatively un-pre-processed data
    exptevents, stim, spike_dict, state_dict, tags, stimparam, exptparams = baphy_load_data(parmfilepath,options)

    # pre-process event list (event_times) to only contain useful events

    # extract each trial
    tag_mask_start="TRIALSTART"
    tag_mask_stop="TRIALSTOP"
    ffstart=(exptevents['name'] == tag_mask_start)
    ffstop=(exptevents['name'] == tag_mask_stop)
    TrialCount=np.max(exptevents.loc[ffstart,'Trial'])
    event_times=pd.concat([exptevents.loc[ffstart,['start']].reset_index(),
                          exptevents.loc[ffstop,['end']].reset_index()], axis=1)
    event_times['name']="TRIAL"
    event_times=event_times.drop(columns=['index'])

    stim_dict={}
    stim1_dict={}
    stim2_dict={}
    state_dict={}

    # make stimulus events unique to each trial
    this_event_times=event_times.copy()
    rasterfs=options['rasterfs']
    BigStimMatrix=stimparam[-1]
    state=np.zeros([3,stim.shape[1],stim.shape[2]])
    single_stream_trials = (BigStimMatrix[0,1,:]==-1)
    state[1,:,single_stream_trials]=1
    prebins=int(exptparams['TrialObject'][1]['PreTrialSilence']*rasterfs)
    samplebins=int(exptparams['TrialObject'][1]['ReferenceHandle'][1]['Duration']*rasterfs)

    for trialidx in range(0,TrialCount):
        event_name="TRIAL{0}".format(trialidx)
        this_event_times.loc[trialidx,'name']=event_name
        stim1_dict[event_name]=stim[:,:,trialidx,0]
        stim2_dict[event_name]=stim[:,:,trialidx,1]
        stim_dict[event_name]=stim[:,:,trialidx,2]

        s=np.zeros([3,stim_dict[event_name].shape[1]])
        rslot=np.argmax(np.diff(BigStimMatrix[:,0,trialidx])==0)+1
        rbin=prebins+rslot*samplebins
        s[0,rbin:]=1
        single_stream_trial = int(BigStimMatrix[0,1,trialidx]==-1)
        s[1,:]=single_stream_trial
        tarslot=np.argmin(BigStimMatrix[:,0,trialidx]>0)-1
        s[2,:]=BigStimMatrix[tarslot,0,trialidx]
        state_dict[event_name]=s

    state_dict['BigStimMatrix']=BigStimMatrix

    event_times=pd.concat([event_times, this_event_times])

    # add stim events
    stim_mask="Stim ,"
    ffstim=(exptevents['name'].str.contains(stim_mask))
    stim_event_times=exptevents.loc[ffstim,['name','start','end']]
    event_times=pd.concat([event_times, stim_event_times])

    # sort by when the event occured in experiment time
    event_times=event_times.sort_values(by=['start','end'])
    cols=['name','start','end']
    event_times=event_times[cols]
#    for trialidx in range(0,TrialCount):
#       rslot=np.argmax(np.diff(BigStimMatrix[:,0,trialidx])==0)+1
#       rbin=prebins+rslot*samplebins
#       state[0,rbin:,trialidx]=1
#
#       tarslot=np.argmin(BigStimMatrix[:,0,trialidx]>0)-1
#       state[2,:,trialidx]=BigStimMatrix[tarslot,0,trialidx]
#
#    state_dict['repeating_phase']=np.reshape(state[0,:,:].T,[-1,1])
#    state_dict['single_stream']=np.reshape(state[0,:,:].T,[-1,1])
#    state_dict['targetid']=np.reshape(state[0,:,:].T,[-1,1])

    return event_times, spike_dict, stim_dict, state_dict, stim1_dict, stim2_dict


def spike_time_to_raster(spike_dict,fs=100,event_times=None):
    """
    convert list of spike times to a raster of spike rate, with duration
    matching max end time in the event_times list
    """

    if event_times is not None:
        maxtime=np.max(event_times["end"])

    maxbin=int(fs*maxtime)
    unitcount=len(spike_dict.keys())
    raster=np.zeros([unitcount,maxbin])

    cellids=sorted(spike_dict)
    for i,key in enumerate(cellids):
        for t in spike_dict[key]:
            b=int(np.floor(t*fs))
            if b<maxbin:
                raster[i,b]+=1

    return raster,cellids


def dict_to_signal(stim_dict,fs=100,event_times=None,signal_name='stim',recording_name='rec'):

    maxtime=np.max(event_times["end"])
    maxbin=int(fs*maxtime)

    tags=list(stim_dict.keys())
    chancount=stim_dict[tags[0]].shape[0]

    z=np.zeros([chancount,maxbin])

    empty_stim=nems.signal.Signal(matrix=z,fs=fs,name=signal_name,epochs=event_times,recording=recording_name)
    stim=empty_stim.replace_epochs(stim_dict)

    return stim

def baphy_load_recording(cellid,batch,options):

    #print(options)
    options['rasterfs'] = int(options.get('rasterfs',100))
    options['stimfmt'] = options.get('stimfmt','ozgf')
    options['chancount'] =  int(options.get('chancount',18))
    options['pertrial'] = int(options.get('pertrial',False))
    options['includeprestim'] = options.get('includeprestim',1)

    options['pupil'] = int(options.get('pupil',False))
    options['pupil_deblink'] = int(options.get('pupil_deblink',1))
    options['pupil_median'] = int(options.get('pupil_deblink',1))
    options['stim'] = int(options.get('stim',True))
    options['runclass'] = options.get('runclass',None)
    options['cellid'] = options.get('cellid',cellid)
    options['batch']=int(batch)

    d=db.get_batch_cell_data(batch=batch, cellid=cellid, label='parm')
    files=list(d['parm'])

    for i,parmfilepath in enumerate(files):

        if options["runclass"]=="RDT":
            event_times, spike_dict, stim_dict, state_dict, stim1_dict, stim2_dict = baphy_load_dataset_RDT(parmfilepath,options)
        else:
            event_times, spike_dict, stim_dict, state_dict = baphy_load_dataset(parmfilepath,options)

            d2=event_times.loc[0].copy()
            if (i==0) and (d2['name']=='PASSIVE_EXPERIMENT'):
                d2['name']='PRE_PASSIVE'
                event_times=event_times.append(d2)
            elif d2['name']=='PASSIVE_EXPERIMENT':
                d2['name']='POST_PASSIVE'
                event_times=event_times.append(d2)

        # generate spike raster
        raster_all,cellids=spike_time_to_raster(spike_dict,fs=options['rasterfs'],event_times=event_times)

        # generate response signal
        t_resp=nems.signal.Signal(fs=options['rasterfs'],matrix=raster_all,name='resp',recording=cellid,chans=cellids,epochs=event_times)
        if i==0:
            resp=t_resp
        else:
            resp=resp.concatenate_time([resp,t_resp])

        try:
            options['pupil']=True
            # create pupil signal if it exists
            rlen=raster_all.shape[1]
            plen=state_dict['pupiltrace'].shape[1]
            if plen>rlen:
                state_dict['pupiltrace']=state_dict['pupiltrace'][:,0:-(plen-rlen)]
            elif rlen>plen:
                state_dict['pupiltrace']=state_dict['pupiltrace'][:,0:-(rlen-plen)]

            # generate pupil signals
            t_pupil=nems.signal.Signal(fs=options['rasterfs'],matrix=state_dict['pupiltrace'],name='pupil',recording=cellid,chans=['pupil'],epochs=event_times)

            if i==0:
                pupil=t_pupil
            else:
                pupil=pupil.concatenate_time([pupil,t_pupil])
        except:
            options['pupil']=False

        if options['stim']:
            t_stim=dict_to_signal(stim_dict,fs=options['rasterfs'],event_times=event_times)
            t_stim.recording=cellid

            if i==0:
                print("i={0} starting".format(i))
                stim=t_stim
            else:
                print("i={0} concatenating".format(i))
                stim=stim.concatenate_time([stim,t_stim])

        if options['stim'] and options["runclass"]=="RDT":
            t_BigStimMatrix=state_dict['BigStimMatrix']
            del state_dict['BigStimMatrix']

            t_stim1=dict_to_signal(stim1_dict,fs=options['rasterfs'],event_times=event_times,
                                   signal_name='stim1',recording_name=cellid)
            t_stim2=dict_to_signal(stim2_dict,fs=options['rasterfs'],event_times=event_times,
                                   signal_name='stim2',recording_name=cellid)
            t_state=dict_to_signal(state_dict,fs=options['rasterfs'],event_times=event_times,
                                   signal_name='state',recording_name=cellid)
            t_state.chans=['repeating_phase', 'single_stream', 'targetid']

            if i==0:
                print("i={0} starting".format(i))
                stim1=t_stim1
                stim2=t_stim2
                state=t_state
                BigStimMatrix=t_BigStimMatrix
            else:
                print("i={0} concatenating".format(i))
                stim1=stim1.concatenate_time([stim1,t_stim1])
                stim2=stim2.concatenate_time([stim2,t_stim2])
                state=state.concatenate_time([state,t_state])
                BigStimMatrix=np.concatenate((BigStimMatrix,t_BigStimMatrix),axis=2)

    resp.meta=options

    if options['pupil']:
        signals={'resp': resp, 'pupil': pupil}
    else:
        signals={'resp': resp}

    if options['stim']:
        signals['stim']=stim

    if options['stim'] and options["runclass"]=="RDT":
        signals['stim1']=stim1
        signals['stim2']=stim2
    if options["runclass"]=="RDT":
        signals['state']=state
        #signals['stim'].meta={'BigStimMatrix': BigStimMatrix}

    rec=nems.recording.Recording(signals=signals)

#    if options["runclass"]=="RDT":
#        rec['meta']={'BigStimMatrix': BigStimMatrix}

    return rec

def baphy_data_path(options):

    data_path="/auto/data/tmp/batch{0}_fs{1}_{2}{3}/".format(options["batch"],options["rasterfs"],options['stimfmt'],options["chancount"])

    return data_path