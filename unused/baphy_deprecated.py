#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:10:22 2018

@author: svd
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
import nems0.signal
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






