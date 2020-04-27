#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017
@author: svd, changes added by njs
"""

import logging
import re
import os
import os.path
import scipy
import scipy.io
import scipy.ndimage.filters
import scipy.signal
import numpy as np
import json
import sys
import io
import datetime
import glob
from math import isclose
import copy

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from nems import get_setting
import nems.signal
import nems.recording
import nems.db as db
import nems_lbhb.behavior as beh
from nems.recording import Recording
from nems.recording import load_recording
from nems.utils import recording_filename_hash
import nems_lbhb.io as io
import nems.epoch as ep

# TODO: Replace catch-all `except:` statements with except SpecificError,
#       or add some other way to help with debugging them.

# paths to baphy data -- standard locations on elephant
stim_cache_dir = '/auto/data/tmp/tstim/'  # location of cached stimuli
spk_subdir = 'sorted/'   # location of spk.mat files relative to parmfiles

log = logging.getLogger(__name__)

# ===================== Loading a baphy experiment into a recording object =======================
# standard pipeline is:
#   baphy_load_recording_file  - load the cached .tgz file based on options dictionary
#   baphy_load_recording_uri   - caches a .tgz recording file *entry point for nems
#   baphy_load_recording       - packages baphy data into experiment
#       baphy_load_dataset     - does most of the epoch processing / naming based on exptevents
#       baphy_load_data        - basically just loads data from mfiles
# TODO - consider how / where behavior.py code and BAPHYExperiment can be used to streamline
# this pipeline and improve labeling of behavior trials etc.
# ===============================================================================================

def baphy_load_recording_file(**options):
    """
    This is the main entry point to load a baphy recording.

    "simple" wrapper to load recording, calls get_recording_uri to figure
    out cache file and create if necessary. then load

    :param options:
    specify files with list of raw ids or mfile list or cellid/siteid/cellids+batch
        or specify a cellid/batch and it will find the relevant mfiles/rawids
    specify signals with resp=True/False, stim=True/False, pupil=True/False, etc
    other options as in other places (rasterfs, stimfmt, etc...)
    model fit use case options includes
    :return:
    """
    uri, cells_to_extract = baphy_load_recording_uri(**options)

    rec = load_recording(uri)
    rec.meta['cells_to_extract'] = cells_to_extract

    return rec

def baphy_load_recording_uri(recache=False, **options):
    """
    Meant to be a "universal loader" for baphy recordings.
    First figure out hash for options dictionary
    If cached file exists and return its name
    If it doesn't exist, call baphy_load_recording, save the cache file and
    return the filename.

    input:
        options: dictionary

        required fields:
            batch - (int) batch number
            cellid - single cellid string, list of cellids, or siteid
                If siteid is passed, return superset of cells. i.e. if some
                cells aren't present in one of the files that is found for this batch,
                don't load that file. To override this behavior, pass rawid list.

    return:
        data_file : string
        URI for recording file specified in options

    TODO: web-support for URI, currently just local filenames

    (CRH - 9/21/2018)
    """

    mfilename = options.get('mfilename', None)
    batch = options.get('batch', None)
    cellid = options.get('cellid', None)

    if options.get('siteid') is not None:
        raise DeprecationWarning("Use cellid to specify recording site. e.g. options['cellid']='DRX005c'")

    if ((cellid is None) | (batch is None)) & (mfilename is None):
        raise ValueError("options dict must include (siteid, batch) or mfilename")

    # parse cellid. Update cellid, siteid, rawid in options dictionary
    # if cellid/batch not specified, find them based on mfile.
    cells_to_extract, options = parse_cellid(options)
    siteid = options['siteid']

    # fill in remaining default options
    options = fill_default_options(options)

    use_API = get_setting('USE_NEMS_BAPHY_API')

    data_file = recording_filename_hash(
        siteid, options, uri_path=get_setting('NEMS_RECORDINGS_DIR'))
    if use_API:
        _, f = os.path.split(data_file)
        host = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT'))
        data_uri = host + '/recordings/' + str(batch) + '/' + f
    else:
        data_uri = data_file

    if not use_API and (not os.path.exists(data_file) or recache == True):
        log.info("Generating recording")
        # rec.meta is set = options in the following function
        rec = baphy_load_recording(**options)
        log.info('Caching recording: %s', data_file)
        rec.save(data_file)

    else:
        log.info('Cached recording found: %s', data_uri)

    return data_uri, cells_to_extract


# ============================ baphy loading "utils" ==================================
def parse_cellid(options):
    """
    figure out if cellid is
        1) single cellid
        2) list of cellids
        3) a siteid

    using this, add the field 'siteid' to the options dictionary. If siteid was passed,
    define cellid as a list of all cellids recorded at this site, for this batch.

    options: dictionary
            batch - (int) batch number
            cellid - single cellid string, list of cellids, or siteid
                If siteid is passed, return superset of cells. i.e. if some
                cells aren't present in one of the files that is found for this batch,
                don't load that file. To override this behavior, pass rawid list.

    returns updated options dictionary and the cellid to extract from the recording
        NOTE: The reason we keep "cellid to extract" distinct from the options dictionary
        is so that it doesn't muck with the cached recording hash. e.g. if you want to analyze
        cell1 from a site where you recorded cells1-4, you don't want a different recording
        cached for each cell.
    """

    options = options.copy()

    mfilename = options.get('mfilename', None)
    cellid = options.get('cellid', None)
    batch = options.get('batch', None)
    rawid = options.get('rawid', None)
    cells_to_extract = None

    if ((cellid is None) | (batch is None)) & (mfilename is None):
        raise ValueError("must provide cellid and batch or mfilename")

    siteid = None
    cell_list = None
    if type(cellid) is list:
        cell_list = cellid
    elif (type(cellid) is str) & ('-' not in cellid):
        siteid = cellid

    if mfilename is not None:
        # simple, db-free case. Just a pass through.
        pass
        if batch is None:
            options['batch'] = 0

    elif cell_list is not None:
        # list of cells was passed
        siteid = cellid.split('-')[0]
        cell_list_all, rawid = db.get_stable_batch_cells(batch=batch, cellid=cell_list,
                                             rawid=rawid)
        options['cellid'] = cell_list_all
        options['rawid'] = rawid
        options['siteid'] = cell_list[0].split('-')[0]
        cells_to_extract = cell_list

    elif siteid is not None:
        # siteid was passed, figure out if electrode numbers were specified.
        chan_nums = None
        if '.e' in siteid:
            args = siteid.split('.')
            siteid = args[0]
            chan_lims = args[1].replace('e', '').split(':')
            chan_nums = np.arange(int(chan_lims[0]), int(chan_lims[1])+1)

        cell_list, rawid = db.get_stable_batch_cells(batch=batch, cellid=siteid,
                                             rawid=rawid)

        if chan_nums is not None:
            cells_to_extract = [c for c in cell_list if int(c.split('-')[1]) in chan_nums]
        else:
            cells_to_extract = cell_list

        options['cellid'] = cell_list
        if len(rawid) != 0:
            options['rawid'] = rawid
        options['siteid'] = siteid

    elif cellid is not None:
        # single cellid was passed, want list of all cellids. First, get rawids
        cell_list, rawid = db.get_stable_batch_cells(batch=batch, cellid=cellid,
                                                     rawid=rawid)
        # now, use rawid to get all stable cellids across these files
        siteid = cell_list[0].split('-')[0]
        cell_list, rawid = db.get_stable_batch_cells(batch=batch, cellid=siteid,
                                                     rawid=rawid)

        options['cellid'] = cell_list
        options['rawid'] = rawid
        options['siteid'] = siteid
        cells_to_extract = [cellid]

    if (len(cells_to_extract) == 0) & (mfilename is None):
        raise ValueError("No cellids found! Make sure cellid/batch is specified correctly, "
                            "or that you've specified an mfile.")

    return list(cells_to_extract), options


def fill_default_options(options):
    """
    fill in default options. use options after adding defaults to specify
    metadata hash
    """

    options = options.copy()

    # set default options if missing
    options['rasterfs'] = int(options.get('rasterfs', 100))
    options['stimfmt'] = options.get('stimfmt', 'ozgf')
    options['chancount'] = int(options.get('chancount', 18))
    options['pertrial'] = int(options.get('pertrial', False))
    options['includeprestim'] = options.get('includeprestim', 1)
    options['pupil'] = int(options.get('pupil', False))
    options['rem'] = int(options.get('rem', False))
    options['pupil_eyespeed'] = int(options.get('pupil_eyespeed', False))
    if options['pupil'] or options['rem']:
        options = io.set_default_pupil_options(options)

    #options['pupil_deblink'] = int(options.get('pupil_deblink', 1))
    #options['pupil_deblink_dur'] = options.get('pupil_deblink_dur', 1)
    #options['pupil_median'] = options.get('pupil_median', 0)
    #options["pupil_offset"] = options.get('pupil_offset', 0.75)
    options['resp'] = int(options.get('resp', True))
    options['stim'] = int(options.get('stim', True))
    options['runclass'] = options.get('runclass', None)
    options['rawid'] = options.get('rawid', None)

    if options['stimfmt'] in ['envelope', 'parm']:
        log.info("Setting chancount=0 for stimfmt=%s", options['stimfmt'])
        options['chancount'] = 0

    return options


def baphy_load_data(parmfilepath, **options):
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
    # default_options={'rasterfs':100, 'includeprestim':True,
    #                  'stimfmt':'ozgf', 'chancount':18,
    #                  'cellid': 'all'}
    # options = options.update(default_options)

    # add .m extension if missing
    if parmfilepath[-2:] != ".m":
        parmfilepath += ".m"
    # load parameter file
    log.info('Loading parameters: %s', parmfilepath)
    globalparams, exptparams, exptevents = io.baphy_parm_read(parmfilepath)
    # TODO: use paths that match LBHB filesystem? new s3 filesystem?
    #       or make s3 match LBHB?

    # figure out stimulus cachefile to load
    if options['stim']:
        if (options['stimfmt']=='parm') & exptparams['TrialObject'][1]['ReferenceClass'].startswith('Torc'):
            import nems_lbhb.strf.torc_subfunctions as tsf
            TorcObject = exptparams['TrialObject'][1]['ReferenceHandle'][1]
            stim, tags, stimparam = tsf.generate_torc_spectrograms(
                      TorcObject, rasterfs=options['rasterfs'], single_cycle=False)
            # NB stim is a dict rather than a 3-d array

        elif exptparams['runclass']=='VOC_VOC':
            stimfilepath1 = io.baphy_stim_cachefile(exptparams, parmfilepath, use_target=False, **options)
            stimfilepath2 = io.baphy_stim_cachefile(exptparams, parmfilepath, use_target=True, **options)
            print("Cached stim: {0}, {1}".format(stimfilepath1, stimfilepath2))
            # load stimulus spectrogram
            stim1, tags1, stimparam1 = baphy_load_specgram(stimfilepath1)
            stim2, tags2, stimparam2 = baphy_load_specgram(stimfilepath2)
            stim = np.concatenate((stim1,stim2), axis=2)
            if exptparams['TrialObject'][1]['ReferenceHandle'][1]['SNR'] >= 100:
                t2 = [t+'_0dB' for t in tags2]
                tags = np.concatenate((tags1,t2))
                eventmatch='Reference1'
            else:
                t1 = [t+'_0dB' for t in tags1]
                tags = np.concatenate((t1,tags2))
                eventmatch = 'Reference2'
            #import pdb
            #pdb.set_trace()
            for i in range(len(exptevents)):
                if eventmatch in exptevents.loc[i,'name']:
                    exptevents.loc[i,'name'] = exptevents.loc[i,'name'].replace('.wav','.wav_0dB')
                    exptevents.loc[i,'name'] = exptevents.loc[i,'name'].replace('Reference1','Reference')
                    exptevents.loc[i,'name'] = exptevents.loc[i,'name'].replace('Reference2','Reference')

            stimparam = stimparam1
        else:
            stimfilepath = io.baphy_stim_cachefile(exptparams, parmfilepath, **options)
            print("Cached stim: {0}".format(stimfilepath))
            # load stimulus spectrogram
            stim, tags, stimparam = baphy_load_specgram(stimfilepath)

        if options["stimfmt"]=='envelope' and \
            exptparams['TrialObject'][1]['ReferenceClass']=='SSA':
            # SSA special case
            stimo=stim.copy()
            maxval=np.max(np.reshape(stimo,[2,-1]),axis=1)
            print('special case for SSA stim!')
            ref=exptparams['TrialObject'][1]['ReferenceHandle'][1]
            stimlen=ref['PipDuration']+ref['PipInterval']
            stimbins=int(stimlen*options['rasterfs'])

            stim=np.zeros([2,stimbins,6])
            prebins=int(ref['PipInterval']/2*options['rasterfs'])
            durbins=int(ref['PipDuration']*options['rasterfs'])
            stim[0,prebins:(prebins+durbins),0:3]=maxval[0]
            stim[1,prebins:(prebins+durbins),3:]=maxval[1]
            tags=["{}+ONSET".format(ref['Frequencies'][0]),
                  "{}+{:.2f}".format(ref['Frequencies'][0],ref['F1Rates'][0]),
                  "{}+{:.2f}".format(ref['Frequencies'][0],ref['F1Rates'][1]),
                  "{}+ONSET".format(ref['Frequencies'][1]),
                  "{}+{:.2f}".format(ref['Frequencies'][1],ref['F1Rates'][0]),
                  "{}+{:.2f}".format(ref['Frequencies'][1],ref['F1Rates'][1])]

    else:
        stim = np.array([])

        if options['runclass'] is None:
            stim_object = 'ReferenceHandle'
        elif 'runclass' in exptparams.keys():
            runclass = exptparams['runclass'].split("_")
            if (len(runclass) > 1) and (runclass[1] == options["runclass"]):
                stim_object = 'TargetHandle'
            else:
                stim_object = 'ReferenceHandle'
        else:
            stim_object = 'ReferenceHandle'

        if (options['runclass']=='VOC') & (exptparams['runclass']=='VOC_VOC'):
            # special kludge for clean + noisy VOC expts
            raise Warning("VOC_VOC files not supported")
        else:
            tags = exptparams['TrialObject'][1][stim_object][1]['Names']
        tags, tagids = np.unique(tags, return_index=True)
        stimparam = []

    # figure out spike file to load
    pp, bb = os.path.split(parmfilepath)
    spkfilepath = pp + '/' + spk_subdir + re.sub(r"\.m$", ".spk.mat", bb)
    print("Spike file: {0}".format(spkfilepath))

    # load spike times
    if options['resp']:
        sortinfo, spikefs = baphy_load_spike_data_raw(spkfilepath)

        # adjust spike and event times to be in seconds since experiment started
        exptevents, spiketimes, unit_names = io.baphy_align_time(
                exptevents, sortinfo, spikefs, options['rasterfs']
                )

        # assign cellids to each unit
        siteid = globalparams['SiteID']
        unit_names = [(siteid + "-" + x) for x in unit_names]
        # print(unit_names)

        # test for special case where psuedo cellid suffix has been added to
        # cellid by stripping anything after a "_" underscore in the cellid (list)
        # provided
        pcellids = options['cellid'] if (type(options['cellid']) is list) \
           else [options['cellid']]
        cellids = []
        pcellidmap = {}
        for pcellid in pcellids:
            t = pcellid.split("_")
            t[0] = t[0].lower()
            cellids.append(t[0])
            pcellidmap[t[0]] = pcellid
        print(pcellidmap)
        # pull out a single cell if 'all' not specified
        spike_dict = {}
        for i, x in enumerate(unit_names):
            if (cellids[0] == 'all'):
                spike_dict[x] = spiketimes[i]
            elif (x.lower() in cellids):
                spike_dict[pcellidmap[x.lower()]] = spiketimes[i]
        #import pdb
        #pdb.set_trace()
        if not spike_dict:
            raise ValueError('No matching cellid in baphy spike file')
    else:
        # no spike data, use baphy-recorded timestamps.
        # TODO: get this working with old baphy files that don't record explicit timestamps
        # in that case, just assume real time is the sum of trial durations.
        spike_dict = {}
        #import pdb; pdb.set_trace()
        exptevents = io.baphy_align_time_baphyparm(exptevents, finalfs=options['rasterfs'])

    state_dict = {}
    if options['pupil']:
        try:
            pupilfilepath = re.sub(r"\.m$", ".pup.mat", parmfilepath)
            options['verbose'] = False
            if options['pupil_eyespeed']:
                pupildata, ptrialidx = imo.load_pupil_trace(
                        pupilfilepath, exptevents, **options
                        )
                try:
                    state_dict['pupil_eyespeed'] = pupildata['pupil_eyespeed']
                    state_dict['pupiltrace'] = pupildata['pupil']
                except:
                    log.info('No eyespeed data exists for this recording')
                    state_dict['pupiltrace'] = pupildata

            else:
                pupiltrace, ptrialidx = io.load_pupil_trace(
                        pupilfilepath, exptevents, **options
                        )
                state_dict['pupiltrace'] = pupiltrace

        except ValueError:
            raise ValueError("Error loading pupil data: " + pupilfilepath)

    if options['rem']:
        try:
            rem_options = io.load_rem_options(pupilfilepath)
            rem_options['verbose'] = False
            #rem_options['rasterfs'] = options['rasterfs']
            is_rem, rem_options = io.get_rem(pupilfilepath=pupilfilepath,
                              exptevents=exptevents, **rem_options)
            is_rem = is_rem.astype(float)
            new_len = int(len(is_rem) * options['rasterfs'] / rem_options['rasterfs'])
            is_rem = resample(is_rem, new_len)
            is_rem[is_rem>0.01] = 1
            is_rem[is_rem<=0.01] = 0
            state_dict['rem'] = is_rem
        except:
            log.info("REM load failed. Skipping.")

    return (exptevents, stim, spike_dict, state_dict,
            tags, stimparam, exptparams)


def baphy_load_dataset(parmfilepath, **options):
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
    exptevents, stim, spike_dict, state_dict, tags, stimparam, exptparams = \
        baphy_load_data(parmfilepath, **options)

    # if runclass is BVT, add behavior outcome column (to be used later)
    # very kludgy
    # TODO - Figure out nice way to interface BAPHYExperiment with nems_lbhb.behavior
    # with this loading procedure.
    # CRH 12/10/2019
    if (exptparams['runclass'] == 'BVT'):
        BVT = True
    else:
        BVT = False
    if (exptparams['runclass'] == 'BVT') & (exptparams.get('BehaveObjectClass','DMS') != 'Passive'):
        exptevents = beh.create_trial_labels(exptparams, exptevents)
        active_BVT = True
    else:
        active_BVT = False

    # Figure out how to handle "false alarm" trials. Truncate after first lick
    # if not passive or classical conditioning
    if exptparams.get('BehaveObjectClass','DMS') in ['Passive', 'ClassicalConditioning']:
        remove_post_lick = False
    else:
        remove_post_lick = True
    if exptparams.get('BehaveObjectClass','DMS') in ['ClassicalConditioning']:
        active_CC=True
    else:
        active_CC=False

    # pre-process event list (event_times) to only contain useful events
    # extract each trial
    log.info('Creating trial events')
    tag_mask_start = "TRIALSTART"
    tag_mask_stop = "TRIALSTOP"
    ffstart = exptevents['name'].str.startswith(tag_mask_start)
    # Set trial stops to the beginning of the next trial for continuous data
    # loading
    ffstop = exptevents['name'].str.startswith(tag_mask_start)

    # end at the end of last trial
    final_trial = np.argwhere(((exptevents['name'] == tag_mask_stop)==True).values)[-1][0]
    ffstop.iloc[final_trial] = True
    # "start" of last TRIALSTOP event
    final_trial_end0 = exptevents["end"].max()
    #final_trial_end0 = exptevents.iloc[final_trial]['end']
    final_trial_end = np.floor(final_trial_end0*options['rasterfs']) / options['rasterfs']
    end_events = (exptevents['end'] >= final_trial_end)
    exptevents.loc[end_events, 'end'] = final_trial_end

    log.info('Setting end for {} events from {} to {}'.format(
        np.sum(end_events), final_trial_end0, final_trial_end))
    # set first True to False (the start of the first trial)
    first_true = np.argwhere((ffstop == True).values)[0][0]
    ffstop.iloc[first_true] = False

    TrialCount = np.max(exptevents.loc[ffstart, 'Trial'])

    event_times = pd.concat([exptevents.loc[ffstart, ['start']].reset_index(),
                             exptevents.loc[ffstop, ['end']].reset_index()],
                            axis=1)
    event_times['name'] = "TRIAL"
    event_times = event_times.drop(columns=['index'])

    if remove_post_lick:
        # during most baphy behaviors, the trial terminates following an early lick
        # (false alarm), but events may be listed after that time.
        # this code removes post-FA events and truncates any epochs interupted by
        # by the lick. Exception is classical conditioning, where licks are recorded
        # but don't affect stimulus presentation
        log.info('Removing post-response stimuli')
        keepevents = np.full(len(exptevents), True, dtype=bool)

        for trialidx in range(1, TrialCount+1):
            # remove stimulus events after TRIALSTOP or STIM,OFF event
            fftrial_stop = (exptevents['Trial'] == trialidx) & \
                ((exptevents['name'] == "STIM,OFF") |
                 (exptevents['name'] == "OUTCOME,VEARLY") |
                 (exptevents['name'] == "OUTCOME,EARLY") |
                 (exptevents['name'] == "TRIALSTOP"))
            if np.sum(fftrial_stop):
                trialstoptime = np.min(exptevents[fftrial_stop]['start'])

                fflate = (exptevents['Trial'] == trialidx) & \
                    exptevents['name'].str.startswith('Stim , ') & \
                    (exptevents['start'] > trialstoptime)
                fftrunc = (exptevents['Trial'] == trialidx) & \
                    exptevents['name'].str.startswith('Stim , ') & \
                    (exptevents['start'] <= trialstoptime) & \
                    (exptevents['end'] > trialstoptime)

            for i, d in exptevents.loc[fflate].iterrows():
                # print("{0}: {1} - {2} - {3}>{4}"
                #       .format(i, d['Trial'], d['name'], d['end'], start))
                # remove Pre- and PostStimSilence as well
                keepevents[(i-1):(i+2)] = False

            for i, d in exptevents.loc[fftrunc].iterrows():
                log.debug("Truncating event %d early at %.3f", i, trialstoptime)
                exptevents.loc[i, 'end'] = trialstoptime
                # also trim post stim silence
                exptevents.loc[i + 1, 'start'] = trialstoptime
                exptevents.loc[i + 1, 'end'] = trialstoptime

        print("Keeping {0}/{1} events that precede responses"
              .format(np.sum(keepevents), len(keepevents)))
        exptevents = exptevents[keepevents].reset_index()

    # add event characterizing outcome of each behavioral
    # trial (if behavior)
    log.info('Creating trial outcome events')
    ff_lick_dur = (exptevents['name'] == 'LICK')
    note_map = {'OUTCOME,FALSEALARM': 'FA_TRIAL',
                'OUTCOME,EARLY': 'FA_TRIAL',
                'OUTCOME,VEARLY': 'FA_TRIAL',
                'OUTCOME,MISS': 'MISS_TRIAL',
                'OUTCOME,MATCH': 'HIT_TRIAL',
                'BEHAVIOR,PUMPON,Pump': 'HIT_TRIAL'}
    this_event_times = event_times.copy()
    any_behavior = False

    if active_BVT:
        # CRH - using the labeled soundTrial events in exptevents
        # this labels all sounds. i.e. a REF can be a correct_reject
        # at this point, event_times is just labeling baphy trials though
        # so just take last "soundTrial" and label the trial that
        for trialidx in range(1, TrialCount+1):
            ff = exptevents[exptevents.Trial==trialidx]
            ff = ff[ff.soundTrial!='NULL']
            try:
                label = ff['soundTrial'].iloc[-1]
                this_event_times.loc[trialidx-1, 'name'] = label
            except:
                # was labeled as NULL, since sound never played
                this_event_times.loc[trialidx-1, 'name'] = 'EARLY_TRIAL'

        any_behavior = True
    else:
        for trialidx in range(1, TrialCount+1):
            # determine behavioral outcome, log event time to add epochs
            # spanning each trial
            ff = (((exptevents['name'] == 'OUTCOME,FALSEALARM')
                | (exptevents['name'] == 'OUTCOME,EARLY')
                | (exptevents['name'] == 'OUTCOME,VEARLY')
                | (exptevents['name'] == 'OUTCOME,MISS')
                | (exptevents['name'] == 'OUTCOME,MATCH')
                | (exptevents['name'] == 'BEHAVIOR,PUMPON,Pump'))
                & (exptevents['Trial'] == trialidx))

            for i, d in exptevents.loc[ff].iterrows():
                # print("{0}: {1} - {2} - {3}"
                #       .format(i, d['Trial'], d['name'], d['end']))
                dtrial = this_event_times.loc[trialidx-1]
                this_event_times.loc[trialidx-1, 'name'] = note_map[d['name']]
                any_behavior = True
                fdur = (ff_lick_dur
                        & (exptevents['start'] < dtrial['start'] + 0.5)
                        & (exptevents['end'] > dtrial['start'] + 0.001))
                if np.sum(fdur) & (note_map[d['name']]=='HIT_TRIAL') & \
                        (exptevents.loc[fdur,'start'].min() < dtrial['start'] + 0.5):
                    log.info(f'Erroneous early lick in HIT trial {trialidx}, deleting')
                    exptevents.loc[fdur,'name']='MISSEDLICK'

    # CRH add check, just in case user messed up when doing experiment
    # and selected: physiology yes, passive, but set behavior control to active
    # in this case, behavior didn't run, file got created with _p_, but baphy
    # still tried to label trials.
    any_behavior = any_behavior & (exptparams.get('BehaveObjectClass','DMS') != 'Passive')

    # figure out length of entire experiment
    file_start_time = np.min(event_times['start'])
    file_stop_time = np.max(event_times['end'])
    te = pd.DataFrame(index=[0], columns=(event_times.columns))

    if any_behavior:
        # only concatenate newly labeled trials if events occured that reflect
        # behavior. There's probably a less kludgy way of checking for this
        # before actually running through the above loop
        event_times = pd.concat([event_times, this_event_times])
        te.loc[0] = [file_start_time, file_stop_time, 'ACTIVE_EXPERIMENT']
    else:
        te.loc[0] = [file_start_time, file_stop_time, 'PASSIVE_EXPERIMENT']
    event_times = event_times.append(te)

    # ADD epoch for FILENAME
    b = os.path.splitext(os.path.basename(parmfilepath))[0]
    te.loc[0] = [file_start_time, file_stop_time, 'FILE_'+b]
    event_times = event_times.append(te)

    # ff = (exptevents['Trial'] == 3){}
    # exptevents.loc[ff]

    stim_dict = {}

    if options.get('pertrial', 0):
        # NOT COMPLETE!
        raise ValueError('pertrial not supported')
        # make stimulus events unique to each trial
        this_event_times = event_times.copy()
        for eventidx in range(0, TrialCount):
            event_name = "TRIAL{0}".format(eventidx)
            this_event_times.loc[eventidx, 'name'] = event_name
            if options['stim']:
                stim_dict[event_name] = stim[:, :, eventidx]
        event_times = pd.concat([event_times, this_event_times])

    else:
        # generate stimulus events unique to each distinct stimulus
        log.info('Aligning events between stim and response')
        ff_tar_events = exptevents['name'].str.endswith('Target') | \
                        exptevents['name'].str.endswith('Target+NoLight') | \
                        exptevents['name'].str.endswith('Target+Light')

        ff_tar_pre = exptevents['name'].str.startswith('Pre') & ff_tar_events
        ff_tar_dur = exptevents['name'].str.startswith('Stim') & ff_tar_events
        ff_tar_post = exptevents['name'].str.startswith('Post') & ff_tar_events

        ff_lick_dur = (exptevents['name'] == 'LICK')
        ff_pre_all = exptevents['name'] == ""
        ff_post_all = ff_pre_all.copy()

        snr_suff=""
        if 'SNR' in exptparams['TrialObject'][1]['ReferenceHandle'][1].keys():
            SNR = exptparams['TrialObject'][1]['ReferenceHandle'][1]['SNR']
            if SNR<100:
                log.info('Noisy stimulus (SNR<100), appending tag to epoch names')
                snr_suff="_{}dB".format(SNR)

        for eventidx in range(0, len(tags)):

            if options['stim']:
                # save stimulus for this event as separate dictionary entry
                if type(stim) is dict:
                    stim_dict["STIM_" + tags[eventidx] + snr_suff] = stim[tags[eventidx]]
                else:
                    stim_dict["STIM_" + tags[eventidx] + snr_suff] = stim[:, :, eventidx]
            else:
                stim_dict["STIM_" + tags[eventidx] + snr_suff] = np.array([[]])
            # complicated experiment-specific part
            tag_mask_start = (
                    "PreStimSilence , " + tags[eventidx] + " , Reference"
                    )
            tag_mask_stop = (
                    "PostStimSilence , " + tags[eventidx] + " , Reference"
                    )

            ffstart = (exptevents['name'] == tag_mask_start)
            if np.sum(ffstart) > 0:
                ffstop = (exptevents['name'] == tag_mask_stop)
            else:
                ffstart = (exptevents['name'].str.contains(tag_mask_start))
                ffstop = (exptevents['name'].str.contains(tag_mask_stop))

            # remove start events that don't have a complementary stop
            tff1, = np.where(ffstart)
            tff2, = np.where(ffstop)
            for tff in tff1:
                if tff+2 not in tff2:
                    ffstart[tff]=False

            # create intial list of stimulus events
            this_event_times = pd.concat(
                    [exptevents.loc[ffstart, ['start']].reset_index(),
                     exptevents.loc[ffstop, ['end']].reset_index()],
                    axis=1
                    )
            this_event_times = this_event_times.drop(columns=['index'])
            this_event_times['name'] = "STIM_" + tags[eventidx] + snr_suff

            # screen for conflicts with target events
            keepevents = np.ones(len(this_event_times)) == 1
            keeppostevents = np.ones(len(this_event_times)) == 1
            for i, d in this_event_times.iterrows():
                if remove_post_lick:
                    fdur = ((ff_tar_dur | ff_lick_dur)
                            & (exptevents['start'] < d['end'] - 0.001)
                            & (exptevents['end'] > d['start'] + 0.001))
                else:
                    fdur = (ff_tar_dur
                            & (exptevents['start'] < d['end'] - 0.001)
                            & (exptevents['end'] > d['start'] + 0.001))

                if np.sum(fdur) and \
                   (exptevents['start'][fdur].min() < d['start'] + 0.5):
                    # assume fully overlapping, delete automaticlly
                    # print("Stim (event {0}: {1:.2f}-{2:.2f} {3}"
                    #       .format(eventidx,d['start'], d['end'],d['name']))
                    # print("??? But did it happen?"
                    #       "? Conflicting target: {0}-{1} {2}"
                    #       .format(exptevents['start'][j],
                    #               exptevents['end'][j],
                    #               exptevents['name'][j]))
                    keepevents[i] = False
                    keeppostevents[i] = False
                elif np.sum(fdur):
                    # truncate reference period
                    # print("adjusting {0}-{1}={2}".format(this_event_times['end'][i],
                    #        exptevents['start'][fdur].min(),
                    #        this_event_times['end'][i]-exptevents['start'][fdur].min()))
                    this_event_times.loc[i, 'end'] = \
                       exptevents['start'][fdur].min()
                    keeppostevents[i] = False

            if np.sum(keepevents == False):
                print("Removed {0}/{1} events that overlap with target"
                      .format(np.sum(keepevents == False), len(keepevents)))

            # create final list of these stimulus events
            this_event_times = this_event_times[keepevents]
            tff1, = np.where(ffstart)
            tff2, = np.where(ffstop)
            try:
                ffstart[tff1[keepevents == False]] = False
                ffstop[tff2[keeppostevents == False]] = False
            except:
                import pdb; pdb.set_trace()

            event_times = event_times.append(this_event_times,
                                             ignore_index=True)
            this_event_times['name'] = "REFERENCE"
            event_times = event_times.append(this_event_times,
                                             ignore_index=True)
            # event_times = pd.concat([event_times, this_event_times])

            ff_pre_all = ff_pre_all | ffstart
            ff_post_all = ff_post_all | ffstop

        # generate list of corresponding pre/post events
        this_event_times2 = pd.concat(
                [exptevents.loc[ff_pre_all, ['start']],
                 exptevents.loc[ff_pre_all, ['end']]],
                axis=1
                )
        this_event_times2['name'] = 'PreStimSilence'
        this_event_times3 = pd.concat(
                [exptevents.loc[ff_post_all, ['start']],
                 exptevents.loc[ff_post_all, ['end']]],
                axis=1
                )
        this_event_times3['name'] = 'PostStimSilence'

        event_times = event_times.append(this_event_times2, ignore_index=True)
        event_times = event_times.append(this_event_times3, ignore_index=True)

        # create list of target events
        this_event_times = pd.concat(
                [exptevents.loc[ff_tar_pre, ['start']].reset_index(),
                 exptevents.loc[ff_tar_post, ['end']].reset_index()],
                axis=1
                )
        this_event_times = this_event_times.drop(columns=['index'])
        this_event_times['name'] = "TARGET"
        event_times = event_times.append(this_event_times, ignore_index=True)

        #import pdb; pdb.set_trace()
        for i,e in exptevents[ff_tar_events | ff_lick_dur].iterrows():
            name = e['name']
            elements = name.split(" , ")

            if elements[0] == "PreStimSilence":
                name="PreStimSilence"
            elif elements[0] == "Stim":
                if not active_CC:
                    name="TAR_" + elements[1]
                else:
                    name="STIM_" + elements[1]
                e['start'] = exptevents.loc[i-1]['start']
                e['end'] = exptevents.loc[i+1]['end']
            elif elements[0] == "PostStimSilence":
                name="PostStimSilence"
            else:
                name = "LICK"
                licklen = 0.1
                e['end']=e['start'] + licklen
            # print('adding {} {}-{}'.format(name,e['start'], e['end']))
            te = pd.DataFrame(index=[0], columns=(event_times.columns),
                              data=[[e['start'], e['end'], name]])
            event_times = event_times.append(te, ignore_index=True)

        # event_times = pd.concat(
        #         [event_times, this_event_times2, this_event_times3]
        #         )

    #import pdb; pdb.set_trace()

    # add behavior events
    if exptparams['runclass'] == 'PTD' and any_behavior:
        # special events for tone in noise task
        tar_idx_freq = exptparams['TrialObject'][1]['TargetIdxFreq']
        tar_snr = exptparams['TrialObject'][1]['RelativeTarRefdB']
        common_tar_idx, = np.where(tar_idx_freq == np.max(tar_idx_freq))

        if (isinstance(tar_idx_freq, (int))
                or len(tar_idx_freq) == 1
                or np.isinf(tar_snr[0])):
            diff_event = 'PURETONE_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (np.max(common_tar_idx) < 2):
            diff_event = 'EASY_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (2 in common_tar_idx):
            diff_event = 'MEDIUM_BEHAVIOR'
        elif np.isfinite(tar_snr[0]) & (np.min(common_tar_idx) > 2):
            diff_event = 'HARD_BEHAVIOR'
        else:
            diff_event = 'PURETONE_BEHAVIOR'
        te = pd.DataFrame(index=[0], columns=(event_times.columns))
        te.loc[0] = [file_start_time, file_stop_time, diff_event]
        event_times = event_times.append(te, ignore_index=True)
        # event_times=pd.concat([event_times, te])

    # sort by when the event occurred in experiment time

    event_times = event_times.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    event_times = event_times.drop(columns=['index'])

    return event_times, spike_dict, stim_dict, state_dict


def baphy_load_recording(**options):
    """
    CRH 03-13-2020

    This function should only be called from baphy_generate_recording_uri!
    It should NOT be called directly. It now assumes that cellid has been
    parsed and that cellid/siteid/batch/rawid/mfilename etc. are all taken care
    of already. Given this information, it generates a nems recording using
    the options dictionary.

    TODO: This should interfact with BAPHYExperiment in the future, somehow...

    Returns
        rec: recording

    "old to-do's" are these still relevant?
    # TODO: add logic for options['resp']==False so this doesn't barf if
    #       cellid/batch/celllist/siteid aren't specified
    # TODO: break apart different signal loaders, make them work even if
    #        resp=False
    # TODO: (load all signals for one recording, make recording) =>
    #           move functionality to BAPHYExperiment wrapper
    #       then concatenate recordings

    """

    # STEP 1: FIGURE OUT FILE(S) and SIGNAL(S) TO LOAD

    options = fill_default_options(options)
    meta = options
    mfilename = options.get('mfilename', None)
    cellid = options.get('cellid', None)
    batch = options.get('batch', None)
    siteid = options.get('siteid', None)

    if not options.get('resp', True):
        # eg, load training + pupil data
        if mfilename is None:
            raise ValueError("must specify mfilename if resp==False")
        rec_name = siteid

    rec_name = siteid

    if mfilename is None:
        # query database to find all baphy files that belong to this site/batch
        d = db.get_batch_cell_data(batch=batch, cellid=siteid, label='parm',
                                   rawid=options['rawid'])
        dni = d.reset_index()
        files = list(set(list(d['parm'])))
        files.sort()
        goodtrials = np.array([], dtype=bool)
    else:
        if type(mfilename) is list:
            files = mfilename
        else:
            files=[mfilename]
        goodtrials = np.array([], dtype=bool)
        dni = None

    if len(files) == 0:
       raise ValueError('Data not found for cell {0}/batch {1}'.format(
               cellid,batch))

    # STEP 2: LOOP THROUGH FILES, LOAD RELEVANT SIGNALS FOR EACH
    for i, parmfilepath in enumerate(files):
        # load the file and do a bunch of preprocessing:
        if options["runclass"] == "RDT":
            log.info("loading RDT data")
            event_times, spike_dict, stim_dict, \
                state_dict, stim1_dict, stim2_dict = \
                baphy_load_dataset_RDT(parmfilepath, **options)
        else:
            event_times, spike_dict, stim_dict, state_dict = \
                baphy_load_dataset(parmfilepath, **options)

            d2 = event_times.loc[0].copy()
            if (i == 0) and (d2['name'] == 'PASSIVE_EXPERIMENT'):
                d2['name'] = 'PRE_PASSIVE'
                event_times = event_times.append(d2)
            elif d2['name'] == 'PASSIVE_EXPERIMENT':
                d2['name'] = 'POST_PASSIVE'
                event_times = event_times.append(d2)

        tt = event_times[event_times['name'].str.startswith('TRIAL')]
        trialcount = len(tt)

        # if loading from database, check to see if goodtrials are specified
        # so that bad trials can be masked out
        _goodtrials = np.ones(trialcount, dtype=bool)
        if dni is not None:
            s_goodtrials = dni.loc[dni['parm'] ==
                                   parmfilepath, 'goodtrials'].values[0]

            if (s_goodtrials is not None) and len(s_goodtrials):
                log.info("goodtrials not empty: %s", s_goodtrials)
                s_goodtrials = re.sub("[\[\]]", "", s_goodtrials)
                g = s_goodtrials.split(" ")
                _goodtrials = np.zeros(trialcount, dtype=bool)
                for b in g:
                    b1 = b.split(":")
                    if len(b1) == 1:
                        # single trial in list, simulate colon syntax
                        b1 = b1 + b1
                    _goodtrials[(int(b1[0])-1):int(b1[1])] = True

        goodtrials = np.concatenate((goodtrials, _goodtrials))

        # generate response signal
        t_resp = nems.signal.PointProcess(
                fs=options['rasterfs'], data=spike_dict,
                name='resp', recording=rec_name, chans=list(spike_dict.keys()),
                epochs=event_times
                )
        if i == 0:
            resp = t_resp
        else:
            # concatenate onto end of main response signal
            resp = resp.append_time(t_resp)

        if options['pupil']:

            # create pupil signal if it exists
            if i == 0:
                rlen = int(t_resp.shape[1])
            else:
                rlen = int(resp.shape[1]) - int(pupil.shape[1])

            pcount = state_dict['pupiltrace'].shape[0]
            plen = state_dict['pupiltrace'].shape[1]
            if plen > rlen:
                state_dict['pupiltrace'] = \
                    state_dict['pupiltrace'][:, 0:-(plen-rlen)]
            elif rlen > plen:
                state_dict['pupiltrace'] = \
                    np.append(state_dict['pupiltrace'],
                              np.ones([pcount, rlen - plen]) * np.nan,
                              axis=1)

            # generate pupil signals
            t_pupil = nems.signal.RasterizedSignal(
                    fs=options['rasterfs'], data=state_dict['pupiltrace'],
                    name='pupil', recording=rec_name, chans=['pupil'],
                    epochs=event_times)

            if i == 0:
                pupil = t_pupil
            else:
                pupil = pupil.concatenate_time([pupil, t_pupil])

            print("rlen={}  plen={}".format(resp.ntimes, pupil.ntimes))
            max_this=t_resp.epochs['end'].max()
            max_all=resp.epochs['end'].max()
            print('resp max times: this={:.15f} all={:.15f}'.format(max_this,max_all))
            max_this=t_pupil.epochs['end'].max()
            max_all=pupil.epochs['end'].max()
            print('pupil max times: this={:.15f} all={:.15f}'.format(max_this,max_all))

        if (options['pupil_eyespeed']) and ('pupil_eyespeed' in state_dict.keys()):
            # create pupil signal if it exists
            rlen = int(t_resp.ntimes)
            pcount = state_dict['pupil_eyespeed'].shape[0]
            plen = state_dict['pupil_eyespeed'].shape[1]
            if plen > rlen:
                state_dict['pupil_eyespeed'] = \
                    state_dict['pupil_eyespeed'][:, 0:-(plen-rlen)]
            elif rlen > plen:
                state_dict['pupil_eyespeed'] = \
                    np.append(state_dict['pupil_eyespeed'],
                              np.ones([pcount, rlen - plen]) * np.nan,
                              axis=1)

            # generate pupil signals
            t_pupil_s = nems.signal.RasterizedSignal(
                    fs=options['rasterfs'], data=state_dict['pupil_eyespeed'],
                    name='pupil_eyespeed', recording=rec_name, chans=['pupil_eyespeed'],
                    epochs=event_times)

            if i == 0:
                pupil_speed = t_pupil_s
            else:
                pupil_speed = pupil_speed.concatenate_time([pupil_speed, t_pupil_s])

        if options['rem'] and (state_dict.get('rem') is not None):

            # create pupil signal if it exists
            rlen = int(t_resp.ntimes)
            plen = state_dict['rem'].shape[0]

            if plen > rlen:
                state_dict['rem'] = \
                    state_dict['rem'][0:-(plen-rlen)]
            elif rlen > plen:
                state_dict['rem'] = \
                    np.append(state_dict['rem'],
                              np.ones(rlen - plen) * np.nan,
                              axis=0)
            print(np.nansum(state_dict['rem']))
            # generate pupil signals
            t_rem = nems.signal.RasterizedSignal(
                    fs=options['rasterfs'],
                    data=np.reshape(state_dict['rem'],[1,-1]),
                    name='rem', recording=rec_name, chans=['rem'],
                    epochs=event_times)

            if i == 0:
                rem = t_rem
            else:
                rem = rem.concatenate_time([rem, t_rem])
            print(np.nansum(rem.as_continuous()))
        else:
            options['rem'] = 0

        if options['stim'] and options["runclass"] == "RDT":
            log.info("concatenating RDT stim")

            t_stim1 = nems.signal.TiledSignal(
                fs=options['rasterfs'], data=stim1_dict,
                name='fg', epochs=event_times,
                recording=rec_name
            )
            t_stim2 = nems.signal.TiledSignal(
                fs=options['rasterfs'], data=stim2_dict,
                name='bg', epochs=event_times,
                recording=rec_name
            )
            BigStimMatrix = state_dict['BigStimMatrix'].copy()
            del state_dict['BigStimMatrix']
            t_state = nems.signal.TiledSignal(
                fs=options['rasterfs'], data=state_dict,
                name='state', epochs=event_times,
                recording=rec_name
                )
            t_state.chans = ['repeating', 'dual_stream', 'target_id']
            t_state = t_state.rasterize()
            x = t_state.loc['target_id'].as_continuous()
            tars = np.unique(x[~np.isnan(x)])

            if i == 0:
                stim1 = t_stim1
                stim2 = t_stim2
                state = t_state
            else:
                stim1 = stim1.append_time(t_stim1)
                stim2 = stim2.append_time(t_stim2)
                state = state.append_time(t_state)

        if options['stim']:
            # accumulate dictionaries
            # CRH replaced cellid w/ site (for when cellid is list)
            t_stim = nems.signal.TiledSignal(
                    data=stim_dict, fs=options['rasterfs'], name='stim',
                    epochs=event_times, recording=rec_name
                    )

            if i == 0:
                print("i={0} starting".format(i))
                stim = t_stim
            else:
                print("i={0} concatenating".format(i))
                # TODO implement concatenate_time for SignalDictionary
                # this basicall just needs to merge the data dictionaries
                # a la : new_dict={**stim._data,**t_stim.data}
                stim = stim.append_time(t_stim)

    resp.meta = options
    signals = {'resp': resp}

    if options['stim']:
        signals['stim'] = stim

    if options['pupil']:
        signals['pupil'] = pupil
    if (options['pupil_eyespeed']) & ('pupil_eyespeed' in state_dict.keys()):
        signals['pupil_eyespeed'] = pupil_speed
    if options['rem']:
        signals['rem'] = rem

    if options['stim'] and options["runclass"] == "RDT":
        signals['fg'] = stim1
        signals['bg'] = stim2
    if options["runclass"] == "RDT":
        signals['state'] = state
        #signals['stim'].meta={'BigStimMatrix': BigStimMatrix}
    meta['files']=files
    rec = nems.recording.Recording(signals=signals, meta=meta, name=siteid)

    if goodtrials.size > np.sum(goodtrials):
        log.info(goodtrials)
        # remove epochs from bad trials (not goodtrials), avoid a
        #log.info(goodtrials)
        bad_bounds = rec['resp'].get_epoch_bounds('TRIAL')[~goodtrials]
        all_bounds = rec['resp'].epochs[['start','end']].values

        bad_epochs = ep.epoch_contained(all_bounds, bad_bounds)
        rec['resp'].epochs.loc[bad_epochs]
        new_epochs = rec['resp'].epochs.drop(rec['resp'].epochs.index[bad_epochs])

        #import pdb; pdb.set_trace()
        for key, s in rec.signals.items():
            s.epochs = new_epochs

        # mask out trials outside of goodtrials range, specified in celldb
        # usually during meska save
        #trial_epochs = rec['resp'].get_epoch_indices('TRIAL')
        #good_epochs = trial_epochs[goodtrials]
        #good_epochs[:, 1] += 1
        #rec = rec.create_mask(good_epochs)
        #log.info('masking and resetting epochs for good trials')
        #rec = rec.apply_mask(reset_epochs=True)

    return rec


def baphy_load_recording_rasterized(**options):
    """
    Wrapper for baphy_load_recording

    Calls baphy_load_recording (nonrasterized) and then rasterizes resp and
    stim signals.
    """
    rec = baphy_load_recording(**options)

    if 'resp' in rec.signals:
        rec['resp'] = rec['resp'].rasterize()
    if 'stim' in rec.signals:
        rec['stim'] = rec['stim'].rasterize()

    return rec


def baphy_load_specgram(stimfilepath):

    matdata = scipy.io.loadmat(stimfilepath, chars_as_strings=True)

    stim = matdata['stim']

    stimparam = matdata['stimparam'][0][0]

    try:
        # case 1: loadstimfrombaphy format
        # remove redundant tags from tag list and stimulus array
        d = matdata['stimparam'][0][0][0][0]
        d = [x[0] for x in d]
        tags, tagids = np.unique(d, return_index=True)

        stim = stim[:, :, tagids]
    except:
        # loadstimbytrial format. don't want to filter by unique tags.
        # field names within stimparam don't seem to be preserved
        # in this load format??
        d = matdata['stimparam'][0][0][2][0]
        tags = [x[0] for x in d]

    return stim, tags, stimparam


def baphy_stim_cachefile_DEP(exptparams, parmfilepath=None, **options):
    """
    generate cache filename generated by loadstimfrombaphy

    code adapted from loadstimfrombaphy.m
    """
    raise Warning('DEPRECATED FUNCTION??')
    if 'truncatetargets' not in options:
        options['truncatetargets'] = 1
    if 'pertrial' not in options:
        options['pertrial'] = False

    if options['pertrial']:
        # loadstimbytrial cache filename format
        pp, bb = os.path.split(parmfilepath)
        bb = bb.split(".")[0]
        dstr = "loadstimbytrial_{0}_ff{1}_fs{2}_cc{3}_trunc{4}.mat".format(
                     bb, options['stimfmt'], options['rasterfs'],
                     options['chancount'], options['truncatetargets']
                     )
        return stim_cache_dir + dstr

    # otherwise use standard load stim from baphy format
    if options['runclass'] is None:
        RefObject = exptparams['TrialObject'][1]['ReferenceHandle'][1]
    elif 'runclass' in exptparams.keys():
        runclass = exptparams['runclass'].split("_")
        if (len(runclass) > 1) and (runclass[1] == options["runclass"]):
            RefObject = exptparams['TrialObject'][1]['TargetHandle'][1]
        else:
            RefObject = exptparams['TrialObject'][1]['ReferenceHandle'][1]
    else:
        RefObject = exptparams['TrialObject'][1]['ReferenceHandle'][1]

    dstr = RefObject['descriptor']
    if dstr == 'Torc':
        if 'RunClass' in exptparams['TrialObject'][1].keys():
            dstr += '-'+exptparams['TrialObject'][1]['RunClass']
        else:
            dstr += '-TOR'

    # include all parameter values, even defaults, in filename
    fields = RefObject['UserDefinableFields']
    for cnt1 in range(0, len(fields), 3):
        if RefObject[fields[cnt1]] == 0:
            RefObject[fields[cnt1]] = int(0)
            # print(fields[cnt1])
            # print(RefObject[fields[cnt1]])
            # print(dstr)
        dstr = "{0}-{1}".format(dstr, RefObject[fields[cnt1]])

    dstr = re.sub(r":", r"", dstr)

    if 'OveralldB' in exptparams['TrialObject'][1]:
        OveralldB = exptparams['TrialObject'][1]['OveralldB']
        dstr += "-{0}dB".format(OveralldB)
    else:
        OveralldB = 0

    dstr += "-{0}-fs{1}-ch{2}".format(
            options['stimfmt'], options['rasterfs'], options['chancount']
            )

    if options['includeprestim']:
        dstr += '-incps1'

    dstr = re.sub(r"[ ,]", r"_", dstr)
    dstr = re.sub(r"[\[\]]", r"", dstr)

    return stim_cache_dir + dstr + '.mat'


def baphy_load_spike_data_raw(spkfilepath, channel=None, unit=None):

    matdata = scipy.io.loadmat(spkfilepath, chars_as_strings=True)

    sortinfo = matdata['sortinfo']
    if sortinfo.shape[0] > 1:
        sortinfo = sortinfo.T
    sortinfo = sortinfo[0]

    # figure out sampling rate, used to convert spike times into seconds
    spikefs = matdata['rate'][0][0]

    return sortinfo, spikefs


def spike_time_to_raster(spike_dict, fs=100, event_times=None):
    """
    convert list of spike times to a raster of spike rate, with duration
    matching max end time in the event_times list
    """

    if event_times is not None:
        maxtime = np.max(event_times["end"])

    maxbin = np.int(np.ceil(fs*maxtime))
    unitcount = len(spike_dict.keys())
    raster = np.zeros([unitcount, maxbin])

    cellids = sorted(spike_dict)
    for i, key in enumerate(cellids):
        for t in spike_dict[key]:
            b = int(np.floor(t*fs))
            if b < maxbin:
                raster[i, b] += 1

    return raster, cellids


def dict_to_signal(stim_dict, fs=100, event_times=None, signal_name='stim',
                   recording_name='rec'):

    maxtime = np.max(event_times["end"])
    maxbin = int(fs*maxtime)

    tags = list(stim_dict.keys())
    chancount = stim_dict[tags[0]].shape[0]

    z = np.zeros([chancount, maxbin])

    empty_stim = nems.signal.RasterizedSignal(
            data=z, fs=fs, name=signal_name,
            epochs=event_times, recording=recording_name
            )
    stim = empty_stim.replace_epochs(stim_dict)
    # stim = stim.normalize('minmax')

    return stim

# ========================= Specialized loader fns for RDT data ================================
def baphy_load_dataset_RDT_old(parmfilepath, **options):
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

    options['pupil'] = options.get('pupil', False)
    options['stim'] = options.get('stim', True)
    options['runclass'] = options.get('runclass', None)

    # get the relatively un-pre-processed data
    exptevents, stim, spike_dict, state_dict, tags, stimparam, exptparams = \
        baphy_load_data(parmfilepath, **options)

    # pre-process event list (event_times) to only contain useful events

    # extract each trial
    tag_mask_start = "TRIALSTART"
    tag_mask_stop = "TRIALSTOP"
    ffstart = exptevents['name'].str.startswith(tag_mask_start)
    ffstop = (exptevents['name'] == tag_mask_stop)
    TrialCount = np.max(exptevents.loc[ffstart, 'Trial'])
    event_times = pd.concat(
            [exptevents.loc[ffstart, ['start']].reset_index(),
             exptevents.loc[ffstop, ['end']].reset_index()],
            axis=1
            )
    event_times['name'] = "TRIAL"
    event_times = event_times.drop(columns=['index'])

    stim_dict = {}
    stim1_dict = {}
    stim2_dict = {}
    state_dict = {}

    # make stimulus events unique to each trial
    this_event_times = event_times.copy()
    rasterfs = options['rasterfs']
    BigStimMatrix = stimparam[-1]
    state = np.zeros([3, stim.shape[1], stim.shape[2]])
    single_stream_trials = (BigStimMatrix[0, 1, :] == -1)
    state[1, :, single_stream_trials] = 1
    prebins = int(exptparams['TrialObject'][1]['PreTrialSilence']*rasterfs)
    samplebins = int(
            exptparams['TrialObject'][1]['ReferenceHandle'][1]['Duration']
            * rasterfs
            )

    for trialidx in range(0, TrialCount):
        event_name = "TRIAL{0}".format(trialidx)
        this_event_times.loc[trialidx, 'name'] = event_name
        stim1_dict[event_name] = stim[:, :, trialidx, 0]
        stim2_dict[event_name] = stim[:, :, trialidx, 1]
        stim_dict[event_name] = stim[:, :, trialidx, 2]

        s = np.zeros([3, stim_dict[event_name].shape[1]])
        rslot = np.argmax(np.diff(BigStimMatrix[:, 0, trialidx]) == 0) + 1
        rbin = prebins + rslot*samplebins
        s[0, rbin:] = 1
        single_stream_trial = int(BigStimMatrix[0, 1, trialidx] == -1)
        s[1, :] = single_stream_trial
        tarslot = np.argmin(BigStimMatrix[:, 0, trialidx] > 0) - 1
        s[2, :] = BigStimMatrix[tarslot, 0, trialidx]
        state_dict[event_name] = s

    state_dict['BigStimMatrix'] = BigStimMatrix

    event_times = pd.concat([event_times, this_event_times])

    # add stim events
    stim_mask = "Stim ,"
    ffstim = (exptevents['name'].str.contains(stim_mask))
    stim_event_times = exptevents.loc[ffstim, ['name', 'start', 'end']]
    event_times = pd.concat([event_times, stim_event_times])

    # sort by when the event occured in experiment time
    event_times = event_times.sort_values(by=['start', 'end'])
    cols = ['name', 'start', 'end']
    event_times = event_times[cols]
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

    return (event_times, spike_dict, stim_dict, state_dict,
            stim1_dict, stim2_dict)

def baphy_load_dataset_RDT(parmfilepath, options, sample_offset,
                           sequence_offset):
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

    #options['pupil'] = options.get('pupil', False)
    #options['stim'] = options.get('stim', True)
    #options['runclass'] = options.get('runclass', None)

    # get the relatively un-pre-processed data
    exptevents, stim, spike_dict, state_dict, tags, stimparam, exptparams = \
        baphy_load_data(parmfilepath, **options)

    # extract each trial
    tag_mask_start = "TRIALSTART"
    tag_mask_stop = "TRIALSTOP"
    ffstart = (exptevents['name'] == tag_mask_start)
    ffstop = (exptevents['name'] == tag_mask_stop)
    TrialCount = np.max(exptevents.loc[ffstart, 'Trial'])
    event_times = pd.concat(
            [exptevents.loc[ffstart, ['start']].reset_index(),
             exptevents.loc[ffstop, ['end']].reset_index()],
            axis=1
            )
    event_times['name'] = "TRIAL"
    event_times = event_times.drop(columns=['index'])

    stim_dict = {}
    stim1_dict = {}
    stim2_dict = {}
    state_dict = {}

    # make stimulus events unique to each trial
    this_event_times = event_times.copy()
    rasterfs = options['rasterfs']
    BigStimMatrix = stimparam[-1]

    prebins = int(exptparams['TrialObject'][1]['PreTrialSilence']*rasterfs)
    samplebins = int(
            exptparams['TrialObject'][1]['ReferenceHandle'][1]['Duration']
            * rasterfs
            )

    bsm = np.rollaxis(BigStimMatrix, -1)
    sequences = np.unique(bsm, axis=0)
    sequence_info = {'name': [], 'start': [], 'end': []}

    for trialidx in range(0, TrialCount):
        event_name = "TRIAL{0}".format(trialidx)
        this_event_times.loc[trialidx, 'name'] = event_name
        stim1_dict[event_name] = stim[:, :, trialidx, 0]
        stim2_dict[event_name] = stim[:, :, trialidx, 1]
        stim_dict[event_name] = stim[:, :, trialidx, 2]

        s = np.zeros([3, stim_dict[event_name].shape[1]])

        # 1 if repeating, 0 if not
        rslot = np.argmax(np.diff(BigStimMatrix[:, 0, trialidx]) == 0) + 1
        rbin = prebins + rslot*samplebins
        s[0, rbin:] = 1

        # 1 if dual stream, 0 if not
        dual_stream_trial = int(BigStimMatrix[0, 1, trialidx] != -1)
        s[1, :] = dual_stream_trial

        # Target ID
        tarslot = np.argmin(BigStimMatrix[:, 0, trialidx] > 0) - 1
        s[2, :] = BigStimMatrix[tarslot, 0, trialidx] + sample_offset
        state_dict[event_name] = s

        bsm_trial = bsm[trialidx]
        sequence_mask = np.all(sequences == bsm_trial, axis=(1, 2))
        sequence_id = np.flatnonzero(sequence_mask)[0] + sequence_offset + 1

        sequence_name = "SEQUENCE{}".format(sequence_id)
        start, end = this_event_times.loc[trialidx, ['start', 'end']].values
        sequence_info['name'].append(sequence_name)
        sequence_info['start'].append(start)
        sequence_info['end'].append(end)

    sequence_times = pd.DataFrame(sequence_info)
    event_times = pd.concat([event_times, this_event_times, sequence_times])

    def callback(match, o=sample_offset):
        i = int(match.group()) + o
        return '{:02d}'.format(i)

    # add stim events
    stim_mask = "Stim ,"
    ffstim = (exptevents['name'].str.contains(stim_mask))
    stim_event_times = exptevents.loc[ffstim, ['name', 'start', 'end']]
    stim_event_times['name'] = stim_event_times['name'] \
        .apply(lambda x: re.sub('\d+', callback, x))
    event_times = pd.concat([event_times, stim_event_times])

    # sort by when the event occured in experiment time
    event_times = event_times.sort_values(by=['start', 'end'])
    cols = ['name', 'start', 'end']
    event_times = event_times[cols]

    return (event_times, spike_dict, stim_dict, state_dict,
            stim1_dict, stim2_dict)


def baphy_load_recording_RDT(cellid, batch, options):
    """
    query NarfData to find baphy files for specified cell/batch and then load
    """

    # print(options)
    #options['rasterfs'] = int(options.get('rasterfs', 100))
    #options['stimfmt'] = options.get('stimfmt', 'ozgf')
    #options['chancount'] = int(options.get('chancount', 18))
    #options['pertrial'] = int(options.get('pertrial', False))
    #options['includeprestim'] = options.get('includeprestim', 1)

    #options['stim'] = int(options.get('stim', True))
    #options['runclass'] = options.get('runclass', None)
    #options['cellid'] = options.get('cellid', cellid)
    #options['batch'] = int(batch)
    #options['rawid'] = options.get('rawid', None)

    d = db.get_batch_cell_data(batch=batch,
                               cellid=cellid,
                               rawid=options['rawid'],
                               label='parm')

    if len(d)==0:
        raise ValueError('cellid/batch entry not found in NarfData')

    files = list(d['parm'])

    sample_offset = 0
    sequence_offset = 0

    for i, parmfilepath in enumerate(files):
        event_times, spike_dict, stim_dict, \
            state_dict, stim1_dict, stim2_dict = \
            baphy_load_dataset_RDT(parmfilepath, options, sample_offset,
                                   sequence_offset)

        epoch_name = 'FILE{}'.format(i)
        epoch_start = event_times['start'].min()
        epoch_end = event_times['end'].max()
        event_times = event_times.append({
            'name': epoch_name,
            'start': epoch_start,
            'end': epoch_end
            }, ignore_index=True)

        # Find maximum sample ID and increment offset
        m = event_times['name'].str.startswith('Stim')
        s = ','.join(event_times.loc[m].name)
        s_id = [int(i) for i in re.findall('\d+', s)]
        s_offset = max(s_id)
        sample_offset += s_offset

        # Find maximum sequence ID and increment offset
        m = event_times['name'].str.startswith('SEQUENCE')
        s = ','.join(event_times.loc[m].name)
        s_id = [int(i) for i in re.findall('\d+', s)]
        s_offset = max(s_id)
        sequence_offset += s_offset

        # generate spike raster
        raster_all, cellids = spike_time_to_raster(
                spike_dict, fs=options['rasterfs'], event_times=event_times
                )

        if 1:
            # generate response signal
            t_resp = nems.signal.PointProcess(
                    fs=options['rasterfs'], data=spike_dict,
                    name='resp', recording=cellid, chans=list(spike_dict.keys()),
                    epochs=event_times
                    )
            if i == 0:
                resp = t_resp
            else:
                # concatenate onto end of main response signal
                resp = resp.append_time(t_resp)
        else:
            # generate response signal
            t_resp = nems.signal.RasterizedSignal(
                    fs=options['rasterfs'], data=raster_all, name='resp',
                    recording=cellid, chans=cellids, epochs=event_times
                    )
            if i == 0:
                resp = t_resp
            else:
                resp = resp.concatenate_time([resp, t_resp])

        if options['stim']:
            t_stim = dict_to_signal(stim_dict, fs=options['rasterfs'],
                                    event_times=event_times)
            t_stim.recording = cellid

            if i == 0:
                log.info("i={0} starting".format(i))
                stim = t_stim
            else:
                log.info("i={0} concatenating".format(i))
                stim = stim.concatenate_time([stim, t_stim])

            t_stim1 = dict_to_signal(
                    stim1_dict, fs=options['rasterfs'],
                    event_times=event_times, signal_name='fg',
                    recording_name=cellid
                    )
            t_stim2 = dict_to_signal(
                    stim2_dict, fs=options['rasterfs'],
                    event_times=event_times, signal_name='bg',
                    recording_name=cellid
                    )
            t_state = dict_to_signal(
                    state_dict, fs=options['rasterfs'],
                    event_times=event_times, signal_name='state',
                    recording_name=cellid
                    )
            t_state.chans = ['repeating', 'dual_stream', 'target_id']
            x = t_state.loc['target_id'].as_continuous()
            tars = np.unique(x[~np.isnan(x)])

            if i == 0:
                stim1 = t_stim1
                stim2 = t_stim2
                state = t_state
            else:
                stim1 = stim1.concatenate_time([stim1, t_stim1])
                stim2 = stim2.concatenate_time([stim2, t_stim2])
                state = state.concatenate_time([state, t_state])

    resp.meta = options
    resp.meta['files'] = files
    signals = {'resp': resp}

    if options['stim']:
        signals['stim'] = stim
        signals['fg'] = stim1
        signals['bg'] = stim2

    signals['state'] = state
    rec = nems.recording.Recording(signals=signals)
    return rec

# ================ Kilosort utils, do these belong here? Make new module? ======================

def get_kilosort_template(batch=None, cellid=None):
    """
    return the waveform template for the given cellid. only works for cellids
    with the current naming scheme i.e. TAR017b-07-2. crh 2018-07-24
    """
    parmfile = db.get_batch_cell_data(batch=batch, cellid=cellid).T.values[0][0]
    path = os.path.dirname(parmfile)+'/sorted/'
    rootname = ('.').join([os.path.basename(parmfile).split('.')[0], 'spk.mat'])
    spkfile = path+rootname
    sortdata = scipy.io.loadmat(spkfile, chars_as_strings=True)

    try:
        chan = int(cellid[-4:-2])
        unit = int(cellid[-1:])
        template = sortdata['sortinfo'][0][chan-1][0][0][unit-1][0]['Template'][chan-1,:]
    except:
        template=np.nan

    return template


def get_kilosort_templates(batch=None):
    """
    Return dataframe containing the waveform template for every cellid in this
    batch. crh 2018-07-24
    """
    cellids = db.get_batch_cells(batch)['cellid']
    df = pd.DataFrame(index=cellids, columns=['template'])

    for c in cellids:
        df.loc[c]['template'] = get_kilosort_template(batch=batch, cellid=c)

    return df

# ========================= DEPRECATED FUNCTIONS ============================

def baphy_load_recording_nonrasterized(**options):
    """
    DEPRECRATED, replaced by baphy_load_recording
    """
    raise DeprecationWarning("replaced by baphy_load_recording")
    return baphy_load_recording(**options)

def baphy_data_path(**options):
    """
    DEPRECATED:
        replaced by baphy_load_recording_uri

    required entries in options dictionary:
        cellid: string or list
            string can be a valid cellid or siteid
            list is a list of cellids from a single(?) site
        batch: integer
        rasterfs
        stimfmt
        chancount
    """
    raise DeprecationWarning("This function is deprecated. Use baphy_load_recording_uri")

    if (options.get('cellid') is None) or \
       (options.get('batch') is None) or \
       (options.get('rasterfs') is None) or \
       (options.get('stimfmt') is None) or \
       (options.get('chancount') is None):
        raise ValueError("cellid,batch,rasterfs,stimfmt,chancount options required")

    recache = options.get('recache', 0)
    if 'recache' in options:
        del options['recache']

    options = fill_default_options(options)
    log.info(options)

    # three ways to select cells
    cellid = options.get('cellid', None)
    if type(cellid) is list:
        cellid = cellid[0].split("-")[0]

    elif cellid is None and options.get('siteid') is not None:
        cellid = options.get('siteid')

    siteid = options.get('siteid', cellid.split("-")[0])

    # TODO : base filename on siteid/cellid plus hash from JSON-ized options
    #data_path = (get_setting('NEMS_RECORDINGS_DIR') + "/{0}/{1}{2}_fs{3}/"
    #             .format(options["batch"], options['stimfmt'],
    #                     options["chancount"], options["rasterfs"]))
    #data_file = data_path + cellid + '.tgz'

    data_file = recording_filename_hash(
            siteid, options, uri_path=get_setting('NEMS_RECORDINGS_DIR'))

    #log.info(data_file)
    #log.info(options)

    if not os.path.exists(data_file) or recache is True:
        #  rec = baphy_load_recording(
        #          options['cellid'], options['batch'], options
        #          )
        if options['runclass'] == 'RDT':
            log.info('cellid: %s', options['cellid'])
            log.info('batch: %s', options['batch'])
            rec = baphy_load_recording_RDT(options['cellid'], options['batch'], options)
        else:
            rec = baphy_load_recording(**options)
        log.info(rec.name)
        log.info(rec.signals.keys())
        rec.save(data_file)

    return data_file
