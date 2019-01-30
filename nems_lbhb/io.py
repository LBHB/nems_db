#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:47:56 2018

@author: svd
"""

import logging
import re
import os
import os.path
import pickle
import scipy.io
import scipy.io as spio
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
from itertools import groupby, repeat, chain, product

import pandas as pd
import matplotlib.pyplot as plt
import nems.signal
import nems.recording
import nems.db as db
from nems.recording import Recording
from nems.recording import load_recording

log = logging.getLogger(__name__)

# paths to baphy data -- standard locations on elephant
stim_cache_dir = '/auto/data/tmp/tstim/'  # location of cached stimuli
spk_subdir = 'sorted/'   # location of spk.mat files relative to parmfiles


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def baphy_mat2py(s):

    s3 = re.sub(r';', r'', s.rstrip())
    s3 = re.sub(r'%', r'#', s3)
    s3 = re.sub(r'\\', r'/', s3)
    s3 = re.sub(r"\.([a-zA-Z0-9]+)'", r"XX\g<1>'", s3)
    s3 = re.sub(r"\.([a-zA-Z0-9]+)\+", r"XX\g<1>+", s3)
    s3 = re.sub(r"\.([a-zA-Z0-9]+) ,", r"XX\g<1> ,", s3)
    s3 = re.sub(r'globalparams\(1\)', r'globalparams', s3)
    s3 = re.sub(r'exptparams\(1\)', r'exptparams', s3)

    s4 = re.sub(r'\(([0-9]*)\)', r'[\g<1>]', s3)

    s5 = re.sub(r'\.([A-Za-z][A-Za-z0-9_]+)', r"['\g<1>']", s4)

    s6 = re.sub(r'([0-9]+) ', r"\g<0>,", s5)
    s6 = re.sub(r'NaN ', r"np.nan,", s6)
    s6 = re.sub(r'Inf ', r"np.inf,", s6)

    s7 = re.sub(r"XX([a-zA-Z0-9]+)'", r".\g<1>'", s6)
    s7 = re.sub(r"XX([a-zA-Z0-9]+)\+", r".\g<1>+", s7)
    s7 = re.sub(r"XX([a-zA-Z0-9]+) ,", r".\g<1> ,", s7)
    s7 = re.sub(r',,', r',', s7)
    s7 = re.sub(r',Hz', r'Hz', s7)
    s7 = re.sub(r'NaN', r'np.nan', s7)
    s7 = re.sub(r'zeros\(([0-9,]+)\)', r'np.zeros([\g<1>])', s7)
    s7 = re.sub(r'{(.*)}', r'[\g<1>]', s7)

    s8 = re.sub(r" , REF-[0-9]+", r" , Reference", s7)
    s8 = re.sub(r" , TARG-[0-9]+", r" , Reference", s8)

    return s8


def baphy_parm_read(filepath):
    log.info("Loading {0}".format(filepath))

    f = io.open(filepath, "r")
    s = f.readlines(-1)

    globalparams = {}
    exptparams = {}
    exptevents = {}

    for ts in s:
        sout = baphy_mat2py(ts)
        # print(sout)
        try:
            exec(sout)
        except KeyError:
            ts1 = sout.split('= [')
            ts1 = ts1[0].split(',[')

            s1 = ts1[0].split('[')
            sout1 = "[".join(s1[:-1]) + ' = {}'
            try:
                exec(sout1)
            except:
                s2 = sout1.split('[')
                sout2 = "[".join(s2[:-1]) + ' = {}'
                try:
                    exec(sout2)
                except:
                    s3 = sout2.split('[')
                    sout3 = "[".join(s3[:-1]) + ' = {}'
                    try:
                        exec(sout3)
                    except:
                        s4 = sout3.split('[')
                        sout4 = "[".join(s4[:-1]) + ' = {}'
                        exec(sout4)
                        exec(sout3)
                    exec(sout2)
                exec(sout1)
            exec(sout)
        except NameError:
            log.info("NameError on: {0}".format(sout))
        except:
            log.info("Other error on: {0} to {1}".format(ts,sout))

    # special conversions

    # convert exptevents to a DataFrame:
    t = [exptevents[k] for k in exptevents]
    d = pd.DataFrame(t)
    if 'ClockStartTime' in d.columns:
        exptevents = d.drop(['Rove', 'ClockStartTime'], axis=1)
    elif 'Rove' in d.columns:
        exptevents = d.drop(['Rove'], axis=1)
    else:
        exptevents = d
    # rename columns to NEMS standard epoch names
    exptevents.columns = ['name', 'start', 'end', 'Trial']
    for i in range(len(exptevents)):
        if exptevents.loc[i, 'end'] == []:
            exptevents.loc[i, 'end'] = exptevents.loc[i, 'start']

    if 'ReferenceClass' not in exptparams['TrialObject'][1].keys():
        exptparams['TrialObject'][1]['ReferenceClass'] = \
           exptparams['TrialObject'][1]['ReferenceHandle'][1]['descriptor']
    # CPP special case, deletes added commas
    if exptparams['TrialObject'][1]['ReferenceClass'] == 'ContextProbe':
        tags = exptparams['TrialObject'][1]['ReferenceHandle'][1]['Names']  # gets the list of tags
        tag_map = {oldtag: re.sub(r' , ', r'  ', oldtag)
                   for oldtag in tags}  # eliminates commas with regexp and maps old tag to new commales tag
        # places the commaless tags back in place
        exptparams['TrialObject'][1]['ReferenceHandle'][1]['Names'] = list(tag_map.values())
        # extends the tag map adding pre stim and post prefix, and Reference sufix
        epoch_map = dict()
        for sufix, tag in product(['PreStimSilence', 'Stim', 'PostStimSilence'], tags):
            key = '{} , {} , Reference'.format(sufix, tag)
            val = '{} , {} , Reference'.format(sufix, tag_map[tag])
            epoch_map[key] = val
        # replaces exptevents names using the map, i.e. get rid of commas
        exptevents.replace(epoch_map, inplace=True)

    return globalparams, exptparams, exptevents


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


def baphy_stim_cachefile(exptparams, parmfilepath=None, **options):
    """
    generate cache filename generated by loadstimfrombaphy

    code adapted from loadstimfrombaphy.m
    """

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


def baphy_align_time(exptevents, sortinfo, spikefs, finalfs=0):

    # number of channels in recording (not all necessarily contain spikes)
    chancount = len(sortinfo)

    # figure out how long each trial is by the time of the last spike count.
    # this method is a hack!
    # but since recordings are longer than the "official"
    # trial end time reported by baphy, this method preserves extra spikes
    TrialCount = np.max(exptevents['Trial'])
    TrialLen_sec = np.array(
            exptevents.loc[exptevents['name'] == "TRIALSTOP"]['start']
            )
    TrialLen_spikefs = np.concatenate(
            (np.zeros([1, 1]), TrialLen_sec[:, np.newaxis]*spikefs), axis=0
            )

    for c in range(0, chancount):
        if len(sortinfo[c]) and sortinfo[c][0].size:
            s = sortinfo[c][0][0]['unitSpikes']
            s = np.reshape(s, (-1, 1))
            unitcount = s.shape[0]
            for u in range(0, unitcount):
                st = s[u, 0]

                # print('chan {0} unit {1}: {2} spikes'.format(c,u,st.shape[1]))
                for trialidx in range(1, TrialCount+1):
                    ff = (st[0, :] == trialidx)
                    if np.sum(ff):
                        utrial_spikefs = np.max(st[1, ff])
                        TrialLen_spikefs[trialidx, 0] = np.max(
                                [utrial_spikefs, TrialLen_spikefs[trialidx, 0]]
                                )

    # using the trial lengths, figure out adjustments to trial event times.
    if finalfs:
        log.info('rounding Trial offset spike times'
                 ' to even number of rasterfs bins')
        # print(TrialLen_spikefs)
        TrialLen_spikefs = (
                np.ceil(TrialLen_spikefs / spikefs*finalfs)
                / finalfs*spikefs
                )
        # print(TrialLen_spikefs)

    Offset_spikefs = np.cumsum(TrialLen_spikefs)
    Offset_sec = Offset_spikefs / spikefs  # how much to offset each trial

    # adjust times in exptevents to approximate time since experiment started
    # rather than time since trial started (native format)
    for Trialidx in range(1, TrialCount+1):
        # print("Adjusting trial {0} by {1} sec"
        #       .format(Trialidx,Offset_sec[Trialidx-1]))
        ff = (exptevents['Trial'] == Trialidx)
        exptevents.loc[ff, ['start', 'end']] = (
                exptevents.loc[ff, ['start', 'end']] + Offset_sec[Trialidx-1]
                )

        # ff = ((exptevents['Trial'] == Trialidx)
        #       & (exptevents['end'] > Offset_sec[Trialidx]))
        # badevents, = np.where(ff)
        # print("{0} events past end of trial?".format(len(badevents)))
        # exptevents.drop(badevents)

    log.info("{0} trials totaling {1:.2f} sec".format(TrialCount, Offset_sec[-1]))

    # convert spike times from samples since trial started to
    # (approximate) seconds since experiment started (matched to exptevents)
    totalunits = 0
    spiketimes = []  # list of spike event times for each unit in recording
    unit_names = []  # string suffix for each unit (CC-U)
    chan_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for c in range(0, chancount):
        if len(sortinfo[c]) and sortinfo[c][0].size:
            s = sortinfo[c][0][0]['unitSpikes']
            comment = sortinfo[c][0][0][0][0][2][0]
            log.debug('Comment: %s', comment)

            s = np.reshape(s, (-1, 1))
            unitcount = s.shape[0]
            for u in range(0, unitcount):
                st = s[u, 0]
                uniquetrials = np.unique(st[0, :])
                # print('chan {0} unit {1}: {2} spikes {3} trials'
                #       .format(c, u, st.shape[1], len(uniquetrials)))

                unit_spike_events = np.array([])
                for trialidx in uniquetrials:
                    ff = (st[0, :] == trialidx)
                    this_spike_events = (st[1, ff]
                                         + Offset_spikefs[np.int(trialidx-1)])
                    if (comment != []):
                        if (comment == 'PC-cluster sorted by mespca.m'):
                            # remove last spike, which is stray
                            this_spike_events = this_spike_events[:-1]
                    unit_spike_events = np.concatenate(
                            (unit_spike_events, this_spike_events), axis=0
                            )
                    # print("   trial {0} first spike bin {1}"
                    #       .format(trialidx,st[1,ff]))

                totalunits += 1
                if chancount <= 8:
                    unit_names.append("{0}{1}".format(chan_names[c], u+1))
                else:
                    unit_names.append("{0:02d}-{1}".format(c+1, u+1))
                spiketimes.append(unit_spike_events / spikefs)

    return exptevents, spiketimes, unit_names


def set_default_pupil_options(options):

    options = options.copy()
    options["rasterfs"] = options.get('rasterfs', 100)
    options["pupil_offset"] = options.get('pupil_offset', 0.75)
    options["pupil_deblink"] = options.get('pupil_deblink', True)
    options["pupil_deblink_dur"] = options.get('pupil_deblink_dur', (1/3))
    options["pupil_median"] = options.get('pupil_median', 0)
    options["pupil_smooth"] = options.get('pupil_smooth', 0)
    options["pupil_highpass"] = options.get('pupil_highpass', 0)
    options["pupil_lowpass"] = options.get('pupil_lowpass', 0)
    options["pupil_bandpass"] = options.get('pupil_bandpass', 0)
    options["pupil_derivative"] = options.get('pupil_derivative', '')
    options["pupil_mm"] = options.get('pupil_mm', False)
    options["pupil_eyespeed"] = options.get('pupil_eyespeed', False)
    options["rem_units"] = options.get('rem_units', 'mm')
    options["rem_min_pupil"] = options.get('rem_min_pupil', 0.2)
    options["rem_max_pupil"] = options.get('rem_max_pupil', 1)
    options["rem_max_pupil_sd"] = options.get('rem_max_pupil_sd', 0.05)
    options["rem_min_saccade_speed"] = options.get('rem_min_saccade_speed', 0.5)
    options["rem_min_saccades_per_minute"] = options.get('rem_min_saccades_per_minute', 0.01)
    options["rem_max_gap_s"] = options.get('rem_max_gap_s', 15)
    options["rem_min_episode_s"] = options.get('rem_min_episode_s', 30)
    options["verbose"] = options.get('verbose', True)

    return options


def load_pupil_trace(pupilfilepath, exptevents=None, **options):
    """
    returns big_rs which is pupil trace resampled to options['rasterfs']
    and strialidx, which is the index into big_rs for the start of each
    trial. need to make sure the big_rs vector aligns with the other signals
    """

    options = set_default_pupil_options(options)

    rasterfs = options["rasterfs"]
    pupil_offset = options["pupil_offset"]
    pupil_deblink = options["pupil_deblink"]
    pupil_deblink_dur = options["pupil_deblink_dur"]
    pupil_median = options["pupil_median"]
    pupil_mm = options["pupil_mm"]
    pupil_eyespeed = options["pupil_eyespeed"]
    verbose = options["verbose"]
    options['pupil'] = options.get('pupil', True)
    #rasterfs = options.get('rasterfs', 100)
    #pupil_offset = options.get('pupil_offset', 0.75)
    #pupil_deblink = options.get('pupil_deblink', True)
    #pupil_deblink_dur = options.get('pupil_deblink_dur', (1/3))
    #pupil_median = options.get('pupil_median', 0)
    #pupil_mm = options.get('pupil_mm', False)
    #pupil_eyespeed = options.get('pupil_eyespeed', False)
    #verbose = options.get('verbose', False)

    if options["pupil_smooth"]:
        raise ValueError('pupil_smooth not implemented. try pupil_median?')
    if options["pupil_highpass"]:
        raise ValueError('pupil_highpass not implemented.')
    if options["pupil_lowpass"]:
        raise ValueError('pupil_lowpass not implemented.')
    if options["pupil_bandpass"]:
        raise ValueError('pupil_bandpass not implemented.')
    if options["pupil_derivative"]:
        raise ValueError('pupil_derivative not implemented.')

    # we want to use exptevents TRIALSTART events as the ground truth for the time when each trial starts.
    # these times are set based on openephys data, since baphy doesn't log exact trial start times
    if exptevents is None:
        # if exptevents hasn't been loaded and corrected for spike data yet, do that so that we have accurate trial
        # start times.
        # key exptevents are exptevents['name'].str.startswith('TRIALSTART')
        parmfilepath = pupilfilepath.replace(".pup.mat",".m")
        globalparams, exptparams, exptevents = baphy_parm_read(parmfilepath)
        pp, bb = os.path.split(parmfilepath)
        spkfilepath = pp + '/' + spk_subdir + re.sub(r"\.m$", ".spk.mat", bb)
        log.info("Spike file: {0}".format(spkfilepath))
        # load spike times
        sortinfo, spikefs = baphy_load_spike_data_raw(spkfilepath)
        # adjust spike and event times to be in seconds since experiment started
        exptevents, spiketimes, unit_names = baphy_align_time(
                exptevents, sortinfo, spikefs, rasterfs
                )

    try:
        basename = os.path.basename(pupilfilepath).split('.')[0]
        abs_path = os.path.dirname(pupilfilepath)
        pupildata_path = os.path.join(abs_path, "sorted", basename + '.pickle')

        with open(pupildata_path, 'rb') as fp:
            pupildata = pickle.load(fp)

        # hard code to use minor axis for now
        options['pupil_variable_name'] = 'minor_axis'
        log.info("Using default pupil_variable_name: " +
                 options['pupil_variable_name'])
        log.info("Using CNN results for pupiltrace")

        pupil_diameter = pupildata['cnn']['a'] * 2

        pupil_diameter = pupil_diameter[:-1, np.newaxis]

        log.info("pupil_diameter.shape: " + str(pupil_diameter.shape))

        if pupil_eyespeed:
            try:
                eye_speed = pupildata['cnn']['eyespeed'][:-1, np.newaxis]
                log.info("loaded eye_speed")
            except:
                pupil_eyespeed = False
                log.info("eye_speed requested but file does not exist!")

    except:
        matdata = scipy.io.loadmat(pupilfilepath)

        log.info("Attempted to load pupil from CNN analysis, but file didn't exist. Loading from pup.mat")

        p = matdata['pupil_data']
        params = p['params']
        if 'pupil_variable_name' not in options:
            options['pupil_variable_name'] = params[0][0]['default_var'][0][0][0]
            log.info("Using default pupil_variable_name: " +
                     options['pupil_variable_name'])
        if 'pupil_algorithm' not in options:
            options['pupil_algorithm'] = params[0][0]['default'][0][0][0]
            log.info("Using default pupil_algorithm: " + options['pupil_algorithm'])

        results = p['results'][0][0][-1][options['pupil_algorithm']]
        pupil_diameter = np.array(results[0][options['pupil_variable_name']][0][0])
        if pupil_diameter.shape[0] == 1:
            pupil_diameter = pupil_diameter.T
        log.info("pupil_diameter.shape: " + str(pupil_diameter.shape))

        if pupil_eyespeed:
            try:
                eye_speed = np.array(results[0]['eye_speed'][0][0])
                log.info("loaded eye_speed")
            except:
                pupil_eyespeed = False
                log.info("eye_speed requested but file does not exist!")


    fs_approximate = 30  # approx video framerate
    if pupil_deblink:
        dp = np.abs(np.diff(pupil_diameter, axis=0))
        blink = np.zeros(dp.shape)
        blink[dp > np.nanmean(dp) + 6*np.nanstd(dp)] = 1
        # CRH add following line 7-19-2019
        # (blink should be = 1 if pupil_dia goes to 0)
        blink[[isclose(p, 0, abs_tol=0.5) for p in pupil_diameter[:-1]]] = 1
        smooth_width = int(fs_approximate*pupil_deblink_dur)
        box = np.ones([smooth_width]) / smooth_width
        blink = np.convolve(blink[:, 0], box, mode='same')
        blink[blink > 0] = 1
        blink[blink <= 0] = 0
        onidx, = np.where(np.diff(blink) > 0)
        offidx, = np.where(np.diff(blink) < 0)

        if onidx[0] > offidx[0]:
            onidx = np.concatenate((np.array([0]), onidx))
        if len(onidx) > len(offidx):
            offidx = np.concatenate((offidx, np.array([len(blink)])))
        deblinked = pupil_diameter.copy()
        if pupil_eyespeed:
            deblinked_eye_speed = eye_speed.copy()
        for i, x1 in enumerate(onidx):
            x2 = offidx[i]
            if x2 < x1:
                log.info([i, x1, x2])
                log.info("WHAT'S UP??")
            else:
                # print([i,x1,x2])
                deblinked[x1:x2, 0] = np.linspace(
                        deblinked[x1], deblinked[x2-1], x2-x1
                        )
                if pupil_eyespeed:
                    deblinked_eye_speed[x1:x2, 0] = np.nan

        if verbose:
            plt.figure()
            if pupil_eyespeed:
                plt.subplot(2, 1, 1)
            plt.plot(pupil_diameter, label='Raw')
            plt.plot(deblinked, label='Deblinked')
            plt.xlabel('Frame')
            plt.ylabel('Pupil')
            plt.legend()
            plt.title("Artifacts detected: {}".format(len(onidx)))
            if pupil_eyespeed:
                plt.subplot(2, 1, 2)
                plt.plot(eye_speed, label='Raw')
                plt.plot(deblinked_eye_speed, label='Deblinked')
                plt.xlabel('Frame')
                plt.ylabel('Eye speed')
                plt.legend()
        pupil_diameter = deblinked
        if pupil_eyespeed:
            eye_speed = deblinked_eye_speed

    if pupil_eyespeed:
        returned_measurement = eye_speed
    else:
        returned_measurement = pupil_diameter

    # resample and remove dropped frames

    # find and parse pupil events
    pp = ['PUPIL,' in x['name'] for i, x in exptevents.iterrows()]
    trials = list(exptevents.loc[pp, 'Trial'])
    ntrials = len(trials)
    timestamp = np.zeros([ntrials+1])
    firstframe = np.zeros([ntrials+1])
    for i, x in exptevents.loc[pp].iterrows():
        t = x['Trial'] - 1
        s = x['name'].split(",[")
        p = eval("["+s[1])
        # print("{0} p=[{1}".format(i,s[1]))
        timestamp[t] = p[0]
        firstframe[t] = int(p[1])
    pp = ['PUPILSTOP' in x['name'] for i, x in exptevents.iterrows()]
    lastidx = np.argwhere(pp)[-1]

    s = exptevents.iloc[lastidx[0]]['name'].split(",[")
    p = eval("[" + s[1])
    timestamp[-1] = p[0]
    firstframe[-1] = int(p[1])

    # align pupil with other events, probably by
    # removing extra bins from between trials
    ff = exptevents['name'].str.startswith('TRIALSTART')
    start_events = exptevents.loc[ff, ['start']].reset_index()
    start_events['StartBin'] = (
            np.round(start_events['start'] * rasterfs)
            ).astype(int)
    start_e = list(start_events['StartBin'])
    ff = (exptevents['name'] == 'TRIALSTOP')
    stop_events = exptevents.loc[ff, ['start']].reset_index()
    stop_events['StopBin'] = (
            np.round(stop_events['start'] * rasterfs)
            ).astype(int)
    stop_e = list(stop_events['StopBin'])

    # calculate frame count and duration of each trial
    #svd/CRH fix: use start_e to determine trial duration
    duration = np.diff(np.append(start_e, stop_e[-1]) / rasterfs)

    # old method: use timestamps in pupil recording, which don't take into account correction for sampling bin size
    # that may be coarser than the video sampling rate
    #duration = np.diff(timestamp) * 24*60*60

    frame_count = np.diff(firstframe)

    if pupil_eyespeed & options['pupil']:
        l = ['pupil', 'pupil_eyespeed']
    elif pupil_eyespeed:
        l = ['pupil_eyespeed']
    elif options['pupil']:
        l = ['pupil']

    big_rs_dict = {}

    for signal in l:
        if signal == 'pupil_eyespeed':
            pupil_eyespeed = True
        else:
            pupil_eyespeed = False

        # warp/resample each trial to compensate for dropped frames
        strialidx = np.zeros([ntrials + 1])
        big_rs = np.array([])
        all_fs = np.empty([ntrials])

        for ii in range(0, ntrials):

            if signal == 'pupil_eyespeed':
                d = eye_speed[
                        int(firstframe[ii]):int(firstframe[ii]+frame_count[ii]), 0
                        ]
            else:
                d = pupil_diameter[
                        int(firstframe[ii]):int(firstframe[ii]+frame_count[ii]), 0
                        ]
            fs = frame_count[ii] / duration[ii]
            all_fs[ii] = fs
            t = np.arange(0, len(d)) / fs
            if pupil_eyespeed:
                d = d * fs  # convert to px/s before resampling
            ti = np.arange(
                    (1/rasterfs)/2, duration[ii]+(1/rasterfs)/2, 1/rasterfs
                    )
            # print("{0} len(d)={1} len(ti)={2} fs={3}"
            #       .format(ii,len(d),len(ti),fs))
            di = np.interp(ti, t, d)
            big_rs = np.concatenate((big_rs, di), axis=0)
            if (ii < ntrials-1) and (len(big_rs) > start_e[ii+1]):
                big_rs = big_rs[:start_e[ii+1]]
            elif ii == ntrials-1:
                big_rs = big_rs[:stop_e[ii]]

            strialidx[ii+1] = len(big_rs)

        if pupil_median:
            kernel_size = int(round(pupil_median*rasterfs/2)*2+1)
            big_rs = scipy.signal.medfilt(big_rs, kernel_size=kernel_size)

        # shift pupil (or eye speed) trace by offset, usually 0.75 sec
        offset_frames = int(pupil_offset*rasterfs)
        big_rs = np.roll(big_rs, -offset_frames)

        # svd pad with final pupil value (was np.nan before)
        big_rs[-offset_frames:] = big_rs[-offset_frames]

        # shape to 1 x T to match NEMS signal specs
        big_rs = big_rs[np.newaxis, :]

        if pupil_mm:
            try:
                #convert measurements from pixels to mm
                eye_width_px = matdata['pupil_data']['results'][0][0]['eye_width'][0][0][0]
                eye_width_mm = matdata['pupil_data']['params'][0][0]['eye_width_mm'][0][0][0]
                big_rs = big_rs*(eye_width_mm/eye_width_px)
            except:
                print("couldn't convert pupil to mm")

        if verbose:
            #plot framerate for each trial (for checking camera performance)
            plt.figure()
            plt.plot(all_fs.T)
            plt.xlabel('Trial')
            plt.ylabel('Sampling rate (Hz)')

        if verbose:
            plt.show()

        if len(l)==2:
            big_rs_dict[signal] = big_rs

    if len(l)==2:
        return big_rs_dict, strialidx
    else:
        return big_rs, strialidx


def get_rem(pupilfilepath, exptevents=None, **options):
    """
    Find rapid eye movements based on pupil and eye-tracking data.

    Inputs:

        pupilfilepath: Absolute path of the pupil file (to be loaded by
        nems_lbhb.io.load_pupil_trace).

        exptevents:

        options: Dictionary of analysis parameters
            rasterfs: Sampling rate (default: 100)
            rem_units: If 'mm', convert pupil to millimeters and eye speed to
              mm/s while loading (default: 'mm')
            rem_min_pupil: Minimum pupil size during REM episodes (default: 0.2)
            rem_max_pupil: Maximum pupil size during REM episodes (mm, default: 1)
            rem_max_pupil_sd: Maximum pupil standard deviation during REM episodes
             (default: 0.05)
            rem_min_saccade_speed: Minimum eye movement speed to consider eye
             movement as saccade (default: 0.01)
            rem_min_saccades_per_minute: Minimum saccades per minute during REM
             episodes (default: 0.01)
            rem_max_gap_s: Maximum gap to fill in between REM episodes
             (seconds, default: 15)
            rem_min_episode_s: Minimum duration of REM episodes to keep
             (seconds, default: 30)
            verbose: Plot traces and identified REM episodes (default: True)

    Returns:

        is_rem: Numpy array of booleans, indicating which time bins occured
         during REM episodes (True = REM)

        options: Dictionary of parameters used in analysis

    ZPS 2018-09-24: Initial version.
    """

    #Set analysis parameters from defaults, if necessary.
    options = set_default_pupil_options(options)

    rasterfs = options["rasterfs"]
    units = options["rem_units"]
    min_pupil = options["rem_min_pupil"]
    max_pupil = options["rem_max_pupil"]
    max_pupil_sd = options["rem_max_pupil_sd"]
    min_saccade_speed = options["rem_min_saccade_speed"]
    min_saccades_per_minute = options["rem_min_saccades_per_minute"]
    max_gap_s = options["rem_max_gap_s"]
    min_episode_s = options["rem_min_episode_s"]
    verbose = options["verbose"]

    #Load data.
    load_options = options.copy()
    load_options["verbose"] = False
    if units == 'mm':
        load_options["pupil_mm"] = True
    elif units == "px":
        load_options["pupil_mm"] = False
    elif units == 'norm_max':
        raise ValueError("TODO: support for norm pupil diam/speed by max")
        load_options['norm_max'] = True

    load_options["pupil_eyespeed"] = True
    pupil_trace, _ = load_pupil_trace(pupilfilepath, exptevents, **load_options)
    pupil_size = pupil_trace["pupil"]
    eye_speed = pupil_trace["pupil_eyespeed"]

    pupil_size = pupil_size[0,:]
    eye_speed = eye_speed[0,:]

    #Find REM episodes.

    #(1) Very small pupil sizes often indicate that the pupil is occluded by the
    #eyelid or underlit. In either case, measurements of eye position are
    #unreliable, so we remove these frames of the trace before analysis.
    pupil_size[np.nan_to_num(pupil_size) < min_pupil] = np.nan
    eye_speed[np.nan_to_num(pupil_size) < min_pupil] = np.nan

    #(2) Rapid eye movements are similar to saccades. In our data,
    #these appear as large, fast spikes in the speed at which pupil moves.
    #To mark epochs when eye is moving more quickly than usual, threshold
    #eye speed, then smooth by calculating the rate of saccades per minute.
    saccades = np.nan_to_num(eye_speed) > min_saccade_speed
    minute = np.ones(rasterfs*60)/(rasterfs*60)
    saccades_per_minute = np.convolve(saccades, minute, mode='same')

    #(3) To distinguish REM sleep from waking - since it seeems that ferrets
    #can sleep with their eyes open - look for periods when pupil is constricted
    #and doesn't show slow oscillations (which may indicate a different sleep
    #stage or quiet waking).
    #  10-second moving average of pupil size:
    ten_seconds = np.ones(rasterfs*10)/(rasterfs*10)
    smooth_pupil_size = np.convolve(pupil_size, ten_seconds, mode='same');
    # 10-second moving standard deviation of pupil size:
    pupil_sd = pd.Series(smooth_pupil_size)
    pupil_sd = pupil_sd.rolling(rasterfs*10).std()
    pupil_sd = np.array(pupil_sd)
    rem_episodes = (np.nan_to_num(smooth_pupil_size) < max_pupil) & \
                   (np.isfinite(smooth_pupil_size)) & \
                   (np.nan_to_num(pupil_sd) < max_pupil_sd) & \
                   (np.nan_to_num(saccades_per_minute) > min_saccades_per_minute)

    #(4) Connect episodes that are separated by a brief gap.
    rem_episodes = run_length_encode(rem_episodes)
    brief_gaps = []
    for i,episode in enumerate(rem_episodes):
        is_gap = not(episode[0])
        gap_time = episode[1]
        if is_gap and gap_time/rasterfs < max_gap_s:
            rem_episodes[i] = (True, gap_time)
            brief_gaps.append((True, gap_time))
        else:
            brief_gaps.append((False, gap_time))

    #(5) Remove brief REM episodes.
    rem_episodes = run_length_encode(run_length_decode(rem_episodes))
    brief_episodes = []
    for i,episode in enumerate(rem_episodes):
        is_rem_episode = episode[0]
        episode_time = episode[1]
        if is_rem_episode and episode_time/rasterfs < min_episode_s:
            rem_episodes[i] = (False, episode_time)
            brief_episodes.append((True, episode_time))
        else:
            brief_episodes.append((False, episode_time))

    is_rem = run_length_decode(rem_episodes)

    #Plot
    if verbose:

        samples = pupil_size.size
        minutes = samples/(rasterfs*60)
        time_ax = np.linspace(0, minutes, num=samples)

        is_brief_gap = run_length_decode(brief_gaps)
        is_brief_episode = run_length_decode(brief_episodes)
        rem_dur = np.array([t for is_rem,t in rem_episodes if is_rem])/(rasterfs*60)

        fig, ax = plt.subplots(4,1)
        if len(rem_dur) == 0:
            title_str = 'no REM episodes'
        elif len(rem_dur) == 1:
            title_str = '1 REM episode, duration: {:0.2f} minutes'.\
                format(rem_dur[0])
        else:
            title_str = '{:d} REM episodes, mean duration: {:0.2f} minutes'.\
                format(len(rem_dur), rem_dur.mean())
        title_str = '{:s}\n{:s}'.format(pupilfilepath, title_str)
        fig.suptitle(title_str)

        ax[0].autoscale(axis='x', tight=True)
        ax[0].plot(time_ax, eye_speed, color='0.5')
        ax[0].plot([time_ax[0], time_ax[-1]], \
                [min_saccade_speed, min_saccade_speed], 'k--')
        ax[0].set_ylabel('Eye speed')

        ax[1].autoscale(axis='x', tight=True)
        ax[1].plot(time_ax, saccades_per_minute, color='0', linewidth=2)
        ax[1].plot([time_ax[0], time_ax[-1]], \
                [min_saccades_per_minute, min_saccades_per_minute], 'k--')
        l0, = ax[1].plot(time_ax[is_rem.nonzero()], \
                   saccades_per_minute[is_rem.nonzero()], 'r.')
        l1, = ax[1].plot(time_ax[is_brief_gap.nonzero()], \
                         saccades_per_minute[is_brief_gap.nonzero()], 'b.')
        l2, = ax[1].plot(time_ax[is_brief_episode.nonzero()], \
                         saccades_per_minute[is_brief_episode.nonzero()], 'y.')
        ax[1].set_ylabel('Saccades per minute')

        ax[0].legend((l0,l1,l2), \
                     ('REM', 'Brief gaps (included)', 'Brief episodes (excluded)'), \
                     frameon=False)

        ax[2].autoscale(axis='x', tight=True)
        ax[2].plot(time_ax, pupil_size, color='0.5')
        ax[2].plot(time_ax, smooth_pupil_size, color='0', linewidth=2)
        ax[2].plot([time_ax[0], time_ax[-1]], \
                [max_pupil, max_pupil], 'k--')
        ax[2].plot(time_ax[is_rem.nonzero()], \
                smooth_pupil_size[is_rem.nonzero()], 'r.')
        ax[2].plot(time_ax[is_brief_gap.nonzero()], \
                smooth_pupil_size[is_brief_gap.nonzero()], 'b.')
        ax[2].plot(time_ax[is_brief_episode.nonzero()], \
                smooth_pupil_size[is_brief_episode.nonzero()], 'y.')
        ax[2].set_ylabel('Pupil size')

        ax[3].autoscale(axis='x', tight=True)
        ax[3].plot(time_ax, pupil_sd, color='0', linewidth=2)
        ax[3].plot([time_ax[0], time_ax[-1]], \
                [max_pupil_sd, max_pupil_sd], 'k--')
        ax[3].plot(time_ax[is_rem.nonzero()], \
                pupil_sd[is_rem.nonzero()], 'r.')
        ax[3].plot(time_ax[is_brief_gap.nonzero()], \
                pupil_sd[is_brief_gap.nonzero()], 'b.')
        ax[3].plot(time_ax[is_brief_episode.nonzero()], \
                pupil_sd[is_brief_episode.nonzero()], 'y.')
        ax[3].set_ylabel('Pupil SD')
        ax[3].set_xlabel('Time (min)')

        plt.show()

    return is_rem, options


def run_length_encode(a):
    """
    Takes a 1-dimensional array, returns a list of tuples (elem, n), where
    elem is each symbol in the array, and n is the number of times it appears
    consecutively. For example, if given the array:
        np.array([False, True, True, True, False, False])
    the function will return:
        [(False, 1), (True, 3), (False, 2)]

    ZPS 2018-09-24: Helper function for get_rem.
    """
    return [(k, len(list(g))) for k,g in groupby(a)]

def run_length_decode(a):
    """
    Reverses the operation performed by run_length_encode.

    ZPS 2018-09-24: Helper function for get_rem.
    """
    a = [list(repeat(elem,n)) for (elem,n) in a]
    a = list(chain.from_iterable(a))
    return np.array(a)


def cache_rem_options(pupilfilepath, cachepath=None, **options):

    jsonfilepath = pupilfilepath.replace('.pup.mat', '.rem.json')
    if cachepath is not None:
        pp, bb = os.path.split(jsonfilepath)
        jsonfilepath = os.path.join(cachepath, bb)

    fh = open(jsonfilepath, 'w')
    json.dump(options, fh)
    fh.close()


def load_rem_options(pupilfilepath, cachepath=None, **options):

    jsonfilepath = pupilfilepath.replace('.pup.mat','.rem.json')
    if cachepath is not None:
        pp, bb = os.path.split(jsonfilepath)
        jsonfilepath = os.path.join(cachepath, bb)

    if os.path.exists(jsonfilepath):
        fh = open(jsonfilepath, 'r')
        options = json.load(fh)
        fh.close()
        return options
    else:
        raise ValueError("REM options file not found.")

def baphy_pupil_uri(pupilfilepath, **options):
    """
    return uri to pupil signal file
    if cache file doesn't exists, process the pupil data based on the contents
    of the relevant pup.mat file (pupilfilepath) and save to cache file.
    Then return cached filename.

    Processing:
        pull out pupil trace determined with specified algorithm
        warp time to match trial times in baphy paramter file
        extract REM trace if velocity signal exists

    Cache file location currently hard-coded to:
        /auto/data/nems_db/recordings/pupil/

    """
    #options['rasterfs']=100
    #options['pupil_mm']=True
    #options['pupil_median']=0.5
    #options['pupil_deblink']=True
    #options['units']='mm'
    #options['verbose']=False

    options = set_default_pupil_options(options)

    cacheroot = "/auto/data/nems_db/recordings/pupil/"

    pp, pupbb = os.path.split(pupilfilepath)
    pp_animal, pen = os.path.split(pp)
    pp_daq, animal = os.path.split(pp_animal)
    cachebb = pupbb.replace(".pup.mat","")
    cachepath = os.path.join(cacheroot, animal, )

    parmfilepath = pupilfilepath.replace(".pup.mat",".m")
    pp, bb = os.path.split(parmfilepath)

    globalparams, exptparams, exptevents = baphy_parm_read(parmfilepath)
    spkfilepath = pp + '/' + spk_subdir + re.sub(r"\.m$", ".spk.mat", bb)
    log.info("Spike file: {0}".format(spkfilepath))
    # load spike times
    sortinfo, spikefs = baphy_load_spike_data_raw(spkfilepath)
    # adjust spike and event times to be in seconds since experiment started

    exptevents, spiketimes, unit_names = baphy_align_time(
            exptevents, sortinfo, spikefs, options["rasterfs"])
    print('Creating trial events')
    tag_mask_start = "TRIALSTART"
    tag_mask_stop = "TRIALSTOP"
    ffstart = exptevents['name'].str.startswith(tag_mask_start)
    ffstop = (exptevents['name'] == tag_mask_stop)
    TrialCount = np.max(exptevents.loc[ffstart, 'Trial'])
    event_times = pd.concat([exptevents.loc[ffstart, ['start']].reset_index(),
                             exptevents.loc[ffstop, ['end']].reset_index()],
                            axis=1)
    event_times['name'] = "TRIAL"
    event_times = event_times.drop(columns=['index'])

    pupil_trace, ptrialidx = load_pupil_trace(pupilfilepath=pupilfilepath,
                                   exptevents=exptevents, **options)

    is_rem, options = get_rem(pupilfilepath=pupilfilepath,
                              exptevents=exptevents, **options)

    pupildata = np.stack([pupil_trace, is_rem], axis=1)
    t_pupil = nems.signal.RasterizedSignal(
            fs=options['rasterfs'], data=pupildata,
            name='pupil', recording=cachebb, chans=['pupil', 'rem'],
            epochs=event_times)


    return pupil_trace, is_rem, options
