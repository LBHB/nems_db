#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:47:56 2018

@author: svd

A bunch of routines for loading data from baphy/matlab

"""

from functools import lru_cache
from pathlib import Path
import logging
import re
import os
import os.path
import pickle

import pylab as pl
import scipy.io
import scipy.io as spio
import scipy.ndimage.filters
import scipy.signal
from scipy.interpolate import interp1d
import numpy as np
import collections
import json
import hashlib
import io
import datetime
import glob
from math import isclose
import copy
from itertools import groupby, repeat, chain, product
import h5py

from nems_lbhb import OpenEphys as oe
from nems_lbhb import SettingXML as oes
import pandas as pd
import matplotlib.pyplot as plt
import nems0.signal
import nems0.recording
import nems0.db as db
import nems0.epoch as ep
from nems0.recording import Recording
from nems0.recording import load_recording
import nems_lbhb.behavior as behavior
from nems_lbhb import runclass
from nems0.uri import load_resource

log = logging.getLogger(__name__)

# paths to baphy data -- standard locations on elephant
stim_cache_dir = '/auto/data/tmp/tstim/'  # location of cached stimuli
spk_subdir = 'sorted/'  # location of spk.mat files relative to parmfiles


# =================================================================

def baphy_align_time_openephys(events, timestamps, raw_rasterfs=30000, rasterfs=None, baphy_legacy_format=False):
    '''
    Parameters
    ----------
    events : DataFrame
        Events stored in BAPHY parmfile
    timestamps : array
        Array of timestamps (in seconds) as read in from openephys
    baphy_legacy_format : bool
        If True, assume that all data before the onset of the first trial are
        discarded (i.e., as is the case when aligning times using the spike
        times file. This results in the first trial having a start timestamp of
        0.
    '''
    if rasterfs is not None:
        timestamps = np.round(timestamps * rasterfs) / rasterfs

    n_baphy = events['Trial'].max()
    n_oe = len(timestamps)
    if n_baphy != n_oe:
        mesg = f'Number of trials in BAPHY ({n_baphy}) and ' \
               'OpenEphys ({n_oe}) do not match'
        raise ValueError(mesg)

    if baphy_legacy_format:
        timestamps = timestamps - timestamps[0]

    events = events.copy()
    for i, timestamp in enumerate(timestamps):
        m = events['Trial'] == i + 1
        events.loc[m, ['start', 'end']] += timestamp
    return events


###############################################################################
# Openephys utility functions
###############################################################################
def load_trial_starts_openephys(openephys_folder):
    '''
    Load trial start times (seconds) from OpenEphys DIO

    Parameters
    ----------
    openephys_folder : str or Path
        Path to OpenEphys folder
    '''
    event_file = Path(openephys_folder) / 'all_channels.events'
    data = oe.load(str(event_file))
    header = data.pop('header')
    df = pd.DataFrame(data)
    ts = df.query('(channel == 0) & (eventType == 3) & (eventId == 1)')

    return (ts['timestamps'].values) / float(header['sampleRate'])


def load_sampling_rate_openephys(openephys_folder):
    '''
    Load sampling rate (samples/sec) from OpenEphys DIO

    Parameters
    ----------
    openephys_folder : str or Path
        Path to OpenEphys folder
    '''
    event_file = Path(openephys_folder) / 'all_channels.events'
    data = oe.load(str(event_file))
    return int(data['header']['sampleRate'])


def load_continuous_openephys(fh):
    '''
    Read continous OpenEphys dataset

    Parameters
    ----------
    fh : {str, file-like object, buffer}
        If a file-like object or buffer, will read directly from it. If a
        string, will open the file first (and close upon exiting).

    Unlike the version provided by OpenEphys, this one can handle reading from
    buffered streams (e.g., such as that provided by a tarfile) or existing
    files.

    Example
    -------
    import tarfile

    parmfile = '/auto/data/daq/Nameko/NMK004/NMK004e06_p_NON.m'
    manager = io.BAPHYExperiment(parmfile)
    filename = manager.openephys_tarfile_relpath / '126_CH1.continuous'
    tar_fh = tarfile.open(manager.openephys_tarfile, 'r:gz')
    fh = tar_fh.extractfile(str(filename))
    ch_data = load_continous_openephys(fh)
    '''
    if not isinstance(fh, io.IOBase):
        fh = open(fh, 'rb')
        do_close = True
    else:
        do_close = False

    header = oe.readHeader(fh)
    scale = float(header['bitVolts'])
    ts_dtype = np.dtype('<i8')
    n_dtype = np.dtype('<u2')
    record_number_dtype = np.dtype('>u2')
    data_dtype = np.dtype('>i2')

    timestamps = []
    record_number = []
    data = []

    SAMPLES_PER_RECORD = 1024

    while True:
        try:
            b = fh.read(ts_dtype.itemsize)
            ts = np.frombuffer(b, ts_dtype, 1)[0]
            b = fh.read(n_dtype.itemsize)
            n = np.frombuffer(b, n_dtype, 1)[0]
            if n != SAMPLES_PER_RECORD:
                raise IOError('Found corrupt record')
            b = fh.read(record_number_dtype.itemsize)
            rn = np.frombuffer(b, record_number_dtype, 1)[0]
            b = fh.read(data_dtype.itemsize * n)
            d = np.frombuffer(b, data_dtype, n) * scale
            _ = fh.read(10)

            timestamps.append(ts)
            record_number.append(rn)
            data.append(d)
        except ValueError:
            # We have reached end of file?
            break

    timestamps = np.array(timestamps)
    record_number = np.array(record_number)
    data = np.concatenate(data)

    if do_close:
        fh.close()

    return {
        'header': header,
        'timestamps': timestamps,
        'data': data,
        'record_number': record_number,
    }


###############################################################################
# Unsorted functions
###############################################################################
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
    s3 = re.sub(r';$', r'', s.rstrip())
    s3 = re.sub(r'%', r'#', s3)
    s3 = re.sub(r'\\', r'/', s3)
    s3 = re.sub(r'\.wav[-_]', r'-',s3) # MLE 2019 05 29 kludge to avoid .wav file sufix to be considered as matlab struct.field
    s3 = re.sub(r"\.([a-zA-Z0-9]+)'", r"XX\g<1>'", s3)
    s3 = re.sub(r"\.([a-zA-Z0-9]+)'", r"XX\g<1>'", s3)
    s3 = re.sub(r"\.([a-zA-Z0-9]+)\+", r"XX\g<1>+", s3)
    s3 = re.sub(r"\.([a-zA-Z0-9]+) ,", r"XX\g<1> ,", s3)
    s3 = re.sub(r'globalparams\(1\)', r'globalparams', s3)
    s3 = re.sub(r'exptparams\(1\)', r'exptparams', s3)

    # special case for notes with () in them
    s4 = re.sub(r'\(([0-9]*)\) ,', r'-\g<1> ,', s3)
    s4 = re.sub(r'\(([0-9]*)\)', r'[\g<1>]', s4)

    s5 = re.sub(r'\.wav', r"",
                s4)  # MLE eliminates .wav file sufix to not confuse with field ToDo: elimiate .wav from param files ?
    s5 = re.sub(r'\.([A-Za-z][A-Za-z0-9_]+)', r"['\g<1>']", s5)

    s6 = re.sub(r'([0-9]+) ', r"\g<0>,", s5)
    x = s6.split('=')
    if len(x) > 1:
        if ';' in x[1]:
            x[1] = re.sub(r'\[', r"np.array([[", x[1])
            x[1] = re.sub(r'\]', r"]])", x[1])
            x[1] = re.sub(r';', '],[', x[1])
        x[1] = re.sub('true ', 'True,', x[1])
        x[1] = re.sub('false ', 'False,', x[1])
        x[1] = re.sub('true]', 'True]', x[1])
        x[1] = re.sub('false]', 'False]', x[1])
        x[1] = re.sub('true$', 'True', x[1])
        x[1] = re.sub('false$', 'False', x[1])
        x[1] = re.sub(r'NaN ', r"np.nan,", x[1])
        x[1] = re.sub(r'Inf ', r"np.inf,", x[1])
        x[1] = re.sub(r'NaN,', r"np.nan,", x[1])
        x[1] = re.sub(r'Inf,', r"np.inf,", x[1])
        x[1] = re.sub(r'NaN\]', r"np.nan]", x[1])
        x[1] = re.sub(r'Inf\]', r"np.inf]", x[1])
        s6 = "=".join(x)

    s7 = re.sub(r"XX([a-zA-Z0-9]+)'", r".\g<1>'", s6)
    s7 = re.sub(r"XX([a-zA-Z0-9]+)'", r".\g<1>'", s7)
    s7 = re.sub(r"XX([a-zA-Z0-9]+)\+", r".\g<1>+", s7)
    s7 = re.sub(r"XX([a-zA-Z0-9]+) ,", r".\g<1> ,", s7)
    s7 = re.sub(r',,', r',', s7)
    s7 = re.sub(r',Hz', r'Hz', s7)
    s7 = re.sub(r'NaN', r'np.nan', s7)
    s7 = re.sub(r'zeros\(([0-9,]+)\)', r'np.zeros([\g<1>])', s7)
    if s7.count('{') == s7.count('}'):
        s7 = s7.replace('{', '[')
        s7 = s7.replace('}', ']')
        # s7 = re.sub(r'{(.*?)}', r'[\g<1>]', s7) # Replace {*} by [*]. (.*?) because that finds shortest matches possible.
    else:
        raise RuntimeError('matlab->python string conversion failed because there were an unequal number of { and }')

    s8 = re.sub(r" , REF-[0-9]+", r" , Reference", s7)
    s8 = re.sub(r" , TARG-[0-9]+", r" , Reference", s8)

    return s8

def get_parmfile_format(parmfile):

    asfile = Path(parmfile).with_suffix('.m')
    aspath = Path(parmfile).with_suffix('')
    if os.path.exists(asfile):
        return 'baphy'
    elif os.path.exists(aspath):
        return 'psi'
    else:
        return None

def adjust_parmfile_name(parmfile):

    asfile = Path(parmfile).with_suffix('.m')
    aspath = Path(parmfile).with_suffix('')
    if os.path.exists(asfile):
        return asfile
    elif os.path.exists(aspath):
        return aspath
    else:
        return None


def psi_parm_read(filepath):
    """
    read parameter from psi datafolder. Relevant info in event_log.csv
    and trial_log.csv
    match spect of baphy loader:
       exptevents.columns = ['name', 'start', 'end', 'Trial']
    :param filepath:
    :return:
    """
    eventfile = Path(filepath) / "event_log.csv"
    trialfile = Path(filepath) / "trial_log.csv"
    globalfile = Path(filepath) / "globalparams.json"

    prestimsilence=0.5
    poststimsilence=0.5

    root1, parmfile = os.path.split(filepath)
    root2, penname = os.path.split(root1)
    root3, ferret = os.path.split(root2)
    parts = parmfile.split('_')
    siteid = parts[0][:-2]
    runclass = parts[2];
    ctime = datetime.datetime.fromtimestamp(os.path.getctime(filepath))

    if os.path.isfile(globalfile):
        with open(globalfile, 'r') as f:
            globalparams = json.load(f)
        globalparams['Tester'] = globalparams['experimenter']
        globalparams['Ferret'] = globalparams['animal']
        if globalparams['training']=='Physiology+behavior':
            globalparams['Physiology'] = 'Yes -- Behavior'
        else:
            globalparams['Physiology'] = 'Yes -- Passive'
    else:
        log.info('***** Kludge alert!! Hard coding many baphy settings. *****')
        globalparams = {}
        globalparams['rawfilename'] = os.path.join(filepath, 'raw')
        globalparams['Tester'] = 'jwingert'
        globalparams['Ferret'] = 'Prince'
        globalparams['runclass'] = runclass

        if parts[2]=='a':
            globalparams['Physiology'] = 'Yes -- Behavior'
        else:
            globalparams['Physiology'] = 'Yes -- Passive'
        globalparams['SiteID'] = siteid

    globalparams['HWSetup'] = 17
    globalparams['date'] = ctime.strftime("%Y-%m-%d")
    globalparams['SiteID'] = siteid
    globalparams['HWparams'] = {'DAQSystem': 'Open-Ephys'}
    globalparams['NumberOfElectrodes'] = 384
    globalparams['stim_system'] = 'psi'
    globalparams['Module'] = 'psi'
    globalparams['ExperimentComplete'] = 1
    globalparams['tempMfile'] = filepath

    T = pd.read_csv(trialfile)
    E = pd.read_csv(eventfile)
    rparms = T.loc[0]
    TrialObject = {1: {
        'ReferenceClass': 'BigNat',
        'ReferenceHandle': {1: {'PreStimSilence': prestimsilence,
                                'PostStimSilence': poststimsilence,
                                'SoundPath': rparms.background_wav_sequence_path,
                                'Duration': rparms.background_wav_sequence_duration,
                                'Normalization': rparms.background_wav_sequence_normalization ,
                                'FixedAmpScale': rparms.background_wav_sequence_norm_fixed_scale ,
                                'fit_range': rparms.background_wav_sequence_fit_range,
                                'fit_reps': rparms.background_wav_sequence_fit_reps,
                                'test_range': rparms.background_wav_sequence_test_range,
                                'test_reps': rparms.background_wav_sequence_test_reps,
                                'iti_duration': rparms.iti_duration,
                            'Names': {},
                            'descriptor': 'BigNat'
                            }},
        'TargetClass': 'Tone',
        'TargetHandle': {1:{ 'descriptor': 'Tone'
                        }},
        'OveralldB': rparms.background_wav_sequence_level}
    }
    exptparams = {'runclass': runclass,
                  'StartTime': ctime.strftime("%H:%M:%S"),
                  'BehaveObjectClass': 'psi-go-nogo',
                  'TrialObject': TrialObject,
                  'TotalRepetitions': 1,
                  'Repetition': 1,
                  }

    event_count = len(E)
    cols = E.columns
    if 'trial' in cols:
        E['Trial'] = E['trial']
        guess_trials = False
    else:
        E['Trial'] = 1
        guess_trials = True
    E['duration'] = 0
    for ee, r in E.iterrows():
        if str(r['info'])!='nan':
            info = json.loads(r['info'])
            if 'duration' in info.keys():
                if ~np.isinf(info['duration']):
                    E.loc[ee,'duration'] = info['duration']

        if r['event'] == 'background_added':
            E_pausenext = E.loc[(E['event']=='background_paused') &
                                (E['timestamp']>r['timestamp']) &
                                (E['timestamp']<r['timestamp']+E.loc[ee,'duration'])]
            if (len(E_pausenext)>0):
                E.loc[ee, 'duration'] = E_pausenext['timestamp'].min()-E.loc[ee, 'timestamp']
                #print(f"{E.loc[ee, 'timestamp']} {E.loc[ee, 'duration']}")
    E['start'] = E['timestamp']
    E['end'] = E['start'] + E['duration']
    E['name'] = E['event']
    E['Info'] = E['info']
    experiment_end = E.loc[E.event=='experiment_end','start'].max()
    if experiment_end==0:
        experiment_end = E.loc[E.event == 'trial_end', 'start'].max()

    E.loc[E.end>experiment_end,'end']=experiment_end

    E.loc[E['Info'].astype(str)=='nan','Info']=''
    E.loc[E['Trial']==0,'Trial']=1

    exptevents=E[['start','end','name','Trial','Info']].copy()

    if guess_trials:
        start_events = exptevents.loc[exptevents['name']=='trial_start']
        for ee in range(len(exptevents)):
            tt=(start_events['start']<=exptevents.loc[ee,'start']).sum()
            if tt>1:
                exptevents.loc[ee,'Trial']=tt

    if 'trial_number' not in T.columns:
        T['trial_number']=np.arange(T.shape[0],dtype=int)+1
    trialstarts = exptevents.loc[exptevents['name']=='trial_start',['Trial','start']].values
    trialstarts2 = T[['trial_number', 'trial_start']].values
    Tlen = trialstarts.shape[0]
    if trialstarts2.shape[0]<trialstarts.shape[0]:
        Tlen=trialstarts2.shape[0]
        exptevents['old_trial']=exptevents['Trial'].copy()
        exptevents['Trial'] = 0
        for t in range(1,Tlen+1):
            mm = np.argmin(np.abs(trialstarts[:,1]-trialstarts2[t-1,1]))
            exptevents.loc[exptevents['old_trial']==(mm+1),'Trial']=t
        for i,r in exptevents.loc[exptevents.Trial==0].iterrows():
            try:
                next_trial = trialstarts[trialstarts[:,1]>r['start'],0].min()
            except:
                next_trial = Tlen
                pass
            print(i, next_trial, r['start'],r['end'], r['name'], r['Info'])
        log.info(f"Trial counts adjusted E: {trialstarts.shape[0]} to T: {trialstarts2.shape[0]} ")
        exptevents = exptevents.drop(columns='old_trial')
        #dmean=np.sum(np.abs(trialstarts[:Tlen,1]-trialstarts2[:Tlen,1]))
        #if dmean > 0:
        #    log.info(f"Trial counts E: {trialstarts.shape[0]} T: {trialstarts2.shape[0]} ")
        #    log.info(f"WARNING!! Truncated start time diff: {dmean} sec")
    elif trialstarts2.shape[0]>trialstarts.shape[0]:
        raise ValueError("Trial log has more trials than event log???")

    if (exptevents['Trial'] > Tlen).sum() > 0:
        log.info('Removing events after last trial')
    exptevents=exptevents.loc[(exptevents['Trial']<=Tlen) & (exptevents['Trial']>0)]

    # DONT adjust timestamps to reference current trial rather than absolute
    # target_events = exptevents.loc[exptevents['name']=='target_start'].copy()

    #for ee, r in target_events.iterrows():
    #    exptevents.loc[exptevents['Trial']==r['Trial'],'start'] -= r['start']
    #    exptevents.loc[exptevents['Trial']==r['Trial'],'end'] -= r['start']

    tstart_events = exptevents.loc[exptevents['name']=='target_start'].reset_index(drop=True)
    tstart_events['name']='TRIALSTART'
    tstart_events['end'] = tstart_events['start']
    tstart_events['Info'] = ''
    tstop_events = tstart_events.copy()
    tstop_events['name'] = 'TRIALSTOP'
    tstop_events['Trial'] -= 1
    tstop_events.loc[tstop_events.Trial==0,['start','end']] = exptevents['end'].max()
    tstop_events.loc[tstop_events.Trial==0,['Trial']] = Tlen
    tstop_events = tstop_events.sort_values(by='start').reset_index(drop=True)

    bg_events = exptevents.loc[exptevents['name']=='background_added'].copy()
    Names = []
    for ee, r in bg_events.iterrows():
        info = json.loads(r['Info'])
        Names.append(f'{info["metadata"]["filename"]}.wav')
        bg_events.loc[ee, 'name'] = f'Stim , {info["metadata"]["filename"]}.wav , Reference'
        bg_events.loc[ee, 'Info'] = ''
    Names = list(set(Names))
    Names.sort()
    TrialObject[1]['ReferenceHandle'][1]['Names'] = Names

    tar_events = exptevents.loc[(exptevents['name'] == 'target_start') &
                                (exptevents['Trial']<=Tlen)].copy()
    for ee, r in tar_events.iterrows():
        trialinfo = T.loc[T['trial_number']==r['Trial']].iloc[0]
        snr = trialinfo['snr']
        target_freq = trialinfo['target_tone_frequency']
        trial_type = trialinfo['trial_type']
        if 'nogo' in trial_type:
            name = f'Stim , {target_freq}+-InfdB , Catch'
        else:
            name = f'Stim , {target_freq}+{snr}dB , Target'
        tar_events.loc[ee, 'name'] = name
        tar_events.loc[ee, 'Info'] = ''
    stimevents = pd.concat([bg_events, tar_events])
    prestimevents=stimevents.copy()
    starts = prestimevents['start'].copy()
    prestimevents['start'] = starts-prestimsilence
    prestimevents['end'] = starts
    prestimevents['name'] = prestimevents['name'].str.replace('Stim ,','PreStimSilence ,')
    poststimevents = stimevents.copy()
    stops = poststimevents['end'].copy()
    poststimevents['start'] = stops
    poststimevents['end'] = stops+poststimsilence
    poststimevents['name'] = poststimevents['name'].str.replace('Stim ,', 'PostStimSilence ,')

    trial_number = T['trial_number']
    videostart = T['psivideo_frame_ts']
    videostart -= videostart[0]
    videoframes = T['psivideo_frames_written']
    videoname = videoframes.apply(lambda x: f"PSIVIDEO,{x:.0f}")
    d_ = {'start': videostart, 'end': videostart, 'name': videoname,
          'Trial': trial_number, 'Info': ''}
    videoevents = pd.DataFrame(d_)

    response_ts = T['response_ts']
    response_outcome = T['score']
    response_name = response_outcome.apply(lambda x: f"LICK , {x}")

    d_ = {'start': response_ts, 'end': response_ts, 'name': response_name,
          'Trial': trial_number, 'Info': ''}
    responseevents = pd.DataFrame(d_)
    responseevents = responseevents.loc[np.isfinite(response_ts)]

    trial_outcomes = response_outcome.apply(lambda x: f"{x}_TRIAL")
    n_outcomes = len(trial_outcomes)
    d_ = {'start': tstart_events['start'][:n_outcomes],
          'end': tstop_events['start'][:n_outcomes],
          'name': trial_outcomes, 'Trial': trial_number, 'Info': ''}
    outcomeevents = pd.DataFrame(d_)

    exptevents = pd.concat([exptevents, tstart_events, tstop_events, stimevents,
                            prestimevents, poststimevents, videoevents,
                            responseevents, outcomeevents], ignore_index=True)
    exptevents = exptevents.sort_values(by=['Trial','start']).reset_index(drop=True)

    globalparams['rawfilecount'] = Tlen

    return globalparams, exptparams, exptevents

def baphy_parm_read(filepath, evpread=True):
    log.info("Loading {0}".format(filepath))

    if os.path.isdir(filepath):
        return psi_parm_read(filepath)

    s = load_resource(str(filepath))
    if type(s) is str:
        s = s.split("\n")
    # f = io.open(filepath, "r")
    # s = f.readlines(-1)

    globalparams = {}
    exptparams = {}
    exptevents = {}
    for ts in s:
        # print(ts)
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
            import pdb;
            pdb.set_trace()
        except SyntaxError:
            log.info("SyntaxError parsing this baphy config line: {0}".format(sout))
            # import pdb; pdb.set_trace()
        except Exception as e:
            log.info("Other error on: {0} to {1}".format(ts, sout))
            import pdb;
            pdb.set_trace()

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
    # import pdb
    # pdb.set_trace()
    exptevents.columns = ['name', 'start', 'end', 'Trial']
    for i in range(len(exptevents)):
        if exptevents.loc[i, 'end'] == []:
            exptevents.loc[i, 'end'] = exptevents.loc[i, 'start']

    if evpread:
        try:
            # get lick events from evp file
            evpfile = Path(filepath).with_suffix('.evp')
            lick_events = get_lick_events(evpfile, name='LICK')
            log.info("evp file for licks: %s", evpfile)

            # add evp lick events, delete baphy lick events
            exptevents = exptevents[~(exptevents.name == 'LICK')]
            exptevents = pd.concat([exptevents, lick_events], ignore_index=True)
        except:
            log.info("Failed loading evp file. Still zipped?")

    if 'ReferenceClass' not in exptparams['TrialObject'][1].keys():
        exptparams['TrialObject'][1]['ReferenceClass'] = \
            exptparams['TrialObject'][1]['ReferenceHandle'][1]['descriptor']
    # CPP special case, deletes added commas ToDo this might be unneccesary, the task is done in MLE code.
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


def baphy_convert_user_definable_fields(x):
    '''
    Converts all occurances of the `'UserDefinableFields'` list to a
    dictionary. This is recursive, so it will scan the full dataset returned by
    `baphy_parm_read`.

    Example
    ------
    >>> data = {
            'descriptor': 'NoiseSample',
            'UserDefinableFields': ['PreStimSilence', 'edit', 0,
                                    'PostStimSilence', 'edit', 0,
                                    'Duration', 'edit', 0.3,]
        }

    >>> baphy_convert_user_definable_fields(data)
    >>> print(data)
    {'descriptor': 'NoiseSample',
     'UserDefinableFields': {'PreStimSilence': 0, 'PostStimSilence': 0, 'Duration': 0.3}
    '''
    if isinstance(x, dict) and 'UserDefinableFields' in x:
        userdef = x.pop('UserDefinableFields')
        keys = userdef[::3]
        values = userdef[2::3]
        x['UserDefinableFields'] = dict(zip(keys, values))
    if isinstance(x, dict):
        for v in x.values():
            baphy_convert_user_definable_fields(v)
    elif isinstance(x, (tuple, list)):
        for v in x:
            baphy_convert_user_definable_fields(v)


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
    if options['pupil'] or options['rem']:
        options = set_default_pupil_options(options)

    # options['pupil_deblink'] = int(options.get('pupil_deblink', 1))
    # options['pupil_deblink_dur'] = options.get('pupil_deblink_dur', 1)
    # options['pupil_median'] = options.get('pupil_median', 0)
    # options["pupil_offset"] = options.get('pupil_offset', 0.75)
    options['raw'] = int(options.get('raw', False))
    options['mua'] = int(options.get('mua', False))
    options['resp'] = int(options.get('resp', True))
    options['stim'] = int(options.get('stim', True))
    options['runclass'] = options.get('runclass', None)
    options['rawid'] = options.get('rawid', None)
    options['facemap'] = options.get('facemap', False)
    options['facepca'] = options.get('facepca', False)

    if options['stimfmt'] in ['envelope', 'parm']:
        log.info("Setting chancount=0 for stimfmt=%s", options['stimfmt'])
        options['chancount'] = 0

    return options


def parse_loadkey(loadkey=None, batch=None, siteid=None, cellid=None,
                  **options):
    """
    :param loadkey:  nems load string (eg, "ozgf.fs100.ch18")
    :param options:  pre-loaded options, will be overwritten by loadkey contents
    :return: options dictionary
    """

    options = fill_default_options(options)

    # remove any preprocessing keywords in the loader string.
    if '-' in loadkey:
        loader = nems0.utils.escaped_split(loadkey, '-')[0]
    else:
        loader = loadkey
    log.info('loader=%s', loader)

    ops = loader.split(".")

    # updates some some defaults

    options.update({'rasterfs': 100, 'chancount': 0})
    if ops[0] in ['nostim', 'psth', 'ns', 'evt']:
        options.update({'stim': False, 'stimfmt': 'parm'})
    else:
        options['stimfmt'] = ops[0]
    if options['stimfmt'] == 'env':
        options['stimfmt'] = 'envelope'
    # computed, but not saved anywhere?
    load_pop_file = ("pop" in ops)

    for op in ops[1:]:
        if op.startswith('fs'):
            options['rasterfs'] = int(op[2:])
        elif op.startswith('ch'):
            options['chancount'] = int(op[2:])

        elif op.startswith('fmap'):
            options['facemap'] = int(op[4:])

        elif op.startswith('fpca'):
            if len(op[4:]):
                options['facepca'] = int(op[4:])
            else:
                options['facepca'] = 1

        elif op == 'pup':
            options.update({'pupil': True, 'rem': 1})
            # options.update({'pupil': True, 'pupil_deblink': True,
            #                'pupil_deblink_dur': 1,
            #                'pupil_median': 0, 'rem': 1})
        elif op == 'dlc':
            options.update({'dlc': True})
        elif op == 'mono':
            options.update({'mono': True})
        elif op == 'rem':
            options['rem'] = True

        elif 'eysp' in ops:
            options['pupil_eyespeed'] = True
        elif op == 'voc':
            options.update({'runclass': 'VOC'})
        elif op == 'bin':
            options.update({'binaural': 'crude'})
        elif op == 'bin0':
            options.update({'binaural': 'crude', 'binsplit': False})

    if 'stimfmt' not in options.keys():
        raise ValueError('Valid stim format (ozgf, gtgram, psth, parm, env, evt) not specified in loader=' + loader)
    if (options['stimfmt'] == 'ozgf') and (options['chancount'] <= 0):
        raise ValueError('Stim format ozgf requires chancount>0 (.chNN) in loader=' + loader)

    # these fields are now optional (vs. xform_wrappers)
    if siteid is not None:
        options['siteid'] = siteid

    if batch is not None:
        options["batch"] = batch
        if int(batch) in [263, 294]:
            options["runclass"] = "VOC"

    if cellid is not None:
        options["cellid"] = cellid

    return options


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
    if 'use_target' in options:
        if options['use_target']:
            RefObject = exptparams['TrialObject'][1]['TargetHandle'][1]
        else:
            RefObject = exptparams['TrialObject'][1]['ReferenceHandle'][1]
    elif options['runclass'] is None:
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
            dstr += '-' + exptparams['TrialObject'][1]['RunClass']
        else:
            dstr += '-TOR'

    # include all parameter values, even defaults, in filename
    fields = RefObject['UserDefinableFields']
    if options['stimfmt'] == 'envelope':
        x_these_fields = ['F0s', 'ComponentsNumber'];
    else:
        x_these_fields = [];

    for cnt1 in range(0, len(fields), 3):
        if RefObject[fields[cnt1]] == 0:
            RefObject[fields[cnt1]] = int(0)
            # print(fields[cnt1])
            # print(RefObject[fields[cnt1]])
            # print(dstr)
        if fields[cnt1] in x_these_fields:
            if type(RefObject[fields[cnt1]]) is int:
                l = ['X']
            else:
                l = ['X' for i in range(len(RefObject[fields[cnt1]]))]
            dstr = "{0}-{1}".format(dstr, "__".join(l))
        else:
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

    if len(dstr) > 250:
        dhash = hashlib.sha1(dstr.encode('ascii')).hexdigest()
        dstr = dstr[:200] + '_' + dhash

    filepath = stim_cache_dir + dstr + '.mat'

    return filepath


def parm_tbp(exptparams, **options):
    """
    generate parameterized spectrograms for TBP ref/tar stimuli in stim_dict format

    :param exptparams:
    :param options:
    :return:
    """
    ref = exptparams['TrialObject'][1]['ReferenceHandle'][1]
    tar = exptparams['TrialObject'][1]['TargetHandle'][1]

    ref_names = ref['Names']
    tar_names = tar['Names']
    if tar['descriptor'] == 'ToneInNoise':
        tar_tone_names = tar['Tone'][1]['Names']
        tar_noise_bands = np.array(tar['ToneBands']) - 1
        tar_fixed_band = tar['ToneFixedBand']
        if len(tar_fixed_band) == 0:
            tar_tone_bands = tar_noise_bands
        else:
            tar_tone_bands = [int(tar_fixed_band) - 1] * len(tar_noise_bands)

        # _, tar_tone_channels = np.unique(tar_tone_bands, return_index=True)
        # assume there's only one target tone frequency!
        tar_tone_channels = np.full_like(tar_tone_bands, 0)

        tar_snrs = tar['SNRs']
        # import pdb; pdb.set_trace()
    elif tar['descriptor'] == 'Tone':
        # import pdb;
        # pdb.set_trace()
        tar_tone_names = tar['Names']
        tar_noise_bands = np.arange(len(tar_tone_names))
        tar_tone_bands = np.arange(len(tar_tone_names))
        tar_tone_channels = tar_tone_bands.copy()
        tar_snrs = np.full(len(tar_tone_names), np.inf)

    else:
        raise ValueError(f"Unsupported TargetClass {tar['descriptor']}")

    stim_dict = {}
    total_bands = len(ref_names) + len(set(tar_tone_bands))
    fs = options['rasterfs']
    prebins = int(fs * ref['PreStimSilence'])
    durbins = int(fs * ref['Duration'])
    postbins = int(fs * ref['PostStimSilence'])
    total_bins = prebins + durbins + postbins
    for i, r in enumerate(ref_names):
        s = np.zeros((total_bands, total_bins))
        s[i, prebins] = 1
        stim_dict[r] = s
    for i, t in enumerate(tar_names):
        s = np.zeros((total_bands, total_bins))
        if np.isfinite(tar_snrs[i]):
            s[tar_noise_bands[i], prebins] = 1
            s[len(ref_names) + tar_tone_channels[i], prebins] = 10 ** (tar_snrs[i] / 20)
        elif tar_snrs[i] > 0:
            s[len(ref_names) + tar_tone_channels[i], prebins] = 1
        else:
            s[tar_noise_bands[i], prebins] = 1
        stim_dict[t] = s
    tags = list(stim_dict.keys())
    stimparam = {'chans': ref_names + list(set(tar_tone_names))}

    return stim_dict, tags, stimparam


def labeled_line_stim(exptparams, **options):
    """
    generate parameterized "spectrogram" of stimulus where onset of each unique stim/tar/ref/cat event
    is coded in each row

    :param exptparams:
    :param options:
    :return:
    """
    ref = exptparams['TrialObject'][1]['ReferenceHandle'][1]
    tar = exptparams['TrialObject'][1]['TargetHandle'][1]

    ref_names = ref['Names']
    tar_names = tar['Names']
    all_names = ref_names + tar_names

    stim_dict = {}
    total_bands = len(ref_names) + len(tar_names)
    fs = options['rasterfs']
    prebins = int(fs * ref['PreStimSilence'])
    durbins = int(fs * ref['Duration'])
    postbins = int(fs * ref['PostStimSilence'])
    total_bins = prebins + durbins + postbins
    for i, r in enumerate(all_names):
        if i == 0:
            prebins = int(fs * ref['PreStimSilence'])
            durbins = int(fs * ref['Duration'])
            postbins = int(fs * ref['PostStimSilence'])
            total_bins = prebins + durbins + postbins
        elif i == len(ref_names):
            # shift to using tar lengths
            prebins = int(fs * tar['PreStimSilence'])
            durbins = int(fs * tar['Duration'])
            postbins = int(fs * tar['PostStimSilence'])
            total_bins = prebins + durbins + postbins

        s = np.zeros((total_bands, total_bins))
        s[i, prebins] = 1
        stim_dict[r] = s
    tags = list(stim_dict.keys())
    stimparam = {'chans': all_names}

    return stim_dict, tags, stimparam


def baphy_load_stim(exptparams, parmfilepath, epochs=None, **options):

    if options['stimfmt'] in ['gtgram', 'nenv', 'lenv']:
        # &(exptparams['TrialObject'][1]['ReferenceClass'] == 'BigNat'):

        stim, tags, stimparam = runclass.NAT_stim(epochs, exptparams, **options)

    elif (options['stimfmt'] == 'parm') & exptparams['TrialObject'][1]['ReferenceClass'].startswith('Torc'):
        import nems_lbhb.strf.torc_subfunctions as tsf
        TorcObject = exptparams['TrialObject'][1]['ReferenceHandle'][1]
        stim, tags, stimparam = tsf.generate_torc_spectrograms(
            TorcObject, rasterfs=options['rasterfs'], single_cycle=False)
        # adjust so that all power is >0
        for k in stim.keys():
            stim[k] = stim[k] + 5

        # NB stim is a dict rather than a 3-d array

    elif (options['stimfmt'] == 'parm') & \
            (exptparams['TrialObject'][1]['ReferenceClass'] == 'NoiseBurst'):

        # NB stim is a dict rather than a 3-d array
        stim, tags, stimparam = parm_tbp(exptparams, **options)

    elif (options['stimfmt'] == 'll'):

        # NB stim is a dict rather than a 3-d array
        stim, tags, stimparam = labeled_line_stim(exptparams, **options)

    elif exptparams['runclass'] == 'VOC_VOC':
        stimfilepath1 = baphy_stim_cachefile(exptparams, parmfilepath, use_target=False, **options)
        stimfilepath2 = baphy_stim_cachefile(exptparams, parmfilepath, use_target=True, **options)
        log.info("Cached stim: {0}, {1}".format(stimfilepath1, stimfilepath2))
        # load stimulus spectrogram
        stim1, tags1, stimparam1 = baphy_load_specgram(stimfilepath1)
        stim2, tags2, stimparam2 = baphy_load_specgram(stimfilepath2)
        stim = np.concatenate((stim1, stim2), axis=2)
        if exptparams['TrialObject'][1]['ReferenceHandle'][1]['SNR'] >= 100:
            t2 = [t + '_0dB' for t in tags2]
            tags = np.concatenate((tags1, t2))
            eventmatch = 'Reference1'
        else:
            t1 = [t + '_0dB' for t in tags1]
            tags = np.concatenate((t1, tags2))
            eventmatch = 'Reference2'
        # import pdb
        # pdb.set_trace()
        for i in range(len(exptevents)):
            if eventmatch in exptevents.loc[i, 'name']:
                exptevents.loc[i, 'name'] = exptevents.loc[i, 'name'].replace('.wav', '.wav_0dB')
                exptevents.loc[i, 'name'] = exptevents.loc[i, 'name'].replace('Reference1', 'Reference')
                exptevents.loc[i, 'name'] = exptevents.loc[i, 'name'].replace('Reference2', 'Reference')

        stimparam = stimparam1
    else:
        stimfilepath = baphy_stim_cachefile(exptparams, parmfilepath, **options)
        log.info("Cached stim: {0}".format(stimfilepath))
        # load stimulus spectrogram
        stim, tags, stimparam = baphy_load_specgram(stimfilepath)

    if options["stimfmt"] == 'envelope' and \
            exptparams['TrialObject'][1]['ReferenceClass'] == 'SSA':
        # SSA special case
        stimo = stim.copy()
        maxval = np.max(np.reshape(stimo, [2, -1]), axis=1)
        log.info('special case for SSA stim!')
        ref = exptparams['TrialObject'][1]['ReferenceHandle'][1]
        stimlen = ref['PipDuration'] + ref['PipInterval']
        stimbins = int(stimlen * options['rasterfs'])

        stim = np.zeros([2, stimbins, 6])
        prebins = int(ref['PipInterval'] / 2 * options['rasterfs'])
        durbins = int(ref['PipDuration'] * options['rasterfs'])
        stim[0, prebins:(prebins + durbins), 0:3] = maxval[0]
        stim[1, prebins:(prebins + durbins), 3:] = maxval[1]
        tags = ["{}+ONSET".format(ref['Frequencies'][0]),
                "{}+{:.2f}".format(ref['Frequencies'][0], ref['F1Rates'][0]),
                "{}+{:.2f}".format(ref['Frequencies'][0], ref['F1Rates'][1]),
                "{}+ONSET".format(ref['Frequencies'][1]),
                "{}+{:.2f}".format(ref['Frequencies'][1], ref['F1Rates'][0]),
                "{}+{:.2f}".format(ref['Frequencies'][1], ref['F1Rates'][1])]

    snr_suff = ""
    if 'SNR' in exptparams['TrialObject'][1]['ReferenceHandle'][1].keys():
        SNR = exptparams['TrialObject'][1]['ReferenceHandle'][1]['SNR']
        if SNR < 100:
            log.info('Noisy stimulus (SNR<100), appending tag to epoch names')
            snr_suff = "_{}dB".format(SNR)

    if exptparams['runclass'] == 'CPN':
        # clean up NTI sequence tags
        # import pdb; pdb.set_trace()
        # sequence001:5-6-2-3-5
        tags = ["-".join(t.split("  ")).replace(" ", "") if t.startswith("sequence") else t for t in tags]

    if (epochs is not None):
        # additional processing steps to convert stim into a dictionary with keys that match epoch names
        # specific to BAPHYExperiment loader.

        if (type(stim) is not dict):
            stim_dict = {}
            for eventidx in range(0, len(tags)):
                # save stimulus for this event as separate dictionary entry
                stim_dict["STIM_" + tags[eventidx] + snr_suff] = stim[:, :, eventidx]
            stim = stim_dict

        if (type(stim) is dict):
            keys = list(stim.keys())
            new_stim = {}
            new_keys = []
            for k in keys:
                matches = list(set(epochs[epochs.name.str.endswith(k)].name.values))
                for nk in matches:
                    new_stim[nk] = stim[k]
            stim = new_stim

    # stim_dict = {}
    # for eventidx in range(0, len(tags)):
    #    # save stimulus for this event as separate dictionary entry
    #    if type(stim) is dict:
    #        stim_dict["STIM_" + tags[eventidx] + snr_suff] = stim[tags[eventidx]]
    #    else:
    #        stim_dict["STIM_" + tags[eventidx] + snr_suff] = stim[:, :, eventidx]

    return stim, tags, stimparam


def baphy_load_spike_data_raw(spkfilepath, channel=None, unit=None):
    matdata = scipy.io.loadmat(spkfilepath, chars_as_strings=True)

    sortinfo = matdata['sortinfo']
    if sortinfo.shape[0] > 1:
        sortinfo = sortinfo.T
    sortinfo = sortinfo[0]
    spikedata = {}
    spikedata['sortinfo'] = sortinfo
    # figure out sampling rate, used to convert spike times into seconds
    spikedata['spikefs'] = matdata['rate'][0][0]
    if 'baphy_fmt' in matdata:
        spikedata['baphy_fmt'] = matdata['baphy_fmt']
    else:
        spikedata['baphy_fmt'] = 1

    return spikedata


def baphy_align_time_BAD(exptevents, sortinfo, spikefs, finalfs=0):
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
        (np.zeros([1, 1]), TrialLen_sec[:, np.newaxis] * spikefs), axis=0
    )

    for c in range(0, chancount):
        if len(sortinfo[c]) and sortinfo[c][0].size:
            s = sortinfo[c][0][0]['unitSpikes']
            s = np.reshape(s, (-1, 1))
            unitcount = s.shape[0]
            for u in range(0, unitcount):
                st = s[u, 0]

                # print('chan {0} unit {1}: {2} spikes'.format(c,u,st.shape[1]))
                for trialidx in range(1, TrialCount + 1):
                    ff = (st[0, :] == trialidx)
                    if np.sum(ff):
                        utrial_spikefs = np.max(st[1, ff])
                        TrialLen_spikefs[trialidx, 0] = np.max(
                            [utrial_spikefs, TrialLen_spikefs[trialidx, 0]]
                        )

    # using the trial lengths, figure out adjustments to trial event times.
    if finalfs:
        log.debug('rounding Trial offset spike times to even number of rasterfs bins')
        # print(TrialLen_spikefs)
        TrialLen_spikefs = np.ceil(TrialLen_spikefs / spikefs * finalfs) / finalfs * spikefs
    Offset_spikefs = np.cumsum(TrialLen_spikefs)
    Offset_sec = Offset_spikefs / spikefs  # how much to offset each trial

    # adjust times in exptevents to approximate time since experiment started
    # rather than time since trial started (native format)
    for Trialidx in range(1, TrialCount + 1):
        ff = (exptevents['Trial'] == Trialidx)
        exptevents.loc[ff, ['start', 'end']] = (
                exptevents.loc[ff, ['start', 'end']] + Offset_sec[Trialidx - 1]
        )

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
            log.info('Comment: %s', comment)

            s = np.reshape(s, (-1, 1))
            unitcount = s.shape[0]
            for u in range(0, unitcount):
                st = s[u, 0]
                uniquetrials = np.unique(st[0, :])

                unit_spike_events = np.array([])
                for trialidx in uniquetrials:
                    ff = (st[0, :] == trialidx)
                    this_spike_events = (st[1, ff]
                                         + Offset_spikefs[np.int(trialidx - 1)])
                    if (comment != []):
                        if (comment == 'PC-cluster sorted by mespca.m'):
                            # remove last spike, which is stray
                            this_spike_events = this_spike_events[:-1]
                    unit_spike_events = np.concatenate(
                        (unit_spike_events, this_spike_events), axis=0
                    )

                totalunits += 1
                if chancount <= 8:
                    unit_names.append("{0}{1}".format(chan_names[c], u + 1))
                else:
                    unit_names.append("{0:02d}-{1}".format(c + 1, u + 1))
                spiketimes.append(unit_spike_events / spikefs)

    return exptevents, spiketimes, unit_names


def baphy_align_time(exptevents, sortinfo, spikefs, finalfs=0, sortidx=0):
    # number of channels in recording (not all necessarily contain spikes)
    chancount = len(sortinfo)
    while chancount > 1 and sortinfo[chancount - 1].size == 0:
        chancount -= 1
    # figure out how long each trial is by the time of the last spike count.
    # this method is a hack!
    # but since recordings are longer than the "official"
    # trial end time reported by baphy, this method preserves extra spikes
    TrialCount = int(np.max(exptevents['Trial']))

    hit_trials = exptevents[exptevents.name == "BEHAVIOR,PUMPON,Pump"].Trial
    max_event_times = exptevents.groupby('Trial')['end'].max().values
    # import pdb; pdb.set_trace()
    TrialStart_sec = np.array(
        exptevents.loc[exptevents['name'] == "TRIALSTART"]['start']
    )
    TrialLen_sec = np.array(
        exptevents.loc[exptevents['name'] == "TRIALSTOP"]['start']
    )
    if TrialStart_sec.sum()>0:
        TrialLen_sec -= TrialStart_sec
        offset_exists = True
    else:
        offset_exists = False

    if len(hit_trials):
        TrialLen_sec[hit_trials - 1] = max_event_times[hit_trials - 1]

    TrialLen_spikefs = np.concatenate(
        (np.zeros([1, 1]), TrialLen_sec[:, np.newaxis] * spikefs), axis=0
    )
    if ~offset_exists:
        for ch in range(0, chancount):
            if len(sortinfo[ch]) and len(sortinfo[ch][0]) >= sortidx + 1 and sortinfo[ch][0][sortidx].size:
                s = sortinfo[ch][0][sortidx]['unitSpikes']
                s = np.reshape(s, (-1, 1))
                unitcount = s.shape[0]
                for u in range(0, unitcount):
                    st = s[u, 0]

                    # print('chan {0} unit {1}: {2} spikes'.format(c,u,st.shape[1]))
                    for trialidx in range(1, TrialCount + 1):
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
                np.ceil(TrialLen_spikefs / spikefs * finalfs) / finalfs * spikefs
        )
        # TrialLen_spikefs = (
        #        np.ceil(TrialLen_spikefs / spikefs*finalfs + 1) / finalfs*spikefs
        #        )
        # print(TrialLen_spikefs)

    Offset_spikefs = np.cumsum(TrialLen_spikefs)
    Offset_sec = Offset_spikefs / spikefs  # how much to offset each trial
    # adjust times in exptevents to approximate time since experiment started
    # rather than time since trial started (native format)
    for Trialidx in range(1, TrialCount + 1):
        # print("Adjusting trial {0} by {1} sec"
        #       .format(Trialidx,Offset_sec[Trialidx-1]))
        ff = (exptevents['Trial'] == Trialidx)
        if offset_exists:
            s_ = TrialStart_sec[Trialidx-1]
            s_fs = np.ceil(s_ * finalfs) / finalfs
            exptevents.loc[ff, ['start', 'end']] = \
                    exptevents.loc[ff, ['start', 'end']] - s_ + s_fs
            Offset_spikefs[Trialidx - 1] = s_fs * spikefs
        else:
            exptevents.loc[ff, ['start', 'end']] = (
                    exptevents.loc[ff, ['start', 'end']] + Offset_sec[Trialidx - 1]
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
        if len(sortinfo[c]) and len(sortinfo[c][0]) >= sortidx + 1 and sortinfo[c][0][sortidx].size:
            s = sortinfo[c][0][sortidx]['unitSpikes']
            comment = sortinfo[c][0][sortidx][0][0][2][0]
            log.debug('Comment: %s', comment)

            s = np.reshape(s, (-1, 1))
            unitcount = s.shape[0]
            for u in range(0, unitcount):
                st = s[u, 0]
                # if st.size:
                if st.shape[0] == 3:
                    log.debug("{} {}".format(u, str(st.shape)))
                    uniquetrials = np.unique(st[0, :])
                    # print('chan {0} unit {1}: {2} spikes {3} trials'
                    #       .format(c, u, st.shape[1], len(uniquetrials)))

                    unit_spike_events = np.array([])
                    for trialidx in range(1, TrialCount + 1):
                        ff = (st[0, :] == trialidx)
                        try:
                            this_spike_events = (st[1, ff]
                                                 + Offset_spikefs[np.int(trialidx - 1)])
                        except:
                            import pdb
                            pdb.set_trace()
                        if len(comment) > 0:
                            if comment == 'PC-cluster sorted by mespca.m':
                                # remove last spike, which is stray
                                this_spike_events = this_spike_events[:-1]
                        unit_spike_events = np.concatenate(
                            (unit_spike_events, this_spike_events), axis=0
                        )
                        # print("   trial {0} first spike bin {1}"
                        #       .format(trialidx,st[1,ff]))

                    totalunits += 1
                    if chancount <= 8:
                        # svd -- avoid letter channel names from now on?
                        # unit_names.append("{0}{1}".format(chan_names[c], u+1))
                        unit_names.append("{0:02d}-{1}".format(c + 1, u + 1))
                    else:
                        unit_names.append("{0:02d}-{1}".format(c + 1, u + 1))
                    spiketimes.append(unit_spike_events / spikefs)

                # else:
                # TODO - Incorporate this, but deal with cases of truly missing data. This is
                # designed only for cases where e.g. a single cellid doens't spike during one
                # of many files (for example during a passive), not cases where the cellid
                # is just missed (like cases due to append units)
                #    # append empty list for units that had no spikes
                #    if chancount <= 8:
                #        unit_names.append("{0}{1}".format(chan_names[c], u+1))
                #    else:
                #        unit_names.append("{0:02d}-{1}".format(c+1, u+1))
                #    spiketimes.append([])
    return exptevents, spiketimes, unit_names


def baphy_align_time_baphyparm(exptevents, finalfs=0, **options):
    TrialCount = int(np.max(exptevents['Trial']))

    TrialStarts = exptevents.loc[exptevents['name'].str.startswith("TRIALSTART")]['name']

    def _get_start_time(x):
        d = x.split(",")
        if len(d) > 2:
            time = datetime.datetime.strptime(d[1].strip() + " " + d[2], '%Y-%m-%d %H:%M:%S.%f')
        else:
            time = datetime.datetime(2000, 1, 1)
        return time

    def _get_time_diff_seconds(x):
        d = x.split(",")
        if len(d) > 2:
            time = datetime.datetime.strptime(d[1].strip() + " " + d[2], '%Y-%m-%d %H:%M:%S.%f')
        else:
            time = datetime.datetime(2000, 1, 1)
        return time

    TrialStartDateTime = TrialStarts.apply(_get_start_time)

    # time first trial started, all epoch times will be measured in seconds from this time
    timezero = TrialStartDateTime.iloc[0]

    def _get_time_diff_seconds(x, timezero=0):

        return (x - timezero).total_seconds()

    TrialStartSeconds = TrialStartDateTime.apply(_get_time_diff_seconds, timezero=timezero)

    Offset_sec = TrialStartSeconds.values

    if np.sum(Offset_sec) == 0:
        log.info('No timestamps in baphy events, inferring trial times from durations')
        Offset_sec = exptevents.loc[exptevents.name == 'TRIALSTOP', 'start'].values
        Offset_sec = np.roll(Offset_sec, 1)
        Offset_sec[0] = 0
        Offset_sec = np.cumsum(Offset_sec)

    exptevents['start'] = exptevents['start'].astype(float)
    exptevents['end'] = exptevents['end'].astype(float)

    # adjust times in exptevents to approximate time since experiment started
    # rather than time since trial started (native format)
    for Trialidx in range(1, TrialCount + 1):
        # print("Adjusting trial {0} by {1} sec"
        #       .format(Trialidx,Offset_sec[Trialidx-1]))
        ff = (exptevents['Trial'] == Trialidx)
        exptevents.loc[ff, ['start', 'end']] = (
                exptevents.loc[ff, ['start', 'end']] + Offset_sec[Trialidx - 1]
        )

    if finalfs:
        exptevents['start'] = np.round(exptevents['start'] * finalfs) / finalfs
        exptevents['end'] = np.round(exptevents['end'] * finalfs) / finalfs

    log.info("{0} trials totaling {1:.2f} sec".format(TrialCount, Offset_sec[-1]))

    return exptevents


def set_default_pupil_options(options):
    options = options.copy()
    options["rasterfs"] = options.get('rasterfs', 100)
    options['pupil'] = options.get('pupil', 1)
    options['pupil_analysis_method'] = options.get('pupil_analysis_method', 'cnn')  # or 'matlab'
    if options['pupil_analysis_method'] == 'cnn':
        options['pupil_variable_name'] = options.get('pupil_variable_name', 'area')
    else:
        options['pupil_variable_name'] = options.get('pupil_variable_name', 'minor_axis')
    options["pupil_offset"] = options.get('pupil_offset', 0.75)
    options["pupil_deblink"] = options.get('pupil_deblink', True)
    options["pupil_deblink_dur"] = options.get('pupil_deblink_dur', 1)
    options["pupil_median"] = options.get('pupil_median', 0)
    options["pupil_smooth"] = options.get('pupil_smooth', 0)
    options["pupil_highpass"] = options.get('pupil_highpass', 0)
    options["pupil_lowpass"] = options.get('pupil_lowpass', 0)
    options["pupil_bandpass"] = options.get('pupil_bandpass', 0)
    options["pupil_derivative"] = options.get('pupil_derivative', '')
    options["pupil_mm"] = options.get('pupil_mm', False)
    options["rem"] = options.get('rem', True)
    options["rem_units"] = options.get('rem_units', 'mm')
    options["rem_min_pupil"] = options.get('rem_min_pupil', 0.2)
    options["rem_max_pupil"] = options.get('rem_max_pupil', 1)
    options["rem_max_pupil_sd"] = options.get('rem_max_pupil_sd', 0.05)
    options["rem_min_saccade_speed"] = options.get('rem_min_saccade_speed', 0.5)
    options["rem_min_saccades_per_minute"] = options.get('rem_min_saccades_per_minute', 0.01)
    options["rem_max_gap_s"] = options.get('rem_max_gap_s', 15)
    options["rem_min_episode_s"] = options.get('rem_min_episode_s', 30)
    options["pupil_artifacts"] = options.get("pupil_artifacts",
                                             False)  # load boolean signal indicating "bad" pupil chunks
    options["verbose"] = options.get('verbose', False)

    return options


def load_pupil_trace(pupilfilepath, exptevents=None, **options):
    """
    returns big_rs which is pupil trace resampled to options['rasterfs']
    and strialidx, which is the index into big_rs for the start of each
    trial. need to make sure the big_rs vector aligns with the other signals
    """

    options = set_default_pupil_options(options)

    pupilfilepath = get_pupil_file(pupilfilepath, **options)

    rasterfs = options["rasterfs"]
    pupil_offset = options["pupil_offset"]
    pupil_deblink = options["pupil_deblink"]
    pupil_deblink_dur = options["pupil_deblink_dur"]
    pupil_median = options["pupil_median"]
    pupil_mm = options["pupil_mm"]
    verbose = options["verbose"]
    options['pupil'] = options.get('pupil', True)

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
        from nems_lbhb.baphy_experiment import BAPHYExperiment

        experiment = BAPHYExperiment.from_pupilfile(pupilfilepath)
        trial_starts = experiment.get_trial_starts()
        exptevents = experiment.get_baphy_events()

        # parmfilepath = pupilfilepath.replace(".pup.mat",".m")
        # globalparams, exptparams, exptevents = baphy_parm_read(parmfilepath)
        # pp, bb = os.path.split(parmfilepath)
        # spkfilepath = pp + '/' + spk_subdir + re.sub(r"\.m$", ".spk.mat", bb)
        # log.info("Spike file: {0}".format(spkfilepath))
        ## load spike times
        # sortinfo, spikefs = baphy_load_spike_data_raw(spkfilepath)
        ## adjust spike and event times to be in seconds since experiment started
        # exptevents, spiketimes, unit_names = baphy_align_time(
        #        exptevents, sortinfo, spikefs, rasterfs
        #        )

    loading_pcs = 0
    if 'SVD.pickle' in pupilfilepath:
        loading_pcs = options.get('facemap', 0)

        log.info("SVD.pickle file, assuming single matrix: %s", pupilfilepath)
        with open(pupilfilepath, 'rb') as fp:
            pupildata = pickle.load(fp)

        pupil_diameter = pupildata[:, :loading_pcs]

        log.info("pupil_diameter.shape: %s", str(pupildata.shape))
        log.info("keeping %d channels: ", loading_pcs)

    elif '.pickle' in pupilfilepath:
        with open(pupilfilepath, 'rb') as fp:
            pupildata = pickle.load(fp)

        # hard code to use minor axis for now
        options['pupil_variable_name'] = options.get('pupil_variable_name', 'minor_axis')
        log.info("Using pupil_variable_name: %s", options['pupil_variable_name'])
        log.info("Using CNN results for pupiltrace")
        if options['pupil_variable_name'] == 'minor_axis':
            pupil_diameter = pupildata['cnn']['a'] * 2
        elif options['pupil_variable_name'] == 'major_axis':
            pupil_diameter = pupildata['cnn']['b'] * 2
        elif options['pupil_variable_name'] == 'area':
            pupil_diameter = np.pi * pupildata['cnn']['b'] * pupildata['cnn']['a']
        else:
            raise ValueError(f"Pupil variable name {options['pupil_variable_name']} is unknown")
        # missing frames/frames that couldn't be decoded were saved as nans
        # pad them here
        nan_args = np.argwhere(np.isnan(pupil_diameter))

        for arg in nan_args:
            arg = arg[0]
            log.info("padding missing pupil frame {0} with adjacent ellipse params".format(arg))
            try:
                pupil_diameter[arg] = pupil_diameter[arg - 1]
            except:
                pupil_diameter[arg] = pupil_diameter[arg - 1]

        pupil_diameter = pupil_diameter[:-1, np.newaxis]
        # figure out which extra pupil signals to load
        # keep these in a signal called pupil_extras where
        # each channel name corresponds to one of these signals
        pupil_extras_keys = ['excluded_frames', 'eyespeed',
                             'eyelid_left_x', 'eyelid_left_y',
                             'eyelid_top_x', 'eyelid_top_y',
                             'eyelid_right_x', 'eyelid_right_y',
                             'eyelid_bottom_x', 'eyelid_bottom_y']
        pupil_extras = {}
        for k in pupil_extras_keys:
            if k in pupildata['cnn'].keys():
                log.info(f"Found extra pupil signal: {k}, loading into signal pupil extras")
                if k == 'excluded_frames':
                    # special case here, because not a time course
                    artifacts = np.zeros(pupil_diameter.shape).astype(bool)
                    for a in pupildata['cnn']['excluded_frames']:
                        artifacts[a[0]:a[1]] = True
                    pupil_extras[k] = artifacts
                else:
                    pupil_extras[k] = np.array(pupildata['cnn'][k])[:-1, np.newaxis]

        log.info("pupil_diameter.shape: " + str(pupil_diameter.shape))

    elif '.pup.mat' in pupilfilepath:

        matdata = scipy.io.loadmat(pupilfilepath)

        pupil_extras = {}  # for backwards compaitbility with the new pupil fits

        p = matdata['pupil_data']
        params = p['params']
        if ('pupil_variable_name' not in options):
            options['pupil_variable_name'] = params[0][0]['default_var'][0][0][0]
            log.debug("Using default pupil_variable_name: %s", options['pupil_variable_name'])
        elif (options['pupil_variable_name'] == 'area'):
            log.info("Ignoring default pupil variable and using pupil area")
        if 'pupil_algorithm' not in options:
            options['pupil_algorithm'] = params[0][0]['default'][0][0][0]
            log.debug("Using default pupil_algorithm: %s", options['pupil_algorithm'])

        results = p['results'][0][0][-1][options['pupil_algorithm']]
        if options['pupil_variable_name'] == 'area':
            pupil_diameter = np.pi * np.array(results[0]["minor_axis"][0][0]) * np.array(
                results[0]["major_axis"][0][0]) / 2
        else:
            pupil_diameter = np.array(results[0][options['pupil_variable_name']][0][0])
        if pupil_diameter.shape[0] == 1:
            pupil_diameter = pupil_diameter.T
        log.info("pupil_diameter.shape: " + str(pupil_diameter.shape))

    fs_approximate = 30  # approx video framerate
    if pupil_deblink & ~loading_pcs:
        dp = np.abs(np.diff(pupil_diameter, axis=0))
        blink = np.zeros(dp.shape)
        blink[dp > np.nanmean(dp) + 4 * np.nanstd(dp)] = 1
        # CRH add following line 7-19-2019
        # (blink should be = 1 if pupil_dia goes to 0)
        blink[[isclose(p, 0, abs_tol=0.5) for p in pupil_diameter[:-1]]] = 1
        smooth_width = int(fs_approximate * pupil_deblink_dur)
        box = np.ones([smooth_width]) / smooth_width
        blink = np.convolve(blink[:, 0], box, mode='same')
        blink[blink > 0] = 1
        blink[blink <= 0] = 0
        onidx, = np.where(np.diff(blink) > 0)
        offidx, = np.where(np.diff(blink) < 0)

        if (len(onidx) == 0) and (len(offidx) == 0):
            log.info("WARNING - Tried to deblink but didn't find any blinks. Continue loading pupil trace...")
        else:
            if onidx[0] > offidx[0]:
                onidx = np.concatenate((np.array([0]), onidx))
            if len(onidx) > len(offidx):
                offidx = np.concatenate((offidx, np.array([len(blink)])))
            deblinked = pupil_diameter.copy()

            for i, x1 in enumerate(onidx):
                x2 = offidx[i]
                if x2 < x1:
                    log.info([i, x1, x2])
                    log.info("WHAT'S UP??")
                else:
                    # print([i,x1,x2])
                    deblinked[x1:x2, 0] = np.linspace(
                        deblinked[x1], deblinked[x2 - 1], x2 - x1
                    ).squeeze()

            if verbose:
                plt.figure()
                plt.plot(pupil_diameter, label='Raw')
                plt.plot(deblinked, label='Deblinked')
                plt.xlabel('Frame')
                plt.ylabel('Pupil')
                plt.legend()
                plt.title("Artifacts detected: {}".format(len(onidx)))
            log.info("Deblink: artifacts detected: {}".format(len(onidx)))
            pupil_diameter = deblinked

    # resample and remove dropped frames

    # find and parse pupil events
    pp = ['PUPIL,' in x['name'] for i, x in exptevents.iterrows()]
    trials = list(exptevents.loc[pp, 'Trial'])
    ntrials = len(trials)
    timestamp = np.zeros([ntrials + 1])
    firstframe = np.zeros([ntrials + 1])
    for i, x in exptevents.loc[pp].iterrows():
        t = int(x['Trial'] - 1)
        s = x['name'].split(",[")
        p = eval("[" + s[1])
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
    # svd/CRH fix: use start_e to determine trial duration
    duration = np.diff(np.append(start_e, stop_e[-1]) / rasterfs)

    # old method: use timestamps in pupil recording, which don't take into account correction for sampling bin size
    # that may be coarser than the video sampling rate
    # duration = np.diff(timestamp) * 24*60*60

    frame_count = np.diff(firstframe)

    if loading_pcs:
        # facemap stuff
        l = ['pupil']
    elif options['pupil']:
        l = ['pupil'] + list(pupil_extras.keys())

    big_rs_dict = {}
    for signal in l:
        extras = False
        # warp/resample each trial to compensate for dropped frames
        strialidx = np.zeros([ntrials + 1])
        # big_rs = np.array([[]])
        all_fs = np.empty([ntrials])

        for ii in range(0, ntrials):
            if loading_pcs:
                d = pupil_diameter[int(firstframe[ii]):int(firstframe[ii] + frame_count[ii]), :]
            elif signal == 'pupil':
                d = pupil_diameter[
                    int(firstframe[ii]):int(firstframe[ii] + frame_count[ii]), 0
                    ]
            elif signal in pupil_extras_keys:
                extras = True
                d = pupil_extras[signal][
                    int(firstframe[ii]):int(firstframe[ii] + frame_count[ii]), 0
                    ]

            fs = frame_count[ii] / duration[ii]
            all_fs[ii] = fs
            t = np.arange(0, d.shape[0]) / fs
            ti = np.arange(
                (1 / rasterfs) / 2, duration[ii] + (1 / rasterfs) / 2, 1 / rasterfs
            )
            # print("{0} len(d)={1} len(ti)={2} fs={3}"
            #       .format(ii,len(d),len(ti),fs))
            _f = interp1d(t, d, axis=0, fill_value="extrapolate")
            di = _f(ti)
            if ii == 0:
                big_rs = di
            else:
                big_rs = np.concatenate((big_rs, di), axis=0)
            if (ii < ntrials - 1) and (len(big_rs) > start_e[ii + 1]):
                big_rs = big_rs[:start_e[ii + 1]]
            elif ii == ntrials - 1:
                big_rs = big_rs[:stop_e[ii]]

            strialidx[ii + 1] = big_rs.shape[0]

        if (pupil_median) & (signal == 'pupil'):
            kernel_size = int(round(pupil_median * rasterfs / 2) * 2 + 1)
            big_rs = scipy.signal.medfilt(big_rs, kernel_size=(kernel_size, 1))

        # shift pupil (or extras) trace by offset, usually 0.75 sec
        offset_frames = int(pupil_offset * rasterfs)
        big_rs = np.roll(big_rs, -offset_frames, axis=0)

        # svd pad with final pupil value (was np.nan before)
        big_rs[-offset_frames:] = big_rs[-offset_frames]

        # shape to 1 x T to match NEMS signal specs. or transpose if 2nd dim already exists
        if big_rs.ndim == 1:
            big_rs = big_rs[np.newaxis, :]
        else:
            big_rs = big_rs.T

        if pupil_mm:
            try:
                # convert measurements from pixels to mm
                eye_width_px = matdata['pupil_data']['results'][0][0]['eye_width'][0][0][0]
                eye_width_mm = matdata['pupil_data']['params'][0][0]['eye_width_mm'][0][0][0]
                big_rs = big_rs * (eye_width_mm / eye_width_px)
            except:
                log.info("couldn't convert pupil to mm")

        if verbose:
            # plot framerate for each trial (for checking camera performance)
            plt.figure()
            plt.plot(all_fs.T)
            plt.xlabel('Trial')
            plt.ylabel('Sampling rate (Hz)')

        if verbose:
            plt.show()

        if len(l) >= 2:
            big_rs_dict[signal] = big_rs

    if len(l) >= 2:
        return big_rs_dict, strialidx
    else:
        return big_rs, strialidx


def load_dlc_trace(dlcfilepath, exptevents=None, return_raw=False, verbose=False,
                   rasterfs=30, dlc_threshold=-1, fill_invalid='interpolate', max_gap=2,
                   **options):
    """
    returns big_rs which is pupil trace resampled to options['rasterfs']
    and strialidx, which is the index into big_rs for the start of each
    trial. need to make sure the big_rs vector aligns with the other signals

    testing:
    parmfile = '/auto/data/daq/Clathrus/training2022/Clathrus_2022_01_11_TBP_1.m'
    dlcfilepath = '/auto/data/daq/Clathrus/training2022/sorted/Clathrus_2022_01_11_TBP_1.lickDLC_resnet50_multividJan14shuffle1_1030000.h5'
    """

    # todo : figure out filename from parm file path.
    # options = set_default_pupil_options(options)
    # pupilfilepath = get_pupil_file(pupilfilepath, **options)

    options['dlc'] = True

    # if options["dlc_smooth"]:
    #    raise ValueError('pupil_smooth not implemented. try pupil_median?')

    dataframe = pd.read_hdf(dlcfilepath)
    scorer = dataframe.columns.get_level_values(0)[0]
    bodyparts = dataframe[scorer].columns.get_level_values(0)

    num_frames = dataframe.shape[0]
    names_bodyparts = list(bodyparts.unique(level=0))
    num_bodyparts = len(names_bodyparts)

    data_array = np.zeros((num_bodyparts * 2, num_frames))
    list_bodyparts = []

    for i, bp in enumerate(names_bodyparts):
        x = dataframe[scorer][bp]['x'].values
        y = dataframe[scorer][bp]['y'].values
        threshold_check = dataframe[scorer][bp]['likelihood'].values > dlc_threshold
        bad_frame_count = (~threshold_check).sum()
        assume_videofs=30
        if bad_frame_count==0:
            pass
            # no bad frames
        elif (fill_invalid == 'interpolate'):
            invalid_onsets = np.where(np.diff(threshold_check.astype(int))==-1)[0]+1
            invalid_offsets = np.where(np.diff(threshold_check.astype(int))==1)[0]+1
            if invalid_onsets[0] > invalid_offsets[0]:
                invalid_onsets = np.concatenate(([0], invalid_onsets))
            if invalid_onsets[-1] > invalid_offsets[-1]:
                invalid_offsets = np.concatenate((invalid_offsets, [len(threshold_check)]))
            for (a, b) in zip(invalid_onsets, invalid_offsets):
                if (a > 0) & (b < len(x)) & ((b-a)/assume_videofs <= max_gap ):
                    x[a:b] = np.linspace(x[a-1], x[b], b-a)
                    y[a:b] = np.linspace(y[a - 1], y[b], b - a)
                else:
                    x[a:b] = np.nan
                    y[a:b] = np.nan

        elif (fill_invalid == 'mean'):
            log.info(f"{bp}: {bad_frame_count} bad samples, filling in with mean")
            if (fill_invalid == 'mean') & (bad_frame_count > 0) & (max_gap > 0):

                x[~threshold_check] = np.nanmean(x[threshold_check])
                y[~threshold_check] = np.nanmean(y[threshold_check])

                if bad_frame_count > 0:
                    log.info(f"{bp}: {bad_frame_count} bad samples, filling in with mean")
        else: # nan
            x[~threshold_check] = np.nan
            y[~threshold_check] = np.nan
            if bad_frame_count > 0:
                log.info(f"{bp}: {bad_frame_count} bad samples, replacing with nan")
        data_array[2 * i] = x
        data_array[2 * i + 1] = y

        list_bodyparts.append(bp + "_x")
        list_bodyparts.append(bp + "_y")

    if verbose:
        log.info(data_array.shape)  # should be number of bodyparts*2 x number of frames
        log.info(list_bodyparts)  # should be each bodypart twice
        log.info(data_array)  # check that values match

    if return_raw:
        return data_array, list_bodyparts

    # we want to use exptevents TRIALSTART events as the ground truth for the time when each trial starts.
    # these times are set based on openephys data, since baphy doesn't log exact trial start times
    if exptevents is None:
        from nems_lbhb.baphy_experiment import BAPHYExperiment

        experiment = BAPHYExperiment.from_pupilfile(pupilfilepath)
        trial_starts = experiment.get_trial_starts()
        exptevents = experiment.get_baphy_events()

    fs_approximate = 30  # approx video framerate
    # resample and remove dropped frames

    # find and parse lick trace events
    #pp = ['LICK,' in x['name'] for i, x in exptevents.iterrows()]
    pp = exptevents['name'].str.startswith('LICK,')
    if pp.sum() > 0:
        # yes, there were LICK events, load baphy style
        trials = list(exptevents.loc[pp, 'Trial'])
        ntrials = len(trials)
        timestamp = np.zeros([ntrials + 1])
        firstframe = np.zeros([ntrials + 1])
        for i, x in exptevents.loc[pp].iterrows():
            t = int(x['Trial'] - 1)
            s = x['name'].split(",[")
            p = eval("[" + s[1])
            # print("{0} p=[{1}".format(i,s[1]))
            timestamp[t] = p[0]
            firstframe[t] = int(p[1])

        pp = ['LICKSTOP' in x['name'] for i, x in exptevents.iterrows()]
        lastidx = np.argwhere(pp)[-1]

        s = exptevents.iloc[lastidx[0]]['name'].split(",[")
        p = eval("[" + s[1])
        timestamp[-1] = p[0]
        firstframe[-1] = int(p[1])

        # align DLC signals with other events, probably by
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

    else:
        # assume PSIVIDEO format instead
        pp = exptevents['name'].str.startswith('PSIVIDEO')
        trials = list(exptevents.loc[pp, 'Trial'])
        ntrials = len(trials)
        timestamp = np.zeros([ntrials + 2])
        firstframe = np.zeros([ntrials + 2])
        for i, x in exptevents.loc[pp].iterrows():
            t = int(x['Trial'])
            s = x['name'].split(",")

            timestamp[t] = x['start']
            firstframe[t] = int(s[1])
            #print(f"Trial {t+1} time={timestamp[t]:.2f} frames={firstframe[t]:.0f}")
        nominal_fr = int(np.round(firstframe[-2]/timestamp[-2]))
        final_frames = data_array.shape[1]-firstframe[-2]
        firstframe[-1] = data_array.shape[1]
        timestamp[-1] = timestamp[-2]+final_frames/nominal_fr
        start_e = (timestamp[:-1] * rasterfs).astype(int)
        stop_e = (timestamp[1:] * rasterfs).astype(int)
        if stop_e[0]==0:
            start_e = start_e[1:]
            stop_e = stop_e[1:]
            timestamp = timestamp[1:]
            firstframe = firstframe[1:]

    # calculate frame count and duration of each trial
    duration = np.diff(np.append(start_e, stop_e[-1]) / rasterfs)

    frame_count = np.diff(firstframe)
    l = list_bodyparts

    big_rs_dict = {}
    for sigidx, signal in enumerate(l):
        extras = False
        # warp/resample each trial to compensate for dropped frames
        strialidx = np.zeros([len(frame_count) + 1])
        # big_rs = np.array([[]])
        all_fs = np.empty([len(frame_count)])

        for ii in range(len(frame_count)):
            d = data_array[sigidx, int(firstframe[ii]):int(firstframe[ii] + frame_count[ii])]

            fs = frame_count[ii] / duration[ii]
            all_fs[ii] = fs
            t = np.arange(0, d.shape[0]) / fs
            ti = np.arange(
                (1 / rasterfs) / 2, duration[ii] + (1 / rasterfs) / 2, 1 / rasterfs
            )
            # print("{0} len(d)={1} len(ti)={2} fs={3}"
            #       .format(ii,len(d),len(ti),fs))
            _f = interp1d(t, d, axis=0, fill_value="extrapolate")
            di = _f(ti)
            if ii == 0:
                big_rs = di
            else:
                big_rs = np.concatenate((big_rs, di), axis=0)
            if (ii < ntrials - 1) and (len(big_rs) > start_e[ii + 1]):
                big_rs = big_rs[:start_e[ii + 1]]
            elif ii == ntrials - 1:
                big_rs = big_rs[:stop_e[ii]]

            strialidx[ii + 1] = big_rs.shape[0]

        # if (pupil_median) & (signal == 'pupil'):
        #    kernel_size = int(round(pupil_median*rasterfs/2)*2+1)
        #    big_rs = scipy.signal.medfilt(big_rs, kernel_size=(kernel_size,1))

        # shape to 1 x T to match NEMS signal specs. or transpose if 2nd dim already exists
        if big_rs.ndim == 1:
            big_rs = big_rs[np.newaxis, :]
        else:
            big_rs = big_rs.T

        if verbose and (sigidx == 0):
            # plot framerate for each trial (for checking camera performance)
            plt.figure()
            plt.plot(all_fs.T)
            plt.xlabel('Trial')
            plt.ylabel('Sampling rate (Hz)')

        if verbose:
            plt.show()

        if len(l) >= 2:
            big_rs_dict[signal] = big_rs

    log.info('done creating big_rs')

    if len(l) >= 2:
        return big_rs_dict, strialidx
    else:
        return big_rs, strialidx


def load_facepca_trace(facepcafilepath, exptevents=None, return_raw=False,
                       verbose=False, rasterfs=30, pc_count=1, fill_invalid='mean',
                       **options):
    """
    returns big_rs which is pupil trace resampled to options['rasterfs']
    and strialidx, which is the index into big_rs for the start of each
    trial. need to make sure the big_rs vector aligns with the other signals

    testing:
    parmfile = '/auto/data/daq/Amanita/AMT020/AMT020a11_p_NAT.m'
    facepcafilepath = '/auto/data/daq/Amanita/AMT020/sorted/AMT020a11_p_NAT.facePCs.h5'
    """

    options['facepca'] = pc_count

    # if options["dlc_smooth"]:
    #    raise ValueError('pupil_smooth not implemented. try pupil_median?')

    data = {}
    f = h5py.File(facepcafilepath, 'r')
    for k in f.keys():
        data[k] = np.array(f.get(k))

    data_array = data['projection'][:, :pc_count]

    num_frames = data_array.shape[0]

    if verbose:
        log.info(f"Face PC data shape: {data_array.shape}")

    if return_raw:
        return data_array

    # we want to use exptevents TRIALSTART events as the ground truth for the time when each trial starts.
    # these times are set based on openephys data, since baphy doesn't log exact trial start times
    if exptevents is None:
        raise ValueError('Must provide exptevents. Inference not currently supported.')
        # TODO fix this
        from nems_lbhb.baphy_experiment import BAPHYExperiment

        experiment = BAPHYExperiment.from_pupilfile(pupilfilepath)
        trial_starts = experiment.get_trial_starts()
        exptevents = experiment.get_baphy_events()

    fs_approximate = 30  # approx video framerate
    # resample and remove dropped frames

    # find and parse lick trace events
    pp = ['PUPIL,' in x['name'] for i, x in exptevents.iterrows()]

    trials = list(exptevents.loc[pp, 'Trial'])
    ntrials = len(trials)
    timestamp = np.zeros([ntrials + 1])
    firstframe = np.zeros([ntrials + 1])
    for i, x in exptevents.loc[pp].iterrows():
        t = int(x['Trial'] - 1)
        s = x['name'].split(",[")
        p = eval("[" + s[1])
        # print("{0} p=[{1}".format(i,s[1]))
        timestamp[t] = p[0]
        firstframe[t] = int(p[1])
    pp = ['PUPILSTOP' in x['name'] for i, x in exptevents.iterrows()]
    lastidx = np.argwhere(pp)[-1]

    s = exptevents.iloc[lastidx[0]]['name'].split(",[")
    p = eval("[" + s[1])
    timestamp[-1] = p[0]
    firstframe[-1] = int(p[1])

    # align Face PC signals with other events, probably by
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
    duration = np.diff(np.append(start_e, stop_e[-1]) / rasterfs)

    frame_count = np.diff(firstframe)

    big_rs_list = []
    for sigidx in range(pc_count):
        signal = f"fpc{sigidx}"
        extras = False
        # warp/resample each trial to compensate for dropped frames
        strialidx = np.zeros([ntrials + 1])
        # big_rs = np.array([[]])
        all_fs = np.empty([ntrials])

        for ii in range(0, ntrials):
            d = data_array[int(firstframe[ii]):int(firstframe[ii] + frame_count[ii]), sigidx]

            fs = frame_count[ii] / duration[ii]
            all_fs[ii] = fs
            t = np.arange(0, d.shape[0]) / fs
            ti = np.arange(
                (1 / rasterfs) / 2, duration[ii] + (1 / rasterfs) / 2, 1 / rasterfs
            )
            # print("{0} len(d)={1} len(ti)={2} fs={3}"
            #       .format(ii,len(d),len(ti),fs))
            _f = interp1d(t, d, axis=0, fill_value="extrapolate")
            di = _f(ti)
            if ii == 0:
                big_rs = di
            else:
                big_rs = np.concatenate((big_rs, di), axis=0)
            if (ii < ntrials - 1) and (len(big_rs) > start_e[ii + 1]):
                big_rs = big_rs[:start_e[ii + 1]]
            elif ii == ntrials - 1:
                big_rs = big_rs[:stop_e[ii]]

            strialidx[ii + 1] = big_rs.shape[0]

        # if (pupil_median) & (signal == 'pupil'):
        #    kernel_size = int(round(pupil_median*rasterfs/2)*2+1)
        #    big_rs = scipy.signal.medfilt(big_rs, kernel_size=(kernel_size,1))

        # shape to 1 x T to match NEMS signal specs. or transpose if 2nd dim already exists
        if big_rs.ndim == 1:
            big_rs = big_rs[np.newaxis, :]
        else:
            big_rs = big_rs.T
        big_rs_list.append(big_rs)
        if verbose and (sigidx == 0):
            # plot framerate for each trial (for checking camera performance)
            plt.figure()
            plt.plot(all_fs.T)
            plt.xlabel('Trial')
            plt.ylabel('Sampling rate (Hz)')

        if verbose:
            plt.show()

    big_rs = np.concatenate(big_rs_list, axis=0)
    log.info('done creating big_rs')

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
    # find appropriate pupil file
    pupilfilepath = get_pupil_file(pupilfilepath)

    # Set analysis parameters from defaults, if necessary.
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

    # Load data.
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

    pupil_size = pupil_size[0, :]
    eye_speed = eye_speed[0, :]

    # Find REM episodes.

    # (1) Very small pupil sizes often indicate that the pupil is occluded by the
    # eyelid or underlit. In either case, measurements of eye position are
    # unreliable, so we remove these frames of the trace before analysis.
    pupil_size[np.nan_to_num(pupil_size) < min_pupil] = np.nan
    eye_speed[np.nan_to_num(pupil_size) < min_pupil] = np.nan

    # (2) Rapid eye movements are similar to saccades. In our data,
    # these appear as large, fast spikes in the speed at which pupil moves.
    # To mark epochs when eye is moving more quickly than usual, threshold
    # eye speed, then smooth by calculating the rate of saccades per minute.
    saccades = np.nan_to_num(eye_speed) > min_saccade_speed
    minute = np.ones(rasterfs * 60) / (rasterfs * 60)
    saccades_per_minute = np.convolve(saccades, minute, mode='same')

    # (3) To distinguish REM sleep from waking - since it seeems that ferrets
    # can sleep with their eyes open - look for periods when pupil is constricted
    # and doesn't show slow oscillations (which may indicate a different sleep
    # stage or quiet waking).
    #  10-second moving average of pupil size:
    ten_seconds = np.ones(rasterfs * 10) / (rasterfs * 10)
    smooth_pupil_size = np.convolve(pupil_size, ten_seconds, mode='same');
    # 10-second moving standard deviation of pupil size:
    pupil_sd = pd.Series(smooth_pupil_size)
    pupil_sd = pupil_sd.rolling(rasterfs * 10).std()
    pupil_sd = np.array(pupil_sd)
    rem_episodes = (np.nan_to_num(smooth_pupil_size) < max_pupil) & \
                   (np.isfinite(smooth_pupil_size)) & \
                   (np.nan_to_num(pupil_sd) < max_pupil_sd) & \
                   (np.nan_to_num(saccades_per_minute) > min_saccades_per_minute)

    # (4) Connect episodes that are separated by a brief gap.
    rem_episodes = run_length_encode(rem_episodes)
    brief_gaps = []
    for i, episode in enumerate(rem_episodes):
        is_gap = not (episode[0])
        gap_time = episode[1]
        if is_gap and gap_time / rasterfs < max_gap_s:
            rem_episodes[i] = (True, gap_time)
            brief_gaps.append((True, gap_time))
        else:
            brief_gaps.append((False, gap_time))

    # (5) Remove brief REM episodes.
    rem_episodes = run_length_encode(run_length_decode(rem_episodes))
    brief_episodes = []
    for i, episode in enumerate(rem_episodes):
        is_rem_episode = episode[0]
        episode_time = episode[1]
        if is_rem_episode and episode_time / rasterfs < min_episode_s:
            rem_episodes[i] = (False, episode_time)
            brief_episodes.append((True, episode_time))
        else:
            brief_episodes.append((False, episode_time))

    is_rem = run_length_decode(rem_episodes)

    # Plot
    if verbose:

        samples = pupil_size.size
        minutes = samples / (rasterfs * 60)
        time_ax = np.linspace(0, minutes, num=samples)

        is_brief_gap = run_length_decode(brief_gaps)
        is_brief_episode = run_length_decode(brief_episodes)
        rem_dur = np.array([t for is_rem, t in rem_episodes if is_rem]) / (rasterfs * 60)

        fig, ax = plt.subplots(4, 1)
        if len(rem_dur) == 0:
            title_str = 'no REM episodes'
        elif len(rem_dur) == 1:
            title_str = '1 REM episode, duration: {:0.2f} minutes'. \
                format(rem_dur[0])
        else:
            title_str = '{:d} REM episodes, mean duration: {:0.2f} minutes'. \
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

        ax[0].legend((l0, l1, l2), \
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
    return [(k, len(list(g))) for k, g in groupby(a)]


def run_length_decode(a):
    """
    Reverses the operation performed by run_length_encode.

    ZPS 2018-09-24: Helper function for get_rem.
    """
    a = [list(repeat(elem, n)) for (elem, n) in a]
    a = list(chain.from_iterable(a))
    return np.array(a)


def cache_rem_options(pupilfilepath, cachepath=None, **options):
    pupilfilepath = get_pupil_file(pupilfilepath)

    options['verbose'] = False
    if '.pickle' in pupilfilepath:
        jsonfilepath = pupilfilepath.replace('.pickle', '.rem.json')
    else:
        jsonfilepath = pupilfilepath.replace('.pup.mat', '.rem.json')

    if cachepath is not None:
        pp, bb = os.path.split(jsonfilepath)
        jsonfilepath = os.path.join(cachepath, bb)

    fh = open(jsonfilepath, 'w')
    json.dump(options, fh)
    fh.close()


def load_rem_options(pupilfilepath, cachepath=None, **options):
    pupilfilepath = get_pupil_file(pupilfilepath)

    if '.pickle' in pupilfilepath:
        jsonfilepath = pupilfilepath.replace('.pickle', '.rem.json')
    else:
        jsonfilepath = pupilfilepath.replace('.pup.mat', '.rem.json')
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


def get_pupil_file(pupilfilepath, **options):
    """
    For backwards compatibility in pupil/rem functions. Default is to load the
    pupil fit from the CNN model fit. However, for some older recordings, this
    may not exist and so you may still want to load the pup.mat file. This
    is a helper function to find which pupil file to load
    6-28-2019, CRH

    options dict added 08.17.2020. In options. specific "pupil_analysis_method" to
        specifically say if you matlab / python results for pupil. Default is to use python:
        (options['pupil_analysis_method']='cnn'). If the method you ask for doesn't exist, you'll
        get a log message warning, but it will then try to load the other option.
    """
    pupilfilepath = str(pupilfilepath)
    if ('.pickle' in pupilfilepath) & os.path.isfile(pupilfilepath) & (options['pupil_analysis_method'] == 'cnn'):
        log.info("Loading CNN pupil fit from .pickle file")
        return pupilfilepath

    if (options['pupil_analysis_method'] == 'cnn') & (
            (not os.path.isfile(pupilfilepath)) | ('pup.mat' in pupilfilepath)):

        if not os.path.isfile(pupilfilepath):
            pp, bb = os.path.split(pupilfilepath)
            pupilfilepath = pp + '/sorted/' + bb.split('.')[0] + '.pickle'

            if os.path.isfile(pupilfilepath):
                log.info("Loading CNN pupil fit from .pickle file")
                return pupilfilepath
            else:
                raise FileNotFoundError("Pupil analysis not found")

        elif os.path.isfile(pupilfilepath):
            pp, bb = os.path.split(pupilfilepath)
            CNN_pupilfilepath = pp + '/sorted/' + bb.split('.')[0] + '.pickle'

            if os.path.isfile(CNN_pupilfilepath):
                log.info("Loading CNN pupil fit from .pickle file")
                return CNN_pupilfilepath
            else:
                log.info("WARNING: CNN pupil fit doesn't exist, " \
                         "Loading pupil fit from .pup.mat file")
                return pupilfilepath

    elif ('pup.mat' in pupilfilepath) & (options['pupil_analysis_method'] == 'matlab'):
        if os.path.isfile(pupilfilepath):
            return pupilfilepath
        else:
            raise FileNotFoundError("Asked for matlab pupil analysis, but results file doesn't exist." \
                                    "Check that this video has been analyzed and/or try setting options['pupil_analyis_method']='cnn'")

    else:
        raise FileNotFoundError("Pupil analysis not found")


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
    # options['rasterfs']=100
    # options['pupil_mm']=True
    # options['pupil_median']=0.5
    # options['pupil_deblink']=True
    # options['units']='mm'
    # options['verbose']=False

    options = set_default_pupil_options(options)

    cacheroot = "/auto/data/nems_db/recordings/pupil/"

    pp, pupbb = os.path.split(pupilfilepath)
    pp_animal, pen = os.path.split(pp)
    pp_daq, animal = os.path.split(pp_animal)
    cachebb = pupbb.replace(".pup.mat", "")
    cachepath = os.path.join(cacheroot, animal, )

    parmfilepath = pupilfilepath.replace(".pup.mat", ".m")
    pp, bb = os.path.split(parmfilepath)

    globalparams, exptparams, exptevents = baphy_parm_read(parmfilepath)
    spkfilepath = pp + '/' + spk_subdir + re.sub(r"\.m$", ".spk.mat", bb)
    log.info("Spike file: {0}".format(spkfilepath))
    # load spike times
    spikedata = baphy_load_spike_data_raw(spkfilepath)
    # adjust spike and event times to be in seconds since experiment started

    exptevents, spiketimes, unit_names = baphy_align_time(
        exptevents, spikedata['sortinfo'], spikedata['spikefs'], options["rasterfs"])
    log.info('Creating trial events')
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
    t_pupil = nems0.signal.RasterizedSignal(
            fs=options['rasterfs'], data=pupildata,
            name='pupil', recording=cachebb, chans=['pupil', 'rem'],
            epochs=event_times)

    return pupil_trace, is_rem, options


def load_raw_pupil(pupilfilepath, fs=None):
    """
    Simple function to read continuous pupil trace in w/o baphy trial start alignment etc.

    Right now, only works for .pickle pupil files.
    """

    with open(pupilfilepath, 'rb') as fp:
        pupildata = pickle.load(fp)

    pupil_diameter = pupildata['cnn']['a'] * 2

    # missing frames/frames that couldn't be decoded were saved as nans
    # pad them here
    nan_args = np.argwhere(np.isnan(pupil_diameter))

    for arg in nan_args:
        arg = arg[0]
        log.info("padding missing pupil frame {0} with adjacent ellipse params".format(arg))
        try:
            pupil_diameter[arg] = pupil_diameter[arg - 1]
        except:
            pupil_diameter[arg] = pupil_diameter[arg - 1]

    pupil_diameter = pupil_diameter[:-1, np.newaxis]

    return pupil_diameter


def load_raw_photometry(photofilepath, fs=None, framen=0):
    """
    Simple function to read (and process photometry trace). Will ask for user input to define ROI based
    on the first frame of the video file.
    """
    import av
    video_container = av.open(photofilepath)
    video_stream = [s for s in video_container.streams][0]

    F_mag1 = []
    F_mag2 = []
    for i, packet in enumerate(video_container.demux(video_stream)):
        if i % 1000 == 0:
            log.info("frame: {}".format(i))

        if i < framen:
            frame = packet.decode()[0]

        elif i == framen:
            frame = packet.decode()[0]
            frame_ = np.asarray(frame.to_image().convert('LA'))
            plt.imshow(frame_[:, :, 0])
            print("Before closing image, locate the center of your ROI!")
            plt.show()
            print("left ROI: ")
            x = int(input("x1 center: "))
            y = int(input("y1 center: "))
            print("right ROI: ")
            x2 = int(input("x2 center: "))
            y2 = int(input("y2 center: "))

            # define roi:
            roi_width = 4
            x1_range = np.arange(x - roi_width, x + roi_width)
            y1_range = np.arange(y - roi_width, y + roi_width)

            x2_range = np.arange(x2 - roi_width, x2 + roi_width)
            y2_range = np.arange(y2 - roi_width, y2 + roi_width)

            roi1 = frame_[:, :, 0][x1_range, :][:, y1_range]
            roi2 = frame_[:, :, 0][x2_range, :][:, y2_range]
            fmag1 = np.mean(roi1)
            fmag2 = np.mean(roi2)
            F_mag1.append(fmag1)
            F_mag2.append(fmag2)

        else:
            try:
                frame = packet.decode()[0]
                frame_ = np.asarray(frame.to_image().convert('LA'))
                roi1 = frame_[:, :, 0][x1_range, :][:, y1_range]
                roi2 = frame_[:, :, 0][x2_range, :][:, y2_range]
                fmag1 = np.mean(roi1)
                fmag2 = np.mean(roi2)
                F_mag1.append(fmag1)
                F_mag2.append(fmag2)
            except:
                print('end of file reached')

    return np.array(F_mag1)[:, np.newaxis], np.array(F_mag2)[:, np.newaxis]


def evpread(filename, options):
    """
    VERY crude first pass at reading in evp file using python.
    For now, just reads in aux chans. Created to load lick data
    CRH 06.19.2020
    """

    auxchans = options.get('auxchans', [])

    f = open(filename, 'rb')
    header = np.fromfile(f, count=10, dtype=np.int32)

    spikechancount = header[1]
    auxchancount = header[2]
    lfpchancount = header[6]
    trials = header[5]
    aux_fs = header[4]

    if len(auxchans) > 0:
        auxchansteps = np.diff(np.concatenate(([0], [a + 1 for a in auxchans], [auxchancount + 1]))) - 1
    else:
        auxchansteps = auxchancount

    # loop over trials
    trialidx = []
    aux_trialidx = []
    for tt in range(trials):
        # read trial header
        trheader = np.fromfile(f, count=3, dtype=np.int32)
        if trheader.size == 0:
            break

        if sum(trheader) != 0:
            ta = []

            # seek through spikedata
            f.seek(trheader[0] * 2 * spikechancount, 1)

            # read in aux data for this trial
            if (auxchancount > 0) & (len(auxchans) > 0):
                for ii in range(auxchancount):
                    if auxchansteps[ii] > 0:
                        f.seek(trheader[1] * 2 * auxchansteps[ii], 1)
                    else:
                        ta.append(np.fromfile(f, count=trheader[1], dtype=np.int16))

                if tt == 0:
                    aux_trialidx.append(trheader[1])
                else:
                    aux_trialidx.append(trheader[1] + aux_trialidx[tt - 1])

            else:
                f.seek(trheader[1] * 2 * auxchancount, 1)

            # seek through lfp data
            f.seek(trheader[2] * 2 * lfpchancount, 1)

            # stack data over channels
            if tt == 0:
                ra = np.stack(ta)
            else:
                ra = np.concatenate((ra, np.stack(ta)), axis=-1)

            # which trials are extracted
            trialidx.append(tt + 1)

        else:
            # skip to next trial
            f.seek((trheader[0] * spikechancount) +
                   (trheader[1] * auxchancount) +
                   (trheader[2] * lfpchancount) * 2, 1)

    # pack and return results
    pack = collections.namedtuple('evpdata', field_names='trialidx aux_fs aux_trialidx aux_data')
    output = pack(trialidx=trialidx,
                  aux_fs=aux_fs, aux_trialidx=aux_trialidx, aux_data=ra)

    f.close()

    return output


def get_lick_events(evpfile, name='LICK'):
    """
    Load analog lick data from evp file. Create dataframe of
    lick events in the style of nems exptevents: columns = [name, start, end, Trial]
    """
    lickdata = evpread(evpfile, {'auxchans': [0]})
    startidx = lickdata.aux_trialidx
    startidx = np.append(0, startidx[:-1])
    endidx = startidx[1:]
    endidx = np.append(endidx, -1)
    trialidx = lickdata.trialidx
    fs = lickdata.aux_fs
    lick_trace = lickdata.aux_data

    s = []
    t = []
    for tidx, eidx, sidx in zip(trialidx, endidx, startidx):
        data = lick_trace[0, sidx:eidx]
        lickedges = np.diff(data - data.mean())
        lickidx = np.argwhere(lickedges > 0).squeeze()

        if lickidx.size == 0:
            pass
        elif lickidx.size == 1:
            s.append(lickidx / fs)
            t.append(tidx)
        elif lickidx.size > 1:
            s.extend(lickidx / fs)
            t.extend([tidx] * len(lickidx))

    # build dataframe
    df = pd.DataFrame(columns=['name', 'start', 'end', 'Trial'], index=range(len(s)))
    df['start'] = s
    df['end'] = s
    df['Trial'] = t
    df['name'] = name

    return df


def get_mean_spike_waveform(cellid, usespkfile=None):
    """
    Return 1-D numpy array containing the mean sorted spike waveform
    :cellid: str
    :usespkfile: Bool or None

        1. The legacy one (usepkpfile==False), finds the phy generated npy files containing the waveforms
        and mean waveforms for the classified clusters. This methods might fail when the Kilosort output has not cluste IDs
        2. The new method (usespkfile==True), asumes there is a mean wavefomr in the sorted info generated when sort jobs
        are finished and pushed to the database. This is a little circuitous since the phy npy waveforms are saved inside
        matlab structs, that then have to be reloaded into python here...
        This method is not backwards compatible, but it should hold consistency with ulterior analysis (?) and its the
        preffered method

        3. if  usespkfile is None, tries new then legacy approaches
        2022-08-16 MLE.
    """
    if type(cellid) != str:
        raise ValueError("cellid must be string type")

    # new method
    def usematfile():
        cparts = cellid.split("-")
        chan = int(cparts[1])
        unit = int(cparts[2])
        sql = f"SELECT runclassid, path, respfile from sCellFile where cellid = '{cellid}'"
        d = db.pd_query(sql)

        if len(d) == 0:
            # log.info(f"No files for {cellid}")
            # mwf = np.array([])
            raise RuntimeError(f"No files for {cellid}")

        for i in range(len(d)):
            spkfilepath = os.path.join(d['path'][i], d['respfile'][i])
            matdata = scipy.io.loadmat(spkfilepath, chars_as_strings=True)
            sortinfo = matdata['sortinfo']
            if sortinfo.shape[0] > 1:
                sortinfo = sortinfo.T

            try:
                mwf = sortinfo[0][chan - 1][0][0].flatten()[unit - 1]['MeanWaveform'].squeeze()
            except:
                # log.info(f"Can't get Mean Waveform for {cellid} {i} {d['respfile'][i]}")
                continue

            if len(mwf) > 0:
                # log.info(f"Good Mean Waveform for {cellid} {i} {d['respfile'][i]} len={len(mwf)}")
                break
            else:
                log.info(f"Empty Mean Waveform for {cellid} {i} {d['respfile'][i]} len={len(mwf)}")

        else:
            # log.info(f"Empty Mean Waveform for {cellid} {i} {d['respfile'][i]} len={len(mwf)}")
            raise RuntimeError(f"Could not find a non Empty Mean Waveform across all {i} {cellid} files")

        return mwf

    # legacy method
    def usenpyfile():
        # get KS_cluster (if it exists... this is a new feature)
        sql = f"SELECT kilosort_cluster_id from gSingleRaw where cellid = '{cellid}'"
        kid = db.pd_query(sql).iloc[0][0]

        # find phy results
        site = cellid.split('-')[0]

        # finds the files associated with this sorted neuron. A single neuron might be present across experiments,
        # so there can be multiple files, and we need to find the Kilosort run that encompases all of them
        # e.g: [/auto/data/daq/Amanita/AMT020/AMT020a14_p_CPN.m , .../AMT020a15_p_CPN.m] -> ...
        # ... /auto/data/daq/Amanita/AMT020/tmp/KiloSort/AMT020a_14_15_KiloSort2_minfr_goodchannels_to0

        sql = f"SELECT stimpath, stimfile from sCellFile where cellid = '{cellid}'"
        cell_df = db.pd_query(sql)
        stimpath = cell_df.stimpath.unique()
        assert len(stimpath) == 1
        path = Path(stimpath[0]) / 'tmp' / 'KiloSort'

        # get the pair of numbers following the penetration, prior the underscore: AMT020a09_p_CPN.m -> 9
        _rns = np.sort(cell_df.stimfile.str.extract(r'\D(\d{2})_').values.squeeze())
        _rns = '_'.join([str(int(ii)) for ii in _rns])
        sortpath = list(path.glob(f"{site}_{_rns}_KiloSort*"))

        if len(sortpath) == 0:
            raise ValueError(f"Couldn't find find directory for cellid: {cellid}")
        elif len(sortpath) > 1:
            raise ValueError(f"multiple paths for {cellid} sort, which one is correct?:\n"
                             f"{list(sortpath)}")

        sortpath = sortpath[0]

        # get all waveforms for this sorted file
        try:
            w = np.load(sortpath / 'results' / 'wft_mwf.npy')
        except:
            w = np.load(sortpath / 'results' / 'mean_waveforms.npy')
        clust_ids = pd.read_csv(sortpath / 'results' / 'cluster_group.tsv', sep='\t').cluster_id
        kidx = np.argwhere(clust_ids.values == kid)[0][0]

        # get waveform
        mwf = w[:, kidx]
        return mwf


    if usespkfile is None:
        try:
            log.info('looking in spike.mat files')
            mwf = usematfile()
        except:
            log.info('failed. looking in spike.npw files')
            mwf = usenpyfile()
    elif usespkfile is True:
        mwf = usematfile()
    elif usespkfile is False:
        mwf = usenpyfile()
    else:
        raise ValueError(f'parameter usepkpfile has to be bool or None but is {usespkfile}')

    return mwf


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
        raise ValueError("must provide cellid and batch, or mfilename")

    siteid = None
    cell_list = None
    if type(cellid) is list:
        cell_list = cellid
    elif (type(cellid) is str) & (('%' in cellid) | ('*' in cellid)):
        cellid = cellid.replace('*', '%')
        cell_data = db.pd_query(f"SELECT cellid FROM Batches WHERE batch=%s and cellid like %s",
                                (batch, cellid))
        cell_list = cell_data['cellid'].to_list()
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
            chan_nums = np.arange(int(chan_lims[0]), int(chan_lims[1]) + 1)

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
        if len(cell_list) == 0:
            print(f'empty cell_list for cellid={cellid}, batch={batch}, rawid={rawid}?')
            import pdb;
            pdb.set_trace()
        siteid = cell_list[0].split('-')[0]
        cell_list, rawid = db.get_stable_batch_cells(batch=batch, cellid=siteid,
                                                     rawid=rawid)

        options['cellid'] = cell_list
        options['rawid'] = rawid
        options['siteid'] = siteid
        cells_to_extract = [cellid]

    if options['cellid'] is not None:
        cellids = options['cellid'] if (type(options['cellid']) is list) \
            else [options['cellid']]
        units = []
        channels = []
        for cellid in cellids:
            t = cellid.split("_")
            # print(cellids)
            # test for special case where psuedo cellid suffix has been added to
            # cellid by stripping anything after a "_" underscore in the cellid (list)
            # provided
            scf = []
            for rawid_ in rawid:  # rawid is actually a list of rawids
                scf_ = db.get_cell_files(t[0], rawid=rawid_)
                scf_ = scf_[['rawid', 'cellid', 'channum', 'unit']].drop_duplicates()
                assert len(scf_) == 1
                scf.append(scf_)
            assert len(scf) == len(rawid)
            channels.append(scf[0].iloc[0].channum)
            units.append(scf[0].iloc[0].unit)
        options['channels'] = channels
        options['units'] = units

    if (len(cells_to_extract) == 0) & (mfilename is None):
        raise ValueError("No cellids found! Make sure cellid/batch is specified correctly, "
                         "or that you've specified an mfile.")

    return list(cells_to_extract), options