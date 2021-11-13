from functools import lru_cache
from pathlib import Path
import logging
import re
import os
import os.path
import pickle
import pathlib as pl

import numpy as np
import scipy.io
import scipy.io as spio
import scipy.ndimage.filters
import scipy.signal
import json
import sys
import tarfile
import datetime
import glob
from math import isclose
import copy
from itertools import groupby, repeat, chain, product

from nems_lbhb import runclass, baphy_io
from nems_lbhb import OpenEphys as oe
from nems_lbhb import SettingXML as oes
import pandas as pd
import matplotlib.pyplot as plt
import nems.epoch as ep
import nems.signal
import nems.recording
import nems.db as db
from nems.recording import Recording
from nems.recording import load_recording
import nems_lbhb.behavior as behavior
import nems_lbhb.behavior_plots as bplot
import nems_lbhb.baphy_io as io
from nems.utils import recording_filename_hash
from nems import get_setting

log = logging.getLogger(__name__)

stim_cache_dir = '/auto/data/tmp/tstim/'  # location of cached stimuli

# special decorator that returns copies of cached objects.
# this is useful for cases where you don't want to accidentally
# mutate the native object returned by a cache function
def copying_lru_cache(maxsize=10, typed=False):
    def decorator(f):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(f)
        def wrapper(*args, **kwargs):
            return copy.deepcopy(cached_func(*args, **kwargs))
        return wrapper
    return decorator

###############################################################################
# Main entry-point for BAPHY experiments
###############################################################################
class BAPHYExperiment:
    '''
    Facilitates managing the various files and datasets associated with a
    single BAPHY experiment:

        >>> parmfile = '/auto/data/daq/Nameko/NMK004/NMK004e06_p_NON.m'
        >>> manager = BAPHYExperiment(parmfile)
        >>> print(manager.pupilfile)
        /auto/data/daq/Nameko/NMK004/NMK004e06_p_NON.pup.mat
        >>> print(manager.spikefile)
        /auto/data/daq/Nameko/NMK004/sorted/NMK004e06_p_NON.spk.mat
        >>> manager.get_trial_starts()
        array([ 31.87386667,  44.28406667,  56.64683333,  68.99676667,
                81.3689    , 102.33266667, 114.70163333, 127.0711    ,
               139.4586    , 151.82203333, 167.28496667, 179.64786667,
               192.05813333, 204.47266667, 216.8878    , 230.98546667,
               243.3703    , 255.7584    , 268.1709    , 280.60616667])
    '''
    @classmethod
    def from_spikefile(cls, spikefile):
        '''
        Initialize class from a spike filename.

        Useful if you are debugging code so that you don't have to manually
        edit the filename to arrive at the parmfilename.
        '''
        spikefile = Path(spikefile)
        folder = spikefile.parent.parent
        parmfile = folder / spikefile.name.rsplit('.', 2)[0]
        parmfile = parmfile.with_suffix('.m')
        return cls(parmfile)

    @classmethod
    def from_pupilfile(cls, pupilfile):
        if 'sorted' in pupilfile:
            # using new pupil analysis, which is save in sorted dir
            pp, bb = os.path.split(pupilfile)
            fn = bb.split('.')[0]
            path = os.path.split(pp)[0]
            parmfile = Path(os.path.join(path, fn)).with_suffix('.m')
        else:
            parmfile = Path(str(pupilfile).rsplit('.', 2)[0])
            parmfile = parmfile.with_suffix('.m')

        return cls(parmfile)

    def __init__(self, parmfile=None, batch=None, cellid=None, rawid=None):
        # Make sure that the '.m' suffix is present! In baphy_load data I see
        # that there's care to make sure it's present so I assume not all
        # functions are careful about the suffix.

        # New init options. CRH 2/15/2020. Want to be able to load 
        #   1) multiple parmfiles into single baphy experiment
        #   2) find parmfiles using batch/cellid

        # if don't pass parmfile, must pass both batch and cellid (can 
        # extract individual cellids later if needed). rawid is optional

        if parmfile is None:
            # use database to find the correct parmfiles / cellids to load
            parse_cellid_options = {
                'batch': batch,
                'cellid': cellid,
                'rawid': rawid
            }
            cells_to_extract, nops = io.parse_cellid(parse_cellid_options)
            self.batch = batch
            self.siteid = cellid[:7]
            self.cells_to_extract = cells_to_extract # this is all cells to be extracted from the recording
            self.cells_to_load = nops['cellid']      # this is all "stable" cellids across these rawid
            self.channels_to_load = nops['channels']
            self.units_to_load = nops['units']
            self.rawid = nops['rawid']
            # get list of corresponding parmfiles at this site for these rawids
            d = db.get_batch_cell_data(batch=batch, cellid=self.siteid, label='parm', rawid=self.rawid)
            files = list(set(list(d['parm'])))
            files.sort()
            self.parmfile = [Path(f).with_suffix('.m') for f in files]


        # db-free options for loading specific cellids / parmfiles
        # here, must assume cellid = cells_to_load = cells_to_extract 
        # (so, might end up caching an extra, redundant, recording,
        # but this is the 'safe' way to do it.)
        elif (type(parmfile) is list): # or (type(parmfile) is str):
            if type(parmfile) is str:
                parmfile=[parmfile]
            self.parmfile = [Path(p).with_suffix('.m') for p in parmfile]
            self.batch = None
            if rawid is None:
                stems = [pl.Path(p).stem + '.m' for p in parmfile]
                print(stems)
                stemstring = "'" + "','".join(stems) + "'"
                rawdata = db.pd_query(f"SELECT * from gDataRaw where parmfile in ({stemstring})")
                rawid = []
                for s in stems:
                    rawid.append(rawdata.loc[rawdata.parmfile==s,'id'].values[0])

            self.rawid = rawid
            
            if cellid is not None:
                if type(cellid) is list:
                    self.cells_to_extract = cellid
                    self.cells_to_load = cellid
                    self.siteid = cellid[0][:7]
                elif type(cellid) is str:
                    self.cells_to_extract = [cellid]
                    self.cells_to_load = [cellid]
                    self.siteid = cellid[:7]
                else:
                    raise TypeError
            else:
                self.siteid = os.path.split(parmfile[0])[-1][:7]
                self.cells_to_load = None
                self.cells_to_extract = None
        else:
            self.parmfile = [Path(parmfile).with_suffix('.m')]
            self.siteid = os.path.split(parmfile)[-1][:7]
            self.batch = None
            self.rawid = [rawid] # todo infer from parmfile instad of parsing
            if cellid is not None:
                
                if type(cellid) is list:
                    self.cells_to_extract = cellid
                    self.cells_to_load = cellid
                    self.channels_to_load = [int(c.split("-")[1]) for c in cellid]
                    self.units_to_load = [int(c.split("-")[2]) for c in cellid]
                    self.siteid = cellid[0][:7]
                elif type(cellid) is str:
                    t = cellid.split("-")
                    self.cells_to_extract = [cellid]
                    self.cells_to_load = [cellid]
                    self.channels_to_load = [int(t[1])]
                    self.units_to_load = [int(t[2])]
                    self.siteid = cellid[:7]
                else:
                    raise TypeError
            else:
                self.sited = os.path.split(parmfile)[-1][:7]
                self.cells_to_load = None
                self.cells_to_extract = None
                self.units_to_load = None

        #if np.any([not p.exists() for p in self.parmfile]):
        #    raise IOError(f'Not all parmfiles in {self.parmfile} were found')

        # we cannot assume all parmfiles come from same folder/experiment (site+number)
        self.folder = self.parmfile[0].parent
        self.experiment = [p.name.split('_', 1)[0] for p in self.parmfile]

        # full file name will be unique though, so this is a list
        self.experiment_with_runclass = [Path(p.stem) for p in self.parmfile]

        # add some new attributes for purposes of caching recordings
        self.animal = str(self.parmfile[0].parent).split(os.path.sep)[4]
        penname = str(self.parmfile[0].parent).split(os.path.sep)[5]
        # if batch is None, set batch = 'animal/siteid', unless "training" in parmfile.
        # If training, save in training director by setting batch = 'animal/trainingXXXX'
        if (self.batch is None) & ('training' in penname):
            self.batch = os.path.sep.join([self.animal, penname])
        elif self.batch is None:
            self.batch = os.path.sep.join([self.animal, self.siteid])
        else:
            pass
    
    @property
    @lru_cache(maxsize=128)
    def openephys_folder(self):
        folders = []
        for e,er in zip(self.experiment, self.experiment_with_runclass):
            path = self.folder / 'raw' / e
            candidates = list(path.glob(er.as_posix() + '*'))
            if len(candidates) > 1:
                raise ValueError('More than one candidate found')
            if len(candidates) == 0:
                raise ValueError('No candidates found')
            folders += candidates
        return folders

    @property
    @lru_cache(maxsize=128)
    def openephys_tarfile(self):
        '''
        Return path to OpenEphys tarfile containing recordings
        '''
        path = [(self.folder / 'raw' / e).with_suffix('.tgz') for e in self.experiment]
        return path

    @property
    @lru_cache(maxsize=128)
    def openephys_tarfile_relpath(self):
        '''
        Return relative path in OpenEphys tarfile that represents the "parent"
        of all files (e.g., *.continuous) stored within, e.g.:

            filename = manager.openephys_tarfile_relpath / '126_CH1.continuous'
            import tarfile
            with tarfile.open(manager.openephys_tarfile, 'r:gz') as fh:
                fh.open(filename)
        '''
        return [f.relative_to(t.parent) for f,t in zip(self.openephys_folder, self.openephys_tarfile)]
        #parent = self.openephys_tarfile.parent
        #return self.openephys_folder.relative_to(parent)

    @property
    @lru_cache(maxsize=128)
    def pupilfile(self):
        return [p.with_suffix('.pup.mat') for p in self.parmfile]

    @property
    @lru_cache(maxsize=128)
    def spikefile(self):
        filenames = [self.folder / 'sorted' / s for s in self.experiment_with_runclass]
        if np.any([not f.with_suffix('.spk.mat').exists() for f in filenames]):
            raise IOError("Spike file doesn't exist") 
        else:
            return [f.with_suffix('.spk.mat') for f in filenames]
    
    @property
    @lru_cache(maxsize=128)
    def behavior(self):
        exptparams = self.get_baphy_exptparams()
        behavior = np.any([e['BehaveObjectClass'] != 'Passive' for e in exptparams])
        return behavior

    @property
    @lru_cache(maxsize=128)
    def correction_method(self):
        # figure out appropriate time correction method
        try:
            # TODO add check for openephys files. For now,
            # just use spikes if a spikefile exists
            self.spikefile
            method = 'spikes'
        except IOError:
            method = 'baphy'
        return method

    @lru_cache(maxsize=128)
    def get_trial_starts(self, method='openephys'):
        if method == 'openephys':
            return [io.load_trial_starts_openephys(openephys_folder) for openephys_folder in self.openephys_folder]
        raise ValueError(f'Method "{method}" not supported')

    @lru_cache(maxsize=128)
    def get_raw_sampling_rate(self, method='openephys'):
        if method == 'openephys':
            return [io.load_sampling_rate_openephys(openephys_folder) for openephys_folder in self.openephys_folder]
        raise ValueError(f'Method "{method}" not supported')

    @lru_cache(maxsize=128)
    def get_baphy_events(self, correction_method='openephys', rasterfs=None, **kw):
        baphy_events = self.get_baphy_exptevents()
    
        if correction_method is None:
            return baphy_events
        elif correction_method == 'baphy':
            return [io.baphy_align_time_baphyparm(ev) for ev in baphy_events]
        elif correction_method == 'openephys':
            trial_starts = self.get_trial_starts('openephys')
            raw_rasterfs = self.get_raw_sampling_rate('openephys')
            return [io.baphy_align_time_openephys(ev, ts, rfs, rasterfs) 
                    for ev, ts, rfs in zip(baphy_events,trial_starts, raw_rasterfs)]
        elif correction_method == 'spikes':
            spikedata = self._get_spikes()
            exptevents = [io.baphy_align_time(ev, spd['sortinfo'], spd['spikefs'], kw['rasterfs'])[0] for (ev, spd)
                                in zip(baphy_events, spikedata)]
            return exptevents
        mesg = 'Unsupported correction method "{correction_method}"'
        raise ValueError(mesg)

    @copying_lru_cache(maxsize=128)
    def get_behavior_events(self, correction_method=None, **kw):

        if correction_method is None:
            correction_method = self.correction_method

        exptparams = self.get_baphy_exptparams()
        exptevents = self.get_baphy_events(correction_method=correction_method, **kw)
        behavior_events = []
        for ep, ev in zip(exptparams, exptevents):
            try:
                # TODO support for different Behavior Objects
                # consider also finding invalid trials and epochs within trials
                ev = behavior.create_trial_labels(ep, ev)
                behavior_events.append(behavior.mark_invalid_trials(ep, ev, **kw))
            except KeyError:
                # passive file, just return exptevents df
                behavior_events.append(ev)

        return behavior_events

    @copying_lru_cache(maxsize=128)
    def get_baphy_exptevents(self, raw=False):
        exptevents = [ep[-1] for ep in self._get_baphy_parameters(userdef_convert=False)]
        if raw:
            return exptevents
        else: 
            # do basic preprocessing of baphy events. For example, in PTD data,
            # when OverlapRefTar = Yes, two overlapping events are created (Ref / Target).
            # We want to merge these for the sake of behavior analysis etc.
            exptparams = self.get_baphy_exptparams()
            
            # deal with overlapping ref/tar epochs
            OverlapRefTar = [e['TrialObject'][1].get('OverlapRefTar','No') for e in exptparams]
            exptevents = [_merge_refTar_epochs(e, o) for e, o in zip(exptevents, OverlapRefTar)]
            
            # truncate FA trials
            log.info('Remove post-response events')
            # typically, this isn't an issue with OEP recordings. However, for MANTA recordings,
            # where spikes aren't collected continuously, not removing these post-response epochs
            # will mess up time alignment, which occurs within `get_baphy_events`, which calls
            # baphy_io.baphy_align_time -- crh 01.04.2021
            exptevents = [_truncate_trials(e) for e in exptevents]

            #other preprocessing steps...

            return exptevents

    @copying_lru_cache(maxsize=128)
    def get_baphy_exptparams(self):
        exptparams = [ep[1] for ep in self._get_baphy_parameters(userdef_convert=False)]
        return exptparams

    @copying_lru_cache(maxsize=128)
    def get_baphy_globalparams(self):
        globalparams = [ep[0] for ep in self._get_baphy_parameters(userdef_convert=False)]
        return globalparams

    def get_recording_uri(self, generate_if_missing=True, cellid=None, loadkey=None, recache=False, **kwargs):
        '''
        This is where the kwargs contents are critical (for generating the correct 
        recording file hash)

        TODO: loadkey parsing?
        '''
        if loadkey is not None:
            kwargs = baphy_io.parse_loadkey(loadkey=loadkey, batch=self.batch)
        else:
            kwargs = io.fill_default_options(kwargs)

        # add BAPHYExperiment version to recording options
        # kwargs.update({'version': 'BAPHYExperiment.2'})
        kwargs.update({'version': 'BAPHYExperiment.3'}) # version 3 added pupil extras to recording signals

        # add parmfiles / cells_to_load list - these are unique ids for the recording
        kwargs.update({'mfiles': [str(i) for i in self.parmfile]})
        kwargs.update({'cell_list': self.cells_to_load})

        # add batch to cache recording in the correct location
        kwargs.update({'siteid': self.siteid})
        kwargs.update({'batch': self.batch})

        # see if can load from cache, if not, call generate_recording
        data_file = recording_filename_hash(
                self.experiment[0][:7], kwargs, uri_path=get_setting('NEMS_RECORDINGS_DIR'))

        use_API = get_setting('USE_NEMS_BAPHY_API')

        if use_API:
            _, f = os.path.split(data_file)
            host = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT'))
            data_uri = host + '/recordings/' + str(self.batch) + '/' + f
            log.info('Cached recording: %s', data_uri)
        else:
            if ((not os.path.exists(data_file)) & generate_if_missing) | recache:
                # strip unhashable fields (list) from the kwargs (to play nice with lru_caching / copying)
                del kwargs['mfiles']; del kwargs['cell_list']
                rec = self.generate_recording(**kwargs)
                log.info('Caching recording: %s', data_file)
                rec.save(data_file)
            else:
                log.info('Cached recording: %s', data_file)
            data_uri = data_file

        return data_uri


    @lru_cache(maxsize=128)
    def get_recording(self, recache=False, generate_if_missing=True, loadkey=None, **kwargs):
        '''
        Steps to building a recording:
            1) Figure out which signals to load
            2) For each parmfile
                load the (correctly aligned events)
                load the appropriate time series (signals)
                create signals
                append time for each parmfile
            3) Package all signals into recording
        '''
        if loadkey is not None:
            kwargs = baphy_io.parse_loadkey(loadkey=loadkey, batch=self.batch)

        # see if can load from cache, if not, call generate_recording
        data_file = self.get_recording_uri(generate_if_missing=False, recache=recache, **kwargs)
        
        # take care of the recache inside get_recording_uri 
        # do we even need this if/else block here??? crh 01.22.2021
        if (not os.path.exists(data_file)): # | recache:
            kwargs.update({'mfiles': None})
            rec = self.generate_recording(**kwargs)
            log.info('Caching recording: %s', data_file)
            rec.save(data_file)
        else:
            log.info('Cached recording found')
            rec = load_recording(data_file)

        rec.meta['cells_to_extract'] = self.cells_to_extract

        return rec

    def generate_recording(self, rawchans=None, **kwargs):
        rec_name = self.experiment[0][:7]     
        
        # figure out signals to load, then load them (as lists)
        raw = kwargs.get('raw', False)
        resp = kwargs.get('resp', False)
        pupil = kwargs.get('pupil', False)
        stim = kwargs.get('stim', False)
        
        # default sampling rates depend on what signals are loaded
        if raw:
            kwargs['rasterfs'] = kwargs.get('rasterfs', 400)
        else:
            kwargs['rasterfs'] = kwargs.get('rasterfs', 100)
            
        # stim, lfp, photometry etc.

        # get correction method
        correction_method = self.correction_method

        # get raw exptevents
        raw_exptevents = self.get_baphy_exptevents()

        # load aligned baphy events
        if raw:
            exptevents = self.get_baphy_events(correction_method='openephys', **kwargs)
        elif self.behavior:
            exptevents = self.get_behavior_events(correction_method=correction_method, **kwargs)
        else:
            exptevents = self.get_baphy_events(correction_method=correction_method, **kwargs)

        # trim epoch names, remove behavior columns labels etc.
        exptparams = self.get_baphy_exptparams()
        globalparams = self.get_baphy_globalparams()
    
        # get good/bad trials, if database
        goodtrials = [None] * len(self.parmfile)

        d = db.get_batch_cell_data(batch=self.batch, cellid=self.siteid, label='parm',
                                   rawid=self.rawid)
        if len(d) > 0:
            for i, parm in enumerate(self.parmfile):
                trialcount = exptevents[i][exptevents[i]['name'].str.startswith('TRIALSTART')].shape[0]

                s_goodtrials = d.loc[d['parm'].str.strip('.m')==str(parm).strip('.m'), 'goodtrials'].values[0]
                if (s_goodtrials is not None) and (len(s_goodtrials)>0):
                    log.info("goodtrials not empty: %s", s_goodtrials)
                    s_goodtrials = re.sub("[\[\]]", "", s_goodtrials)
                    g = re.split('" "|,',s_goodtrials)  #Split by space or by comma
                    _goodtrials = np.zeros(trialcount, dtype=bool)

                    for b in g:
                        b1 = b.split(":")
                        if (len(b1) == 1) or (len(b1[1])==0):
                            # single trial in list, simulate colon syntax
                            b1[1] = str(trialcount+1)
                        _goodtrials[(int(b1[0])-1):int(b1[1])] = True

                    goodtrials[i] = list(_goodtrials)
        
        baphy_events = [baphy_events_to_epochs(bev, parm, gparm, i, goodtrials=gtrials, **kwargs) for i, (bev, parm, gparm, gtrials) in enumerate(zip(exptevents, exptparams, globalparams, goodtrials))]

        # add speciality parsing of baphy_events for each parmfile. For example, tweaking epoch names etc. 
        for i, (bev, param) in enumerate(zip(baphy_events, exptparams)):
            if param['runclass']=='TBP':
                # for TBP, we need to update events to tweak certain target names if they belong to targetDistSet 2, i.e. reminder targets
                # also need to update the soundObject names accordingly in exptparams. 
                # NOTE: This will not update the result returned by self.get_baphy_exptparams, 
                # but it will update this local exptparams that gets used for signal generation
                baphy_events[i], exptparams[i] = runclass.TBP(bev, param)
            if param['runclass']=='CPN':
                #ToDo: format epochs for clarity and define if AllPermutations or Triplets
                baphy_events[i], exptparams[i] = runclass.CPN(bev, param)
                
        signals = {}
        
        if raw:
            if rawchans is None:
                rawchans = np.arange(globalparams[0]['NumberOfElectrodes'])
            fs = kwargs['rasterfs']
            d, t0 = self.get_continuous_data(chans=rawchans, rasterfs=fs)
            #import pdb;pdb.set_trace()
            for i in range(len(baphy_events)):
                #s = np.round(baphy_events[i].loc[:,'start'] * float(fs)) - np.round(t0[i]/fs)
                #e = np.round(baphy_events[i].loc[:,'end'] * float(fs)) - np.round(t0[i]/fs)
                #diff = np.round((baphy_events[i].loc[:,'end'] - baphy_events[i].loc[:,'start']) * float(fs))
                
                baphy_events[i].loc[:,'start'] -= np.round(t0[i])/fs
                baphy_events[i].loc[:,'end'] -= np.round(t0[i])/fs
                
            raw_sigs = [nems.signal.RasterizedSignal(
                        fs=kwargs['rasterfs'], data=r,
                        name='raw', recording=rec_name, chans=[str(c+1) for c in rawchans],
                        epochs=e)
                        for e, r in zip(baphy_events, d)]
            signals['raw'] = nems.signal.RasterizedSignal.concatenate_time(raw_sigs)
            
        if resp:
            spike_dicts = self.get_spike_data(raw_exptevents, **kwargs)
            resp_sigs = [nems.signal.PointProcess(
                         fs=kwargs['rasterfs'], data=sp,
                         name='resp', recording=rec_name, chans=list(sp.keys()),
                         epochs=baphy_events[i]) 
                         for i, sp in enumerate(spike_dicts)]

            for i, r in enumerate(resp_sigs):
                if i == 0:
                    signals['resp'] = r
                else:
                    signals['resp'] = signals['resp'].append_time(r)
            
        if pupil:

            def check_length(ps, rs):
                for i, (p, r) in enumerate(zip(ps, rs)):
                    rlen = r.ntimes
                    plen = p.as_continuous().shape[1]
                    if plen > rlen:
                        ps[i] = p._modified_copy(p.as_continuous()[:, 0:-(plen-rlen)])
                    elif rlen > plen:
                        pcount = p.as_continuous().shape[0]
                        ps[i] = p._modified_copy(np.append(p.as_continuous(), 
                                                np.ones([pcount, rlen - plen]) * np.nan, axis=1))
                return ps

            p_traces = self.get_pupil_trace(exptevents=exptevents, **kwargs)
            if np.all([type(p_traces[i][0]) is not np.ndarray for i in range(len(p_traces))]):
                # multiple 'pupil signals'
                # one always has to be the pupil trace itself
                pupil_sigs = [nems.signal.RasterizedSignal(
                        fs=kwargs['rasterfs'], data=p[0]['pupil'],
                        name='pupil', recording=rec_name, chans=['pupil'],
                        epochs=baphy_events[i])
                        for (i, p) in enumerate(p_traces)]

                # the rest are "pupil" extras to be packed into a single signal
                extra_sigs = [sig for sig in p_traces[0][0].keys() if sig!='pupil']        
                extra_sigs = [nems.signal.RasterizedSignal(
                            fs=kwargs['rasterfs'], data=np.concatenate([p[0][sig] for sig in extra_sigs], axis=0),
                            name='pupil_extras', recording=rec_name, chans=extra_sigs,
                            epochs=baphy_events[i])
                            for (i, p) in enumerate(p_traces)]
                # make sure each pupil signal is the same len as resp, if resp exists
                if resp:
                    pupil_sigs = check_length(pupil_sigs, resp_sigs)
                    extra_sigs = check_length(extra_sigs, resp_sigs)

                signals['pupil_extras'] = nems.signal.RasterizedSignal.concatenate_time(extra_sigs)
                signals['pupil'] = nems.signal.RasterizedSignal.concatenate_time(pupil_sigs)

            else:
                # at least one of the pupil files doesn't have "extras". Just load pupil
                pupil_sigs = [nems.signal.RasterizedSignal(
                            fs=kwargs['rasterfs'], data=p[0],
                            name='pupil', recording=rec_name, chans=['pupil'],
                            epochs=baphy_events[i]) if type(p[0]) is np.ndarray else 
                            nems.signal.RasterizedSignal(
                            fs=kwargs['rasterfs'], data=p[0]['pupil'],
                            name='pupil', recording=rec_name, chans=['pupil'],
                            epochs=baphy_events[i])
                            for (i, p) in enumerate(p_traces)]
                # make sure each pupil signal is the same len as resp, if resp exists
                if resp:
                    pupil_sigs = check_length(pupil_sigs, resp_sigs)

                signals['pupil'] = nems.signal.RasterizedSignal.concatenate_time(pupil_sigs)

        if stim:
            #import pdb; pdb.set_trace()
            stim_sigs = [nems.signal.TiledSignal(
                            data=io.baphy_load_stim(exptparams[i], str(p), epochs=baphy_events[i], **kwargs)[0],
                            fs=kwargs['rasterfs'], name='stim',
                            epochs=baphy_events[i], recording=rec_name)
                        for i, p in enumerate(self.parmfile)]

            signals['stim'] = nems.signal.TiledSignal.concatenate_time(stim_sigs)

        if len(signals)==0:
            # make a dummy signal
            fs = kwargs['rasterfs']
            #import pdb; pdb.set_trace()
            file_sigs = [nems.signal.RasterizedSignal(
                          fs=fs, data=np.zeros((1,int(np.max(baphy_events[i]['end'])*fs)))+i,
                          name='fileidx', recording=rec_name, chans=['fileidx'],
                          epochs=baphy_events[i])
                          for (i, p) in enumerate(baphy_events)]
            signals['fileidx'] = nems.signal.RasterizedSignal.concatenate_time(file_sigs)

        meta = kwargs
        meta['files'] = [str(p) for p in self.parmfile]
        rec = nems.recording.Recording(signals=signals, meta=meta, name=rec_name)

        return rec

    # ==================== DATA EXTRACTION METHODS =====================

    def get_continuous_data(self, chans, rasterfs=None):
        '''
        WARNING: This is a beta method. The interface and return value may
        change.
        chans (list or numpy slice): which electrodes to load data from
        '''
        # get filenames (TODO: can this be sped up?)
        #with tarfile.open(self.openephys_tarfile, 'r:gz') as tar_fh:
        #    log.info("Finding filenames in tarfile...")
        #    filenames = [f.split('/')[-1] for f in tar_fh.getnames()]
        #    data_files = sorted([f for f in filenames if 'CH' in f], key=len)

        continuous_data_list = []
        timestamp0_list = []
        
        # iterate through each openephys_folder
        for openephys_folder,tarfile_fullpath,tarfile_relpath in \
               zip(self.openephys_folder,self.openephys_tarfile,self.openephys_tarfile_relpath):
            # Use xml settings instead of the tar file. Much faster. Also, takes care
            # of channel mapping (I think)
            recChans, _ = oes.GetRecChs(str(openephys_folder / 'settings.xml'))
            connector = [i for i in recChans.keys()][0]
            #import pdb; pdb.set_trace()
            # handle channel remapping
            info = oes.XML2Dict(str(openephys_folder / 'settings.xml'))
            mapping = info['SIGNALCHAIN']['PROCESSOR']['Filters/Channel Map']['EDITOR']
            mapping_keys = [k for k in mapping.keys() if 'CHANNEL' in k]
            for k in mapping_keys:
                ch_num = mapping[k].get('Number')
                if ch_num in recChans[connector]:
                    recChans[connector][ch_num]['name_mapped'] = 'CH'+mapping[k].get('Mapping')

            recChans = [recChans[connector][i]['name_mapped'] \
                                for i in recChans[connector].keys()]
            data_files = [connector + '_' + c + '.continuous' for c in recChans]
            all_chans = np.arange(len(data_files))
            idx = all_chans[chans].tolist()
            selected_data = np.take(data_files, idx)
            continuous_data = []
            timestamp0 = []
            for filename in selected_data:
                full_filename = openephys_folder / filename
                if os.path.isfile(full_filename):
                    log.info('%s already extracted, load faster...', filename)
                    data = io.load_continuous_openephys(str(full_filename))
                    if rasterfs is None:
                        continuous_data.append(data['data'][np.newaxis, :])
                        timestamp0.append(data['timestamps'][0])
                    else:
                        resample_new_size = int(np.round(len(data['data']) * rasterfs / int(data['header']['sampleRate'])))
                        d = scipy.signal.resample(data['data'], resample_new_size)
                        continuous_data.append(d[np.newaxis, :])
                        timestamp0.append(data['timestamps'][0] * rasterfs / int(data['header']['sampleRate']))
                else:
                    with tarfile.open(tarfile_fullpath, 'r:gz') as tar_fh:
                        log.info("Extracting / loading %s...", filename)
                        full_filename = tarfile_relpath / filename
                        with tar_fh.extractfile(str(full_filename)) as fh:
                            data = io.load_continuous_openephys(fh)
                            if rasterfs is None:
                                continuous_data.append(data['data'][np.newaxis, :])
                                timestamp0.append(data['timestamps'][0])
                            else:
                                resample_new_size = int(np.round(len(data['data']) * rasterfs / int(data['header']['sampleRate'])))
                                d = scipy.signal.resample(data['data'], resample_new_size)
                                continuous_data.append(d[np.newaxis, :])
                                timestamp0.append(data['timestamps'][0] * rasterfs / int(data['header']['sampleRate']))

            continuous_data_list.append(np.concatenate(continuous_data, axis=0))
            timestamp0_list.append(timestamp0[0])

        return continuous_data_list, timestamp0_list
    

    def get_spike_data(self, exptevents, **kw):
        #for i, f in enumerate(self.parmfile):
        #    fn = str(f).split('/')[-1]
        #    exptevents[i].to_pickle('/auto/users/hellerc/code/scratch/exptevents_io_{}.pickle'.format(fn))
        spikedata = self._get_spikes()
        if self.correction_method == 'spikes':
            spikedicts = []
            for file_ind in range(len(exptevents)):
                # (ev, (sp, fs)) in zip(exptevents, spikes_fs):
                spikedict = {}
                spiketimes = []
                unit_names = []
                _, spiketimes, unit_names = io.baphy_align_time(exptevents[file_ind], spikedata[file_ind]['sortinfo'],
                                                                  spikedata[file_ind]['spikefs'], kw['rasterfs'])

                if self.cells_to_load is not None:
                    for i in range(len(self.cells_to_load)):
                        if self.channels_to_load is not None:
                            # Use channel_to_load and units_to_load
                            chan_unit_str = '{:02d}-{}'.format(self.channels_to_load[i], self.units_to_load[i])
                        else:
                            # Use cells_to_load, strip out siteid
                            chan_unit_str = self.cells_to_load[i][self.cells_to_load[i].find('-') + 1:]
                        try:
                            mi = unit_names.index(chan_unit_str)
                        except ValueError:
                            #import pdb; pdb.set_trace()
                            raise RuntimeError(
                                f'{chan_unit_str} was asked to be loaded, but wasn''t found in the spk.mat file')
                        spikedict[self.cells_to_load[i]] = spiketimes[mi]
                else:
                    for i, unit_name in enumerate(unit_names):
                        spikedict[self.siteid + "-" + unit_name] = spiketimes[i]
                spikedicts.append(spikedict)
        else:
            raise NotImplementedError
        return spikedicts


    def get_pupil_trace(self, exptevents=None, **kwargs):
        if exptevents is not None:
            return [io.load_pupil_trace(str(p), exptevents=e, **kwargs) for e, p in zip(exptevents, self.pupilfile)]
        else:
            return [io.load_pupil_trace(str(p), **kwargs) for p in self.pupilfile]

    # ===================================================================

    # ==================== BEHAVIOR METRIC METHODS ======================
    def get_behavior_performance(self, trials=None, tokens=None, **kwargs):
        """
        Return dictionary of behavior performance metrics
            If trial is not None, only compute behavior metrics
            over specified BAPHY trials (counting from first active parmfile)

            If tokens is not None, only compute metrics 
            over specified sound tokens (counting from first active parmfile)

            Use kwargs to specify valid trials (see behavior.compute_metrics 
                for docs). Also, make sure includes sampling rate, otherwise
                won't know how to align time stamps
        """
        if not self.behavior:
            raise ValueError("No behavior detected in this experiment")

        # add dummy rasterfs for creating aligned timestamps. Not critically important what
        # this is for the behavior metrics
        rasterfs = kwargs.get('rasterfs', 100)
        kwargs['rasterfs'] = rasterfs

        # get aligned exptevents for behavior files
        events = self.get_behavior_events(correction_method=self.correction_method, **kwargs)
        params = self.get_baphy_exptparams()

        for i, (bev, param) in enumerate(zip(events, params)):
            if param['runclass']=='TBP':
                # for TBP, we need to update events to tweak certain target names if they belong to targetDistSet 2, i.e. reminder targets
                # also need to update the soundObject names accordingly in exptparams. 
                # NOTE: This will not update the result returned by self.get_baphy_exptparams, 
                # but it will update this local exptparams that gets used for signal generation
                events[i], params[i] = runclass.TBP(bev, param)

        behave_file = [True if (p['BehaveObjectClass'] != 'Passive') else False for p in params]
        if len(behave_file) > 1:
            events = [e for i, e in enumerate(events) if behave_file[i]]
        elif behave_file[0] == True:
            pass
        # assume params same for all files. This is a bit kludgy... Think it
        # should work?
        beh_params = params[np.min(np.where(behave_file)[0])]
        
        # stack events
        events = self._stack_events(events)

        # run behavior analysis
        kwargs.update({'trial_numbers': trials, 'sound_trial_numbers': tokens})
        
        metrics = behavior.compute_metrics(beh_params, events, **kwargs)    

        return metrics

    def _stack_events(self, exptevents):
        """
        Merge list of exptevents into single df for behavior calculations
        """
        epochs = []
        for i, ev in enumerate(exptevents):
            if i == 0 :
                epochs.append(ev)
            else:
                ev['end'] += epochs[-1]['end'].max()
                ev['start'] += epochs[-1]['end'].max()
                ev['Trial'] += epochs[-1]['Trial'].max()
                ev.loc[ev.soundTrialidx!=0, 'soundTrialidx'] += epochs[-1]['soundTrialidx'].max()
                epochs.append(ev)
        return pd.concat(epochs, ignore_index=True)
        '''
        offset = 0
        trial_offset = 0
        token_offset = 0
        epochs = []
        for ev in exptevents:
            if (ev['start'].min() == offset) & (offset != 0):
                offset += ev['end'].max()
                trial_offset += ev['Trial'].max()
                token_offset += ev['soundTrialidx'].max()
            else:
                ev['end'] += offset
                ev['start'] += offset
                ev['Trial'] += trial_offset
                ev.loc[ev.soundTrialidx!=0, 'soundTrialidx'] += token_offset
                offset += ev['end'].max()
                trial_offset += ev['Trial'].max()
                token_offset += ev['soundTrialidx'].max()
            epochs.append(ev)
        return pd.concat(epochs, ignore_index=True)
        '''

    # ======================= PLOTTING METHODS ======================
    def plot_RT_histogram(self, trials=None, tokens=None, bins=None, ax=None, **options):
        """
        extract RTs and pass to behavior_plots.plot_RT_histogram fn

        options:    behavioral trial options (e.g. which trials are invalid -- see behavior.compute_metrics)
        trials:     BAPHY trials (list/array of ints) over which to compute metrics. If None, use all valid trials specified in options dict.
        tokens:     epoch tokens (list/array of ints) over which to compute metrics. If None, use all valid sound tokens according to options dict.
        bins:       int or range for specifying histogram resolution. If int, 
                        histogram will be plotted from 0 to 2 sec in "bins" number of time bins
        """
        bev = self.get_behavior_events(**options)
        bev = self._stack_events(bev)
        options.update({'trial_numbers': trials, 'sound_trial_numbers': tokens})
        bev = behavior.mark_invalid_trials(self.get_baphy_exptparams()[0], bev, **options)

        # get RTs for each sound
        tars = bev.loc[bev.name.str.contains('Target') & (bev.invalidSoundTrial == False), 'name'].unique().tolist()
        catch =bev.loc[bev.name.str.contains('Catch') & (bev.invalidSoundTrial == False), 'name'].unique().tolist()
        ref = bev.loc[bev.name.str.contains('Reference') & (bev.invalidSoundTrial == False), 'name'].unique().tolist()
        epochs = tars + catch
        keys = [k.split(' , ')[1] for k in epochs]
        RTs = {k: [] for k in keys + ['Reference']}
        for e in epochs + ref:
            if e in ref:
                RTs['Reference'].extend(bev.loc[(bev.name==e) & (bev.invalidSoundTrial == False), 'RT'])
            else:
                RTs[e.split(' , ')[1]].extend(bev.loc[(bev.name==e) & (bev.invalidSoundTrial == False), 'RT'])
        
        perf = self.get_behavior_performance(trials=trials, tokens=tokens, **options)
        di = perf['DI']
        ax = bplot.plot_RT_histogram(RTs, bins=bins, DI=di, ax=ax)

        return ax

    # ===================================================================
    # Methods below this line just pass through to the functions for now.
    def _get_baphy_parameters(self, userdef_convert=False):
        '''
        Parameters
        ----------
        userdef_convert : bool
            If True, find all instances of the `UserDefinableFields` key in the
            BAPHY parms data and convert them to dictionaries. See
            :func:`baphy_convert_user_definable_fields` for example.
        '''
        # Returns tuple of global, expt and events
        result = [io.baphy_parm_read(p) for p in self.parmfile]
        if userdef_convert:
            [io.baphy_convert_user_definable_fields(r) for r in result]
        return result

    def _get_spikes(self):
        return [io.baphy_load_spike_data_raw(str(s)) for s in self.spikefile]


# ==============  epoch manipulation functions  ================

def baphy_events_to_epochs(exptevents, exptparams, globalparams, fidx, goodtrials=None, **options):
    """
    Modify exptevents dataframe for nems epochs.
    This includes cleaning up event names and moving behavior
    labels to name columnn, if they exist. This is basically
    just a (slightly) cleaned up version of baphy_load_dataset.

    goodtrials (added 08.08.2021) -- which baphy trials to keep
    """
    epochs = []

    log.info('Creating trial epochs')
    trial_epochs = _make_trial_epochs(exptevents, exptparams, fidx, **options)
    epochs.append(trial_epochs)

    log.info('Creating stim epochs')
    stim_epochs = _make_stim_epochs(exptevents, exptparams, **options)
    epochs.append(stim_epochs)

    log.info('Creating Light epochs')
    light_epochs = _make_light_epochs(exptevents, exptparams, **options)
    #if light_epochs != []:
    if light_epochs is not None:
        epochs.append(light_epochs)

    # this step includes removing post lick events for 
    # active files
    behavior = False
    if exptparams['BehaveObjectClass'] not in ['ClassicalConditioning', 'Passive']:
        log.info('Creating behavior epochs')
        behavior = True
        behavior_epochs = _make_behavior_epochs(exptevents, exptparams, **options)
        epochs.append(behavior_epochs)
        
    epochs = pd.concat(epochs, ignore_index=True)#, sort=False)
    file_start_time = epochs['start'].min()
    file_end_time = epochs['end'].max()
    te = pd.DataFrame(index=[0], columns=(epochs.columns))
    if behavior:
        # append ACTIVE epoch
        te.loc[0, 'start'] = file_start_time
        te.loc[0, 'end'] = file_end_time
        te.loc[0, 'name']= 'ACTIVE_EXPERIMENT'

    else:
        # append PASSIVE epoch
        te.loc[0, 'start'] = file_start_time
        te.loc[0, 'end'] = file_end_time
        te.loc[0, 'name']= 'PASSIVE_EXPERIMENT'

    epochs = epochs.append(te, ignore_index=True)
    # append file name epoch
    mfilename = os.path.split(globalparams['tempMfile'])[-1].split('.')[0]

    te.loc[0, 'start'] = file_start_time
    te.loc[0, 'end'] = file_end_time
    te.loc[0, 'name'] = 'FILE_'+mfilename

    epochs = epochs.append(te, ignore_index=True)
    
    epochs = epochs.sort_values(by=['start', 'end'], 
                    ascending=[1, 0]).reset_index()
    epochs = epochs.drop(columns=['index'])

    epochs = _trim_epoch_columns(epochs)

    # touch up last trial
    # edit end values to make round according to rasterfs
    final_trial_end = np.floor(epochs[epochs.name=='TRIAL'].end.max() * options['rasterfs']) / options['rasterfs']
    end_events = (epochs['end'] >= final_trial_end)
    epochs.loc[end_events, 'end'] = final_trial_end

    start_events = (epochs['start'] >= final_trial_end)
    epochs = epochs[~start_events]

    # get rid of weird floating point precision 
    epochs.at[:, 'start'] = [np.round(x, 5) for x in epochs['start'].values]
    epochs.at[:, 'end'] = [np.round(x, 5) for x in epochs['end'].values]

    # Remove any duplicate epochs (that are getting created somewhere???)
    epochs = epochs.drop_duplicates()

    # if goodtrials exist, only keep epochs within "good" baphy trials
    if goodtrials is not None:
        bad_bounds = epochs[epochs.name=='TRIAL'][~np.array(goodtrials)][['start', 'end']].values
        all_bounds = epochs[['start','end']].values

        bad_epochs = ep.epoch_contained(all_bounds, bad_bounds)
        epochs = epochs.drop(epochs.index[bad_epochs])

    return epochs


def _make_trial_epochs(exptevents, exptparams, fidx=None, **options):
    """
    Define baphy trial epochs
    """
    # sort of hacky. This means that if behavior classification 
    # was run and it's NOT classical conditioning, you should truncate
    # trials after licks
    #remove_post_lick = ('soundTrial' in exptevents.columns) & \
    #                        (exptparams['BehaveObjectClass'] != 'ClassicalConditioning')

    trial_events = exptevents[exptevents['name'].str.startswith('TRIALSTART')].copy()
    end_events = exptevents[exptevents['name'].str.startswith('TRIALSTOP')]
    trial_events.at[:, 'end'] = end_events['start'].values
    trial_events.at[:, 'name'] = 'TRIAL'

    #if remove_post_lick:
    #   trial_events =  _remove_post_lick(trial_events, exptevents, **options)
    #   trial_events =  _remove_post_stim_off(trial_events, exptevents, **options)

    trial_events = trial_events.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    baphy_trials = trial_events.copy()
    names = [f'BAPHYTRIAL{i+1}_FILE{fidx+1}' for i in range(baphy_trials.shape[0])]
    baphy_trials.name = names
    trial_events = pd.concat([trial_events, baphy_trials]).sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    trial_events = trial_events.drop(columns=['index'])
    trial_events = trial_events.drop(columns=['level_0'])

    return trial_events


def _make_stim_epochs(exptevents, exptparams, **options):
    """
    Define epochs for each unique stimulus that was played (if FA trial,
    stim might not play so we should truncate that epoch  at STIM,OFF /
    TRIALSTOP / OUTCOME,VEARLY(?) / OUTCOME,EARLY(?)
    if that's the case)
    """
    # sort of hacky. This means that if behavior classification 
    # was run and it's NOT classical conditioning, you should truncate
    # trials after licks
    remove_post_lick = ('soundTrial' in exptevents.columns) & \
                            (exptparams['BehaveObjectClass'] != 'ClassicalConditioning')

    # reference events (including spont)
    ref_tags = exptevents[exptevents.name.str.contains('Reference') & \
                            (~exptevents.name.str.contains('PreStimSilence') & \
                            ~exptevents.name.str.contains('PostStimSilence'))].name.unique()

    ref_s_tags = exptevents[exptevents.name.str.contains('Reference') & \
                            exptevents.name.str.contains('PreStimSilence')].name.unique()
    ref_e_tags = exptevents[exptevents.name.str.contains('Reference') & \
                            exptevents.name.str.contains('PostStimSilence')].name.unique()
    ref_starts = exptevents[exptevents.name.isin(ref_s_tags)].copy()
    ref_ends = exptevents[exptevents.name.isin(ref_e_tags)].copy()

    ref_events = exptevents[exptevents.name.isin(ref_tags)].copy()
    # new_tags = ['STIM_'+t.split(',')[1].replace(' ', '') for t in ref_events.name]
    new_tags = [f"STIM_{'-'.join([b.strip().replace(' ', '') for b in t.split(',')[1:-1]])}" for t in ref_events.name]

    #import pdb; pdb.set_trace()

    ref_events.at[:, 'name'] = new_tags
    ref_events.at[:, 'start'] = ref_starts.start.values
    ref_events.at[:, 'end'] = ref_ends.end.values
    ref_events2 = ref_events.copy()
    ref_events2.at[:, 'name'] = 'REFERENCE'

    ref_events = pd.concat([ref_events, ref_events2], ignore_index=True)

    # target events (including spont)
    tar_tags = exptevents[exptevents.name.str.contains('Target') & \
                            ~exptevents.name.str.contains('Silence')].name.unique()
    tar_s_tags = exptevents[exptevents.name.str.contains('Target') & \
                            exptevents.name.str.contains('PreStimSilence')].name.unique()
    tar_e_tags = exptevents[exptevents.name.str.contains('Target') & \
                            exptevents.name.str.contains('PostStimSilence')].name.unique()
    tar_starts = exptevents[exptevents.name.isin(tar_s_tags)].copy()
    tar_ends = exptevents[exptevents.name.isin(tar_e_tags)].copy()
    
    tar_events = exptevents[exptevents.name.isin(tar_tags)].copy()
    new_tags = ['TAR_' + t.split(',')[1].replace(' ', '') for t in tar_events.name]
    tar_events.at[:, 'name'] = new_tags
    tar_events.at[:, 'start'] = tar_starts.start.values
    tar_events.at[:, 'end'] = tar_ends.end.values
    tar_events2 = tar_events.copy()
    tar_events2.at[:, 'name'] = 'TARGET'
    tar_events = pd.concat([tar_events, tar_events2], ignore_index=True)

    # Catch events (including spont)
    cat_tags = exptevents[exptevents.name.str.contains('Catch') & \
                            ~exptevents.name.str.contains('Silence')].name.unique()
    cat_s_tags = exptevents[exptevents.name.str.contains('Catch') & \
                            exptevents.name.str.contains('PreStimSilence')].name.unique()
    cat_e_tags = exptevents[exptevents.name.str.contains('Catch') & \
                            exptevents.name.str.contains('PostStimSilence')].name.unique()
    cat_starts = exptevents[exptevents.name.isin(cat_s_tags)].copy()
    cat_ends = exptevents[exptevents.name.isin(cat_e_tags)].copy()
    
    cat_events = exptevents[exptevents.name.isin(cat_tags)].copy()
    new_tags = ['CAT_' + t.split(',')[1].replace(' ', '') for t in cat_events.name]
    cat_events.at[:, 'name'] = new_tags
    cat_events.at[:, 'start'] = cat_starts.start.values
    cat_events.at[:, 'end'] = cat_ends.end.values
    cat_events2 = cat_events.copy()
    cat_events2.at[:, 'name'] = 'CATCH'
    cat_events = pd.concat([cat_events, cat_events2], ignore_index=True)

    # pre/post stim events
    sil_tags = exptevents[exptevents.name.str.contains('Silence')].name.unique()
    sil_events = exptevents[exptevents.name.isin(sil_tags)].copy()
    new_tags = [t.split(',')[0].replace(' ', '') for t in sil_events.name]
    sil_events.at[:, 'name'] = new_tags

    # lick events
    lick_events = exptevents[exptevents.name=='LICK']

    #if remove_post_lick:
        # crh 01.05.2021 - this should get taken care of in exptevents loading now
        #ref_events = _remove_post_lick(stim_events, exptevents)
        #cat_events = _remove_post_stim_off(cat_events, exptevents)
        #ref_events = _remove_post_stim_off(ref_events, exptevents)

    # concatenate events together
    stim_events = pd.concat([ref_events, tar_events, cat_events, sil_events, lick_events], ignore_index=True)

    #if remove_post_lick:
    #    stim_events = _remove_post_lick(stim_events, exptevents)
    #    stim_events = _remove_post_stim_off(stim_events, exptevents)

    stim_events = stim_events.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    stim_events = stim_events.drop(columns=['index'])

    return stim_events


def _make_light_epochs(exptevents, exptparams, **options):
    """
    Look for / define light epochs (for optogenetics).
    Quick and dirty, probably needs improvment -- CRH 10.21.2020
    """ 
    light_events = exptevents[exptevents['name'].str.contains('LightStim')].copy()
    
    if light_events.shape[0] >= 1:
        light_events.at[:, 'name'] = 'LIGHTON'
        light_events = light_events.sort_values(
                by=['start', 'end'], ascending=[1, 0]
                ).reset_index()
        light_events = light_events.drop(columns=['index'])
        return light_events

    else:
        return None


def _make_behavior_epochs(exptevents, exptparams, **options):
    """
    Add soundTrial events to the epochs dataframe. i.e. move them into 
    the name column so that they stick around when these columns get
    stripped later for nems packaging.
    """
    if 'soundTrial' not in exptevents.columns:
        raise KeyError("soundTrial not in exptevents. Behavior analysis code \
                        has not been run yet, shouldn't be making \
                            behavior epochs")

    # add column for invalid baphy trials
    exptevents = behavior.mark_invalid_trials(exptparams, exptevents, **options)
    # TODO : make sure this works for different Behavior Objects (eg, ClassicalConditioning)

    baphy_outcomes = ['HIT_TRIAL', 
                      'MISS_TRIAL', 
                      'CORRECT_REJECT_TRIAL',
                      'INCORRECT_HIT_TRIAL',
                      'EARLY_TRIAL',
                      'FALSE_ALARM_TRIAL']
    baphy_outcomes_tf = exptevents.name.isin(baphy_outcomes)
    baphy_behavior_events = exptevents[baphy_outcomes_tf].copy()

    behavior_events = exptevents[(exptevents.soundTrial != 'NULL')].copy()
    tokens = [t.replace('_TRIAL', '_TOKEN') for t in behavior_events.soundTrial]
    behavior_events.loc[:, 'name'] = tokens

    # invalid baphy trial events
    invalid_trials = exptevents[exptevents.invalidTrial].Trial.unique()
    invalid_events = exptevents[baphy_outcomes_tf & exptevents.Trial.isin(invalid_trials)].copy()
    invalid_events.loc[:, 'name'] = 'INVALID_BAPHY_TRIAL'

    behavior_events = pd.concat([baphy_behavior_events,
                                behavior_events, invalid_events], ignore_index=True)

    #behavior_events = _remove_post_lick(behavior_events, exptevents)
    #behavior_events = _remove_post_stim_off(behavior_events, exptevents)

    behavior_events = behavior_events.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    behavior_events = behavior_events.drop(columns=['index'])

    return behavior_events


def _remove_post_lick(events, exptevents, **options):
    # screen for FA / Early trials in which we need to truncate / chop out references
    trunc_trials = exptevents[exptevents.name.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL', 'CORRECT_REJECT_TRIAL'])].Trial.unique()
    #lick_time = exptevents[exptevents.Trial.isin(trunc_trials) & (exptevents.name=='LICK')].start
    #lick_trial = exptevents[exptevents.Trial.isin(trunc_trials) & (exptevents.name=='LICK')].Trial.unique() #.values
    #import pdb; pdb.set_trace()
    #if len(lick_time) != len(trunc_trials):
    #    import pdb;pdb.set_trace()
    #   raise ValueError('More than one lick recorded on a FA trial, whats up??')
    
    for t in trunc_trials:  
        fl = exptevents[(exptevents.Trial==t) & (exptevents.name=='LICK')].iloc[0]['start']
        e = events[events.Trial==t]
        # truncate events that overlapped with lick
        if options.get('truncate_postlick', False):
            events.at[e[e.end > fl].index, 'end'] = fl
            # remove events that started after the lick completely
            events = events.drop(e[(e.start.values > fl) & (e.end.values >= fl)].index)
        else:
            events = events.drop(e[e.end >= fl].index)

    return events


def _truncate_trials(exptevents, **options):
    """
    Catch-all for removing stim epoch data post stim-off. Meant to replace former baphy code that
    deals with MANTA recordings. This operates on exptevents (raw) before running align time.
    For compatibility later on, need to make sure you never orphan a prestim / sound / poststim on
    its own. They need to be in triplets.
    """
    log.info("Removing post-reponse data")
    
    trunc_trials = exptevents[exptevents.name.isin(['STIM,OFF'])].Trial.unique()
    events = exptevents.copy()
    for t in trunc_trials:
        toff = exptevents[(exptevents.Trial==t) & exptevents.name.isin(['STIM,OFF', 'TRIALSTOP'])]['start'].min()
        e = events[events.Trial==t]

        # for "stim" events, make sure to not orphan a 
        # pre/post/stim epoch (keep all, or delete all)
        for idx, r in e.iterrows():
            name = r.loc['name']
            if 'Stim , ' in name:
                if toff > r.loc['start']:
                    # keep all stim events for this sound and
                    # if toff < poststim end, set toff to poststim end
                    if (toff < events.loc[idx+1, 'end']):
                        toff = events.loc[idx+1, 'end']
                    pass
                else:
                    events = events.drop(idx)
                    events = events.drop(idx-1)
                    events = events.drop(idx+1)

        # for remaining events, just brute force truncate
        # truncate partial events
        events.at[e[(e.end > toff) & ~e.name.str.contains('.*Stim.*', regex=True) & ~e.name.str.contains('TRIALSTOP')].index, 'end'] = toff
        if events.loc[(events.Trial==t) & (events.name=='TRIALSTOP'),'start'].min() < toff:
            #print(t)
            #import pdb; pdb.set_trace()
            events.loc[(events.Trial==t) & (events.name=='TRIALSTOP'),'start']=toff
            events.loc[(events.Trial==t) & (events.name=='TRIALSTOP'),'end']=toff
        # remove events that start after toff
        events = events.drop(e[(e.start.values > toff) &
                               (e.end.values >= toff) &
                                ~e.name.str.contains('.*Stim.*') &
                                ~e.name.str.contains('TRIALSTOP')].index)
        
    return events
'''
# this is the old code in baphy.py. Seems to break stuff for some batch 309 because it
# removes some TRIALSTOPS which screws up aligning time. (e.g. see BRT006d-a1)
    keepevents = np.full(len(exptevents), True, dtype=bool)
    TrialCount = exptevents.Trial.max()
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

    log.info("Keeping {0}/{1} events that precede responses"
            .format(np.sum(keepevents), len(keepevents)))
    exptevents = exptevents[keepevents].reset_index()

    return exptevents
'''


def _remove_post_stim_off(events, exptevents, **options):
    # screen for trials where sound was turned off early. These will largey overlap with the events
    # detected by _remove_post_lick, except in weird cases, for example, in catch behaviors where
    # targets can come in the middle of a string of refs, but refs are turned off post target hit

    log.info("Removing data post stim-off")
    trunc_trials = exptevents[exptevents.name.isin(['STIM,OFF'])].Trial.unique()
    for t in trunc_trials:
        toff = exptevents[(exptevents.Trial==t) & (exptevents.name=='STIM,OFF')].iloc[0]['start']
        e = events[events.Trial==t]
        if options.get('truncate_postlick', False):
            # truncate partial events
            events.at[e[e.end > toff].index, 'end'] = toff
            # remove events that start after toff
            events = events.drop(e[(e.start.values > toff) & (e.end.values >= toff)].index)
        else:
            #import pdb; pdb.set_trace()
            # cruder, but simpler. remove events that end after toff
            events = events.drop(e[e.end >= toff].index)
    return events


def _trim_epoch_columns(epochs):
    cols = [c for c in epochs.columns if c not in ['start', 'end', 'name']]
    return epochs.drop(columns=cols)


def _merge_refTar_epochs(exptevents, OverlapRefTar):
    """
    If OverlapRefTar = Yes, merge overlapping reference / target events
    """
    if OverlapRefTar=='No':
        return exptevents
    else:
        # for every target, if there's a preceding reference with an identical start time, merge them
        target_prestims = exptevents[exptevents.name.str.contains('PreStimSilence , .* , Target', regex=True)]
        # get preceding ref stim names (if prestim time stamp matches)
        ref_idx = target_prestims.index-3
        refs = exptevents.loc[ref_idx]
        refs = refs[refs.start.values == target_prestims.start.values]

        # set the poststim duration to the ref post stim (to get rid of long poststim tails)
        exptevents.at[target_prestims.index+2, 'end'] = exptevents.loc[refs.index+2, 'end'].values

        exptevents = exptevents.drop(refs.index)   # drop ref pre stim
        exptevents = exptevents.drop(refs.index+1) # drop ref stim
        exptevents = exptevents.drop(refs.index+2) # drop ref poststim

        return exptevents
