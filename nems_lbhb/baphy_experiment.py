from functools import lru_cache
from pathlib import Path
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
import tarfile
import datetime
import glob
from math import isclose
import copy
from itertools import groupby, repeat, chain, product

from nems_lbhb import runclass
from nems_lbhb import OpenEphys as oe
from nems_lbhb import SettingXML as oes
import pandas as pd
import matplotlib.pyplot as plt
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

    def __init__(self, parmfile=None, batch=None, siteid=None, rawid=None):
        # Make sure that the '.m' suffix is present! In baphy_load data I see
        # that there's care to make sure it's present so I assume not all
        # functions are careful about the suffix.

        # New init options. CRH 2/15/2020. Want to be able to load 
        #   1) multiple parmfiles into single baphy experiment
        #   2) find parmfiles using batch/siteid

        # if don't pass parmfile, must pass both batch and siteid (can 
        # extract individual cellids later if needed). rawid is optional

        if parmfile is None:
            self.batch = batch
            self.siteid = siteid
            self.rawid = rawid
            d = db.get_batch_cell_data(batch=batch, cellid=siteid, label='parm',
                                   rawid=rawid)
            files = list(set(list(d['parm'])))
            files.sort()
            self.parmfile = [Path(f).with_suffix('.m') for f in files]

        elif type(parmfile) is list:
            self.parmfile = [Path(p).with_suffix('.m') for p in parmfile]
            self.siteid = os.path.split(parmfile[0])[-1][:7]
            self.batch = None
        else:
            self.parmfile = [Path(parmfile).with_suffix('.m')]
            self.siteid = os.path.split(parmfile)[-1][:7]
            self.batch = None

        #if np.any([not p.exists() for p in self.parmfile]):
        #    raise IOError(f'Not all parmfiles in {self.parmfile} were found')

        # we can assume all parmfiles come from same folder/experiment (site)
        self.folder = self.parmfile[0].parent
        self.experiment = self.parmfile[0].name.split('_', 1)[0]

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
        path = self.folder / 'raw' / self.experiment
        candidates = list(path.glob(self.experiment_with_runclass + '*'))
        if len(candidates) > 1:
            raise ValueError('More than one candidate found')
        if len(candidates) == 0:
            raise ValueError('No candidates found')
        return candidates[0]

    @property
    @lru_cache(maxsize=128)
    def openephys_tarfile(self):
        '''
        Return path to OpenEphys tarfile containing recordings
        '''
        path = self.folder / 'raw' / self.experiment
        return path.with_suffix('.tgz')

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
        parent = self.openephys_tarfile.parent
        return self.openephys_folder.relative_to(parent)

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
            return io.load_trial_starts_openephys(self.openephys_folder)
        raise ValueError(f'Method "{method}" not supported')

    @lru_cache(maxsize=128)
    def get_baphy_events(self, correction_method='openephys', **kw):
        baphy_events = self.get_baphy_exptevents()
    
        if correction_method is None:
            return baphy_events
        if correction_method == 'baphy':
            return [io.baphy_align_time_baphyparm(ev) for ev in baphy_events]
        if correction_method == 'openephys':
            trial_starts = self.get_trial_starts('openephys')
            return io.baphy_align_time_openephys(baphy_events, trial_starts, **kw)
        if correction_method == 'spikes':
            spikes_fs = self._get_spikes()
            exptevents = [io.baphy_align_time(ev, sp, fs, kw['rasterfs'])[0] for (ev, (sp, fs)) 
                                in zip(baphy_events, spikes_fs)]
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
            OverlapRefTar = [e['TrialObject'][1]['OverlapRefTar'] for e in exptparams]
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

    def get_recording_uri(self, generate_if_missing=True, cellid=None, **kwargs):

        kwargs = io.fill_default_options(kwargs)

        # add BAPHYExperiment version to recording options
        kwargs.update({'version': 'BAPHYExperiment.1'})
        kwargs.update({'mfiles': [str(i) for i in self.parmfile]})

        # add batch to cache recording in the correct location
        kwargs.update({'siteid': self.siteid})
        kwargs.update({'batch': self.batch})

        # see if can load from cache, if not, call generate_recording
        data_file = recording_filename_hash(
                self.experiment[:7], kwargs, uri_path=get_setting('NEMS_RECORDINGS_DIR'))

        use_API = get_setting('USE_NEMS_BAPHY_API')

        if use_API:
            _, f = os.path.split(data_file)
            host = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT'))
            data_uri = host + '/recordings/' + str(self.batch) + '/' + f
            log.info('Cached recording: %s', data_uri)
        else:
            if (not os.path.exists(data_file)) & generate_if_missing:
                kwargs.update({'mfiles': None})
                rec = self.generate_recording(**kwargs)
                log.info('Caching recording: %s', data_file)
                rec.save(data_file)
            else:
                log.info('Cached recording: %s', data_file)
            data_uri = data_file

        return data_uri


    @lru_cache(maxsize=128)
    def get_recording(self, recache=False, generate_if_missing=True, **kwargs):
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
        # see if can load from cache, if not, call generate_recording
        data_file = self.get_recording_uri(generate_if_missing=False, **kwargs)
        
        if (not os.path.exists(data_file)) | recache:
            kwargs.update({'mfiles': None})
            rec = self.generate_recording(**kwargs)
            log.info('Caching recording: %s', data_file)
            rec.save(data_file)
        else:
            log.info('Cached recording found')
            rec = load_recording(data_file)

        return rec

    def generate_recording(self, **kwargs):
        rec_name = self.experiment[:7]      
        
        # figure out signals to load, then load them (as lists)
        resp = kwargs.get('resp', False)
        pupil = kwargs.get('pupil', False)
        stim = kwargs.get('stim', False)

        # stim, lfp, photometry etc.

        # get correction method
        correction_method = self.correction_method

        # get raw exptevents
        raw_exptevents = self.get_baphy_exptevents()
        
        # load aligned baphy events
        if self.behavior:
            exptevents = self.get_behavior_events(correction_method=correction_method, **kwargs)
        else:
            exptevents = self.get_baphy_events(correction_method=correction_method, **kwargs)

        # trim epoch names, remove behavior columns labels etc.
        exptparams = self.get_baphy_exptparams()
        globalparams = self.get_baphy_globalparams()
        baphy_events = [baphy_events_to_epochs(bev, parm, gparm, **kwargs) for (bev, parm, gparm) in zip(exptevents, exptparams, globalparams)]

        #import pdb; pdb.set_trace()

        # add speciality parsing of baphy_events for each parmfile. For example, tweaking epoch names etc. 
        for i, (bev, param) in enumerate(zip(baphy_events, exptparams)):
            if param['runclass']=='TBP':
                # for TBP, we need to update events to tweak certain target names if they belong to targetDistSet 2, i.e. reminder targets
                # also need to update the soundObject names accordingly in exptparams. 
                # NOTE: This will not update the result returned by self.get_baphy_exptparams, 
                # but it will update this local exptparams that gets used for signal generation
                baphy_events[i], exptparams[i] = runclass.TBP(bev, param)
        
    
        signals = {}
        if resp:
            spike_dicts = self.get_spike_data(raw_exptevents, **kwargs)
            spike_dicts = [dict(zip([self.siteid + "-" + x for x in d.keys()], d.values())) for
                                    d in spike_dicts]
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
            
            p_traces = self.get_pupil_trace(exptevents=exptevents, **kwargs)
            pupil_sigs = [nems.signal.RasterizedSignal(
                          fs=kwargs['rasterfs'], data=p[0],
                          name='pupil', recording=rec_name, chans=['pupil'],
                          epochs=baphy_events[i])
                          for (i, p) in enumerate(p_traces)]

            # make sure each pupil signal is the same len as resp, if resp exists
            if resp:
                for i, (p, r) in enumerate(zip(pupil_sigs, resp_sigs)):
                    rlen = r.ntimes
                    plen = p.as_continuous().shape[1]
                    if plen > rlen:
                        pupil_sigs[i] = p._modified_copy(p.as_continuous()[:, 0:-(plen-rlen)])
                    elif rlen > plen:
                        pcount = p.as_continuous().shape[0]
                        pupil_sigs[i] = p._modified_copy(np.append(p.as_continuous(), 
                                                np.ones([pcount, rlen - plen]) * np.nan, axis=1))

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

    def get_continuous_data(self, chans):
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

        # Use xml settings instead of the tar file. Much faster. Also, takes care
        # of channel mapping (I think)
        recChans, _ = oes.GetRecChs(str(self.openephys_folder / 'settings.xml'))
        connector = [i for i in recChans.keys()][0]
        #import pdb; pdb.set_trace()
        # handle channel remapping
        info = oes.XML2Dict(str(self.openephys_folder / 'settings.xml'))
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
        for filename in selected_data:
            full_filename = self.openephys_folder / filename
            if os.path.isfile(full_filename):
                log.info('%s already extracted, load faster...', filename)
                data = io.load_continuous_openephys(str(full_filename))
                continuous_data.append(data['data'][np.newaxis, :])
            else:
                with tarfile.open(self.openephys_tarfile, 'r:gz') as tar_fh:
                    log.info("Extracting / loading %s...", filename)
                    full_filename = self.openephys_tarfile_relpath / filename
                    with tar_fh.extractfile(str(full_filename)) as fh:
                        data = io.load_continuous_openephys(fh)
                        continuous_data.append(data['data'][np.newaxis, :])

        continuous_data = np.concatenate(continuous_data, axis=0)

        return continuous_data         

    def get_spike_data(self, exptevents, **kw):
        #for i, f in enumerate(self.parmfile):
        #    fn = str(f).split('/')[-1]
        #    exptevents[i].to_pickle('/auto/users/hellerc/code/scratch/exptevents_io_{}.pickle'.format(fn))
        spikes_fs = self._get_spikes()
        if self.correction_method == 'spikes':
            spikedicts = [io.baphy_align_time(ev, sp, fs, kw['rasterfs'])[1:3] for (ev, (sp, fs)) 
                                    in zip(exptevents, spikes_fs)]
            spike_dict = []
            for sd in spikedicts:
                units = sd[1]
                spiketimes = sd[0]
                d = {}
                for i, unit in enumerate(units):
                    d[unit] = spiketimes[i]
                spike_dict.append(d)
        else:
            raise NotImplementedError

        return spike_dict

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

def baphy_events_to_epochs(exptevents, exptparams, globalparams, **options):
    """
    Modify exptevents dataframe for nems epochs.
    This includes cleaning up event names and moving behavior
    labels to name columnn, if they exist. This is basically
    just a (slightly) cleaned up version of baphy_load_dataset.
    """

    epochs = []

    log.info('Creating trial epochs')
    trial_epochs = _make_trial_epochs(exptevents, exptparams, **options)
    epochs.append(trial_epochs)

    log.info('Creating stim epochs')
    stim_epochs = _make_stim_epochs(exptevents, exptparams, **options)
    epochs.append(stim_epochs)

    log.info('Creating Light epochs')
    light_epochs = _make_light_epochs(exptevents, exptparams, **options)
    if light_epochs != []:
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

    # Final step, remove any duplicate epochs (that are getting created somewhere???)
    epochs = epochs.drop_duplicates()

    return epochs


def _make_trial_epochs(exptevents, exptparams, **options):
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
    trial_events = trial_events.drop(columns=['index'])

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
    new_tags = ['STIM_'+t.split(',')[1].replace(' ', '') for t in ref_events.name]
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
        return []


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
        events.at[e[(e.end > toff) & ~e.name.str.contains('.*Stim.*') & ~e.name.str.contains('TRIALSTOP')].index, 'end'] = toff
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