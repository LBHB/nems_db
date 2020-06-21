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
import nems_lbhb.io as io

log = logging.getLogger(__name__)

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
            self.parmfile = [Path(f) for f in files]

        elif type(parmfile) is list:
            self.parmfile = [Path(p).with_suffix('.m') for p in parmfile]
        
        else:
            self.parmfile = [Path(parmfile).with_suffix('.m')]
           
        if np.any([not p.exists() for p in self.parmfile]):
            raise IOError(f'Not all parmfiles in {self.parmfile} were found')

        # we can assume all parmfiles come from same folder/experiment (site)
        self.folder = self.parmfile[0].parent
        self.experiment = self.parmfile[0].name.split('_', 1)[0]

        # full file name will be unique though, so this is a list
        self.experiment_with_runclass = [Path(p.stem) for p in self.parmfile]
    

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
    def get_baphy_exptevents(self):
        exptevents = [ep[-1] for ep in self._get_baphy_parameters(userdef_convert=False)]
        return exptevents

    @copying_lru_cache(maxsize=128)
    def get_baphy_exptparams(self):
        exptparams = [ep[1] for ep in self._get_baphy_parameters(userdef_convert=False)]
        return exptparams

    @copying_lru_cache(maxsize=128)
    def get_baphy_globalparams(self):
        globalparams = [ep[0] for ep in self._get_baphy_parameters(userdef_convert=False)]
        return globalparams

    @lru_cache(maxsize=128)
    def get_recording(self, **kwargs):
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
        
        rec_name = self.experiment      
        
        # figure out signals to load, then load them (as lists)
        resp = kwargs.get('resp', False)
        pupil = kwargs.get('pupil', False)
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

            signals['pupil'] = nems.signal.RasterizedSignal.concatenate_time(pupil_sigs)

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
        for i, f in enumerate(self.parmfile):
            fn = str(f).split('/')[-1]
            exptevents[i].to_pickle('/auto/users/hellerc/code/scratch/exptevents_io_{}.pickle'.format(fn))
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
        behave_file = [True if (p['BehaveObjectClass'] != 'Passive') else False for p in params]
        if len(behave_file) > 1:
            events = [e for i, e in enumerate(events) if behave_file[i]]
        elif behave_file[0] == True:
            events = events
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

    return epochs


def _make_trial_epochs(exptevents, exptparams, **options):
    """
    Define baphy trial epochs
    """
    # sort of hacky. This means that if behavior classification 
    # was run and it's NOT classical conditioning, you should truncate
    # trials after licks
    remove_post_lick = ('soundTrial' in exptevents.columns) & \
                            (exptparams['BehaveObjectClass'] != 'ClassicalConditioning')
    

    trial_events = exptevents[exptevents['name'].str.startswith('TRIALSTART')].copy()
    end_events = exptevents[exptevents['name'].str.startswith('TRIALSTOP')]
    trial_events.at[:, 'end'] = end_events['start'].values
    trial_events.at[:, 'name'] = 'TRIAL'

    if remove_post_lick:
       trial_events =  _remove_post_lick(trial_events, exptevents)

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
                            ~exptevents.name.str.contains('Silence')].name.unique()
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

    # pre/post stim events
    sil_tags = exptevents[exptevents.name.str.contains('Silence')].name.unique()
    sil_events = exptevents[exptevents.name.isin(sil_tags)].copy()
    new_tags = [t.split(',')[0].replace(' ', '') for t in sil_events.name]
    sil_events.at[:, 'name'] = new_tags

    # lick events
    lick_events = exptevents[exptevents.name=='LICK']

    # concatenate events together
    stim_events = pd.concat([ref_events, tar_events, sil_events, lick_events], ignore_index=True)

    if remove_post_lick:
        stim_events = _remove_post_lick(stim_events, exptevents)

    stim_events = stim_events.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    stim_events = stim_events.drop(columns=['index'])

    return stim_events


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

    behavior_events = _remove_post_lick(behavior_events, exptevents)

    behavior_events = behavior_events.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index()
    behavior_events = behavior_events.drop(columns=['index'])

    return behavior_events


def _remove_post_lick(events, exptevents):
    # screen for FA / Early trials in which we need to truncate / chop out references
    trunc_trials = exptevents[exptevents.name.isin(['FALSE_ALARM_TRIAL', 'EARLY_TRIAL'])].Trial.unique()
    lick_time = exptevents[exptevents.Trial.isin(trunc_trials) & (exptevents.name=='LICK')].start
    lick_trial = exptevents[exptevents.Trial.isin(trunc_trials) & (exptevents.name=='LICK')].Trial.values

    #if len(lick_time) != len(trunc_trials):
    #    import pdb;pdb.set_trace()
    #   raise ValueError('More than one lick recorded on a FA trial, whats up??')
    
    for t in trunc_trials:  
        fl = lick_time.iloc[lick_trial==t].iloc[0]
        e = events[events.Trial==t]
        # truncate events that overlapped with lick
        events.at[e[e.end > fl].index, 'end'] = fl
        # remove events that started after the lick completely
        events = events.drop(e[(e.start.values > fl) & (e.end.values >= fl)].index)
    
    return events

def _trim_epoch_columns(epochs):
    cols = [c for c in epochs.columns if c not in ['start', 'end', 'name']]
    return epochs.drop(columns=cols)