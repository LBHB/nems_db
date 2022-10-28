import numpy as np
import matplotlib.pyplot as plt
import nems0.db as nd  # NEMS database functions -- NOT celldb
import nems_lbhb.baphy as nb   # baphy-specific functions
import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems0.recording as recording
import numpy as np
import TwoStim_helpers as ts
import nems0.preprocessing as preproc
import nems0.metrics.api as nmet
import pickle as pl
import pandas as pd
import sys
import os
import re
import itertools
sys.path.insert(0,'/auto/users/luke/Code/Python/Utilities')
import fitEllipse as fE
import nems0.epoch as ep
import logging
log = logging.getLogger(__name__)

def calc_psth_metrics_Greg(batch, cellid,):
    import numpy as np
    import SPO_helpers as sp
    import nems0.preprocessing as preproc
    import nems0.metrics.api as nmet
    import nems0.metrics.corrcoef
    import copy
    import nems0.epoch as ep
    import scipy.stats as sst
    from nems_lbhb.gcmodel.figures.snr import compute_snr
    from nems0.preprocessing import generate_psth_from_resp
    import logging
    log = logging.getLogger(__name__)

    start_win_offset=0  #Time (in sec) to offset the start of the window used to calculate threshold, exitatory percentage, and inhibitory percentage
    options = {}

    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = {'rasterfs': 100,
               'stim': False,
               'resp': True}
    rec = manager.get_recording(**options)

    passive = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT']
    rec['resp'] = rec['resp'].extract_channels([cellid])

    if passive.shape[0] >= 2:
        #if OLP test was sorted in here as well, slice it out of the epochs and data
        print(f"Multiple ({passive.shape[0]}) OLPs found in {cellid}")
        runs = passive.shape[0] - 1
        max_run = (passive['end'] - passive['start']).reset_index(drop=True).idxmax()
        if runs != max_run:
            print(f"There are {runs+1} OLPs, the longest is run {max_run+1}. Using last run but make sure that is what you want.")
        else:
            print(f"The {runs+1} run is also the longest run: {max_run+1}, using last run.")
        good_start = passive.iloc[-1,1]
        rec['resp']._data = {key: val[val >= good_start] - good_start for key,val in rec['resp']._data.items()}
        rec['resp'].epochs = rec['resp'].epochs.loc[rec['resp'].epochs['start'] >= good_start,:].reset_index(drop=True)
        rec['resp'].epochs['start'] = rec['resp'].epochs['start'] - good_start
        rec['resp'].epochs['end'] = rec['resp'].epochs['end'] - good_start

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs=100

    #Greg spont rate subtraction with std norm
    prestimsilence = resp.extract_epoch('PreStimSilence')
    # average over reps(0) and time(-1), preserve neurons
    spont_rate = np.expand_dims(np.nanmean(prestimsilence, axis=(0, -1)), axis=1)
    ##STD OVER PRESETIM ONLY
    std_per_neuron = resp._data.std(axis=1, keepdims=True)
    std_per_neuron[std_per_neuron == 0] = 1
    norm_spont = resp._modified_copy(data=(resp._data - spont_rate) / std_per_neuron)

    file = os.path.splitext(rec.meta['files'][0])[0]
    experiment_name = file[-3:]
    #get dictionary of parameters for experiment
    if experiment_name == 'OLP':
        params = get_expt_params(resp, manager, cellid)
    else:
        params = {}

    epcs=rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2=rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    prestim=epcs.iloc[0]['end']
    poststim=ep2['end']-ep2['start']
    lenstim=ep2['end']

    stim_epochs = ep.epoch_names_matching(resp.epochs, 'STIM_')
    if paths:
        bg_dir = f"/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{paths[0]}/"
        bg_nm = os.listdir(bg_dir)
        bg_names = [os.path.splitext(name)[0][2:].replace('_', '') for name in bg_nm]
        fg_dir = f"/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{paths[1]}/"
        fg_nm = os.listdir(fg_dir)
        fg_names = [os.path.splitext(name)[0][2:].replace('_', '') for name in fg_nm]
        bg_names.append('null'), fg_names.append('null')

        bg_epochs = [b.split('_')[1].split('-')[0] for b in stim_epochs]
        fg_epochs = [f.split('_')[2].split('-')[0] for f in stim_epochs]
        bg_epochs = [b[2:] if b != 'null' else b for b in bg_epochs]
        fg_epochs = [f[2:] if f != 'null' else f for f in fg_epochs]

        bool_bg, bool_fg = [], []
        for (b, f) in zip(bg_epochs, fg_epochs):
            bool_bg.append(b in bg_names)
            bool_fg.append(f in fg_names)

        mask = np.logical_and(np.asarray(bool_bg), np.asarray(bool_fg))
        stim_epochs = [b for a, b in zip(mask, stim_epochs) if a]

        good_epochs = resp.epochs.loc[(resp.epochs['name'].isin(stim_epochs)), :]
        starts = good_epochs.start
        ends = good_epochs.end

        ff_start = resp.epochs.start.isin(starts)
        ff_ends = resp.epochs.end.isin(ends)
        ff_silence = resp.epochs.name.isin(stim_epochs + ['PreStimSilence', 'PostStimSilence', 'REFERENCE'])

        resp.epochs = resp.epochs.loc[ff_silence & (ff_start | ff_ends), :]
        rec['resp'].epochs = resp.epochs.loc[ff_silence & (ff_start | ff_ends), :]

    epoch_repetitions = [resp.count_epoch(cc) for cc in stim_epochs]
    full_resp = np.empty((max(epoch_repetitions), len(stim_epochs),
                          (int(lenstim) * rec['resp'].fs)))
    full_resp[:] = np.nan
    for cnt, epo in enumerate(stim_epochs):
        resps_list = resp.extract_epoch(epo)
        full_resp[:resps_list.shape[0], cnt, :] = resps_list[:, 0, :]

    ##base reliability
    # gets two subsamples across repetitions, and takes the mean across reps
    rep1 = np.nanmean(full_resp[0:-1:2, ...], axis=0)
    rep2 = np.nanmean(full_resp[1:full_resp.shape[0] + 1:2, ...], axis=0)

    resh1 = np.reshape(rep1, [-1])
    resh2 = np.reshape(rep2, [-1])

    corcoef = sst.pearsonr(resh1[:], resh2[:])[0]

    ##average response
    pre_bin = int(prestim * rec['resp'].fs)
    post_bin = int(full_resp.shape[-1] - (poststim * rec['resp'].fs))

    raster = np.squeeze(full_resp[..., pre_bin:post_bin])

    S = tuple([*range(0, len(raster.shape), 1)])
    avg_resp = np.nanmean(np.absolute(raster), axis=S)

    ##signal to noise
    snr = compute_snr(resp)

    #Calculate suppression for each sound pair.
    # epochs with two sounds in them
    if paths:
        epcs_twostim = resp.epochs.loc[(resp.epochs['name'].str.count('-0-1') == 2) &
                                   (resp.epochs['name'].isin(stim_epochs)), :].copy()
    else:
        epcs_twostim = resp.epochs[resp.epochs['name'].str.count('-0-1') == 2].copy()

    twostims = np.unique(epcs_twostim.name.values.tolist())
    supp_array = np.empty((len(twostims)))
    supp_array[:] = np.nan

    for cnt, stimmy in enumerate(twostims.tolist()):
        ABepo = resp.extract_epoch(stimmy)
        sep = get_sep_stim_names(stimmy)
        Aepo = resp.extract_epoch('STIM_'+sep[0]+'_null')
        Bepo = resp.extract_epoch('STIM_null_'+sep[1])
        lenA, lenB = Aepo.shape[0], Bepo.shape[0]
        min_rep = np.min((Aepo.shape[0], Bepo.shape[0]))
        lin_resp = (Aepo[:min_rep, :, :] + Bepo[:min_rep, :, :])

        mean_lin = np.nanmean(np.squeeze(lin_resp), axis=(0,1))
        mean_combo = np.nanmean(np.squeeze(ABepo), axis=(0,1))
        supp_array[cnt] = mean_lin - mean_combo

    spike_times=rec['resp']._data[cellid]
    count=0
    for index, row in epcs.iterrows():
        count+=np.sum((spike_times > row['start']) & (spike_times < row['end']))
    SR=count/(epcs['end']-epcs['start']).sum()

    resp=rec['resp'].rasterize()
    resp=add_stimtype_epochs(resp)
    ps=resp.select_epochs(['PreStimSilence']).as_continuous()
    ff = np.isfinite(ps)
    SR_rast=ps[ff].mean()*resp.fs
    SR_std=ps[ff].std()*resp.fs





def get_expt_params(resp, manager, cellid):
    '''Greg added function that takes a loaded response and returns a dict that
    contains assorted parameters that are useful for plotting'''
    params = {}

    e = resp.epochs
    expt_params = manager.get_baphy_exptparams()  # Using Charlie's manager
    if len(expt_params) == 1:
        ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
    if len(expt_params) > 1:
        ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]

    params['experiment'], params['fs'] = cellid.split('-')[0], resp.fs
    params['PreStimSilence'], params['PostStimSilence'] = ref_handle['PreStimSilence'], ref_handle[
        'PostStimSilence']
    params['Duration'], params['SilenceOnset'] = ref_handle['Duration'], ref_handle['SilenceOnset']
    params['max reps'] = e[e.name.str.startswith('STIM')].pivot_table(index=['name'], aggfunc='size').max()
    params['stim length'] = int(e.loc[e.name.str.startswith('REF')].iloc[0]['end']
                                - e.loc[e.name.str.startswith('REF')].iloc[0]['start'])
    params['combos'] = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
                        'Half BG/Half FG', 'Full BG/Half FG']
    params['Background'], params['Foreground'] = ref_handle['Background'], ref_handle['Foreground']

    soundies = list(ref_handle['SoundPairs'].values())
    params['pairs'] = [tuple([j for j in (soundies[s]['bg_sound_name'].split('.')[0],
                                          soundies[s]['fg_sound_name'].split('.')[0])])
                       for s in range(len(soundies))]
    params['units'], params['response'] = resp.chans, resp
    params['rec'] = resp  # could be rec, was using for PCA function, might need to fix with spont/std

    return params


def get_sep_stim_names(stim_name):
    seps = [m.start() for m in re.finditer('_(\d|n)', stim_name)]
    if len(seps) < 2 or len(seps) > 2:
        return None
    else:
        return [stim_name[seps[0] + 1:seps[1]], stim_name[seps[1] + 1:]]


def parse_stim_type(stim_name):
    stim_sep = get_sep_stim_names(stim_name)
    if stim_sep is None:
        stim_type = None
    else:
        if stim_sep[0] == 'null':
            stim_type = 'B'
        elif stim_sep[1] == 'null':
            stim_type = 'A'
        elif stim_sep[0] == stim_sep[1]:
            stim_type = 'C'
        else:
            stim_type = 'I'
    return stim_type


def add_stimtype_epochs(sig):
    import pandas as pd
    df0 = sig.epochs.copy()
    df0['name'] = df0['name'].apply(parse_stim_type)
    df0 = df0.loc[df0['name'].notnull()]
    sig.epochs = pd.concat([sig.epochs, df0])
    return sig
