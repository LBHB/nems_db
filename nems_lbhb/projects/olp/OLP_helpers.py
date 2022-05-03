import numpy as np
import scipy.stats as sst
import pandas as pd

def manual_fix_units(cell_list):
    '''I don't know why these units are incorrectly saved. But it was just these two, add
    others if there keeps being problem, or just figure out the issue.'''
    for i in range(len(cell_list)):
        if cell_list[i] == 'TBR025a-21-2':
            cell_list[i] = 'TBR025a-21-1'
        elif cell_list[i] == 'TBR025a-60-2':
            cell_list[i] = 'TBR025a-60-1'

    return cell_list


def get_load_options(batch):
    '''Get options that will load baphyexperiment. Currently (4/22/22) only gtgram set up for
    binaural stimuli, need to set up for monaural.'''
    if batch == 339:
        options = {'rasterfs': 100,
                   'stim': True,
                   'resp': True,
                   'stimfmt': 'gtgram'}
    else:
        options = {'rasterfs': 100,
                   'stim': False,
                   'resp': True}
    return options


def remove_olp_test(rec):
    '''In some cases the olp test and real olp are sorted together and will both come up. The real one
    is always last of OLPs run. So take that one.'''
    if passive.shape[0] >= 2:
        # if OLP test was sorted in here as well, slice it out of the epochs and data
        print(f"Multiple ({passive.shape[0]}) OLPs found in {cellid}")
        runs = passive.shape[0] - 1
        max_run = (passive['end'] - passive['start']).reset_index(drop=True).idxmax()
        if runs != max_run:
            print(
                f"There are {runs + 1} OLPs, the longest is run {max_run + 1}. Using last run but make sure that is what you want.")
        else:
            print(f"The {runs + 1} run is also the longest run: {max_run + 1}, using last run.")
        good_start = passive.iloc[-1, 1]
        rec['resp']._data = {key: val[val >= good_start] - good_start for key, val in rec['resp']._data.items()}
        rec['resp'].epochs = rec['resp'].epochs.loc[rec['resp'].epochs['start'] >= good_start, :].reset_index(drop=True)
        rec['resp'].epochs['start'] = rec['resp'].epochs['start'] - good_start
        rec['resp'].epochs['end'] = rec['resp'].epochs['end'] - good_start

    return rec


def remove_spont_rate_std(resp):
    '''Remove the spont rate using prestimsilence from the response with std normalization. STD taken
    over prestim only.'''
    prestimsilence = resp.extract_epoch('PreStimSilence')
    # average over reps(0) and time(-1), preserve neurons
    spont_rate = np.expand_dims(np.nanmean(prestimsilence, axis=(0, -1)), axis=1)

    std_per_neuron = resp._data.std(axis=1, keepdims=True)
    std_per_neuron[std_per_neuron == 0] = 1
    norm_spont = resp._modified_copy(data=(resp._data - spont_rate) / std_per_neuron)

    ##This was Luke's way of geting SR. It gets the same thing I do/100
    # spike_times = rec['resp']._data[cellid]
    # count = 0
    # for index, row in epcs.iterrows():
    #     count += np.sum((spike_times > row['start']) & (spike_times < row['end']))
    # SR = count / (epcs['end'] - epcs['start']).sum()

    ##A second way of doing the same thing?
    # ps = resp.select_epochs(['PreStimSilence']).as_continuous()
    # ff = np.isfinite(ps)
    # SR_rast = ps[ff].mean() * resp.fs
    # SR_std = ps[ff].std() * resp.fs

    return norm_spont, spont_rate[0][0], std_per_neuron[0][0]


def get_expt_params(resp, manager, cellid):
    '''General helpful parameters that will come up later, probably.'''
    params = {}

    e = resp.epochs
    expt_params = manager.get_baphy_exptparams()  # Using Charlie's manager
    if len(expt_params) == 1:
        ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
    if len(expt_params) > 1:
        ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]

    params['experiment'], params['fs'] = cellid.split('-')[0], resp.fs
    params['PreStimSilence'], params['PostStimSilence'] = ref_handle['PreStimSilence'], ref_handle['PostStimSilence']
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
    params['rec'] = resp #could be rec, was using for PCA function, might need to fix with spont/std
    params['bg_folder'], params['fg_folder'] = ref_handle['BG_Folder'], ref_handle['FG_Folder']
    if 'Binaural' in ref_handle.keys():
        params['Binaural'] = ref_handle['Binaural']
    else:
        params['Binaural'] = 'No'

    return params


def path_tabor_get_epochs(stim_epochs, rec, resp, params):
    '''I can't figure out what this code was for. I think I stopped using the variable paths in this
    a while ago but I can't remember what it was for because things work without it. It seems like
    this code doesn't handle spaces in filenames whereas epochs don't have them. I don't know 4/22/22'''
    stim_epochs, rec, resp = ohel.path_tabor_get_epochs(stim_epochs, rec, resp, params)

    bg_dir = f"/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{params['bg_folder']}/"
    bg_nm = os.listdir(bg_dir)
    bg_names = [os.path.splitext(name)[0][2:].replace('_', '') for name in bg_nm]
    fg_dir = f"/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/{params['fg_folder']}/"
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
    # resp.epochs = resp.epochs.loc[ff_silence & (ff_start | ff_ends), :]
    # full = pd.concat([good_epochs, selected_silences]).sort_values(by=['start', 'end'], ascending=[True, False], ignore_index=True)

    return stim_epochs, rec, resp


def calc_base_reliability(full_resp):
    '''Calculates a correlation coeffient, or how reliable the neuron response is by taking two
    subsamples across repetitions, and takes the mean across the repetitions.'''
    rep1 = np.nanmean(full_resp[0:-1:2, ...], axis=0)
    rep2 = np.nanmean(full_resp[1:full_resp.shape[0] + 1:2, ...], axis=0)

    resh1 = np.reshape(rep1, [-1])
    resh2 = np.reshape(rep2, [-1])

    corcoef = sst.pearsonr(resh1[:], resh2[:])[0]

    return corcoef


def calc_average_response(full_resp, params):
    '''Calculates the average response across the stimulus.'''
    pre_bin = int(params['prestim'] * params['fs'])
    post_bin = int(full_resp.shape[-1] - (params['poststim'] * params['fs']))

    raster = np.squeeze(full_resp[..., pre_bin:post_bin])

    S = tuple([*range(0, len(raster.shape), 1)])
    avg_resp = np.nanmean(np.absolute(raster), axis=S)

    return avg_resp


def label_pair_type(stim):
    '''Take all epochs that have two full sounds playing and assign a type to them based on the
    location of each sound that was played. This is important to binaural stimuli:
    11: both sounds played contralateral
    12: BG played contra- FG played ipsilateral
    21: BG played ipsi- FG played contralateral
    22: both sounds played ipsilateral
    If Binaural parameter is No (or doesn't exist in previous animals), type is set to 11.'''
    if len(stim.split('_')[1].split('-')) >= 4:
        type_label = stim.split('_')[1].split('-')[3]+stim.split('_')[2].split('-')[3]
    else:
        type_label = '11'

    return type_label

def label_ep_type(ep_name):
    '''Labels epochs that have one or two stimuli in it. First position refers to BG, second
    to FG. 0 means null, 1 means primary speaker, 2 means secondary speaker'''
    seps = (ep_name.split('_')[1], ep_name.split('_')[2])

    if len(seps[0].split('-')) >= 4 or len(seps[1].split('-')) >= 4:
        if seps[0] != 'null' and seps[1] != 'null':
            stim_type = seps[0].split('-')[3] + seps[1].split('-')[3]
        else:
            if seps[0] == 'null':
                stim_type = '0' + seps[1].split('-')[3]
            elif seps[1] == 'null':
                stim_type = seps[0].split('-')[3] + '0'
    else:
        if seps[0] == 'null':
            stim_type = '01'
        elif seps[1] == 'null':
            stim_type = '10'
        else:
            stim_type = '11'

    return stim_type