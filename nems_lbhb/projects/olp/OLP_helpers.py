import numpy as np
import scipy.stats as sst
import pandas as pd
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import glob
import seaborn as sb
import re
import nems0.epoch as ep
import joblib as jl
from nems_lbhb import baphy_io
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nems_lbhb.stats import jack_mean_err



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
    # Maybe add this as the default? Could save time?
    # options = {'rasterfs': fs, 'stim': True, 'stimfmt': 'lenv', 'resp': True, 'recache': False}

    return options


def remove_olp_test(rec):
    '''DEFUNCT as of 2023_05. Made fitting smarter, so it doesn't have to all run on one machine
    and it can handle neuropixel recordings that contain multiple real OLP files, this cannot.

    In some cases the olp test and real olp are sorted together and will both come up. The real one
    is always last of OLPs run. So take that one.'''
    passive = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT']
    if passive.shape[0] >= 2:
        # if OLP test was sorted in here as well, slice it out of the epochs and data
        print(f"Multiple ({passive.shape[0]}) OLPs found in")
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


def remove_olp_test_nonOLP(rec):
    '''DEFUNCT as of 2023_05. Made fitting smarter, so it doesn't have to all run on one machine
    and it can handle neuropixel recordings that contain multiple real OLP files, this cannot.

    2023_01_04. For the batch 341 that has the prediction. The signal has multiple files, including non-OLP ones
    so I changed remove_olp_test to be flexible to pick only OLP and if there are multiple OLPs, only the second one.
    In some cases the olp test and real olp are sorted together and will both come up. The real one
    is always last of OLPs run. So take that one.'''
    passive = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT']
    filenames = ep.epoch_names_matching(rec['resp'].epochs, 'FILE_')
    file_ends = [aa.split('_')[-1] for aa in filenames]

    if passive.shape[0] >= 2 and len(file_ends) > 1:
        print(f"Multiple ({passive.shape[0]}) FILEs found in rec['resp'], {file_ends}.")
        OLPs = [cc for cc, aa in enumerate(file_ends) if aa == 'OLP']
        print(f"Index of OLP(s) are {OLPs}")
        if len(OLPs) > 1:
            print(f"There are {len(OLPs)} OLP files, keeping the second one, index {max(OLPs)}.")
            OLP_idx = max(OLPs)
            print(f"Keeping file filenames {filenames[OLP_idx]}")
        elif len(OLPs) == 1:
            print(f"There is only 1 OLP file, keeping index {max(OLPs)}")
            OLP_idx = max(OLPs)
            print(f"Keeping file filenames {filenames[OLP_idx]}")

        good_start = passive.iloc[OLP_idx, 1]
        good_end = passive.iloc[OLP_idx, 2]

        rec['resp'].epochs = rec['resp'].epochs.loc[(rec['resp'].epochs['start'] < good_end) &
                                                    (rec['resp'].epochs['start'] >= good_start)].reset_index(drop=True)
        rec['resp']._data = {key: val[(val >= good_start) & (val < good_end)] - good_start for key, val in rec['resp']._data.items()}
        rec['resp'].epochs['start'] = rec['resp'].epochs['start'] - good_start
        rec['resp'].epochs['end'] = rec['resp'].epochs['end'] - good_start
        #
        #
        # rec['resp']._data = {key: val[val >= good_start] - good_start for key, val in rec['resp']._data.items()}
        # rec['resp'].epochs = rec['resp'].epochs.loc[rec['resp'].epochs['start'] >= good_start, :].reset_index(
        #     drop=True)
        # rec['resp'].epochs['start'] = rec['resp'].epochs['start'] - good_start
        # rec['resp'].epochs['end'] = rec['resp'].epochs['end'] - good_start

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
        expt_params = manager.get_baphy_exptparams()
        whichparams = [aa for aa in range(len(expt_params)) if expt_params[aa]['runclass']=='OLP']
        # You can ultimately make this so that ref_handle['Combos'] must == 'Manual' to ensure the second is the real one
        if len(whichparams) == 1:
            ref_handle = expt_params[whichparams[0]]['TrialObject'][1]['ReferenceHandle'][1]
        else:
            print(f"There are {len(whichparams)} OLPs for {cellid}, using the last one.")
            ref_handle = expt_params[whichparams[-1]]['TrialObject'][1]['ReferenceHandle'][1]

    params['experiment'], params['fs'] = cellid.split('-')[0], resp.fs
    params['PreStimSilence'], params['PostStimSilence'] = ref_handle['PreStimSilence'], ref_handle['PostStimSilence']
    params['Duration'] = ref_handle['Duration']
    params['SilenceOnset'] = ref_handle['SilenceOnset']
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
    ## Took this out for fitting ARM data, not sure why ref_handle for that experiment doesn't have BG_Folder ONLY
    # params['bg_folder'], params['fg_folder'] = ref_handle['BG_Folder'], ref_handle['FG_Folder']
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


def remove_clicks(w, max_threshold=15, verbose=False):
    '''SVD made in 2022_09. Same as the matlab function when OLP is called, it takes the high power
    clicks away from an RMS normalized signal and log scales things above the threshold.'''
    w_clean = w

    # log compress everything > 67% of max
    crossover = 0.67 * max_threshold
    ii = (w>crossover)
    w_clean[ii] = crossover + np.log(w_clean[ii]-crossover+1);
    jj = (w<-crossover)
    w_clean[jj] = -crossover - np.log(-w_clean[jj]-crossover+1);

    if verbose:
       print(f'bins compressed down: {ii.sum()} up: {jj.sum()} max {np.abs(w).max():.2f}-->{np.abs(w_clean).max():.2f}')

    return w_clean


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
    if len(ep_name.split('_')) == 1 or ep_name[:5] != 'STIM_':
        stim_type = None
    elif len(list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])) == 2:
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])
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
    else:
        stim_type = None

    return stim_type


def label_synth_type(ep_name):
        '''Labels epochs that have one or two stimuli in it based on what kind of synthetic sound it is.
        N = Normal RMS, C = Cochlear, T = Temporal, S = Spectral, U = Spectrotemporal, M = spectrotemporal
        modulation, A = Non-RMS normalized unsynethic'''
        if len(ep_name.split('_')) == 1 or ep_name[:5] != 'STIM_':
            synth_type = None
        # elif len(ep_name.split('_')) == 3:
        #     seps = (ep_name.split('_')[1], ep_name.split('_')[2])
        elif len(list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])) == 2:
            seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])
            if len(seps[0].split('-')) >= 5 or len(seps[1].split('-')) >= 5:
                if seps[0] != 'null':
                    synth_type = seps[0].split('-')[4]
                elif seps[1] != 'null':
                    synth_type = seps[1].split('-')[4]
                else:
                    raise ValueError(f"Something went wrong with {ep_name}, both parts are 'null'")

            else:
                synth_type = 'A'
        else:
            synth_type = None

        return synth_type


def label_dynamic_ep_type(ep_name):
    '''Labels epochs that have one or two stimuli in it according to its duration (dynamic stimuli.
    First position refers to BG, second to FG. n means null, f means full length, h means half length'''
    if len(ep_name.split('_')) == 1 or ep_name[:5] != 'STIM_':
        stim_type = None
    elif len(list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])) == 2:
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", ep_name)[0])

        if len(seps[0].split('-')) >= 2 or len(seps[1].split('-')) >= 2:
            if seps[0] != 'null' and seps[1] != 'null':
                if seps[0].split('-')[1] == '0':
                    btype = 'f'
                else:
                    btype = 'h'
                if seps[1].split('-')[1] == '0':
                    ftype = 'f'
                else:
                    ftype = 'h'
                stim_type = btype + ftype
            else:
                if seps[0] == 'null':
                    if seps[1].split('-')[1] == '0':
                        ftype = 'f'
                    else:
                        ftype = 'h'
                    stim_type = 'n' + ftype
                elif seps[1] == 'null':
                    if seps[0].split('-')[1] == '0':
                        btype = 'f'
                    else:
                        btype = 'h'
                    stim_type = btype + 'n'
    else:
        stim_type = None
        print(f"None of your labeling things worked for {ep_name}, you should look into that.")

    return stim_type


def add_stimtype_epochs(sig):
    '''Mostly unneeded, just replaces epochs with their type instead of just
    labeling them as in label_ep_type'''
    import pandas as pd
    df0 = sig.epochs.copy()
    df0['name'] = df0['name'].apply(label_ep_type)
    df0 = df0.loc[df0['name'].notnull()]
    sig.epochs = pd.concat([sig.epochs, df0])
    return sig


def r_noise_corrected(X,Y,N_ac=200):
    '''This one is directly from Luke's'''
    import nems0.metrics.corrcoef
    Xac = nems0.metrics.corrcoef._r_single(X, N_ac,0)
    Yac = nems0.metrics.corrcoef._r_single(Y, N_ac,0)
    repcount = X.shape[0]
    rs = np.zeros((repcount,repcount))
    for nn in range(repcount):
        for mm in range(repcount):
            X_ = X[mm, :]
            Y_ = Y[nn, :]
            # remove all nans from pred and resp
            ff = np.isfinite(X_) & np.isfinite(Y_)

            if (np.sum(X_[ff]) != 0) and (np.sum(Y_[ff]) != 0):
                rs[nn,mm] = np.corrcoef(X_[ff],Y_[ff])[0, 1]
            else:
                rs[nn,mm] = 0
    #rs=rs[np.triu_indices(rs.shape[0],1)]
    #plt.figure(); plt.imshow(rs)
    return np.mean(rs)/(np.sqrt(Xac) * np.sqrt(Yac))


def get_binaural_adjacent_epochs(stim):
    '''Takes a simulus name and finds the two epochs if you were to reverse
    the BG and FG ipsi to contra or vice versa'''
    s, a, b = stim.split('_')
    if len(a.split('-')) >= 4 or len(b.split('-')) >= 4:
        #Swap BG first
        q, w, e, r = a.split('-')[:4]
        if r == '1':
            newr = '2'
        elif r == '2':
            newr = '1'
        newa = '-'.join([q, w, e, newr])
        dA = '_'.join([s, newa, b])

        #Swap FG next
        q, w, e, r = b.split('-')[:4]
        if r == '1':
            newr = '2'
        elif r == '2':
            newr = '1'
        newb = '-'.join([q, w, e, newr])
        dB = '_'.join([s, a, newb])

        return dA, dB


def quadrants_by_FR(weight_df, threshold=0.05, quad_return=5):
    '''Only works well pre- '_start' '_end' period, as this will only filter across the whole signal.

    Filters a dataframe by a FR threshold with spont subtracted. quad_returns says which
    filtered quadrants to return. If you give a list it'll output as a dictionary with keys
    which quadrant, if a single integer it'll just be the dataframe outputted. Default is 5
    which takes a combination of BG+/FG+, BG+/FG-, and BG-/FG+.'''
    quad1 = weight_df.loc[(weight_df.bg_FR<=-threshold) & (weight_df.fg_FR>=threshold)]
    quad2 = weight_df.loc[(np.abs(weight_df.bg_FR)<=threshold) & (weight_df.fg_FR>=threshold)]
    quad3 = weight_df.loc[(weight_df.bg_FR>=threshold) & (weight_df.fg_FR>=threshold)]
    quad4 = weight_df.loc[(weight_df.bg_FR<=-threshold) & (np.abs(weight_df.fg_FR)<=threshold)]
    quad6 = weight_df.loc[(weight_df.bg_FR>=threshold) & (np.abs(weight_df.fg_FR)<=threshold)]
    quad10 = pd.concat([quad2, quad3, quad6], axis=0)
    quad7 = weight_df.loc[(weight_df.bg_FR<=-threshold) & (weight_df.fg_FR<=-threshold)]
    quad8 = weight_df.loc[(np.abs(weight_df.bg_FR)<=threshold) & (weight_df.fg_FR<=-threshold)]
    quad9 = weight_df.loc[(weight_df.bg_FR>=threshold) & (weight_df.fg_FR<=-threshold)]
    quad5 = weight_df.loc[(np.abs(weight_df.bg_FR)<threshold) & (np.abs(weight_df.fg_FR)<threshold)]
    dfs = [quad1, quad2, quad3, quad4, quad5, quad6, quad7, quad8, quad9, quad10]
    if isinstance(quad_return, list):
        quads = {qq:dfs[qq - 1] for qq in quad_return}
    elif isinstance(quad_return, int):
        quads = dfs[quad_return - 1]
    else:
        raise ValueError(f"quad_return input {quad_return} is not a list or int.")

    return quads, threshold


def histogram_subplot_handler(df_dict, yax='cells', tags=None):
    if not tags:
        tags = [ta for ta in df_dict.keys()]
    dfs = [qu for qu in df_dict.values()]
    if len(dfs) == 9:
        fig, axes = plt.subplots(3, 3, sharex=True, sharey=False, figsize=(8, 8))
    elif len(dfs) == 4:
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))
    else:
        fig, axes = plt.subplots(1, len(dfs), sharex=True, sharey=True, figsize=(8, len(dfs)+1))

    ax = axes.ravel()
    for aa, tt, qq in zip(ax, tags, dfs):
        weight_hist(qq, tag=tt, y=yax, ax=aa)


def weight_hist(df, tag=None, y='cells', ax=None):
    edges=np.arange(-2,2,.05)
    if not ax:
        fig, ax = plt.subplots()
        title = True
    else:
        title = False

    if y == 'cells':
        ax.hist(df.weightsA, bins=edges, histtype='step')
        ax.hist(df.weightsB, bins=edges, histtype='step')
        ax.set_ylabel('Number of Cells', fontweight='bold', fontsize=8)
    elif y == 'percent':
        na, xa = np.histogram(df.weightsA, bins=edges)
        na = na / na.sum() * 100
        nb, xb = np.histogram(df.weightsB, bins=edges)
        nb = nb / nb.sum() * 100
        ax.hist(xa[:-1], xa, weights=na, histtype='step')
        ax.hist(xb[:-1], xb, weights=nb, histtype='step')
        ax.set_ylabel('Percent', fontweight='bold', fontsize=8)
    else:
        raise ValueError(f"y value {y} is not supported, put either 'cells' or 'percent'")

    ax.set_xlabel('Weight', fontweight='bold', fontsize=8)
    if title == True:
        ax.set_title(f"{str(df.Animal.unique())} - {tag}", fontweight='bold', fontsize=12)
        ax.legend(('Background', 'Foreground'), fontsize=7)
    if title == False:
        ax.set_title(f'{tag}', fontweight='bold', fontsize=8)
        ax.legend(('Background', 'Foreground'), fontsize=4)


def get_sound_statistics_full(weight_df, cuts=None, fs=100):
    '''Largely DEFUNCT as of 2023_05. Made fitting smarter, so it doesn't have to all run on one machine
    and it can handle neuropixel recordings that contain multiple real OLP files, this cannot. Replaced
    with get_sound_statistics_from_df().

    Updated 2022_09_13. Added mean relative gain for each sound. The rel_gain is BG or FG
    respectively.
    Updated 2022_09_12. Now it can take a DF that has multiple synthetic conditions and pull
    the stats for the synthetic sounds. The dataframe will label these by column synth_kind
    and you should pull out them that way, because they all have the same name in the name
    column. Additionally, RMS normalization stats were added in RMS_norm and max_norm powers.
    5/12/22 Takes a cellid and batch and figures out all the sounds that were played
    in that experiment and calculates some stastistics it plots side by side. Also outputs
    those numbers in a cumbersome dataframe'''
    lfreq, hfreq, bins = 100, 24000, 48
    cid, btch = weight_df.cellid.iloc[0], weight_df.batch.iloc[0]
    manager = BAPHYExperiment(cellid=cid, batch=btch)
    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    BG_folder, FG_folder = ref_handle['BG_Folder'], ref_handle['FG_Folder']

    bbs = list(set([bb.split('_')[1][:2] for bb in weight_df.epoch]))
    ffs = list(set([ff.split('_')[2][:2] for ff in weight_df.epoch]))
    bbs.sort(key=int), ffs.sort(key=int)

    synths = list(weight_df.synth_kind.unique())
    kind_dict = {'M': 'SpectrotemporalMod', 'U': 'Spectrotemporal', 'T': 'Temporal',
                  'S': 'Spectral', 'C': 'Cochlear'}

    # if 'N' in synths and 'A' in synths:
    #     synths.remove('A')

    syn_df = []
    for syn in synths:
        # This is getting the mean rel gain for each sound (FG rel gain for FGs, etc)
        synth_df = weight_df.loc[weight_df.synth_kind==syn].copy()
        bg_df = synth_df[['BG', 'BG_rel_gain']]
        fg_df = synth_df[['FG', 'FG_rel_gain']]

        bg_mean = bg_df.groupby(by='BG').agg(mean=('BG_rel_gain', np.mean)).reset_index().\
            rename(columns={'BG': 'short_name'})
        fg_mean = fg_df.groupby(by='FG').agg(mean=('FG_rel_gain', np.mean)).reset_index().\
            rename(columns={'FG': 'short_name'})
        mean_df = pd.concat([bg_mean, fg_mean])

        # This is just loading the sounds and stuffs
        if syn=='A' or syn=='N':
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{BG_folder}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{ff}*.wav'))[0] for ff in ffs]
        else:
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'{BG_folder}/{kind_dict[syn]}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'{FG_folder}/{kind_dict[syn]}/{ff}*.wav'))[0] for ff in ffs]

        paths = bg_paths + fg_paths
        bgname = [bb.split('/')[-1].split('.')[0] for bb in bg_paths]
        fgname = [ff.split('/')[-1].split('.')[0] for ff in fg_paths]
        names = bgname + fgname

        Bs, Fs = ['BG'] * len(bgname), ['FG'] * len(fgname)
        labels = Bs + Fs

        sounds = []
        means = np.empty((bins, len(names)))
        means[:] = np.NaN
        for cnt, sn, pth, ll in zip(range(len(labels)), names, paths, labels):
            sfs, W = wavfile.read(pth)
            spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

            # to measure rms power... for rms-normed signals:
            rms_normed = np.std(remove_clicks(W / W.std(), 15))
            # for max-normed signals:
            max_normed = np.std(W / np.abs(W).max()) * 5

            dev = np.std(spec, axis=1)

            freq_dev = np.std(spec, axis=0)
            freq_mean = np.nanmean(spec, axis=1)
            x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
            csm = np.cumsum(freq_mean)
            big = np.max(csm)

            freq75 = x_freq[np.abs(csm - (big * 0.75)).argmin()]
            freq25 = x_freq[np.abs(csm - (big * 0.25)).argmin()]
            freq50 = x_freq[np.abs(csm - (big * 0.5)).argmin()]
            bandw = np.log2(freq75 / freq25)

            means[:, cnt] = freq_mean

            # 2022_09_23 Adding power spectrum stats
            temp = np.abs(np.fft.fft(spec, axis=1))
            freq = np.abs(np.fft.fft(spec, axis=0))

            temp_ps = np.sum(np.abs(np.fft.fft(spec, axis=1)), axis=0)[1:].std()
            freq_ps = np.sum(np.abs(np.fft.fft(spec, axis=0)), axis=1)[1:].std()

            sounds.append({'name': sn.split('_')[0],
                           'type': ll,
                           'synth_kind': syn,
                           'Tstationary': np.nanmean(dev),
                           'bandwidth': bandw,
                           '75th': freq75,
                           '25th': freq25,
                           'center': freq50,
                           'spec': spec,
                           'mean_freq': freq_mean,
                           'Fstationary_wrong': np.std(freq_mean),
                           'Fstationary': np.nanmean(freq_dev),
                           'RMS_norm_power': rms_normed,
                           'max_norm_power': max_normed,
                           'temp_ps': temp,
                           'freq_ps': freq,
                           'temp_ps_std': temp_ps,
                           'freq_ps_std': freq_ps,
                           'short_name': sn[2:].split('_')[0].replace(' ', '')})

            if cuts:
                one, two = spec[:, cuts[0]:int(cuts[1] * fs)], spec[:, int(cuts[1] * fs):]
                t_dev_start, t_dev_end = np.std(one, axis=1), np.std(two, axis=1)
                f_dev_start, f_dev_end = np.std(one, axis=0), np.std(two, axis=0)


                freq_mean_start, freq_mean_end = np.nanmean(one, axis=1), np.nanmean(two, axis=1)
                x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
                csm_start, csm_end = np.cumsum(freq_mean_start), np.cumsum(freq_mean_end)
                big_start, big_end = np.max(csm_start), np.max(csm_end)

                freq75_start = x_freq[np.abs(csm_start - (big_start * 0.75)).argmin()]
                freq25_start = x_freq[np.abs(csm_start - (big_start * 0.25)).argmin()]
                bandw_start = np.log2(freq75_start / freq25_start)

                freq75_end = x_freq[np.abs(csm_end - (big_end * 0.75)).argmin()]
                freq25_end = x_freq[np.abs(csm_end - (big_end * 0.25)).argmin()]
                bandw_end = np.log2(freq75_end / freq25_end)

                sounds[cnt]['Tstationary_start'] = np.nanmean(t_dev_start)
                sounds[cnt]['Tstationary_end'] = np.nanmean(t_dev_end)
                sounds[cnt]['Fstationary_start'] = np.nanmean(f_dev_start)
                sounds[cnt]['Fstationary_end'] = np.nanmean(f_dev_end)
                sounds[cnt]['bandwidth_start'] = bandw_start
                sounds[cnt]['bandwidth_end'] = bandw_end

        sound_df = pd.DataFrame(sounds)
        # Merge the relative gain data into the DF of sounds
        sound_df = pd.merge(sound_df, mean_df, on='short_name').rename(columns={'mean': 'rel_gain'})

        # Add mod spec calculations to sound_df, 2022_08_26
        mods = np.empty((sound_df.iloc[0].spec.shape[0], sound_df.iloc[0].spec.shape[1],
                         len(sound_df)))
        mods[:] = np.NaN
        mod_list = []
        for cnt, ii in enumerate(sound_df.name):
            row = sound_df.loc[sound_df.name == ii]
            spec = row['spec'].values[0]
            mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))
            mods[:, :, cnt] = mod
            mod_list.append(mod)
        avmod = np.nanmean(mods, axis=2)
        norm_list = [aa - avmod for aa in mod_list]
        avmod = avmod[:, :, np.newaxis]
        normmod = mods - avmod
        clow, chigh = np.min(normmod), np.max(normmod)
        sound_df['modspec'] = mod_list
        sound_df['normmod'] = norm_list
        # selfsounds['normmod'] = norm_list

        trimspec = [aa[24:, 30:69] for aa in sound_df['modspec']]
        negs = [aa[:, :20] for aa in trimspec]
        negs = [aa[:, ::-1] for aa in negs]
        poss = [aa[:, -20:] for aa in trimspec]
        trims = [(nn + pp) / 2 for (nn, pp) in zip(negs, poss)]
        sound_df['trimspec'] = trims

        # Collapses across each access
        ots = [np.nanmean(aa, axis=0) for aa in trims]
        ofs = [np.nanmean(aa, axis=1) for aa in trims]

        tbins, fbins = 100, 48

        wt = np.fft.fftshift(np.fft.fftfreq(tbins, 1 / tbins))
        wf = np.fft.fftshift(np.fft.fftfreq(fbins, 1 / 6))

        wt2 = wt[50:70]
        wf2 = wf[24:]

        cumwt = [np.cumsum(aa) / np.sum(aa) for aa in ots]
        bigt = [np.max(aa) for aa in cumwt]
        freq50t = [wt2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumwt, bigt)]

        cumft = [np.cumsum(aa) / np.sum(aa) for aa in ofs]
        bigf = [np.max(aa) for aa in cumft]
        freq50f = [wf2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumft, bigf)]

        sound_df['avgwt'], sound_df['avgft'] = ots, ofs
        sound_df['cumwt'], sound_df['cumft'] = cumwt, cumft
        sound_df['t50'], sound_df['f50'] = freq50t, freq50f
        sound_df['meanT'], sound_df['meanF'] = ots, ofs
        # End mod spec addition 2022_08_26
        syn_df.append(sound_df)

    main_df = pd.concat(syn_df)

    return main_df


def get_sound_statistics(weight_df, plot=True):
    '''Largely DEFUNCT as of 2023_05. Although I can see this having some use for plotting still, but
    probably just cannibalize it to a better function, if you want the plot.

    5/12/22 Takes a cellid and batch and figures out all the sounds that were played
    in that experiment and calculates some stastistics it plots side by side. Also outputs
    those numbers in a cumbersome dataframe'''
    lfreq, hfreq, bins = 100, 24000, 48
    cid, btch = weight_df.cellid.iloc[0], weight_df.batch.iloc[0]
    manager = BAPHYExperiment(cellid=cid, batch=btch)
    expt_params = manager.get_baphy_exptparams()  # Using Charlie's manager
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    BG_folder, FG_folder = ref_handle['BG_Folder'], ref_handle['FG_Folder']

    bbs = list(set([bb.split('_')[1][:2] for bb in weight_df.epoch]))
    ffs = list(set([ff.split('_')[2][:2] for ff in weight_df.epoch]))
    bbs.sort(key=int), ffs.sort(key=int)

    bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'{BG_folder}/{bb}*.wav'))[0] for bb in bbs]
    fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'{FG_folder}/{ff}*.wav'))[0] for ff in ffs]
    paths = bg_paths + fg_paths
    bgname = [bb.split('/')[-1].split('.')[0] for bb in bg_paths]
    fgname = [ff.split('/')[-1].split('.')[0] for ff in fg_paths]
    names = bgname + fgname

    Bs, Fs = ['BG'] * len(bgname), ['FG'] * len(fgname)
    labels = Bs + Fs

    sounds = []
    means = np.empty((bins, len(names)))
    means[:] = np.NaN
    for cnt, sn, pth, ll in zip(range(len(labels)), names, paths, labels):
        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        dev = np.std(spec, axis=1)

        freq_mean = np.nanmean(spec, axis=1)
        x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
        csm = np.cumsum(freq_mean)
        big = np.max(csm)

        freq75 = x_freq[np.abs(csm - (big * 0.75)).argmin()]
        freq25 = x_freq[np.abs(csm - (big * 0.25)).argmin()]
        freq50 = x_freq[np.abs(csm - (big * 0.5)).argmin()]
        bandw = np.log2(freq75 / freq25)

        means[:, cnt] = freq_mean

        sounds.append({'name': sn,
                       'type': ll,
                       'std': dev,
                       'bandwidth': bandw,
                       '75th': freq75,
                       '25th': freq25,
                       'center': freq50,
                       'spec': spec,
                       'mean_freq': freq_mean,
                       'freq_stationary': np.std(freq_mean)})

    sound_df = pd.DataFrame(sounds)

    # Add mod spec calculations to sound_df, 2022_08_26
    mods = np.empty((sound_df.iloc[0].spec.shape[0], sound_df.iloc[0].spec.shape[1],
                     len(sound_df)))
    mods[:] = np.NaN
    mod_list = []
    for cnt, ii in enumerate(sound_df.name):
        row = sound_df.loc[sound_df.name == ii]
        spec = row['spec'].values[0]
        mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))
        mods[:, :, cnt] = mod
        mod_list.append(mod)
    avmod = np.nanmean(mods, axis=2)
    norm_list = [aa - avmod for aa in mod_list]
    avmod = avmod[:, :, np.newaxis]
    normmod = mods - avmod
    clow, chigh = np.min(normmod), np.max(normmod)
    sound_df['modspec'] = mod_list
    sound_df['normmod'] = norm_list
    # selfsounds['normmod'] = norm_list

    trimspec = [aa[24:, 30:69] for aa in sound_df['modspec']]
    negs = [aa[:, :20] for aa in trimspec]
    negs = [aa[:, ::-1] for aa in negs]
    poss = [aa[:, -20:] for aa in trimspec]
    trims = [(nn + pp) / 2 for (nn, pp) in zip(negs, poss)]
    sound_df['trimspec'] = trims

    # Collapses across each access
    ots = [np.nanmean(aa, axis=0) for aa in trims]
    ofs = [np.nanmean(aa, axis=1) for aa in trims]

    tbins, fbins = 100, 48

    wt = np.fft.fftshift(np.fft.fftfreq(tbins, 1 / tbins))
    wf = np.fft.fftshift(np.fft.fftfreq(fbins, 1 / 6))

    wt2 = wt[50:70]
    wf2 = wf[24:]

    cumwt = [np.cumsum(aa) / np.sum(aa) for aa in ots]
    bigt = [np.max(aa) for aa in cumwt]
    freq50t = [wt2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumwt, bigt)]

    cumft = [np.cumsum(aa) / np.sum(aa) for aa in ofs]
    bigf = [np.max(aa) for aa in cumft]
    freq50f = [wf2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumft, bigf)]

    sound_df['avgwt'], sound_df['avgft'] = ots, ofs
    sound_df['cumwt'], sound_df['cumft'] = cumwt, cumft
    sound_df['t50'], sound_df['f50'] = freq50t, freq50f
    sound_df['meanT'], sound_df['meanF'] = ots, ofs
    # End mod spec addition 2022_08_26

    # allmean = np.nanmean(means, axis=1, keepdims=True)
    # norm_mean = [aa / allmean for aa in sound_df.mean_freq]
    # freq_stationarity = [np.std(aa) for aa in allmean]
    # sound_df['norm_mean'],  = norm_mean
    # sound_df['freq_stationary'] = freq_stationarity

    ss = sound_df.explode('std')
    # frs = sound_df.explode('norm_mean')
    # frs = sound_df.explode('mean_freq')
    snames = [dd[2:] for dd in sound_df.name]

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(18, 8))

        sb.barplot(x='name', y='std', palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
                   data=ss, ci=68, ax=ax[0], errwidth=1)
        ax[0].set_xticklabels(snames, rotation=90, fontweight='bold', fontsize=7)
        ax[0].set_ylabel('Non-stationariness', fontweight='bold', fontsize=12)
        ax[0].spines['top'].set_visible(True), ax[0].spines['right'].set_visible(True)
        ax[0].set(xlabel=None)

        sb.barplot(x='name', y='bandwidth',
                   palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
                   data=sound_df, ax=ax[1])
        ax[1].set_xticklabels(snames, rotation=90, fontweight='bold', fontsize=7)
        ax[1].set_ylabel('Bandwidth (octaves)', fontweight='bold', fontsize=12)
        ax[1].spines['top'].set_visible(True), ax[1].spines['right'].set_visible(True)
        ax[1].set(xlabel=None)

        sb.barplot(x='name', y='freq_stationary',
                   palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
                   data=sound_df, ax=ax[2])
        ax[2].set_xticklabels(snames, rotation=90, fontweight='bold', fontsize=7)
        ax[2].set_ylabel('Frequency Non-stationariness', fontweight='bold', fontsize=12)
        ax[2].spines['top'].set_visible(True), ax[2].spines['right'].set_visible(True)
        ax[2].set(xlabel=None)

        fig.tight_layout()

    return sound_df


def plot_sound_stats(sound_df, metrics, labels=None, synth_kind=None, lines=None, sort=True):
    '''2023_06_13. Updated to include lines with averages of the stats with SEM
    2022_09_14. This is a way to look at the sound stats (passed as a list) from a sound_df and compare them. But also, if you add
    lines, a dictionary, which passes keys as those matching something found in stats, with a cutoff. That cut off will
    be drawn as a line on that subplot for that stat and it will also tell you what sounds are below that threshold,
    returning those sounds in a dictionary where the stat is a key and the values are a list of 'bad' sounds. Labels
    are optional, passing it will look prettier than it defaulting the labels to what the sound stat in the df.'''
    from scipy import stats
    if synth_kind:
        sound_df = sound_df.loc[sound_df.synth_kind == synth_kind]
    else:
        synth_kind = 'Natural'
    sound_df.rename(columns={'std': 'Tstationary', 'freq_stationary': 'Fstationary', 'RMS_norm_power': 'RMS_power',
                             'max_norm_power': 'max_power'}, inplace=True)
    sound_df['short_name'] = [dd.replace('_', '') for dd in sound_df.short_name]
    try:
        sound_df = sound_df.drop_duplicates('short_name')
    except:
        pass

    if isinstance(metrics, list):
        lens = len(metrics)
    elif isinstance(metrics, str):
        lens, stats = 1, [metrics]

    if lens <= 3:
        hh, ww = 1, lens
    else:
        hh, ww = int(np.ceil(lens / 3)), 3
    sound_df['Tstationary'] = [np.mean(aa) for aa in sound_df['Tstationary']]

    if sort:
        bgs, fgs = sound_df.loc[sound_df.type=='BG'], sound_df.loc[sound_df.type=='FG']
        bgs = bgs.sort_values('short_name')
        vv, oo, nn = fgs.loc[fgs.Vocalization=='Yes'], fgs.loc[fgs.Vocalization=='Other'], fgs.loc[fgs.Vocalization=='No']
        vv, oo, nn = vv.sort_values('short_name'), oo.sort_values('short_name'), nn.sort_values('short_name')
        sound_df = pd.concat([bgs, vv, oo, nn])

    fig, axes = plt.subplots(hh, ww, figsize=(ww * 5, hh * 5))
    axes = np.ravel(axes)

    bads, stat = {}, {}
    for cnt, (ax, st) in enumerate(zip(axes, metrics)):
        sb.barplot(x='short_name', y=st,
                   palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
                   data=sound_df, ci=68, ax=ax)
        ax.set_xticklabels(sound_df.short_name, rotation=90, fontsize=7) #fontweight='bold')
        if labels:
            ax.set_ylabel(labels[cnt], fontweight='bold', fontsize=12)
        else:
            ax.set_ylabel(metrics[cnt], fontweight='bold', fontsize=12)
        ax.spines['top'].set_visible(True), ax.spines['right'].set_visible(True)
        ax.set(xlabel=None)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)

        if sort:
            both = sound_df[['type', st, 'Vocalization']]
            bgs, fgs = both.loc[both.type=='BG'], both.loc[both.type=='FG']
            vv, oo, nn = fgs.loc[fgs.Vocalization == 'Yes'], fgs.loc[fgs.Vocalization == 'Other'], fgs.loc[
                fgs.Vocalization == 'No']
            bg_med, vv_med, oo_med, nn_med = np.median(bgs[st]), np.median(vv[st]), np.median(oo[st]), np.median(nn[st])
            fg_med = np.median(fgs[st])
            bg_med, bg_err = jack_mean_err(bgs[st], do_median=True)
            fg_med, fg_err = jack_mean_err(fgs[st], do_median=True)
            vv_med, vv_err = jack_mean_err(vv[st], do_median=True)
            oo_med, oo_err = jack_mean_err(oo[st], do_median=True)
            nn_med, nn_err = jack_mean_err(nn[st], do_median=True)
            stat[f'bg_{st}'], stat[f'fg_{st}'] = (bg_med, bg_err), (fg_med, fg_err)
            stat[f'voc_{st}'], stat[f'oth_{st}'], stat[f'nonv_{st}'] = (vv_med, vv_err), (oo_med, oo_err), (nn_med, nn_err)
            bglen, vvlen, oolen, nnlen = len(bgs), len(vv), len(oo), len(nn)
            xmin, xmax = ax.get_xlim()
            ax.hlines(bg_med, xmin=xmin, xmax=xmin+bglen, ls='--', color='dodgerblue')
            ax.hlines(vv_med, xmin=xmin+bglen, xmax=xmin+bglen+vvlen, ls='--', color='olivedrab')
            ax.hlines(oo_med, xmin=xmin+bglen+vvlen, xmax=xmin+bglen+vvlen+oolen, ls='--', color='olivedrab')
            ax.hlines(nn_med, xmin=xmin+bglen+vvlen+nnlen, xmax=xmin+bglen+vvlen+oolen+nnlen, ls='--', color='olivedrab')

        else:
            both = sound_df[['type', st]]
            bgs, fgs = both.loc[both.type=='BG'], both.loc[both.type=='FG']
            bg_med, bg_err = jack_mean_err(bgs[st], do_median=True)
            fg_med, fg_err = jack_mean_err(fgs[st], do_median=True)
            stat[f'bg_{st}'], stat[f'fg_{st}'] = (bg_med, bg_err), (fg_med, fg_err)
            # std = sound_df[['type', st]].groupby(by='type', as_index=False).sem()
            bglen, fglen = sound_df['type'].value_counts()
            xmin, xmax = ax.get_xlim()

            ax.hlines(bg_med, xmin=xmin, xmax=xmin+bglen, ls='--', color='dodgerblue')
            ax.hlines(fg_med, xmin=xmin+bglen, xmax=xmax, ls='--', color='olivedrab')

        # if fill:
        #     ax.fill_between([xmin, xmin+bglen], avg.loc[avg.type=='BG'][st].values[0] + std.loc[std.type=='BG'][st].values[0],
        #                     avg.loc[avg.type == 'BG'][st].values[0] - std.loc[std.type == 'BG'][st].values[0],
        #                        alpha=0.3, color='dodgerblue')
        #     ax.fill_between([xmin+bglen, xmax], avg.loc[avg.type=='FG'][st].values[0] + std.loc[std.type=='FG'][st].values[0],
        #                     avg.loc[avg.type == 'FG'][st].values[0] - std.loc[std.type == 'FG'][st].values[0],
        #                        alpha=0.3, color='olivedrab')

        stat[f'bg_{st}'], stat[f'fg_{st}']
        # tt = stats.ttest_ind(sound_df.loc[sound_df.type=='BG'][st], sound_df.loc[sound_df.type=='FG'][st])
        tt = stats.mannwhitneyu(sound_df.loc[sound_df.type=='BG'][st], sound_df.loc[sound_df.type=='FG'][st])
        stat[f'stats_{st}'] = tt.pvalue
        if cnt == 0:
            ax.set_title(f'Synth: {synth_kind}, p={np.around(tt.pvalue, 3)}', fontsize=8, fontweight='bold')
        else:
            ax.set_title(f'p={np.around(tt.pvalue, 3)}', fontsize=8, fontweight='bold')

        if lines:
            if st in lines.keys():
                xmin, xmax = ax.get_xlim()
                ax.hlines(lines[st], xmin=xmin, xmax=xmax, ls=':', color='black')
                ax.set_xlim(xmin, xmax)
                bad_df = sound_df.loc[sound_df[st] <= lines[st]]
                bads[st] = bad_df.short_name.tolist()
    # axes[0].set_title(f"Synth: {synth_kind}", fontsize=10, fontweight='bold')
    fig.tight_layout()
    if lines:
        return bads
    else:
        return stat


def sound_stat_violin(df, mets, met_labels):
    '''2023_07_03. Quickly made this as an alternative and more concise version of the sound stat bar plot.
    This just makes a violin plot of the given statistics you ask for. Make sure you also provide a
    corresponding list of how you want them to be labeled. DF you pass is the straight df you load
    and the bad sounds will get taken out in the function and the sound_df generated.'''
    bads = ['CashRegister', 'Heels', 'Castinets', 'Dice']  # RMS Power Woodblock
    df = df.loc[df['BG'].apply(lambda x: x not in bads)]
    df = df.loc[df['FG'].apply(lambda x: x not in bads)]
    sound_df = ohel.get_sound_statistics_from_df(df, percent_lims=[15,85], append=False)
    sounds = sound_df.loc[sound_df.synth_kind=='N']

    fig, ax = plt.subplots(1, len(mets), figsize=(len(mets)*3,4))
    for cnt, mt in enumerate(mets):
        sn.violinplot(data=sounds, x="type", y=mt, ax=ax[cnt])
        ax[cnt].set_xlabel('')
        ax[cnt].set_xticklabels(sound_df.type.unique().tolist(), fontweight='bold', fontsize=10)
        ax[cnt].set_ylabel(met_labels[cnt], fontweight='bold', fontsize=10)

        tt = stats.ttest_ind(sounds.loc[sounds.type=='BG'][mt], sounds.loc[sounds.type=='FG'][mt]).pvalue
        ax[cnt].set_title(f"p={np.around(tt, 5)}")
    fig.tight_layout()


def add_sound_stats(weight_df, sound_df):
    '''Only for use with get_sound_statistics_full (above). It separately takes the sound_df created by
    that function and adds it to your data table (weight_df). Largely DEFUNCT by get_sound_statistics_from_df
    in 2023_05, which skips this step by having the option to append within the function, making for
    easier handling of this without all these stupid lines of code.

    Updated 2022_09_23. Added t50 and f50 and modspec stuff to weight_df
    Updated 2022_09_13. Previously it just added the T, band, and F stats to the dataframe.
    I updated it so that it takes synth kind into account when adding the statistics, and
    also adds RMS and max power for the sounds.'''
    BGdf, FGdf = sound_df.loc[sound_df.type == 'BG'], sound_df.loc[sound_df.type == 'FG']
    BGmerge, FGmerge = pd.DataFrame(), pd.DataFrame()
    BGmerge['BG'] = [aa[2:].replace(' ', '') for aa in BGdf.name]
    BGmerge['BG_Tstationary'] = BGdf.Tstationary.tolist()
    BGmerge['BG_bandwidth'] = BGdf.bandwidth.tolist()
    BGmerge['BG_Fstationary'] = BGdf.Fstationary.tolist()
    BGmerge['BG_Tstationary_start'] = BGdf.Tstationary_start.tolist()
    BGmerge['BG_bandwidth_start'] = BGdf.bandwidth_start.tolist()
    BGmerge['BG_Fstationary_start'] = BGdf.Fstationary_start.tolist()
    BGmerge['BG_Tstationary_end'] = BGdf.Tstationary_end.tolist()
    BGmerge['BG_bandwidth_end'] = BGdf.bandwidth_end.tolist()
    BGmerge['BG_Fstationary_end'] = BGdf.Fstationary_end.tolist()
    BGmerge['BG_RMS_power'] = BGdf.RMS_norm_power.tolist()
    BGmerge['BG_max_power'] = BGdf.max_norm_power.tolist()
    BGmerge['BG_temp_ps'] = BGdf.temp_ps.tolist()
    BGmerge['BG_temp_ps_std'] = BGdf.temp_ps_std.tolist()
    BGmerge['BG_freq_ps'] = BGdf.freq_ps.tolist()
    BGmerge['BG_freq_ps_std'] = BGdf.freq_ps_std.tolist()
    # BGmerge['BG_avgwt'] = BGdf.avgwt.tolist()
    # BGmerge['BG_avgft'] = BGdf.avgft.tolist()
    # BGmerge['BG_cumwt'] = BGdf.cumwt.tolist()
    # BGmerge['BG_cumft'] = BGdf.cumft.tolist()
    BGmerge['BG_t50'] = BGdf.t50.tolist()
    BGmerge['BG_f50'] = BGdf.f50.tolist()
    BGmerge['synth_kind'] = BGdf.synth_kind.tolist()

    FGmerge['FG'] = [aa[2:].replace(' ', '') for aa in FGdf.name]
    FGmerge['FG_Tstationary'] = FGdf.Tstationary.tolist()
    FGmerge['FG_bandwidth'] = FGdf.bandwidth.tolist()
    FGmerge['FG_Fstationary'] = FGdf.Fstationary.tolist()
    FGmerge['FG_Tstationary_start'] = FGdf.Tstationary_start.tolist()
    FGmerge['FG_bandwidth_start'] = FGdf.bandwidth_start.tolist()
    FGmerge['FG_Fstationary_start'] = FGdf.Fstationary_start.tolist()
    FGmerge['FG_Tstationary_end'] = FGdf.Tstationary_end.tolist()
    FGmerge['FG_bandwidth_end'] = FGdf.bandwidth_end.tolist()
    FGmerge['FG_Fstationary_end'] = FGdf.Fstationary_end.tolist()
    FGmerge['FG_RMS_power'] = FGdf.RMS_norm_power.tolist()
    FGmerge['FG_max_power'] = FGdf.max_norm_power.tolist()
    FGmerge['FG_temp_ps'] = FGdf.temp_ps.tolist()
    FGmerge['FG_temp_ps_std'] = FGdf.temp_ps_std.tolist()
    FGmerge['FG_freq_ps'] = FGdf.freq_ps.tolist()
    FGmerge['FG_freq_ps_std'] = FGdf.freq_ps_std.tolist()
    # FGmerge['FG_avgwt'] = FGdf.avgwt.tolist()
    # FGmerge['FG_avgft'] = FGdf.avgft.tolist()
    # FGmerge['FG_cumwt'] = FGdf.cumwt.tolist()
    # FGmerge['FG_cumft'] = FGdf.cumft.tolist()
    FGmerge['FG_t50'] = FGdf.t50.tolist()
    FGmerge['FG_f50'] = FGdf.f50.tolist()
    FGmerge['synth_kind'] = FGdf.synth_kind.tolist()

    weight_df = pd.merge(right=BGmerge, left=weight_df, on=['BG', 'synth_kind'], validate='m:1')
    weight_df = pd.merge(right=FGmerge, left=weight_df, on=['FG', 'synth_kind'], validate='m:1')

    return weight_df


def get_sound_statistics_from_df(df, percent_lims=[15, 85], area=None, append=True, fs=100):
    '''2023_05_22. Updated to include new spectral correlation metric. Also now takes an input that dictates by
    what amount of the power spectrum you will be filtering a sound for its bandwidth and spectral correlation.

    2023_05_16. Updated to take an input df that has unique paths for each BG and FGs uniquely used
    throughout the dataframe, so it doesn't do it for extras and the path is what references back. Should
    be easy to add new sound statistics to the big df as we decide on them.

    Updated 2022_09_13. Added mean relative gain for each sound. The rel_gain is BG or FG
    respectively.
    Updated 2022_09_12. Now it can take a DF that has multiple synthetic conditions and pull
    the stats for the synthetic sounds. The dataframe will label these by column synth_kind
    and you should pull out them that way, because they all have the same name in the name
    column. Additionally, RMS normalization stats were added in RMS_norm and max_norm powers.
    5/12/22 Takes a cellid and batch and figures out all the sounds that were played
    in that experiment and calculates some stastistics it plots side by side. Also outputs
    those numbers in a cumbersome dataframe'''
    lfreq, hfreq, bins = 100, 24000, 48

    if area:
        df = df.loc[df.area==area]

    synths = list(df.synth_kind.unique())
    if len(synths)==2 or synths==['N']:
        synths.sort(reverse=True)
        simple_names = []

    df['BG_filt_name'], df['FG_filt_name'] = [dd.replace(' ', '') for dd in df['BG']], [dd.replace(' ', '') for dd in df['FG']]
    df['BG_filt_name'], df['FG_filt_name'] = [dd.replace('_', '') for dd in df['BG_filt_name']], \
                                             [dd.replace('_', '') for dd in df['FG_filt_name']]

    # To avoid passing cuts as a parameter, just use it if it's in the dataframe, if not, pass none
    try:
        cuts = df.fit_segment.unique()[0].split('-')
        # Pulls it out of dataframe as a string (in ms), convert to ints and s
        cuts[0], cuts[1] = int(cuts[0]), int(cuts[1]) / 1000
    except:
        cuts = None

    # Going to make a dataframe for FGs and for BGs separately
    the_dfs = {}
    for ll in ['BG', 'FG']:
        # Split any dataframe into the synth kinds (as those have the same names) and do the same thing for each
        # before recombining
        syn_df, bad_dict = [], {}
        for syn in synths:
            # This is getting the mean rel gain for each sound (FG rel gain for FGs, etc)
            synth_df = df.loc[df.synth_kind == syn].copy()
            synth_df = synth_df.sort_values(f'{ll}_filt_name')
            bad_idx = []

            gain_df = synth_df[[f'{ll}_filt_name', f'{ll}_rel_gain']]
            mean_df = gain_df.groupby(by=f'{ll}_filt_name').agg(mean=(f'{ll}_rel_gain', np.mean)).reset_index(). \
                rename(columns={f'{ll}_filt_name': f'{ll}_short_name'})

            # get the paths to the specific sounds used within this subset of the data
            paths = [list(synth_df.loc[synth_df[f'{ll}_filt_name']==dd][f'{ll}_path'])[0] for dd in mean_df[f'{ll}_short_name']]
            # paths = list(synth_df[f'{ll}_path'].unique())
            names = [bb.split('/')[-1].split('.')[0] for bb in paths]
            flt_names = [dd[2:].replace(' ', '') for dd in names]
            flt_names = [dd.replace('_', '') for dd in flt_names]

            sounds = []
            means = np.empty((bins, len(names)))
            means[:] = np.NaN

            for cnt, sn, pth, flt in zip(range(len(paths)), names, paths, flt_names):
                sfs, W = wavfile.read(pth)
                spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

                # to measure rms power... for rms-normed signals:
                rms_normed = np.std(remove_clicks(W / W.std(), 15))
                # for max-normed signals:
                max_normed = np.std(W / np.abs(W).max()) * 5

                dev = np.std(spec, axis=1)

                freq_dev = np.std(spec, axis=0)
                freq_mean = np.nanmean(spec, axis=1)
                x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
                csm = np.cumsum(freq_mean)
                big = np.max(csm)

                # 2023_05_22. New spectral correlation metric
                lower, upper = percent_lims[0] / 100, percent_lims[1] / 100
                bin_high = np.abs(csm - (big * upper)).argmin()
                bin_low = np.abs(csm - (big * lower)).argmin()
                bandwidth = np.log2(x_freq[bin_high] / x_freq[bin_low])

                # Chops the spectrogram before calculating spectral metric
                cut_spec = spec[bin_low:bin_high, :]
                cc = np.corrcoef(cut_spec)
                cpow = cc[np.triu_indices(cut_spec.shape[0], k=1)].mean()

                freq_range = (int(x_freq[bin_low]), int(x_freq[bin_high]))

                freq75 = x_freq[np.abs(csm - (big * 0.75)).argmin()]
                freq25 = x_freq[np.abs(csm - (big * 0.25)).argmin()]
                freq50 = x_freq[np.abs(csm - (big * 0.5)).argmin()]
                bandw = np.log2(freq75 / freq25)

                means[:, cnt] = freq_mean

                # 2022_09_23 Adding power spectrum stats
                temp = np.abs(np.fft.fft(spec, axis=1))
                freq = np.abs(np.fft.fft(spec, axis=0))

                temp_ps = np.sum(np.abs(np.fft.fft(spec, axis=1)), axis=0)[1:].std()
                freq_ps = np.sum(np.abs(np.fft.fft(spec, axis=0)), axis=1)[1:].std()

                if len(synths) == 2 or synths==['N']:
                    if flt in simple_names:
                        bad_idx.append(cnt)
                        print(f'Saving index cnt={cnt} to take {flt}, or {sn} out.')
                    else:
                        simple_names.append(flt)
                        print(f'Adding {flt}, or {sn}. While cnt = {cnt}')



                sounds.append({f'{ll}_name': sn, #.split('_')[0],
                               'synth_kind': syn,
                               f'{ll}_Tstationary': np.nanmean(dev),
                               f'{ll}_Fcorr': cpow,
                               f'{ll}_bandwidth': bandwidth,
                               'bw_percent': f'{percent_lims[0]}/{percent_lims[1]}',
                               f'{ll}_freq_range': freq_range,
                               f'{ll}_75th': freq75,
                               f'{ll}_25th': freq25,
                               f'{ll}_bw_25/75': bandw,
                               f'{ll}_center': freq50,
                               f'{ll}_spec': spec,
                               f'{ll}_mean_freq': freq_mean,
                               f'{ll}_Fstationary_wrong': np.std(freq_mean),
                               f'{ll}_Fstationary': np.nanmean(freq_dev),
                               f'{ll}_RMS_power': rms_normed,
                               f'{ll}_max_power': max_normed,
                               f'{ll}_temp_ps': temp,
                               f'{ll}_freq_ps': freq,
                               f'{ll}_temp_ps_std': temp_ps,
                               f'{ll}_freq_ps_std': freq_ps,
                               # f'{ll}_short_name': sn[2:].split('_')[0].replace(' ', ''),
                               # f'{ll}_short_name': pth.split('/')[-1].split('.')[0][2:].replace(' ',''),
                               f'{ll}_short_name': flt,
                               f'{ll}_path': pth})

                if cuts:
                    start_gain_df = synth_df[[f'{ll}_filt_name', f'{ll}_rel_gain_start']]
                    start_mean_df = start_gain_df.groupby(by=f'{ll}_filt_name').agg(mean=(f'{ll}_rel_gain_start', np.mean)).reset_index(). \
                        rename(columns={f'{ll}_filt_name': f'{ll}_short_name'})
                    start_mean_df.rename(columns={'mean': f'{ll}_rel_gain_avg_start'}, inplace=True)

                    end_gain_df = synth_df[[f'{ll}_filt_name', f'{ll}_rel_gain_end']]
                    end_mean_df = end_gain_df.groupby(by=f'{ll}_filt_name').agg(mean=(f'{ll}_rel_gain_end', np.mean)).reset_index(). \
                        rename(columns={f'{ll}_filt_name': f'{ll}_short_name'})
                    end_mean_df.rename(columns={'mean': f'{ll}_rel_gain_avg_end'}, inplace=True)

                    one, two = spec[:, cuts[0]:int(cuts[1] * fs)], spec[:, int(cuts[1] * fs):]
                    t_dev_start, t_dev_end = np.std(one, axis=1), np.std(two, axis=1)
                    f_dev_start, f_dev_end = np.std(one, axis=0), np.std(two, axis=0)

                    freq_mean_start, freq_mean_end = np.nanmean(one, axis=1), np.nanmean(two, axis=1)
                    x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
                    csm_start, csm_end = np.cumsum(freq_mean_start), np.cumsum(freq_mean_end)
                    big_start, big_end = np.max(csm_start), np.max(csm_end)

                    freq75_start = x_freq[np.abs(csm_start - (big_start * 0.75)).argmin()]
                    freq25_start = x_freq[np.abs(csm_start - (big_start * 0.25)).argmin()]
                    bandw_start = np.log2(freq75_start / freq25_start)

                    freq75_end = x_freq[np.abs(csm_end - (big_end * 0.75)).argmin()]
                    freq25_end = x_freq[np.abs(csm_end - (big_end * 0.25)).argmin()]
                    bandw_end = np.log2(freq75_end / freq25_end)

                    # 2023_05_22. New spectral metric
                    lower, upper = percent_lims[0] / 100, percent_lims[1] / 100
                    bin_high_start = np.abs(csm_start - (big_start * upper)).argmin()
                    bin_low_start = np.abs(csm_start - (big_start * lower)).argmin()
                    bandwidth_start = np.log2(x_freq[bin_high_start] / x_freq[bin_low_start])
                    freq_range_start = (int(x_freq[bin_low_start]), int(x_freq[bin_high_start]))

                    # Chops the spectrogram before calculating spectral metric
                    cut_spec_start = one[bin_low_start:bin_high_start, :]
                    cc_start = np.corrcoef(cut_spec_start)
                    cpow_start = cc_start[np.triu_indices(cut_spec_start.shape[0], k=1)].mean()

                    bin_high_end = np.abs(csm_end - (big_end * upper)).argmin()
                    bin_low_end = np.abs(csm_end - (big_end * lower)).argmin()
                    bandwidth_end = np.log2(x_freq[bin_high_end] / x_freq[bin_low_end])
                    freq_range_end = (int(x_freq[bin_low_end]), int(x_freq[bin_high_end]))

                    cut_spec_end = two[bin_low_start:bin_high_end, :]
                    cc_end = np.corrcoef(cut_spec_end)
                    cpow_end = cc_end[np.triu_indices(cut_spec_end.shape[0], k=1)].mean()

                    # sounds[cnt][f'{ll}_rel_gain_avg_start'] = start_mean_df[f'{ll}_rel_gain_avg_start']
                    # sounds[cnt][f'{ll}_rel_gain_avg_end'] = end_mean_df[f'{ll}_rel_gain_avg_end']
                    sounds[cnt][f'{ll}_Tstationary_start'] = np.nanmean(t_dev_start)
                    sounds[cnt][f'{ll}_Tstationary_end'] = np.nanmean(t_dev_end)
                    sounds[cnt][f'{ll}_Fstationary_start'] = np.nanmean(f_dev_start)
                    sounds[cnt][f'{ll}_Fstationary_end'] = np.nanmean(f_dev_end)
                    sounds[cnt][f'{ll}_bandwidth_25/75_start'] = bandw_start
                    sounds[cnt][f'{ll}_bandwidth_25/75_end'] = bandw_end
                    sounds[cnt][f'{ll}_bandwidth_start'] = bandwidth_start
                    sounds[cnt][f'{ll}_bandwidth_end'] = bandwidth_end
                    sounds[cnt][f'{ll}_freq_range_start'] = freq_range_start
                    sounds[cnt][f'{ll}_freq_range_end'] = freq_range_end
                    sounds[cnt][f'{ll}_Fcorr_start'] = cpow_start
                    sounds[cnt][f'{ll}_Fcorr_end'] = cpow_end

            sound_df = pd.DataFrame(sounds)
            # Merge the relative gain data into the DF of sounds
            sound_df = pd.merge(sound_df, mean_df, on=f'{ll}_short_name').rename(columns={'mean': f'{ll}_rel_gain_avg'})
            if cuts:
                sound_df = pd.merge(sound_df, start_mean_df, on=f'{ll}_short_name')
                sound_df = pd.merge(sound_df, end_mean_df, on=f'{ll}_short_name')

            # Add mod spec calculations to sound_df, 2022_08_26
            mods = np.empty((sound_df.iloc[0][f'{ll}_spec'].shape[0], sound_df.iloc[0][f'{ll}_spec'].shape[1],
                             len(sound_df)))
            mods[:] = np.NaN
            mod_list = []
            for cnt, ii in enumerate(sound_df[f'{ll}_name']):
                row = sound_df.loc[sound_df[f'{ll}_name'] == ii]
                spec = row[f'{ll}_spec'].values[0]
                mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))
                mods[:, :, cnt] = mod
                mod_list.append(mod)
            avmod = np.nanmean(mods, axis=2)
            norm_list = [aa - avmod for aa in mod_list]
            avmod = avmod[:, :, np.newaxis]
            normmod = mods - avmod
            clow, chigh = np.min(normmod), np.max(normmod)
            sound_df[f'{ll}_modspec'] = mod_list
            sound_df[f'{ll}_normmod'] = norm_list
            # selfsounds['normmod'] = norm_list

            trimspec = [aa[24:, 30:69] for aa in sound_df[f'{ll}_modspec']]
            negs = [aa[:, :20] for aa in trimspec]
            negs = [aa[:, ::-1] for aa in negs]
            poss = [aa[:, -20:] for aa in trimspec]
            trims = [(nn + pp) / 2 for (nn, pp) in zip(negs, poss)]
            sound_df[f'{ll}_trimspec'] = trims

            # Collapses across each access
            ots = [np.nanmean(aa, axis=0) for aa in trims]
            ofs = [np.nanmean(aa, axis=1) for aa in trims]

            tbins, fbins = 100, 48

            wt = np.fft.fftshift(np.fft.fftfreq(tbins, 1 / tbins))
            wf = np.fft.fftshift(np.fft.fftfreq(fbins, 1 / 6))

            wt2 = wt[50:70]
            wf2 = wf[24:]

            cumwt = [np.cumsum(aa) / np.sum(aa) for aa in ots]
            bigt = [np.max(aa) for aa in cumwt]
            freq50t = [wt2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumwt, bigt)]

            cumft = [np.cumsum(aa) / np.sum(aa) for aa in ofs]
            bigf = [np.max(aa) for aa in cumft]
            freq50f = [wf2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumft, bigf)]

            sound_df[f'{ll}_avgwt'], sound_df[f'{ll}_avgft'] = ots, ofs
            sound_df[f'{ll}_cumwt'], sound_df[f'{ll}_cumft'] = cumwt, cumft
            sound_df[f'{ll}_t50'], sound_df[f'{ll}_f50'] = freq50t, freq50f
            sound_df[f'{ll}_meanT'], sound_df[f'{ll}_meanF'] = ots, ofs

            sound_df.drop(bad_idx, inplace=True)

            # End mod spec addition 2022_08_26
            syn_df.append(sound_df)

        main_df = pd.concat(syn_df)

        # 2023_05_17. This is where you pick what things move on to the next round from sound_df
        edit_df = main_df[[f'{ll}_name', 'synth_kind', f'{ll}_Tstationary', f'{ll}_bandwidth', f'{ll}_Fcorr',
                           # f'{ll}_Fstationary_wrong', f'{ll}_Fstationary',
                           f'{ll}_freq_range', f'{ll}_RMS_power',
                           f'{ll}_max_power', f'{ll}_temp_ps_std', f'{ll}_freq_ps_std',
                           f'{ll}_short_name', f'{ll}_path', f'{ll}_rel_gain_avg', f'{ll}_t50', f'{ll}_f50',
                           # f'{ll}_25th', f'{ll}_75th'
                           ]]

        # Adds the stuff that takes place in the cut loop, if it exists
        if cuts:
            cut_df = main_df[[f'{ll}_rel_gain_avg_start', f'{ll}_rel_gain_avg_end',
                              f'{ll}_Tstationary_start', f'{ll}_Tstationary_end',
                              # f'{ll}_Fstationary_start', f'{ll}_Fstationary_end',
                              f'{ll}_Fcorr_start', f'{ll}_Fcorr_end',
                              f'{ll}_freq_range_start', f'{ll}_freq_range_end',
                              f'{ll}_bandwidth_start', f'{ll}_bandwidth_end']]
            edit_df = pd.concat([edit_df, cut_df], axis=1)

        the_dfs[ll] = edit_df

    # 2023_05_17. Option either lets you return your old df with this appended on it, or the sounds by themselves
    if append == True:
        # print(f'Before append, df is len={len(df)}')
        # print(f'Before append, the_dfs["BG"] is len={len(the_dfs["BG"])}')
        # the_dfs['BG'].rename(columns={'BG_short_name': 'BG', 'BG_rel_gain': 'BG_rel_gain_all'}, inplace=True)
        the_dfs['BG'].rename(columns={'BG_short_name': 'BG_filt_name'}, inplace=True)
        df = pd.merge(right=the_dfs['BG'], left=df, on=['BG_filt_name', 'synth_kind', 'BG_path'], validate='m:1')
        # print(f"After merging with the_dfs['BG'], df is now len={len(df)}")
        # print(f'Before append, the_dfs["FG"] is len={len(the_dfs["FG"])}')
        # the_dfs['FG'].rename(columns={'FG_short_name': 'FG', 'FG_rel_gain': 'FG_rel_gain_all'}, inplace=True)
        the_dfs['FG'].rename(columns={'FG_short_name': 'FG_filt_name'}, inplace=True)
        df = pd.merge(right=the_dfs['FG'], left=df, on=['FG_filt_name', 'synth_kind', 'FG_path'], validate='m:1')
        # print(f"After merging with the_dfs['FG'], df is now len={len(df)}")
        df['bw_percent'] = f'{percent_lims[0]}/{percent_lims[1]}'

        #2023_08_02. Adding spectral overlap.
        if cuts:
            suffixes = ['', '_start', '_end']
        else:
            suffixes = ['']
        for ss in suffixes:
            BG_low, BG_high = [aa[0] for aa in df[f'BG_freq_range{ss}']], [aa[1] for aa in df[f'BG_freq_range{ss}']]
            FG_low, FG_high = [aa[0] for aa in df[f'FG_freq_range{ss}']], [aa[1] for aa in df[f'FG_freq_range{ss}']]

            overlap_max_low = [np.max([dd, aa]) for dd, aa in zip(BG_low, FG_low)]
            overlap_min_high = [np.min([dd, aa]) for dd, aa in zip(BG_high, FG_high)]

            octave_ol = [np.log2(dd / aa) for dd, aa in zip(overlap_min_high, overlap_max_low)]
            overlap = [dd if dd > 0 else 0 for dd in octave_ol]

            df[f'BG_spectral_overlap{ss}'] = [(oo / bbw) * 100 for oo, bbw in zip(overlap, list(df[f'BG_bandwidth{ss}']))]
            df[f'FG_spectral_overlap{ss}'] = [(oo / bbw) * 100 for oo, bbw in zip(overlap, list(df[f'FG_bandwidth{ss}']))]

        return df

    else:
        for aa in the_dfs.keys():
            the_dfs[aa]['bw_percent'] = f'{percent_lims[0]}/{percent_lims[1]}'
        the_dfs['BG']['type'], the_dfs['FG']['type'] = 'BG', 'FG'

        bg_rn = {key: (key[3:] if len(key.split('_')) > 1 else key) for key in the_dfs['BG'].columns.to_list() if
                 key[:2] == 'BG'}
        fg_rn = {key: (key[3:] if len(key.split('_')) > 1 else key) for key in the_dfs['FG'].columns.to_list() if
                 key[:2] == 'FG'}
        bgs, fgs = the_dfs['BG'].rename(columns=bg_rn), the_dfs['FG'].rename(columns=fg_rn)
        bgs, fgs = bgs.sort_values(by='name'), fgs.sort_values(by='name')

        dfs = pd.concat([bgs, fgs])

        return dfs


def plot_example_specs(sound_df, sound_idx, lfreq=100, hfreq=24000, bins=48):
    more = int(np.ceil(len(sound_idx)/2))
    x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
    ticks = [0, 10, 20, 30, 40, bins-1]
    labels = [int(np.ceil(x_freq[aa])) for aa in ticks]

    f, axes = plt.subplots(2, more, sharex=True, sharey=True, figsize=(8, 3*more))
    ax = axes.ravel(order='F')
    for AX, ii in zip(ax, sound_idx):
        row = sound_df.iloc[ii]
        AX.imshow(row['spec'], aspect='auto', origin='lower')
        AX.set_yticks(ticks)
        AX.set_yticklabels(labels)
        AX.set_ylabel("Frequency (Hz)", fontweight='bold', fontsize=8)
        AX.set_xlabel("Time (s)", fontweight='bold', fontsize=8)
        AX.set_title(f"{row.type}: {row['name'][2:]}", fontweight='bold', fontsize=10)


def filter_weight_df(df, suffixes=['_end', '_start'], fr_thresh=0.03, r_thresh=0.6, quad_return=3, bin_kind='11', area=None,
                     synth_kind='A', weight_lims=[-1, 2], bads=True, bad_filt={'RMS_power': 0.95, 'max_power': 0.3}):
    '''2022_11_09. Takes a df of weights and you can specify a bunch of filters I commonly use and will apply them all
    separately, which is nice because I've always done this as a clusterfuck.'''
    if area:
        df = df.loc[df.area==area]
    if fr_thresh:
        filt_labels = [[f"bg_FR{fl}", f"fg_FR{fl}"] for fl in suffixes]
        for ff in filt_labels:
            if quad_return == 3:
                df = df.loc[(df[ff[0]] >= fr_thresh) & (df[ff[1]] >= fr_thresh)]
            elif quad_return == 2:
                df = df.loc[(np.abs(df[ff[0]]) <= fr_thresh) & (df[ff[1]] >= fr_thresh)]
            elif quad_return == 6:
                df = df.loc[(df[ff[0]] >= fr_thresh) & (np.abs(df[ff[1]]) <= fr_thresh)]
            else:
                raise ValueError(f"quad_return parameter must be 3, 2, or 6, you gave {quat_return}.")
    if r_thresh:
        r_filt_labels = [[f"r{fl}", f"r{fl}"] for fl in suffixes]
        for rf in r_filt_labels:
            df = df.loc[(df[rf[0]] >= r_thresh) & (df[rf[1]] >= r_thresh)]
    if bin_kind:
        df = df.loc[df.kind==bin_kind]
    if synth_kind:
        df = df.loc[df.synth_kind==synth_kind]
    if weight_lims:
        if isinstance(weight_lims, int):
            weight_lims = [-np.abs(weight_lims), np.abs(weight_lims)]
        if len(weight_lims) == 2:
            w_filt_labels = [[f"weightsA{fl}", f"weightsB{fl}"] for fl in suffixes]
            for wf in w_filt_labels:
                df = df.loc[(df[wf[0]] < weight_lims[1]) & (df[wf[0]] > weight_lims[0]) &
                            (df[wf[1]] < weight_lims[1]) & (df[wf[1]] > weight_lims[0])]
        else:
            raise ValueError(f"You put '{weight_lims}' as your weight cuts, make it two values or a single int.")
    if bads == True:
        sound_df = get_sound_statistics_full(df)
        if synth_kind == 'A':
            stat = 'max_power'
            bad_dict = plot_sound_stats(sound_df, [stat], labels=['Max Power'],
                                             lines={stat: bad_filt[stat]}, synth_kind=synth_kind)
            bads = list(bad_dict[stat])
        else:
            stat = 'RMS_power'
            bad_dict = plot_sound_stats(sound_df, [stat], labels=['RMS Power'],
                                             lines={stat: bad_filt[stat]}, synth_kind='N')
            bads = list(bad_dict[stat])
        df = df.loc[df['BG'].apply(lambda x: x not in bads)]
        df = df.loc[df['FG'].apply(lambda x: x not in bads)]

    return df


def filter_synth_df_by(df_full, use='N', suffixes=['', '_start', '_end'], fr_thresh=0.03, r_thresh=0.6,
                       quad_return=3, bin_kind='11', weight_lims=[-1.5, 2.5], area=None):
    '''2022_12_01. You give it a DF with many synthetic conditions, specify the one you want to be the condition
    that is used as the filter for a variety of metrics and it returns the dataframe. Also can do area or not.'''
    df_full = df_full.loc[df_full.kind==bin_kind]
    df_full['filt_id'] = df_full.cellid + '-' + df_full.BG + '-' + df_full.FG

    if area:
        df_full = df_full.loc[df_full.area==area]

    synths = list(df_full.synth_kind.unique())
    synth_dfs = {x: df_full.loc[df_full.synth_kind==x].sort_values(by=['filt_id']).reset_index(drop=True) for x in synths}

    df = synth_dfs[use]

    if fr_thresh:
        filt_labels = [[f"bg_FR{fl}", f"fg_FR{fl}"] for fl in suffixes]
        for ff in filt_labels:
            if quad_return == 3:
                df = df.loc[(df[ff[0]] >= fr_thresh) & (df[ff[1]] >= fr_thresh)]
            elif quad_return == 2:
                df = df.loc[(np.abs(df[ff[0]]) <= fr_thresh) & (df[ff[1]] >= fr_thresh)]
            elif quad_return == 6:
                df = df.loc[(df[ff[0]] >= fr_thresh) & (np.abs(df[ff[1]]) <= fr_thresh)]
            else:
                raise ValueError(f"quad_return parameter must be 3, 2, or 6, you gave {quad_return}.")
    if r_thresh:
        r_filt_labels = [[f"r{fl}", f"r{fl}"] for fl in suffixes]
        for rf in r_filt_labels:
            df = df.loc[(df[rf[0]] >= r_thresh) & (df[rf[1]] >= r_thresh)]

    if weight_lims:
        if isinstance(weight_lims, int):
            weight_lims = [-np.abs(weight_lims), np.abs(weight_lims)]
        if len(weight_lims) == 2:
            w_filt_labels = [[f"weightsA{fl}", f"weightsB{fl}"] for fl in suffixes]
            for wf in w_filt_labels:
                df = df.loc[(df[wf[0]] < weight_lims[1]) & (df[wf[0]] > weight_lims[0]) &
                            (df[wf[1]] < weight_lims[1]) & (df[wf[1]] > weight_lims[0])]

    idxs = df.index.tolist()

    dfs_filtered = [dd.loc[dd.index.isin(idxs)] for dd in synth_dfs.values()]

    full_suffixes = ['', '_start', '_end']
    bad_idxs = []
    if weight_lims:
        for ddf in dfs_filtered:
            if isinstance(weight_lims, int):
                weight_lims = [-np.abs(weight_lims), np.abs(weight_lims)]
            if len(weight_lims) == 2:
                w_filt_labels = [[f"weightsA{fl}", f"weightsB{fl}"] for fl in full_suffixes]
                for wf in w_filt_labels:
                    rejects = ddf.loc[(ddf[wf[0]] >= weight_lims[1]) | (ddf[wf[0]] <= weight_lims[0]) |
                                (ddf[wf[1]] >= weight_lims[1]) | (ddf[wf[1]] <= weight_lims[0])]
                    if len(rejects) > 0:
                        reject_idxs = rejects.index.tolist()
                        bad_idxs.append(reject_idxs)

        flat_bads = list(set([y for x in bad_idxs for y in x]))

        dfs_filtered = [dd.loc[~dd.index.isin(flat_bads)] for dd in dfs_filtered]


    cells = [cc.cellid.tolist() for cc in dfs_filtered]
    check = [True for aa in cells if cells[0] == aa]
    if False in check:
        raise ValueError("It's possible the orders of the dataframes are not the same and your indexing will be weird.")

    big_df = pd.concat(dfs_filtered)

    # This is just going to remove a few outlier weights, if you really want to be careful, make it extract that row from
    # every synthetic type dataframe.


    big_df['filt_by'] = use

    return big_df


def get_cut_info(df, prebins=50, postbins=50, trialbins=200, fs=100):
    '''2023_01_03. Made this so that you can take a dataframe with no metrics and calculate the masks
    for the assorted cut fits that were applied using the fit_segment parameter. I didn't want it to
    have to load files again to get ref_handle, so the bins are input options. For OLP, they should
    always be the defaults, so it shouldn't matter for this purpose.'''
    start, dur = [int(int(aa) / 1000 * fs) for aa in (list(set(df.fit_segment))[0].split('-'))]
    rs = [aa for aa in list(df.columns) if ('weightsA' in aa) and ('_pred' not in aa)]
    cut_labels = [f"_{aa.split('_')[1]}" if len(aa.split('_'))==2 else '' for aa in rs]

    antidur = trialbins - prebins - postbins - dur

    full = [False] * prebins + [True] * dur + [True] * antidur + [True] * postbins
    goods = [False] * prebins + [True] * dur + [False] * antidur + [False] * postbins
    bads = [False] * prebins + [False] * dur + [True] * antidur + [False] * postbins
    full_nopost = [False] * prebins + [True] * dur + [True] * antidur + [False] * postbins
    cut_list = [full, goods, bads, full_nopost]

    cuts_info = {cut_labels[i]: cut_list[i] for i in range(len(cut_list))}

    return cuts_info


def add_animal_name_col(df):
    '''2023_05_17. This was added as part of the initial fitting that I made.

    2023_01_12. Just takes a dataframe and decides what the animal ID was, useful for splitting based on animals.
    You'll have to manually add dividers of _A and _B for different animals if trying to divide by hemispheres, which
    are typically denoted by experiment number.'''
    animals = []
    cellids = df.cellid.tolist()
    for id in cellids:
        cell = id.split('-')[0]
        animal = cell[:3]
        exp = int(cell[3:6])
        if animal == 'CLT':
            if exp < 24:
                animal = animal + '_A'
            else:
                animal = animal + '_B'
        elif animal == 'PRN':
            if exp < 40:
                animal = animal + '_A'
            else:
                animal = animal + '_B'
        animals.append(animal)

    df['animal'] = animals
    return df


def merge_dynamic_error(weight_df, dynamic_path='cache_dyn', SNR=0):
    '''2023_07_05. Turned what I was doing into a function. This simply takes your existing big weight_df
    and loads the dynamic calculations you did using enqueue_dynamic.py and script_dynamic.py and combines
    them to a single relevant df. This you should then pass to ofig.plot_dynamic_error()'''
    # Only take cells in layers that I care about
    weight_df = weight_df.loc[(weight_df.layer=='NA') | (weight_df.layer=='5') | (weight_df.layer=='44') | (weight_df.layer=='13') |
                    (weight_df.layer=='4') | (weight_df.layer=='56') | (weight_df.layer=='16') | (weight_df.layer=='BS')]
    # Take out the dynamic runs that have the full-full
    full_FRs = weight_df.loc[(weight_df.olp_type=='dynamic') & (weight_df.dyn_kind=='ff') & (weight_df.SNR==SNR)]
    # Create dataframe to ultimately paste the full-full FRs on top of the half ones
    FR_df = full_FRs[['cellid', 'area', 'BG', 'FG', 'bg_FR', 'fg_FR']]
    FR_df = FR_df.rename(columns={"bg_FR": "full_bg_FR", "fg_FR": "full_fg_FR"})

    # Load all the jobs that did the dynamic calculation for us and make them into one DF
    saved_paths = glob.glob(f"/auto/users/hamersky/{dynamic_path}/*")
    dyn_df = []
    for path in saved_paths:
        df = jl.load(path)
        dyn_df.append(df)
    dyn_df = pd.concat(dyn_df)
    # merge the dynamic dataframe with the corresponding rows from the full df so we have all the info
    dyn_stuff = pd.merge(weight_df, dyn_df, on=['epoch', 'dyn_kind', 'parmfile', 'cellid'])

    # only keep the layers and SNR we want
    filt = dyn_stuff
    filt = filt.loc[(filt.layer=='NA') | (filt.layer=='5') | (filt.layer=='44') | (filt.layer=='13') |
                    (filt.layer=='4') | (filt.layer=='56') | (filt.layer=='16') | (filt.layer=='BS')]
    filt = filt.loc[dyn_stuff.SNR==SNR]

    # Add the column that contains the full_FR to the filtered dynamic added dataframe
    full_df = pd.merge(filt, FR_df, on=['cellid', 'area', 'BG', 'FG'])

    return full_df


def filter_across_condition(df, synth_show=['11','12','21','22'], filt_kind='synth_kind',
                            snr_threshold=0.12, r_cut=0.4, rel_cut=2.5, weight_lim=[-0.5, 2], suffix=['']):
    '''2023_07_30. Takes a dataframe filtered by synthetic olp and applies several filters across all
    synthetic conditions for that cell and stimulus pair. It will end with a dataframe that has an even
    length of all of the synthetic conditions input in synth_show. All of the snr, r, and rg tests
    must pass for all of the synthetic conditions to keep it.'''
    synth_df = df.loc[df[filt_kind].isin(synth_show)]
    synth_df['filt_name'] = synth_df['cellid'] + '-' + synth_df['BG'] + '-' + synth_df['FG']

    new_df = pd.DataFrame()
    unique_ids = synth_df.filt_name.unique().tolist()

    cc = 1
    for uid in unique_ids:
        bools_summary = []
        for sf in suffix:
            id_df = synth_df.loc[synth_df.filt_name==uid]
            rr = id_df[f'r{sf}']>=r_cut
            snr = (id_df[f'bg_snr{sf}']>=snr_threshold) & (id_df[f'fg_snr{sf}']>=snr_threshold)
            rg = (id_df[f'FG_rel_gain{sf}'] <= rel_cut) & (id_df[f'FG_rel_gain{sf}'] >= -rel_cut)
            wb = (id_df[f'weightsA{sf}'] >= weight_lim[0]) & (id_df[f'weightsA{sf}'] <= weight_lim[1])
            wf = (id_df[f'weightsB{sf}'] >= weight_lim[0]) & (id_df[f'weightsB{sf}'] <= weight_lim[1])

            bools = rr.values.tolist() + snr.values.tolist() + rg.values.tolist() + wb.values.tolist() + wf.values.tolist()
            summary = all(i for i in bools)
            bools_summary.append(summary)

        full_summary = all(i for i in bools_summary)

        if full_summary == True:
            print(f"Adding {cc}")
            cc += 1
            new_df = pd.concat([new_df, id_df])

    ret_df = new_df.drop(['filt_name'], axis=1)
    return ret_df


def label_vocalization(filt, species):
    '''2023_07_27. These manual bricks of labels were taking up room on my scratch file and annoying me so I made this.
    Pass it a dataframe and specify the species and it'll just simply add it.'''
    if species == 'ferret':
        voc_labels = {'Bell': 'No', 'Branch': 'No', 'Bugle': 'No', 'CashRegister': 'No', 'Castinets': 'No',
                      'Chickens': 'Other', 'Dice': 'No', 'Dolphin': 'Other', 'Fight': 'Yes', 'FightSqueak': 'Yes',
                      'Fight_Squeak': 'Yes', 'FireCracker': 'No', 'Geese': 'Other', 'Gobble': 'Yes',
                      'Gobble_High': 'Yes', 'Heels': 'No', 'Keys': 'No', 'KitGroan': 'Yes', 'KitHigh': 'Yes',
                      'KitWhine': 'Yes', 'Kit_Groan': 'Yes', 'Kit_High': 'Yes', 'Kit_Low': 'Yes',
                      'Kit_Whine': 'Yes', 'ManA': 'Other', 'ManB': 'Other', 'Tsik': 'Other', 'TwitterB': 'Other',
                      'Typing': 'No', 'WomanA': 'Other', 'WomanB': 'Other', 'Woodblock': 'No', 'Xylophone': 'No'}
        # voc_labels = {'Bell': 'No', 'Branch': 'No', 'Bugle': 'No', 'CashRegister': 'No', 'Castinets': 'No',
        #               'Chickens': 'No', 'Dice': 'No', 'Dolphin': 'No', 'Fight': 'Yes', 'FightSqueak': 'Yes',
        #               'Fight_Squeak': 'Yes', 'FireCracker': 'No', 'Geese': 'No', 'Gobble': 'Yes',
        #               'Gobble_High': 'Yes', 'Heels': 'No', 'Keys': 'No', 'KitGroan': 'Yes', 'KitHigh': 'Yes',
        #               'KitWhine': 'Yes', 'Kit_Groan': 'Yes', 'Kit_High': 'Yes', 'Kit_Low': 'Yes',
        #               'Kit_Whine': 'Yes', 'ManA': 'Human', 'ManB': 'Human', 'Tsik': 'No', 'TwitterB': 'No',
        #               'Typing': 'No', 'WomanA': 'Human', 'WomanB': 'Human', 'Woodblock': 'No', 'Xylophone': 'No'}
        # If you're doing this you don't want to split by hemisphere, so get rid of that modifier
        filt['animal'] = filt.animal.str[:3]
        filt['Vocalization'] = filt['FG'].map(voc_labels)
        filt['animal_voc'] = filt['animal'] + '_' + filt['Vocalization'].replace({'Yes':'voc', 'No': 'non'})

    elif species == 'marmoset':
        voc_labels = {'Alarm': 'Yes', 'Bell': 'No', 'Blacksmith': 'No', 'Branch': 'No', 'CashRegister': 'No',
                      'Castinets': 'No', 'Chickens': 'No', 'Chirp': 'Yes', 'Dice': 'No', 'Geese': 'No',
                      'Heels': 'No', 'Keys': 'No', 'Loud_Shrill': 'Yes', 'ManA': 'No', 'ManB': 'No',
                      'Phee': 'Yes', 'Seep': 'Yes', 'Trill': 'Yes', 'Tsik': 'Yes', 'TsikEk': 'Yes',
                      'Tsik_Ek': 'Yes', 'TwitterA': 'Yes', 'TwitterB': 'Yes', 'Typing': 'No', 'WomanA': 'No',
                      'WomanB': 'No', 'Woodblock': 'No', 'Xylophone': 'No'}
        filt['Vocalization'] = filt['FG'].map(voc_labels)
        noise_labels = {'Alarm': 'Yes', 'Chirp': 'No', 'Loud_Shrill': 'Yes', 'Phee': 'No', 'Seep': 'Yes', 'Trill': 'No',
                        'Tsik': 'Yes', 'TsikEk': 'No', 'Tsik_Ek': 'No', 'TwitterA': 'No', 'TwitterB': 'Yes'}
        filt['noisy'] = filt['FG'].map(noise_labels)
        filt.noisy.fillna('non', inplace=True)
        filt['fg_noise'] = filt['animal'] + '_' + filt['Vocalization'].replace({'Yes': 'voc', 'No': 'non'}) + '_' + \
                           filt['noisy'].replace({'Yes': 'noise', 'No': 'quiet'})
        filt['animal_voc'] = filt['animal'] + '_' + filt['Vocalization'].replace({'Yes':'voc', 'No': 'non'})

    elif species == 'sounds':
        voc_labels = {'Bell': 'No', 'Branch': 'No', 'Bugle': 'No', 'CashRegister': 'No', 'Castinets': 'No',
                      'Chickens': 'Other', 'Dice': 'No', 'Dolphin': 'Other', 'Fight': 'Yes', 'FightSqueak': 'Yes',
                      'Fight_Squeak': 'Yes', 'FireCracker': 'No', 'Geese': 'Other', 'Gobble': 'Yes',
                      'Gobble_High': 'Yes', 'GobbleHigh': 'Yes', 'Heels': 'No', 'Keys': 'No', 'KitGroan': 'Yes', 'KitHigh': 'Yes',
                      'KitWhine': 'Yes', 'Kit_Groan': 'Yes', 'Kit_High': 'Yes', 'Kit_Low': 'Yes', 'KitLow': 'Yes',
                      'Kit_Whine': 'Yes', 'ManA': 'Other', 'ManB': 'Other', 'Tsik': 'Other', 'TwitterB': 'Other',
                      'Typing': 'No', 'WomanA': 'Other', 'WomanB': 'Other', 'Woodblock': 'No', 'Xylophone': 'No'}
        bgs, fgs = filt.loc[filt.type=='BG'], filt.loc[filt.type=='FG']
        fgs['Vocalization'] = fgs['short_name'].map(voc_labels)
        bgs['Vocalization'] = 'Background'
        filt = pd.concat([bgs, fgs])

    return filt


def df_filters(filt, snr_threshold=0.12, rel_cut=2.5, r_cut=0.4, weight_lim=[-0.5, 2]):
    '''2023_07_31. Does the basic filtering on a dataframe to save me from having to do it over and over.'''
    if snr_threshold:
        filt = filt.loc[(filt.bg_snr >= snr_threshold) & (filt.fg_snr >= snr_threshold)]
    if rel_cut:
        filt = filt.loc[(filt[f'FG_rel_gain'] <= rel_cut) & (filt[f'FG_rel_gain'] >= -rel_cut)]
    if weight_lim:
        filt = filt.loc[((filt[f'weightsA'] >= weight_lim[0]) & (filt[f'weightsA'] <= weight_lim[1])) &
                        ((filt[f'weightsB'] >= weight_lim[0]) & (filt[f'weightsB'] <= weight_lim[1]))]
    if r_cut:
        filt = filt.dropna(axis=0, subset='r')
        filt = filt.loc[filt.r >= r_cut]

    return filt


def add_spike_widths(weight_df, save_name='ferrets_with_spikes', cutoff=None):
    '''2023_08_01. Give it your big dataframe and let it figure out which ones have spike width in them and then
    give you those back, merging it with your DF. It also saves it, so you don't have to watch it load each time.'''
    weight_df['site'] = [dd[:7] for dd in weight_df['cellid']]
    sites = weight_df.site.unique().tolist()

    width_df = pd.DataFrame()
    fails, goods = [], []
    for ss in sites:
        try:
            site_info = baphy_io.get_spike_info(siteid=ss)
            site_info['cellid'] = site_info.index
            widths = site_info[['cellid', 'sw']]
            width_df = pd.concat([width_df, widths])
            print(f'Adding {ss}')
            goods.append(ss)
        except:
            print(f'{ss} did NOT work')
            fails.append(ss)
    width_df.rename(columns={'sw': 'spike_width'}, inplace=True)

    weight_dff = pd.merge(right=width_df, left=weight_df, on=['cellid'])

    if cutoff:
        if isinstance(cutoff, dict):
            weight_dff['width'], weight_dff['cutoff'] = np.NaN, np.NaN
            filter_critters = {key: val for key, val in cutoff.items() if len(key)==3}
            filter_names = [dd for dd in filter_critters.keys()]

            weird_filt = pd.DataFrame()
            for nn, cc in filter_critters.items():
                weight_dff.loc[(weight_dff.animal==nn), 'cutoff'] = cc
                weight_dff.loc[((weight_dff.animal==nn) & (weight_dff['spike_width'] >= cc)), 'width'] = 'broad'
                weight_dff.loc[((weight_dff.animal==nn) & (weight_dff['spike_width'] < cc)), 'width'] = 'narrow'

            other_critters = {key: val for key, val in cutoff.items() if len(key) > 3}
            if len(other_critters)>=1:
                other_cut =list(other_critters.values())[0]
                weight_dff.loc[(~weight_dff.animal.isin(filter_names)), 'cutoff'] = other_cut
                weight_dff.loc[((~weight_dff.animal.isin(filter_names)) &
                                (weight_dff['spike_width'] >= other_cut)), 'width'] = 'broad'
                weight_dff.loc[((~weight_dff.animal.isin(filter_names)) &
                                (weight_dff['spike_width'] < other_cut)), 'width'] = 'narrow'

            weight_dff.loc[weight_dff['spike_width'] >= cc, 'width'] = 'broad'
            weight_dff.loc[weight_dff['spike_width'] < cc, 'width'] = 'narrow'

        elif isinstance(cutoff, float):
            weight_dff['width'], weight_dff['sw_cutoff'] = np.NaN, cutoff
            weight_dff.loc[weight_dff['spike_width'] >= cutoff, 'width'] = 'broad'
            weight_dff.loc[weight_dff['spike_width'] < cutoff, 'width'] = 'narrow'

    save_path = f'/auto/users/hamersky/olp_analysis/{save_name}'
    jl.dump(weight_dff, save_path)

    return weight_dff


def get_spectral_overlap_stats_and_paths(filt, area=None):
    '''2023_08_03. Stupid little function that takes a dataframe and gets the sound metrics for it, returning to
    you a dataframe that has all of the BG/FG combos and the spectral overlap and bandwidth information for each
    pair. It also has the frequency range and paths for those files so you can put it into a function that plots
    a pair side by side.'''
    sound_df = ohel.get_sound_statistics_from_df(filt, percent_lims=[15, 85], area=area, append=True)

    new_df_bg = sound_df[['BG', 'FG', 'BG_spectral_overlap', 'BG_rel_gain', 'BG_bandwidth', 'BG_path']]
    new_df_fg = sound_df[['BG', 'FG', 'FG_spectral_overlap', 'FG_rel_gain', 'FG_bandwidth', 'FG_path']]

    bg_path_df = sound_df[['BG', 'BG_path', 'BG_freq_range']].drop_duplicates()
    fg_path_df = sound_df[['FG', 'FG_path', 'FG_freq_range']].drop_duplicates()

    bg_mean = new_df_bg.groupby(by=['BG', 'FG']).mean().reset_index()
    fg_mean = new_df_fg.groupby(by=['BG', 'FG']).mean().reset_index()

    combined_for_path = pd.concat([bg_mean, fg_mean], axis=1)
    combined_for_path = combined_for_path.loc[:,~combined_for_path.columns.duplicated()]

    combined_for_path['BG_path'], combined_for_path['FG_path'] = np.NaN, np.NaN
    BG_path_dict = {key:val for key, val in zip(bg_path_df['BG'], bg_path_df['BG_path'])}
    FG_path_dict = {key:val for key, val in zip(fg_path_df['FG'], fg_path_df['FG_path'])}
    BG_freq_dict = {key:val for key, val in zip(bg_path_df['BG'], bg_path_df['BG_freq_range'])}
    FG_freq_dict = {key:val for key, val in zip(fg_path_df['FG'], fg_path_df['FG_freq_range'])}
    combined_for_path['BG_freq_range'] = combined_for_path['BG'].map(BG_freq_dict)
    combined_for_path['FG_freq_range'] = combined_for_path['FG'].map(FG_freq_dict)
    combined_for_path['BG_path'] = combined_for_path['BG'].map(BG_path_dict)
    combined_for_path['FG_path'] = combined_for_path['FG'].map(FG_path_dict)

    return combined_for_path


def plot_spectral_overlap_specs(dff, BG, FG):
    '''2023_08_03. Uses the output of ohel.get_spectral_overlap_stats_and_pairs() and you give it a BG and FG
    identity that you want it to plot the spectrograms of and indicate the spectral overlap side by side.'''
    row = dff.loc[(dff.BG==BG) & (dff.FG==FG)]

    fig, ax = plt.subplots(1, 1, figsize=(15,5))

    bgsfs, bgW = wavfile.read(row['BG_path'].values[0])
    bgspec = gtgram(bgW, bgsfs, 0.02, 0.01, 48, 100, 24000)
    fgsfs, fgW = wavfile.read(row['FG_path'].values[0])
    fgspec = gtgram(fgW, fgsfs, 0.02, 0.01, 48, 100, 24000)
    specs = np.concatenate([bgspec, fgspec], axis=1)

    ax.imshow(specs, aspect='auto', origin='lower', extent=[0, specs.shape[1], 0, specs.shape[0]],
                 cmap='gray_r')
    ax.vlines([bgspec.shape[-1]], 0, 48, linestyle='-', color='black')
    x_freq = np.logspace(np.log2(100), np.log2(24000), num=48, base=2)

    ax.set_xticks([50,150])
    ax.set_xticklabels([f'BG: {BG}', f'FG: {FG}'], fontweight='bold', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontweight='bold', fontsize=10)

    int_bins = [int(xx) for xx in x_freq]
    BG_freqs = row['BG_freq_range'].values[0]
    BG_bins = [int_bins.index(dd) for dd in list(BG_freqs)]
    FG_freqs = row['FG_freq_range'].values[0]
    FG_bins = [int_bins.index(dd) for dd in list(FG_freqs)]
    yticks = list(np.concatenate([[0], BG_bins, FG_bins, [47]]))
    ylabels = [int(x_freq[dd]) for dd in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.hlines(BG_bins, 0, bgspec.shape[-1], linestyle=':', color='black')
    ax.hlines(FG_bins, bgspec.shape[-1], bgspec.shape[-1] + fgspec.shape[-1], linestyle=':', color='black')

    ax.spines['top'].set_visible(True), ax.spines['right'].set_visible(True)
    ax.set_title(f"BG Spec Overlap: {int(row['BG_spectral_overlap'].values[0])}%, "
                 f"Bandwidth: {np.around(row['BG_bandwidth'].values[0], 1)} --- "
                 f"FG Spec Overlap: {int(row['FG_spectral_overlap'].values[0])}%, "
                 f"Bandwidth: {np.around(row['FG_bandwidth'].values[0], 1)}",
                 fontsize=12, fontweight='bold')

    return row


def plot_spike_width_distributions(filt, split_critter=None, line=0.375):
    '''2023_08_03. Dumb function that takes a filtered dataframe and plots distributions of spike width. If you
    specify a critter to separate out (only 1 on this one, I should've made it more) it'll separate that animal
    from the rest, just make sure you manually give it the cutoff lines you want to make.'''
    filtt = filt.copy()
    filtt = filtt.drop_duplicates('cellid')

    little, big = filtt.spike_width.min(), filtt.spike_width.max()
    edges = np.arange(0, np.around(big, 1), .005)

    if split_critter:
        lonely_ferret = filtt.loc[filtt.animal==split_critter]
        other_ferrets = filtt.loc[filtt.animal!=split_critter]
        if isinstance(line, float):
            lines = [line, line, line]
        elif isinstance(line, list):
            if len(line)==2:
                line = line + [line[-1]]
        fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=False, sharex=True)
        to_plots, names = [lonely_ferret, other_ferrets, filtt], [f'{split_critter}', 'Others', 'Everyone']
        for ax, plot_df, nm, ln in zip(axes, to_plots, names, line):
            na, xa = np.histogram(plot_df['spike_width'], bins=edges)
            na = na / na.sum() * 100
            ax.hist(xa[:-1], xa, weights=na, histtype='step', color='dimgrey', linewidth=2)
            ax.set_xlabel('Spike Width', fontsize=10, fontweight='bold')
            ymin, ymax = ax.get_ylim()
            ax.vlines([ln], 0, ymax, colors='black', linestyles=':', lw=1)
            ax.set_title(f'{nm}, n={len(plot_df)}, cut={ln}', fontsize=10, fontweight='bold')
            axes[0].set_ylabel('Percentage of Cells', fontsize=10, fontweight='bold')

    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        na, xa = np.histogram(filtt['spike_width'], bins=edges)
        na = na / na.sum() * 100
        ax.hist(xa[:-1], xa, weights=na, histtype='step', color='dimgrey', linewidth=2)
        ax.set_xlabel('Spike Width', fontsize=10, fontweight='bold')
        ymin, ymax = ax.get_ylim()
        ax.vlines([line], 0, ymax, colors='black', linestyles=':', lw=1)
        ax.set_title(f'n={len(filtt)}, cut={line}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Percentage of Cells', fontsize=10, fontweight='bold')


def run_sound_stats_reg(df, r_cut=0.4, snr_threshold=0.12, suffix='', area='A1', synth=None,
            xs=['Fcorr', 'Tstationary', 'bandwidth'],
            category='Vocalization', no_dict=False, shuffle=True):
    '''2023_08_03. Been sitting here a while. Takes a filtered dataframe and regresses a bunch of columns in the
    dataframe you put in.'''
    df = df.loc[df.area==area].copy()

    if category:
        if category=='Vocalization':
            voc_labels_fg = {'Yes': 1, 'No': 0, 'Other': 2}
            # df['Vocalization'] = df['Vocalization'].map(voc_labels)
            df['BG_Vocalization'] = 3
            df['FG_Vocalization'] = df['Vocalization'].map(voc_labels_fg)
            vocal_key = {0: 'Non-Vocalization', 1: 'Ferret Vocalization', 2: 'Other Vocalization', 3: 'Background'}

    xs = [f'{xx}{suffix}' for xx in xs]
    y = [f'rel_gain{suffix}']

    if r_cut:
        df = df.dropna(axis=0, subset='r')
        df = df.loc[df[f'r{suffix}'] >= r_cut]
    df = df.copy()

    # fr_thresh = 0.03
    if snr_threshold:
        if suffix == '_start' or suffix == '_end':
            df = df.loc[(df.bg_FR_start >= snr_threshold) & (df.fg_FR_start >= snr_threshold)
                        & (df.bg_FR_end >= snr_threshold) & (df.fg_FR_end >= snr_threshold)]
        else:
            df = df.loc[(df.bg_snr >= snr_threshold) & (df.fg_snr >= snr_threshold)]

    df.rename(columns={'bg_snr': 'BG_snr', 'fg_snr': 'FG_snr'}, inplace=True)

    sound_df = get_sound_statistics_from_df(df, percent_lims=[15, 85], append=True,)
    # sound_df.loc[sound_df.BG_spectral_overlap <= 0, 'BG_spectral_overlap'] = 0.0

    if category == 'Vocalization':
        nms = xs + y + [category]
    else:
        nms = xs + y
    bx, fx = [f'BG_{bb}' for bb in nms], [f'FG_{ff}' for ff in nms]
    if category:
        if isinstance(category, list):
            bx, fx = ['BG', 'synth_kind', 'cellid', 'layer'] + bx + category, ['FG', 'synth_kind', 'cellid', 'layer'] + fx + category
        elif isinstance(category, str):
            bx, fx = ['BG', 'synth_kind', 'cellid', 'layer'] + bx, ['FG', 'synth_kind', 'cellid',
                                                                                 'layer'] + fx
            # bx, fx = ['BG', 'synth_kind', 'cellid', 'layer'] + bx + [category], ['FG', 'synth_kind', 'cellid', 'layer'] + fx + [category]
    else:
        bx, fx = ['BG', 'synth_kind', 'cellid', 'layer'] + bx, ['FG', 'synth_kind', 'cellid', 'layer'] + fx
    # bx, fx = ['BG', 'synth_kind', 'cellid', 'layer'] + bx, ['FG', 'synth_kind', 'cellid', 'layer'] + fx

    bgs, fgs = sound_df[bx], sound_df[fx]
    bg_rn = {key:(key[3:] if len(key.split('_'))>1 else 'name') for key in bgs.columns.to_list() if key[:2]=='BG'}
    fg_rn = {key:(key[3:] if len(key.split('_'))>1 else 'name') for key in fgs.columns.to_list() if key[:2]=='FG'}
    bgs, fgs = bgs.rename(columns=bg_rn), fgs.rename(columns=fg_rn)

    if shuffle==True:
        if isinstance(category, list):
            shuffle_list = ['full'] + xs + category
        elif isinstance(category, str):
            shuffle_list = ['full'] + xs + [category]
    else:
        shuffle_list = ['full']

    ests = {}
    for shuff in shuffle_list:
        to_reg = pd.concat([bgs, fgs])
        if shuff != 'full':
            # if shuff=='Vocalization':
            #     fgs_voc = to_reg.loc[to_reg.Vocalization!=3]
            #     bgs_voc = to_reg.loc[to_reg.Vocalization==3]
            #     fgs_voc[shuff] = np.random.permutation(fgs_voc[shuff].values)
            #     to_reg = pd.concat([fgs_voc, bgs_voc])
            # else:
            to_reg[shuff] = np.random.permutation(to_reg[shuff].values)

        if synth:
            to_reg = to_reg.loc[to_reg.synth_kind==synth]

        for xx in xs:
            to_reg[xx] -= to_reg[xx].mean()
            to_reg[xx] /= to_reg[xx].std()

        string = ' + '.join(xs)
        if category:
            if isinstance(category, list):
                cats = [f'C({cc})' for cc in category]
                cat_string = ' + '.join(cats)
            elif isinstance(category, str):
                cat_string = f'C({category})'
            fit_string = ' + '.join([string, cat_string])
        else:
            fit_string = string

        mod = smf.ols(formula=f'{y[0]} ~ {fit_string}', data=to_reg)
        est = mod.fit()

        ests[shuff] = est

    if category=='Vocalization':
        vars = list(ests['full'].params.index)
        cat_vars = vars[:len(vocal_key)]
        vocal_dict = {cat_vars[key]:val for key, val in vocal_key.items()}
    else:
        vocal_dict = {}

    return ests, vocal_dict


def get_olp_filter(weight_df, kind='vanilla', metric=False):
    '''2023_08_08. The day I got tired of running tons of dumb little piecemeal code every time I started
    a new console. This will take the dataframe you load and apply the appropriate filters based on what
    you are trying to ultimately do. Inputs for kind include vanilla, binaural, synthetic, '''
    # Adds a vocalization column that kind of manually labels the FGs as the type they are
    # Also, removes the tags I left on the animal column specifying hemisphere
    filt = label_vocalization(weight_df, species='ferret')
    # Keep both primary and secondary areas for now,
    filt = filt.loc[(filt.area == 'A1') | (filt.area == 'PEG')]
    # Rename some layers that are named funny because of the labelling GUI, not a meaningful distinction
    filt.loc[filt.layer == '4', 'layer'] = '44'
    filt.loc[filt.layer == '5', 'layer'] = '56'
    filt.loc[filt.layer == 'BS', 'layer'] = '13'
    # Save only certain layers that are cortical
    filt = filt.loc[(filt.layer == 'NA') | (filt.layer == '5') | (filt.layer == '44') | (filt.layer == '13') |
                    (filt.layer == '4') | (filt.layer == '56') | (filt.layer == '16') | (filt.layer == 'BS')]

    # Keep the vanilla OLP parameters, this being dynamic full-full, binaural contra-contra, and no SNR
    if kind == 'vanilla':
        filt = filt.loc[filt.dyn_kind == 'ff']
        filt = filt.loc[filt.kind == '11']
        filt = filt.loc[filt.SNR == 0]

        # This deals with synthetics and how to deal with the natural ('N') and non-RMS natural ('A'),
        # Which constitute multiple presentations of the same stimuli to a cell, which we want to average them
        # It also deals with that some PRN days had multiple OLP sessions played to the same cell
        # Each OLP has its own vanilla control within with a different epoch name, so those all need to be
        # averaged across so one cell doesn't get represented 4 times in the average. This does not yet deal
        # with instances where one electrode site implant received the same stimulus across days, those are still
        # being treated as unique instances.
        prn = filt.loc[filt.animal == 'PRN']
        prn = prn.loc[(prn.synth_kind == 'N') | (prn.synth_kind == 'A')]
        prn_filt = prn.groupby(by=['cellid', 'BG', 'FG', 'Vocalization', 'animal', 'area', 'BG_path', 'FG_path'],
                               as_index=False).mean()
        # Moves along with everyone who isn't Prince (which means each day was a unique penetration)
        # For now going to average the synthetics (only applies to Clathrus) and keep everything else.
        filt = filt.loc[((filt.synth_kind == 'N') & (filt['animal'] == 'CLT') & (filt['olp_type'] == 'synthetic')) |
                        ((filt.synth_kind == 'A') & (filt['animal'] == 'CLT') & (filt['olp_type'] == 'synthetic')) |
                        ((filt.synth_kind == 'A') & (filt['animal'] == 'CLT') & (filt['olp_type'] != 'synthetic')) |
                        ((filt.synth_kind == 'A') & (filt['animal'].isin(['TNC', 'ARM'])))]
        others = filt.loc[filt.animal != 'PRN']
        others_filt = others.groupby(by=['cellid', 'BG', 'FG', 'Vocalization', 'animal', 'area', 'BG_path', 'FG_path'],
                                     as_index=False).mean()
        # Put these two modified dataframes back together. They lost some info in the average, but most categorical
        # columns that are needed should be preserved manually.
        filt = pd.concat([others_filt, prn_filt])
        # Synth_kind couldn't be kept because some are 'A', but with it gone and the averages of N and A performed
        # just call everything N because it's broadly the natural stimuli
        filt['synth_kind'] = 'N'
        # Some functions need layer or kind in there to filter them out, so just readd that.
        filt['layer'], filt['kind'] = 'NA', '11'
    elif kind == 'binaural':
        filt = filt.loc[filt.dyn_kind == 'ff']
        filt = filt.loc[filt.SNR == 0]
        filt = filt.loc[filt.olp_type == 'binaural']
    elif kind == 'SNR':
        filt = filt.loc[filt.dyn_kind == 'ff']
        filt = filt.loc[filt.kind == '11']
        # SNR run is constructed around olp_type=='dynamic', but there will be a whole bunch of other
        # dynamic instances at 0 SNR, so you want the ones that are matched to an off-SNR one, hence
        # a new column that identifies SNR 10 instances, gets their cellid and sounds played and then
        # gets all other times that cell heard those sounds from the dataframe.
        # The utility of getting rid of binaural and synthetic ahead of time is days in Prince when
        # SNR, binaural, and synthetic were all played.
        filt = filt.loc[(filt.olp_type != 'binaural') & (filt.olp_type != 'synthetic')]
        filt['filt_name'] = filt['filt_name'] = filt['cellid'] + '-' + filt['BG'] + '-' + filt['FG']
        snr10 = filt.loc[filt.SNR == 10]
        epoch_names = snr10.filt_name.tolist()
        filt = filt.loc[filt.filt_name.isin(epoch_names)]
        filt = filt.drop(labels=['filt_name'], axis=1)
    elif kind == 'synthetic':
        filt = filt.loc[filt.dyn_kind == 'ff']
        filt = filt.loc[filt.kind == '11']
        filt = filt.loc[filt.olp_type == 'synthetic']
    elif kind == 'sounds':
        filt = filt.loc[filt.dyn_kind == 'ff']
        filt = filt.loc[filt.kind == '11']
        filt = filt.loc[filt.SNR == 0]
        filt = filt.loc[((filt.synth_kind=='A') & (filt.olp_type!='synthetic')) |
                        ((filt.synth_kind=='N') & (filt.olp_type=='synthetic'))]

    # sound_df = ohel.get_sound_statistics_from_df(filt, percent_lims=[15,85], append=False)
    # bad_dict = ohel.plot_sound_stats(sound_df, ['max_power', 'RMS_power'], labels=['Max Power', 'RMS Power'],
    #                                  lines={'RMS_power': 0.95, 'max_power': 0.3}, synth_kind='N')
    # bads = ['CashRegister', 'Heels', 'Castinets', 'Dice']  # RMS Power Woodblock for 'N'
    # Get rid of the sounds that, through the lines above, would have been decided to not meet our criteria
    bads = ['Branch', 'CashRegister', 'Heels', 'Woodblock', 'Castinets', 'Dice', 'Tsik', 'TwitterB']  # RMS power + noisy marm
    filt = filt.loc[filt['BG'].apply(lambda x: x not in bads)]
    filt = filt.loc[filt['FG'].apply(lambda x: x not in bads)]

    if metric == True:
        filt = df_filters(filt, snr_threshold=0.12, rel_cut=2.5, r_cut=0.4, weight_lim=[-0.5, 2])

    return filt