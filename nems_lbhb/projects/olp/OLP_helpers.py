import numpy as np
import scipy.stats as sst
import pandas as pd
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nems.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import glob
import seaborn as sb

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
    elif len(ep_name.split('_')) == 3:
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
    else:
        stim_type = None

    return stim_type


def label_synth_type(ep_name):
        '''Labels epochs that have one or two stimuli in it based on what kind of synthetic sound it is.
        N = Normal RMS, C = Cochlear, T = Temporal, S = Spectral, U = Spectrotemporal, M = spectrotemporal
        modulation, A = Non-RMS normalized unsynethic'''
        if len(ep_name.split('_')) == 1 or ep_name[:5] != 'STIM_':
            synth_type = None
        elif len(ep_name.split('_')) == 3:
            seps = (ep_name.split('_')[1], ep_name.split('_')[2])
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
    import nems.metrics.corrcoef
    Xac = nems.metrics.corrcoef._r_single(X, N_ac,0)
    Yac = nems.metrics.corrcoef._r_single(Y, N_ac,0)
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
    '''Filters a dataframe by a FR threshold with spont subtracted. quad_returns says which
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


def get_sound_statistics_full(weight_df):
    '''Updated 2022_09_13. Added mean relative gain for each sound. The rel_gain is BG or FG
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

            freq_mean = np.nanmean(spec, axis=1)
            x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
            csm = np.cumsum(freq_mean)
            big = np.max(csm)

            freq75 = x_freq[np.abs(csm - (big * 0.75)).argmin()]
            freq25 = x_freq[np.abs(csm - (big * 0.25)).argmin()]
            freq50 = x_freq[np.abs(csm - (big * 0.5)).argmin()]
            bandw = np.log2(freq75 / freq25)

            means[:, cnt] = freq_mean

            sounds.append({'name': sn.split('_')[0],
                           'type': ll,
                           'synth_kind': syn,
                           'std': dev,
                           'bandwidth': bandw,
                           '75th': freq75,
                           '25th': freq25,
                           'center': freq50,
                           'spec': spec,
                           'mean_freq': freq_mean,
                           'freq_stationary': np.std(freq_mean),
                           'RMS_norm_power': rms_normed,
                           'max_norm_power': max_normed,
                           'short_name': sn[2:].split('_')[0].replace(' ', '')})

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
    '''5/12/22 Takes a cellid and batch and figures out all the sounds that were played
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


def plot_sound_stats(sound_df, stats, labels=None, synth_kind='N', lines=None):
    '''2022_09_14. This is a way to look at the sound stats (passed as a list) from a sound_df and compare them. But also, if you add
    lines, a dictionary, which passes keys as those matching something found in stats, with a cutoff. That cut off will
    be drawn as a line on that subplot for that stat and it will also tell you what sounds are below that threshold,
    returning those sounds in a dictionary where the stat is a key and the values are a list of 'bad' sounds. Labels
    are optional, passing it will look prettier than it defaulting the labels to what the sound stat in the df.'''
    sound_df = sound_df.loc[sound_df.synth_kind == synth_kind]
    sound_df.rename(columns={'std': 'Tstationary', 'freq_stationary': 'Fstationary', 'RMS_norm_power': 'RMS_power',
                             'max_norm_power': 'max_power'}, inplace=True)
    if isinstance(stats, list):
        lens = len(stats)
    elif isinstance(stats, str):
        lens, stats = 1, [stats]

    if lens <= 3:
        hh, ww = 1, lens
    else:
        hh, ww = int(np.ceil(lens / 3)), 3
    sound_df['Tstationary'] = [np.mean(aa) for aa in sound_df['Tstationary']]

    fig, axes = plt.subplots(hh, ww, figsize=(ww * 5, hh * 5))
    axes = np.ravel(axes)

    bads = {}
    for cnt, (ax, st) in enumerate(zip(axes, stats)):
        sb.barplot(x='short_name', y=st,
                   palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
                   data=sound_df, ci=68, ax=ax)
        ax.set_xticklabels(sound_df.short_name, rotation=90, fontweight='bold', fontsize=7)
        if labels:
            ax.set_ylabel(labels[cnt], fontweight='bold', fontsize=12)
        else:
            ax.set_ylabel(stats[cnt], fontweight='bold', fontsize=12)
        ax.spines['top'].set_visible(True), ax.spines['right'].set_visible(True)
        ax.set(xlabel=None)

        if st in lines.keys():
            xmin, xmax = ax.get_xlim()
            ax.hlines(lines[st], xmin=xmin, xmax=xmax, ls=':', color='black')
            ax.set_xlim(xmin, xmax)
            bad_df = sound_df.loc[sound_df[st] <= lines[st]]
            bads[st] = bad_df.short_name.tolist()
    axes[0].set_title(f"Synth: {synth_kind}", fontsize=10, fontweight='bold')
    return bads


def add_sound_stats(weight_df, sound_df):
    '''Updated 2022_09_13. Previously it just added the T, band, and F stats to the dataframe.
    I updated it so that it takes synth kind into account when adding the statistics, and
    also adds RMS and max power for the sounds.'''
    BGdf, FGdf = sound_df.loc[sound_df.type == 'BG'], sound_df.loc[sound_df.type == 'FG']
    BGmerge, FGmerge = pd.DataFrame(), pd.DataFrame()
    BGmerge['BG'] = [aa[2:].replace(' ', '') for aa in BGdf.name]
    BGmerge['BG_Tstationary'] = [np.nanmean(aa) for aa in BGdf['std']]
    BGmerge['BG_bandwidth'] = BGdf.bandwidth.tolist()
    BGmerge['BG_Fstationary'] = BGdf.freq_stationary.tolist()
    BGmerge['BG_RMS_power'] = BGdf.RMS_norm_power.tolist()
    BGmerge['BG_max_power'] = BGdf.max_norm_power.tolist()
    BGmerge['synth_kind'] = BGdf.synth_kind.tolist()

    FGmerge['FG'] = [aa[2:].replace(' ', '') for aa in FGdf.name]
    FGmerge['FG_Tstationary'] = [np.nanmean(aa) for aa in FGdf['std']]
    FGmerge['FG_bandwidth'] = FGdf.bandwidth.tolist()
    FGmerge['FG_Fstationary'] = FGdf.freq_stationary.tolist()
    FGmerge['FG_RMS_power'] = FGdf.RMS_norm_power.tolist()
    FGmerge['FG_max_power'] = FGdf.max_norm_power.tolist()
    FGmerge['synth_kind'] = FGdf.synth_kind.tolist()

    weight_df = pd.merge(right=BGmerge, left=weight_df, on=['BG', 'synth_kind'], validate='m:1')
    weight_df = pd.merge(right=FGmerge, left=weight_df, on=['FG', 'synth_kind'], validate='m:1')

    return weight_df



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
