from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import nems_lbhb.TwoStim_helpers as ts
from scipy import stats
import scipy.ndimage.filters as sf
import glob
from nems.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
from pathlib import Path
from nems.analysis.gammatone.gtgram import gtgram
import nems_lbhb.projects.olp.OLP_helpers as ohel


df_filtered, weight_df, titles, bins = [], [], [], []

def scatter_weights(weight_df=weight_df, title=titles, bins=bins):
    #Colorful scatter plot of BG weights v FG weights - moved to opl
    fig, ax = plt.subplots()
    g = sns.scatterplot(x='weightsA', y='weightsB', data=weight_df, hue='Animal')
    ax.set_title(f"{title}")

    plt.xlabel('Background Weights')
    plt.ylabel('Foreground Weights')
    plt.xlim((-2, 2))
    plt.ylim((-2, 2))
    plt.title(f"{title}")
    plt.gca().set_aspect(1)

def heatmap_weights(weight_df=weight_df, title=titles, bins=np.arange(-2,2,.05)):
    #Same plot as the previous one but using the weight dataframe
    gi=~np.isnan(weight_df['weightsA']) & ~np.isnan(weight_df['weightsB'])
    plt.figure(figsize=(5,5));  plt.hist2d(weight_df['weightsA'][gi],weight_df['weightsB'][gi],bins=bins)
    plt.xlim((-.5, 1.5)); plt.ylim((-.5, 1.5))
    plt.gca().set_aspect(1)
    plt.xlabel('Background Weight')
    plt.ylabel('Foreground Weight')
    plt.title(f"{title}")


def filter_epochs_by_file_names(pathidx, weight_df=weight_df):
    # Takes an array of two numbers [BG, FG] corresponding with which respective file in OLP
    # you want to grab epochs from. Accounts for differences in my naming schemes ('_' v ' ')
    import glob

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Background{pathidx[0]}/*.wav'))
    bg_names_spaces = [bb.split('/')[-1].split('.')[0][2:] for bb in bg_dir]
    bg_names_nospace = [bb.replace('_', '') for bb in bg_names_spaces if '_' in bb]
    bg_names = bg_names_spaces + bg_names_nospace

    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Foreground{pathidx[1]}/*.wav'))
    fg_names_spaces = [ff.split('/')[-1].split('.')[0][2:] for ff in fg_dir]
    fg_names_nospace = [ff.replace('_', '') for ff in fg_names_spaces if '_' in ff]
    fg_names = fg_names_spaces + fg_names_nospace

    bool_bg = weight_df['namesA'].str.contains('|'.join(bg_names))
    bool_fg = weight_df['namesB'].str.contains('|'.join(fg_names))
    filtered_df = weight_df[bool_bg & bool_fg]

    return filtered_df


def get_keyword_sound_type(kw, weight_df=weight_df, pathidx=[2, 3], scat_type='suppression',
                           single=True, exact=False, kw2=None):
    # When plotting FR vs weights this will find which sound type (BG/FG) your keyword
    # belongs to to return some info to pass to the plotting function

    df_filtered = weight_df

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{pathidx[0]}/*.wav'))
    bg_names_spaces = [bb.split('/')[-1].split('.')[0][2:] for bb in bg_dir]
    bg_names_underscore = [bb.replace(' ', '_') for bb in bg_names_spaces if ' ' in bb]
    bg_names_nospace = [bb.replace(' ', '') for bb in bg_names_spaces if ' ' in bb]
    bg_names = bg_names_spaces + bg_names_underscore + bg_names_nospace

    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Foreground{pathidx[1]}/*.wav'))
    fg_names_spaces = [ff.split('/')[-1].split('.')[0][2:] for ff in fg_dir]
    fg_names_underscore = [ff.replace(' ', '_') for ff in fg_names_spaces if ' ' in ff]
    fg_names_nospace = [ff.replace(' ', '') for ff in fg_names_spaces if ' ' in ff]
    fg_names = fg_names_spaces + fg_names_underscore + fg_names_nospace

    if kw2:
        kws = [kw, kw2]
        if exact:
            if (kw in bg_names and kw2 in bg_names) or (kw in fg_names and kw2 in fg_names):
                raise ValueError("Both keywords are the same category. Try Again")
        else:
            if any(kw in namebb for namebb in bg_names) & any(kw2 in nameb for nameb in bg_names) or \
                    any(kw in nameff for nameff in fg_names) & any(kw2 in namef for namef in fg_names):
                raise ValueError("Both keywords are the same category. Try Again")
        print(f"Two keywords. Some interactive plotting features may still need"
              f"to be worked out.")
    else:
        kws = [kw]

    for nn, key in enumerate(kws):
        if exact:
            kw_ep = key + '-'   #hacky(?) way of filtering out epochs where kw is part of it
            if '_' in key or ' ' in key:
                raise ValueError(f"Your keyword {key} has a space or underscore in it, "
                                 f"that isn't supported when exact==True.")

            if key in bg_names and key in fg_names:
                raise ValueError(f"Your keyword '{key}' is in BG and FG, this would never happen, "
                                 f"something is wrong.")
            elif key in bg_names:
                kw_info = {'sound_type': 'BG', 'xcol': 'BG_FR', 'ycol': 'weightsA', 'keyword': key}
                df_filtered = df_filtered[df_filtered['namesA'].str.contains(kw_ep)].copy()
                print(f"Keyword '{key}' is a BG. Filtering dataframe.")
            elif key in fg_names:
                kw_info = {'sound_type': 'FG', 'xcol': 'FG_FR', 'ycol': 'weightsB', 'keyword': key}
                df_filtered = df_filtered[df_filtered['namesB'].str.contains(kw_ep)].copy()
                print(f"Keyword '{key}' is a FG. Filtering dataframe.")
            else:
                raise ValueError(f"Your keyword '{key}' is in neither sound type.")

        else:
            if any(key in nameb for nameb in bg_names) & any(key in namef for namef in fg_names):
                raise ValueError(f"Your keyword '{key}' is in BG and FG, be more specific.")
            elif any(key in nameb for nameb in bg_names):
                if sum([key in name for name in bg_names_spaces]) > 1:
                    print(f"Caution: keyword '{key}' is found multiple times in BG list, consider being more specific.")
                kw_info = {'sound_type': 'BG', 'xcol': 'BG_FR', 'ycol': 'weightsA', 'keyword': key}
                df_filtered = df_filtered[df_filtered['namesA'].str.contains(key)].copy()
                print(f"Keyword '{key}' is a BG. Filtering dataframe.")
            elif any(key in namef for namef in fg_names):
                if sum([key in name for name in fg_names_spaces]) > 1:
                    print(f"Caution: keyword '{key}' is found multiple times in FG list, consider being more specific.")
                kw_info = {'sound_type': 'FG', 'xcol': 'FG_FR', 'ycol': 'weightsB', 'keyword': key}
                df_filtered = df_filtered[df_filtered['namesB'].str.contains(key)].copy()
                print(f"Keyword '{key}' is a FG. Filtering dataframe.")
            else:
                raise ValueError(f"Your keyword '{key}' is in neither sound type.")

        if nn == 0:
            kw1_info = kw_info
        if nn == 1:
            print(f"Combining kw_info dictionaries.")
            kw1_info['sound_type2'] = kw_info['sound_type']
            kw1_info['keyword2'] = kw_info['keyword']
            kw_info = kw1_info
            if len(df_filtered) == 0:
                raise ValueError(f"Keywords '{kw}' and '{kw2}' were never played as a soundpair"
                                 f"so the dataframe would be empty.")
            else:
                print(f"The sound pairings of '{kw}' and '{kw2}' were played {len(df_filtered)} times.")

    if single:
        kw_info['fn'] = plot_single_psth
        fn_args = {'df_filtered': df_filtered, 'sound_type': kw_info['sound_type']}
    else:
        kw_info['fn'] = plot_psth_scatter
        fn_args = {'df_filtered': df_filtered, 'scatter': scat_type}

    return df_filtered, kw_info, fn_args


def get_keyword_list(kws, weight_df=weight_df, pathidx=[2, 3],
                           scat_type='suppression', single=True):
    # When plotting FR vs weights this will find which sound type (BG/FG) your keyword
    # belongs to to return some info to pass to the plotting function

    df_filtered = weight_df
    dfs = []

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{pathidx[0]}/*.wav'))
    bg_names_spaces = [bb.split('/')[-1].split('.')[0][2:] for bb in bg_dir]
    bg_names_underscore = [bb.replace(' ', '_') for bb in bg_names_spaces if ' ' in bb]
    bg_names_nospace = [bb.replace(' ', '') for bb in bg_names_spaces if ' ' in bb]
    bg_names = bg_names_spaces + bg_names_underscore + bg_names_nospace

    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Foreground{pathidx[1]}/*.wav'))
    fg_names_spaces = [ff.split('/')[-1].split('.')[0][2:] for ff in fg_dir]
    fg_names_underscore = [ff.replace(' ', '_') for ff in fg_names_spaces if ' ' in ff]
    fg_names_nospace = [ff.replace(' ', '') for ff in fg_names_spaces if ' ' in ff]
    fg_names = fg_names_spaces + fg_names_underscore + fg_names_nospace

    if len(kws) == 1:
        kws = [kws]

    for nn, key in enumerate(kws):
        if any(key in nameb for nameb in bg_names) & any(key in namef for namef in fg_names):
            raise ValueError(f"Your keyword '{key}' is in BG and FG, be more specific.")
        elif any(key in nameb for nameb in bg_names):
            if sum([key in name for name in bg_names_spaces]) > 1:
                print(f"Caution: keyword '{key}' is found multiple times in BG list, consider being more specific.")
            kw_info = {'sound_type': 'BG', 'xcol': 'BG_FR', 'ycol': 'weightsA', 'keyword': key}
            filtered = df_filtered[df_filtered['namesA'].str.contains(key)].copy()
            print(f"Keyword '{key}' is a BG. Filtering dataframe.")
        elif any(key in namef for namef in fg_names):
            if sum([key in name for name in fg_names_spaces]) > 1:
                print(f"Caution: keyword '{key}' is found multiple times in FG list, consider being more specific.")
            kw_info = {'sound_type': 'FG', 'xcol': 'FG_FR', 'ycol': 'weightsB', 'keyword': key}
            filtered = df_filtered[df_filtered['namesB'].str.contains(key)].copy()
            print(f"Keyword '{key}' is a FG. Filtering dataframe.")
        else:
            raise ValueError(f"Your keyword '{key}' is in neither sound type.")

        if nn == 0:
            kw1_info = kw_info
        if nn > 0:
            print(f"Combining kw_info dictionaries.")
            kw1_info[f'sound_type{nn+1}'] = kw_info['sound_type']

        dfs.append(filtered)

    kw_info['keyword'] = kws
    df_filtered = pd.concat(dfs)

    if single:
        kw_info['fn'] = plot_single_psth
        fn_args = {'df_filtered': df_filtered, 'sound_type': kw_info['sound_type']}
    else:
        kw_info['fn'] = plot_psth_scatter
        fn_args = {'df_filtered': df_filtered, 'scatter': scat_type}

    return df_filtered, kw_info, fn_args


def get_resp_and_stim_info_old_format(cellid_and_stim_str, df_filtered):
    cellid, stimA, stimB = cellid_and_stim_str.split(':')
    cell_df = df_filtered.loc[cellid]
    this_cell_stim = cell_df[(cell_df['namesA'] == stimA) & (cell_df['namesB'] == stimB)].iloc[0]
    animal_id = cellid[:3]

    if animal_id == 'HOD' or animal_id == 'TBR':
        batch = 333
    elif animal_id == 'ARM':
        # got to code in for batch to differentiate between A1 and PEG batches,
        # where can I get that info above?
        batch = 0

    fs=100
    expt = BAPHYExperiment(cellid=cellid, batch=batch)
    rec = expt.get_recording(rasterfs=fs, resp=True, stim=False)
    resp = rec['resp'].rasterize()

    BG, FG = stimA.split('-')[0], stimB.split('-')[0]

    #parts for spectrograms now
    if animal_id == 'HOD':
        folder_ids = [1,1]
    elif animal_id == 'TBR':
        folder_ids = [2,3]
    elif animal_id == 'ARM':
        folder_ids = [1,2]

    parms = {'cellid': cellid, 'animal_id': animal_id, 'BG': BG, 'FG': FG, 'resp': resp,
             'this_cell_stim': this_cell_stim, 'fs': fs, 'folder_ids': folder_ids}

    return parms



def get_resp_and_stim_info(cellid_and_stim_str, df_filtered):
    cellid, stimA, stimB = cellid_and_stim_str.split(':')
    cell_df = df_filtered.loc[df_filtered.cellid == cellid]
    this_cell_stim = cell_df[(cell_df['BG'] == stimA) & (cell_df['FG'] == stimB)].iloc[0]
    animal_id = cellid[:3]

    if animal_id == 'HOD' or animal_id == 'TBR':
        batch = 333
    elif animal_id == 'ARM':
        # got to code in for batch to differentiate between A1 and PEG batches,
        # where can I get that info above?
        batch = 0
    elif animal_id == 'CLT':
        batch = 340

    fs=100
    expt = BAPHYExperiment(cellid=cellid, batch=batch)
    rec = expt.get_recording(rasterfs=fs, resp=True, stim=False)
    resp = rec['resp'].rasterize()

    BG, FG = stimA.split('-')[0], stimB.split('-')[0]

    #parts for spectrograms now
    if animal_id == 'HOD':
        folder_ids = [1,1]
    elif animal_id == 'TBR':
        folder_ids = [2,3]
    elif animal_id == 'ARM':
        folder_ids = [1,2]
    elif animal_id == 'CLT':
        folder_ids = [2,3]

    parms = {'cellid': cellid, 'animal_id': animal_id, 'BG': BG, 'FG': FG, 'resp': resp,
             'this_cell_stim': this_cell_stim, 'fs': fs, 'folder_ids': folder_ids}

    return parms


def df_to_weight_df(df):
    weight_df = pd.concat(df['weight_dfR'].values,keys=df.index)
    # BGgroups = pd.concat(df['WeightAgroupsR'].values,keys=df.index)
    # FGgroups = pd.concat(df['WeightBgroupsR'].values,keys=df.index)
    animal_list = [anim[0][:3] for anim in weight_df.index]
    cells = [cell[0] for cell in weight_df.index]
    weight_df.insert(4, 'Animal', animal_list)
    spont_list = [float(df.loc[cell]['spont_rate']) for cell in cells]
    weight_df.insert(5, 'spont_rate', spont_list)
    #Add suppression column to weights_df
    supp_df = pd.DataFrame()
    for cll in df.index:
        supp = df.loc[cll,'suppression']
        fr = df.loc[cll, 'FR']

        names = [ts.get_sep_stim_names(sn) for sn in df.loc[cll,'pair_names']]
        BGs, FGs = [rr[0] for rr in names], [qq[1] for qq in names]
        cell_df = pd.DataFrame({'suppression': supp,
                               'BG_FR': fr[:,0],
                               'FG_FR': fr[:,1],
                               'Combo_FR': fr[:,2],
                               'namesA': BGs,
                               'namesB': FGs,
                               'cellid': cll})
        supp_df = supp_df.append(cell_df)

    supp_df = supp_df.set_index('cellid', append=True)
    supp_df = supp_df.swaplevel(0,1)
    supp_df = supp_df.set_index(['namesA','namesB'], append=True)
    weight_df = weight_df.set_index(['namesA','namesB'], append=True)
    joint = pd.concat([weight_df, supp_df], axis=1)
    weight_df = joint.reset_index(['namesA','namesB'])
    weight_df['BG_FRns'] = weight_df['BG_FR'] - weight_df['spont_rate']
    weight_df['FG_FRns'] = weight_df['FG_FR'] - weight_df['spont_rate']

    return weight_df


def quadrants_by_FRns(weight_df, threshold=0.05, quad_return=5):
    '''Filters a dataframe by a FR threshold with spont subtracted. quad_returns says which
    filtered quadrants to return. If you give a list it'll output as a dictionary with keys
    which quadrant, if a single integer it'll just be the dataframe outputted. Default is 5
    which takes a combination of BG+/FG+, BG+/FG-, and BG-/FG+.'''
    quad1 = weight_df.loc[(weight_df.BG_FRns<=-threshold) & (weight_df.FG_FRns>=threshold)]
    quad2 = weight_df.loc[(np.abs(weight_df.BG_FRns)<=threshold) & (weight_df.FG_FRns>=threshold)]
    quad3 = weight_df.loc[(weight_df.BG_FRns>=threshold) & (weight_df.FG_FRns>=threshold)]
    quad4 = weight_df.loc[(weight_df.BG_FRns<=-threshold) & (np.abs(weight_df.FG_FRns)<=threshold)]
    quad6 = weight_df.loc[(weight_df.BG_FRns>=threshold) & (np.abs(weight_df.FG_FRns)<=threshold)]
    quad10 = pd.concat([quad2, quad3, quad6], axis=0)
    quad7 = weight_df.loc[(weight_df.BG_FRns<=-threshold) & (weight_df.FG_FRns<=-threshold)]
    quad8 = weight_df.loc[(np.abs(weight_df.BG_FRns)<=threshold) & (weight_df.FG_FRns<=-threshold)]
    quad9 = weight_df.loc[(weight_df.BG_FRns>=threshold) & (weight_df.FG_FRns<=-threshold)]
    quad5 = weight_df.loc[(np.abs(weight_df.BG_FRns)<threshold) & (np.abs(weight_df.FG_FRns)<threshold)]
    dfs = [quad1, quad2, quad3, quad4, quad5, quad6, quad7, quad8, quad9, quad10]
    if isinstance(quad_return, list):
        quads = {qq:dfs[qq - 1] for qq in quad_return}
    elif isinstance(quad_return, int):
        quads = dfs[quad_return - 1]
    else:
        raise ValueError(f"quad_return input {quad_return} is not a list or int.")

    return quads, threshold


def weight_hist_dual(df, tag=None):
    '''Dataframe give should be df_filtered'''
    bins=np.arange(-2,2,.05)
    animals = list(df.Animal.unique())
    dfs = [df.loc[df.Animal==aa] for aa in animals]
    fig, axes = plt.subplots(1, len(animals), figsize=(5*len(animals),5))
    for (ax, Df) in zip(axes, dfs):
        to_plot = Df.loc[:, ['weightsA', 'weightsB']].values.T
        ax.hist(to_plot[0,:],bins=bins,histtype='step')
        ax.hist(to_plot[1,:],bins=bins,histtype='step')
        ax.legend(('Background','Foreground'))
        ax.set_xlabel('Weight')
        ax.set_title(f"{str(df.Animal.unique())} - {tag}")


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


def plot_psth(cellid_and_stim_str, df_filtered=weight_df, plot_error=True):
    # Version of popup psth plot that includes spectrograms below psth
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile
    import scipy.ndimage.filters as sf

    parms = get_resp_and_stim_info(cellid_and_stim_str, df_filtered)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    weightBG, weightFG = this_cell_stim['weightsA'], this_cell_stim['weightsB']
    epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1']
            # , f'STIM_{BG}-0.5-1_{FG}-0-1']

    resp = resp.extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    f = plt.figure(figsize=(12, 9))
    psth = plt.subplot2grid((4, 3), (0, 0), rowspan=2, colspan=3)
    specA = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
    specB = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)
    ax = [psth, specA, specB]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs ) - prestim

    bg_alone, fg_alone = epochs[0], epochs[1]
    r_mean = {e:np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
    r_mean['Weight Model'] = (r_mean[bg_alone] * weightBG) + (r_mean[fg_alone] * weightFG)
    r_mean['Linear Sum'] = r_mean[bg_alone] + r_mean[fg_alone]

    colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray', 'black']
    styles = ['-', '-', '-', '-', ':']

    for e, c, s in zip(r_mean.keys(), colors, styles):
        ax[0].plot(time, sf.gaussian_filter1d(r_mean[e], sigma=1)
             * fs, color=c, linestyle=s, label=e)
    ax[0].legend(('Bg, weight={:.2f}'.format(this_cell_stim.weightsA),
                  'Fg, weight={:.2f}'.format(this_cell_stim.weightsB),
                  'Both',
                  'Weight Model, r={:.2f}'.format(this_cell_stim.r),
                  'Linear Sum'))
    ax[0].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['suppression']}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymax*0.5, ymax, color='black', lw=0.75, ls=':')
    ax[0].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))

    bg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
              f'Background{folder_ids[0]}/{BG}.wav'
    fg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
              f'Foreground{folder_ids[1]}/{FG}.wav'

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([0, 20, 40, 60, 80]), ax[2].set_yticks([])
    ax[2].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)


def plot_psth_scatter(cellid_and_stim_str, df_filtered=weight_df, scatter='suppression'):
    # Version of popup psth plot that includes spectrograms below psth
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile
    import scipy.ndimage.filters as sf

    parms = get_resp_and_stim_info(cellid_and_stim_str, df_filtered)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    weightBG, weightFG = this_cell_stim['weightsA'], this_cell_stim['weightsB']
    epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1']
            # , f'STIM_{BG}-0.5-1_{FG}-0-1']

    resp = resp.extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    f = plt.figure(figsize=(12, 9))
    psth = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=3)
    specA = plt.subplot2grid((4, 5), (2, 0), rowspan=1, colspan=3)
    specB = plt.subplot2grid((4, 5), (3, 0), rowspan=1, colspan=3)
    scat = plt.subplot2grid((4, 5), (0, 3), rowspan=2, colspan=2)
    ax = [psth, specA, specB, scat]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs ) - prestim

    bg_alone, fg_alone, combo = epochs[0], epochs[1], epochs[2]
    r_mean = {e:np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
    r_mean['Weight Model'] = (r_mean[bg_alone] * weightBG) + (r_mean[fg_alone] * weightFG)
    r_mean['Linear Sum'] = r_mean[bg_alone] + r_mean[fg_alone]

    colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray', 'black']
    styles = ['-', '-', '-', '-', ':']

    for e, c, s in zip(r_mean.keys(), colors, styles):
        ax[0].plot(time, sf.gaussian_filter1d(r_mean[e], sigma=1)
                    * fs, color=c, linestyle=s, label=e)
    ax[0].legend((f'BG, weight={np.around(this_cell_stim.weightsA, 2)}, FR={np.around(this_cell_stim.BG_FRns, 3)}',
                  f'FG, weight={np.around(this_cell_stim.weightsB, 2)}, FR={np.around(this_cell_stim.FG_FRns, 3)}',
                  'Both',
                  'Weight Model, r={:.2f}'.format(this_cell_stim.r),
                  'Linear Sum'))
    ax[0].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['suppression']}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymax*0.5, ymax, color='black', lw=0.75, ls=':')
    ax[0].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))

    bg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
              f'Background{folder_ids[0]}/{BG}.wav'
    fg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
              f'Foreground{folder_ids[1]}/{FG}.wav'

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([0, 20, 40, 60, 80]), ax[2].set_yticks([])
    ax[2].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)

    prebins = int(prestim * fs)

    bg, fg = r_mean[bg_alone][prebins:-prebins], r_mean[fg_alone][prebins:-prebins]
    both = r_mean[combo][prebins:-prebins]
    suppression = r_mean['Linear Sum'][prebins:-prebins] - r_mean[combo][prebins:-prebins]
    if scatter == 'suppression':
        a = ax[3].scatter(bg, fg, c=suppression, cmap='inferno', s=15)
        ax[3].set_title('color = Suppression')
    elif scatter == 'combo':
        a = ax[3].scatter(bg, fg, c=both, cmap='inferno', s=15)
        ax[3].set_title('color = BG+FG Response')
    else:
        raise ValueError(f"Somehow you inputted '{scatter}' for scatter, use 'suppression' or 'combo.'")
    ax[3].set_xlabel('r(BG)'), ax[3].set_ylabel('r(FG)')
    f.colorbar(a)
    f.tight_layout()


def plot_single_psth(cellid_and_stim_str, sound_type, df_filtered=df_filtered):
    # Version of popup psth plot that includes spectrograms below psth
    parms = get_resp_and_stim_info(cellid_and_stim_str, df_filtered)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    if sound_type == 'BG':
        epochs = f'STIM_{BG}-0-1_null'
        color = 'deepskyblue'
    elif sound_type == 'FG':
        epochs = f'STIM_null_{FG}-0-1'
        color = 'yellowgreen'
    else:
        raise ValueError(f"sound_type must be 'BG' or 'FG', {sound_type} is invalid.")

    resp = parms['resp'].extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    sem = np.squeeze(stats.sem(r[epochs], axis=0, nan_policy='omit'))
    r_mean = np.squeeze(np.nanmean(r[epochs], axis=0))

    f = plt.figure(figsize=(12, 6))
    psth = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)
    spec = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3)
    ax = [psth, spec]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs].shape[-1]) / fs) - prestim

    ax[0].plot(time, sf.gaussian_filter1d(r_mean, sigma=1)
         * fs, color=color, label=epochs)

    ax[0].fill_between(time, sf.gaussian_filter1d((r_mean - sem) * fs, sigma=1),
                    sf.gaussian_filter1d((r_mean + sem) * fs, sigma=1),
                    alpha=0.3, color='grey')
    ax[0].legend((f"{sound_type}, weight={np.around(this_cell_stim['weightsA'],2)}\n"
                  f"{sound_type}, firing rate={this_cell_stim['BG_FR']}",
                  ' '))
    ax[0].set_title(f"{cellid_and_stim_str}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymax*0.5, ymax, color='black', lw=0.75, ls=':')
    ax[0].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))

    BG, FG = int(BG[:2]), int(FG[:2])

    if sound_type == 'BG':
        path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Background{folder_ids[0]}/*.wav'))[BG - 1]
    elif sound_type =='FG':
        path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
                           f'Foreground{folder_ids[1]}/*.wav'))
    else:
        raise ValueError(f"sound_type must be 'BG' or 'FG', {sound_type} is invalid.")

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)



def psth_responses_by_kw(cellid_and_stim_str, df_filtered, kw, sound_type, sigma=2, save=False):
    # Version of popup psth plot that includes spectrograms below psth
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile
    import scipy.ndimage.filters as sf
    from pathlib import Path
    from scipy import stats
    import glob

    parms = get_resp_and_stim_info(cellid_and_stim_str, df_filtered)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    colors = ['deepskyblue', 'yellowgreen', 'grey', 'silver']

    if animal_id == 'TBR':
        BGf, FGf = BG.replace(' ', ''), FG.replace(' ', '')

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Foreground{folder_ids[1]}/*.wav'))
    bg_path, fg_path = bg_dir[int(BG[:2]) - 1], fg_dir[int(FG[:2]) - 1]

    epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1',
              f'STIM_{BG}-0.5-1_{FG}-0-1']

    resp = parms['resp'].extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs) - prestim

    f = plt.figure(figsize=(15,9))
    psth = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=5)
    specBG = plt.subplot2grid((4, 5), (2, 0), rowspan=1, colspan=5)
    specFG = plt.subplot2grid((4, 5), (3, 0), rowspan=1, colspan=5)

    rBG, rFG = np.nanmean(r[epochs[0]][:,0,:], axis=0), np.nanmean(r[epochs[1]][:,0,:],axis=0)
    rlin = rBG + rFG

    ax = [psth, specBG, specFG]

    for e, c in zip(epochs, colors):
        ax[0].plot(time, sf.gaussian_filter1d(np.nanmean(r[e][:,0,:], axis=0), sigma=sigma)
             * fs, color=c, label=e)
    ax[0].plot(time, sf.gaussian_filter1d(rlin, sigma=sigma) * fs, color='grey', ls=':', label='Linear Sum')
    ax[0].legend((f'{epochs[0]}, weight={np.around(this_cell_stim.weightsA,2)}',
                  f'{epochs[1]}, weight={np.around(this_cell_stim.weightsB,2)}',
                  f'{epochs[2]}',
                  f'{epochs[3]}',
                  'Linear Sum'))
    ax[0].set_title(f"{cellid} - BG: {BG} - FG: {FG} - sigma={sigma}", weight='bold')
    # ax[0].set_xlim([0-(prestim/2), time[-1]])
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymin, ymax, color='black', lw=0.75, ls=':')

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)
    ymin2, ymax2 = ax[1].get_ylim()
    ax[1].vlines((spec.shape[-1]+1)/2, ymin2, ymax2, color='white', lw=0.75, ls=':')

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([0, 20, 40, 60, 80]), ax[2].set_yticks([])
    ax[2].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)
    ymin3, ymax3 = ax[2].get_ylim()
    ax[2].vlines((spec.shape[-1]+1)/2, ymin3, ymax3, color='white', lw=0.75, ls=':')

    ax[2].set_xlabel('Seconds', weight='bold')
    ax[1].set_ylabel(f"Background:\n{BG}", weight='bold', labelpad=-80, rotation=0)
    ax[2].set_ylabel(f"Foreground:\n{FG}", weight='bold', labelpad=-80, rotation=0)

    if save:
        site = cellid.split('-')[0]
        path = f"/home/hamersky/OLP PSTHs/{sound_type}/{kw}/{animal_id}/"
        # if os.path.isfile(path):
        Path(path).mkdir(parents=True, exist_ok=True)

        plt.savefig(path + f"{cellid} - {BG} - {FG} - sigma{sigma}.png")
        plt.close()


def plot_weight_psth(cellid_and_stim_str, df_filtered, save=False):
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile
    import scipy.ndimage.filters as sf
    from pathlib import Path
    from scipy import stats
    import glob

    parms = get_resp_and_stim_info(cellid_and_stim_str, df_filtered)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    weightBG, weightFG = this_cell_stim['weightsA'], this_cell_stim['weightsB']
    epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1']
    # , f'STIM_{BG}-0.5-1_{FG}-0-1']

    resp = resp.extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    f = plt.figure(figsize=(9, 5))
    psth = plt.subplot2grid((14, 3), (4, 0), rowspan=7, colspan=6)
    specA = plt.subplot2grid((14, 3), (0, 0), rowspan=2, colspan=6)
    specB = plt.subplot2grid((14, 3), (2, 0), rowspan=2, colspan=6)
    ax = [specA, specB, psth]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs) - prestim

    bg_alone, fg_alone = epochs[0], epochs[1]
    r_mean = {e: np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
    r_mean['Weight Model'] = (r_mean[bg_alone] * weightBG) + (r_mean[fg_alone] * weightFG)
    # r_mean['Linear Sum'] = r_mean[bg_alone] + r_mean[fg_alone]

    colors = ['deepskyblue', 'yellowgreen', 'dimgray', 'lightcoral']
    styles = ['-', '-', '-', '-']

    for e, c, s in zip(r_mean.keys(), colors, styles):
        ax[2].plot(time, sf.gaussian_filter1d(r_mean[e], sigma=1)
                   * fs, color=c, linestyle=s, label=e)
    ax[2].legend((f"BG, weight={np.around(this_cell_stim.weightsA, 3)}",
                  f"FG, weight={np.around(this_cell_stim.weightsB, 3)}",
                  f"BG+FG Combo",
                  f"Model Prediction, r={np.around(this_cell_stim.r, 2)}"), fontsize=7)
    # ax[2].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['suppression']}")
    ymin, ymax = ax[2].get_ylim()
    ax[2].vlines([0, 1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    # ax[2].vlines(0.5, ymax * 0.5, ymax, color='black', lw=0.75, ls=':')
    ax[2].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))
    ax[2].set_xlabel('Time (s)', fontweight='bold', size=10)
    ax[2].set_ylabel('spk/s', fontweight='bold', size=10)

    folder_ids = [2,3]
    # if FG == '12TsikEk':
    #     FG = '12Tsik Ek'

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Foreground{folder_ids[1]}/*.wav'))
    bg_path, fg_path = bg_dir[int(BG[:2]) - 1], fg_dir[int(FG[:2]) - 1]
    #
    # bg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
    #           f'Background{folder_ids[0]}/{BG}.wav'
    # fg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
    #           f'Foreground{folder_ids[1]}/{FG}.wav'
    xf = 100
    xmin, xmax = ax[2].get_xlim()
    low, high = xmin * xf, xmax * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[0].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[0].set_xlim(low, high)
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xticklabels([]), ax[0].set_yticklabels([])
    ax[0].spines['top'].set_visible(False), ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False), ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel(f"BG: {BG[2:]}", rotation=0, fontweight='bold',
                     size=8, labelpad=-35)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[1].set_xticklabels([]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)
    ax[1].set_ylabel(f"FG: {FG[2:]}", rotation=0, fontweight='bold',
                     size=8, labelpad=-35)

    ymin, ymax = ax[1].get_ylim()
    ax[0].vlines([0, 1 * fs], ymin, ymax, color='black', lw=0.5)
    ax[0].hlines([ymin + 2, ymax], 0, 1 * fs, color='black', lw=0.5)
    ax[1].vlines([0, 1 * fs], ymin, ymax, color='black', lw=0.5)
    ax[1].hlines([ymin + 1, ymax], 0, 1 * fs, color='black', lw=0.5)
    ax[0].set_title(f"{cellid}", fontweight='bold', size=12)

    if save:
        if this_cell_stim.r >= 0.8:
            path = f"/home/hamersky/OLP PSTHs Weights/"
            # if os.path.isfile(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            print(f"Saving {cellid}")
            plt.savefig(path + f"{cellid} - {BG} - {FG}.png")
            plt.close()
        else:
            print(f"r is too low {this_cell_stim.r}, not saving.")
            plt.close()


def plot_model_diagram_parts(cellid_and_stim_str, df_filtered):
    '''This function works a little backwards. But it takes this string of a stimulus
    and the weight_df and turns it into a schematic of the linear weighted model. Cool.
    But, generating the stimulus string is dumb and should get simplified. But if it's
    just for this function I'm not too worried right now, going to make a wrapper for it in
    /OLP_figures 2022_08_25'''
    parms = get_resp_and_stim_info(cellid_and_stim_str, df_filtered)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    weightBG, weightFG = this_cell_stim['weightsA'], this_cell_stim['weightsB']
    epo = this_cell_stim.epoch
    epochs = [f"STIM_{epo.split('_')[1]}_null", f"STIM_null_{epo.split('_')[2]}", epo]

    resp = resp.extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    f = plt.figure(figsize=(10, 9))
    psthA = plt.subplot2grid((14, 7), (1, 0), rowspan=2, colspan=3)
    specA = plt.subplot2grid((14, 7), (0, 0), rowspan=1, colspan=3)
    specB = plt.subplot2grid((14, 7), (4, 0), rowspan=1, colspan=3, sharey=specA)
    psthB = plt.subplot2grid((14, 7), (5, 0), rowspan=2, colspan=3, sharex=psthA, sharey=psthA)
    specC = plt.subplot2grid((14, 7), (9, 0), rowspan=1, colspan=3,  sharey=specA)
    psthC = plt.subplot2grid((14, 7), (10, 0), rowspan=2, colspan=3, sharex=psthA, sharey=psthA)
    psthW = plt.subplot2grid((14, 7), (1, 4), rowspan=2, colspan=3, sharex=psthA, sharey=psthA)
    psthW1 = plt.subplot2grid((14, 7), (5, 4), rowspan=2, colspan=3, sharex=psthA, sharey=psthA)


    ax = [specA, psthA, specB, psthB, specC, psthC, psthW, psthW1]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs) - prestim

    bg_alone, fg_alone = epochs[0], epochs[1]
    r_mean = {e: np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
    r_mean['Weight Model'] = (r_mean[bg_alone] * weightBG) + (r_mean[fg_alone] * weightFG)
    # r_mean['Linear Sum'] = r_mean[bg_alone] + r_mean[fg_alone]
    r_mean['weight_bg'] = r_mean[bg_alone] * weightBG
    r_mean['weight_fg'] = r_mean[fg_alone] * weightFG


    colors = ['deepskyblue', 'yellowgreen', 'dimgray', 'lightcoral', 'deepskyblue', 'yellowgreen']
    styles = ['-', '-', '-', '-', '-', '-']
    axnum = [1, 3, 5, 5, 6, 7]

    for e, c, s, a in zip(r_mean.keys(), colors, styles, axnum):
        ax[a].plot(time, sf.gaussian_filter1d(r_mean[e], sigma=1)
                   * fs, color=c, linestyle=s, label=e)
    ax[5].legend((f"BG+FG Combo",
                  f"Model Prediction, r={np.around(this_cell_stim.r, 2)}"), fontsize=6)
    # ax[2].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['suppression']}")
    ymin, ymax = ax[1].get_ylim()
    for e, c, s, a in zip(r_mean.keys(), colors, styles, axnum):
        ax[a].vlines([0, 1.0], ymin, ymax, color='black', lw=0.75, ls='--')
        ax[a].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))
        ax[a].set_xlabel('Time (s)', fontweight='bold', size=10)
        ax[a].set_ylabel('spk/s', fontweight='bold', size=10)
    ax[6].set_xticks([]), ax[6].set_yticks([]), ax[6].set_xlabel('')
    ax[6].set_xticklabels([]), ax[6].set_yticklabels([]), ax[6].set_ylabel('')
    ax[7].set_xticks([]), ax[7].set_yticks([]), ax[7].set_xlabel('')
    ax[7].set_xticklabels([]), ax[7].set_yticklabels([]), ax[7].set_ylabel('')

    folder_ids = parms['folder_ids']
    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Foreground{folder_ids[1]}/*.wav'))
    bg_path = [bb for bb in bg_dir if epo.split('_')[1].split('-')[0][:2] in bb][0]
    fg_path = [ff for ff in fg_dir if epo.split('_')[2].split('-')[0][:2] in ff][0]

    xf = 100
    xmin, xmax = ax[1].get_xlim()
    low, high = xmin * xf, xmax * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    WA = W
    ax[0].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[0].set_xlim(low, high)
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xticklabels([]), ax[0].set_yticklabels([])
    ax[0].spines['top'].set_visible(False), ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False), ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel(f"BG: {epo.split('_')[1][2:].split('-')[0]}", rotation=0, fontweight='bold',
                     size=8, labelpad=-5)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([]), ax[2].set_yticks([])
    ax[2].set_xticklabels([]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)
    ax[2].set_ylabel(f"FG: {epo.split('_')[2][2:].split('-')[0]}", rotation=0, fontweight='bold',
                     size=8, labelpad=-5)

    Ws = WA + W
    spec = gtgram(Ws, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[4].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
    ax[4].set_xlim(low, high)
    ax[4].set_xticks([]), ax[4].set_yticks([])
    ax[4].set_xticklabels([]), ax[4].set_yticklabels([])
    ax[4].spines['top'].set_visible(False), ax[4].spines['bottom'].set_visible(False)
    ax[4].spines['left'].set_visible(False), ax[4].spines['right'].set_visible(False)
    ax[4].set_ylabel(f"BG+FG", rotation=0, fontweight='bold',
                     size=8, labelpad=-5)

    ymin, ymax = ax[1].get_ylim()
    ax[0].vlines([0, 1 * fs], ymin, ymax, color='black', lw=0.5)
    ax[0].hlines([ymin + 2, ymax], 0, 1 * fs, color='black', lw=0.5)
    ax[2].vlines([0, 1 * fs], ymin, ymax, color='black', lw=0.5)
    ax[2].hlines([ymin + 1, ymax], 0, 1 * fs, color='black', lw=0.5)
    ax[4].vlines([0, 1 * fs], ymin, ymax, color='black', lw=0.5)
    ax[4].hlines([ymin + 1, ymax], 0, 1 * fs, color='black', lw=0.5)
    ax[0].set_title(f"{cellid}", fontweight='bold', size=12)
    ax[4].set_title(f"Weight BG = {np.around(this_cell_stim.weightsA,2)}\n"
                    f"Weight FG = {np.around(this_cell_stim.weightsB,2)}",
                    fontweight='bold', size=8)


def get_cellstring_old_format(cell, BG, FG, weight_df):
    '''Pretty simple, just gets a cell string in a format that the above two functions take.
    It doesn't have to be this way, but it was an easy existing framework for it.'''
    cellid_and_stim_strs = [index[0] + ':' + nameA + ':' + nameB for index, nameA, nameB in \
                            zip(weight_df.index.values,
                                weight_df['namesA'], weight_df['namesB'])]
    cellid_and_stim_str = [ee for ee in cellid_and_stim_strs if ee.split(':')[0] == cell
                           if ee.split(':')[1][:2] == BG if ee.split(':')[2][:2] == FG][0]

    return cellid_and_stim_str


def get_cellstring(cell, BG, FG, weight_df):
    '''Changed on 2022_08_25 to reflect current naming conventions in weight_df.
    Pretty simple, just gets a cell string in a format that the above two functions take.
    It doesn't have to be this way, but it was an easy existing framework for it.'''
    cellid_and_stim_strs = [index + ':' + nameA + ':' + nameB for index, nameA, nameB in \
                            zip(weight_df.cellid.values,
                                weight_df['BG'], weight_df['FG'])]
    cellid_and_stim_str = [ee for ee in cellid_and_stim_strs if ee.split(':')[0] == cell
                           if ee.split(':')[1] == BG if ee.split(':')[2] == FG][0]

    return cellid_and_stim_str


def split_psth_multiple_units(df_filtered, sortby='random', order='low',
                              folder_ids=[2,3], sigma=2):
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile
    import scipy.ndimage.filters as sf
    from pathlib import Path
    from scipy import stats
    import glob
    import random

    rows = 15
    fig, axes = plt.subplots(rows + 1, 3, sharey='row')
    if sortby == 'random':
        locs = random.sample(range(0, len(df_filtered)), rows)
        locs.sort()
        print(f"Choosing random indexes, displaying df_filtered indexes {locs}")
        df_plot = df_filtered.iloc[locs]
    else:
        if sortby in list(df_filtered.columns):
            df_sort = df_filtered.sort_values(sortby).copy()
            if order == 'high':
                df_plot = df_sort[-rows:]
            elif order == 'low':
                df_plot = df_sort[:rows]
            else:
                raise ValueError(f"order input '{order}' is wrong. Input 'high' or 'low.'")
        else:
            raise ValueError(f"sortby value '{sortby}' is not a column in the df. Try again.")
    axes = np.ravel(axes)

    cellid_and_stim_strs = [index[0] + ':' + nameA + ':' + nameB for index, nameA, nameB in \
                            zip(df_plot.index.values,
                                df_plot['namesA'], df_plot['namesB'])]
    parms = get_resp_and_stim_info(cellid_and_stim_strs[0], df_plot)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Foreground{folder_ids[1]}/*.wav'))
    bg_path, fg_path = bg_dir[int(BG[:2]) - 1], fg_dir[int(FG[:2]) - 1]

    # Plot the spectrograms above the respective traces, it'll look cool
    sfs, W = wavfile.read(bg_path)
    specBG = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    sfs, W = wavfile.read(fg_path)
    specFG = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    specCombo = specBG + specFG
    labls = [f'BG: {BG[2:]}', f'FG: {FG[2:]}', f'Combo: {BG[2:]} + {FG[2:]}']

    axes[0].imshow(specBG, aspect='auto', origin='lower', extent=[0, specBG.shape[1], 0, specBG.shape[0]])
    axes[1].imshow(specFG, aspect='auto', origin='lower', extent=[0, specFG.shape[1], 0, specFG.shape[0]])
    axes[2].imshow(specCombo, aspect='auto', origin='lower', extent=[0, specCombo.shape[1], 0, specCombo.shape[0]])
    for nn, ll in enumerate(labls):
        axes[nn].spines['top'].set_visible(False), axes[nn].spines['bottom'].set_visible(False)
        axes[nn].spines['left'].set_visible(False), axes[nn].spines['right'].set_visible(False)
        axes[nn].set_yticks([]), axes[nn].set_xticks([])
        axes[nn].set_title(ll, weight='bold')
    axes[0].set_ylabel(f"Sort Order: {order}", rotation=0, weight='bold', labelpad=60)

    # Add the data
    axn = 3
    colors = ['deepskyblue', 'yellowgreen', 'grey']

    for css in cellid_and_stim_strs:
        parms = get_resp_and_stim_info(css, df_plot)
        cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()
        epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1']

        resp = parms['resp'].extract_channels([cellid])
        r = resp.extract_epochs(epochs)

        rBG, rFG = np.nanmean(r[epochs[0]][:, 0, :], axis=0), np.nanmean(r[epochs[1]][:, 0, :], axis=0)
        rlin = rBG + rFG

        prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
        time = (np.arange(0, r[epochs[0]].shape[-1]) / fs) - prestim

        axes[axn].set_ylabel(f'\n{cellid}\n{sortby}: {np.around(this_cell_stim.suppression,5)}',\
                             rotation=0, labelpad=60, weight='bold')

        for e, c in zip(epochs, colors):
            axes[axn].plot(time, sf.gaussian_filter1d(np.nanmean(r[e][:,0,:], axis=0), sigma=sigma)
                 * fs, color=c)
            ymin, ymax = axes[axn].get_ylim()
            axes[axn].vlines([0, 1.0], ymin, ymax, color='black', lw=0.75, ls='--')
            axes[axn].vlines(0.5, ymin, ymax, color='black', lw=0.75, ls=':')
            if e == epochs[-1]:
                axes[axn].plot(time, sf.gaussian_filter1d(rlin, sigma=sigma) * fs, color='grey', ls=':')
            axn += 1


def split_psth_highest_lowest(df_filtered, sortby='suppression', rows=15, folder_ids=[2,3], sigma=2):

    if rows * 2 > len(df_filtered):
        print(f"You wanted {rows} rows but there are only {len(df_filtered)} units at this site,"
              f"adjusting rows to make things more tidy.")
        rows = int(np.ceil(len(df_filtered)/2))

    fig, axes = plt.subplots(rows + 1, 6)
    if sortby in list(df_filtered.columns):
        df_sort = df_filtered.sort_values(sortby).copy()
    else:
        raise ValueError(f"sortby value '{sortby}' is not a column in the df. Try again.")
    axes = np.ravel(axes)

    cellid_and_stim_strs = [index[0] + ':' + nameA + ':' + nameB for index, nameA, nameB in \
                            zip(df_sort.index.values,
                                df_sort['namesA'], df_sort['namesB'])]
    parms = get_resp_and_stim_info(cellid_and_stim_strs[0], df_sort)
    cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Foreground{folder_ids[1]}/*.wav'))
    bg_path, fg_path = bg_dir[int(BG[:2]) - 1], fg_dir[int(FG[:2]) - 1]

    # Plot the spectrograms above the respective traces, it'll look cool
    sfs, W = wavfile.read(bg_path)
    specBG = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    sfs, W = wavfile.read(fg_path)
    specFG = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    specCombo = specBG + specFG
    labls = [f'BG: {BG[2:]}', f'FG: {FG[2:]}', f'Combo: {BG[2:]} + {FG[2:]}']
    lblor = ['low', 'high']

    ord = 0
    for odr in lblor:
        axes[ord].imshow(specBG, aspect='auto', origin='lower', extent=[0, specBG.shape[1], 0, specBG.shape[0]])
        if ord == 0:
            axes[ord].set_ylabel(f"Sort Order: {odr}", rotation=0, weight='bold', labelpad=60)
        for nn, ll in enumerate(labls):
            axes[nn+ord].spines['top'].set_visible(False), axes[nn+ord].spines['bottom'].set_visible(False)
            axes[nn+ord].spines['left'].set_visible(False), axes[nn+ord].spines['right'].set_visible(False)
            axes[nn+ord].set_yticks([]), axes[nn+ord].set_xticks([])
            axes[nn+ord].set_title(ll, weight='bold')
        ord += 1
        axes[ord].imshow(specFG, aspect='auto', origin='lower', extent=[0, specFG.shape[1], 0, specFG.shape[0]])
        ord += 1
        axes[ord].imshow(specCombo, aspect='auto', origin='lower', extent=[0, specCombo.shape[1], 0, specCombo.shape[0]])
        if ord == 5:
            axes[ord].set_ylabel(f"Sort Order: {odr}\nShowing {rows*2}/{len(df_sort)} units",
                                 rotation=0, weight='bold', labelpad=60)
            axes[ord].yaxis.set_label_position("right")
        ord += 1

    # Add the data
    axn = 6
    colors = ['deepskyblue', 'yellowgreen', 'grey']
    df_plot_high, df_plot_low = df_sort[-rows:], df_sort[:rows]
    cass_high, cass_low = cellid_and_stim_strs[-rows:], cellid_and_stim_strs[:rows]
    dfs, casses = [df_plot_low, df_plot_high], [cass_low, cass_high]

    for (df, cass) in zip(dfs, casses):
        for css in cass:
            parms = get_resp_and_stim_info(css, df)
            cellid, animal_id, BG, FG, resp, this_cell_stim, fs, folder_ids = parms.values()
            epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1']
            print(f"Plotting {cellid} which has suppression {this_cell_stim.suppression}.")

            resp = parms['resp'].extract_channels([cellid])
            r = resp.extract_epochs(epochs)

            rBG, rFG = np.nanmean(r[epochs[0]][:, 0, :], axis=0), np.nanmean(r[epochs[1]][:, 0, :], axis=0)
            rlin = rBG + rFG

            prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
            time = (np.arange(0, r[epochs[0]].shape[-1]) / fs) - prestim

            if axn % 2 == 0:
                axes[axn].set_ylabel(f'\n{cellid}\n{sortby}: {np.around(this_cell_stim.suppression,5)}',\
                                     rotation=0, labelpad=60, weight='bold', fontsize = 8)
            else:
                axes[axn+2].set_ylabel(f'\n{cellid}\n{sortby}: {np.around(this_cell_stim.suppression,5)}',\
                     rotation=0, labelpad=60, weight='bold', fontsize=8)
                axes[axn+2].yaxis.set_label_position("right")

            rsps = [np.nanmean(r[ep][:,0,:], axis=0) for ep in epochs]
            rsps.append(rlin)
            ymax, ymin = np.max([np.max(rs) for rs in rsps]) * fs, np.min([np.min(rs) for rs in rsps]) * fs
            axes[axn+1].set_yticks([]), axes[axn+2].set_yticks([])

            for e, c in zip(epochs, colors):
                axes[axn].plot(time, sf.gaussian_filter1d(np.nanmean(r[e][:,0,:], axis=0), sigma=sigma)
                     * fs, color=c)
                # ymin, ymax = axes[axn].get_ylim()
                axes[axn].set_ylim([ymin, ymax])
                axes[axn].vlines([0, 1.0], ymin, ymax, color='black', lw=0.75, ls='--')
                axes[axn].vlines(0.5, ymin, ymax, color='black', lw=0.75, ls=':')
                if e == epochs[-1]:
                    axes[axn].plot(time, sf.gaussian_filter1d(rlin, sigma=sigma) * fs, color='grey', ls=':')
                axn += 1
            axn += 3
        axn = 9


def histogram_summary_plot(weight_df, threshold=0.05):
    '''Pretty niche plot that will plot BG+/FG+ histograms and compare BG and FG weights,
    then plot BG+/FG- histogram and BG-/FG+ histogram separate and then compare BG and FG
    again in a bar graph. I guess you could put any thresholded quadrants you want, but the
    default is the only that makes sense. Last figure on APAN/SFN poster.'''
    quad, _ = ohel.quadrants_by_FR(weight_df, threshold=threshold, quad_return=[3, 2, 6])
    quad3, quad2, quad6 = quad.values()

    f = plt.figure(figsize=(15, 7)) # I made the height 5 for NGP retreat poster
    histA = plt.subplot2grid((7, 17), (0, 0), rowspan=5, colspan=3)
    meanA = plt.subplot2grid((7, 17), (0, 4), rowspan=5, colspan=2)
    histB = plt.subplot2grid((7, 17), (0, 8), rowspan=5, colspan=3)
    histC = plt.subplot2grid((7, 17), (0, 11), rowspan=5, colspan=3)
    meanB = plt.subplot2grid((7, 17), (0, 15), rowspan=5, colspan=2)
    ax = [histA, meanA, histB, histC, meanB]

    edges = np.arange(-1, 2, .05)
    na, xa = np.histogram(quad3.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(quad3.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[0].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
    ax[0].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
    ax[0].legend(('Background', 'Foreground'), fontsize=7)
    ax[0].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=12)
    ax[0].set_title(f"Respond to both\nBG and FG alone", fontweight='bold', fontsize=15)
    ax[0].set_xlabel("Mean Weight", fontweight='bold', fontsize=12)
    ymin, ymax = ax[0].get_ylim()

    BG1, FG1 = np.mean(quad3.weightsA), np.mean(quad3.weightsB)
    BG1sem, FG1sem = stats.sem(quad3.weightsA), stats.sem(quad3.weightsB)
    ttest1 = stats.ttest_ind(quad3.weightsA, quad3.weightsB)
    ax[1].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
    ax[1].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')
    ax[1].set_ylabel('Mean Weight', fontweight='bold', fontsize=12)
    # ax[1].set_ylim(0, 0.79)
    if ttest1.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"{ttest1.pvalue:.3f}"
    ax[1].set_title(title, fontsize=8)

    BG2, FG2 = np.mean(quad6.weightsA), np.mean(quad2.weightsB)
    BG2sem, FG2sem = stats.sem(quad6.weightsA), stats.sem(quad2.weightsB)
    ttest2 = stats.ttest_ind(quad6.weightsA, quad2.weightsB)
    ax[4].bar("BG", BG2, yerr=BG2sem, color='deepskyblue')
    ax[4].bar("FG", FG2, yerr=FG2sem, color='yellowgreen')
    ax[4].set_ylabel("Weight", fontweight='bold', fontsize=12)
    # ax[4].set_ylim(0, 0.79)
    if ttest2.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"{ttest2.pvalue:.3f}"
    ax[4].set_title(title, fontsize=8)
    mean_big = np.max([BG1, FG1, BG2, FG2])
    ax[1].set_ylim(0, mean_big+(mean_big*0.1))
    ax[4].set_ylim(0, mean_big+(mean_big*0.1))

    na, xa = np.histogram(quad6.weightsA, bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(quad2.weightsB, bins=edges)
    nb = nb / nb.sum() * 100
    ax[2].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
    ax[3].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
    ax[2].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=12)
    ax[2].set_title(f"Respond to BG\nalone only", fontweight='bold', fontsize=15)
    ax[3].set_title(f"Respond to FG\nalone only", fontweight='bold', fontsize=15)
    ax[2].set_xlabel("Weight", fontweight='bold', fontsize=12)
    ax[3].set_xlabel("Weight", fontweight='bold', fontsize=12)
    biggest = np.max([na,nb])
    # ax[2].set_ylim(ymin, ymax), ax[3].set_ylim(ymin, ymax)
    ax[2].set_ylim(ymin, biggest), ax[3].set_ylim(ymin, biggest)
    ax[3].set_yticks([])

    return ttest1, ttest2, [quad3.shape[0], quad2.shape[0], quad6.shape[0]]


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

def histogram_filter_by_r(weight_df, r_threshold=0.75, tags=None,
                          cell_threshold=0.05, quad_return=[3,2,6]):
    df_dict, _ = quadrants_by_FRns(weight_df, threshold=cell_threshold, quad_return=quad_return)
    if 0 > r_threshold > 1:
        raise ValueError(f"Threshold {r_threshold} given for r_threshold must be between 0 and 1.")
    if not tags:
        tags = [ta for ta in df_dict.keys()]
    dfs = [qu for qu in df_dict.values()]
    fig, axes = plt.subplots(1, len(dfs), sharex=True, sharey=True, figsize=(8,len(dfs)+1))
    ax = axes.ravel()
    for aa, tt, qq in zip(axes, tags, dfs):
        edges = np.arange(0, 1, .025)
        na, xa = np.histogram(qq.r, bins=edges)
        na = na / na.sum() * 100
        aa.hist(xa[:-1], xa, weights=na, histtype='step', color='black')
        big, small = len(qq), len(qq.loc[qq.r<r_threshold])
        aa.set_title(f"{tt}\n{small}/{big}, {int((small/big)*100)}%", fontweight='bold', fontsize=16)
        ymin, ymax = aa.get_ylim()
        aa.vlines(r_threshold, ymin, ymax, color='black', lw=0.75, ls='--')
        aa.set_ylim(ymin,ymax)
        xmin, xmax = aa.get_ylim()
        aa.fill_between([0, r_threshold], ymin, ymax, color='black', alpha=0.1)
        aa.set_xlim()

    ax[0].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=12)
    ax[1].set_xlabel('r-value', fontweight='bold', fontsize=12)
    fig.subplots_adjust(top=0.85)


def save_psths_bgfg_sort(weight_df):
    '''This is what I used to save PSTHs to /OLP PSTHs/ which was sorted by BG or FG sound'''
    kws = ['Chimes', 'Gravel', 'Insect', 'Rain', 'Rock', 'Stream', 'Thunder', 'Waterfall', 'Waves', 'Wind', 'Alarm',
           'Chirp', 'Shrill', 'Phee', 'Seep', 'Trill', 'TwitterA', 'TwitterB', 'Ek']
    for kw in kws:
        df_filtered, plotids, fnargs = get_keyword_sound_type(kw, weight_df=weight_df,
                                                              single=True)
        cellid_and_stim_strs = [index[0] + ':' + nameA + ':' + nameB for index, nameA, nameB in \
                                zip(df_filtered.index.values,
                                    df_filtered['namesA'], df_filtered['namesB'])]

        for css in cellid_and_stim_strs:
            psth_responses_by_kw(css, df_filtered, kw, plotids['sound_type'], sigma=2, save=True)


# Interactive plotting helpers
def generate_interactive_plot(df, xcolumn='bg_FR', ycolumn='fg_FR',
                              threshold=0.05, sigma=None, sort=True):
    '''Little wrapper function of taking a dataframe and making the interactive scatter
    plot. You can define what x and y values should be, but default and what I'd probably
    always use is FR. Sigma of 2 seems to be nice, but do you. Made and put here 2022_08_31.'''
    plotids, df_filtered, fnargs = {'xcol': xcolumn, 'ycol': ycolumn,
                                    'fn':interactive_plot_click_psth_heatmap}, \
                                   df.copy(), {'df_filtered': df, 'sigma':sigma, 'sort':sort}
    cellid_and_stim_strs= [index+':'+nameA+':'+nameB for index,nameA,nameB in \
                          zip(df_filtered.cellid.values,
                              df_filtered['BG'],df_filtered['FG'])]
    f, ax = plt.subplots(1,1)
    phi=interactive_scatter(df_filtered[plotids['xcol']].values,
                             df_filtered[plotids['ycol']].values,
                             cellid_and_stim_strs, plotids,
                             ax=ax, fn=plotids['fn'], thresh=threshold, fnargs=fnargs)


def interactive_plot_click_psth_heatmap(cellid_and_stim_str,
                                        df_filtered=weight_df, sigma=None, sort=True):
    '''The business of the interactive scatter'''

    # Kind of old way of getting cell parameters, but it works. I'd prefer to nixx the
    # cellid_and_stim_str thing.
    parms = get_resp_and_stim_info(cellid_and_stim_str, df_filtered)
    cellid, animal_id, BG, FG, response, this_cell_stim, fs, folder_ids, pre, dur = parms.values()

    weightBG, weightFG = this_cell_stim['weightsA'], this_cell_stim['weightsB']
    epo = this_cell_stim.epoch
    epochs = [f"STIM_{epo.split('_')[1]}_null", f"STIM_null_{epo.split('_')[2]}", epo]

    resp = response.extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    # Make the husk of this plot
    fig, axes = plt.subplots(figsize=(15, 9))
    psth = plt.subplot2grid((11, 16), (0, 0), rowspan=4, colspan=8)
    specA = plt.subplot2grid((11, 16), (5, 0), rowspan=2, colspan=8)
    specB = plt.subplot2grid((11, 16), (8, 0), rowspan=2, colspan=8)
    BGheat = plt.subplot2grid((11, 16), (0, 10), rowspan=2, colspan=5)
    FGheat = plt.subplot2grid((11, 16), (3, 10), rowspan=2, colspan=5)
    combheat = plt.subplot2grid((11, 16), (6, 10), rowspan=2, colspan=5)
    diffheat = plt.subplot2grid((11, 16), (9, 10), rowspan=2, colspan=5)
    cbar_main = plt.subplot2grid((11, 16), (3, 15), rowspan=2, colspan=1)
    cbar_diff = plt.subplot2grid((11, 16), (9, 15), rowspan=2, colspan=1)
    ax = [psth, specA, specB, BGheat, FGheat, combheat, diffheat, cbar_main, cbar_diff]

    # Gather some helpful plotting things including responses
    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs ) - prestim

    bg_alone, fg_alone, combo = epochs[0], epochs[1], epochs[2]
    r_mean = {e:np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
    r_mean['Weight Model'] = (r_mean[bg_alone] * weightBG) + (r_mean[fg_alone] * weightFG)
    r_mean['Linear Sum'] = r_mean[bg_alone] + r_mean[fg_alone]

    colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray', 'black']
    styles = ['-', '-', '-', '-', ':']

    # Plotting the psth
    for e, c, s in zip(r_mean.keys(), colors, styles):
        ax[0].plot(time, sf.gaussian_filter1d(r_mean[e], sigma=1)
                    * fs, color=c, linestyle=s, label=e)
    ax[0].legend((f'BG, weight={np.around(this_cell_stim.weightsA, 2)}, FR={np.around(this_cell_stim.bg_FR, 3)}',
                  f'FG, weight={np.around(this_cell_stim.weightsB, 2)}, FR={np.around(this_cell_stim.fg_FR, 3)}',
                  'Both',
                  'Weight Model, r={:.2f}'.format(this_cell_stim.r),
                  'Linear Sum'))
    ax[0].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['supp']}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymax*0.5, ymax, color='black', lw=0.75, ls=':')
    ax[0].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))

    # Getting spectrograms to plot BG and FG
    folder_ids = parms['folder_ids']
    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{folder_ids[0]}/*.wav'))
    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Foreground{folder_ids[1]}/*.wav'))
    bg_path = [bb for bb in bg_dir if epo.split('_')[1].split('-')[0][:2] in bb][0]
    fg_path = [ff for ff in fg_dir if epo.split('_')[2].split('-')[0][:2] in ff][0]

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)
    ax[1].set_ylabel(f"BG: {BG}", fontweight='bold', fontsize=8)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([0, 20, 40, 60, 80]), ax[2].set_yticks([])
    ax[2].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)
    ax[2].set_xlabel('Time (s)', fontweight='bold', fontsize=8)
    ax[2].set_ylabel(f"FG: {FG}", fontweight='bold', fontsize=8)

    # Get all other cells from this site
    site = cellid[:7]
    this_site = df_filtered.loc[(df_filtered['cellid'].str.contains(site)) &
                                (df_filtered.epoch == epo)]
    site_ids = this_site.cellid.tolist()
    num_ids = [cc[8:] for cc in site_ids]

    # Normalize response and extract the BG_null, null_FG, and combo epochs
    norm_spont, SR, STD = ohel.remove_spont_rate_std(response)
    epor = norm_spont.extract_epochs(epochs)
    resp_plot = np.stack([np.nanmean(aa, axis=0) for aa in list(epor.values())])

    # Normalize response to max response for each unit, across all three stim
    for nn in range(resp_plot.shape[1]):
        # max_val = np.max(np.abs(resp_plot[:,nn,int(prestim*fs):int((prestim+dur)*fs)]))
        max_val = np.max(np.abs(resp_plot[:,nn,:]))
        resp_plot[:,nn,:] = resp_plot[:,nn,:] / max_val

    # Get difference array before smoothing
    ls_array = resp_plot[0,:,:] + resp_plot[1,:,:]
    # diff_array = resp_plot[2,:,:] - resp_plot[1,:,:]
    diff_array = resp_plot[2,:,:] - ls_array

    if sort == True:
        sort_array = diff_array[:, int(prestim*fs):int((prestim+dur)*fs)]
        means = list(np.nanmean(sort_array, axis=1))
        indexes = list(range(len(means)))
        sort_df = pd.DataFrame(list(zip(means, indexes)), columns=['mean', 'idx'])
        sort_df = sort_df.sort_values('mean', ascending=False)
        sort_list = sort_df.idx
        diff_array = diff_array[sort_list, :]
        resp_plot = resp_plot[:, sort_list, :]
        num_array = np.asarray(num_ids)
        num_ids = list(num_array[sort_list])
        print(num_ids)

    # Smooth if you have given it a sigma by which to smooth
    if sigma:
        resp_plot = sf.gaussian_filter1d(resp_plot, sigma, axis=2)
        diff_array = sf.gaussian_filter1d(diff_array, sigma, axis=1)
    # Get the min and max of the array, find the biggest magnitude and set max and min
    # to the abs and -abs of that so that the colormap is centered at zero
    cmax, cmin = np.max(resp_plot), np.min(resp_plot)
    biggest = np.maximum(np.abs(cmax),np.abs(cmin))
    cmax, cmin = np.abs(biggest), -np.abs(biggest)

    # Plot BG, FG, Combo
    for (ww, qq) in enumerate(range(3,6)):
        dd = ax[qq].imshow(resp_plot[ww, :, :], vmin=cmin, vmax=cmax,
                           cmap='bwr', aspect='auto', origin='lower',
                           extent=[time[0], time[-1], 0, len(this_site)])
        ax[qq].vlines([int(pre), int(pre+dur)], ymin=0, ymax=len(this_site),
                      color='black', lw=1, ls=':')
        ax[qq].set_ylabel('Unit', fontweight='bold', fontsize=8)
        ax[qq].set_yticks([*range(0, len(this_site))])
        ax[qq].set_yticklabels(num_ids, fontsize=5)
        ax[qq].set_title(f"{epochs[ww]}", fontsize=8, fontweight='bold')
    ax[5].set_xlabel('Time (s)', fontweight='bold', fontsize=8)
    # Add the colorbar to the axis to the right of these, the diff will get separate cbar
    fig.colorbar(dd, ax=ax[7])
    ax[7].spines['top'].set_visible(False), ax[7].spines['right'].set_visible(False)
    ax[7].spines['bottom'].set_visible(False), ax[7].spines['left'].set_visible(False)
    ax[7].set_yticks([]), ax[7].set_xticks([])

    # Plot the difference heatmap with its own colorbar
    dmax, dmin = np.max(diff_array), np.min(diff_array)
    biggestd = np.maximum(np.abs(dmax),np.abs(dmin))
    dmax, dmin = np.abs(biggestd), -np.abs(biggestd)
    ddd = ax[6].imshow(diff_array, vmin=dmin, vmax=dmax,
                           cmap='PiYG', aspect='auto', origin='lower',
                           extent=[time[0], time[-1], 0, len(this_site)])
    ax[6].vlines([0, int(dur)], ymin=0, ymax=len(this_site),
                 color='black', lw=1, ls=':')
    ax[6].set_yticks([*range(0, len(this_site))])
    ax[6].set_yticklabels(num_ids, fontsize=5)
    ax[6].set_title(f"Difference (Combo - FG)", fontsize=8, fontweight='bold')
    ax[6].set_xlabel('Time (s)', fontsize=8, fontweight='bold')

    fig.colorbar(ddd, ax=ax[8])
    ax[8].spines['top'].set_visible(False), ax[8].spines['right'].set_visible(False)
    ax[8].spines['bottom'].set_visible(False), ax[8].spines['left'].set_visible(False)
    ax[8].set_yticks([]), ax[8].set_xticks([])


def get_resp_and_stim_info(cellid_and_stim_str, df_filtered):
    '''Finished and moved here from OLP_fitting_Greg on 2022_08_31. Kind of an antiquated
    way of getting cell info using this cellid_and_stim_str format of loading. It's fine.'''
    cellid, stimA, stimB = cellid_and_stim_str.split(':')
    cell_df = df_filtered.loc[df_filtered.cellid == cellid]
    this_cell_stim = cell_df[(cell_df['BG'] == stimA) & (cell_df['FG'] == stimB)].iloc[0]
    animal_id = cellid[:3]

    if animal_id == 'HOD' or animal_id == 'TBR':
        batch = 333
    elif animal_id == 'ARM':
        # got to code in for batch to differentiate between A1 and PEG batches,
        # where can I get that info above?
        batch = 0
    elif animal_id == 'CLT':
        batch = 340
    else:
        raise ValueError(f"You need to add {animal_id} to the list in get_resp_and_stim_info.")

    fs=100
    expt = BAPHYExperiment(cellid=cellid, batch=batch)
    rec = expt.get_recording(rasterfs=fs, resp=True, stim=False)
    expt_params = expt.get_baphy_exptparams()  # Using Charlie's manager
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    resp = rec['resp'].rasterize()

    BG, FG = stimA.split('-')[0], stimB.split('-')[0]

    #parts for spectrograms now
    if animal_id == 'HOD':
        folder_ids = [1,1]
    elif animal_id == 'TBR':
        folder_ids = [2,3]
    elif animal_id == 'ARM':
        folder_ids = [1,2]
    elif animal_id == 'CLT':
        folder_ids = [2,3]
    else:
        folder_ids = [2,3] # Any other animals for the foreseeable future should use this

    parms = {'cellid': cellid, 'animal_id': animal_id, 'BG': BG, 'FG': FG, 'resp': resp,
             'this_cell_stim': this_cell_stim, 'fs': fs, 'folder_ids': folder_ids,
             'prestim': ref_handle['PreStimSilence'], 'stim_dur': ref_handle['Duration']}

    return parms


def interactive_scatter(x, y, names, ids, ax=None, fn=None, fnargs={}, dv=None,
                      thresh=None, color='animal', **kwargs):
    '''Most of the guts here are taken from Luke's function about interactive plotting.
    I really don't know what everything does exactly but it points to my above function
    when you click on a scatter point. Moved from OLP_fitting_Greg 2022_08_31.'''
    if ax is None:
        ax = plt.gca()
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = 'none'
    if 'sound_type' in ids:
        kwargs['sound_type'] = ids['sound_type']
    good_inds = np.where(np.isfinite(x + y))[0]
    x = x[good_inds]
    y = y[good_inds]
    names = [names[g] for g in good_inds]
    if type(fn) is list:
        for i in range(len(fnargs)):
            if 'pth' in fnargs[i].keys():
                fnargs[i]['pth'] = [fnargs[i]['pth'][gi] for gi in good_inds]
    plot_df = pd.DataFrame(data=list(zip(x, y, [n[:3] for n in names])), index=names,
                           columns=[ids['xcol'], ids['ycol'], 'animal'])
    if color == 'FG':
        color = [aa.split(':')[2].split('-')[0][2:].replace('_', '') for aa in plot_df.index]
    # plot_df = pd.DataFrame(data=list(zip(x, y, [n[:3] for n in names])), index=names,
    #                        columns=['weightBG', 'weightFG', 'animal'])
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=ids['xcol'], y=ids['ycol'], data=plot_df, hue=color, picker=5, ax=ax)
    # ax = sns.scatterplot(x='weightBG', y='weightFG', data=plot_df, hue='animal', picker=5, ax =ax)
    if thresh:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.vlines([thresh, -thresh], ymin, ymax, color='black', lw=0.5)
        ax.hlines([thresh, -thresh], xmin, xmax, color='black', lw=0.5)

    art = ax.collections[0]
    if 'keyword' in ids:
        ax.set_title(f"{ids['keyword']}: {ids['sound_type']}")

    # art, = ax.plot(x, y, picker=5, **kwargs)
    # art=ax.scatter(x,y,picker=5)

    def onpick(event):
        if event.artist == art:
            # ind = good_inds[event.ind[0]]
            ind = event.ind[0]
            print('onpick scatter: {}: {} ({},{})'.format(ind, names[ind], np.take(x, ind), np.take(y, ind)))
            if dv is not None:
                dv[0] = names[ind]
            if fn is None:
                print('fn is none?')
            elif type(fn) is list:
                for fni, fna in zip(fn, fnargs):
                    fni(names[ind], **fna)
                    # fni(names[ind],**fna,ind=ind)
            else:
                fn(names[ind], **fnargs)

    def on_plot_hover(event):
        for curve in ax.get_lines():
            if curve.contains(event)[0]:
                print('over {0}'.format(curve.get_gid()))

    ax.figure.canvas.mpl_connect('pick_event', onpick)
    return art
