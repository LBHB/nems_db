import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from nems0.analysis.gammatone.gtgram import gtgram
import pandas as pd
import nems_lbhb.projects.olp.OLP_helpers as ohel
from pydub import AudioSegment
from pydub.playback import play
import copy
import joblib as jl
import nems_lbhb.projects.olp.OLP_figures as ofig

# filename = 'full_stats_df_v1'
# jl.dump(sound_dff, f"/auto/data/sounds/BigNat/v1/saved_dfs/{filename}")

path = '/auto/data/sounds/BigNat/v1/saved_dfs/full_df_specs_v1'      # Full save, has waveforms/specs, too big
path = '/auto/data/sounds/BigNat/v1//saved_dfs/full_stats_df_v1'     # Use me, everything but got rid of wave/spec
sound_df = jl.load(path)



xcol = 'Tstationary'
ycol = 'Fcorr'


plot_stat_scatter_marginals(sound_df, xcol='Tstationary', ycol='Fcorr')
plot_stat_scatter_marginals(sound_df, xcol='Fcorr', ycol='bandwidth')
plot_stat_scatter_marginals(sound_df, xcol='Tstationary', ycol='bandwidth')

cols = ['Tstationary', 'Fcorr', 'bandwidth']
spans = {}
for col in cols:
    span = (sound_df[col].min(), sound_df[col].max())
    distance = span[1] - span[0]
    blocks = np.linspace(span[0], span[1], 5)

    spans[col] = blocks

ts = [dd if nn != len(list(spans[cols[0]])) for dd in list(spans[cols[0]])]


def plot_stat_scatter_marginals(dff, xcol='Tstationary', ycol='Fcorr'):
    col_names = {'Tstationary': 'Temporal\nvariance',
                 'Fcorr': 'Spectral\ncorrelation',
                 'bandwidth': 'Bandwidth'}

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    main = plt.subplot2grid((10, 10), (3, 0), rowspan=7, colspan=7)
    xx = plt.subplot2grid((10, 10), (0, 0), rowspan=2, colspan=7, sharex=main)
    yy = plt.subplot2grid((10, 10), (3, 8), rowspan=7, colspan=2, sharey=main)
    ax = [xx, yy, main]

    ax[-1].scatter(dff[xcol], dff[ycol], s=2, color='black')
    ax[-1].set_xlabel(col_names[xcol], fontsize=10, fontweight='bold')
    ax[-1].set_ylabel(col_names[ycol], fontsize=10, fontweight='bold')

    for aa, col in enumerate([xcol, ycol]):
        span = (dff[col].min(), dff[col].max())
        distance = span[1] - span[0]
        if col != 'bandwidth':
            edges = np.arange(span[0], span[1], distance / 50)
        else:
            edges = np.arange(span[0], span[1], distance / 20)

        na, xa = np.histogram(dff[col], bins=edges)
        na = na / na.sum() * 100
        median = np.median(dff[col])
        median, se = ofig.jack_mean_err(dff[col], do_median=True)

        if aa==0:
            ax[aa].hist(xa[:-1], xa, weights=na, histtype='step', color='purple', linewidth=2, orientation='vertical')
            lil, big = ax[aa].get_ylim()
            ax[aa].vlines(median, ymin=lil, ymax=big, ls=':', color='black', lw=1)
            lill, bigg = ax[-1].get_ylim()
            ax[-1].vlines(median, ymin=lill, ymax=bigg, ls=':', color='black', lw=1)
        if aa==1:
            ax[aa].hist(xa[:-1], xa, weights=na, histtype='step', color='purple', linewidth=2, orientation='horizontal')
            lil, big = ax[aa].get_xlim()
            ax[aa].hlines(median, xmin=lil, xmax=big, ls=':', color='black', lw=1)
            lill, bigg = ax[-1].get_xlim()
            ax[-1].hlines(median, xmin=lill, xmax=bigg, ls=':', color='black', lw=1)
        # ax[aa].set_xlabel(col_names[col], fontsize=10, fontweight='bold')
        ax[aa].set_title(f"Median: {np.around(median, 2)} se {np.around(se, 2)}")
        # meds[col_names[aa]] = median




cols = ['Tstationary', 'Fcorr', 'bandwidth']
col_names = ['Temporal\nvariance', 'Spectral\ncorrelation', 'Bandwidth']
fig, ax = plt.subplots(1, 3, figsize=(10,4))

meds = {}
for aa, col in enumerate(cols):
    edges = np.arange(-1, 2, .05)

    span = (dff[col].min(), dff[col].max())
    distance = span[1]-span[0]
    if col != 'bandwidth':
        edges = np.arange(span[0], span[1], distance/50)
    else:
        edges = np.arange(span[0], span[1], distance / 20)

    na, xa = np.histogram(dff[col], bins=edges)
    na = na / na.sum() * 100
    median = np.median(dff[col])
    median, se = ofig.jack_mean_err(dff[col], do_median=True)

    ax[aa].hist(xa[:-1], xa, weights=na, histtype='step', color='purple', linewidth=2, orientation='vertical')
    lil, big = ax[aa].get_ylim()
    ax[aa].vlines(median, ymin=lil, ymax=big, ls=':', color='black', lw=1)
    ax[aa].set_xlabel(col_names[aa], fontsize=10, fontweight='bold')
    ax[aa].set_title(f"Median: {np.around(median, 2)} se {np.around(se, 2)}")
    meds[col_names[aa]] = median

ax[0].set_ylabel('Percent of cells', fontsize=10, fontweight='bold')
fig.tight_layout()


# ax[axn].set_ylabel('Percentage of cells', fontweight='bold', fontsize=12)
ax[axn].set_title(f"{aaa} - 0-0.5s", fontweight='bold', fontsize=12)
ax[axn].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_xlabel("Mean Weight", fontweight='bold', fontsize=12)
ax[3].set_xlabel("Mean Weight", fontweight='bold', fontsize=12)

generate_interactive_plot(sound_df, xcolumn='Tstationary', ycolumn='Fcorr')

# Interactive plotting helpers
def generate_interactive_plot(df, xcolumn='bg_FR', ycolumn='fg_FR', sort=True):
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




from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def V(x,y,z):
     return np.cos(10*x) + np.cos(10*y) + np.cos(10*z) + 2*(x**2 + y**2 + z**2)

X,Y = np.mgrid[-1:1:100j, -1:1:100j]
Z_vals = [ -0.5, 0, 0.9 ]
num_subplots = len( Z_vals)

fig = plt.figure(figsize=(10, 4))
for i,z in enumerate( Z_vals):
    ax = fig.add_subplot(1 , num_subplots , i+1, projection='3d')
    ax.contour(X, Y, V(X,Y,z), cmap=cm.gnuplot)
    ax.set_title('z = %.2f'%z, fontsize=30)
fig.savefig('contours.png', facecolor='grey', edgecolor='none')









go = False
if go==True:
    # SOUND_ROOT = f"/auto/data/sounds/BigNat/v1/*.wav"
    SOUND_ROOT = f"/auto/data/sounds/BigNat/v1/"
    path_dir = glob.glob((SOUND_ROOT+'*.wav'))

    # Use if starting fresh
    sound_df = get_sound_stats_(path_dir, percent_lims=[15,85], self_name=True, plot=True, save_dir='jpeg', num=0)

    # Use if picking up where you left off
    back_dir = list(glob.glob((SOUND_ROOT+'saves/backup*')))[-1]
    backie = jl.load(back_dir)
    save_num = int(back_dir.split('-')[-1]) + 1
    update_path_dir = path_dir[save_num:]

    sound_df = get_sound_stats_(update_path_dir, percent_lims=[15,85], self_name=True,
                                plot=True, save_dir='jpeg', num=save_num, backup=backie)


def get_sound_stats_(filelist, percent_lims=[15, 85], self_name=False, plot=False, save_dir=None, num=0, backup=None):
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

    if isinstance(backup, pd.DataFrame):
        input_counts = dict(backup['input'].value_counts())
        how_many_total = len(filelist) + len(backup)
    else:
        input_counts = {}
        how_many_total = len(filelist)

    if how_many_total <= num:
        raise ValueError(f"You're done. It's over. The dataframe already has {how_many_total} items.")

    sounds = []
    for path in filelist:
        sound_root = '/'.join(path.split('/')[:-1])+'/'
        short_name = '_'.join(path.split('/')[-1].split('_')[2:]).split('.')[0]
        filename = path.split('/')[-1]

        sfs, W = wavfile.read(path)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        # to measure rms power... for rms-normed signals:
        rms_normed = np.std(ohel.remove_clicks(W / W.std(), 15))
        # for max-normed signals:
        # max_normed = np.std(W / np.abs(W).max()) * 5
        # dev = np.std(spec, axis=1)

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

        #the frequency metric
        cpow = cc[np.triu_indices(cut_spec.shape[0], k=1)].mean()

        #the time metric
        cut_dev = np.std(cut_spec, axis=1)

        freq_range = (int(x_freq[bin_low]), int(x_freq[bin_high]))

        # temp = np.abs(np.fft.fft(spec, axis=1))
        # freq = np.abs(np.fft.fft(spec, axis=0))
        sound_dict = {'name': short_name,
                       'Tstationary': np.nanmean(cut_dev),
                       'Fcorr': cpow,
                       'bandwidth': bandwidth,
                       'bw_percent': f'{percent_lims[0]}/{percent_lims[1]}',
                       'freq_range': freq_range,
                       'RMS_power': rms_normed,
                       'spec': spec,
                       'waveform': W,
                       'root': sound_root,
                       'file': filename}

        if self_name==True:
            base_short = ''
            while base_short=='':
                song = AudioSegment.from_wav(path)
                play(song)
                base_short = input(f'Give me a name for {short_name}:')
            if base_short=='again':
                song = AudioSegment.from_wav(path)
                play(song)
                base_short = input(f"Let's try again. Give me a name for {short_name}:")
            if base_short in list(input_counts.keys()):
                input_counts[base_short] += 1
                baby_name = f'{base_short}{input_counts[base_short]}'
                print(f'{base_short} has already appeared {input_counts[base_short]-1}x, so this one is called {baby_name}')
            else:
                baby_name = base_short
                input_counts[base_short] = 1
                print(f'First time {base_short} was used, naming {baby_name}.')

            nickname_dict = {'nickname': baby_name, 'input': base_short}
            sound_dict = nickname_dict | sound_dict
            disp_dict = dict((k, sound_dict[k]) for k in ('nickname', 'Tstationary', 'Fcorr', 'bandwidth'))
            lines = [f'{key}: {value}' if isinstance(value,str) else f'{key}: {np.around(value,2)}' for key, value in disp_dict.items()]
            print(lines)
            # print('\n'.join(lines))

        sounds.append(sound_dict)

        if plot==True:
            fig, axes = plt.subplots(figsize=(18, 7))
            wave = plt.subplot2grid((8, 12), (0, 0), rowspan=2, colspan=8)
            spe = plt.subplot2grid((8, 12), (3, 0), rowspan=4, colspan=8)
            cross = plt.subplot2grid((8, 12), (3, 8), rowspan=4, colspan=4, aspect='equal')
            ax = [wave, spe, cross]

            ax[0].plot(W, color='black')
            ax[0].set_xlim(0,len(W)), ax[0].set_xticks([])
            if 'nickname' in list(sound_dict.keys()):
                ax[0].set_title(f"Name: {sound_dict['nickname']}\nFilename: {filename}", fontweight='bold', fontsize=12)
            else:
                ax[0].set_title(f"Filename: {filename}", fontweight='bold', fontsize=12)

            ax[1].imshow(np.sqrt(spec), aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]],
                 cmap='gray_r')
            ax[1].set_yticks([0, bin_low, bin_high, bins])
            ax[1].set_yticklabels([lfreq, freq_range[0], freq_range[1], hfreq])
            ax[1].hlines([bin_low, bin_high], 0, spec.shape[1], lw=0.5, ls=(0, (5, 10)), color='black')
            ax[1].set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            ax[1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            length = list(np.arange(0, spec.shape[1], 50))
            ax[1].set_xticks(length)
            ax[1].set_xticklabels([aa/100 for aa in length])
            ax[1].set_title(f'Temporal Variance: {np.around(np.nanmean(cut_dev), 2)}\n'
                            f'Bandwidth ({percent_lims[0]}-{percent_lims[1]}%): {np.around(bandwidth, 2)}',
                            fontsize=12, fontweight='bold')

            ax[2].imshow(cc, aspect='equal', origin='lower', extent=[0,cc.shape[1], 0, cc.shape[0]],
                 cmap='inferno')
            ax[2].set_title(f'Spectral\nCorrelation: {np.around(cpow, 3)}', fontsize=12, fontweight='bold')
            ax[2].set_yticks([0, cc.shape[1]]), ax[2].set_yticklabels([freq_range[0], freq_range[1]])
            ax[2].set_xticks([0, cc.shape[1]]), ax[2].set_xticklabels([freq_range[0], freq_range[1]])
            ax[2].set_xlabel('Frequency Bin', fontsize=10, fontweight='bold')
            fig.tight_layout()
            if save_dir:
                plt.savefig(f'{sound_root}/{save_dir}/{filename}.jpeg')
                if self_name==True:
                    plt.savefig(f"{sound_root}/{save_dir}_short/{format(num+1, '03d')}{baby_name}.jpeg")
            plt.close()

        if num in list(np.arange(1,how_many_total, 5)):
            print(f"Saving a backup of what you've done so far up to {num}, {filename}")
            save_df = pd.DataFrame(sounds)
            if isinstance(backup, pd.DataFrame):
                save_df = pd.concat([backup, save_df], axis=0).reset_index(drop=True)
                print("Merging dataframes.")
            jl.dump(save_df, f"{sound_root}/saves/backup-num-{num}")

        num+=1

    dff = pd.DataFrame(sounds)
    if isinstance(backup, pd.DataFrame):
        dff = pd.concat([backup, dff], axis=0).reset_index(drop=True)
    return dff




