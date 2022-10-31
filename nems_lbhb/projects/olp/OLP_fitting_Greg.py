import nems_lbhb.TwoStim_helpers as ts
import nems_lbhb.projects.olp.OLP_plot_helpers as opl
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
# OLP_cell_metrics_db_path='/auto/users/luke/Projects/OLP/NEMS/celldat_A1_v1.h5'

fit = False
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_add_spont.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_resp.h5'

batch, fs, paths = 333, 100, None
if batch == 328:
    titles = 'A1'
elif batch == 329:
    titles = 'PEG'
elif batch == 333:
    titles = 'HOD + TBR'

batch = 339 #Binaural ferret OLP
batch = 340 #All ferret OLP

# PSTH metrics that have to do with one stimulus at a time
if fit == True:
    cell_df = nd.get_batch_cells(batch)
    cell_list = cell_df['cellid'].tolist()
    for i in range(len(cell_list)):
        if cell_list[i] == 'TBR025a-21-2':
            cell_list[i] = 'TBR025a-21-1'
        elif cell_list[i] == 'TBR025a-60-2':
            cell_list[i] = 'TBR025a-60-1'

    metrics=[]
    for cellid in cell_list:
        metrics_ = ts.calc_psth_metrics(batch, cellid)
        print('****rAAm: {} rBBm: {}'.format(metrics_['rAAm'], metrics_['rBBm']))
        metrics.append(metrics_)

    df=pd.DataFrame(data=metrics)
    df['modelspecname']='dlog_fir2x15_lvl1_dexp1'
    df['cellid']=cell_list
    df = df.set_index('cellid')

    df = df.apply(ts.type_by_psth, axis=1);
    df['batch']=batch

    df = df.apply(ts.calc_psth_weight_resp, axis=1, fs=fs)

    os.makedirs(os.path.dirname(OLP_cell_metrics_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df_store=copy.deepcopy(df)
    def drop_get_error(row):
        row['weight_dfR'] = row['weight_dfR'].copy().drop(columns='get_error')
        return row
    df_store=df_store.apply(drop_get_error,axis=1)
    df_store=df_store.drop(columns=['get_nrmseR'])
    cols_to_keep = df_store[['weight_dfR', 'pair_names', 'suppression', 'FR', 'animal',
        'modelspecname', 'corcoef', 'avg_resp', 'snr', 'batch', 'weightsR', 'EfitR',
        'nMSER', 'nfR', 'rR', 'namesR', 'namesAR', 'namesBR', 'WeightAgroupsR',
        'WeightBgroupsR', 'norm_spont', 'spont_rate', 'rec']]
    store['df'] = cols_to_keep.copy()
    store.close()
else:
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df=store['df']
    store.close()




# filter dataframe by metrics
# df = df.loc[(df['corcoef'] >= 0.2) & (df['avg_resp'] >= 0.025)]
df = df.loc[(df['corcoef'] >= 0.2)]
weight_df = opl.df_to_weight_df(df)

# Filtered big df by certain epochs for sounds used with Hood (if [1,1])
weight_df = opl.filter_epochs_by_file_names([1,1], weight_df=weight_df)


#Get quadrant filter on the dataframe of weights
quad, threshold = opl.quadrants_by_FRns(weight_df, threshold=0.05, quad_return=[3,2,6])
quad, threshold = opl.quadrants_by_FRns(weight_df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])




#makes scatter
fig, ax = plt.subplots()
ax.scatter(weight_df['BG_FRns'], weight_df['FG_FRns'], s=2)
ymin,ymax = ax.get_ylim()
xmin,xmax = ax.get_xlim()
ax.vlines([threshold, -threshold], ymin, ymax, ls='-', color='black', lw=0.5)
ax.hlines([threshold, -threshold], xmin, xmax, ls='-', color='black', lw=0.5)
ax.set_ylim(-0.2,0.4)
ax.set_xlim(-0.2,0.4)
ax.set_xlabel('BG Alone FR', fontweight='bold', fontsize=7)
ax.set_ylabel('FG Alone FR', fontweight='bold', fontsize=7)
ax.set_aspect('equal')

#makes scatter of each animal colored with labels
tbr = weight_df.loc[weight_df.Animal=='TBR']
hod = weight_df.loc[weight_df.Animal=='HOD']

fig, ax = plt.subplots()
ax.scatter(tbr['BG_FRns'], tbr['FG_FRns'], s=1, label=f'Tabor\nn={len(tbr)}', color='purple')
ax.scatter(hod['BG_FRns'], hod['FG_FRns'], s=1, label=f'Hood\nn={len(hod)}', color='orange')
ax.legend(loc='upper left', fontsize=8)
ymin,ymax = ax.get_ylim()
xmin,xmax = ax.get_xlim()
ax.vlines([threshold, -threshold], ymin, ymax, ls='-', color='black', lw=0.5)
ax.hlines([threshold, -threshold], xmin, xmax, ls='-', color='black', lw=0.5)
ax.set_ylim(-0.2,0.4)
ax.set_xlim(-0.2,0.4)
ax.set_xlabel('BG Alone FR', fontweight='bold', fontsize=9)
ax.set_ylabel('FG Alone FR', fontweight='bold', fontsize=9)
ax.set_aspect('equal')



#Use if quad_return == 1
opl.weight_hist(quad, tag=None, y='percent')

#Use if quad_return > 1
opl.histogram_subplot_handler(quad, yax='percent', tags=['BG+ / FG+', 'BG+ / FG-', 'BG- / FG+'])

#Very specific last figure from APAN/SFN poster
ttest1, ttest2 = opl.histogram_summary_plot(weight_df)

#histograms of r values filtered below a r value
opl.histogram_filter_by_r(weight_df, r_threshold=0.75, tags=None,
                          cell_threshold=0.05, quad_return=[3,2,6])


threshold = 0.05
#make FR scatter and filter weight_df by a particular sound
kw = 'TwitterB'
df_filtered, plotids, fnargs = opl.get_keyword_sound_type(kw, weight_df=weight_df,
                                                          scat_type='suppression',
                                                          single=True, exact=True)

plotids, df_filtered, fnargs = {'xcol': 'BG_FRns', 'ycol': 'FG_FRns', 'fn':opl.plot_psth_scatter}, \
                               df_filtered.copy(), {'df_filtered': df_filtered, 'scatter': 'suppression'}


#Two keywords to get only a single sound pair across all units
kw = 'Chimes'
kw2 = 'Chirp'
df_filtered, plotids, fnargs = opl.get_keyword_sound_type(kw, weight_df=weight_df,
                                                          scat_type='suppression',
                                                          single=True, exact=False, kw2=kw2)

#filter df by a list of keywords, regardless of type
kws = ['TwitterB', 'Ek', 'Chirp']
df_filtered, plotids, fnargs = opl.get_keyword_list(kws, weight_df=weight_df,
                                                          scat_type='suppression',
                                                          single=True)

kws = ['TwitterB', 'Ek', 'Chirp', 'Tsik', 'Alarm']
df_filtered, plotids, fnargs = opl.get_keyword_list(kws, weight_df=weight_df,
                                                          scat_type='suppression',
                                                          single=True)

quad, threshold = opl.quadrants_by_FRns(df_filtered, threshold=0.05, quad_return=[2,3,6])

quad3 = quad[3]
dft = quad3.loc[quad3.Animal=='TBR']
dfh = quad3.loc[quad3.Animal=='HOD']
fig,ax =plt.subplots(1, 2, sharex=True,sharey=True)
ax[0].scatter(dfh.weightsA, dfh.weightsB)
ax[1].scatter(dft.weightsA, dft.weightsB)
ax[0].set_title('Hood')
ax[1].set_title('Tabor')
ax[0].set_xlabel('BG Weight')
ax[0].set_ylabel('FG Weight')


plotids, df_filtered, fnargs = {'xcol': 'BG_FRns', 'ycol': 'FG_FRns', 'fn':opl.plot_psth_scatter}, \
                               df_filtered.copy(), {'df_filtered': df_filtered, 'scatter': 'suppression'}
threshold=0.05

plotids, df_filtered, fnargs = {'xcol': 'weightsA', 'ycol': 'weightsB', 'fn':opl.plot_psth_scatter}, \
                               df_filtered.copy(), {'df_filtered': df_filtered, 'scatter': 'suppression'}

weightthresh = 3
weight_df = df_filtered
weight_df = weight_df.loc[(weight_df.weightsB>=-weightthresh) & (weight_df.weightsB<=weightthresh)]
weight_df = weight_df.loc[(weight_df.weightsA>=-weightthresh) & (weight_df.weightsA<=weightthresh)]
df_filtered = weight_df

anm = 'HOD'
df_filtered = df_filtered.loc[df_filtered.Animal==anm]
# Run this once you have plotids, df_filtered, fnargs defined based on what you want to plot
cellid_and_stim_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(df_filtered.index.values,
                          df_filtered['namesA'],df_filtered['namesB'])]

# Makes interactive plot, works with standard scatter or single keyword, untested with dual keyword
f, ax = plt.subplots(1,1)
phi=ts.scatterplot_print(df_filtered[plotids['xcol']].values,
                         df_filtered[plotids['ycol']].values,
                         cellid_and_stim_strs, plotids,
                         ax=ax,fn=plotids['fn'], thresh=threshold, color='namesB', fnargs=fnargs)
ax.set_title(f"{titles}")



opl.weight_hist_dual(df_filtered, f"All - {plotids['keyword']}")
opl.weight_hist(df_filtered, f"All - {plotids['keyword']}")

#Makes a figure of all the PSTHs separated and sorted
opl.split_psth_multiple_units(df_filtered, sortby='suppression', order='low', sigma=2)

opl.split_psth_highest_lowest(df_filtered, sortby='suppression', rows=18, sigma=2)


#If you just want standard scatter of BG weight v FG weight
plotids, df_filtered, fnargs = {'xcol': 'weightsA', 'ycol': 'weightsB', 'fn':opl.plot_psth}, \
                               weight_df.copy(), {'df_filtered': weight_df}

#Adding 2022_08_29 for new weight_df names
weight_df = weight_df.rename(columns={'BG':'namesA', 'FG':'namesB', 'bg_FR':'BG_FRns',
                              'fg_FR':'FG_FRns'})
plotids, df_filtered, fnargs = {'xcol': 'BG_FRns', 'ycol': 'FG_FRns', 'fn':opl.plot_psth_scatter}, \
                               weight_df.copy(), {'df_filtered': weight_df, 'scatter': 'suppression'}
cellid_and_stim_strs= [index+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(df_filtered.cellid.values,
                          df_filtered['namesA'],df_filtered['namesB'])]
f, ax = plt.subplots(1,1)
phi=ts.scatterplot_print(df_filtered[plotids['xcol']].values,
                         df_filtered[plotids['ycol']].values,
                         cellid_and_stim_strs, plotids,
                         ax=ax,fn=plotids['fn'], thresh=threshold, fnargs=fnargs)



#interactive plot using firing rates
plotids, df_filtered, fnargs = {'xcol': 'BG_FRns', 'ycol': 'FG_FRns', 'fn':opl.plot_psth_scatter}, \
                               weight_df.copy(), {'df_filtered': weight_df, 'scatter': 'suppression'}


# Run this once you have plotids, df_filtered, fnargs defined based on what you want to plot
cellid_and_stim_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(df_filtered.index.values,
                          df_filtered['namesA'],df_filtered['namesB'])]

# Makes interactive plot, works with standard scatter or single keyword, untested with dual keyword
f, ax = plt.subplots(1,1)
phi=ts.scatterplot_print(df_filtered[plotids['xcol']].values,
                         df_filtered[plotids['ycol']].values,
                         cellid_and_stim_strs, plotids,
                         ax=ax,fn=plotids['fn'], thresh=threshold, fnargs=fnargs)
ax.set_title(f"{titles}")


#Really simple and early scatters of weight of whatever dataframe you put in
opl.scatter_weights(weight_df, titles)
opl.heatmap_weights(weight_df, titles)



cell, BG, FG = 'TBR012a-04-1', '10', '12'  #maybe a good bad one
cell, BG, FG = 'TBR035a-13-1', '10', '07'  #maybe a good one
cell, BG, FG = 'TBR036a-23-1', '10', '12'  #maybe a good one
cell, BG, FG = 'TBR012a-31-1', '10', '07'  #maybe a good one
cell, BG, FG = 'TBR008a-43-1', '10', '07'  #maybe a good one
cell, BG, FG = 'TBR008a-15-1', '10', '15'  #maybe a good one


##This is good for making the PSTH with model fit and BG, FG, BG+FG with specs above
cellid_and_stim_str = opl.get_cellstring(cell, BG, FG, weight_df)
opl.plot_weight_psth(cellid_and_stim_str, weight_df, False)

#Makes the model diagram figure parts, not useful for anything but
opl.plot_model_diagram_parts(cellid_and_stim_str, weight_df)



##adding stuff to get out just the 1/2 alone, same 1/2 alone + full other, full both
full_sound = 'FG'
cell = df.loc['TBR011a-42-1']
pair_names = cell['pair_names']
pairs = [(pp.split('1_')[0][5:-3], pp.split('1_')[1][:-4]) for pp in pair_names]

sound = 0

if full_sound == 'FG':
    half_epochs = [f"STIM_{rr[0]}-0.5-1_{rr[1]}-0-1" for rr in pairs]
    lone_epochs = [f"STIM_null_{rr[1]}-0-1" for rr in pairs]
elif full_sound == 'BG':
    half_epochs = [f"STIM_{rr[0]}-0-1_{rr[1]}-0.5-1" for rr in pairs]
    lone_epochs = [f"STIM_{rr[0]}-0-1_null" for rr in pairs]
full_epochs = [str(aa) for aa in pair_names]

resp = cell['rec']['resp'].rasterize()
rh = resp.extract_epochs(half_epochs)
rf = resp.extract_epochs(full_epochs)
rl = resp.extract_epochs(lone_epochs)

hh = list(rh.keys())[sound]
ff = list(rf.keys())[sound]
ll = lone_epochs[sound]
half = np.squeeze(np.mean(rh[hh], axis=0))[:100]
full = np.squeeze(np.mean(rf[ff], axis=0))[:100]
lone = np.squeeze(np.mean(rl[ll], axis=0))[:100]


import scipy.ndimage.filters as sf

time = (np.arange(0, half.shape[0]) / fs)
half_smooth = sf.gaussian_filter1d(half, sigma=2)
full_smooth = sf.gaussian_filter1d(full, sigma=2)
lone_smooth = sf.gaussian_filter1d(lone, sigma=2)

maxdiff = np.max(np.abs(full_smooth-lone_smooth))

fullhalf = np.abs(full_smooth - half_smooth) / maxdiff
lonehalf = np.abs(lone_smooth - half_smooth) / maxdiff
signal = np.abs(lone_smooth - full_smooth) / maxdiff
# fullhalf = ((full_smooth - half_smooth)**2) #/ (full_smooth**2 + half_smooth**2)
# lonehalf = ((lone_smooth - half_smooth)**2) #/ (lone_smooth**2 + half_smooth**2)

var = np.mean(lonehalf[:int(len(time)/2)], axis=0)
time_idx = [i for i, v in enumerate(list(fullhalf)) if v <= var and i >= 50]

fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,6))
ax[0].plot(time, lone_smooth * fs, label=f"{full_sound} Alone", color='deepskyblue')
ax[0].plot(time, full_smooth * fs, label=f"Both Full", color='yellowgreen')
ax[0].plot(time, half_smooth * fs, label=f"{full_sound} Full/Other Half", color='grey')
ymin,ymax = ax[0].get_ylim()
ax[0].vlines(time[int(half.shape[0]/2)], ymin, ymax, ls=':')
ax[0].legend(fontsize=9)
ax[0].set_ylabel('Firing Rate', fontweight='bold', fontsize=10)

ax[1].plot(time, lonehalf, label='lone/half', color='deepskyblue')
ax[1].plot(time, fullhalf, label='full/half', color='yellowgreen')
ymin, ymax = ax[1].get_ylim()
ax[1].vlines(time[int(half.shape[0]/2)], ymin, ymax, ls=':')
ax[1].vlines(time[(time_idx[0])], ymin, ymax, ls='-', lw=1, color='black')
ax[1].hlines(np.mean(var), time[0], time[-1], ls='-', lw=1, color='black')
ax[1].legend(fontsize=9)
ax[1].set_ylabel("Normalized Difference", fontweight='bold', fontsize=10)
ax[1].set_xlabel("Time (s)", fontweight='bold', fontsize=10)
ax[0].set_title(f"{cell.name} - Pair {sound} - {pairs[sound]}", fontweight='bold', fontsize=14)
ax[1].set_title(f"Variability: {np.around(var, 3)} - Time: {int((time[time_idx[0]]-0.5) * 1000)}ms",
                fontweight='bold', fontsize=11)


#DO THIS FOR SEVERAL CELLS -
quadrant = 3
cellid = 'HOD005b-01-1'
cellid = 'TBR036a-55-1'

cells = 'HOD005b'
cells = 'TBR035a'
allcells = list(df.index)
cell_list = [i for i in allcells if i[:7]==cells]
var_list, cell_li = [], []
for cc in cell_list:
    qidx = get_filtered_pair_idxs(df, cc, quadrant)
    if len(qidx) > 0:
        vartime = plot_dynamic_stimuli_noplot(cc, pair_idx=qidx)
        var_list.append(vartime)
        cell_li.append(cc)
    else:
        print('Skipping')
av_var = [np.mean(i) for i in var_list]
std = [np.std(i) for i in var_list]
varr = np.asarray(av_var)
stdarr = np.asarray(std)
fig,ax = plt.subplots()
labs = [i[-4:] for i in cell_li]
ax.bar(labs, varr)
ax.set_xticklabels(labs, rotation=90)


vartime= plot_dynamic_stimuli(cellid)


qidx = get_filtered_pair_idxs(df, cellid, quadrant)
vartime = plot_dynamic_stimuli(cellid, pair_idx=qidx)


def plot_dynamic_stimuli_noplot(cellid, full_sound='FG', pair_idx=None):
    import scipy.ndimage.filters as sf

    cell = df.loc[cellid]
    pair_names = cell['pair_names']
    if pair_idx:
        pair_names = [pair_names[i] for i in pair_idx]
        print('Removing some pairs')
    pairs = [(pp.split('1_')[0][5:-3], pp.split('1_')[1][:-4]) for pp in pair_names]

    if full_sound == 'FG':
        half_epochs = [f"STIM_{rr[0]}-0.5-1_{rr[1]}-0-1" for rr in pairs]
        lone_epochs = [f"STIM_null_{rr[1]}-0-1" for rr in pairs]
    elif full_sound == 'BG':
        half_epochs = [f"STIM_{rr[0]}-0-1_{rr[1]}-0.5-1" for rr in pairs]
        lone_epochs = [f"STIM_{rr[0]}-0-1_null" for rr in pairs]
    full_epochs = [str(aa) for aa in pair_names]

    resp = cell['rec']['resp'].rasterize()
    rh = resp.extract_epochs(half_epochs)
    rf = resp.extract_epochs(full_epochs)
    rl = resp.extract_epochs(lone_epochs)

    vartime = []
    dA, dAB, signal = [], [], []
    Half, Lone, Full = [], [], []
    for sound in range(len(pairs)):
        hh = list(rh.keys())[sound]
        ff = list(rf.keys())[sound]
        ll = lone_epochs[sound]
        half = np.squeeze(np.mean(rh[hh], axis=0))[:100]
        full = np.squeeze(np.mean(rf[ff], axis=0))[:100]
        lone = np.squeeze(np.mean(rl[ll], axis=0))[:100]

        time = (np.arange(0, half.shape[0]) / fs)
        half_smooth = sf.gaussian_filter1d(half, sigma=2)
        full_smooth = sf.gaussian_filter1d(full, sigma=2)
        lone_smooth = sf.gaussian_filter1d(lone, sigma=2)

        maxdiff = np.max(np.abs(full_smooth-lone_smooth))

        Half.append(half_smooth)
        Full.append(full_smooth)
        Lone.append(lone_smooth)

        dAB.append(np.abs(full_smooth - half_smooth) / maxdiff)
        dA.append(np.abs(lone_smooth - half_smooth) / maxdiff)
        signal.append(np.abs(lone_smooth - full_smooth) / maxdiff)

    Half, Full, Lone = np.stack(Half), np.stack(Full), np.stack(Lone)
    dAB = np.stack(dAB) * np.stack(signal)
    dA = np.stack(dA) * np.stack(signal)
    var = np.mean(dA[:,:int(len(time)/2)], axis=1)

    for aa in range(len(pairs)):
        time_idx = [i for i, v in enumerate(list(dAB[aa,:])) if v <= var[aa] and i >= 50]
        vartime.append(int((time[time_idx[0]]-0.5) * 1000))
    return vartime


def plot_dynamic_stimuli(cellid, full_sound='FG', pair_idx=None):
    import scipy.ndimage.filters as sf

    cell = df.loc[cellid]
    pair_names = cell['pair_names']
    if pair_idx:
        pair_names = [pair_names[i] for i in pair_idx]
        print('Removing some pairs')
    pairs = [(pp.split('1_')[0][5:-3], pp.split('1_')[1][:-4]) for pp in pair_names]

    if full_sound == 'FG':
        half_epochs = [f"STIM_{rr[0]}-0.5-1_{rr[1]}-0-1" for rr in pairs]
        lone_epochs = [f"STIM_null_{rr[1]}-0-1" for rr in pairs]
    elif full_sound == 'BG':
        half_epochs = [f"STIM_{rr[0]}-0-1_{rr[1]}-0.5-1" for rr in pairs]
        lone_epochs = [f"STIM_{rr[0]}-0-1_null" for rr in pairs]
    full_epochs = [str(aa) for aa in pair_names]

    resp = cell['rec']['resp'].rasterize()
    rh = resp.extract_epochs(half_epochs)
    rf = resp.extract_epochs(full_epochs)
    rl = resp.extract_epochs(lone_epochs)

    fig, axes = plt.subplots(6,4, sharex=True, figsize=(15,10))
    ax = np.ravel(axes, 'F')

    vartime = []
    dA, dAB, signal = [], [], []
    Half, Lone, Full = [], [], []
    for sound in range(len(pairs)):
        hh = list(rh.keys())[sound]
        ff = list(rf.keys())[sound]
        ll = lone_epochs[sound]
        half = np.squeeze(np.mean(rh[hh], axis=0))[:100]
        full = np.squeeze(np.mean(rf[ff], axis=0))[:100]
        lone = np.squeeze(np.mean(rl[ll], axis=0))[:100]

        time = (np.arange(0, half.shape[0]) / fs)
        half_smooth = sf.gaussian_filter1d(half, sigma=2)
        full_smooth = sf.gaussian_filter1d(full, sigma=2)
        lone_smooth = sf.gaussian_filter1d(lone, sigma=2)

        maxdiff = np.max(np.abs(full_smooth-lone_smooth))

        Half.append(half_smooth)
        Full.append(full_smooth)
        Lone.append(lone_smooth)

        dAB.append(np.abs(full_smooth - half_smooth) / maxdiff)
        dA.append(np.abs(lone_smooth - half_smooth) / maxdiff)
        signal.append(np.abs(lone_smooth - full_smooth) / maxdiff)

    Half, Full, Lone = np.stack(Half), np.stack(Full), np.stack(Lone)
    dAB = np.stack(dAB) * np.stack(signal)
    dA = np.stack(dA) * np.stack(signal)
    var = np.mean(dA[:,:int(len(time)/2)], axis=1)

    xis = 0
    for aa in range(len(pairs)):
        ax[xis].plot(time, Lone[aa,:] * fs, label=f"{full_sound} Alone", color='deepskyblue')
        ax[xis].plot(time, Full[aa,:] * fs, label=f"Both Full", color='yellowgreen')
        ax[xis].plot(time, Half[aa,:] * fs, label=f"{full_sound} Full/Other Half", color='grey')
        ymin,ymax = ax[xis].get_ylim()
        ax[xis].vlines(time[int(Half[aa,:].shape[0]/2)], ymin, ymax, ls=':')
        if aa == 0:
            ax[xis].legend(fontsize=5)
        if pair_idx:
            pn = pair_idx[aa]
        else:
            pn = aa
        ax[xis].set_title(f"Pair {pn} - {pairs[aa]}", fontsize=8, fontweight='bold')

        xis += 1

        time_idx = [i for i, v in enumerate(list(dAB[aa,:])) if v <= var[aa] and i >= 50]
        vartime.append(int((time[time_idx[0]]-0.5) * 1000))

        ax[xis].plot(time, dA[aa,:], label='lone/half', color='deepskyblue')
        ax[xis].plot(time, dAB[aa,:], label='full/half', color='yellowgreen')
        ymin, ymax = ax[xis].get_ylim()
        ax[xis].vlines(time[int(Half[aa,:].shape[0]/2)], ymin, ymax, ls=':')
        ax[xis].vlines(time[(time_idx[0])], ymin, ymax, ls='-', lw=1, color='black')
        ax[xis].hlines(var[aa], time[0], time[-1], ls='-', lw=1)
        ax[xis].set_title(f"Variance = {np.around(var[aa],3)}, "
                          f"{int((time[time_idx[0]]-0.5) * 1000)}ms", fontsize=6)
        if aa == 0:
            ax[xis].legend(fontsize=7)

        xis += 1

        fig.suptitle(f"{cell.name}", fontweight='bold', fontsize=10)

    dAB_mean = np.mean(np.stack(dAB) * np.stack(signal), axis=0)
    dA_mean = np.mean(np.stack(dA) * np.stack(signal), axis=0)

    time_idx = [i for i, v in enumerate(list(dAB_mean)) if v <= np.mean(var) and i >=50]

    fig, ax = plt.subplots()
    ax.plot(time, dA_mean, label='dA')
    ax.plot(time, dAB_mean, label='dAB')
    ax.legend()
    ymin,ymax = ax.get_ylim()
    ax.vlines(time[int(Half[aa,:].shape[0]/2)], ymin, ymax, ls=':')
    ax.vlines(time[(time_idx[0])], ymin,ymax, ls='-', lw=1, color='black')
    ax.hlines(np.mean(var), time[0], time[-1], ls='-', lw=1, color='black')
    ax.set_title(f"{cell.name} - Variance={np.around(np.mean(var),3)} - "
                 f"{int((time[time_idx[0]]-0.5) * 1000)}ms")

    return vartime

def get_filtered_pair_idxs(df, cellid, quadrant):
    cell_df = df.loc[df.index==cellid]
    cell_weights = opl.df_to_weight_df(cell_df)
    quad, threshold = opl.quadrants_by_FRns(cell_weights, threshold=0.05, quad_return=[quadrant])
    new_df = quad[quadrant]
    for aa in range(len(new_df)):
        print(f"BR_FRns: {np.around(new_df.iloc[aa].BG_FRns, 4)} - "
              f"FG_FRns: {np.around(new_df.iloc[aa].FG_FRns,4)} - "
              f"BG: {new_df.iloc[aa].namesA[:-4]} - FG: {new_df.iloc[aa].namesB[:-4]}")
    dff = new_df.reset_index()
    quad_df = dff.loc[dff.cellid==cellid]
    qidx = [i for i in quad_df.level_1]

    # quad, threshold = opl.quadrants_by_FRns(weight_df, threshold=0.05, quad_return=[quadrant])
    # dff = quad[quadrant]
    # dff = dff.reset_index()
    # quad_df = dff.loc[dff.cellid==cellid]
    # qidx = [i for i in quad_df.level_1]
    return qidx




#DO THIS FOR SEVERAL SOUNDS AVERAGED TOGETHER -
#copy for matrix
dA, dAB, signal = [], [], []
for sound in range(len(pairs)):
    hh = list(rh.keys())[sound]
    ff = list(rf.keys())[sound]
    ll = lone_epochs[sound]
    half = np.squeeze(np.mean(rh[hh], axis=0))[:100]
    full = np.squeeze(np.mean(rf[ff], axis=0))[:100]
    lone = np.squeeze(np.mean(rl[ll], axis=0))[:100]


    import scipy.ndimage.filters as sf

    time = (np.arange(0, half.shape[0]) / fs)
    half_smooth = sf.gaussian_filter1d(half, sigma=2)
    full_smooth = sf.gaussian_filter1d(full, sigma=2)
    lone_smooth = sf.gaussian_filter1d(lone, sigma=2)

    maxdiff = np.max(np.abs(full_smooth-lone_smooth))

    dAB.append(np.abs(full_smooth - half_smooth) / maxdiff)
    dA.append(np.abs(lone_smooth - half_smooth) / maxdiff)
    signal.append(np.abs(lone_smooth - full_smooth) / maxdiff)

dAB = np.mean(np.stack(dAB) * np.stack(signal), axis=0)
dA = np.mean(np.stack(dA) * np.stack(signal), axis=0)

fig, ax = plt.subplots()
ax.plot(dA, label='dA')
ax.plot(dAB, label='dAB')
ax.legend()
ymin,ymax = ax.get_ylim()
ax.vlines(50, ymin,ymax, ls=':')
ax.set_title(cell.name)






#sliding bins manually instead of smooth
bins = len(half)
width = 3
last_bin = bins-width
half_bin = np.array([np.mean(half[ii:ii+width]) for ii in range(last_bin)])
full_bin = np.array([np.mean(full[ii:ii+width]) for ii in range(last_bin)])
lone_bin = np.array([np.mean(lone[ii:ii+width]) for ii in range(last_bin)])

fullcomp = ((full_bin - half_bin)**2) / (full_bin**2 + half_bin**2)
lonecomp = ((lone_bin - half_bin)**2) / (lone_bin**2 + half_bin**2)

fig, ax = plt.subplots()
ax.plot(fullcomp, label='full/half')
ax.plot(lonecomp, label='lone/half')
ax.legend()

ww = ((full - half)**2) / (full**2 + half**2)

fig,ax = plt.subplots()
ax.plot(half, label='half')
ax.plot(full, label='full')
ax.plot(lone, label='lone')
ax.legend()

epdf =




##remake figure the way Stephen wants with only tabor
weight_df = weight_df.loc[weight_df.Animal=='TBR']
weightthresh = 3
weight_df = weight_df.loc[(weight_df.weightsB>=-weightthresh) & (weight_df.weightsB<=weightthresh)]
weight_df = weight_df.loc[(weight_df.weightsA>=-weightthresh) & (weight_df.weightsA<=weightthresh)]


metric = 'weightsB'
anims = weight_df.Animal.unique()
df1 = weight_df.loc[weight_df.Animal==anims[0]]
fgs1 = list(df1['namesB'].value_counts().index)
fg_names1 = [aa.split('-')[0][2:].replace('_', '') for aa in fgs1]




new_fgs1_title = [aa.split('-')[0][2:].replace('_', '') for aa in fgs1 if aa != 'All']
new_fgs1_title.insert(0, 'All')

fig, axes = plt.subplots(4, 6, sharex=True, sharey=True)
ax = np.ravel(axes)
fgs1.insert(0, 'All')

axn, titl = 0, 0
plot_list = []
for axn, fg1 in enumerate(fgs1):

    if fg1 == 'All':
        plot_df, anm = weight_df, 'All'
    else:
        plot_df = weight_df.loc[weight_df['namesB'] == fg1]

    quad_sub, threshold = opl.quadrants_by_FRns(plot_df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
    mean_weights = [np.mean(quad_sub[aa+1][metric]) for aa in range(len(quad_sub))]
    weight_array = np.reshape(mean_weights, (3,3))
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    nn = len(plot_df)
    ax[axn].set_title(f"{new_fgs1_title[axn]} - n={nn}", fontweight='bold', size=10), ax[axn].set_aspect('equal')
    plot_list.append(weight_array)

ax[0].set_yticks([]), ax[0].set_xticks([])

plot_list = np.stack(plot_list, axis=0)
vmi = np.nanmin(plot_list)
vma = np.nanmax(plot_list)
if vma > abs(vmi):
    vmi = -vma
else:
    vma = -vmi

for cnt, aa in enumerate(plot_list):
    im = ax[cnt].imshow(aa, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')

plt.colorbar(im)

fig.tight_layout()




#Makes graphic of scatters of FRs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
import numpy as np
fig, axes = plt.subplots(4, 6)
axes = np.ravel(axes)

fgs = list(weight_df['namesB'].value_counts().index)
fg_cnt = list(weight_df['namesB'].value_counts().values)
fgs.insert(0, 'All'), fg_cnt.insert(0, len(weight_df))

for (ax, fg, cnt) in zip(axes, fgs, fg_cnt):
    ax.set_aspect('equal')

    if fg == 'All':
        plot_df, anm = weight_df, 'All'
    else:
        if len(weight_df.loc[weight_df['namesB']==fg].Animal.unique()) == 1:
            anm = weight_df.loc[weight_df['namesB'] == fg].Animal.unique()[0]
        else:
            raise ValueError(f"FG {fg} was played to multiple animals.")
        plot_df = weight_df.loc[weight_df['namesB'] == fg]
    im = ax.scatter(plot_df['BG_FRns'], plot_df['FG_FRns'], s=3, color='black')
    ymin,ymax = ax.get_ylim()
    xmin,xmax = ax.get_xlim()
    low, high = min(xmin,ymin), max(xmax,ymax)
    ax.set_ylim(low, high)
    ax.set_xlim(low, high)
    ax.vlines([threshold, -threshold], low, high, ls='-', color='deepskyblue', lw=0.5)
    ax.hlines([threshold, -threshold], low, high, ls='-', color='deepskyblue', lw=0.5)
    ax.set_ylabel("FG FR"), ax.set_xlabel("BG FR")

    quad_sub, _= opl.quadrants_by_FRns(plot_df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
    total_len = len(plot_df)
    counts = [len(qd) for cc, qd in quad_sub.items()]
    percents = [int(np.around((cc/sum(counts)) * 100, 0)) for cc in counts]
    max_percent = max(percents)

    ax.set_title(f"FG: {fg} - n={cnt}\nAnimal: {anm} - Max: {max_percent}%", fontweight='bold')

    alp = 0.9
    n = len(percents)
    mm = max(percents) + 5
    colors = plt.cm.cool(np.linspace(0,1,mm))
    ax.add_patch(mpatches.Rectangle((low, threshold), abs(threshold+low), abs(high-threshold),
                                 linewidth=1, facecolor=colors[percents[0]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((-threshold, threshold), abs(threshold*2), abs(high-threshold),
                                 linewidth=1, facecolor=colors[percents[1]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((threshold, threshold), abs(high-threshold), abs(high-threshold),
                                 linewidth=1, facecolor=colors[percents[2]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((low, -threshold), abs(threshold+low), abs(threshold*2),
                                 linewidth=1, facecolor=colors[percents[3]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((-threshold, -threshold), abs(threshold*2), abs(threshold*2),
                                 linewidth=1, facecolor=colors[percents[4]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((threshold, -threshold), abs(high-threshold), abs(threshold*2),
                                 linewidth=1, facecolor=colors[percents[5]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((low, low), abs(threshold+low), abs(low+threshold),
                                 linewidth=1, facecolor=colors[percents[6]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((-threshold, low), abs(threshold*2), abs(low+threshold),
                                 linewidth=1, facecolor=colors[percents[7]], alpha=alp))
    ax.add_patch(mpatches.Rectangle((threshold, low), abs(high-threshold), abs(low+threshold),
                                 linewidth=1, facecolor=colors[percents[8]], alpha=alp))
fig.tight_layout()

##remake figure the way Stephen wants
weightthresh = 3
weight_df = weight_df.loc[(weight_df.weightsB>=-weightthresh) & (weight_df.weightsB<=weightthresh)]
weight_df = weight_df.loc[(weight_df.weightsA>=-weightthresh) & (weight_df.weightsA<=weightthresh)]


metric = 'weightsB'
anims = weight_df.Animal.unique()
df1 = weight_df.loc[weight_df.Animal==anims[0]]
fgs1 = list(df1['namesB'].value_counts().index)
df2 = weight_df.loc[weight_df.Animal==anims[1]]
fgs2 = list(df2['namesB'].value_counts().index)
num_fgs2 = len(fgs2)
fg_names1 = [aa.split('-')[0][2:].replace('_', '') for aa in fgs1]
fg_names2 = [aa.split('-')[0][2:].replace('_', '') for aa in fgs2]

names2, names2idx = [], []
fgs2.append('null')
for ww in fg_names1:
    if ww in fg_names2:
        names2.append(ww)
        names2idx.append(fg_names2.index(ww))
    else:
        names2.append('null')
        names2idx.append(-1)
fg2_unique = [aa for aa in range(num_fgs2) if aa not in names2idx]
names2idx = names2idx + fg2_unique
new_fgs2 = [fgs2[aa] for aa in names2idx]
fg_names2.append('null')
new_fgs2_title = [fg_names2[aa] for aa in names2idx]

if len(new_fgs2) != len(fgs1):
    dif = len(new_fgs2) - len(fgs1)
    new_fgs1 = fgs1 + ['null'] * dif
else:
    new_fgs1 = fgs1

new_fgs1_title = [aa.split('-')[0][2:].replace('_', '') for aa in new_fgs1 if aa != 'All']
for nn, aa in enumerate(new_fgs1_title):
    if aa == 'null':
        new_fgs1_title[nn] = new_fgs2_title[nn]

#hacky add for lab meeting
keeps = [0, 1, 2, 4, 6]
new_fgs1 = [new_fgs1[i] for i in keeps]
new_fgs1_title = [new_fgs1_title[i] for i in keeps]
new_fgs2 = [new_fgs2[i] for i in keeps]
new_fgs2_title = [new_fgs2_title[i] for i in keeps]

new_fgs1_title.insert(0, 'All')

fig, axes = plt.subplots(3, len(new_fgs1)+1, sharex=True, sharey=True)
ax = np.ravel(axes, 'F')
new_fgs1.insert(0, 'All'), new_fgs2.insert(0, 'All')

axn, titl = 0, 0
plot_list = []
for (fg1, fg2) in zip(new_fgs1, new_fgs2):

    if fg1 == 'All':
        plot_df, anm = weight_df, 'All'
        anm1, anm2 = anims[0], anims[1]
        plot1, plot2 = weight_df.loc[weight_df['Animal']==anm1], weight_df.loc[weight_df['Animal']==anm2]
    elif fg1 == 'null':
        plot_df = weight_df.loc[weight_df['namesB']==fg2]
        plot1, anm1 = np.empty((3,3)), anims[0]
        plot2, anm2 = weight_df.loc[weight_df['namesB']==fg2], anims[1]
        plot1[:] = np.NaN
    elif fg2 == 'null':
        plot_df = weight_df.loc[weight_df['namesB']==fg1]
        plot1, anm1 = weight_df.loc[weight_df['namesB']==fg1], anims[0]
        plot2, anm2 = np.empty((3,3)), anims[1]
        plot2[:] = np.NaN
    else:
        anm1, anm2 = anims[0], anims[1]
        plot_df = weight_df.loc[(weight_df['namesB'] == fg1) | (weight_df['namesB'] == fg2)]
        plot1 = weight_df.loc[weight_df['namesB'] == fg1]
        plot2 = weight_df.loc[weight_df['namesB'] == fg2]

    quad_sub, threshold = opl.quadrants_by_FRns(plot_df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
    mean_weights = [np.mean(quad_sub[aa+1][metric]) for aa in range(len(quad_sub))]
    weight_array = np.reshape(mean_weights, (3,3))
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"{new_fgs1_title[titl]}", fontweight='bold'), ax[axn].set_aspect('equal')
    plot_list.append(weight_array)
    axn += 1

    if fg1 != 'null':
        quad_sub, threshold = opl.quadrants_by_FRns(plot1, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
        mean_weights = [np.mean(quad_sub[aa+1][metric]) for aa in range(len(quad_sub))]
        weight_array = np.reshape(mean_weights, (3,3))
        nn = len(plot1)
    else:
        weight_array = plot1
        nn = 0
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"n={nn}", size=8), ax[axn].set_aspect('equal')
    plot_list.append(weight_array)
    axn += 1

    if fg2 != 'null':
        quad_sub, threshold = opl.quadrants_by_FRns(plot2, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
        mean_weights = [np.mean(quad_sub[aa+1][metric]) for aa in range(len(quad_sub))]
        weight_array = np.reshape(mean_weights, (3,3))
        nn = len(plot2)
    else:
        weight_array = plot2
        nn = 0
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"n={nn}", size=8), ax[axn].set_aspect('equal')
    plot_list.append(weight_array)
    axn += 1
    titl += 1

ax[0].set_ylabel('Combined', fontweight='bold')
ax[1].set_ylabel(f"{anims[0]}", fontweight='bold')
ax[2].set_ylabel(f"{anims[1]}", fontweight='bold')
ax[0].set_yticks([]), ax[0].set_xticks([])

plot_list = np.stack(plot_list, axis=0)
vmi = np.nanmin(plot_list)
vma = np.nanmax(plot_list)
# vmi = -vma
for cnt, aa in enumerate(plot_list):
    im = ax[cnt].imshow(aa, vmin=vmi, vmax=vma, aspect='equal', cmap='YlGnBu')

plt.colorbar(im)

fig.tight_layout()



##remake figure the way Stephen wants but percentages
weightthresh = 3
weight_df = weight_df.loc[(weight_df.weightsB>=-weightthresh) & (weight_df.weightsB<=weightthresh)]
weight_df = weight_df.loc[(weight_df.weightsA>=-weightthresh) & (weight_df.weightsA<=weightthresh)]


anims = weight_df.Animal.unique()
df1 = weight_df.loc[weight_df.Animal==anims[0]]
fgs1 = list(df1['namesB'].value_counts().index)
df2 = weight_df.loc[weight_df.Animal==anims[1]]
fgs2 = list(df2['namesB'].value_counts().index)
num_fgs2 = len(fgs2)
fg_names1 = [aa.split('-')[0][2:].replace('_', '') for aa in fgs1]
fg_names2 = [aa.split('-')[0][2:].replace('_', '') for aa in fgs2]

names2, names2idx = [], []
fgs2.append('null')
for ww in fg_names1:
    if ww in fg_names2:
        names2.append(ww)
        names2idx.append(fg_names2.index(ww))
    else:
        names2.append('null')
        names2idx.append(-1)
fg2_unique = [aa for aa in range(num_fgs2) if aa not in names2idx]
names2idx = names2idx + fg2_unique
new_fgs2 = [fgs2[aa] for aa in names2idx]
fg_names2.append('null')
new_fgs2_title = [fg_names2[aa] for aa in names2idx]

if len(new_fgs2) != len(fgs1):
    dif = len(new_fgs2) - len(fgs1)
    new_fgs1 = fgs1 + ['null'] * dif
else:
    new_fgs1 = fgs1

new_fgs1_title = [aa.split('-')[0][2:].replace('_', '') for aa in new_fgs1 if aa != 'All']
for nn, aa in enumerate(new_fgs1_title):
    if aa == 'null':
        new_fgs1_title[nn] = new_fgs2_title[nn]

# counts1 = [len(df1.loc[df1.namesB==i]) for i in new_fgs1]
# counts2 = [len(df2.loc[df1.namesB==i]) for i in new_fgs2]
#Need to automate this
keeps = [0, 1, 2, 4, 6]
new_fgs1 = [new_fgs1[i] for i in keeps]
new_fgs1_title = [new_fgs1_title[i] for i in keeps]
new_fgs2 = [new_fgs2[i] for i in keeps]
new_fgs2_title = [new_fgs2_title[i] for i in keeps]

new_fgs1_title.insert(0, 'All')

fig, axes = plt.subplots(3, len(new_fgs1)+1, sharex=True, sharey=True)
ax = np.ravel(axes, 'F')
new_fgs1.insert(0, 'All'), new_fgs2.insert(0, 'All')

axn, titl = 0, 0
plot_list = []
for (fg1, fg2) in zip(new_fgs1, new_fgs2):

    if fg1 == 'All':
        plot_df, anm = weight_df, 'All'
        anm1, anm2 = anims[0], anims[1]
        plot1, plot2 = weight_df.loc[weight_df['Animal']==anm1], weight_df.loc[weight_df['Animal']==anm2]
    elif fg1 == 'null':
        plot_df = weight_df.loc[weight_df['namesB']==fg2]
        plot1, anm1 = np.empty((3,3)), anims[0]
        plot2, anm2 = weight_df.loc[weight_df['namesB']==fg2], anims[1]
        plot1[:] = np.NaN
    elif fg2 == 'null':
        plot_df = weight_df.loc[weight_df['namesB']==fg1]
        plot1, anm1 = weight_df.loc[weight_df['namesB']==fg1], anims[0]
        plot2, anm2 = np.empty((3,3)), anims[1]
        plot2[:] = np.NaN
    else:
        anm1, anm2 = anims[0], anims[1]
        plot_df = weight_df.loc[(weight_df['namesB'] == fg1) | (weight_df['namesB'] == fg2)]
        plot1 = weight_df.loc[weight_df['namesB'] == fg1]
        plot2 = weight_df.loc[weight_df['namesB'] == fg2]

    quad_sub, threshold = opl.quadrants_by_FRns(plot_df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
    counts = [len(qd) for cc, qd in quad_sub.items()]
    percents = [int(np.around((cc / sum(counts)) * 100, 0)) for cc in counts]
    percent_array = np.reshape(percents, (3,3))

    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"{new_fgs1_title[titl]}", fontweight='bold'), ax[axn].set_aspect('equal')
    plot_list.append(percent_array)
    axn += 1

    if fg1 != 'null':
        quad_sub, threshold = opl.quadrants_by_FRns(plot1, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
        counts = [len(qd) for cc, qd in quad_sub.items()]
        percents = [int(np.around((cc / sum(counts)) * 100, 0)) for cc in counts]
        percent_array = np.reshape(percents, (3, 3))
        nn = len(plot1)
    else:
        percent_array = plot1
        nn = 0
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"n={nn}", size=8), ax[axn].set_aspect('equal')
    plot_list.append(percent_array)
    axn += 1

    if fg2 != 'null':
        quad_sub, threshold = opl.quadrants_by_FRns(plot2, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
        counts = [len(qd) for cc, qd in quad_sub.items()]
        percents = [int(np.around((cc / sum(counts)) * 100, 0)) for cc in counts]
        percent_array = np.reshape(percents, (3, 3))
        nn = len(plot2)
    else:
        percent_array = plot2
        nn = 0
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"n={nn}", size=8), ax[axn].set_aspect('equal')
    plot_list.append(percent_array)
    axn += 1
    titl += 1

ax[0].set_ylabel('Combined', fontweight='bold')
ax[1].set_ylabel(f"{anims[0]}", fontweight='bold')
ax[2].set_ylabel(f"{anims[1]}", fontweight='bold')
ax[0].set_yticks([]), ax[0].set_xticks([])

plot_list = np.stack(plot_list, axis=0)
vmi = np.nanmin(plot_list)
vma = np.nanmax(plot_list)
# vmi = -vma
for cnt, aa in enumerate(plot_list):
    im = ax[cnt].imshow(aa, vmin=vmi, vmax=vma, aspect='equal', cmap='inferno')

plt.colorbar(im)

fig.tight_layout()

##remake figure the way Stephen wants using BGs
weightthresh = 3
weight_df = weight_df.loc[(weight_df.weightsB>=-weightthresh) & (weight_df.weightsB<=weightthresh)]
weight_df = weight_df.loc[(weight_df.weightsA>=-weightthresh) & (weight_df.weightsA<=weightthresh)]



fgs = list(df2['namesA'].value_counts().index)
fg_cnt = list(df2['namesA'].value_counts().values)

metric = 'weightsA'
anims = weight_df.Animal.unique()
df1 = weight_df.loc[weight_df.Animal==anims[0]]
fgs1 = list(df1['namesA'].value_counts().index)
df2 = weight_df.loc[weight_df.Animal==anims[1]]
fgs2 = list(df2['namesA'].value_counts().index)
num_fgs2 = len(fgs2)
fg_names1 = [aa.split('-')[0][2:].replace('_', '') for aa in fgs1]
fg_names2 = [aa.split('-')[0][2:].replace('_', '') for aa in fgs2]

names2, names2idx = [], []
fgs2.append('null')
for ww in fg_names1:
    if ww in fg_names2:
        names2.append(ww)
        names2idx.append(fg_names2.index(ww))
    else:
        names2.append('null')
        names2idx.append(-1)
fg2_unique = [aa for aa in range(num_fgs2) if aa not in names2idx]
names2idx = names2idx + fg2_unique
new_fgs2 = [fgs2[aa] for aa in names2idx]
fg_names2.append('null')
new_fgs2_title = [fg_names2[aa] for aa in names2idx]

if len(new_fgs2) != len(fgs1):
    dif = len(new_fgs2) - len(fgs1)
    new_fgs1 = fgs1 + ['null'] * dif
else:
    new_fgs1 = fgs1

new_fgs1_title = [aa.split('-')[0][2:].replace('_', '') for aa in new_fgs1 if aa != 'All']
for nn, aa in enumerate(new_fgs1_title):
    if aa == 'null':
        new_fgs1_title[nn] = new_fgs2_title[nn]
new_fgs1_title.insert(0, 'All')

fig, axes = plt.subplots(3, len(new_fgs1)+1, sharex=True, sharey=True)
ax = np.ravel(axes, 'F')
new_fgs1.insert(0, 'All'), new_fgs2.insert(0, 'All')

axn, titl = 0, 0
plot_list = []
for (fg1, fg2) in zip(new_fgs1, new_fgs2):

    if fg1 == 'All':
        plot_df, anm = weight_df, 'All'
        anm1, anm2 = anims[0], anims[1]
        plot1, plot2 = weight_df.loc[weight_df['Animal']==anm1], weight_df.loc[weight_df['Animal']==anm2]
    elif fg1 == 'null':
        plot_df = weight_df.loc[weight_df['namesA']==fg2]
        plot1, anm1 = np.empty((3,3)), anims[0]
        plot2, anm2 = weight_df.loc[weight_df['namesA']==fg2], anims[1]
        plot1[:] = np.NaN
    elif fg2 == 'null':
        plot_df = weight_df.loc[weight_df['namesA']==fg1]
        plot1, anm1 = weight_df.loc[weight_df['namesA']==fg1], anims[0]
        plot2, anm2 = np.empty((3,3)), anims[1]
        plot2[:] = np.NaN
    else:
        anm1, anm2 = anims[0], anims[1]
        plot_df = weight_df.loc[(weight_df['namesA'] == fg1) | (weight_df['namesA'] == fg2)]
        plot1 = weight_df.loc[(weight_df['namesA'] == fg1) & (weight_df['Animal'] == anims[0])]
        plot2 = weight_df.loc[(weight_df['namesA'] == fg2) & (weight_df['Animal'] == anims[1])]

    quad_sub, threshold = opl.quadrants_by_FRns(plot_df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
    mean_weights = [np.mean(quad_sub[aa+1][metric]) for aa in range(len(quad_sub))]
    weight_array = np.reshape(mean_weights, (3,3))
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"{new_fgs1_title[titl]}", fontweight='bold'), ax[axn].set_aspect('equal')
    plot_list.append(weight_array)
    axn += 1

    if fg1 != 'null':
        quad_sub, threshold = opl.quadrants_by_FRns(plot1, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
        mean_weights = [np.mean(quad_sub[aa+1][metric]) for aa in range(len(quad_sub))]
        weight_array = np.reshape(mean_weights, (3,3))
        nn = len(plot1)
    else:
        weight_array = plot1
        nn = 0
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"n={nn}", size=8), ax[axn].set_aspect('equal')
    plot_list.append(weight_array)
    axn += 1

    if fg2 != 'null':
        quad_sub, threshold = opl.quadrants_by_FRns(plot2, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
        mean_weights = [np.mean(quad_sub[aa+1][metric]) for aa in range(len(quad_sub))]
        weight_array = np.reshape(mean_weights, (3,3))
        nn = len(plot2)
    else:
        weight_array = plot2
        nn = 0
    # ax[axn].imshow(weight_array, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')
    ax[axn].set_title(f"n={nn}", size=8), ax[axn].set_aspect('equal')
    plot_list.append(weight_array)
    axn += 1
    titl += 1

ax[0].set_ylabel('Combined', fontweight='bold')
ax[1].set_ylabel(f"{anims[0]}", fontweight='bold')
ax[2].set_ylabel(f"{anims[1]}", fontweight='bold')
ax[0].set_yticks([]), ax[0].set_xticks([])

plot_list = np.stack(plot_list, axis=0)
vmi = np.nanmin(plot_list)
vma = np.nanmax(plot_list)
vmi = -vma
for cnt, aa in enumerate(plot_list):
    im = ax[cnt].imshow(aa, vmin=vmi, vmax=vma, aspect='equal', cmap='bwr')

plt.colorbar(im)

fig.tight_layout()


#Makes graphic of scatters of FRs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
import numpy as np
fig, axes = plt.subplots(4, 6)
axes = np.ravel(axes)

fgs = list(weight_df['namesB'].value_counts().index)
fg_cnt = list(weight_df['namesB'].value_counts().values)
fgs.insert(0, 'All'), fg_cnt.insert(0, len(weight_df))

for (ax, fg, cnt) in zip(axes, fgs, fg_cnt):
    ax.set_aspect('equal')

    if fg == 'All':
        plot_df, anm = weight_df, 'All'
    else:
        if len(weight_df.loc[weight_df['namesB']==fg].Animal.unique()) == 1:
            anm = weight_df.loc[weight_df['namesB'] == fg].Animal.unique()[0]
        else:
            raise ValueError(f"FG {fg} was played to multiple animals.")
        plot_df = weight_df.loc[weight_df['namesB'] == fg]
    im = ax.scatter(plot_df['BG_FRns'], plot_df['FG_FRns'], s=3, color='black')
    ymin,ymax = ax.get_ylim()
    xmin,xmax = ax.get_xlim()
    low, high = min(xmin,ymin), max(xmax,ymax)
    ax.set_ylim(low, high)
    ax.set_xlim(low, high)
    ax.vlines([threshold, -threshold], low, high, ls='-', color='deepskyblue', lw=0.5)
    ax.hlines([threshold, -threshold], low, high, ls='-', color='deepskyblue', lw=0.5)
    ax.set_ylabel("FG FR"), ax.set_xlabel("BG FR")

    quad_sub, _= opl.quadrants_by_FRns(plot_df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])
    total_len = len(plot_df)
    counts = [len(qd) for cc, qd in quad_sub.items()]
    percents = [int(np.around((cc/sum(counts)) * 100, 0)) for cc in counts]
    max_percent = max(percents)

    ax.set_title(f"FG: {fg} - n={cnt}\nAnimal: {anm} - Max: {max_percent}%", fontweight='bold')


fig.tight_layout()