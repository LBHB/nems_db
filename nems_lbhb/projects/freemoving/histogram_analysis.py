import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from nems0 import db, preprocessing
import seaborn as sns

runclassid = 132
rasterfs = 100
# sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
# dparm = db.pd_query(sql)
from functools import partial
from scipy.ndimage import maximum_filter, binary_erosion
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.stats import  kruskal
sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

# peak detections
def detect_peaks(image, neighbors=None, threshold=0.5):
    """
    Detects local peaks and troughs in image above a threshold using
    ndimage.maximum_filter and returns values and locations

    :param image: A 2D matrix/image
    :param neighbors: shape of surrounding pixel array to use as the local filter (n*n array)
    :param threshold: (0-1) percentage of maximum peak value to use as threshold
    :return: A list of peak/trough values and their coordinates as well as a mask of all the peaks/troughs in the image
    """

    if neighbors is None:
        neighbors = [3, 3]

    # Take abs of image so peaks and troughs are detected
    image_abs = abs(image)

    # define an n-connected  neighborhood
    neighborhood = np.ones((neighbors[0], neighbors[1]))

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image_abs, footprint=neighborhood) == image_abs
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image_abs == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    # peak coordinates
    peak_loc = np.array(np.where(detected_peaks == 1)).transpose()

    # peak values
    peak_values = []
    for peak in peak_loc:
        peak_value = image[peak[0], peak[1]]
        peak_values.append(peak_value)

    # find max value and set threshold to identify relatively large peaks and create list with coordinates
    max_peak = max(abs(np.array(peak_values)))

    largest_peaks = {peak_value: peak_loc for peak_value, peak_loc in zip(peak_values, peak_loc) if
                     abs(peak_value) >= threshold * max_peak}

    return largest_peaks, detected_peaks

# plot functions
def onclick(xy, event):
    from matplotlib import colors
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f, axes=%s' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata, event.inaxes.get_title()))

    f, ax = plt.subplots(1,3, sharex=True, sharey=True)
    df = xy['df']
    df = df.loc[df['layer'] == event.inaxes.get_title().split(' ')[1]]
    layer = event.inaxes.get_title().split(' ')[1]
    x_stat = xy['x']
    y_stat = xy['y']

    closest_x = df.loc[df[x_stat].between((event.xdata - 0.1), (event.xdata + 0.1), inclusive=False), 'cellid']
    nearest_cell_i = np.abs(df[df['cellid'].isin(closest_x)][y_stat].values - event.ydata).argmin()
    nearest_cell_id = [df[df['cellid'].isin(closest_x)]['cellid'].values[nearest_cell_i]]

    tc1 = df[df['cellid'].isin(nearest_cell_id)]['tc1'].values[0]
    tc2 = df[df['cellid'].isin(nearest_cell_id)]['tc2'].values[0]
    tc = df[df['cellid'].isin(nearest_cell_id)]['tc'].values[0]
    sc = df[df['cellid'].isin(nearest_cell_id)]['scc'].values[0]
    si = df[df['cellid'].isin(nearest_cell_id)]['si'].values[0]
    xbins = tc.shape[0]
    ybins = tc.shape[1]
    imopts = {'cmap': 'viridis', 'interpolation': 'none', 'extent': [0, xbins*20, 0, ybins*20]}
    ax[0].imshow(tc1.T, **imopts)
    ax[0].set_title('Split 1')
    ax[1].imshow(tc2.T, **imopts)
    ax[1].set_title('Split 2')
    ax[2].imshow(tc.T, **imopts)
    ax[2].set_title('All data')
    ax[0].set_ylabel("pixels y")

    axs = ax.flatten()
    for ax in axs:
        ax.set_xlabel("pixels x")

    f.suptitle(f'ID: {nearest_cell_id[0]} \n layer: {layer}, SC: {np.round(sc, decimals=2)}, SI: {np.round(si, decimals=2)}')
    f.tight_layout()
    plt.savefig(f'/auto/users/wingertj/data/{nearest_cell_id[0]}.pdf', dpi=400)

def interactive_spatial_scatter(df, x_stat, y_stat):
    clrs = [matplotlib.colors.to_rgb(c) for c in matplotlib.pyplot.rcParams['axes.prop_cycle'].by_key()['color']]
    f, ax = plt.subplots(1, 3, sharey=True, sharex=True)
    layer_13_cellids = df.loc[df['layer'] == '13', 'cellid'].unique()
    ax[0].scatter([df.loc[(df['layer'] == '13') & (df['cellid'] == cell), x_stat].values[0] for cell in
                   layer_13_cellids],
                  [df.loc[(df['layer'] == '13') & (df['cellid'] == cell), y_stat].values[0] for cell in
                   layer_13_cellids], c=clrs[0], alpha=0.3)
    ax[0].set_title('Layer 13')
    layer_4_cellids = df.loc[df['layer'] == '44', 'cellid'].unique()
    ax[1].scatter([df.loc[(df['layer'] == '44') & (df['cellid'] == cell), x_stat].values[0] for cell in
                   layer_4_cellids],
                  [df.loc[(df['layer'] == '44') & (df['cellid'] == cell), y_stat].values[0] for cell in
                   layer_4_cellids], c=clrs[1], alpha=0.3)
    ax[1].set_title('Layer 44')
    layer_56_cellids = df.loc[df['layer'] == '56', 'cellid'].unique()
    ax[2].scatter([df.loc[(df['layer'] == '56') & (df['cellid'] == cell), x_stat].values[0] for cell in
                   layer_56_cellids],
                  [df.loc[(df['layer'] == '56') & (df['cellid'] == cell), y_stat].values[0] for cell in
                   layer_56_cellids], color=clrs[2], alpha=0.3)
    ax[2].set_title('Layer 56')
    ax[0].set_ylabel('Spatial Information (bits)')
    ax[0].set_xlabel('Spatial Correlation')
    ax[1].set_xlabel('Spatial Correlation')
    ax[2].set_xlabel('Spatial Correlation')
    cid = f.canvas.mpl_connect('button_press_event', partial(onclick, {'df': df, 'x':x_stat, 'y':y_stat}))
    return f, ax

batch = 348
siteids, cellids = db.get_batch_sites(batch)

sql = f"SELECT distinct left(cellid,7) as siteid,stimpath,stimfile from sCellFile where runclassid={runclassid}"
dallfiles = db.pd_query(sql)
siteids = dallfiles['siteid'].unique().tolist()

hist_df = pd.read_pickle('/auto/users/wingertj/data/spatial_hist_df.pkl')
hist_df.loc[hist_df['layer']=='5', 'layer'] = '56'
hist_df.loc[hist_df['layer']=='3', 'layer'] = '13'
hist_df.loc[hist_df['layer']=='BS', 'layer'] = '13'
hist_df.drop(hist_df[hist_df['layer']=='16'].index, inplace=True)
hist_df = hist_df.sort_values('layer')

sig_df = hist_df.loc[hist_df['scc'] > hist_df['scc_threshold']]
sig_df_si = hist_df.loc[hist_df['si'] > hist_df['si_threshold']]

sig_l4 = len(sig_df.loc[sig_df['layer'] == '44'])
all_l4 = len(hist_df.loc[hist_df['layer']=='44'])
percent_sig4 = (sig_l4 / all_l4) * 100
sig_l56 = len(sig_df.loc[sig_df['layer'] == '56'])
all_l56 = len(hist_df.loc[hist_df['layer']=='56'])
percent_sig56 = (sig_l56 / all_l56) * 100
sig_l13 = len(sig_df.loc[sig_df['layer'] == '13'])
all_l13 = len(hist_df.loc[hist_df['layer']=='13'])
percent_sig13 = (sig_l13 / all_l13) * 100

# significance tests - krukal wallis? talk to stephen
data_scc = [sig_df[sig_df['layer'] == '13']['scc'].values, sig_df[sig_df['layer'] == '44']['scc'].values, sig_df[sig_df['layer'] == '56']['scc'].values]
stat_scc, pval_scc = kruskal(data_scc[0], data_scc[1], data_scc[2], nan_policy='omit')

data_si = [sig_df[sig_df['layer'] == '13']['si'].values, sig_df[sig_df['layer'] == '44']['si'].values, sig_df[sig_df['layer'] == '56']['si'].values]
stat_si, pval_si = kruskal(data_si[0], data_si[1], data_si[2], nan_policy='omit')

# multiple t-tests
import numpy as np
rng = np.random.default_rng()
from scipy.stats import permutation_test
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

for xi in range(3):
    for yi in range(3):
        res = permutation_test((data_scc[xi], data_scc[yi]), statistic, vectorized=True,
                               n_resamples=1000, alternative='two-sided', random_state=rng)
        print(f"{xi}/{yi}: stat: {res.statistic}, pval: {res.pvalue}")

for xi in range(3):
    for yi in range(3):
        res = permutation_test((data_si[xi], data_si[yi]), statistic, vectorized=True,
                               n_resamples=1000, alternative='two-sided', random_state=rng)
        print(f"{xi}/{yi}: stat: {res.statistic}, pval: {res.pvalue}")

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].pie([sig_l13, all_l13-sig_l13], labels=['Sig', 'Non-sig'], labeldistance=None, autopct='%.0f%%')
ax[0].set_title(f'Layer 1-3\n n={all_l13}')
ax[1].pie([sig_l4, all_l4-sig_l4], autopct='%.0f%%')
ax[1].set_title(f'Layer 4\n n={all_l4}')
ax[2].pie([sig_l56, all_l56-sig_l56], autopct='%.0f%%')
ax[2].set_title(f'Layer 5-6 \n n={all_l56}')
fig.legend()
fig.suptitle("Significant spatially stable neurons")
plt.tight_layout()
plt.savefig('/auto/users/wingertj/data/hist_pie.pdf', dpi=400)

#significant spatial

f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.005})
sns.stripplot(x="layer", y="si", data=sig_df, ax=ax_top, alpha=0.3, zorder=1, dodge=True)
sns.stripplot(x="layer", y="si", data=sig_df, ax=ax_bottom, alpha=0.3, zorder=1, dodge=True)
sns.pointplot(x="layer", y="si", data=sig_df, ax=ax_bottom, estimator=np.mean, n_boot=1000, errorbar="se", capsize=0.3, color='black', join=False)
ax_top.set_ylim(bottom=3)   # those limits are fake
ax_bottom.set_ylim(0,3)

sns.despine(ax=ax_bottom)
sns.despine(ax=ax_top, bottom=True)

ax = ax_top
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

ax2 = ax_bottom
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
yticks = np.linspace(0, 3, 3)
ax2.set_yticks(yticks)
plt.savefig('/auto/users/wingertj/data/hist_si.pdf', dpi=400)

# g = sns.catplot(data=sig_df, x='layer', y='si', kind='strip', alpha=0.3, zorder=1, dodge=True)
# sns.pointplot(x="layer", y="si", data=sig_df, estimator=np.mean, n_boot=1000, errorbar="se", capsize=0.3, color='black', join=False)
# g.axes[0,0].set_ylabel("Spatial Information (bits)")

g1 = sns.catplot(data=sig_df, x='layer', y='scc', kind='strip', alpha=0.3, zorder=1, dodge=True)
sns.pointplot(x="layer", y="scc", data=sig_df, estimator=np.mean, n_boot=1000, errorbar="se", capsize=0.3, color='black', join=False)
g1.axes[0,0].set_ylabel("Spatial Correlation")
plt.savefig('/auto/users/wingertj/data/hist_scc.pdf', dpi=400)

# #all units
# ax2 = sns.catplot(data=hist_df, x='layer', y='si', kind='strip', alpha=0.3, zorder=1, dodge=True)
# sns.pointplot(x="layer", y="si", data=sig_df, estimator=np.mean, n_boot=1000, errorbar="se", capsize=0.3, markers="_", color='black', join=False, zorder=10, err_kws={'linewidth': 0.5})
#
# ax3 = sns.catplot(data=hist_df, x='layer', y='scc', kind='strip', alpha=0.3, zorder=1, dodge=True)
# sns.pointplot(x="layer", y="scc", data=sig_df, estimator=np.mean, n_boot=1000, errorbar="se", capsize=0.3,  markers="_", color='black', join=False, zorder=10, err_kws={'linewidth': 0.5})

fis, _ = interactive_spatial_scatter(sig_df, 'scc', 'si')
plt.savefig('/auto/users/wingertj/data/interactive_scatter.pdf', dpi=400)

# test peak location
f, ax = plt.subplots(5, 5)
ax = ax.flatten()
cell_plots = sig_df.loc[sig_df['layer'] == '56', 'cellid'].unique()
kernel = Gaussian2DKernel(x_size=5, y_size=5, x_stddev=3, y_stddev=3)
def filter_nan_gaussian_david(arr, sigma=3):
    """Allows intensity to leak into the nan area.
    According to Davids answer:
        https://stackoverflow.com/a/36307291/7128154
    """
    from scipy import ndimage
    gauss = arr.copy()
    gauss[np.isnan(gauss)] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)

    norm = np.ones(shape=arr.shape)
    norm[np.isnan(arr)] = 0
    norm = ndimage.gaussian_filter(
            norm, sigma=sigma, mode='constant', cval=0)

    # avoid RuntimeWarning: invalid value encountered in true_divide
    norm = np.where(norm==0, 1, norm)
    gauss = gauss/norm
    gauss[np.isnan(arr)] = np.nan
    return gauss

for i, cell_plot in enumerate(cell_plots[:25]):
    tc = sig_df[sig_df['cellid']==cell_plot]['tc'].values[0]
    tc_nan_mask = np.isnan(tc)
    tc_smoothed = filter_nan_gaussian_david(tc)
    tc_smoothed[tc_nan_mask] = np.nan
    imopts = {'cmap': 'viridis', 'interpolation': 'none', 'origin': 'lower'}
    ax[i].imshow(tc_smoothed, **imopts)
    # get max
    max_val = np.nanmax(tc_smoothed)
    max_loc = np.where(tc_smoothed == max_val)
    print(np.where(tc_smoothed==max_val))
    ax[i].scatter(max_loc[1], max_loc[0], color='red')
plt.savefig('/auto/users/wingertj/data/example_hist_maxes.pdf', dpi=400)
#plot rough guess of where cells max tuning curve values are located
def jitter_point(point, jitter):
    import random
    x, y = point
    x += random.uniform(-jitter, jitter)
    y += random.uniform(-jitter, jitter)
    return (x, y)

ft, axt = plt.subplots(1,1)
all_cells = sig_df['cellid'].unique()
for cell_plot in all_cells:
    tc = sig_df[sig_df['cellid']==cell_plot]['tc'].values[0]
    if (tc.shape[0] > 36):
        continue
    else:
        tc_nan_mask = np.isnan(tc)
        tc_smoothed = filter_nan_gaussian_david(tc)
        tc_smoothed[tc_nan_mask] = np.nan
        # get max
        max_val = np.nanmax(tc_smoothed)
        max_loc = np.where(tc_smoothed==max_val)
        jt = jitter_point((max_loc[0][0], max_loc[1][0]), jitter=0.5)
        axt.scatter(jt[0], jt[1], alpha=0.3, color='red')
yticks = np.arange(0, 30, 5)
xticks = np.arange(0, 40, 5)
axt.set_yticks(yticks)
axt.set_xticks(xticks)
plt.ylim(max(plt.ylim()), min(plt.ylim()))
axt.set_yticklabels(yticks[::-1]*20)
axt.set_xticklabels(xticks*20)
axt.set_ylabel("y pixels")
axt.set_xlabel("x pixels")
axt.set_title('Spatial tuning curve peaks')
plt.savefig('/auto/users/wingertj/data/all_hist_maxes.pdf', dpi=400)


bp = []