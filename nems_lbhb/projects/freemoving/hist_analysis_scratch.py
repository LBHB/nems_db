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

    f, ax = plt.subplots(1,3, sharex=True)
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
    imopts = {'cmap': 'viridis', 'interpolation': 'none', 'origin': 'lower'}
    ax[0].imshow(tc1, **imopts)
    ax[0].set_title('Split 1')
    ax[1].imshow(tc2, **imopts)
    ax[1].set_title('Split 2')
    ax[2].imshow(tc, **imopts)
    ax[2].set_title('All data')

    f.suptitle(f'ID: {nearest_cell_id[0]} \n layer: {layer}, SC: {np.round(sc, decimals=2)}, SI: {np.round(si, decimals=2)}')
    f.tight_layout()

def interactive_spatial_scatter(df, x_stat, y_stat):
    f, ax = plt.subplots(1, 3, sharey=True, sharex=True)
    layer_13_cellids = df.loc[df['layer'] == '13', 'cellid'].unique()
    ax[0].scatter([df.loc[(df['layer'] == '13') & (df['cellid'] == cell), x_stat].values[0] for cell in
                   layer_13_cellids],
                  [df.loc[(df['layer'] == '13') & (df['cellid'] == cell), y_stat].values[0] for cell in
                   layer_13_cellids], color='green')
    ax[0].set_title('Layer 13')
    layer_4_cellids = df.loc[df['layer'] == '44', 'cellid'].unique()
    ax[1].scatter([df.loc[(df['layer'] == '44') & (df['cellid'] == cell), x_stat].values[0] for cell in
                   layer_4_cellids],
                  [df.loc[(df['layer'] == '44') & (df['cellid'] == cell), y_stat].values[0] for cell in
                   layer_4_cellids], color='orange')
    ax[1].set_title('Layer 44')
    layer_56_cellids = df.loc[df['layer'] == '56', 'cellid'].unique()
    ax[2].scatter([df.loc[(df['layer'] == '56') & (df['cellid'] == cell), x_stat].values[0] for cell in
                   layer_56_cellids],
                  [df.loc[(df['layer'] == '56') & (df['cellid'] == cell), y_stat].values[0] for cell in
                   layer_56_cellids], color='blue')
    ax[2].set_title('Layer 56')
    ax[0].set_ylabel('Spatial Information (bits)')
    ax[0].set_xlabel('Spatial Correlation')
    ax[1].set_xlabel('Spatial Correlation')
    ax[2].set_xlabel('Spatial Correlation')
    cid = f.canvas.mpl_connect('button_press_event', partial(onclick, {'df': df, 'x':x_stat, 'y':y_stat}))


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

ax = sns.catplot(data=sig_df, x='layer', y='si', kind='strip', alpha=0.5, zorder=1)
sns.pointplot(x="layer", y="si", data=sig_df, errorbar="se", capsize=0.3, markers="_", color='black', join=False, zorder=10, err_kws={'linewidth': 0.5})

ax1 = sns.catplot(data=sig_df, x='layer', y='scc', kind='strip', zorder=1, dodge=True)
sns.pointplot(x="layer", y="scc", data=sig_df, errorbar="se", capsize=0.3,  markers="_", color='black', join=False, zorder=10)

interactive_spatial_scatter(sig_df, 'scc', 'si')

# test peak location
f, ax = plt.subplots(5, 5)
ax = ax.flatten()
cell_plots = sig_df.loc[sig_df['layer'] == '56', 'cellid'].unique()
kernel = Gaussian2DKernel(x_size=5, y_size=5, x_stddev=3, y_stddev=3)
for i, cell_plot in enumerate(cell_plots[:25]):
    tc1 = sig_df[sig_df['cellid']==cell_plot]['tc1'].values[0]
    tc1_nan_mask = np.isnan(tc1)
    tc1_smoothed = convolve(tc1, kernel)
    tc1_smoothed[tc1_nan_mask] = np.nan
    threshold = np.nanmax(tc1_smoothed)*0.7
    imopts = {'cmap': 'viridis', 'interpolation': 'none', 'origin': 'lower'}
    ax[i].imshow(tc1_smoothed, **imopts)
    ct = ax[i].contour(tc1_smoothed, levels=[threshold], colors='black')
    largest_peaks, all_peaks = detect_peaks(tc1_smoothed, neighbors=[7,7], threshold=0.7)
    peaks = list(largest_peaks.keys())
    # for peak in peaks:
    #     ax[i].scatter(largest_peaks[peak][1], largest_peaks[peak][0], color='red')

bp = []