from nems.tools.json import save_model, load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import re
from matplotlib.cm import get_cmap

# plotting function
def decoder_scatter(df, siteids, target_chans):
    from matplotlib.cm import get_cmap
    import matplotlib.pyplot as plt

    cmap = get_cmap('hot')
    cmap1 = get_cmap('cool')
    cmap_grey =get_cmap('Greys')

    clr = np.linspace(0, 1, len(target_chans)+1)
    f, ax = plt.subplots(3, 1)
    axlabels = []
    for si, siteid in enumerate(siteids):
        site_df = df[df['siteid'] == siteid].copy()
        for li, tlyr in enumerate(site_df['tlyr'].unique()):
            for pi, tchan in enumerate(target_chans):
                if si == 0:
                    if li == 0:
                        axlabels.append(tchan)
                try:
                    respmean_ll = site_df.loc[
                        (site_df['tlyr'] == tlyr) & (site_df['rlyr'] == tlyr) & (site_df['isig'] == 'resp') & (
                                    site_df['tch'] == tchan)]['r_test'].values[0]
                    respmean_rl = site_df.loc[
                        (site_df['tlyr'] == tlyr) & (site_df['rlyr'] != tlyr) & (site_df['isig'] == 'resp') & (
                                    site_df['tch'] == tchan)]['r_test'].values[0].mean()

                    if siteid.startswith('PRN'):
                        ax[li].scatter(pi + 2 * pi, respmean_ll, label=tchan, color=cmap(clr[pi]), alpha=0.5)
                        ax[li].scatter(pi + 1 + 2 * pi, respmean_rl, label=tchan, color='grey', alpha=0.5)
                    else:
                        ax[li].scatter(pi, respmean_ll, label=tchan, color=cmap1(clr[pi]), alpha=0.5)
                        ax[li].scatter(pi + 1 + 2 * pi, respmean_rl, label=tchan, color='grey', alpha=0.5)
                except:
                    print(siteid, tlyr, tchan)
                    continue
            x_ticks = [i+2*i for i in range(len(axlabels))]
            ax[li].set_xticks(x_ticks)
            ax[li].set_xticklabels(axlabels)

    return f, ax

def model_comparison(df1, df2, siteids, target_chans, df_labels = ['df1', 'df2']):
    from matplotlib.cm import get_cmap
    import matplotlib.pyplot as plt

    cmap = get_cmap('hot')
    cmap1 = get_cmap('cool')
    cmap_grey =get_cmap('Greys')

    color_dict = {'13': 'green', '4': 'yellow', '56': 'blue'}

    clr = np.linspace(0, 1, len(target_chans)+1)
    f, ax = plt.subplots(1, len(target_chans))

    for si, siteid in enumerate(siteids):
        site_df1 = df1[df1['siteid'] == siteid].copy()
        site_df2 = df2[df2['siteid'] == siteid].copy()
        for li, tlyr in enumerate(site_df1['tlyr'].unique()):
            for pi, tchan in enumerate(target_chans):
                try:
                    respmean_ll1 = site_df1.loc[
                        (site_df1['tlyr'] == tlyr) & (site_df1['rlyr'] == tlyr) & (site_df1['isig'] == 'resp') & (
                                    site_df1['tch'] == tchan)]['r_test'].values[0]
                    respmean_ll2 = site_df2.loc[
                        (site_df2['tlyr'] == tlyr) & (site_df2['rlyr'] == tlyr) & (site_df2['isig'] == 'resp') & (
                                    site_df2['tch'] == tchan)]['r_test'].values[0]

                    ax[pi].scatter(respmean_ll1, respmean_ll2, color=color_dict[tlyr])
                    ax[pi].set_title(tchan)
                    ax[pi].set_xlabel(df_labels[0])
                    ax[pi].set_ylabel(df_labels[1])
                    ax[pi].set_box_aspect(1)
                    xvals = [0, 0.25, 0.5, 0.75, 1.0]
                    yvals = [0, 0.25, 0.5, 0.75, 1.0]
                    ax[pi].plot(xvals, yvals, color='black', linestyle='--')
                except:
                    print(siteid, tlyr, tchan)
                    continue
    return f, ax


full_model_save_path = Path('/auto/users/wingertj/models/spatial_decoding/')
model_save_path = Path('/auto/users/wingertj/models/decoding_layer_removal')

# make giant dataframe
full_model = pd.read_pickle(str(full_model_save_path/'decoder_df.pkl'))
model_df = pd.read_pickle(str(model_save_path/'decoder_df.pkl'))
all_df = pd.concat([full_model, model_df],axis=0).reset_index(drop=True)
# add cellnumber to dataframe
all_df['cellnum'] = [len(all_df['cellids'].values[i]) for i in range(len(all_df['cellids'].values))]

# pick channels of interest
target_chans = ['d', 'theta', 'v', 'front_x', 'front_y']

# #filter out sites with less than 10 cells
# high_cell_df = all_df.loc[(all_df['cellnum']>10) & (all_df['fircaus']=='True')].reset_index(drop=True).copy()
# high_cell_ids = high_cell_df['siteid'].unique()
# # target_chans = model_df['tch'].unique()
#
# decoder_scatter(high_cell_df, high_cell_ids, target_chans)

#filter out sites with less than 10 cells
anti_causal_df = all_df.loc[(all_df['cellnum']>=10) & (all_df['fircaus']=='True')].reset_index(drop=True).copy()
anti_causal_ids = anti_causal_df['siteid'].unique()
# target_chans = model_df['tch'].unique()

#filter out sites with less than 10 cells
causal_df = all_df.loc[(all_df['cellnum']>=10) & (all_df['fircaus']=='') & (all_df['tlyr']!='rlyr')].reset_index(drop=True).copy()
causal_ids = causal_df['siteid'].unique()
# target_chans = model_df['tch'].unique()

#full causal model
full_causal_df = all_df.loc[(all_df['fircaus']=='') & (all_df['tlyr']!='rlyr')].reset_index(drop=True).copy()
full_causal_ids = full_causal_df['siteid'].unique()

#shared sites between anticausal and causal filters
shared_ids = list(set(anti_causal_ids).intersection(set(causal_ids)))
f, ax = decoder_scatter(causal_df, shared_ids, target_chans)
f.suptitle('causal')
f1, ax1 = decoder_scatter(anti_causal_df, shared_ids, target_chans)
f1.suptitle('anti-causal')

# quick comparison of causal and anti-causal models
model_comparison(causal_df, anti_causal_df, shared_ids, target_chans, df_labels=['causal', 'anti-causal'])


bp = []
