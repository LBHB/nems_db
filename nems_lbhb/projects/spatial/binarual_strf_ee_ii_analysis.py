import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nems0 import db
from nems0.xform_helper import load_model_xform, fit_model_xform

from nems.models import LN
from matplotlib.cm import get_cmap
from pathlib import Path
from functools import partial

log = logging.getLogger(__name__)

data_save_path = Path('/auto/users/wingertj/data/')
df_label = '.strf_analysis.pkl'
### functions ###
def site_strf_plot(df, cellids, statname, num, showmask=True, showsitename=True, save=False, savepath=None, saveopts=None):
    from matplotlib import colors
    # divnorm = colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=10)
    # pcolormesh(your_data, cmap="coolwarm", norm=divnorm)
    for cellid in cellids:
        sitedf = df[df['cell_id'] == cellid]
        f, ax = plt.subplots(num, (2+showmask), sharex=True, sharey=True)
        min_stat = sitedf[statname].values.min()
        max_stat = sitedf[statname].values.max()
        stat_range = np.linspace(min_stat, max_stat, num)
        # cmaxs = []
        # cmins = []
        # for cnum, stat_near in enumerate(stat_range):
        #     stat_index = np.abs(sitedf[statname].values - stat_near).argmin()
        #     cellname = sitedf['cell_ids'].values[stat_index]
        #     cmins.append(sitedf[sitedf['cell_ids'] == cellname]['c_strf'].values[0].min())
        #     cmaxs.append(sitedf[sitedf['cell_ids'] == cellname]['c_strf'].values[0].max())
        # cmax = max(cmaxs)
        # cmin = max(cmins)
        # imopts = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'None', 'cmap': 'bwr', 'vmin': cmin, 'vmax':cmax}
        # imopts = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'None', 'cmap': 'bwr', 'extent':[0, 10*20, 200, 20000]}
        yticks = np.linspace(200, 20000, 3)
        ylabels = [0.2, 2, 20]
        xticks = np.linspace(0,10*20, 5)
        for cnum, stat_near in enumerate(stat_range):
            stat_index = np.abs(sitedf[statname].values - stat_near).argmin()
            stat_value = sitedf[statname].values[stat_index]
            cellname = sitedf['cell_ids'].values[stat_index]
            r_test = sitedf['r_test'].values[stat_index]
            vmin = min([np.nanmin(sitedf[sitedf['cell_ids'] == cellname]['c_strf'].values[0]),
                        np.nanmin(sitedf[sitedf['cell_ids'] == cellname]['i_strf'].values[0])])
            vmax = max([np.nanmax(sitedf[sitedf['cell_ids'] == cellname]['c_strf'].values[0]),
                        np.nanmax(sitedf[sitedf['cell_ids'] == cellname]['i_strf'].values[0])])
            divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
            imopts = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'None', 'cmap': 'bwr',
                      'extent': [0, 10 * 20, 200, 20000], 'norm':divnorm}
            ax[cnum, 0].imshow(sitedf[sitedf['cell_ids'] == cellname]['c_strf'].values[0], **imopts)
            ax[cnum, 0].set_yticks(yticks)
            ax[cnum, 0].set_xticks(xticks)
            ax[cnum, 0].set_yticklabels(ylabels)
            ax[cnum, 0].set_ylabel("frequency (kHz)")
            ax[0, 0].set_title('Contra', fontsize=12)
            # ax[cnum, 1].annotate(f"signed %: {np.round(stat_value, decimals=2)}", xy=(100, 17000), fontsize=12, weight='bold')
            ax[cnum, 1].annotate(f"r: {np.round(r_test, decimals=2)}", xy=(100, 13000), fontsize=12, weight='bold')
            ax[cnum, 1].imshow(sitedf[sitedf['cell_ids'] == cellname]['i_strf'].values[0], **imopts)
            ax[cnum, 1].set_yticks(yticks)
            ax[cnum, 1].set_xticks(xticks)
            ax[cnum, 1].set_yticklabels(ylabels)
            ax[0, 1].set_title('Ipsi', fontsize=12)
            if showmask:
                ax[cnum, 2].imshow(sitedf[sitedf['cell_ids'] == cellname]['mask'].values[0], **imopts)
        ax[cnum, 1].set_xlabel("time lag (ms)")
        ax[cnum, 0].set_xlabel("time lag (ms)")
        if showsitename:
            f.suptitle(cellid[:7])
        plt.tight_layout()
        if save:
            f.savefig(savepath/f'{cellid[:7]}_binaural_strf_examples.pdf', **saveopts)

def onclick(xy, event):
    from matplotlib import colors
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    f, ax = plt.subplots(3,1, sharex=True)
    imopts = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'None', 'cmap': 'bwr',
              'extent': [0, 10 * 20, 200, 20000]}
    yticks = np.linspace(200, 20000, 3)
    ylabels = [0.2, 2, 20]
    xticks = np.linspace(0, 10 * 20, 5)
    x_stat = xy['x']
    y_stat = xy['y']
    # closest_x = rank_df[rank_df['total_sign_change'].between((event.xdata-0.1), (event.xdata+0.1), inclusive=False)]['cell_ids']
    closest_x = rank_df[rank_df[x_stat].between((event.xdata - 0.1), (event.xdata + 0.1), inclusive=False)][
        'cell_ids']
    # nearest_cell_i = np.abs(rank_df[rank_df['cell_ids'].isin(closest_x)]['contra_epercent'].values - event.ydata).argmin()
    nearest_cell_i = np.abs(rank_df[rank_df['cell_ids'].isin(closest_x)][y_stat].values - event.ydata).argmin()
    nearest_cell_id = [rank_df[rank_df['cell_ids'].isin(closest_x)]['cell_ids'].values[nearest_cell_i]]
    max_val = max([np.abs(rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['c_strf'].values[0]).max(),
                   np.abs(rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['i_strf'].values[0]).max(),
                  np.abs(rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['c_strf'].values[0]-rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['i_strf'].values[0]).max()])
    # min_val = min([rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['c_strf'].values[0].min(),
    #                rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['i_strf'].values[0].min()])
    # try:
    #     divnorm = colors.TwoSlopeNorm(vmin=min_val, vcenter=0., vmax=max_val)
    # except:
    #     divnorm = colors.Normalize(vmin=min_val, vmax=max_val)
    # imopts = {'origin': 'lower', 'interpolation': 'None', 'cmap': 'bwr'}
    cstrf = rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['c_strf'].values[0]
    istrf = rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['i_strf'].values[0]
    # cmask = np.invert(rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['econtra_masks'].values[0]|rank_df[rank_df['cell_ids'].isin(nearest_cell_id)]['icontra_masks'].values[0])
    # cstrf[cmask] = np.nan
    # istrf[cmask] = np.nan
    ax[0].imshow(cstrf, **imopts, vmax=max_val, vmin=-max_val)
    f.suptitle(f"{nearest_cell_id[0]} \n {x_stat}: {np.round(rank_df[rank_df['cell_ids']==nearest_cell_id[0]][x_stat].values[0], decimals=3)}\n"
                    f"{y_stat}: {np.round(rank_df[rank_df['cell_ids']==nearest_cell_id[0]][y_stat].values[0], decimals=3)}")
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(ylabels)
    ax[0].set_ylabel("frequency (kHz)")
    ax[0].set_title("Contra")
    ax[1].imshow(istrf, **imopts, vmax=max_val, vmin=-max_val)
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(ylabels)
    ax[1].set_ylabel("frequency (kHz)")
    ax[1].set_title("Ipsi")
    ax[2].imshow(cstrf-istrf, **imopts, vmax=max_val, vmin=-max_val)
    ax[2].set_yticks(yticks)
    ax[2].set_xticks(xticks)
    ax[2].set_yticklabels(ylabels)
    ax[2].set_xlabel("time lag (ms)")
    ax[2].set_ylabel("frequency (kHz)")
    ax[2].set_title("Difference")
    plt.tight_layout()

def interactive_strf_scatter(df, cellids, x_stat, y_stat):
    f, ax = plt.subplots(1,1)
    clrs = get_cmap('turbo')
    clr_space = np.linspace(0,1, len(cellids))
    for sitenum, cellid in enumerate(cellids):
        site_df = df[rank_df['cell_id'] ==cellid]
        ax.scatter(site_df[x_stat].values, site_df[y_stat].values, color=clrs(clr_space[sitenum]))
    cid = f.canvas.mpl_connect('button_press_event', partial(onclick, {'x':x_stat, 'y':y_stat}))
    ax.set_ylabel(y_stat)
    ax.set_xlabel(x_stat)

def strf_scatter_stat_highlight(df, cellids, x_stat, y_stat, hightlight=True, savepath = '', savefig={}):
    f, ax = plt.subplots(1, 1)
    site_mean_dict = {}
    if hightlight == True:
        for sitenum, cellid in enumerate(cellids):
            site_df = df[rank_df['cell_id'] == cellid]
            site_mean_xstat = np.nanmean(site_df[x_stat].values)
            site_mean_dict[site_mean_xstat] = cellid
        # find random high x stat site and random low x stat site to highlight
        site_means = np.array(list(site_mean_dict.keys()))
        site_means = site_means[~np.isnan(site_means)]
        qts = np.percentile(site_means, [25, 50, 75])
        high_vals = site_means[site_means > qts[2]]
        low_vals = site_means[site_means < qts[0]]
        rand_high = site_mean_dict[high_vals[np.random.randint(0, len(high_vals), size=1)[0]]]
        rand_low = site_mean_dict[low_vals[np.random.randint(0, len(low_vals), size=1)[0]]]
    else:
        rand_high = 'None'
        rand_low = 'None'
    for sitenum, cellid in enumerate(cellids):
        site_df = df[rank_df['cell_id'] == cellid]
        if cellid == rand_high:
            continue
            # ax.scatter(site_df[x_stat].values, site_df[y_stat].values, color='red')
        elif cellid == rand_low:
            continue
            # ax.scatter(site_df[x_stat].values, site_df[y_stat].values, color='blue')
        else:
            ax.scatter(site_df[x_stat].values, site_df[y_stat].values, color='#ebe9e8')
    # plot high and low values
    if hightlight == True:
        site_df = df[rank_df['cell_id'] == rand_high]
        ax.scatter(site_df[x_stat].values, site_df[y_stat].values, color='red')
        site_df = df[rank_df['cell_id'] == rand_low]
        ax.scatter(site_df[x_stat].values, site_df[y_stat].values, color='blue')
    cid = f.canvas.mpl_connect('button_press_event', partial(onclick, {'x': x_stat, 'y': y_stat}))
    ax.set_ylabel(y_stat)
    ax.set_xlabel(x_stat)
    ax.set_title("Binaural STRF interactions")
    if savefig:
        f.savefig(savepath, **savefig)

def df_perm_test(df, stat, cellids, permnum=1000):
    # true test statistic
    f, ax = plt.subplots(1, 1)
    true_mean_stat_var = []
    # true_mean_var_meancontra = []
    # true_mean_diffsum = []
    for rank in ranks:
        rank_df = df.copy()
        site_stat_var = []
        site_meancontra_var = []
        site_diffsum_var = []
        for sitenum, cellid in enumerate(cellids):
            site_df = rank_df[rank_df['cell_id'] == cellid]
            cell_n = len(site_df['diff'].values)
            if cell_n < 5:
                continue
            site_stat_var.append(np.var(site_df[stat].values))
            # site_meancontra_var.append(np.var(site_df['mean(contra)/mean(abs(contra))'].values))
            # site_diffsum_var.append(np.var(site_df['diffsum_ratio'].values))
        true_mean_stat_var.append(np.mean(site_stat_var))
        # true_mean_var_meancontra.append(np.mean(site_meancontra_var))
        # true_mean_diffsum.append(np.mean(site_diffsum_var))

    perm_stat_var = []
    # perm_meancotnra_var = []
    # perm_diffsum_var = []
    for permutenum in range(permnum):
        for rank in ranks:
            rank_df = df.copy()
            # permute site_id
            rank_df['cell_id'] = rank_df['cell_id'].sample(frac=1).reset_index(drop=True)
            site_stat_var = []
            # site_cisum_var = []
            # site_diffsum_var = []
            for sitenum, cellid in enumerate(cellids):
                site_df = rank_df[rank_df['cell_id'] ==cellid]
                cell_n = len(site_df['diff'].values)
                if cell_n < 5:
                    continue
                site_stat_var.append(np.var(site_df[stat].values))
                # site_meancontra_var.append(np.var(site_df['mean(contra)/mean(abs(contra))'].values))
                # site_diffsum_var.append(np.var(site_df['diffsum_ratio'].values))
            perm_stat_var.append(np.mean(site_stat_var))
            # perm_meancotnra_var.append(np.mean(site_meancontra_var))
            # perm_diffsum_var.append(np.mean(site_diffsum_var))
    ax.hist(perm_stat_var, bins=30)
    ax.axvline(true_mean_stat_var[0])
    ax.set_title(f"Mean site variance of {stat}")

# binaural_strf_df = pd.read_pickle(data_save_path/'binaural_strf_df_0.pkl')
binaural_strf_df = pd.read_pickle(data_save_path/'binaural_strf_df_nan.pkl')
r_test_thresh = 0.2
binaural_strf_df = binaural_strf_df[binaural_strf_df['r_test'] > r_test_thresh]
ranks = list(set(binaural_strf_df['rank']))
ranks.sort()
cellids = list(set(binaural_strf_df['cell_id']))
f, ax = plt.subplots(6, len(ranks))
bwr = get_cmap('bwr')
stat_name = 'masked signed(i+c)/(Ni+Nc)'
# stat_name = 'masked signed(i+c)/(Ni+Nc)'
if stat_name == 'masked diffsum_ratio':
    other_stat_name = 'masked signed(i+c)/(Ni+Nc)'
    dos_thresh = 1
elif stat_name == 'masked signed(i+c)/(Ni+Nc)':
    other_stat_name = 'masked diffsum_ratio'
    dos_thresh = 0
# for col, rank in enumerate(ranks):
#     rank_df = binaural_strf_df[binaural_strf_df['rank'] == rank]
#     diff_sum = np.log2(rank_df['diff'].values/rank_df['sum'].values)
#     ax[0, col].hist(diff_sum, bins=20)
#     ax[0, col].set_ylabel("counts")
#     ax[0, col].set_title(f"rank {rank}")
#     ax[0, col].set_xlabel("std(contra strf - ipsi strf)\n/std(contra strf + ipsi strf)")
#     ax[0, col].axvline(x=np.log2(dos_thresh), color='black')
#     # masked
#     diff_sum_masked = np.log2(rank_df['masked diff'].values/rank_df['masked sum'].values)
#     ax[3, col].hist(diff_sum_masked, bins=20)
#     ax[3, col].set_ylabel("counts")
#     ax[3, col].set_xlabel("std(contra strf - ipsi strf)\n/std(contra strf + ipsi strf)")
#     ax[3, col].axvline(x=np.log2(dos_thresh), color='black')
#     ax[0, col].sharex(ax[3, col])
#     site_ic_means = []
#     site_diffsum_mags = []
#     masked_site_ic_means = []
#     masked_site_diffsum_mags = []
#     for cellid in cellids[:15]:
#         site_df = rank_df[rank_df['cell_id'] ==cellid]
#         cell_n = len(site_df['diff'].values)
#         if cell_n < 1:
#             continue
#         max_stat = site_df[stat_name].values.max()
#         min_stat = site_df[stat_name].values.min()
#         near_stats = [max_stat, dos_thresh, min_stat]
#         if rank == 16:
#             f2, ax2 = plt.subplots(len(near_stats), 2)
#             for cnum, val in enumerate(near_stats):
#                 stat_index = np.abs(site_df[stat_name].values - val).argmin()
#                 stat_value = site_df[stat_name].values[stat_index]
#                 cellname = site_df['cell_ids'].values[stat_index]
#                 ax2[cnum, 0].imshow(site_df[site_df['cell_ids']==cellname]['c_strf'].values[0], origin='lower', aspect='auto')
#                 ax2[cnum, 0].set_title(f"{stat_name}: {stat_value}")
#                 ax2[cnum, 1].imshow(site_df[site_df['cell_ids']==cellname]['i_strf'].values[0], origin='lower', aspect='auto')
#             plt.tight_layout()
#         site_ic_mean = np.mean(site_df['i_strf_std'].values/site_df['c_strf_std'].values)
#         site_ic_means.append(site_ic_mean)
#         site_ic_std = np.std(site_df['i_strf_std'].values/site_df['c_strf_std'].values)
#         # site_diffsum_mag = sum(site_df['diff'].values/site_df['sum'].values > dos_thresh)/len(site_df['diff'].values)
#         site_diffsum_mag = sum(site_df[stat_name].values > dos_thresh) / len(
#             site_df[stat_name].values)
#         site_diffsum_mags.append(site_diffsum_mag)
#         ax[1, col].scatter(site_ic_mean, site_ic_std, c=bwr(site_diffsum_mag), s=6)
#         ax[1, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")
#         ax[1, col].set_ylabel("std across cells")
#         #masked
#         masked_site_ic_mean = np.mean(site_df['masked i_strf_std'].values/site_df['masked c_strf_std'])
#         masked_site_ic_means.append(masked_site_ic_mean)
#         masked_site_ic_std = np.std(site_df['masked i_strf_std'].values/site_df['masked c_strf_std'])
#         # masked_site_diffsum_mag = sum(site_df['masked diff'].values/site_df['masked sum'].values > dos_thresh)/len(site_df['masked diff'].values)
#         masked_site_diffsum_mag  = sum(site_df[stat_name].values > dos_thresh) / len(
#             site_df[stat_name].values)
#         masked_site_diffsum_mags.append(masked_site_diffsum_mag)
#         ax[4, col].scatter(masked_site_ic_mean, masked_site_ic_std, c=bwr(masked_site_diffsum_mag), s=6)
#         ax[4, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")
#         ax[4, col].set_ylabel("std across cells")
#     site_ic_means_low = [sicm for sicm, sdsm in zip(site_ic_means, site_diffsum_mags) if sdsm < 0.5]
#     site_ic_means_high = [sicm for sicm, sdsm in zip(site_ic_means, site_diffsum_mags) if sdsm > 0.5 ]
#     masked_site_ic_means_low = [sicm for sicm, sdsm in zip(masked_site_ic_means, masked_site_diffsum_mags) if sdsm < 0.5]
#     masked_site_ic_means_high = [sicm for sicm, sdsm in zip(masked_site_ic_means, masked_site_diffsum_mags) if sdsm > 0.5 ]
#     ax[2, col].hist(site_ic_means_low, bins=10, color='blue', alpha=0.5)
#     ax[2, col].hist(site_ic_means_high, bins=10, color='red', alpha=0.5)
#     ax[2, col].set_ylabel("counts")
#     ax[2, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")
#     ax[5, col].hist(masked_site_ic_means_low, bins=10, color='blue', alpha=0.5)
#     ax[5, col].hist(masked_site_ic_means_high, bins=10, color='red', alpha=0.5)
#     ax[5, col].set_ylabel("counts")
#     ax[5, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")

ranks = [16]
for col, rank in enumerate(ranks):
    rank_df = binaural_strf_df[binaural_strf_df['rank'] == rank]
    f_, ax_ = plt.subplots(1,1)
    f_2, ax_2 = plt.subplots(1, 1)
    interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='sum(abs(contra-ipsi))/sum(abs(contra))', y_stat='mean(contra)/mean(abs(contra))')
    interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='sum(abs(contra-ipsi))/sum(abs(contra))', y_stat='mean(contra)/mean(abs(contra))')
    interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='std(hcontra-hipsi)/(std(hcontra)+std(hipsi))', y_stat='(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))')
    interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))', y_stat='mean(contra)/mean(abs(contra))')
    tmp_fig_savepath = data_save_path/'binaural_strf_highlight_std_cminusi.pdf'
    saveopts = {'dpi': 600, 'format': 'pdf'}
    strf_scatter_stat_highlight(df=rank_df, cellids=cellids, x_stat='std(hcontra-hipsi)/(std(hcontra)+std(hipsi))', y_stat='mean(contra)/mean(abs(contra))', savepath=tmp_fig_savepath, savefig=saveopts)
    tmp_fig_savepath = data_save_path / 'binaural_strf_highlight_stdc_minus_stdi.pdf'
    strf_scatter_stat_highlight(df=rank_df, cellids=cellids, x_stat='(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))', y_stat='mean(contra)/mean(abs(contra))', savepath=tmp_fig_savepath, savefig=saveopts)
    tmp_fig_savepath = data_save_path/'binaural_strf_nohighlight_std_cminusi.pdf'
    saveopts = {'dpi': 600, 'format': 'pdf'}
    strf_scatter_stat_highlight(df=rank_df, cellids=cellids, x_stat='std(hcontra-hipsi)/(std(hcontra)+std(hipsi))', y_stat='mean(contra)/mean(abs(contra))', hightlight=False, savepath=tmp_fig_savepath, savefig=saveopts)
    tmp_fig_savepath = data_save_path / 'binaural_strf_nohighlight_stdc_minus_stdi.pdf'
    strf_scatter_stat_highlight(df=rank_df, cellids=cellids, x_stat='(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))', y_stat='mean(contra)/mean(abs(contra))', hightlight=False, savepath=tmp_fig_savepath, savefig=saveopts)
    # interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='std(hcontra-hipsi)/(std(hcontra)+std(hipsi))', y_stat='mean(contra)/mean(abs(contra))')
    # interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='std(hcontra-hipsi)/(std(hcontra)+std(hipsi))', y_stat='ic_ratio')
    # interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))',
    #                          y_stat='ic_ratio')
    # interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='mean(contra)/mean(abs(contra))',
    #                          y_stat='ic_ratio')
    # interactive_strf_scatter(df=rank_df, cellids=cellids, x_stat='mean(contra)/mean(abs(contra))',
    #                          y_stat='(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))')

    # clrs = get_cmap('turbo')
    # clr_space = np.linspace(0,1, len(cellids))
    # f_3, ax_3 = plt.subplots(1, 1)
    # for sitenum, cellid in enumerate(cellids):
    #     site_df = rank_df[rank_df['cell_id'] ==cellid]
    #     ax_2.scatter(site_df['sum(abs(contra-ipsi))/sum(abs(contra))'].values, site_df['mean(contra)/mean(abs(contra))'].values, color=clrs(clr_space[sitenum]))
    #
    #     ax_3.scatter(site_df['sum(abs(contra-ipsi))/sum(abs(contra+hipsi))'].values, site_df['mean(contra)/mean(abs(contra))'].values, color=clrs(clr_space[sitenum]))
    # cid2 = f_2.canvas.mpl_connect('button_press_event', partial(onclick, {'x':'sum(abs(contra-ipsi))/sum(abs(contra))', 'y':'mean(contra)/mean(abs(contra))'}))
    # cid3 = f_3.canvas.mpl_connect('button_press_event', partial(onclick, {'x':'sum(abs(contra-ipsi))/sum(abs(contra+hipsi))', 'y':'mean(contra)/mean(abs(contra))'}))

    # flip_states = [0.0, 1.0]
    # for flip_state in flip_states:
    #     total_flips = rank_df[rank_df['total_sign_change'].values == flip_state]['cell_ids'].values
    #     row_col = int(np.ceil(len(total_flips)**0.5))
    #     f_2, ax_2 = plt.subplots(row_col, row_col)
    #     ax_2s = ax_2.flatten()
    #     f_3, ax_3 = plt.subplots(row_col, row_col)
    #     ax_3s = ax_3.flatten()
    #     for cn, flipped_cell in enumerate(total_flips):
    #         min_val = np.min([rank_df[rank_df['cell_ids'] == flipped_cell]['c_strf'].values[0].min(),
    #                           rank_df[rank_df['cell_ids'] == flipped_cell]['i_strf'].values[0].min()])
    #         max_val = np.max([rank_df[rank_df['cell_ids'] == flipped_cell]['c_strf'].values[0].max(),
    #                           rank_df[rank_df['cell_ids'] == flipped_cell]['i_strf'].values[0].max()])
    #         ax_2s[cn].imshow(rank_df[rank_df['cell_ids'] == flipped_cell]['c_strf'].values[0], vmax=max_val, vmin=min_val)
    #         ax_3s[cn].imshow(rank_df[rank_df['cell_ids'] == flipped_cell]['i_strf'].values[0], vmax=max_val, vmin=min_val)

    # cellids = list(set(rank_df['cell_id']))
    # if stat_name == 'masked diffsum_ratio':
    #     stat_vals = np.log2(rank_df[stat_name].values)
    #     # masked
    #     masked_stat_values = np.log2(rank_df[stat_name].values)
    #     x_thresh = np.log2(dos_thresh)
    # else:
    #     stat_vals = rank_df[stat_name].values
    #     masked_stat_values = rank_df[stat_name].values
    #     x_thresh = dos_thresh
    # ax[0, col].hist(stat_vals, bins=20)
    # ax[0, col].set_ylabel("counts")
    # ax[0, col].set_title(f"rank {rank}")
    # ax[0, col].set_xlabel(stat_name)
    # ax[0, col].axvline(x=x_thresh, color='black')
    # # masked
    # ax[3, col].hist(masked_stat_values, bins=20)
    # ax[3, col].set_ylabel("counts")
    # ax[3, col].set_xlabel(stat_name)
    # ax[3, col].axvline(x=x_thresh, color='black')
    # ax[0, col].sharex(ax[3, col])
    # max_stat = rank_df[stat_name].values.max()
    # min_stat = rank_df[stat_name].values.min()
    # near_stats = [max_stat, dos_thresh, min_stat]
    # # if rank == 16:
    # #     f2, ax2 = plt.subplots(len(near_stats), 2)
    # #     for cnum, val in enumerate(near_stats):
    # #         stat_index = np.abs(rank_df[stat_name].values - val).argmin()
    # #         stat_value = rank_df[stat_name].values[stat_index]
    # #         other_stat = rank_df[other_stat_name].values[stat_index]
    # #         cellname = rank_df['cell_ids'].values[stat_index]
    # #         ax2[cnum, 0].imshow(rank_df[rank_df['cell_ids'] == cellname]['c_strf'].values[0], origin='lower',
    # #                             aspect='auto')
    # #         ax2[cnum, 0].set_title(f"{cellname}\n{stat_name}: {stat_value}\n{other_stat_name}: {other_stat}")
    # #         ax2[cnum, 1].imshow(rank_df[rank_df['cell_ids'] == cellname]['i_strf'].values[0], origin='lower',
    # #                             aspect='auto')
    # #     plt.tight_layout()
    # site_ic_means = []
    # site_diffsum_mags = []
    # masked_site_ic_means = []
    # masked_site_diffsum_mags = []
    # # f3, ax3 = plt.subplots(len(cellids),1)
    # hist_bins = np.linspace(-1, 1, 20, endpoint=False)
    # pop_stat_std = np.std(rank_df[stat_name].values)
    # site_ee_percents = {}
    # site_stat_mean = []
    # site_stat_dict = {}
    # site_cisum = []
    # site_hsumhdiff_mean = []
    # site_hsumhdiff_mean_std = []
    # site_cisum_std = []
    # site_meancontra = []
    # site_meancontra_std = []
    # for sitenum, cellid in enumerate(cellids):
    #     site_df = rank_df[rank_df['cell_id'] ==cellid]
    #     cell_n = len(site_df['diff'].values)
    #     if cell_n < 1:
    #         continue
    #
    #     site_hsumhdiff_mean.append(np.mean(site_df['sum(abs(contra-ipsi))/sum(abs(contra+hipsi))'].values))
    #     site_hsumhdiff_mean_std.append(np.std(site_df['sum(abs(contra-ipsi))/sum(abs(contra+hipsi))'].values))
    #     site_cisum.append(np.mean(site_df['sum(abs(contra-ipsi))/sum(abs(contra))'].values))
    #     site_cisum_std.append(np.std(site_df['sum(abs(contra-ipsi))/sum(abs(contra))'].values))
    #     site_meancontra.append(np.mean(site_df['mean(contra)/mean(abs(contra))'].values))
    #     site_meancontra_std.append(np.std(site_df['mean(contra)/mean(abs(contra))'].values))
    #     percent_ee = sum(site_df[stat_name].values > pop_stat_std)/len(site_df[stat_name].values)
    #     site_ee_percents[percent_ee] = cellid
    #     site_stat_mean.append(site_df[stat_name].values.mean())
    #     site_stat_dict[site_df[stat_name].values.mean()] = cellid
    #     # ax3[sitenum].hist(site_df[stat_name].values, bins=hist_bins, alpha=0.6)
    #     # max_stat = site_df[stat_name].values.max()
    #     # min_stat = site_df[stat_name].values.min()
    #     # near_stats = [max_stat, dos_thresh, min_stat]
    #     # if rank == 16:
    #     #     f2, ax2 = plt.subplots(len(near_stats), 2)
    #     #     for cnum, val in enumerate(near_stats):
    #     #         stat_index = np.abs(site_df[stat_name].values - val).argmin()
    #     #         stat_value = site_df[stat_name].values[stat_index]
    #     #         cellname = site_df['cell_ids'].values[stat_index]
    #     #         ax2[cnum, 0].imshow(site_df[site_df['cell_ids']==cellname]['c_strf'].values[0], origin='lower', aspect='auto')
    #     #         ax2[cnum, 0].set_title(f"{stat_name}: {stat_value}")
    #     #         ax2[cnum, 1].imshow(site_df[site_df['cell_ids']==cellname]['i_strf'].values[0], origin='lower', aspect='auto')
    #     #     plt.tight_layout()
    #     # max_stat = site_df[stat_name].values.max()
    #     # min_stat = site_df[stat_name].values.min()
    #     # near_stats = [max_stat, dos_thresh, min_stat]
    #     # if rank == 16:
    #     #     f2, ax2 = plt.subplots(len(near_stats), 2)
    #     #     for cnum, val in enumerate(near_stats):
    #     #         stat_index = np.abs(site_df[stat_name].values - val).argmin()
    #     #         stat_value = site_df[stat_name].values[stat_index]
    #     #         cellname = site_df['cell_ids'].values[stat_index]
    #     #         ax2[cnum, 0].imshow(site_df[site_df['cell_ids']==cellname]['c_strf'].values[0], origin='lower', aspect='auto')
    #     #         ax2[cnum, 0].set_title(f"{stat_name}: {stat_value}")
    #     #         ax2[cnum, 1].imshow(site_df[site_df['cell_ids']==cellname]['i_strf'].values[0], origin='lower', aspect='auto')
    #     #     plt.tight_layout()
    #     site_ic_mean = np.mean(site_df['i_strf_std'].values/site_df['c_strf_std'].values)
    #     site_ic_means.append(site_ic_mean)
    #     site_ic_std = np.std(site_df['i_strf_std'].values/site_df['c_strf_std'].values)
    #     # site_diffsum_mag = sum(site_df['diff'].values/site_df['sum'].values > dos_thresh)/len(site_df['diff'].values)
    #     site_diffsum_mag = sum(site_df[stat_name].values > dos_thresh) / len(
    #         site_df[stat_name].values)
    #     site_diffsum_mags.append(site_diffsum_mag)
    #     ax[1, col].scatter(site_ic_mean, site_ic_std, c=bwr(site_diffsum_mag), s=6)
    #     ax[1, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")
    #     ax[1, col].set_ylabel("std across cells")
    #     #masked
    #     masked_site_ic_mean = np.mean(site_df['masked i_strf_std'].values/site_df['masked c_strf_std'])
    #     masked_site_ic_means.append(masked_site_ic_mean)
    #     masked_site_ic_std = np.std(site_df['masked i_strf_std'].values/site_df['masked c_strf_std'])
    #     # masked_site_diffsum_mag = sum(site_df['masked diff'].values/site_df['masked sum'].values > dos_thresh)/len(site_df['masked diff'].values)
    #     masked_site_diffsum_mag  = sum(site_df[stat_name].values > dos_thresh) / len(
    #         site_df[stat_name].values)
    #     masked_site_diffsum_mags.append(masked_site_diffsum_mag)
    #     ax[4, col].scatter(masked_site_ic_mean, masked_site_ic_std, c=bwr(masked_site_diffsum_mag), s=6)
    #     ax[4, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")
    #     ax[4, col].set_ylabel("std across cells")
    # f6, ax6 = plt.subplots(1,1)
    # f7, ax7 = plt.subplots(1, 1)
    # for i in range(len(site_cisum)):
    #     ax6.scatter(site_cisum[i], site_meancontra[i], color=clrs(clr_space[i]))
    #     ax6.errorbar(site_cisum[i], site_meancontra[i], xerr=site_cisum_std[i], yerr=site_meancontra_std[i],
    #                  color=clrs(clr_space[i]), alpha=0.5)
    #     ax6.annotate(cellids[i], (site_cisum[i], site_meancontra[i]))
    #     ax7.scatter(site_hsumhdiff_mean[i], site_meancontra[i], color=clrs(clr_space[i]))
    #     ax7.errorbar(site_hsumhdiff_mean[i], site_meancontra[i], xerr=site_hsumhdiff_mean_std[i], yerr=site_meancontra_std[i],
    #                  color=clrs(clr_space[i]), alpha=0.5)
    #     ax7.annotate(cellids[i], (site_hsumhdiff_mean[i], site_meancontra[i]))
    #
    # f4, ax4 = plt.subplots(1,1)
    # # ax4.hist(rank_df[stat_name].values, bins=hist_bins, histtype='bar', density=True)
    # # high_pop_ee = max(list(site_ee_percents.keys()))
    # high_pop_ee = min(list(site_ee_percents.keys()))
    # stat_range = np.linspace(min(list(site_stat_dict.keys())), max(list(site_stat_dict.keys())), 15)
    # site_names = []
    # site_mean_vals = np.array(list(site_stat_dict.keys()))
    # for mean_stat in stat_range:
    #     site_index = np.abs(site_mean_vals-mean_stat).argmin()
    #     site_names.append(site_stat_dict[site_mean_vals[site_index]])
    # lowest_site = site_stat_dict[min(list(site_stat_dict.keys()))]
    # max_site = site_stat_dict[max(list(site_stat_dict.keys()))]
    # # ax4.hist(rank_df[rank_df['cell_id'] == site_ee_percents[high_pop_ee]][stat_name].values, bins=hist_bins,histtype='bar', density=True, color='red')
    # ax4.hist(site_stat_mean, bins=20)
    # site_ic_means_low = [sicm for sicm, sdsm in zip(site_ic_means, site_diffsum_mags) if sdsm < 0.5]
    # site_ic_means_high = [sicm for sicm, sdsm in zip(site_ic_means, site_diffsum_mags) if sdsm > 0.5 ]
    # masked_site_ic_means_low = [sicm for sicm, sdsm in zip(masked_site_ic_means, masked_site_diffsum_mags) if sdsm < 0.5]
    # masked_site_ic_means_high = [sicm for sicm, sdsm in zip(masked_site_ic_means, masked_site_diffsum_mags) if sdsm > 0.5 ]
    # ax[2, col].hist(site_ic_means_low, bins=10, color='blue', alpha=0.5)
    # ax[2, col].hist(site_ic_means_high, bins=10, color='red', alpha=0.5)
    # ax[2, col].set_ylabel("counts")
    # ax[2, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")
    # ax[5, col].hist(masked_site_ic_means_low, bins=10, color='blue', alpha=0.5)
    # ax[5, col].hist(masked_site_ic_means_high, bins=10, color='red', alpha=0.5)
    # ax[5, col].set_ylabel("counts")
    # ax[5, col].set_xlabel("mean std(ipsi strf)/\nstd(contra strf)")
    #
    # good_example_cells = ['SLJ019a-A-157-1', 'PRN014b-225-1', 'PRN048a-230-1', 'CLT039c-002-1']

    # site_strf_plot(rank_df, cellids=good_example_cells, statname=stat_name, num=4, showmask=False, save=True, savepath=data_save_path, saveopts={'dpi':600, 'format':'pdf'})
# quick permutation test idea
df = binaural_strf_df[binaural_strf_df['rank'] == rank]
def df_perm_test(df, stat, cellids, permnum=1000, norm_to_perm = True):
    # true test statistic
    f, ax = plt.subplots(1, 1)
    true_mean_stat_var = []
    true_mean_stat_varofvar = []
    # true_mean_var_meancontra = []
    # true_mean_diffsum = []
    for rank in ranks:
        rank_df = df.copy()
        site_stat_var = []
        site_meancontra_var = []
        site_diffsum_var = []
        for sitenum, cellid in enumerate(cellids):
            site_df = rank_df[rank_df['cell_id'] == cellid]
            cell_n = len(site_df['diff'].values)
            if cell_n < 5:
                continue
            site_stat_var.append(np.var(site_df[stat].values))
            # site_meancontra_var.append(np.var(site_df['mean(contra)/mean(abs(contra))'].values))
            # site_diffsum_var.append(np.var(site_df['diffsum_ratio'].values))

        true_mean_stat_var.append(np.mean(site_stat_var))
        true_mean_stat_varofvar.append(np.std(site_stat_var))
        # true_mean_var_meancontra.append(np.mean(site_meancontra_var))
        # true_mean_diffsum.append(np.mean(site_diffsum_var))

    perm_stat_var = []
    # perm_meancotnra_var = []
    # perm_diffsum_var = []
    for permutenum in range(permnum):
        for rank in ranks:
            rank_df = df.copy()
            # permute site_id
            rank_df['cell_id'] = rank_df['cell_id'].sample(frac=1).reset_index(drop=True)
            site_stat_var = []
            # site_cisum_var = []
            # site_diffsum_var = []
            for sitenum, cellid in enumerate(cellids):
                site_df = rank_df[rank_df['cell_id'] ==cellid]
                cell_n = len(site_df['diff'].values)
                if cell_n < 5:
                    continue
                site_stat_var.append(np.var(site_df[stat].values))
                # site_meancontra_var.append(np.var(site_df['mean(contra)/mean(abs(contra))'].values))
                # site_diffsum_var.append(np.var(site_df['diffsum_ratio'].values))
            perm_stat_var.append(np.mean(site_stat_var))
            # perm_meancotnra_var.append(np.mean(site_meancontra_var))
            # perm_diffsum_var.append(np.mean(site_diffsum_var))
    ax.hist(perm_stat_var, bins=30)
    ax.axvline(true_mean_stat_var[0])
    ax.set_title(f"Mean site variance of {stat}")
    # ax[1].hist(perm_meancotnra_var, bins=30)
    # ax[1].axvline(true_mean_var_meancontra[0])
    # ax[2].hist(perm_diffsum_var, bins=30)
    # ax[2].axvline(true_mean_diffsum[0])
    if true_mean_stat_var > np.mean(perm_stat_var):
        pvalue = (sum(np.array(perm_stat_var)>true_mean_stat_var)+1)/(sum(perm_stat_var<true_mean_stat_var)+1)
    else:
        pvalue = (sum(np.array(perm_stat_var) < true_mean_stat_var)+1)/(sum(np.array(perm_stat_var) > true_mean_stat_var)+1)

    mean_perm_val = np.mean(perm_stat_var)
    if norm_to_perm:
        return true_mean_stat_var/mean_perm_val, true_mean_stat_varofvar/mean_perm_val, mean_perm_val/mean_perm_val, np.std(perm_stat_var/mean_perm_val), pvalue
    else:
        return true_mean_stat_var, true_mean_stat_varofvar, np.mean(perm_stat_var), np.std(perm_stat_var), pvalue

statlist = ['ic_ratio', 'mean(contra)/mean(abs(contra))', 'sum(abs(contra-ipsi))/sum(abs(contra))', '(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))', 'std(hcontra-hipsi)/(std(hcontra)+std(hipsi))']
stat_dict = {}
for stat in statlist:
    true_stat, true_varofvar, perm_stat, perm_varofvar, pvalue = df_perm_test(df=binaural_strf_df[binaural_strf_df['rank'] == rank], stat=stat, cellids=cellids, permnum=1000)
    stat_dict[stat] = [true_stat, true_varofvar, perm_stat, perm_varofvar, pvalue]

plot_stats = ['mean(contra)/mean(abs(contra))', 'std(hcontra-hipsi)/(std(hcontra)+std(hipsi))', '(std(hcontra)-std(hipsi))/(std(hcontra)+std(hipsi))']
plot_stats_labels = ['mean(contra)/\nmean(abs(contra))', 'std(hcontra-hipsi)/\n(std(hcontra)+std(hipsi))', '(std(hcontra)-std(hipsi))/\n(std(hcontra)+std(hipsi))']
x_labels = plot_stats
stat_list = []
stat_list_varofvar = []
perm_stat_list = []
perm_stat_list_varofvar = []
pvalues = []
for stat in plot_stats:
    stat_list.append(stat_dict[stat][0][0])
    stat_list_varofvar.append(stat_dict[stat][1][0])
    perm_stat_list.append(stat_dict[stat][2])
    perm_stat_list_varofvar.append(stat_dict[stat][3])
    pvalues.append(stat_dict[stat][4])
plot_stats_dict = {'Mean Site Variance': stat_list, 'Mean Permutated Variance':perm_stat_list}
stat_varofvar = {'Mean Site Variance':stat_list_varofvar, 'Mean Permutated Variance':perm_stat_list_varofvar}
stat_pvalues = {'Mean Site Variance':pvalues, 'Mean Permutated Variance':pvalues}

x = np.arange(len(x_labels))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
colors = ['blue', 'grey']
for stat_type, value in plot_stats_dict.items():
    offset = width * multiplier
    if stat_type == 'Mean Permutated Variance':
        rects = ax.bar(x + offset, value, width, xerr=stat_varofvar[stat_type], yerr=stat_varofvar[stat_type], label=stat_type, color=colors[multiplier])
    else:
        rects = ax.bar(x + offset, value, width, label=stat_type, color=colors[multiplier])
    if multiplier == 0:
        ax.bar_label(rects, labels=[f"p: {pval}" if pval >= 0.01 else "p:<0.01" for pval in stat_pvalues[stat_type]], padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('% Permutated Mean Site Variance')
ax.set_title('Mean site variance for STRF statistics')
ax.set_xticks(x +width/2, plot_stats_labels)
ax.legend(loc='upper right', ncols=2)
# ax.set_ylim(0, 250)
tmp_fig_savepath = data_save_path / 'binaural_strf_stat_summary.pdf'
saveopts = {'dpi': 600, 'format': 'pdf'}
fig.savefig(tmp_fig_savepath, **saveopts)

plt.tight_layout()
bp = []
