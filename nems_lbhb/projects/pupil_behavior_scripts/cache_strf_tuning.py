"""
Compute STRFs for each cell in batch 307 and 309. Cache these into
the pupil-behavior ms directory to be loaded by "pupil_behavior_dump_csv"

Also, save strf images to pdf file along w/ snr and bf for visual inspection.

crh - 3/6/2020
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import nems_lbhb.strf.strf as strf
from nems_lbhb.strf.torc_subfunctions import strfplot
import nems_lbhb.baphy as nb
from nems0.recording import Recording
import nems0.db as nd
from nems import get_setting

fs = 1000

# data frame cache
path = get_setting("NEMS_RESULTS_DIR")
df_295_filename = os.path.join(path, str(295), 'd_tuning.csv')
df_307_filename = os.path.join(path, str(307), 'd_tuning.csv')
df_309_filename = os.path.join(path, str(309), 'd_tuning.csv')
df_313_filename = os.path.join(path, str(313), 'd_tuning.csv')

# pdf figure cache
# pdf_path = '/home/charlie/Desktop/lbhb/code/nems_db/nems_lbhb/pupil_behavior_scripts/strf_tuning/'

# ================================= batch 307 ==================================
cells_307 = nd.get_batch_cells(307).cellid
df_307 = pd.DataFrame(index=cells_307, columns=['BF', 'SNR', 'STRF', 'StimParms'])
for cellid in cells_307:
    print('analyzing cell: {0}, batch {1}'.format(cellid, 307))

    ops = {'batch': 307, 'pupil': 1, 'rasterfs': fs, 'cellid': cellid, 'stim': 0}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    r = rec.copy()

    # get strf and save results
    out = strf.tor_tuning(cellid, rec=r, plot=False)
    df_307.loc[cellid] = [out.Best_Frequency_Hz, out.Signal_to_Noise, [out.STRF], [out.StimParams]]

# cache bf / snr results
df_307[['BF', 'SNR']].to_csv(df_307_filename)

if 0:
    # plot results, 16 strfs per page
    page_starts = np.arange(0, len(cells_307), 16) 
    for j, page in enumerate(page_starts):
        f, ax = plt.subplots(4, 4, figsize=(16, 12))

        results = df_307.iloc[page:(page+16)]
        tleng = 0.75
        for i, a in enumerate(ax.flatten()):
            try:
                strfplot(results.iloc[i]['STRF'][0], results.iloc[i]['StimParms'][0]['lfreq'], tleng, axs=a, smooth=True)
                bf = results.iloc[i]['BF']
                snr = np.round(results.iloc[i]['SNR'], 3)
                a.set_title(results.iloc[i].name + '\n BF: {0}, SNR: {1}'.format(bf, snr), fontsize=8)
            except:
                pass

        f.tight_layout()
        f.savefig(pdf_path+'307_strfs_{}.png'.format(j))
    plt.close('all')

# ================================= batch 309 ===================================
cells_309 = nd.get_batch_cells(309).cellid
df_309 = pd.DataFrame(index=cells_309, columns=['BF', 'SNR', 'STRF', 'StimParms'])
for cellid in cells_309:
    print('analyzing cell: {0}, batch {1}'.format(cellid, 309))

    if 'ley046g' in cellid:
        # torcs switched. just use first half of data
        ops = {'batch': 309, 'pupil': 1, 'rasterfs': fs, 'cellid': cellid, 'stim': 0,
            'rawid': [131491, 131492, 131493]}
    else:
        ops = {'batch': 309, 'pupil': 1, 'rasterfs': fs, 'cellid': cellid, 'stim': 0}
    uri = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    r = rec.copy()

    # get strf and save results
    out = strf.tor_tuning(cellid, rec=r, plot=False)
    df_309.loc[cellid] = [out.Best_Frequency_Hz, out.Signal_to_Noise, [out.STRF], [out.StimParams]]

# cache bf / snr results
df_309[['BF', 'SNR']].to_csv(df_309_filename)

if 0:
    # plot results, 16 strfs per page
    page_starts = np.arange(0, len(cells_309), 16) 
    for j, page in enumerate(page_starts):
        f, ax = plt.subplots(4, 4, figsize=(16, 12))

        results = df_309.iloc[page:(page+16)]
        tleng = 0.75
        for i, a in enumerate(ax.flatten()):
            try:
                strfplot(results.iloc[i]['STRF'][0], results.iloc[i]['StimParms'][0]['lfreq'], tleng, axs=a, smooth=True)
                bf = results.iloc[i]['BF']
                snr = np.round(results.iloc[i]['SNR'], 3)
                a.set_title(results.iloc[i].name + '\n BF: {0}, SNR: {1}'.format(bf, snr), fontsize=8)
            except:
                pass

        f.tight_layout()
        f.savefig(pdf_path+'309_strfs_{}.png'.format(j))
    plt.close('all')


# ================================= batch 313 ===================================
cells_313 = nd.get_batch_cells(313).cellid
df_313 = pd.DataFrame(index=cells_313, columns=['BF', 'SNR', 'STRF', 'StimParms'])
for cellid in cells_313:
    print('analyzing cell: {0}, batch {1}'.format(cellid, 313))

    if 'ley046g' in cellid:
        # torcs switched. just use first half of data
        ops = {'batch': 313, 'pupil': 0, 'rasterfs': fs, 'cellid': cellid, 'stim': 0,
            'rawid': [131491, 131492, 131493]}
    elif 'bbl032f-a1' in cellid:
        ops = {'batch': 313, 'pupil': 0, 'rasterfs': fs, 'cellid': cellid, 'stim': 0,
            'rawid': [120899, 120901, 120903]}
    elif 'bbl041e-a1' in cellid:
        ops = {'batch': 313, 'pupil': 0, 'rasterfs': fs, 'cellid': cellid, 'stim': 0,
            'rawid': [121418, 121419, 121420]}
    else:
        ops = {'batch': 313, 'pupil': 0, 'rasterfs': fs, 'cellid': cellid, 'stim': 0}
    uri, _ = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    r = rec.copy()

    # get strf and save results
    if 'TTL038d-a1' in cellid: # WHY ??
        pass
    else:
        out = strf.tor_tuning(cellid, rec=r, plot=False)
        df_313.loc[cellid] = [out.Best_Frequency_Hz, out.Signal_to_Noise, [out.STRF], [out.StimParams]]

# cache bf / snr results
df_313[['BF', 'SNR']].to_csv(df_313_filename)


# ================================= batch 295 ===================================
cells_295 = nd.get_batch_cells(295).cellid
df_295 = pd.DataFrame(index=cells_295, columns=['BF', 'SNR', 'STRF', 'StimParms'])
for cellid in cells_295:
    print('analyzing cell: {0}, batch {1}'.format(cellid, 295))

    ops = {'batch': 295, 'pupil': 0, 'rasterfs': fs, 'cellid': cellid, 'stim': 0}
    uri, _ = nb.baphy_load_recording_uri(**ops)
    rec = Recording.load(uri)
    r = rec.copy()

    out = strf.tor_tuning(cellid, rec=r, plot=False)
    df_295.loc[cellid] = [out.Best_Frequency_Hz, out.Signal_to_Noise, [out.STRF], [out.StimParams]]

# cache bf / snr results
df_295[['BF', 'SNR']].to_csv(df_295_filename)
