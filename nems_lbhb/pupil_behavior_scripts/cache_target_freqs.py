"""
Cache target frequency for each active file for each cell
"""
import nems.db as nd
import pandas as pd
import numpy as np

fpath = '/auto/users/hellerc/code/nems_db/nems_lbhb/pupil_behavior_scripts/'

# ================================= batch 307 ====================================
perfile_df = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_307_fil.csv', index_col=0)
df_307 = pd.DataFrame()
cells_307 = nd.get_batch_cells(307).cellid
for cellid in cells_307:
    sql = "SELECT sCellFile.stimfile, gData.svalue, gData.value from sCellFile INNER JOIN \
                gData on (sCellFile.rawid=gData.rawid AND gData.name='Tar_Frequencies') \
                INNER JOIN gDataRaw on sCellFile.rawid=gDataRaw.id WHERE gDataRaw.behavior='active' \
                AND sCellFile.cellid=%s order by sCellFile.rawid"
    d = nd.pd_query(sql, params=(cellid,))
    pf_labels = np.unique([l for l in perfile_df[perfile_df['cellid']==cellid]['state_chan_alt'] if 'ACTIVE' in l])

    if len(pf_labels) != 0:
        for i in range(0, d.shape[0]):
            tf = d.iloc[i]['value']
            if tf is None:
                tf = int(d.iloc[i]['svalue'].strip('[]').split(' ')[0])
            _df = pd.DataFrame({'cellid': cellid, 'tar_freq': tf, 'state_chan_alt': pf_labels[i]}, index=[0])
            df_307 = df_307.append(_df)
    else:
        pass

df_307.to_csv(fpath + 'd_307_tar_freqs.csv')

# ================================= batch 309 ====================================
perfile_df = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_309_fil.csv', index_col=0)
df_309 = pd.DataFrame()
cells_309 = nd.get_batch_cells(309).cellid
for cellid in cells_309:
    sql = "SELECT sCellFile.stimfile, gData.svalue, gData.value from sCellFile INNER JOIN \
                gData on (sCellFile.rawid=gData.rawid AND gData.name='Tar_Frequencies') \
                INNER JOIN gDataRaw on sCellFile.rawid=gDataRaw.id WHERE gDataRaw.behavior='active' \
                AND sCellFile.cellid=%s order by sCellFile.rawid"
    d = nd.pd_query(sql, params=(cellid,))
    pf_labels = np.unique([l for l in perfile_df[perfile_df['cellid']==cellid]['state_chan_alt'] if 'ACTIVE' in l])

    if (len(pf_labels) != 0) & (len(pf_labels)==d.shape[0]):
        for i in range(0, d.shape[0]):
            tf = d.iloc[i]['value']
            if tf is None:
                tf = int(d.iloc[i]['svalue'].strip('[]').split(' ')[0])
            _df = pd.DataFrame({'cellid': cellid, 'tar_freq': tf, 'state_chan_alt': pf_labels[i]}, index=[0])
            df_309 = df_309.append(_df)
    else:
        pass

df_309.to_csv(fpath + 'd_309_tar_freqs.csv')

# ================================= batch 295 ====================================
perfile_df = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_295_fil_stategain.csv', index_col=0)
df_295 = pd.DataFrame()
cells_295 = nd.get_batch_cells(295).cellid
for cellid in cells_295:
    sql = "SELECT sCellFile.stimfile, gData.svalue, gData.value from sCellFile INNER JOIN \
                gData on (sCellFile.rawid=gData.rawid AND gData.name='Tar_Frequencies') \
                INNER JOIN gDataRaw on sCellFile.rawid=gDataRaw.id WHERE gDataRaw.behavior='active' \
                AND sCellFile.cellid=%s order by sCellFile.rawid"
    d = nd.pd_query(sql, params=(cellid,))
    pf_labels = np.unique([l for l in perfile_df[perfile_df['cellid']==cellid]['state_chan_alt'] if ('ACTIVE' in l)])

    if (len(pf_labels) != 0) & (len(pf_labels)==d.shape[0]):
        for i in range(0, d.shape[0]):
            tf = d.iloc[i]['value']
            if tf is None:
                tf = int(d.iloc[i]['svalue'].strip('[]').split(' ')[0])
            _df = pd.DataFrame({'cellid': cellid, 'tar_freq': tf, 'state_chan_alt': pf_labels[i]}, index=[0])
            df_295 = df_295.append(_df)
    else:
        import pdb; pdb.set_trace()
        pass

df_295.to_csv(fpath + 'd_295_tar_freqs.csv')