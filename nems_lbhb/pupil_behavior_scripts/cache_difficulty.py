"""
Cache target frequency for each active file for each cell
"""
import nems.db as nd
import pandas as pd
import numpy as np
import os
from nems import get_setting

fpath = get_setting('NEMS_RESULTS_DIR') 

# rawids for sessions with TONEinTORCs SVD style
flipsetrawids=[120446, 120448, 120451, 120475, 120477, 120525,
    120528, 120530, 120538, 120539, 120542, 120544]
# rawids for sessions with TONEinTORCs DS style
straddlesetrawids=[120110, 120111, 120163, 120165, 120184, 120185,
    120188, 120190, 120199, 120201, 120207, 120209, 120211, 120214,
    120234, 120234, 120254, 120256, 120258, 120260, 120272, 120273,
    120274, 120275, 120293, 120283, 120285, 120286, 120289, 120290,
    120293, 120310, 120311, 120312, 120313, 120314, 120316, 120317,
    120435, 120436, 120437]

# ================================= batch 307 ====================================
perfile_df = pd.read_csv(os.path.join(fpath, str(307), 'd_pup_fil_sdexp.csv'), index_col=0)
df_307 = pd.DataFrame()
cells_307 = nd.get_batch_cells(307).cellid
for cellid in cells_307:
    _, rawid = nd.get_stable_batch_cells(batch=307, cellid=cellid)
    sql = "SELECT value, svalue, rawid from gData where name='Trial_TargetIdxFreq' and rawid in {}".format(tuple(rawid))
    d = nd.pd_query(sql, params=())
    sql =  "SELECT value, svalue, rawid from gData where name='Trial_RelativeTarRefdB' and rawid in {}".format(tuple(rawid))
    d2 =  nd.pd_query(sql, params=())
    sql = "SELECT behavior, id from gDataRaw where id in {0}".format(tuple(rawid))
    da = nd.pd_query(sql)

    d = d[d.rawid.isin([r for r in da.id if da[da.id==r]['behavior'].values=='active'])]
    d2 = d2[d2.rawid.isin([r for r in da.id if da[da.id==r]['behavior'].values=='active'])]
    d2.columns = [c+'_rel' for c in d2.columns]
    d = pd.concat([d, d2], axis=1)

    pf_labels = np.unique([l for l in perfile_df[perfile_df['cellid']==cellid]['state_chan_alt'] if 'ACTIVE' in l])

    if len(pf_labels) != 0:
        diff = None
        for i in range(0, d.shape[0]):
            tf = d.iloc[i]['svalue']
            if (tf is not None):
                tf = np.array([float(x) for x in d.iloc[i]['svalue'].strip('[]').split(' ')])
                reltar = np.array([float(x) for x in d.iloc[i]['svalue_rel'].strip('[]').split(' ')])
                tcount = len(tf)
                if (tcount==5) & np.isinf(reltar[0]) & (tf[0]==0.8):
                    diff = 0 # puretone                
                elif (tcount==5) & (sum(tf==0.2)==5):
                    diff = 2 # medium
                elif (tcount==5) & (tf[0]==0.3):
                    diff = 1 # easy
                elif (tcount==5) & (tf[0]==0.1):
                    diff = 3 # hard
                elif (tcount==5) & (np.isinf(reltar[0])):
                    diff = 0 # pure tone (or should this get marked as easy?)
                elif d.iloc[i]['rawid'] in flipsetrawids:
                    if reltar[1]<0.5:
                        diff = 1
                    else:
                        diff = 3
                elif d.iloc[i]['rawid'] in straddlesetrawids:
                    if (reltar[0]>reltar[1]):
                        diff = 1
                    else:
                        diff = 3
            else:
                diff = 0 # tone only

            # figure out difficulty
            
            _df = pd.DataFrame({'cellid': cellid, 'difficulty': diff, 'state_chan_alt': pf_labels[i]}, index=[0])
            df_307 = df_307.append(_df)
    else:
        pass

df_307.to_csv(os.path.join(fpath, str(307), 'd_difficulty.csv'))

# ================================= batch 309 ====================================
perfile_df = pd.read_csv(os.path.join(fpath, str(309), 'd_pup_fil_sdexp.csv'), index_col=0)
df_309 = pd.DataFrame()
cells_309 = nd.get_batch_cells(309).cellid
for cellid in cells_309:
    _, rawid = nd.get_stable_batch_cells(batch=309, cellid=cellid)
    sql = "SELECT value, svalue, rawid from gData where name='Trial_TargetIdxFreq' and rawid in {}".format(tuple(rawid))
    d = nd.pd_query(sql, params=())
    sql =  "SELECT value, svalue, rawid from gData where name='Trial_RelativeTarRefdB' and rawid in {}".format(tuple(rawid))
    d2 =  nd.pd_query(sql, params=())
    sql = "SELECT behavior, id from gDataRaw where id in {0}".format(tuple(rawid))
    da = nd.pd_query(sql)

    d = d[d.rawid.isin([r for r in da.id if da[da.id==r]['behavior'].values=='active'])]
    d2 = d2[d2.rawid.isin([r for r in da.id if da[da.id==r]['behavior'].values=='active'])]
    d2.columns = [c+'_rel' for c in d2.columns]
    d = pd.concat([d, d2], axis=1)

    pf_labels = np.unique([l for l in perfile_df[perfile_df['cellid']==cellid]['state_chan_alt'] if 'ACTIVE' in l])

    if len(pf_labels) != 0:
        diff = None
        for i in range(0, d.shape[0]):
            tf = d.iloc[i]['svalue']
            if (tf is not None):
                tf = np.array([float(x) for x in d.iloc[i]['svalue'].strip('[]').split(' ')])
                reltar = np.array([float(x) for x in d.iloc[i]['svalue_rel'].strip('[]').split(' ')])
                tcount = len(tf)
                if (tcount==5) & np.isinf(reltar[0]) & (tf[0]==0.8):
                    diff = 0 # puretone                
                elif (tcount==5) & (sum(tf==0.2)==5):
                    diff = 2 # medium
                elif (tcount==5) & (tf[0]==0.3):
                    diff = 1 # easy
                elif (tcount==5) & (tf[0]==0.1):
                    diff = 3 # hard
                elif (tcount==5) & (np.isinf(reltar[0])):
                    diff = 0 # pure tone (or should this get marked as easy?)
                elif d.iloc[i]['rawid'] in flipsetrawids:
                    if reltar[1]<0.5:
                        diff = 1
                    else:
                        diff = 3
                elif d.iloc[i]['rawid'] in straddlesetrawids:
                    if (reltar[0]>reltar[1]):
                        diff = 1
                    else:
                        diff = 3
            else:
                diff = 0 # tone only

            # figure out difficulty
            
            _df = pd.DataFrame({'cellid': cellid, 'difficulty': diff, 'state_chan_alt': pf_labels[i]}, index=[0])
            df_309 = df_309.append(_df)
    else:
        pass

df_309.to_csv(os.path.join(fpath, str(309), 'd_difficulty.csv'))