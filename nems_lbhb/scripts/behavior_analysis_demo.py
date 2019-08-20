from nems_lbhb import io
from nems import db
import numpy as np

def str_asarray(s):
    s = s.replace('[','').replace(']','')
    els = s.split(' ')
    arr = np.zeros(len(els))
    for i, e in enumerate(els):
        arr[i] = e
    return arr

sql="SELECT gDataRaw.*, gData.svalue as DI, d2.svalue as pumpdur" + \
    " FROM gDataRaw INNER JOIN gData" + \
    " ON gDataRaw.id=gData.rawid AND gData.name='uDiscriminationIndex'" + \
    " INNER JOIN gData d2 ON gDataRaw.id=d2.rawid AND d2.name='Behave_PumpDuration'" + \
    " WHERE gDataRaw.cellid like %s"
params = ("AMT%T%", )
df = db.pd_query(sql, params)
df['DI'] = [str_asarray(di) if di is not None else np.nan for di in df['DI']]

r = df.iloc[-1]
mfilename = r['resppath'] + r['parmfile']
print("loading "+mfilename)

globalparams, exptparams, exptevents = io.baphy_parm_read(mfilename)

