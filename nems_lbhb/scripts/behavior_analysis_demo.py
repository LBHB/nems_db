from nems_lbhb import io
from nems import db


sql="SELECT gDataRaw.*, gData.svalue as DI, d2.svalue as pumpdur" + \
    " FROM gDataRaw INNER JOIN gData" + \
    " ON gDataRaw.id=gData.rawid AND gData.name='uDiscriminationIndex'" + \
    " INNER JOIN gData d2 ON gDataRaw.id=d2.rawid AND d2.name='Behave_PumpDuration'" + \
    " WHERE gDataRaw.cellid like %s"
params = ("AMT%T%", )
df = db.pd_query(sql, params)


r = df.iloc[-1]
mfilename = r['resppath'] + r['parmfile']
print("loading "+mfilename)

globalparams, exptparams, exptevents = io.baphy_parm_read(mfilename)

