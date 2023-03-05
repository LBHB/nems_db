from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, butter, sosfilt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from datetime import date, datetime

from nems0 import db
import nems0.epoch as ep
from nems0.utils import smooth

import nems0.plots.api as nplt

sql = "SELECT sCellFile.*, gSingleRaw.isolation, gPenetration.well, gPenetration.pendate" +\
       " FROM sCellFile INNER JOIN gCellMaster ON sCellFile.masterid=gCellMaster.id" +\
       " INNER JOIN gPenetration ON gCellMaster.penid=gPenetration.id" +\
       " INNER JOIN gSingleRaw ON sCellFile.singlerawid=gSingleRaw.id" +\
       " WHERE sCellFile.cellid like 'PRN%%' order by cellid"
df_cell = db.pd_query(sql)


datestr0 = df_cell.loc[0,'pendate']
def days_since(datestr):
    d1 = datetime.strptime(datestr, "%Y-%m-%d")
    d0 = datetime.strptime(datestr0, "%Y-%m-%d")
    delta = d1 - d0
    return delta.days

df_cell['siteid']=df_cell['cellid'].apply(db.get_siteid)
df_cell['day']=df_cell['pendate'].apply(days_since)
df_cell['SU']=df_cell['isolation']>=95

df_single_cell = df_cell.groupby('cellid').agg(
    {'well': 'first',
     'siteid': 'first',
     'day': 'first',
     'SU': 'first',
     'rawid': 'count'})
df_site = df_single_cell.groupby('siteid').agg(
    {'well': 'first',
     'day': 'first',
     'SU': 'sum',
     'rawid': 'count'})
df_site.columns = ['well','day','SU','N']

d1 = df_site[['well','day','SU']].reset_index()
d2 = df_site[['well','day','N']].reset_index()
d1['Type']='SU'
d2['Type']='SU+MU'
cols = ['siteid','Well','Day','N units','Type']
d1.columns=cols
d2.columns=cols
d = pd.concat([d1, d2], ignore_index=True)

f,ax=plt.subplots(1,1)
sns.scatterplot(data=d, x='Day', y='N units', hue='Well', style='Type', palette='deep', ax=ax)
ax.set_title('Prince left hemisphere')
