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

animal="PRN"
animal="SLJ"

sql = "SELECT sCellFile.*, gSingleRaw.isolation, gPenetration.well, gPenetration.pendate" +\
       " FROM sCellFile INNER JOIN gCellMaster ON sCellFile.masterid=gCellMaster.id" +\
       " INNER JOIN gPenetration ON gCellMaster.penid=gPenetration.id" +\
       " INNER JOIN gSingleRaw ON sCellFile.singlerawid=gSingleRaw.id" +\
       f" WHERE sCellFile.cellid like '{animal}%%' order by cellid"
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
df_cell['isAC'] = (df_cell['area'] == 'A1') | (df_cell['area'] == 'BS') | \
     (df_cell['area'] == 'PEG')
df_cell['AC_SU'] = df_cell['SU'] & df_cell['isAC']

df_single_cell = df_cell.groupby('cellid').agg(
    {'well': 'first',
     'siteid': 'first',
     'day': 'first',
     'isAC': 'first',
     'AC_SU': 'first',
     'SU': 'first',
     'rawid': 'count'})
df_site = df_single_cell.groupby('siteid').agg(
    {'well': 'first',
     'day': 'first',
     'isAC': 'sum',
     'AC_SU': 'sum',
     'SU': 'sum',
     'rawid': 'count'})
df_site.columns = ['well','day','isAC','AC_SU','SU','N']

d0 = df_site[['well','day','AC_SU']].reset_index()
d1 = df_site[['well','day','SU']].reset_index()
d2 = df_site[['well','day','N']].reset_index()
d0['Type']='AC+SU'
d1['Type']='SU'
d2['Type']='SU+MU'
cols = ['siteid','Well','Day','N units','Type']
d0.columns=cols
d1.columns=cols
d2.columns=cols
d = pd.concat([d0, d1, d2], ignore_index=True)

f,ax=plt.subplots(1,1,figsize=(8,4))
sns.scatterplot(data=d, x='Day', y='N units', hue='Well', style='Type', palette='deep', ax=ax)
ax.set_title(f'{animal} cell census')
for i,r in df_site.iterrows():
    ax.text(r.day, r.N, i, fontsize=10, rotation=90, va='bottom', ha='center')

