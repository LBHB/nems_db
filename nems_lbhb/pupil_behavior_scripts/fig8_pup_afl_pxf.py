"""
Unique variance explained by pupil, afl, and pxf. Bar plot.
Point is to highlight no substantial variance explained by pxf (no interaction between
arousal effects and task condition)
CRH 04/22/2020
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import nems.db as nd

# load model results
m = 'psth.fs20.pup-ld-st.pup.afl.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'
m_p0 = 'psth.fs20.pup-ld-st.pup0.afl.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'
m_b0 = 'psth.fs20.pup-ld-st.pup.afl0.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'
m_bp0 = 'psth.fs20.pup-ld-st.pup.afl.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'
mpxf = 'psth.fs20.pup-ld-st.pup.afl.pxf-ref-psthfr.s_sdexp.S_jk.nf20-basic'

modelnames = tuple([m, m_p0, m_b0, m_bp0, mpxf])
sql = "SELECT cellid, r_test, se_test, modelname FROM Results WHERE batch=307 and modelname in {}".format(modelnames)
r = nd.pd_query(sql)
r['r_test'] = r['r_test'] ** 2
r = r[~r.cellid.str.contains('AMT')]

rA1 = r.pivot(columns='modelname', index='cellid', values='r_test')

A1_upup = rA1[m] - rA1[m_p0]
A1_ubeh = rA1[m] - rA1[m_b0]
A1_upxf = rA1[m] - rA1[m_bp0]
A1 = pd.DataFrame(index=A1_upup.index, columns=['upup', 'ubeh', 'upxf'], data=pd.concat([A1_upup, A1_ubeh, A1_upxf], axis=1).values)
A1['area'] = 'A1'

sql = "SELECT cellid, r_test, se_test, modelname FROM Results WHERE batch=309 and modelname in {}".format(modelnames)
r = nd.pd_query(sql)
r['r_test'] = r['r_test'] ** 2
r = r[~r.cellid.str.contains('AMT')]

rIC = r.pivot(columns='modelname', index='cellid', values='r_test')

IC_upup = rIC[m] - rIC[m_p0]
IC_ubeh = rIC[m] - rIC[m_b0]
IC_upxf = rIC[m] - rIC[m_bp0]
IC = pd.DataFrame(index=IC_upup.index, columns=['upup', 'ubeh', 'upxf'], data=pd.concat([IC_upup, IC_ubeh, IC_upxf], axis=1).values)
IC['area'] = 'IC'

df = pd.concat([A1, IC])
df = df.melt(value_vars=['upup', 'ubeh', 'upxf'], id_vars='area')

f, ax = plt.subplots(1, 1)

sns.barplot(data=df, x='variable', y='value', hue='area', errwidth=2, errcolor='k', edgecolor='k', lw=2, ax=ax)
ax.set_ylabel(r'Unique $R^{2}$')
ax.set_xticks(range(0, 3))
ax.set_xticklabels(['pup', 'afl', 'pxf'])
ax.set_xlabel('State channel')

plt.show()

# compare pxf vs. pxf0 directly
pxf = rIC[mpxf].mean()
pxf0 = rIC[m_bp0].mean()
pval = ss.wilcoxon(rIC[mpxf], rIC[m_bp0]).pvalue
print("IC r_test: \n pxf: {0}, pxf0: {1}, pval: {2} \n".format(pxf, pxf0, pval))

pxf = rA1[mpxf].mean()
pxf0 = rA1[m_bp0].mean()
pval = ss.wilcoxon(rA1[mpxf], rA1[m_bp0]).pvalue
print("A1 r_test: \n pxf: {0}, pxf0: {1}, pval: {2}".format(pxf, pxf0, pval))


