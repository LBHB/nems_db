import pickle
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob
import os.path
import pickle
import pandas as pd
from pathlib import Path
from scipy import stats
import seaborn as sns

#dataroot="/auto/users"
dataroot="/Volumes/users"

def load_cv(cv_path, glob='*summary.pkl'):
    results = []
    for filename in Path(cv_path).glob(glob):
        cell = filename.stem.rsplit('-', 1)[0]
        with open(filename, 'rb') as fh:
            try:
                cv = pickle.load(fh)
                cv['cell'] = cell
                results.append(cv)
            except:
                print(f'Error loading {filename}')
    return pd.concat(results)

cv = load_cv(dataroot+ '/bburan/analysis/RDT/token_based/results/cv', '*summaryV2.pkl').set_index(['cell', 'model'])
scv = load_cv(dataroot+ '/bburan/analysis/RDT/token_based/results/shuffle_cv', '*summaryV2.pkl').set_index(['cell', 'model'])

with open(dataroot+ '/bburan/analysis/RDT/token_based/results/summarized.pkl', 'rb') as fh:
    sr, rate, rg, enh, tp, sp, observed = pd.read_pickle(fh, compression=None)
cell_area_map = observed.groupby('cell').area.first()

m = pd.concat([cv, scv], keys=['no', 'yes'], names=['shuffled'])
m = m.reset_index().join(cell_area_map, on='cell').set_index(['area'] + m.index.names)['r_squared'].xs('enh', level='model')

sns.set_context('paper')
sns.set_style('whitegrid')
def colorize(bp, fc, ec):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=ec)
    for patch in bp['boxes']:
        patch.set(facecolor=fc)
    return patch

figure, ax = plt.subplots(1, 1, figsize=(4, 4))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c',]

bp = ax.boxplot(m.loc['A1', 'yes'], positions=[0], widths=0.5, patch_artist=True)
colorize(bp, colors[1], 'black')
bp = ax.boxplot(m.loc['A1', 'no'], positions=[1], widths=0.5, patch_artist=True)
colorize(bp, colors[2], 'black')
bp = ax.boxplot(m.loc['PEG', 'yes'], positions=[3], widths=0.5, patch_artist=True)
h1 = colorize(bp, colors[1], 'black')
bp = ax.boxplot(m.loc['PEG', 'no'], positions=[4], widths=0.5, patch_artist=True)
h2 = colorize(bp, colors[2], 'black')

ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(['A1', 'PEG'])
ax.legend([h1, h2], ['S', 'Full'])
ax.set_ylabel('Mean pred. corr.')

sns.despine(figure, offset=10)
figure.savefig('/tmp/PSTH-pred.pdf')

meanpred=np.zeros((2,2))
sempred=np.zeros((2,2))

x = m.loc['A1'].unstack('shuffled').dropna()
print('A1', stats.ttest_rel(x['yes'], x['no']))
print(x.mean(), x.median(), x.std()/np.sqrt(x.count()))
meanpred[0,:]=x.mean().values
sempred[0,:]=(x.std()/np.sqrt(x.count())).values

x = m.loc['PEG'].unstack('shuffled').dropna()
print('PEG', stats.ttest_rel(x['yes'], x['no']))
print(x.mean(), x.median(), x.std()/np.sqrt(x.count()))
meanpred[1,:]=x.mean().values
sempred[1,:]=(x.std()/np.sqrt(x.count())).values

fig = plt.figure(figsize=(4,4))
plt.bar(np.arange(2)-0.28, meanpred[:,0]-0.2, yerr=sempred[:,0], bottom=0.2,width=0.2)
plt.bar(np.arange(2)+0.28, meanpred[:,1]-0.2, yerr=sempred[:,1], bottom=0.2,width=0.2)

plt.legend(['S','full'])
ax_mean=plt.gca()
ax_mean.set_xticks(np.arange(0,2))
ax_mean.set_xticklabels(['A1','PEG'])
ax_mean.set_ylabel('mean pred corr.')
#plt.ax_remove_box(ax_mean)

sns.despine(fig, offset=10)
fig.savefig('/tmp/PSTH-pred-meanbar.pdf')


m = pd.concat([cv, scv], keys=['no', 'yes'], names=['shuffled'])['r_squared']
#m = m.reset_index().join(cell_area_map, on='cell').set_index(['area'] + m.index.names)['r_squared'].xs('enh', level='model')

figure, axes = plt.subplots(2, 2, figsize=(8, 8))

x = m.unstack('shuffled').xs('global', level='model')
axes[0, 0].plot(x['yes'], x['no'], 'k.')
axes[0, 0].plot([-0.5, 1], [-0.5, 1], 'r-')
axes[0, 0].set_xlabel('Stream-independent (shuffled)')
axes[0, 0].set_ylabel('Stream-independent')

x = m.unstack('shuffled').xs('enh', level='model').dropna()
print(stats.ttest_rel(x['no'], x['yes']))

axes[0, 1].plot(x['yes'], x['no'], 'k.')
axes[0, 1].plot([-0.5, 1], [-0.5, 1], 'r-')
axes[0, 1].set_xlabel('Stream-dependent (shuffled)')
axes[0, 1].set_ylabel('Stream-dependent')

x = m.xs('no', level='shuffled').unstack('model').dropna()
print(stats.ttest_rel(x['global'], x['enh']))
axes[1, 0].plot(x['global'], x['enh'], 'k.')
axes[1, 0].plot([-0.5, 1], [-0.5, 1], 'r-')
axes[1, 0].set_xlabel('Stream-independent')
axes[1, 0].set_ylabel('Stream-dependent')

x.mean()
