from pathlib import Path
import datetime
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 10,
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import nems
import nems.db as nd
import nems_lbhb.xform_wrappers as xwrap
import nems.epoch as ep
from pathlib import Path

from nems.xform_helper import load_model_xform
from nems_lbhb.xform_wrappers import generate_recording_uri, split_pop_rec_by_mask
from nems.recording import load_recording

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
    MODELGROUPS, POP_MODELGROUPS, HELDOUT, MATCHED, EQUIVALENCE_MODELS_SINGLE, EQUIVALENCE_MODELS_POP,
    POP_MODELS, ALL_FAMILY_POP,
    SIG_TEST_MODELS,
    get_significant_cells, snr_by_batch, NAT4_A1_SITES, NAT4_PEG_SITES, PLOT_STAT, DOT_COLORS, DOT_MARKERS, base_path,
    linux_user, ALL_FAMILY_MODELS, VERSION, count_fits, compute_snr, int_path, 
    a1, peg, base_path, single_column_short, single_column_tall, column_and_half_short, column_and_half_tall, column_and_half_vshort
)

a1 = 322
peg = 323

a1_sparseness_path = int_path  / str(a1) / 'sparseness_data.csv'
peg_sparseness_path = int_path  / str(peg) / 'sparseness_data.csv'

def sparseness(r, p, cellid="cell", verbose=False, ax=None, colors=None):
    n_r,xr = np.histogram(r, bins=np.arange(np.ceil(r.max()))-0.0)
    n_p,xp = np.histogram(p, bins=np.arange(np.ceil(r.max()))-0.0)
    c = np.corrcoef(r, p)[0,1]

    S_r = 1 - (r.mean())**2 / (r**2).mean()
    S_p = 1 - (p.mean())**2 / (p**2).mean()

    if verbose:
        if ax is None:
            f,ax = plt.subplots(figsize=(2,1.5))
        maxbins=np.min([12,len(n_r)])
        print(maxbins)
        if colors is None:
            colors=[None, None]
        ax.bar(xr[:maxbins]-0.2, n_r[:maxbins], width=0.4, color=colors[0], #edgecolor='black', linewidth=0.5,
               label=f"Actual {S_r:.2f}")
        ax.bar(xp[:maxbins]+0.2, n_p[:maxbins], width=0.4, color=colors[1], #edgecolor='black', linewidth=0.5,
               label=f"Predicted {S_p:.2f}")
        ax.set_xlabel("Spikes/sec")
        ax.set_ylabel("N occurrences")
        
        ax.legend(frameon=False, fontsize=7)
    return S_r, S_p, c


def sparseness_example(batch, cellid, modelname, rec=None, ax=None, tag="", colors=None):

    if rec is None:
        xf, ctx = load_model_xform(cellid, batch, modelname, eval_model=True)
        val = ctx['val']
        val = val.apply_mask()
    else:
        xf, ctx = load_model_xform(cellid, batch, modelname, eval_model=False)
        val = rec

    val_ = ctx['modelspec'].evaluate(val)

    fs = val['resp'].fs
    this_resp = val['resp'].extract_channels(chans=[cellid])._data[0,:] * fs
    this_pred = val_['pred']._data[0,:] * fs

    c = np.corrcoef(this_resp, this_pred)[0,1]

    if ax is None:
        f,ax=plt.subplots()
    S_r, S_p, c = sparseness(this_resp, this_pred, cellid=cellid, verbose=True, ax=ax, colors=colors)
    #ax.set_title(f"{cellid} {tag}")
    print(f"{cellid} r_test={c:.3f} orig r_test={ctx['modelspec'].meta['r_test'][0,0]:.3f} S_r={S_r:.3f} S_p={S_p:.3f}")

    return S_r, S_p


def sparseness_by_batch(batch, modelnames=None,
                        pop_reference_model=ALL_FAMILY_POP[2],
                        save_path=None, force_regenerate=False, rec=None):

    if (save_path is not None) & (not force_regenerate):
        if os.path.exists(save_path):
            sparseness_data = pd.read_csv(save_path, index_col=0)
            return sparseness_data

    if modelnames is None:
        modelnames=[ALL_FAMILY_MODELS[0],ALL_FAMILY_MODELS[2],ALL_FAMILY_MODELS[3] ]
        #modelnames=[ALL_FAMILY_MODELS[2],ALL_FAMILY_MODELS[3] ]

    d = nd.batch_comp(batch, modelnames)
    cellids = d.index

    if rec is None:
        xf0, ctx0 = load_model_xform(cellids[0], batch, pop_reference_model, eval_model=True)
        val = ctx0['val']
        val=val.apply_mask()
        del ctx0
    else:
        val=rec

    sparseness_data = pd.DataFrame()

    for j, cellid in enumerate(cellids):
        r_test_all = d.loc[cellid].values.max()
        if r_test_all>0.2:
            for i,m in enumerate(modelnames):

                xf, ctx = load_model_xform(cellid, batch, m, eval_model=False)
                val_ = ctx['modelspec'].evaluate(val)

                fs = val['resp'].fs
                this_resp = val['resp'].extract_channels(chans=[cellid])._data[0,:] * fs
                this_pred = val_['pred']._data[0,:] * fs

                c = np.corrcoef(this_resp, this_pred)[0,1]
                #c2 = np.corrcoef(this_resp._data[0,:], original_pred._data[0,:])[0,1]
                #print(c2, d.loc[cellid].values)

                S_r, S_p, c = sparseness(this_resp, this_pred, verbose=False)
                print(f"{j} {cellid} r_test={c:.3f} orig r_test={d.loc[cellid].values[i]:.3f} S_r={S_r:.3f} S_p={S_p:.3f}")

                sparseness_data = sparseness_data.append({'cellid': cellid, 'S_r': S_r, 'S_p': S_p, 'r_test': c,
                                                          'r_test_all': r_test_all, 'model': i},
                                                         ignore_index=True)

    if save_path is not None:
        print(f"Saving sparseness_data to {save_path}")
        sparseness_data.to_csv(save_path)

    return sparseness_data


def sparseness_plot(sparseness_data, modelnames):
    f,ax = plt.subplots(1,len(modelnames)+1, figsize=(4*(len(modelnames)+1),3))
    for i in range(len(modelnames)):
        m = modelnames[i].split("_")[1].split("-")[0]
        ax[i].plot([0,1],[0,1],'k--')
        sparseness_data.loc[sparseness_data.model==i].plot.scatter('S_r','S_p', s=5, c='r_test', ax=ax[i], vmin=0.1, vmax=0.9)
        ax[i].set_title(m)

    d = sparseness_data.loc[sparseness_data.model==1].merge(sparseness_data.loc[sparseness_data.model==2], how='inner', on='cellid',
                                                           suffixes=('_dnn','_ln'))
    ax[len(modelnames)].plot([0,1],[0,1],'k--')
    d.plot.scatter('S_p_ln','S_p_dnn', s=5, c='r_test_all_dnn', ax=ax[len(modelnames)], vmin=0.1, vmax=0.9)
    ax[len(modelnames)].set_title('diff')

    return f


def sparseness_figs():
    
    batch=322
    cellid='ARM030a-40-2'
    modelname=ALL_FAMILY_MODELS[2]
    modelname_ln=ALL_FAMILY_MODELS[3]
    xf, ctx = load_model_xform(cellid, batch, modelname, eval_model=True)
    val = ctx['val']
    val = val.apply_mask()

    f1,ax=plt.subplots(2,2, figsize=column_and_half_tall) #, sharex=True, sharey=True)
    sparseness_example(batch, cellid, modelname, rec=val, ax=ax[0, 0], tag="CNN", colors=['gray', DOT_COLORS['1D CNNx2']])
    sparseness_example(batch, cellid, modelname_ln, rec=val, ax=ax[0, 1], tag="LN", colors=['gray', DOT_COLORS['pop LN']])
    ax[0,1].set_yticks([])

    a1_sparseness_path = int_path  / str(a1) / 'sparseness_data.csv'
    peg_sparseness_path = int_path  / str(peg) / 'sparseness_data.csv'
    modelnames = [ALL_FAMILY_MODELS[0], ALL_FAMILY_MODELS[2], ALL_FAMILY_MODELS[3]]
    pop_reference_model = ALL_FAMILY_POP[2]

    batch=a1
    sparseness_data_a1 = sparseness_by_batch(batch, modelnames=modelnames,
                                             pop_reference_model=pop_reference_model,
                                             save_path=a1_sparseness_path, force_regenerate=False, rec=None)
    batch=peg
    sparseness_data_peg = sparseness_by_batch(batch, modelnames=modelnames,
                                              pop_reference_model=pop_reference_model,
                                              save_path=peg_sparseness_path, force_regenerate=False, rec=None)

    sparseness_data_a1['area']='A1'
    sparseness_data_peg['area']='PEG'
    sparseness_data = pd.concat([sparseness_data_a1,sparseness_data_peg], ignore_index=True)

    d = sparseness_data_a1.loc[sparseness_data_a1.model==1].merge(sparseness_data_a1.loc[sparseness_data_a1.model==2], how='inner', on='cellid',
                                                                  suffixes=('_dnn','_ln'))
    ax[1, 0].plot([0,1],[0,1],'k--')
    d.plot.scatter('S_p_ln','S_p_dnn', s=5, c='r_test_all_dnn', ax=ax[1, 0], vmin=0.1, vmax=0.9)
    ax[1, 0].set_title('Predicted sparseness')
    ax[1, 0].set_aspect('equal')
    ax[1, 0].set_xlabel('pop LN')
    ax[1, 0].set_ylabel('1D CNN')

    r_test_min=0.3
    print(f"r_test_min={r_test_min}")
    sd_r = sparseness_data.loc[sparseness_data['r_test_all']>r_test_min, ['area','cellid','model','S_r']].copy()
    sd_r['model']="act"
    sd_r.columns=['area','cellid','model','S']
    sd_r = sd_r.drop_duplicates()
    sd_p = sparseness_data.loc[sparseness_data['r_test_all']>r_test_min, ['area','cellid','model','S_p']].copy()
    sd_p = sd_p.loc[sd_p['model']>0]
    sd_p['model']=sd_p['model'].astype(str)
    sd_p.columns=['area','cellid','model','S']
    sd = pd.concat([sd_p,sd_r],ignore_index=True)
    sd.loc[sd['model']=='1.0','model']='1D CNN'
    sd.loc[sd['model']=='2.0','model']='pop LN'
    sd['label'] = sd['area'] + " " +sd['model']
    #tres=results.loc[(results[PLOT_STAT]<1) & results[PLOT_STAT]>-0.05]

    #f,ax=plt.subplots()
    sns.stripplot(x='label', y='S', hue='label', data=sd, zorder=0,
                  palette=['gray', DOT_COLORS['1D CNNx2'], DOT_COLORS['pop LN']]*2,
                  hue_order=['A1 act', 'A1 1D CNN', 'A1 pop LN', 'PEG act', 'PEG 1D CNN', 'PEG pop LN'],
                  order=['A1 act', 'A1 1D CNN', 'A1 pop LN', 'PEG act', 'PEG 1D CNN', 'PEG pop LN'],
                  jitter=0.2, size=2, ax=ax[1,1]) #[1,1]
    sns.boxplot(x='label', y='S', data=sd, boxprops={'facecolor': 'None', 'linewidth': 1},
                showcaps=False, showfliers=False, whiskerprops={'linewidth': 0},
                order=['A1 act','A1 1D CNN','A1 pop LN','PEG act','PEG 1D CNN','PEG pop LN'], ax=ax[1,1]) #[1,1]
    plt.xticks(rotation=45, fontsize=6, ha='right')
    ax[1,1].legend_.remove()
    ax[1,1].set_xlabel('')
    ax[1,1].set_title(f"r_test_min={r_test_min}")

    ref_models = ['A1 act','A1 1D CNN','PEG act','PEG 1D CNN']
    test_models = ['A1 1D CNN','A1 pop LN','PEG 1D CNN','PEG pop LN']

    tests = [[m1,m2,st.mannwhitneyu(sd.loc[sd['label']==m1,'S'], sd.loc[sd['label']==m2,'S'], alternative='two-sided')]
             for m1, m2 in zip(ref_models, test_models)]
    print(pd.DataFrame(tests, columns=['ref','test','MannWhitney u,p']))
    print(sd.groupby('label').median())

    f1.tight_layout()

    return f1, tests

if __name__ == '__main__':
    
    f1 = sparseness_figs()

    




