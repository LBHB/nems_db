"""
Project population data onto custom axes
    e.g. principle noise axis, TAR/REF discrimination axis, HIT/MISS axis...
"""

import nems_lbhb.baphy as nb
import nems_lbhb.dimensionality_reduction as dr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

site = 'TAR010c'
batch = 307
rasterfs = 40
ops = {'cellid': site, 'batch': 307, 'rasterfs': rasterfs, 'stimfmt': 'parm'}
rec = nb.baphy_load_recording_file(**ops)
rec['resp'] = rec['resp'].rasterize()

# recs w/ different masks for plotting below
hit_rec = rec.copy()
hit_rec = hit_rec.and_mask(['HIT_TRIAL'])
miss_rec = rec.copy()
miss_rec = miss_rec.and_mask(['MISS_TRIAL'])
pass_rec = rec.copy()
pass_rec = pass_rec.and_mask(['PASSIVE_EXPERIMENT'])

# mask passive trials, correct trials, miss trials. Keep all ref and target
rec = rec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL', 'MISS_TRIAL'])

# ==================  add signal with projection of resp onto the noise axis =============================
rec = dr.get_noise_projection(rec, epochs=None, collapse=True)

# ================  add signal with projection onto TARGET vs. REFERENCE discrim axis ====================
# define by collapsing over the first 200ms evoked data of REF/TAR.
# need to create specialized epochs for this
prestim_bounds = rec['resp'].get_epoch_bounds('PreStimSilence')
prestim_len = prestim_bounds[0][1] - prestim_bounds[0][0]
tar_200 = rec.epochs[rec.epochs.name=='TARGET']
tar_200['start'] += prestim_len
tar_200['end'] = tar_200['start'] + 0.2

ref_200 = rec.epochs[rec.epochs.name=='REFERENCE']
ref_200['start'] += prestim_len
ref_200['end'] = ref_200['start'] + 0.2
# add epochs
rec['resp'].add_epoch('TARGET_first_200ms', tar_200[['start', 'end']].values)
rec['resp'].add_epoch('REFERENCE_first_200ms', ref_200[['start', 'end']].values)
rec.epochs.update(rec['resp'].epochs)

rec = dr.get_discrimination_projection(rec, epoch1='TARGET_first_200ms', epoch2='REFERENCE_first_200ms', collapse=True)

# ============================== add choice discrimination projection ================================
# axis separating TARGET_first_200ms for HIT and MISS trials
# need to first make a mask over active TARGET_first_200ms. Then, get axis between HIT / MISS
rec_choice = rec.copy()
rec_choice = rec.create_mask(True)
rec_choice = rec_choice.and_mask(['HIT_TRIAL', 'MISS_TRIAL'])
rec_choice = rec_choice.and_mask(['TARGET_first_200ms'])
rec = dr.get_discrimination_projection(rec_choice, epoch1='HIT_TRIAL', epoch2='MISS_TRIAL', collapse=True)


# ================================ Visualize projections =====================================
projections = ['noise_projection', 
                   'TARGET_first_200ms_vs_REFERENCE_first_200ms_projection',
                   'HIT_TRIAL_vs_MISS_TRIAL_projection']
titles = ['Projection onto Noise axis', 'Projection onto REF/TAR axis', 'Project onto HIT/MISS axis']
for projection, title in zip(projections, titles):

    hit = rec[projection].extract_epoch('TARGET', mask=hit_rec['mask'])
    miss = rec[projection].extract_epoch('TARGET', mask=miss_rec['mask'])
    hitr = rec[projection].extract_epoch('REFERENCE', mask=hit_rec['mask'])
    missr = rec[projection].extract_epoch('REFERENCE', mask=miss_rec['mask'])
    passt = rec[projection].extract_epoch('TARGET', mask=pass_rec['mask'])
    passr = rec[projection].extract_epoch('REFERENCE', mask=pass_rec['mask'])

    t = np.linspace(0, hit.shape[-1] / rasterfs, hit.shape[-1])
    tr = np.linspace(0, hitr.shape[-1] / rasterfs, hitr.shape[-1])

    f, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    m = hit.mean(axis=0).squeeze()
    sem = hit.std(axis=0).squeeze() / np.sqrt(hit.shape[0])
    ax[0].plot(t, m, color='purple', lw=2, label='HIT')
    ax[0].fill_between(t, m-sem, m+sem, color='purple', lw=0, alpha=0.3)

    m = miss.mean(axis=0).squeeze()
    sem = miss.std(axis=0).squeeze() / np.sqrt(miss.shape[0])
    ax[0].plot(t, m, color='gold', lw=2, label='MISS')
    ax[0].fill_between(t, m-sem, m+sem, color='gold', lw=0, alpha=0.3)

    m = passt.mean(axis=0).squeeze()
    sem = passt.std(axis=0).squeeze() / np.sqrt(passt.shape[0])
    ax[0].plot(t, m, color='red', lw=2, label='PASSIVE')
    ax[0].fill_between(t, m-sem, m+sem, color='red', lw=0, alpha=0.3)

    ax[0].set_title('TARGET {}'.format(title))
    ax[0].set_xlabel('Time')
    ax[0].legend()


    m = hitr.mean(axis=0).squeeze()
    sem = hitr.std(axis=0).squeeze() / np.sqrt(hitr.shape[0])
    ax[1].plot(tr, m, color='purple', lw=2, label='HIT')
    ax[1].fill_between(tr, m-sem, m+sem, color='purple', lw=0, alpha=0.3)

    m = missr.mean(axis=0).squeeze()
    sem = missr.std(axis=0).squeeze() / np.sqrt(missr.shape[0])
    ax[1].plot(tr, m, color='gold', lw=2, label='MISS')
    ax[1].fill_between(tr, m-sem, m+sem, color='gold', lw=0, alpha=0.3)

    m = passr.mean(axis=0).squeeze()
    sem = passr.std(axis=0).squeeze() / np.sqrt(passr.shape[0])
    ax[1].plot(tr, m, color='red', lw=2, label='PASSIVE')
    ax[1].fill_between(tr, m-sem, m+sem, color='red', lw=0, alpha=0.3)

    ax[1].set_title('REFERENCE {}'.format(title))
    ax[1].set_xlabel('Time')
    ax[1].legend()

    f.tight_layout()

plt.show()