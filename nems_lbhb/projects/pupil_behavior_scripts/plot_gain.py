"""
Simple script to inspect gain/dc signals over time for sdexp model fits
"""

import matplotlib.pyplot as plt
import nems0.xform_helper as xhelp

cellid = 'BRT016f-a1'
batch = 309
modelname = 'psth.fs20.pup-ld-st.pup.afl.pxf-ref-psthfr.s_sdexp.S_jk.nf20-basic'
modelname0p = 'psth.fs20.pup-ld-st.pup0.afl.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'
modelname0b = 'psth.fs20.pup-ld-st.pup.afl0.pxf0-ref-psthfr.s_sdexp.S_jk.nf20-basic'

xf, ctx = xhelp.load_model_xform(cellid=cellid, batch=batch, modelname=modelname)
xf0p, ctx0p = xhelp.load_model_xform(cellid=cellid, batch=batch, modelname=modelname0p)
xf0b, ctx0b = xhelp.load_model_xform(cellid=cellid, batch=batch, modelname=modelname0b)

r = ctx['val'].apply_mask()
r0p = ctx0p['val'].apply_mask()
r0b = ctx0b['val'].apply_mask()

f, ax = plt.subplots(1, 1)

ax.plot(r0p['gain']._data.T, label='pupil shuffle', alpha=0.2)
ax.plot(r0b['gain']._data.T, label='beh shuffle', alpha=0.2)
ax.plot(r['gain']._data.T, label='full model', color='k')


ax.legend(frameon=False)
ax.set_ylabel('gain')
ax.set_xlabel('Time')

# plot the psth in the different behavior conditions
# for the full model, and the pupil shuffled model
f, ax = plt.subplots(1, 2)
state_chans = [f for f in ctx['val'].epochs.name.unique() if 'FILE' in f]

for chan in state_chans:
    rec = ctx['val']
    rec = rec.and_mask(chan) 
    rec0 = ctx0p['val']
    rec0 = rec0.and_mask(chan)

    full_model = rec['pred'].extract_epoch('REFERENCE', mask=rec['mask'])
    pup0_model = rec0['pred'].extract_epoch('REFERENCE', mask=rec0['mask'])

    ax[0].set_title('Full model')
    ax[0].plot(full_model.mean(axis=(0, 1)), label=chan)

    ax[1].set_title('pup0 model')
    ax[1].plot(pup0_model.mean(axis=(0, 1)), label=chan)
    ax[1].legend(frameon=False)

plt.show()