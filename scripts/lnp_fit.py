import os
import logging
import copy

import numpy as np
import matplotlib.pyplot as plt

import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split
from nems_db.xform_wrappers import generate_recording_uri
from nems_lbhb.lnp_helpers import simulate_spikes

log = logging.getLogger(__name__)

#cellid = 'TAR010c-18-1'
cellid = 'TAR010c-13-1'  # does great for all models
#cellid = 'bbl086b-02-1'  # does crappy for all models
#cellid = 'TAR017b-33-3'  # good for gc model but not so much for others
#cellid = "BRT034f-42-1"  # fine on linear but terrible on GC
#cellid = 'TAR010c-15-4'  # okay on linear but fails on gc
#cellid = 'TAR010c-40-1'  # better than linear by a bit cont. strf clearly fit
batch = 289

modelname = (
        # Retrieve the data with 200Hz sampling and 18 spectral channels
        "ozgf.fs100.ch18"
        # Load the data into a recording object
        "-ld"
        # Split into training and validation sets
        "-splitep"
        "_"  # loaders -> modules
        # Apply log transformation to the stimulus (fixed, no parameters)
        "dlog.f"
        # Spectral filter (nems.modules.weight_channels)
        "-wc.18x1.g"
        # Temporal filter (nems.modules.fir)
        "-fir.1x15"
        # Scale, currently init to 1.
        #"-scl.1"
        # Level shift, usually init to mean response (nems.modules.levelshift)
        "-lvl.1"
        # Nonlinearity (nems.modules.nonlinearity -> double_exponential)
        "-dexp.1"  # TODO: try relu instead of dexp? (rectification)
        "_"  # modules -> fitters
        # Set initial values and do a rough "pre-fit"
        # Initialize fir coeffs to L2-norm of random values
        "-init.lnp.t8"#.L2f"
        # Do the full fits
        #"-lnp.t5"
        #"-nestspec"
        )

# Equivalent to:
#modelname = "ozgf.fs200.ch18-ld-splitep_dlog.f-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init.t3-lnp.t5"

log.info('Initializing modelspec(s) for cell/batch %s/%d...',
         cellid, int(batch))

# Segment modelname for meta information
kws = escaped_split(modelname, '_')
modelspecname = "-".join(kws[1:-1])
loadkey = kws[0]
fitkey = kws[-1]

meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
        'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
        'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
        'githash': os.environ.get('CODEHASH', ''),
        'recording': loadkey}

uri_key = escaped_split(loadkey, '-')[0]
recording_uri = generate_recording_uri(cellid, batch, uri_key)
registry_args = {'cellid': cellid, 'batch': int(batch)}
xfspec = xhelp.generate_xforms_spec(modelname=modelname, meta=meta,
                                    xforms_kwargs=registry_args)

# actually do the fit
ctx = {}
for i, xfa in enumerate(xfspec):
    ctx = xforms.evaluate_step(xfa, ctx)


m = ctx['modelspec']
e = ctx['est']
v = ctx['val']
r = ctx['rec']

p = m.phi()


# Plot spikes vs sim to check model behavior
rate_vector = v['pred'].as_continuous().flatten()
resp = v['resp'].as_continuous().flatten()
ff = v['mask'].as_continuous().flatten() & np.isfinite(rate_vector) \
        & np.isfinite(resp)
rate_vector = rate_vector[ff]
resp = (resp[ff] > 0).astype('int')  # ignore multiple spikes per bin
n_sim = 9
sims = [simulate_spikes(rate_vector) for i in range(n_sim)]
merged = np.vstack((resp, *sims))

fig, axes = plt.subplots(8, 2)
cutoff = int(resp.size/8)
i = 0
j = 0
for k, row in enumerate(axes):
    for ax in row:
        plt.sca(ax)
        if k % 2 == 0:
            plt.plot(rate_vector[i*cutoff:(i+1)*cutoff])
            i += 1
        else:
            plt.imshow(merged[:, j*cutoff:(j+1)*cutoff], aspect='auto',
                       cmap='Greys')
            j += 1


fig.suptitle('1 resp (1st index) and {} sims for data split into 8 segments'
             .format(n_sim))


# Add the simulations as signals in val so that they can be
# viewed in recording browser.
for i, s in enumerate(sims):
    resp_array = copy.deepcopy(v['resp']).as_continuous()
    resp_array[:, ff] = s
    spikes_sig = copy.deepcopy(v['resp'])._modified_copy(resp_array)
    ctx['val']['spikes%d'%i] = spikes_sig

# Separate plot with lots more simulations and not split by time
#fig = plt.figure()
#new_sims = [simulate_spikes(rate_vector) for i in range(n_sim*10)]
#new_merged = np.vstack((resp, *new_sims))
#plt.imshow(new_merged, aspect='auto', cmap='Greys')
#fig.suptitle('1 resp (1st index) and {} sims'.format(n_sim*10))


# TODO: add spike simulations to context so they can be viewed in browser

# TODO: re-examine initial conditions for dexp - they assume data is averaged
#       so probably getting initialized to some weird stuff for this model
