import os
import sys

import nems0.db as nd
import nems0.utils
from nems_lbhb.projects.olp.OLP_fit import OLP_fit_cell_pred_individual
from nems_lbhb.projects.olp.OLP_fit import OLP_fit_partial_weights_individual

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems0.utils.progress_fun = nd.update_job_tick
else:
    queueid = 0

if queueid:
    print("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# all parameters are pased as string, ensure proper datatypes
cellid = sys.argv[1]
batch = int(sys.argv[2])
try:
    modelname = sys.argv[3]
except:
    modelname = None

print(f"Running OLP Prediction cell {cellid}")


# first run 2022_12 with Mateo
# Also for 2023_04_25 new code to run all the data including Prince, with things I added along the way
# _ = OLP_fit_cell_pred_individual(cellid, 344, threshold=None, snip=[0, 0.5], pred=False,
#                                  fit_epos='syn', fs=100)

# cellid = 'CLT019a-015-2'
# batch = 345
# modelname = "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"

if modelname:
    _ = OLP_fit_partial_weights_individual(cellid, batch, snip=[0, 0.5], pred=False, fs=100, modelname=modelname)
else:
    _ = OLP_fit_partial_weights_individual(cellid, batch, snip=[0, 0.5], pred=False, fs=100, modelname=None)

# Mark completed in the queue. Note that this should happen last thing!
# Otherwise the job might still crash after being marked as complete.
if queueid:
    nd.update_job_complete(queueid)
