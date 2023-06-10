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
modelname = sys.argv[3]

print(f"Running OLP Prediction cell {cellid}")


# first run 2022_12 with Mateo
# Also for 2023_04_25 new code to run all the data including Prince, with things I added along the way
# _ = OLP_fit_cell_pred_individual(cellid, 344, threshold=None, snip=[0, 0.5], pred=False,
#                                  fit_epos='syn', fs=100)

_ = OLP_fit_partial_weights_individual(cellid, batch, snip=[0, 0.5], pred=True, fs=100, modelname=modelname)

# Mark completed in the queue. Note that this should happen last thing!
# Otherwise the job might still crash after being marked as complete.
if queueid:
    nd.update_job_complete(queueid)
