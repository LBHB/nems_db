#Example of how to cache parameters for REM analysis.
#ZPS 2018-10-01

import pandas as pd
import nems.db as nd
from nems_lbhb.baphy_io import get_rem, cache_rem_options, load_rem_options
from os.path import basename

#Load the file paths for pupil data.
batch_cell_data = nd.get_batch_cell_data(batch=289)
parmfiles = batch_cell_data['parm'].unique().tolist()
parmfiles = [s.replace('.m', '') for s in parmfiles]
pupilfilepaths = [s + '.pup.mat' for s in parmfiles]

#Choose a recording.
recording_to_analyze = pupilfilepaths[0]
#Try analyzing using the default parameters.
_, options = get_rem(recording_to_analyze)
#Look at the results, adjust the parameters if necessary.
options["rem_max_pupil_sd"] = 0.03
#Check if the changes to the parameters gives better results.
_, options = get_rem(recording_to_analyze, **options)
#Once the results are acceptable, save them in the cache.
cache_rem_options(recording_to_analyze, **options)

#To review all analyzed recordings in the batch:
for recording in pupilfilepaths:
    try:
        get_rem(recording, **load_rem_options(recording))
    except ValueError:
        continue

#To review the REM parameters for the batch:
rem_options = []
for recording in pupilfilepaths:
    try:
        options = load_rem_options(recording)
        options["recording"] = basename(recording)
        rem_options.append(options)
    except ValueError:
        continue
rem_options = pd.DataFrame(rem_options)
rem_options.set_index('recording')
