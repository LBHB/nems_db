from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt

from nems_lbhb.baphy_experiment import BAPHYExperiment

from nems0.analysis.gammatone.gtgram import gtgram
import nems0.epoch as ep
from nems0 import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
import pandas as pd

# all sites that need to be run
batch = 348
siteids, cellids = db.get_batch_sites(batch)

full_model_save_path = Path('/auto/users/wingertj/models/spatial_decoding/')
model_save_path = Path('/auto/users/wingertj/models/decoding_layer_removal')

# get dataframes of current models
full_model = pd.read_pickle(str(full_model_save_path/'decoder_df.pkl'))
model_df = pd.read_pickle(str(model_save_path/'decoder_df.pkl'))
all_df = pd.concat([full_model, model_df],axis=0).reset_index(drop=True)

# get sites that are already done
anti_causal_df = all_df.loc[(all_df['fircaus']=='True')].reset_index(drop=True).copy()
anti_causal_ids = anti_causal_df['siteid'].unique()

causal_df = all_df.loc[(all_df['fircaus']=='') & (all_df['tlyr']!='rlyr')].reset_index(drop=True).copy()
causal_ids = causal_df['siteid'].unique()

# get list of sites for anti-causal models and causal models
# anti_causal_run = [id for id in siteids if id not in anti_causal_ids]
# anti_causal_modelnames = ['-rlyr.13.-perms.10.-input.resp.-firfilt.True', '-rlyr.4.-perms.10.-input.resp.-firfilt.True', '-rlyr.56.-perms.10.-input.resp.-firfilt.True']

causal_run = [id for id in siteids if id not in causal_ids]
causal_modelnames = ['-rlyr.13.-perms.10.-input.resp.-firfilt.False', '-rlyr.4.-perms.10.-input.resp.-firfilt.False', '-rlyr.56.-perms.10.-input.resp.-firfilt.False']

full_run = siteids
full_modelnames = ['-rlyr..-perms.0.-input.resp.-firfilt.True']

executable_path = '/auto/users/wingertj/nems_db/nems_lbhb/projects/freemoving/decode_fit_wrapper'
script_path = '/auto/users/wingertj/nems_db/nems_lbhb/projects/freemoving/decoder_model_fit_script.py'
GPU_job=True

run_sites = [full_run, causal_run]
run_models = [full_modelnames, causal_modelnames]
# modelnames = ['dc.dlc_dist.nmse']
# modelnames = ['-rlyr.13.-perms.10.-input.resp.-firfilt.True', '-rlyr.4.-perms.10.-input.resp.-firfilt.True', '-rlyr.56.-perms.10.-input.resp.-firfilt.True']

force_rerun = False
run_in_lbhb = True

for siteids, modelnames in zip(run_sites, run_models):
    if run_in_lbhb:
        # first models, run locally so that recordings get generated.
        r = db.enqueue_models(siteids, batch, modelnames, executable_path=executable_path,
                              script_path=script_path, GPU_job=GPU_job, user="wingertj",
                              linux_user='wingertj', force_rerun=force_rerun)
        for a,b in r:
            print(a,b)
    else:
        # exacloud

        # exacloud queue settings:
        exa_executable_path = '/home/users/davids/anaconda3/envs/nems/bin/python'
        exa_script_path = '/home/users/davids/nems_db/scripts/nems0_scripts/fit_single.py'
        ssh_key = '/home/svd/.ssh/id_rsa'
        user = "davids"
        lbhb_user = "svd"

        enqueue_exacloud_models(
            cellist=siteids, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=exa_executable_path, script_path=exa_script_path, useGPU=GPU_job)

