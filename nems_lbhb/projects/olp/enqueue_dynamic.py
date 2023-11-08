import nems0.db as nd
import joblib as jl

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/hamersky/miniconda3/envs/olp/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/hamersky/nems_db/nems_lbhb/projects/olp/script_dynamic.py'


# 2023_07_03.
# path = '/auto/users/hamersky/olp_analysis/2023-05-17_batch344_0-500_metrics' #full one with PRNB layers and paths
# path = '/auto/users/hamersky/olp_analysis/2023-07-21_batch344_0-500_metric'
path = '/auto/users/hamersky/olp_analysis/2023-09-21_batch344_0-500_final'
weight_df = jl.load(path)

batch = int(weight_df.batch.unique()[0])
half_parms = weight_df.loc[(weight_df.dyn_kind=='fh') | (weight_df.dyn_kind=='hf')][['parmfile', 'cellid']].drop_duplicates()
# half_parms = half_parms[100:103]

# iterates over every mode, checks what cells have not been fitted with it and runs the fit command.
for nn, (_, row) in enumerate(half_parms.iterrows()):
    note = f"OLP_dynamic_calc_{row['cellid']}_{row['parmfile']}"
    args = [row['cellid'], batch, row['parmfile']]
    print(note)
    out = nd.add_job_to_queue(args, note, force_rerun=True,
                              user="greg", codeHash="master",
                              executable_path=executable_path, script_path=script_path,
                              priority=1, GPU_job=0, reserve_gb=0)

    for oo in out:
        print(oo)

print(f'\nenqueued {nn+1} jobs')
