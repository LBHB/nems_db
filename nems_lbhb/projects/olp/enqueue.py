import itertools as itt
import nems0.db as nd

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/hamersky/miniconda3/envs/olp/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/hamersky/nems_db/nems_lbhb/projects/olp/script.py'

# some neurons were added and renamed,and must be reprocecced
# batch = 341
batch = 344
cell_df = nd.get_batch_cells(batch)
cell_list = cell_df['cellid'].tolist()
# cell_list = cell_list[0:5]
# cell_list = [cc for cc in cell_list if cc[:3]=='CLT']
# cell_list = [cc for cc in cell_list if int(cc[3:6]) >= 27]
# cell_list = [cc for cc in cell_list if int(cc[3:6]) < 46]
# cell_list = ohel.manual_fix_units(cell_list)  # So far only useful for two TBR cells


# iterates over every mode, checks what cells have not been fitted with it and runs the fit command.
for nn, cellid in enumerate(cell_list):
    # note = f'OLP_prediction_{cellid}_0-500ms'
    note = f'OLP_weights_{cellid}_0-500ms'
    args = [cellid]
    print(note)
    out = nd.add_job_to_queue(args, note, force_rerun=True,
                              user="greg", codeHash="master",
                              executable_path=executable_path, script_path=script_path,
                              priority=1, GPU_job=0, reserve_gb=0)

    for oo in out:
        print(oo)

print(f'\nenqueued {nn+1} jobs')

