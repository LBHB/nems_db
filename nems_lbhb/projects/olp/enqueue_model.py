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
# batch = 344  # ferrets



# 2023_06_09. Adding Stephen modelfits
# batch=345  # stephen modelfits
batch =328  # marm
cell_df = nd.get_batch_cells(batch)
cell_list = cell_df['cellid'].tolist()
cell_list = cell_list[111:119]
# modelname = "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x70-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"
# modelname = "gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.Nx1x120-fir.25x1x120-wc.120xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb5-lite.tf.lr1e4"

# cell_list = cell_list[0:5]
# cell_list = [cc for cc in cell_list if cc[:3]=='CLT']
# cell_list = [cc for cc in cell_list if int(cc[3:6]) >= 27]
# cell_list = [cc for cc in cell_list if int(cc[3:6]) < 46]
# cell_list = ohel.manual_fix_units(cell_list)  # So far only useful for two TBR cells


# iterates over every mode, checks what cells have not been fitted with it and runs the fit command.
for nn, cellid in enumerate(cell_list):
    # note = f'OLP_prediction_{cellid}_0-500ms'
    note = f'OLP_weights_{cellid}_0-500ms'
    try:
        args = [cellid, batch, modelname]
    except:
        args = [cellid, batch]
    print(note)
    out = nd.add_job_to_queue(args, note, force_rerun=True,
                              user="greg", codeHash="master",
                              executable_path=executable_path, script_path=script_path,
                              priority=1, GPU_job=0, reserve_gb=0)

    for oo in out:
        print(oo)

print(f'\nenqueued {nn+1} jobs')
