import os
import datetime


def kamiak_batch(cellids, batch, modelnames, output_path):
    subdirectory = str(datetime.datetime.now())
    for i, m in enumerate(modelnames):
        for j, c in enumerate(cellids):
            name = f'{m}__{batch}__{c}'
            filename = f'{name}.srun'
            jobname = f'NEMS_model{i}_cell{j}'
            directory_path = os.path.join(output_path, subdirectory)
            full_path = os.path.join(directory_path, filename)

            contents = ("#!/bin/bash\n"
                        "#SBATCH --partition=kamiak\n"
                        f"#SBATCH --job-name={jobname}\n"
                        f"#SBATCH --output=/home/jacob.pennington/nems_logs/{name}.out\n"
                        f"#SBATCH --error=/home/jacob.pennington/nems_logs/{name}.err\n"
                        "#SBATCH --time=1-23:59:00\n"
                        "#SBATCH --nodes=1\n"
                        "#SBATCH --ntasks-per-node=1\n"
                        "python3 /home/jacob.pennington/nems_scripts/fit_xforms.py c batch m")

            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
            with open(full_path, 'w+') as f:
                f.write(contents)


# load results

# (Assuming files have already been copied back from kamiak)

# Need: directory containing the files
#       batch

# 1: Get list of cellids for batch

# 2: For cellid in cellids:
#     If directory/c/ doesn't exist:
#         skip
#     else:
#         xfspec, ctx = xforms.load_analysis(directory/cellid/)
#         modelspec = ctx['modelspec']
#         modelspec.meta['modelpath'] = destination
#         modelspec.meta['figurefile'] = destination+'figure.0000.png'
#         nd.update_results_table(modelspec)
#         destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
#                        batch, cellid, ms.get_modelspec_longname(modelspec))
#w