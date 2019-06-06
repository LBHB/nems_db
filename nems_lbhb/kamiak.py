import os
import datetime
import stat

import nems_lbhb.baphy as nb


def kamiak_batch(cellids, batch, modelnames, output_path):
    # Put in folder for today's date
    subdirectory = str(datetime.datetime.now()).split(' ')[0]
    directory_path = os.path.join(output_path, subdirectory)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

    # Create a manifest of the recording names needed
    manifest_path = os.path.join(directory_path, 'manifest.sh')
    recording_entries = []
    script_entries = []
    remote_host = "jacob.pennington@kamiak.wsu.edu"
    recordings = "/home/jacob.pennington/nems/recordings/"
    remote_recordings = f"{remote_host}:{recordings}"
    scripts = "/home/jacob.pennington/slurm_scripts/"
    remote_scripts = f"{remote_host}:{scripts}"

    for j, c in enumerate(cellids):
        # TODO: Don't hardcode the loader options, parse from modelname
        options = {'cellid': c, 'batch': batch, 'stim': 1,
                   'stimfmt': 'ozgf', 'chancount': 18, 'rasterfs': 100}
        # Record unique recording URIs (may be shared for c with same siteid)
        recording_uri = nb.baphy_load_recording_uri(**options)
        if recording_uri not in recording_entries:
            recording_entries.append(recording_uri)
            local_rec = f'{recordings}/{recording_uri.split("/")[-1]}'

        for i, m in enumerate(modelnames):
            name = f'{m}__{batch}__{c}'
            filename = f'{name}.srun'
            jobname = f'NEMS_model{i}_cell{j}'
            full_path = os.path.join(directory_path, filename)

            contents = ("#!/bin/bash\n"
                        "#SBATCH --partition=kamiak\n"
                        f"#SBATCH --job-name={jobname}\n"
                        f"#SBATCH --output=/home/jacob.pennington/nems_logs/{subdirectory}/{name}.out\n"
                        f"#SBATCH --error=/home/jacob.pennington/nems_logs/{subdirectory}/{name}.err\n"
                        "#SBATCH --time=1-23:59:00\n"
                        "#SBATCH --nodes=1\n"
                        "#SBATCH --ntasks-per-node=1\n"
                        "python3 /home/jacob.pennington/nems_scripts/fit_xforms.py "
                        f"'{c}' '{batch}' '{m}' '{local_rec}'")

            with open(full_path, 'w+') as script:
                script.write(contents)
            script_entries.append(full_path)

    # Write recording_uri list to manifest

    # Copy recording files in separate scp commands since
    # a full batch might become very large for a single copy
    manifest_lines = [f'scp {e} {remote_recordings}/{e.split("/")[-1]}'
                      for e in recording_entries]
    # But all scripts can be copied over in one command
    all_scripts = ' '.join(script_entries)
    manifest_lines += [f'scp {all_scripts} {remote_scripts}/{subdirectory}/']
    manifest_lines.insert(0, f'ssh {remote_host} "mkdir -p {scripts}/{subdirectory}"')
    manifest_lines.insert(0, '#!/bin/bash')
    manifest_contents = '\n'.join([str(e) for e in manifest_lines])
    with open(manifest_path, 'w+') as manifest:
        manifest.write(manifest_contents)
    st = os.stat(manifest_path)
    os.chmod(manifest_path, st.st_mode | stat.S_IEXEC)  # make it executable


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