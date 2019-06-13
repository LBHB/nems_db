import os
import datetime
import stat
import logging

import nems_lbhb.baphy as nb
import nems.db as nd
import nems.xforms as xforms

log = logging.getLogger(__name__)


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
            remote_rec = f'{recordings}/{recording_uri.split("/")[-1]}'

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
                        f"'{c}' '{batch}' '{m}' '{remote_rec}' '{recording_uri}'")

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


def kamiak_array(cellids, batch, modelnames, output_path):
    # Put in folder for today's date
    subdirectory = str(datetime.datetime.now()).split(' ')[0]
    directory_path = os.path.join(output_path, subdirectory)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

    # Create a manifest of the recording names needed
    manifest_path = os.path.join(directory_path, 'manifest.sh')
    args_path = os.path.join(directory_path, 'jobs.txt')
    script_path = os.path.join(directory_path, 'batch.srun')
    recording_entries = []
    args_entries = []
    remote_host = "jacob.pennington@kamiak.wsu.edu"
    recordings = "/home/jacob.pennington/nems/recordings/"
    remote_recordings = f"{remote_host}:{recordings}"
    scripts = "/home/jacob.pennington/slurm_scripts/"
    remote_scripts = f"{remote_host}:{scripts}"
    logs = "/home/jacob.pennington/nems_logs/"
    jobs = f"{scripts}/{subdirectory}/jobs.txt"
    failed_jobs = jobs[:-4] + '_failed.txt'

    for j, c in enumerate(cellids):
        # TODO: Don't hardcode the loader options, parse from modelname
        options = {'cellid': c, 'batch': batch, 'stim': 1,
                   'stimfmt': 'ozgf', 'chancount': 18, 'rasterfs': 100}
        # Record unique recording URIs (may be shared for c with same siteid)
        recording_uri = nb.baphy_load_recording_uri(**options)
        if recording_uri not in recording_entries:
            recording_entries.append(recording_uri)
            remote_rec = f'{recordings}/{recording_uri.split("/")[-1]}'

        for i, m in enumerate(modelnames):

            args = f"{c} {batch} {m} {remote_rec} {recording_uri}"
            args_entries.append(args)

    script_contents = (
              "#!/bin/bash\n"
              "#SBATCH --partition=kamiak\n"
              "#SBATCH --job-name=NEMS\n"
              f"#SBATCH --array=1-{len(args_entries)}\n"
              f"#SBATCH --output={logs}/{subdirectory}/NEMS.out\n"
              f"#SBATCH --error={logs}/{subdirectory}/NEMS.err\n"
              "#SBATCH --time=1-23:59:00\n"
              "#SBATCH --nodes=1\n"
              "#SBATCH --ntasks-per-node=1\n"
              "\n"
              f"job=$(sed \"${{SLURM_ARRAY_TASK_ID}}q;d\" {jobs})\n"
              "args=($job)\n"
              "\n"
              "# Call cleanup function on cancelled jobs\n"
              "function clean_up {\n"
              "    echo \"${args[0]} ${args[1]} ${args[2]}"
              f"    ${{args[3]}} ${{args[4]}}\" >> {failed_jobs}\n"
              "}\n"
              "# trap termination signals\n"
              "trap 'clean_up' SIGINT SIGTERM\n"
              "python3 /home/jacob.pennington/nems_scripts/fit_xforms.py "
              "${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]}"
              )

    # Write recording_uri list to manifest

    # Copy recording files in separate scp commands since
    # a full batch might become very large for a single copy
    manifest_lines = [f'rsync -avx --ignore-existing {e} {remote_recordings}/{e.split("/")[-1]}'
                      for e in recording_entries]
    # But all scripts can be copied over in one command
    manifest_lines.append(f'scp {args_path} {remote_scripts}/{subdirectory}/')
    manifest_lines.append(f'scp {script_path} {remote_scripts}/{subdirectory}/')
    manifest_lines.insert(0, f'ssh {remote_host} "mkdir -p {scripts}/{subdirectory}"')
    manifest_lines.insert(0, f'ssh {remote_host} "mkdir -p {logs}/{subdirectory}"')
    manifest_lines.insert(0, '#!/bin/bash')
    manifest_contents = '\n'.join([e for e in manifest_lines])
    args_contents = '\n'.join([a for a in args_entries])
    with open(manifest_path, 'w+') as manifest:
        manifest.write(manifest_contents)
    st = os.stat(manifest_path)
    os.chmod(manifest_path, st.st_mode | stat.S_IEXEC)  # make it executable
    with open(args_path, 'w+') as args:
        args.write(args_contents)
    with open(script_path, 'w+') as script:
        script.write(script_contents)


def kamiak_to_database(cellids, batch, modelnames, source_path):
# Assumes files have already been copied back from kamiak and
# stored in source_path
    for cellid in cellids:
        for modelname in modelnames:
            path = os.path.join(source_path, batch, cellid, modelname)
            if not os.path.exists(path):
                log.warning("missing fit for: \n%s\n%s\n%s\n"
                            "using path: %s\n",
                            batch, cellid, modelname, path)
            else:
                 xfspec, ctx = xforms.load_analysis(path)
                 modelspec = ctx['modelspec']
                 preview = modelspec.meta.get('figurefile', None)
                 nd.update_results_table(modelspec, preview=preview)
                 if 'log' not in ctx:
                     ctx['log'] = 'missing log'
                 xforms.save_analysis(None, None, ctx['modelspec'], xfspec,
                                      ctx['figures'], ctx['log'])
