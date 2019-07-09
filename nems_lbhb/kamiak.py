import os
import datetime
import stat
import logging
import itertools

import nems_lbhb.baphy as nb
import nems.db as nd
import nems.xforms as xforms
from nems import get_setting

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
    reverse_manifest_path = os.path.join(directory_path, 'reverse_manifest.sh')
    args_path = os.path.join(directory_path, 'jobs.txt')
    script_path = os.path.join(directory_path, 'batch.srun')
    recording_entries = []
    args_entries = []
    remote_host = "jacob.pennington@kamiak.wsu.edu"
    recordings = "/home/jacob.pennington/nems/recordings/"
    results = f"/home/jacob.pennington/nems/results/{batch}"
    remote_recordings = f"{remote_host}:{recordings}"
    remote_results = f"{remote_host}:{results}"
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
              f"#SBATCH --output={logs}/{subdirectory}/NEMS.%A_%a.out\n"
              f"#SBATCH --error={logs}/{subdirectory}/NEMS.%A_%a.err\n"
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
              "${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]}\n"
              "echo \"task ${SLURM_ARRAY_TASK_ID} complete\""
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

    reverse_manifest_lines = [f'rsync -avx {remote_results} $1']
    reverse_manifest_lines.insert(0, '#!/bin/bash')
    reverse_manifest_contents = '\n'.join([r for r in reverse_manifest_lines])

    args_contents = '\n'.join([a for a in args_entries])

    with open(manifest_path, 'w+') as manifest:
        manifest.write(manifest_contents)
    os.chmod(manifest_path, 0o777) # open up permissions for other users
    st = os.stat(manifest_path)
    os.chmod(manifest_path, st.st_mode | stat.S_IEXEC)  # make it executable

    with open(reverse_manifest_path, 'w+') as reverse_manifest:
        reverse_manifest.write(reverse_manifest_contents)
    os.chmod(reverse_manifest_path, 0o777)
    rst = os.stat(reverse_manifest_path)
    os.chmod(reverse_manifest_path, rst.st_mode | stat.S_IEXEC)

    with open(args_path, 'w+') as args:
        args.write(args_contents)
    with open(script_path, 'w+') as script:
        script.write(script_contents)


def kamiak_to_database(cellids, batch, modelnames, source_path,
                       executable_path=None, script_path=None):
    # Assumes files have already been copied back from kamiak and
    # stored in source_path
#    missing = []
#    for cellid in cellids:
#        for modelname in modelnames:
#            path = os.path.join(source_path, batch, cellid, modelname)
#            if not os.path.exists(path):
#                log.warning("missing fit for: \n%s\n%s\n%s\n"
#                            "using path: %s\n",
#                            batch, cellid, modelname, path)
#                missing.append((batch, cellid, modelname))
#            else:
#                 xfspec, ctx = xforms.load_analysis(path)
#                 preview = ctx['modelspec'].meta.get('figurefile', None)
#                 if 'log' not in ctx:
#                     ctx['log'] = 'missing log'
#                 xforms.save_analysis(None, None, ctx['modelspec'], xfspec,
#                                      ctx['figures'], ctx['log'])
#                 nd.update_results_table(ctx['modelspec'], preview=preview)

    user = 'jacob'
    linux_user = 'nems'
    allowqueuemaster = 1
    waitid = 0
    parmstring = ''
    rundataid = 0
    priority = 1
    reserve_gb = 0
    codeHash = 'kamiak'

    if executable_path in [None, 'None', 'NONE', '']:
        executable_path = get_setting('DEFAULT_EXEC_PATH')
    if script_path in [None, 'None', 'NONE', '']:
        script_path = get_setting('DEFAULT_SCRIPT_PATH')

    combined = [(c, b, m) for c, b, m in
                itertools.product(cellids, [batch], modelnames)]
    notes = ['%s/%s/%s' % (c, b, m) for c, b, m in combined]
    commandPrompts = ["%s %s %s %s %s" % (executable_path, script_path,
                                          c, b, m)
                      for c, b, m in combined]

    engine = nd.Engine()
    for (c, b, m), note, commandPrompt in zip(combined, notes, commandPrompts):
        path = os.path.join(source_path, batch, c, m)
        if not os.path.exists(path):
            log.warning("missing fit for: \n%s\n%s\n%s\n"
                        "using path: %s\n",
                        batch, c, m, path)
            continue
        else:
             xfspec, ctx = xforms.load_analysis(path)
             preview = ctx['modelspec'].meta.get('figurefile', None)
             if 'log' not in ctx:
                 ctx['log'] = 'missing log'
             xforms.save_analysis(None, None, ctx['modelspec'], xfspec,
                                  ctx['figures'], ctx['log'])
             nd.update_results_table(ctx['modelspec'], preview=preview)

        conn = engine.connect()
        sql = 'SELECT * FROM tQueue WHERE note="' + note +'"'
        r = conn.execute(sql)
        if r.rowcount>0:
            # existing job, figure out what to do with it
            x=r.fetchone()
            queueid = x['id']
            complete = x['complete']

            if complete == 1:
                # Do nothing - the queue already shows a complete job
                pass

            elif complete == 2:
                # Change dead to complete
                sql = "UPDATE tQueue SET complete=1, killnow=0 WHERE id={}".format(queueid)
                r = conn.execute(sql)

            else:
                # complete in [-1, 0] -- already running or queued
                # Do nothing
                pass

        else:
            # New job
            sql = "INSERT INTO tQueue (rundataid,progname,priority," +\
                   "reserve_gb,parmstring,allowqueuemaster,user," +\
                   "linux_user,note,waitid,codehash,queuedate) VALUES"+\
                   " ({},'{}',{}," +\
                   "{},'{}',{},'{}'," +\
                   "'{}','{}',{},'{}',NOW())"
            sql = sql.format(rundataid, commandPrompt, priority, reserve_gb,
                             parmstring, allowqueuemaster, user, linux_user,
                             note, waitid, codeHash)
            r = conn.execute(sql)

        conn.close()
