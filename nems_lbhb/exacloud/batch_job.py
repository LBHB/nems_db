import argparse
import re
import subprocess
import sys
import datetime
from pathlib import Path


def write_batch_file(job_arguments,
                     queueid=None,
                     time_limit=10,
                     reserve_gb=0,
                     use_gpu=False,
                     high_mem=False,
                     exclude=None):
    """Parses the arguments and creates the slurm sbatch file.

    Batch files are saved in the users home directory in "job_history".

    :param job_arguments: Arguments to srun.
    :param queueid: Queueid for updating queuemaster.
    :param time_limit: Hours that the job will run for. Hard cap, after which the job will be terminated.
    :param high_mem: Whether or not the job requires a GPU with a larger amount of memory. See ACC docs for GPU.
    :param exclude: List of nodes to exclude. Comma separated values, no spaces.

    :return: The file location of the batch file.
    """
    job_dir = Path.home() / 'job_history'
    # create job dir if doesn't exist
    job_dir.mkdir(exist_ok=True, parents=True)

    dt_string = datetime.datetime.now().strftime('%Y-%m-%d-T%H%M%S')
    job_file_name = dt_string + '_slurmjob.sh'
    job_file_loc = job_dir / job_file_name

    job_log_name = dt_string + '_jobid'
    job_log_loc = job_dir / job_log_name

    # first two components of args (i.e. exec and script)
    job_name = []
    # chop if paths
    for arg in job_arguments[:2]:
        if Path(arg).exists():
            job_name.append(Path(arg).name)
        else:
            job_name.append(arg)
    job_name = ':'.join(job_name)

    job_comment = ' '.join(job_arguments[2:])

    # format time into HH:MM:SS
    time_mins = round(time_limit * 60)
    time_hours = time_mins // 60
    time_mins -= time_hours * 60

    reserve_gb=int(reserve_gb)

    with open(job_file_loc, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'#SBATCH --account=lbhb\n')
        f.write(f'#SBATCH --time={time_hours:02d}:{time_mins:02d}:00\n')
        f.write(f'#SBATCH --cpus-per-task=1\n')
        if use_gpu:
            if reserve_gb==0:
                reserve_gb=48
        else:
            if reserve_gb==0:
                reserve_gb=8
        f.write(f'#SBATCH --mem={reserve_gb}G\n')
        f.write(f'#SBATCH --job-name={job_name}\n')
        f.write(f'#SBATCH --comment="{job_comment}"\n')
        f.write(f'#SBATCH --output={str(job_log_loc)}%j_log.out\n')

        if queueid is not None:  # to work with queuemaster need to add in queueid env
            f.write(f'#SBATCH --export=ALL,QUEUEID={queueid},LD_LIBRARY_PATH=:/home/users/davids/miniconda3/envs/nems0/lib/:/home/users/davids/miniconda3/envs/nems0/lib/python3.9/site-packages/nvidia/cudnn/lib:/home/users/davids/miniconda3/envs/nems/lib/:/home/users/davids/miniconda3/envs/nems/lib/python3.9/site-packages/nvidia/cudnn/lib:/home/users/davids/miniconda3/envs/nems/lib/:/home/users/davids/miniconda3/envs/nems/lib/python3.9/site-packages/nvidia/cudnn/lib\n')

        if use_gpu:
            f.write(f'#SBATCH --partition=gpu\n')
            if high_mem:
                f.write(f'#SBATCH --gres=disk:5,gpu:v100:1\n')
            else:
                f.write(f'#SBATCH --gres=disk:5,gpu:1\n')

            #f.write(f'module add /home/exacloud/software/modules/cuda/10.1.243\n')
            #f.write(f'module add /home/exacloud/software/modules/cudnn/7.6-10.1\n')
            #f.write(f'module add /home/exacloud/software/modules/cuda/11.4.2\n')
            #f.write(f'module add /home/exacloud/software/modules/cudnn/8.2.4.15-11.4\n')
        else:
            f.write(f'#SBATCH --partition=exacloud\n')
            f.write(f'#SBATCH --gres=disk:5\n')

        if exclude is not None:
            f.write(f'#SBATCH --exclude={exclude}\n')
  
        #f.write('conda activate nems\n')

        f.write(' '.join(['srun'] + job_arguments))
        f.write('\n')

    return str(job_file_loc)


def queue_batch_file(job_file_loc):
    """Calls sbatch on the job file. Searches for jobid to return.

    :param job_file_loc: Location of the job file.

    :return: Returns tuple of stdout, stderr.
    """
    ret = subprocess.run(['sbatch', str(job_file_loc)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = ret.stdout.decode()  # stdout is a bytes object

    try:
        jobid = re.match(r'^Submitted batch job (\d*)$', str(stdout)).group(1)
    except (IndexError, AttributeError):
        jobid = None

    return jobid, stdout


if __name__ == '__main__':
    """Creates and runs a slurm batch file. 
    
    Named arguments must be passed in before unnamed arguments. 
    
    Ex:
    $ python batch_job.py --queueid=1234567 --time_limit=10 python fit_single.py cellid batch modelname
    """
    # parse arguments in order to collect all args into list, except for QUEUEID
    parser = argparse.ArgumentParser(description='Run jobs on exacloud!')
    parser.add_argument('--queueid', default=None, help='The tQueue QID.')
    parser.add_argument('--time_limit', type=float, default=10, help='The time limit for the job in hours.')
    parser.add_argument('--reserve_gb', type=float, default=0, help='Maximum GB to require (0 means use defaults).')
    parser.add_argument('--use_gpu', action='store_true', help='Whether to start a CPU or GPU job.')
    parser.add_argument('--high_mem', action='store_true', help='Whether to use GPU with more memory.')
    parser.add_argument('--exclude', default=None, help='List of nodes to exclude. Comma separated values, no spaces.')

    parser.add_argument('arguments', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    job_file_loc = write_batch_file(args.arguments, args.queueid, args.time_limit, args.reserve_gb, 
                                    args.use_gpu)
    jobid, stdout = queue_batch_file(job_file_loc)

    if jobid is not None:
        print(jobid)
        sys.exit(0)
    else:
        sys.exit(stdout)
