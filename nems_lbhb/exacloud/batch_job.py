import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def write_batch_file(job_arguments, queueid=None):
    """Parses the arguments and creates the slurm sbatch file.

    Batch files are saved in the users home directory in "job_history".

    :param job_arguments: Arguments to srun.
    :param queueid: Queueid for updating queuemaster.

    :return: The file location of the batch file.
    """
    job_dir = Path.home() / 'job_history'
    # create job dir if doesn't exist
    job_dir.mkdir(exist_ok=True, parents=True)

    dt_string = datetime.now().strftime('%Y-%m-%d-T%H%M%S')
    job_file_name = dt_string + 'slurmjob.sh'
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

    with open(job_file_loc, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --account=lbhb\n')
        f.write('#SBATCH --time=10:00:00\n')
        f.write('#SBATCH --partition=gpu\n')
        f.write('#SBATCH --cpus-per-task=1')
        f.write('#SBATCH --mem=4G\n')
        f.write('#SBATCH --gres=disk:5\n')
        f.write(f'#SBATCH --job-name={job_name}\n')
        f.write(f'#SBATCH --comment="{job_comment}"\n')
        f.write(f'#SBATCH --output={str(job_log_loc)}%j_log.out\n')
        if queueid is not None:  # to work with queuemaster need to add in queueid env
            f.write(f'#SBATCH --export=ALL,QUEUEID={queueid}\n')
        f.write(' '.join(['srun'] + job_arguments))
        f.write('\n')

    return str(job_file_loc)


def queue_batch_file(job_file_loc):
    """Calls sbatch on the job file.

    :param job_file_loc: Location of the job file.

    :return: Returns tuple of stdout, stderr.
    """
    ret = subprocess.run(['sbatch', str(job_file_loc)], capture_output=True)
    print(f'STDOUT: {ret.stdout}')
    return ret.stdout, ret.stderr


if __name__ == '__main__':
    """Creates and runs a slurm batch file. 
    
    If queueid is passed in, it MUST be the first named argument. Ex:
    
    $ python batch_job.py --queueid=1234567 python fit_single.py cellid batch modelname
    """
    # parse arguments in order to collect all args into list, except for QUEUEID
    parser = argparse.ArgumentParser(description='Run jobs on exacloud!')
    parser.add_argument('--queueid', default=None, help='The tQueue QID.')

    parser.add_argument('arguments', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    job_file_loc = write_batch_file(args.arguments, args.queueid)
    queue_batch_file(job_file_loc)
