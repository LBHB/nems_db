import argparse
import subprocess
import sys


def get_job_state(jobid):
    """Gets the job state for a given jobid.

    :param jobid: Job ID of a job on exacloud.

    :return: Returns the job state code.
    """
    # use squeue, limiting to just the jobid and state options, with no header
    ret = subprocess.run(['squeue', '-j', jobid, '-h', '-o', '%t'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = ret.stdout.decode()  # stdout is a bytes object
    return stdout


def is_job_alive(job_state):
    """Returns a bool of whether the job is alive or not.

    :param job_state: An exacloud job state code. See https://slurm.schedmd.com/sacct.html#lbAG

    :return: A bool of whether alive or not.
    """
    alive_codes = [
        'CD',  # completed
        'PD',  # pending
        'R',   # running
        'RQ',  # requeued
    ]

    dead_codes = ['BF', 'CA', 'DL', 'F', 'NF', 'OOM', 'PR', 'RS', 'RV', 'S', 'TO']

    if job_state in alive_codes:
        return True
    elif job_state in dead_codes:
        return False
    else:
        raise AttributeError(f'Job state "{job_state}" not recognized. Error occurred.')


if __name__ == '__main__':
    """Parses args to get job state."""
    parser = argparse.ArgumentParser(description='Get job state on exacloud.')
    parser.add_argument('jobid', help='The job ID of the desired job.')

    args = parser.parse_args()
    try:
        job_state = get_job_state(args.jobid)
        print(int(is_job_alive(job_state)))
        sys.exit(0)
    except:
        sys.exit(1)
