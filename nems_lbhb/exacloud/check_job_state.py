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
    if not stdout:
        raise ValueError(f'Did not get anything from stdout for jobid "{jobid}".')
    return stdout.strip()


def is_job_alive(job_state):
    """Returns a bool of whether the job is alive or not.

    :param job_state: An exacloud job state code. See https://slurm.schedmd.com/sacct.html#lbAG

    :return: A bool of whether alive or not.
    """
    alive_codes = [
        'PENDING',
        'RUNNING',
        'REQUEUED',
    ]

    dead_codes = ['BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY',
                  'PREEMPTED', 'RESIZING', 'REVOKED', 'SUSPENDED', 'TIMEOUT']

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
