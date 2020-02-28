import datetime
import itertools
import logging
from contextlib import contextmanager

from nems import db

log = logging.getLogger(__name__)

tQueue = db.Tables()['tQueue']

@contextmanager
def db_session():
    """Context manager to handle database connections."""
    session = db.Session()

    try:
        yield session
        session.commit()
    except:
        session.rollback()
    finally:
        session.close()


def enqueue_exacloud_models(cellist, batch, modellist, user, linux_user, executable_path,
                            script_path, time_limit=10, useGPU=False):
    """Enqueues models similarly to nems.db.enqueue_models, except on the Exacloud cluster at ACC.

    :param celllist: List of cells to include in analysis.
    :param batch: Batch number cells originate from.
    :param modellist: List of models to run.
    :param user: Username of one starting the job.
    :param linux_user: OHSU username.
    :param executable_path: Executable used to run script.
    :param script_path: Script to run.
    :param time_limit: How long the job will run for. Jobs will terminated if not complete by the end of the time limit.
    :param useGPU: Whether or not to be GPU job. Currently unused.
    """
    # if batch_path in [None, 'None', 'NONE', '']:
    #     batch_path = Path(r'/home/exacloud/lustre1/LBHB/code/nems_db/nems_lbhb/exacloud/batch_job.py')

    # extra parameters for future
    time_limit = f'--time_limit={time_limit}'
    use_gpu = '--use_gpu' if useGPU else ''
    extra_options = ' '.join([time_limit, use_gpu])

    # Convert to list of tuples b/c product object only useable once.
    combined = list(itertools.product(cellist, [batch], modellist))
    log.info(combined)

    queue_items = []

    for cell, b, model in combined:

        progname = ' '.join([extra_options, executable_path, script_path, cell, b, model])
        note = '/'.join([cell, b, model])

        queue_item = tQueue(
            progname=progname,
            machinename='exacloud',
            queuedate=datetime.datetime.now(),
            user=user,
            linux_user=linux_user,
            note=note,
            allowqueuemaster=18,  # exacloud specific code
        )

        queue_items.append(queue_item)

    with db_session() as session:
        session.add_all(queue_items)


def enqueue_single_exacloud_model(cell, batch, model, user, linux_user, executable_path,
                                  script_path, time_limit=10, useGPU=False):
    """Enqueues a single model. See `enqueue_exacloud_models` for parameter information."""
    enqueue_exacloud_models(cellist=[cell], batch=batch, modellist=[model], user=user, linux_user=linux_user,
                            executable_path=executable_path, script_path=script_path, time_limit=time_limit,
                            useGPU=useGPU)
