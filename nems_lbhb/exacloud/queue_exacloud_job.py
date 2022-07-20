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


def enqueue_exacloud_models(cellist, batch, modellist, user, linux_user, executable_path, script_path, priority=1,
                            time_limit=14, reserve_gb=0, useGPU=False, high_mem=False, exclude=None, force_rerun=False):
    """Enqueues models similarly to nems0.db.enqueue_models, except on the Exacloud cluster at ACC.

    :param celllist: List of cells to include in analysis.
    :param batch: Batch number cells originate from.
    :param modellist: List of models to run.
    :param user: Username of one starting the job.
    :param linux_user: OHSU username.
    :param executable_path: Executable used to run script.
    :param script_path: Script to run.
    :param time_limit: Max hours the job will run for. Jobs will terminated if not complete by the end of the time limit.
    :param reserve_gb: Max GB required for the job. Job will fail if memory use goes above this level.
    :param useGPU: Whether or not to be GPU job.
    :param high_mem: Whether or not GPU should be a higher memory one.
    :param exclude: List of nodes to exclude. Comma separated values, no spaces.
    """
    # if batch_path in [None, 'None', 'NONE', '']:
    #     batch_path = Path(r'/home/exacloud/lustre1/LBHB/code/nems_db/nems_lbhb/exacloud/batch_job.py')

    # extra parameters for future
    time_limit = f'--time_limit={time_limit}'
    use_gpu = '--use_gpu' if useGPU else ''
    reserve_gb = f'--reserve_gb={reserve_gb}' if reserve_gb else ''
    high_mem = '--high_mem' if high_mem else ''
    exclude = f'--exclude={exclude}' if exclude is not None else ''
    extra_options = ' '.join([time_limit, reserve_gb, use_gpu, high_mem, exclude])

    # Convert to list of tuples b/c product object only useable once.
    combined = list(itertools.product(cellist, [str(batch)], modellist))
    #log.info(combined)

    queue_items = []

    engine = db.Engine()
    conn = engine.connect()

    for cell, b, model in combined:
        add_msg_str=''
        progname = ' '.join([extra_options, executable_path, script_path, cell, b, model])
        if "*" in cell:
            progname = ' '.join([extra_options, executable_path, script_path, f"'{cell}'", b, model])
            add_msg_str = f", subbed '{cell}' for {cell} in tQueue progname."
        note = '/'.join([cell, b, model])

        sql = f"SELECT * FROM Results WHERE batch={b} and cellid='{cell}' and modelname='{model}'"
        rres = conn.execute(sql)
        
        if (rres.rowcount==0) | force_rerun:
            sql = 'SELECT * FROM tQueue WHERE allowqueuemaster=18 AND note="' + note +'"'
            r = conn.execute(sql)
            if r.rowcount>0:
                # existing job, figure out what to do with it

                x=r.fetchone()
                queueid = x['id']
                complete = x['complete']
                if force_rerun:
                    if complete == 1:
                        message = "Resetting existing queue entry for: %s\n" % note
                        sql = f"UPDATE tQueue SET complete=0, killnow=0, progname='{progname}', user='{user}', priority={priority} WHERE id={queueid}"
                        r = conn.execute(sql)

                    elif complete == 2:
                        message = "Dead queue entry for: %s exists, resetting." % note
                        sql = f"UPDATE tQueue SET complete=0, killnow=0, progname='{progname}', user='{user}', priority={priority} WHERE id={queueid}"
                        r = conn.execute(sql)

                    else:  # complete in [-1, 0] -- already running or queued
                        message = "Incomplete entry for: %s exists, skipping." % note

                else:

                    if complete == 1:
                        message = "Completed entry for: %s exists, skipping."  % note
                    elif complete == 2:
                        message = "Dead entry for: %s exists, skipping."  % note
                    else:  # complete in [-1, 0] -- already running or queued
                        message = "Incomplete entry for: %s exists, skipping." % note
        
                log.info(message)
            else:
                # new job
                queue_item = tQueue(
                    progname=progname,
                    machinename='exacloud',
                    queuedate=datetime.datetime.now(),
                    user=user,
                    linux_user=linux_user,
                    note=note,
                    priority=priority,
                    allowqueuemaster=18,  # exacloud specific code
                )

                queue_items.append(queue_item)
                message = f"Added exacloud job: {note}{add_msg_str}"
        else:
            message = "Model fit for: %s exists, skipping."  % note

        log.info(message)

    with db_session() as session:
        session.add_all(queue_items)


def enqueue_single_exacloud_model(cell, batch, model, user, linux_user, executable_path,
                                  script_path, time_limit=10, useGPU=False, high_mem=False, force_rerun=False):
    """Enqueues a single model. See `enqueue_exacloud_models` for parameter information."""
    enqueue_exacloud_models(cellist=[cell], batch=batch, modellist=[model], user=user, linux_user=linux_user,
                            executable_path=executable_path, script_path=script_path, time_limit=time_limit,
                            useGPU=useGPU, high_mem=high_mem, force_rerun=force_rerun)
