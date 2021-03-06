import os
import subprocess
import datetime
import sys
import logging
import itertools
import json

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
import pandas.io.sql as psql
import sqlite3

import nems_db.util
from nems_db import get_setting
from nems.utils import recording_filename_hash

log = logging.getLogger(__name__)
__ENGINE__ = None


###### Functions for establishing connectivity, starting a session, or
###### referencing a database table


def Engine():
    '''Returns a mysql engine object. Creates the engine if necessary.
    Otherwise returns the existing one.'''
    global __ENGINE__

    uri = _get_db_uri()
    if not __ENGINE__:
        __ENGINE__ = create_engine(uri, pool_recycle=7200)

    return __ENGINE__

    #except Exception as e:
    #    log.exception("Error when attempting to establish a database "
    #                  "connection.", e)
    #    raise(e)


def Session():
    '''Returns a mysql session object.'''
    engine = Engine()
    return sessionmaker(bind=engine)()


def Tables():
    '''Returns a dictionary containing Narf database table objects.'''
    engine = Engine()
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    tables = {
            'NarfUsers': Base.classes.NarfUsers,
            'NarfAnalysis': Base.classes.NarfAnalysis,
            'NarfBatches': Base.classes.NarfBatches,
            'NarfResults': Base.classes.NarfResults,
            'tQueue': Base.classes.tQueue,
            'tComputer': Base.classes.tComputer,
            'sCellFile': Base.classes.sCellFile,
            'sBatch': Base.classes.sBatch,
            'gCellMaster': Base.classes.gCellMaster,
            }
    return tables


def sqlite_test():

    creds = nems_db.util.ensure_env_vars(['NEMS_RECORDINGS_DIR'])
    dbfilepath = os.path.join(creds['NEMS_RECORDINGS_DIR'],'nems.db')

    conn = sqlite3.connect(dbfilepath)
    sql = "SELECT name FROM sqlite_master WHERE type='table' and name like 'Narf%'"
    r = conn.execute(sql)
    d = r.fetchone()

    if d is None:
        print("Tables missing, need to reinitialize database?")

        print("Creating db")
        scriptfilename = '/auto/users/svd/python/nems_db/nems_db/nems.db.sqlite.sql'
        cursor = conn.cursor()

        print("Reading Script...")
        scriptFile = open(scriptfilename, 'r')
        script = scriptFile.read()
        scriptFile.close()

        print("Running Script...")
        cursor.executescript(script)

        conn.commit()
        print("Changes successfully committed")

    conn.close()

    return 1



def _get_db_uri():
    '''Used by Engine() to establish a connection to the database.'''
    creds = nems_db.util.ensure_env_vars(
            ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASS',
             'MYSQL_DB', 'MYSQL_PORT', 'SQL_ENGINE',
             'NEMS_RECORDINGS_DIR']
            )

    if creds['SQL_ENGINE'] == 'mysql':
        db_uri = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                creds['MYSQL_USER'], creds['MYSQL_PASS'], creds['MYSQL_HOST'],
                creds['MYSQL_PORT'], creds['MYSQL_DB']
                )
    elif creds['SQL_ENGINE'] == 'sqlite':
        dbfilepath = os.path.join(creds['NEMS_RECORDINGS_DIR'],'nems.db')

        db_uri = 'sqlite:///' + dbfilepath

    return db_uri


def pd_query(sql=None, params=()):
    """
    execture an SQL command and return the results in a dataframe
    params:
        sql: string
            query to execute
            use fprintf formatting, eg
                sql = "SELECT * FROM table WHERE name=%s"
                params = ("Joe")

    TODO : sqlite compatibility?
    """

    if sql is None:
        raise ValueError ("parameter sql required")
    engine = Engine()
    # print(sql)
    # print(params)
    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


###### Functions that access / manipulate the job queue. #######

def enqueue_models(celllist, batch, modellist, force_rerun=False,
                   user="nems", codeHash="master", jerbQuery='',
                   executable_path=None, script_path=None,
                   priority=1):
    """Call enqueue_single_model for every combination of cellid and modelname
    contained in the user's selections.

    for each cellid in celllist and modelname in modellist, will create jobs
    that execute this command on a cluster machine:

    <executable_path> <script_path> <cellid> <batch> <modelname>

    e.g.:
    /home/nems/anaconda3/bin/python /home/nems/nems/fit_single_model.py \
       TAR010c-18-1 271 ozgf100ch18_dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1_basic

    Arguments:
    ----------
    celllist : list
        List of cellid selections made by user.
    batch : string
        batch number selected by user.
    modellist : list
        List of modelname selections made by user.
    force_rerun : boolean (default=False)
        If true, models will be fit even if a result already exists.
        If false, models with existing results will be skipped.
    user : string (default="nems")
        Typically the login name of the user who owns the job
    codeHash : string (default="master")
        Git hash string identifying a commit for the specific version of the
        code repository that should be used to run the model fit.
        Can also accept the name of a branch.
    jerbQuery : dict
        Dict that will be used by 'jerb find' to locate matching jerbs
    executable_path : string (defaults to nems' python3 executable)
        Path to executable python (or other command line program)
    script_path : string (defaults to nems' copy of nems/nems_fit_single.py)
        First parameter to pass to executable

    Returns:
    --------
    (queueids, messages) : list
        Returns a tuple of the tQueue id and results message for each
        job that was either updated in or added to the queue.

    See Also:
    ---------
    Narf_Analysis : enqueue_models_callback

    """

    # some parameter values, mostly for backwards compatibility with other
    # queueing approaches
    if user:
        user = user
    else:
        user = 'None'
    linux_user = 'nems'
    allowqueuemaster = 1
    waitid = 0
    parmstring = ''
    rundataid = 0

    engine = Engine()
    conn = engine.connect()

    if executable_path in [None, 'None', 'NONE', '']:
        executable_path = get_setting('DEFAULT_EXEC_PATH')
    if script_path in [None, 'None', 'NONE', '']:
        script_path = get_setting('DEFAULT_SCRIPT_PATH')

    # Convert to list of tuples b/c product object only useable once.
    combined = [(c, b, m) for c, b, m in
                itertools.product(celllist, [batch], modellist)]

    notes = ['%s/%s/%s' % (c, b, m) for c, b, m in combined]
    commandPrompts = ["%s %s %s %s %s" % (executable_path, script_path,
                                          c, b, m)
                      for c, b, m in combined]

    queueids = []
    messages = []
    for note, commandPrompt in zip(notes, commandPrompts):
        sql = 'SELECT * FROM tQueue WHERE note="' + note +'"'

        r = conn.execute(sql)
        if r.rowcount>0:
            # existing job, figure out what to do with it

            x=r.fetchone()
            queueid = x['id']
            complete = x['complete']
            if force_rerun:
                if complete == 1:
                    message = "Resetting existing queue entry for: %s\n" % note
                    sql = "UPDATE tQueue SET complete=0, killnow=0 WHERE id={}".format(queueid)
                    r = conn.execute(sql)

                elif complete == 2:
                    message = "Dead queue entry for: %s exists, resetting.\n" % note
                    sql = "UPDATE tQueue SET complete=0, killnow=0 WHERE id={}".format(queueid)
                    r = conn.execute(sql)

                else:  # complete in [-1, 0] -- already running or queued
                    message = "Incomplete entry for: %s exists, skipping.\n" % note

            else:

                if complete == 1:
                    message = "Completed entry for: %s exists, skipping.\n"  % note
                elif complete == 2:
                    message = "Dead entry for: %s exists, skipping.\n"  % note
                else:  # complete in [-1, 0] -- already running or queued
                    message = "Incomplete entry for: %s exists, skipping.\n" % note

        else:
            # new job
            sql = "INSERT INTO tQueue (rundataid,progname,priority," +\
                   "parmstring,allowqueuemaster,user," +\
                   "linux_user,note,waitid,codehash,queuedate) VALUES"+\
                   " ({},'{}',{}," +\
                   "'{}',{},'{}'," +\
                   "'{}','{}',{},'{}',NOW())"
            sql = sql.format(rundataid, commandPrompt, priority, parmstring,
                  allowqueuemaster, user, linux_user, note, waitid, codeHash)
            r = conn.execute(sql)
            queueid = r.lastrowid
            message = "Added new entry for: %s.\n"  % note

        queueids.append(queueid)
        messages.append(message)

    conn.close()

    return zip(queueids, messages)


def enqueue_models_old(celllist, batch, modellist, force_rerun=False,
                   user="nems", codeHash="master", jerbQuery='',
                   executable_path=None, script_path=None):
    """Call enqueue_single_model for every combination of cellid and modelname
    contained in the user's selections.

    for each cellid in celllist and modelname in modellist, will create jobs
    that execute this command on a cluster machine:

    <executable_path> <script_path> <cellid> <batch> <modelname>

    e.g.:
    /home/nems/anaconda3/bin/python /home/nems/nems/fit_single_model.py \
       TAR010c-18-1 271 ozgf100ch18_dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1_basic

    Arguments:
    ----------
    celllist : list
        List of cellid selections made by user.
    batch : string
        batch number selected by user.
    modellist : list
        List of modelname selections made by user.
    force_rerun : boolean (default=False)
        If true, models will be fit even if a result already exists.
        If false, models with existing results will be skipped.
    user : string (default="nems")
        Typically the login name of the user who owns the job
    codeHash : string (default="master")
        Git hash string identifying a commit for the specific version of the
        code repository that should be used to run the model fit.
        Can also accept the name of a branch.
    jerbQuery : dict
        Dict that will be used by 'jerb find' to locate matching jerbs
    executable_path : string (defaults to nems' python3 executable)
        Path to executable python (or other command line program)
    script_path : string (defaults to nems' copy of nems/nems_fit_single.py)
        First parameter to pass to executable

    Returns:
    --------
    (queueids, messages) : list
        Returns a tuple of the tQueue id and results message for each
        job that was either updated in or added to the queue.

    See Also:
    ---------
    Narf_Analysis : enqueue_models_callback

    """
    session = Session()
    db_tables = Tables()
    NarfResults = db_tables['NarfResults']
    tQueue = db_tables['tQueue']

    if executable_path in [None, 'None', 'NONE', '']:
        executable_path = get_setting('DEFAULT_EXEC_PATH')
    if script_path in [None, 'None', 'NONE', '']:
        script_path = get_setting('DEFAULT_SCRIPT_PATH')

    # Convert to list of tuples b/c product object only useable once.
    combined = [(c, b, m) for c, b, m in
                itertools.product(celllist, [batch], modellist)]

    if not force_rerun:
        existing_results = psql.read_sql_query(
                session.query(NarfResults.cellid, NarfResults.modelname,
                              NarfResults.batch)
                .filter(NarfResults.cellid.in_(celllist))
                .filter(NarfResults.batch == batch)
                .filter(NarfResults.modelname.in_(modellist))
                .statement,
                session.bind
                )
        removals = [r for r in existing_results.itertuples()]
        combined = [t for t in combined if t not in removals]

    notes = ['%s/%s/%s' % (c, b, m) for c, b, m in combined]
    commandPrompts = ["%s %s %s %s %s" % (executable_path, script_path,
                                          c, b, m)
                      for c, b, m in combined]

    queue_data = (
            session.query(tQueue)
            .filter(tQueue.note.in_(notes))
            .all()
            )

    queueids = []
    messages = []
    for note, commandPrompt in zip(notes, commandPrompts):
        # TODO: find a way to avoid this nested loop?
        #       but should be faster than doing nested queries at least.
        exists = False
        for i, qdata in enumerate(queue_data):
            if qdata.note == note:
                exists = True
                queue_data.pop(i)
                break

        if exists and (int(qdata.complete) <= 0):
            # incomplete entry for note already exists, skipping
            message = "Incomplete entry for: %s exists, skipping.\n" % note
            job = qdata
        elif exists and (int(qdata.complete) == 2):
            # dead queue entry for note exists, resetting
            # update complete and progress status each to 0
            message = "Dead queue entry for: %s exists, resetting.\n" % note
            qdata.complete = 0
            qdata.progress = 0
            job = qdata
            job.codeHash = codeHash
            # update command prompt incase a new executable or script path
            # has been provided.
            job.commandPrompt = commandPrompt
        elif exists and (int(qdata.complete) == 1):
            # resetting existing queue entry for note
            # update complete and progress status each to 0
            message = "Resetting existing queue entry for: %s\n" % note
            qdata.complete = 0
            qdata.progress = 0
            job = qdata
            # update codeHash on re-run
            job.codeHash = codeHash
            # update command prompt incase a new executable or script path
            # has been provided.
            job.commandPrompt = commandPrompt
        else:
            # result must not have existed, or status value was greater than 2
            # add new entry
            message = "Adding job to queue for: %s\n" % note
            job = _add_model_to_queue(
                commandPrompt, note, user, codeHash, jerbQuery
                )
            session.add(job)

        queueids.append(job.id)
        messages.append(message)
        session.commit()

    session.close()
    return zip(queueids, messages)


def enqueue_single_model(cellid, batch, modelname, user=None,
                         force_rerun=False, codeHash="master", jerbQuery='',
                         executable_path=None, script_path=None):

    zipped = enqueue_models([cellid], batch, [modelname], force_rerun,
                            user, codeHash, jerbQuery, executable_path,
                            script_path)

    queueid, message = next(zipped)
    return queueid, message


def enqueue_single_model_duplicate(
        cellid, batch, modelname, user=None,
        session=None,
        force_rerun=False, codeHash="master", jerbQuery='',
        executable_path=None, script_path=None):
    """
    Adds one model to the queue to be fitted for a single cell/batch

    Inputs:
    -------
    if executable_path is None:
        executable_path = "/home/nems/anaconda3/bin/python"
    if script_path is None:
        script_path = "/home/nems/nems_db/nems_fit_single.py"

    Returns:
    --------
    queueid : int
        id (primary key) that was assigned to the new tQueue entry, or -1.
    message : str
        description of the action taken, to be reported to the console by
        the calling enqueue_models function.
    """
    if session is None:
        session = Session()

    db_tables = Tables()
    NarfResults = db_tables['NarfResults']
    tQueue = db_tables['tQueue']

    if executable_path is None:
        executable_path = "/home/nems/anaconda3/bin/python"

    if script_path is None:
        script_path = "/home/nems/nems_db/nems_fit_single.py"

    commandPrompt = ("{0} {1} {2} {3} {4}"
                     .format(executable_path, script_path,
                             cellid, batch, modelname)
                     )

    note = "%s/%s/%s" % (cellid, batch, modelname)

    result = (session.query(NarfResults)
              .filter(NarfResults.cellid == cellid)
              .filter(NarfResults.batch == batch)
              .filter(NarfResults.modelname == modelname)
              .first()
              )
    if result and not force_rerun:
        log.info("Entry in NarfResults already exists for: %s, skipping.\n",
                 note)
        return -1, 'skip'

    # query tQueue to check if entry with same cell/batch/model already exists
    qdata = (
        session.query(tQueue)
        .filter(tQueue.note == note)
        .first()
    )

    job = None
    message = None

    if qdata and (int(qdata.complete) <= 0):
        # TODO:
        # incomplete entry for note already exists, skipping
        # update entry with same note? what does this accomplish?
        # moves it back into queue maybe?
        message = "Incomplete entry for: %s exists, skipping.\n" % note
        job = qdata
    elif qdata and (int(qdata.complete) == 2):
        # TODO:
        # dead queue entry for note exists, resetting
        # update complete and progress status each to 0
        # what does this do? doesn't look like the sql is sent right away,
        # instead gets assigned to [res,r]
        message = "Dead queue entry for: %s exists, resetting.\n" % note
        qdata.complete = 0
        qdata.progress = 0
        job = qdata
        job.codeHash = codeHash
        # update command prompt incase a new executable or script path
        # has been provided.
        job.commandPrompt = commandPrompt
    elif qdata and (int(qdata.complete) == 1):
        # TODO:
        # resetting existing queue entry for note
        # update complete and progress status each to 0
        # same as above, what does this do?
        message = "Resetting existing queue entry for: %s\n" % note
        qdata.complete = 0
        qdata.progress = 0
        job = qdata
        # update codeHash on re-run
        job.codeHash = codeHash
        # update command prompt incase a new executable or script path
        # has been provided.
        job.commandPrompt = commandPrompt
    else:
        # result must not have existed, or status value was greater than 2
        # add new entry
        message = "Adding job to queue for: %s\n" % note
        job = _add_model_to_queue(
            commandPrompt, note, user, codeHash, jerbQuery
            )
        session.add(job)

    session.commit()
    queueid = job.id

    return queueid, message


def _add_model_to_queue(commandPrompt, note, user, codeHash, jerbQuery,
                        priority=1, rundataid=0):
    """
    Returns:
    --------
    job : tQueue object instance
        tQueue object with variables assigned inside function based on
        arguments.

    See Also:
    ---------
    Narf_Analysis: dbaddqueuemaster

    """

    # TODO: why is narf version checking for list vs string on prompt and note?
    #       won't they always be a string passed from enqueue function?
    #       or want to be able to add multiple jobs manually from command line?
    #       will need to rewrite with for loop to to add this functionality in
    #       the future if needed.

    tQueue = Tables()['tQueue']
    job = tQueue()

    if user:
        user = user
    else:
        user = 'None'
    linux_user = 'nems'
    allowqueuemaster = 1
    waitid = 0
    dt = str(datetime.datetime.now().replace(microsecond=0))

    job.rundataid = rundataid
    job.progname = commandPrompt
    job.priority = priority
    job.parmstring = ''
    job.queuedate = dt
    job.allowqueuemaster = allowqueuemaster
    job.user = user
    job.linux_user = linux_user
    job.note = note
    job.waitid = waitid
    job.codehash = codeHash

    return job


def update_job_complete(queueid=None):
    """
    mark job queueid complete in tQueue
    svd old-fashioned way of doing it
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    engine = Engine()
    conn = engine.connect()
    sql = "UPDATE tQueue SET complete=1 WHERE id={}".format(queueid)
    r = conn.execute(sql)
    conn.close()

    return r
    """
    # fancy sqlalchemy method?
    session = Session()
    qdata = (
            session.query(tQueue)
            .filter(tQueue.id == queueid)
            .first()
            )
    if not qdata:
        # Something went wrong - either no matching id, no matching note,
        # or mismatch between id and note
        log.info("Invalid query result when checking for queueid & note match")
        log.info("/n for queueid: %s"%queueid)
    else:
        qdata.complete = 1
        session.commit()

    session.close()
    """


def update_job_start(queueid=None):
    """
    in tQueue, mark job as active and progress set to 1
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    engine = Engine()
    conn = engine.connect()
    sql = ("UPDATE tQueue SET complete=-1,progress=1 WHERE id={}"
           .format(queueid))
    r = conn.execute(sql)
    conn.close()
    return r


def update_job_tick(queueid=None):
    """
    update current machine's load in the cluster db and tick off a step
    of progress in the fit in tQueue
    """
    if queueid is None:
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
        else:
            log.warning("queueid not specified or found in os.environ")
            return 0

    path = nems_db.util.__file__
    i = path.find('nems_db/util')
    qsetload_path = (path[:i] + 'bin/qsetload')
    result = subprocess.run(qsetload_path, stdout=subprocess.PIPE)
    r = result.returncode

    if r:
        log.warning('Error executing qsetload')
        log.warning(result.stdout.decode('utf-8'))

    engine = Engine()
    conn = engine.connect()
    # tick off progress, job is live
    sql = ("UPDATE tQueue SET progress=progress+1 WHERE id={}"
           .format(queueid))
    r = conn.execute(sql)
    conn.close()

    return r


#### Results / performance logging

def save_results(stack, preview_file, queueid=None):
    """
    save performance data from modelspec to NarfResults
    pull some information out of the queue table if queueid provided
    """

    session = Session()
    db_tables = Tables()
    tQueue = db_tables['tQueue']
    NarfUsers = db_tables['NarfUsers']
    # Can't retrieve user info without queueid, so if none was passed
    # use the default blank user info
    if queueid:
        job = (
            session.query(tQueue)
            .filter(tQueue.id == queueid)
            .first()
        )
        username = job.user
        narf_user = (
            session.query(NarfUsers)
            .filter(NarfUsers.username == username)
            .first()
        )
        labgroup = narf_user.labgroup
    else:
        username = ''
        labgroup = 'SPECIAL_NONE_FLAG'

    results_id = update_results_table(stack, preview_file, username, labgroup)

    session.close()

    return results_id


def update_results_table(modelspec, preview=None,
                         username="svd", labgroup="lbhb"):
    db_tables = Tables()
    NarfResults = db_tables['NarfResults']
    session = Session()
    cellids = modelspec[0]['meta'].get('cellids',
                       [modelspec[0]['meta']['cellid']])

    for cellid in cellids:
        batch = modelspec[0]['meta']['batch']
        modelname = modelspec[0]['meta']['modelname']

        r = (
            session.query(NarfResults)
            .filter(NarfResults.cellid == cellid)
            .filter(NarfResults.batch == batch)
            .filter(NarfResults.modelname == modelname)
            .first()
        )
        collist = ['%s' % (s) for s in NarfResults.__table__.columns]
        attrs = [s.replace('NarfResults.', '') for s in collist]
        removals = [
            'id', 'lastmod'
        ]
        for col in removals:
            attrs.remove(col)

        if not r:
            r = NarfResults()
            if preview:
                r.figurefile = preview
            r.username = username
            r.public = 1
            if not labgroup == 'SPECIAL_NONE_FLAG':
                try:
                    if not labgroup in r.labgroup:
                        r.labgroup += ', %s' % labgroup
                except TypeError:
                    # if r.labgroup is none, can't check if user.labgroup is in it
                    r.labgroup = labgroup
            fetch_meta_data(modelspec, r, attrs, cellid)
            session.add(r)
        else:
            if preview:
                r.figurefile = preview
            # TODO: This overrides any existing username or labgroup assignment.
            #       Is this the desired behavior?
            r.username = username
            r.public=1
            if not labgroup == 'SPECIAL_NONE_FLAG':
                try:
                    if not labgroup in r.labgroup:
                        r.labgroup += ', %s' % labgroup
                except TypeError:
                    # if r.labgroup is none, can't check if labgroup is in it
                    r.labgroup = labgroup
            fetch_meta_data(modelspec, r, attrs, cellid)
        r.cellid = cellid
        session.commit()
        results_id = r.id

    session.close()

    return results_id


def fetch_meta_data(modelspec, r, attrs, cellid=None):
    """Assign attributes from model fitter object to NarfResults object.

    Arguments:
    ----------
    modelspec : nems modelspec with populated metadata dictionary
        Stack containing meta data, modules, module names et cetera
        (see nems_modules).
    r : sqlalchemy ORM object instance
        NarfResults object, either a blank one that was created before calling
        this function or one that was retrieved via a query to NarfResults.

    Returns:
    --------
    Nothing. Attributes of 'r' are modified in-place.

    """

    r.lastmod = datetime.datetime.now().replace(microsecond=0)

    for a in attrs:
        # list of non-numerical attributes, should be blank instead of 0.0
        if a in ['modelpath', 'modelfile', 'githash']:
            default = ''
        else:
            default = 0.0
        # TODO: hard coded fix for now to match up stack.meta names with
        # narfresults names.
        # Either need to maintain hardcoded list of fields instead of pulling
        # from NarfResults, or keep meta names in fitter matched to columns
        # some other way if naming rules change.
        #if 'fit' in a:
        #    k = a.replace('fit', 'est')
        #elif 'test' in a:
        #    k = a.replace('test', 'val')
        #else:
        #    k = a
        v=_fetch_attr_value(modelspec, a, default, cellid)
        setattr(r, a, v)
        log.debug("modelspec: meta {0}={1}".format(a,v))



def _fetch_attr_value(modelspec, k, default=0.0, cellid=None):
    """Return the value of key 'k' of modelspec[0]['meta'], or default."""

    # if modelspec[0]['meta'][k] is a string, return it.
    # if it's an ndarray or anything else with indices, get the first index;
    # otherwise, just get the value. Then convert to scalar if np data type.
    # or if key doesn't exist at all, return the default value.
    if k in modelspec[0]['meta']:
        v = modelspec[0]['meta'][k]
        if not isinstance(v, str):
            try:
                if cellid is not None and type(v==list):
                    cellids = modelspec[0]['meta']['cellids']
                    i = [index for index, value in enumerate(cellids) if value == cellid]
                    v = modelspec[0]['meta'][k][i[0]]
                else:
                    v = modelspec[0]['meta'][k][0]
            except BaseException:
                v = modelspec[0]['meta'][k]
            finally:
                try:
                    v = np.asscalar(v)
                    if np.isnan(v):
                        log.warning("value for %s, converting to 0.0 to avoid errors when"
                                    " saving to mysql", k)
                        v = 0.0
                except BaseException:
                    pass
        else:
            v = modelspec[0]['meta'][k]
    else:
        v = default

    return v

def get_batch(name=None, batchid=None):
    # eg, sql="SELECT * from NarfBatches WHERE batch=301"
    engine = Engine()
    params = ()
    sql = "SELECT * FROM sBatch WHERE 1"
    if not batchid is None:
        sql += " AND id=%s"
        params = params+(batchid,)

    if not name is None:
       sql += " AND name like %s"
       params = params+("%"+name+"%",)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d

def get_batch_cells(batch=None, cellid=None, rawid=None):
    # eg, sql="SELECT * from NarfBatches WHERE batch=301"
    engine = Engine()
    params = ()
    sql = "SELECT DISTINCT cellid,batch FROM NarfData WHERE 1"
    if batch is not None:
        sql += " AND batch=%s"
        params = params+(batch,)

    if cellid is not None:
       sql += " AND cellid like %s"
       params = params+(cellid+"%",)

    if rawid is not None:
        sql+= " AND rawid = %s"
        params=params+(rawid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


def get_batch_cell_data(batch=None, cellid=None, rawid=None, label=None):

    engine = Engine()
    # eg, sql="SELECT * from NarfData WHERE batch=301 and cellid="
    params = ()
    sql = ("SELECT DISTINCT NarfData.*,sCellFile.goodtrials" +
           " FROM NarfData LEFT JOIN sCellFile " +
           " ON (NarfData.rawid=sCellFile.rawid " +
           " AND NarfData.cellid=sCellFile.cellid)" +
           " WHERE 1")
    if batch is not None:
        sql += " AND NarfData.batch=%s"
        params = params+(batch,)

    if cellid is not None:
        sql += " AND NarfData.cellid like %s"
        params = params+(cellid+"%",)

    if rawid is not None:
        sql += " AND NarfData.rawid IN %s"
        rawid = tuple([str(i) for i in list(rawid)])
        params = params+(rawid,)

    if label is not None:
        sql += " AND NarfData.label like %s"
        params = params + (label,)
    sql += " ORDER BY NarfData.filepath"
    print(sql)
    d = pd.read_sql(sql=sql, con=engine, params=params)
    if label == 'parm':
        d['parm'] = d['filepath']
    else:
        d.set_index(['cellid', 'groupid', 'label', 'rawid', 'goodtrials'], inplace=True)
        d = d['filepath'].unstack('label')

    return d


def get_batches(name=None):
    # eg, sql="SELECT * from NarfBatches WHERE batch=301"
    engine = Engine()
    params = ()
    sql = "SELECT *,id as batch FROM sBatch WHERE 1"
    if name is not None:
        sql += " AND name like %s"
        params = params+("%"+name+"%",)
    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


def get_cell_files(cellid=None, runclass=None):
    # eg, sql="SELECT * from sCellFile WHERE cellid like "TAR010c-30-1"
    engine = Engine()
    params = ()
    sql = ("SELECT sCellFile.*,gRunClass.name, gSingleRaw.isolation FROM sCellFile INNER JOIN "
           "gRunClass on sCellFile.runclassid=gRunClass.id "
           " INNER JOIN "
           "gSingleRaw on sCellFile.rawid=gSingleRaw.rawid and sCellFile.cellid=gSingleRaw.cellid WHERE 1")
    if cellid is not None:
        sql += " AND sCellFile.cellid like %s"
        params = params+("%"+cellid+"%",)
    if runclass is not None:
        sql += " AND gRunClass.name like %s"
        params = params+("%"+runclass+"%",)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


# temporary function while we migrate databases
# (don't have access to gRunClass right now, so need to use rawid)
def get_cell_files2(cellid=None, runclass=None, rawid=None):
    engine = Engine()
    params = ()
    sql = ("SELECT sCellFile.* FROM sCellFile WHERE 1")

    if not cellid is None:
        sql += " AND sCellFile.cellid like %s"
        params = params+("%"+cellid+"%",)
    if not runclass is None:
        sql += " AND gRunClass.name like %s"
        params = params+("%"+runclass+"%",)
    if not rawid is None:
        sql+=" AND sCellFile.rawid = %s"
        params = params+(rawid,)


    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


def get_isolation(cellid=None, batch=None):
    engine = Engine()
    sql = ("SELECT min_isolation FROM NarfBatches WHERE cellid = {0}{1}{2} and batch = {3}".format("'",cellid,"'",batch))

    d = pd.read_sql(sql=sql, con=engine)
    return d


def get_cellids(rawid=None):
    engine = Engine()
    sql = ("SELECT distinct(cellid) FROM sCellFile WHERE 1")

    if rawid is not None:
        sql+=" AND rawid = {0} order by cellid".format(rawid)
    else:
        sys.exit('Must give rawid')

    cellids = pd.read_sql(sql=sql,con=engine)['cellid']

    return cellids


def list_batches(name=None):

    d = get_batches(name)

    for x in range(0, len(d)):
        print("{} {}".format(d['batch'][x], d['name'][x]))

    return d


def get_data_parms(rawid=None, parmfile=None):
    # get parameters stored in gData associated with a rawfile
    engine = Engine()
    if rawid is not None:
        sql = ("SELECT gData.* FROM gData INNER JOIN "
               "gDataRaw ON gData.rawid=gDataRaw.id WHERE gDataRaw.id={0}"
               .format(rawid))
        # sql="SELECT * FROM gData WHERE rawid={0}".format(rawid)

    elif parmfile is not None:
        sql = ("SELECT gData.* FROM gData INNER JOIN gDataRaw ON"
               "gData.rawid=gDataRaw.id WHERE gDataRaw.parmfile = '{0}'"
               .format(parmfile))
        log.info(sql)
    else:
        pass

    d = pd.read_sql(sql=sql, con=engine)

    return d


def batch_comp(batch=301, modelnames=None, cellids=['%'], stat='r_test'):
    NarfResults = Tables()['NarfResults']
    if modelnames is None:
        modelnames = ['parm100pt_wcg02_fir15_pupgainctl_fit01_nested5',
                      'parm100pt_wcg02_fir15_pupgain_fit01_nested5',
                      'parm100pt_wcg02_fir15_stategain_fit01_nested5'
                      ]

    session = Session()
    results=None
    for mn in modelnames:
        #     .filter(NarfResults.cellid.in_(cellids))
        tr = psql.read_sql_query(
                session.query(NarfResults)
                .filter(NarfResults.batch == batch)
                .filter(NarfResults.modelname == mn)
                .statement,
                session.bind
                )
        tc=tr[['cellid',stat]]
        tc=tc.set_index('cellid')
        tc.columns=[mn]
        if results is None:
            results=tc
        else:
            results=results.join(tc)

    session.close()

    return results


def get_results_file(batch, modelnames=None, cellids=None):
    NarfResults = Tables()['NarfResults']
    session = Session()
    query = (
        session.query(NarfResults)
        .filter(NarfResults.batch == batch)
        .order_by(desc(NarfResults.lastmod))
        )

    if modelnames is not None:
        if not isinstance(modelnames, list):
            raise ValueError("Modelnames should be specified as a list, "
                             "got %s", str(type(modelnames)))
        query = query.filter(NarfResults.modelname.in_(modelnames))

    if cellids is not None:
        if not isinstance(cellids, list):
            raise ValueError("Cellids should be specified as a list, "
                             "got %s", str(type(cellids)))
        query = query.filter(NarfResults.cellid.in_(cellids))

    results = psql.read_sql_query(query.statement, session.bind)
    session.close()

    if results.empty:
        raise ValueError("No result exists for:\n"
                         "batch: {0}\nmodelnames: {1}\ncellids: {2}\n"
                         .format(batch, modelnames, cellids))
    else:
        return results


def get_stable_batch_cells(batch=None, cellid=None, rawid=None,
                             label ='parm'):
    '''
    Used to return only the information for units that were stable across all
    rawids that match this batch and site/cellid.
    '''
    if (batch is None) | (cellid is None):
        raise ValueError

    # eg, sql="SELECT * from NarfData WHERE batch=301 and cellid="
    engine = Engine()
    params = ()
    sql = "SELECT cellid FROM NarfData WHERE 1"

    if type(cellid) is list:
        sql_rawids = "SELECT rawid FROM NarfData WHERE 1"  # for rawids
    else:
        sql_rawids = "SELECT DISTINCT rawid FROM NarfData WHERE 1"  # for rawids


    if batch is not None:
        sql += " AND batch=%s"
        sql_rawids += " AND batch=%s"
        params = params+(batch,)

    if label is not None:
       sql += " AND label = %s"
       sql_rawids += " AND label = %s"
       params = params+(label,)

    if cellid is not None:
        if type(cellid) is list:
            cellid = tuple(cellid)
            sql += " AND cellid IN %s"
            sql_rawids += " AND cellid IN %s"
            params = params+(cellid,)
        else:
            sql += " AND cellid like %s"
            sql_rawids += " AND cellid like %s"
            params = params+(cellid+"%",)

    if rawid is not None:
        sql += " AND rawid IN %s"
        if type(rawid) is not list:
            rawid = [rawid]
        rawid=tuple([str(i) for i in rawid])
        params = params+(rawid,)
        print(params)
        d = pd.read_sql(sql=sql, con=engine, params=params)

        cellids = np.sort(d['cellid'].value_counts()[d['cellid'].value_counts()==len(rawid)].index.values)

        # Make sure cellids is a list
        if type(cellids) is np.ndarray and type(cellids[0]) is np.ndarray:
            cellids = list(cellids[0])
        elif type(cellids) is np.ndarray:
            cellids = list(cellids)
        else:
            pass

        log.debug('Returning cellids: {0}, stable across rawids: {1}'.format(cellids, rawid))

        return cellids, list(rawid)

    else:
        rawid = pd.read_sql(sql=sql_rawids, con=engine, params=params)
        if type(cellid) is tuple:
            rawid = rawid['rawid'].value_counts()[rawid['rawid'].value_counts()==len(cellid)]
            rawid = rawid.index.tolist()
        else:
            rawid = rawid['rawid'].tolist()

        if type(cellid) is tuple:
            siteid = cellid[0].split('-')[0]
        else:
            siteid = cellid.split('-')[0]

        cellids, rawid = get_stable_batch_cells(batch, siteid, rawid)

        return cellids, rawid


def get_wft(cellid=None):
    engine = Engine()
    params = ()
    sql = "SELECT meta_data FROM gSingleCell WHERE 1"

    sql += " and cellid =%s"
    params = params+(cellid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)
    if d.values[0][0] is None:
        print('no meta_data information for {0}'.format(cellid))
        return -1

    wft = json.loads(d.values[0][0])
    ## 1 is fast spiking, 0 is regular spiking
    celltype = int(wft['wft_celltype'])

    return celltype


def get_gSingleCell_meta(cellid=None, fields=None):

    engine = Engine()
    params = ()
    sql = "SELECT meta_data FROM gSingleCell WHERE 1"

    sql += " and cellid =%s"
    params = params+(cellid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)
    if d.values[0][0] is None:
        print('no meta_data information for {0}'.format(cellid))
        return -1
    else:
        dic = json.loads(d.values[0][0])
        if type(fields) is list:
            out = {}
            for f in fields:
                out[f] = dic[f]

        elif type(fields) is str:
            out = dic[fields]
        elif fields is None:
            out = {}
            fields = dic.keys()
            for f in fields:
                out[f] = dic[f]

        return out

def get_rawid(cellid, run_num):
    """
    Used to return the rawid corresponding to given run number. To be used if
    you have two files at a given site that belong to the same batch but were
    sorted separately and you only want to load cellids from one of the files.

    ex. usage in practice would be to pass a sys arg cellid followed by the
    run_num:

        cellid = 'TAR017b-04-1_04'

        This specifies cellid and run_num. So parse this string and pass as args
        to this function to return rawid
    """
    engine = Engine()
    params = ()
    sql = "SELECT rawid FROM sCellFile WHERE 1"

    if cellid is not None:
        sql += " AND cellid like %s"
        params = params+(cellid+"%",)

    if run_num is not None:
        sql += " AND respfile like %s"
        params = params+(cellid[:-5]+run_num+"%",)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return [d['rawid'].values[0]]


#### NarfData management

def save_recording_to_db(recfilepath, meta=None, user="nems", labgroup="",
                         public=True):
    """
    expects recfilepath == "/path/to/data/<exptname>_<hash>.tgz"

    """
    engine = Engine()
    conn = engine.connect()

    path, base = os.path.split(recfilepath)
    base = base.split("_")
    pre = base[0]
    hsh = base[1].split(".")[0]
    batch = int(meta.get("batch", 0))
    if batch > 0:
        path, batchstr = os.path.split(path)

    file_hash = recording_filename_hash(name=pre, meta=meta, uri_path=path,
                                        uncompressed=False)
    meta_string = json.dumps(meta, sort_keys=True)

    if file_hash != recfilepath:
        raise ValueError("meta does not produce hash matching recfilepath")

    sql = "INSERT INTO NarfData (batch,hash,meta,filepath,label," + \
          "username,labgroup,public) VALUES" + \
          " ({},'{}','{}','{}','{}','{}','{}',{})"
    sql = sql.format(batch, hsh, meta_string, recfilepath, "recording",
                     user, labgroup, int(public))
    r = conn.execute(sql)
    dataid = r.lastrowid
    log.info("Added new entry %d for: %s.", dataid, recfilepath)

    return dataid
