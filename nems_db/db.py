import os
import datetime
import sys
import pandas as pd
import numpy as np
import logging
import nems_db.util
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
import pandas.io.sql as psql

log = logging.getLogger(__name__)

# TODO: instead of doing all this connection HERE (which occurs during import),
#       we should instead have a "connect()" function that returns a session.
#       We should remove these global vars!
#       - started below with Session and _get_db_uri functions
#       - hard part is replacing calls in this module and all that depend
#       - on it.    --jacob 3/25/2018

creds = nems_db.util.ensure_env_vars(
        ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASS', 'MYSQL_DB', 'MYSQL_PORT']
        )
db_uri = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
        creds['MYSQL_USER'], creds['MYSQL_PASS'], creds['MYSQL_HOST'],
        creds['MYSQL_PORT'], creds['MYSQL_DB']
        )

# sets how often sql alchemy attempts to re-establish connection engine
# TODO: query db for time-out variable; set this to some fraction of that
POOL_RECYCLE = 7200

# Create a database connection engine
engine = create_engine(db_uri, pool_recycle=POOL_RECYCLE)

# TODO: How to handle moving these inside a function?
#       Could return them all in a dict each. Then any module that needs
#       Tables can just import 'get_db_tables' function and use ala
#       tables = get_db_tables()
#       NarfResults = tables['NarfResults']
# Create base class to mirror existing database schema
Base = automap_base()
Base.prepare(engine, reflect=True)

NarfUsers = Base.classes.NarfUsers
NarfAnalysis = Base.classes.NarfAnalysis
NarfBatches = Base.classes.NarfBatches
NarfResults = Base.classes.NarfResults
tQueue = Base.classes.tQueue
tComputer = Base.classes.tComputer
sCellFile = Base.classes.sCellFile
sBatch = Base.classes.sBatch
gCellMaster = Base.classes.gCellMaster

# import this when another module needs to use the database connection.
# used like a class - ex: 'session = Session()'
Session = sessionmaker(bind=engine)


# TODO: Come up with a better naming scheme for these? Goes against pep8
#       but sort of makes sense with the sqlalchemy theme
#def Session():
#    uri = _get_db_uri()
#    #engine = create_engine(uri, pool_recyle=POOL_RECYCLE)
#    engine = create_engine(uri)
#    return sessionmaker(bind=engine)


def Engine():
    uri = _get_db_uri()
    return create_engine(uri, pool_recyle=POOL_RECYCLE)


def Tables():
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


def _get_db_uri():
    creds = nems_db.util.ensure_env_vars(
            ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASS',
             'MYSQL_DB', 'MYSQL_PORT']
            )
    db_uri = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
            creds['MYSQL_USER'], creds['MYSQL_PASS'], creds['MYSQL_HOST'],
            creds['MYSQL_PORT'], creds['MYSQL_DB']
            )
    return db_uri


def enqueue_models(celllist, batch, modellist, force_rerun=False,
                   user=None, codeHash="master", jerbQuery='', ):
    """Call enqueue_single_model for every combination of cellid and modelname
    contained in the user's selections.

    Arguments:
    ----------
    celllist : list
        List of cellid selections made by user.
    batch : string
        batch number selected by user.
    modellist : list
        List of modelname selections made by user.
    force_rerun : boolean
        If true, models will be fit even if a result already exists.
        If false, models with existing results will be skipped.
    user : TODO
    codeHash : string
        Git hash string identifying a commit for the specific version of the
        code repository that should be used to run the model fit.
        Can also accept the name of a branch.
    jerbQuery : dict
        Dict that will be used by 'jerb find' to locate matching jerbs

    Returns:
    --------
    pass_fail : list
        List of strings indicating success or failure for each job that
        was supposed to be queued.

    See Also:
    ---------
    . : enqueue_single_model
    Narf_Analysis : enqueue_models_callback

    """
    # Not yet ready for testing - still need to coordinate the supporting
    # functions with the model queuer.
    session = Session()

    pass_fail = []
    for model in modellist:
        for cell in celllist:
            queueid, message = _enqueue_single_model(
                cell, batch, model, force_rerun, user,
                session, codeHash, jerbQuery,
            )
            if queueid:
                pass_fail.append(
                    '\n queueid: {0},'
                    '\n message: {1}'
                    .format(queueid, message)
                )
            else:
                pass_fail.append(
                    '\nFailure: {0}, {1}, {2}'
                    .format(cell, batch, model)
                )

    # Can return pass_fail instead if prefer to do something with it in views
    log.info('\n'.join(pass_fail))

    session.close()
    return


def _enqueue_single_model(
        cellid, batch, modelname, force_rerun, user,
        session, codeHash, jerbQuery
):
    """Adds a particular model to the queue to be fitted.

    Returns:
    --------
    queueid : int
        id (primary key) that was assigned to the new tQueue entry, or -1.
    message : str
        description of the action taken, to be reported to the console by
        the calling enqueue_models function.

    See Also:
    ---------
    Narf_Analysis : enqueue_single_model

    """

    # TODO: anything else needed here? this is syntax for nems_fit_single
    #       command prompt wrapper in main nems folder.
    commandPrompt = (
	" /home/nems/anaconda3/bin/python"
        " /home/nems/nems_db/nems_fit_single.py {0} {1} {2}"
        .format(cellid, batch, modelname)
    )

    note = "%s/%s/%s" % (cellid, batch, modelname)

    result = (
        session.query(NarfResults)
        .filter(NarfResults.cellid == cellid)
        .filter(NarfResults.batch == batch)
        .filter(NarfResults.modelname == modelname)
        .first()
    )
    if result and not force_rerun:
        log.info(
            "Entry in NarfResults already exists for: %s, skipping.\n" %
            note)
        session.close()
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
        message = "Incomplete entry for: %s already exists, skipping.\n" % note
        job = qdata
    elif qdata and (int(qdata.complete) == 2):
        # TODO:
        # dead queue entry for note exists, resetting
        # update complete and progress status each to 0
        # what does this do? doesn't look like the sql is sent right away,
        # instead gets assigned to [res,r]
        message = "Dead queue entry for: %s already exists, resetting.\n" % note
        qdata.complete = 0
        qdata.progress = 0
        job = qdata
        job.codeHash = codeHash
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


def update_job_complete(queueid):
    # mark job complete
    # svd old-fashioned way of doing
    #sql="UPDATE tQueue SET complete=1 WHERE id={}".format(queueid)
    #result = conn.execute(sql)
    # conn.close()
    conn = engine.connect()
    # tick off progress, job is live
    sql = "UPDATE tQueue SET complete=1 WHERE id={}".format(queueid)
    r = conn.execute(sql)
    conn.close()
    return r
    """
    ession = Session()
    # also filter based on note? - should only be one result to match either
    # filter, but double checks to make sure there's no conflict
    #note = "{0}/{1}/{2}".format(cellid, batch, modelname)
    #.filter(tQueue.note == note)
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


def update_job_start(queueid):
    conn = engine.connect()
    # mark job as active and progress set to 1
    sql = ("UPDATE tQueue SET complete=-1,progress=1 WHERE id={}"
           .format(queueid))
    r = conn.execute(sql)
    conn.close()
    return r


def update_job_tick(queueid=0):
    path = os.path.dirname(nems_config.defaults.__file__)
    i = path.find('nems/nems_config')
    qsetload_path = (path[:i + 5] + 'misc/cluster/qsetload')
    r=os.system(qsetload_path)
    if r:
        log.warning('Error executing qsetload')

    if queueid:
        conn = engine.connect()
        # tick off progress, job is live
        sql = "UPDATE tQueue SET progress=progress+1 WHERE id={}".format(queueid)
        r = conn.execute(sql)
        conn.close()


    return r


def save_results(stack, preview_file, queueid=None):
    session = Session()

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

"""
Start new nems functions here
"""


def update_results_table(modelspec, preview=None, username="svd", labgroup="lbhb"):
    session = Session()

    cellid = modelspec[0]['meta']['cellid']
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
        r.public=1
        if not labgroup == 'SPECIAL_NONE_FLAG':
            try:
                if not labgroup in r.labgroup:
                    r.labgroup += ', %s' % labgroup
            except TypeError:
                # if r.labgroup is none, can't check if user.labgroup is in it
                r.labgroup = labgroup
        fetch_meta_data(modelspec, r, attrs)
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
        fetch_meta_data(modelspec, r, attrs)

    session.commit()
    results_id = r.id
    session.close()

    return results_id


def fetch_meta_data(modelspec, r, attrs):
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
        v=_fetch_attr_value(modelspec, a, default)
        setattr(r, a, v)
        log.debug("modelspec: meta {0}={1}".format(a,v))


def _fetch_attr_value(modelspec, k, default=0.0):
    """Return the value of key 'k' of modelspec[0]['meta'], or default."""

    # if modelspec[0]['meta'][k] is a string, return it.
    # if it's an ndarray or anything else with indices, get the first index;
    # otherwise, just get the value. Then convert to scalar if np data type.
    # or if key doesn't exist at all, return the default value.
    if k in modelspec[0]['meta']:
        if modelspec[0]['meta'][k]:
            if not isinstance(modelspec[0]['meta'][k], str):
                try:
                    v = modelspec[0]['meta'][k][0]
                except BaseException:
                    v = modelspec[0]['meta'][k]
                finally:
                    try:
                        v = np.asscalar(v)
                    except BaseException:
                        pass
            else:
                v = modelspec[0]['meta'][k]
        else:
            v = default
    else:
        v = default

    return v

def get_batch(name=None, batchid=None):
    # eg, sql="SELECT * from NarfBatches WHERE batch=301"
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
    params = ()
    sql = "SELECT DISTINCT cellid,batch FROM NarfData WHERE 1"
    if not batch is None:
        sql += " AND batch=%s"
        params = params+(batch,)

    if not cellid is None:
       sql += " AND cellid like %s"
       params = params+(cellid+"%",)

    if not rawid is None:
        sql+= " AND rawid = %s"
        params=params+(rawid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d

def get_batch_cell_data(batch=None, cellid=None, rawid=None, label=None
                        ):
    # eg, sql="SELECT * from NarfData WHERE batch=301 and cellid="
    params = ()
    sql = "SELECT * FROM NarfData WHERE 1"
    if not batch is None:
        sql += " AND batch=%s"
        params = params+(batch,)

    if not cellid is None:
       sql += " AND cellid like %s"
       params = params+(cellid+"%",)

    if not rawid is None:
       sql += " AND rawid=%s"
       params = params+(rawid,)

    if not label is None:
       sql += " AND label like %s"
       params = params+(label,)


    d = pd.read_sql(sql=sql, con=engine, params=params)
    d.set_index(['cellid', 'groupid', 'label', 'rawid'], inplace=True)
    d=d['filepath'].unstack('label')

    return d

def get_batches(name=None):
    # eg, sql="SELECT * from NarfBatches WHERE batch=301"
    params = ()
    sql = "SELECT *,id as batch FROM sBatch WHERE 1"
    if not name is None:
        sql += " AND name like %s"
        params = params+("%"+name+"%",)
    d = pd.read_sql(sql=sql, con=engine, params=params)

    return d


def get_cell_files(cellid=None, runclass=None):
    # eg, sql="SELECT * from sCellFile WHERE cellid like "TAR010c-30-1"
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

    sql = ("SELECT min_isolation FROM NarfBatches WHERE cellid = {0}{1}{2} and batch = {3}".format("'",cellid,"'",batch))

    d = pd.read_sql(sql=sql, con=engine)
    return d

def get_cellids(rawid=None):
   sql = ("SELECT distinct(cellid) FROM sCellFile WHERE 1")

   if rawid is not None:
       sql+=" AND rawid = {0} order by cellid".format(rawid)
   else:
       sys.exit('Must give rawid')

   cellids = pd.read_sql(sql=sql,con=engine)['cellid']

   return cellids



def list_batches(name=None):
    d = get_batches(name)
    for x in range(0,len(d)):
        print("{} {}".format(d['batch'][x],d['name'][x]))

    return d


def get_data_parms(rawid=None, parmfile=None):
    # get parameters stored in gData associated with a rawfile

    if rawid is not None:
        sql = ("SELECT gData.* FROM gData INNER JOIN "
               "gDataRaw ON gData.rawid=gDataRaw.id WHERE gDataRaw.id={0}"
               .format(rawid))
        #sql="SELECT * FROM gData WHERE rawid={0}".format(rawid)
    elif parmfile is not None:
        sql = ("SELECT gData.* FROM gData INNER JOIN gDataRaw ON"
               "gData.rawid=gDataRaw.id WHERE gDataRaw.parmfile = '{0}'"
               .format(parmfile))
        log.info(sql)
    else:
        pass

    d = pd.read_sql(sql=sql, con=engine)

    return d


def batch_comp(batch, modelnames=[], cellids=['%']):

    modelnames = ['parm100pt_wcg02_fir15_pupgainctl_fit01_nested5',
                  'parm100pt_wcg02_fir15_pupgain_fit01_nested5',
                  'parm100pt_wcg02_fir15_stategain_fit01_nested5'
                  ]
    batch = 301
    cellids = ['%']

    session = Session()

    #     .filter(NarfResults.cellid.in_(cellids))
    results = psql.read_sql_query(
        session.query(NarfResults)
        .filter(NarfResults.batch == batch)
        .filter(NarfResults.modelname.in_(modelnames))
        .statement,
        session.bind
    )

    session.close()

    return results


def get_results_file(batch, modelnames=[], cellids=['%']):

    session = Session()

    #     .filter(NarfResults.cellid.in_(cellids))
    results = psql.read_sql_query(
        session.query(NarfResults)
        .filter(NarfResults.batch == batch)
        .filter(NarfResults.modelname.in_(modelnames))
        .filter(NarfResults.cellid.in_(cellids))
        .order_by(desc(NarfResults.lastmod))
        .statement,
        session.bind
    )

    session.close()

    if results.empty:
        raise ValueError("No result exists for:\n"
                         "batch: {0}\nmodelnames: {1}\ncellids: {2}\n"
                         .format(batch, modelnames, cellids))
    else:
        return results

def get_stable_batch_cell_data(batch=None, cellid=None):
    '''
    Used to return only the information for units that were stable across all
    rawids that match this batch and site (cellid)
    '''
    # eg, sql="SELECT * from NarfData WHERE batch=301 and cellid="
    params = ()
    sql = "SELECT * FROM NarfData WHERE 1"
    sql_rawids = "SELECT DISTINCT rawid FROM NarfData WHERE 1" # for rawids
    
    if not batch is None:
        sql += " AND batch=%s"
        sql_rawids += " AND batch=%s"
        params = params+(batch,)

    if not cellid is None:
       sql += " AND cellid like %s"
       sql_rawids += " AND cellid like %s"
       params = params+(cellid+"%",)

    
    rawids = pd.read_sql(sql=sql_rawids, con=engine, params=params)
    
    for i, rawid in enumerate(rawids['rawid']):
        if i == 0:
            sql += " AND rawid=%s"
            params = params+(rawid,)
        else:
            sql += " OR rawid=%s"
            params = params+(rawid,)

    d = pd.read_sql(sql=sql, con=engine, params=params)
    d.set_index(['cellid', 'groupid', 'label', 'rawid'], inplace=True)
    d=d['filepath'].unstack('label')

    return d