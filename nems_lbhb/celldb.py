import datetime
import json
import numpy as np

from sqlalchemy import create_engine, desc, exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

from nems0 import db
from nems0 import get_setting

import pandas as pd



def LBHB_Tables():

    engine = db.Engine()
    Base = automap_base()
    Base.prepare(engine, reflect=True)
    tables = {
            'gCellMaster': Base.classes.gCellMaster,
            'gSingleCell': Base.classes.gSingleCell,
            'sCellFile': Base.classes.sCellFile,
            }
    return tables

def get_single_cell_data(cellid):
    """
    :param cellid: single cellid or siteid
    :return:
    """
    sql = f"SELECT * FROM gSingleCell WHERE cellid like '{cellid}%%'"
    d = db.pd_query(sql)

    return d

def get_site_data(siteid):
    """
    :param siteid:
    :return:
    """
    sql = f"SELECT * FROM gCellMaster WHERE siteid like '{siteid}'"
    d = db.pd_query(sql)

    return d

def update_table(tablename, key, value, **datadict):
    """
    :param cellid: eg, 'TAR010c-01-1' (should be an existing cellid)
    :param datadict: eg, {'layer': '2/3', 'phototag': 'E, mdlx', 'spikeshape': '0.122,0.328,0.309'}
          keys must match existing columns of gSingleCell (the table that stores site properties)
    :return:
    """
    session = db.Session()
    t = LBHB_Tables()[tablename]

    session.query(t).filter(getattr(t, key) == value).update(datadict)

    session.commit()


def update_single_cell_data(cellid, **datadict):
    """
    :param cellid: eg, 'TAR010c-01-1' (should be an existing cellid)
    :param datadict: eg, {'layer': '2/3', 'phototag': 'E, mdlx', 'spikeshape': '0.122,0.328,0.309'}
        keys must match existing columns of gSingleCell (the table that stores cell properties)
    :return:
    """
    update_table('gSingleCell', 'cellid', cellid, **datadict)

    return

def update_site_data(siteid, **datadict):
    """
    :param siteid: eg, 'TAR010c' (should be an existing siteid)
    :param datadict: eg, {'channeldepths': '1050,1000,1000,...0', 'layerboundaries': '200,500,800'}
        keys must match existing columns of gCellMaster (the table that stores site properties)
    :return:
    """
    update_table('gCellMaster', 'siteid', siteid, **datadict)



class celldb():
    ENGINE = None
    user = None
    animal = None
    training = 1
    MYSQL_USER = None
    MYSQL_PASS = None
    MYSQL_HOST = None
    MYSQL_PORT = 3306
    MYSQL_DB = None
    TESTMODE = False

    def __init__(self):
        self.MYSQL_USER = get_setting('MYSQL_USER')
        self.MYSQL_PASS = get_setting('MYSQL_PASS')
        self.MYSQL_HOST = get_setting('MYSQL_HOST')
        self.MYSQL_PORT = get_setting('MYSQL_PORT')
        self.MYSQL_DB = get_setting('MYSQL_DB')

    def Engine(self):
        '''Returns a mysql engine object. Creates the engine if necessary.
        Otherwise returns the existing one.'''
        global __ENGINE__

        uri = self.get_db_uri()
        if self.ENGINE is None:
            self.ENGINE = create_engine(uri, pool_recycle=1600)

        return self.ENGINE

    def Session(self):
        '''Returns a mysql session object.'''
        engine = self.Engine()
        return sessionmaker(bind=engine)()

    def Tables(self):
        '''Returns a dictionary containing Narf database table objects.'''
        engine = self.Engine()
        Base = automap_base()
        Base.prepare(engine, reflect=True)

        tables = {
            'gAnimal': Base.classes.gAnimal,
            'gPenetration': Base.classes.gPenetration,
            'gCellMaster': Base.classes.gCellMaster,
            'gDataRaw': Base.classes.gDataRaw,
            'gData': Base.classes.gData,
            'gSingleCell': Base.classes.gSingleCell,
            'gSingleRaw': Base.classes.gSingleRaw,
        }

        return tables

    def get_db_uri(self):
        '''Used by Engine() to establish a connection to the database.'''

        db_uri = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
            self.MYSQL_USER, self.MYSQL_PASS, self.MYSQL_HOST,
            self.MYSQL_PORT, self.MYSQL_DB
        )
        return db_uri

    def pd_query(self, sql=None, params=None):
        """
        execute an SQL command and return the results in a dataframe
        """

        if sql is None:
            raise ValueError("parameter sql required")
        engine = self.Engine()

        d = None
        try:
            d = pd.read_sql_query(sql=sql, con=engine, params=params)
        except exc.SQLAlchemyError as OpErr:
            if OpErr._message().count('Lost connection to MySQL server during query') > 0:
                log.warning('Lost connection to MySQL server during query, trying again.')
                d = pd.read_sql_query(sql=sql, con=engine, params=params)
        if d is None:
            raise ValueError(f"pd_query error on sql={sql} params={params}")
        return d

    def sqlexec(self, sql):
        engine = self.Engine()
        conn = engine.connect()
        conn.execute(sql)

        return True

    def sqlinsert(self, table, d):
        """
        TODO: what if d is too long?
        """
        engine = self.Engine()
        conn = engine.connect()

        # creating column list for insertion
        cols = "`,`".join([str(i) for i in d.columns.tolist()])

        # Insert DataFrame in a bulk insert
        sql = f"REPLACE INTO `{table}` (`" + cols + "`) VALUES (" + "%s," * (len(d.columns.tolist()) - 1) + "%s)"
        rows = []
        for i, row in d.iterrows():
            rows.append(tuple(row.values))
        if self.TESTMODE:
            print(sql)
            return sql
        else:
            res = conn.execute(sql, rows)
            return res.lastrowid

    def sqlupdate(self, table, id, d=None, idfield='id'):
        """
        TODO: what if d is too long?
        """
        engine = self.Engine()
        conn = engine.connect()

        fv = []
        for k, v in d.items():
            if type(v) is str:
                fv.append(f"{k}='{v}'")
            else:
                fv.append(f"{k}={v}")

        if type(id) is str:
            id = "'" + id + "'"
        sql = f"UPDATE {table} SET {','.join(fv)} WHERE {idfield}={id}"

        if self.TESTMODE:
            print(sql)
        else:
            conn.execute(sql)

    def get_users(self):
        d = self.pd_query("SELECT id,userid,email FROM gUserPrefs WHERE active")
        return d

    def get_animals(self, name=None, lab="lbhb", species="ferret", active=True):
        if active:
            active_string = " AND onschedule<2"
        else:
            active_string = ""
        if name is None:
            d = self.pd_query(f"SELECT * FROM gAnimal WHERE lab=%s {active_string} AND species=%s", (lab, species))
        else:
            d = self.pd_query("SELECT * FROM gAnimal WHERE animal=%s", (name,))

        return d

    def get_penetration(self, penname):
        sql = f"SELECT * FROM gPenetration WHERE penname='{penname}'"
        print(sql)
        return self.pd_query(sql)

    def get_rawdata(self, siteid=None, rawid=None):
        if rawid is None:
            wherestr = f"WHERE gDataRaw.cellid='{siteid}'"
        else:
            wherestr = f"WHERE gDataRaw.id={rawid}"
        sql = "SELECT gDataRaw.*, gCellMaster.penid" + \
              " FROM gDataRaw INNER JOIN gCellMaster" + \
              f" ON gDataRaw.masterid=gCellMaster.id {wherestr} ORDER BY id"

        return self.pd_query(sql)

    def get_current_penname(self, animal=None, training=None,
                            force_next=False, create_if_missing=True):
        """
        get penname for animal/training condition. if latest pendate is before today,
        increment penname
        :param animal:
        :param training:
        :return: penname: str
        """
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training

        d_animal = self.get_animals(name=animal)
        if len(d_animal) > 0:
            cellprefix = d_animal.loc[0, 'cellprefix']
        else:
            raise ValueError(f"animal {animal} not found in celldb.gAnimal")

        sql = 'SELECT max(id) as maxid FROM gPenetration WHERE training=%s AND animal=%s'
        lastpendata = self.pd_query(sql, (training, animal))
        today = datetime.date.today().strftime("%Y-%m-%d")
        need_to_create = False

        if len(lastpendata) == 0:
            print('No penetrations exist for this animal. Guessing info from scratch.')
            pennum = 1
        else:
            sql = f"SELECT * FROM gPenetration WHERE id={lastpendata.loc[0, 'maxid']}"
            lastpendata = self.pd_query(sql)
            penname = lastpendata.loc[0, 'penname']
            pendate = lastpendata.loc[0, 'pendate']

            if (pendate == today) & ~force_next:
                pennum = int(penname[3:6])
            else:
                print(f"Last pendate: {pendate} incrementing from {penname}")
                pennum = int(penname[3:6]) + 1
                need_to_create = True
        if training:
            penname = f"{cellprefix}{pennum:03d}T"
        else:
            penname = f"{cellprefix}{pennum:03d}"

        if need_to_create and create_if_missing:
            self.create_penetration(animal=animal, training=training, penname=penname)

        return penname

    def get_current_site(self, siteid=None, animal=None, training=None, penname=None, force_next=False):
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training

        if siteid is not None:
            sql = f"SELECT * FROM gCellMaster WHERE siteid='{siteid}' ORDER BY id"
        else:
            if (penname is None):
                penname = self.get_current_penname(animal, training)
            sql = f"SELECT * FROM gCellMaster WHERE penname='{penname}' ORDER BY id"
        sitedata = self.pd_query(sql)
        if len(sitedata) == 0:
            print(f"No current site for penetration {penname}?")
            return None
        else:
            return sitedata.iloc[-1]

    def create_penetration(self, user=None, animal=None, training=None,
                           penname=None, well=1,
                           NumberOfElectrodes=0, HWSetup=0):
        if user is None:
            user = self.user
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training
        if penname is None:
            penname = self.get_current_penname(animal, training=training, force_next=True)

        today = datetime.date.today().strftime("%Y-%m-%d")

        racknotes = ''
        speakernotes = 'Free-field Manger'
        probenotes = ''
        electrodenotes = ''
        ear = ''
        d = pd.DataFrame({
            'penname': penname,
            'animal': animal,
            'well': well,
            'pendate': today,
            'who': user,
            'fixtime': datetime.datetime.now().strftime("%H:%M"),
            'ear': ear,
            'rackid': HWSetup,
            'racknotes': racknotes,
            'speakernotes': speakernotes,
            'probenotes': probenotes,
            'electrodenotes': electrodenotes,
            'training': training,
            'numchans': NumberOfElectrodes,
            'addedby': user,
            'info': 'psilbhb.celldb'
        }, index=[0])

        self.sqlinsert('gPenetration', d)
        print(f"Created new penetration {penname}")
        self.create_site(penname=penname, animal=animal,
                         training=training, user=user)
        return penname

    def create_site(self, penname=None, siteid=None, animal=None,
                    training=None, user=None,
                    area="", comments=""):
        if animal is None:
            animal = self.animal
        if training is None:
            training = self.training

        if siteid is None:

            pendata = self.get_penetration(penname).loc[0]
            if user is None:
                user = pendata.addedby
            sitedata = self.get_current_site(penname=penname, force_next=True)
            if sitedata is not None:
                siteletter = sitedata.siteid[-1]
                newsiteletter = chr(ord(siteletter) + 1)
                siteid = sitedata.siteid[:-1] + newsiteletter
            else:
                siteid = penname + 'a'
        else:
            penname = siteid[:-1]
            pendata = self.get_penetration(penname).loc[0]

        d = pd.DataFrame({'siteid': siteid,
                          'cellid': siteid,
                          'penid': pendata.id,
                          'penname': penname,
                          'animal': pendata.animal,
                          'well': pendata.well,
                          'training': pendata.training,
                          'findtime': datetime.datetime.now().strftime("%H:%M"),
                          'area': area,
                          'comments': comments,
                          'addedby': user,
                          'info': 'psilbhb.celldb'}, index=[0])
        masterid = self.sqlinsert('gCellMaster', d)
        print(f'Added gCellMaster entry {siteid}')

        cellid = siteid + "-001-1"
        # sitedata = self.get_current_site(self, penname=penname)
        # masterid = sitedata['id']

        d = pd.DataFrame({'siteid': siteid,
                          'cellid': cellid,
                          'masterid': masterid,
                          'penid': pendata.id,
                          'channel': '001-',
                          'unit': 1,
                          'channum': 1,
                          'addedby': user,
                          'info': 'psilbhb.celldb'}, index=[0])
        self.sqlinsert('gSingleCell', d)
        print(f'Added gSingleCell entry {cellid}')

        return siteid

    def create_rawfile(self, siteid=None,
                       runclass=None, filenum=0, behavior="passive", timejuice=0,
                       pupil=False, psi_format=True,
                       ):
        if siteid is None:
            sitedata = self.get_current_site()
            siteid = sitedata.siteid
        else:
            sitedata = self.get_current_site(siteid=siteid)
        penname = sitedata['penname']
        if runclass is None:
            raise ValueError('Three-letter runclass must be specified for new_raw_file')
        if behavior is None:
            behavior = "passive"

        masterid = sitedata.id
        sql = f"SELECT * FROM gRunClass where name='{runclass}'"
        runclassdata = self.pd_query(sql)
        if len(runclassdata) == 1:
            runclassid = runclassdata.loc[0, 'id']
            stimclass = runclassdata.loc[0, 'stimclass']
            task = runclassdata.loc[0, 'task']
        else:
            raise ValueError(f"runclass {runclass} not found in gRunClass")
        year = datetime.datetime.now().strftime("%Y")
        month = datetime.datetime.now().strftime("%m")
        day = datetime.datetime.now().strftime("%d")

        # determine path + parmfilename
        rawdata = self.get_rawdata(siteid)
        if len(rawdata) > 0:
            prevparmfile = rawdata.iloc[-1]['parmfile']
            resppath = rawdata.iloc[-1]['resppath']
            if (filenum==0) and sitedata.training:
                filenum = int(prevparmfile.split(".")[0].split("_")[-1]) + 1
            elif filenum==0:
                filenum = int(prevparmfile[len(siteid):(len(siteid) + 2)]) + 1
        else:
            dataroot = '/auto/data/daq'
            if sitedata.training:
                resppath = f'{dataroot}/{sitedata.animal}/training{year}/'
            else:
                resppath = f'{dataroot}/{sitedata.animal}/{penname}/'
        if behavior == 'passive':
            bstr = 'p'
        else:
            bstr = 'a'
        if sitedata.training:
            filestr = f'{filenum:d}'
            parmbase = f"{sitedata.animal}_{year}_{month}_{day}_{runclass}_{filestr}"
        else:
            filestr = f'{filenum:02d}'
            parmbase = f"{siteid}{filestr}_{bstr}_{runclass}"

        pupilfile = ""
        if psi_format:
            parmfile = parmbase
            evpfilename = f"{resppath}{parmbase}/reward_contact_analog.zarr"
            rawpath = f'{resppath}{parmbase}/raw/'
            if pupil:
                pupilfile = f"{resppath}{parmbase}/recording.avi"

        else:
            parmfile = parmbase + ".m"
            if pupil:
                pupilfile = f"{resppath}{parmbase}.avi"

            evpfilename = f"{resppath}{parmbase}.evp"
            rawpath = f'{resppath}raw/{siteid}{filestr}/'

        d = pd.DataFrame({
            'cellid': siteid,
            'masterid': masterid,
            'runclass': runclass,
            'runclassid': runclassid,
            'training': sitedata.training,
            'resppath': resppath,
            'parmfile': parmfile,
            'eyewin': pupil,
            'eyecalfile': pupilfile,
            'respfileevp': evpfilename,
            'respfile': rawpath,
            'task': task,
            'stimclass': stimclass,
            'behavior': behavior,
            'timejuice': timejuice,
            'fixtime': datetime.datetime.now().strftime("%H:%M"),
            'time': datetime.datetime.now().strftime("%H:%M"),
            'addedby': sitedata.addedby,
            'info': 'psilbhb.celldb'}, index=[0])
        rawid = self.sqlinsert('gDataRaw', d)
        print(f'Added gDataRaw entry {parmbase}')
        rawdata = {'rawid': rawid, 'resppath': resppath, 'parmbase': parmbase,
                   'pupil_file': pupilfile, 'rawpath': rawpath}
        return rawdata

    def save_data(self, rawid, datadict, parmtype=0, keep_existing=False):

        if type(datadict) is not dict:
            raise ValueError(f"Parmeter datadict must be a dict")

        fn = datadict.keys()

        if keep_existing:
            namestr = "'" + "','".join(fn) + "'"

            sql = f'DELETE FROM gData WHERE rawid={rawid}' + \
                  f" AND name in ({namestr}) AND parmtype={parmtype}"
        else:
            sql = f'DELETE FROM gData WHERE rawid={rawid} AND parmtype={parmtype}'
        self.sqlexec(sql)

        rawdata = self.get_rawdata(rawid=rawid)
        if len(rawdata) == 0:
            raise ValueError(f"rawid={rawid} not in gDataRaw")

        masterid = rawdata.loc[0, 'masterid']
        siteid = rawdata.loc[0, 'cellid']
        penid = rawdata.loc[0, 'penid']
        d = pd.DataFrame()
        for i, (k, v) in enumerate(datadict.items()):
            jv = json.dumps(v)
            d.loc[i, 'name'] = k
            d.loc[i, 'svalue'] = jv
        d['siteid'] = siteid
        d['penid'] = masterid
        d['masterid'] = masterid
        d['rawid'] = rawid
        d['datatype'] = 1
        d['parmtype'] = parmtype
        d['addedby'] = c.user
        d['info'] = 'psilbhb.celldb'

        self.sqlinsert('gData', d)

        print(f'Saved {len(d)} data items for rawid {rawid}')

    def read_data(self, rawid):
        sql = f"SELECT * FROM gData WHERE rawid={rawid} ORDER BY parmtype,id"

        d = self.pd_query(sql)
        sidx = d['datatype'] == 1
        d['value'] = d['value'].astype(object)
        # d.loc[sidx, 'value'] = d.loc[sidx, 'svalue']

        try:
            d.loc[sidx, 'value'] = d.loc[sidx, 'svalue'].apply(json.loads)
        except:
            pass
        try:
            sidx = np.isnan(d['value'].astype(float))
            d.loc[sidx, 'value'] = d.loc[sidx, 'svalue']
        except:
            pass
        return d

def __main__():
    c = celldb()
    c.animal = "Test"
    c.training = True
    c.user = "david"
    c.TESTMODE = True
    penname = c.get_current_penname()
    sitedata = c.get_current_site()

    # resppath, parmbase, rawpath, rawid = c.create_rawfile(runclass="TSP", behavior="active", pupil=True)
    # resppath, parmbase, rawpath, rawid = c.create_rawfile(runclass="NON", behavior="passive", pupil=False)

    # c.sqlupdate('gDataRaw', rawid, {'trials': 60, 'corrtrials': 50})

    rawid = 146632
    datadict = {'Trial_Paradigm': 'GoNogo',
                'Trial_CatchFrac': 0.5,
                'Trial_Durations': [1.0, 2.0, 3.0],
                'ReferenceClass': 'NaturalSounds',
                'TargetClass': 'Tone',
                'Tar_Frequency': 1000}
    parmtype = 0
    # c.save_data(rawid, datadict, parmtype=parmtype)

    rawid = 146342
    rawid = 146632

    d = c.read_data(rawid)
    print(d[['name', 'value']])
