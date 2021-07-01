
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

from nems import db

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
    gSingleCell = LBHB_Tables()['gSingleCell']

    session.query(gSingleCell).\
        filter(gSingleCell.cellid == cellid).\
        update(datadict)

    session.commit()





def update_single_cell_data(cellid, **datadict):
    """
    :param cellid: eg, 'TAR010c-01-1' (should be an existing cellid)
    :param datadict: eg, {'layer': '2/3', 'phototag': 'E, mdlx', 'spikeshape': '0.122,0.328,0.309'}
          keys must match existing columns of gSingleCell (the table that stores site properties)
    :return:
    """
    session = db.Session()
    gSingleCell = LBHB_Tables()['gSingleCell']

    session.query(gSingleCell).\
        filter(gSingleCell.cellid == cellid).\
        update(datadict)

    session.commit()

def update_site_data(siteid, **datadict):
    """
    :param siteid: eg, 'TAR010c' (should be an existing siteid)
    :param datadict: eg, {'layer': '2/3', 'phototag': 'E, mdlx', 'spikeshape': '0.122,0.328,0.309'}
        keys must match existing columns of gSingleCell (the table that stores site properties)
    :return:
    """
    session = db.Session()
    gSingleCell = LBHB_Tables()['gSingleCell']

    session.query(gSingleCell).\
        filter(gSingleCell.cellid == cellid).\
        update(datadict)

    session.commit()

