#!/usr/bin/env python3

# This script runs nems_main.fit_single_model from the command line

import nems_db.xform_wrappers as nw
import nems0.utils
import sys
import os

import logging
log = logging.getLogger(__name__)

try:
    import nems_db.db as nd
    db_exists = True
except Exception as e:
    # If there's an error import nems0.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems0.db, can't update tQueue")
    print(e)
    db_exists = False

if __name__ == '__main__':

    # leftovers from some industry standard way of parsing inputs

    #parser = argparse.ArgumentParser(description='Generetes the topic vector and block of an author')
    #parser.add_argument('action', metavar='ACTION', type=str, nargs=1, help='action')
    #parser.add_argument('updatecount', metavar='COUNT', type=int, nargs=1, help='pubid count')
    #parser.add_argument('offset', metavar='OFFSET', type=int, nargs=1, help='pubid offset')
    #args = parser.parse_args()
    #action=parser.action[0]
    #updatecount=parser.updatecount[0]
    #offset=parser.offset[0]
    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems0.utils.progress_fun = nd.update_job_tick

    else:
        queueid = 0

    if queueid:
        print("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)

    if len(sys.argv)<4:
        print('syntax: nems_fit_single cellid batch modelname')
        exit(-1)

    cellid=sys.argv[1]
    batch=sys.argv[2]
    modelname=sys.argv[3]

    print("Running nems_queue_demo with parameters ({0},{1},{2})".format(cellid,batch,modelname))

    print("Insert your code here")
    # Mark completed in the queue. Note that this should happen last thing!
    # Otherwise the job might still crash after being marked as complete.
    if queueid:
        nd.update_job_complete(queueid)


