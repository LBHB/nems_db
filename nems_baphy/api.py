import re
import os
import io
from pathlib import Path
import mimetypes

from flask import abort, Response, request
from flask_restful import Resource

import nems_lbhb.baphy as baphy
import nems

# Define some regexes for sanitizing inputs
RECORDING_REGEX = re.compile(r"[\-_a-zA-Z0-9]+\.tar\.gz$")
CELLID_REGEX = re.compile(r"^[\-_a-zA-Z0-9]+$")
BATCH_REGEX = re.compile(r"^\d+$")

nems_recordings_dir = nems.get_setting('NEMS_RECORDINGS_DIR')
nems_results_dir = nems.get_setting('NEMS_RESULTS_DIR')


def valid_recording_filename(recording_filename):
    ''' Input Sanitizer.  True iff the filename has a valid format. '''
    matches = RECORDING_REGEX.match(recording_filename)
    return matches


def valid_cellid(cellid):
    ''' Input Sanitizer.  True iff the cellid has a valid format. '''
    matches = CELLID_REGEX.match(cellid)
    return matches


def valid_batch(batch):
    ''' Input Sanitizer.  True iff the batch has a valid format. '''
    matches = BATCH_REGEX.match(batch)
    return matches


def ensure_valid_cellid(cellid):
    if not valid_cellid(cellid):
        abort(400, 'Invalid cellid:' + cellid)


def ensure_valid_batch(batch):
    if not valid_batch(batch):
        abort(400, 'Invalid batch:' + batch)


def ensure_valid_recording_filename(rec):
    if not valid_recording_filename(rec):
        abort(400, 'Invalid recording:' + rec)


def not_found():
    abort(404, "Resource not found. ")


class BaphyInterface(Resource):
    '''
    An interface to BAPHY that returns NEMS-compatable signal objects
    '''
    def __init__(self, **kwargs):
        # self.host = kwargs['host'],
        # self.port = kwargs['port'],
        # self.user = kwargs['user'],
        # self.pass = kwargs['pass'],
        # self.db = kwargs['db']
        # self.db = ... # TODO: Connect to database HERE
        pass

    def get(self, batch, cellid):
        '''
        Queries the MySQL database, finds the file, and returns
        the corresponding data in a NEMS-friendly Recording object.
        '''
        ensure_valid_batch(batch)
        ensure_valid_cellid(cellid)
        batch = int(batch)

        options = request.args.copy()

        # TODO: Sanitize arguments

        # TODO: Wrapping this in a try/catch is acceptable given how likely it is to fail,
        #       and then give clients some feedback about what went wrong:

        rec = baphy.baphy_load_recording(cellid, batch, options)

        if rec:
            targz = rec.as_targz()
            return Response(targz, status=200, mimetype='application/gzip')
        else:
            abort(400, 'load_recording_from_baphy returned None')

    def put(self, batch, cellid):
        abort(400, 'Not yet implemented')

    def delete(self, batch, cellid):
        abort(400, 'Not yet Implemented')

class GetRecording(Resource):
    '''
    An interface to BAPHY that returns NEMS-compatable signal objects
    '''
    def __init__(self, **kwargs):
        pass

    def get(self, batch, file):
        '''
        Queries the MySQL database, finds the file, and returns
        the corresponding data in a NEMS-friendly Recording object.
        '''
        filepath = os.path.join(nems_recordings_dir, batch, file)

        try:
            f = io.open(filepath, 'rb')
            return Response(f, status=200, mimetype='application/gzip')
        except:
            #print(f"looking for file: filepath")
            abort(400, f'file {filepath} not found')

    def put(self, batch, file):
        abort(400, 'Not yet implemented')

    def delete(self, batch, file):
        abort(400, 'Not yet Implemented')


class GetDaq(Resource):
    '''
    An interface to BAPHY that returns NEMS-compatable signal objects
    '''
    def __init__(self, **kwargs):
        pass

    def get(self, animal, site, file):
        '''
        Queries the MySQL database, finds the file, and returns
        the corresponding data in a NEMS-friendly Recording object.
        '''
        filepath = os.path.join(nems_recordings_dir, 'daq', animal, site, file)

        try:
            f = io.open(filepath, 'rb')
            return Response(f, status=200, mimetype='application/text')
        except:
            abort(400, f'file {filepath} not found')

    def put(self, batch, file):
        abort(400, 'Not yet implemented')

    def delete(self, batch, file):
        abort(400, 'Not yet Implemented')


class UploadResults(Resource):
    '''
    An interface for uploading and downloading NEMS results
    '''
    ALLOWED_EXTENSIONS = set(['tgz','json','png'])

    def __init__(self, **kwargs):
        pass

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get(self, batch, cellid, path, file):
        '''
        find the right path
        '''
        fullpath = os.path.join(nems_results_dir, batch, cellid, path, file)
        mimetype, encoding = mimetypes.guess_type(file)
        print(f"Guessing mimetype {mimetype}")
        try:
            f = io.open(fullpath, 'rb')
            return Response(f, status=200, mimetype=mimetype)
        except:
            abort(400, 'file not found')

    def put(self, batch, cellid, path, file):
        data = request.data
        print('data received: len: {}'.format(len(data)))
        fullpath = os.path.join(nems_results_dir, batch, cellid, path)
        filename = os.path.abspath(os.path.join(fullpath, file))
        print('save to : ' + filename)
        if not os.path.exists(fullpath):
           os.makedirs(fullpath, 0o777)
        os.chmod(fullpath, 0o777)
        f = os.open(filename, os.O_RDWR|os.O_CREAT)
        os.write(f, data)
        os.close(f)

    def delete(self, batch, file):
        abort(400, 'Not yet Implemented')


class UploadQueueLog(Resource):
    '''
    An interface to BAPHY that returns NEMS-compatable signal objects
    '''
    def __init__(self, **kwargs):
        pass

    def get(self, batch, cellid, path, file):
       abort(400, 'Not yet implemented')

    def put(self, queueid):
        data = request.data
        print('data received: len: {}'.format(len(data)))
        log_dir = Path('/auto/data/web/celldb/queue')
        rounded_queueid = int(queueid) // 1000 * 1000
        log_dir /= str(rounded_queueid)
        if not log_dir.exists():
            log_dir.mkdir(parents=True, mode=0o777, exist_ok=True)
        log_loc = log_dir / (str(queueid) + '.out')

        print('save to : ' + str(log_loc))
        f = os.open(log_loc, os.O_RDWR|os.O_CREAT)
        os.write(f, data)
        os.close(f)
        os.chmod(log_loc, 0o777)

    def delete(self, batch, file):
        abort(400, 'Not yet Implemented')
