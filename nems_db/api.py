import os
import io
import re
import json
import logging
log = logging.getLogger(__name__)

from flask import abort, Response, request
from flask_restful import Resource

from .query import grep_dirtree

# Define some regexes for sanitizing inputs
RECORDING_REGEX = re.compile(r"[\-_a-zA-Z0-9]+\.tar\.gz$")
CELLID_REGEX = re.compile(r"^[\-_a-zA-Z0-9]+$")
BATCH_REGEX = re.compile(r"^\d+$")
QUERY_REGEX = re.compile(r"[\-_a-zA-Z0-9\.]+$")


def as_path(recording, modelname, fitter, date):
    ''' Returns a relative path. '''
    if not recording and modelname and fitter and date:
        raise ValueError('Not all necessary fields defined!')
    # TODO: Use an URL lib for creating valid URLs instead of this:
    url = recording + '/' + modelname + '/' + fitter + '/' + date + '/'
    return url


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


class UploadResultsInterface(Resource):
    '''
    An interface for uploading any kind of file to a filesystem
    hierarchy stored on disk (or perhaps in Amazon S3).
    TODO: Require credentials for PUT!
    '''
    def __init__(self, **kwargs):
        self.upload_dir = kwargs['upload_dir']

    def put(self, objpath):
        # TODO: Ensure filepath is not insane
        if objpath[-5:] == '.json':
#            j = request.json
#            if not j:
#                print(str(request))
#                abort(400, "Object was not a json?!")
            # TODO: Verify it is a modelspec and a JSON
            bytesobj = io.BytesIO(request.data)
        elif objpath[-7:] == 'log.txt':
            # TODO: Verify it is a log file (plain text)
            bytesobj = io.BytesIO(request.data)
        elif objpath[-4:] == '.png':
            # TODO: Verify it is a PNG file
            bytesobj = io.BytesIO(request.data)
        else:
            abort(400, "Objpath not allowed.")

        filepath = os.path.join(self.upload_dir, objpath)

        # Create any necessary directories
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        if os.path.exists(filepath):
            abort(409, 'File already exists; will not overwrite.')

        # If here, it must be a new file and thus safe to write to disk
        with open(filepath, 'wb') as f:
            f.write(bytesobj.read())

        return Response(None, status=200)

    def post(self, rec):
        abort(400, 'Not implemented. Use PUT!')

    def delete(self, rec):
        abort(400, 'Not implemented. Deleting should never be needed!')


class UploadRecordingInterface(Resource):
    '''
    An interface for uploading NEMS-compatable .tar.gz recordings only.
    TODO: Require credentials for PUT!
    '''
    def __init__(self, **kwargs):
        self.upload_dir = kwargs['upload_dir']

    def get(self, recording_filepath):
        abort(400, 'Not implemented; nginx serves out GET requests faster.')

    def put(self, recording_filepath):
        if not recording_filepath[-7:] == '.tar.gz':
            abort(400, 'File must end with .tar.gz')
        # Create subdirectory if needed
        filepath = os.path.join(self.upload_dir, recording_filepath)
        if os.path.exists(filepath):
            abort(409, 'File exists; not going to overwrite.')
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        # TODO: Check that request has correct mime type
        d = io.BytesIO(request.data)
        with open(filepath, 'wb') as f:
            f.write(d.read())
        return Response(status=200)

    def post(self, recording_filepath):
        abort(400, 'Not implemented; use PUT instead')

    def delete(self, recording_filepath):
        abort(400, 'Not implemented; http DELETE is a little dangerous.')


class QueryInterface(Resource):
    '''
    An interface for retrieving lists of matching results, which may
    be subsequently retrieved through ResultInterface methods.
    '''
    def __init__(self, **kwargs):
        # TODO: Connect to MySQL using kwargs that pass connection info
        # This is not the right way to do it, but works until db.py is refactored:
        # self.session = Session()
        self.search_dir = kwargs['search_dir']
        self.results_uri = kwargs['results_uri']

    def get(self, **kwargs):
        '''
        Returns a JSON of URIs that matched the kwargs.
        '''
        query_options = {'include': '*.json',
                         'contains': ''}

        opts = request.args

        if opts.get('only'):
            filt = opts.get('only')
            if filt == 'modelspecs':
                query_options['include'] = 'modelspec*.json'
            elif filt == 'xfspecs':
                query_options['include'] = 'xfspec.json'
            else:
                abort(400, 'only must be "modelspecs" or "xfspecs"')

        if opts.get('contains'):
            cont = opts.get('contains')
            if QUERY_REGEX.match(cont):
                query_options['contains'] = cont
            else:
                m = 'Query must contain only characters [a-zA-Z0-9\-\_\.]'
                abort(400, m)

        results = grep_dirtree(self.search_dir,
                               contains=query_options['contains'],
                               include=query_options['include'])

        # Make those results into URIs again
        uris = [self.results_uri + r for r in results]
        js = json.dumps(uris)

        # TODO: A better way would be to use Results and MySQL
        # filters = []
        # for key, val in kwargs.items():
        #     col = getattr(Results, key)
        #     if isinstance(val, str):
        #         filters.append(col.ilike(val))
        #
        # objs = self.session.query(Results).filter(and_(*filters)).all()
        #
        # results = [(r.recording, r.modelanme, r.fitter, r.date) for r in objs]

        return Response(js, status=200, mimetype='application/json')
