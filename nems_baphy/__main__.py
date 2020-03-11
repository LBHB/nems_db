from flask import Flask
from flask_restful import Api
from nems_baphy.api import BaphyInterface, GetRecording, UploadResults,UploadQueueLog
from nems_db.util import ensure_env_vars
from nems import get_settings, get_setting

req_env_vars = ['NEMS_BAPHY_API_HOST',
                'NEMS_BAPHY_API_PORT',
                'MYSQL_HOST',
                'MYSQL_USER',
                'MYSQL_PASS',
                'MYSQL_DB',
                'MYSQL_PORT',
                'NEMS_RECORDINGS_DIR',
                'NEMS_RESULTS_DIR']

# Load the credentials, throwing an error if any are missing
creds = get_settings()

for v in req_env_vars:
    if creds.get(v,None) is None:
        raise ValueError('Setting %s not specified in nems/configs/', v)

app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = creds['NEMS_RESULTS_DIR']

api.add_resource(BaphyInterface,
                 '/baphy/<string:batch>/<string:cellid>',
                 resource_class_kwargs={'host': creds['MYSQL_HOST'],
                                        'user': creds['MYSQL_USER'],
                                        'pass': creds['MYSQL_PASS'],
                                        'db':   creds['MYSQL_DB'],
                                        'port': creds['MYSQL_PORT']})
api.add_resource(GetRecording,
                 '/recordings/<string:batch>/<string:file>',
                 resource_class_kwargs={})

api.add_resource(UploadResults,
                 '/results/<string:batch>/<string:cellid>/<string:path>/<string:file>',
                 resource_class_kwargs={})

api.add_resource(UploadQueueLog,
                 '/queuelog/<string:queueid>',
                 resource_class_kwargs={})

app.run(port=int(creds['NEMS_BAPHY_API_PORT']),
        host=creds['NEMS_BAPHY_API_HOST'])
