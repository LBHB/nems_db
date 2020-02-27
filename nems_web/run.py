"""Module for launching the nems_analysis flask app.

Launches the app and hosts the server at the domain/ip and port
specified. See flask documentation for additional app.run options.

Note:
-----
debug should NEVER be set to True when launching the server for a
'production' environment (i.e. when hosting for public use). Per the
flask documentation, this would allow users to run "arbitrary code" on
the server, potentially causing harm.

"""

import nems_web
from nems_web.nems_analysis import app #socketio
# from gevent.wsgi import WSGIServer

#from OpenSSL import SSL
#context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
# TODO: fix path spec when add in SSL
#context.use_privatekey_file(np.path + '/nems_config/host.key')
#context.use_certificate_file(np.path + '/nems_config/host.cert')

# TODO: figure out how to correctly configure the certificate for
#       desired host name, then turn this back on for neuralprediction.
#context = (
#        np.path + '/nems_config/host.cert', np.path + '/nems_config/host.key'
#        )

# TODO: app.run() not meant to be used in production, just for testing
#       according to Flask docs. Need to replace with better server.
#       flask.pocoo.org/docs/0.12/deploying/wsgi-standalone/
port = int(nems_web.get_setting('PORT'))
debug = (str(nems_web.get_setting('DEBUG')).lower() in ("yes", "true", "1", True))
print('Starting nems_web server on port {}, debug={}'.format(port,debug))
app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
#http_server = WSGIServer(('', 8000), app)
#http_server.serve_forever()
