"""View functions for "Fit Single Now" and "Enqueue Models" buttons.

These functions communicate with modelfit.py and are called by flask
when the browser navigates to their app.route URL signatures.
fit_single_model_view calls modelfit.fit_single_model
for the cell, batch and model selection passed via AJAX.
enqueue_models_view calls enqueue_models for the list of
cell selections, batch selection, and list of model selections
passed via AJAX.
Both functions return json-serialized summary information for the user
to indicate the success/failure and results of the model fit/queue.

See Also:
---------
. : modelfit.py

"""

import logging

import ast
from flask import jsonify, request
from flask_login import login_required

from nems_web.nems_analysis import app
from nems.db import enqueue_models
from nems.modelspec import _lookup_fn_at
from nems_web.account_management.views import get_current_user
from nems_lbhb.kamiak import kamiak_to_database
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

log = logging.getLogger(__name__)


@app.route('/enqueue_models')
@login_required
def enqueue_models_view():
    """Call modelfit.enqueue_models with user selections as args."""

    user = get_current_user()

    # Only pull the numerals from the batch string, leave off the description.
    bSelected = request.args.get('bSelected')[:3]
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    codeHash = request.args.get('codeHash')
    execPath = request.args.get('execPath')
    scriptPath = request.args.get('scriptPath')
    force_rerun = request.args.get('forceRerun', type=int)
    useKamiak = request.args.get('useKamiak', type=int)
    kamiakFunction = request.args.get('kamiakFunction')  # fn to generate scripts
    kamiakPath = request.args.get('kamiakPath')  # path to store output in
    loadKamiak = request.args.get('loadKamiak', type=int)  # check to load results
    kamiakResults = request.args.get('kamiakResults')  # path to results
    useGPU = request.args.get('useGPU', type=int)  # path to results
    useExacloud = request.args.get('useExacloud', type=int)
    exaOHSU = request.args.get('exaOHSU')
    exaExec = request.args.get('exaExec')
    exaScript = request.args.get('exaScript')
    exaLimit = request.args.get('exaLimit')
    exaExclude = request.args.get('exaExclude')
    exaHighMem = request.args.get('exaHighMem', type=int)

    if loadKamiak:
        kamiak_to_database(cSelected, bSelected, mSelected, kamiakResults,
                           execPath, scriptPath)
        return jsonify(data=True)

    elif useExacloud:
        log.info('Starting exacloud jobs!')
        enqueue_exacloud_models(cellist=cSelected, batch=bSelected, modellist=mSelected, user=user.username,
                                linux_user=exaOHSU, executable_path=exaExec, script_path=exaScript,
                                time_limit=exaLimit, useGPU=useGPU, high_mem=exaHighMem, exclude=exaExclude)
        return jsonify(data=True)

    elif useKamiak:
        # kamiakFunction should be a stringified pointer to a function
        # that takes a list of cellids, a batch, a list of modelnames,
        # and a directory where the output should be stored,
        # Ex: kamiakScript = 'nems_lbhb.utils.my_kamiak_function'
        try:
            kamiak_script = _lookup_fn_at(kamiakFunction, ignore_table=True)
            kamiak_script(cSelected, bSelected, mSelected, kamiakPath)
            return jsonify(data=True)
        except AttributeError:
            log.warning('kamiakFunction doesnt exist or is improperly defined')
            return jsonify(data=False)
    else:
        if not codeHash:
            codeHash = 'master'
        if not execPath:
            execPath = None
        if not scriptPath:
            scriptPath = None

        enqueue_models(
                cSelected, bSelected, mSelected,
                force_rerun=bool(force_rerun), user=user.username,
                codeHash=codeHash, executable_path=execPath,
                script_path=scriptPath, GPU_job=useGPU
                )

        return jsonify(data=True)


@app.route('/add_jerb_kv')
def add_jerb_kv():
    """Take key, list of values, and existing JSON object (query) from input
    then combine them into a new JSON object with the key and values added."""

    key = request.args.get('key')
    values = request.args.get('val')
    query = request.args.get('query')

    if not query:
        query = {}
    else:
        # Evaluate JSON-formatted string as a dict
        query = ast.literal_eval(query)

    if not values:
        val_list = []
    else:
        values = values.replace(' ', '')
        val_list = values.split(',')

    query[key] = val_list
    return jsonify(newQuery=query)
