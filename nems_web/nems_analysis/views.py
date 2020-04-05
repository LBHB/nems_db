"""View functions for the main page of the nems_analysis web interface.

UI Upate functions handle the initial state of the landing page as well as
refreshing analysis, batch, cell and model selectors when a user makes a
selection. The general procedure for each function is:
    -Get the user's selections from an AJAX call.
    -Query the database for a list of new entries for the dependent selector,
        based on the user's selections.
    -Return a JSON-serialized version of the list of new entries.

Analysis Editor functions populate the fields of the Analysis Editor modal,
add the data in the fields to the database, and/or delete an analysis from
the database.

Table Selection functions implement callbacks for the Preview button (so far),
as well as any other buttons whose behavior depends on selections within the
results table (as opposed to the selectors).

Miscellaneous functions handle other simple UI features that don't fit in
any other category (so far, just one function to serve error_log.txt).

"""

import logging
import copy
import datetime
import json
import itertools
from base64 import b64encode
from urllib.parse import urlparse
from collections import namedtuple

from flask import (
        render_template, jsonify, request,
        )
from flask_login import login_required
import pandas.io.sql as psql
import pandas as pd
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc, or_

from nems_web.nems_analysis import app, bokeh_version
from nems.db import Session, Tables
from nems_web.nems_analysis.ModelFinder import ModelFinder
from nems_db.plots import PLOT_TYPES
from nems_web.account_management.views import get_current_user
from nems_web.run_custom.script_utils import scan_for_scripts
from nems.uri import load_resource
from nems_web.utilities.enclosure import split_by_enclosure
from nems.utils import escaped_split

log = logging.getLogger(__name__)


# TODO: Move these options to configs
#       namedtuple is a temporary hack to force object-like attributes
ui_opt = namedtuple('ui_opt', 'cols rowlimit sort measurelist required_cols detailcols iso snr snri')
n_ui = ui_opt(
    cols=['r_test', 'r_fit', 'n_parms'],
    rowlimit=500,
    sort='cellid',
    # specifies which columns from narf results can be used to quantify
    # performance for plots
    measurelist=[
            'r_test', 'r_ceiling', 'r_fit', 'r_active', 'mse_test',
            'mse_fit', 'mi_test', 'mi_fit', 'nlogl_test',
            'nlogl_fit', 'cohere_test', 'cohere_fit',
            ],
    # any columns in this list will always show on results table
    required_cols=['cellid', 'modelname'],
    # specifies which columns to display in 'details' tab for analysis
    detailcols=['id', 'status', 'question', 'answer'],
    # default minimum values for filtering out cells before plotting
    iso=0,
    snr=0,
    snri=0,
    )

##################################################################
####################   UI UPDATE FUNCTIONS  ######################
##################################################################


@app.route('/')
def main_view():
    """Initialize the nems_analysis landing page.

    Queries the database to get lists of available analyses, batches,
    status filters, tag filters, and results columns.
    Specifies defaults for results columns, row limit and sort column.

    Returns:
    --------
    main.html : template
        The landing page template rendered with variables for analysislist,
        batchlist, collist, defaultcols, measurelist, defaultrowlimit,
        sortlist, defaultsort, statuslist, and taglist.

    """

    # TODO: figure out how to integrate sec_lvl/superuser mode
    #       maybe need to add sec_lvl column to analysis/batches/results?
    #       then can compare in query ex: if user.sec_lvl > analysis.sec_lvl
    user = get_current_user()
    session = Session()
    db_tables = Tables()
    Results = db_tables['Results']
    Analysis = db_tables['Analysis']
    Batches = db_tables['Batches']
    sBatch = db_tables['sBatch']

    # .all() returns a list of tuples, so it's necessary to pull the
    # name elements out into a list by themselves.
    analyses = (
            session.query(Analysis)
            .filter(or_(
                    int(user.sec_lvl) == 9,
                    Analysis.public == '1',
                    Analysis.labgroup.ilike('%{0}%'.format(user.labgroup)),
                    Analysis.username == user.username,
                    ))
            .order_by(asc(Analysis.id))
            .all()
            )
    analysislist = [
            a.name for a in analyses
            ]
    analysis_ids = [
            a.id for a in analyses
            ]

    batchids = [
            i[0] for i in
            session.query(Batches.batch)
            .distinct()
            #.filter(or_(
            #        int(user.sec_lvl) == 9,
            #        Batches.public == '1',
            #        Batches.labgroup.ilike('%{0}%'.format(user.labgroup)),
            #        Batches.username == user.username,
            #        ))
            .all()
            ]
    batchnames = []
    for i in batchids:
        name = (
                session.query(sBatch.name)
                .filter(sBatch.id == i)
                .first()
                )
        if not name:
            batchnames.append('')
        else:
            batchnames.append(name.name)
    batchlist = [
            (batch + ': ' + batchnames[i])
            for i, batch in enumerate(batchids)
            ]
    batchlist.sort()

    # Default settings for results display.
    # TODO: let user choose their defaults and save for later sessions
    # cols are in addition to cellid, modelname and batch,
    # which are set up to be required
    defaultcols = n_ui.cols
    defaultrowlimit = n_ui.rowlimit
    defaultsort = n_ui.sort
    measurelist = n_ui.measurelist
    statuslist = [
            i[0] for i in
            session.query(Analysis.status)
            .filter(Analysis.name.in_(analysislist))
            .distinct().all()
            ]

    # Separate tags into list of lists of strings.
    tags = [
            i[0].split(",") for i in
            session.query(Analysis.tags)
            .filter(Analysis.name.in_(analysislist))
            .distinct().all()
            ]
    # Flatten list of lists into a single list of all tag strings
    # and remove leading and trailing whitespace.
    taglistbldupspc = [i for sublist in tags for i in sublist]
    taglistbldup = [t.strip() for t in taglistbldupspc]
    # Reform the list with only unique tags
    taglistbl = list(set(taglistbldup))
    # Finally, remove any blank tags and sort the list.
    taglist = [t for t in taglistbl if t != '']
    taglist.sort()

    # Returns all columns in the format 'Results.columnName,'
    # then removes the leading 'Results.' from each string
    collist = ['%s'%(s) for s in Results.__table__.columns]
    collist = [s.replace('Results.', '') for s in collist]
    sortlist = copy.deepcopy(collist)
    # Remove cellid and modelname from options toggles- make them required.
    required_cols = n_ui.required_cols
    for col in required_cols:
        collist.remove(col)

    # imported at top from PlotGenerator
    plotTypeList = PLOT_TYPES
    # imported at top from nems_web.run_scrits.script_utils
    scriptList = scan_for_scripts()

    session.close()

    return render_template(
            'main.html', analysislist=analysislist, analysis_ids=analysis_ids,
            batchlist=batchlist, collist=collist, defaultcols=defaultcols,
            measurelist=measurelist, defaultrowlimit=defaultrowlimit,
            sortlist=sortlist, defaultsort=defaultsort, statuslist=statuslist,
            taglist=taglist, plotTypeList=plotTypeList, username=user.username,
            seclvl=int(user.sec_lvl), iso=n_ui.iso, snr=n_ui.snr,
            snri=n_ui.snri, scripts=scriptList, bokeh_version=bokeh_version
            )


@app.route('/update_batch')
def update_batch():
    """Update current batch selection after an analysis is selected."""

    session = Session()
    Analysis = Tables()['Analysis']
    blank = 0

    aSelected = request.args.get('aSelected', type=str)
    batch = (
            session.query(Analysis.batch)
            .filter(Analysis.name == aSelected)
            .first()
            )
    try:
        batch = batch.batch
    except Exception as e:
        log.info(e)
        batch = ''
        blank = 1

    session.close()

    return jsonify(batch=batch, blank=blank)


_previous_update_models = ('', '', '', [])
@app.route('/update_models')
def update_models():
    """Update the list of modelnames in the model selector after an
    analysis is selected.

    """

    session = Session()
    Analysis = Tables()['Analysis']

    aSelected = request.args.get('aSelected', type=str)
    extraModels = request.args.get('extraModels', type=str)
    extraAnalyses = request.args.get('extraAnalyses', type=str)
    search = request.args.get('modelSearch', type=str)

    extraModels = [s.strip() for s in extraModels.split(',')]
    extraAnalyses = [s.strip() for s in extraAnalyses.split(',')]

    global _previous_update_models
    pA, xM, xA, pM = _previous_update_models
    if (aSelected == pA) and (extraModels == xM) and (extraAnalyses == xA):
        model_list = pM
    else:
        modeltree = (
                session.query(Analysis.modeltree)
                .filter(Analysis.name == aSelected)
                .first()
                )
        # Pass modeltree string from Analysis to a ModelFinder constructor,
        # which will use a series of internal methods to convert the tree string
        # to a list of model names.
        # Then add any additional models specified in extraModels, and add
        # model_lists from extraAnalyses.
        if modeltree and modeltree[0]:
            model_list = _get_models(modeltree[0])
            extraModels = [m for m in extraModels if
                           (m not in model_list and m.strip() != '')]
            model_list.extend(extraModels)
            if extraAnalyses:
                analyses = (
                        session.query(Analysis.modeltree)
                        .filter(Analysis.name.in_(extraAnalyses))
                        .all()
                        )
                for t in [a.modeltree for a in analyses]:
                    extras = [m for m in _get_models(t) if m not in model_list]
                    model_list.extend(extras)
            _previous_update_models = (aSelected, extraModels,
                                       extraAnalyses, model_list)

        else:
            _previous_update_models = ('', '', '', [])
            return jsonify(modellist="Model tree not found.")

    session.close()

    filtered_models = simple_search(search, model_list)

    return jsonify(modellist=filtered_models)


def _get_models(modeltree):
    load, mod, fit = _get_trees(modeltree)
    if load and mod and fit:
        loader = ModelFinder(load).modellist
        model = ModelFinder(mod).modellist
        fitter = ModelFinder(fit).modellist
        combined = itertools.product(loader, model, fitter)
        model_list = ['_'.join(m) for m in combined]
    else:
        # Probably an old modeltree that doesn't have a separate
        # specification of loaders/preprocessors and fitters/postprocessors
        model_list = ModelFinder(mod, sep='_').modellist

    return model_list


_previous_update_cells = ('', [])
@app.route('/update_cells')
def update_cells():
    """Update the list of cells in the cell selector after a batch
    is selected (this will cascade from an analysis selection).

    Also updates current batch in Analysis for current analysis.

    """

    session = Session()
    db_tables = Tables()
    Batches = db_tables['Batches']
    sBatch = db_tables['sBatch']
    Analysis = db_tables['Analysis']

    # Only get the numerals for the selected batch, not the description.
    bSelected = request.args.get('bSelected')
    aSelected = request.args.get('aSelected')
    search = request.args.get('cellSearch')

    global _previous_update_cells
    previous_batch, previous_celllist = _previous_update_cells
    if bSelected == previous_batch:
        celllist = previous_celllist
    else:
        celllist = [
                i[0] for i in
                session.query(Batches.cellid)
                .filter(Batches.batch == bSelected[:3])
                .all()
                ]
        _previous_update_cells = (bSelected, celllist)

    batchname = (
            session.query(sBatch)
            .filter(sBatch.id == bSelected[:3])
            .first()
            )
    if batchname:
        batch = str(bSelected[:3] + ': ' + batchname.name)
    else:
        batch = bSelected
    analysis = (
            session.query(Analysis)
            .filter(Analysis.name == aSelected)
            .first()
            )
    # don't change batch association if batch is blank
    if analysis and bSelected:
        analysis.batch = batch

    session.commit()
    session.close()

    # remove pairs
    filtered_cellids = [c for c in celllist if '+' not in c]
    # remove siteids
    filtered_cellids = [c for c in filtered_cellids if '-' in c]
    # filter by search string
    filtered_cellids = simple_search(search, filtered_cellids)

    return jsonify(celllist=filtered_cellids)


_previous_update_results = ('', [], [], [], '')
@app.route('/update_results')
def update_results():
    """Update the results table after a batch, cell or model selection
    is changed.

    """

    user = get_current_user()
    session = Session()
    Results = Tables()['Results']

    nullselection = """
            MUST SELECT A BATCH AND ONE OR MORE CELLS AND
            ONE OR MORE MODELS BEFORE RESULTS WILL UPDATE
            """

    bSelected = request.args.get('bSelected')
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    colSelected = request.args.getlist('colSelected[]')
    # If no batch, cell or model is selected, display an error message.
    if (len(bSelected) == 0) or (not cSelected) or (not mSelected):
        return jsonify(resultstable=nullselection)

    global _previous_update_results
    pB, pCe, pM, pCo, pR = _previous_update_results
    if ((pB == bSelected) and (pCe == cSelected) and (pM == mSelected)
         and (pCo == colSelected)):
        resultstable = pR
    else:

        # Only get numerals for selected batch.
        bSelected = bSelected[:3]
        # Use default value of 500 if no row limit is specified.
        rowlimit = request.args.get('rowLimit', 500)
        ordSelected = request.args.get('ordSelected')
        # Parse string into appropriate sqlalchemy method
        if ordSelected == 'asc':
            ordSelected = asc
        elif ordSelected == 'desc':
            ordSelected = desc
        sortSelected = request.args.get('sortSelected', 'cellid')

        # Always add cellid and modelname to column lists,
        # since they are required for selection behavior.
        cols = [
                getattr(Results, 'cellid'),
                getattr(Results, 'modelname'),
                ]
        cols += [
                getattr(Results, c) for c in colSelected
                if hasattr(Results, c)
                ]

        # Package query results into a DataFrame
        results = psql.read_sql_query(
                Query(cols, session)
                .filter(Results.batch == bSelected)
                .filter(Results.cellid.in_(cSelected))
                .filter(NarfResults.modelname.in_(mSelected))
                .filter(or_(
                        int(user.sec_lvl) == 9,
                        NarfResults.public == '1',
                        NarfResults.labgroup.ilike('%{0}%'.format(user.labgroup)),
                        NarfResults.username == user.username,
                        ))
                .order_by(ordSelected(getattr(NarfResults, sortSelected)))
                .limit(rowlimit).statement,
                session.bind
                )
        with pd.option_context('display.max_colwidth', None):
            resultstable = results.to_html(
                    index=False, classes="table-hover table-condensed",
                    )
        _previous_update_results = (bSelected, cSelected, mSelected, colSelected,
                                    resultstable)

    session.close()

    return jsonify(resultstable=resultstable)


_previous_update_analysis = ([], [], [], [])
@app.route('/update_analysis')
def update_analysis():
    """Update list of analyses after a tag and/or filter selection changes."""

    user = get_current_user()
    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    tagSelected = request.args.getlist('tagSelected[]')
    statSelected = request.args.getlist('statSelected[]')

    global _previous_update_analysis
    previous_tags, previous_stats, previous_analyses, previous_ids = _previous_update_analysis
    if (previous_tags == tagSelected) and (previous_stats == statSelected):
        analysislist = previous_analyses
        analysis_ids = previous_ids
    else:
        # If special '__any' value is passed, set tag and status to match any
        # string in ilike query.
        if '__any' in tagSelected:
            tagStrings = [NarfAnalysis.tags.ilike('%%')]
        else:
            tagStrings = [
                    NarfAnalysis.tags.ilike('%{0}%'.format(tag))
                    for tag in tagSelected
                    ]
        if '__any' in statSelected:
            statStrings = [NarfAnalysis.status.ilike('%%')]
        else:
            statStrings = [
                    NarfAnalysis.status.ilike('%{0}%'.format(stat))
                    for stat in statSelected
                    ]
        analyses = (
                session.query(NarfAnalysis)
                .filter(or_(*tagStrings))
                .filter(or_(*statStrings))
                .filter(or_(
                        int(user.sec_lvl) == 9,
                        NarfAnalysis.public == '1',
                        NarfAnalysis.labgroup.ilike('%{0}%'.format(user.labgroup)),
                        NarfAnalysis.username == user.username,
                        ))
                .order_by(asc(NarfAnalysis.id))
                .all()
                )
        analysislist = [
                a.name for a in analyses
                ]
        analysis_ids = [
                a.id for a in analyses
                ]

    session.close()

    return jsonify(analysislist=analysislist, analysis_ids=analysis_ids)


@app.route('/update_analysis_details')
def update_analysis_details():
    """Update contents of the analysis details popover when the analysis
    selection is changed.

    """

    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    # TODO: Find a better/centralized place to store these options.
    # Columns to display in detail popup - add/subtract here if desired.
    detailcols = n_ui.detailcols

    aSelected = request.args.get('aSelected')

    cols = [
            getattr(NarfAnalysis, c) for c in detailcols
            if hasattr(NarfAnalysis, c)
            ]

    # Package query results into a DataFrame
    results = psql.read_sql_query(
            Query(cols, session)
            .filter(NarfAnalysis.name == aSelected)
            .statement,
            session.bind
            )

    detailsHTML = """"""

    if results.size > 0:
        for col in detailcols:
            # Use a single line for id and status columns
            if (col == 'id') or (col == 'status'):
                detailsHTML += """
                    <p><strong>%s</strong>: %s</p>
                    """ % (col, results[col].iat[0])
            # Use a header + paragraph for everything else
            else:
                detailsHTML += """
                    <h5><strong>%s</strong>:</h5>
                    <p>%s</p>
                    """ % (col, results[col].iat[0])

    session.close()

    return jsonify(details=detailsHTML)


@app.route('/update_status_options')
def update_status_options():

    user = get_current_user()
    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    statuslist = [
        i[0] for i in
        session.query(NarfAnalysis.status)
        .filter(or_(
                NarfAnalysis.public == '1',
                NarfAnalysis.labgroup.ilike('%{0}%'.format(user.labgroup)),
                NarfAnalysis.username == user.username,
                ))
        .distinct().all()
        ]

    session.close()

    return jsonify(statuslist=statuslist)


@app.route('/update_tag_options')
def update_tag_options():

    user = get_current_user()
    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    tags = [
        i[0].split(",") for i in
        session.query(NarfAnalysis.tags)
        .filter(or_(
                NarfAnalysis.public == '1',
                NarfAnalysis.labgroup.ilike('%{0}%'.format(user.labgroup)),
                NarfAnalysis.username == user.username,
                ))
        .distinct().all()
        ]
    # Flatten list of lists into a single list of all tag strings
    # and remove leading and trailing whitespace.
    taglistbldupspc = [i for sublist in tags for i in sublist]
    taglistbldup = [t.strip() for t in taglistbldupspc]
    # Reform the list with only unique tags
    taglistbl = list(set(taglistbldup))
    # Finally, remove any blank tags and sort the list.
    taglist = [t for t in taglistbl if t != '']
    taglist.sort()

    session.close()

    return jsonify(taglist=taglist)


##############################################################################
################      edit/delete/new  functions for Analysis Editor #########
##############################################################################


@app.route('/edit_analysis', methods=['GET', 'POST'])
@login_required
def edit_analysis():
    """Take input from Analysis Editor modal and save it to the database.

    Button : Edit Analysis

    """

    user = get_current_user()
    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    modTime = datetime.datetime.now().replace(microsecond=0)

    eName = request.args.get('name')
    eId = request.args.get('id')
    eStatus = request.args.get('status')
    eTags = request.args.get('tags')
    eQuestion = request.args.get('question')
    eAnswer = request.args.get('answer')
    eLoad = request.args.get('load')
    eMod = request.args.get('mod')
    eFit = request.args.get('fit')
    eTree = json.dumps([eLoad, eMod, eFit])

    if eId == '__none':
        checkExists = False
    else:
        checkExists = (
                session.query(NarfAnalysis)
                .filter(NarfAnalysis.id == eId)
                .first()
                )

    if checkExists:
        a = checkExists
        if (
                a.public
                or (user.labgroup in a.labgroup)
                or (a.username == user.username)
                ):
            a.name = eName
            a.status = eStatus
            a.question = eQuestion
            a.answer = eAnswer
            a.tags = eTags
            try:
                a.lastmod = modTime
            except:
                a.lastmod = str(modTime)
            a.modeltree = eTree
        else:
            log.info("You do not have permission to modify this analysis.")
            return jsonify(
                    success=("failed")
                    )
    # If it doesn't exist, add new sql alchemy object with the
    # appropriate attributes, which should get assigned to a new id
    else:
        # TODO: Currently copies user's labgroup by default.
        #       Is that the behavior we want?
        try:
            a = NarfAnalysis(
                    name=eName, status=eStatus, question=eQuestion,
                    answer=eAnswer, tags=eTags, batch='',
                    lastmod=modTime, modeltree=eTree, username=user.username,
                    labgroup=user.labgroup, public='0'
                    )
        except:
            a = NarfAnalysis(
                    name=eName, status=eStatus, question=eQuestion,
                    answer=eAnswer, tags=eTags, batch='',
                    lastmod=str(modTime), modeltree=eTree,
                    username=user.username, labgroup=user.labgroup, public='0'
                    )

        session.add(a)

    addedName = a.name
    session.commit()
    session.close()

    # Clear out cached analysis selection, otherwise the simple caching logic
    # thinks the analysis selection didn't change so it will load previous
    # modeltree
    global _previous_update_models
    _previous_update_models = ('', '', '', [])

    # After handling submissions, return user to main page so that it
    # refreshes with new analysis included in list
    return jsonify(success="Analysis %s saved successfully." % addedName)


@app.route('/get_current_analysis')
def get_current_analysis():
    """Populate the Analysis Editor form with the database contents for the
    currently selected analysis.

    """

    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    aSelected = request.args.get('aSelected')
    # If no analysis was selected, fill fields with blank text to
    # mimic 'New Analysis' behavior.
    if len(aSelected) == 0:
        return jsonify(
                name='', status='', tags='', question='',
                answer='', tree='',
                )

    a = (
        session.query(NarfAnalysis)
        .filter(NarfAnalysis.id == aSelected)
        .first()
        )

    load, mod, fit = _get_trees(a.modeltree)

    session.close()

    return jsonify(
            id=a.id, name=a.name, status=a.status, tags=a.tags,
            question=a.question, answer=a.answer, load=load, mod=mod, fit=fit,
            )


@app.route('/check_analysis_exists')
def check_analysis_exists():
    """Check for a duplicate analysis name when an Analysis Editor form is
    submitted. If a duplicate exists, warn the user before overwriting.

    """

    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    nameEntered = request.args.get('nameEntered')
    analysisId = request.args.get('analysisId')

    exists = False
    result = (
            session.query(NarfAnalysis)
            .filter(NarfAnalysis.name == nameEntered)
            .first()
            )

    # only set to True if id is different, so that
    # overwriting own analysis doesn't cause flag
    if result and (
            analysisId == '__none' or
            (int(result.id) != int(analysisId))
            ):
        exists = True

    session.close()

    return jsonify(exists=exists)


@app.route('/delete_analysis')
@login_required
def delete_analysis():
    """Delete the selected analysis from the database."""

    user = get_current_user()
    session = Session()
    NarfAnalysis = Tables()['NarfAnalysis']

    success = False
    aSelected = request.args.get('aSelected')
    if len(aSelected) == 0:
        return jsonify(success=success)

    result = (
            session.query(NarfAnalysis)
            .filter(NarfAnalysis.id == aSelected)
            .first()
            )
    if result is None:
        return jsonify(success=success)

    if (result.public
            or (result.username == user.username)
            or (user.labgroup in result.labgroup)):
        success = True
        session.delete(result)
        session.commit()
    else:
        log.info("You do not have permission to delete this analysis.")
        return jsonify(success=success)

    session.close()

    return jsonify(success=success)


def _get_trees(modeltree):
    try:
        load, mod, fit = json.loads(modeltree)
    except ValueError:
        # Modeltree is still using old paradigm with only one entry
        load, mod, fit = '', modeltree, ''

    return load, mod, fit


####################################################################
###############     TABLE SELECTION FUNCTIONS     ##################
####################################################################


@app.route('/get_preview')
def get_preview():
    """Queries the database for the filepath to the preview image
    for the selected cell, batch and model combination(s)

    """

    session = Session()
    NarfResults = Tables()['NarfResults']

    # Only get the numerals for the selected batch, not the description.
    bSelected = request.args.get('bSelected', type=str)[:3]
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')

    figurefile = None
    path = (
            session.query(NarfResults)
            .filter(NarfResults.batch == bSelected)
            .filter(NarfResults.cellid.in_(cSelected))
            .filter(NarfResults.modelname.in_(mSelected))
            .first()
            )

    if not path:
        session.close()
        return jsonify(image='missing preview')
    else:
        figurefile = str(path.figurefile)
        session.close()

    # Another temporary compatibility hack to convert
    # s3://... to https://
    if figurefile.startswith('s3'):
        prefix = 'https://s3-us-west2.amazonaws.com'
        parsed = urlparse(figurefile)
        bucket = parsed.netloc
        path = parsed.path
        figurefile = prefix + '/' + bucket + '/' + path

    # TODO: this should eventually be the only thing that gets
    #       called - above try/except ugliness is temporary for
    #       backwards compatibility
    image_bytes = load_resource(figurefile)
    b64img = str(b64encode(image_bytes))[2:-1]
    return jsonify(image=b64img)


###############################################################################
#################   SAVED SELECTIONS    #######################################
###############################################################################


@app.route('/get_saved_selections')
def get_saved_selections():
    session = Session()
    NarfUsers = Tables()['NarfUsers']

    user = get_current_user()
    user_entry = (
            session.query(NarfUsers)
            .filter(NarfUsers.username == user.username)
            .first()
            )
    if not user_entry:
        return jsonify(response="user not logged in, can't load selections")
    selections = user_entry.selections
    null = False
    if not selections:
        null = True
    session.close()
    return jsonify(selections=selections, null=null)


@app.route('/set_saved_selections', methods=['GET', 'POST'])
def set_saved_selections():
    user = get_current_user()
    if not user.username:
        return jsonify(
                response="user not logged in, can't save selections",
                null=True,
                )
    session = Session()
    NarfUsers = Tables()['NarfUsers']

    saved_selections = request.args.get('stringed_selections')
    user_entry = (
            session.query(NarfUsers)
            .filter(NarfUsers.username == user.username)
            .first()
            )
    user_entry.selections = saved_selections
    session.commit()
    session.close()

    return jsonify(response='selections saved', null=False)


def simple_search(query, collection):
    '''
    Filter cellids or modelnames by simple space-separated search strings.

    Parameters:
    ----------
    query : str
        Search string, see syntax below.
    collection : list
        List of items to compare the search string against.
        Ex: A list of cellids.

    Returns:
    -------
    filtered_collection : list
        Collection with all non-matched entries removed.

    Syntax:
    ------
    space : OR
    &     : AND
    !     : NEGATE
    (In that order of precedence)

    Example:
    collection = ['AMT001', BRT000', 'TAR002', 'TAR003']

    search('AMT', collection)
    >>  ['AMT001']

    search('AMT TAR', collection)
    >>  ['AMT001', 'TAR002', 'TAR003']

    search ('AMT TAR&!003', collection)
    >>  ['AMT001', 'TAR002']

    search ('AMT&TAR', collection)
    >> []


    '''
    filtered_collection = []
    for c in collection:
        for s in query.split(' '):
            if ('&' in s) or ('!' in s):
                ands = s.split('&')
                all_there = True
                for a in ands:
                    negate = ('!' in a)
                    b = a.replace('!', '')
                    if (b in c) or (c in b):
                        if negate:
                            all_there = False
                            break
                        else:
                            continue
                    else:
                        if negate:
                            continue
                        else:
                            all_there = False
                            break
                if all_there:
                    filtered_collection.append(c)
            else:
                if (s in c) or (c in s):
                    filtered_collection.append(c)
                    break
                else:
                    pass

    return filtered_collection
