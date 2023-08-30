"""Miscellaneous view functions.

Contents so far:
    status_report

"""
import logging
import itertools
from base64 import b64encode

import pandas.io.sql as psql
import pandas as pd
from flask import request, render_template, jsonify

from nems_web.nems_analysis import app, bokeh_version
from nems0.db import Session, Tables
from nems_web.reports.reports import Performance_Report, Fit_Report

log = logging.getLogger(__name__)


@app.route('/batch_performance', methods=['GET', 'POST'])
def batch_performance():
    session = Session()
    Results = Tables()['Results']

    cSelected = request.form['cSelected']
    bSelected = request.form['bSelected'][:3]
    mSelected = request.form['mSelected']
    findAll = request.form['findAll']

    cSelected = cSelected.split(',')
    mSelected = mSelected.split(',')

    if int(findAll):
        results = psql.read_sql_query(
                session.query(
                        Results.cellid, Results.modelname,
                        Results.r_test
                        )
                .filter(Results.batch == bSelected)
                .statement,
                session.bind
                )
    else:
        results = psql.read_sql_query(
                session.query(
                        Results.cellid, Results.modelname,
                        Results.r_test
                        )
                .filter(Results.batch == bSelected)
                .filter(Results.cellid.in_(cSelected))
                .filter(Results.modelname.in_(mSelected))
                .statement,
                session.bind
                )
    # get back list of models that matched other query criteria
    results_models = [
            m for m in
            list(set(results['modelname'].values.tolist()))
            ]
    # filter mSelected to match results models so that list is in the
    # same order as on web UI
    ordered_models = [
            m for m in mSelected
            if m in results_models
            ]

    report = Performance_Report(results, bSelected, ordered_models)
    report.generate_plot()

    session.close()
    return render_template(
            'batch_performance.html', script=report.script, div=report.div,
            bokeh_version=bokeh_version,
            )


@app.route('/fit_report')
def fit_report():
    session = Session()
    db_tables = Tables()
    tQueue = db_tables['tQueue']
    Results = db_tables['Results']

    cSelected = request.args.getlist('cSelected[]')
    bSelected = request.args.get('bSelected')[:3]
    mSelected = request.args.getlist('mSelected[]')

    multi_index = pd.MultiIndex.from_product(
            [mSelected, cSelected], names=['modelname', 'cellid']
            )
    status = pd.DataFrame(index=multi_index, columns=['yn'])

    tuples = list(itertools.product(cSelected, [bSelected], mSelected))
    notes = ['{0}/{1}/{2}'.format(t[0], t[1], t[2]) for t in tuples]

    qdata = psql.read_sql_query(
            session.query(tQueue)
            .filter(tQueue.note.in_(notes))
            .statement,
            session.bind,
            )

    results = psql.read_sql_query(
            session.query(
                    Results.cellid, Results.batch,
                    Results.modelname,
                    )
            .filter(Results.batch == bSelected)
            .filter(Results.cellid.in_(cSelected))
            .filter(Results.modelname.in_(mSelected))
            .statement,
            session.bind
            )

    for i, t in enumerate(tuples):
        yn = 0.3  # missing
        try:
            complete = qdata.loc[qdata['note'] == notes[i], 'complete'].iloc[0]
            if complete < 0:
                yn = 0.4  # in progress
            elif complete == 0:
                yn = 0.5  # not started
            elif complete == 1:
                yn = 0.6  # finished
            elif complete == 2:
                yn = 0  # dead entry
            else:
                pass  # unknown value, so leave as missing?
        except:
            try:
                result = results.loc[
                        (results['cellid'] == t[0])
                        & (results['batch'] == int(t[1]))
                        & (results['modelname'] == t[2]),
                        'cellid'
                        ].iloc[0]
                yn = 0.6
            except:
                pass
        status['yn'].loc[t[2], t[0]] = yn

    status.reset_index(inplace=True)
    status = status.pivot(index='cellid', columns='modelname', values='yn')
    status = status[status.columns].astype(float)
    report = Fit_Report(status)
    report.generate_plot()

    session.close()

    image = str(b64encode(report.img_str))[2:-1]
    return jsonify(image=image)
