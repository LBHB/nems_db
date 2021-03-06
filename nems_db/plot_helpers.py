import logging

import pandas.io.sql as psql

import nems_db.plots as plots
from nems.db import Session, Tables, get_batch_cells

log = logging.getLogger(__name__)


def plot_filtered_batch(batch, models, measure, plot_type,
                        only_fair=True, include_outliers=False, display=True,
                        extra_cols=[], snr=0.0, iso=0.0, snr_idx=0.0):
    cells = get_batch_cells(batch)['cellid'].tolist()
    cells = get_filtered_cells(cells, snr, iso, snr_idx)
    plot = get_plot(cells, models, batch, measure, plot_type, only_fair,
                    include_outliers, display)
    plot.generate_plot()
    return plot


def get_plot(cells, models, batch, measure, plot_type, only_fair=True,
             include_outliers=False, display=True):
    session = Session()
    Results = Tables()['Results']

    results_df = psql.read_sql_query(
            session.query(Results)
            .filter(Results.batch == batch)
            .filter(Results.cellid.in_(cells))
            .filter(Results.modelname.in_(models))
            .statement, session.bind
            )
    results_models = [
            m for m in
            list(set(results_df['modelname'].values.tolist()))
            ]
    ordered_models = [
            m for m in models
            if m in results_models
            ]
    PlotClass = getattr(plots, plot_type)
    plot = PlotClass(
            data=results_df, measure=measure, models=ordered_models,
            fair=only_fair, outliers=include_outliers, display=display
            )
    session.close()
    return plot


def get_filtered_cells(cells, batch, snr=0.0, iso=0.0, snr_idx=0.0):
    """Removes cellids from list if they do not meet snr/iso criteria."""
    session = Session()
    Batches = Tables()['Batches']

    snr = max(snr, 0)
    iso = max(iso, 0)
    snr_idx = max(snr_idx, 0)

    db_criteria = psql.read_sql_query(
            session.query(Batches)
            .filter(Batches.cellid.in_(cells))
            .filter(Batches.min_snr_index >= snr_idx)
            .filter(Batches.min_isolation >= iso)
            .filter(Batches.est_snr >= snr)
            .filter(Batches.val_snr >= snr)
            .statement, session.bind
            )

    return list(set(db_criteria['cellid'].values.tolist()))
