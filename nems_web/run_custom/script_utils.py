""" Helper functions for filtering cells and forming data matrix. """

import pkgutil

import pandas as pd
import numpy as np
import pandas.io.sql as psql

from nems.db import Session, Tables


def scan_for_scripts():
    #package = ns
    #script_names = [
    #        modname for importer, modname, ispkg
    #        in pkgutil.iter_modules(package.__path__)
    #        ]
    script_names = ['NotImplemented']
    return script_names


def filter_cells(session, batch, cells, min_snr=0, min_iso=0, min_snri=0):
    """ Returns a list of cells that don't meet the minimum snr/iso/snri
    criteria specified. The calling function can then remove them from the
    cell list if desired (ex: for cell in bad_cells, cell_list.remove(cell))

    Arguments:
    ----------
    batch : int
        The batch number to query.
    cells : list
        A list of the cellids to be checked.
    min_snr : float
        The minimum signal to noise ratio desired.
    min_iso : float
        The minimum isolation value desired.
    min_snri : float
        The minimum signal to noise ratio index desired.
    session : object
        An open database session object for querying NarfBatches.

    """

    bad_cells = []
    session = Session()
    NarfBatches = Tables()['NarfBatches']

    for cellid in cells:
        dbCriteria = (
                session.query(NarfBatches)
                .filter(NarfBatches.batch == batch)
                .filter(NarfBatches.cellid.ilike(cellid))
                .first()
                )
        if dbCriteria:

            db_snr = min(dbCriteria.est_snr, dbCriteria.val_snr)
            db_iso = dbCriteria.min_isolation
            db_snri = dbCriteria.min_snr_index

            a = (min_snr > db_snr)
            b = (min_iso > db_iso)
            c = (min_snri > db_snri)

            if a or b or c:
                bad_cells.append(cellid)

                # Uncomment section below to include verbose output of
                # why individual cells were 'bad'

                #filterReason = ""
                #if a:
                #    filterReason += (
                #            "min snr: %s -- was less than criteria: %s\n"
                #            %(db_snr, min_snr)
                #            )
                #if b:
                #    filterReason += (
                #            "min iso: %s -- was less than criteria: %s\n"
                #            %(db_iso, min_iso)
                #            )
                #if c:
                #    filterReason += (
                #            "min snr index: %s -- was less than criteria: %s\n"
                #            %(db_snri, min_snri)
                #            )
                #print(
                #    "Removing cellid: %s,\n"
                #    "because: %s"
                #    %(cellid, filterReason)
                #    )
        else:
            print(
                "No entry in NarfBatches for cellid: {0} in batch: {1}"
                .format(cellid, batch)
                )
            bad_cells.append(cellid)

    print("Number of bad cells to snr/iso criteria: {0}".format(len(bad_cells)))
    print("Out of total cell count: {0}".format(len(cells)))

    return bad_cells


def form_data_array(
        session, batch, cells, models, columns=None, only_fair=True,
        include_outliers=False,
        ):

    # TODO: figure out a good way to form this from the existing
    #       dataframe instead of making a new one then copying over.
    #       Should be able to just re-index then apply some
    #       lambda function over vectorized dataframe for filtering?
    session = Session()
    NarfResults = Tables()['NarfResults']

    data = psql.read_sql_query(
            session.query(NarfResults)
            .filter(NarfResults.batch == batch)
            .filter(NarfResults.cellid.in_(cells))
            .filter(NarfResults.modelname.in_(models))
            .statement,
            session.bind
            )
    if not columns:
        columns = data.columns.values.tolist()

    multiIndex = pd.MultiIndex.from_product(
            [cells, models], names=['cellid', 'modelname'],
            )
    newData = pd.DataFrame(
            index=multiIndex, columns=columns,
            )
    newData.sort_index()

    # columns that don't contain performance data - this will be excluded
    # from outlier checks.
    non_comp_columns = [
            'id', 'cellid', 'modelname', 'batch', 'n_parms', 'figurefile',
            'githash', 'lastmod', 'score', 'sparsity', 'modelpath', 'modelfile',
            'username', 'labgroup', 'public',
            ]
    for c in cells:
        for m in models:
            dataRow = data.loc[(data.cellid == c) & (data.modelname == m)]

            # if col is in the non_comp list, just copy the value if it exists
            # then move on to the next iteration.
            for col in columns:
                if col in non_comp_columns:
                    try:
                        newData[col].loc[c,m] = dataRow[col].values.tolist()[0]
                        continue
                    except:
                        newData[col].loc[c,m] = np.nan
                        continue
                value = np.nan
                newData[col].loc[c,m] = value
                # If loop hits a continue, value will be left as NaN.
                # Otherwise, will be assigned a value from data
                # after passing all checks.
                try:
                    value = dataRow[col].values.tolist()[0]
                except Exception as e:
                    # Error should mean no value was recorded,
                    # so leave as NaN.
                    # No need to run outlier checks if value is missing.
                    print("No %s recorded for %s,%s"%(col,m,c))
                    continue

                if not include_outliers:
                    # If outliers is false, run a bunch of checks based on
                    # measure and if a check fails, step out of the loop.

                    # Comments for each check are copied from
                    # from Narf_Analysis : compute_data_matrix

                    # "Drop r_test values below threshold"
                    a1 = (col == 'r_test')
                    b1 = (value < dataRow['r_floor'].values.tolist()[0])
                    a2 = (col == 'r_ceiling')
                    b2 = (
                        dataRow['r_test'].values.tolist()[0]
                        < dataRow['r_floor'].values.tolist()[0]
                        )
                    a3 = (col == 'r_floor')
                    b3 = b1
                    if (a1 and b1) or (a2 and b2) or (a3 and b3):
                        continue

                    # "Drop MI values greater than 1"
                    a1 = (col == 'mi_test')
                    b1 = (value > 1)
                    a2 = (col == 'mi_fit')
                    b2 = (0 <= value <= 1)
                    if (a1 and b1) or (a2 and not b2):
                        continue

                    # "Drop MSE values greater than 1.1"
                    a1 = (col == 'mse_test')
                    b1 = (value > 1.1)
                    a2 = (col == 'mse_fit')
                    b2 = b1
                    if (a1 and b1) or (a2 and b2):
                        continue

                    # "Drop NLOGL outside normalized region"
                    a1 = (col == 'nlogl_test')
                    b1 = (-1 <= value <= 0)
                    a2 = (col == 'nlogl_fit')
                    b2 = b1
                    if (a1 and b1) or (a2 and b2):
                        continue

                    # TODO: is this still used? not listed in NarfResults
                    # "Drop gamma values that are too low"
                    a1 = (col == 'gamma_test')
                    b1 = (value < 0.15)
                    a2 = (col == 'gamma_fit')
                    b2 = b1
                    if (a1 and b1) or (a2 and b2):
                        continue

                # TODO: is an outlier check needed for cohere_test
                #       and/or cohere_fit?

                # If value existed and passed outlier checks,
                # re-assign it to the proper DataFrame position
                # to overwrite the NaN value.
                newData[col].loc[c,m] = value

    if only_fair:
        # If fair is checked, drop all rows that contain a NaN value for
        # every column.
        for c in cells:
            for m in models:
                if newData.loc[c,m].isnull().values.all():
                    newData.drop(c, level='cellid', inplace=True)
                    break


    # Swap the 0th and 1st levels so that modelname is the primary index,
    # since most plots group by model.
    newData = newData.swaplevel(i=0, j=1, axis=0)

    return newData
