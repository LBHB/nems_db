import os
import copy

import pylab as pl
import numpy as np
import itertools
from functools import partial

from lbhb.analysis import rdt

import nems0.analysis.api
import nems0.fitters.api
import nems0.preprocessing
import nems0.plots.api
import nems0.initializers
import nems0.metrics.corrcoef
import nems0.analysis.test_prediction
import nems0.xforms
import nems0.plots.file

from nems import db


def do_fit(batch, cell, wcg_n, fir_n, shuffle_phase, shuffle_stream):

    recording = rdt.io.load_recording(batch, cell, True, True)
    est_times, val_times = rdt.preprocessing.split_est_val(recording, False)

    modelspec = rdt.modules.create_modelspec(recording, wcg_n, fir_n, True, 'dual')
    modelspec = nems0.priors.set_mean_phi(modelspec)
    model_name = '_'.join(m['id'] for m in modelspec)

    if shuffle_phase:
        model_name = f'shufflePhase_{model_name}'
        recording['repeating'] = recording['repeating'].shuffle_time()

    if shuffle_stream:
        model_name = f'shuffleStream_{model_name}'
        recording = rdt.preprocessing.shuffle_streams(recording)

    est_recording = rdt.preprocessing.select_times(recording, est_times,
                                                   random_only=False,
                                                   dual_only=True)

    val_recording = rdt.preprocessing.select_times(recording, val_times,
                                                   random_only=False,
                                                   dual_only=True)

    # Fit all but the gain term. Do not include dexp (note the modelspec
    # slice).
    mapper = partial(nems0.fitters.mappers.simple_vector, subset=[0, 1, 3])
    prefit_modelspec, = nems0.analysis.api.fit_basic(
        est_recording,
        modelspec[:-1],
        fitter=nems0.fitters.api.scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 1000}},
    )

    # Copy over the prefit modelspec items
    modelspec[:-1] = copy.deepcopy(prefit_modelspec)

    # Fit all but the gain term. Include dexp this time.
    mapper = partial(nems0.fitters.mappers.simple_vector, subset=[0, 1, 3, 4])
    strf_modelspec, = nems0.analysis.api.fit_basic(
        est_recording,
        modelspec,
        fitter=nems0.fitters.api.scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-8, 'maxiter': 1000}},
    )

    # Now fit gain, level and dexp
    mapper = partial(nems0.fitters.mappers.simple_vector, subset=[2, 3, 4])
    final_modelspecs = nems0.analysis.api.fit_basic(
        est_recording,
        strf_modelspec,
        fitter=nems0.fitters.api.scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-10, 'maxiter': 5000}},
        mapper=mapper,
    )

    est_pred, val_pred = nems0.analysis.api \
        .generate_prediction(est_recording, val_recording, final_modelspecs)

    final_modelspecs = nems0.analysis.api \
        .standard_correlation(est_pred, val_pred, final_modelspecs)

    destination = f'/auto/data/nems_db/results/{batch}/{cell}/{model_name}/'
    figurefile = os.path.join(destination, 'figure.0000.png')

    final_modelspecs[0][0]['meta'].update({
        'batch': batch,
        'cellid': cell,
        'modelname': model_name,
        'loader': 'custom',
        'fitkey': 'custom',
        'recording': 'custom',
        'username': 'nems',
        'labgroup': 'lbhb',
        'public': 1,
        'githash': 'TODO',
        'modelpath': destination,
        'figurefile': figurefile,
    })

    if not os.path.exists(destination):
        os.makedirs(destination)

    nems0.xforms.save_analysis(destination, recording,
                              modelspecs=final_modelspecs, xfspec=['custom'],
                              log='None', figures=[])

    context = {'est': est_pred, 'val': val_pred, 'modelspecs': final_modelspecs}
    f = nems0.plots.api.quickplot(context);
    f.savefig(os.path.join(destination, 'quickplot.png'))


def main():
    import argparse

    qid = os.environ.get('QUEUEID', None)
    parser = argparse.ArgumentParser(description='Fit cell from batch to model')
    parser.add_argument('cell', type=str, help='Batch ID containing data')
    parser.add_argument('batch', type=str, help='Cell ID to fit')
    parser.add_argument('--wcg_n', type=int, help='wcg rank', default=2)
    parser.add_argument('--fir_n', type=int, help='FIR ntaps', default=15)
    parser.add_argument('--shuffle-phase', action='store_true', help='Shuffle phase')
    parser.add_argument('--shuffle-stream', action='store_true', help='Shuffle stream')
    parser.add_argument('model', type=str, help='Model name (ignored)', nargs='?')

    args = parser.parse_args()
    if qid is not None:
        db.update_job_start(qid)
        nems0.utils.progress_fun = db.update_job_tick

    do_fit(args.batch, args.cell, args.wcg_n, args.fir_n, args.shuffle_phase,
           args.shuffle_stream)

    if qid is not None:
        db.update_job_complete(qid)


if __name__ == '__main__':
    main()
