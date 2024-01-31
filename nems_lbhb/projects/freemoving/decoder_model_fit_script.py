from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
# import nems0.epoch as ep
# import nems0.plots.api as nplt
from nems0.utils import smooth
# from nems_lbhb.xform_wrappers import generate_recording_uri
# from nems_lbhb.baphy_experiment import BAPHYExperiment
# from nems_lbhb.plots import plot_waveforms_64D
# from nems_lbhb.preprocessing import impute_multi
# from nems_lbhb import baphy_io
# from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
#     free_scatter_sum, dlc2dist

from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential, LevelShift, ReLU
from nems.layers.base import Layer, Phi, Parameter

import os
import sys
import logging
from pathlib import Path
import subprocess

from os.path import basename, join
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from nems0 import db
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems_lbhb.projects.freemoving import decoder_tools
from nems.tools import json
from nems.tools.json import save_model
from distutils.util import strtobool

log = logging.getLogger(__name__)

force_SDB = True
try:
    if 'SDB-' in sys.argv[3]:
        force_SDB = True
except:
    pass
if force_SDB:
    os.environ['OPENBLAS_VERBOSE'] = '2'
    os.environ['OPENBLAS_CORETYPE'] = 'sandybridge'

import nems0.xform_helper as xhelp
import nems0.utils
from nems0.uri import save_resource
from nems0 import get_setting

if force_SDB:
    log.info('Setting OPENBLAS_CORETYPE to sandybridge')

try:
    import nems0.db as nd

    db_exists = True
except Exception as e:
    # If there's an error import nems0.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems0.db, can't update tQueue")
    print(e)
    db_exists = False

if __name__ == '__main__':

    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems0.utils.progress_fun = nd.update_job_tick
        if 'SLURM_JOB_ID' in os.environ:
            jobid = os.environ['SLURM_JOB_ID']
            nd.update_job_pid(jobid)
            nd.update_startdate()
            comment = ' '.join(sys.argv[1:])
            update_comment = ['sacctmgr', '-i', 'modify', 'job', f'jobid={jobid}', 'set', f'Comment="{comment}"']
            subprocess.run(update_comment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.info(f'Set comment string to: "{comment}"')
    else:
        queueid = 0

    if queueid:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)
        log.info("HOSTNAME={}".format(os.environ.get('HOSTNAME', 'unknown')))

    if len(sys.argv) < 4:
        print('syntax: decoder_model_fit_script.py siteid batch modelname')
        exit(-1)

    decoded_variables = ['d', 'theta', 'v', 'front_x', 'front_y']
    dlc_chans = 8
    rasterfs = 25
    siteid = sys.argv[1]
    batch = int(sys.argv[2])
    modelname = sys.argv[3]

    mparams = modelname.split('.')
    layer_name = mparams[mparams.index('-rlyr')+1]
    perm_num = int(mparams[mparams.index('-perms')+1])
    inputsig = [mparams[mparams.index('-input')+1],]
    firfilt = bool(strtobool(mparams[mparams.index('-firfilt') + 1]))

    # get data from A1 units
    print(f"loading recording...siteid/batch/rasterfs: {siteid}/{batch}/{rasterfs}")
    recs = decoder_tools.load_free_data(siteid, batch=batch, rasterfs=rasterfs,
                                    dlc_chans=dlc_chans, compute_position=False, remove_layer=layer_name, layer_perms=perm_num)
    print("number of recs: {}".format(len(recs)))
    for rec in recs:
        lyr_removed = rec.meta['rlyr']
        print('rec layer removed: {}'.format(lyr_removed))
        epoch_regex = "^STIM_"
        epochs = rec.epochs
        is_stim = epochs['name'].str.startswith('STIM')
        stim_epoch_list = epochs[is_stim]['name'].tolist()
        stim_epochs = list(set(stim_epoch_list))
        stim_epochs.sort()
        # how many times are stimuli repeated
        ep_counts = ep.epoch_occurrences(epochs, regex=epoch_regex)
        print(f"stim occurances: {ep_counts}")
        e = ep_counts.idxmax()
        cellids = rec['resp'].chans

        signames = inputsig
        print(f'input signal: {inputsig[0]}')
        for signame in signames:
            jackknifes = 10
            for jack in range(jackknifes):
                # ask about difference in mask by epoch vs mask by time
                if signame == 'resp':
                    est = rec.jackknife_mask_by_time(jackknifes,jack,invert=False)
                    val = rec.jackknife_mask_by_time(jackknifes, jack, invert=True)
                else:
                    est = rec.jackknife_mask_by_epoch(jackknifes,jack,'TRIAL',invert=False)
                    val = rec.jackknife_mask_by_epoch(jackknifes,jack,'TRIAL',invert=True)

                p = val['psth'].extract_epoch(e)
                r = val['resp'].extract_epoch(e)
                d = val['diff'].extract_epoch(e)

                # For a model that uses multiple inputs, we package the input data into
                # a dictionary. This way, layers can specify which inputs they need using the
                # assigned keys.

                val = val.apply_mask()
                est = est.apply_mask()

                # change 'diff' to 'resp' for non-PSTH subbed data
                input_sig_name = signame # 'resp'
                target_sig_names = ['dist', 'dlc']
                for target_sig_name in target_sig_names:
                    print("target chan: {}".format(target_sig_name))
                    for i, target_chan in enumerate(est[target_sig_name].chans):
                        if target_chan in decoded_variables:
                            print(f"target chan: {target_chan}")
                            input = est[input_sig_name]._data.T
                            target = est[target_sig_name]._data.T
                            test_input = val[input_sig_name]._data.T
                            test_target = val[target_sig_name]._data.T

                            targetchan=i # which channel of dist signal we're decoding

                            # calculate timebins that are valid in decoded variable
                            good_timebins = (np.isnan(target).sum(axis=1) == 0)
                            good_test_timebins = (np.isnan(test_target).sum(axis=1) == 0)

                            # select only good timebins
                            input = input[good_timebins,:]
                            test_input = test_input[good_test_timebins,:]

                            target = target[good_timebins,targetchan][:,np.newaxis]
                            test_target = test_target[good_test_timebins,targetchan][:,np.newaxis]

                            # select subset of cells (this can be driven off the database)
                            input = input[:, :]
                            test_input = test_input[:, :]

                            m, s = target.mean(axis=0, keepdims=True), target.std(axis=0, keepdims=True)
                            target = (target-m)/s
                            test_target = (test_target-m)/s

                            cellcount = input.shape[1]
                            dimcount = target.shape[1]
                            fircount = 5
                            # dimcount = 5
                            l2count = 10

                            # layers = [
                            #     WeightChannels(shape=(cellcount,1,5)),
                            #     FIR(shape=(5,1,5)),
                            #     ReLU(shape=(5,), no_shift=False, no_offset=False, no_gain=True),
                            #     WeightChannels(shape=(5,5)),
                            #     ReLU(shape=(5,), no_shift=False, no_offset=False, no_gain=True),
                            #     WeightChannels(shape=(5,1)),
                            #     DoubleExponential(shape=(dimcount,)),
                            # ]

                            layers = [
                                WeightChannels(shape=(cellcount,1,l2count)),
                                FIR(shape=(fircount,1,l2count), include_anticausal=firfilt),
                                ReLU(shape=(l2count,), no_shift=False, no_offset=False, no_gain=True),
                                WeightChannels(shape=(l2count,l2count)),
                                ReLU(shape=(l2count,), no_shift=False, no_offset=False, no_gain=True),
                                WeightChannels(shape=(l2count,dimcount)),
                                DoubleExponential(shape=(dimcount,)),
                            ]

                            modelstring = f"-wc.Nx1x{l2count}-fir.{fircount}x1x{l2count}-fircaus.{str(firfilt)}-relu.{l2count}-wc.{l2count}x{l2count}-relu.{l2count}-wc.{l2count}x{dimcount}-de.{dimcount}"
                            # tolerance = 1e-2
                            tolerance = 1e-3
                            # max_iter = 200
                            max_iter = 5000
                            cost_function = 'nmse'
                            fitter_options = {'cost_function': cost_function, 'early_stopping_delay': 50,
                                              'early_stopping_patience': 100,
                                              'early_stopping_tolerance': tolerance,
                                              'learning_rate': 1e-3, 'epochs': max_iter,
                                              }

                            model = Model(layers=layers)
                            model.name = f'id.{siteid}.tlyr.{layer_name}.rlyr.{lyr_removed}-isig.{signame}-tsig.{target_sig_name}-tch.{target_chan}_jk{jack}_{modelstring}_{cost_function}'
                            model = model.sample_from_priors()
                            model = model.sample_from_priors()
                            log.info(f'Site: {siteid}')
                            log.info(f'Model: {model.name}')

                            use_tf = True
                            if use_tf:
                                input_ = np.expand_dims(input, axis=0)
                                test_input_ = np.expand_dims(test_input, axis=0)
                                target_ = np.expand_dims(target, axis=0)

                                model = model.fit(input=input_, target=target_, backend='tf',
                                                  fitter_options=fitter_options, batch_size=None)

                                prediction = model.predict(input_, batch_size=None)[0,:,:]
                                test_prediction = model.predict(test_input_, batch_size=None)[0,:,:]
                                ccf = np.corrcoef(prediction[:,0],target[:,0])[0,1]
                                cct = np.corrcoef(test_prediction[:,0],test_target[:,0])[0,1]

                            model.meta['r_fit'] = ccf
                            model.meta['r_test'] = cct
                            model.meta['prediction'] = test_prediction
                            model.meta['resp'] = test_target
                            model.meta['siteid'] = siteid
                            model.meta['batch'] = rec.meta['batch']
                            model.meta['modelname'] = model.name
                            model.meta['cellids'] = est['resp'].chans
                            model.meta['cellid'] = siteid
                            model.meta['rlyr'] = rec.meta['rlyr']

                            model_save_path = Path('/auto/users/wingertj/models/decoding_layer_removal/')
                            print(f"saving model to {model_save_path/model.name}")
                            save_model(model, model_save_path/model.name)
                        else:
                            continue

    log.info("Done with fit.")

    # Mark completed in the queue. Note that this should happen last thing!
    # Otherwise the job might still crash after being marked as complete.
    if db_exists & bool(queueid):
        nd.update_job_complete(queueid)

