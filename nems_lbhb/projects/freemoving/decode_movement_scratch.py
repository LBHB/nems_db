from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#from scipy.interpolate import LinearNDInterpolator
#from scipy.ndimage import gaussian_filter

from nems0 import db, preprocessing
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


from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential, LevelShift, ReLU
from nems.layers.base import Layer, Phi, Parameter
from decoder_tools import xy_plot_animation

# if USE_DB:
#     siteid = "PRN034a"
    # siteid = "PRN010a"
    # siteid = "PRN009a"
    # siteid = "PRN011a"
    # siteid = "PRN022a"
    # siteid = "PRN047a"
    # siteid = "PRN015a"
runclassid = 132
rasterfs = 25
# sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
#dparm = db.pd_query(sql)

sql = f"SELECT distinct left(cellid,7) as siteid,stimpath,stimfile from sCellFile where runclassid={runclassid}"
dallfiles = db.pd_query(sql)
siteids = dallfiles['siteid'].unique().tolist()
decoded_dfs = []
siteids = ['PRN020a']
for siteid in siteids:
    # siteid=siteids[0]
    sql = f"SELECT count(cellid) as cellcount,stimpath,stimfile from sCellFile where cellid like '{siteid}%%' AND runclassid={runclassid} AND area='A1' group by stimpath,stimfile"
    dparminfo = db.pd_query(sql)

    parmfile = [r.stimpath+r.stimfile for i,r in dparminfo.iterrows()]
    cellids=None

# else:
#     parmfile = ["/auto/data/daq/Prince/PRN015/PRN015a01_a_NTD",
#                 "/auto/data/daq/Prince/PRN015/PRN015a02_a_NTD"]
#     cellids = None

## load the recording

    # decoded_dict['siteid'] = siteid
    try:
        ex = BAPHYExperiment(parmfile=parmfile, cellid=cellids)
        print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

        recache = True

        # load recording
        # rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
        #                        dlc=True, recache=recache, rasterfs=rasterfs,
        #                        dlc_threshold=0.2, fill_invalid='interpolate')
        rec = ex.get_recording(resp=True,
                               dlc=True, recache=recache, rasterfs=rasterfs,
                               dlc_threshold=0.2, fill_invalid='interpolate')
    except:
        continue

    # generate 'dist' signal from dlc signal
    rec = dlc2dist(rec, ref_x0y0=None, smooth_win=5, norm=False, verbose=False)

    # grab A1 units
    try:
        depth_info = baphy_io.get_depth_info(siteid=siteid)
        A1_units = depth_info.loc[depth_info['area']== 'A1'].index.tolist()
        A1_in_rec = [chan for chan in rec['resp'].chans if chan in A1_units]
        if len(A1_in_rec) == 0:
            A1_in_rec = [chan for chan in rec['resp'].chans if chan[:7] + chan[15:] in A1_units]
        # compute PSTH for repeated stimuli
        epoch_regex = "^STIM_"
        rec['resp'] = rec['resp'].extract_channels(A1_in_rec)
        rec['resp'] = rec['resp'].rasterize()
        rec['psth'] = preprocessing.generate_average_sig(rec['resp'], 'psth', epoch_regex=epoch_regex)
    except:
        continue
    # compute PSTH for repeated stimuli
    # epoch_regex = "^STIM_"
    # rec['resp'] = rec['resp'].extract_channels(A1_units)
    # rec['resp'] = rec['resp'].rasterize()
    # rec['psth'] = preprocessing.generate_average_sig(rec['resp'], 'psth', epoch_regex=epoch_regex)

    # # generate quick and dirty raw xy histograms
    # x = rec['dlc'][2]
    # y = rec['dlc'][3]
    # pixel_bin = 20
    # xbins = int(np.floor(np.nanmax(x)/pixel_bin))
    # ybins = int(np.floor(np.nanmax(y)/pixel_bin))
    # # initialize hist matrix
    # spike_hist = np.empty((xbins, ybins, len(x)))
    # spike_hist[:] = np.nan
    # occupation_hist = np.empty((xbins, ybins, len(x)))
    # occupation_hist[:] = np.nan
    # all_neuron_maps = []
    # all_neuron_half1 = []
    # all_neuron_half2 = []
    # for s in range(len(rec['resp'].chans)):
    #     rast = rec['resp'][s]
    #     cellid = rec['resp'].chans[s]
    #     for i in range(len(rast)):
    #         try:
    #             x_index = int(np.floor(x[i]/pixel_bin))
    #             y_index = int(np.floor(y[i]/pixel_bin))
    #             spike_hist[x_index, y_index, i] = rast[i]
    #             occupation_hist[x_index, y_index, i] = 1/rasterfs
    #         except:
    #             continue
    #
    #     spike_map = np.nansum(spike_hist, axis=2)
    #     occupation_map = np.nansum(occupation_hist, axis=2)
    #     rate_map = spike_map/occupation_map
    #     all_neuron_maps.append(rate_map)
    #     half_bin = round(len(spike_hist[0, 0, :])/2)
    #     spike_map_half1 = np.nansum(spike_hist[:, :, :half_bin], axis=2)/np.nansum(occupation_hist[:, :, :half_bin], axis=2)
    #     all_neuron_half1.append(spike_map_half1)
    #     spike_map_half2 = np.nansum(spike_hist[:, :, half_bin:], axis=2)/np.nansum(occupation_hist[:, :, half_bin:], axis=2)
    #     all_neuron_half2.append(spike_map_half2)
    #
    # #plot binned firing rate histograms for full session
    # f = plt.figure(figsize=(4, 10))
    # for i in range(10):
    #     ax = plt.subplot(10,3,i*3+1)
    #     ax.imshow(all_neuron_half1[i].T, origin='lower')
    #     ax.set_title("first half \n" + rec['resp'].chans[i])
    #
    #     ax = plt.subplot(10,3,i*3+2)
    #     ax.imshow(all_neuron_half2[i].T, origin='lower')
    #     ax.set_title("second half \n" + rec['resp'].chans[i])
    #
    #     ax = plt.subplot(10,3,i*3+3)
    #     ax.imshow(all_neuron_maps[i].T, origin='lower')
    #     ax.set_title("total \n" + rec['resp'].chans[i])
    # plt.tight_layout()


    # use diff to predict dist
    rec['diff'] = rec['resp']._modified_copy(data = rec['resp']._data-rec['psth']._data)


    epochs = rec.epochs
    is_stim = epochs['name'].str.startswith('STIM')
    stim_epoch_list = epochs[is_stim]['name'].tolist()
    stim_epochs = list(set(stim_epoch_list))
    stim_epochs.sort()
    # how many times are stimuli repeated
    ep_counts = ep.epoch_occurrences(epochs, regex=epoch_regex)
    print(ep_counts)
    e = ep_counts.idxmax()
    cellids = rec['resp'].chans

    signames = ['resp']
    for signame in signames:
        #est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
        jackknifes = 1
        for jack in range(jackknifes):
            # ask about difference in mask by epoch vs mask by time
            if signame == 'resp':
                est = rec.jackknife_mask_by_time(jackknifes,jack,invert=False)
                val = rec.jackknife_mask_by_time(jackknifes, jack, invert=True)
            else:
                est = rec.jackknife_mask_by_epoch(jackknifes,jack,'TRIAL',invert=False)
                val = rec.jackknife_mask_by_epoch(jackknifes,jack,'TRIAL',invert=True)

            # e = stim_epochs[0]
            p = val['psth'].extract_epoch(e)
            r = val['resp'].extract_epoch(e)
            d = val['diff'].extract_epoch(e)

            # f,ax=plt.subplots(3,1)
            # # cid=96
            # cid = -1
            # ax[0].imshow(p[:,cid,:], aspect='auto', interpolation='none')
            # ax[1].imshow(r[:,cid,:], aspect='auto', interpolation='none')
            # ax[2].imshow(d[:,cid,:], aspect='auto', interpolation='none')
            #
            # ax[0].set_title(f'psth {cellids[cid]} - {e}')
            # ax[1].set_title(f'single trial {cellids[cid]} - {e}')
            # ax[2].set_title(f'diff {cellids[cid]} - {e}')
            # plt.tight_layout()


            # For a model that uses multiple inputs, we package the input data into
            # a dictionary. This way, layers can specify which inputs they need using the
            # assigned keys.

            val = val.apply_mask()
            est = est.apply_mask()

            # change 'diff' to 'resp' for non-PSTH subbed data
            input_sig_name = signame # 'resp'
            target_sig_names = ['dlc']
            target_chans = ['front_x', 'front_y']
            for target_sig_name in target_sig_names:
                # input = est[input_sig_name]._data.T
                # target = est[target_sig_name]._data.T
                # test_input = val[input_sig_name]._data.T
                # test_target = val[target_sig_name]._data.T
                for i, target_chan in enumerate(est[target_sig_name].chans):
                    if target_chan in target_chans:
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

                        layers = [
                            WeightChannels(shape=(cellcount,1,5)),
                            FIR(shape=(5,1,5)),
                            ReLU(shape=(5,), no_shift=False, no_offset=False, no_gain=True),
                            WeightChannels(shape=(5,5)),
                            ReLU(shape=(5,), no_shift=False, no_offset=False, no_gain=True),
                            WeightChannels(shape=(5,1)),
                            DoubleExponential(shape=(dimcount,)),
                        ]
                            #LevelShift(shape=(dimcount,))

                        model = Model(layers=layers)
                        model = model.sample_from_priors()
                        model = model.sample_from_priors()

                        tolerance = 1e-5
                        max_iter = 200

                        use_tf = True
                        if use_tf:
                            input_ = np.expand_dims(input, axis=0)
                            test_input_ = np.expand_dims(test_input, axis=0)
                            target_ = np.expand_dims(target, axis=0)
                            test_target_ = np.expand_dims(test_target, axis=0)

                            fitter_options = {'cost_function': 'nmse', 'early_stopping_delay': 10,
                                              'early_stopping_patience': 5,
                                              'early_stopping_tolerance': tolerance,
                                              'learning_rate': 1e-2, 'epochs': max_iter,
                                              }
                            model = model.fit(input=input_, target=target_, backend='tf',
                                              fitter_options=fitter_options, batch_size=None)

                            prediction = model.predict(input_, batch_size=None)[0,:,:]
                            test_prediction = model.predict(test_input_, batch_size=None)[0,:,:]

                            # f,ax=plt.subplots(1,2)
                            # ax[0].scatter(prediction*s+m,target*s+m,s=1)
                            ccf = np.corrcoef(prediction[:,0],target[:,0])[0,1]
                            # ax[0].set_title(f'targetchan={rec[target_sig_name].chans[targetchan]}, cc={ccf:.3f}')
                            # ax[1].scatter(test_prediction*s+m,test_target*s+m,s=1)
                            cct = np.corrcoef(test_prediction[:,0],test_target_[:,0])[0,1]
                            # ax[1].set_title(f'TEST targetchan={rec[target_sig_name].chans[targetchan]}, cc={cct:.3f}')
                            d = {
                                'siteid': [siteid],
                                'input signal': [signame],
                                'test input': [test_input_],
                                'test target': [test_target_],
                                'test prediction': [test_prediction],
                                'target signal': [target_sig_name],
                                'target channel': [target_chan],
                                'jackknife': [jack],
                                'test cc': [cct],
                                'model': [model]
                            }
                            # decoded_df.append([siteid, target_sig_name, target_chan, targetchan, ccf, cct])
                            # decoded_dict['target signal'] = target_sig_name
                            # decoded_dict['target'] = target_chan
                            # decoded_dict['target idx'] = targetchan
                            # decoded_dict['cc_fit'] = ccf
                            # decoded_dict['cc_test'] = cct
                            tmpdf = pd.DataFrame.from_dict(data=d)
                            decoded_dfs.append(tmpdf)

                            # f,ax=plt.subplots(1,1)
                            # ax.scatter(test_prediction,test_target,s=1)
                            # cc = np.corrcoef(test_prediction[:,0],test_target[:,0])[0,1]
                            # ax.set_title(f'targetchan={rec["dist"].chans[targetchan]}, cc={cc:.3f}')
                        else:
                            fitter_options = {'cost_function': 'nmse', 'options': {'ftol': tolerance, 'gtol': tolerance/10, 'maxiter': max_iter}}

                            #model.layers[-1].skip_nonlinearity()
                            #model=model.fit(input=input, target=target, fitter_options=fitter_options)

                            #model.layers[-1].unskip_nonlinearity()

                            model =model.fit(input=input, target=target, fitter_options=fitter_options)

                            prediction = model.predict(input)


                            f,ax=plt.subplots(1,1)
                            ax.scatter(prediction,target,s=1)
                            cc = np.corrcoef(prediction[:,0],target[:,0])[0,1]
                            ax.set_title(f'targetchan={rec["dist"].chans[targetchan]}, cc={cc:.3f}')
                    else:
                        continue
decoded_df = pd.concat(decoded_dfs)


# decoded_df.to_pickle('/auto/users/wingertj/data/decoding_df.pkl')

bp = []
