import nems_lbhb.projects.olp.OLP_plot_helpers as opl
import nems.db as nd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
import nems_lbhb.projects.olp.OLP_Binaural_plot as obip



sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
fs, paths = 100, None

fit = False
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_add_spont.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_resp.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_corr.h5'  #New for testig corr

batch = 328 #Ferret A1
batch = 329 #Ferret PEG
batch = 333 #Marmoset (HOD+TBR)
batch = 340 #All ferret OLP
batch = 339 #Binaural ferret OLP

if fit == True:
    cell_df = nd.get_batch_cells(batch)
    cell_list = cell_df['cellid'].tolist()
    cell_list = ohel.manual_fix_units(cell_list) #So far only useful for two TBR cells
    cell_list = [cc for cc in cell_list if cc[:6] == "CLT007"]
    cell_list = cell_list[:5]
    # cellid, parmfile = 'CLT007a-009-2', None

    metrics=[]
    for cellid in cell_list:
        cell_metric = ofit.calc_psth_metrics(batch, cellid)
        cell_metric.insert(loc=0, column='cellid', value=cellid)
        print(f"Adding cellid {cellid}.")
        metrics.append(cell_metric)

    df = pd.concat(metrics)
    df.reset_index()

    os.makedirs(os.path.dirname(OLP_cell_metrics_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df_store=copy.deepcopy(df)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df=store['df']
    store.close()

# df = df.query("cellid == 'CLT008a-006-2'")
# df = df.query("cellid == 'CLT007a-002-1'")

weights = False
OLP_weights_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full.h5' #weight + corr
if weights == True:
    weight_df = ofit.fit_weights(df, batch, fs)

    os.makedirs(os.path.dirname(OLP_weights_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_weights_db_path)
    df_store=copy.deepcopy(weight_df)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_weights_db_path)
    weight_df=store['df']
    store.close()


sound_stats = False
OLP_stats_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_sound_stats.h5' #weight + corr
if sound_stats == True:
    sound_df = ohel.get_sound_statistics(weight_df, plot=False)
    weight_df = ohel.add_sound_stats(weight_df, sound_df)

    os.makedirs(os.path.dirname(OLP_stats_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_stats_db_path)
    df_store=copy.deepcopy(weight_df)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_stats_db_path)
    weight_df=store['df']
    store.close()

from nems_lbhb.baphy_experiment import BAPHYExperiment
import copy
import nems.epoch as ep
import nems.preprocessing as preproc
import SPO_helpers as sp
import glob
from nems.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import re
import itertools



def get_sep_stim_names(stim_name):
    seps = [m.start() for m in re.finditer('_(\d|n)', stim_name)]
    if len(seps) < 2 or len(seps) > 2:
        return None
    else:
        return [stim_name[seps[0] + 1:seps[1]], stim_name[seps[1] + 1:]]

weight_list = []
batch = 339
fs = 100
lfreq, hfreq, bins = 100, 24000, 48
threshold = 0.1
cell_df = nd.get_batch_cells(batch)
cell_list = cell_df['cellid'].tolist()
cell_list = ohel.manual_fix_units(cell_list) #So far only useful for two TBR cells

fit_epochs = ['10', '01', '20', '02', '11', '12', '21', '22']
loader = 'env100'
modelspecs_dir = '/auto/users/luke/Code/nems/modelspecs'

for cellid in cell_list:
    loadkey = 'ns.fs100'

    manager = BAPHYExperiment(cellid=cellid, batch=batch)

    options = {'rasterfs': 100,
               'stim': False,
               'resp': True}
    rec = manager.get_recording(**options)

    #GET sound envelopes and get the indices for chopping?
    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    FG_folder, fgidx = ref_handle['FG_Folder'], list(set(ref_handle['Foreground']))
    fgidx.sort(key=int)

    idxstr = [str(ff).zfill(2) for ff in fgidx]

    fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'{FG_folder}/{ff}*.wav'))[0] for ff in idxstr]
    fgname = [ff.split('/')[-1].split('.')[0].replace(' ', '') for ff in fg_paths]
    ep_fg = [f"STIM_null_{ff}" for ff in fgname]

    env_cuts = {}
    for nm, pth in zip(fgname, fg_paths):
        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        env = np.nanmean(spec, axis=0)
        cutoff = np.max(env) * threshold

        aboves = np.squeeze(np.argwhere(env >= cutoff))
        belows = np.squeeze(np.argwhere(env < cutoff))

        env_cuts[nm] = (aboves, belows)

        # f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
        # ax[0].plot(env)
        # ax[0].hlines(cutoff, 0, 100, ls=':'
        # ax[0].set_title(f"{nm}")
        # ax[1].plot(env[aboves])
        # ax[2].plot(env[belows])


    rec['resp'].fs = fs
    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())

    _, SR, _ = ohel.remove_spont_rate_std(resp)

    stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_')

    val = rec.copy()
    val['resp'] = val['resp'].rasterize()
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    est_sub = None

    df0 = val['resp'].epochs.copy()
    df2 = val['resp'].epochs.copy()
    df0['name'] = df0['name'].apply(ohel.label_ep_type)
    df0 = df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])

    val['resp'].epochs = df3
    val_sub = copy.deepcopy(val)
    val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)

    val = val_sub
    fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR / rec['resp'].fs)
    val['resp'] = val['resp'].transform(fn)

    print('calc weights')

    signame = 'resp'
    do_plot = False
    find_mse_confidence = True
    get_nrmse_fn = True

    #where twostims fit actually begins
    epcs = val.epochs[val.epochs['name'].str.count('-0-1') >= 1].copy()
    sepname = epcs['name'].apply(get_sep_stim_names)
    epcs['nameA'] = [x[0] for x in sepname.values]
    epcs['nameB'] = [x[1] for x in sepname.values]

    # epochs with two sounds in them
    epcs_twostim = epcs[epcs['name'].str.count('-0-1') == 2].copy()

    A, B, AB, sepnames = ([], [], [], [])  # re-defining sepname
    for i in range(len(epcs_twostim)):
        if any((epcs['nameA'] == epcs_twostim.iloc[i].nameA) & (epcs['nameB'] == 'null')) \
                and any((epcs['nameA'] == 'null') & (epcs['nameB'] == epcs_twostim.iloc[i].nameB)):
            A.append('STIM_' + epcs_twostim.iloc[i].nameA + '_null')
            B.append('STIM_null_' + epcs_twostim.iloc[i].nameB)
            AB.append(epcs_twostim['name'].iloc[i])
            sepnames.append(sepname.iloc[i])

    #Calculate weights
    weights = np.zeros((2, len(AB)))
    weights_h = np.zeros((2, len(AB)))
    weights_l = np.zeros((2, len(AB)))
    weights_lp = np.zeros((2, len(AB)))
    Efit = np.zeros((5,len(AB)))
    nMSE = np.zeros(len(AB))
    nf = np.zeros(len(AB))
    r = np.zeros(len(AB))
    get_error=[]

    for i in range(len(AB)):
        names=[[A[i]],[B[i]],[AB[i]]]
        weights[:,i], weights_h[:,i], weights_l[:,i], weights_lp[:,i], \
        Efit[:,i], nMSE[i], nf[i], get_nrmse, r[i], ge = \
            calc_psth_weights_of_model_responses_list(
                val,names,signame,do_plot=None,find_mse_confidence=None,
                get_nrmse_fn=None, window=None, cuts=env_cuts)
        get_error.append(ge)
        if do_plot and find_mse_confidence:
            plt.title('{}, signame={}'.format(AB[i],signame))

    window=None

    weight_df = pd.DataFrame(
        [epcs_twostim['nameA'].values, epcs_twostim['nameB'].values,
         weights[0, :], weights[1, :],
         weights_h[0, :], weights_h[1, :],
         weights_l[0, :], weights_l[1, :],
         weights_lp[0, :], weights_lp[1, :],
         Efit, nMSE, nf, r,
         get_error])
    weight_df = weight_df.T
    weight_df.columns = ['namesA', 'namesB', 'weightsA', 'weightsB', 'weightsAh', 'weightsBh',
                         'weightsAl', 'weightsBl', 'weightsAlp', 'weightsBlp',
                         'Efit', 'nMSE', 'nf', 'r', 'get_error']
    cols = ['namesA', 'namesB', 'weightsA', 'weightsB', 'nMSE']
    print(weight_df[cols])

    weight_df = weight_df.astype({'weightsA': float, 'weightsB': float,
                                  'weightsAh': float, 'weightsBh': float,
                                  'weightsAl': float, 'weightsBl': float,
                                  'weightsAlp': float, 'weightsBlp': float,
                                  'nMSE': float, 'nf': float, 'r': float})
    val_range = lambda x: max(x) - min(x)
    val_range.__name__ = 'range'
    MI = lambda x: np.mean([np.abs(np.diff(pair)) / np.abs(np.sum(pair)) for pair in itertools.combinations(x, 2)])
    MI.__name__ = 'meanMI'
    MIall = lambda x: (max(x) - min(x)) / np.abs(max(x) + min(x))
    MIall.__name__ = 'meanMIall'

    fns = ['count', val_range, 'std', MI, MIall, 'sum']
    WeightAgroups = weight_df.groupby('namesA')[['weightsA', 'weightsB']].agg(fns)
    WeightAgroups = WeightAgroups[WeightAgroups['weightsA']['count'] > 1]
    WeightBgroups = weight_df.groupby('namesB')[['weightsA', 'weightsB']].agg(fns)
    WeightBgroups = WeightBgroups[WeightBgroups['weightsA']['count'] > 1]

    cols = ['count', 'range', 'meanMIall']
    print('Grouped by A, A weight metrics:')
    print(WeightAgroups['weightsA'][cols])
    print('Grouped by A, B weight metrics:')
    print(WeightAgroups['weightsB'][cols])
    print('Grouped by B, A weight metrics:')
    print(WeightBgroups['weightsA'][cols])
    print('Grouped by B, B weight metrics:')
    print(WeightBgroups['weightsB'][cols])

    names = AB
    namesA = A
    namesB = B
    D = locals()
    D = {k: D[k] for k in (
    'weights', 'Efit', 'nMSE', 'nf', 'get_nrmse', 'r', 'names', 'namesA', 'namesB', 'weight_df', 'WeightAgroups',
    'WeightBgroups')}

    d = {k + 'R': v for k, v in D.items()}

    weightDF = d['weight_dfR']
    weightDF.insert(loc=0, column='cellid', value=cellid)

    weight_list.append(weightDF)
    

   def drop_get_error(row):
        row['weight_dfR'] = row['weight_dfR'].copy().drop(columns=['get_error', 'Efit'])
        return row

    df0 = df0.copy().drop(columns='get_nrmseR')
    df0 = df0.apply(drop_get_error, axis=1)

    weight_df = pd.concat(df0['weight_dfR'].values, keys=df0.cellid).reset_index(). \
        drop(columns='level_1')
    ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df.namesA, weight_df.namesB)]
    weight_df = weight_df.drop(columns=['namesA', 'namesB'])
    weight_df['epoch'] = ep_names

    weights_df = pd.merge(right=weight_df, left=df, on=['cellid', 'epoch'])
    if df.shape[0] != weights_df.shape[0] or weight_df.shape[0] != weights_df.shape[0]:
        raise ValueError("Resulting weights_df does not match length of parts, some epochs were dropped.")


# return weights_C, Efit_C, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I


def calc_psth_weights_of_model_responses_list(val, names, signame='pred', do_plot=False, find_mse_confidence=True,
                                              get_nrmse_fn=True, window=None, cuts=None):
    # prestimtime=0.5;#1;
    PSS = val[signame].epochs[val[signame].epochs['name'] == 'PreStimSilence'].iloc[0]
    prestimtime = PSS['end'] - PSS['start']
    REF = val[signame].epochs[val[signame].epochs['name'] == 'REFERENCE'].iloc[0]
    total_duration = REF['end'] - REF['start']
    POSS = val[signame].epochs[val[signame].epochs['name'] == 'PostStimSilence'].iloc[0]
    poststimtime = POSS['end'] - POSS['start']
    duration = total_duration - prestimtime - poststimtime

    post_duration_pad = .5  # Include stim much post-stim time in weight calcs
    time = np.arange(0, val[signame].extract_epoch(names[0][0]).shape[-1]) / val[signame].fs - prestimtime

    if cuts is None:
        subsets = 1
        weights_h, weights_l, weights_lp = None, None, None
    else:
        subsets = 4 #if I pass my fg envelope filter it'll fit normal, high power, low power
                    # plus poststim and low power without post stim
        #get which cut set to use
        fs = val['resp'].fs
        Fg = names[1][0].split('_')[2].split('-')[0]
        high_power, low_power = cuts[Fg]
        binstim = int((duration+poststimtime) * fs)
        maxbin = np.max([np.max(high_power), np.max(low_power)])
        post_bins = np.asarray(range(maxbin+1,binstim))
        low_wpost = np.concatenate((low_power, post_bins))
        postbin = int(poststimtime * fs)

        high_power_pad, low_power_pad = high_power + postbin, low_power + postbin
        lwpost_pad, full = low_wpost + postbin, np.asarray(range(0,binstim)) + postbin
        filters = [full, high_power_pad, low_power_pad, lwpost_pad]

    for subset in range(subsets):
        sig1 = np.concatenate([val[signame].extract_epoch(n).squeeze()[filters[subset]] for n in names[0]])
        sig2 = np.concatenate([val[signame].extract_epoch(n).squeeze()[filters[subset]] for n in names[1]])
        # sig_SR=np.ones(sig1.shape)
        sigO = np.concatenate([val[signame].extract_epoch(n).squeeze()[filters[subset]] for n in names[2]])

        # fsigs=np.vstack((sig1,sig2,sig_SR)).T
        fsigs = np.vstack((sig1, sig2)).T
        ff = np.all(np.isfinite(fsigs), axis=1) & np.isfinite(sigO)
        close_to_zero = np.array([np.allclose(fsigs[ff, i], 0, atol=1e-17) for i in (0, 1)])
        if all(close_to_zero):
            # Both input signals have all their values close to 0. Set weights to 0.
            weights = np.zeros(2)
            rank = 1
        elif any(close_to_zero):
            weights_, residual_sum, rank, singular_values = np.linalg.lstsq(np.expand_dims(fsigs[ff, ~close_to_zero], 1),
                                                                            sigO[ff], rcond=None)
            weights = np.zeros(2)
            weights[~close_to_zero] = weights_
        else:
            weights, residual_sum, rank, singular_values = np.linalg.lstsq(fsigs[ff, :], sigO[ff], rcond=None)
            # residuals = ((sigO[ff]-(fsigs[ff,:]*weights).sum(axis=1))**2).sum()

        if subset == 0:
            # calc CC between weight model and actual response
            pred = np.dot(weights, fsigs[ff, :].T)
            cc = np.corrcoef(pred, sigO[ff])
            r_weight_model = cc[0, 1]

            # norm_factor = np.std(sigO[ff])
            norm_factor = np.mean(sigO[ff] ** 2)

        if subset == 0:
            weights_f = weights
        elif subset == 1:
            weights_h = weights
        elif subset == 2:
            weights_l = weights
        elif subset == 3:
            weights_lp = weights
    #
    # if rank == 1:
    #     min_nMSE = 1
    #     min_nRMSE = 1
    # else:
    #     # min_nrmse = np.sqrt(residual_sum[0]/ff.sum())/norm_factor
    #     pred = np.dot(weights, fsigs[ff, :].T)
    #     min_nRMSE = np.sqrt(((sigO[ff] - pred) ** 2).mean()) / np.sqrt(
    #         norm_factor)  # minimim normalized root mean squared error
    #     min_nMSE = ((sigO[ff] - pred) ** 2).mean() / norm_factor  # minimim normalized mean squared error
    #
    # # create NMSE caclulator for later
    # if get_nrmse_fn:
    #     def get_nrmse(weights=weights):
    #         pred = np.dot(weights, fsigs[ff, :].T)
    #         nrmse = np.sqrt(((pred - sigO[ff]) ** 2).mean(axis=-1)) / norm_factor
    #         return nrmse
    # else:
    #     get_nrmse = np.nan
    #
    # def get_error(weights=weights, get_what='error'):
    #
    #     if get_what == 'sigA':
    #         return fsigs[ff, 0]
    #     elif get_what == 'sigB':
    #         return fsigs[ff, 1]
    #     elif get_what == 'sigAB':
    #         return sigO[ff]
    #     elif get_what == 'pred':
    #         return np.dot(weights, fsigs[ff, :].T)
    #     elif get_what == 'error':
    #         pred = np.dot(weights, fsigs[ff, :].T)
    #         return pred - sigO[ff]
    #     else:
    #         raise RuntimeError('Invalid get_what parameter')

    # if not find_mse_confidence:
    #     weights[close_to_zero] = np.nan
    #     return weights, np.nan, min_nMSE, norm_factor, get_nrmse, r_weight_model, get_error

    # #    sigF=weights[0]*sig1 + weights[1]*sig2 + weights[2]
    # #    plt.figure();
    # #    plt.plot(np.vstack((sig1,sig2,sigO,sigF)).T)
    # #    wA_ = np.linspace(-2, 4, 100)
    # #    wB_ = np.linspace(-2, 4, 100)
    # #    wA, wB = np.meshgrid(wA_,wB_)
    # #    w=np.vstack((wA.flatten(),wB.flatten())).T
    # #    sigF2=np.dot(w,fsigs[ff,:].T)
    # #    mse = ((sigF2-sigO[ff].T) ** 2).mean(axis=1)
    # #    mse = np.reshape(mse,(len(wA_),len(wB_)))
    # #    plt.figure();plt.imshow(mse,interpolation='none',extent=[wA_[0],wA_[-1],wB_[0],wB_[-1]],origin='lower',vmax=.02,cmap='viridis_r');plt.colorbar()
    #
    # def calc_nrmse_matrix(margin, N=60, threshtype='ReChance'):
    #     # wsearcha=(-2, 4, 100)
    #     # wsearchb=wsearcha
    #     # margin=6
    #     if not hasattr(margin, "__len__"):
    #         margin = np.float(margin) * np.ones(2)
    #     wA_ = np.hstack((np.linspace(weights[0] - margin[0], weights[0], N),
    #                      (np.linspace(weights[0], weights[0] + margin[0], N)[1:])))
    #     wB_ = np.hstack((np.linspace(weights[1] - margin[1], weights[1], N),
    #                      (np.linspace(weights[1], weights[1] + margin[1], N)[1:])))
    #     wA, wB = np.meshgrid(wA_, wB_)
    #     w = np.stack((wA, wB), axis=2)
    #     nrmse = get_nrmse(w)
    #     # range_=mse.max()-mse.min()
    #     if threshtype == 'Absolute':
    #         thresh = nrmse.min() * np.array((1.4, 1.6))
    #         thresh = nrmse.min() * np.array((1.02, 1.04))
    #         As = wA[(nrmse < thresh[1]) & (nrmse > thresh[0])]
    #         Bs = wB[(nrmse < thresh[1]) & (nrmse > thresh[0])]
    #     elif threshtype == 'ReChance':
    #         thresh = 1 - (1 - nrmse.min()) * np.array((.952, .948))
    #         As = wA[(nrmse < thresh[1]) & (nrmse > thresh[0])]
    #         Bs = wB[(nrmse < thresh[1]) & (nrmse > thresh[0])]
    #     return nrmse, As, Bs, wA_, wB_
    #
    # if min_nRMSE < 1:
    #     this_threshtype = 'ReChance'
    # else:
    #     this_threshtype = 'Absolute'
    # margin = 6
    # As = np.zeros(0)
    # Bs = np.zeros(0)
    # attempt = 0
    # did_estimate = False
    # while len(As) < 20:
    #     attempt += 1
    #     if (attempt > 1) and (len(As) > 0) and (len(As) > 2) and (not did_estimate):
    #         margin = np.float(margin) * np.ones(2)
    #         m = np.abs(weights[0] - As).max() * 3
    #         if m == 0:
    #             margin[0] = margin[0] / 2
    #         else:
    #             margin[0] = m
    #
    #         m = np.abs(weights[1] - Bs).max() * 3
    #         if m == 0:
    #             margin[1] = margin[1] / 2
    #         else:
    #             margin[1] = m
    #         did_estimate = True
    #     elif attempt > 1:
    #         margin = margin / 2
    #     if attempt > 1:
    #         print('Attempt {}, margin = {}'.format(attempt, margin))
    #     nrmse, As, Bs, wA_, wB_ = calc_nrmse_matrix(margin, threshtype=this_threshtype)
    #
    #     if attempt == 8:
    #         print('Too many attempts, break')
    #         break
    #
    # try:
    #     efit = fE.fitEllipse(As, Bs)
    #     center = fE.ellipse_center(efit)
    #     phi = fE.ellipse_angle_of_rotation(efit)
    #     axes = fE.ellipse_axis_length(efit)
    #
    #     epars = np.hstack((center, axes, phi))
    # except:
    #     print('Error fitting ellipse: {}'.format(sys.exc_info()[0]))
    #     print(sys.exc_info()[0])
    #     epars = np.full([5], np.nan)
    # #    idxA = (np.abs(wA_ - weights[0])).argmin()
    # #    idxB = (np.abs(wB_ - weights[1])).argmin()
    # if do_plot:
    #     plt.figure();
    #     plt.imshow(nrmse, interpolation='none', extent=[wA_[0], wA_[-1], wB_[0], wB_[-1]], origin='lower',
    #                cmap='viridis_r');
    #     plt.colorbar()
    #     ph = plt.plot(weights[0], weights[1], Color='k', Marker='.')
    #     plt.plot(As, Bs, 'r.')
    #
    #     if not np.isnan(epars).any():
    #         a, b = axes
    #         R = np.arange(0, 2 * np.pi, 0.01)
    #         xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    #         yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
    #         plt.plot(xx, yy, color='k')
    #
    # #    plt.figure();plt.plot(get_nrmse(weights=(xx,yy)))
    # #    plt.figure();plt.plot(get_nrmse(weights=(As,Bs)))
    # weights[close_to_zero] = np.nan



    # return weights_f, weights_h, weights_l, weights_lp,\
    #        epars, nrmse.min(), norm_factor, get_nrmse, r_weight_model, get_error
    return weights_f, weights_h, weights_l, weights_lp,\
               np.nan, np.nan, norm_factor, np.nan, r_weight_model, np.nan



#add mod spec to sound_df
mods = np.empty((sound_df.iloc[0].spec.shape[0], sound_df.iloc[0].spec.shape[1],
                 len(sound_df)))
mods[:] = np.NaN
mod_list = []
for cnt, ii in enumerate(sound_df.name):
    row = sound_df.loc[sound_df.name==ii]
    spec = row['spec'].values[0]
    mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))
    mods[:, :, cnt] = mod
    mod_list.append(mod)
avmod = np.nanmean(mods, axis=2)
norm_list = [aa - avmod for aa in mod_list]
avmod = avmod[:,:,np.newaxis]
normmod = mods - avmod
clow, chigh = np.min(normmod), np.max(normmod)
sound_df['modspec'] = mod_list
sound_df['normmod'] = norm_list
# selfsounds['normmod'] = norm_list

trimspec = [aa[24:, 30:69] for aa in sound_df['modspec']]
negs = [aa[:, :20] for aa in trimspec]
negs = [aa[:, ::-1] for aa in negs]
poss = [aa[:, -20:] for aa in trimspec]
trims = [(nn + pp) /2 for (nn, pp) in zip(negs, poss)]
sound_df['trimspec'] = trims

ots = [np.nanmean(aa, axis=0) for aa in trims]
ofs = [np.nanmean(aa, axis=1) for aa in trims]

wt2 = wt[50:70]
wf2 = wf[24:]

cumwt = [np.cumsum(aa)/np.sum(aa) for aa in ots]
bigt = [np.max(aa) for aa in cumwt]
freq50t = [wt2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumwt, bigt)]

cumft = [np.cumsum(aa)/np.sum(aa) for aa in ofs]
bigf = [np.max(aa) for aa in cumft]
freq50f = [wf2[np.abs(cc - (bb * 0.5)).argmin()] for (cc, bb) in zip(cumft, bigf)]

sound_df['cumwt'], sound_df['cumft'] = cumwt, cumft
sound_df['t50'], sound_df['f50'] = freq50t, freq50f
sound_df['meanT'], sound_df['meanF'] = ots, ofs









#plots a bunch of wt and wf lines with average, cumsum, and 50%
f, axes = plt.subplots(2, 3, figsize=(12,7))
ax = axes.ravel()
for aa in ots[:20]:
    ax[0].plot(wt2, aa, color='deepskyblue')
for aa in ots[20:]:
    ax[0].plot(wt2, aa, color='yellowgreen')
ax[0].set_xlabel('wt (Hz)', fontweight='bold', fontsize=8)
ax[0].set_ylabel('Average', fontweight='bold', fontsize=8)

for aa in cumwt[:20]:
    ax[1].plot(aa, color='deepskyblue')
for aa in cumwt[20:]:
    ax[1].plot(aa, color='yellowgreen')
ax[1].set_ylabel('Cumulative Sum', fontweight='bold', fontsize=8)
ax[1].set_xlabel('wt (Hz)', fontweight='bold', fontsize=8)

bgs, fgs = np.nanmean(freq50t[:20]), np.nanmean(freq50t[20:])
ax[2].boxplot([freq50t[:20], freq50t[20:]], labels=['BG','FG'])
ax[2].set_ylabel('Median', fontweight='bold', fontsize=8)

# f, ax = plt.subplots(1, 3, figsize=(12,5))
for aa in ofs[:20]:
    ax[3].plot(wf2, aa, color='deepskyblue')
for aa in ofs[20:]:
    ax[3].plot(wf2, aa, color='yellowgreen')
ax[3].set_xlabel('wf (cycles/s)', fontweight='bold', fontsize=8)
ax[3].set_ylabel('Average', fontweight='bold', fontsize=8)

for aa in cumft[:20]:
    ax[4].plot(aa, color='deepskyblue')
for aa in cumft[20:]:
    ax[4].plot(aa, color='yellowgreen')
ax[4].set_ylabel('Cumulative Sum', fontweight='bold', fontsize=8)
ax[4].set_xlabel('wf (cycles/s)', fontweight='bold', fontsize=8)

bgs, fgs = np.nanmean(freq50f[:20]), np.nanmean(freq50f[20:])
ax[5].boxplot([freq50f[:20], freq50f[20:]], labels=['BG','FG'])
ax[5].set_ylabel('Median', fontweight='bold', fontsize=8)

##reproduce the jittered scatters with mod spec stuff
BGdf, FGdf = sound_df.loc[sound_df.type == 'BG'], sound_df.loc[sound_df.type == 'FG']
BGmerge, FGmerge = pd.DataFrame(), pd.DataFrame()
BGmerge['BG'] = [aa.replace(' ', '') for aa in BGdf.name]
BGmerge['BG_wt'] = BGdf.t50.tolist()
BGmerge['BG_wf'] = BGdf.f50.tolist()

FGmerge['FG'] = [aa.replace(' ', '') for aa in FGdf.name]
FGmerge['FG_wt'] = FGdf.t50.values.tolist()
FGmerge['FG_wf'] = FGdf.f50.tolist()

weight_df = pd.merge(right=BGmerge, left=weight_df, on=['BG'], validate='m:1')
weight_df = pd.merge(right=FGmerge, left=weight_df, on=['FG'], validate='m:1')




quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=0.03, quad_return=3)

quad = quad.copy()


quad['jitter_BG_wt'] = quad['BG_wt'] + np.random.normal(0, 0.075, len(quad))
quad['jitter_FG_wt'] = quad['FG_wt'] + np.random.normal(0, 0.075, len(quad))

quad['jitter_BG_wf'] = quad['BG_wf'] + np.random.normal(0, 0.0075, len(quad))
quad['jitter_FG_wf'] = quad['FG_wf'] + np.random.normal(0, 0.0075, len(quad))

##
from scipy import stats

f, ax = plt.subplots(1, 2, figsize=(10,5))
sb.scatterplot(x='jitter_BG_wt', y='weightsB', data=quad, ax=ax[0], s=3)
sb.scatterplot(x='jitter_FG_wt', y='weightsA', data=quad, ax=ax[0], s=3)
ax[0].set_xlabel('wt (Hz)', fontweight='bold', fontsize=8)
ax[0].set_ylabel('Weight', fontweight='bold', fontsize=8)

sb.scatterplot(x='jitter_BG_wf', y='weightsB', data=quad, ax=ax[1], s=3)
sb.scatterplot(x='jitter_FG_wf', y='weightsA', data=quad, ax=ax[1], s=3)
ax[1].set_xlabel('wf (cycles/s)', fontweight='bold', fontsize=8)
ax[1].set_ylabel('')
f.suptitle("how that sound effects the weight of others")

Y = np.concatenate((quad['weightsB'].values, quad['weightsA'].values))
X = np.concatenate((quad['BG_wt'].values, quad['FG_wt'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[0].get_xlim())
y = reg.slope*x + reg.intercept
ax[0].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[0].legend()

X = np.concatenate((quad['BG_wf'].values, quad['FG_wf'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[1].get_xlim())
y = reg.slope*x + reg.intercept
ax[1].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[1].legend()

##
f, ax = plt.subplots(1, 2, figsize=(10,5))
sb.scatterplot(x='jitter_BG_wt', y='weightsA', data=quad, ax=ax[0], s=3)
sb.scatterplot(x='jitter_FG_wt', y='weightsB', data=quad, ax=ax[0], s=3)
ax[0].set_xlabel('wt (Hz)', fontweight='bold', fontsize=8)
ax[0].set_ylabel('Weight', fontweight='bold', fontsize=8)

sb.scatterplot(x='jitter_BG_wf', y='weightsA', data=quad, ax=ax[1], s=3)
sb.scatterplot(x='jitter_FG_wf', y='weightsB', data=quad, ax=ax[1], s=3)
ax[1].set_xlabel('wf (cycles/s)', fontweight='bold', fontsize=8)
ax[1].set_ylabel('')
f.suptitle("how that sound is weighted")

Y = np.concatenate((quad['weightsA'].values, quad['weightsB'].values))
X = np.concatenate((quad['BG_wt'].values, quad['FG_wt'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[0].get_xlim())
y = reg.slope*x + reg.intercept
ax[0].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[0].legend()

X = np.concatenate((quad['BG_wf'].values, quad['FG_wf'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[1].get_xlim())
y = reg.slope*x + reg.intercept
ax[1].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[1].legend()


#barplot of these hopefully
fig, ax = plt.subplots(2, 1, figsize=(5, 8))

sb.barplot(x='name', y='t50', palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
           data=sound_df, ci=68, ax=ax[0], errwidth=1)
ax[0].set_xticklabels(sound_df.name, rotation=90, fontweight='bold', fontsize=7)
ax[0].set_ylabel('wt (Hz)', fontweight='bold', fontsize=12)
ax[0].spines['top'].set_visible(True), ax[0].spines['right'].set_visible(True)
ax[0].set(xlabel=None)

sb.barplot(x='name', y='f50',
           palette=["lightskyblue" if x == 'BG' else 'yellowgreen' for x in sound_df.type],
           data=sound_df, ax=ax[1])
ax[1].set_xticklabels(sound_df.name, rotation=90, fontweight='bold', fontsize=7)
ax[1].set_ylabel('wf (cycles/s)', fontweight='bold', fontsize=12)
ax[1].spines['top'].set_visible(True), ax[1].spines['right'].set_visible(True)
ax[1].set(xlabel=None)

fig.tight_layout()




##Prep for big mod spec figrues
quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=0.03, quad_return=3)
quad = quad.copy()

bgsub = quad[['BG', 'weightsA', 'weightsB']].copy()
fgsub = quad[['FG', 'weightsB', 'weightsA']].copy()

bgsub.rename(columns={'BG':'name', 'weightsA':'selfweight', 'weightsB':'effectweight'}, inplace=True)
fgsub.rename(columns={'FG':'name', 'weightsB':'selfweight', 'weightsA':'effectweight'}, inplace=True)
weights = pd.concat([bgsub, fgsub], axis=0)
means = weights.groupby('name').agg('mean')
selfy = weights.groupby('name').agg(selfweight=('selfweight',np.mean)).reset_index()
effect = weights.groupby('name').agg(effectweight=('effectweight',np.mean)).reset_index()
# selfsort = selfy.sort_values('selfweight').reset_index()
# effectsort = effect.sort_values('effectweight').reset_index()

fn = lambda x: x[2:].replace(' ', '')
sound_df['sound'] = sound_df.name.apply(fn)
sound_df.rename(columns={'name':'fullname', 'sound':'name'}, inplace=True)

selfsounds = selfy.merge(sound_df, on='name').sort_values('selfweight')
effectsounds = effect.merge(selfsounds, on='name').sort_values('effectweight', ascending=False)
self_sort = selfsounds.fullname
effect_sort = effectsounds.fullname

w = 13
h = 3
t = 1
tbins = 100
fbins = 48
lfreq = 100
hfreq = 24000

tmod = (tbins / t) / 2
xbound = tmod * 0.4
wt = np.fft.fftshift(np.fft.fftfreq(tbins, 1 / tbins))
wf = np.fft.fftshift(np.fft.fftfreq(fbins, 1 / 6))

##plot mod specs in order ascending of their weight
f, axes = plt.subplots(h*2, w, figsize=(18,8))
ax = axes.ravel()
AX = list(np.arange(0,13)) + list(np.arange(26,39)) + list(np.arange(52,65))

for aa, snd in zip(AX, self_sort):
    row = selfsounds.loc[selfsounds.fullname == snd]
    spec = row['spec'].values[0]
    mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))

    ax[aa].imshow(spec, aspect='auto', origin='lower')
    ax[aa+13].imshow(np.sqrt(mod), aspect='auto', origin='lower',
                 extent=(wt[0]+0.5, wt[-1]+0.5, wf[0], wf[-1]))
    ax[aa].set_yticks([]), ax[aa].set_xticks([])
    ax[aa+13].set_xlim(-xbound, xbound)
    ax[aa+13].set_ylim(0,np.max(wf))
    if aa == 0 or aa == 13 or aa == 26:
        ax[aa+13].set_ylabel("wf (cycles/s)", fontweight='bold', fontsize=6)
    if aa >= 52:
        ax[aa+13].set_xlabel("wt (Hz)", fontweight='bold', fontsize=6)
    ax[aa].set_title(f"{row['name'].values[0]}: {np.around(row['selfweight'].values[0], 3)}", fontweight='bold', fontsize=8)

##plot mod specs in order descending of the weight they cause in paired sound
f, axes = plt.subplots(h*2, w, figsize=(18,8))
ax = axes.ravel()
AX = list(np.arange(0,13)) + list(np.arange(26,39)) + list(np.arange(52,65))

for aa, snd in zip(AX, effect_sort):
    row = effectsounds.loc[effectsounds.fullname == snd]
    spec = row['spec'].values[0]
    mod = np.fft.fftshift(np.abs(np.fft.fft2(spec)))

    ax[aa].imshow(spec, aspect='auto', origin='lower')
    ax[aa+13].imshow(np.sqrt(mod), aspect='auto', origin='lower',
                 extent=(wt[0]+0.5, wt[-1]+0.5, wf[0], wf[-1]))
    ax[aa].set_yticks([]), ax[aa].set_xticks([])
    ax[aa+13].set_xlim(-xbound, xbound)
    ax[aa+13].set_ylim(0,np.max(wf))
    if aa == 0 or aa == 13 or aa == 26:
        ax[aa+13].set_ylabel("wf (cycles/s)", fontweight='bold', fontsize=6)
    if aa >= 52:
        ax[aa+13].set_xlabel("wt (Hz)", fontweight='bold', fontsize=6)
    ax[aa].set_title(f"{row['name'].values[0]}: {np.around(row['effectweight'].values[0], 3)}", fontweight='bold', fontsize=8)

##plot normed mod specs in order ascending of their weight
f, axes = plt.subplots(h*2, w, figsize=(18,8))
ax = axes.ravel()
AX = list(np.arange(0,13)) + list(np.arange(26,39)) + list(np.arange(52,65))

for aa, snd in zip(AX, self_sort):
    row = selfsounds.loc[selfsounds.fullname == snd]
    spec = row['spec'].values[0]
    mod = row['normmod'].values[0]

    ax[aa].imshow(spec, aspect='auto', origin='lower')
    ax[aa+13].imshow(mod, aspect='auto', origin='lower',
                 extent=(wt[0]+0.5, wt[-1]+0.5, wf[0], wf[-1]), vmin=clow, vmax=chigh)
    ax[aa].set_yticks([]), ax[aa].set_xticks([])
    ax[aa+13].set_xlim(-xbound, xbound)
    ax[aa+13].set_ylim(0,np.max(wf))
    if aa == 0 or aa == 13 or aa == 26:
        ax[aa+13].set_ylabel("wf (cycles/s)", fontweight='bold', fontsize=6)
    if aa >= 52:
        ax[aa+13].set_xlabel("wt (Hz)", fontweight='bold', fontsize=6)
    ax[aa].set_title(f"{row['name'].values[0]}: {np.around(row['selfweight'].values[0], 3)}", fontweight='bold', fontsize=8)



##plot single mod specs
obip.lot_mod_spec(2)




quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=0.03, quad_return=3)

quad = quad.copy()


quad['jitter_BG_Tstationary'] = quad['BG_Tstationary'] + np.random.normal(0, 0.2, len(quad))
quad['jitter_FG_Tstationary'] = quad['FG_Tstationary'] + np.random.normal(0, 0.2, len(quad))

quad['jitter_BG_bandwidth'] = quad['BG_bandwidth'] + np.random.normal(0, 0.02, len(quad))
quad['jitter_FG_bandwidth'] = quad['FG_bandwidth'] + np.random.normal(0, 0.02, len(quad))

quad['jitter_BG_Fstationary'] = quad['BG_Fstationary'] + np.random.normal(0, 0.2, len(quad))
quad['jitter_FG_Fstationary'] = quad['FG_Fstationary'] + np.random.normal(0, 0.2, len(quad))

from scipy import stats

f, ax = plt.subplots(1, 3, figsize=(12,5))
sb.scatterplot(x='jitter_BG_Tstationary', y='weightsB', data=quad, ax=ax[0], s=3)
sb.scatterplot(x='jitter_FG_Tstationary', y='weightsA', data=quad, ax=ax[0], s=3)
ax[0].set_xlabel('Non-stationariness', fontweight='bold', fontsize=8)
ax[0].set_ylabel('Weight', fontweight='bold', fontsize=8)

sb.scatterplot(x='jitter_BG_bandwidth', y='weightsB', data=quad, ax=ax[1], s=3)
sb.scatterplot(x='jitter_FG_bandwidth', y='weightsA', data=quad, ax=ax[1], s=3)
ax[1].set_xlabel('Bandwidth', fontweight='bold', fontsize=8)
ax[1].set_ylabel('')

sb.scatterplot(x='jitter_BG_Fstationary', y='weightsB', data=quad, ax=ax[2], s=3)
sb.scatterplot(x='jitter_FG_Fstationary', y='weightsA', data=quad, ax=ax[2], s=3)
ax[2].set_xlabel('Frequency Non-stationariness', fontweight='bold', fontsize=8)
ax[2].set_ylabel('')
f.suptitle("how that sound effects the weight of others")

Y = np.concatenate((quad['weightsB'].values, quad['weightsA'].values))
X = np.concatenate((quad['BG_Tstationary'].values, quad['FG_Tstationary'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[0].get_xlim())
y = reg.slope*x + reg.intercept
ax[0].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[0].legend()

X = np.concatenate((quad['BG_bandwidth'].values, quad['FG_bandwidth'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[1].get_xlim())
y = reg.slope*x + reg.intercept
ax[1].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[1].legend()

X = np.concatenate((quad['BG_Fstationary'].values, quad['FG_Fstationary'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[2].get_xlim())
y = reg.slope*x + reg.intercept
ax[2].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[2].legend()

f, ax = plt.subplots(1, 3, figsize=(12,5))
sb.scatterplot(x='jitter_BG_Tstationary', y='weightsA', data=quad, ax=ax[0], s=3)
sb.scatterplot(x='jitter_FG_Tstationary', y='weightsB', data=quad, ax=ax[0], s=3)
ax[0].set_xlabel('Non-stationariness', fontweight='bold', fontsize=8)
ax[0].set_ylabel('Weight', fontweight='bold', fontsize=8)

sb.scatterplot(x='jitter_BG_bandwidth', y='weightsA', data=quad, ax=ax[1], s=3)
sb.scatterplot(x='jitter_FG_bandwidth', y='weightsB', data=quad, ax=ax[1], s=3)
ax[1].set_xlabel('Bandwidth', fontweight='bold', fontsize=8)
ax[1].set_ylabel('')

sb.scatterplot(x='jitter_BG_Fstationary', y='weightsA', data=quad, ax=ax[2], s=3)
sb.scatterplot(x='jitter_FG_Fstationary', y='weightsB', data=quad, ax=ax[2], s=3)
ax[2].set_xlabel('Frequency Non-stationariness', fontweight='bold', fontsize=8)
ax[2].set_ylabel('')
f.suptitle("how that sound is weighted")

Y = np.concatenate((quad['weightsA'].values, quad['weightsB'].values))
X = np.concatenate((quad['BG_Tstationary'].values, quad['FG_Tstationary'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[0].get_xlim())
y = reg.slope*x + reg.intercept
ax[0].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[0].legend()

X = np.concatenate((quad['BG_bandwidth'].values, quad['FG_bandwidth'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[1].get_xlim())
y = reg.slope*x + reg.intercept
ax[1].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[1].legend()

X = np.concatenate((quad['BG_Fstationary'].values, quad['FG_Fstationary'].values))
reg = stats.linregress(X, Y)
x = np.asarray(ax[2].get_xlim())
y = reg.slope*x + reg.intercept
ax[2].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[2].legend()

#plot, with regression, FR v weighted FR
from scipy import stats
quad['FRbg*weightA'] = quad['bg_FR'] * quad['weightsA']
quad['FRfg*weightB'] = quad['fg_FR'] * quad['weightsB']

f, ax = plt.subplots(1, 2, figsize=(12,7), sharex=True, sharey=True)
sb.scatterplot(x='bg_FR', y='FRbg*weightA', data=quad, ax=ax[0], s=3, color='deepskyblue')
ax[0].set_aspect('equal')
ax[0].set_ylabel('Weighted FR', fontweight='bold', fontsize=10)
ax[0].set_xlabel('Background FR', fontweight='bold', fontsize=10)
ax[0].spines['top'].set_visible(True), ax[0].spines['right'].set_visible(True)

sb.scatterplot(x='fg_FR', y='FRfg*weightB', data=quad, ax=ax[1], s=3, color='yellowgreen')
ax[1].set_aspect('equal')
ax[1].set_ylabel('Weighted Foreground FR', fontweight='bold', fontsize=10)
ax[1].set_xlabel('Foreground FR', fontweight='bold', fontsize=10)
ax[1].spines['top'].set_visible(True), ax[1].spines['right'].set_visible(True)
xmin, xmax = ax[0].get_xlim()
ax[0].set_xlim(xmin-0.03, xmax-0.2)
ax[0].set_ylim(xmin-0.03, xmax-0.2)

X, Y = quad['bg_FR'], quad['FRbg*weightA']
reg = stats.linregress(X, Y)
x = np.asarray(ax[0].get_xlim())
y = reg.slope*x + reg.intercept
ax[0].plot(x, y, color='deepskyblue', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[0].legend()

X, Y = quad['fg_FR'], quad['FRfg*weightB']
reg = stats.linregress(X, Y)
x = np.asarray(ax[1].get_xlim())
y = reg.slope*x + reg.intercept
ax[1].plot(x, y, color='yellowgreen', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[1].legend()

#plot, with regression, FR v weight
f, ax = plt.subplots(1, 2, figsize=(5,7), sharex=True, sharey=True)
sb.scatterplot(x='bg_FR', y='weightsA', data=quad, ax=ax[0], s=3, color='deepskyblue')
ax[0].set_aspect('equal')
ax[0].set_ylabel('Weight', fontweight='bold', fontsize=10)
ax[0].set_xlabel('Background FR', fontweight='bold', fontsize=10)
ax[0].spines['top'].set_visible(True), ax[0].spines['right'].set_visible(True)

sb.scatterplot(x='fg_FR', y='weightsB', data=quad, ax=ax[1], s=3, color='yellowgreen')
ax[1].set_aspect('equal')
ax[1].set_ylabel('Weight', fontweight='bold', fontsize=10)
ax[1].set_xlabel('Foreground FR', fontweight='bold', fontsize=10)
ax[1].spines['top'].set_visible(True), ax[1].spines['right'].set_visible(True)

X, Y = quad['bg_FR'], quad['weightsA']
reg = stats.linregress(X, Y)
x = np.asarray(ax[0].get_xlim())
y = reg.slope*x + reg.intercept
ax[0].plot(x, y, color='deepskyblue', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[0].legend()

X, Y = quad['fg_FR'], quad['weightsB']
reg = stats.linregress(X, Y)
x = np.asarray(ax[1].get_xlim())
y = reg.slope*x + reg.intercept
ax[1].plot(x, y, color='yellowgreen', label=f"slope: {reg.slope:.3f}\n"
                        f"coef: {reg.rvalue:.3f}\n"
                        f"p = {reg.pvalue:.3f}")
ax[1].legend()




#plot all the sound stats for all the sounds
sound_df = ohel.get_sound_statistics(weight_df, plot=True)

##plot the quad of spectrogram examples for by sound stats
lfreq, hfreq, bins = 100, 24000, 48
sound_idx = [4, 9, 19, 36]
ohel.plot_example_specs(sound_df, sound_idx)



subset = quad.loc[quad.BG=='Waterfall']


f, ax = plt.subplots(1,2)
ax[0].semilogx(x_freq, freq_mean)
ymin, ymax = ax[0].get_ylim()
ax[0].vlines([freq75, freq50, freq25], ymin, ymax)
ax[1].imshow(spec, aspect='auto', origin='lower')


cell = 'CLT007a-009-2'
bg, fg = 'Waterfall', 'Keys'


cells = obip.get_cell_names(df)
cells['site'] = cells.cellid.str[:6]

a1df = weight_df.loc[weight_df.area == 'A1']
pegdf = weight_df.loc[weight_df.area == 'PEG']

a1 = cells.loc[cells.area == 'A1']
peg = cells.loc[cells.area == 'PEG']

site = 'CLT019'
sites = cells.loc[cells.site == site]
for ss in sites.cellid:
    print(ss)


cell = 'CLT007a-009-2'
cell = 'CLT007a-019-1'
cell = 'CLT019a-031-2'
cell = 'CLT019a-055-1'
df = weight_df
cell = df.cellid.unique()[0]

pairs = obip.get_pair_names(cell, df)

bg = 'Chimes'
fg = 'Geese'


obip.plot_binaural_psths(df, cell, bg, fg, batch, save=True, close=False)

# check = df.loc[(df.cellid == cell) & (df.BG == bg) & (df.FG == fg)]
#
#
# for ci in sites.cellid:
#     obip.plot_binaural_psths(df, ci, bg, fg, batch, save=True, close=True)



quad, threshold = ohel.quadrants_by_FR(a1df, threshold=0.05, quad_return=[1,2,3,4,5,6,7,8,9])

#makes scatter
fig, ax = plt.subplots()
ax.scatter(weight_df['bg_FR'], weight_df['fg_FR'], s=2)
ymin,ymax = ax.get_ylim()
xmin,xmax = ax.get_xlim()
ax.vlines([threshold, -threshold], ymin, ymax, ls='-', color='black', lw=0.5)
ax.hlines([threshold, -threshold], xmin, xmax, ls='-', color='black', lw=0.5)
ax.set_ylim(-0.2,0.4)
ax.set_xlim(-0.2,0.4)
ax.set_xlabel('BG Alone FR', fontweight='bold', fontsize=7)
ax.set_ylabel('FG Alone FR', fontweight='bold', fontsize=7)
ax.set_aspect('equal')

df11 = weight_df.loc[(weight_df.kind=='11') & (weight_df.area=='A1')]
df12 = weight_df.loc[(weight_df.kind=='12') & (weight_df.area=='A1')]
df21 = weight_df.loc[(weight_df.kind=='21') & (weight_df.area=='A1')]
df22 = weight_df.loc[(weight_df.kind=='22') & (weight_df.area=='A1')]


fig, ax = plt.subplots()
ax.scatter(df11['bg_FR'], df11['fg_FR'], s=1, label=f'BG contra/FG contra', color='purple')
ax.scatter(df12['bg_FR'], df12['fg_FR'], s=1, label=f'BG contra/FG ipsi', color='orange')
ax.scatter(df21['bg_FR'], df21['fg_FR'], s=1, label=f'BG ipsi/FG contra', color='black')
ax.scatter(df22['bg_FR'], df22['fg_FR'], s=1, label=f'BG ipsi/FG ipsi', color='yellow')
ax.legend(loc='upper left', fontsize=8)
ymin,ymax = ax.get_ylim()
xmin,xmax = ax.get_xlim()
ax.vlines([threshold, -threshold], ymin, ymax, ls='-', color='black', lw=0.5)
ax.hlines([threshold, -threshold], xmin, xmax, ls='-', color='black', lw=0.5)
ax.set_ylim(-0.2,0.4)
ax.set_xlim(-0.2,0.4)
ax.set_xlabel('BG Alone FR', fontweight='bold', fontsize=9)
ax.set_ylabel('FG Alone FR', fontweight='bold', fontsize=9)
ax.set_aspect('equal')

#makes scatter
fig, ax = plt.subplots()
ax.scatter(a1df['bg_FR'], a1df['fg_FR'], color='purple', label='A1', s=2)
# ax.scatter(pegdf['bg_FR'], pegdf['fg_FR'], color='orange', label='PEG', s=2)
ymin,ymax = ax.get_ylim()
xmin,xmax = ax.get_xlim()
ax.vlines([threshold, -threshold], ymin, ymax, ls='-', color='black', lw=0.5)
ax.hlines([threshold, -threshold], xmin, xmax, ls='-', color='black', lw=0.5)
ax.set_ylim(-0.2,0.4)
ax.set_xlim(-0.2,0.4)
ax.set_xlabel('BG Alone FR', fontweight='bold', fontsize=7)
ax.set_ylabel('FG Alone FR', fontweight='bold', fontsize=7)
ax.set_aspect('equal')
ax.spines['top'].set_visible(True), ax.spines['right'].set_visible(True)

#Use if quad_return == 1
opl.weight_hist(quad, tag=None, y='percent')

#Use if quad_return > 1
opl.histogram_subplot_handler(quad, yax='percent', tags=['BG+ / FG+', 'BG+ / FG-', 'BG- / FG+'])
