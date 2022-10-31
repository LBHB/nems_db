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

#testing synthetic
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/synthetic_test.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/synthetic.h5'  #<-- the one you ran yesterday
##synthetic Full With sound stats and weights
OLP_stats_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5' #weight + corr


batch = 328 #Ferret A1
batch = 329 #Ferret PEG
batch = 333 #Marmoset (HOD+TBR)
batch = 340 #All ferret OLP

##Get clathrus synthetic
clt = [dd for dd in cell_list if dd[:3] == 'CLT']
clt = clt[460:]

batch = 339 #Binaural ferret OLP

if fit == True:
    cell_df = nd.get_batch_cells(batch)
    cell_list = cell_df['cellid'].tolist()
    cell_list = ohel.manual_fix_units(cell_list) #So far only useful for two TBR cells
    # cell_list = [cc for cc in cell_list if (cc[:6] == "CLT022") or (cc[:6] == 'CLT023')]
    # cell_list = [cc for cc in cell_list if (cc[:6] == "CLT030") or (cc[:6] == 'CLT033')]
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

#testing synthetic
OLP_weights_db_path = '/auto/users/hamersky/olp_analysis/synth_test_weights.h5' #weight + corr <--Used on full
##
OLP_weights_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full.h5' #weight + corr
OLP_weights_db_path = '/auto/users/hamersky/olp_analysis/Synthetic_weights.h5' #weight + corr

if weights == True:
    weight_df = ofit.fit_weights(df, batch, fs)

    os.makedirs(os.path.dirname(OLP_weights_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_weights_db_path)
    df_store = copy.deepcopy(weight_df)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_weights_db_path)
    weight_df=store['df']
    store.close()


sound_stats = False

#testing synthetic
OLP_stats_db_path = '/auto/users/hamersky/olp_analysis/synethtic_test_full_sound.h5' #weight + corr
##
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


##zippy greg stuff before vacation to get rid of quiet FGs
# weight_dff = weight_df.loc[weight_df.kind=='11']
# print(len(weight_dff))
weight_dff = weight_df.loc[weight_df.FG != 'Heels']
print(len(weight_dff))
weight_dff = weight_dff.loc[weight_dff.FG != 'KitWhine']
print(len(weight_dff))
weight_dff = weight_dff.loc[weight_dff.FG != 'Typing']
print(len(weight_dff))
weight_dff = weight_dff.loc[weight_dff.FG != 'Dice']
print(len(weight_dff))
weight_dff = weight_dff.loc[weight_dff.FG != 'CashRegister']
print(len(weight_dff))
weight_dff = weight_dff.loc[weight_dff.FG != 'KitGroan']
print(len(weight_dff))
weight_dff = weight_dff.loc[weight_dff.FG != 'Keys']
print(len(weight_dff))
quad, threshold = ohel.quadrants_by_FR(weight_dff, threshold=0.03, quad_return=3)
obip.binaural_weight_hist(weight_dff)

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
threshold = 0.75
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

    prebins = int(ref_handle['PreStimSilence'] * options['rasterfs'])
    postbins = int(ref_handle['PostStimSilence'] * options['rasterfs'])
    durbins = int(ref_handle['Duration'] * options['rasterfs'])
    trialbins = durbins + postbins

    env_cuts = {}
    for nm, pth in zip(fgname, fg_paths):
        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        env = np.nanmean(spec, axis=0)
        cutoff = np.max(env) * threshold

        # aboves = np.squeeze(np.argwhere(env >= cutoff))
        # belows = np.squeeze(np.argwhere(env < cutoff))

        highs, lows, whole_thing = env >= cutoff, env < cutoff, env > 0
        prestimFalse = np.full((prebins,), False)
        poststimTrue = np.full((trialbins - len(env),), True)
        poststimFalse = np.full((trialbins - len(env),), False)

        full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
        aboves = np.concatenate((prestimFalse, highs, poststimFalse))
        belows = np.concatenate((prestimFalse, lows, poststimFalse))
        belows_post = np.concatenate((prestimFalse, lows, poststimTrue))

        env_cuts[nm] = [full, aboves, belows, belows_post]

        f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
        ax[0].plot(env)
        ax[0].hlines(cutoff, 0, 100, ls=':')
        ax[0].set_title(f"{nm}")
        ax[1].plot(env[aboves])
        ax[2].plot(env[belows])

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

    print(f'calc weights {cellid}')

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
    subsets = len(list(env_cuts.values())[0])
    weights = np.zeros((2, len(AB), subsets))
    Efit = np.zeros((5,len(AB), subsets))
    nMSE = np.zeros((len(AB), subsets))
    nf = np.zeros((len(AB), subsets))
    r = np.zeros((len(AB), subsets))
    cut_len = np.zeros((len(AB), subsets-1))
    get_error=[]

    for i in range(len(AB)):
        names=[[A[i]],[B[i]],[AB[i]]]
        Fg = names[1][0].split('_')[2].split('-')[0]
        cut_list = env_cuts[Fg]

        for ss, cut in enumerate(cut_list):
            weights[:,i,ss], Efit[:,i,ss], nMSE[i,ss], nf[i,ss], _, r[i,ss], _ = \
                    calc_psth_weights_of_model_responses_list(val, names,
                                                              signame='resp', cuts=cut)
            if ss != 0:
                cut_len[i, ss-1] = np.sum(cut)
            # get_error.append(ge)

    if subsets == 4:
        weight_df = pd.DataFrame(
            [epcs_twostim['nameA'].values, epcs_twostim['nameB'].values,
             weights[0, :, 0], weights[1, :, 0], nMSE[:, 0], nf[:, 0], r[:, 0],
             weights[0, :, 1], weights[1, :, 1], nMSE[:, 1], nf[:, 1], r[:, 1], cut_len[:,0],
             weights[0, :, 2], weights[1, :, 2], nMSE[:, 2], nf[:, 2], r[:, 2], cut_len[:,1],
             weights[0, :, 3], weights[1, :, 3], nMSE[:, 3], nf[:, 3], r[:, 3], cut_len[:,2],])
        weight_df = weight_df.T
        weight_df.columns = ['namesA', 'namesB', 'weightsA', 'weightsB', 'nMSE', 'nf', 'r',
                             'weightsA_h', 'weightsB_h', 'nMSE_h', 'nf_h', 'r_h', 'h_idxs',
                             'weightsA_l', 'weightsB_l', 'nMSE_l', 'nf_l', 'r_l', 'l_idxs',
                             'weightsA_lp', 'weightsB_lp', 'nMSE_lp', 'nf_lp', 'r_lp', 'lp_idxs']
        cols = ['namesA', 'namesB', 'weightsA', 'weightsB', 'nMSE']
        print(weight_df[cols])

        weight_df = weight_df.astype({'weightsA': float, 'weightsB': float,
                                      'weightsA_h': float, 'weightsB_h': float,
                                      'weightsA_l': float, 'weightsB_l': float,
                                      'weightsA_lp': float, 'weightsB_lp': float,
                                      'nMSE': float, 'nf': float, 'r': float,
                                      'nMSE_h': float, 'nf_h': float, 'r_h': float,
                                      'nMSE_l': float, 'nf_l': float, 'r_l': float,
                                      'nMSE_lp': float, 'nf_lp': float, 'r_lp': float,
                                      'h_idxs': float, 'l_idxs': float, 'lp_idxs': float})

    else:
        raise ValueError(f"Only {subsets} subsets. You got lazy and didn't make this part"
                         f"flexible yet.")


    weight_df.insert(loc=0, column='cellid', value=cellid)

    weight_list.append(weight_df)

weight_df0 = pd.concat(weight_list)


ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df0.namesA, weight_df0.namesB)]
weight_df0 = weight_df0.drop(columns=['namesA', 'namesB'])
weight_df0['epoch'] = ep_names

weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
weight_df0['threshold'] = str(int(threshold * 100))
if df.shape[0] != weights_df.shape[0] or weight_df.shape[0] != weights_df.shape[0]:
    raise ValueError("Resulting weights_df does not match length of parts, some epochs were dropped.")

##load here.
OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_partial_weights20.h5'  # weight + corr
OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_partial_weights.h5'  # weight + corr

part_weights = False
if part_weights == True:
    os.makedirs(os.path.dirname(OLP_partialweights_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_partialweights_db_path)
    df_store=copy.deepcopy(weight_df0)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_partialweights_db_path)
    weight_df0=store['df']
    store.close()




def calc_psth_weights_of_model_responses_list(val, names, signame='resp',
                                              get_nrmse_fn=False, cuts=None):

    sig1 = np.concatenate([val[signame].extract_epoch(n).squeeze()[cuts] for n in names[0]])
    sig2 = np.concatenate([val[signame].extract_epoch(n).squeeze()[cuts] for n in names[1]])
    # sig_SR=np.ones(sig1.shape)
    sigO = np.concatenate([val[signame].extract_epoch(n).squeeze()[cuts] for n in names[2]])

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

    # calc CC between weight model and actual response
    pred = np.dot(weights, fsigs[ff, :].T)
    cc = np.corrcoef(pred, sigO[ff])
    r_weight_model = cc[0, 1]

    # norm_factor = np.std(sigO[ff])
    norm_factor = np.mean(sigO[ff] ** 2)

    if rank == 1:
        min_nMSE = 1
        min_nRMSE = 1
    else:
        # min_nrmse = np.sqrt(residual_sum[0]/ff.sum())/norm_factor
        pred = np.dot(weights, fsigs[ff, :].T)
        min_nRMSE = np.sqrt(((sigO[ff] - pred) ** 2).mean()) / np.sqrt(
            norm_factor)  # minimim normalized root mean squared error
        min_nMSE = ((sigO[ff] - pred) ** 2).mean() / norm_factor  # minimim normalized mean squared error

    # create NMSE caclulator for later
    if get_nrmse_fn:
        def get_nrmse(weights=weights):
            pred = np.dot(weights, fsigs[ff, :].T)
            nrmse = np.sqrt(((pred - sigO[ff]) ** 2).mean(axis=-1)) / norm_factor
            return nrmse
    else:
        get_nrmse = np.nan

    weights[close_to_zero] = np.nan
    return weights, np.nan, min_nMSE, norm_factor, get_nrmse, r_weight_model, get_error


##################FINISHED FRIDAY, WANT TO PLOT THE FOUR CONDITIONS NOW
env_threshold = '75'
quad_threshold= 0.05
# area = 'A1'

edges = np.arange(-1,2,.05)

df = weight_df0.loc[weight_df0['threshold'] == env_threshold]

quad0, _ = ohel.quadrants_by_FR(df, threshold=quad_threshold, quad_return=3)
quad0 = quad0.loc[quad0.kind == '11']
quad0.loc[quad0['l_idxs'] <= 5, 'weightsA_l'] = np.NaN
quad0.loc[quad0['l_idxs'] <= 5, 'weightsB_l'] = np.NaN
quad0.loc[quad0['h_idxs'] <= 5, 'weightsA_h'] = np.NaN
quad0.loc[quad0['h_idxs'] <= 5, 'weightsB_h'] = np.NaN


f = plt.figure(figsize=(15, 12))
hist11 = plt.subplot2grid((13, 16), (0, 0), rowspan=5, colspan=3)
mean11 = plt.subplot2grid((13, 16), (0, 4), rowspan=5, colspan=2)
hist12 = plt.subplot2grid((13, 16), (0, 8), rowspan=5, colspan=3, sharey=hist11)
mean12 = plt.subplot2grid((13, 16), (0, 12), rowspan=5, colspan=2, sharey=mean11)
hist21 = plt.subplot2grid((13, 16), (7, 0), rowspan=5, colspan=3, sharey=hist11)
mean21 = plt.subplot2grid((13, 16), (7, 4), rowspan=5, colspan=2, sharey=mean11)
hist22 = plt.subplot2grid((13, 16), (7, 8), rowspan=5, colspan=3, sharey=hist11)
mean22 = plt.subplot2grid((13, 16), (7, 12), rowspan=5, colspan=2, sharey=mean11)
ax = [hist11, hist12, hist21, hist22, mean11, mean12, mean21, mean22]


titles = ['Full weights', f'Above {env_threshold}% env',
          f'Below {env_threshold}% env no post', f'Below {env_threshold}% env with post']
Aw = ['weightsA', 'weightsA_h', 'weightsA_l', 'weightsA_lp']
Bw = ['weightsB', 'weightsB_h', 'weightsB_l', 'weightsB_lp']


ttests = {}
DF = quad0
for aa, (tt, aw, bw) in enumerate(zip(titles, Aw, Bw)):
    na, xa = np.histogram(DF[aw], bins=edges)
    na = na / na.sum() * 100
    nb, xb = np.histogram(DF[bw], bins=edges)
    nb = nb / nb.sum() * 100

    ax[aa].hist(xa[:-1], xa, weights=na, histtype='step', color='deepskyblue')
    ax[aa].hist(xb[:-1], xb, weights=nb, histtype='step', color='yellowgreen')
    ax[aa].legend(('Background', 'Foreground'), fontsize=6)
    ax[aa].set_ylabel('Percentage\nof cells', fontweight='bold', fontsize=10)
    ax[aa].set_title(f"{tt}", fontweight='bold', fontsize=12)
    ax[aa].set_xlabel("Weight", fontweight='bold', fontsize=10)
    ymin, ymax = ax[aa].get_ylim()

    BG1, FG1 = np.mean(DF[aw]), np.mean(DF[bw])
    BG1sem, FG1sem = stats.sem(DF[aw]), stats.sem(DF[bw])
    ttest = stats.ttest_ind(DF[aw], DF[bw])
    ax[aa+4].bar("BG", BG1, yerr=BG1sem, color='deepskyblue')
    ax[aa+4].bar("FG", FG1, yerr=FG1sem, color='yellowgreen')
    ax[aa+4].set_ylabel('Mean Weight', fontweight='bold', fontsize=10)
    if ttest.pvalue < 0.001:
        title = 'p<0.001'
    else:
        title = f"{ttest.pvalue:.3f}"
    ax[aa + 4].set_title(title, fontsize=8)


f.suptitle(f"{area}", fontweight='bold', fontsize=12)










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



def sound_metric_scatter(df, x_metrics, y_metric, x_labels, area='A1', threshold=0.03,
                         jitter=[0.2,0.03,0.25],
                         quad_return=3, metric_filter=None, synth_kind='N', bin_kind='11'):
    '''Makes a series of scatterplots that compare a stat of the sounds to some metric of data. In
    a usual situation it would be Tstationariness, bandwidth, and Fstationariness compared to relative
    gain. Can also be compared to weights.
    y_metric refers to the FIRST one it will input, for relative_gain this is not an issue. If you want
    to differentiate between weights the sound affects in others vs how that sound is weighted itself,
    input the one as it relates to BG, so 'weightsB' will be 'how that sound effects others' and will
    know to make the metric 'weightsA' for the FGs, for example.
    When inputting x_metric names, always make it a list. All entries should be found in the df being
    passed, but you should remove the BG_ or FG_ prefix.
    Made into a function from OLP_analysis_main on 2022_09_07'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=quad_return)
    quad = quad.loc[(quad.area==area) & (quad.synth_kind==synth_kind) & (quad.kind==bin_kind)]
    quad = quad.copy()

    # I use 2.5 for relative gain, I'm sure weights have one too...
    if metric_filter:
        quad = quad.loc[quad[y_metric] <= metric_filter]
        quad = quad.loc[quad[y_metric] >= -metric_filter]

    if y_metric=='BG_rel_gain':
        y_metric2, title, ylabel = 'FG_rel_gain', 'Relative Gain', 'Relative Gain'
    elif y_metric=='weightsB':
        y_metric2, title, ylabel = 'weightsA', 'How this sound effects a concurrent sound', 'Weight'
    elif y_metric=='weightsA':
        y_metric2, title, ylabel = 'weightsB', 'How this sound itself is weighted', 'Weight'
    else:
        y_metric2, title, ylabel = y_metric, y_metric, y_metric

    fig, axes = plt.subplots(1, len(x_metrics), figsize=(len(x_metrics)*4, 5))

    for cnt, (ax, met) in enumerate(zip(axes, x_metrics)):
        # Add a column that is the data for that metric, but jittered, for viewability
        quad[f'jitter_BG_{met}'] = quad[f'BG_{met}'] + np.random.normal(0, jitter[cnt], len(quad))
        quad[f'jitter_FG_{met}'] = quad[f'FG_{met}'] + np.random.normal(0, jitter[cnt], len(quad))
        # Do the plotting
        sb.scatterplot(x=f'jitter_BG_{met}', y=y_metric, data=quad, ax=ax, s=3, color='deepskyblue')
        sb.scatterplot(x=f'jitter_FG_{met}', y=y_metric2, data=quad, ax=ax, s=3, color='yellowgreen')
        ax.set_xlabel(x_labels[cnt], fontweight='bold', fontsize=10)
        if cnt==0:
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=10)
        else:
            ax.set_ylabel('')

        # Run a regression
        Y = np.concatenate((quad[y_metric].values, quad[y_metric2].values))
        X = np.concatenate((quad[f'BG_{met}'].values, quad[f'FG_{met}'].values))
        reg = stats.linregress(X, Y)
        x = np.asarray(ax.get_xlim())
        y = reg.slope * x + reg.intercept
        ax.plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                                                 f"coef: {reg.rvalue:.3f}\n"
                                                 f"p = {reg.pvalue:.3f}")
        ax.legend()


    fig.suptitle(f"{title} - RMS Nat", fontweight='bold', fontsize=10)







    Y = np.concatenate((quad['weightsB'].values, quad['weightsA'].values))
    X = np.concatenate((quad['BG_Tstationary'].values, quad['FG_Tstationary'].values))
    reg = stats.linregress(X, Y)
    x = np.asarray(ax.get_xlim())
    y = reg.slope * x + reg.intercept
v
    ax[0].legend()

    X = np.concatenate((quad['BG_bandwidth'].values, quad['FG_bandwidth'].values))
    reg = stats.linregress(X, Y)
    x = np.asarray(ax[1].get_xlim())
    y = reg.slope * x + reg.intercept
    ax[1].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                                             f"coef: {reg.rvalue:.3f}\n"
                                             f"p = {reg.pvalue:.3f}")
    ax[1].legend()

    X = np.concatenate((quad['BG_Fstationary'].values, quad['FG_Fstationary'].values))
    reg = stats.linregress(X, Y)
    x = np.asarray(ax[2].get_xlim())
    y = reg.slope * x + reg.intercept
    ax[2].plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                                             f"coef: {reg.rvalue:.3f}\n"
                                             f"p = {reg.pvalue:.3f}")
    ax[2].legend()











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




#
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

### SOME SCATTERS THAT ARE NICE
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
