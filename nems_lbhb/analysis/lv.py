"""
Code for new(?) latent variable analysis associated with Aim 3b of R01 A1
"""


from sklearn.decomposition import PCA
import nems0.db as nd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

params = {'legend.fontsize': 12,
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)

from scipy.stats import linregress
from sklearn.decomposition import PCA

from scipy.optimize import fmin, minimize
from scipy.signal import resample
from nems0.preprocessing import generate_psth_from_resp
from nems_lbhb.preprocessing import mask_high_repetion_stims

from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.preprocessing import make_state_signal
from nems_lbhb.tin_helpers import sort_targets, compute_ellipse, load_tbp_recording, pb_regress, \
   get_sound_labels, plot_average_psths, site_tuning_avg, site_tuning_curves
from nems0.xform_helper import fit_model_xform, load_model_xform
import nems0.db as nd
from nems0.metrics.lv import cc_err


outpath="/auto/users/svd/projects/lv/pupil_nat/"
savefigs=False

options = {'resp': True, 'pupil': True, 'rasterfs': 10}
ref_modelname = "ozgf.fs100.ch18-ld-norm.l1-sev_wc.18x50.g-fir.1x20x50-relu.50.f-wc.50x60-relu.60.f-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4"

def get_population_cellids(batch, modelname=None):
    if modelname is None:
        modelname = ref_modelname

    def get_siteid(s):
       return s.split("-")[0]

    all_siteids={}
    rep_cellids={}
    
    d = nd.batch_comp(batch=batch, modelnames=[modelname])
    d['siteid'] = d.index.map(get_siteid)

    siteids = list(set(d['siteid'].tolist()))
    
    all_siteids[batch]=siteids
    
    site_cellids = [d.loc[d.index.str.startswith(s)].index.values[0] for s in siteids]
    site_cellids.sort()
    
    rep_cellids[batch]=site_cellids

    return all_siteids, rep_cellids

# define cost functions

# apply additive (dc) LV to each neurons' prediction
# pred = pred_0 + (d+g*state) * lv , with d,g as gain and offset for each neuron
def lv_mod_dc(d, g, state, lv, pred, showdetails=False):
    pred=pred.copy()
    if showdetails:
        f,ax=plt.subplots(d.shape[1],1,figsize=(10,5))
        if d.shape[1]==1:
            ax=[ax]
    for l in range(d.shape[1]):
        pred += (d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]) * lv[l:(l+1),:]
        if showdetails:
            ax[l].imshow((d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]), aspect='auto', interpolation='none', origin='lower', cmap='bwr')
    return pred 

# apply mulitiplicative LV to each neurons' prediction. fast version. identical output to slow?
# pred = pred_0 + (d+g*state) * lv , with d,g as gain and offset for each neuron
def lv_mod(d, g, state, lv, pred0, showdetails=False):
    """
    returns
       pred: cell X time matrix of scaled prediction
    """
    if showdetails:
        f,ax=plt.subplots(d.shape[1],1,figsize=(10,5))
        if d.shape[1]==1:
            ax=[ax]
    sf = np.zeros(pred0.shape)
    for l in range(d.shape[1]):
        # slow: apply each scaling term sequentially
        #pred *= np.exp((d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]) * lv[l:(l+1),:])
        # faster(?): compute all scaling terms then apply at once (outside of loop)
        sf += (d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]) * lv[l:(l+1),:]
        if showdetails:
            ax[l].imshow((d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]), aspect='auto', interpolation='none', origin='lower', cmap='bwr')
    pred = pred0 * np.exp(sf)
    return pred 

err_counter=0

# cost function to compute error between predicted and actual noise correlation matrices
def cc_err_defunct(w, pred, indep_noise, lv, pred0, state, active_cc, passive_cc, aidx, pidx, 
           pcproj_std=None, pc_axes=None):
    global err_counter
    _w=np.reshape(w,[pred0.shape[0], -1])
    lv_count=int(_w.shape[1]/3)
    p = lv_mod(_w[:,lv_count:(lv_count*2)], _w[:,(lv_count*2):(lv_count*3)], state, lv, pred) + (_w[:,0:lv_count] @ state) *indep_noise
    pascc = np.cov(p[:,pidx] - pred0[:,pidx])
    actcc = np.cov(p[:,aidx] - pred0[:,aidx])
    E = np.sum((pascc-passive_cc)**2) / np.sum(passive_cc**2) + np.sum((actcc-active_cc)**2) / np.sum(active_cc**2)
    if (err_counter % 1000) == 0:
        print(f"{err_counter}: {E}")
    err_counter+=1
    return E

# cost function to compute error between predicted and actual noise correlation matrices, with additional term to match
def cc_err_bak(w, pred, indep_noise, lv, pred0, state, active_cc, passive_cc, aidx, pidx,
            pcproj_std=None, pc_axes=None):
    global err_counter
    _w=np.reshape(w,[pred0.shape[0], -1])
    lv_count=int(_w.shape[1]/3)
    p = lv_mod(_w[:,lv_count:(lv_count*2)], _w[:,(lv_count*2):(lv_count*3)], state, lv, pred) + (_w[:,0:lv_count] @ state) *indep_noise
    pascc = np.cov(p[:,pidx] - pred0[:,pidx])
    actcc = np.cov(p[:,aidx] - pred0[:,aidx])

    if pc_axes is not None:
        pcproj = (p-pred).T.dot(pc_axes.T).T
        pp_std = pcproj.std(axis=1)
        E = np.sum((pascc-passive_cc)**2) / np.sum(passive_cc**2) + np.sum((actcc-active_cc)**2) / np.sum(active_cc**2) + \
           np.sum((pcproj_std-pp_std)**2)*10
    else:
        E = np.sum((pascc-passive_cc)**2) / np.sum(passive_cc**2) + np.sum((actcc-active_cc)**2) / np.sum(active_cc**2)

    if (err_counter % 1000) == 0:
        print(f"{err_counter}: {E}")
    err_counter+=1
    return E

def cc_err_withlv(w, pred, indep_noise, lv, pred0, state, active_cc, passive_cc, aidx, pidx,
                  pcproj_std=None, pc_axes=None):
    global err_counter
    _w=np.reshape(w,[pred0.shape[0], -1])
    lv_count=int(_w.shape[1]/3)
    p = lv_mod(_w[:,lv_count:(lv_count*2)], _w[:,(lv_count*2):(lv_count*3)], state, lv, pred) + (_w[:,0:lv_count] @ state) *indep_noise
    pascc = np.cov(p[:,pidx] - pred0[:,pidx])
    actcc = np.cov(p[:,aidx] - pred0[:,aidx])
    E = cc_err({'pred_lv': p, 'pred': pred0}, group_idx=[aidx, pidx], group_cc=[active_cc, passive_cc], 
                  pcproj_std=pcproj_std, pc_axes=pc_axes)

    if (err_counter % 1000) == 0:
        print(f"{err_counter}: {E}")
    err_counter+=1
    return E

# cost function to minimize error in noise correlation and overall prediction, with independent additive noise only (no LV)
# accepts identical parameters as LV cost functions so it can be slotted in as a control model
def cc_err_nolv(w, pred, indep_noise, lv, pred0, state, active_cc, passive_cc, aidx, pidx,
                pcproj_std=None, pc_axes=None):
    global err_counter
    _w=np.reshape(w,[pred0.shape[0], -1])
    lv_count=int(_w.shape[1]/3)
    p = (_w[:,0:lv_count] @ state) *indep_noise
    pascc = np.cov(p[:,pidx] - pred0[:,pidx])
    actcc = np.cov(p[:,aidx] - pred0[:,aidx])
    
    if pc_axes is not None:
        pcproj = (p-pred).T.dot(pc_axes.T).T
        pp_std = pcproj.std(axis=1)
        E = np.sum((pascc-passive_cc)**2) / np.sum(passive_cc**2) + np.sum((actcc-active_cc)**2) / np.sum(active_cc**2) + \
           np.sum((pcproj_std-pp_std)**2)*10
    else:
        E = np.sum((pascc-passive_cc)**2) / np.sum(passive_cc**2) + np.sum((actcc-active_cc)**2) / np.sum(active_cc**2)

    if (err_counter % 1000) == 0:
        print(f"{err_counter}: {E}")
    err_counter+=1
    return E

def lv_wrapper(siteid="TAR010c", 
               batch=322, 
               modelname="psth.fs10.pup-ld-st.pup-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont",
               maxiter=10000):

    d = nd.pd_query(f"SELECT batch,cellid FROM Batches WHERE batch=%s AND cellid like %s", (batch, siteid+"%",))
    cellid=d['cellid'].values[0]

    xf,ctx = load_model_xform(cellid, batch, modelname)


    ### define large/small masks
    rec=ctx['val'].copy()
    
    # code hacked from nems_lbhb.preprocessing.pupil_mask
    pupil_data = rec['pupil'].extract_epoch('REFERENCE')
    pupil_data = np.tile(np.nanmean(pupil_data, axis=-1),
    		     [1, pupil_data.shape[-1]])[:, np.newaxis, :]
    pup_median = np.median(pupil_data.flatten()[~np.isnan(pupil_data.flatten())])
    
    for condition in ['large','small']:
        if condition == 'large':
    	    mask = ((pupil_data > pup_median) & (~np.isnan(pupil_data)))
    	    op_mask = ((pupil_data <= pup_median) & (~np.isnan(pupil_data)))
        elif condition == 'small':
            mask = ((pupil_data <= pup_median) & (~np.isnan(pupil_data)))
            op_mask = ((pupil_data > pup_median) & (~np.isnan(pupil_data)))

        # perform AND mask with existing mask
        if 'mask' in rec.signals:
            mask = (mask & rec['mask'].extract_epoch('REFERENCE'))
            op_mask = (op_mask & rec['mask'].extract_epoch('REFERENCE'))
        elif 'mask' not in rec.signals:
            pass

        rec['mask_'+condition] = rec['mask'].replace_epochs({'REFERENCE': mask})

    rm=rec.apply_mask()

    #rec=mask_high_repetion_stims(rec)
    #rec = generate_psth_from_resp(rec, resp_sig='resp', epoch_regex='^STIM_')
    #rm=rec.apply_mask()
    pred_sig='pred'

    resp0 = rm['resp']._data
    state0 = rm['state']._data.copy()
    pred0 = rm[pred_sig]._data.copy()
    mask_large0 = rm['mask_large']._data.copy().astype(int)

    # resample from 50 Hz to 10 Hz
    #ds_ratia=10
    ds_ratio=1
    L0 = resp0.shape[1]
    L = int(np.ceil(L0/ds_ratio))
    dL = int(L0-L*ds_ratio)

    if ds_ratio>1:
        resp = resample(np.pad(resp0, ((0,0),(0,dL))), L, axis=1)
        state = resample(np.pad(state0, ((0,0),(0,dL))), L, axis=1)
        pred = resample(np.pad(pred0, ((0,0),(0,dL))), L, axis=1)
        mask_large = np.round(resample(np.pad(mask_large0, ((0,0),(0,dL))), L, axis=1))
    else:

        resp=resp0
        state=state0
        pred=pred0
        mask_large=mask_large0

    cellcount=resp.shape[0]
    state_count=state.shape[0]

    cmap='bwr'

    print(f"L={L}, L0={L0}")

    from nems_lbhb.dimensionality_reduction import TDR
    import re

    epoch_regex = '^STIM_'
    stims = (rec.epochs['name'].value_counts() >= 8)
    stims = [stims.index[i] for i, s in enumerate(stims) if bool(re.search(epoch_regex, stims.index[i])) and s == True]

    # can't simply extract evoked for refs because can be longer/shorted if it came after target 
    # and / or if it was the last stim. So, masking prestim / postim doesn't work. Do it manually
    d = rec['resp'].extract_epochs(stims, mask=rec['mask'])

    R = [v.mean(axis=0) for (k, v) in d.items()]
    #R = [np.reshape(np.transpose(v,[1,0,2]),[v.shape[1],-1]) for (k, v) in d.items()]
    Rall_u = np.hstack(R).T

    pca = PCA(n_components=2)
    pca.fit(Rall_u)
    pc_axes = pca.components_

    pc_axes.shape

    #state_fit = np.concatenate((state,state[1:,:]), axis=0)
    state_fit = state
    pred_fit = pred
    resp_fit = resp

    dup_count=2
    if dup_count>1:
        print(f'duplicating state var(s) {dup_count} X')
        state_fit=np.concatenate((state_fit,state_fit[1:,:]),axis=0)

    lv_count=state_fit.shape[0]
    # lv_count = 1   # static
    # lv_count = 2   # pupil only

    indep_noise = np.random.randn(*pred_fit.shape)
    lv = np.random.randn(*state_fit[:lv_count,:].shape)
    pred0 = pred_fit.copy()

    actual_cc = np.cov(resp_fit-pred0)
    u,s,vh=np.linalg.svd(actual_cc)
    pc1=u[:,0]

    med_pupil = np.median(state_fit[1,:])

    # compute noise correlation for active and passive conditions separately
    large_idx = mask_large.astype(bool)[0,:]
    small_idx = (1-mask_large).astype(bool)[0,:]
    large_cc = np.cov(resp_fit[:,large_idx]-pred0[:,large_idx])
    small_cc = np.cov(resp_fit[:,small_idx]-pred0[:,small_idx])

    f,ax=plt.subplots(1,3,figsize=(12,4))
    mm = np.max(np.concatenate((small_cc,large_cc)))
    ax[0].imshow(small_cc,clim=[-mm,mm],cmap=cmap)
    ax[0].set_title('small')
    ax[1].imshow(large_cc,clim=[-mm,mm],cmap=cmap)
    ax[1].set_title('large')
    d=large_cc-small_cc
    mmd=np.max(np.abs(d))
    im=ax[2].imshow(large_cc-small_cc,clim=[-mmd,mmd],cmap=cmap,interpolation='none')
    ax[2].set_title('large-small')
    #plt.colorbar(im,ax=ax[2])

    indep_noise.shape, lv.shape, small_idx.shape, small_idx.sum()

    options={'gtol': 1e-04, 'maxfun': maxiter, 'maxiter': maxiter}

    # variance of projection onto PCs
    pcproj0 = (resp-pred).T.dot(pc_axes.T).T
    pcproj_std = pcproj0.std(axis=1)

    # no-LV fit, just independent noise
    w0 = np.zeros((cellcount,lv_count*3))
    w0[:,0]=0.05

    # first fit without independent noise to push out to LVs
    err_counter=0
    res = minimize(cc_err_nolv, w0, options=options, method='L-BFGS-B',
                   args=(pred_fit, indep_noise, lv, pred0, state_fit, large_cc, small_cc, large_idx, small_idx, pcproj_std, pc_axes))
    w_nolv=np.reshape(res.x,[-1, lv_count*3])

    # intially perform only-LV fit, no independent noise
    # initialize
    if dup_count<2:
        w0 = np.zeros((cellcount,lv_count*3))
        w0[:,0]=0.05
        w0[:,lv_count*2]=pc1/10
    else:
        w0 = np.random.randn(cellcount, lv_count*3)/100
        w0[:,0]=0.05

    # first fit without independent noise to push out to LVs
    print('fitting only LV terms first...')
    res = minimize(cc_err_withlv, w0, options=options, method='L-BFGS-B',
                   args=(pred_fit, indep_noise*0, lv, pred0, state_fit, large_cc, small_cc, large_idx, small_idx, pcproj_std, pc_axes))
    w1=np.reshape(res.x,[-1, lv_count*3])

    w=w1.copy()

    # now perform fit weights for both LV and indep noise

    # second fit WITH independent noise to allow for independent noise
    print('... then fitting indep + LV terms...')
    res = minimize(cc_err_withlv, w1, options=options, method='L-BFGS-B',
                   args=(pred_fit, indep_noise, lv, pred0, state_fit, large_cc, small_cc, large_idx, small_idx, pcproj_std, pc_axes))
    w=np.reshape(res.x,[-1, lv_count*3])


    f,ax=plt.subplots(3,3,figsize=(12,4),sharey='row',sharex=True)
    ax[0,0].plot(w_nolv[:,0:lv_count])
    ax[0,0].plot(w0[:,0],'--')
    ax[0,0].set_title('indep only')
    ax[1,0].plot(w_nolv[:,lv_count:(lv_count*2)])
    ax[1,0].plot(w0[:,lv_count:(lv_count*2)],'--')
    ax[2,0].plot(w_nolv[:,(lv_count*2):])
    ax[2,0].plot(w0[:,(lv_count*2):],'--');

    ax[0,1].plot(w1[:,0:lv_count])
    ax[0,1].plot(w0[:,0],'--')
    ax[0,1].set_title('lv only')
    ax[1,1].plot(w1[:,lv_count:(lv_count*2)])
    ax[1,1].plot(w0[:,lv_count:(lv_count*2)],'--')
    ax[2,1].plot(w1[:,(lv_count*2):])
    ax[2,1].plot(w0[:,(lv_count*2):],'--');

    ax[0,2].plot(w[:,0:lv_count])
    ax[0,2].plot(w0[:,0],'--')
    ax[0,2].set_title('full indep+lv')
    ax[1,2].plot(w[:,lv_count:(lv_count*2)])
    ax[1,2].plot(w1[:,lv_count:(lv_count*2)],'--')
    ax[2,2].plot(w[:,(lv_count*2):])
    ax[2,2].plot(w1[:,(lv_count*2):],'--');


    ## generate predictions with indep noise and LV
    ## resample back up to L0-dL from L
    #L0 = resp0.shape[1]
    #L = int(np.ceil(L0/ds_ratio))
    #dL = int(L0-L*ds_ratio)
    #pred = resample(np.pad(pred0, ((0,0),(0,dL))), L, axis=1)
    #
    # resample back to L0+dL then remove last dL bins. Resulting vector should match len(mm)?

    pred_data = rec[pred_sig]._data.copy()
    mm = rec['mask']._data[0,:]
    #pred_indep = pred + (w[:,0:lv_count] @ state_fit) *indep_noise
    pred_indep = pred + (w_nolv[:,0:lv_count] @ state_fit) *indep_noise
    if ds_ratio>1:
        pred_data[:,mm] = resample(pred_indep, L*ds_ratio, axis=1)[:, 0:((L*ds_ratio)-dL)]
    else:
        pred_data[:,mm] = pred_indep

    rec['pred_indep'] = rec[pred_sig]._modified_copy(data=pred_data)

    pred_lv = lv_mod(w[:,lv_count:(lv_count*2)], w[:,(lv_count*2):(lv_count*3)], state_fit, lv[:,:state.shape[1]], pred, showdetails=True) + (w[:,0:lv_count] @ state_fit) *indep_noise
    pred_data = rec[pred_sig]._data.copy()
    if ds_ratio>1:
        pred_data[:,mm] = resample(pred_lv, L*ds_ratio, axis=1)[:, 0:((L*ds_ratio)-dL)]
    else:
        pred_data[:,mm] = pred_lv
    rec['pred_lv'] = rec[pred_sig]._modified_copy(data=pred_data)

    w_nopup=w.copy()
    #w_nopup[:,1]=0
    w_nopup[:,(lv_count+1):(lv_count*2)]=0
    w_nopup[:,(lv_count*2+1):]=0
    pred_nopup = lv_mod(w_nopup[:,lv_count:(lv_count*2)], w_nopup[:,(lv_count*2):(lv_count*3)], state_fit, lv[:,:state.shape[1]], pred, showdetails=False) + (w[:,0:lv_count] @ state_fit) *indep_noise
    pred_data = rec[pred_sig]._data.copy()
    if ds_ratio>1:
        pred_data[:,mm] = resample(pred_nopup, L*ds_ratio, axis=1)[:, 0:((L*ds_ratio)-dL)]
    else:
        pred_data[:,mm] = pred_nopup
    rec['pred_nopup'] = rec[pred_sig]._modified_copy(data=pred_data)

    ## display noise corr. matrices
    f,ax = plt.subplots(4,5, figsize=(10,8), sharex=True, sharey=True)

    pred0=pred.copy()

    large_cc = np.cov(resp[:,large_idx]-pred0[:,large_idx])
    small_cc = np.cov(resp[:,small_idx]-pred0[:,small_idx])

    mm=np.max(np.abs(small_cc)) * 0.5

    sm_cc = np.cov(pred[:,small_idx]-pred0[:,small_idx])
    lg_cc = np.cov(pred[:,large_idx]-pred0[:,large_idx])
    ax[0,0].imshow(sm_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1,0].imshow(lg_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[3,0].imshow((large_cc-small_cc) - (lg_cc-sm_cc),aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[0,0].set_title(siteid + ' pred')

    ax[0,0].set_ylabel('small')
    ax[1,0].set_ylabel('large')
    ax[2,0].set_ylabel('large-small')
    ax[3,0].set_ylabel('d_sim-d_act')

    sm_cc = np.cov(pred_indep[:,small_idx]-pred0[:,small_idx])
    lg_cc = np.cov(pred_indep[:,large_idx]-pred0[:,large_idx])
    ax[0,1].imshow(sm_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1,1].imshow(lg_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[2,1].imshow(lg_cc-sm_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[3,1].imshow((large_cc-small_cc) - (lg_cc-sm_cc),aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[0,1].set_title('pred + indep noise')

    sm_cc = np.cov(pred_lv[:,small_idx]-pred0[:,small_idx])
    lg_cc = np.cov(pred_lv[:,large_idx]-pred0[:,large_idx])
    ax[0,2].imshow(sm_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1,2].imshow(lg_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[2,2].imshow(lg_cc-sm_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[3,2].imshow((large_cc-small_cc) - (lg_cc-sm_cc),aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[0,2].set_title('pred + lv')

    ax[0,3].imshow(small_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1,3].imshow(large_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[2,3].imshow(large_cc-small_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[0,3].set_title('actual resp')

    sm_cc = np.cov(pred_nopup[:,small_idx]-pred0[:,small_idx])
    lg_cc = np.cov(pred_nopup[:,large_idx]-pred0[:,large_idx])
    ax[0,4].imshow(sm_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[1,4].imshow(lg_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[2,4].imshow(lg_cc-sm_cc,aspect='auto',interpolation='none',clim=[-mm,mm], cmap='bwr', origin='lower')
    ax[0,4].set_title('pred_nopup + lv')

    if savefigs:
        outfile = f"noise_corr_sim_{siteid}_{batch}.pdf"
        print(f"saving to {outpath}/{outfile}")
        f.savefig(f"{outpath}/{outfile}")

    rt=rec.copy()

    #a=tdr_axes
    a=pc_axes

    # project onto first two PCs
    rt['rpc'] = rt['resp']._modified_copy(rt['resp']._data.T.dot(a.T).T[0:2, :])
    rt['ppc_pred'] = rt[pred_sig]._modified_copy(rt[pred_sig]._data.T.dot(a.T).T[0:2, :])
    rt['ppc_indep'] = rt['pred_indep']._modified_copy(rt['pred_indep']._data.T.dot(a.T).T[0:2, :])
    rt['ppc_lv'] = rt['pred_lv']._modified_copy(rt['pred_lv']._data.T.dot(a.T).T[0:2, :])
    rt['ppc_nopup'] = rt['pred_nopup']._modified_copy(rt['pred_nopup']._data.T.dot(a.T).T[0:2, :])

    units = rt['resp'].chans
    e=rt['resp'].epochs
    r_large = rt.copy()
    r_large['mask']=r_large['mask_large']
    r_small = rt.copy()
    r_small['mask']=r_small['mask_small']

    conditions = ['small', 'large']
    cond_recs = [r_small, r_large]

    d = rec['resp'].get_epoch_bounds('PreStimSilence')
    PreStimBins = int(np.round(np.mean(np.diff(d))*rec['resp'].fs))
    d = rec['resp'].get_epoch_bounds('PostStimSilence')
    PostStimBins = int(np.round(np.mean(np.diff(d))*rec['resp'].fs))

    ChunkSec=0.5
    ChunkBins = int(np.round(ChunkSec*rec['resp'].fs))
    PreStimBins, PostStimBins, ChunkBins

    #cmaps = [[BwG(int(c)) for c in np.linspace(0,255,len(ref_stims))], 
    #         [gR(int(c)) for c in np.linspace(0,255,len(sounds))]]
    from nems_lbhb.tin_helpers import make_tbp_colormaps, compute_ellipse
    siglist = ['ppc_pred', 'ppc_indep', 'ppc_lv', 'rpc', 'ppc_nopup']
    f,ax=plt.subplots(len(conditions),len(siglist),sharex=True,sharey=True, figsize=(2*len(siglist),4))
    for ci, to, r in zip(range(len(conditions)), conditions, cond_recs):
        for j, sig in enumerate(siglist):
            #colors = cmaps[0]
            for i,k in enumerate(stims):
                try:
                    p = r[sig].extract_epoch(k, mask=r['mask'])
                    if p.shape[0]>2:
                        psamples = p.shape[2]
                        for c in range(PreStimBins-ChunkBins,psamples-PostStimBins,ChunkBins):
                            g = np.isfinite(p[:,0,c])
                            x = np.nanmean(p[g,0,c:(c+ChunkBins)], axis=1)
                            y = np.nanmean(p[g,1,c:(c+ChunkBins)], axis=1)
                            #c=list(colors(i))
                            #c[-1]=0.2
                            #ax[ci, j].plot(x,y,'.', color=c, label=k)
                            e = compute_ellipse(x, y)
                            ax[ci, j].plot(e[0], e[1])
                            if c==(PreStimBins-ChunkBins):
                                ax[ci,j].plot(x.mean(),y.mean(),'k*',markersize=5)
                except:
                    #print(f'no matches for {k}')
                    pass

            ax[ci,j].set_title(f"{to}-{sig}")
    #ax[ci, 0].legend()
    #ax[ci, 0].set_title(to + " REF/TAR")

    ax[0,0].set_ylabel(siteid)
    ax[1,0].set_xlabel('PC1')
    ax[1,0].set_ylabel('PC2')

    if savefigs:
        outfile = f"pop_latent_sim_{siteid}_{batch}.pdf"
        print(f"saving to {outpath}/{outfile}")
        f.savefig(f"{outpath}/{outfile}")

    return rt
