import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems.preprocessing import make_state_signal



def sort_refs(refs):
    idx = np.argsort([int(r.strip('STIM_')[0]) for r in refs])
    return np.array(refs)[idx].tolist()

def sort_targets(targets):
    """
    sort target epoch strings by freq, then by snr, then by targets tag (N1, N2 etc.)
    """
    f = []
    snrs = []
    labs = []
    try:
        for t in targets:
            f.append(int(t.strip('TAR_').strip('CAT_').split('+')[0]))
            snr = t.split('+')[1].split('dB')[0]
            if snr=='Inf': snr=np.inf
            elif snr=='-Inf': snr=-np.inf
            else: snr=int(snr)
            snrs.append(snr)
            try:
                labs.append(int(t.split('+')[-1].split(':N')[-1]))
            except:
                labs.append(np.nan)
        tar_df = pd.DataFrame(data=np.stack([f, snrs, labs]).T, columns=['freq', 'snr', 'n']).sort_values(by=['freq', 'snr', 'n'])
        sidx = tar_df.index
        return np.array(targets)[sidx].tolist()
    except:
        return targets

def sort_refs(refs):
    """
    Sort ref strings by frequency
    """
    idx = np.argsort([int(f.split('STIM_')[1]) for f in refs])
    return np.array(refs)[idx].tolist()


def get_snrs(targets):
    """
    return list of snrs for each target
    """
    snrs = []
    for t in targets:
        snr = t.split('+')[1].split('dB')[0]
        if snr=='Inf': snr=np.inf 
        elif snr=='-Inf': snr=-np.inf
        else: snr=int(snr)
        snrs.append(snr)
    return snrs


def get_tar_freqs(targets):
    """
    return list of target freqs
    """
    return [int(t.strip('TAR_').strip('CAT_').split('+')[0]) for t in targets]

def get_freqs(targets):
    """
    return list of target freqs
    """
    return [int(t.strip('STIM_').strip('TAR_').strip('CAT_').split('+')[0]) for t in targets]

def make_tbp_colormaps(ref_stims=None, tar_stims=None, use_tar_freq_idx=0):

    if ref_stims is None:
       N_ref = 256
       N_tar = 256
       mid_ref = 128
    else:
       N_ref = len(ref_stims)
       N_tar = len(tar_stims)

       tar_freqs=get_freqs(tar_stims)
       ref_freqs=get_freqs(ref_stims)

       mid_ref = np.abs(np.array(ref_freqs) - tar_freqs[use_tar_freq_idx]).argmin()
       #mid_ref = np.where(np.array([tar_freqs[use_tar_freq_idx]==r 
       #                             for r in ref_freqs]))[0][0]

       #print(tar_freqs,ref_freqs,mid_ref)

    #print(f"N_ref={N_ref} mid_ref={mid_ref} N_tar={N_tar}") 
    vals = np.ones((N_ref, 4))
    mid_gray = 210/256
    vals[:mid_ref, 0] = np.linspace((1-mid_gray), mid_gray, mid_ref)
    vals[:mid_ref, 1] = np.linspace((1-mid_gray), mid_gray, mid_ref)
    vals[:mid_ref, 2] = np.linspace(mid_gray, mid_gray, mid_ref)
    vals[mid_ref:, 0] = np.linspace(mid_gray, 0, N_ref-mid_ref)
    vals[mid_ref:, 1] = np.linspace(mid_gray, 168/256, N_ref-mid_ref)
    vals[mid_ref:, 2] = np.linspace(mid_gray, 0, N_ref-mid_ref)
    BwG = ListedColormap(vals, 'BwG')
    
    vals = np.ones((N_tar, 4))
    vals[:, 0] = np.linspace(mid_gray, 1, N_tar)
    vals[:, 1] = np.linspace(mid_gray, 1-mid_gray, N_tar)
    vals[:, 2] = np.linspace(mid_gray, 1-mid_gray, N_tar)
    gR = ListedColormap(vals, 'gR')

    return BwG, gR

BwG, gR = make_tbp_colormaps()

def compute_ellipse(x, y):
    inds = np.isfinite(x) & np.isfinite(y)
    x= x[inds]
    y = y[inds]
    data = np.vstack((x, y))
    mu = np.mean(data, 1)
    data = data.T - mu
    D, V = np.linalg.eig(np.divide(np.matmul(data.T, data), data.shape[0] - 1))
    # order = np.argsort(D)[::-1]
    # D = D[order]
    # V = abs(V[:, order])
    t = np.linspace(0, 2 * np.pi, 100)
    e = np.vstack((np.sin(t), np.cos(t)))  # unit circle
    VV = np.multiply(V, np.sqrt(D))  # scale eigenvectors
    e = np.matmul(VV, e).T + mu  # project circle back to orig space
    e = e.T
    return e

def load_tbp_recording(siteid, batch, **options):
    options['resp']=True
    options['pupil']=options.get('pupil', True)
    options['rasterfs']=options.get('rasterfs', 50)
    
    manager = BAPHYExperiment(siteid=siteid, batch=batch)
    rec = manager.get_recording(**options)
    rec['resp'] = rec['resp'].rasterize()

    onsetsec = 0.1
    offsetsec = 0.4

    onset = int(onsetsec * options['rasterfs'])
    offset = int(offsetsec * options['rasterfs'])

    # PCA on trial averaged responses
    ref_stims, sounds, all_sounds = get_sound_labels(rec)
    """
    try:
        targets = [f for f in rec['resp'].epochs.name.unique() if 'TAR_' in f]
        catch = [f for f in rec['resp'].epochs.name.unique() if 'CAT_' in f]

        sounds = targets + catch

        ref_stims = [x for x in rec['resp'].epochs.name.unique() if 'STIM_' in x]
        idx = np.argsort([int(s.split('_')[-1]) for s in ref_stims])
        ref_stims = np.array(ref_stims)[idx].tolist()

        sounds=sort_targets(sounds)
    except:
        ref_stims=['REFERENCE']
        sounds = ['TARGET']

    all_sounds = ref_stims + sounds
    """
    rall = rec.and_mask(['ACTIVE_EXPERIMENT', 'PASSIVE_EXPERIMENT'])

    # can't simply extract evoked for refs because can be longer/shorted if it came after target
    # and / or if it was the last stim.So, masking prestim / postim doesn't work.Do it manually
    d = rall['resp'].extract_epochs(all_sounds, mask=rall['mask'])
    d = {k: v[~np.isnan(v[:, :, onset:offset].sum(axis=(1, 2))), :, :] for (k, v) in d.items()}
    d = {k: v[:, :, onset:offset] for (k, v) in d.items()}

    Rall_u = np.vstack([d[k].sum(axis=2).mean(axis=0) for k in d.keys()])

    pca = PCA(n_components=2)
    pca.fit(Rall_u)
    pc_axes = pca.components_

    # project onto first PC and plot trial psth
    rec['pc1'] = rec['resp']._modified_copy(rec['resp']._data.T.dot(pc_axes.T).T[[0], :])
    rec['pc2'] = rec['resp']._modified_copy(rec['resp']._data.T.dot(pc_axes.T).T[[1], :])
    
    #rec = make_state_signal(rec, state_signals=['pupil','active'], permute_signals=['pupil'])
    rec = make_state_signal(rec, state_signals=['pupil','active'])

    return rec


def get_sound_labels(rec):

    targets = [f for f in rec['resp'].epochs.name.unique() if 'TAR_' in f]
    catch = [f for f in rec['resp'].epochs.name.unique() if 'CAT_' in f]
    tar_stims = targets + catch

    ref_stims = [x for x in rec['resp'].epochs.name.unique() if 'STIM_' in x]
    idx = np.argsort([int(s.split('_')[-1]) for s in ref_stims])
    ref_stims = np.array(ref_stims)[idx].tolist()
    
    tar_stims=sort_targets(tar_stims)
    all_stims = ref_stims + tar_stims
    
    return ref_stims, tar_stims, all_stims


def pb_regress(rec):

    r = rec.copy()
    r = r.and_mask(['HIT_TRIAL','CORRECT_REJECT_TRIAL', 'PASSIVE_EXPERIMENT'])

    ref_stims, tar_stims, all_stims = get_sound_labels(r)

    cellids = r['resp'].chans
    cellcount = len(cellids)
    states = r['state'].chans
    statecount = r['state'].shape[0]
    stimcount=len(all_stims)
    
    rr= slice(int(0.1*r['resp'].fs), int(0.4*r['resp'].fs))

    betas = np.zeros((cellcount, statecount, stimcount))
    for i, k in enumerate(all_stims):
        resp = r['resp'].extract_epoch(k, mask=r['mask'])
        state = r['state'].extract_epoch(k, mask=r['mask'])
        g = np.isfinite(resp[:,0,10]) & np.isfinite(state[:,0,19])
        resp = np.mean(resp[g,:,rr],axis=2)
        state = np.mean(state[g,:,rr],axis=2)
        if np.std(state[:,0]-state[:,2])==0:
            print(f'skipping {k}')
        else:
            #state[:,1]= 0
            for c in range(cellcount):
                beta_hat = np.linalg.lstsq(state,resp[:,c])[0]
                betas[c,:,i]=beta_hat

    return betas, cellids, states, all_stims


def plot_average_psths(rec):

    # get some high-level properties about the recording
    fs=rec['resp'].fs
    onsetsec = 0.1
    offsetsec = 0.1

    onset = int(onsetsec * rec['resp'].fs)
    offset = int(offsetsec * rec['resp'].fs)
    stim_len = int(0.5*fs)

    ref_stims, sounds, all_sounds = get_sound_labels(rec)

    cellids = rec['resp'].chans
    siteid = cellids[0].split("-")[0]
    e=rec['resp'].epochs
    r_active = rec.copy().and_mask(['HIT_TRIAL','CORRECT_REJECT_TRIAL'])
    r_passive = rec.copy().and_mask(['PASSIVE_EXPERIMENT'])
    r_miss = rec.copy().and_mask(['MISS_TRIAL'])
    stim_len = int(0.5*rec['resp'].fs)

    conditions = ['active correct','passive', 'miss']
    cond_recs = [r_active, r_passive, r_miss]

    cellcount = len(cellids)
    celloffset = 0

    f,ax=plt.subplots(cellcount+2, 4, figsize=(4,(cellcount+2)*0.4), sharex=True, sharey='row')

    ii=0
    #cmaps = [plt.cm.get_cmap('viridis', len(ref_stims)+2),
    #         plt.cm.get_cmap('Reds', len(sounds)+1)]
    cmaps=make_tbp_colormaps(ref_stims, sounds)

    for cat,labels,colors in zip(['noise','target'],[ref_stims,sounds],cmaps):
        #tar = int(cat=='target')

        for to,r in zip(['Passive','Active'],[r_passive, r_active ]):

            act = int(to=='Active')
            mean_cell = np.zeros((cellcount,len(labels), stim_len))

            for jj in range(cellcount):

                c = jj+celloffset
                cellid=cellids[c]

                for i,k in enumerate(labels):
                    try:
                        p1 = r['resp'].extract_epoch(k, mask=r['mask'])
                        g = np.isfinite(p1[:,c,10])
                        x = np.nanmean(p1[g,c,:stim_len], axis=0) * fs
                        tt = np.arange(x.shape[0])/r['resp'].fs - onsetsec
                        ax[jj,ii].plot(tt, x, color=colors(i), linewidth=0.5, label=k)
                        mean_cell[jj,i,:] = x

                    except:
                        #print(f'no matches for {k}')
                        mean_cell[jj,i,:] = np.nan
                        pass

                if (ii==0):
                    cid="-".join(cellid.split("-")[1:])
                    ax[jj,0].text(tt[0],np.max(mean_cell[jj,:,:]), cid, fontsize=8)

            mean_cell = np.mean(mean_cell,axis=0)
            for i,k in enumerate(labels):
                ax[cellcount,ii].plot(tt, mean_cell[i,:], color=colors(i), linewidth=0.5, label=k)
            if ii==0:
                ax[cellcount,0].set_ylabel('mean', fontsize=8)

            ax[0,ii].set_title(f"{siteid} {to} {cat}")
            ii += 1

    #ax[cellcount-1,3].legend();
    for xx, k in zip(range(len(ref_stims)), ref_stims):
        _k = k.split("_")[1]
        ax[cellcount+1,0].text(xx*(tt[-1]-tt[0])/len(ref_stims)+tt[0], 0, _k, fontsize=8, color=cmaps[0](xx),ha='center',va='bottom',rotation=90)
        #print((xx, 0, k, cmaps[0](xx)))

    for yy, k in zip(range(len(sounds)), sounds):
        _k = k.split("_")[1].split("+")[0]
        ax[cellcount+1,0].text(0, yy+1, k, fontsize=8, color=cmaps[1](yy),ha='center',va='bottom')
        #print((tt[0], yy+1, k, cmaps[1](yy)))

    ax[cellcount+1,0].set_ylim([0,len(sounds)]);
    for a in ax[cellcount+1,:]:
        a.set_axis_off()

    return f


def site_tuning_avg(rec, betas):
    f,ax = plt.subplots(2,2,figsize=(12,6),sharey='row')
    ref_stims, tar_stims, all_stims = get_sound_labels(rec)
    for ii,rr in enumerate([slice(len(ref_stims)),slice(len(ref_stims),len(all_stims))]):
        if ii==0:
            cat='REF'
        else:
            cat='TAR'
        mean_ap = np.stack([betas[:, 0, rr], betas[:, 0, rr]+betas[:, 2, rr], betas[:,2,rr]], axis=0)
        pop_mean_ap = np.mean(mean_ap,axis=1)
        sns.stripplot(data=mean_ap[2], dodge=True, edgecolor='white', linewidth=0.5,
                      marker='o', size=5, ax=ax[0,ii])
        ax[0,ii].plot(pop_mean_ap[2])
        ax[0,ii].set_xticklabels([])

        ax[1,ii].plot(pop_mean_ap.T)
        ax[1,ii].set_xticks(np.arange(pop_mean_ap.shape[1]))
        ax[1,ii].set_xticklabels([s.split("+")[0] for s in all_stims[rr]], rotation=90)

    ax[1,1].legend(('pas','act','diff'))
    ax[0,0].set_ylabel('fraction change')

    return f


def site_tuning_curves(rec, betas):
    ref_stims, tar_stims, all_stims = get_sound_labels(rec)
    cellids = rec['resp'].chans
    siteid = cellids[0].split("-")[0]

    # tuning curve, passive vs. active
    cellcount = len(cellids)
    stimcount=betas.shape[2]

    cat_idx = np.array([c for c in range(len(tar_stims)) if tar_stims[c].startswith("CAT")])
    tar_idxs = np.array([c for c in range(len(tar_stims)) if not tar_stims[c].startswith("CAT")])

    tar_freqs = [s.split("_")[1].split("+")[0] for s in tar_stims]
    tar_idx = [t for t in range(len(tar_stims)) if '0dB' in tar_stims[t]][0]
    tar_freq = tar_freqs[tar_idx]
    cat_freqs = [tar_freqs[c] for c in cat_idx]
    tar_freq_idx = np.where([tar_freq in s for s in ref_stims])[0][0]


    rows=int(np.ceil(cellcount/3))
    f2,ax2=plt.subplots(rows, 6, figsize=(6,rows*1))

    split_by='behavior'
    #split_by='pupil'

    for c,cellid in enumerate(cellids):
        _i = int(np.floor(c/rows))
        _j = int(c % rows)

        # ref:
        rr = slice(len(ref_stims))
        if split_by=='behavior':
            mean_ap = [betas[c, 0, rr], betas[c, 0, rr]+betas[c, 2, rr]]
        else:
            mean_ap = [betas[c, 0, rr] - betas[c, 1, rr]/2, betas[c, 0, rr] + betas[c, 1, rr]/2]            

        a = ax2[_j, _i*2]
        a.plot(mean_ap[0],color='lightgray')
        a.plot(mean_ap[1],color='#11bb11')
        a.set_xlim([-2, len(ref_stims)])

        # tar
        rr = slice(len(ref_stims),stimcount)
        if split_by=='behavior':
            mean_ap = [betas[c, 0, rr], betas[c, 0, rr]+betas[c, 2, rr]]
        else:
            mean_ap = [betas[c, 0, rr] - betas[c, 1, rr]/2, betas[c, 0, rr] + betas[c, 1, rr]/2]            

        a = ax2[_j, _i*2+1]
        for c,c1 in zip(cat_idx, list(cat_idx[1:])+[len(tar_stims)]):
            a.plot(np.arange(c,c1),mean_ap[0][c:c1],color='lightgray')
            a.plot(np.arange(c,c1),mean_ap[1][c:c1],color='red')

        a.plot(cat_idx,mean_ap[0][cat_idx],'.',color='lightgray')
        a.set_xlim([-1, len(tar_stims)])

        mm=np.max([ax2[_j, _i*2].get_ylim(), ax2[_j, _i*2+1].get_ylim()])
        for a in [ax2[_j, _i*2], ax2[_j, _i*2+1]]:
            a.set_ylim([-mm/10,mm*1.1])
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_axis_off()

        ax2[_j, _i*2].plot([-2, -2],[0, mm],'k',linewidth=4)
        ax2[_j, _i*2].plot([tar_freq_idx, tar_freq_idx],[0, mm],'k--',linewidth=0.5)
        cid="-".join(cellid.split("-")[1:])
        ax2[_j, _i*2].text(0,mm,cid,fontsize=8)

        if (_j==0) and (_i==2):
            for c,cx,cf in zip(cat_idx,list(cat_idx[1:]-1)+[len(tar_stims)-1],cat_freqs):
                ax2[_j, _i*2+1].text(cx,mean_ap[0][cx],cf,fontsize=8,va='bottom',ha='center')

    plt.suptitle(f"{siteid} split by {split_by}");
    if split_by=='behavior':
        ax2[0,-1].legend(['pas','act'], frameon=False)
    else: 
        ax2[0,-1].legend(['sm','lg'], frameon=False)

    return f2
