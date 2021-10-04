import matplotlib.pyplot as plt
import numpy as np
import os
import io
import pandas as pd

#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils
import nems.db as nd
import nems.recording as recording
import nems.epoch as ep

#import nems.baphy as nb
import nems_lbhb.xform_wrappers as nw
import nems_lbhb.stateplots as sp

#params = {'legend.fontsize': 8,
#          'figure.figsize': (8, 6),
#          'axes.labelsize': 8,
#          'axes.titlesize': 8,
#          'xtick.labelsize': 8,
#          'ytick.labelsize': 8,
#          'pdf.fonttype': 42,
#          'ps.fonttype': 42}
#plt.rcParams.update(params)


def ev_pupil(cellid, batch, presilence=0.35):
    modelname = "psth.fs20.pup-st.pup.beh_stategain.3_init.st-basic"

    print('Finding recording for cell/batch {0}/{1}...'.format(cellid, batch))

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    # generate xfspec, which defines sequence of events to load data,
    # generate modelspec, fit data, plot results and save
    recording_uri = nw.generate_recording_uri(cellid, batch, loader)
    print(recording_uri)

    rec = recording.load_recording(recording_uri)
    pupil = rec['pupil']
    active = pupil.epoch_to_signal('ACTIVE_EXPERIMENT')
    active.chans = ['active']

    p = pupil.as_continuous()[0, :]
    b = active.as_continuous()[0, :]
    g = np.isfinite(p) & np.isfinite(b)
    cc_pup_beh = np.corrcoef(p[g], b[g])[0, 1]

    pshift = np.int(pupil.fs * 0.75)
    #pupil = pupil._modified_copy(np.roll(pupil._data, (0, pshift)))
    d = pupil._data
    pupil = pupil._modified_copy(np.roll(d / np.nanmax(d), (0, pshift)))
    #pupil = pupil._modified_copy(d / np.nanmax(d))
    #pupil = pupil._modified_copy(np.roll(d, (0, pshift)))

    trials = pupil.get_epoch_indices('TRIAL')
    targets = pupil.get_epoch_indices('TARGET')
    pt_blocks = pupil.get_epoch_indices('PURETONE_BEHAVIOR').tolist()
    easy_blocks = pupil.get_epoch_indices('EASY_BEHAVIOR').tolist()
    hard_blocks = pupil.get_epoch_indices('HARD_BEHAVIOR').tolist()
    passive_blocks = pupil.get_epoch_indices('PASSIVE_EXPERIMENT').tolist()
    behavior_blocks = pupil.get_epoch_indices('ACTIVE_EXPERIMENT')

    blocks=[]
    for p in passive_blocks:
        p.append('passive')
        blocks.append(p)
    for p in pt_blocks:
        p.append('puretone')
        blocks.append(p)
    for p in easy_blocks:
        p.append('easy')
        blocks.append(p)
    for p in hard_blocks:
        p.append('hard')
        blocks.append(p)

    blocks.sort()
    #print(blocks)
    trialbins = int(pupil.fs * 6)
    prebins = int(pupil.fs *presilence)

    ev=[]
    ev_prenorm=[]
    label=[]
    beh_lickrate=[]
    beh_lickrate_norm=[]
    beh_all = {}
    for block in blocks:
        k = block[-1]
        label.append(k)
        block_trials = ep.epoch_intersection(trials, np.array([block[0:2]]))
        tcount=block_trials.shape[0]
        for t in range(tcount):
            block_trials[t,1]=block_trials[t,0]+trialbins
            if block_trials[t,1]>pupil.shape[1]:
                block_trials[t,1]=pupil.shape[1]

        tev = pupil.extract_epoch(block_trials)[:,0,:]

        tev0 = np.nanmean(tev[:,:prebins],axis=1)
#        m = tev0 > 0.3 * np.nanmax(tev0)
#        print(block)
#        print("{}-{} mean {} ({}/{} big)".format(
#                np.nanmin(tev0),np.nanmax(tev0),np.nanmean(tev0),
#                np.sum(m), len(tev0)))
#        tev = tev[m, :]

        ev.append(tev)
        ev_prenorm.append(tev - np.mean(tev[:,:prebins],axis=1,keepdims=True))
        if k not in beh_all.keys():
            beh_all[k]=np.array([])
        beh_all[k] = np.append(beh_all[k], tev.ravel())
        beh_lickrate.append((k,np.nanmean(ev[-1],axis=0)))
        beh_lickrate_norm.append((k,np.nanmean(ev_prenorm[-1],axis=0)))

        #print("{}: {} trials, {} bins".format(k,tev.shape[0],tev.shape[1]))

    perf_blocks={'hits': pupil.get_epoch_indices('HIT_TRIAL'),
                 'misses': pupil.get_epoch_indices('MISS_TRIAL'),
                 'fas': pupil.get_epoch_indices('FA_TRIAL')}

    ev=[]
    ev_prenorm=[]
    perf_lickrate=[]
    perf_lickrate_norm=[]
    perf_all = {}
    for k,block in perf_blocks.items():

        block_trials = ep.epoch_intersection(trials, block)
        tcount=block_trials.shape[0]
        for t in range(tcount):
            block_trials[t,1]=block_trials[t,0]+trialbins
            if block_trials[t,1]>pupil.shape[1]:
                block_trials[t,1]=pupil.shape[1]
        t = pupil.extract_epoch(block_trials, allow_empty=True)
        if t.size:
            tev = t[:,0,:]
        else:
            tev = np.ones((1,trialbins)) * np.nan
        perf_all[k] = t.ravel()

        ev.append(tev)
        ev_prenorm.append(tev - np.mean(tev[:,:prebins],axis=1,keepdims=True))

        perf_lickrate.append((k,np.nanmean(ev[-1],axis=0)))
        perf_lickrate_norm.append((k,np.nanmean(ev_prenorm[-1],axis=0)))

        #print("{}: {} trials, {} bins".format(k,tev.shape[0],tev.shape[1]))

    return beh_lickrate, beh_lickrate_norm, beh_all, \
           perf_lickrate, perf_lickrate_norm, perf_all, cc_pup_beh


def ev_pupil_plot(ev_data_tuple_list, title=None, ax=None, fs=100,
                  prestimsilence=0.35, linecolors=None, fillcolors=None):
    if ax is None:
        fh=plt.figure()
        ax=plt.gca()
    trialbins=len(ev_data_tuple_list[0][1])
    tt=np.arange(trialbins) / fs - prestimsilence

    label=[]
    cc=0
    opt={}
    for k, ev in ev_data_tuple_list:
        label.append(k)
        if len(ev.shape)>1 and ev.shape[1]>1:
            if fillcolors is not None:
                opt = {'facecolor': fillcolors[cc], 'alpha': 0.5}
            mm=np.nanmean(ev,axis=1)
            ee=np.nanstd(ev,axis=1)/np.sqrt(ev.shape[1])
            plt.fill_between(tt, mm-ee, mm+ee, **opt)
        else:
            mm = ev
        if linecolors is not None:
            opt = {'color': linecolors[cc]}
        plt.plot(tt, mm, **opt)
        cc += 1
    plt.legend(label, loc='upper left', frameon=False)
    plt.xlabel('trial time (s)')
    plt.ylabel('pupil diameter')
    if title is not None:
        plt.title(title)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax


# being main script
save_figures = False

batch = 307
d=nd.get_batch_cells(batch=batch)
cellids=list(d['cellid'])

ucellids=[cellids[0]]
batches=[batch]
for c in cellids[1:]:
    site1=ucellids[-1].split("-")[0]
    site2=c.split("-")[0]
    if site1 != site2:
        ucellids.append(c)
        batches.append(batch)

batch=309
d = nd.get_batch_cells(batch=batch)
cellids = list(d['cellid'])

for c in cellids[1:]:
    site1 = ucellids[-1].split("-")[0]
    site2 = c.split("-")[0]
    if site1 != site2:
        ucellids.append(c)
        batches.append(batch)

cellids = ucellids
print(cellids)
print(batches)

siteids=[]
active=[]
passive=[]
active_full=[]
passive_full=[]
hard=[]
easy=[]
puretone=[]
hits=[]
fas=[]
misses=[]
hits_full=[]
fas_full=[]
misses_full=[]
presilence=0.35
fs=20
perf_all={}
beh_all={}
cc_pup_beh = []
for cellid,batch in zip(cellids,batches):

    beh_lickrate, beh_lickrate_norm, tbeh_all, \
    perf_lickrate, perf_lickrate_norm, tperf_all, ccpb = \
         ev_pupil(cellid, batch, presilence=presilence)

    siteids.append(cellid.split("-")[0])
    cc_pup_beh.append(ccpb)

    cc=0
    pset=[]
    aset=[]
    pfullset=[]
    afullset=[]
    for k, ev in beh_lickrate_norm:
        if k=='passive':
            pset.append(np.reshape(ev,[-1,1]))
            pfullset.append(np.reshape(beh_lickrate[cc][1],[-1,1]))
            kalt='passive'
        else:
            aset.append(np.reshape(ev,[-1,1]))
            afullset.append(np.reshape(beh_lickrate[cc][1],[-1,1]))
            if k=='hard':
                hard.append(np.reshape(ev,[-1,1]))
            elif k=='easy':
                easy.append(np.reshape(ev,[-1,1]))
            else:
                puretone.append(np.reshape(ev,[-1,1]))
            kalt='active'

        t = tbeh_all[k]
        t = t[np.isfinite(t)]
        if kalt in beh_all.keys():
            beh_all[kalt] = np.append(beh_all[kalt], t)
        else:
            beh_all[kalt] = t

        cc+=1

    passive.append(np.mean(np.concatenate(pset,axis=1),axis=1, keepdims=True))
    passive_full.append(np.mean(np.concatenate(pfullset,axis=1),axis=1, keepdims=True))
    active.append(np.mean(np.concatenate(aset,axis=1),axis=1, keepdims=True))
    active_full.append(np.mean(np.concatenate(afullset,axis=1),axis=1, keepdims=True))

    cc=0
    for k, ev in perf_lickrate_norm:
        if k=='hits':
            hits.append(np.reshape(ev,[-1,1]))
            hits_full.append(np.reshape(perf_lickrate[cc][1],[-1,1]))
        elif k=='misses':
            misses.append(np.reshape(ev,[-1,1]))
            misses_full.append(np.reshape(perf_lickrate[cc][1],[-1,1]))
        else:
            fas.append(np.reshape(ev,[-1,1]))
            fas_full.append(np.reshape(perf_lickrate[cc][1],[-1,1]))
        cc+=1

        t = tperf_all[k]
        t = t[np.isfinite(t)]
        if cellid==cellids[0]:
            perf_all[k] = t
        else:
            perf_all[k] = np.append(perf_all[k],t)

ylim1 = [0.46, 0.66]
ylim2 = [-0.04, 0.05]

f=plt.figure(figsize=(9,8))
ax=plt.subplot(3,3,1)
d=[('passive',np.concatenate(passive_full,axis=1)),
   ('active',np.concatenate(active_full,axis=1))]
ev_pupil_plot(d, title="beh vs. pupil", fs=fs, prestimsilence=presilence,
              ax=ax,
              linecolors=[sp.line_colors['passive'],sp.line_colors['active']],
              fillcolors=[sp.line_colors['passive'],sp.line_colors['active']])
ax.set_ylim(ylim1)

ax=plt.subplot(3,3,2)
d=[('passive',np.concatenate(passive,axis=1)),
   ('active',np.concatenate(active,axis=1))]
ev_pupil_plot(d, title="beh vs. d_pupil", fs=fs, prestimsilence=presilence,
              ax=ax,
              linecolors=[sp.line_colors['passive'],sp.line_colors['active']],
              fillcolors=[sp.line_colors['passive'],sp.line_colors['active']])
ax.set_ylim(ylim2)

prebins=int(presilence*fs)
print(prebins)
ev_start=int(presilence*fs)
ev_end=int((presilence+3)*fs)
print("Averaging evoked response over {}-{} sec".format(ev_start/fs-presilence, ev_end/fs-presilence))

passive_d=np.mean(np.concatenate(passive,axis=1)[ev_start:ev_end,:],axis=0)
passive_0=np.mean(np.concatenate(passive_full,axis=1)[:prebins,:],axis=0)

active_d=np.mean(np.concatenate(active,axis=1)[ev_start:ev_end,:],axis=0)
active_0=np.mean(np.concatenate(active_full,axis=1)[:prebins,:],axis=0)
print(passive_d.shape)
print(active_d.shape)

ax=plt.subplot(3,3,3)
for i in range(len(passive_0)):
    plt.plot([passive_0[i],active_0[i]],[passive_d[i],active_d[i]],
             color='darkgray',linewidth=0.5)

h1,=plt.plot(passive_0,passive_d,'.',markersize=10, label='passive',
             mfc=sp.fill_colors['passive'], mec=sp.fill_colors['passive'])
h2,=plt.plot(active_0,active_d,'.',markersize=10, label='active',
             mfc=sp.fill_colors['active'], mec=sp.fill_colors['active'])
plt.xlabel('pre-trial pupil diameter')
plt.ylabel('evoked pupil change')
plt.legend(handles=(h1,h2), frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax=plt.subplot(3,3,5)
d=[('hard',np.concatenate(hard,axis=1)),
   ('easy',np.concatenate(easy,axis=1)),
   ('puretone',np.concatenate(puretone,axis=1))]
lc=[sp.line_colors['hard'],sp.line_colors['easy'],sp.line_colors['puretone']]
fc=[sp.fill_colors['hard'],sp.fill_colors['easy'],sp.fill_colors['puretone']]

ev_pupil_plot(d, title="difficulty vs. d_pupil", fs=fs, prestimsilence=presilence,
              linecolors=lc, fillcolors=lc, ax=ax)
ax.set_ylim(ylim2)

ax=plt.subplot(3,3,6)
b = np.linspace(0,1,31)
w = (b[1]-b[0])/2
x = b[1:]-w
p = np.histogram(beh_all['passive'], bins=b)[0]
p = p / np.sum(p)
a = np.histogram(beh_all['active'], bins=b)[0]
a = a / np.sum(a)

ax.bar(x-w/2, p, width=w, color=sp.fill_colors['passive'])
ax.bar(x+w/2, a, width=w, color=sp.fill_colors['active'])

lc=[sp.line_colors['false_alarm'],sp.line_colors['hit'],sp.line_colors['miss']]
fc=[sp.fill_colors['false_alarm'],sp.fill_colors['hit'],sp.fill_colors['miss']]

ax=plt.subplot(3,3,7)
d=[('fas',np.concatenate(fas_full,axis=1)),
   ('hits',np.concatenate(hits_full,axis=1)),
   ('misses',np.concatenate(misses_full,axis=1))]
ev_pupil_plot(d, title="performance vs. d_pupil", fs=fs, prestimsilence=presilence,
              linecolors=lc, fillcolors=lc, ax=ax)
ax.set_ylim(ylim1)

ax=plt.subplot(3,3,8)
d=[('fas',np.concatenate(fas,axis=1)),
   ('hits',np.concatenate(hits,axis=1)),
   ('misses',np.concatenate(misses,axis=1))]
ev_pupil_plot(d, title="performance vs. pupil", fs=fs, prestimsilence=presilence,
              linecolors=lc, fillcolors=lc, ax=ax)
ax.set_ylim(ylim2)


prebins=int(presilence*fs)
evbins=prebins+int(fs)  # look at first second post trial start

hits_d=np.mean(np.concatenate(hits,axis=1)[ev_start:ev_end,:],axis=0)
hits_0=np.mean(np.concatenate(hits_full,axis=1)[:prebins,:],axis=0)
misses_d=np.mean(np.concatenate(misses,axis=1)[ev_start:ev_end,:],axis=0)
misses_0=np.mean(np.concatenate(misses_full,axis=1)[:prebins,:],axis=0)
fas_d=np.mean(np.concatenate(fas,axis=1)[ev_start:ev_end,:],axis=0)
fas_0=np.mean(np.concatenate(fas_full,axis=1)[:prebins,:],axis=0)

print(hits_d.shape)
print(fas_d.shape)

ax=plt.subplot(3,3,9)
#for i in range(len(passive_0)):
#    plt.plot([passive_0[i],active_0[i]],[passive_d[i],active_d[i]],color='darkgray',linewidth=0.5)

h3,=plt.plot(fas_0, fas_d,'.',markersize=12, label='fas',
             mfc=sp.fill_colors['false_alarm'], mec=sp.fill_colors['false_alarm'])
h2,=plt.plot(misses_0, misses_d,'.',markersize=10, label='misses',
             mfc=sp.fill_colors['miss'], mec=sp.fill_colors['miss'])
h1,=plt.plot(hits_0, hits_d,'.',markersize=10, label='hits',
             mfc=sp.fill_colors['hit'], mec=sp.fill_colors['hit'])
plt.xlabel('pre-trial pupil diameter')
plt.ylabel('evoked pupil change')
plt.legend(handles=(h1,h2,h3), frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()

if save_figures:
    f.savefig('/auto/users/svd/projects/pupil-behavior/PTD_ev_pupil_sharpened.pdf')
    f.savefig('/auto/users/svd/projects/pupil-behavior/PTD_ev_pupil_sharpened.png')

    df=pd.DataFrame(data=np.vstack([passive_0, active_0, passive_d, active_d, cc_pup_beh]).T,
                    columns=['baseline_p','baseline_a','evoked_p','evoked_a', 'cc_pup_beh'],
                    index=siteids)
    df.to_csv('/auto/users/svd/projects/pupil-behavior/PTD_ev_pupil.per_site.csv')
