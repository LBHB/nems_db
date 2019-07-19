import pandas as pd
import numpy as np
import nems.plots.api as nplt
import nems.preprocessing as preproc
import matplotlib.pyplot as plt

def load(rec, **context):
    eval_conds=['A','B','C','I',['A','B'],['C','I']]
    return {'rec':  add_condition_epochs(rec), 'evaluation_conditions':eval_conds}

def add_condition_epochs(rec):
    df0=rec['resp'].epochs.copy()
    df2=rec['resp'].epochs.copy()
    df0['name']=df0['name'].apply(parse_stim_type)
    df0=df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])
    rec['resp'].epochs=df3
    return rec

def parse_stim_type(stim_name):
    stim_sep = stim_name.split('+')
    if len(stim_sep) == 1:
        stim_type = None
    elif stim_sep[1] == 'null':
        stim_type = 'B'
    elif stim_sep[2] == 'null':
        stim_type = 'A'
    elif stim_sep[1] == stim_sep[2]:
        stim_type = 'C'
    else:
        stim_type = 'I'
    return stim_type

def plot_all_vals_(modelspec, val, figures=None, IsReload=False, **context):
    if figures is None:
        figures = []
    if not IsReload:
        fig = plot_all_vals(val[0],modelspec)
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))

    return {'figures': figures}

def add_coherence_as_state(rec,permute=False,baseline=True,**context):
    coh = rec['resp'].epoch_to_signal('C')
    inc = rec['resp'].epoch_to_signal('I')
    if permute:
        coh = coh.shuffle_time(rand_seed=0,mask=rec['mask'])
        inc = inc.shuffle_time(rand_seed=1,mask=rec['mask'])
    rec = preproc.concatenate_state_channel(rec, coh, state_signal_name='state',generate_baseline=baseline)
    rec = preproc.concatenate_state_channel(rec, inc, state_signal_name='state')   
    rec = preproc.concatenate_state_channel(rec, coh, state_signal_name='state_raw',generate_baseline=baseline)
    rec = preproc.concatenate_state_channel(rec, inc, state_signal_name='state_raw')
    chans=['Coherent','Incoherent']
    if baseline:
        chans.insert(0,'baseline')
    rec.signals['state'].chans=chans
    rec.signals['state_raw'].chans=chans
    
    
    
#    rec['Coherence'] = rec['resp'].epoch_to_signal('C')
#    inc = rec['resp'].epoch_to_signal('I').as_continuous()
#    C = rec['Coherence'].as_continuous() + 2*inc
#    rec['Coherence']._data=C
#    rec = preproc.concatenate_state_channel(rec, rec['Coherence'], state_signal_name='state')
#    rec = preproc.concatenate_state_channel(rec, rec['Coherence'], state_signal_name='state_raw')
#    rec.signals['state'].chans=['baseline','Coherence']
#    rec.signals['state_raw'].chans=['baseline','Coherence']
    return {'rec': rec}

def plot_all_vals(val,modelspec,signames=['resp','pred'],channels=[0,0,1],subset=None,plot_singles_on_dual=False):
    #NOTE TO SELF: Not sure why channels=[0,0,1]. Setting it as default, but when called by plot_linear_and_weighted_psths it should be [0,0,0]
    from nems.plots.timeseries import timeseries_from_epoch
    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler
    signame_good_epochs='resp'
    if val[signames[0]].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    extracted = val[signame_good_epochs].extract_epoch(epochname)
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)
    
    epochs=val[signame_good_epochs].epochs
    epochs=epochs[epochs['name'] ==  epochname].iloc[occurrences]
    st_mask=val[signame_good_epochs].epochs['name'].str.contains('ST')
    inds=[]
    for index, row in epochs.iterrows():
        matchi = (val[signame_good_epochs].epochs['start'] == row['start']) & (val[signame_good_epochs].epochs['end'] == row['end'])
        matchi = matchi & st_mask
        inds.append(np.where(matchi)[0][0])
         
    names=val[signame_good_epochs].epochs['name'].iloc[inds].tolist()
    
    A=[];B=[];
    for name in names:
        nm=name.split('+')
        A.append(nm[1])
        B.append(nm[2])

    if subset is None:
        plot_order=['STIM_T+si464+null', 'STIM_T+null+si464', 'STIM_T+si464+si464',
                'STIM_T+si516+null', 'STIM_T+null+si516', 'STIM_T+si516+si516',
                'STIM_T+si464+si516', 'STIM_T+si516+si464']
        figsize=(10, 15)
    elif subset == 'C+I':
        plot_order=['STIM_T+si464+si464','STIM_T+si516+si516',
                'STIM_T+si464+si516', 'STIM_T+si516+si464']
        figsize=(19, 5)
    plot_order.reverse()
    order = np.array([names.index(nm) for nm in plot_order])
    names_short=[n.replace('STIM_T+','').replace('si464','1').replace('si516','2').replace('null','_') for n in names]
#    names2=sorted(names,key=lambda x: plot_order.index(x))
    
 #   idmap = dict((id,pos) for pos,id in enumerate(plot_order))
    
    sigs = [val[s] for s in signames]
    title = ''
    nplt=len(plot_order)
    gs_kw = dict(hspace=0,left=0.04,right=.99)
    fig, ax = plt.subplots(nrows=nplt, ncols=1, figsize=figsize,sharey=True,gridspec_kw=gs_kw)
    if signames==['resp','pred']:
        [axi.set_prop_cycle(cycler('color', ['k','#1f77b4', 'r']) + cycler('linestyle', ['-', '-', '--'])) for axi in ax]
    else:
        #['resp', 'lin_model']:
        [axi.set_prop_cycle(cycler('color', ['k','g','r']) + cycler(linestyle=['-', 'dotted','-'])+ cycler(linewidth=[1,2,1])) for axi in ax]
    allsigs =np.hstack([s.as_continuous()[-1,:] for s in sigs])    
    yl=[np.nanmin(allsigs), np.nanmax(allsigs)]
    prestimtime=val['stim'].epochs.loc[0].end
    for i in range(nplt):
        timeseries_from_epoch(sigs, epochname, title=title,
                         occurrences=occurrences[order[i]],ax=ax[i],channels=channels,linestyle=None,linewidth=None)
        if names_short[order[i]] in ['1+_','2+_']:
            #timeseries_from_epoch([val['stim']], epochname, title=title,
            #             occurrences=occurrences[order[i]],ax=ax[i])
            ep=val['stim'].extract_epoch(names[order[i]]).squeeze()
            ep=80+20*np.log10(ep.T)
            ep=ep/ep.max()*yl[1]
            time_vector = np.arange(0, len(ep)) / val['stim'].fs
            ax[i].plot(time_vector-prestimtime,ep,'--',color='#ff7f0e')
        if plot_singles_on_dual:
            snA=names_short[order[i]][:2]+'_'
            snB='_'+names_short[order[i]][1:]
            snA_=names[names_short.index(snA)]
            snB_=names[names_short.index(snB)]
            epA=val[signames[0]].extract_epoch(snA_).squeeze()
            epB=val[signames[0]].extract_epoch(snB_).squeeze()
            time_vector = np.arange(0, len(epA)) / val['resp'].fs
            ax[i].plot(time_vector-prestimtime,epA,'--',color=(1,.5,0),linewidth=1.5)
            ax[i].plot(time_vector-prestimtime,epB,'--',color=(0,.5,1),linewidth=1.5)
        ax[i].set_ylabel(names_short[order[i]],rotation=0,horizontalalignment='right',verticalalignment='bottom')

    if modelspec is not None:
        ax[0].set_title('{}: {}'.format(modelspec[0]['meta']['cellid'],modelspec[0]['meta']['modelname']))
    [axi.get_xaxis().set_visible(False) for axi in ax[:-1]]
    [axi.get_yaxis().set_ticks([]) for axi in ax]  
    [axi.get_legend().set_visible(False) for axi in ax[:-1]]
    [axi.set_xlim([.8-1, 4.5-1]) for axi in ax]
    yl_margin = .01*(yl[1]-yl[0])
    [axi.set_ylim((yl[0]-yl_margin, yl[1]+yl_margin)) for axi in ax]
    if plot_singles_on_dual:
        ls=[signames[0]+' A',signames[0]+' B']
    else:
        ls=['Stim']
    ax[nplt-1].legend(signames+ls)
    return fig

def plot_linear_and_weighted_psths(val,SR,signame='resp',weights=None,subset=None,addsig=None):
    #smooth and subtract SR
    import copy
    fn = lambda x : np.atleast_2d(smooth(x.squeeze(), 3, 2) - SR/val[signame].fs)
    val[signame]=val[signame].transform(fn)
    if addsig is not None:
        fn = lambda x : np.atleast_2d(smooth(x.squeeze(), 3, 2) - SR/val[addsig].fs)
        val[addsig]=val[addsig].transform(fn)
    lin_weights=[[1,1],[1,1]]
    epcs=val[signame].epochs[val[signame].epochs['name'] == 'PreStimSilence'].copy()
    epcs_offsets=[epcs['end'].iloc[0], 0]

    inp=copy.deepcopy(val[signame])
    out, l_corrs=generate_weighted_model_signals(inp,lin_weights,epcs_offsets)
    val[signame+'_lin_model']=out
    if weights is None:
        sigz=[signame,signame+'_lin_model']
        if addsig is not None:
           sigz.append(addsig)
        plot_singles_on_dual=True
        w_corrs=None
    else:
        val[signame+'_weighted_model'], w_corrs=generate_weighted_model_signals(val[signame],weights,epcs_offsets)
        sigz=[signame,signame+'_lin_model',signame+'_weighted_model']
        plot_singles_on_dual=False
    fh=plot_all_vals(val,None,signames=sigz,channels=[0,0,0],subset=subset,plot_singles_on_dual=plot_singles_on_dual)
    return fh, w_corrs, l_corrs

def plot_linear_and_weighted_psths_(batch,cellid,weights=None,subset=None):
    rec_file = nw.generate_recording_uri(cellid, batch, loadkey='env.fs100')
    rec=recording.load_recording(rec_file)
    rec['resp'] = rec['resp'].extract_channels([cellid])
    rec['resp'].fs=200
    
    SR=get_SR(rec)
    
    #COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    
    val['resp']=add_condition_epochs(val['resp'])

    fh, w_corrs, l_corrs = plot_linear_and_weighted_psths(val,SR,signame='resp',weights=weights,subset=subset,addsig='resp')
    return fh, w_corrs, l_corrs

def plot_linear_and_weighted_psths_model(modelspec, val, rec, figures=None, IsReload=False, **context):
    if figures is None:
        figures = []
    if not IsReload:
        SR=0    #SR=get_SR(rec)
        fig,_,_ = plot_linear_and_weighted_psths(val,SR,signame='pred',subset='C+I',addsig='resp')
        phi = modelspec[1]['phi']
        if 'g' in phi:
            g=phi['g'].copy()
            if g.shape[1]==2:
                gn=np.hstack((np.full((2,1),1),g))+1
            else:
                gn=g/g[:,:1] +1
            yl=fig.axes[0].get_ylim()
            th=fig.axes[0].text(fig.axes[0].get_xlim()[1], yl[1]+.2*np.diff(yl),
                '     gain  \nA: {: .2f} \nB: {: .2f} \nA-B: {: .2f} '.format(
                gn[0][2],gn[1][2],gn[0][2]-gn[1][2]),
                verticalalignment='top',horizontalalignment='right')
            th2=fig.axes[2].text(fig.axes[0].get_xlim()[1], yl[1]+0*np.diff(yl),
                '     gain  \nA: {: .2f} \nB: {: .2f} \nA-B: {: .2f} '.format(
                gn[0][1],gn[1][1],gn[0][1]-gn[1][1]),
                verticalalignment='top',horizontalalignment='right')
        fig.axes[0].set_title('{}: {}'.format(modelspec[0]['meta']['cellid'],modelspec[0]['meta']['modelname']))
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))

    return {'figures': figures}

def generate_weighted_model_signals(sig_in,weights,epcs_offsets):
    sig_in=sig_in.copy()
    sig_out=sig_in.copy()
    sig_out._data = np.full(sig_out._data.shape, np.nan)
    types=['C','I']
    epcs=sig_in.epochs[sig_in.epochs['name'].str.contains('STIM')].copy()
    epcs['type']=epcs['name'].apply(parse_stim_type)
    orig_epcs=sig_in.epochs.copy()
    sig_in.epochs['start']=sig_in.epochs['start']+epcs_offsets[0]
    sig_in.epochs['end']=sig_in.epochs['end']+epcs_offsets[1]
    EA=np.array([n.split('+')[1] for n in epcs['name']])
    EB=np.array([n.split('+')[2] for n in epcs['name']])
    corrs={}
    for _weights,_type in zip(weights,types):
        inds=np.nonzero(epcs['type'] == _type)[0]
        for ind in inds:
            r=sig_in.extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                if (len(indA) > 0) & (len(indB) > 0):
                    rA=sig_in.extract_epoch(epcs.iloc[indA[0]]['name'])
                    rB=sig_in.extract_epoch(epcs.iloc[indB[0]]['name'])
                    sig_out=sig_out.replace_epoch(epcs.iloc[ind]['name'],_weights[0]*rA+_weights[1]*rB,preserve_nan=False)
                    R=sig_out.extract_epoch(epcs.iloc[ind]['name'])
        
        
        ins=sig_in.extract_epochs(epcs.iloc[inds]['name'])
        ins=np.hstack([ins[k] for k in ins.keys()]).flatten()
        outs=sig_out.extract_epochs(epcs.iloc[inds]['name'])
        outs=np.hstack([outs[k] for k in outs.keys()]).flatten()
        ff = np.isfinite(ins) & np.isfinite(outs)    
        cc = np.corrcoef(ins[ff], outs[ff])
        corrs[_type]=cc[0,1]
    sig_in.epochs=orig_epcs.copy()
    sig_out.epochs=orig_epcs.copy()
    return sig_out, corrs

def get_SR(rec):
    epcs=rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    cellid=list(rec['resp']._data.keys())[0]
    spike_times=rec['resp']._data[cellid]
    count=0
    for index, row in epcs.iterrows():
        count+=np.sum((spike_times > row['start']) & (spike_times < row['end']))
    return count/(epcs['end']-epcs['start']).sum()
    
def smooth(x,window_len=11,passes=2,window='flat'):
    import numpy as np
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    
    
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=s 
    for passnum in range(passes):
        y=np.convolve(w/w.sum(),y,mode='valid')
    return y