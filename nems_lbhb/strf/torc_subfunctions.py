import numpy as np
from numpy.matlib import repmat
import scipy as sp
from matplotlib import pyplot as plt


def interpft(x,ny,dim=0):
    '''
    Function to interpolate using FT method, based on matlab interpft()
    :param x: array for interpolation
    :param ny: length of returned vector post-interpolation
    :param dim: performs interpolation along dimension DIM, default 0
    :return: interpolated data
    '''

    if dim >= 1:                                         #if interpolating along columns, dim = 1
        x = np.swapaxes(x,0,dim)                         #temporarily swap axes so calculations are universal regardless of dim
    if len(x.shape) == 1:                                #interpolation should always happen along same axis ultimately
        x = np.expand_dims(x,axis=1)

    siz = x.shape
    [m, n] = x.shape

    a = np.fft.fft(x,m,0)
    nyqst = int(np.ceil((m+1)/2))
    b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)

    if np.remainder(m,2)==0:
        b[nyqst,:] = b[nyqst,:]/2
        b[nyqst+ny-m,:] = b[nyqst,:]

    y = np.fft.irfft(b,b.shape[0],0)
    y = y * ny / m
    y = np.reshape(y, [y.shape[0],siz[1]])
    y = np.squeeze(y)

    if dim >= 1:                                        #switches dimensions back here to get desired form
        y = np.swapaxes(y,0,dim)

    return y



def insteadofbin(resp,binsize,mf=1):
    '''
    Downsample spike histogram from resp (resolution mf) to resolution by binsize(ms)
    :param resp: response data (comes from spikeperiod
    :param binsize: calculated size of bins (basep/stimtime = binsize)
    :param mf: multiplication factor, default 1
    :return: returns downsampled spike histogram
    '''

    if len(resp.shape) >= 2:
        [spikes,records] = resp.shape
    else:
        resp = np.expand_dims(resp, axis=1)
        [spikes,records] = resp.shape

    outlen = int(spikes / binsize / mf)

    if outlen - np.floor(outlen/1) * 1 > 1:
        print('Non-integer # bins. Result may be distorted')

    outlen = np.round(outlen)
    dsum = np.zeros([outlen,records])

    for rec in range(records):
        temprec = np.fft.fft(resp[:,rec])             #fft for each

        if outlen % 2 == 0:                           #if even length, create middle point
            temprec[np.ceil((outlen-1)/2)+1] = np.abs(temprec[np.ceil(outlen-1)/2]+1)

        dsum[:, rec] = np.fft.ifft(np.concatenate((temprec[0:int(np.ceil((outlen - 1) / 2) + 1)], np.conj(
            np.flipud(temprec[1:int(np.floor((outlen - 1) / 2) + 1)]))))).real

    return dsum



def get_strf_tuning(strf0, strfemp, StimParams):

    # figure out some tuning properties
    maxoct = int(np.log2(StimParams['hfreq']/StimParams['lfreq']))
    stepsize2 = maxoct / strf0.shape[0]
    fs=1000

    smooth = [100,strf0.shape[1]]
    strfsmooth = interpft(strf0, smooth[0], 0)
    strfempsmooth = interpft(strfemp, smooth[0], 0)

    ff = np.exp(np.linspace(np.log(StimParams['lfreq']),np.log(StimParams['hfreq']),strfsmooth.shape[0]))

    mm = np.mean(strfsmooth[:,:7] * (1*(strfsmooth[:,:7] > 0)), 1)
    if sum(np.abs(mm)) > 0:
        bfidx = int(sum(((mm == np.max(mm)).ravel().nonzero())))
        bf = np.round(ff[bfidx])
        bfshiftbins = (maxoct / 2 - np.log2(bf / StimParams['lfreq'])) / stepsize2
    else:
        bfidx = 1
        bf = 0
        bfshiftbins = 0

    mmneg = np.mean(strfsmooth[:,:7] * (1*(strfsmooth[:,:7] < 0)), 1)
    if sum(np.abs(mm)) > 0:
        wfidx = int(sum(((mmneg == np.min(mmneg)).ravel().nonzero())))
        wf = np.round(ff[wfidx])
        wfshiftbins = (maxoct / 2 - np.log2(wf / StimParams['lfreq'])) / stepsize2
    else:
        wfidx = 1
        wf = 0
        wfshiftbins = 0

    if -mmneg[wfidx] > mm[bfidx]:
        # If stronger negative component, calculate latency with neg
        shiftbins = wfshiftbins
        irsmooth = -interpft(strfsmooth[wfidx, :], 250)
        irempsmooth = interpft(strfempsmooth[wfidx], 250, 0)
    else:
        # Use positives
        shiftbins = bfshiftbins
        irsmooth = interpft(strfsmooth[bfidx, :], 250)
        irempsmooth = interpft(strfempsmooth[bfidx], 250)

    mb = 0
    # Find significantly modulated time bins
    sigmod = np.asarray((irsmooth-mb > irempsmooth*2).ravel().nonzero())
    sigmod = sigmod[np.logical_and(sigmod>=7,sigmod<124)]

    if len(sigmod) > 3:
        latbin = sigmod[0]
        dd = np.concatenate([np.diff(sigmod),[41]])
        durbin = sigmod[np.min((dd[0:] > 40).ravel().nonzero())]
        lat = int(np.round(latbin * 1000 / fs))
        offlat = int(np.round(durbin * 1000 / fs))
        #print("onset/offset latency:", lat, offlat)
    else:
        latbin = 0
        lat = 0
        durbin = 0
        offlat = 0
        print('no significant onset latency\n')

    return bf, lat, offlat, bfidx*strf0.shape[0]/100, \
           latbin*strf0.shape[1]/250, durbin*strf0.shape[1]/250


def strfplot(strf0, lfreq, tleng, smooth=0, noct=5, axs=None):
    '''
    Plots STRF using smoothing and interpolation
    :param strf0: strf data
    :param lfreq: lowest stimulus frequency
    :param tleng: length of a stimulus
    :param smooth: default 0, smoothing factor
    :param noct: number of octaves
    :param siglev: not used right now
    :param axs: default None, tell us what axis we're going to be on
    :return: list of frequencies to populate axis ticks, useful in subsequent plots
    '''
    if axs == None:
        fig, axs = plt.subplots()

    if smooth:
        if smooth == 1:
            smooth = [100, 250]
        strfdata = interpft(interpft(strf0, smooth[0], 0), smooth[1], 1)

    else:
        strfdata = strf0

<<<<<<< HEAD
    axs.imshow(strfdata, cmap=None, norm=None, aspect='auto', extent=[0, tleng, 0, noct], origin='lower', )
=======
    axs.imshow(strfdata, cmap=None, norm=None, aspect='auto', extent=[0, tleng, 0, noct], origin='lower')
>>>>>>> 84f0ee15ff44ef5ce02ebbadbbd841b990728629

    if lfreq:
        freqappend = lfreq
        freqticks = []
        for fff in range(noct+1):
            if fff != 0:
                freqappend = freqappend * 2
            freqticks.append(freqappend)
        axs.set_yticks(np.arange(noct))
        axs.set_yticklabels(freqticks)

    return freqticks,strfdata



def strf_torc_pred(stim, strfest):
    '''
    Small function to make a prediction of strf
    :param stim: strf data
    :param strfest: stimulus data
    :return: predicted strf
    '''
    [stimX,stimT,numstim] = stim.shape
    pred = np.zeros((stimT, numstim))

    for rec in range(numstim):
        for X in range(stimX):
            tr = (np.fft.ifft(np.fft.fft(stim[X, :, rec]) * np.fft.fft(strfest[X, :]))).real

            pred[:, rec] = pred[:, rec] + tr

    return pred



def makepsth(dsum,fhist,startime,endtime,mf=1):
    '''
    Creates a period histogram according to period implied by input freq fhist
    :param dsum: spike data
    :param fhist: the frequency for which the histogram is performed
    :param startime: the start of the histogram data (ms)
    :param endtime: the end of the histogram data (ms)
    :param mf: multiplication factor
    :return: psth
    '''
    if fhist == 0:
        fhist = 1000/(endtime-startime)
    dsum = dsum[:]

    period = int(1000 * (1/fhist) * mf)
    startime = startime * mf
    endtime = endtime * mf

    if endtime > len(dsum):
        endtime = len(dsum)

    fillmax = int(np.ceil(endtime/period) * period)
    if fillmax > endtime:
        dsum[endtime+1:fillmax] = np.nan
        endtime = fillmax

    dsum[:startime] = np.nan
    repcount = int(fillmax / period)
    dsum = np.reshape(dsum[:endtime],(period,repcount),order='F')

    wrapdata = np.nansum(dsum,1)
    cnorm =  np.sum(np.logical_not(np.isnan(dsum)),1)

    return wrapdata,cnorm



def get_snr(spdata,stim,basep,mf,allrates):
    '''
    Gets snr using spike data given in spdata, spike data and stims should include all inverse-repeat pairs of TORCs
    valid only with TORC stimuli
    :param spdata: spike data
    :param stim: stimulus data, scaled
    :param basep: length of base
    :param mf:multiplication factor
    :param allrates: list of all rates used in TORCs
    :return: snr
    '''

    #spdata[(np.isnan(spdata)).ravel().nonzero()] = 0
    [numdata,numsweeps,numrecs] = spdata.shape
    [_,stimtime,_] = stim.shape

    if spdata.shape[2] != stim.shape[2]:
        print('Number of records and stimuli are not equal')

    #Response variability as a function of frequency per stimulus (-pair)#
    #-------------------------------------------------------------------#
    n = numsweeps * numdata / mf / basep

    reps_per_sweep = np.floor(numdata/basep)
    # remove incomplete rep
    tmp = spdata[range(int(reps_per_sweep * basep)), :, :]
    if reps_per_sweep>1:
        #remove first rep
        tmp=tmp[basep:, :, :]
        reps_per_sweep -= 1
    spdata2trim = np.reshape(tmp, [int(basep), int(reps_per_sweep * numsweeps), numrecs], order='F')

    # don't need this any more
    #n = numsweeps * np.floor(n / numsweeps)
    #if n/numsweeps > 1:
    #    spdata2trim = np.delete(spdata2, np.arange(0,n,n/numsweeps), axis=1)

    n = spdata2trim.shape[1]

    vrf = 1e6 * np.square(stimtime) / np.square(basep) / n * np.square(
        np.nanstd(np.fft.fft(spdata2trim, spdata2trim.shape[0], 0), ddof=1, axis=1))

    ##############Response power as a function of frequency##############
    #--------------------------------------------------------------------
    spikeperiod = np.squeeze(np.nanmean(spdata2trim,1))

    #"downsample spike histogram using insteadofbin()"#
    if basep/stimtime != 1/mf:
        spikeperiod = insteadofbin(spikeperiod, basep / stimtime, mf)

    spikeperiod = spikeperiod * 1e3 / basep * stimtime

    prf = np.square(np.abs(np.fft.fft(spikeperiod,spikeperiod.shape[0],0)))

    ######Variability of the STRF estimate (for TORC stimuli)############
    #-------------------------------------------------------------------#
    stimpospolarity = np.arange(0,stim.shape[2],1).astype(int)

    stim2 = stim[:,:,stimpospolarity]

    freqindex = np.swapaxes(np.round(np.abs(allrates) * basep/1000),0,1)
    freqindx = freqindex[:,stimpospolarity]

    # These are 1/a^2 for each stimulus
    AA = 2 * sum(freqindx > 0 ) / np.mean(np.mean(np.square(stim2),axis=0),axis=0)

    #"Estimate of the total power (ps) and error power (pes) of the STRF"
    pt = 0
    pes = 0

    for rec in range(stimpospolarity.shape[0]):
        pt = pt + 1 / np.square(stimtime) * AA[rec] * np.sum((2 * prf[freqindx[:,rec].astype(int),rec]))       #keep adding pt through each rep
        pes = pes + 1 / np.square(stimtime) * AA[rec] * np.sum((2 * vrf[freqindx[:,rec].astype(int),rec]))     #keep adding pes through each rep

    snr = pt/pes - 1
    return snr


def generate_torc_spectrograms(TorcObject, rasterfs=None, single_cycle=True):
    """
    soundobject should be the parameters for a Torc object from baphy (eg,
      exptparams['TrialObject'][1]['ReferenceHandle'][1]
    default rasterfs is native max modulation rate of torc_subfunctions
    single_cycle - don't generate full stimulus. If false repeat as needed and
        Pre-/Post- Silence

    return:
       TorcValues - dict of 2d arrays, one per TORC
       tags - baphy stimulus names associated with each index
       StimParams - important parameters of the Torc
    """
    fs=1000 # hard code fs since it seems to have to be 1000 to actually work

    TorcNames = TorcObject['Names']
    RefDuration = TorcObject['Duration']
    referencecount = len(TorcNames)

    numrecs = referencecount
    mf = int(fs/1000)  # used to be int(fs/1000)
    stdur = int(RefDuration*1000)

    # change nesting to TORCs(StimParam(...))
    TorcParams = dict.fromkeys(TorcNames)
    all_freqs = list()
    all_velos = list()
    all_hfreq = list()
    all_lfreq = list()

    for tt, torc in zip(TorcObject['Params'].keys(), TorcNames):
        #TorcParams[torc] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][tt + 1]     #insert Params 1-30 to torcs 1-30 now TORCs(Params(...)) nested other way
        TorcParams[torc] = TorcObject['Params'][tt]
        freqs = TorcParams[torc]['Scales']
        velos = TorcParams[torc]['Rates']
        all_freqs.append(freqs)
        all_velos.append(velos)
        highestfreqs = TorcParams[torc]['HighestFrequency']
        lowestfreqs = TorcParams[torc]['LowestFrequency']
        all_hfreq.append(highestfreqs)
        all_lfreq.append(lowestfreqs)

    frqs = np.unique(np.concatenate(all_freqs))
    vels = np.unique(np.concatenate(all_velos))
    HighestFrequency = int(np.unique(all_hfreq))
    LowestFrequency = int(np.unique(all_lfreq))
    Octaves = np.log2(HighestFrequency/LowestFrequency)

    StimParams = dict()
    StimParams['lfreq'] = LowestFrequency
    StimParams['hfreq'] = HighestFrequency
    StimParams['octaves'] = int(Octaves)

    Params = dict()
    N = np.size(frqs) * np.size(vels)        #aka nrips
    W = vels                                 #array of ripple velocities
    T = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    Params['T'] = T

    Ompos = [x for x in frqs if x >= 0]
    Omnegzero = np.flip([x for x in frqs if x <= 0])

    Omega = np.swapaxes(np.stack((Ompos,Omnegzero)),0,1)

    numvels = len(W)
    numfrqs = np.size(Omega,0)
    numstim = len(TorcNames)

    waveParams = np.empty([2,numvels,numfrqs,2,numstim])

    ##This part in MATLAB makes T, octaves, maxv, maxf, saf, numcomp
    basep = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    StimParams['basep'] = basep
    maxvel = np.max(np.abs(W))
    maxfrq = np.max(np.abs(Omega))
    saf = int(np.ceil(maxvel*2 + 1000/basep))
    numcomp = int(np.ceil(maxfrq*2*Octaves + 1))
    Params['numcomp'] = numcomp

    ##function [ststims,freqs]=stimprofile(waveParams,W,Omega,lfreq,hfreq,numcomp,T,saf);
    [_,Ws,Omegas,lr,numstim] = waveParams.shape
    [a,b] = Omega.shape
    [d] = W.shape

    if a*b*d != Omegas*Ws*lr:
        print('Omega and.or W do not match waveParams')

    sffact = saf/1000                                                        #lower sample rate
    leng = int(np.round(T*sffact))
    Params['leng'] = leng

    # Create dictionary of TORCs
    TorcValues = dict()
    for key,value in TorcParams.items():
        y_sum = torcmaker(value, Params)
        TorcValues[key] = y_sum

    tags = list(TorcValues.keys())

    if not single_cycle:
        # Process response signal to eliminate bins of silence before and after stimulus
        if rasterfs is None:
            rasterfs=saf # Nyquist rate for catching TORC modulations
        PreStimBins = int(TorcObject['PreStimSilence']*rasterfs)
        PostStimBins = int(TorcObject['PostStimSilence']*rasterfs)
        num_cycles = int(np.ceil(stdur/StimParams['basep']))

        for k in TorcValues.keys():
            t = repmat(TorcValues[k],1,num_cycles)
            final_samples=int(t.shape[1]*rasterfs/saf)
            t=sp.signal.resample(t, final_samples, axis=1)

            TorcValues[k]=np.concatenate([np.zeros((numcomp,PreStimBins))-5, t,
                                          np.zeros((numcomp,PostStimBins))-5], axis=1)

    return TorcValues, tags, StimParams


def torcmaker(TORC,Params):
    '''
    Returns the TORC - option to plot each torc commented out
    This is a fairly core calculation
    :param TORC: TORC data from torc dictionary
    :param Params: TorcObject containing info about the torc
    :return: the torc itself
    '''
    lfreq = TORC['LowestFrequency']
    hfreq = TORC['HighestFrequency']
    Scales = TORC['Scales']
    Amplitude = TORC['RippleAmplitude']
    Phase = TORC['Phase']
    Rate = TORC['Rates']

    octaves = np.log2(hfreq)-np.log2(lfreq)
    normed_scales = [s*octaves for s in Scales]
    cycles_per_sec = 1000/Params['T']
    normed_tempmod= [t/cycles_per_sec for t in Rate]
    numcomp = Params['numcomp']
    leng = Params['leng']

    # somehow we've figured out that final spectrogram should be
    # (numcomp spectral dimension rows) X (leng time samples per TORC cycle)

    stimHolder = np.zeros((numcomp,leng),dtype=complex)
    c = [np.floor(numcomp/2), np.floor(leng/2)]

    for i, (vel,phs,amp,scl) in enumerate(zip(normed_tempmod,Phase,Amplitude,normed_scales)):
        #print("ripple {}: vel={}, phi={}, amp={}, scl={}".format(i,vel,phs,amp,scl))

        # figure out index in fourier domain for each ripple
        v1=int(vel+c[1])
        v2=int(c[1]-vel)
        s1=int(scl+c[0])
        s2=int(c[0]-scl)

        stimHolder[s1,v1] = (amp/2)*np.exp(1j*(phs-90)*np.pi/180)
        stimHolder[s2,v2] = (amp/2)*np.exp(-1j*(phs-90)*np.pi/180)

    y_sum = (np.fft.ifft2(np.fft.ifftshift(stimHolder*(leng*numcomp)))).real
    return y_sum



def strf_est_core(stacked, TorcObject, fs, INC1stCYCLE=0, jackN=0):
    '''
    Estimate STRF from TORCs: main subfunction, with options to plot and compare to matlab values
    :param stacked: spike raster, time (sound start to stop) x torcidx x repetition
    :param TorcObject: torc parameter data
    :param fs: bin rate of stacked (stacked will be rebinned to match max sampling rate of torcs)
    :param INC1stCYCLE: default 0, if 1, include first 250ms of stacked in STRF estimation (0 removes transient reponse)
    :param jackN: default 0, if >1, computer jackknifes on strfest to measure error bars
    :return: strfest, snr (usually >0.2 is good), stim (taken from Scaled Torcs, collapses torc dict to matrix),strfemp,StimParams (some useful params)
    '''

    #referencecount = TorcObject['MaxIndex']
    #TorcNames = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Names"]
    TorcNames = TorcObject['Names']
    RefDuration = TorcObject['Duration']
    referencecount = len(TorcNames)

    numrecs = referencecount
    mf = int(fs/1000)  # used to be int(fs/1000)
    stdur = int(RefDuration*1000)

    # change nesting to TORCs(StimParam(...))
    TorcParams = dict.fromkeys(TorcNames)
    all_freqs = list()
    all_velos = list()
    all_hfreq = list()
    all_lfreq = list()

    for tt, torc in zip(TorcObject['Params'].keys(), TorcNames):
        #TorcParams[torc] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][tt + 1]     #insert Params 1-30 to torcs 1-30 now TORCs(Params(...)) nested other way
        TorcParams[torc] = TorcObject['Params'][tt]
        freqs = TorcParams[torc]['Scales']
        velos = TorcParams[torc]['Rates']
        all_freqs.append(freqs)
        all_velos.append(velos)
        highestfreqs = TorcParams[torc]['HighestFrequency']
        lowestfreqs = TorcParams[torc]['LowestFrequency']
        all_hfreq.append(highestfreqs)
        all_lfreq.append(lowestfreqs)

    frqs = np.unique(np.concatenate(all_freqs))
    vels = np.unique(np.concatenate(all_velos))
    HighestFrequency = int(np.unique(all_hfreq))
    LowestFrequency = int(np.unique(all_lfreq))
    Octaves = np.log2(HighestFrequency/LowestFrequency)

    StimParams = dict()
    StimParams['lfreq'] = LowestFrequency
    StimParams['hfreq'] = HighestFrequency
    StimParams['octaves'] = int(Octaves)

    Params = dict()
    N = np.size(frqs) * np.size(vels)        #aka nrips
    W = vels                                 #array of ripple velocities
    T = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    Params['T'] = T

    Ompos = [x for x in frqs if x >= 0]
    Omnegzero = np.flip([x for x in frqs if x <= 0])

    Omega = np.swapaxes(np.stack((Ompos,Omnegzero)),0,1)

    numvels = len(W)
    numfrqs = np.size(Omega,0)
    numstim = len(TorcNames)

    waveParams = np.empty([2,numvels,numfrqs,2,numstim])

    ##This part in MATLAB makes T, octaves, maxv, maxf, saf, numcomp
    basep = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    StimParams['basep'] = basep
    maxvel = np.max(np.abs(W))
    maxfrq = np.max(np.abs(Omega))
    saf = int(np.ceil(maxvel*2 + 1000/basep))
    numcomp = int(np.ceil(maxfrq*2*Octaves + 1))
    Params['numcomp'] = numcomp

    ##function [ststims,freqs]=stimprofile(waveParams,W,Omega,lfreq,hfreq,numcomp,T,saf);
    [_,Ws,Omegas,lr,numstim] = waveParams.shape
    [a,b] = Omega.shape
    [d] = W.shape

    if a*b*d != Omegas*Ws*lr:
        print('Omega and.or W do not match waveParams')

    sffact = saf/1000                                                        #lower sample rate
    leng = int(np.round(T*sffact))
    Params['leng'] = leng

    # Create dictionary of TORCs
    TorcValues = dict()
    for key,value in TorcParams.items():
        y_sum = torcmaker(value, Params)
        TorcValues[key] = y_sum

    # ######plot all torcs in subplots -!!!eventually make this optional input?
    # row = int(np.ceil(math.sqrt(len(TorcValues.keys()))))
    # col = int(np.floor(math.sqrt(len(TorcValues.keys()))))
    #
    # if row * col < len(TorcValues.keys()):
    #     row = row + 1
    #
    # torcfig, torcaxs = plt.subplots(row, col, sharex=False, sharey=False, squeeze=False)
    #
    # axsall = np.ravel(torcaxs)
    #
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    #
    # for ttt,(key,value) in enumerate(TorcValues.items()):
    #     axsall[ttt].imshow(value)
    #######

    # Scaling our stimulus (TORCs)
    ModulationDepth = 0.9                                                  #default set in matlab program
    base1= 0
    xSize = int(np.round(10*numcomp/Octaves))
    tSize = int(10*saf*basep/1000)
    ScaledTorcs = dict()

    for key,value in TorcValues.items():
        [xsiz, tsiz] = value.shape
        temp = value

        if xSize != xsiz & tSize != tsiz:
            temp1 = interpft(interpft(temp, xSize, 0), tSize, 1)

            scl = np.max(np.abs([np.min(np.min(temp1)), np.max(np.max(temp1))]))    #largest |value| is scale factor

            temp2 = base1 + temp*ModulationDepth/scl

        ScaledTorcs[key] = temp2

    [stimX,stimT] = temp.shape
    binsize = int(basep/stimT)

    strfest = np.zeros([stimX,stimT])

    ##only loop over real (nonNaN) reps, may be diff number of reps for diff torcs
    try:
        if stacked.shape[0] <= fs/4:
            realrepcount = np.max(np.logical_not(np.isnan(stacked[1,:,1])).ravel().nonzero())+1
        else:
            realrepcount = np.max(np.logical_not(np.isnan(stacked[int(np.round(fs/4))+1,:,1])).ravel().nonzero())+1
    except:
        print('stacked matrix is empty, dropping to debugger')
        import pdb; pdb.set_trace()

    if INC1stCYCLE == 1:
        if stacked.shape[0]>basep:
            print('Including 1st TORC Cycle')
        FirstStimTime = 0
    elif INC1stCYCLE > 0:
        FirstStimTime = INC1stCYCLE
    else:
        FirstStimTime = basep

    # Get things ready for snr inputs
    stim = np.stack(list(ScaledTorcs.values()),axis=2)                  #take my lovely dictionary and smoosh it for now

    a1rv = []                                                           #create empty list for all the Torcs' respective velocities
    for key,values in TorcParams.items():
        torcrate = values['Rates']
        a1rv.append(torcrate)
    allrates = np.asarray(a1rv)

    if stacked.shape[1]>1:
        snr = get_snr(stacked, stim, basep, mf, allrates)
    else:
        snr = 0

    if not jackN:         #normalize by the num of reps that were presented
        for rep in range(realrepcount):
            for rec in range(numstim):

                if stacked.shape[0] <= fs/4:
                    thisrepcount = sum(np.logical_not(np.isnan(stacked[0,:,rec])))
                else:
                    thisrepcount = sum(sum(np.logical_not(np.isnan(stacked[int(np.round(fs/4)+1):,:,0]))) > 0).astype(int)

                if thisrepcount == 0:
                    thisrepcount = 1

                spkdata = stacked[:,rep,rec]

                if fs != 1000:
                    spkdata = sp.signal.resample(stacked,1000)

                if len(spkdata) < stdur:
                    spkdata = np.concatenate((spkdata,np.ones(stdur-len(spkdata))*np.nan),axis=None)

                [Dsum,cnorm] = makepsth(spkdata.copy(), int(1000 / basep), int(FirstStimTime), stdur, mf)

                #Time to normalize by number of cycles. May be variable for TORCs, since they could be shorter than length in exptparams
                Dsum = Dsum / (cnorm + (cnorm == 0))

                if binsize > 1/mf:
                    Dsum = insteadofbin(Dsum, binsize, mf)
                Dsum = Dsum * (1000/binsize)                                           #normalization by bin size

                if sum(np.isnan(Dsum)) or not thisrepcount:
                    print('NaN STRF: rep {} rec {}'.format(rep,rec))
                else:
                    Stim = stim[:,:,rec]
                    strftemp = np.zeros(strfest.shape)

                for abc in range(stimX):
                    stimrow = Stim[abc,:]
                    strftemp[abc,:] = np.fft.ifft(np.conj(np.fft.fft(stimrow)) * np.fft.fft(np.squeeze(Dsum))).real

                strftemp = strftemp / stimT                                                     #normalization
                strftemp = strftemp * (2 * N / np.mean(np.mean(np.square(Stim)))/numstim)       #normalization

                strfest = strfest + strftemp / thisrepcount

        strfemp = np.zeros(strfest.shape)

    else:
        strfj = np.zeros([stimX,stimT,jackN])

        stackedj = stacked
        stackedj[:FirstStimTime,:,:] = np.nan

        mm = int(np.floor(stackedj.shape[0] / basep))
        stackedj = np.reshape(stackedj[:mm*basep,:,:],(basep,mm*stackedj.shape[1],stackedj.shape[2]),order='F').copy()

        print('strfest with %d jackknifes\n' % jackN)

        for rec in range(numrecs):
            for bb in range(jackN):
                stackedt = stackedj[:,:,rec]
                nnbins = np.squeeze(np.asarray(np.nonzero(np.ravel(np.logical_not(np.isnan(stackedt)),order='F'))))
                totalbins = len(nnbins)
                bbexcl = np.asarray(list(range(int(np.ceil(bb * totalbins / jackN)),int(np.ceil((bb+1) * totalbins / jackN)))))
                stackedravel = np.ravel(stackedt,order='F')
                stackedravel[nnbins[bbexcl]] = np.nan
                stackedtt = np.reshape(stackedravel, [basep, mm * stacked.shape[1]], order='F').copy()

                dsum = np.nanmean(stackedtt,1)
                dsum[np.isnan(dsum)] = 0

                if binsize > 1/mf:
                    Dsum = insteadofbin(dsum,binsize,mf)
                Dsum = Dsum * (1000/binsize)
                Stim = stim[:,:,rec]
                strftemp = np.zeros(strfest.shape)

                for abc in range(stimX):
                    stimrow = Stim[abc,:]
                    strftemp[abc, :] = np.fft.ifft(np.conj(np.fft.fft(stimrow)) * np.fft.fft(np.squeeze(Dsum))).real

                strftemp = strftemp / stimT                                                     #normalization
                strftemp = strftemp * (2 * N / np.mean(np.mean(np.square(Stim)))/numstim)       #normalization

                strfj[:,:,bb] = strfj[:,:,bb] + strftemp

                if np.any(np.isnan(strfest[0])):
                    print('NaN STRF: rep {} rec {}'.format(rep, rec))

        mm = np.mean(strfj,2)
        ee = np.std(strfj,2,ddof=1) * np.sqrt(jackN-1)
        strfest = mm
        strfemp = ee

    return strfest,snr,stim,strfemp,StimParams
