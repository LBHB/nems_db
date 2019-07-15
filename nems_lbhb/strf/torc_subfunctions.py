import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import io as sio


def interpft(x,ny,dim=0):                                                         #input torc array, new sampling, which dimension to interpolate on
    '''
    Function to interpolate using FT method, based on matlab interpft()
    :param x: array for interpolation
    :param ny: length of returned vector post-interpolation
    :param dim: performs interpolation along dimension DIM, default 0
    :return: interpolated data
    '''

    # if not dim:
    #     x = np.expand_dims(x,axis=1)
    #     if np.isscalar(x):
    #         nshifts = 1
    #     else:
    #         nshifts = 0
    # else:
    #     x = np.swapaxes(x,0,dim)

    if dim >= 1:                                                                   #if interpolating along columns, dim = 1
        x = np.swapaxes(x,0,dim)                                                   #temporarily swap axes so calculations are universal regardless of dim
    if len(x.shape) == 1:
        x = np.expand_dims(x,axis=1)

    siz = x.shape                                                                  #what is the torc size overall
    [m, n] = x.shape                                                               #unique var for each torc dimension

    if ny > m:                                                                     #if you will be increasing samples (should be)
        incr = 1                                                                   #assign this variable - not very useful but doesn't hurt, could be useful

    a = np.fft.fft(x,m,0)                                                          #do FT along rows, shape unaltered
    nyqst = int(np.ceil((m+1)/2))                                                  #nyqst num calculated
    b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)  #insert a field of zeros to expand dim to new, using nyqst as break point

    if np.remainder(m,2)==0:                                                       #this hasn't come up yet
        b[nyqst,:] = b[nyqst,:]/2                                                  #presumably dealing with remainder
        b[nyqst+ny-m,:] = b[nyqst,:]                                               #somehow

    y = np.fft.irfft(b,b.shape[0],0)                                               #take inverse FT (real) using new dimension generated along dim 0 of b

    #if all(np.isreal(x)):                                                         #checks to make sure everything is real
    #    y = y.real                                                                #it is, don't know when this would come up

    y = y * ny / m                                                                 #necessary conversion...
    #y = y[1:ny:incr,:]                                                            #

    y = np.reshape(y, [y.shape[0],siz[1]])                                         #should preserve shape

    y = np.squeeze(y)

    if dim >= 1:                                                                   #as above, if interpolating along columns
        y = np.swapaxes(y,0,dim)                                                   #swap axes back and y will be correct

    return y                                                                       #returned value



def insteadofbin(resp,binsize,mf=1):
    '''
    ###SUBFUNCTION - insteadofbin()#####################################################
    #Downsample spike histogram from resp (resolution mf) to resolution by binsize(ms)##
    #"Does by sinc-filtering and downsampling instead of binning," whatever that means##
    #function dsum = insteadofbin(resp,binsize,mf);                                   ##

    :param resp: response data (comes from spikeperiod
    :param binsize: calculated size of bins (basep/stimtime = binsize)
    :param mf: multiplication factor, default 1
    :return: returns downsampled spike histogram
    '''

    if len(resp.shape) >= 2:                                                       #added in jackN phase ot account for Dsum input having one dimension in bython (250x1)
        [spikes,records] = resp.shape                                              #break response into its dimensions
    else:
        resp = np.expand_dims(resp, axis=1)
        [spikes,records] = resp.shape

    outlen = int(spikes / binsize / mf)                                            #

    if outlen - np.floor(outlen/1) * 1 > 1:                                        #basically what matlab fxn mod(x,y)
        print('Non-integer # bins. Result may be distorted')                       #check probably for special circumstances

    outlen = np.round(outlen)                                                      #original comments say "round or ceil or floor?" oh well
    dsum = np.zeros([outlen,records])                                              #empty an array to fill below, going to be output

    for rec in range(records):                                                     #going through all records
        temprec = np.fft.fft(resp[:,rec])                                          #fft for each

        if outlen % 2 == 0:                                                        #if even length, create middle point
            temprec[np.ceil((outlen-1)/2)+1] = np.abs(temprec[np.ceil(outlen-1)/2]+1)

        dsum[:, rec] = np.fft.ifft(np.concatenate((temprec[0:int(np.ceil((outlen - 1) / 2) + 1)], np.conj(
            np.flipud(temprec[1:int(np.floor((outlen - 1) / 2) + 1)]))))).real

    return dsum



def strfplot(strf0, lfreq, tleng, smooth=0, noct=5, siglev=5,axs=None):
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

        # supposedly python does colorbar scaling on its own, thanks
        # if len(clim) != 0:
        #     clim = clim
        # elif np.max(np.max(np.abs(stdata))):
        #     clim = [-1*np.max(np.max(np.abs(stdata))),1*np.max(np.max(np.abs(stdata)))]
        # else:
        #     clim = [-1,1]

        #f = plt.gcf

        axs.imshow(strfdata, cmap=None, norm=None, aspect='auto', extent=[0, tleng, 0, noct], origin='lower', )

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

    period = int(1000 * (1/fhist) * mf)                                            # in samples
    startime = startime * mf                                                       #      "
    endtime = endtime * mf                                                         #      "

    if endtime > len(dsum):
        endtime = len(dsum)

    fillmax = int(np.ceil(endtime/period) * period)
    if fillmax > endtime:
        dsum[endtime+1:fillmax] = np.nan
        endtime = fillmax

    dsum[:startime] = np.nan
    repcount = int(fillmax / period)
    dsum = np.reshape(dsum[:endtime],(period,repcount),order='F')

    wrapdata = np.nansum(dsum,1)                                                  #get 250 list of how many 1s there were
    cnorm =  np.sum(np.logical_not(np.isnan(dsum)),1)                             #get 250 list of how many 1s were possible

    return wrapdata,cnorm

    ##There's extra code that SVD 'hacked' to allow for includsion of the first TORC



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
    # ###
    # rmat = sio.loadmat('/auto/users/hamersky/r.mat')
    # spdata = rmat['r']
    # ###
    spdata[(np.isnan(spdata)).ravel().nonzero()] = 0                                   #get rid of not numbers
    [numdata,numsweeps,numrecs] = spdata.shape                                         #dims to vars
    [stimfreq,stimtime,numstims] = stim.shape                                          #dims to vars

    invlist = []                                                                       #no clue what this is for actually, it's subsequent uses are commented out

    if spdata.shape[2] != stim.shape[2]:                                               #just a check
        print('Number of records and stimuli are not equal')                           #doubt this'll happen

    # for key, value in ScaledTorcs.items():                                             #pretty unnecessary
    #     [stimfreq,stimtime] = value.shape                                              #a better way to get these I think

    #Response variability as a function of frequency per stimulus (-pair)#
    #-------------------------------------------------------------------#
    n = numsweeps * numdata / mf / basep                                                                       #
    tmp = spdata[range(int(np.round(np.floor(n / numsweeps) * mf * basep))),:,:]                               #temp is the same as spdata
    spdata2 = np.reshape(tmp, [int(basep * mf), int(numsweeps * np.floor(n / numsweeps)), numrecs], order='F') #moves dims from 0 axis to 1, new shape 250,30,30
    n = numsweeps * np.floor(n / numsweeps)                                                                    #n should still be same as above

    if n/numsweeps > 1:                                                                                        #it will
        spdata2trim = np.delete(spdata2, np.arange(0,n,n/numsweeps), axis=1)                                   #squish out a third of axis 1

    n = spdata2trim.shape[1]                                                                                   #new axis 1 defines this (2/3 of previous n)

    # if not invlist.shape[0] < 2:
    #     vrf = 1e6 * np.square(stimtime) / np.square(basep) / n * np.square(np.std(
    #         np.fft.fft((spdata2trim[:, :, invlist[0, :]] - spdata2trim[:, :, invlist[1.:]]) / 2, spdata2trim.shape[0], 0),
    #         ddof=1, axis=1))
    # else:
    vrf = 1e6 * np.square(stimtime) / np.square(basep) / n * np.square(                                        #
        np.std(np.fft.fft(spdata2trim, spdata2trim.shape[0], 0), ddof=1, axis=1))                              #

    ##############Response power as a function of frequency##############
    #-------------------------------------------------------------------#
    spikeperiod = np.squeeze(np.mean(spdata2trim,1))                                                           #

    #"downsample spike histogram using insteadofbin()"#
    if basep/stimtime != 1/mf:                                                                                 #
        spikeperiod = insteadofbin(spikeperiod, basep / stimtime, mf)                                              #

    spikeperiod = spikeperiod * 1e3 / basep * stimtime                                                         #

    # if not invlist.shape[0] < 2:
    #     spikeperiod = (spikeperiod[:,invlist[0,:]] - spikeperiod[:,invlist[1,:]])/2

    prf = np.square(np.abs(np.fft.fft(spikeperiod,spikeperiod.shape[0],0)))                                    #

    ######Variability of the STRF estimate (for TORC stimuli)############
    #-------------------------------------------------------------------#
    # if not invlist.shape[0] < 2:
    #     stimpospolarity = invlist[0,:]
    # else:
    stimpospolarity = np.arange(0,stim.shape[2],1).astype(int)                                                 #

    stim2 = stim[:,:,stimpospolarity]                                                                          #

    freqindex = np.swapaxes(np.round(np.abs(allrates) * basep/1000),0,1)                                       #
    freqindx = freqindex[:,stimpospolarity]                                                                    #

    AA = 2 * sum(freqindx > 0 ) / np.mean(np.mean(np.square(stim2),axis=0),axis=0)                             #These are 1/a^2 for each stimulus

    #"Estimate of the total power (ps) and error power (pes) of the STRF"
    pt = 0                                                                                                     #start at zero cause we're going to be adding
    pes = 0                                                                                                    #

    for rec in range(stimpospolarity.shape[0]):                                                                #
        pt = pt + 1 / np.square(stimtime) * AA[rec] * np.sum((2 * prf[freqindx[:,rec].astype(int),rec]))       #keep adding pt through each rep
        pes = pes + 1 / np.square(stimtime) * AA[rec] * np.sum((2 * vrf[freqindx[:,rec].astype(int),rec]))     #keep adding pes through each rep

    snr = pt/pes - 1                                                                                           #calc snr
    return snr

###Function to return the TORC - option to plot each Torc###
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
        #print("ripple {}: stimHolder[s1,v1]={}, stimholder[s2,v2]={}".format(i,stimHolder[s1,v1],stimHolder[s2,v2]))
        #######if you want to look at your ripple at any point
        #plt.figure()
        #plt.imshow(np.abs(stimHolder))

    y_sum = (np.fft.ifft2(np.fft.ifftshift(stimHolder*(leng*numcomp)))).real
    return y_sum



def strf_est_core(stacked,TorcObject,exptparams,fs,INC1stCYCLE=0,jackN=0):
    '''
    Estimate STRF from TORCs: main subfunction, with options to plot and compare to matlab values
    :param stacked: spike raster, time (sound start to stop) x torcidx x repetition
    :param TorcObject: torc parameter data
    :param exptparams: certain necessary experiment parameters from initial extraction
    :param fs: bin rate of stacked (stacked will be rebinned to match max sampling rate of torcs)
    :param INC1stCYCLE: default 0, if 1, include first 250ms of stacked in STRF estimation (0 removes transient reponse)
    :param jackN: default 0, if >1, computer jackknifes on strfest to measure error bars
    :return: strfest, snr (usually >0.2 is good), stim (taken from Scaled Torcs, collapses torc dict to matrix),strfemp,StimParams (some useful params)
    '''

    referencecount = TorcObject['MaxIndex']
    TorcNames = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Names"]
    TorcKeys = exptparams["TrialObject"][1]["ReferenceHandle"][1].keys()
    RefDuration = TorcObject['Duration']

    numrecs = referencecount
    mf = int(fs/1000)
    stdur = int(RefDuration*1000)
    ddur = int(RefDuration*1000)
    stonset = 0

    ###change nesting to TORCs(StimParam(...))
    TorcParams = dict.fromkeys(TorcNames)                                                    #Create dict from TorcNames
    all_freqs = list()                                                                       #Create empty list of freqs
    all_velos = list()                                                                       #Create empty list of velos
    all_hfreq = list()
    all_lfreq = list()

    for tt, torc in enumerate(TorcNames):                                                    #Number them
        TorcParams[torc] = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Params"][tt + 1]     #insert Params 1-30 to torcs 1-30 now TORCs(Params(...)) nested other way
        freqs = TorcParams[torc]['Scales']                                                   #Add all TORCs' Scales value as var
        velos = TorcParams[torc]['Rates']                                                    #Add all TORCs' Rates value as var
        all_freqs.append(freqs)                                                              #
        all_velos.append(velos)                                                              #
        highestfreqs = TorcParams[torc]['HighestFrequency']
        lowestfreqs = TorcParams[torc]['LowestFrequency']
        all_hfreq.append(highestfreqs)
        all_lfreq.append(lowestfreqs)

    frqs = np.unique(np.concatenate(all_freqs))                                              #Smoosh to one array and output unique elements
    vels = np.unique(np.concatenate(all_velos))                                              #temporal spectra
    HighestFrequency = int(np.unique(all_hfreq))
    LowestFrequency = int(np.unique(all_lfreq))
    Octaves = np.log2(HighestFrequency/LowestFrequency)

    StimParams = dict()
    StimParams['lfreq'] = LowestFrequency
    StimParams['hfreq'] = HighestFrequency
    StimParams['octaves'] = int(Octaves)

    Params = dict()
    N = np.size(frqs) * np.size(vels)        #aka nrips                                      #Seems more concise than MATLAB version --N = size(unique([cat(2,a1rv{:});cat(2,a1rf{:})]','rows'),1)
    W = vels                                                                                 #array of ripple velocities
    T = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))          #
    Params['T'] = T

    Ompos = [x for x in frqs if x >= 0]                                                      #Get positive frequencies
    Omneg = [x for x in frqs if x < 0]                                                       #Get negative frequencies
    Omnegzero = np.flip([x for x in frqs if x <= 0])                                         #just used for populating an array a few lines down

    Omega = np.swapaxes(np.stack((Ompos,Omnegzero)),0,1)                                     #Make an array for main output Omega

    numvels = len(W)                                                                         #
    numfrqs = np.size(Omega,0)                                                               #Used to make empty array to be populated by params
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
    [ap,Ws,Omegas,lr,numstim] = waveParams.shape                               #wave params is 5D, define size of each dimension
    [a,b] = Omega.shape                                                        #splitting ripple freqs to matrix nums
    [d] = W.shape                                                              #splitting W into same

    if a*b*d != Omegas*Ws*lr:
        print('Omega and.or W do not match waveParams')

    #ln(num)/ln(2) = log2(num)
    #numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
    ####vvv!!!!!not working right freqrange
    freqrange = np.logspace(LowestFrequency, HighestFrequency, numcomp+1, endpoint=True, base=2.0)#make log2 scale from low to high freq binned by how many comps
    sffact = saf/1000                                                        #lower sample rate
    leng = int(np.round(T*sffact))                                           #stim duration with lower sampling rounded
    t = np.arange(leng)                                                      #make array for time that takes into account 0 is really first
    Params['leng'] = leng

    w0 = 1000/T
    W0 = 1/(np.log2(HighestFrequency/LowestFrequency))

    k = np.round(W/w0)
    l = np.round(Omega/W0)

    ststims = np.zeros([numcomp, leng, numstim])                            #to be our output of TORCS

    cnew = [np.floor(numcomp/2)+1, np.floor(leng/2)+1]

    ###this part is important###
    TorcValues = dict()                                                                #make new dict for function output
    for key,value in TorcParams.items():                                               #cycle through with all the different TORC names and respective values
        y_sum = torcmaker(value, Params)                                                #output of function (TORC) assigned to variable
        TorcValues[key] = y_sum                                                        #update dict with the key you are on and the value the function just returned

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

    ########checkmatlab######
    # import scipy.io as sio                                             #
    # StStims = sio.loadmat('/auto/users/hamersky/stims.mat')            #load Matlab file as dictionary
    # ststims = StStims['ststims']                                       #pull out the data as variable
    # [mlx,mlt,torcnum] = ststims.shape                                  #assign variables to the shapes
    # renametorc = TorcNames                                             #call TorcNames for here
    #
    # MatlabTorcs = dict()                                               #open new dict to be comparable to TorcValues
    # for iii in range(torcnum):                                         #go through in range of 'numstims'
    #     whichtorc = ststims[:,:,iii]                                   #each iteration save that torc as variable
    #     eee = renametorc[iii]                                          #get the corresponding torcname as num we're on
    #     MatlabTorcs[eee] = whichtorc                                   #assign the key and the value, looks like TorcValues
    #
    # ststims = np.around(ststims[:,:,:],decimals=4)
    #     for nn,(key,value) in enumerate(TorcValues.items()):
    #         torccc = np.around(value,decimals=4)
    #         if np.all(torccc == ststims[:,:,nn]):
    #             print(f'{nn} True')
    #         else:
    #             print(f'{nn} False')
    #
    # for qwer in range(spdata2trimmat.shape[0]):
    #     if np.all(spdata2trimmat[qwer, 17, 16] == spdata2trim[qwer, 17, 16]):
    #         print(qwer, "True")
    #     else:
    #         print(qwer, "False")
    #########################

    ###Going into the stimscale#########################################################
    #function stim = stimscale(stim,REGIME,OPTION1,OPTION2,tsiz,xsiz); -our data made moddep 0.9 [] 250 30
    # 3)'moddep': Specified modulation depth;
    #	OPTION1 is the modulation depth as a fraction of 1. The default is 0.9.
    #	OPTION2 is not required.

    ModulationDepth = 0.9                                                              #default set in matlab program
    base1= 0                                                                           #hard coded in matlab
    numtorc = len(TorcValues.keys())                                                   #how many torcs there are, again
    xSize = int(np.round(10*numcomp/Octaves))   #30                                    #new x sampling rate
    tSize = int(10*saf*basep/1000)              #250                                   #new t sampling rate
    ScaledTorcs = dict()                                                               #new dictionary for scaled torcs

    for key,value in TorcValues.items():                                               #go through this with all torcs
        [xsiz, tsiz] = value.shape                                                     #basic dims of torc
        temp = value                                                                   #pull out numbers to usable variable

        if xSize != xsiz & tSize != tsiz:                                              #when new sampling rate doesn't equal old vals
            temp1 = interpft(interpft(temp, xSize, 0), tSize, 1)                           #add points, yielding bigger array

            scl = np.max(np.abs([np.min(np.min(temp1)), np.max(np.max(temp1))]))       #largest |value| is scale factor

            temp2 = base1 + temp*ModulationDepth/scl                                   #transform original torc values with moddep and calc'd scale

        ScaledTorcs[key] = temp2                                                       #populate dictionary with new values
    ####################################################################################

    #Back to main strf_est_core function (matlab) - scaled and all good
    [stimX,stimT] = temp.shape                                                                   #have var for dims of torc
    binsize = int(basep/stimT)                                                                   #number of bins

    strfest = np.zeros([stimX,stimT])                                                            #empty array same size as torc

    ##only loop over real (nonNaN) reps, may be diff number of reps for diff torcs
    #in my code, 'stacked' is matlab's 'r'
    if stacked.shape[0] <= fs/4:                                                                     #multiple cycles
        realrepcount = np.max(np.logical_not(np.isnan(stacked[1,:,1])).ravel().nonzero())+1            #
    else:                                                                                        #go to here, all 10 are real and used
        realrepcount = np.max(np.logical_not(np.isnan(stacked[int(np.round(fs/4))+1,:,1])).ravel().nonzero())+1

    #[pp,bb]=fileparts(mfilename)                                           ##don't know what this is for...##

    if INC1stCYCLE == 1:                                                               #Some maybe useless code, cause I defaulted var to 0
        if stacked.shape[0]>250:                                                           #
            print('Including 1st TORC Cycle')                                          #Stim starts on 0 cause not excluding
        FirstStimTime = 0                                                              #
    elif INC1stCYCLE > 0:                                                              #
        FirstStimTime = INC1stCYCLE                                                    #
    else:                                                                              #
        FirstStimTime = basep                                                          #This is what will be happening

    #Get things ready for snr inputs#
    stim = np.stack(list(ScaledTorcs.values()),axis=2)                                 #take my lovely dictionary and smoosh it for now

    a1rv = []                                                                          #create empty list for all the Torcs' respective velocities
    for key,values in TorcParams.items():                                              #iterate through dict
        torcrate = values['Rates']                                                     #access the rate for each key,value combo
        a1rv.append(torcrate)                                                          #add each time this iterates to a list of lists
    allrates = np.asarray(a1rv)                                                        #transform list of list into an array

    if stacked.shape[1]>1:
        snr = get_snr(stacked, stim, basep, mf, allrates)
    else:
        snr = 0
    ###
    # rmat = sio.loadmat('/auto/users/hamersky/r.mat')
    # rmat = rmat['r']


    if not jackN:                                                                      #normalize by the num of reps that were presented
        for rep in range(realrepcount):                                                #for this particular record (can be variable for DMS or
            for rec in range(numstim):                                                 #any data set that was interrupted in mid of a rep

                if np.all(stacked[0] <= fs/4):                                                 #
                    thisrepcount = sum(np.logical_not(np.isnan(stacked[0,:,rec])))     #this line is direct translation from matlab, wasn't used so can't test actual functionality with numbers, does exact same thing though
                else:
                    thisrepcount = sum(sum(np.logical_not(np.isnan(stacked[int(np.round(fs/4)+1):,:,0]))) > 0).astype(int)

                if thisrepcount == 0:
                    thisrepcount = 1

                spkdata = stacked[:,rep,rec]
                # spkdata = (rmat[:,rep,rec]).astype('float64')

                ###Probably not going to happen in this case
                # if fs != 1000:                                                         #this might not be a great translation
                #     spkdata = sp.signal.resample(stacked,1000)                         #probably not, couldn't find ==resample()
                if fs != 1000:
                    spkdata = sp.signal.resample(stacked,1000)

                if len(spkdata) < stdur:
                    spkdata = np.concatenate((spkdata,np.ones(stdur-len(spkdata))*np.nan),axis=None)
                ###next part will

                [Dsum,cnorm] = makepsth(spkdata.copy(), int(1000 / basep), int(FirstStimTime), stdur, mf)

                #Time to normalize by #cycles. May be variable for TORCs, since they could be shorter than length in exptparams
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
        # stackedj = rmat.astype('float64')
        stackedj[:FirstStimTime,:,:] = np.nan

        mm = int(np.floor(stackedj.shape[0] / basep))
        stackedj = np.reshape(stackedj[:mm*basep,:,:],(basep,mm*stackedj.shape[1],stackedj.shape[2]),order='F').copy()

        print('strfest with %d jackknifes\n' % jackN)

        for rec in range(numrecs):
            for bb in range(jackN):
                stackedt = stackedj[:,:,rec]
                nnbins = np.squeeze(np.asarray(np.nonzero(np.ravel(np.logical_not(np.isnan(stackedt)),order='F'))))
                #nnbins = np.squeeze(np.asarray(np.nonzero(np.ravel(np.logical_not(np.isnan(stackedt)).ravel().nonzero()))))
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