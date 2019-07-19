import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from nems_lbhb.strf.torc_subfunctions import interpft, insteadofbin, makepsth, torcmaker

def strf_plot_prepare(strf,Params):
    '''
    Determines Best Frequency
    :param strf: whichever STRF array you are looking for bf on
    :param Params: some stimulus parameters
    :return: best frequency
    '''
    maxoct = int(np.log2(Params['hfreq'] / Params['lfreq']))
    stepsize = maxoct / strf.shape[0]

    smooth = [100, strf.shape[1]]
    strfsmooth = interpft(strf, smooth[0], 0)

    ff = np.exp(np.linspace(np.log(Params['lfreq']), np.log(Params['hfreq']), strfsmooth.shape[0]))

    mm = np.mean(strfsmooth[:, :7] * (1 * (strfsmooth[:, :7] > 0)), 1)
    if sum(np.abs(mm)) > 0:
        bfidx = int(sum(((mm == np.max(mm)).ravel().nonzero())))
        bf = np.round(ff[bfidx])
    else:
        bfidx = 1
        bf = 0

    return bf


def base_model(data):
    '''
    Creates the model that will be trained on the stimulus and response data
    :param data: tell it out many features there will be
    :return: the model itself
    '''
    model = Sequential()
    model.add(Dense(1,input_shape=(data.shape[1],),kernel_initializer='normal',activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model


def model_strf(stimulus,response):
    '''
    Create an STRF based on training of the model with torc and response data
    :param stimulus: TORCs, with time lags and concatenated across TORCs
    :param response: average response across repetitions for each TORC
    :return: ground truth
    '''
    data = np.swapaxes(stimulus,0,1)
    model = base_model(data)
    model.fit(data, response, batch_size=data.shape[0], epochs=25000, verbose=0)

    fitvals = model.layers[0].get_weights()[0].squeeze()
    fitH = np.reshape(fitvals, (15, 25), order='F')

    return fitH


def stimulus_compiler(stimDict):
    '''
    Generate from TORC data a single array of all time delays and all torcs
    :param stimDict: Dictionary containing all the TORC stimuli
    :return: array of all stimuli combined
    '''
    DelayTorcs = dict()
    for tt, (key, value) in enumerate(stimDict.items()):
        for jj in range(value.shape[1]):
            rolledtor = np.roll(value, (jj), axis=1)
            if jj == 0:
                torcdel = rolledtor
            else:
                torcdel = np.concatenate((torcdel, rolledtor), axis=0)
        DelayTorcs[key] = torcdel
        if tt == 0:
            stimall = torcdel
        else:
            stimall = np.concatenate((stimall, torcdel), axis=1)

    return stimall


def response_compiler(stacked,Params):
    '''
    Create from response data a single vector of average responses to torcs
    :param stacked: raster data
    :param Params: some parameters
    :return: a 1D of responses
    '''
    FirstStimTime = Params['basep']
    RespCat = dict()
    for rep in range(stacked.shape[1]):
        for rec in range(stacked.shape[2]):

            spkdata = stacked[:,rep,rec]

            [Dsum,cnorm] = makepsth(spkdata.copy(), int(1000 / Params['basep']), int(FirstStimTime), Params['stdur'], Params['mf'])

            #Time to normalize by #cycles. May be variable for TORCs, since they could be shorter than length in exptparams
            Dsum = Dsum / (cnorm + (cnorm == 0))

            if Params['binsize'] > 1/Params['mf']:
                Dsum = insteadofbin(Dsum, Params['binsize'], Params['mf'])
            Dsum = Dsum * (1000/Params['binsize'])                                           #normalization by bin size

            if rec == 0:
                respcatval = Dsum
            else:
                respcatval = np.concatenate((respcatval,Dsum),axis=None)

        RespCat[rep] = respcatval

    avgResp = np.mean(np.array(list(RespCat.values())),axis=0)

    return avgResp


def strf_input_gen(stacked,TorcObject,exptparams,fs=1000):
    '''
    Generates the stimuli and response data that will be fed into our model
    :param stacked: Response data
    :param TorcObject: Stimulus info
    :param exptparams: Some experiment parameters
    :param fs: sampling frequency
    :return: Stimulus and response data as well as some useful parameters
    '''
    TorcNames = exptparams["TrialObject"][1]["ReferenceHandle"][1]["Names"]
    RefDuration = TorcObject['Duration']
    mf = int(fs/1000)
    stdur = int(RefDuration*1000)

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

    frqs = np.unique(np.concatenate(all_freqs))  # Smoosh to one array and output unique elements
    vels = np.unique(np.concatenate(all_velos))  # temporal spectra
    HighestFrequency = int(np.unique(all_hfreq))
    LowestFrequency = int(np.unique(all_lfreq))
    Octaves = np.log2(HighestFrequency / LowestFrequency)

    Params = dict()
    Params['stdur'] = stdur
    Params['mf'] = mf
    Params['lfreq'] = LowestFrequency
    Params['hfreq'] = HighestFrequency
    Params['octaves'] = int(Octaves)
    W = vels                                                                                 #array of ripple velocities
    T = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    Params['T'] = T

    Ompos = [x for x in frqs if x >= 0]                                                      #Get positive frequencies
    Omnegzero = np.flip([x for x in frqs if x <= 0])                                         #just used for populating an array a few lines down

    Omega = np.swapaxes(np.stack((Ompos,Omnegzero)),0,1)                                     #Make an array for main output Omega

    numvels = len(W)                                                                         #
    numfrqs = np.size(Omega,0)                                                               #Used to make empty array to be populated by params
    numstim = len(TorcNames)

    waveParams = np.empty([2,numvels,numfrqs,2,numstim])

    ##This part in MATLAB makes T, octaves, maxv, maxf, saf, numcomp
    basep = int(np.round(fs/min(np.abs(np.diff(np.unique([x for x in W if x != 0]))))))
    maxvel = np.max(np.abs(W))
    maxfrq = np.max(np.abs(Omega))
    saf = int(np.ceil(maxvel*2 + 1000/basep))
    numcomp = int(np.ceil(maxfrq*2*Octaves + 1))
    Params['numcomp'] = numcomp
    Params['basep'] = basep

    ##function [ststims,freqs]=stimprofile(waveParams,W,Omega,lfreq,hfreq,numcomp,T,saf);
    [ap,Ws,Omegas,lr,numstim] = waveParams.shape                               #wave params is 5D, define size of each dimension
    [a,b] = Omega.shape                                                        #splitting ripple freqs to matrix nums
    [d] = W.shape                                                              #splitting W into same

    if a*b*d != Omegas*Ws*lr:
        print('Omega and.or W do not match waveParams')

    sffact = saf/1000                                                        #lower sample rate
    leng = int(np.round(T*sffact))                                           #stim duration with lower sampling rounded
    Params['leng'] = leng

    ###this part is important###
    TorcValues = dict()                                                                #make new dict for function output
    for key,value in TorcParams.items():                                               #cycle through with all the different TORC names and respective values
        y_sum = torcmaker(value, Params)                                                #output of function (TORC) assigned to variable
        TorcValues[key] = y_sum                                                        #update dict with the key you are on and the value the function just returned

    ModulationDepth = 0.9                                                              #default set in matlab program
    xSize = int(np.round(10*numcomp/Octaves))                                          #new x sampling rate
    tSize = int(10*saf*basep/1000)                                                     #new t sampling rate
    ScaledTorcs = dict()                                                               #new dictionary for scaled torcs
    for key,value in TorcValues.items():                                               #go through this with all torcs
        [xsiz, tsiz] = value.shape                                                     #basic dims of torc
        temp = value                                                                   #pull out numbers to usable variable
        if xSize != xsiz & tSize != tsiz:                                              #when new sampling rate doesn't equal old vals
            temp1 = interpft(interpft(temp, xSize, 0), tSize, 1)                           #add points, yielding bigger array

            scl = np.max(np.abs([np.min(np.min(temp1)), np.max(np.max(temp1))]))       #largest |value| is scale factor

            temp2 = temp*ModulationDepth/scl                                   #transform original torc values with moddep and calc'd scale
        ScaledTorcs[key] = temp2

    # Create S* 375x25x30, stimulus data
    stimall = stimulus_compiler(ScaledTorcs)

    [stimX,stimT] = temp.shape                                                                   #have var for dims of torc
    binsize = int(basep/stimT)
    Params['binsize'] = binsize

    # create response data normalized to bins and averaged across repetitions
    avgResp = response_compiler(stacked,Params)

    return stimall,avgResp,Params