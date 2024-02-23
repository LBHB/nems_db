import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from nems0.analysis.gammatone.gtgram import gtgram

# 2024_01_19 to make /auto/data/sounds/backgrounds/v3
SOUND_ROOT = f"/auto/data/sounds/backgrounds/v2/*.wav"
path_dir = glob.glob((SOUND_ROOT))
path_dir.sort()

num, save, start, name = 1, 'Chimes', 100, '00cat78rec1chimesinthewindexcerpt1'
num, save, start, name = 2, 'Stream', 0, '00cat516rec1streamexcerpt1'
num, save, start, name = 3, 'Bees', 0, 'cat23rec1beesbuzzingexcerpt1'
num, save, start, name = 4, 'Blender', 46, 'cat31rec1blenderexcerpt1'
num, save, start, name = 5, 'Bulldozer', 72, 'cat45rec1bulldozersoundideas1141excerpt1'
num, save, start, name = 6, 'Bus', 100, 'cat47rec1busexcerpt1'
num, save, start, name = 7, 'Idling', 20, 'cat61rec1idlingsoundideas522excerpt1'
num, save, start, name = 8, 'Chainsaw', 0, 'cat73rec1chainsawexcerpt1'
num, save, start, name = 9, 'Coffee', 62, 'cat87rec1coffeemachineexcerpt1'
num, save, start, name = 10, 'Dentist', 93, 'cat106rec1dentistdrillsoundideas1803excerpt1'
num, save, start, name = 11, 'Jackhammer', 0, 'cat129rec1jackhammerexcerpt1'
num, save, start, name = 12, 'Drill', 58, 'cat140rec1handdrillsoundideas1245excerpt1'
num, save, start, name = 13, 'Film', 50, 'cat149rec1filmreelfreesoundsbone666138excerpt1'
num, save, start, name = 14, 'Hairdryer', 26, 'cat185rec1hairdryerexcerpt1'
num, save, start, name = 15, 'Metal Music', 0, 'cat193rec2heavymetalaliceinchainsrainwhenidie239secexcerpt1'
num, save, start, name = 16, 'Tuning', 52, 'cat255rec1orchestratuningupexcerpt1'
num, save, start, name = 17, 'Rain', 54, 'cat287rec1rainexcerpt1'
num, save, start, name = 18, 'Rock Tumble', 70, 'cat301rec1envsoundsrocktumblingsoundideas38272secexcerpt1'
num, save, start, name = 19, 'Wind', 0, 'cat312rec1windexcerpt1'
num, save, start, name = 20, 'Sander', 41, 'cat313rec1electricsanderexcerpt1'
num, save, start, name = 21, 'Shaver', 5, 'cat329rec1electricshaverexcerpt1'
num, save, start, name = 22, 'Thunder', 18, 'cat368rec1thunderexcerpt1'
num, save, start, name = 23, 'Train Bell', 3, 'cat373rec1trainwarningbellexcerpt1'
num, save, start, name = 24, 'Train Whistle', 50, 'cat375rec1trainwhistleexcerpt1'
num, save, start, name = 25, 'Waves', 54, 'cat403rec1wavesexcerpt1'
num, save, start, name = 26, 'Blinds', 0, 'cat409rec1windowblindsexcerpt1'
num, save, start, name = 27, 'Rock Music', 0, 'cat434rec160srockbeatlesshecameinthroughthebathroomwindowexcerpt1'
num, save, start, name = 28, 'Waterfall', 0, 'cat534rec1waterfallexcerpt1'
num, save, start, name = 29, 'Wipers', 0, 'cat538rec1windshieldwipersexcerpt1'
num, save, start, name = 30, 'Gravel', 0, 'cat558rec1tirerollingongravelexcerpt1'
num, save, start, name = 31, 'Insect Buzz', 75, 'cat565rec1insectsbuzzingexcerpt1'
num, save, start, name = 32, 'Wildebeest', 9, 'cat567rec1animalherdwildebeestexcerpt1'
num, save, start, name = 33, 'Tractor', 51, 'cat581rec1tractorexcerpt1'
num, save, start, name = 34, 'Drain', 9, 'cat598rec1waterdrainingexcerpt1'


ROOT = f"/auto/data/sounds/backgrounds/v2/"
SAVEROOT = f"/auto/data/sounds/backgrounds/v3/"

plot_wav_with_power(ROOT, name)

create_truncated_wav(ROOT, name, start, seconds=3, number=format(num, '02d'), savename=save, SAVEROOT=SAVEROOT)

#####
#####

# 2024_01_11 stuff
# SOUND_ROOT = f"/auto/users/lbhb/sounds/*.wav"
# path_dir = glob.glob((SOUND_ROOT))
# path_dir.sort()

name_start_dir = ['00cat668_rec1_ferret_fights_Athena-Violet001_excerpt1',    #save='Fight AV1', num=43, start=66
                 'cat668_rec1_ferret_fights_Athena-Violet001_excerpt2',       #save='Fight AV2', num=44, start=26
                 'cat668_rec1_ferret_fights_Athena-Violet001_excerpt3',       #save='Fight AV3', num=45, start=12
                 'cat668_rec1_ferret_fights_Athena-Violet001_excerpt4',       #save='Fight AV4', num=46, start=53
                 'cat668_rec1_ferret_fights_Athena-Violet001_excerpt5',       #save='Fight AV5', num=47, start=89
                 'cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt1',      #save='Fight JV1', num=48, start=11
                 'cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt2',      #save='Fight JV2', num=49, start=180
                 'cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt3',      #save='Fight JV3', num=50, start=97
                 'cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt4',      #save='Fight JV4', num=51, start=5
                 '00cat668_rec3_ferret_kits_51p9-8-10-11-12-13-14_excerpt1',  #save='Kit High 51p1', num=52, start=79
                 'cat668_rec3_ferret_kits_51p9-8-10-11-12-13-14_excerpt2',    #save='Kit High 51p2', num=53, start=0
                 'cat668_rec3_ferret_kits_51p9-8-10-11-12-13-14_excerpt3',    #save='Kit High 51p3', num=54, start=7
                 'cat668_rec3_ferret_kits_51p9-8-10-11-12-13-14_excerpt4',    #save='Kit High 51p4', num=55, start=7
                 'cat668_rec3_ferret_kits_51p9-8-10-11-12-13-14_excerpt5',    #save='Kit High 51p5', num=56, start=7
                 'cat668_rec4_ferret_kits_54p1-2-3-4-6_excerpt1',             #save='Kit Groan 54p1', num=57, start=1
                 'cat668_rec4_ferret_kits_54p1-2-3-4-6_excerpt2',             #save='Kit Groan 54p2', num=58, start=2
                 'cat668_rec4_ferret_kits_54p1-2-3-4-6_excerpt3',             #save='Kit Groan 54p3', num=59, start=6
                 'cat668_rec4_ferret_kits_54p1-2-3-4-6_excerpt4',             #save='Kit Groan 54p4', num=60, start=3
                 'cat668_rec5_ferret_kits_0p1-2-3-4-5-6_excerpt1',            #save='Kit Groan 0p1', num=61, start=6
                 'cat668_rec5_ferret_kits_0p1-2-3-4-5-6_excerpt2',            #save='Kit Groan 0p2', num=62, start=1
                 'cat668_rec5_ferret_kits_0p1-2-3-4-5-6_excerpt3',            #save='Kit Groan 0p3', num=63, start=7
                 'cat668_rec5_ferret_kits_0p1-2-3-4-5-6_excerpt4',            #save='Kit Groan 0p4', num=64, start=1
                 'cat668_rec6_ferret_kits_18p1-2-3-5-7-8_excerpt1',           #save='Kit Whine 18p1', num=65, start=0
                 'cat668_rec6_ferret_kits_18p1-2-3-5-7-8_excerpt2',           #save='Kit Whine 18p2', num=66, start=2
                 'cat668_rec6_ferret_kits_18p1-2-3-5-7-8_excerpt3',           #save='Kit Whine 18p3', num=67, start=6
                 'cat668_rec6_ferret_kits_18p1-2-3-5-7-8_excerpt7',           #save='Kit Whine 18p4', num=68, start=5
                 '00cat668_rec7_ferret_oxford_male_chopped_excerpt1',         #save='Gobble OX1', num=69, start=0
                 'cat668_rec7_ferret_oxford_male_chopped_excerpt2',           #save='Gobble OX2', num=70, start=0
                 'cat668_rec7_ferret_oxford_male_chopped_excerpt3',           #save='Gobble OX3', num=71, start=0
                 'cat668_rec7_ferret_oxford_male_chopped_excerpt4',           #save='Gobble OX4', num=72, start=53
                 'cat668_rec7_ferret_oxford_male_chopped_excerpt5',           #save='Gobble OX5', num=73, start=0
                 'cat668_rec7_ferret_oxford_male_chopped_excerpt6']           #save='Gobble OX6', num=74, start=59

ROOT = f"/auto/users/lbhb/sounds/"

plot_wav_with_power(ROOT, name)

create_truncated_wav(ROOT, name, start, seconds=3, number='GRH', savename=save)
create_truncated_wav(ROOT, name, start, seconds=1, number=f'{num}', savename=save)


##To make 'GRHFight JV2'
name = 'cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt2'
filepath = ROOT + name + '.wav'
fs, W = wavfile.read(filepath)
factor=int(fs/100)
start = 180*factor
one_sec = W[start:]

name = 'cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt3'
filepath = ROOT + name + '.wav'
fs, W = wavfile.read(filepath)
start = 152*factor
factor = int(fs/100)
sec_sec = W[start:start + 35280]

dd = np.concatenate((one_sec, sec_sec), axis=0)
spec = gtgram(dd, fs, 0.02, 0.01, 48, 100, 20000)
get_norm(spec, name, threshold=0.025, hfreq=20000)
number, seconds, savename, 'GRH', 3, 'Fight JV2'
SAVE_PATH = f'{ROOT}{seconds}sec/{number}{savename}.wav'
wavfile.write(SAVE_PATH, fs, dd)
####

##To make 042, 1second
name = 'ferretb2001R'
savename = "Kit Squeak"
filepath = ROOT + name + '.wav'
fs, W = wavfile.read(filepath)
factor = int(fs/100)
start, start2 = 13, 71
start = start*factor
start2 = start2 * factor
first = W[start:start + int(((factor*100) / 2))]
second = W[start2:start2 + int(((factor*100) / 2))]
one_sec = np.concatenate((first, second), axis=0)

print(len(one_sec))
seconds, number, hfreq, lfreq = 1, 42, 20000, 100
spec = gtgram(one_sec, fs, 0.02, 0.01, 48, lfreq, hfreq)
get_norm(spec, name, threshold=0.025, hfreq=hfreq)
SAVE_PATH = f'{ROOT}{seconds}sec/{number}{savename}.wav'

wavfile.write(SAVE_PATH, fs, one_sec)
####


def plot_wav_with_power(ROOT, name):
    filepath = ROOT + name + '.wav'
    fs, W = wavfile.read(filepath)
    spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 20000)
    get_norm(spec, name, threshold=0.025, hfreq=20000)

def create_truncated_wav(ROOT, name, start, seconds=3, number='',
                         lfreq=100, hfreq=20000, savename=name, SAVEROOT=None):
    filepath = ROOT + name + '.wav'
    fs, W = wavfile.read(filepath)
    factor = int(fs/100)
    print(f'fs is {fs}')
    start = start * factor
    three_sec = W[start:start + ((factor * 100) * seconds)]
    print(len(three_sec))
    spec = gtgram(three_sec, fs, 0.02, 0.01, 48, lfreq, hfreq)
    # get_z(spec, name, threshold=0.15)
    get_norm(spec, name, threshold=0.025, hfreq=hfreq)
    # savename = name.replace('_', '+').replace('-', '=')
    if SAVEROOT:
        SAVE_PATH = f'{SAVEROOT}{number}{savename}.wav'
    else:
        SAVE_PATH = f'{ROOT}{seconds}sec/{number}{savename}.wav'
    NAME = SAVE_PATH.split('/')[-1].split('.')[0]
    print(str(NAME))
    print(name)
    print(spec.shape)

    wavfile.write(SAVE_PATH, fs, three_sec)


## End of stuff Greg added 2024_01_09

##Old Stuff
def get_norm(spec, name, threshold=0.05, hfreq=20000):
    '''Plots the spectrogram you're working with, with a panel below of the z-score
    and below that the difference in adjacent z-scores. Vertical lines are placed
    when the z-score difference between two bins exceeds a defined threshold.'''
    import numpy as np
    av = spec.mean(axis=0)
    big = np.max(av)
    norm = av/big

    nonstation = np.nanmean(np.std(spec, axis=1),axis=0)
    freq_mean = np.average(spec, axis=1)
    mean_idx = np.argmax(freq_mean, axis=0)
    x_freq = np.logspace(np.log2(100), np.log2(hfreq), num=48, base=2)
    cf = x_freq[mean_idx]

    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].imshow(spec, aspect='auto', origin='lower')
    ax[0].set_ylabel('Frequency (kHz)')
    freqs = np.logspace(np.log2(100), np.log2(hfreq), 48, base=2)
    idxs = [0,12,24,36,47]
    freq = np.round([freqs[i] for i in idxs])/1000
    ax[0].set_yticks(idxs)
    ax[0].set_yticklabels(freq)
    ax[1].plot(norm)
    ax[1].set_ylabel('normalized')
    ax[0].vlines(50,3,45,color='white', ls=':')
    ticks = np.arange(0, spec.shape[1], 5)
    ax[1].set_xticks(ticks)
    di = np.diff(norm)
    ax[2].plot(di)
    ax[2].set_ylabel('difference')
    goods = []
    for x in range(len(di)):
        if x == 0 and di[x] > threshold:
            goods.append(x)
        if x != 0:
            if di[x] > threshold and di[x-1] < threshold:
                goods.append(x)
    min,max = ax[1].get_ylim()
    ax[1].vlines(goods, min, max, ls=':')
    ax[2].vlines(goods, min, max, ls=':')
    fig.suptitle(f'{name} - threshold: {threshold}\nNon-stationariness {np.round(nonstation)} -'
                 f'Center Frequency {np.round(cf)}')

def get_z(spec, name, threshold=0.15):
    '''Plots the spectrogram you're working with, with a panel below of the z-score
    and below that the difference in adjacent z-scores. Vertical lines are placed
    when the z-score difference between two bins exceeds a defined threshold.'''
    import numpy as np
    av = spec.mean(axis=0)
    me = av.mean()
    sd = np.std(av)
    zz = (av - me) / sd
    di = np.diff(zz)
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].imshow(spec, aspect='auto', origin='lower')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[1].plot(zz)
    ax[1].set_ylabel('z-score')
    ax[0].vlines(50,3,45,color='white', ls=':')
    ticks = np.arange(0, spec.shape[1], 5)
    ax[1].set_xticks(ticks)
    ax[2].plot(di)
    ax[2].set_ylabel('z-score difference')
    goods = np.where(di > threshold)[0].tolist()
    min,max = ax[1].get_ylim()
    ax[1].vlines(goods, min, max, ls=':')
    ax[2].vlines(goods, min, max, ls=':')
    fig.suptitle(f'{name}')


def full_spec(ROOT, name, kind, factor=441):
    '''Displays an entire 4s spectrogram and z-score for looking around to
    decide what chunk you want'''
    filepath = ROOT + name + '.wav'
    fs, W = wavfile.read(filepath)
    spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 20000)
    # get_z(spec, name, threshold=0.15)
    get_norm(spec, name, threshold=0.025, hfreq=20000)

    np.logspace(np.log2(100), np.log2(20000), 48, base=2)