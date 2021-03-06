import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

from nems.analysis.gammatone.gtgram import gtgram
from nems import recording, signal
from nems import xforms, get_setting
import random

def make_three_trials(all_trials, all_epochs, stim_dict, cut_dir, fs, cats,
                      trial_length=15, number_bins=6, overlap=2):
    '''Will create the three complementary (interleaved?) trials and add them to the directories'''
    trial, trial_labels, bin_start_times = empty_triad(trial_length, number_bins, fs, overlap)

    all_trials, all_epochs = generate_triad(all_trials, all_epochs, trial, trial_labels,
                                            bin_start_times, cats, fs, seg_dict, seg_label_list)

    return all_trials, all_epochs

def cut_decider(num_segmenters, seg_length, segment_min_s = 0.5, segment_max_s = 2):
    '''Tell it how many different sets of cuts you want per sound and how long the songs
    you'll be chopping, along with your minimum and maximum length of the segments'''
    cut_dir = []
    for bb in range(num_segmenters):
        cuts = segment_gen(seg_length, segment_min_s, segment_max_s)
        cut_dir.append(cuts)

    return cut_dir

def load_wavs(sounds, classes):
    '''This will load all of the wave files and keep it tidy with the name of the sound, its class
    and it will also check to make sure they're all sampled at the same rate, and if not, it will
    resample them.'''
    stim_dict = {}
    fs_list = []
    for nn, (c, s) in enumerate(zip(sound_classes, sound_files)):
        fs, W = wavfile.read(s)
        basename = os.path.splitext(os.path.basename(s))[0]
        stim_dict[nn] = (c, basename, W)
        fs_list.append(fs)

    uniques = np.unique(fs_list)
    length = int(len(W) / int(uniques))
    print(f'The sampling frequencies in this set are {uniques}')
    # Need to ask Stephen how to up sample, for my dummy data shouldn't be hard.
    # print(f'The length of unique is {len(uniques)}')
    # if len(uniques) > 1:

    return stim_dict, uniques[0], length

def empty_triad(length=15,bins=6,fs=48000,overlap=None):
    '''Basic trial structure, you specify the length and how many
    bins for samples you want (with the option of overlap, where you define
    how many bins should have overlap and it'll randomly pick em), and
    it'll create that in seconds before multiplying by the sampling
    rate

    It'll create the three complementary trials and also will create a list (trial_fills)
    with the keys that will later be the sounds to populate the trial with.'''
    # length,bins,fs,overlap = 15,6,48000,2
    trial = np.zeros((3, length * fs))
    # trial = [np.zeros(length*fs) * 3 for i in range(3)]

    bin_length = int(length / bins * fs)
    bin_start_times = [bin_length * i for i in range(bins)]
    trial_labels = [[0] * bins for i in range(3)]
    extra_starts = [[] * r for r in range(3)]
    if overlap:
        if (bins/overlap).is_integer():
            extra = random.sample(range(0,bins),bins)
            # n = int(bins/3)
            # overlap_positions = [extra[i * n:(i + 1) * n] for i in range((len(extra) + n - 1) // n)]
            for qq in range(len(extra)):
                # extra_starts[qq % 3].append(bin_start_times[extra[qq]])
                extra_starts[qq % 3].append(extra[qq])
            A_starts = [extra_starts[1],extra_starts[2],extra_starts[0]]
            B_starts = [extra_starts[2],extra_starts[0],extra_starts[1]]
            for tri in range(3):
                for ab in extra_starts[tri]:
                    trial_labels[tri][ab] = f'A{ab+1}+B{ab+1}'
                for aa in A_starts[tri]:
                    trial_labels[tri][aa] = f'A{aa+1}'
                for bb in B_starts[tri]:
                    trial_labels[tri][bb] = f'B{bb+1}'

    return trial, trial_labels, bin_start_times

def segment_gen(len_sec, min_seg=0.5, max_seg=2.0):
    '''Creates a list that gets used as the positions within a wav file to make cuts
    that make use of the entire sound segment. For more universal function I made it so
    the number of pieces it creates corresponds to the number of seconds. I have a version
    where you can define it, but I wasn't yet able to have it handle extreme cases and
    parameters well yet. This version works well for my 3 or 4s stimuli at the moment'''
    if max_seg >= len_sec:
        add_list = []
        print('The longest segment length you want is too long for your sound file!')
    elif min_seg < 0:
        add_list = []
        print('The minimum segment cannot be below zero!')
    elif max_seg - min_seg < 0:
        add_list = []
        print('Minimum segment length must be smaller than the maximum segment length!')
    else:
        pieces = int(len_sec)
        list, add_list = [min_seg] * pieces, []
        ii, time_remaining, total, max_add = 1, len_sec - np.sum(list), 0, max_seg - min_seg
        # print(f'Length {len_sec}, Remaining {time_remaining}, Min {min_seg}, Pieces {pieces}, Max to add {max_add}')
        for aa in range(pieces):
            if time_remaining < max_add and aa == pieces - 1:
                sample = np.round(time_remaining,1)
                # print(f'I used this to add {time_remaining}')
            elif time_remaining > max_add and aa == pieces - 1:
                print('There is an error from the random generator and you might consider rerunning this.')
            if max_add <= time_remaining and aa != pieces - 1:
                sample = np.round(np.random.uniform(0, max_add), 1)
                # print(f'max_add {max_add} <= time_remaining {time_remaining}')
            if max_add > time_remaining and aa != pieces - 1:
                sample = np.round(np.random.uniform(0, time_remaining), 1)
            add_list.append(sample)
            # print(f'On the {aa+1} loop I added {sample} to give {add_list}, tr {time_remaining - sample}')
            time_remaining = time_remaining - sample
        zip_list = zip(list, np.abs(add_list))
        cut_list = [x + y for (x, y) in zip_list]
        print(f'Generated a list of cuts {cut_list} seconds in duration.')

    return cut_list

def sound_segmenter(stim_dict, cuts, fs, seg_label_list=None, seg_dict=None):
    '''Takes your dictionary of sounds (labeled and categorized) and iterates through chopping
    each sound by however many cuts lists one has. If it is the first time you are running it
    it will make a new output dictionary that contains the label and the segment waveform, as
    well as a list of labels that correspond to the dictionary keys.'''
    if not seg_dict:
        seg_dict = {}
    if not seg_label_list:
        seg_label_list = []

    for c, s in stim_dict.items():
        for nn, cc in enumerate(cuts):
            start, stop = 0, 1
            samples_list = [0]
            for vv in cc:
                seg_samples = vv * fs  # convert seconds to samples
                sample_add = samples_list[-1] + seg_samples
                samples_list.append(int(sample_add))
            seconds_list = [x / fs for x in samples_list]
            while stop <= len(cc):
                segment = stim_dict[c][2][samples_list[start]:samples_list[stop]]
                label = f'{stim_dict[c][1]}_{seconds_list[start]}-{seconds_list[stop]}s_dur-{cc[start]}_cutids-{nn}_CLASS-0{int(stim_dict[c][0])}'
                seg_dict[label] = segment
                seg_label_list.append(label)
                start += 1
                stop += 1
    return seg_dict, seg_label_list

def populate_triad(trial, trial_labels, start_times, cats, fs, seg_dict, seg_label_list):
    bin_length = (start_times[1] - start_times[0]) / fs
    start_times_s = start_times / fs

    sound_dict = {}
    sort_df = pd.DataFrame(columns=['Label', 'Name', 'Class', 'On', 'Off', 'Segment'],)
    real_epochs = pd.DataFrame(columns=['Name', 'Class', 'On', 'Off'])
    # sound_list = []
    for bn, tt in enumerate(start_times_s):
        wave_A, sound_dict, sort_df = sound_assigner('A', bn, tt, cats, seg_label_list,
                                                     seg_dict, sound_dict, sort_df, bin_length, fs)
        wave_B, sound_dict, sort_df = sound_assigner('B', bn, tt, cats, seg_label_list,
                                                     seg_dict, sound_dict, sort_df, bin_length, fs)

        wave_AB = wave_A + wave_B
        #Going to have to just tell it that both are there from the 12 in sort_df and pull both rows
        sound_dict[f'A{bn+1}+B{bn+1}'] = wave_AB
        # sound_list.append(f'A{bn+1}+B{bn+1}')

    sort_df = sort_df.set_index('Label')

    for num, pop in enumerate(trial_labels):
        trial[start_times[num]:int(start_times[num]+(bin_length*fs))] = sound_dict[pop]
        if '+' in pop:
            A, B = pop.split('+')
            row = {'Name': sort_df.loc[A]['Name'],
               'Class': sort_df.loc[A]['Class'], 'On': sort_df.loc[A]['On'],
               'Off': sort_df.loc[A]['Off']}
            real_epochs = real_epochs.append(row, ignore_index=True)
            row = {'Name': sort_df.loc[B]['Name'],
               'Class': sort_df.loc[B]['Class'], 'On': sort_df.loc[B]['On'],
               'Off': sort_df.loc[B]['Off']}
            real_epochs = real_epochs.append(row, ignore_index=True)
        else:
            row = {'Name': sort_df.loc[pop]['Name'],
               'Class': sort_df.loc[pop]['Class'], 'On': sort_df.loc[pop]['On'],
               'Off': sort_df.loc[pop]['Off']}
            real_epochs = real_epochs.append(row, ignore_index=True)

    return trial, real_epochs

def sound_assigner(trial_letter, count, start, cats, seg_label_list,
                   seg_dict, sound_dict, sort_df, bin_length,
                   fs):
    cat = random.randrange(1, cats + 1, 1)
    good_list = []
    for hh in seg_label_list:
        if f'CLASS-0{cat}' in hh:
            good_list.append(hh)
    # if 'B' in trial_letter:         # So two overlapping sounds can't be of exactly the same clip, maybe needless
    #     label_df = sort_df.copy().set_index(['Label'])
    #     label_A = label_df.loc[f'A{count + 1}']['Name']
    #     for jj in seg_label_list:
    #         if f'{label_A}' in jj:
    #             good_list.remove(jj)
    random.shuffle(good_list)
    add_label = random.choice(good_list)
    add_wave = seg_dict[add_label]

    seg_len = len(add_wave) / fs

    if 'A' in trial_letter:  #The first segment gets randomly placed, sound B will be relative to it to ensure overlap
        max_jitter_length = np.round(bin_length - seg_len, 1)
        jitter = np.round(np.random.uniform(0, max_jitter_length), 1)
        # on_time = start + jitter
        pre = np.zeros(int(jitter * fs))
        post = np.zeros(int((bin_length - (jitter + seg_len)) * fs))

    if 'B' in trial_letter:
        df = sort_df.copy().set_index(['Label'])
        A_on, A_off = df.loc[f'A{count + 1}']['On'] - start, df.loc[f'A{count + 1}']['Off'] - start

        max_start = np.round(A_off - 0.5, 1)
        if max_start + seg_len > bin_length:
            max_start = bin_length - seg_len
        min_start = np.round((A_on + 0.5) - seg_len, 1)
        if min_start < 0:
            min_start = 0
        max_range = np.round(max_start - A_on, 1)
        min_range = np.round(min_start - A_on, 1)

        jitter_values, jit = [i for i in np.arange(-bin_length,bin_length,0.25)], 0
        jitter_list = [i for i in jitter_values if min_range < i < max_range]
        if len(jitter_list) == 0:
            jitter_list = [0]
        offset_from_A = random.choice(jitter_list)
        jitter = A_on + offset_from_A
        pre = np.zeros(int(jitter * fs))
        post = np.zeros(int((bin_length - (jitter + seg_len)) * fs))

    # off_time = on_time + seg_len
    wave = np.concatenate((pre, add_wave, post))
    if len(wave) != int((bin_length * fs)):
        for ab in range(int((bin_length * fs) - len(wave))):
            wave = np.append(wave, 0)

    sound_dict[f'{trial_letter}{count + 1}'] = wave
    # sound_list.append(f'A{bn+1}')
    row = {'Label': f'{trial_letter}{count + 1}', 'Name': 'STIM_' + add_label.split('_')[0], 'Class': f"CLASS_0{cat}", 'On': np.round(jitter + start, 2),
           'Off': np.round(jitter + start + seg_len, 2), 'Segment': add_label.split('_')[1]}
    sort_df = sort_df.append(row, ignore_index=True)

    return wave, sound_dict, sort_df

def generate_triad(all_trials, all_epochs, trial, trial_labels, bin_start_times,
                   cats, fs, seg_dict, seg_label_list):
    '''Takes everything we constructed for that trial structure and actually makes them,
    giving us a dictionary with which trial it is we've created with it's corresponding waveform
    as well as one that corresponds to the list of epochs in the trial.'''
    for trl in range(len(trial)):
        single_trial, single_epoch = populate_triad(trial[trl], trial_labels[trl], bin_start_times, cats, fs, seg_dict, seg_label_list)
        trial_count = len(all_trials) + 1
        if trial_count <= 9:
            all_trials[f'Trial_00{trial_count}'] = single_trial
            all_epochs[f'Trial_00{trial_count}'] = single_epoch
        if 9 < trial_count < 100:
            all_trials[f'Trial_0{trial_count}'] = single_trial
            all_epochs[f'Trial_0{trial_count}'] = single_epoch
        if trial_count >= 100:
            all_trials[f'Trial_{trial_count}'] = single_trial
            all_epochs[f'Trial_{trial_count}'] = single_epoch

    return all_trials, all_epochs


BAPHY_ROOT = "/Users/grego/baphy"
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")
class_labels = ['Ferret']
cats = len(class_labels)

dir1 = os.path.join(BAPHY_ROOT, "SoundObjects", "@FerretVocal", "Sounds_set4", "*.wav")
# dir2 = os.path.join(BAPHY_ROOT, "SoundObjects", "@Speech", "sounds", "*sa1.wav")
set1 = glob.glob(dir1)
# set2 = glob.glob(dir2)
sound_set1 = set1
# sound_set2 = set2

sound_classes = (np.zeros(len(set1)) + 1)
sound_files = set1
# sound_classes = np.concatenate((np.zeros(len(set1)) + 1,
#                                 np.zeros(len(set2)) + 2,))
# sound_files = set1 + set2



unique_triads = 1
num_cut_sets = 5
all_trials, all_epochs = {}, {}

all_trials, all_epochs = {}, {}
#Load all the sound files we may use and gives them class assignments, too
stim_dict, fs, seg_len = load_wavs(sound_files,sound_classes)

#Make a bunch of sets of cut points and then cut all the files you loaded
cut_dir = cut_decider(num_cut_sets, seg_len)
seg_dict, seg_label_list = sound_segmenter(stim_dict, cut_dir, fs)

#Take and select some segments for each segment position and then populate each of three trials and
#add it to the directories to keep track of the trial waveforms and corresponding epochs
for mm in range(unique_triads):
    all_trials, all_epochs = make_three_trials(all_trials, all_epochs, stim_dict,
                                               cut_dir, fs, cats, 15, 6, 2)

