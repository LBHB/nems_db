import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nems0 import db

plt.ion()


#Class that contains dataframe for a single animal
#with plotting and analysis functions
class behav:
    def __init__(self, animal, task, days='single', day=0, migrate_only=True,
                 non_migrate_blocks=''):

        self.animal = animal
        self.task = task
        self.days = days
        self.day = day
        self.migrate_only = migrate_only
        self.non_migrate_blocks = non_migrate_blocks
        self.dataframe, self.all_data_extensive = self.import_data()


    def import_data(self):
        sql = f"SELECT * FROM gDataRaw WHERE runclass = '{self.task}' AND cellid like '{self.animal}%' AND training=1 AND bad=0 order by id"

        d_rawfiles = db.pd_query(sql)

        # filter out "bad" sessions
        good_sessions = np.isfinite(d_rawfiles['trials']) & (d_rawfiles['trials'] > 10)

        d_rawfiles = d_rawfiles.loc[good_sessions].reset_index(drop=True)

        # plt.figure()
        # plt.plot(d_rawfiles['trials'])

        # Identify which experiments had variable snr
        # snr_indexes = np.arange(39, 53)
        if self.days == 'single':
            if self.day == 0:
                real_task_indexes = [(d_rawfiles.shape[0]),]
            else:
                real_task_indexes = [(d_rawfiles.shape[0] - (1 + self.day)),]
        elif self.days == 'all':
            real_task_indexes = np.arange(0, (d_rawfiles.shape[0]))

        all_data = []
        all_data_extensive = []

        for i in real_task_indexes:
            valid = True
            session_id = i

            r = d_rawfiles.loc[session_id]
            path = os.path.join(r['resppath'], r['parmfile'])
            triallog = os.path.join(path, 'trial_log.csv')
            d = pd.read_csv(triallog)

            if self.task == 'NFB':
                if self.migrate_only == True:
                    if 'migrate_trial' in d.columns:
                        if self.non_migrate_blocks:
                            if 1 in d['migrate_trial'].unique():
                                valid = False
                        elif self.non_migrate_blocks == False:
                            if 1 not in d['migrate_trial'].unique():
                                valid = False
                        d['trial_duration'] = d['response_start'] - d['np_start']
                        d['spatial_config'] = 'same'
                        d.loc[d['bg_channel'] == d['fg_channel'], 'spatial_config'] = 'diff'
                        d.loc[d['bg_channel'] == -1, 'spatial_config'] = 'dichotic'

                        interesting_data = d[
                            ['trial_number', 'response', 'correct', 'response_time', 'hold_duration', 'snr',
                             'spatial_config', 'migrate_trial', 'trial_duration']].copy()
                        interesting_data['parmfile'] = r['parmfile']
                        interesting_data['session_id'] = session_id
                        interesting_data['day'] = r['cellid']
                    else:
                        valid = False
                else:
                    if 'snr' in d.columns:
                        if 'migrate_trial' in d.columns:
                            valid = False
                        d['trial_duration'] = d['response_start'] - d['np_start']
                        d['spatial_config'] = 'same'
                        d.loc[d['bg_channel'] == d['fg_channel'], 'spatial_config'] = 'diff'
                        d.loc[d['bg_channel'] == -1, 'spatial_config'] = 'dichotic'

                        interesting_data = d[
                            ['trial_number', 'response', 'correct', 'response_time', 'hold_duration', 'snr',
                             'spatial_config', 'response_time', 'trial_duration']].copy()
                        interesting_data['parmfile'] = r['parmfile']
                        interesting_data['session_id'] = session_id
                        interesting_data['day'] = r['cellid']
                    else:
                        valid = False

            elif self.task == 'VOW':
                # Constructs spatial config columns
                d['trial_duration'] = d['response_start'] - d['np_start']
                d['spatial_config'] = 'monaural'
                d.loc[d['s1_name'] == d['s2_name'], 'spatial_config'] = 'binaural'
                # Constructs pitch column
                import re
                def label_based_on_three_digit_number(text):
                    pattern = r'(\d{3})'
                    match = re.search(pattern, text)
                    if match:
                        extracted_number = match.group(1)
                        if extracted_number in ["106", "151", "201"]:
                            return extracted_number
                    return "Not Matched"
                d['combined_stim'] = d['s1_name']
                d.loc[d['s1_name'].isna(), 'combined_stim'] = d['s2_name']
                d['pitch'] = d['combined_stim'].apply(label_based_on_three_digit_number)
                interesting_data = d[
                    ['trial_number', 'response', 'correct', 'response_time', 'snr', 'spatial_config', 'pitch',
                     'trial_duration']].copy()
                interesting_data['parmfile'] = r['parmfile']
                interesting_data['session_id'] = session_id
                interesting_data['day'] = r['cellid']

            if valid == True:
                all_data.append(interesting_data)
                all_data_extensive.append(d)

        interesting_data_all = pd.concat(all_data, ignore_index=True)
        interesting_data_all['early'] = (interesting_data_all['response'] == 'early_np').astype(float)
        interesting_data_all['incorrect'] = ((interesting_data_all['response'] != 'early_np') & \
                                             (interesting_data_all['correct'] == False)).astype(float)
        interesting_data_all['correct_val'] = (interesting_data_all['correct']).astype(float)

        return interesting_data_all, all_data_extensive

    #Prints all dataframe columns, and all unique days
    def variables(self):
        print(self.dataframe.columns)
        print(self.dataframe['day'].unique())

    def sample_plots(self, XVARIABLE, day_list=[], kind='bar'):

        if len(day_list) == 0:
            plot_frame = self.dataframe.copy()
        else:
            plot_frame = self.dataframe[self.dataframe['day'].isin(day_list)].copy()

        YVARIABLE = 'early_nosepoke'
        plot_frame['early_nosepoke'] = (plot_frame['response'] == 'early_np').astype(bool)
        series2 = plot_frame.groupby(XVARIABLE)[YVARIABLE].mean()
        ax = series2.plot(kind=kind)
        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0, 1])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'{YVARIABLE} by {XVARIABLE} for session {plot_frame["day"].unique()[0]}',
                  fontsize = 8)
        counts = plot_frame.groupby(XVARIABLE)[YVARIABLE].count().values
        for i, p in enumerate(ax.patches):
            ax.annotate(str(counts[i]), (p.get_x() * 1.005, p.get_height() * 0.005))

        plt.tight_layout()
        plt.show()
        print(plot_frame.groupby(XVARIABLE)[YVARIABLE].count())

        YVARIABLE = 'correct'
        Non_nose_poke = plot_frame.loc[(plot_frame['early_nosepoke'] == False)]
        series3 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()
        ax2 = series3.plot(kind=kind)
        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0, 1])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'{YVARIABLE} for session {Non_nose_poke["day"].unique()[0]} without early_nps',
                  fontsize = 8)
        counts2 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count().values
        for i, p in enumerate(ax2.patches):
            ax2.annotate(str(counts2[i]), (p.get_x() * 1.005, p.get_height() * 0.005))
        plt.tight_layout()
        plt.show()
        print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())

    def perform_over_time(self, XVARIABLE, day_list=[], kind='bar'):

        if len(day_list) == 0:
            plot_frame = self.dataframe.copy()
        else:
            plot_frame = self.dataframe[self.dataframe['day'].isin(day_list)].copy()

        YVARIABLE = 'early_nosepoke'
        plot_frame['early_nosepoke'] = (plot_frame['response'] == 'early_np').astype(bool)
        for i, v in enumerate(XVARIABLE):
            for j, w in enumerate(plot_frame[v].unique()):
                series = plot_frame[plot_frame[v] == w].groupby('day')[YVARIABLE].mean()
                series.plot(kind='line', label=f'{w}')
        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0, 1])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'{YVARIABLE} by {XVARIABLE} for session {plot_frame["day"].unique()[0]}',
                  fontsize = 8)
        plt.legend()
        ticks = np.arange(len(day_list))
        plt.xticks(ticks=ticks, labels=day_list, rotation=90)
        plt.tight_layout()
        plt.show()
        print(plot_frame.groupby(XVARIABLE)[YVARIABLE].count())

        YVARIABLE = 'correct'
        Non_nose_poke = plot_frame.loc[(plot_frame['early_nosepoke'] == False)]
        for i, v in enumerate(XVARIABLE):
            for j, w in enumerate(Non_nose_poke[v].unique()):
                series = Non_nose_poke[Non_nose_poke[v] == w].groupby('day')[YVARIABLE].mean()
                series.plot(kind='line', label=f'{w}')
        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0, 1])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'{YVARIABLE} for session {Non_nose_poke["day"].unique()[0]} without early_nps',
                  fontsize = 8)
        plt.legend()
        ticks = np.arange(len(day_list))
        plt.xticks(ticks=ticks, labels=day_list, rotation=90)
        plt.tight_layout()
        plt.show()
        print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())

    def remove_remind_trials(self):
        remind_trials = (self.dataframe['correct'].shift(1) == False) & (self.dataframe['response'].shift(1) != 'early_np')
        self.dataframe = self.dataframe[~remind_trials]

    def extract_post_migrate(self):
        false_trial_indexes = self.dataframe['migrate_trial'] == True
        false_shifted = false_trial_indexes.shift(-1, fill_value=False)
        self.dataframe = self.dataframe[false_shifted]


    def plot_trial_duration(self, XVARIABLE, day_list=[], kind='bar', YVARIABLE = 'trial_duration'):

        if len(day_list) == 0:
            plot_frame = self.dataframe.copy()
        else:
            plot_frame = self.dataframe[self.dataframe['day'].isin(day_list)].copy()

        # Non_nose_poke = plot_frame.loc[(plot_frame['early_nosepoke'] == False)]
        correct = plot_frame.loc[plot_frame['correct']]
        series3 = correct.groupby(XVARIABLE)[YVARIABLE].mean()
        ax2 = series3.plot(kind=kind)
        plt.axhline(y=0.5, linestyle='--')
        # plt.ylim([0, 1])
        plt.ylim([0.8, 1.5])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'{YVARIABLE} for session {correct["day"].unique()[0]} without early_nps',
                  fontsize = 8)
        counts2 = correct.groupby(XVARIABLE)[YVARIABLE].count().values
        for i, p in enumerate(ax2.patches):
            ax2.annotate(str(counts2[i]), (p.get_x() * 1.005, p.get_height() * 1.005))
        plt.tight_layout()
        plt.show()
        print(correct.groupby(XVARIABLE)[YVARIABLE].count())

if __name__ == "__main__":

#     #runclass NFB = 2afc, VOW = vowel
    example = behav('LMD', 'NFB', days='all', migrate_only=True, non_migrate_blocks='')
    print(example.dataframe.shape)
    example.remove_remind_trials()
    print(example.dataframe.shape)

    # example.variables()
    XVARIABLE = ['snr']
    day_list = []

    example.sample_plots(XVARIABLE, day_list)
    example.perform_over_time(XVARIABLE, day_list)
    # example.plot_trial_duration(XVARIABLE, day_list)

