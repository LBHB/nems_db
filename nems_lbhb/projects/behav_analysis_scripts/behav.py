import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nems0 import db

plt.ion()


#Class that contains dataframe for a single animal
#with plotting and analysis functions
class behav:
    def __init__(self, animal, task, days='single', day=0, migrate_only=True):

        self.animal = animal
        self.task = task
        self.days = days
        self.day = day
        self.migrate_only = migrate_only
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
                        d['trial_duration'] = d['response_start'] - d['np_start']
                        d['spatial_config'] = 'same'
                        d.loc[d['bg_channel'] == d['fg_channel'], 'spatial_config'] = 'diff'
                        d.loc[d['bg_channel'] == -1, 'spatial_config'] = 'dichotic'

                        interesting_data = d[
                            ['trial_number', 'response', 'correct', 'response_time', 'hold_duration', 'snr',
                             'spatial_config', 'migrate_trial', 'response_time', 'trial_duration']].copy()
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
                     'response_time', 'trial_duration']].copy()
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
        series2.plot(kind=kind)
        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0, 1])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'{YVARIABLE} by {XVARIABLE} for session {self.dataframe["day"][0]}',
                  fontsize = 8)
        plt.tight_layout()
        plt.show()
        print(plot_frame.groupby(XVARIABLE)[YVARIABLE].count())

        YVARIABLE = 'correct'
        Non_nose_poke = plot_frame.loc[(plot_frame['early_nosepoke'] == False)]
        series3 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()
        series3.plot(kind=kind)
        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0, 1])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'Percent correct for session {self.dataframe["day"][0]} without early_nps',
                  fontsize = 8)
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
        plt.title(f'{YVARIABLE} by {XVARIABLE} for session {self.dataframe["day"][0]}',
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
        plt.title(f'Percent correct for session {self.dataframe["day"][0]} without early_nps',
                  fontsize = 8)
        plt.legend()
        ticks = np.arange(len(day_list))
        plt.xticks(ticks=ticks, labels=day_list, rotation=90)
        plt.tight_layout()
        plt.show()
        print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())

if __name__ == "__main__":

#     #runclass NFB = 2afc, VOW = vowel
    my_test = behav('LMD', 'NFB', days='all', migrate_only=True)
    my_test.variables()
    XVARIABLE = ['snr']
    day_list = ['LMD063Ta', 'LMD064Ta', 'LMD065Ta', 'LMD066Ta']
    # my_test.sample_plots(XVARIABLE, day_list, kind='line')
    my_test.perform_over_time(XVARIABLE, day_list)
