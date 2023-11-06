import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nems0 import db

# plt.ion()
#
# font_size = 8
# params = {'legend.fontsize': font_size - 2,
#           'figure.figsize': (8, 6),
#           'axes.labelsize': font_size,
#           'axes.titlesize': font_size,
#           'axes.spines.right': False,
#           'axes.spines.top': False,
#           'xtick.labelsize': font_size,
#           'ytick.labelsize': font_size,
#           'pdf.fonttype': 42,
#           'ps.fonttype': 42}
#
# plt.rcParams.update(params)


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
        sql = f"SELECT * FROM gDataRaw WHERE runclass = '{self.task}' AND cellid like '{self.animal}%' AND bad=0 order by id"

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
                        d['rt'] = d['response_ts']-d['trial_start']
                        d['trial_duration'] = d['response_start'] - d['np_start']
                        d['spatial_config'] = 'same'
                        d.loc[d['bg_channel'] == d['fg_channel'], 'spatial_config'] = 'diff'
                        d.loc[d['bg_channel'] == -1, 'spatial_config'] = 'dichotic'

                        interesting_data = d[
                            ['trial_number', 'response', 'correct', 'response_time', 'hold_duration', 'snr',
                             'rt','np_start','response_start','spatial_config', 'migrate_trial', 'trial_duration', 'fg_i', 'bg_i', 'trial_is_repeat']].copy()
                        interesting_data['parmfile'] = r['parmfile']
                        interesting_data['session_id'] = session_id
                        interesting_data['day'] = r['cellid']
                    else:
                        valid = False
                else:
                    if 'snr' in d.columns:
                        if 'migrate_trial' in d.columns:
                            valid = False
                        d['rt'] = d['response_ts']-d['trial_start']
                        d['trial_duration'] = d['response_start'] - d['np_start']
                        d['spatial_config'] = 'same'
                        d.loc[d['bg_channel'] == d['fg_channel'], 'spatial_config'] = 'diff'
                        d.loc[d['bg_channel'] == -1, 'spatial_config'] = 'dichotic'

                        interesting_data = d[
                            ['trial_number', 'response', 'correct', 'response_time', 'hold_duration', 'snr',
                             'rt','np_start','response_start','spatial_config', 'response_time', 'trial_duration', 'fg_i', 'bg_i']].copy()
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
                     'trial_duration', 'fg_i', 'bg_i', 'trial_is_repeat']].copy()
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
        interesting_data_all['early_nosepoke'] = (interesting_data_all['response'] == 'early_np').astype(bool)


        #Finds and labels remind trials
        interesting_data_all['prev_is_wrong'] = False
        interesting_data_all.loc[interesting_data_all.index>0,'prev_is_wrong'] = (
                (interesting_data_all.iloc[:-1]['correct'] == False) &
                (interesting_data_all.iloc[:-1]['response'] != 'early_np')).values
        interesting_data_all['remind_trial'] = interesting_data_all['prev_is_wrong'] & interesting_data_all['trial_is_repeat']

        # interesting_data_all['difference']['bg_i'] = (interesting_data_all.bg_i != interesting_data_all.bg_i.shift()).astype(int)
        # interesting_data_all['difference']['fg_i'] = (
        #             interesting_data_all.fg_i != interesting_data_all.fg_i.shift()).astype(int)
        # interesting_data_all['difference']['snr'] = (interesting_data_all.snr != interesting_data_all.snr.shift()).astype(int)
        # interesting_data_all['difference']['spatial_config'] = (interesting_data_all.spatial_config != interesting_data_all.spatial_config.shift()).astype(int)


        # if self.task == 'NFB':
        #     interesting_data_all['difference'][''] ==
        # elif self.task == 'VOW':
        #     interesting_data_all['remind_trial'] ==


        return interesting_data_all, all_data_extensive

    #Prints all dataframe columns, and all unique days
    def variables(self):
        print(self.dataframe.columns)
        print(self.dataframe['day'].unique())

    def sample_plots(self, XVARIABLE, day_list=[], kind='bar', YVARIABLE='correct', include_remind = False, only_0_db=False,
                     figsize = None):

        if len(day_list) == 0:
            plot_frame = self.dataframe.copy()
        else:
            plot_frame = self.dataframe[self.dataframe['day'].isin(day_list)].copy()

        # plot_frame = plot_frame.iloc[:100]

        if YVARIABLE=='correct':
            Non_nose_poke = plot_frame.loc[(plot_frame['early_nosepoke'] == False)]
            # Non_nose_poke = plot_frame
        else:
            Non_nose_poke = plot_frame

        if not include_remind:
            Non_nose_poke = Non_nose_poke[Non_nose_poke['remind_trial'] == False]

        # Non_nose_poke = Non_nose_poke.iloc[-75:]
        Non_nose_poke = Non_nose_poke[Non_nose_poke['response'] != 'early_np']

        if only_0_db:
            Non_nose_poke = Non_nose_poke[Non_nose_poke['snr'] == 0]

        series2 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()

        f,ax=plt.subplots(figsize = figsize)
        series2.plot(kind=kind, ax=ax)
        # plt.axhline(y=0.5, linestyle='--')
        # plt.ylim([0, 1])
        # plt.xlabel(XVARIABLE)
        # plt.ylabel(YVARIABLE)
        # plt.title(f'{YVARIABLE} by {XVARIABLE} for session {plot_frame["day"].unique()[0]}',
        #           fontsize = 8)
        # counts = plot_frame.groupby(XVARIABLE)[YVARIABLE].count().values
        # for i, p in enumerate(ax.patches):
        #     ax.annotate(str(counts[i]), (p.get_x() * 1.005, p.get_height() * 0.005))
        #
        # plt.tight_layout()
        # plt.show()
        # print(plot_frame.groupby(XVARIABLE)[YVARIABLE].count())
        #
        # YVARIABLE = 'correct'
        #series3 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()
        #ax2 = series3.plot(kind=kind)
        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0.5, 1.5])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'{YVARIABLE} for session {Non_nose_poke["day"].unique()[0]} without early_nps',
                  fontsize = 8)
        counts2 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count().values
        for i, p in enumerate(counts2):
            ax.text(i, 0, str(counts2[i]), ha='center', va='bottom')
        plt.tight_layout()
        # plt.show()
        print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())

        # import seaborn as sns
        #
        #
        # Non_nose_poke = Non_nose_poke[Non_nose_poke['response'] != 'early_np']
        # data = Non_nose_poke.groupby('day')[['migrate_trial', 'response_time']].mean().reset_index()
        #
        # sns.stripplot(x='migrate_trial', y=f'{YVARIABLE}', data=Non_nose_poke, hue='day', jitter=True, palette='Set2', size=6)
        # plt.xlabel('Trial type')
        # plt.ylabel('Response time (s)')
        # # plt.title(f'Response time for {parmfile}')
        return f

    def perform_over_time(self, XVARIABLE, YVARIABLE='correct', day_list=[], kind='bar', only_0_db=False):

        if len(day_list) == 0:
            plot_frame = self.dataframe.copy()
        else:
            plot_frame = self.dataframe[self.dataframe['day'].isin(day_list)].copy()


        # YVARIABLE = 'early_nosepoke'
        # plt.figure()
        # for i, v in enumerate(XVARIABLE):
        #     for j, w in enumerate(plot_frame[v].unique()):
        #         series = plot_frame[plot_frame[v] == w].groupby('day')[YVARIABLE].mean()
        #         series.plot(kind='line', label=f'{w}')
        # plt.axhline(y=0.5, linestyle='--')
        # plt.ylim([0, 1])
        # plt.xlabel(XVARIABLE)
        # plt.ylabel(YVARIABLE)
        # plt.title(f'{YVARIABLE} by {XVARIABLE} for session {plot_frame["day"].unique()[0]}',
        #           fontsize = 8)
        # plt.legend()
        # ticks = np.arange(len(day_list))
        # plt.xticks(ticks=ticks, labels=day_list, rotation=90)
        # plt.tight_layout()
        # plt.show()
        # print(plot_frame.groupby(XVARIABLE)[YVARIABLE].count())

        f,ax = plt.subplots()
        # YVARIABLE = 'correct'
        Non_nose_poke = plot_frame.loc[(plot_frame['early_nosepoke'] == False)]
        Non_nose_poke = Non_nose_poke[Non_nose_poke['remind_trial'] == False]
        if only_0_db:
            Non_nose_poke = Non_nose_poke[Non_nose_poke['snr'] == 0]
        for i, v in enumerate(XVARIABLE):
            for j, w in enumerate(Non_nose_poke[v].unique()):
                series = Non_nose_poke[Non_nose_poke[v] == w].groupby('day')[YVARIABLE].mean()
                series.plot(kind='line', label=f'{w}', ax=ax)
                print(series)
        ax.axhline(y=0.5, linestyle='--')
        ax.set_ylim([0, 1])
        ax.set_xlabel(XVARIABLE)
        ax.set_ylabel(YVARIABLE)
        ax.set_title(f'{YVARIABLE} for session {Non_nose_poke["day"].unique()[0]} without early_nps',
                  fontsize = 8)
        plt.legend()
        ticks = np.arange(len(day_list))
        ax.set_xticks(ticks=ticks, labels=day_list, rotation=90)
        f.tight_layout()
        #plt.show()
        print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())
        return f

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

    def rolling_avg(self, XVARIABLE, day_list=[], kind='bar', YVARIABLE='correct', include_remind=False):
        if len(day_list) == 0:
            plot_frame = self.dataframe.copy()
        else:
            plot_frame = self.dataframe[self.dataframe['day'].isin(day_list)].copy()


        if YVARIABLE == 'correct':
            Non_nose_poke = plot_frame.loc[(plot_frame['early_nosepoke'] == False)]
            # Non_nose_poke = plot_frame
        else:
            Non_nose_poke = plot_frame

        if not include_remind:
            Non_nose_poke = Non_nose_poke[Non_nose_poke['remind_trial'] == False]

        Non_nose_poke['rolling_avg'] = Non_nose_poke['correct'].rolling(20, min_periods=1).mean()
        series2 = Non_nose_poke.groupby(XVARIABLE)['rolling_avg']

        f, ax = plt.subplots()
        series2.plot(kind='line', ax=ax)    # plt.axhline(y=0.5, linestyle='--')



        plt.axhline(y=0.5, linestyle='--')
        plt.ylim([0, 1])
        plt.xlabel(XVARIABLE)
        plt.ylabel(YVARIABLE)
        plt.title(f'Rolling average of {YVARIABLE} for session {Non_nose_poke["day"].unique()[0]} without early_nps',
                  fontsize=8)
        counts2 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count().values
        for i, p in enumerate(counts2):
            ax.text(i, 0, str(counts2[i]), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())

def lemon_space_snr_bar():
    example = behav('LMD', 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
    #example.remove_remind_trials()

    # example.variables()
    XVARIABLE = ['spatial_config','snr']

    start_site = 'LMD066Ta'
    d=example.dataframe
    d = d.loc[(d['day']>=start_site) & (d['snr'].isin([0,-10,-20,-30]))]
    example.dataframe = d
    days=d.loc[d['day']>=start_site,'day'].unique().tolist()
    f = example.sample_plots(XVARIABLE, day_list=days)

    # for day in days:
    #     example.sample_plots(XVARIABLE, day_list=[day])

    # d=example.dataframe
    # d=d.loc[(d['early_nosepoke']==False) ]
    # dday=d.groupby(['day','snr'])[['correct']].mean().reset_index()
    # dday = dday.pivot(columns=['snr'], index=['day'], values=['correct'])
    #
    # plt.figure()
    # dday.plot()
    return f

def slippy_space_snr_bar():
    example = behav('SLJ', 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
    #example.remove_remind_trials()

    # example.variables()
    XVARIABLE = ['spatial_config','snr']

    start_site = 'SLJ057Ta'
    d=example.dataframe
    d = d.loc[(d['day']>=start_site) & (d['snr'].isin([0,-5,-10,-15,-20,-30]))]
    example.dataframe = d
    days=d.loc[d['day']>=start_site,'day'].unique().tolist()
    f=example.sample_plots(XVARIABLE, day_list=days)

    #for day in days:
    #  example.sample_plots(XVARIABLE, day_list=[day])

    #d=example.dataframe
    #d=d.loc[(d['early_nosepoke']==False) ]
    #dday=d.groupby(['day','snr'])[['correct']].mean().reset_index()
    #dday = dday.pivot(columns=['snr'], index=['day'], values=['correct'])
    #
    # plt.figure()
    # dday.plot()
    return f

def lemon_space_snr_subplots():
    plot_data = behav('LMD', 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
    df = plot_data.dataframe.copy()

    t = [-30, -20, -10, 0]

    df = df.loc[~df['snr'].isin([-15, -5])]
    snrs = df.snr.unique().tolist()
    snrs.sort()
    snr_mask = {ss: kk for kk, ss in enumerate(snrs)}
    df['snr_mask'] = [snr_mask[dd] for dd in df['snr']]

    # day_list = ['LMD068Ta', 'LMD069Ta', 'LMD070Ta', 'LMD071Ta', 'LMD072Ta']
    # df = df[df['day'].isin(day_list)]

    df = df[df['response'] != 'early_np']
    df = df[df['remind_trial'] == False]


    df['valid_day'] = True
    for cnt, day in enumerate(df['day'].unique()):
        valid = True
        tempdf = df[df['day'] == day].copy()
        for snr in tempdf['snr'].unique():
            if len(tempdf[tempdf['snr'] == snr]) < 15:
                valid = False
        if valid == False:
            df.loc[df['day'] == day, 'valid_day'] = False

    df_good_days = df[df['valid_day'] == True]

    from matplotlib import cm
    greys = cm.get_cmap('YlOrRd', 12)
    cols = greys(np.linspace(.3, .9, len(df['day'].unique())))

    types = ['same', 'dichotic', 'diff']
    type_dict = {'same': 'ipsi', 'dichotic': 'diotic', 'diff': 'contra'}
    # perfect wonderful first working figure
    fig, axes = plt.subplots(1, 4, figsize=(6, 2.5), sharey=True)
    lastax = axes[-1]
    
    colors = ['black', 'blue', 'red']
    i = 0
    for cc, (ax, tt) in enumerate(zip(axes[:3], types)):
        if i != 3:
            dff = df_good_days.loc[df_good_days['spatial_config'] == tt]
            for cnt, day in enumerate(dff['day'].unique()):
                series = dff[dff['day'] == day].groupby('snr_mask')['correct'].mean()
                if len(series) > 2:
                    # series = series.reset_index()
                    # series_mask = np.isfinite(series.astype(np.double))
                    ax.plot(series, label=f'{day[3:6]}', color=cols[cnt], marker='.', ls='-', alpha=0.25)
                    # ax.ylim([0, 1])
                    ax.set_title(f'{type_dict[tt]}', fontweight='bold', fontsize=12)
                    if tt == 'same':
                        ax.set_ylabel('Percent correct', fontweight='bold', fontsize=10)
                    ax.set_xlabel('snr', fontweight='bold', fontsize=10)
                    ax.set_xticks(range(len(snrs)))
                    ax.set_xticklabels([int(dd) for dd in snrs])
                    # if cc == 0:
                    #     ax.legend()
                    #     ax.set_ylabel('Percent correct', fontweight='bold', fontsize=10)

            mean_series = dff.groupby('snr_mask')['correct'].mean()
            dff_days = dff.groupby(['day', 'snr_mask'])['correct'].mean()
            sem_series = dff_days.groupby('snr_mask').sem()
            ax.plot(mean_series.index, mean_series.values, color='black', marker='o', ls='-')
            ax.errorbar(x=mean_series.index, y=mean_series.values, yerr=sem_series, label=type_dict[types[i]], color='black', marker='o', ls='-')
            ax.axhline(0.5,ls='--',lw=1)

            #
            lastax.plot(mean_series.index, mean_series.values, color=colors[i], ls='-')
            lastax.errorbar(x=mean_series.index, y=mean_series.values, yerr=sem_series, label=type_dict[types[i]], color=colors[i],
                         ls='-')


        i += 1
    # fig.suptitle('Performance across spatial condition (all), before migrate')
    lastax.axhline(0.5,ls='--',lw=1)
    lastax.set_xticks(range(len(snrs)))
    lastax.set_xticklabels([int(dd) for dd in snrs])
    lastax.set_xlabel('FG SNR')
    lastax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.tight_layout()

    #plt.savefig('/auto/users/sticknej/code/correlation/lemon_101323b.pdf')
    plt.show()
    return fig

def slippy_space_snr_subplots():
    plot_data = behav('SLJ', 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
    df = plot_data.dataframe.copy()

    t = [-30, -20, -10, -5, 0, 100]

    df = df[df['snr'] != 5]
    df = df[df['snr'] != 15]

    snrs = df.snr.unique().tolist()
    snrs.sort()
    snr_mask = {ss: kk for kk, ss in enumerate(snrs)}
    df['snr_mask'] = [snr_mask[dd] for dd in df['snr']]

    # day_list = ['LMD068Ta', 'LMD069Ta', 'LMD070Ta', 'LMD071Ta', 'LMD072Ta']
    # df = df[df['day'].isin(day_list)]

    testdf = df[df['response'] != 'early_np']
    testdf = testdf[testdf['remind_trial'] == False]
    testdf = testdf.loc[(testdf.spatial_config=='diff') & (testdf.snr==0)].groupby('day').mean()['correct']
    good_days=testdf.index[testdf>=0.80].to_list()
    print(good_days)
    df = df[df['day'].isin(good_days)]
    #df = df[~df['day'].str.startswith('SLJ05')]
    #df = df[~df['day'].isin(['SLJ064Ta', 'SLJ066Ta', 'SLJ068Ta'])]
    df = df[df['response'] != 'early_np']
    df = df[df['remind_trial'] == False]


    df['valid_day'] = True
    for cnt, day in enumerate(df['day'].unique()):
        valid = True
        tempdf = df[df['day'] == day].copy()
        for snr in tempdf['snr'].unique():
            if len(tempdf[tempdf['snr'] == snr]) < 20:
                valid = False
        if valid == False:
            df.loc[df['day'] == day, 'valid_day'] = False

    df_good_days = df[df['valid_day'] == True]

    from matplotlib import cm
    greys = cm.get_cmap('YlOrRd', 12)
    cols = greys(np.linspace(.3, .9, len(df['day'].unique())))

    types = ['same', 'dichotic', 'diff']
    type_dict = {'same': 'ipsi', 'dichotic': 'diotic', 'diff': 'contra'}
    # perfect wonderful first working figure
    fig, axes = plt.subplots(1, 4, figsize=(6, 2.5), sharey=True)

    colors = ['black', 'blue', 'red']
    i =0
    for cc, (ax, tt) in enumerate(zip(axes, types)):
        if i != 3:
            dff = df_good_days.loc[df_good_days['spatial_config'] == tt]
            for cnt, day in enumerate(dff['day'].unique()):
                series = dff[dff['day'] == day].groupby('snr_mask')['correct'].mean()
                if len(series) > 2:
                    # series = series.reset_index()
                    # series_mask = np.isfinite(series.astype(np.double))
                    ax.plot(series, label=f'{day[3:6]}', color=cols[cnt], marker='.', ls='-', alpha=0.25)
                    ax.set_title(f'{type_dict[tt]}', fontweight='bold', fontsize=12)
                    if tt == 'same':
                        ax.set_ylabel('Percent correct', fontweight='bold', fontsize=10)
                    ax.set_xlabel('snr', fontweight='bold', fontsize=10)
                    ax.set_xticks(range(len(snrs)))
                    ax.set_xticklabels([int(dd) for dd in snrs])
                    # if cc == 0:
                    #     ax.legend()
                    #     ax.set_ylabel('Percent correct', fontweight='bold', fontsize=10)

            # mean_series = dff.groupby('day').mean().groupby('snr_mask')['correct'].mean()
            dff_days = dff.groupby(['day', 'snr_mask'])['correct'].mean()
            mean_series = dff_days.groupby('snr_mask').mean()
            sem_series = dff_days.groupby('snr_mask').sem()
            ax.plot(mean_series.index, mean_series.values, color='black', marker='o', ls='-')
            ax.errorbar(x=mean_series.index, y=mean_series.values, yerr=sem_series, label=type_dict[types[i]], 
                        color='black', marker='o', ls='-')
            ax.axhline(0.5,ls='--',lw=1)
            
            #
            lastax = axes[3]
            lastax.plot(mean_series.index, mean_series.values, color=colors[i], ls='-')
            lastax.errorbar(x=mean_series.index, y=mean_series.values, yerr=sem_series, label=type_dict[types[i]], color=colors[i],
                            ls='-')
            
        i += 1
    # fig.suptitle('Performance across spatial condition (all), before migrate')
    lastax.axhline(0.5,ls='--',lw=1)
    lastax.set_xticks(range(len(snrs)))
    lastax.set_xticklabels([int(dd) for dd in snrs])
    lastax.set_xlabel('FG SNR')
    lastax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.tight_layout()

    #plt.savefig('/auto/users/sticknej/code/correlation/slippy_101323b.pdf')
    plt.show()
    return fig



def lemon_dlc_trace_plot(parmfile='LemonDisco_2023_10_05_NFB_2'):

    rawdata = db.pd_query(f"SELECT * FROM gDataRaw WHERE parmfile like '{parmfile}'")
    siteid = rawdata.iloc[0]['cellid']
    #siteid = 'LMD077Ta'
    behav_lmd = behav('LMD', 'NFB', days='all', migrate_only=True, non_migrate_blocks=False)

    df = behav_lmd.dataframe.copy()
    df = df[df['day'] == siteid]

    from nems_lbhb import baphy_experiment, baphy_io

    psifile = [f'/auto/data/daq/LemonDisco/training2023/{parmfile}', ]
    ex = baphy_experiment.BAPHYExperiment(parmfile=psifile)
    rec = ex.get_recording(resp=False, stim=False, dlc=True, recache=False)
    fs = rec['dlc'].fs
    all_trials = rec['dlc'].extract_epoch("TRIAL")
    if all_trials.shape[-1]/fs > 10:
        all_trials = all_trials[:,:,:int(10*fs)]
    mask_migrate = (df['migrate_trial'] == 1) & (df['correct'] == True) & (df['response'] == 'spout_1')
    mask_nonmigrate = (df['migrate_trial'] == 0) & (df['correct'] == True) & (df['response'] == 'spout_1')
    mask_migrate2 = (df['migrate_trial'] == 1) & (df['correct'] == True) & (df['response'] == 'spout_2')
    mask_nonmigrate2 = (df['migrate_trial'] == 0) & (df['correct'] == True) & (df['response'] == 'spout_2')
    mask_migrate_err = (df['migrate_trial'] == 1) & (df['correct'] == False) & (df['response'] == 'spout_2')
    mask_nonmigrate_err = (df['migrate_trial'] == 0) & (df['correct'] == False) & (df['response'] == 'spout_2')

    font_size = 8
    params = {'legend.fontsize': font_size - 2,
              'figure.figsize': (8, 6),
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'axes.spines.right': False,
              'axes.spines.top': False,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'pdf.fonttype': 42,
              'ps.fonttype': 42}
    plt.rcParams.update(params)


    f=plt.figure(figsize=(3, 2))

    T = 300
    tt = np.arange(T) / rec['dlc'].fs
    meanstartx = np.mean(all_trials[:, 0, 0])
    scaleby=1000

    h2 = plt.plot(tt, (all_trials[mask_nonmigrate, 0, :T].T-meanstartx)/scaleby, color='lightblue', lw=0.5)
    h1 = plt.plot(tt, (all_trials[mask_migrate, 0, :T].T-meanstartx)/scaleby, color='green', lw=0.5)
    h2 = plt.plot(tt, (meanstartx - all_trials[mask_nonmigrate2, 0, :T].T)/scaleby, color='lightblue', lw=0.5)
    h1 = plt.plot(tt, (meanstartx - all_trials[mask_migrate2, 0, :T].T)/scaleby, color='green', lw=0.5)

    #h3 = plt.plot(tt, all_trials[mask_migrate_err, 0, :T].T, color='lightgreen', lw=0.5)
    #h4 = plt.plot(tt[:100], all_trials[mask_nonmigrate_err, 0, :100].T, color='lightblue', lw=0.5)
    for i,r in df.reset_index().iterrows():
        my_trial_len = r['rt']
        if np.isfinite(my_trial_len):
            stopbin = int(my_trial_len*fs)
            colors = {1: 'darkgreen', 0: 'blue'}

            if r['correct']==False:
                pass
            else:
                if r['response']=='spout_1':
                    y = (all_trials[i,0,stopbin]-meanstartx)/scaleby
                else:
                    y = (meanstartx - all_trials[i,0,stopbin])/scaleby
                plt.plot(tt[stopbin],y,'.', color=colors[r['migrate_trial']])


    plt.legend((h1[0], h2[0]), ('Dynamic', 'Static', 'migrate+err', 'nonmigrate+err'),
               frameon=False)
    plt.xlabel('Time (sec)')
    plt.ylabel('Horizontal position (L -> R)')
    #plt.title('Target at right')
    plt.tight_layout()
    f.savefig(f'/home/svd/Documents/onedrive/proposals/r01_bcp/figures/raw/{siteid}_dlc_dynamic.pdf')

def slippy_dlc_trace_plot(parmfile = 'SlipperyJack_2023_10_03_NFB_1'):
    from nems0 import db
    rawdata = db.pd_query(f"SELECT * FROM gDataRaw WHERE parmfile like '{parmfile}'")
    siteid = rawdata.iloc[0]['cellid']
    #siteid = 'SLJ064Ta'
    behav_lmd = behav('SLJ', 'NFB', days='all', migrate_only=True, non_migrate_blocks='')

    df = behav_lmd.dataframe.copy()
    df = df[df['day'] == siteid]

    from nems_lbhb import baphy_experiment, baphy_io

    psifile = [f'/auto/data/daq/SlipperyJack/training2023/{parmfile}', ]
    ex = baphy_experiment.BAPHYExperiment(parmfile=psifile)
    rec = ex.get_recording(resp=False, stim=False, dlc=True, recache=False)
    fs=rec['dlc'].fs
    epochs=rec['dlc'].epochs.copy()
    trial_epochs=rec['dlc'].epochs.loc[rec['dlc'].epochs.name == 'TRIAL'].reset_index()
    all_trials = rec['dlc'].extract_epoch("TRIAL")
    for i,r in df.reset_index().iterrows():
        trial_start_time = trial_epochs.loc[i,'start']
        try:
            lick_event = epochs.loc[(epochs.start>trial_start_time) & (epochs.name.str.startswith('LICK')),'start'].values[0]
        except:
            lick_event = np.nan
        my_trial_len = lick_event-trial_start_time
        my_trial_len = r['rt']
        #my_trial_len = r['trial_duration'] # r['response_start'] - trial_epochs.loc[i,'start']
        if np.isfinite(my_trial_len):
            stopbin = int(my_trial_len*fs)
            all_trials[i, :, stopbin:] = np.nan

    # mask_migrate = (df['migrate_trial'] == 1) & (df['correct'] == True) & (df['response'] == 'spout_1')
    # mask_nonmigrate = (df['migrate_trial'] == 0) & (df['correct'] == True) & (df['response'] == 'spout_1')
    # mask_migrate2 = (df['migrate_trial'] == 1) & (df['correct'] == True) & (df['response'] == 'spout_2')
    # mask_nonmigrate2 = (df['migrate_trial'] == 0) & (df['correct'] == True) & (df['response'] == 'spout_2')
    # mask_migrate_err = (df['migrate_trial'] == 1) & (df['correct'] == False) & (df['response'] == 'spout_2')
    # mask_nonmigrate_err = (df['migrate_trial'] == 0) & (df['correct'] == False) & (df['response'] == 'spout_2')

    #mask_spout1 = (df['correct'] == True) & (df['remind_trial'] == 0) & (df['response'] == "spout_1")
    #mask_spout2 = (df['correct'] == True) & (df['remind_trial'] == 0) & (df['response'] == "spout_2")
    #mask_spout1 = (df['correct'] == True) & (df['remind_trial'] == 0) & (df['response'] == "spout_1")
    #mask_spout2 = (df['correct'] == True) & (df['remind_trial'] == 0) & (df['response'] == "spout_2")
    mask_spout1 = (df['correct'] == False) & (df['remind_trial'] == 0) & (df['response'] == "spout_1")
    mask_spout2 = (df['correct'] == False) & (df['remind_trial'] == 0) & (df['response'] == "spout_2")
    mask_correct = (df['correct'] == True) & (df['remind_trial'] == 0)
    mask_incorrect = (df['correct'] == False) & (df['remind_trial'] == 0)

    # plot params
    font_size = 8
    params = {'legend.fontsize': font_size - 2,
              'figure.figsize': (8, 6),
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'axes.spines.right': False,
              'axes.spines.top': False,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'pdf.fonttype': 42,
              'ps.fonttype': 42}
    plt.rcParams.update(params)


    plt.figure(figsize=(10, 5))
    T = 200
    tt = np.arange(T) / rec['dlc'].fs
    h2 = plt.plot(tt, all_trials[mask_spout1, 0, :T].T, color='lightblue', lw=0.5)
    h1 = plt.plot(tt, all_trials[mask_spout2, 0, :T].T, color='green', lw=0.5)
    #h1 = plt.plot(tt, all_trials[mask_correct, 0, :T].T-all_trials[mask_correct, 2, :T].T, color='green', lw=0.5)
    #h3 = plt.plot(tt, all_trials[mask_correct, 2, :T].T, color='yellow', lw=0.5)
    meanstartx = np.mean(all_trials[:, 0, 0])
    # h2 = plt.plot(tt, meanstartx * 2 - all_trials[mask_correct, 0, :T].T, color='lightblue', lw=0.5)
    # h1 = plt.plot(tt, meanstartx * 2 - all_trials[mask_incorrect, 0, :T].T, color='green', lw=0.5)

    # h3 = plt.plot(tt, all_trials[mask_incorrect, 0, :T].T, color='lightgreen', lw=0.5)
    # h4 = plt.plot(all_trials[mask_nonmigrate_err, 0, :100].T, color='lightblue', lw=0.5)
    plt.xlim([0,1.5])
    #plt.ylim([0,150])

    #plt.legend((h1[0], h2[0]), ('correct', 'incorrect'))
    plt.legend((h1[0], h2[0]), (f"spout2 n={mask_spout2.sum()}", f"spout1 n={mask_spout1.sum()}"))
    plt.xlabel('Time (sec)')
    plt.ylabel('Horizontal position (left -> right)')
    # plt.title(f'Target at right, {siteid}')
    plt.title(f'{siteid}')

    plt.show()



def lemon_dlc_rt_comparison():
    import seaborn as sns

    parmfile = 'LemonDisco_2023_10_09_NFB_3'
    from nems0 import db
    rawdata = db.pd_query(f"SELECT * FROM gDataRaw WHERE parmfile like '{parmfile}'")
    siteid = rawdata.iloc[0]['cellid']
    behav_lmd = behav('LMD', 'NFB', days='all', migrate_only=True, non_migrate_blocks=False)

    df = behav_lmd.dataframe.copy()
    df = df[df['day'] == siteid]

    from nems_lbhb import baphy_experiment, baphy_io

    psifile = [f'/auto/data/daq/LemonDisco/training2023/{parmfile}', ]
    ex = baphy_experiment.BAPHYExperiment(parmfile=psifile)
    rec = ex.get_recording(resp=False, stim=False, dlc=True, recache=False)
    rec['dlc'].fs
    all_trials = rec['dlc'].extract_epoch("TRIAL")

    switch_trials = np.zeros((all_trials.shape[0]))
    for i in range(all_trials.shape[0]):
        over_410 = False
        under_210 = False

        if all_trials[i,0,:300].max() > 410:
            over_410 = True

        if all_trials[i,0,:300].min() < 210:
            under_210 = True

        switch_trials[i] = (over_410 and under_210)

    df['switch_trial'] = pd.Series(switch_trials.tolist()).values

    df_correct = df[df['correct'] == True].copy()
    df_correct['migrate_trial_type'] = 'Non-migrate'
    df_correct.loc[(df_correct['migrate_trial'] == True) & (df_correct['switch_trial'] == True), 'migrate_trial_type'] = 'Migrate indirect'
    df_correct.loc[(df_correct['migrate_trial'] == True) & (df_correct['switch_trial'] == False), 'migrate_trial_type'] = 'Migrate direct'

    #YVARIABLE = 'response_time'
    YVARIABLE = 'rt'
    sns.stripplot(x='migrate_trial_type', y=YVARIABLE, data=df_correct, jitter=True, palette='Set2', size=6)
    plt.xlabel('Trial type')
    plt.ylabel('Response time (s)')
    plt.title(f'Response time for {parmfile}')

    means = df_correct.groupby('migrate_trial_type')['response_time'].mean()

    ax = plt.gca()
    for day, mean in means.items():
        ax.text(day, mean, f"Mean: {mean:.2f}", ha='center', va='bottom')

    plt.show()



    blah = 'blah'

def day_range_comparison(XVARIABLE, day_list=[], kind='bar', YVARIABLE='correct', include_remind = False):

    example = behav('SLJ', 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
    # # #runclass NFB = 2afc, VOW = vowel
    # #
    # example.variables()
    XVARIABLE = ['spatial_config', 'snr']
    YVARIABLE = 'correct'
    # # #
    day_list1 = ['SLJ063Ta', 'SLJ065Ta', 'SLJ067Ta']
    day_list2 = ['SLJ064Ta', 'SLJ066Ta', 'SLJ068Ta']

    plot_frame_saline = example.dataframe[example.dataframe['day'].isin(day_list1)].copy()
    plot_frame_drug = example.dataframe[example.dataframe['day'].isin(day_list2)].copy()

    # plot_frame = plot_frame.iloc[:100]


    Non_nose_poke_saline = plot_frame_saline.loc[(plot_frame_saline['early_nosepoke'] == False)]
    plot_frame_drug= plot_frame_drug.loc[(plot_frame_drug['early_nosepoke'] == False)]


    if not include_remind:
        Non_nose_poke_saline = Non_nose_poke_saline[Non_nose_poke_saline['remind_trial'] == False]
        plot_frame_drug = plot_frame_drug[plot_frame_drug['remind_trial'] == False]

    # Non_nose_poke = Non_nose_poke.iloc[-75:]
    series2 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()

    f,ax=plt.subplots()
    series2.plot(kind=kind, ax=ax)
    # plt.axhline(y=0.5, linestyle='--')
    # plt.ylim([0, 1])
    # plt.xlabel(XVARIABLE)
    # plt.ylabel(YVARIABLE)
    # plt.title(f'{YVARIABLE} by {XVARIABLE} for session {plot_frame["day"].unique()[0]}',
    #           fontsize = 8)
    # counts = plot_frame.groupby(XVARIABLE)[YVARIABLE].count().values
    # for i, p in enumerate(ax.patches):
    #     ax.annotate(str(counts[i]), (p.get_x() * 1.005, p.get_height() * 0.005))
    #
    # plt.tight_layout()
    # plt.show()
    # print(plot_frame.groupby(XVARIABLE)[YVARIABLE].count())
    #
    # YVARIABLE = 'correct'
    #series3 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()
    #ax2 = series3.plot(kind=kind)
    plt.axhline(y=0.5, linestyle='--')
    plt.ylim([0, 1])
    plt.xlabel(XVARIABLE)
    plt.ylabel(YVARIABLE)
    plt.title(f'{YVARIABLE} for session {Non_nose_poke["day"].unique()[0]} without early_nps',
              fontsize = 8)
    counts2 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count().values
    for i, p in enumerate(counts2):
        ax.text(i, 0, str(counts2[i]), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())

def slippy_dcz_effect():
    example = behav('SLJ', 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
    XVARIABLE = ['day']
    YVARIABLE = 'correct'
    day_list3 = ['SLJ063Ta', 'SLJ064Ta', 'SLJ065Ta', 'SLJ066Ta', 'SLJ067Ta',  'SLJ068Ta', 'SLJ069Ta']
    example.sample_plots(XVARIABLE, day_list3, YVARIABLE=YVARIABLE, only_0_db=True, figsize=[2,3])

    font_size = 8
    params = {'legend.fontsize': font_size - 2,
              'figure.figsize': (8, 6),
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'axes.spines.right': False,
              'axes.spines.top': False,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'pdf.fonttype': 42,
              'ps.fonttype': 42}
    plt.rcParams.update(params)
    plt.savefig('/auto/users/sticknej/code/correlation/slippy_101323c.pdf')
    plt.show()


def lemon_migrate_rt():
    example = behav('LMD', 'NFB', days='all', migrate_only=True, non_migrate_blocks=False)


    df = example.dataframe.copy()
    day_list = ['LMD074Ta', 'LMD075Ta', 'LMD076Ta', 'LMD077Ta', 'LMD078Ta',
                'LMD079Ta', 'LMD080Ta', 'LMD081Ta', 'LMD082Ta', 'LMD083Ta']
    df = df[df['day'].isin(day_list)]

    df = df[df['response'] != 'early_np']
    df = df[df['correct'] == True]

    dff = df.groupby(['day', 'migrate_trial'])['rt'].mean().reset_index()
    migrate_mean = df.groupby

    import seaborn as sns

    fig, ax = plt.subplots()
    ax = sns.pointplot(x='migrate_trial', y='rt', data=dff, jitter=True, color='black', size=6)
    ax = sns.stripplot(x='migrate_trial', y='rt', data=dff, jitter=True, palette='Set2', size=6)


    df_mean = dff.groupby('migrate_trial')['rt'].mean()
    df_sem = dff.groupby('migrate_trial')['rt'].sem()



    font_size = 8
    params = {'legend.fontsize': font_size - 2,
              'figure.figsize': (8, 6),
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'axes.spines.right': False,
              'axes.spines.top': False,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'pdf.fonttype': 42,
              'ps.fonttype': 42}
    plt.rcParams.update(params)

    ax.set_xticklabels(['Stationary', 'Dynamic'])
    plt.xlabel('Trial type')
    plt.ylabel('Reaction time (s)')

    plt.savefig('/auto/users/sticknej/code/correlation/lemon_101323c.pdf')

    plt.show()




if __name__ == "__main__":
    print("running main")

    # lemon_space_snr_bar()
    # slippy_space_snr_bar()

    # lemon_space_snr_subplots()
    # slippy_space_snr_subplots()
    # slippy_dcz_effect()
    lemon_migrate_rt()

    # lemon_dlc_trace_plot()
    # lemon_dlc_rt_comparison()
    # slippy_dlc_trace_plot('SlipperyJack_2023_10_03_NFB_1')
    # slippy_dlc_trace_plot('SlipperyJack_2023_10_04_NFB_1')
    # slippy_dlc_trace_plot('SlipperyJack_2023_10_05_NFB_1')
    # slippy_dlc_trace_plot('SlipperyJack_2023_10_06_NFB_1')


    # example = behav('SLJ', 'NFB', days='all', migrate_only=True, non_migrate_blocks=True)
    # # # #runclass NFB = 2afc, VOW = vowel
    # # #
    # # example.variables()
    XVARIABLE = ['migrate_trial', 'day']
    YVARIABLE = 'correct'
    # # # # #
    # day_list1 = ['SLJ063Ta', 'SLJ065Ta', 'SLJ067Ta']
    # day_list2 = ['SLJ064Ta', 'SLJ066Ta', 'SLJ068Ta']
    day_list3 = ['SLJ063Ta', 'SLJ064Ta', 'SLJ065Ta', 'SLJ066Ta', 'SLJ067Ta',  'SLJ068Ta',
                 'SLJ069Ta', 'SLJ070Ta', 'SLJ071Ta', 'SLJ072Ta']
    # day_list = ['LMD074Ta', 'LMD075Ta', 'LMD076Ta', 'LMD077Ta', 'LMD078Ta',
    #             'LMD079Ta', 'LMD080Ta', 'LMD081Ta', 'LMD082Ta']
    # day_list = ['LMD082Ta', 'LMD083Ta']


    # example.rolling_avg(XVARIABLE, day_list=['LMD074Ta'])
    # example.rolling_avg(XVARIABLE, day_list=['LMD073Ta'])


    # example.sample_plots(XVARIABLE, day_list3, YVARIABLE=YVARIABLE)
    # example.sample_plots(XVARIABLE, day_list, YVARIABLE='hold_duration')
    # example.sample_plots(XVARIABLE, day_list, YVARIABLE='trial_duration')
    # example.sample_plots(XVARIABLE, day_list1, YVARIABLE=YVARIABLE, only_0_db=True)
    # example.sample_plots(XVARIABLE, day_list2, YVARIABLE=YVARIABLE, only_0_db=True)

    # example.sample_plots(XVARIABLE, day_list=['SLJ071Ta'], YVARIABLE='early_nosepoke')

    # example.perform_over_time(XVARIABLE, YVARIABLE, day_list3, only_0_db=True)
    # example.perform_over_time(XVARIABLE, YVARIABLE, day_list2)
    # example.plot_trial_duration(XVARIABLE, day_list)
    # example.plot_trial_duration(XVARIABLE, day_list=['SLJ071Ta'], YVARIABLE='early_np')


