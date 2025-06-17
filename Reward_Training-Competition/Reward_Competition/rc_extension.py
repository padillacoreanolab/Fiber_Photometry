import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from rtc_extension import RTC
from experiment_class import Experiment
from scipy.stats import pearsonr
from trial_class import Trial
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import itertools
from matplotlib.colors import LinearSegmentedColormap




class Reward_Competition(RTC):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)

    """*************************READING CSV AND STORING AS DF**************************"""
    def read_manual_scoring(self, csv_file_path):
        """
        Reads in and creates a dataframe to store wins and loses and tangles
        """
        df = pd.read_excel(csv_file_path)
        df.columns = df.columns.str.strip().str.lower()
        filtered_columns = df.filter(like='winner').dropna(axis=1, how='all').columns.tolist()
        col_to_keep = ['file name'] + ['subject'] + filtered_columns + ['tangles']
        df = df[col_to_keep]
        self.trials_df = df

    def find_matching_trial(self, file_name):
        return self.trials.get(file_name, None)  # Return the trial if exact match, else None

    def merge_data(self):
        """
        Merges all data into a dataframe for analysis.
        """
        self.trials_df['file name'] = self.trials_df.apply(lambda row: row['subject'] + '-' + '-'.join(row['file name'].split('-')[-2:]), axis=1)

        self.trials_df['trial'] = self.trials_df.apply(lambda row: self.find_matching_trial(row['file name']), axis=1)
        
        # Debugging: Check how many trials fail to match
        print("Total rows:", len(self.trials_df))
        print("Rows with missing trials:", self.trials_df['trial'].isna().sum())

        self.trials_df['sound cues'] = self.trials_df['trial'].apply(lambda x: x.rtc_events.get('sound cues', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.trials_df['port entries'] = self.trials_df['trial'].apply(lambda x: x.rtc_events.get('port entries', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.trials_df['sound cues onset'] = self.trials_df['sound cues'].apply(lambda x: x.onset_times if x else None)
        self.trials_df['port entries onset'] = self.trials_df['port entries'].apply(lambda x: x.onset_times if x else None)
        self.trials_df['port entries offset'] = self.trials_df['port entries'].apply(lambda x: x.offset_times if x else None)
        
        # Creates a column for subject name
        self.trials_df['subject_name'] = self.trials_df['file name'].str.split('-').str[0]

        # Create a new column that stores an array of winners for each row
        winner_columns = [col for col in self.trials_df.columns if 'winner' in col.lower()]
        self.trials_df['winner_array'] = self.trials_df[winner_columns].apply(lambda row: row.values.tolist(), axis=1)
        
        # drops all other winner columns leaving only winner_array.
        self.trials_df.drop(columns=winner_columns, inplace=True)

        # drops all rows without dopamine data.
        self.trials_df.dropna(subset=['trial'], inplace=True)
        self.trials_df.reset_index(drop=True, inplace=True)


    """***********************FINDING RANKS******************************"""
    def find_ranks_using_ds(self, file_path):
        def creating_new_df(individuals_array):
            columns = ['ID']
            # Using a loop to add the suffix to each string
            # building columns
            suffix1 = 'w'
            suffix2 = 'm'
            w_array = []
            m_array = []
            for w in individuals_array:
                w_array.append(w + suffix1)
            for m in individuals_array:
                m_array.append(m + suffix2)
            columns.extend(w_array)
            columns.extend(m_array)
            calculations = ['w', 'l', 'w2', 'l2', 'DS']
            columns.extend(calculations)
            # entering base values into the new data frame
            empty_dataframe = pd.DataFrame(columns=columns)
            empty_dataframe['ID'] = individuals_array
            # fill in the data frame with zeros
            pd.set_option('future.no_silent_downcasting', True)
            new_dataframe = empty_dataframe.fillna(0).infer_objects(copy=False)
            new_dataframe[calculations] = new_dataframe[calculations].astype(float)
            return new_dataframe, w_array, m_array
        # Read the Excel file with header
        df = pd.read_excel(file_path, header=0)
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        # Remove unneeded columns
        # to_keep = ['subject_wins', 'other_mouse_wins', 'subject', 'agent']
        to_keep = ['mouse_1_wins', 'mouse_2_wins', 'subject', 'agent']
        # Keep only the specified columns
        df = df[to_keep]
        # splitting matches to find specific individuals
        individuals = set()
        for subjects1 in df['agent']:
                individuals.add(subjects1)
        individuals_list = list(individuals)
        individuals_array = np.array(individuals_list)
        all_individuals_array = np.sort(individuals_array)  # More explicit

        # creating a dataframe using function.
        new_dataframe, w_array, m_array = creating_new_df(all_individuals_array)


        # finding the mice in each match and naming them as mouse 1 or 2 and finding all matches of these mice
        for n, row in df.iterrows():
            
            all_individuals_list = list(all_individuals_array)  # Convert array to list
            mouse1 = df.at[n, 'subject']
            # locate mouse 1 location in all_individuals_array
            index_of_mouse1 = all_individuals_list.index(mouse1)
            mouse2 = df.at[n, 'agent']
            # locate mouse 2 location in all_individuals_array
            index_of_mouse2 = all_individuals_list.index(mouse2)
            mouse1w = df.at[n, 'mouse_1_wins']
            mouse2w = df.at[n, 'mouse_2_wins']
            new_dataframe.loc[index_of_mouse1, mouse2 + 'w'] += int(mouse1w)
            new_dataframe.loc[index_of_mouse2, mouse1 + 'w'] += int(mouse2w)
    
            # total matches between the 2 mice
            mouse_m = mouse1w + mouse2w
            new_dataframe.at[index_of_mouse1, mouse2 + 'm'] += int(mouse_m)
            new_dataframe.at[index_of_mouse2, mouse1 + 'm'] += int(mouse_m)
        

        for i in range(len(all_individuals_array)):
            # w = sum of P(ij) win-rate    P(ij)=(individual1/opponent)
            w = 0
            # l = sum of P(ji) loss-rate   P(ji)=1-(individual1/opponent)
            l = 0
            for j in range(len(all_individuals_array)):
                num_wins = new_dataframe.at[i, w_array[j]]
                num_matches = new_dataframe.at[i, m_array[j]]
                # ensure that there are matches between the two
                if pd.isna(num_matches) or num_matches == 0:
                    continue
                if math.isnan(num_wins):
                    num_wins = 0
                pw = num_wins / num_matches
                pl = 1 - pw
                w += pw
                l += pl
            new_dataframe.loc[i, ['w', 'l']] = float(w), float(l)  # More efficient

            """new_dataframe.at[i, 'w'] = float(w)
            new_dataframe.at[i, 'l'] = float(l)"""

        # calculate the w2 and l2 values
        for i in range(len(all_individuals_array)):
            # w2 = sum of all P(ij) * w(j)
            w2 = 0
            # l2 = sum of all (1-P(ij)) * l(j)
            l2 = 0
            for j in range(len(all_individuals_array)):
                num_wins = new_dataframe.at[i, w_array[j]]
                num_matches = new_dataframe.at[i, m_array[j]]
                # ensure that there are matches between the two
                if math.isnan(num_matches) or num_matches == 0:
                    continue
                if math.isnan(num_wins):
                    num_wins = 0
                pw = num_wins / num_matches
                pl = 1 - pw
                w_opponent = new_dataframe.at[j, 'w']
                l_opponent = new_dataframe.at[j, 'l']
                w2 += pw * w_opponent
                l2 += pl * l_opponent
            new_dataframe.at[i, 'w2'] = float(w2)
            new_dataframe.at[i, 'l2'] = float(l2)

        # calculating David's score = w + w2 - l - l2
        for i in range(len(all_individuals_array)):
            w = new_dataframe.at[i, 'w']
            l = new_dataframe.at[i, 'l']
            w2 = new_dataframe.at[i, 'w2']
            l2 = new_dataframe.at[i, 'l2']
            ds = w + w2 - l - l2
            new_dataframe.at[i, 'DS'] = float(ds)

        # necessary output data
        data_to_keep = ['ID', 'DS']
        new_dataframe = new_dataframe[data_to_keep]

        # for cohort 1 and 2 and combined remove n8
        new_dataframe = new_dataframe.loc[new_dataframe['ID'] != 'n8']  # n8 didn't have any expression

        new_dataframe["Prefix"] = new_dataframe["ID"].str.extract(r"([a-zA-Z]+)")  # Extract letter prefix ('n' or 'p')
        new_dataframe["Number"] = new_dataframe["ID"].str.extract(r"(\d+)").astype(int)  # Extract numeric part

        # Step 3: Assign cages dynamically
        new_dataframe["Cage"] = new_dataframe.apply(lambda row: f"{row['Prefix']}{1 if row['Number'] <= 4 else 2}", axis=1)

        # Step 4: Rank within each cage
        new_dataframe["Rank"] = new_dataframe.groupby("Cage")["DS"].rank(ascending=False, method="min").astype("Int64")

        # Drop helper columns
        new_dataframe = new_dataframe.drop(columns=["Prefix", "Number"])
        return new_dataframe
    
    def merging_ranks(self, rank_df, df=None):
        if df is None:
            df=self.trials_df
        df_merged = df.merge(rank_df, left_on="subject_name", right_on="ID", how="left")

        # Step 2: Drop redundant "ID" column if needed
        df_merged.drop(columns=["ID"], inplace=True)
        df = df_merged
        return df

    """***********************REMOVING TRIALS WITH TANGLES******************************"""
    def remove_tangles(self):
        """
        Extracts tangles, removes the corresponding indices from arrays in other columns, and processes the dataframe.
        """

        # Extract 'tangles'
        self.trials_df['tangles_array'] = self.trials_df['trial'].apply(
            lambda x: x.rtc_events.get('tangles', []) if isinstance(x, object) and hasattr(x, 'rtc_events') else []
        )

        # Function to remove indices from an array based on tangles_array
        def remove_indices_from_array(array, indices_to_remove):
            return [item for idx, item in enumerate(array) if idx not in indices_to_remove]

        # Remove corresponding indices from 'winner_array' and other arrays
        self.trials_df['filtered_winner_array'] = self.trials_df.apply(
            lambda row: remove_indices_from_array(row['winner_array'], row['tangles_array']), axis=1
        )
        self.trials_df['filtered_sound_cues'] = self.trials_df.apply(
            lambda row: remove_indices_from_array(row['sound cues onset'], row['tangles_array']), axis=1
        )

        # For now, pass through port entries without filtering
        self.trials_df['filtered_port_entries'] = self.trials_df['port entries onset']
        self.trials_df['filtered_port_entry_offset'] = self.trials_df['port entries offset']

        # Only update 'first_bout' if it exists
        if 'first_bout' in self.trials_df.columns:
            self.trials_df['first_bout'] = self.trials_df.apply(
                lambda row: 'tangle' if len(row['tangles_array']) > 0 and row['tangles_array'][0] == 1 else row['first_bout'],
                axis=1
            )

        # Drop temporary columns
        self.trials_df.drop(columns=['tangles_array'], inplace=True)
        if 'tangles' in self.trials_df.columns:
            self.trials_df.drop(columns=['tangles'], inplace=True)

    """**********************ISOLATING WINNING/LOSING TRIALS************************"""
    def split_by_winner(self):
        """
        Builds self.winner_df and self.loser_df from self.da_df, excluding any 'tie' events.
        """
        df = self.da_df.copy()
        id_cols    = ['subject_name', 'file name', 'trial']
        prune_cols = [c for c in df.columns if c not in id_cols]

        def _clean_seq(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v if isinstance(v, list) else []

        def _prune_row(row, keep_win: bool):
            wins = _clean_seq(row['filtered_winner_array'])
            keep = []
            for i, w in enumerate(wins):
                if w == 'tie':
                    continue                # skip ties
                if keep_win and w == row['subject_name']:
                    keep.append(i)
                if (not keep_win) and (w != row['subject_name']):
                    keep.append(i)
            out = {}
            for col in prune_cols:
                seq = _clean_seq(row[col])
                out[col] = [ seq[i] for i in keep if i < len(seq) ]
            return pd.Series(out)

        # Winners
        win_pruned = df.apply(lambda r: _prune_row(r, True),
                            axis=1, result_type='expand')
        df_win = pd.concat([df[id_cols], win_pruned], axis=1)
        self.winner_df = df_win[df_win[prune_cols[0]].apply(len) > 0]\
                            .reset_index(drop=True)

        # Losers
        lose_pruned = df.apply(lambda r: _prune_row(r, False),
                            axis=1, result_type='expand')
        df_lose = pd.concat([df[id_cols], lose_pruned], axis=1)
        self.loser_df = df_lose[df_lose[prune_cols[0]].apply(len) > 0]\
                            .reset_index(drop=True)



    
    
    



        #  Plot win vs. Loss
    def plot_conditional(self, df_winning, df_losing, method, metric_name, directory_path,
                    custom_xtick_labels=None, 
                    custom_xtick_colors=None, 
                    ylim=None, 
                    nac_color='#15616F',   # Color for NAc
                    mpfc_color='#FFAF00',  # Color for mPFC
                    yticks_increment=1, 
                    figsize=(7,7),  
                    pad_inches=0.1):
        """
        Plotting metrics side by side with win on left and lose on right
        """
        title_suffix = 'NAc'
        title_suffix1 = 'mPFC'
        bar_color = nac_color
        bar_color1 = mpfc_color
        label1 = 'Win'
        label2 = 'Lose'
        def split_by_subject(df1):            
            # Filter out the 'subject_name' column and keep only the relevant columns for response and metric_name
            df_n = df1[df1['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p = df1[df1['subject_name'].str.startswith('p')].drop(columns=['subject_name'])

            # Return filtered dataframes and subject_name column
            return df_n, df_p

        def filter_by_metric(df):
            metric_columns = [col for col in df.columns if col.endswith(metric_name + method)]
            
            # Ensure 'Lick' column is first
            metric_columns.sort(key=lambda x: 'Lick' not in x) 

            # Create two DataFrames, keeping 'subject_name'
            df1 = df[['subject_name', metric_columns[0]]].copy()
            df2 = df[['subject_name', metric_columns[1]]].copy()

            return df1, df2

        win_df, win_df1 = filter_by_metric(df_winning)
        # win_df = lick
        # win_df1 = tone
        lose_df, lose_df1 = filter_by_metric(df_losing)
        # lose_df = lick
        # lose_df1 = tone
        
        # win lick
        win_df_n, win_df_p = split_by_subject(win_df)
        # win tone
        win_df_n1, win_df_p1 = split_by_subject(win_df1)
        # lose lick
        lose_df_n, lose_df_p = split_by_subject(lose_df)
        # lose tone
        lose_df_n1, lose_df_p1 = split_by_subject(lose_df1)

        """4 plots"""
        # plot 1: NAc Lick
        # win
        mean_values = win_df_n.mean()
        sem_values = win_df_n.sem()
        # lose
        mean_values1 = lose_df_n.mean()
        sem_values1 = lose_df_n.sem()
        # title and plot
        title = metric_name + f' Lick DA ({title_suffix})'
        self.ploting_side_by_side(win_df_n, lose_df_n, mean_values, sem_values, mean_values1, sem_values1,
                                  bar_color, figsize, metric_name, ylim, 
                                  yticks_increment, title, directory_path, pad_inches,
                                  label1, label2)
        # plot 2: mPFC Lick
        # win
        mean_values2 = win_df_p.mean()
        sem_values2 = win_df_p.sem()
        # lose
        mean_values3 = lose_df_p.mean()
        sem_values3 = lose_df_p.sem()
        title1 = metric_name +  f' Lick DA ({title_suffix1})'
        self.ploting_side_by_side(win_df_p, lose_df_p, mean_values2, sem_values2, mean_values3, sem_values3,
                                  bar_color1, figsize, metric_name, ylim, 
                                  yticks_increment, title1, directory_path, pad_inches,
                                  'Win', 'Lose')
        
        # plot 3: NAc Tone
        # win
        mean_values4 = win_df_n1.mean()
        sem_values4 = win_df_n1.sem()
        # lose
        mean_values5 = lose_df_n1.mean()
        sem_values5 = lose_df_n1.sem()
        title2 = metric_name + f' Tone DA ({title_suffix})'
        self.ploting_side_by_side(win_df_n1, lose_df_n1, mean_values4, sem_values4, mean_values5, sem_values5,
                                  bar_color, figsize, metric_name, ylim, 
                                  yticks_increment, title2, directory_path, pad_inches,
                                  'Win', 'Lose')
        
        # plot 4: mPFC Tone
        # win
        mean_values6 = win_df_p1.mean()
        sem_values6 = win_df_p1.sem()
        # lose
        mean_values7 = lose_df_p1.mean()
        sem_values7 = lose_df_p1.sem()
        title3 = metric_name +  f' Tone DA ({title_suffix1})'
        self.ploting_side_by_side(win_df_p1, lose_df_p1, mean_values6, sem_values6, mean_values7, sem_values7,
                                  bar_color1, figsize, metric_name, ylim, 
                                  yticks_increment, title3, directory_path, pad_inches,
                                  'Win', 'Lose')
    
    # plots EI peth for every event
    def rc_plot_peth_per_event(self, df, i, directory_path, title='PETH graph for n trials', signal_type='zscore', 
                            error_type='sem', display_pre_time=4, display_post_time=10, yticks_interval=2):
        
        # Plots the PETH for each event index (e.g., each sound cue) across all trials in one figure with subplots.
        
        if df is None:
            df = self.trials_df
        # Determine the indices for the display range
        time_axis = df.iloc[i]['Tone Event_Time_Axis'][0]
        display_start_idx = np.searchsorted(time_axis, -display_pre_time)
        display_end_idx = np.searchsorted(time_axis, display_post_time)
        time_axis = time_axis[display_start_idx:display_end_idx]

        num_events = 19

        # Create subplots arranged horizontally
        fig, axes = plt.subplots(1, num_events, figsize=(5 * num_events, 5), sharey=True)
        print(len(df.iloc[i]['Tone Event_Zscore']))
        for idx, event_index in enumerate(range(num_events)):
            ax = axes[idx]

            tone_event_zscores = df.iloc[i]['Tone Event_Zscore']

            # Ensure it's a list/array and has enough elements
            if not isinstance(tone_event_zscores, (list, np.ndarray)) or len(tone_event_zscores) == 0:
                print(f"Skipping row {i}, 'Tone Event_Zscore' is empty or NaN.")
                continue

            if event_index >= len(tone_event_zscores):
                print(f"Skipping index {event_index}, out of range for row {i}.")
                continue  

            event_traces = tone_event_zscores[event_index]  # Safe access
            
            # Check if event_traces is empty
            if event_traces.size == 0:
                print(f"No data for event {event_index + 1}")
                continue
            
            # Convert list of traces to numpy array
            event_traces = np.array(event_traces)
            
            # Ensure event_traces is 2D (if only one trace, add a new axis to make it 2D)
            if event_traces.ndim == 1:
                event_traces = np.expand_dims(event_traces, axis=0)  # Convert 1D array to 2D
            
            # Truncate the traces to the display range
            event_traces = event_traces[:, display_start_idx:display_end_idx]

            # Plot the traces (without error bars)
            ax.plot(time_axis, event_traces.T, color='b', label=f'Event {event_index + 1}', linewidth=1.5)  # Transpose for proper plotting
            ax.axvline(0, color='black', linestyle='--', label='Event onset')

            # Set the x-ticks to show only the last time, 0, and the very end time
            ax.set_xticks([time_axis[0], 0, time_axis[-1]])
            ax.set_xticklabels([f'{time_axis[0]:.1f}', '0', f'{time_axis[-1]:.1f}'], fontsize=12)

            # Set the y-tick labels with specified interval
            if idx == 0:
                y_min, y_max = ax.get_ylim()
                y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval,
                                    np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval,
                                    yticks_interval)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=12)
            else:
                ax.set_yticks([])  # Hide y-ticks for other subplots

            ax.set_xlabel('Time (s)', fontsize=14)
            if idx == 0:
                ax.set_ylabel(f'{signal_type.capitalize()} dFF', fontsize=14)

            # Set the title for each event
            ax.set_title(f'Event {event_index + 1}', fontsize=14)

            # Remove the right and top spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Adjust layout and add a common title
        plt.suptitle(title, fontsize=16)
        save_path = os.path.join(str(directory_path) + '\\' + str(i) + 'all_PETH.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


    def apply_rc_plot_peth(self, df, directory_path, title='PETH graph for n trials', 
                       signal_type='zscore', error_type='sem', display_pre_time=4, 
                       display_post_time=10, yticks_interval=2):
        
        # Applies the rc_plot_peth_per_event function to all rows in the DataFrame.
        
        # Loop through all rows in the DataFrame
        for i, row in df.iterrows():
            # Call the plotting function for each row
            self.rc_plot_peth_per_event(
                df, 
                i, 
                directory_path,
                title=title, 
                signal_type=signal_type, 
                error_type=error_type, 
                display_pre_time=display_pre_time, 
                display_post_time=display_post_time, 
                yticks_interval=yticks_interval
            )


    def plot_single_trial_heatmaps(self, df, condition, event_type, directory_path, brain_region):
        """
        Plots heatmaps of only the first and last trial of a given condition (win/loss).
        Each heatmap represents **one single trial**, showing Z-score variations over time.
        """
        # Function to filter data by brain region
        
        # Splitting either mPFC or NAc subjects
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            # Return filtered dataframes and subject_name column
            if region == 'mPFC':
                return df_p
            else:
                return df_n
        df = split_by_subject(df, brain_region)

        # Extract data
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]
        first_trial, last_trial = None, None

        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'])  # (num_trials, num_time_bins)
            if len(z_scores) > 0:
                first_trial = z_scores[0]   # First trial's Z-score data
                last_trial = z_scores[-1]   # Last trial's Z-score data
                break  # Only need one subject's trials

        # Convert to 2D arrays (shape: (1, time_bins)) for heatmap
        bin_size = 100

        # Downsample first and last trial data
        first_trial, new_time_axis = self.downsample_data(first_trial, common_time_axis, bin_size)
        last_trial, _ = self.downsample_data(last_trial, common_time_axis, bin_size)  

        # Convert to 2D array for heatmap (since we have only one row)
        first_trial = first_trial[np.newaxis, :]
        last_trial = last_trial[np.newaxis, :]
        # Normalize color scale
        vmin, vmax = min(first_trial.min(), last_trial.min()), max(first_trial.max(), last_trial.max())

        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        if brain_region == "mPFC":
            cmap = 'inferno'
        else:
            cmap = 'viridis'

        for ax, trial_data, title in zip(axes, [first_trial, last_trial], 
                                        [f'First {condition} trial', f'Last {condition} trial']):
            # Plot heatmap
            cax = ax.imshow(trial_data, aspect='auto', cmap=cmap, origin='upper',
                            extent=[common_time_axis[0], common_time_axis[-1], 0, 1],
                            vmin=vmin, vmax=vmax)

            # Formatting
            ax.set_title(title, fontsize=14)
            ax.set_yticks([])  # Remove y-axis ticks (since only one row)
            ax.axvline(0, color='white', linestyle='--', linewidth=2)  # Mark event onset
            if brain_region == "mPFC":
                line_color='blue'
            else:
                line_color='pink'
            ax.axvline(4, color=line_color, linestyle='-', linewidth=2)

        # Set x-axis labels only on the bottom plot
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        axes[-1].set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
        axes[-1].set_xticklabels(['-4', '0', '4', '10'], fontsize=10)

        # Add colorbar to represent Z-score intensity
        cbar = fig.colorbar(cax, ax=axes, orientation='vertical', shrink=0.7, label='Z-score')

        # Save and show
        if directory_path is not None:
            save_path = os.path.join(directory_path, f'{brain_region}_{condition}_Single_Trials_Heatmap.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_first_tone_heatmaps(self, df, condition, event_type, directory_path, brain_region, plot_first=True):
        """
        Plots a heatmap of only the first or last trial of a given condition (win/loss).
        Each heatmap represents **one single trial**, showing Z-score variations over time.
        """
        # Function to filter data by brain region
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            # Return filtered dataframes and subject_name column
            if region == 'mPFC':
                return df_p
            else:
                return df_n

        df = split_by_subject(df, brain_region)

        # Extract data
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]
        first_trial, last_trial = None, None

        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'])  # (num_trials, num_time_bins)
            if len(z_scores) > 0:
                first_trial = z_scores[0]   # First trial's Z-score data
                last_trial = z_scores[-1]   # Last trial's Z-score data
                break  # Only need one subject's trials

        # Convert to 2D arrays (shape: (1, time_bins)) for heatmap
        bin_size = 100  

        # Downsample first and last trial data
        first_trial, new_time_axis = self.downsample_data(first_trial, common_time_axis, bin_size)
        last_trial, _ = self.downsample_data(last_trial, common_time_axis, bin_size)  

        # Convert to 2D array for heatmap (since we have only one row)
        first_trial = first_trial[np.newaxis, :]
        last_trial = last_trial[np.newaxis, :]

        # Normalize color scale
        vmin, vmax = min(first_trial.min(), last_trial.min()), max(first_trial.max(), last_trial.max())

        # Create figure with one subplot (if only one heatmap is to be shown)
        fig, ax = plt.subplots(figsize=(12, 4))

        if brain_region == "mPFC":
            cmap = 'inferno'
        else:
            # colors = ["#08306b", "#4292c6", "#deebf7", "#ffffff"]  
            # cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)
            cmap = 'viridis'

        # Choose the trial to plot (first or last trial based on plot_first argument)
        if plot_first:
            trial_data = first_trial
            title = f'First {condition} trial'
        else:
            trial_data = last_trial
            title = f'Last {condition} trial'

        # Plot heatmap for the selected trial
        cax = ax.imshow(trial_data, aspect='auto', cmap=cmap, origin='upper',
                        extent=[common_time_axis[0], common_time_axis[-1], 0, 1],
                        vmin=vmin, vmax=vmax)

        # Formatting
        ax.set_title(title, fontsize=24)
        ax.set_yticks([])  # Remove y-axis ticks (since only one row)
        ax.axvline(0, color='white', linestyle='--', linewidth=2)  # Mark event onset
        ax.axvline(4, color='pink', linestyle='-', linewidth=2)

        # Set x-axis labels
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=20)
        cbar_ax = fig.add_axes([0.05, 0.1, 0.03, 0.8])
        # Add colorbar to represent Z-score intensity
        cbar = fig.colorbar(cax, cax=cbar_ax, ax=ax, orientation='vertical', shrink=0.7, label='Z-score')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label("Z-score", fontsize=20)
        
        cbar.ax.yaxis.set_ticks_position('left')  # This moves the ticks to the left side
        cbar.ax.yaxis.set_label_position('left') 

        # Save and show the plot
        if directory_path is not None:    
            save_path = os.path.join(directory_path, f'{brain_region}_{condition}_Single_Trial_Heatmap.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_single_psth(self, df, condition, event_type, directory_path, brain_region, y_min, y_max, plot_first=True):
        """
        Plots the PETH of either the first or last bout of either win or loss.
        If plot_first=True, it will plot the first bout. If plot_first=False, it will plot the last bout.
        """
        # Splitting either mPFC or NAc subjects
        if df is None:
            df = self.trials_df
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            # Return filtered dataframes and subject_name column
            if region == 'mPFC':
                return df_p
            else:
                return df_n

        df = split_by_subject(df, brain_region)
        bin_size = 100
        if brain_region == 'mPFC':
            color = '#FFAF00'
        else:
            color = '#15616F'
        
        # Initialize data structures
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]
        first_events = []
        last_events = []
        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'])  # Shape: (num_1D_arrays, num_time_bins)
            
            # Ensure there is at least one 1D array in the row
            if len(z_scores) > 0:
                first_events.append(z_scores[0])   # First 1D array
                last_events.append(z_scores[-1])   # Last 1D array

        # Convert lists to numpy arrays (num_trials, num_time_bins)
        first_events = np.array(first_events)
        last_events = np.array(last_events)

        # Compute mean and SEM
        mean_first = np.mean(first_events, axis=0)
        sem_first = np.std(first_events, axis=0) / np.sqrt(first_events.shape[0])

        mean_last = np.mean(last_events, axis=0)
        sem_last = np.std(last_events, axis=0) / np.sqrt(last_events.shape[0])

        # Choose the event to plot (first or last bout)
        if plot_first:
            mean_peth = mean_first
            sem_peth = sem_first
            title = f'First {condition} bout Z-Score'
        else:
            mean_peth = mean_last
            sem_peth = sem_last
            title = f'Last {condition} bout Z-Score'
        # Create figure with a single subplot
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.set_ylabel('Event Induced Z-scored ΔF/F', fontsize=20)
        ax.tick_params(axis='y', labelleft=True)
        mean_peth, downsampled_time_axis = self.downsample_data(mean_peth, common_time_axis, bin_size)
        sem_peth, _ = self.downsample_data(sem_peth, common_time_axis, bin_size)

        # Plot the selected event
        ax.plot(downsampled_time_axis, mean_peth, color=color, label='Mean DA')
        ax.fill_between(downsampled_time_axis, mean_peth - sem_peth, mean_peth + sem_peth, color=color, alpha=0.4)
        ax.axvline(0, color='black', linestyle='--')  # Mark event onset
        ax.axvline(4, color='pink', linestyle='-')

        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=20)
        # Add a margin to make sure the mean trace doesn't go out of bounds
        ax.set_ylim(y_max, y_min)

        # Save the figure
        if directory_path is not None:
            save_path = os.path.join(str(directory_path) + '\\' + f'{brain_region}_{condition}_PETH.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    def scatter_dominance(self, directory_path, df, metric_name, method, condition, pad_inches=0.1):
        """
        Scatter plot of dominance rank in a cage.
        """
        def filter_by_metric(df, metric_name):
            metric_columns = df.columns[df.columns == f'{metric_name} {method}']

            # Create two DataFrames, keeping 'subject_name'
            df_tone = df[['subject_name', 'Cage', 'Rank'] + metric_columns.tolist()].copy()
            return df_tone
        
        # spliting and copying dataframe into two dataframes for each brain region 
        def split_by_subject(df1):            
            # Filter out the 'subject_name' column and keep only the relevant columns for response and metric_name
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            
            """df_n1 = df2[df2['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p1 = df2[df2['subject_name'].str.startswith('p')].drop(columns=['subject_name'])"""
            # Return filtered dataframes and subject_name column
            return df_n, df_p
        
        df_tone = filter_by_metric(df, metric_name)
        df_tone_n, df_tone_p = split_by_subject(df_tone)

        df_sorted_n = df_tone_n.sort_values(by=['Rank'])
        df_sorted_p = df_tone_p.sort_values(by=['Rank'])

        def scatter_plot(directory_path, df_sorted, method, metric_value, condition, brain_region):
            if brain_region == "mPFC":
                color = '#FFAF00'
            else:
                color = '#15616F'

            # Drop rows where 'Rank' is NaN
            df_sorted = df_sorted.dropna(subset=['Rank'])
            x = df_sorted['Rank']
            y = df_sorted[f'{metric_value} {method}']
            
            if len(x) > 1:  # Pearson requires at least 2 points
                r_value, p_value = pearsonr(x, y)
            else:
                r_value, p_value = float('nan'), float('nan')  # Handle cases with insufficient data
            
            n_value = len(df_sorted)

            # Create the scatter plot with a regression line
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=df_sorted, x='Rank', y=f'{metric_value} {method}', color=color, s=150, edgecolor='black')

            # Add a regression line with R², and remove the shading (confidence interval)
            sns.regplot(data=df_sorted, x='Rank', y=f'{metric_value} {method}', scatter=False, color='black', line_kws={'lw': 2.5}, ci=None)

            print(df_sorted['Rank'].isna().sum())
            print(df_sorted[['Rank']].dropna().head())  # Show non-NaN values

            # Set the x-axis ticks to be separated by increments of 1
            plt.xticks(ticks=range(int(df_sorted['Rank'].min()), int(df_sorted['Rank'].max()) + 1, 1), fontsize=18)
            plt.yticks(fontsize=18)

            # Labels and title with larger fonts
            plt.xlabel('Rank', fontsize=20, labelpad=10)
            plt.ylabel('AUC Event Induced Z-scored ΔF/F', fontsize=20, labelpad=10)
            plt.title(f'{condition} {metric_value} Tone Response to Rank', fontsize=22, fontweight='bold', pad=15)

            # Remove the top and right spines (graph borders)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            # Thicken left and bottom spines
            plt.gca().spines['left'].set_linewidth(2.5)
            plt.gca().spines['bottom'].set_linewidth(2.5)

            # Save the figure
            title = f'{metric_value} {condition} DA response Rank ({brain_region})'
            save_path = os.path.join(str(directory_path), f'{title}.png')
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0.2)
            plt.show()

            return r_value, p_value, n_value

        r_nac, p_nac, n_nac = scatter_plot(directory_path, df_sorted_n, method=method, metric_value=metric_name, condition=condition, brain_region="NAc")
        r_mpfc, p_mpfc, n_mpfc = scatter_plot(directory_path, df_sorted_p, method=method, metric_value=metric_name, condition=condition, brain_region="mPFC")

        print(f"NAc: r={r_nac:.3f}, p={p_nac:.3f}, n={n_nac}")
        print(f"mPFC: r={r_mpfc:.3f}, p={p_mpfc:.3f}, n={n_mpfc}")

        
    """********************************MISC*************************************"""
    def downsample_data(self, data, time_axis, bin_size=10):
            """
            Downsamples the time series data by averaging over bins of 'bin_size' points.
            
            Parameters:
            - data (1D NumPy array): Original Z-score values.
            - time_axis (1D NumPy array): Corresponding time points.
            - bin_size (int): Number of original bins to merge into one.
            
            Returns:
            - downsampled_data (1D NumPy array): Smoothed Z-score values.
            - new_time_axis (1D NumPy array): Adjusted time points.
            """
            num_bins = len(data) // bin_size  # Number of new bins
            data = data[:num_bins * bin_size]  # Trim excess points
            time_axis = time_axis[:num_bins * bin_size]

            # Reshape and compute mean for each bin
            downsampled_data = data.reshape(num_bins, bin_size).mean(axis=1)
            new_time_axis = time_axis.reshape(num_bins, bin_size).mean(axis=1)

            return downsampled_data, new_time_axis



    def compute_subject_mean_traces(self,
                                    df: pd.DataFrame,
                                    event_type: str) -> pd.DataFrame:
        """
        Given a DataFrame (winner_df or loser_df) that has columns
          f"{event_type}_Zscore" and f"{event_type}_Time_Axis",
        compute each subject's mean peri-event trace by stacking
        all of their per-trial traces.

        Returns a DataFrame with columns:
           'subject_name', 'mean_trace', 'time_axis'
        """
        rows = []
        zcol = f"{event_type}_Zscore"
        tcol = f"{event_type}_Time_Axis"

        # group by subject
        for subj, subdf in df.groupby("subject_name"):
            all_traces = []
            for _, row in subdf.iterrows():
                traces = row.get(zcol, []) or []
                for tr in traces:
                    all_traces.append(np.asarray(tr))
            if not all_traces:
                continue
            all_traces = np.vstack(all_traces)         # shape (n_events, n_time)
            mean_tr   = np.nanmean(all_traces, axis=0) # 1d array

            # grab one representative time_axis
            ta0 = np.asarray(subdf.iloc[0][tcol][0])

            rows.append({
                "subject_name": subj,
                "mean_trace":   mean_tr,
                "time_axis":    ta0
            })

        return pd.DataFrame(rows)


    def plot_group_mean_traces(self,
                           subj_traces: pd.DataFrame,
                           event_type: str,
                           brain_region: str,
                           color: str = None,
                           title: str = None,
                           ylim: tuple = None,
                           figsize=(8,5),
                           save_path: str = None):
        """
        Given the per-subject mean traces (output of compute_subject_mean_traces),
        filter by brain_region ('NAc' vs 'mPFC'), compute group mean ± SEM,
        and plot a single PSTH.
        """
        # filter by prefix n* vs p*
        if brain_region == "NAc":
            grp = subj_traces[subj_traces["subject_name"].str.startswith("n")]
            default_color = "#15616F"
        else:
            grp = subj_traces[subj_traces["subject_name"].str.startswith("p")]
            default_color = "#FFAF00"

        if grp.empty:
            print(f"No data for region {brain_region}")
            return

        # assemble array (n_subjects x n_time)
        M = np.vstack(grp["mean_trace"].values)
        mean_trace = np.nanmean(M, axis=0)
        sem_trace  = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])

        # time_axis (shared by all)
        t = grp.iloc[0]["time_axis"]

        # (skip downsampling here; use your downsample_data if desired)
        ds_time, ds_mean, ds_sem = t, mean_trace, sem_trace

        c = color or default_color

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ds_time, ds_mean, color=c, lw=3, label=f"{brain_region} Mean")
        ax.fill_between(ds_time,
                        ds_mean - ds_sem,
                        ds_mean + ds_sem,
                        color=c, alpha=0.4,
                        label="SEM")

        # vertical onset lines
        ax.axvline(0, color="k",      ls="--", lw=2)
        ax.axvline(4, color="#FF69B4",ls="--", lw=2)

        # labels & title
        ax.set_xlabel("Time (s)",         fontsize=14)
        ax.set_ylabel("z-scored ΔF/F",    fontsize=14)
        if title:
            ax.set_title(title, fontsize=18, fontweight='bold')
        else:
            ax.set_title(f"{brain_region} {event_type} PSTH", fontsize=18, fontweight='bold')

        # apply custom y-limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        # ticks & layout
        ax.set_xticks([-4, 0, 4, 10])
        ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)

        # remove top & right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()



    def old_plot_mean_psth(self,
                       winner_df: pd.DataFrame,
                       event_type: str,
                       brain_region: str,
                       bin_size: int = 100,
                       directory_path: str = None):
        """
        OLD METHOD: For each row in winner_df (every session), 
        collapse all per‐event traces → one session trace, then
        average across all sessions (regardless of subject).
        """
        import os, numpy as np, matplotlib.pyplot as plt

        # 1) pick only the region of interest
        prefix = 'n' if brain_region == 'NAc' else 'p'
        df_reg = winner_df[winner_df['subject_name'].str.startswith(prefix)]
        if df_reg.empty:
            print(f"No data for region {brain_region}")
            return

        # 2) collect one PSTH per session
        session_traces = []
        for _, row in df_reg.iterrows():
            # each row has a list of per‐event z‐score arrays
            evts = row[f'{event_type}_Zscore']
            if not isinstance(evts, (list, np.ndarray)) or len(evts) == 0:
                continue

            # stack [n_events x n_timebins]
            mat = np.vstack(evts)
            # mean across events → 1D array of length n_timebins
            session_trace = np.nanmean(mat, axis=0)
            session_traces.append(session_trace)

        if len(session_traces) == 0:
            print("No valid session traces found.")
            return

        # 3) stack all session traces → shape (n_sessions, n_timebins)
        M = np.vstack(session_traces)

        # 4) compute grand mean & SEM across sessions
        mean_psth = np.nanmean(M, axis=0)
        sem_psth  = np.nanstd(M,  axis=0) / np.sqrt(M.shape[0])

        # 5) grab time‐axis from the first row
        common_time_axis = df_reg.iloc[0][f'{event_type}_Time_Axis'][0]
        # downsample for smooth plotting
        def downsample(data, time, bs):
            n = len(data) // bs
            d = data[:n*bs].reshape(n, bs).mean(axis=1)
            t = time[:n*bs].reshape(n, bs).mean(axis=1)
            return d, t

        ds_mean, ts = downsample(mean_psth, common_time_axis, bin_size)
        ds_sem,  _  = downsample(sem_psth,   common_time_axis, bin_size)

        # 6) plot
        color = '#15616F' if brain_region=='NAc' else '#FFAF00'
        plt.figure(figsize=(8,5))
        plt.plot(ts, ds_mean, color=color, lw=3)
        plt.fill_between(ts, ds_mean-ds_sem, ds_mean+ds_sem, color=color, alpha=0.4)
        plt.axvline(0, color='k', ls='--', lw=2)
        plt.axvline(4, color='#FF69B4', ls='-', lw=2)
        plt.xlabel('Time from Cue (s)')
        plt.ylabel('Z-scored ΔF/F')
        plt.title(f'OLD MEAN PSTH: {event_type} ({brain_region})')
        plt.tight_layout()

        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"old_{brain_region}_{event_type}_meanPSTH.png"
            plt.savefig(os.path.join(directory_path, fname), dpi=300, bbox_inches='tight')

        plt.show()



    def plot_win_vs_loss(self,
                         metric_name: str,
                         behavior: str,
                         brain_region: str,
                         directory_path: str = None,
                         color_win: str = '#15616F',
                         color_loss: str = '#FFAF00',
                         figsize: tuple = (5,5),
                         pad_inches: float = 0.2):
        """
        Bar graph: Win vs Loss for a single [behavior metric_name].
        Automatically computes p-value & Cohen's d, places stats box on the right.
        """
        # 1) column key
        col = f"{behavior} {metric_name}"
        for df in (self.winner_df, self.loser_df):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from winner_df or loser_df")

        # 2) pick the right region
        prefix = 'n' if brain_region=='NAc' else 'p'
        win_df  = self.winner_df[self.winner_df['subject_name'].str.startswith(prefix)]
        lose_df = self.loser_df [self.loser_df ['subject_name'].str.startswith(prefix)]

        # 3) collapse each row’s list → single session number, then average across sessions per subject
        def subj_means(df):
            tmp = df[['subject_name', col]].copy()
            tmp[col] = tmp[col].apply(lambda lst:
                                      np.nanmean(lst) if isinstance(lst,(list,np.ndarray))
                                      else np.nan)
            tmp = tmp.dropna(subset=[col])
            # one value per subject
            return tmp.groupby('subject_name')[col].mean().values

        v_win  = subj_means(win_df)
        v_loss = subj_means(lose_df)

        # 4) stats
        tstat, pval = ttest_ind(v_win, v_loss, nan_policy='omit', equal_var=False)
        n1, n2 = len(v_win), len(v_loss)
        sd1, sd2 = np.nanstd(v_win, ddof=1), np.nanstd(v_loss, ddof=1)
        pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
        cohens_d = (np.nanmean(v_win) - np.nanmean(v_loss)) / pooled_sd

        # 5) means & sem
        m1, m2 = np.nanmean(v_win), np.nanmean(v_loss)
        sem1, sem2 = sd1/np.sqrt(n1), sd2/np.sqrt(n2)

        # 6) plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = [0,1]
        w = 0.6

        ax.bar(0, m1, width=w, yerr=sem1, capsize=8,
               color=color_win,  edgecolor='k', label='Win')
        ax.bar(1, m2, width=w, yerr=sem2, capsize=8,
               color=color_loss, edgecolor='k', label='Loss')

        # overlay each subject
        jitter = 0.05
        ax.scatter(np.zeros(n1)+np.random.uniform(-jitter, jitter, n1),
                   v_win,  facecolors='none', edgecolors='black',
                   s=80, alpha=0.8, zorder=3)
        ax.scatter(np.ones(n2)+np.random.uniform(-jitter, jitter, n2),
                   v_loss, facecolors='none', edgecolors='black',
                   s=80, alpha=0.8, zorder=3)

        # labels & title
        ax.set_xticks(x)
        ax.set_xticklabels(['Win','Loss'], fontsize=14)
        ax.set_xlabel('Outcome', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region} {behavior} {metric_name}: Win vs Loss",
                     fontsize=18, pad=12)

        # stats box
        stats_txt = f"p = {pval:.3f}\nCohen’s d = {cohens_d:.2f}"
        ax.text(1.05, 0.5, stats_txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray"))

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # 7) save
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_win_vs_loss.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)

        plt.show()
        return {'t_stat': tstat, 'p_value': pval, 'cohen_d': cohens_d}

    def plot_comp_vs_alone_by_subject(self,
                           df_alone,
                           df_comp,
                           metric_name: str,
                           behavior: str,
                           brain_region: str,
                           brain_color: str,
                           directory_path: str = None,
                           figsize: tuple = (5,5),
                           pad_inches: float = 0.2):
        """
        Compare Reward-Training (alone) vs Reward-Competition (win) by subject.
        Bars are the same color (brain_color), dots lie on the bar centers.
        """
        # 1) column name
        col = f"{behavior} {metric_name}"
        for df in (df_alone, df_comp):
            if col not in df.columns:
                raise KeyError(f"'{col}' missing from input dataframe")

        # 2) filter by brain region
        def pick(df):
            mask = df['subject_name'].str.startswith('n') if brain_region=='NAc' else df['subject_name'].str.startswith('p')
            return df[mask]

        a = pick(df_alone)
        c = pick(df_comp)

        # 3) flatten each row’s list to its mean, then average across sessions per subject
        def subj_means(df):
            tmp = df[['subject_name', col]].copy()
            tmp[col] = tmp[col].apply(lambda lst: np.nanmean(lst) 
                                    if isinstance(lst,(list,np.ndarray)) else np.nan)
            tmp = tmp.dropna(subset=[col])
            return tmp.groupby('subject_name')[col].mean()

        v1 = subj_means(a).values
        v2 = subj_means(c).values

        # 4) stats
        tstat, pval = ttest_ind(v1, v2, nan_policy='omit', equal_var=False)
        n1, n2 = len(v1), len(v2)
        sd1, sd2 = np.nanstd(v1,ddof=1), np.nanstd(v2,ddof=1)
        pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2)/(n1+n2-2))
        cohens_d = (np.nanmean(v1) - np.nanmean(v2)) / pooled_sd

        # 5) means & sem
        m1, m2 = np.nanmean(v1), np.nanmean(v2)
        sem1 = sd1/np.sqrt(n1)
        sem2 = sd2/np.sqrt(n2)

        # 6) plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = [0,1]
        width = 0.6

        # bars
        ax.bar(0, m1, width, yerr=sem1, capsize=8,
            color=brain_color, edgecolor='k', linewidth=2)
        ax.bar(1, m2, width, yerr=sem2, capsize=8,
            color=brain_color, edgecolor='k', linewidth=2)

        # dots on bar centers
        ax.scatter([0]*n1, v1, facecolors='white', edgecolors='black',
                s=120, linewidth=2, zorder=3)
        ax.scatter([1]*n2, v2, facecolors='white', edgecolors='black',
                s=120, linewidth=2, zorder=3)

        # labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Alone', 'Win (Comp)'], fontsize=14)
        ax.set_xlabel('Context', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region} {behavior} {metric_name}", fontsize=18, pad=12)

        # zero line
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        # stats box
        stats_txt = f"p = {pval:.3f}\nCohen’s d = {cohens_d:.2f}"
        ax.text(1.05, 0.5, stats_txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="gray"))

        # clean
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # save
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_alone_vs_comp.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)

        plt.show()
        return {'t_stat': tstat, 'p_value': pval, 'cohen_d': cohens_d}

    def plot_reward_comp_vs_alone(self,
                                df_alone,
                                df_comp,
                                metric_name: str,
                                behavior: str,
                                brain_region: str,
                                directory_path: str = None,
                                color_alone: str = '#15616F',
                                color_comp:  str = '#FFAF00',
                                figsize: tuple = (5,5),
                                pad_inches: float = 0.2):
        """
        A two-bar “Context” plot: Alone vs Competition for a single metric.
        Automatically computes p-value and Cohen’s d, places stats box on right.
        """
        # 1) column key
        col = f"{behavior} {metric_name}"
        if col not in df_alone.columns or col not in df_comp.columns:
            raise KeyError(f"Column {col} missing in one of the dataframes")

        # 2) pick out your region
        def sel_region(df):
            return df[df['subject_name'].str.startswith('n')] if brain_region=='NAc' \
                else df[df['subject_name'].str.startswith('p')]

        # 3) collapse lists → single session mean
        def collapse(df):
            arr = df[col].apply(lambda x: np.nanmean(x)
                                if isinstance(x,(list,np.ndarray)) else np.nan)
            return arr.dropna().values

        arr1 = collapse(sel_region(df_alone))
        arr2 = collapse(sel_region(df_comp))

        # 4) statistics
        tstat, pval = ttest_ind(arr1, arr2,
                                nan_policy='omit',
                                equal_var=False)
        n1, n2 = len(arr1), len(arr2)
        s1, s2 = np.nanstd(arr1,ddof=1), np.nanstd(arr2,ddof=1)
        # pooled SD for Cohen's d
        pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        cohend   = (np.nanmean(arr1)-np.nanmean(arr2)) / pooled_sd

        # 5) means & SEM
        m1, m2 = np.nanmean(arr1), np.nanmean(arr2)
        sem1    = s1/np.sqrt(n1)
        sem2    = s2/np.sqrt(n2)

        # 6) plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = np.array([0,1])
        w = 0.6

        # bars
        ax.bar(0, m1, yerr=sem1, width=w,
            color=color_alone, edgecolor='k', capsize=8,
            label='Alone')
        ax.bar(1, m2, yerr=sem2, width=w,
            color=color_comp,  edgecolor='k', capsize=8,
            label='Competition')

        # jittered points
        jitter = 0.08
        ax.scatter(np.zeros(n1)+np.random.uniform(-jitter,jitter,n1),
                arr1, facecolors='none', edgecolors='gray', s=80, alpha=0.7)
        ax.scatter(np.ones(n2) +np.random.uniform(-jitter,jitter,n2),
                arr2, facecolors='none', edgecolors='gray', s=80, alpha=0.7)

        # labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Alone','Competition'], fontsize=14)
        ax.set_xlabel('Context', fontsize=16)
        ax.set_ylabel(f"{metric_name} Z-scored ΔF/F", fontsize=16)
        ax.set_title(f"{brain_region} {behavior} {metric_name}", fontsize=18, pad=12)

        # stats textbox
        stats_txt = f"p = {pval:.3f}\nCohen’s d = {cohend:.2f}"
        ax.text(1.05, 0.5, stats_txt,
                transform=ax.transAxes,
                va='center', ha='left',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="gray"))

        # clean
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        plt.tight_layout()
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{behavior}_{metric_name}_alone_vs_comp.png"
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', pad_inches=pad_inches)
        plt.show()

        # return stats if you want to log them
        return {'t_stat': tstat, 'p_value': pval, 'cohen_d': cohend}

