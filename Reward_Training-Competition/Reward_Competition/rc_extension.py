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
        self.df = pd.DataFrame()

    """*********************************COMBINE CONSECUTIVE BEHAVIORS***********************************"""
    """def combine_consecutive_rtc_events(self, behavior_name='all', bout_time_threshold=0.5, min_occurrences=1):
        
        Applies the behavior combination logic to all trials within the experiment.
        

        for trial_name, trial_obj in self.trials.items():
            # Ensure the trial has rtc_events attribute
            if not hasattr(trial_obj, 'rtc_events'):
                continue  # Skip if rtc_events is not available

            # Determine which behaviors to process
            if behavior_name == 'all':
                behaviors_to_process = trial_obj.rtc_events.keys()  # Process all behaviors
            else:
                behaviors_to_process = [behavior_name]  # Process a single behavior

            for behavior_event in behaviors_to_process:
                behavior_onsets = np.array(trial_obj.rtc_events[behavior_event].onset)
                behavior_offsets = np.array(trial_obj.rtc_events[behavior_event].offset)

                combined_onsets = []
                combined_offsets = []
                combined_durations = []

                if len(behavior_onsets) == 0:
                    continue  # Skip this behavior if there are no onsets

                start_idx = 0

                while start_idx < len(behavior_onsets):
                    # Initialize the combination window with the first behavior onset and offset
                    current_onset = behavior_onsets[start_idx]
                    current_offset = behavior_offsets[start_idx]

                    next_idx = start_idx + 1

                    # Check consecutive events and combine them if they fall within the threshold
                    while next_idx < len(behavior_onsets) and (behavior_onsets[next_idx] - current_offset) <= bout_time_threshold:
                        # Update the end of the combined bout
                        current_offset = behavior_offsets[next_idx]
                        next_idx += 1

                    # Add the combined onset, offset, and total duration to the list
                    combined_onsets.append(current_onset)
                    combined_offsets.append(current_offset)
                    combined_durations.append(current_offset - current_onset)

                    # Move to the next set of events
                    start_idx = next_idx

                # Filter out bouts with fewer than the minimum occurrences
                valid_indices = []
                for i in range(len(combined_onsets)):
                    num_occurrences = len([onset for onset in behavior_onsets if combined_onsets[i] <= onset <= combined_offsets[i]])
                    if num_occurrences >= min_occurrences:
                        valid_indices.append(i)

                # Update the behavior with the combined onsets, offsets, and durations
                trial_obj.rtc_events[behavior_event].onset = [combined_onsets[i] for i in valid_indices]
                trial_obj.rtc_events[behavior_event].offset = [combined_offsets[i] for i in valid_indices]
                trial_obj.rtc_events[behavior_event].Total_Duration = [combined_durations[i] for i in valid_indices]  # Update Total Duration

                trial_obj.bout_dict = {}  # Reset bout dictionary after processing"""

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
        self.df = df

    def find_matching_trial(self, file_name):
        return self.trials.get(file_name, None)  # Return the trial if exact match, else None

    def merge_data(self):
        """
        Merges all data into a dataframe for analysis.
        """
        self.df['file name'] = self.df.apply(lambda row: row['subject'] + '-' + '-'.join(row['file name'].split('-')[-2:]), axis=1)

        self.df['trial'] = self.df.apply(lambda row: self.find_matching_trial(row['file name']), axis=1)
        
        # Debugging: Check how many trials fail to match
        print("Total rows:", len(self.df))
        print("Rows with missing trials:", self.df['trial'].isna().sum())

        self.df['sound cues'] = self.df['trial'].apply(lambda x: x.rtc_events.get('sound cues', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.df['port entries'] = self.df['trial'].apply(lambda x: x.rtc_events.get('port entries', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.df['sound cues onset'] = self.df['sound cues'].apply(lambda x: x.onset_times if x else None)
        self.df['port entries onset'] = self.df['port entries'].apply(lambda x: x.onset_times if x else None)
        self.df['port entries offset'] = self.df['port entries'].apply(lambda x: x.offset_times if x else None)
        
        # Creates a column for subject name
        self.df['subject_name'] = self.df['file name'].str.split('-').str[0]

        # Create a new column that stores an array of winners for each row
        winner_columns = [col for col in self.df.columns if 'winner' in col.lower()]
        self.df['winner_array'] = self.df[winner_columns].apply(lambda row: row.values.tolist(), axis=1)
        
        # drops all other winner columns leaving only winner_array.
        self.df.drop(columns=winner_columns, inplace=True)

        # drops all rows without dopamine data.
        self.df.dropna(subset=['trial'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)


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
        new_dataframe = new_dataframe.loc[new_dataframe['ID'] != 'n8']

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
            df=self.df
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
        # self.df['tangles'] = [[] for _ in range(len(self.df))]
        
        # Extract 'tangles'
        self.df['tangles_array'] = self.df['trial'].apply(
            lambda x: x.rtc_events.get('tangles', []) if isinstance(x, object) and hasattr(x, 'rtc_events') else []
        )
        
        # Function to remove indices from an array based on tangles_array
        def remove_indices_from_array(array, indices_to_remove):
            return [item for idx, item in enumerate(array) if idx not in indices_to_remove]

        # Remove corresponding indices from 'winner_array' and other arrays
        self.df['filtered_winner_array'] = self.df.apply(
            lambda row: remove_indices_from_array(row['winner_array'], row['tangles_array']), axis=1
        )
        self.df['filtered_sound_cues'] = self.df.apply(
            lambda row: remove_indices_from_array(row['sound cues onset'], row['tangles_array']), axis=1
        )
        """self.df['filtered_port_entries'] = self.df.apply(
            lambda row: remove_indices_from_array(row['port entries onset'], row['tangles_array']), axis=1
        )
        self.df['filtered_port_entry_offset'] = self.df.apply(
            lambda row: remove_indices_from_array(row['port entries offset'], row['tangles_array']), axis=1
        )"""
        
        self.df['filtered_port_entries'] = self.df['port entries onset']
        self.df['filtered_port_entry_offset'] = self.df['port entries offset']

        # Update 'win_or_lose' if the first value of 'tangles_array' is 1
        self.df['first_bout'] = self.df.apply(
            lambda row: 'tangle' if len(row['tangles_array']) > 0 and row['tangles_array'][0] == 1 else row['first_bout'], axis=1
        )

        # drop the 'tangles_array' column if no longer needed
        self.df.drop(columns=['tangles_array'], inplace=True)
        self.df.drop(columns=['tangles'], inplace=True)

    """**********************ISOLATING WINNING/LOSING TRIALS************************"""
    def winning(self):
        def filter_by_winner(row):
            # Ensure 'filtered_winner_array' is a list
            if not isinstance(row['filtered_winner_array'], list):
                print(f"Skipping row {row.name}: filtered_winner_array is not a list.")
                return pd.Series([row['filtered_sound_cues']])

            # Get valid indices where `filtered_winner_array` matches `subject_name`
            valid_indices = [i for i, winner in enumerate(row['filtered_winner_array']) if winner == row['subject_name']]
            
            # print(f"Row {row.name}: Subject {row['subject_name']} - Valid Indices: {valid_indices}")

            # If no indices match, return an empty list
            if not valid_indices:
                return pd.Series([[]])

            # Ensure `filtered_sound_cues` is a list before filtering
            if not isinstance(row['filtered_sound_cues'], list):
                print(f"Skipping row {row.name}: filtered_sound_cues is not a list.")
                return pd.Series([[]])

            # Filter sound cues using valid indices
            filtered_cues = [row['filtered_sound_cues'][i] for i in valid_indices]

            # Print changes per row
            # print(f"Row {row.name}: Before -> {row['filtered_sound_cues']}, After -> {filtered_cues}")

            return pd.Series([filtered_cues])

        # Create a copy of the DataFrame to avoid modifying self.df
        df_copy = self.df.copy()

        # Apply the function to the copied DataFrame
        df_copy[['filtered_sound_cues']] = df_copy.apply(filter_by_winner, axis=1)

        # Drop rows where no filtered_sound_cues remain
        df_filtered = df_copy[df_copy['filtered_sound_cues'].apply(len) > 0]

        return df_filtered


    def losing(self):
        def filter_by_loser(row):
            # Ensure 'filtered_winner_array' is a list
            if not isinstance(row['filtered_winner_array'], list):
                print(f"Skipping row {row.name}: filtered_winner_array is not a list.")
                return pd.Series([row['filtered_sound_cues']])

            # Get indices where `subject_name` is NOT the winner
            valid_indices = [i for i, winner in enumerate(row['filtered_winner_array']) if winner != row['subject_name']]

            # If no losing trials exist, return an empty list
            if not valid_indices:
                return pd.Series([[]])

            # Ensure `filtered_sound_cues` is a list before filtering
            if not isinstance(row['filtered_sound_cues'], list):
                print(f"Skipping row {row.name}: filtered_sound_cues is not a list.")
                return pd.Series([[]])

            # Filter sound cues using valid indices
            filtered_cues = [row['filtered_sound_cues'][i] for i in valid_indices]

            return pd.Series([filtered_cues])

        # Create a copy of the DataFrame to avoid modifying self.df
        df_copy = self.df.copy()

        # Apply the function to the copied DataFrame
        df_copy[['filtered_sound_cues']] = df_copy.apply(filter_by_loser, axis=1)

        # Drop rows where no filtered_sound_cues remain
        df_filtered = df_copy[df_copy['filtered_sound_cues'].apply(len) > 0]

        return df_filtered

    """*************************COMBINING COHORTS*************************"""
    def remove_specified_subjects(self):
        # List of subject names to remove
        subjects_to_remove = ["n4", "n3", "n2", "n1", 'p4']

        # Remove rows where 'subject_names' are in the list
        df_combined = self.df[~self.df['subject_name'].isin(subjects_to_remove)]

        # Display the result
        print(df_combined)
        
        self.df = df_combined

    """*******************************LICKS********************************"""
    def find_first_lick_after_sound_cue(self, df=None):
        """
        Finds the first port entry occurring after 4 seconds following each sound cue.
        If a port entry starts before 4 seconds but extends past it, 
        the function selects the timestamp at 4 seconds after the sound cue.

        Works with any DataFrame that has the required columns.
        """
        if df is None:
            df = self.df  # Default to self.df only if no DataFrame is provided

        first_licks = []  # List to store results

        for index, row in df.iterrows():  # Use df, not self.df
            sound_cues_onsets = row['filtered_sound_cues']
            port_entries_onsets = row['filtered_port_entries']
            port_entries_offsets = row['filtered_port_entry_offset']

            first_licks_per_row = []

            for sc_onset in sound_cues_onsets:
                threshold_time = sc_onset + 4

                # First, check for an ongoing lick:
                ongoing_licks_indices = np.where(
                    (port_entries_onsets < threshold_time) & (port_entries_offsets >= threshold_time)
                )[0]

                if len(ongoing_licks_indices) > 0:
                    first_licks_per_row.append(threshold_time)
                else:
                    # Otherwise, find the first port entry that starts after the threshold
                    future_licks_indices = np.where(port_entries_onsets >= threshold_time)[0]
                    if len(future_licks_indices) > 0:
                        first_licks_per_row.append(port_entries_onsets[future_licks_indices[0]])
                    else:
                        first_licks_per_row.append(None)

            first_licks.append(first_licks_per_row)

        df["first_lick_after_sound_cue"] = first_licks
        return df

    def compute_closest_port_offset(self, lick_column, offset_column, df=None):
        """
        Computes the closest port entry offsets after each lick time and adds them as a new column in the dataframe.
        
        Parameters:
            lick_column (str): The column name for the lick times (e.g., 'first_lick_after_sound_cue').
            offset_column (str): The column name for the port entry offset times (e.g., 'filtered_port_entry_offset').
            new_column_name (str): The name of the new column to store the results. Default is 'closest_port_entry_offsets'.
        
        Returns:
            pd.DataFrame: Updated DataFrame with the new column of closest port entry offsets.
        """
        if df is None:
            df = self.df 
        def find_closest_port_entries(licks, port_entry_offsets):
            """Finds the closest port entry offsets greater than each lick in 'licks'."""
            closest_offsets = []
            
            for lick in licks:
                # Find the indices where port_entry_offset > lick
                valid_indices = np.where(port_entry_offsets > lick)[0]
                
                if len(valid_indices) == 0:
                    closest_offsets.append(np.nan)  # Append NaN if no valid offset is found
                else:
                    # Get the closest port entry offset (the first valid one in the array)
                    closest_offset = port_entry_offsets[valid_indices[0]]
                    closest_offsets.append(closest_offset)
            
            return closest_offsets

        def compute_lick_metrics(row):
            """Compute the closest port entry offsets for each trial."""
            # Extract first_lick_after_sound_cue and filtered_port_entry_offset
            first_licks = np.array(row[lick_column])
            port_entry_offsets = np.array(row[offset_column])
            
            # Get the closest port entry offsets for each lick
            closest_offsets = find_closest_port_entries(first_licks, port_entry_offsets)
            
            return closest_offsets

        # Apply the function to the DataFrame and create a new column with the results
        df['closest_lick_offset'] = df.apply(compute_lick_metrics, axis=1)

    def compute_duration(self, df=None):
        if df is None:
            df = self.df
        df["duration"] = df.apply(lambda row: np.array(row["closest_lick_offset"]) - np.array(row["first_lick_after_sound_cue"]), axis=1)

    """*********************************CALCULATING DOPAMINE RESPONSE***********************************"""
    def compute_event_induced_DA(self, df=None, pre_time=4, post_time=10):
        """
        Computes event-induced DA responses for tone (sound cue) and lick.

        - Tone Event:
        * Time axis: -pre_time to +post_time around each sound cue onset.
        * Baseline: mean of the DA signal in the 4 seconds before the sound cue.

        - Lick Event:
        * Time axis: 0 to +post_time around the lick onset.
        * Baseline: same as the tone event (i.e., computed from the 4 seconds before the sound cue).

        Note: If multiple cues/licks exist, each is processed. The i-th lick uses
            the i-th sound cue's baseline if available. Otherwise, baseline=0.
        """
        if df is None:
            df = self.df

        # -- 1) Determine global dt from all trials --
        min_dt = np.inf
        for _, row in df.iterrows():
            trial_obj = row['trial']
            timestamps = np.array(trial_obj.timestamps)
            if len(timestamps) > 1:
                dt = np.min(np.diff(timestamps))
                min_dt = min(min_dt, dt)
        if min_dt == np.inf:
            print("No valid timestamps found to determine dt.")
            return

        # Two different time axes:
        common_tone_time_axis = np.arange(-pre_time, post_time, min_dt)   # For the sound cue
        common_lick_time_axis = np.arange(0, post_time, min_dt)           # For the lick

        # Lists to store results
        tone_zscores_all = []
        tone_times_all = []
        lick_zscores_all = []
        lick_times_all = []

        for _, row in df.iterrows():
            trial_obj = row['trial']
            timestamps = np.array(trial_obj.timestamps)
            zscore = np.array(trial_obj.zscore)

            # Sound and lick onsets
            sound_cues = row['filtered_sound_cues']
            lick_cues  = row['first_lick_after_sound_cue']

            # ----- Tone (sound cue) -----
            tone_zscores_this_trial = []
            tone_times_this_trial   = []

            if len(sound_cues) == 0:
                # No sound cues => store NaNs
                tone_zscores_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                tone_times_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
            else:
                for sound_event in sound_cues:
                    # Window for the tone
                    window_start = sound_event - pre_time
                    window_end   = sound_event + post_time

                    mask = (timestamps >= window_start) & (timestamps <= window_end)
                    if not np.any(mask):
                        tone_zscores_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                        tone_times_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                        continue

                    # Time relative to the sound cue
                    rel_time = timestamps[mask] - sound_event
                    signal   = zscore[mask]

                    # Baseline: from (sound_event - pre_time) up to the sound_event
                    baseline_mask = (rel_time < 0)
                    baseline = np.nanmean(signal[baseline_mask]) if np.any(baseline_mask) else 0

                    # Baseline-correct
                    corrected_signal = signal - baseline

                    # Interpolate onto [-pre_time, post_time]
                    interp_signal = np.interp(common_tone_time_axis, rel_time, corrected_signal)

                    tone_zscores_this_trial.append(interp_signal)
                    tone_times_this_trial.append(common_tone_time_axis.copy())

            # ----- Lick -----
            lick_zscores_this_trial = []
            lick_times_this_trial   = []

            if len(lick_cues) == 0:
                # No licks => store NaNs
                lick_zscores_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                lick_times_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
            else:
                for i, lick_event in enumerate(lick_cues):
                    # Attempt to pair with the i-th sound cue for baseline
                    if i < len(sound_cues):
                        sound_event_for_baseline = sound_cues[i]
                    else:
                        sound_event_for_baseline = None  # fallback

                    # 1) Compute baseline from the sound cue window
                    if sound_event_for_baseline is not None:
                        # Baseline window: from (sound_event - pre_time) to sound_event
                        baseline_start = sound_event_for_baseline - pre_time
                        baseline_end   = sound_event_for_baseline
                        baseline_mask = (timestamps >= baseline_start) & (timestamps <= baseline_end)
                        if np.any(baseline_mask):
                            baseline_val = np.nanmean(zscore[baseline_mask])
                        else:
                            baseline_val = 0
                    else:
                        baseline_val = 0

                    # 2) Extract the lick window from lick_event (0 to +post_time)
                    window_start = lick_event
                    window_end   = lick_event + post_time
                    mask = (timestamps >= window_start) & (timestamps <= window_end)
                    if not np.any(mask):
                        lick_zscores_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                        lick_times_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                        continue

                    rel_time = timestamps[mask] - lick_event
                    signal   = zscore[mask]

                    # Subtract the tone-based baseline
                    corrected_signal = signal - baseline_val

                    # Interpolate onto [0, post_time]
                    interp_signal = np.interp(common_lick_time_axis, rel_time, corrected_signal)

                    lick_zscores_this_trial.append(interp_signal)
                    lick_times_this_trial.append(common_lick_time_axis.copy())

            # Store in the main lists
            tone_zscores_all.append(tone_zscores_this_trial)
            tone_times_all.append(tone_times_this_trial)
            lick_zscores_all.append(lick_zscores_this_trial)
            lick_times_all.append(lick_times_this_trial)

        # Save columns
        df['Tone Event_Time_Axis'] = tone_times_all
        df['Tone Event_Zscore']    = tone_zscores_all
        df['Lick Event_Time_Axis'] = lick_times_all
        df['Lick Event_Zscore']    = lick_zscores_all

    def compute_tone_da_metrics(self, df=None, mode='standard'):
        if df is None:
            df = self.df
        def compute_da_metrics_for_trial(trial_obj, filtered_sound_cues):
            """Compute DA metrics (AUC, Max Peak, Time of Max Peak, Mean Z-score) for each sound cue, using adaptive peak-following."""
            """if not hasattr(trial_obj, "timestamps") or not hasattr(trial_obj, "zscore"):
                return np.nan"""  # Handle missing attributes

            timestamps = np.array(trial_obj.timestamps)  
            zscores = np.array(trial_obj.zscore)  

            computed_metrics = []
            for cue in filtered_sound_cues:
                start_time = cue
                end_time = cue + 4  # Default window end

                # Extract initial window
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                window_ts = timestamps[mask]
                window_z = zscores[mask]
                
                if len(window_ts) < 2:
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
                    continue
                    # Skip to next cue

                # Compute initial metrics
                auc = np.trapz(window_z, window_ts)  
                max_idx = np.argmax(window_z)
                max_peak = window_z[max_idx]
                peak_time = window_ts[max_idx]
                mean_z = np.mean(window_z)

                computed_metrics.append({
                    "AUC": auc,
                    "Max Peak": max_peak,
                    "Time of Max Peak": peak_time,
                    "Mean Z-score": mean_z,
                    "Adjusted End": end_time  # Store adjusted end
                })
            return computed_metrics
        def compute_ei(df):
            if 'Tone Event_Time_Axis' not in df.columns or 'Mean Tone Event_Zscore' not in df.columns:
                print("Event-induced data not found in behaviors. Please run compute_event_induced_DA() first.")
                return df

            mean_zscores_all = []
            auc_values_all = []
            max_peaks_all = []
            peak_times_all = []

            for _, row in df.iterrows():
                final_time = np.array(row['Tone Event_Time_Axis'], dtype=float)[0]  # Ensure full array
                mean_event_zscore = np.array(row['Mean Tone Event_Zscore'], dtype=float)  # Convert to NumPy array

                if len(final_time) < len(mean_event_zscore):
                    print(f"Warning: final_time length {len(final_time)} < mean_event_zscore length {len(mean_event_zscore)}. Skipping.")
                    continue  

                final_time = final_time[-len(mean_event_zscore):]  # Trim to match z-score length

                # Apply mask for values between 0 and 4 seconds
                mask = (final_time >= 0) & (final_time <= 4)
                time_masked = final_time[mask]
                zscore_masked = mean_event_zscore[mask]

                if len(time_masked) == 0:
                    print("Warning: No values in the selected time range (0-4s). Skipping.")
                    continue  

                # Compute metrics
                mean_z = np.mean(zscore_masked)
                auc = np.trapz(zscore_masked, time_masked)  
                max_idx = np.argmax(zscore_masked)
                max_peak = zscore_masked[max_idx]
                peak_time = time_masked[max_idx]

                print(f"AUC (0-4s): {auc}")

                mean_zscores_all.append(mean_z)
                auc_values_all.append(auc)
                max_peaks_all.append(max_peak)
                peak_times_all.append(peak_time)

            df['Tone Mean Z-score EI'] = mean_zscores_all
            df['Tone AUC EI'] = auc_values_all
            df['Tone Max Peak EI'] = max_peaks_all
            df['Tone Time of Max Peak EI'] = peak_times_all
            return df

        if mode == 'standard':
            # Apply function across all trials
            df["computed_metrics"] = df.apply(
                lambda row: compute_da_metrics_for_trial(row["trial"], row["filtered_sound_cues"]), axis=1)
            df["Tone AUC"] = df["computed_metrics"].apply(lambda x: [item.get("AUC", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Max Peak"] = df["computed_metrics"].apply(lambda x: [item.get("Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Time of Max Peak"] = df["computed_metrics"].apply(lambda x: [item.get("Time of Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Mean Z-score"] = df["computed_metrics"].apply(lambda x: [item.get("Mean Z-score", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Adjusted End"] = df["computed_metrics"].apply(lambda x: [item.get("Adjusted End", np.nan) for item in x] if isinstance(x, list) else np.nan)
            # Drop the "computed_metrics" column if it's no longer needed
            df.drop(columns=["computed_metrics"], inplace=True)
        else:
            compute_ei(df)
                
    def compute_lick_da_metrics(self, df=None, mode='standard'):
        if df is None:
            df = self.df
        """Iterate through trials in the dataframe and compute DA metrics for each lick trial."""
        
        def compute_da_metrics_for_lick(trial_obj, first_lick_after_tones, closest_port_entry_offsets, 
                                        use_adaptive=False, peak_fall_fraction=0.5, allow_bout_extension=False,
                                        use_fractional=False, max_bout_duration=15):
            """Compute DA metrics (AUC, Max Peak, Time of Max Peak, Mean Z-score) for a single trial, 
            iterating over multiple first_lick_after_tone and closest_port_entry_offset values."""
            
            timestamps = np.array(trial_obj.timestamps)  
            zscores = np.array(trial_obj.zscore)  

            computed_metrics = []
            for first_lick_after_tone, closest_port_entry_offset in zip(first_lick_after_tones, closest_port_entry_offsets):
                # Ensure timestamps array is not empty and start_time is before end_time
                if len(timestamps) == 0 or first_lick_after_tone >= closest_port_entry_offset: 
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
                    continue

                # Extract initial window
                mask = (timestamps >= first_lick_after_tone) & (timestamps <= closest_port_entry_offset)
                
                if mask.sum() == 0:  # If no valid timestamps in the window
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
                    continue
                
                window_ts = timestamps[mask]
                window_z = zscores[mask]

                # Compute initial metrics
                auc = np.trapz(window_z, window_ts)  
                max_idx = np.argmax(window_z)
                max_peak = window_z[max_idx]
                peak_time = window_ts[max_idx]
                mean_z = np.mean(window_z)

                # Adaptive peak-following
                if use_adaptive and max_peak >= 0:
                    threshold = max_peak * peak_fall_fraction
                    fall_idx = max_idx

                    while fall_idx < len(window_z) and window_z[fall_idx] > threshold:
                        fall_idx += 1

                    if fall_idx < len(window_z):
                        closest_port_entry_offset = window_ts[fall_idx]  # Adjust end time based on threshold

                    elif allow_bout_extension:
                        # Extend to the full timestamp range if no fall-off is found
                        extended_mask = (timestamps >= first_lick_after_tone)
                        extended_ts = timestamps[extended_mask]
                        extended_z = zscores[extended_mask]

                        peak_idx_ext = np.argmin(np.abs(extended_ts - peak_time))
                        fall_idx_ext = peak_idx_ext

                        while fall_idx_ext < len(extended_z) and extended_z[fall_idx_ext] > threshold:
                            fall_idx_ext += 1

                        if fall_idx_ext < len(extended_ts):
                            closest_port_entry_offset = extended_ts[fall_idx_ext]
                        else:
                            closest_port_entry_offset = extended_ts[-1]

                # Re-extract window with adjusted end time
                final_mask = (timestamps >= first_lick_after_tone) & (timestamps <= closest_port_entry_offset)
                final_ts = timestamps[final_mask]
                final_z = zscores[final_mask]

                if len(final_ts) < 2:
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
                    continue  # Skip to next pair of values

                # Compute final metrics
                auc = np.trapz(final_z, final_ts)
                final_max_idx = np.argmax(final_z)
                final_max_val = final_z[final_max_idx]
                final_peak_time = final_ts[final_max_idx]
                mean_z = np.mean(final_z)

                computed_metrics.append({
                    "AUC": auc,
                    "Max Peak": final_max_val,
                    "Time of Max Peak": final_peak_time,
                    "Mean Z-score": mean_z,
                    "Adjusted End": closest_port_entry_offset  # Store adjusted end
                })
            return computed_metrics
        def compute_ei(df):
            if 'Lick Event_Time_Axis' not in df.columns or 'Mean Lick Event_Zscore' not in df.columns:
                print("Event-induced data not found in behaviors. Please run compute_event_induced_DA() first.")
                return df

            mean_zscores_all = []
            auc_values_all = []
            max_peaks_all = []
            peak_times_all = []

            for _, row in df.iterrows():
                final_time = np.array(row['Lick Event_Time_Axis'], dtype=float)[0] 
                mean_event_zscore = np.array(row['Mean Lick Event_Zscore'], dtype=float)  # Convert to NumPy array

                final_time = final_time[-len(mean_event_zscore):]  # Trim to match z-score length

                # Apply mask for values between 0 and 4 seconds
                mask = (final_time >= 0) & (final_time <= 4)
                time_masked = final_time[mask]
                
                mean_event_zscore = mean_event_zscore[-len(time_masked):]  
                zscore_masked = mean_event_zscore  # Now it matches time_masked
                # zscore_masked = mean_event_zscore[mask]

                if len(time_masked) == 0:
                    print("Warning: No values in the selected time range (0-4s). Skipping.")
                    continue  

                # Compute metrics
                mean_z = np.mean(zscore_masked)
                auc = np.trapz(zscore_masked, time_masked)  
                max_idx = np.argmax(zscore_masked)
                max_peak = zscore_masked[max_idx]
                peak_time = time_masked[max_idx]

                print(f"AUC (0-4s): {auc}")

                mean_zscores_all.append(mean_z)
                auc_values_all.append(auc)
                max_peaks_all.append(max_peak)
                peak_times_all.append(peak_time)

            # Store computed values as lists inside new DataFrame columns
            df['Lick Mean Z-score EI'] = mean_zscores_all
            df['Lick AUC EI'] = auc_values_all
            df['Lick Max Peak EI'] = max_peaks_all
            df['Lick Time of Max Peak EI'] = peak_times_all
            return df

        if mode == 'standard':
            # Apply the function across all trials
            df["lick_computed_metrics"] = df.apply(
                lambda row: compute_da_metrics_for_lick(row["trial"], row["first_lick_after_sound_cue"], 
                                                        row["closest_lick_offset"], use_adaptive=True, 
                                                        peak_fall_fraction=0.5, allow_bout_extension=False), axis=1)
            # Extract the individual DA metrics into new columns
            df["Lick AUC"] = df["lick_computed_metrics"].apply(lambda x: [item.get("AUC", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Lick Max Peak"] = df["lick_computed_metrics"].apply(lambda x: [item.get("Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Lick Time of Max Peak"] = df["lick_computed_metrics"].apply(lambda x: [item.get("Time of Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Lick Mean Z-score"] = df["lick_computed_metrics"].apply(lambda x: [item.get("Mean Z-score", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Lick Adjusted End"] = df["lick_computed_metrics"].apply(lambda x: [item.get("Adjusted End", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df.drop(columns=["lick_computed_metrics"], inplace=True)
        else:
            df = compute_ei(df)


    def find_overall_mean(self, df):
        """
        Finds the overall mean of specified DA metrics per-subject.
        """
        if df is None:
            df = self.df
        
        # Function to compute mean for numerical values, preserving first for categorical ones
        def mean_arrays(group):
            result = {}
            result['Rank'] = group['Rank'].iloc[0]  # Keeps first value
            result['Cage'] = group['Cage'].iloc[0]  # Keeps first value
            result['Tone Event_Time_Axis'] = group['Tone Event_Time_Axis'].iloc[0]  # Time axis should be the same

            # Compute mean for scalar numerical values
            numerical_cols = [
                'Lick AUC', 'Lick Max Peak', 'Lick Mean Z-score',
                'Tone AUC', 'Tone Max Peak', 'Tone Mean Z-score',
                'Lick AUC First', 'Lick AUC Last', 'Lick Max Peak First', 'Lick Max Peak Last',
                'Lick Mean Z-score First', 'Lick Mean Z-score Last',
                'Tone AUC First', 'Tone AUC Last', 'Tone Max Peak First', 'Tone Max Peak Last',
                'Tone Mean Z-score First', 'Tone Mean Z-score Last',
                "Lick AUC EI", "Lick Max Peak EI", "Lick Mean Z-score EI",
                "Tone AUC EI", "Tone Max Peak EI", "Tone Mean Z-score EI",
                "Lick AUC EI First", "Lick Max Peak EI First", "Tone AUC EI First",
                "Tone Max Peak EI First", "Tone Mean Z-score EI First",
                "Lick AUC EI Last", "Lick Max Peak EI Last", "Tone AUC EI Last",
                "Tone Max Peak EI Last", "Tone Mean Z-score EI Last"
            ]
            for col in numerical_cols:
                group[col] = group[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
                group[col] = pd.to_numeric(group[col], errors='coerce')

            for col in numerical_cols:
                result[col] = group[col].mean()

            # Compute element-wise mean for array columns
            array_cols = ["Tone Event_Zscore", "Lick Event_Zscore"]
            for col in array_cols:
                stacked_arrays = np.vstack(group[col].values)  # Stack into 2D array
                result[col] = np.mean(stacked_arrays, axis=0)  # Element-wise mean

            return pd.Series(result)

        df_mean = df.groupby('subject_name').apply(mean_arrays).reset_index()
        
        return df_mean
    
    def find_mean_event_zscore(self, df = None, behavior = 'Tone'):
        if df is None:
            df = self.df
        if 'Tone Event_Zscore' not in df.columns:
            print("Event-induced Z-score data not found in dataframe.")
            return df

        mean_zscores_all = []

        for _, row in df.iterrows():
            event_zscores = np.array(row['Tone Event_Zscore'])  # 2D array (trials × time points)

            if event_zscores.ndim == 2 and event_zscores.size > 0:
                mean_zscores = np.mean(event_zscores, axis=0)  # Compute mean across trials (row-wise)
            else:
                mean_zscores = np.nan  # Handle empty or invalid cases

            mean_zscores_all.append(mean_zscores)

        df[f'Mean {behavior} Event_Zscore'] = mean_zscores_all  # Store the mean 1D array in a new column
        return df

    """*******************************PLOTING**********************************"""
    def ploting_side_by_side(self, df, df1, mean_values, sem_values, mean_values1, sem_values1, bar_color, figsize, metric_name,
                         ylim, yticks_increment, title, directory_path, pad_inches, label1, label2):    
        print(df)
        print(df1)

        bar_width = 0.25  
        gap = 0.05  
        x = np.arange(len(df.columns))  

        fig, ax = plt.subplots(figsize=figsize)

                # Plot individual subject values for both dataframes
        for i, subject in enumerate(df.index):
            ax.scatter(x - bar_width / 2 - gap / 2, df.loc[subject], facecolors='none', edgecolors='gray', s=120, alpha=0.6, linewidth=4, zorder=2)
        for i, subject in enumerate(df1.index):    
            ax.scatter(x + bar_width / 2 + gap / 2, df1.loc[subject], facecolors='none', edgecolors='gray', s=120, alpha=0.6, linewidth=4, zorder=2)

        # Plot bars for the mean with error bars
        bars1 = ax.bar(
            x - bar_width / 2 - gap / 2,  # Adjust x to leave a gap for Tone
            mean_values, 
            yerr=sem_values, 
            capsize=6, 
            color=bar_color, 
            edgecolor='black', 
            linewidth=4, 
            width=bar_width,
            label=label1,  # Label for legend
            error_kw=dict(elinewidth=4, capthick=4, capsize=10, zorder=5)
        )

        bars2 = ax.bar(
            x + bar_width / 2 + gap / 2,  # Adjust x to leave a gap for Lick
            mean_values1, 
            yerr=sem_values1, 
            capsize=6, 
            color=bar_color,
            edgecolor='black', 
            linewidth=4, 
            width=bar_width,
            label=label2,  # Label for legend
            error_kw=dict(elinewidth=4, capthick=4, capsize=10, zorder=5)
        )

        # Set x-ticks in the center of each grouped pair of bars
        # Define the positions for the x-ticks of both bars
        x_left = x - bar_width / 2 - gap / 2  # Position for Tone
        x_right = x + bar_width / 2 + gap / 2  # Position for Lick

        # Combine the tick positions for both
        combined_x_ticks = np.concatenate([x_left, x_right])
        ax.set_xticks(combined_x_ticks)  # All positions for ticks (Tone and Lick)
        # Set the corresponding labels (alternating "Tone" and "Lick")
        combined_labels = [label1] * len(df.columns) + [label2] * len(df.columns)
        ax.set_xticklabels(combined_labels, fontsize=36)

        # Optionally adjust the alignment of the labels if needed
        ax.tick_params(axis='x', which='major', labelsize=36, direction='out', length=6, width=2)
        ax.tick_params(axis='y', which='major', labelsize=36, direction='out', length=10, width=2)

        # Set x-ticks and labels
        ax.set_ylabel(metric_name + ' Z-scored ΔF/F', fontsize=36, labelpad=12)
        ax.set_xlabel("Event", fontsize=40, labelpad=12)
        ax.tick_params(axis='y', labelsize=32)
        ax.tick_params(axis='x', labelsize=32)
        # Add a dashed gray line at y = 0
        ax.axhline(0, color='gray', linestyle='--', linewidth=2, zorder=1)

        # Automatically adjust y-limits based on the data
        if ylim is None:
            all_values = np.concatenate([df.values.flatten(), df1.values.flatten()])
            min_val = np.nanmin(all_values)
            max_val = np.nanmax(all_values)
            ax.set_ylim(0 if min_val > 0 else min_val * 1.1, max_val * 1.1)
        else:
            ax.set_ylim(ylim)
            if ylim[0] < 0:
                ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)

        # Add y-ticks if specified
        if yticks_increment is not None:
            y_min, y_max = ax.get_ylim()
            ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment))

        # Remove unnecessary spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)

        # Statistical Testing
        p_values = []
        print(df.columns)
        for col in df.columns:
            print(f"Processing column: {col}")
            
            # Check if 'Last' column exists in df1, corresponding to the 'First' column in df
            col_last = col.replace('First', 'Last')  # Create the 'Last' column name
            
            # Check if the 'Last' column is in df1
            if col_last in df1.columns:
                print(f"df: {df[col]}")
                print(f"df1: {df1[col_last]}")
                
                # Run t-test between the 'First' and 'Last' columns
                t_stat, p_value = ttest_ind(df[col], df1[col_last], nan_policy='omit', equal_var=False)
                p_values.append(p_value)
                print(f"T-test for {col} and {col_last}: t={t_stat:.3f}, p={p_value:.3e}")  # Print results
            else:
                print(f"Column {col_last} not found in df1. Skipping t-test for {col}.")

        # Function to convert p-values to asterisks
        def get_p_value_asterisks(p_value):
            if p_value < 0.001:
                return "***"
            elif p_value < 0.01:
                return "**"
            elif p_value < 0.05:
                return "*"
            else:
                return None
        # Get the top y-limit to position the significance bar
        y_top = ax.get_ylim()[1] * 0.95  # Position at 95% of max y-limit
        # Add significance annotations (only for significant p-values)
        for i, p_value in enumerate(p_values):
            p_text = get_p_value_asterisks(p_value)  # Convert p-value to asterisk notation

            if p_text:  # Only add bar if significant
                x1 = x[i] - bar_width / 2 - gap / 2  # Left bar
                x2 = x[i] + bar_width / 2 + gap / 2  # Right bar

                # Draw the black significance bar
                ax.plot([x1, x2], [y_top, y_top], color='black', linewidth=6)

                # Add asterisk annotation above the bar
                ax.text((x1 + x2) / 2, y_top + 0.02, p_text, ha='center', fontsize=40, fontweight='bold')
        # Add title
        plt.title(title, fontsize=40, fontweight='bold', pad=24)

        save_path = os.path.join(str(directory_path) + '\\' + f'{title}.png')
        plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
        plt.show()
        
    # Response is tone or lick, metric_name is for AUC, Max Peak, or Mean Z-score
    def plot_da_first_last(self, df, metric_name, method, directory_path, condition='Winning',  
                    brain_region='mPFC',  # New parameter to specify the brain region
                    custom_xtick_labels=None, 
                    custom_xtick_colors=None, 
                    ylim=None, 
                    nac_color='#15616F',   # Color for NAc
                    mpfc_color='#FFAF00',  # Color for mPFC
                    yticks_increment=1, 
                    figsize=(7,7),  
                    pad_inches=0.1):
        """
        Customizable plotting function that plots a single brain region (NAc or mPFC) for first and last win/loss.
        Can use metrics Max Peak, Mean Z-score, and AUC
        """
        if df is None:
            df = self.df
        # Filtering data frame to only keep specified metrics
        def filter_by_metric(df, metric_name):
            # Filter DataFrame columns based on the 'like' condition
            first_column = df[['subject_name', f'Tone {metric_name}{method} First']]
            last_column = df[['subject_name', f'Tone {metric_name}{method} Last']]

            return first_column, last_column

        # spliting and copying dataframe into two dataframes for each brain region 
        def split_by_subjects(df1, df2):            
            # Filter out the 'subject_name' column and keep only the relevant columns for response and metric_name
            df_n = df1[df1['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p = df1[df1['subject_name'].str.startswith('p')].drop(columns=['subject_name'])
            
            df_n1 = df2[df2['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p1 = df2[df2['subject_name'].str.startswith('p')].drop(columns=['subject_name'])
            # Return filtered dataframes and subject_name column
            return df_n, df_p, df_n1, df_p1

        first_df, last_df = filter_by_metric(df, metric_name)

        # Split data into NAc and mPFC, with subject names
        df_nac_f, df_mpfc_f, df_nac_l, df_mpfc_l = split_by_subjects(first_df, last_df)

        # Select the data for the desired brain region
        if brain_region == 'NAc':
            df = df_nac_f
            df1 = df_nac_l
            title_suffix = 'NAc'
            bar_color = nac_color
        elif brain_region == 'mPFC':
            df = df_mpfc_f
            df1 = df_mpfc_l
            title_suffix = 'mPFC'
            bar_color = mpfc_color
        else:
            raise ValueError("brain_region must be either 'NAc' or 'mPFC'")

        title = metric_name + method + ' ' + condition + f' DA ({title_suffix})'

        # Ensure the dataframe contains only numeric data (in case of any non-numeric columns)
        df = df.apply(pd.to_numeric, errors='coerce')

        label1 = 'First'
        label2 = 'Last'
        # Calculate the mean and SEM values for the entire dataframe
        # First
        mean_values = df.mean()
        sem_values = df.sem()

        # Last
        mean_values1 = df1.mean()
        sem_values1 = df1.sem()

        self.ploting_side_by_side(df, df1, mean_values, sem_values, mean_values1, sem_values1,
                                  bar_color, figsize, metric_name, ylim, 
                                  yticks_increment, title, directory_path, pad_inches,
                                  label1, label2)

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
            df = self.df
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

    def plot_specific_peth(self, df, condition, event_type, directory_path, brain_region, y_min, y_max):
        """
        Plots the PETH of the first and last bouts of either win or loss.
        """
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
        bin_size = 100
        if brain_region == 'mPFC':
            color = '#FFAF00'
        else:
            color = '#15616F'
        # Initialize data structures
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]    
        first_events = []
        last_events = []
        """for i, row in df.iterrows():
            print(f"Row {i}: Length of Z-score array: {len(row[f'{event_type} Event_Zscore'])}")"""

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

        mean_first, downsampled_time_axis = self.downsample_data(mean_first, common_time_axis, bin_size)
        sem_first, _ = self.downsample_data(sem_first, common_time_axis, bin_size)
        
        mean_last, _ = self.downsample_data(mean_last, common_time_axis, bin_size)
        sem_last, _ = self.downsample_data(sem_last, common_time_axis, bin_size)
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        axes[0].set_ylabel('Event Induced Z-scored ΔF/F')
        axes[1].set_ylabel('Event Induced Z-scored ΔF/F')

        # Show y-tick labels on both subplots
        axes[0].tick_params(axis='y', labelleft=True)
        axes[1].tick_params(axis='y', labelleft=True)

        for ax, mean_peth, sem_peth, title in zip(
            axes, [mean_first, mean_last], [sem_first, sem_last], 
            [f'First {condition} bout Z-Score', f'Last {condition} bout Z-Score']
        ):
            global_min = min(np.min(mean_first - sem_first), np.min(mean_last - sem_last))
            global_max = max(np.max(mean_first + sem_first), np.max(mean_last + sem_last))

            # Add a margin for better visualization
            margin = 0.1 * (global_max - global_min)
            ax.plot(downsampled_time_axis, mean_peth, color=color, label='Mean DA')
            ax.fill_between(downsampled_time_axis, mean_peth - sem_peth, mean_peth + sem_peth, color=color, alpha=0.4)
            ax.axvline(0, color='black', linestyle='--')  # Mark event onset

            ax.set_title(title, fontsize=18)
            ax.set_xlabel('Time (s)', fontsize=14)
            ax.set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
            ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=12)

            y_min = np.min(mean_peth - sem_peth)
            y_max = np.max(mean_peth + sem_peth)

            # Add a margin to make sure the mean trace doesn't go out of bounds
            margin = 0.1 * (y_max - y_min)  # You can adjust this factor for a larger/smaller margin

            ax.set_ylim(global_min - margin, global_max + margin)

        save_path = os.path.join(str(directory_path) + '\\' + f'{brain_region}_{condition}_PETH.png')
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        # plt.savefig(f'PETH.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.show()

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
            df = self.df
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

    def drop_unnecessary(self, df=None):
        if df is None:
            df = self.df
        # drops lots of unnecessary column to allow for easier observations
        df.drop(columns=['file name', 'port entries onset', 'port entries offset', 'sound cues',
                              'sound cues', 'port entries', 'winner_array', 'filtered_port_entries'], inplace=True)
        
    def first_last(self, df=None):
        """
        Finds the first and last bouts DA of competition
        """
        if df is None:
            df = self.df
        # Extract the first and last values of the array under 'Lick AUC'
        df['Lick AUC First'] = df['Lick AUC'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Lick AUC Last'] = df['Lick AUC'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Lick Max Peak First'] = df['Lick Max Peak'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Lick Max Peak Last'] = df['Lick Max Peak'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Lick Mean Z-score First'] = df['Lick Mean Z-score'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Lick Mean Z-score Last'] = df['Lick Mean Z-score'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Tone AUC First'] = df['Tone AUC'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Tone AUC Last'] = df['Tone AUC'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Tone Max Peak First'] = df['Tone Max Peak'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Tone Max Peak Last'] = df['Tone Max Peak'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Tone Mean Z-score First'] = df['Tone Mean Z-score'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Tone Mean Z-score Last'] = df['Tone Mean Z-score'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Tone AUC EI First'] = df['Tone AUC EI'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Tone AUC EI Last'] = df['Tone AUC EI'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Tone Max Peak EI First'] = df['Tone Max Peak EI'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Tone Max Peak EI Last'] = df['Tone Max Peak EI'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)
        
        df['Tone Mean Z-score EI First'] = df['Tone Mean Z-score EI'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Tone Mean Z-score EI Last'] = df['Tone Mean Z-score EI'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)  

        df['Lick AUC EI First'] = df['Lick AUC EI'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Lick AUC EI Last'] = df['Lick AUC EI'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        df['Lick Max Peak EI First'] = df['Lick Max Peak EI'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Lick Max Peak EI Last'] = df['Lick Max Peak EI'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)
        
        df['Lick Mean Z-score EI First'] = df['Lick Mean Z-score EI'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['Lick Mean Z-score EI Last'] = df['Lick Mean Z-score EI'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)     
    
    """*************************FIRST TONE STUFF*****************************"""
    def keep_first_tone(self, df=None):
        if df is None:
            df = self.df
        df['first_value'] = df['winner_array'].apply(lambda x: x[0] if len(x) > 0 else None)
        df['first_tone'] = df['sound cues onset'].apply(lambda x: x[0] if len(x) > 0 else None).apply(lambda x: np.array([x]))
        df['first_bout'] = df.apply(lambda row: 'win' if row['first_value'] == row['subject_name'] else 'lose', axis=1)

    def keep_last_tone(self, df=None):
        if df is None:
            df = self.df
        df['last_value'] = df['winner_array'].apply(lambda x: x[-1] if len(x) > 0 else None)
        df['last_tone'] = df['sound cues onset'].apply(lambda x: x[-1] if len(x) > 0 else None).apply(lambda x: np.array([x]))
        df['last_bout'] = df.apply(lambda row: 'win' if row['first_value'] == row['subject_name'] else 'lose', axis=1)


    def finding_first_tone_means(self, df=None):
        """
        Finds the mean values of Global and EI DA for both lick and tone and grouping values together by subject. 
        """
        if df is None:
            df = self.df
        df = df.groupby(['subject_name'], as_index=False).agg({
            'Lick AUC Mean': 'mean',
            'Lick Max Peak Mean': 'mean',
            'Lick Mean Z-score Mean': 'mean',
            'Tone AUC Mean': 'mean',
            'Tone Max Peak Mean': 'mean',
            'Tone Mean Z-score Mean': 'mean',
            "Lick AUC Mean EI": 'mean',
            "Lick Max Peak Mean EI": 'mean',
            "Lick Mean Z-score Mean EI": 'mean',
            "Tone AUC Mean EI": 'mean',
            "Tone Max Peak Mean EI": 'mean',
            "Tone Mean Z-score Mean EI": 'mean',
        })
        final_df = df
        return final_df
    
    
    def plot_specific_event_psth(self, event_type, event_index, directory_path, brain_region, y_min, y_max, df=None, condition='Win', bin_size=100):
        """
        Plots the PSTH (mean and SEM) for a specific event bout (0→4 s after event onset)
        across trials, using the same averaging logic as for a bout response.

        Parameters:
            event_type (str): The event type (e.g. 'Tone' or 'Lick').
            event_index (int): 1-indexed event number to plot.
            directory_path (str or None): Directory to save the plot (if None, the plot is not saved).
            brain_region (str): Brain region ('mPFC' or other) to filter subjects.
            y_min (float): Lower bound of the y-axis.
            y_max (float): Upper bound of the y-axis.
            df (DataFrame, optional): DataFrame to use (defaults to self.df).
            bin_size (int, optional): Bin size for downsampling.
        """
        if df is None:
            df = self.df

        # Filter subjects by brain region.
        def split_by_subject(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)
        idx = event_index - 1
        print(f"[DEBUG] PSTH: Plotting {event_type} event index {event_index} (0-indexed {idx}) for brain region {brain_region}")

        selected_traces = []
        for i, row in df.iterrows():
            event_z_list = row.get(f'{event_type} Event_Zscore', [])
            if isinstance(event_z_list, list) and len(event_z_list) > idx:
                trace = np.array(event_z_list[idx])
                print(f"[DEBUG] Row {i}, subject {row['subject_name']}: trace shape {trace.shape}, first 5 values: {trace[:5]}")
                selected_traces.append(trace)
        if len(selected_traces) == 0:
            print(f"No trials have an event at index {event_index} for {event_type}.")
            return

        # Use the common time axis from the first trial's bout.
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][idx]
        selected_traces = np.array(selected_traces)

        mean_trace = np.mean(selected_traces, axis=0)
        sem_trace = np.std(selected_traces, axis=0) / np.sqrt(selected_traces.shape[0])
        
        if hasattr(self, 'downsample_data'):
            mean_trace, downsampled_time_axis = self.downsample_data(mean_trace, common_time_axis, bin_size)
            sem_trace, _ = self.downsample_data(sem_trace, common_time_axis, bin_size)
        else:
            downsampled_time_axis = common_time_axis

        # --- Debug: Check values in the 4–10 s window ---
        post_window = (downsampled_time_axis >= 4) & (downsampled_time_axis <= 10)
        post_vals = mean_trace[post_window]
        print("[DEBUG] PSTH: Final mean trace shape:", mean_trace.shape)
        print("[DEBUG] PSTH: Final mean trace first 10 values:", mean_trace[:10])
        print("[DEBUG] PSTH: 4–10 s window values, first 10:", post_vals[:10])
        print("[DEBUG] PSTH: Max in 4–10 s window:", np.max(post_vals))
        # --- End Debug ---

        # Choose trace color based on brain region.
        trace_color = '#FFAF00' if brain_region == 'mPFC' else '#15616F'

        plt.figure(figsize=(10, 6))
        plt.plot(downsampled_time_axis, mean_trace, color=trace_color, lw=3, label='Mean DA')
        plt.fill_between(downsampled_time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=trace_color, alpha=0.4, label='SEM')
        plt.axvline(0, color='black', linestyle='--', lw=2)
        plt.axvline(4, color='#FF69B4', linestyle='-', lw=2)

        # Force the x-axis to be from -4 to 10 seconds.
        plt.xlabel('Time from Tone Onset (s)', fontsize=30)
        plt.ylabel('Event-Induced z-scored ΔF/F', fontsize=30)
        plt.title(f'{event_type} Event {event_index} {condition} PSTH', fontsize=30, pad=30)
        plt.ylim(y_min, y_max)
        plt.xticks([-4, 0, 4, 10], fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim(-4, 10)  # Force x-axis limits

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)

        if directory_path is not None:
            # Ensure the directory exists.
            os.makedirs(directory_path, exist_ok=True)
            save_path = os.path.join(directory_path, f'{brain_region}_{event_type}_Event{event_index}_PSTH.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_mean_psth(self, df, condition, event_type, directory_path, brain_region, y_min, y_max):
        """
        Plots the PETH of all bouts averaged together.
        """
        if df is None:
            df = self.df

        # Splitting either mPFC or NAc subjects
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)
        bin_size = 100
        if brain_region == 'mPFC':
            trace_color = '#FFAF00'
        else:
            trace_color = '#15616F'
        # Initialize common time axis
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]

        row_means = []  # Store the mean PETH per row
        for _, row in df.iterrows():
            z_scores = np.array(row[f'Mean {event_type} Event_Zscore'])  # Shape: (num_trials, num_time_bins)

            if z_scores.shape[0] > 0:  # Ensure there is data
                row_means.append(z_scores)

        # Convert to numpy array and compute final mean and SEM
        row_means = np.array(row_means)  # Shape: (num_subjects, num_time_bins)
        mean_peth = np.mean(row_means, axis=0)  # Mean across subjects
        sem_peth = np.std(row_means, axis=0) / np.sqrt(row_means.shape[0])  # SEM

        # Downsample data
        mean_trace, downsampled_time_axis = self.downsample_data(mean_peth, common_time_axis, bin_size)
        sem_trace, _ = self.downsample_data(sem_peth, common_time_axis, bin_size)

        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(downsampled_time_axis, mean_trace, color=trace_color, lw=3, label='Mean DA')
        plt.fill_between(downsampled_time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=trace_color, alpha=0.4, label='SEM')
        plt.axvline(0, color='black', linestyle='--', lw=2)
        plt.axvline(4, color='#FF69B4', linestyle='-', lw=2)

        # Set x-axis ticks to [-4, 0, 4, 10]
        plt.xlabel('Time from Tone Onset (s)', fontsize=30)
        plt.ylabel('Event-Induced Z-scored ΔF/F', fontsize=30)
        plt.title(f'{event_type} PSTH', fontsize=30, pad=30)
        plt.ylim(y_min, y_max)
        plt.xticks([-4, 0, 4, 10], fontsize=30)
        plt.xlim(min(-4, downsampled_time_axis[0]), max(10, downsampled_time_axis[-1]))
        plt.yticks(fontsize=30)
        # plt.legend(fontsize=30)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)

        if directory_path is not None:
            save_path = os.path.join(directory_path, f'{brain_region}_{event_type}_{condition}_PSTH.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_mean_per_row_heatmaps(self, df, condition, event_type, directory_path, brain_region):
        """
        Plots a heatmap where each row represents the Z-score time series for each subject/session.
        """

        # Function to filter data by brain region
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)

        subject_names = df['subject_name'].tolist()
        
        # Extract time axis (assumes all subjects share the same time axis)
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]

        # Collect Z-score time series for each subject/session
        mean_per_row = []
        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'])  # Now a 1D array
            mean_per_row.append(z_scores)  # No need to average over trials

        # Convert list of time series to 2D NumPy array (num_subjects, num_time_bins)
        mean_per_row = np.vstack(mean_per_row)

        # Downsample each row
        bin_size = 100  
        downsampled_means = []
        
        for row_mean in mean_per_row:
            downsampled_row, new_time_axis = self.downsample_data(row_mean, common_time_axis, bin_size)
            downsampled_means.append(downsampled_row)

        downsampled_means = np.array(downsampled_means)  # (num_subjects, downsampled_time_bins)

        # Normalize color scale
        if brain_region == "mPFC":
            vmin, vmax = -0.3, 2
        else:
            vmin, vmax = -0.2, 6

        # Define colormap
        cmap = 'inferno' if brain_region == "mPFC" else 'viridis'

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot heatmap
        cax = ax.imshow(downsampled_means, aspect='auto', cmap=cmap, origin='upper',
                        extent=[new_time_axis[0], new_time_axis[-1], 0, len(downsampled_means)],
                        vmin=vmin, vmax=vmax)

        # Formatting
        # ax.set_ylabel('Subjects', fontsize=26)
        ax.set_yticks([])
        ax.axvline(0, color='white', linestyle='--', linewidth=2)  # Event onset
        ax.axvline(4, color="pink", linestyle='-', linewidth=2)

        # Set x-axis labels
        ax.set_xlabel('Time (s)', fontsize=26)
        ax.set_xticks([new_time_axis[0], 0, 4, new_time_axis[-1]])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=26)
        num_subjects = len(subject_names)
        # ytick_positions = np.arange(num_subjects) + 0.5  # Shift by 0.5 to center labels

        # ax.set_yticks(ytick_positions)  # Set new tick positions
        # ax.set_yticklabels(subject_names, fontsize=18, rotation=0)

        # Colorbar
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.7)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_ticks(np.arange(np.ceil(vmin), np.floor(vmax) + 1, 1))  # Ensures whole number intervals
        cbar.set_label("Z-score", fontsize=20)

        # Save and show
        if directory_path is not None:
            save_path = os.path.join(directory_path, f'{brain_region}_{condition}_Row_Mean_Heatmap.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        
        plt.show()

    def reward_comp_vs_alone(self, df_alone, df_comp, metric_name, behavior, brain_region, directory_path,
                         custom_xtick_labels=None, 
                         custom_xtick_colors=None, 
                         ylim=None, 
                         nac_color='#15616F',   # Color for NAc
                         mpfc_color='#FFAF00',  # Color for mPFC
                         yticks_increment=1, 
                         figsize=(5, 7),  
                         pad_inches=0.1):
    
        if behavior == 'Lick':
            behavior = 'Lick '
        else:
            behavior = 'Tone '
        
        if brain_region == 'NAc':
            title = f'NAc {behavior}{metric_name} alone vs competition'
            color = nac_color
        else:
            title = f'mPFC {behavior}{metric_name} alone vs competition'
            color = mpfc_color

        def split_by_subject(df1, brain_region):            
            df_n = df1[df1['subject_name'].str.startswith('n')].copy()
            df_p = df1[df1['subject_name'].str.startswith('p')].copy()
            if brain_region == "NAc":
                return df_n
            else:
                return df_p

        def clean_column(df, col):
            if col not in df.columns:
                print(f"Warning: Column {col} not found in dataframe.")
                return df
            def try_flatten(x):
                if isinstance(x, list) and len(x) == 1:
                    return x[0]
                elif isinstance(x, list):
                    return None  # Drop if list with more than one element
                else:
                    return x
            df[col] = df[col].apply(try_flatten)
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()].copy()
            df[col] = df[col].astype(float)
            return df

        df_competition = split_by_subject(df_comp, brain_region)

        col_1 = f'{behavior}{metric_name}'
        col_2 = f'{behavior}{metric_name} EI'

        alone_df = clean_column(df_alone, col_1)
        competition_df = clean_column(df_competition, col_2)

        # Compute stats
        mean_values = alone_df[col_1].mean()
        sem_values = alone_df[col_1].sem()

        mean_values1 = competition_df[col_2].mean()
        sem_values1 = competition_df[col_2].sem()

        df = alone_df[[col_1]]
        df1 = competition_df[[col_2]]

        self.ploting_side_by_side(df, df1, mean_values, sem_values, mean_values1, sem_values1,
                                color, figsize, metric_name, ylim, 
                                yticks_increment, title, directory_path, pad_inches,
                                'Alone', 'Competition')

    def plot_event_index_grid(self,
                              df: pd.DataFrame,
                              event_type: str,
                              event_index: int,
                              brain_region: str,
                              bin_size: int = 100,
                              ncols: int = 4,
                              figsize_per_plot: tuple = (3, 2),
                              directory_path: str = None):
        """
        Plot each subject's PSTH for a specific event_index in its own subplot,
        mark the first lick and the computed max peak, and return a DataFrame of times.
        """
        import math, os, numpy as np, matplotlib.pyplot as plt

        # Helper to filter by region
        def split_by_subject(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region=='mPFC' else df_n

        df_reg = split_by_subject(df, brain_region)
        idx = event_index - 1

        traces      = []
        labels      = []
        onset_times = []
        peak_rows   = []

        # Collect traces, labels and the underlying metadata
        for _, row in df_reg.iterrows():
            tz = row.get(f'{event_type} Event_Zscore', [])
            ta = row.get(f'{event_type} Event_Time_Axis', [])
            if isinstance(tz, (list, np.ndarray)) and len(tz) > idx:
                trace       = np.array(tz[idx])
                time_axis   = np.array(ta[idx])
                onset       = row['filtered_sound_cues'][idx] \
                              if event_type=='Tone' else row['first_lick_after_sound_cue'][idx]
                peak_abs    = row[f'{event_type} Time of Max Peak'][idx]
                first_lick  = row['first_lick_after_sound_cue'][idx]

                # convert to relative
                peak_rel    = peak_abs - onset
                lick_rel    = first_lick - onset

                traces.append(trace)
                labels.append(row['subject_name'])
                onset_times.append(onset)

                peak_rows.append({
                    'subject_name':  row['subject_name'],
                    'event_type':    event_type,
                    'event_index':   event_index,
                    'brain_region':  brain_region,
                    'first_lick_s':  lick_rel,
                    'peak_time_s':   peak_rel
                })

        if not traces:
            print(f"No data for {event_type} event #{event_index} in {brain_region}")
            return pd.DataFrame()

        # grid layout
        N     = len(traces)
        nrows = math.ceil(N / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(figsize_per_plot[0]*ncols,
                                          figsize_per_plot[1]*nrows),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        color = '#FFAF00' if brain_region=='mPFC' else '#15616F'
        for i, (trace, lbl, onset, peak_info) in enumerate(zip(traces, labels, onset_times, peak_rows)):
            ds_trace, ds_time = self.downsample_data(trace, time_axis, bin_size)

            # find nearest indices for our relative times
            def _closest(t):
                return ds_time[np.abs(ds_time - t).argmin()]

            lick_t = _closest(peak_info['first_lick_s'])
            peak_t = _closest(peak_info['peak_time_s'])

            ax = axes[i]
            ax.plot(ds_time, ds_trace, color=color, lw=1.5)
            ax.axvline(0,                  color='k',      ls='--', lw=1)
            ax.axvline(4,                  color='#FF69B4',ls='-',  lw=1)
            ax.axvline(lick_t,             color='cyan',   ls='-.', lw=1, label='1st lick')
            ax.axvline(peak_t,             color='gray',   ls=':',  lw=1, label='max peak')

            ax.set_title(lbl, fontsize=8)
            ax.set_xlim(ds_time[0], ds_time[-1])
            ax.tick_params(labelsize=6)
            if i % ncols == 0:
                ax.set_ylabel('z ΔF/F', fontsize=6)
            if i // ncols == nrows - 1:
                ax.set_xlabel('Time (s)', fontsize=6)

        # turn off any extra subplots
        for j in range(N, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"{event_type} event #{event_index} PSTH ({brain_region})", fontsize=10)
        plt.tight_layout(rect=[0,0,1,0.96])

        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            out = os.path.join(directory_path,
                               f'{brain_region}_{event_type}_evt{event_index}_grid.png')
            plt.savefig(out, dpi=300, bbox_inches='tight')

        plt.show()

        # return DataFrame of all the peak & lick times
        return pd.DataFrame(peak_rows)

    def plot_tone_and_lick_peaks_with_first_lick(self,
                                                df: pd.DataFrame,
                                                event_index: int,
                                                brain_region: str,
                                                bin_size: int = 100,
                                                ncols: int = 4,
                                                figsize_per_plot: tuple = (3, 2),
                                                directory_path: str = None) -> pd.DataFrame:
        """
        Plots each subject's PSTH for a specific tone event (event_index), and marks:
        • the tone-peak in the 0–4 s window
        • the lick-peak in the 4–10 s window
        • the first lick occurrence after 4 s (relative to cue)
        Returns a DataFrame with columns:
        ['video_name','subject_name','event_index','brain_region',
        'tone_abs_time_s','tone_peak_time_s','tone_peak_amp',
        'lick_peak_time_s','lick_peak_amp',
        'first_lick_time_s']
        """
        import math, os, numpy as np, matplotlib.pyplot as plt

        # 1) Filter by brain region
        def _split(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df_reg = _split(df, brain_region)
        idx    = event_index - 1
        rows   = []
        traces = []

        # 2) Collect trace, time, subject, first-lick, video, and absolute cue time
        for _, row in df_reg.iterrows():
            tone_z     = row.get('Tone Event_Zscore', [])
            tone_t     = row.get('Tone Event_Time_Axis', [])
            first_lick = row.get('first_lick_after_sound_cue', [])
            cues       = row.get('filtered_sound_cues', [])
            video      = row.get('file name', None)
            subj       = row.get('subject_name', None)
            
            # must have this event index everywhere
            if (not isinstance(tone_z, (list, np.ndarray)) or len(tone_z) <= idx
                or len(first_lick) <= idx or len(cues) <= idx):
                continue

            trace    = np.array(tone_z[idx])
            t_axis   = np.array(tone_t[idx])
            cue_abs  = cues[idx]
            lick_abs = first_lick[idx]
            first_rel= lick_abs - cue_abs   # relative time for first lick

            traces.append((trace, t_axis, subj, cue_abs, first_rel, video))

        # 3) Plot grid
        N     = len(traces)
        nrows = math.ceil(N / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            sharex=True, sharey=True
        )
        axes = axes.flatten()
        color = '#FFAF00' if brain_region == 'mPFC' else '#15616F'

        for i, (trace, t_axis, subj, cue_abs, fl_rel, video) in enumerate(traces):
            # Downsample
            ds, dt = self.downsample_data(trace, t_axis, bin_size)

            # Tone peak (0–4s)
            tone_mask = (dt >= 0) & (dt <= 4)
            if tone_mask.any():
                seg = ds[tone_mask]; times = dt[tone_mask]
                t_idx = np.nanargmax(seg)
                t_time, t_amp = times[t_idx], seg[t_idx]
            else:
                t_time = t_amp = np.nan

            # Lick peak (4–10s)
            lick_mask = (dt >= 4) & (dt <= 10)
            if lick_mask.any():
                seg = ds[lick_mask]; times = dt[lick_mask]
                l_idx = np.nanargmax(seg)
                l_time, l_amp = times[l_idx], seg[l_idx]
            else:
                l_time = l_amp = np.nan

            # Record row
            rows.append({
                'video_name':        video,
                'subject_name':      subj,
                'event_index':       event_index,
                'brain_region':      brain_region,
                'tone_abs_time_s':   cue_abs,
                'tone_peak_time_s':  t_time,
                'tone_peak_amp':     t_amp,
                'lick_peak_time_s':  l_time,
                'lick_peak_amp':     l_amp,
                'first_lick_time_s': fl_rel
            })

            # Plot
            ax = axes[i]
            ax.plot(dt, ds, color=color, lw=1.5)

            # Event onset and 4 s marks
            ax.axvline(0, color='k',    ls='--', lw=1)
            ax.axvline(4, color='#F06', ls='-',  lw=1)

            # Tone peak (purple)
            if not np.isnan(t_time):
                ax.axvline(t_time, color='purple', ls=':',  lw=1)

            # Lick peak (purple)
            if not np.isnan(l_time):
                ax.axvline(l_time, color='purple', ls='-.', lw=1)

            # First lick (green dashed)
            ax.axvline(fl_rel, color='green', ls='--', lw=1)

            ax.set_title(subj, fontsize=8)
            ax.set_xlim(dt[0], dt[-1])
            ax.tick_params(labelsize=6)
            if i % ncols == 0:
                ax.set_ylabel('z ΔF/F', fontsize=6)
            if i // ncols == nrows - 1:
                ax.set_xlabel('Time (s)', fontsize=6)

        # Hide any unused axes
        for j in range(N, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Tone event #{event_index} PSTH ({brain_region})", fontsize=10)
        plt.tight_layout(rect=[0,0,1,0.96])

        # Save
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fig.savefig(
                os.path.join(directory_path,
                            f'{brain_region}_Tone_evt{event_index}_grid.png'),
                dpi=300, bbox_inches='tight'
            )
        plt.show()

        # Return peaks DataFrame
        return pd.DataFrame(rows)
