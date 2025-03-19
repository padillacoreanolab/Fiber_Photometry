import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment
from scipy.stats import pearsonr
from trial_class import Trial
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind

class Reward_Competition(Experiment):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)
        # self.trials = {}  # Reset trials to avoid loading from parent class
        self.df = pd.DataFrame()
        if "Cohort_1_2" in experiment_folder_path:
            self.load_rtc1_trials()  # Load 1 RTC trial
        else:
            self.load_rtc2_trials() # load 2 trials for cohort 3

    def load_rtc1_trials(self):
        # Loads each trial folder (block) as a TDTData object and extracts manual annotation behaviors.
        
        trial_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                        if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            trial_obj = Trial(trial_path, '_465A', '_405A')

            self.trials[trial_folder] = trial_obj

    def load_rtc2_trials(self):
        """
        Load two streams of data and splits them into two trials.
        """
        trial_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            trial_obj1 = Trial(trial_path, '_465A', '_405A')
            trial_obj2 = Trial(trial_path, '_465C', '_405C')
            trial_name1 = trial_folder.split('_')[0]                # First mouse of rtc
            trial_name2 = trial_folder.split('_')[1].split('-')[0]  # Second mouse of rtc
            trial_folder1 = trial_name1 + '-' + '-'.join(trial_folder.split('_')[1].split('-')[1:])
            trial_folder2 = trial_name2 + '-' + '-'.join(trial_folder.split('_')[1].split('-')[1:])
            self.trials[trial_folder1] = trial_obj1
            self.trials[trial_folder2] = trial_obj2

    def rc_processing(self, time_segments_to_remove=None):
        """
        Batch processes reward training
        """
        print(self.trials.items())
        for trial_folder, trial in self.trials.items():
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Reward Training Processing {trial_folder}...")
            trial.remove_initial_LED_artifact(t=30)
            trial.remove_final_data_segment(t = 10)
            
            trial.highpass_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            baseline_start, baseline_end = trial.find_baseline_period()  
            # trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            trial.compute_zscore(method = 'standard')
            # trial.compute_zscore(method = 'modified')
            trial.verify_signal()

            # PC0 = Tones
            # PC3 = Box 3
            # PC2 = Box 4
            """
            Using RIG DATA
            """
            trial.behaviors1['sound cues'] = trial.behaviors1.pop('PC0_')
            trial.behaviors1['port entries'] = trial.behaviors1.pop('PC2_', trial.behaviors1.pop('PC3_'))

            # Remove the first entry because it doesn't count
            trial.behaviors1['sound cues'].onset_times = trial.behaviors1['sound cues'].onset[1:]
            trial.behaviors1['sound cues'].offset_times = trial.behaviors1['sound cues'].offset[1:]
            trial.behaviors1['port entries'].onset_times = trial.behaviors1['port entries'].onset[1:]
            trial.behaviors1['port entries'].offset_times = trial.behaviors1['port entries'].offset[1:]

            
            # Finding instances after first tone is played
            port_entries_onset = np.array(trial.behaviors1['port entries'].onset_times)
            port_entries_offset = np.array(trial.behaviors1['port entries'].offset_times)
            first_sound_cue_onset = trial.behaviors1['sound cues'].onset_times[0]
            indices = np.where(port_entries_onset >= first_sound_cue_onset)[0]
            trial.behaviors1['port entries'].onset_times = port_entries_onset[indices].tolist()
            trial.behaviors1['port entries'].offset_times = port_entries_offset[indices].tolist()

            self.combine_consecutive_behaviors1(behavior_name='all', bout_time_threshold=0.5)
            """
            Using csv_data
            """


    """*********************************COMBINE CONSECUTIVE BEHAVIORS***********************************"""
    def combine_consecutive_behaviors1(self, behavior_name='all', bout_time_threshold=0.5, min_occurrences=1):
        """
        Applies the behavior combination logic to all trials within the experiment.
        """

        for trial_name, trial_obj in self.trials.items():
            # Ensure the trial has behaviors1 attribute
            if not hasattr(trial_obj, 'behaviors1'):
                continue  # Skip if behaviors1 is not available

            # Determine which behaviors to process
            if behavior_name == 'all':
                behaviors_to_process = trial_obj.behaviors1.keys()  # Process all behaviors
            else:
                behaviors_to_process = [behavior_name]  # Process a single behavior

            for behavior_event in behaviors_to_process:
                behavior_onsets = np.array(trial_obj.behaviors1[behavior_event].onset)
                behavior_offsets = np.array(trial_obj.behaviors1[behavior_event].offset)

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
                trial_obj.behaviors1[behavior_event].onset = [combined_onsets[i] for i in valid_indices]
                trial_obj.behaviors1[behavior_event].offset = [combined_offsets[i] for i in valid_indices]
                trial_obj.behaviors1[behavior_event].Total_Duration = [combined_durations[i] for i in valid_indices]  # Update Total Duration

                trial_obj.bout_dict = {}  # Reset bout dictionary after processing

    """*************************READING CSV AND STORING AS DF**************************"""
    def read_manual_scoring1(self, csv_file_path):
        """
        Reads in and creates a dataframe to store wins and loses and tangles
        """
        df = pd.read_excel(csv_file_path)
        df.columns = df.columns.str.strip().str.lower()
        filtered_columns = df.filter(like='winner').dropna(axis=1, how='all').columns.tolist()
        col_to_keep = ['file name'] + filtered_columns + ['tangles']
        df = df[col_to_keep]
        self.df = df

    def read_manual_scoring2(self, csv_file_path):
        """
        Reads in and creates a df for trials that recorded 2 mice simultaneously
        
        First change file names so that there is only one mouse before the date, using the subject column.
        """
        df = pd.read_excel(csv_file_path)
        df.columns = df.columns.str.strip().str.lower()
        filtered_columns = df.filter(like='winner').dropna(axis=1, how='all').columns.tolist()
        col_to_keep = ['file name'] + ['subject'] + filtered_columns + ['tangles']
        df = df[col_to_keep]
        self.df = df

    def find_matching_trial(self, file_name):
        return self.trials.get(file_name, None)  # Return the trial if exact match, else None

    def merge_data1(self):
        """
        Merges all data into a dataframe for analysis.
        """
        self.df['trial'] = self.df.apply(lambda row: self.find_matching_trial(row['file name']), axis=1)

        # Debugging: Check how many trials fail to match
        print("Total rows:", len(self.df))
        print("Rows with missing trials:", self.df['trial'].isna().sum())

        # Replace None values temporarily instead of dropping them
        self.df['trial'] = self.df['trial'].fillna("No Match Found")

        # Handle behaviors safely
        self.df['sound cues'] = self.df['trial'].apply(
            lambda x: x.behaviors1.get('sound cues', None) 
            if isinstance(x, object) and hasattr(x, 'behaviors1') else None
        )
        self.df['port entries'] = self.df['trial'].apply(
            lambda x: x.behaviors1.get('port entries', None) 
            if isinstance(x, object) and hasattr(x, 'behaviors1') else None
        )
        self.df['sound cues onset'] = self.df['sound cues'].apply(lambda x: x.onset_times if x else None)
        self.df['port entries onset'] = self.df['port entries'].apply(lambda x: x.onset_times if x else None)
        self.df['port entries offset'] = self.df['port entries'].apply(lambda x: x.offset_times if x else None)

        # Creates a column for subject name
        self.df['subject_name'] = self.df['file name'].str.split('-').str[0]

        # Create a new column that stores an array of winners for each row
        winner_columns = [col for col in self.df.columns if 'winner' in col.lower()]
        self.df['winner_array'] = self.df[winner_columns].apply(lambda row: row.values.tolist(), axis=1)

        # Drops all other winner columns leaving only winner_array.
        self.df.drop(columns=winner_columns, inplace=True)

        # Drop rows without dopamine data only at the end
        self.df = self.df[self.df['trial'] != "No Match Found"]
        self.df.reset_index(drop=True, inplace=True)


    def merge_data2(self):
        """
        Merges data from cohort 3 into a dataframe for analysis
        """
        # Changes file name to match the change in load_rtc2
        self.df['file name'] = self.df.apply(lambda row: row['subject'] + '-' + '-'.join(row['file name'].split('-')[-2:]), axis=1)
        self.df['trial'] = self.df.apply(lambda row: self.find_matching_trial(row['file name']), axis=1)
        
        # Debugging: Check how many trials fail to match
        print("Total rows:", len(self.df))
        print("Rows with missing trials:", self.df['trial'].isna().sum())

        self.df['sound cues'] = self.df['trial'].apply(lambda x: x.behaviors1.get('sound cues', None) if isinstance(x, object) and hasattr(x, 'behaviors1') else None)
        self.df['port entries'] = self.df['trial'].apply(lambda x: x.behaviors1.get('port entries', None) if isinstance(x, object) and hasattr(x, 'behaviors1') else None)
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
        # Extract 'tangles'
        self.df['tangles_array'] = self.df['trial'].apply(
            lambda x: x.behaviors1.get('tangles', []) if isinstance(x, object) and hasattr(x, 'behaviors1') else []
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
        self.df['filtered_port_entries'] = self.df.apply(
            lambda row: remove_indices_from_array(row['port entries onset'], row['tangles_array']), axis=1
        )
        self.df['filtered_port_entry_offset'] = self.df.apply(
            lambda row: remove_indices_from_array(row['port entries offset'], row['tangles_array']), axis=1
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
    def combining_cohorts(self, df1):
        filter_string = "p"

        # Filter df2 to only include rows where 'subject name' contains the filter_string
        filtered_df2 = df1[df1['subject_name'].str.contains(filter_string, na=False)]

        # Concatenate df1 and the filtered df2
        final_df = pd.concat([self.df, filtered_df2], ignore_index=True)
        self.df = final_df


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

                future_licks_indices = np.where(port_entries_onsets >= threshold_time)[0]

                if len(future_licks_indices) > 0:
                    first_licks_per_row.append(port_entries_onsets[future_licks_indices[0]])
                else:
                    ongoing_licks_indices = np.where((port_entries_onsets < threshold_time) & (port_entries_offsets > threshold_time))[0]

                    if len(ongoing_licks_indices) > 0:
                        first_licks_per_row.append(threshold_time)
                    else:
                        first_licks_per_row.append(None)

            first_licks.append(first_licks_per_row)

        df["first_lick_after_sound_cue"] = first_licks  # Add to the given DataFrame

        return df  # Return the modified DataFrame

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
    """*********************************CALCULATING DOPAMINE RESPONSE***********************************"""
    def compute_ei(self, df=None, pre_time=4, post_time=6):
        """
        Computes the peri-event z-scored DA signal for each event in self.behaviors.
        Each event's DA signal is baseline-corrected by subtracting the mean z-score
        from the pre-event period. Two new columns are added to self.behaviors:
        - 'Event_Time_Axis': A common time axis (relative to event onset).
        - 'Event_Zscore': The baseline-corrected z-score signal, interpolated onto the common time axis.
        
        This version uses np.interp without left/right arguments, so any time points outside the 
        real data range are clamped to the first/last available value.
        
        Parameters:
        - pre_time (float): Seconds to include before event onset.
        - post_time (float): Seconds to include after event onset.
        """
        if df is None:
            df = self.df
        if df is None or df.empty:
            print(f"Trial {trial_obj.subject_name}: No behavior events available to compute event-induced DA.")
            return

        # Calculate a common time axis based on the average sampling interval.
        dt = np.mean(np.diff(self.timestamps))
        common_time_axis = np.arange(-pre_time, post_time, dt)

        # Lists to store the common time axis and the interpolated z-score signal for each event.
        event_time_list = []
        event_zscore_list = []

        # Process each event in the behaviors DataFrame.
        for idx, row in self.behaviors.iterrows():
            event_start = row['Event_Start']
            window_start = event_start - pre_time
            window_end = event_start + post_time

            # Identify indices within the peri-event window.
            mask = (self.timestamps >= window_start) & (self.timestamps <= window_end)
            if not np.any(mask):
                # If no data is available, fill with NaNs.
                event_time_list.append(np.full(common_time_axis.shape, np.nan))
                event_zscore_list.append(np.full(common_time_axis.shape, np.nan))
                continue

            # Create a time axis relative to the event onset.
            rel_time = self.timestamps[mask] - event_start
            signal = self.zscore[mask]

            # Compute baseline using the pre-event portion.
            pre_mask = rel_time < 0
            baseline = np.nanmean(signal[pre_mask]) if np.any(pre_mask) else 0

            # Baseline-correct the signal.
            corrected_signal = signal - baseline

            # Interpolate the corrected signal onto the common time axis.
            # Removing left/right arguments clamps values to the boundaries.
            interp_signal = np.interp(common_time_axis, rel_time, corrected_signal)

            event_time_list.append(common_time_axis)
            event_zscore_list.append(interp_signal)

        # Save the computed arrays as new columns in the behaviors DataFrame.
        self.behaviors['Event_Time_Axis'] = event_time_list
        self.behaviors['Event_Zscore'] = event_zscore_list


    def compute_tone_da(self, df=None):
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
                end_time = cue + 6  # Default window end

                # Extract initial window
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                window_ts = timestamps[mask]
                window_z = zscores[mask]
                
                if len(window_ts) < 2:
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
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
                
    def compute_lick_da(self, df=None):
        if df is None:
            df = self.df
        """Iterate through trials in the dataframe and compute DA metrics for each lick trial."""
        
        def compute_da_metrics_for_lick(trial_obj, first_lick_after_tones, closest_port_entry_offsets, 
                                        use_adaptive=True, peak_fall_fraction=0.5, allow_bout_extension=False):
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
        print([item.get("AUC", np.nan) for item in df["lick_computed_metrics"].iloc[0]])

        print(df["lick_computed_metrics"])
        # Drop the "lick_computed_metrics" column if it's no longer needed
        df.drop(columns=["lick_computed_metrics"], inplace=True)

    def find_means(self, df):
        if df is None:
            df = self.df
        df["Lick AUC Mean"] = df["Lick AUC"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Lick Max Peak Mean"] = df["Lick Max Peak"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Lick Mean Z-score Mean"] = df["Lick Mean Z-score"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone AUC Mean"] = df["Tone AUC"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone Max Peak Mean"] = df["Tone Max Peak"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone Mean Z-score Mean"] = df["Tone Mean Z-score"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)

    def find_overall_mean(self, df):
        if df is None:
            df = self.df
        df = df.groupby(['subject_name'], as_index=False).agg({
            'Rank': 'first',  # Keeps the Rank column
            'Cage': 'first',
            'Lick AUC Mean': 'mean',
            'Lick Max Peak Mean': 'mean',
            'Lick Mean Z-score Mean': 'mean',
            'Tone AUC Mean': 'mean',
            'Tone Max Peak Mean': 'mean',
            'Tone Mean Z-score Mean': 'mean',
            'Lick AUC First': 'mean',
            'Lick AUC Last': 'mean',
            'Lick Max Peak First': 'mean',
            'Lick Max Peak Last': 'mean',
            'Lick Mean Z-score First': 'mean',
            'Lick Mean Z-score Last': 'mean',
            'Tone AUC First': 'mean',
            'Tone AUC Last': 'mean',
            'Tone Max Peak First': 'mean',
            'Tone Max Peak Last': 'mean',
            'Tone Mean Z-score First': 'mean',
            'Tone Mean Z-score Last': 'mean',
        })
        final_df = df
        return final_df

    """*******************************PLOTTING**********************************"""
    def ploting_side_by_side(self, df, df1, mean_values, sem_values, mean_values1, sem_values1, bar_color, figsize, metric_name,
                        ylim, yticks_increment, title, directory_path, pad_inches, label1, label2):    
        print(df)
        print(df1)
        # Define bar width for side-by-side bars with gap between them
        bar_width = 0.35  # Width of each bar
        gap = 0.075  # Adjust this value to control the gap between the bars

        # Calculate the x positions for both Tone and Lick, leaving a gap between the bars
        x = np.arange(len(df.columns))  # Positions for the x-ticks

        # Create the plot
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
        y_top = ax.get_ylim()[1]  # Position at 95% of max y-limit

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

        # Set x-ticks in the center of each grouped pair of bars
        # Define the positions for the x-ticks of both bars
        x_left = x - bar_width / 2 - gap / 2  # Position for Tone
        x_right = x + bar_width / 2 + gap / 2  # Position for Lick

        # Combine the tick positions for both
        combined_x_ticks = np.concatenate([x_left, x_right])

        # Set the tick positions
        ax.set_xticks(combined_x_ticks)  # All positions for ticks (Tone and Lick)

        # Set the corresponding labels (alternating "Tone" and "Lick")
        combined_labels = [label1] * len(df.columns) + [label2] * len(df.columns)
        ax.set_xticklabels(combined_labels, fontsize=36)

        # Optionally adjust the alignment of the labels if needed
        ax.tick_params(axis='x', which='major', labelsize=36, direction='out', length=6, width=2)
        
        # Adjust y-tick marks to extend further
        ax.tick_params(axis='y', which='major', labelsize=36, direction='out', length=10, width=2)

        # Increase font sizes
        ax.set_ylabel(metric_name, fontsize=36, labelpad=12)
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

        # Add title
        plt.title(title, fontsize=40, fontweight='bold', pad=24)

        save_path = os.path.join(str(directory_path) + '\\' + f'{title}.png')
        plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
        plt.show()

    # Response is tone or lick, metric_name is for AUC, Max Peak, or Mean Z-score
    def plot_da_first_last(self, df, metric_name, directory_path, condition='Winning',  
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
        Customizable plotting function that plots a single brain region (NAc or mPFC) for both lick and tone.
        Can use metrics Max Peak, Mean Z-score, and AUC
        """
        if df is None:
            df = self.df
        # Filtering data frame to only keep specified metrics
        def filter_by_metric(df, metric_name):
            print(df.columns)
            # Filter DataFrame columns based on the 'like' condition
            first_column = df[['subject_name', f'Tone {metric_name} First']]
            last_column = df[['subject_name', f'Tone {metric_name} Last']]

            return first_column, last_column

        # spliting and copying dataframe into two dataframes for each brain region 
        def split_by_subject(df1, df2):            
            # Filter out the 'subject_name' column and keep only the relevant columns for response and metric_name
            df_n = df1[df1['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p = df1[df1['subject_name'].str.startswith('p')].drop(columns=['subject_name'])
            
            df_n1 = df2[df2['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p1 = df2[df2['subject_name'].str.startswith('p')].drop(columns=['subject_name'])
            # Return filtered dataframes and subject_name column
            return df_n, df_p, df_n1, df_p1

        first_df, last_df = filter_by_metric(df, metric_name)

        # Split data into NAc and mPFC, with subject names
        df_nac_f, df_mpfc_f, df_nac_l, df_mpfc_l = split_by_subject(first_df, last_df)

        print(df_nac_f.columns)
        print(df_nac_l.columns)

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

        title = metric_name + ' ' + condition + f' DA ({title_suffix})'

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

        def filter_by_metric(df):
            metric_columns = df.filter(like=metric_name + method).columns
            print(metric_columns)
            if len(metric_columns) != 2:
                raise ValueError(f"Expected exactly 2 columns, but found {len(metric_columns)}.")

            # Create two DataFrames, keeping 'subject_name'
            df1 = df[['subject_name', metric_columns[0]]].copy()
            df2 = df[['subject_name', metric_columns[1]]].copy()

            return df1, df2
        
        # spliting and copying dataframe into two dataframes for each brain region 
        def split_by_subject(df1):            
            # Filter out the 'subject_name' column and keep only the relevant columns for response and metric_name
            df_n = df1[df1['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p = df1[df1['subject_name'].str.startswith('p')].drop(columns=['subject_name'])
            
            """df_n1 = df2[df2['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p1 = df2[df2['subject_name'].str.startswith('p')].drop(columns=['subject_name'])"""
            # Return filtered dataframes and subject_name column
            return df_n, df_p

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
    def peth_graphs(self):
        pass

    def heatmaps(self):
        pass    

    def scatter_dominance(self, directory_path, df, metric_name, condition, pad_inches=0.1):
        """
        Scatter plot of dominance rank in a cage.
        """
        def filter_by_metric(df, metric_name):
            metric_columns = df.columns[df.columns == f'Tone {metric_name} Mean']
            print(metric_columns)
            
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
        print(df_sorted_n)
        print(df_sorted_p)

        def scatter_plot(directory_path, df_sorted, metric_value, brain_region):
            if brain_region == "mPFC":
                color = '#FFAF00'
            else:
                color = '#15616F'

            # Drop rows where 'Rank' is NaN
            df_sorted = df_sorted.dropna(subset=['Rank'])
            x = df_sorted['Rank']
            y = df_sorted[f'Tone {metric_value} Mean']
            if len(x) > 1:  # Pearson requires at least 2 points
                r_value, p_value = pearsonr(x, y)
            else:
                r_value, p_value = float('nan'), float('nan')  # Handle cases with insufficient data
            
            n_value = len(df_sorted)

            # Create the scatter plot with a regression line
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_sorted, x='Rank', y=f'Tone {metric_value} Mean', color=color, s=100)

            # Add a regression line with R², and remove the shading (confidence interval)
            sns.regplot(data=df_sorted, x='Rank', y=f'Tone {metric_value} Mean', scatter=False, color='black', line_kws={'lw': 2}, ci=None)

            # Set the x-axis ticks to be separated by increments of 1
            plt.xticks(ticks=range(int(df_sorted['Rank'].min()), int(df_sorted['Rank'].max()) + 1, 1))

            # Labels and title
            plt.xlabel('Rank')
            plt.ylabel('Tone Mean AUC')
            plt.title(f'{condition} {metric_value} Tone Response to Rank')
            title = f'{metric_value} DA response Rank ({brain_region})'
            save_path = os.path.join(str(directory_path) + '\\' + f'{title}.png')
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
            return r_value, p_value, n_value

        r_nac, p_nac, n_nac = scatter_plot(df_sorted_n, metric_value=metric_name, brain_region="NAc")
        r_mpfc, p_mpfc, n_mpfc = scatter_plot(df_sorted_p, metric_value=metric_name, brain_region="mPFC")
        print(f"NAc: r={r_nac:.3f}, p={p_nac:.3f}, n={n_nac}")
        print(f"mPFC: r={r_mpfc:.3f}, p={p_mpfc:.3f}, n={n_mpfc}")

        
    """********************************MISC*************************************"""
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

        """df['ei_tone First'] = df['ei_tone'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        df['ei_tone Last'] = df['ei_tone'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)"""