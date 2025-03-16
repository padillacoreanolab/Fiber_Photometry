import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment
from trial_class import Trial
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

class Reward_Competition(Experiment):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)
        # self.trials = {}  # Reset trials to avoid loading from parent class
        self.df = pd.DataFrame()
        # self.load_rtc1_trials()  # Load 1 RTC trial
        if self.behavior_folder_path and 'Cohort_1_2' not in self.behavior_folder_path:
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
            trial_folder1 = trial_name1 + trial_folder.split('_')[1].split('-')[1]
            trial_folder2 = trial_name2 + trial_folder.split('_')[1].split('-')[1]
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
    def combine_consecutive_behaviors1(self, behavior_name='all', bout_time_threshold=1, min_occurrences=1):
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
        # if use excel time:
        # filtered_time_col = df.filter(like='time')
        # filtered_time_col = filtered_time_col.dropna(axis=1, how='all')
        # col_to_keep = ['file name', filtered_columns, 'tangles']
        df = df[col_to_keep]
        self.df = df

    def read_manual_scoring2(self, csv_file_path):
        """
        Reads in and creates a df for trials that recorded 2 mice simultaneously
        
        First change file names so that there is only one mouse before the date, using the subject column.
        """
        pass

    def merge_data(self):
        """
        Merges all data into a dataframe for analysis.
        """
        self.df['trial'] = self.df['file name'].map(self.trials)
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
    
    def merging_ranks(self, rank_df):
        df_merged = self.df.merge(rank_df, left_on="subject_name", right_on="ID", how="left")

        # Step 2: Drop redundant "ID" column if needed
        df_merged.drop(columns=["ID"], inplace=True)
        self.df = df_merged

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
            
            print(f"Row {row.name}: Subject {row['subject_name']} - Valid Indices: {valid_indices}")

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
            print(f"Row {row.name}: Before -> {row['filtered_sound_cues']}, After -> {filtered_cues}")

            return pd.Series([filtered_cues])

        # Apply the function to filter the DataFrame
        self.df[['filtered_sound_cues']] = self.df.apply(filter_by_winner, axis=1)

        # Drop rows where no filtered_sound_cues remain
        self.df = self.df[self.df['filtered_sound_cues'].apply(len) > 0]



    def losing(self):
        def filter_by_loser(row):
            # Ensure 'filtered_winner_array' is a list
            if not isinstance(row['filtered_winner_array'], list):
                print(f"Skipping row {row.name}: filtered_winner_array is not a list.")
                return pd.Series([row['filtered_sound_cues']])

            # Get indices where `subject_name` is NOT the winner
            loser_indices = [i for i, winner in enumerate(row['filtered_winner_array']) if winner != row['subject_name']]

            # If no losing trials exist, return an empty list
            if not loser_indices:
                return pd.Series([[]])

            # Ensure `filtered_sound_cues` is a list before filtering
            if not isinstance(row['filtered_sound_cues'], list):
                return pd.Series([[]])

            # Filter only relevant columns
            filtered_cues = [row['filtered_sound_cues'][i] for i in loser_indices]

            return pd.Series([filtered_cues])

        # Apply the function to filter only the `filtered_sound_cues` column
        self.df[['filtered_sound_cues']] = self.df.apply(filter_by_loser, axis=1)

        # Drop rows where no `filtered_sound_cues` remain
        self.df = self.df[self.df['filtered_sound_cues'].apply(len) > 0]

    

    """*******************************LICKS********************************"""
    def find_first_lick_after_sound_cue(self):
        """
        Finds the first port entry occurring after 4 seconds following each sound cue.
        If a port entry starts before 4 seconds but extends past it, 
        the function selects the timestamp at 4 seconds after the sound cue.
        
        Stores the results as a list in `self.first_lick_after_sound_cue` and adds it to `self.df`.
        """

        first_licks = []  # List to store results

        for index, row in self.df.iterrows():
            sound_cues_onsets = row['filtered_sound_cues'] 
            port_entries_onsets = row['filtered_port_entries']  
            port_entries_offsets = row['filtered_port_entry_offset']  

            first_licks_per_row = []  # Store first licks for this row

            for sc_onset in sound_cues_onsets:
                threshold_time = sc_onset + 4  # Define 4s threshold

                # Find port entries that start AFTER 4 seconds post sound cue
                future_licks_indices = np.where(port_entries_onsets >= threshold_time)[0]

                if len(future_licks_indices) > 0:
                    # If there are port entries after 4s, take the first one
                    first_licks_per_row.append(port_entries_onsets[future_licks_indices[0]])
                else:
                    # Find port entries that START before 4s but continue PAST it
                    ongoing_licks_indices = np.where((port_entries_onsets < threshold_time) & (port_entries_offsets > threshold_time))[0]

                    if len(ongoing_licks_indices) > 0:
                        # If a port entry spans past 4s, use the exact timepoint at 4s
                        first_licks_per_row.append(threshold_time)
                    else:
                        # If no port entry occurs after 4s, append None
                        first_licks_per_row.append(None)

            first_licks.append(first_licks_per_row)  # Append row's results to main list

        # Store results in the object
        self.first_lick_after_sound_cue = first_licks

        # Add to DataFrame
        self.df["first_lick_after_sound_cue"] = first_licks

    def compute_closest_port_offset(self, lick_column, offset_column):
        """
        Computes the closest port entry offsets after each lick time and adds them as a new column in the dataframe.
        
        Parameters:
            lick_column (str): The column name for the lick times (e.g., 'first_lick_after_sound_cue').
            offset_column (str): The column name for the port entry offset times (e.g., 'filtered_port_entry_offset').
            new_column_name (str): The name of the new column to store the results. Default is 'closest_port_entry_offsets'.
        
        Returns:
            pd.DataFrame: Updated DataFrame with the new column of closest port entry offsets.
        """
        
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
        self.df['closest_lick_offset'] = self.df.apply(compute_lick_metrics, axis=1)
    """*********************************CALCULATING DOPAMINE RESPONSE***********************************"""
    def compute_tone_da(self):
        def compute_da_metrics_for_trial(trial_obj, filtered_sound_cues, 
                                        use_adaptive=True, peak_fall_fraction=0.5, 
                                        allow_bout_extension=False):
            """Compute DA metrics (AUC, Max Peak, Time of Max Peak, Mean Z-score) for each sound cue, using adaptive peak-following."""
            """if not hasattr(trial_obj, "timestamps") or not hasattr(trial_obj, "zscore"):
                return np.nan"""  # Handle missing attributes

            timestamps = np.array(trial_obj.timestamps)  
            zscores = np.array(trial_obj.zscore)  

            computed_metrics = []

            for cue in filtered_sound_cues:
                start_time = cue
                end_time = cue + 10  # Default window end

                # Extract initial window
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                window_ts = timestamps[mask]
                window_z = zscores[mask]

                if len(window_ts) < 2:
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
                    continue  # Skip to next cue

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
                        end_time = window_ts[fall_idx]  # Adjust end time
                    elif allow_bout_extension:
                        # Extend to the full timestamp range
                        extended_mask = (timestamps >= start_time)
                        extended_ts = timestamps[extended_mask]
                        extended_z = zscores[extended_mask]

                        peak_idx_ext = np.argmin(np.abs(extended_ts - peak_time))
                        fall_idx_ext = peak_idx_ext

                        while fall_idx_ext < len(extended_z) and extended_z[fall_idx_ext] > threshold:
                            fall_idx_ext += 1

                        if fall_idx_ext < len(extended_ts):
                            end_time = extended_ts[fall_idx_ext]
                        else:
                            end_time = extended_ts[-1]

                # Re-extract window with adjusted end time
                final_mask = (timestamps >= start_time) & (timestamps <= end_time)
                final_ts = timestamps[final_mask]
                final_z = zscores[final_mask]

                if len(final_ts) < 2:
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
                    continue

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
                    "Adjusted End": end_time  # Store adjusted end
                })
            return computed_metrics

        # Apply function across all trials
        self.df["computed_metrics"] = self.df.apply(
            lambda row: compute_da_metrics_for_trial(row["trial"], row["filtered_sound_cues"], 
                                                    use_adaptive=True, peak_fall_fraction=0.5, 
                                                    allow_bout_extension=False), axis=1)
        self.df["Tone AUC"] = self.df["computed_metrics"].apply(lambda x: [item.get("AUC", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Tone Max Peak"] = self.df["computed_metrics"].apply(lambda x: [item.get("Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Tone Time of Max Peak"] = self.df["computed_metrics"].apply(lambda x: [item.get("Time of Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Tone Mean Z-score"] = self.df["computed_metrics"].apply(lambda x: [item.get("Mean Z-score", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Tone Adjusted End"] = self.df["computed_metrics"].apply(lambda x: [item.get("Adjusted End", np.nan) for item in x] if isinstance(x, list) else np.nan)

        # Drop the "computed_metrics" column if it's no longer needed
        self.df.drop(columns=["computed_metrics"], inplace=True)
                
    def compute_lick_da(self):
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
        self.df["lick_computed_metrics"] = self.df.apply(
            lambda row: compute_da_metrics_for_lick(row["trial"], row["first_lick_after_sound_cue"], 
                                                    row["closest_lick_offset"], use_adaptive=True, 
                                                    peak_fall_fraction=0.5, allow_bout_extension=False), axis=1)

        # Extract the individual DA metrics into new columns
        self.df["Lick AUC"] = self.df["lick_computed_metrics"].apply(lambda x: [item.get("AUC", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Lick Max Peak"] = self.df["lick_computed_metrics"].apply(lambda x: [item.get("Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Lick Time of Max Peak"] = self.df["lick_computed_metrics"].apply(lambda x: [item.get("Time of Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Lick Mean Z-score"] = self.df["lick_computed_metrics"].apply(lambda x: [item.get("Mean Z-score", np.nan) for item in x] if isinstance(x, list) else np.nan)
        self.df["Lick Adjusted End"] = self.df["lick_computed_metrics"].apply(lambda x: [item.get("Adjusted End", np.nan) for item in x] if isinstance(x, list) else np.nan)

        # Drop the "lick_computed_metrics" column if it's no longer needed
        self.df.drop(columns=["lick_computed_metrics"], inplace=True)

    def find_means(self):
        self.df["Lick AUC Mean"] = self.df["Lick AUC"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        self.df["Lick Max Peak Mean"] = self.df["Lick Max Peak"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        self.df["Lick Mean Z-score Mean"] = self.df["Lick Mean Z-score"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        self.df["Tone AUC Mean"] = self.df["Tone AUC"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        self.df["Tone Max Peak Mean"] = self.df["Tone Max Peak"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        self.df["Tone Mean Z-score Mean"] = self.df["Tone Mean Z-score"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)

    def find_overall_mean(self):
        mean_df = self.df.groupby(['subject_name'], as_index=False).agg({
            'Rank': 'first',  # Keeps the Rank column
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
            'Tone Mean Z-score Last': 'mean'
        })

        self.df = mean_df

    """*******************************PLOTTING**********************************"""
    def plot_first_last(self, condition='winning'):
        pass

    def plot_dom(self, condition='winning'):
        pass

    def plot_sub(self, condition='winning'):
        pass

    # Response is tone or lick, metric_name is for AUC, Max Peak, or Mean Z-score
    def plot_da(self, metric_name, condition='Winning',  
                    brain_region='mPFC',  # New parameter to specify the brain region
                    custom_xtick_labels=None, 
                    custom_xtick_colors=None, 
                    ylim=None, 
                    nac_color='#15616F',   # Color for NAc
                    mpfc_color='#FFAF00',  # Color for mPFC
                    yticks_increment=None, 
                    figsize=(7,7),  
                    pad_inches=0.1):
        """
        Customizable plotting function that plots a single brain region (NAc or mPFC) for both lick and tone.
        Can use metrics Max Peak, Mean Z-score, and AUC
        """
        # Filtering data frame to only keep specified metrics
        def filter_by_metric(df):
            metric_columns = df.filter(like=metric_name + ' Mean').columns
            if len(metric_columns) != 2:
                raise ValueError(f"Expected exactly 2 columns, but found {len(metric_columns)}.")

            # Create two DataFrames, keeping 'subject_name'
            df1 = df[['subject_name', metric_columns[0]]].copy()
            df2 = df[['subject_name', metric_columns[1]]].copy()

            return df1, df2
        
        # spliting and copying dataframe into two dataframes for each brain region 
        def split_by_subject(df1, df2):            
            # Filter out the 'subject_name' column and keep only the relevant columns for response and metric_name
            df_n = df1[df1['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p = df1[df1['subject_name'].str.startswith('p')].drop(columns=['subject_name'])
            
            df_n1 = df2[df2['subject_name'].str.startswith('n')].drop(columns=['subject_name'])
            df_p1 = df2[df2['subject_name'].str.startswith('p')].drop(columns=['subject_name'])
            # Return filtered dataframes and subject_name column
            return df_n, df_p, df_n1, df_p1

        filtered_df, filtered_df1 = filter_by_metric(self.df)

        # Split data into NAc and mPFC, with subject names
        df_nac, df_mpfc, df_nac1, df_mpfc1 = split_by_subject(filtered_df, filtered_df1)

        # Select the data for the desired brain region
        if brain_region == 'NAc':
            df = df_nac
            df1 = df_nac1 
            mean_values = df_nac.mean()
            sem_values = df_nac.sem()
            title_suffix = 'NAc'
            bar_color = nac_color
        elif brain_region == 'mPFC':
            df = df_mpfc
            df1 = df_mpfc1
            mean_values = df_mpfc.mean()
            sem_values = df_mpfc.sem()
            title_suffix = 'mPFC'
            bar_color = mpfc_color
        else:
            raise ValueError("brain_region must be either 'NAc' or 'mPFC'")

        title = metric_name + ' ' + condition + f' DA Response ({title_suffix})'

        # Ensure the dataframe contains only numeric data (in case of any non-numeric columns)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Calculate the mean and SEM values for the entire dataframe
        # Tone
        mean_values = df.mean()
        sem_values = df.sem()

        # Lick
        mean_values1 = df1.mean()
        sem_values1 = df1.sem()

        # Example: renaming specific columns
        df.rename(columns={'Tone ' + metric_name: 'Tone'}, inplace=True)
        df1.rename(columns={'Lick ' + metric_name: 'Tone'}, inplace=True)

        # Define bar width for side-by-side bars
        bar_width = 1  # Width of each bar
        gap = 0.5  # Adjust this value to control the gap between the bars

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
            label='Tone',  # Label for legend
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
            label='Lick',  # Label for legend
            error_kw=dict(elinewidth=4, capthick=4, capsize=10, zorder=5)
        )

        # Set x-ticks in the center of each grouped pair of bars
        x_ticks = x  # The positions for the x-ticks (center of the grouped bars)
        custom_xtick_labels = ['Tone', 'Lick']  # Labels for the groups

        # Set the positions of x-ticks and their labels
        ax.set_xticks(x)  # Position the ticks at the center of the bars
        ax.set_xticklabels(custom_xtick_labels * len(df.columns), fontsize=36)  # Set the labels, repeating for each group

        # Increase font sizes
        ax.set_ylabel(metric_name, fontsize=36, labelpad=12)
        ax.set_xlabel("Event", fontsize=40, labelpad=12)
        ax.tick_params(axis='y', labelsize=32)
        ax.tick_params(axis='x', labelsize=32)

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
        plt.title(title, fontsize=36, fontweight='bold')

        # Save and display the plot
        plt.savefig(f'{title}{metric_name[0]}.png', transparent=True, bbox_inches='tight', pad_inches=pad_inches)
        plt.tight_layout()
        plt.show()

    """********************************MISC*************************************"""
    def drop_unnecessary(self):
        # drops lots of unnecessary column to allow for easier observations
        self.df.drop(columns=['file name', 'port entries onset', 'port entries offset', 'sound cues',
                              'sound cues', 'port entries', 'winner_array', 'DS', 'filtered_port_entries'], inplace=True)
        
    def first_last(self):
        """
        Finds the first and last bouts DA of competition
        """
        # Extract the first and last values of the array under 'Lick AUC'
        self.df['Lick AUC First'] = self.df['Lick AUC'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        self.df['Lick AUC Last'] = self.df['Lick AUC'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        self.df['Lick Max Peak First'] = self.df['Lick Max Peak'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        self.df['Lick Max Peak Last'] = self.df['Lick Max Peak'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        self.df['Lick Mean Z-score First'] = self.df['Lick Mean Z-score'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        self.df['Lick Mean Z-score Last'] = self.df['Lick Mean Z-score'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        self.df['Tone AUC First'] = self.df['Tone AUC'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        self.df['Tone AUC Last'] = self.df['Tone AUC'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        self.df['Tone Max Peak First'] = self.df['Tone Max Peak'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        self.df['Tone Max Peak Last'] = self.df['Tone Max Peak'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)

        self.df['Tone Mean Z-score First'] = self.df['Tone Mean Z-score'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
        self.df['Tone Mean Z-score Last'] = self.df['Tone Mean Z-score'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)