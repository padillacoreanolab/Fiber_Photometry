import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment
from trial_class import Trial

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
        # self.load_rtc2_trials()  # Load 2 RTC trials instead

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
            print(trial)
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
        print(self.df.columns)

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
        
        # Optionally drop the 'tangles_array' column if no longer needed
        self.df.drop(columns=['tangles_array'], inplace=True)
        
        print(self.df.columns)

