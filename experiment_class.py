import numpy as np
import pandas as pd
import tdt
import os
import matplotlib.pyplot as plt

from trial_class import Trial

class Experiment:
    def __init__(self, experiment_folder_path, behavior_folder_path):
        self.experiment_folder_path = experiment_folder_path
        self.behavior_folder_path = behavior_folder_path
        self.blocks = {}
        self.trials = {}

        self.load_trials()
    
    '''********************************** FOR SINGLE OBJECT  **********************************'''
    def get_first_behavior(self, behaviors=['Investigation', 'Approach', 'Defeat', 'Aggression']):
        """
        Extracts the mean z-score and other details for the first 'Investigation' and 'Approach' behavior events
        from each bout in the bout_dict and stores the values in a new dictionary.

        Parameters:
        - bout_dict (dict): Dictionary containing bout data with behavior events for each bout.
        - behaviors (list): List of behavior events to track (defaults to ['Investigation', 'Approach']).

        Returns:
        - first_behavior_dict (dict): Dictionary containing the start time, end time, duration, 
                                  and mean z-score for each behavior in each bout.
        """
        first_behavior_dict = {}

        # Loop through each bout in the bout_dict
        for bout_name, bout_data in self.bout_dict.items():
            first_behavior_dict[bout_name] = {}  # Initialize the dictionary for this bout
            
            # Loop through each behavior we want to track
            for behavior in behaviors:
                # Check if behavior exists in bout_data and if it contains valid event data
                if behavior in bout_data and isinstance(bout_data[behavior], list) and len(bout_data[behavior]) > 0:
                    # Access the first event for the behavior
                    first_event = bout_data[behavior][0]  # Assuming this is a list of events
                    
                    # Extract the relevant details for this behavior event
                    first_behavior_dict[bout_name][behavior] = {
                        'Start Time': first_event['Start Time'],
                        'End Time': first_event['End Time'],
                        'Total Duration': first_event['End Time'] - first_event['Start Time'],
                        'Mean zscore': first_event['Mean zscore']
                    }
                else:
                    # If the behavior doesn't exist in this bout, add None placeholders
                    first_behavior_dict[bout_name][behavior] = {
                        'Start Time': None,
                        'End Time': None,
                        'Total Duration': None,
                        'Mean zscore': None
                    }


        self.first_behavior_dict = first_behavior_dict

    '''********************************** GROUP PROCESSING **********************************'''
    def load_trials(self):
        """
        Loads each trial folder (block) as a TDTData object and extracts manual annotation behaviors.
        """
        trial_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                        if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            trial_obj = Trial(trial_path)

            self.trials[trial_folder] = trial_obj


    def default_batch_process(self, time_segments_to_remove=None):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for trial_folder, trial in self.trials.items():
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Processing {trial_folder}...")
            trial.remove_initial_LED_artifact(t=30)
            trial.remove_final_data_segment(t = 10)
            
            trial.smooth_and_apply(window_len=int(trial.fs)*1)
            trial.apply_ma_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            # baseline_start, baseline_end = trial.find_baseline_period()  
            # trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            trial.compute_zscore(method = 'standard')

            # trial.remove_short_behaviors(behavior_name='all', min_duration=0.2)
            # trial.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=2, min_occurrences=1)
            trial.verify_signal()

            # Extract manual annotation behaviors
            csv_file_name = f"{trial_folder}.csv"
            csv_file_path = os.path.join(self.behavior_folder_path, csv_file_name)
            trial.extract_manual_annotation_behaviors(csv_file_path)

# added by me
    def remove_time_segments_from_block(self, block_name, time_segments):
        """
        Remove specified time segments from a given block's data.

        Parameters:
        block_name (str): The name of the block (file) to remove time segments from.
        time_segments (list): A list of tuples representing the time segments to remove [(start_time, end_time), ...].
        """
        tdt_data_obj = self.blocks.get(block_name, None)
        if tdt_data_obj is None:
            print(f"Block {block_name} not found.")
            return

        for (start_time, end_time) in time_segments:
            tdt_data_obj.remove_time_segment(start_time, end_time)

        print(f"Removed specified time segments from block {block_name}.")


    '''********************************** PLOTTING **********************************'''
    def plot_all_traces(experiment, behavior_name='all'):
        """
        Plots behavior events for all trials in separate subplots within the same figure.
        """
        num_trials = len(experiment.trials)

        if num_trials == 0:
            print("No trials found in the experiment.")
            return
        
        fig, axes = plt.subplots(nrows=num_trials, figsize=(12, 3 * num_trials), sharex=True)

        # Ensure axes is iterable (in case of a single subplot)
        if num_trials == 1:
            axes = [axes]

        # Loop through trials and plot behavior events in each subplot
        for ax, (trial_name, trial) in zip(axes, experiment.trials.items()):
            trial.plot_behavior_event(behavior_name, ax=ax) 
            ax.set_title(trial_name)

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

