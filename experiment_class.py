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
        self.trials = {}

        self.load_trials()

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

