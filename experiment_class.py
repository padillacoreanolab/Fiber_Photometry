import numpy as np
import pandas as pd
import tdt
import os
import matplotlib.pyplot as plt

from trial_class import Trial

class Experiment:
    def __init__(self, experiment_folder_path, behavior_folder_path, autoload=True):
        self.experiment_folder_path = experiment_folder_path
        self.behavior_folder_path = behavior_folder_path
        self.trials = {}

        if autoload:
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
            trial_obj = Trial(trial_path, '_465A', '_405A')

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
            
            trial.smooth_and_apply(window_len=int(trial.fs)*2)
            trial.apply_ma_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            baseline_start, baseline_end = trial.find_baseline_period()  
            # trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            trial.compute_zscore(method = 'standard')

            trial.verify_signal()


    def group_extract_manual_annotations(self, bout_definitions, first_only=True):
        """
        Extracts behavior bouts and annotations for all trials in the experiment.

        This function:
        1. Iterates through self.trials, looking for behavior CSV files in self.behavior_folder_path.
        2. Calls extract_bouts_and_behaviors for each trial.
        3. Stores the behavior data inside each Trial object.

        Parameters:
        - bout_definitions (list of dict): List defining each bout with:
            - 'prefix': Label used for the bout (e.g., "s1", "s2", "x").
            - 'introduced': Name of the behavior marking the start of the bout.
            - 'removed': Name of the behavior marking the end of the bout.
        - first_only (bool): If True, only the first event in each bout is kept;
                            if False, all events within each bout are retained.
        """
        for trial_name, trial in self.trials.items():
            csv_path = os.path.join(self.behavior_folder_path, f"{trial_name}.csv")
            if os.path.exists(csv_path):
                print(f"Processing behaviors for {trial_name}...")
                trial.extract_bouts_and_behaviors(csv_path, bout_definitions, first_only=first_only)
                trial.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=1)
                # Optionally, you can remove short behaviors:
                # trial.remove_short_behaviors(behavior_name='all', min_duration=0.3)
            else:
                print(f"Warning: No CSV found for {trial_name} in {self.behavior_folder_path}. Skipping.")




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


    def compute_all_da_metrics(self, use_fractional=False, max_bout_duration=10, 
                            use_adaptive=False, allow_bout_extension=False, mode='standard'):
        """
        Iterates over all trials in the experiment and computes DA metrics with the specified windowing options.
        
        Parameters:
        - use_fractional (bool): Whether to limit the window to a maximum duration.
        - max_bout_duration (int): Maximum allowed window duration (in seconds) if fractional analysis is applied.
        - use_adaptive (bool): Whether to adjust the window using adaptive windowing (via local minimum detection).
        - allow_bout_extension (bool): Whether to extend the window if no local minimum is found.
        - mode (str): Either 'standard' to compute metrics using the full standard DA signal, or 'EI' to compute metrics
                        using the event-induced data (i.e. the precomputed 'Event_Time_Axis' and 'Event_Zscore' columns).
        """
        for trial_name, trial in self.trials.items():
            if hasattr(trial, 'compute_da_metrics'):
                print(f"Computing DA metrics for {trial_name} ...")
                trial.compute_da_metrics(use_fractional=use_fractional,
                                        max_bout_duration=max_bout_duration,
                                        use_adaptive=use_adaptive,
                                        allow_bout_extension=allow_bout_extension,
                                        mode=mode)
            else:
                print(f"Warning: Trial '{trial_name}' does not have compute_da_metrics method.")



    def compute_all_event_induced_DA(self, pre_time=4, post_time=15):
        """
        Iterates over all trials in the experiment and computes the event-induced DA signals
        for each trial by calling each Trial's compute_event_induced_DA() method.
        
        Parameters:
        - pre_time (float): Seconds to include before event onset.
        - post_time (float): Seconds to include after event onset.
        """
        for trial_name, trial in self.trials.items():
            print(f"Computing event-induced DA for trial {trial_name} ...")
            trial.compute_event_induced_DA(pre_time=pre_time, post_time=post_time)


