import numpy as np
import pandas as pd
import tdt
import os

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
            
            # Extract manual annotation behaviors
            csv_file_name = f"{trial_folder}.csv"
            csv_file_path = os.path.join(self.behavior_folder_path, csv_file_name)
            trial_obj.extract_manual_annotation_behaviors(csv_file_path)

            self.trials[trial_folder] = trial_obj

    
    '''********************************** TRIAL-SPECIFIC PROCESSING **********************************'''
    def remove_time_segments_from_trial(self, block_name, time_segments):
        """
        Remove specified time segments from a given trial's data.
        """
        tdt_data_obj = self.blocks.get(block_name, None)
        if tdt_data_obj is None:
            print(f"Block {block_name} not found.")
            return

        for (start_time, end_time) in time_segments:
            tdt_data_obj.remove_time_segment(start_time, end_time)

        print(f"Removed specified time segments from block {block_name}.")
