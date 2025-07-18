import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_ROOT)
import os
import pandas as pd
from experiment_class import Experiment            # your existing Experiment base
from slp_trl_class import SleapTrial        # your extended Trial w/ SLEAP
from typing import List, Dict

class SleapExperiment(Experiment):
    def __init__(self,
                 experiment_folder_path: str,
                 behavior_folder_path: str,
                 sleap_folder_path: str,
                 fps: float = 10.0,
                 stream_DA: str = "_465A",
                 stream_ISOS: str = "_405A"):
        # 1) tell Experiment not to auto‐load its TDT trials
        super().__init__(
            experiment_folder_path,
            behavior_folder_path,
            RTC=True        # ← this prevents base.load_trials()
        )

        # 2) now explicitly load your SleapTrial objects
        self.sleap_folder_path    = sleap_folder_path
        self.fps           = fps
        self.stream_DA     = stream_DA
        self.stream_ISOS   = stream_ISOS
        self.trials        = {}    # reset, in case base did anything
        self.all_sleap_features = []

        self.sleap_load_trials()


    def sleap_load_trials(self):
        """
        Loads each trial folder (block) as a TDTData object and extracts manual annotation behaviors.
        """
        trial_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                        if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            trial_obj = SleapTrial(trial_path, '_465A', '_405A')

            self.trials[trial_folder] = trial_obj

    def extract_sleap_features(self) -> pd.DataFrame:
        """
        For each SleapTrial in self.trials:
          1) Find its matching .analysis.h5 by subject_name
          2) load_sleap, filter_sleap_bouts, smooth_locations
          3) add_metadata_and_DA(), compute_pairwise_features()
          4) stamp Subject, Brain_Region, Mouse_ID
        """
        self.all_sleap_features.clear()

        # 0) map subject_name → full path to that subject's .analysis.h5
        h5_files = [
            f for f in os.listdir(self.sleap_folder_path)
            if f.endswith('.analysis.h5')
        ]
        subj_to_h5 = {}
        for fn in h5_files:
            for trial in self.trials.values():
                if trial.subject_name in fn:
                    subj_to_h5[trial.subject_name] = os.path.join(self.sleap_folder_path, fn)

        # 1) loop over SleapTrial objects already in self.trials
        for name, trial in self.trials.items():
            subj = trial.subject_name
            if subj not in subj_to_h5:
                print(f"⚠️  skipping {subj}: no SLEAP file found")
                continue

            h5_path = subj_to_h5[subj]

            # load & crop video tracks to bouts
            trial.load_sleap(h5_path, fps=self.fps)
            trial.filter_sleap_bouts(interp_kind="linear")
            trial.smooth_locations(win=25, poly=3)

            # align DA & metadata, compute pose features
            trial.add_metadata_and_DA()
            df = trial.compute_pairwise_features()

            # stamp trial‑level columns
            df['Subject']      = subj
            df['Mouse_ID']     = subj
            df['Brain_Region'] = 'NAc' if 'nac' in subj.lower() else 'mPFC'
            # intruder_identity is already set by add_metadata_and_DA()

            self.all_sleap_features.append(df)

        # 2) concatenate all trials’ feature‐tables
        if self.all_sleap_features:
            return pd.concat(self.all_sleap_features, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_sleap_features(self, out_csv: str):
        """Dump the concatenated SLEAP + DA feature table."""
        self.all_sleap_features.to_csv(out_csv, index=False)
