import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_ROOT)
import os
import pandas as pd
import numpy as np
from experiment_class import Experiment
from slp_trl_class import SleapTrial
from typing import List, Dict, Optional
import warnings


class SleapExperiment(Experiment):
    def __init__(self,
                 experiment_folder_path: str,
                 behavior_folder_path: str,
                 sleap_folder_path: str,
                 corner_folder_path: str,
                 fps: float = 10.0,
                 stream_DA: str = "_465A",
                 stream_ISOS: str = "_405A"):
        super().__init__(
            experiment_folder_path,
            behavior_folder_path,
            RTC=True
        )

        self.sleap_folder_path    = sleap_folder_path
        self.corner_folder_path = corner_folder_path
        self.fps           = fps
        self.stream_DA     = stream_DA
        self.stream_ISOS   = stream_ISOS
        self.trials        = {} 
        self.all_sleap_features = pd.DataFrame()

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
        trial_dfs: List[pd.DataFrame] = []

        # 0) build subject → corner_h5 lookup
        corner_h5s = [p for p in os.listdir(self.corner_folder_path) if p.endswith('.h5')]
        subj_to_corner: Dict[str,str] = {}
        for fn in corner_h5s:
            for trial in self.trials.values():
                if trial.subject_name in fn:
                    subj_to_corner[trial.subject_name] = os.path.join(self.corner_folder_path, fn)

        # 1) build subject → sleap_h5 lookup (as before)
        sleap_files = [f for f in os.listdir(self.sleap_folder_path) if f.endswith('.analysis.h5')]
        subj_to_sleap: Dict[str,str] = {}
        for fn in sleap_files:
            for trial in self.trials.values():
                if trial.subject_name in fn:
                    subj_to_sleap[trial.subject_name] = os.path.join(self.sleap_folder_path, fn)

        # 2) process each trial
        for name, trial in self.trials.items():
            subj = trial.subject_name
            if subj not in subj_to_sleap:
                print(f"⚠️  skipping {subj}: no SLEAP file")
                continue

            # --- calibrate from corners ---
            corner_path = subj_to_corner.get(subj)
            if corner_path:
                trial.calibrate_from_corners(
                    corner_h5_path=corner_path,
                    real_width_cm=32.0,
                    top_left="Top_Left",
                    top_right="Top_Right",
                )
            else:
                print(f"⚠️  no corner file for {subj}, distances in px")

            # --- now the usual SLEAP pipeline ---
            sleap_path = subj_to_sleap[subj]
            trial.load_sleap(sleap_path, fps=self.fps)
            trial.keep_only_tracks(('subject', 'agent'))

            trial.filter_sleap_bouts(interp_kind="linear")
            trial.smooth_locations(win=25, poly=3)
            trial.add_metadata_and_DA()
            df = trial.compute_pairwise_features()
            df = trial.add_behavior_column(df, time_col="time_s",
                               out_col="behavior_active",
                               mode="all")

            trial_dfs.append(df)

        all_df = pd.concat(trial_dfs, ignore_index=True) if trial_dfs else pd.DataFrame()
        self.all_sleap_features = all_df
        return all_df

    # def extract_sleap_features(
    #     self,
    #     fps: Optional[float] = None,
    #     interp_kind: str = "linear",
    #     smooth_win: int = 25,
    #     smooth_poly: int = 3,
    #     real_width_cm: float = 32.0,
    # ) -> pd.DataFrame:
    #     """
    #     Build a per-trial feature DataFrame by loading SLEAP tracks, interpolating, smoothing,
    #     and computing pairwise features. Stores the concatenated result in `self.all_sleap_features`.

    #     Parameters
    #     ----------
    #     fps : float or None
    #         Override FPS for SLEAP timebase. If None, uses self.fps.
    #     interp_kind : str
    #         Interpolation kind passed to `trial.filter_sleap_bouts`.
    #     smooth_win : int
    #         Savitzky–Golay window length for `trial.smooth_locations`.
    #     smooth_poly : int
    #         Savitzky–Golay polynomial order for `trial.smooth_locations`.
    #     real_width_cm : float
    #         Physical arena width (cm) for corner calibration. If a subject lacks corner file,
    #         distances will remain in pixels.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Concatenated feature table across all processed trials. Empty DataFrame if none processed.
    #     """
    #     trial_dfs: List[pd.DataFrame] = []
    #     fps_val = fps if fps is not None else self.fps

    #     # ---------- 0) subject → corner_h5 ----------
    #     subj_to_corner: Dict[str, str] = {}
    #     if getattr(self, "corner_folder_path", None):
    #         corner_h5s = [p for p in os.listdir(self.corner_folder_path) if p.endswith(".h5")]
    #         for fn in corner_h5s:
    #             for tr in self.trials.values():
    #                 if tr.subject_name in fn:
    #                     subj_to_corner[tr.subject_name] = os.path.join(self.corner_folder_path, fn)

    #     # ---------- 1) subject → sleap_h5 ----------
    #     subj_to_sleap: Dict[str, str] = {}
    #     sleap_files = [f for f in os.listdir(self.sleap_folder_path) if f.endswith(".analysis.h5")]
    #     for fn in sleap_files:
    #         for tr in self.trials.values():
    #             if tr.subject_name in fn:
    #                 subj_to_sleap[tr.subject_name] = os.path.join(self.sleap_folder_path, fn)

    #     # ---------- 2) iterate ----------
    #     for name, trial in self.trials.items():
    #         subj = trial.subject_name

    #         if subj not in subj_to_sleap:
    #             warnings.warn(f"Skipping {subj}: no SLEAP file found in {self.sleap_folder_path}")
    #             continue

    #         corner_path = subj_to_corner.get(subj)
    #         if corner_path:
    #             try:
    #                 trial.calibrate_from_corners(
    #                     corner_h5_path=corner_path,
    #                     real_width_cm=real_width_cm,
    #                     top_left="Top_Left",
    #                     top_right="Top_Right",
    #                 )
    #             except Exception as e:
    #                 warnings.warn(f"{subj}: corner calibration failed ({e}); keeping pixels.")

    #         sleap_path = subj_to_sleap[subj]

    #         try:
    #             print(self.subject_name, "raw:", self._raw_locations.shape,
    #                   "mask.sum:", mask.sum(), "mask.shape:", mask.shape)
    #             # ---- pipeline ----
    #             trial.load_sleap(sleap_path, fps=fps_val)

    #             # sanity check before masking/interp
    #             if getattr(trial, "_raw_locations", None) is None or trial._raw_locations.size == 0:
    #                 warnings.warn(f"{subj}: _raw_locations empty after load; skipping.")
    #                 continue

    #             trial.filter_sleap_bouts(interp_kind=interp_kind)
    #             trial.smooth_locations(win=smooth_win, poly=smooth_poly)
    #             trial.add_metadata_and_DA()

    #             df = trial.compute_pairwise_features()
    #             if df is None or df.empty:
    #                 warnings.warn(f"{subj}: feature DataFrame empty; skipping.")
    #                 continue

    #             trial_dfs.append(df)

    #         except ValueError as ve:
    #             # Specific reshape/size==0 errors bubble up here
    #             warnings.warn(f"{subj}: ValueError during SLEAP processing → {ve}")
    #             continue
    #         except Exception as e:
    #             warnings.warn(f"{subj}: Unexpected error during SLEAP processing → {e}")
    #             continue

    #     all_df = pd.concat(trial_dfs, ignore_index=True) if trial_dfs else pd.DataFrame()
    #     self.all_sleap_features = all_df
    #     return all_df
    
    def save_sleap_features(self, out_csv: str):
        """Dump the concatenated SLEAP + DA feature table."""
        self.all_sleap_features.to_csv(out_csv, index=False)
