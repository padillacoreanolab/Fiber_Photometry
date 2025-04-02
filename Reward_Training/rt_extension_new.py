import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment 
from trial_class import Trial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
from rtc_extension_albert import RTC 

class Reward_Training(RTC):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)
        self.df = pd.DataFrame()  # Optional: reset if needed

    '''*********************************CREATE DF*************************************'''
    def create_base_df(self, directory_path):
        """
        Creates Dataframe that will contain all the data for analysis.
        """
        subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        print(f"Subdirectories: {subdirectories}")  # Debugging line

        # Create the DataFrame with 'file name' column (subdirectories here)
        self.df['file name'] = subdirectories
        self.df['file name'] = self.df['file name'].astype(str)
        
        # matching up data with their file names in the new dataframe
        def find_matching_trial(file_name):
            return self.trials.get(file_name, None)
        
        def expand_filenames(filename):
            parts = filename.split('-')  # Split by '-'
            
            # Extract timestamp (assumes last two parts are the date-time)
            if len(parts) >= 3:
                timestamp = '-'.join(parts[-2:])
            else:
                return [filename]  # Keep unchanged if format is incorrect
            # Extract the prefix (everything before the timestamp)
            prefix = '-'.join(parts[:-2])  # Get all parts before the timestamp
            prefix_parts = prefix.split('_')  # Split by '_'

            # If multiple prefixes exist, return them separately with timestamp
            if len(prefix_parts) > 1:
                return [f"{part}-{timestamp}" for part in prefix_parts]
            return [filename]

        # Apply function and explode into new rows

        # creating column for subject names
        if 'Cohort_1_2' not in directory_path: 
            self.df['file name'] = self.df['file name'].apply(expand_filenames)
            self.df = self.df.explode('file name', ignore_index=True)
        self.df['trial'] = self.df.apply(lambda row: find_matching_trial(row['file name']), axis=1)
        self.df['subject_name'] = self.df['file name'].str.split('-').str[0]
        self.df['sound cues'] = self.df['trial'].apply(lambda x: x.rtc_events.get('sound cues', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.df['port entries'] = self.df['trial'].apply(lambda x: x.rtc_events.get('port entries', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.df['sound cues onset'] = self.df['sound cues'].apply(lambda x: x.onset_times if x else None)
        self.df['port entries onset'] = self.df['port entries'].apply(lambda x: x.onset_times if x else None)
        self.df['port entries offset'] = self.df['port entries'].apply(lambda x: x.offset_times if x else None)

    '''********************************** LICKS **********************************'''
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

        for index, row in df.iterrows():
            
            sound_cues_onsets = row['   sound cues onset']
            port_entries_onsets = row['port entries onset']
            port_entries_offsets = row['port entries offset']

            first_licks_per_row = []

            for sc_onset in sound_cues_onsets:
                threshold_time = sc_onset + 4

                # First, check for an ongoing lick:
                ongoing_licks_indices = np.where(
                    (port_entries_onsets < threshold_time) & (port_entries_offsets > threshold_time)
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



    def compute_EI_DA(self, df=None, pre_time=4, post_time=10):
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
            sound_cues = row['sound cues onset']
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

    def compute_standard_DA(self, df=None, pre_time=4, post_time=10):
        """
        Computes standard DA responses (no baseline subtraction) for each trial.
        
        Two sets of responses are computed:
        1. Tone Event: Data extracted in a window from -pre_time to +post_time around the sound cue onset.
        2. Lick Event: Data extracted in a window from 0 to +post_time around the lick onset.
        
        The function interpolates each extracted segment onto a common time axis.
        """
        import numpy as np

        if df is None:
            df = self.df

        # Determine the smallest sampling interval (dt) across all trials
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

        # Define the common time axes
        common_tone_time_axis = np.arange(-pre_time, post_time, min_dt)  # For sound cue
        common_lick_time_axis = np.arange(0, post_time, min_dt)           # For lick event

        # Lists to store results for each trial
        tone_zscores_all = []
        tone_times_all = []
        lick_zscores_all = []
        lick_times_all = []

        # Process each trial (each row in the dataframe)
        for _, row in df.iterrows():
            trial_obj = row['trial']
            timestamps = np.array(trial_obj.timestamps)
            # Here we assume the DA signal is still in trial_obj.zscore (even if it's not baseline-corrected)
            zscore = np.array(trial_obj.zscore)

            # Get event times
            sound_cues = row['sound cues onset']
            lick_cues  = row['first_lick_after_sound_cue']

            # ----- Tone Event (Sound Cue) -----
            tone_zscores_this_trial = []
            tone_times_this_trial = []
            if len(sound_cues) == 0:
                tone_zscores_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                tone_times_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
            else:
                for sound_event in sound_cues:
                    window_start = sound_event - pre_time
                    window_end   = sound_event + post_time
                    mask = (timestamps >= window_start) & (timestamps <= window_end)
                    if not np.any(mask):
                        tone_zscores_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                        tone_times_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                        continue

                    # Get the signal and relative time (without baseline subtraction)
                    rel_time = timestamps[mask] - sound_event
                    signal   = zscore[mask]
                    # Interpolate onto the common tone time axis
                    interp_signal = np.interp(common_tone_time_axis, rel_time, signal)
                    tone_zscores_this_trial.append(interp_signal)
                    tone_times_this_trial.append(common_tone_time_axis.copy())

            # ----- Lick Event -----
            lick_zscores_this_trial = []
            lick_times_this_trial = []
            if len(lick_cues) == 0:
                lick_zscores_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                lick_times_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
            else:
                for lick_event in lick_cues:
                    window_start = lick_event
                    window_end   = lick_event + post_time
                    mask = (timestamps >= window_start) & (timestamps <= window_end)
                    if not np.any(mask):
                        lick_zscores_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                        lick_times_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                        continue

                    rel_time = timestamps[mask] - lick_event
                    signal   = zscore[mask]
                    # Interpolate onto the common lick time axis
                    interp_signal = np.interp(common_lick_time_axis, rel_time, signal)
                    lick_zscores_this_trial.append(interp_signal)
                    lick_times_this_trial.append(common_lick_time_axis.copy())

            # Append results for this trial
            tone_zscores_all.append(tone_zscores_this_trial)
            tone_times_all.append(tone_times_this_trial)
            lick_zscores_all.append(lick_zscores_this_trial)
            lick_times_all.append(lick_times_this_trial)

        # Store the results in the dataframe
        df['Tone Event_Time_Axis'] = tone_times_all
        df['Tone Event_Zscore']    = tone_zscores_all
        df['Lick Event_Time_Axis'] = lick_times_all
        df['Lick Event_Zscore']    = lick_zscores_all

                
    def compute_da_metrics(self, df=None, bout_duration=4):
        """
        Computes DA metrics for both tone and lick events in a 0→4 s window:
        - AUC
        - Max Peak
        - Time of Max Peak
        - Mean Z-score

        No offsets or adaptive logic is used. For each event (tone or lick),
        the code extracts timestamps in [event_time, event_time + bout_duration]
        and calculates the above metrics.
        """
        if df is None:
            df = self.df  # Default to self.df if not provided

        # Prepare lists to store results for each row
        tone_auc_all = []
        tone_max_peak_all = []
        tone_peak_time_all = []
        tone_mean_z_all = []

        lick_auc_all = []
        lick_max_peak_all = []
        lick_peak_time_all = []
        lick_mean_z_all = []

        for _, row in df.iterrows():
            trial_obj = row['trial']
            if not trial_obj:
                # If there's no valid trial object, append NaNs and continue
                tone_auc_all.append([np.nan])
                tone_max_peak_all.append([np.nan])
                tone_peak_time_all.append([np.nan])
                tone_mean_z_all.append([np.nan])

                lick_auc_all.append([np.nan])
                lick_max_peak_all.append([np.nan])
                lick_peak_time_all.append([np.nan])
                lick_mean_z_all.append([np.nan])
                continue

            # Get timestamps and zscore from the trial
            timestamps = np.array(trial_obj.timestamps)
            zscore = np.array(trial_obj.zscore)

            # Tone onsets
            tone_onsets = row.get('sound cues onset', [])
            if tone_onsets is None:
                tone_onsets = []

            # Lick onsets
            lick_onsets = row.get('first_lick_after_sound_cue', [])
            if lick_onsets is None:
                lick_onsets = []

            # ============== Tone Metrics ==============
            tone_auc = []
            tone_max_peak = []
            tone_peak_time = []
            tone_mean_z = []

            for onset in tone_onsets:
                window_start = onset
                window_end = onset + bout_duration

                mask = (timestamps >= window_start) & (timestamps <= window_end)
                if not np.any(mask):
                    # If no data in the window, store NaNs
                    tone_auc.append(np.nan)
                    tone_max_peak.append(np.nan)
                    tone_peak_time.append(np.nan)
                    tone_mean_z.append(np.nan)
                    continue

                window_ts = timestamps[mask]
                window_z = zscore[mask]

                # Compute metrics
                auc_val = np.trapz(window_z, window_ts)
                max_idx = np.argmax(window_z)
                max_val = window_z[max_idx]
                max_time = window_ts[max_idx]
                mean_val = np.mean(window_z)

                tone_auc.append(auc_val)
                tone_max_peak.append(max_val)
                tone_peak_time.append(max_time)
                tone_mean_z.append(mean_val)

            # ============== Lick Metrics ==============
            lick_auc = []
            lick_max_peak = []
            lick_peak_time = []
            lick_mean_z = []

            for onset in lick_onsets:
                window_start = onset
                window_end = onset + bout_duration

                mask = (timestamps >= window_start) & (timestamps <= window_end)
                if not np.any(mask):
                    lick_auc.append(np.nan)
                    lick_max_peak.append(np.nan)
                    lick_peak_time.append(np.nan)
                    lick_mean_z.append(np.nan)
                    continue

                window_ts = timestamps[mask]
                window_z = zscore[mask]

                # Compute metrics
                auc_val = np.trapz(window_z, window_ts)
                max_idx = np.argmax(window_z)
                max_val = window_z[max_idx]
                max_time = window_ts[max_idx]
                mean_val = np.mean(window_z)

                lick_auc.append(auc_val)
                lick_max_peak.append(max_val)
                lick_peak_time.append(max_time)
                lick_mean_z.append(mean_val)

            # Append results for this row
            tone_auc_all.append(tone_auc)
            tone_max_peak_all.append(tone_max_peak)
            tone_peak_time_all.append(tone_peak_time)
            tone_mean_z_all.append(tone_mean_z)

            lick_auc_all.append(lick_auc)
            lick_max_peak_all.append(lick_max_peak)
            lick_peak_time_all.append(lick_peak_time)
            lick_mean_z_all.append(lick_mean_z)

        # Store in DataFrame columns
        df['Tone AUC'] = tone_auc_all
        df['Tone Max Peak'] = tone_max_peak_all
        df['Tone Time of Max Peak'] = tone_peak_time_all
        df['Tone Mean Z-score'] = tone_mean_z_all

        df['Lick AUC'] = lick_auc_all
        df['Lick Max Peak'] = lick_max_peak_all
        df['Lick Time of Max Peak'] = lick_peak_time_all
        df['Lick Mean Z-score'] = lick_mean_z_all

        return df
