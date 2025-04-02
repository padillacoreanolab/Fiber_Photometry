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
from matplotlib.colors import LinearSegmentedColormap

class Rtc(Experiment):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)
        self.port_bnc = {}  # Reset trials to avoid loading from parent class
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
            del self.trials[trial_folder]
            self.port_bnc[trial_folder1] = 3
            self.port_bnc[trial_folder2] = 2


    def rtc_processing(self, time_segments_to_remove=None):
        """
        Unified processing for RTC recordings that handles both unisubject and multisubject trials.

        For each trial:
        1. Optionally remove designated time segments.
        2. Remove initial LED artifact.
        3. Highpass filter to remove baseline drift.
        4. Align channels, compute dFF, determine baseline period.
        5. Compute standard z-score and verify the signal.
        6. Reassign behavior channels for tone and port entries.
            - For multisubject, use self.port_bnc to decide whether to use PC3_ (port value 3) or PC2_ (port value 2).
            - For unisubject, try PC2_ first, then PC3_.
        7. Remove the first behavior entry (if it is not counting).
        8. Filter port entries so that only those after the first sound cue remain.
        9. Combine consecutive behaviors.
        """
        for trial_folder, trial in self.trials.items():
            # (Optional) Remove time segments if provided.
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Processing trial {trial_folder}...")

            # ----- Preprocessing Steps -----
            trial.remove_initial_LED_artifact(t=30)
            trial.highpass_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            baseline_start, baseline_end = trial.find_baseline_period()
            trial.compute_zscore(method='standard')
            trial.verify_signal()

            # ----- Reassign Behavior Channels -----
            # Sound cues always come from PC0_
            trial.behaviors1['sound cues'] = trial.behaviors1.pop('PC0_')

            # Determine if this trial is multisubject.
            if trial_folder in self.port_bnc:
                # Multisubject: use port info to select the proper channel.
                port_val = self.port_bnc[trial_folder]
                if port_val == 3:
                    trial.behaviors1['port entries'] = trial.behaviors1.pop('PC3_')
                elif port_val == 2:
                    trial.behaviors1['port entries'] = trial.behaviors1.pop('PC2_')
                else:
                    print(f"Warning: Unexpected port value ({port_val}) for trial {trial_folder}")
            else:
                # Unisubject: try PC2_ first; if not available, try PC3_.
                if 'PC2_' in trial.behaviors1:
                    trial.behaviors1['port entries'] = trial.behaviors1.pop('PC2_')
                elif 'PC3_' in trial.behaviors1:
                    trial.behaviors1['port entries'] = trial.behaviors1.pop('PC3_')
                else:
                    print(f"Warning: No port entries channel found for trial {trial_folder}")

            # ----- Post-Processing of Behaviors -----
            # Remove the first (non-counting) entry for both behaviors.
            trial.behaviors1['sound cues'].onset_times = trial.behaviors1['sound cues'].onset[1:]
            trial.behaviors1['sound cues'].offset_times = trial.behaviors1['sound cues'].offset[1:]
            trial.behaviors1['port entries'].onset_times = trial.behaviors1['port entries'].onset[1:]
            trial.behaviors1['port entries'].offset_times = trial.behaviors1['port entries'].offset[1:]

            # Keep only port entries that occur after the first sound cue.
            port_onset = np.array(trial.behaviors1['port entries'].onset_times)
            port_offset = np.array(trial.behaviors1['port entries'].offset_times)
            first_tone = trial.behaviors1['sound cues'].onset_times[0]
            indices = np.where(port_onset >= first_tone)[0]
            trial.behaviors1['port entries'].onset_times = port_onset[indices].tolist()
            trial.behaviors1['port entries'].offset_times = port_offset[indices].tolist()
    """*********************************COMBINE BEHAVIORS AND COHORTS***********************************"""
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

    def combining_cohorts(self, df1):
        df_combined = pd.concat([self.df, df1], ignore_index=True)
        # Filter rows where 'subject_name' is either 'n4', 'n7', or other specified 'n' values
        # List of subject names to remove
        subjects_to_remove = ["n4", "n3", "n2", "n1"]

        # Remove rows where 'subject_names' are in the list
        df_combined = df_combined[~df_combined['subject_name'].isin(subjects_to_remove)]

        # Display the result
        print(df_combined)
        
        self.df = df_combined

    def compute_event_induced_DA(self, df=None, cue_type='sound cues onset', pre_time=4, post_time=10):
        """
        Computes the event-induced DA of a behavior by taking the 4 seconds before the onset of the behavior and normalizing the rest
        of the signal to it.
        """
        if df is None:
            df = self.df
        min_dt = np.inf
        for _, row in df.iterrows():
            trial_obj = row['trial']
            timestamps = np.array(trial_obj.timestamps)
            if len(timestamps) > 1:
                dt = np.min(np.diff(timestamps))  # Find the smallest sampling interval
                min_dt = min(min_dt, dt)

        if min_dt == np.inf:
            print("No valid timestamps found to determine dt.")
            return

        # Define a single global time axis for all trials
        common_time_axis = np.arange(-pre_time, post_time, min_dt)
        event_zscores = []
        event_time_list = []

        # Process each row in the dataframe
        for _, row in df.iterrows():
            trial_obj = row['trial']
            cues = row[cue_type]  # Event start times

            # Convert to numpy arrays
            timestamps = np.array(trial_obj.timestamps)
            zscore = np.array(trial_obj.zscore)

            if len(cues) == 0:
                print(f"Warning: No sound cues for trial {trial_obj}")
                event_zscores.append([np.full(common_time_axis.shape, np.nan)])
                event_time_list.append([np.full(common_time_axis.shape, np.nan)])
                continue

            trial_event_zscores = []
            trial_event_times = []

            # Process each event start time
            for event_start in cues:
                window_start = event_start - pre_time
                window_end = event_start + post_time

                # Find relevant indices
                mask = (timestamps >= window_start) & (timestamps <= window_end)
                if not np.any(mask):
                    print(f"Warning: No timestamps found for event at {event_start}")
                    trial_event_zscores.append(np.full(common_time_axis.shape, np.nan))
                    trial_event_times.append(np.full(common_time_axis.shape, np.nan))
                    continue

                # Time relative to event onset
                rel_time = timestamps[mask] - event_start
                signal = zscore[mask]

                # Compute baseline (pre-event mean)
                pre_mask = rel_time < 0
                baseline = np.nanmean(signal[pre_mask]) if np.any(pre_mask) else 0

                # Baseline-correct the signal
                corrected_signal = signal - baseline

                # Interpolate onto the common time axis
                interp_signal = np.interp(common_time_axis, rel_time, corrected_signal)

                trial_event_zscores.append(interp_signal)
                trial_event_times.append(common_time_axis.copy())  # Store a copy for each event

            event_zscores.append(trial_event_zscores)
            event_time_list.append(trial_event_times)

        # Store results in the dataframe
        df['Tone Event_Time_Axis'] = event_time_list  # Now structured identically to Event_Zscore
        df['Tone Event_Zscore'] = event_zscores

    def compute_lick_ei_DA(self, df=None, pre_time=4, post_time=10):
        if df is None:
            df = self.df
        min_dt = np.inf

        # Determine the smallest time step (min_dt) across all trials
        for _, row in df.iterrows():
            trial_obj = row['trial']
            timestamps = np.array(trial_obj.timestamps)
            if len(timestamps) > 1:
                dt = np.min(np.diff(timestamps))
                min_dt = min(min_dt, dt)

        if min_dt == np.inf:
            print("No valid timestamps found to determine dt.")
            return

        event_zscores = []
        event_time_list = []

        for _, row in df.iterrows():
            trial_obj = row['trial']
            cues = row['first_lick_after_sound_cue']
            sound_cues = row['sound cues onset']

            timestamps = np.array(trial_obj.timestamps)
            zscore = np.array(trial_obj.zscore)

            if len(cues) == 0 or len(sound_cues) == 0:
                print(f"Warning: No valid cues for trial {trial_obj}")
                event_zscores.append([np.full((1,), np.nan)])
                event_time_list.append([np.full((1,), np.nan)])
                continue

            trial_event_zscores = []
            trial_event_times = []

            for event_start in cues:
                try:
                    idx = list(cues).index(event_start)
                    cue_time = sound_cues[idx]
                    lick_start = cues[idx]
                except ValueError:
                    print(f"Warning: Could not find corresponding sound cue for event {event_start}")
                    continue

                # Time window relative to lick onset
                window_start = lick_start - pre_time
                window_end = lick_start + post_time

                # Print to debug the timestamp coverage
                print(f"Trial {trial_obj}: Lick Start = {lick_start}, Window Start = {window_start}, Window End = {window_end}")

                start_idx = np.searchsorted(timestamps, window_start, side="left")
                end_idx = np.searchsorted(timestamps, window_end, side="right")

                rel_time = timestamps[start_idx:end_idx] - lick_start
                signal = zscore[start_idx:end_idx]
                print(len(signal))

                # Print to debug time range
                print(f"Relative Time Min = {np.min(rel_time)}, Max = {np.max(rel_time)}")

                pre_mask = (timestamps >= (cue_time - pre_time)) & (timestamps < cue_time)
                baseline = np.nanmean(zscore[pre_mask]) if np.any(pre_mask) else 0
                corrected_signal = signal - baseline

                common_time_axis = np.arange(-pre_time, post_time + min_dt, min_dt)
                print(len(common_time_axis))

                # Debug common time axis
                print(f"Common Time Axis Min = {np.min(common_time_axis)}, Max = {np.max(common_time_axis)}")

                interp_signal = np.interp(common_time_axis, rel_time, corrected_signal, left=np.nan, right=np.nan)

                trial_event_zscores.append(interp_signal)
                trial_event_times.append(common_time_axis.copy())

            event_zscores.append(trial_event_zscores)
            event_time_list.append(trial_event_times)

        df['Lick Event_Time_Axis'] = event_time_list
        df['Lick Event_Zscore'] = event_zscores

    def compute_tone_da_metrics(self, df=None, mode='standard'):
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
                end_time = cue + 4  # Default window end

                # Extract initial window
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                window_ts = timestamps[mask]
                window_z = zscores[mask]
                
                if len(window_ts) < 2:
                    computed_metrics.append({"AUC": np.nan, "Max Peak": np.nan, "Time of Max Peak": np.nan, "Mean Z-score": np.nan, "Adjusted End": np.nan})
                    continue
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
        def compute_ei(df):
            # EI mode: use event-induced data.
            if 'Tone Event_Time_Axis' not in df.columns or 'Tone Event_Zscore' not in df.columns:
                print("Event-induced data not found in behaviors. Please run compute_event_induced_DA() first.")
                return df

            # Lists to store computed arrays for each row
            mean_zscores_all = []
            auc_values_all = []
            max_peaks_all = []
            peak_times_all = []

            for i, row in df.iterrows():
                # Extract all trials (lists) within the row
                time_axes = np.array(row['Tone Event_Time_Axis'])  # 1D array
                event_zscores = np.array(row['Tone Event_Zscore'])  # 2D array (list of lists)

                # Lists to store per-trial metrics
                mean_zscores = []
                auc_values = []
                max_peaks = []
                peak_times = []

                for time_axis, event_zscore in zip(time_axes, event_zscores):
                    time_axis = np.array(time_axis)
                    event_zscore = np.array(event_zscore)

                    # Mask for time_axis >= 0
                    mask = (time_axis >= 0) & (time_axis <= 4)
                    if not np.any(mask):
                        mean_zscores.append(np.nan)
                        auc_values.append(np.nan)
                        max_peaks.append(np.nan)
                        peak_times.append(np.nan)
                        continue

                    final_time = time_axis[mask]
                    final_z = event_zscore[mask]

                    # Compute metrics
                    mean_z = np.mean(final_z)
                    auc = np.trapz(final_z, final_time)
                    max_idx = np.argmax(final_z)
                    max_peak = final_z[max_idx]
                    peak_time = final_time[max_idx]

                    # Append results for this trial
                    mean_zscores.append(mean_z)
                    auc_values.append(auc)
                    max_peaks.append(max_peak)
                    peak_times.append(peak_time)

                # Append lists of results for this row
                mean_zscores_all.append(mean_zscores)
                auc_values_all.append(auc_values)
                max_peaks_all.append(max_peaks)
                peak_times_all.append(peak_times)

            # Store computed values as lists inside new DataFrame columns
            df['Mean Z-score EI'] = mean_zscores_all
            df['AUC EI'] = auc_values_all
            df['Max Peak EI'] = max_peaks_all
            df['Time of Max Peak EI'] = peak_times_all
            return df

        if mode == 'standard':
            # Apply function across all trials
            df["computed_metrics"] = df.apply(
                lambda row: compute_da_metrics_for_trial(row["trial"], row["sound cues onset"]), axis=1)
            df["Tone AUC"] = df["computed_metrics"].apply(lambda x: [item.get("AUC", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Max Peak"] = df["computed_metrics"].apply(lambda x: [item.get("Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Time of Max Peak"] = df["computed_metrics"].apply(lambda x: [item.get("Time of Max Peak", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Mean Z-score"] = df["computed_metrics"].apply(lambda x: [item.get("Mean Z-score", np.nan) for item in x] if isinstance(x, list) else np.nan)
            df["Tone Adjusted End"] = df["computed_metrics"].apply(lambda x: [item.get("Adjusted End", np.nan) for item in x] if isinstance(x, list) else np.nan)
            # Drop the "computed_metrics" column if it's no longer needed
            df.drop(columns=["computed_metrics"], inplace=True)
        else:
            compute_ei(df)
                
    def compute_lick_da_metrics(self, df=None, mode='standard'):
        if df is None:
            df = self.df
        """Iterate through trials in the dataframe and compute DA metrics for each lick trial."""
        def compute_da_metrics_for_lick(trial_obj, first_lick_after_tones, closest_port_entry_offsets, 
                                        use_adaptive=False, peak_fall_fraction=0.5, allow_bout_extension=False,
                                        use_fractional=False, max_bout_duration=15):
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
        def compute_ei(df):
            # EI mode: use event-induced data.
            if 'Lick Event_Time_Axis' not in df.columns or 'Lick Event_Zscore' not in df.columns:
                print("Event-induced data not found in behaviors. Please run compute_event_induced_DA() first.")
                return df

            # Lists to store computed arrays for each row
            mean_zscores_all = []
            auc_values_all = []
            max_peaks_all = []
            peak_times_all = []

            for i, row in df.iterrows():
                time_axes = [np.array(x) for x in row['Lick Event_Time_Axis']]
                event_zscores = [np.array(x) for x in row['Lick Event_Zscore']]

                # Lists to store per-trial metrics
                mean_zscores = []
                auc_values = []
                max_peaks = []
                peak_times = []

                for time_axis, event_zscore in zip(time_axes, event_zscores):
                    time_axis = np.array(time_axis)
                    event_zscore = np.array(event_zscore)

                    # Mask for time_axis >= 0
                    mask = time_axis >= 0
                    if not np.any(mask):
                        mean_zscores.append(np.nan)
                        auc_values.append(np.nan)
                        max_peaks.append(np.nan)
                        peak_times.append(np.nan)
                        continue

                    final_time = time_axis[mask]
                    final_z = event_zscore[mask]

                    # Compute metrics
                    mean_z = np.mean(final_z)
                    auc = np.trapz(final_z, final_time)
                    max_idx = np.argmax(final_z)
                    max_peak = final_z[max_idx]
                    peak_time = final_time[max_idx]

                    # Append results for this trial
                    mean_zscores.append(mean_z)
                    auc_values.append(auc)
                    max_peaks.append(max_peak)
                    peak_times.append(peak_time)

                # Append lists of results for this row
                mean_zscores_all.append(mean_zscores)
                auc_values_all.append(auc_values)
                max_peaks_all.append(max_peaks)
                peak_times_all.append(peak_times)

            # Store computed values as lists inside new DataFrame columns
            df['Lick Mean Z-score EI'] = mean_zscores_all
            df['Lick AUC EI'] = auc_values_all
            df['Lick Max Peak EI'] = max_peaks_all
            df['Lick Time of Max Peak EI'] = peak_times_all
            return df

        if mode == 'standard':
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
            df.drop(columns=["lick_computed_metrics"], inplace=True)
        else:
            df = compute_ei(df)

    """*********************************************FIND MEANS*********************************************"""
    def find_means(self, df):
        if df is None:
            df = self.df
        df["Lick AUC Mean"] = df["Lick AUC"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Lick Max Peak Mean"] = df["Lick Max Peak"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Lick Mean Z-score Mean"] = df["Lick Mean Z-score"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone AUC Mean"] = df["Tone AUC"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone Max Peak Mean"] = df["Tone Max Peak"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone Mean Z-score Mean"] = df["Tone Mean Z-score"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Lick AUC Mean EI"] = df["Lick AUC EI"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Lick Max Peak Mean EI"] = df["Lick Max Peak EI"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Lick Mean Z-score Mean EI"] = df["Lick Mean Z-score EI"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone AUC Mean EI"] = df["AUC EI"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone Max Peak Mean EI"] = df["Max Peak EI"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)
        df["Tone Mean Z-score Mean EI"] = df["Mean Z-score EI"].apply(lambda x: np.mean(x) if isinstance(x, list) else np.nan)

    def find_overall_mean(self, df):
        if df is None:
            df = self.df
        
        # Function to compute mean for numerical values, preserving first for categorical ones
        def mean_arrays(group):
            result = {}
            result['Rank'] = group['Rank'].iloc[0]  # Keeps first value
            result['Cage'] = group['Cage'].iloc[0]  # Keeps first value
            result['Tone Event_Time_Axis'] = group['Tone Event_Time_Axis'].iloc[0]  # Time axis should be the same

            # Compute mean for scalar numerical values
            numerical_cols = [
                'Lick AUC Mean', 'Lick Max Peak Mean', 'Lick Mean Z-score Mean',
                'Tone AUC Mean', 'Tone Max Peak Mean', 'Tone Mean Z-score Mean',
                'Lick AUC First', 'Lick AUC Last', 'Lick Max Peak First', 'Lick Max Peak Last',
                'Lick Mean Z-score First', 'Lick Mean Z-score Last',
                'Tone AUC First', 'Tone AUC Last', 'Tone Max Peak First', 'Tone Max Peak Last',
                'Tone Mean Z-score First', 'Tone Mean Z-score Last',
                "Lick AUC Mean EI", "Lick Max Peak Mean EI", "Lick Mean Z-score Mean EI",
                "Tone AUC Mean EI", "Tone Max Peak Mean EI", "Tone Mean Z-score Mean EI",
                "Lick AUC EI First", "Lick Max Peak EI First", "Tone AUC EI First",
                "Tone Max Peak EI First", "Tone Mean Z-score EI First",
                "Lick AUC EI Last", "Lick Max Peak EI Last", "Tone AUC EI Last",
                "Tone Max Peak EI Last", "Tone Mean Z-score EI Last"
            ]
            
            for col in numerical_cols:
                result[col] = group[col].mean()

            # Compute element-wise mean for array columns
            array_cols = ["Tone Event_Zscore", "Lick Event_Zscore"]
            for col in array_cols:
                stacked_arrays = np.vstack(group[col].values)  # Stack into 2D array
                result[col] = np.mean(stacked_arrays, axis=0)  # Element-wise mean

            return pd.Series(result)

        df_mean = df.groupby('subject_name').apply(mean_arrays).reset_index()
        
        return df_mean