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
            
            sound_cues_onsets = row['sound cues onset']
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


    '''********************************** DOPAMINE AND METRICS **********************************'''
    def compute_EI_DA(self, df=None, pre_time=4, post_time=10):
        """
        Computes event-induced DA responses for tone (sound cue) and lick.
        For the lick event, it uses the tone-based baseline.
        """
        if df is None:
            df = self.df

        # Determine global dt from all trials.
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

        # Define common time axes.
        common_tone_time_axis = np.arange(-pre_time, post_time, min_dt)   # For tone events.
        common_lick_time_axis = np.arange(0, post_time, min_dt)           # For lick events.

        tone_zscores_all = []
        tone_times_all = []
        lick_zscores_all = []
        lick_times_all = []

        for _, row in df.iterrows():
            trial_obj = row['trial']
            timestamps = np.array(trial_obj.timestamps)
            zscore = np.array(trial_obj.zscore)

            # Get sound cue and lick event times.
            sound_cues = row['sound cues onset']   # tone event onsets
            lick_cues  = row['first_lick_after_sound_cue']  # lick event onsets

            # Process tone events.
            tone_z_this_trial = []
            tone_time_this_trial = []
            if len(sound_cues) == 0:
                tone_z_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                tone_time_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
            else:
                for sound_event in sound_cues:
                    window_start = sound_event - pre_time
                    window_end   = sound_event + post_time

                    mask = (timestamps >= window_start) & (timestamps <= window_end)
                    if not np.any(mask):
                        tone_z_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                        tone_time_this_trial.append(np.full(common_tone_time_axis.shape, np.nan))
                        continue

                    rel_time = timestamps[mask] - sound_event
                    signal   = zscore[mask]

                    pre_mask = rel_time < 0
                    baseline = np.nanmean(signal[pre_mask]) if np.any(pre_mask) else 0

                    corrected_signal = signal - baseline
                    interp_signal = np.interp(common_tone_time_axis, rel_time, corrected_signal)

                    tone_z_this_trial.append(interp_signal)
                    tone_time_this_trial.append(common_tone_time_axis.copy())

            # Process lick events.
            lick_z_this_trial = []
            lick_time_this_trial = []
            if len(lick_cues) == 0:
                lick_z_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                lick_time_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
            else:
                for i, lick_event in enumerate(lick_cues):
                    # Check if the lick event is valid.
                    if lick_event is None:
                        print(f"Warning: Trial {trial_obj} has a None lick event; skipping this lick event.")
                        continue

                    # Use the corresponding sound cue (if available) for baseline.
                    if i < len(sound_cues):
                        sound_event_for_baseline = sound_cues[i]
                    else:
                        sound_event_for_baseline = None

                    if sound_event_for_baseline is not None:
                        baseline_start = sound_event_for_baseline - pre_time
                        baseline_end   = sound_event_for_baseline
                        baseline_mask = (timestamps >= baseline_start) & (timestamps <= baseline_end)
                        if np.any(baseline_mask):
                            baseline_val = np.nanmean(zscore[baseline_mask])
                        else:
                            baseline_val = 0
                    else:
                        baseline_val = 0

                    window_start = lick_event  # Lick event marks t=0 for the lick.
                    window_end   = lick_event + post_time

                    mask = (timestamps >= window_start) & (timestamps <= window_end)
                    if not np.any(mask):
                        lick_z_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                        lick_time_this_trial.append(np.full(common_lick_time_axis.shape, np.nan))
                        continue

                    rel_time = timestamps[mask] - lick_event
                    signal   = zscore[mask]

                    corrected_signal = signal - baseline_val
                    interp_signal = np.interp(common_lick_time_axis, rel_time, corrected_signal)

                    lick_z_this_trial.append(interp_signal)
                    lick_time_this_trial.append(common_lick_time_axis.copy())

            tone_zscores_all.append(tone_z_this_trial)
            tone_times_all.append(tone_time_this_trial)
            lick_zscores_all.append(lick_z_this_trial)
            lick_times_all.append(lick_time_this_trial)

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


    '''********************************** PSTH CODE  **********************************'''
    def plot_specific_event_psth(self, event_type, event_index, directory_path, brain_region, y_min, y_max, df=None, bin_size=100):
        """
        Plots the PSTH (mean and SEM) for a specific event bout (0→4 s after event onset)
        across trials, using the same averaging logic as for a bout response.

        Parameters:
            event_type (str): The event type (e.g. 'Tone' or 'Lick').
            event_index (int): 1-indexed event number to plot.
            directory_path (str or None): Directory to save the plot (if None, the plot is not saved).
            brain_region (str): Brain region ('mPFC' or other) to filter subjects.
            y_min (float): Lower bound of the y-axis.
            y_max (float): Upper bound of the y-axis.
            df (DataFrame, optional): DataFrame to use (defaults to self.df).
            bin_size (int, optional): Bin size for downsampling.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        if df is None:
            df = self.df

        def split_by_subject(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)
        idx = event_index - 1
        print(f"[DEBUG] PSTH: Plotting {event_type} event index {event_index} (0-indexed {idx}) for brain region {brain_region}")

        selected_traces = []
        for i, row in df.iterrows():
            event_z_list = row.get(f'{event_type} Event_Zscore', [])
            if isinstance(event_z_list, list) and len(event_z_list) > idx:
                trace = np.array(event_z_list[idx])
                print(f"[DEBUG] Row {i}, subject {row['subject_name']}: trace shape {trace.shape}, first 5 values: {trace[:5]}")
                selected_traces.append(trace)
        if len(selected_traces) == 0:
            print(f"No trials have an event at index {event_index} for {event_type}.")
            return

        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][idx]
        selected_traces = np.array(selected_traces)

        mean_trace = np.mean(selected_traces, axis=0)
        sem_trace = np.std(selected_traces, axis=0) / np.sqrt(selected_traces.shape[0])
        
        if hasattr(self, 'downsample_data'):
            mean_trace, downsampled_time_axis = self.downsample_data(mean_trace, common_time_axis, bin_size)
            sem_trace, _ = self.downsample_data(sem_trace, common_time_axis, bin_size)
        else:
            downsampled_time_axis = common_time_axis

        # --- Debug: Check values in the 4–10 s window ---
        post_window = (downsampled_time_axis >= 4) & (downsampled_time_axis <= 10)
        post_vals = mean_trace[post_window]
        print("[DEBUG] PSTH: Final mean trace shape:", mean_trace.shape)
        print("[DEBUG] PSTH: Final mean trace first 10 values:", mean_trace[:10])
        print("[DEBUG] PSTH: 4–10 s window values, first 10:", post_vals[:10])
        print("[DEBUG] PSTH: Max in 4–10 s window:", np.max(post_vals))
        # --- End Debug ---

        # Choose trace color based on brain region.
        if brain_region == 'mPFC':
            trace_color = '#FFAF00'
        else:
            trace_color = '#15616F'

        plt.figure(figsize=(10, 6))
        plt.plot(downsampled_time_axis, mean_trace, color=trace_color, lw=3, label='Mean DA')
        plt.fill_between(downsampled_time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=trace_color, alpha=0.4, label='SEM')
        plt.axvline(0, color='black', linestyle='--', lw=2)
        plt.axvline(4, color='#FF69B4', linestyle='-', lw=2)

        # Set x-axis ticks to [-4, 0, 4, 10]
        plt.xlabel('Time from Tone Onset (s)', fontsize=30)
        plt.ylabel('Z-scored ΔF/F', fontsize=30)
        plt.title(f'{event_type} Event {event_index} PSTH', fontsize=30, pad=30)
        plt.ylim(y_min, y_max)
        plt.xticks([-4, 0, 4, 10], fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim(downsampled_time_axis[0], downsampled_time_axis[-1])
        # plt.legend(fontsize=30)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)

        if directory_path is not None:
            save_path = os.path.join(directory_path, f'{brain_region}_{event_type}_Event{event_index}_PSTH.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()





    def plot_event_index_heatmap(self, event_type, max_events, directory_path, brain_region, 
                             vmin, vmax, df=None, bin_size=125):
        """
        Plots a heatmap in which each row is the average DA trace (PSTH) for a given 
        event number (e.g., tone number) across subjects/trials, with tone #1 at the top.
        
        Parameters:
            event_type (str): The event type (e.g., 'Tone' or 'Lick').
            max_events (int): The maximum event number to include.
            directory_path (str or None): Directory to save the plot (if None, the plot is not saved).
            brain_region (str): Brain region used to filter subjects.
            vmin (float): Lower bound of the color scale (Z-score).
            vmax (float): Upper bound of the color scale (Z-score).
            df (DataFrame, optional): DataFrame to use (defaults to self.df).
            bin_size (int, optional): Bin size for downsampling.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        if df is None:
            df = self.df

        def split_by_subject(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)

        first_row = df.iloc[0]
        event_time_axes = first_row.get(f'{event_type} Event_Time_Axis', [])
        if not isinstance(event_time_axes, list) or len(event_time_axes) == 0:
            print(f"No {event_type} event time axes found.")
            return
        common_time_axis = event_time_axes[0]

        print(f"[DEBUG] Heatmap: Averaging {event_type} data for first {max_events} events in brain region {brain_region}")

        averaged_event_traces = []
        for event_idx in range(max_events):
            event_traces = []
            for i, row in df.iterrows():
                event_z_list = row.get(f'{event_type} Event_Zscore', [])
                if isinstance(event_z_list, list) and len(event_z_list) > event_idx:
                    trace = np.array(event_z_list[event_idx])
                    print(f"[DEBUG] Row {i}, subject {row['subject_name']}, event {event_idx+1}: trace shape {trace.shape}, first 5 values: {trace[:5]}")
                    event_traces.append(trace)
            if len(event_traces) > 0:
                avg_trace = np.mean(np.vstack(event_traces), axis=0)
                print(f"[DEBUG] Averaged trace for event {event_idx+1}: first 10 values: {avg_trace[:10]}")
            else:
                avg_trace = np.full(len(common_time_axis), np.nan)
            averaged_event_traces.append(avg_trace)

        heatmap_data = np.array(averaged_event_traces)

        if hasattr(self, 'downsample_data'):
            downsampled_data = []
            for row_trace in heatmap_data:
                down_row, _ = self.downsample_data(row_trace, common_time_axis, bin_size)
                downsampled_data.append(down_row)
            heatmap_data = np.array(downsampled_data)
            dummy = np.zeros(len(common_time_axis))
            _, downsampled_time_axis = self.downsample_data(dummy, common_time_axis, bin_size)
        else:
            downsampled_time_axis = common_time_axis

        # --- Debug: For event 14, print values in the 4–10 s window ---
        debug_event = 14  # 1-indexed event 14
        debug_idx = debug_event - 1
        if debug_idx < heatmap_data.shape[0]:
            event_trace = heatmap_data[debug_idx]
            post_window = (downsampled_time_axis >= 4) & (downsampled_time_axis <= 10)
            post_vals = event_trace[post_window]
            print(f"[DEBUG] Heatmap event {debug_event} (row {debug_idx+1}) mean trace, first 10 values: {event_trace[:10]}")
            print(f"[DEBUG] Heatmap event {debug_event} (row {debug_idx+1}) 4–10 s window, first 10 values: {post_vals[:10]}")
            print(f"[DEBUG] Heatmap event {debug_event} (row {debug_idx+1}) max in 4–10 s window: {np.max(post_vals)}")
        else:
            print(f"[DEBUG] There is no event {debug_event} in the heatmap data.")

        # Invert the y-axis so that tone 1 appears at the top.
        x_min, x_max = downsampled_time_axis[0], downsampled_time_axis[-1]
        extent = [x_min, x_max, max_events, 1]

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(heatmap_data, aspect='auto', 
                    cmap='inferno' if brain_region == 'mPFC' else 'viridis',
                    origin='upper', extent=extent, vmin=vmin, vmax=vmax)

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('Time from Tone Onset (s)', fontsize=30)
        ax.set_ylabel(f'{event_type} Number', fontsize=30)
        
        # Force x-axis ticks to be exactly [-4, 0, 4, 10]
        ax.set_xticks([-4, 0, 4, 10])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=30)
        
        # For the y-axis, we'll display 5 ticks (evenly spaced).
        yticks = np.linspace(1, max_events, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(round(t))}" for t in yticks], fontsize=30)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.axvline(0, color='white', linestyle='--', linewidth=2)
        ax.axvline(4, color='#FF69B4', linestyle='-', linewidth=2)
        ax.set_title(f'{brain_region} {event_type} Averaged Heatmap (Tone 1–{max_events})', 
                    fontsize=32, pad=30)

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label("Event Induced Z-score", fontsize=30)

        if directory_path is not None:
            save_name = f'{brain_region}_{event_type}_Heatmap_Averaged_1to{max_events}.png'
            save_path = os.path.join(directory_path, save_name)
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")

        plt.show()







    def downsample_data(self, data, time_axis, bin_size):
        """
        Downsamples a 1D data array and its corresponding time axis by averaging in bins.

        Parameters:
            data (1D numpy array): The data to be downsampled.
            time_axis (1D numpy array): The corresponding time axis.
            bin_size (int): The number of samples per bin.
        
        Returns:
            downsampled_data (1D numpy array): The downsampled data.
            downsampled_time_axis (1D numpy array): The downsampled time axis.
        """
        import numpy as np

        n = len(data)
        num_bins = n // bin_size  # Number of complete bins.
        if num_bins == 0:
            # If bin_size is larger than the data length, return original.
            return data, time_axis

        # Trim data to fit an integer number of bins.
        trimmed_length = num_bins * bin_size
        data_trimmed = data[:trimmed_length]
        time_trimmed = time_axis[:trimmed_length]

        # Reshape and compute the mean in each bin.
        downsampled_data = data_trimmed.reshape(num_bins, bin_size).mean(axis=1)
        downsampled_time_axis = time_trimmed.reshape(num_bins, bin_size).mean(axis=1)

        return downsampled_data, downsampled_time_axis


    def get_psth_mean_trace(self, event_type, event_index, df=None, bin_size=100):
        """
        Mimics the averaging logic from plot_specific_event_psth:
        For the given event_index (1-indexed), this function returns the mean
        trace across trials (downsampled if available).
        """
        import numpy as np

        if df is None:
            df = self.df

        idx = event_index - 1  # convert 1-indexed to 0-indexed
        selected_traces = []
        for i, row in df.iterrows():
            event_z_list = row.get(f'{event_type} Event_Zscore', [])
            if isinstance(event_z_list, list) and len(event_z_list) > idx:
                selected_traces.append(np.array(event_z_list[idx]))
        if len(selected_traces) == 0:
            print(f"No trials have an event at index {event_index} for {event_type}.")
            return None

        mean_trace = np.mean(np.array(selected_traces), axis=0)
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][idx]
        if hasattr(self, 'downsample_data'):
            mean_trace, _ = self.downsample_data(mean_trace, common_time_axis, bin_size)
        return mean_trace

    def get_heatmap_mean_trace(self, event_type, event_index, df=None, bin_size=125):
        """
        Mimics the averaging logic from plot_event_index_heatmap:
        For the given event_index (1-indexed), this function returns the mean trace
        (averaged across all trials) for that event. (Assumes all events share the same time axis.)
        """
        import numpy as np

        if df is None:
            df = self.df

        idx = event_index - 1  # convert 1-indexed to 0-indexed
        selected_traces = []
        for i, row in df.iterrows():
            event_z_list = row.get(f'{event_type} Event_Zscore', [])
            if isinstance(event_z_list, list) and len(event_z_list) > idx:
                selected_traces.append(np.array(event_z_list[idx]))
        if len(selected_traces) == 0:
            print(f"No trials have an event at index {event_index} for {event_type}.")
            return None

        avg_trace = np.mean(np.vstack(selected_traces), axis=0)
        # Note: In the heatmap function, the common time axis is taken from index 0.
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]
        if hasattr(self, 'downsample_data'):
            avg_trace, _ = self.downsample_data(avg_trace, common_time_axis, bin_size)
        return avg_trace

    def compare_arrays(self, arr1, arr2, tol=1e-6):
        """
        Compares two numpy arrays elementwise and prints out the maximum difference.
        Returns True if they are equal within tolerance, False otherwise.
        """
        import numpy as np
        if arr1 is None or arr2 is None:
            print("One of the arrays is None.")
            return False
        if arr1.shape != arr2.shape:
            print(f"Arrays have different shapes: {arr1.shape} vs {arr2.shape}")
            return False
        if np.allclose(arr1, arr2, atol=tol):
            print("Arrays are the same within the tolerance.")
            return True
        else:
            diff = np.abs(arr1 - arr2)
            max_diff = np.max(diff)
            print(f"Arrays differ. Maximum absolute difference: {max_diff:.6f}")
            print("First 10 differences:", diff.flatten()[:10])
            return False

    def debug_compare_event_14(self, event_type='Tone', event_index=14, df=None, bin_size=100):
        """
        Retrieves the mean event trace for a given event index (e.g., Tone 14)
        using both the PSTH and heatmap averaging logic, and then compares them.
        """
        if df is None:
            df = self.df

        print(f"Comparing {event_type} event {event_index} mean traces using PSTH and Heatmap logic:")

        psth_mean = self.get_psth_mean_trace(event_type, event_index, df, bin_size)
        heatmap_mean = self.get_heatmap_mean_trace(event_type, event_index, df, bin_size)

        print("[DEBUG] PSTH mean trace (first 10 values):", psth_mean[:10] if psth_mean is not None else "None")
        print("[DEBUG] Heatmap mean trace (first 10 values):", heatmap_mean[:10] if heatmap_mean is not None else "None")

        same = self.compare_arrays(psth_mean, heatmap_mean)
        if same:
            print("The arrays for event", event_index, "are the same.")
        else:
            print("The arrays for event", event_index, "differ.")
