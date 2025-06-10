# This file creates a child class of Experiment that is specific to RTC recordings. RTC recordings can either have one mouse or two mice in the same folder.
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

class RTC(Experiment):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path, RTC=True)
        self.port_bnc = {} 
        self.trials_df = pd.DataFrame()
        self.da_df = pd.DataFrame()
        self.load_rtc_trials() 


    '''********************************** Initial Processing  **********************************'''
    def load_rtc_trials(self):
        """
        Unified trial loader for RTC recordings.

        For each folder in self.experiment_folder_path:
        - If the folder name contains an underscore, it is assumed to be multisubject.
            Two Trial objects are created using different channel pairs, and unique keys are generated.
        - Otherwise, a single Trial object is created.
        """
        trial_folders = [
            folder for folder in os.listdir(self.experiment_folder_path)
            if os.path.isdir(os.path.join(self.experiment_folder_path, folder))
        ]
        
        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            
            # Split the folder name by underscore.
            parts = trial_folder.split('_')
            
            # If there's an underscore, assume multisubject.
            if len(parts) > 1:
                # Create two Trial objects with different channel pairs.
                trial_obj1 = Trial(trial_path, '_465A', '_405A')
                trial_obj2 = Trial(trial_path, '_465C', '_405C')
                
                # Extract subject identifiers.
                subject1 = parts[0]  # e.g., "nn3"
                # For the second subject, split the second part by dash to extract the subject ID.
                subject2 = parts[1].split('-')[0]  # e.g., from "nn4-250124-064620" get "nn4"
                
                # Reconstruct a common identifier from the remainder (if present).
                if '-' in parts[1]:
                    # Get everything after the first dash.
                    rest = parts[1].split('-', 1)[1]  # e.g., "250124-064620"
                    trial_key1 = f"{subject1}-{rest}"
                    trial_key2 = f"{subject2}-{rest}"
                else:
                    trial_key1 = subject1
                    trial_key2 = subject2
                
                # Store the trial objects using the generated keys.
                self.trials[trial_key1] = trial_obj1
                self.trials[trial_key2] = trial_obj2
                
                # Record port information for multisubject.
                self.port_bnc[trial_key1] = 2
                self.port_bnc[trial_key2] = 3
            else:
                # Unisubject recording: Create one Trial object.
                trial_obj = Trial(trial_path, '_465A', '_405A')
                self.trials[trial_folder] = trial_obj


    def rtc_processing(self):
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
        """
        for trial_folder, trial in self.trials.items():
            print(f"Processing trial {trial_folder}...")

            # ----- Preprocessing Steps -----
            trial.remove_initial_LED_artifact(t=30)
            trial.highpass_baseline_drift()  # Used specifically for RTC to not smooth
            trial.align_channels()
            trial.compute_dFF()
            # baseline_start, baseline_end = trial.find_baseline_period()
            trial.compute_zscore(method='standard')
            trial.verify_signal()

            # ----- Reassign Behavior Channels -----
            # Sound cues always come from PC0_
            trial.rtc_events['sound cues'] = trial.rtc_events.pop('PC0_')

            # Determine if this trial is multisubject.
            if trial_folder in self.port_bnc:
                # Multisubject: use port info to select the proper channel.
                port_val = self.port_bnc[trial_folder]
                if port_val == 3:
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC3_')
                elif port_val == 2: 
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC2_')
                else:
                    print(f"Warning: Unexpected port value ({port_val}) for trial {trial_folder}")
            else:
                # Unisubject: try PC2_ first; if not available, try PC3_.
                if 'PC2_' in trial.rtc_events:
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC2_')
                elif 'PC3_' in trial.rtc_events:
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC3_')
                else:
                    print(f"Warning: No port entries channel found for trial {trial_folder}")

            # ----- Post-Processing of Behaviors -----
            # Remove the first (non-counting) entry for both behaviors.
            trial.rtc_events['sound cues'].onset_times = trial.rtc_events['sound cues'].onset[1:]
            trial.rtc_events['sound cues'].offset_times = trial.rtc_events['sound cues'].offset[1:]
            trial.rtc_events['port entries'].onset_times = trial.rtc_events['port entries'].onset[1:]
            trial.rtc_events['port entries'].offset_times = trial.rtc_events['port entries'].offset[1:]
            
            valid_sound_cues = [t for t in trial.rtc_events['sound cues'].onset_times if t >= 200]
            trial.rtc_events['sound cues'].onset_times = valid_sound_cues



    def remove_specified_subjects(self):
        """
        Removes specified subjects not used in data analysis
        """
        # List of subject names to remove
        subjects_to_remove = ["n4", "n3", "n2", "n1", 'p4']

        # Remove rows where 'subject_names' are in the list
        df_combined = self.trials_df[~self.trials_df['subject_name'].isin(subjects_to_remove)]
        
        self.trials_df = df_combined


    def extract_da_columns(self):
        """
        Extracts dopamine-related columns from trials_df into da_df.
        """
        da_cols = [
            'subject_name', 'file name', 'trial',
            'filtered_sound_cues', 'filtered_port_entries', 'filtered_port_entry_offset',
            'first_PE_after_sound_cue',
            'Tone_Time_Axis', 'Tone_Zscore',
            'PE_Time_Axis', 'PE_Zscore'
        ]

        # Only keep columns that actually exist in the DataFrame
        available_cols = [col for col in da_cols if col in self.trials_df.columns]
        self.da_df = self.trials_df[available_cols].copy()



    """******************************* PORT ENTRY CALCULATIONS ********************************"""
    def find_first_port_entry_after_sound_cue(self):
        """
        Finds the first port entry occurring after 4 seconds following each sound cue.
        If a port entry starts before 4 seconds but extends past it, 
        the function selects the timestamp at 4 seconds after the sound cue.

        Works with any DataFrame that has the required columns.
        """
        df = self.da_df 

        first_PEs = []  # List to store results

        for index, row in df.iterrows():  # Use df, not self.df
            sound_cues_onsets = row['filtered_sound_cues']
            port_entries_onsets = row['filtered_port_entries']
            port_entries_offsets = row['filtered_port_entry_offset']

            first_PEs_per_row = []

            for sc_onset in sound_cues_onsets:
                threshold_time = sc_onset + 4

                # First, check for an ongoing PE:
                ongoing_PEs_indices = np.where(
                    (port_entries_onsets < threshold_time) & (port_entries_offsets >= threshold_time)
                )[0]

                if len(ongoing_PEs_indices) > 0:
                    first_PEs_per_row.append(threshold_time)
                else:
                    # Otherwise, find the first port entry that starts after the threshold
                    future_PEs_indices = np.where(port_entries_onsets >= threshold_time)[0]
                    if len(future_PEs_indices) > 0:
                        first_PEs_per_row.append(port_entries_onsets[future_PEs_indices[0]])
                    else:
                        first_PEs_per_row.append(None)

            first_PEs.append(first_PEs_per_row)

        df["first_PE_after_sound_cue"] = first_PEs
        return df


    def compute_closest_port_offset(self, PE_column, offset_column):
        """
        Computes the closest port entry offsets after each PE time and adds them as a new column in the dataframe.
        
        Parameters:
            PE_column (str): The column name for the PE times (e.g., 'first_PE_after_sound_cue').
            offset_column (str): The column name for the port entry offset times (e.g., 'filtered_port_entry_offset').
            new_column_name (str): The name of the new column to store the results. Default is 'closest_port_entry_offsets'.
        
        Returns:
            pd.DataFrame: Updated DataFrame with the new column of closest port entry offsets.
        """
        df = self.da_df 

        def find_closest_port_entries(PEs, port_entry_offsets):
            """Finds the closest port entry offsets greater than each PE in 'PEs'."""
            closest_offsets = []
            
            for PE in PEs:
                # Find the indices where port_entry_offset > PE
                valid_indices = np.where(port_entry_offsets > PE)[0]
                
                if len(valid_indices) == 0:
                    closest_offsets.append(np.nan)  # Append NaN if no valid offset is found
                else:
                    # Get the closest port entry offset (the first valid one in the array)
                    closest_offset = port_entry_offsets[valid_indices[0]]
                    closest_offsets.append(closest_offset)
            
            return closest_offsets

        def compute_PE_metrics(row):
            """Compute the closest port entry offsets for each trial."""
            # Extract first_PE_after_sound_cue and filtered_port_entry_offset
            first_PEs = np.array(row[PE_column])
            port_entry_offsets = np.array(row[offset_column])
            
            # Get the closest port entry offsets for each PE
            closest_offsets = find_closest_port_entries(first_PEs, port_entry_offsets)
            
            return closest_offsets

        # Apply the function to the DataFrame and create a new column with the results
        df['closest_PE_offset'] = df.apply(compute_PE_metrics, axis=1)

    def compute_duration(self, df=None):
        if df is None:
            df = self.df
        df["duration"] = df.apply(lambda row: np.array(row["closest_PE_offset"]) - np.array(row["first_PE_after_sound_cue"]), axis=1)


    """******************************* DOPAMINE CALCULATIONS ********************************"""
    def compute_EI_DA(self, pre_time=4, post_time=10):
        """
        Computes event-induced DA (ΔF/F z-score) responses for every tone (sound cue) 
        and PE in self.da_df.

        - Tone:
        * Time axis: -pre_time to +post_time around each sound cue onset.
        * Baseline: mean of the DA signal in the 4 seconds before the sound cue.

        - PE:
        * Time axis: 0 to +post_time around the PE onset.
        * Baseline: same as the tone event baseline (4 s pre-cue).
        * If the i-th PE has no corresponding sound cue, baseline=0.
        """
        df = self.da_df

        # 1) Determine global dt
        min_dt = np.inf
        for _, row in df.iterrows():
            ts = np.array(row['trial'].timestamps)
            if ts.size > 1:
                min_dt = min(min_dt, np.min(np.diff(ts)))
        if min_dt == np.inf:
            print("No valid timestamps found to determine dt.")
            return

        # 2) Common axes
        tone_axis = np.arange(-pre_time, post_time, min_dt)
        PE_axis = np.arange(0,       post_time, min_dt)

        # 3) Containers
        tone_z  = []; tone_t = []
        PE_z  = []; PE_t = []

        for _, row in df.iterrows():
            trial = row['trial']
            ts    = np.array(trial.timestamps)
            zs    = np.array(trial.zscore)
            cues  = row.get('filtered_sound_cues', [])
            PEs = row.get('first_PE_after_sound_cue', [])

            # — Tone —
            this_z_tone = []
            this_t_tone = []
            if not cues:
                this_z_tone.append(np.full(tone_axis.shape, np.nan))
                this_t_tone.append(np.full(tone_axis.shape, np.nan))
            else:
                for cue in cues:
                    mask = (ts >= cue - pre_time) & (ts <= cue + post_time)
                    if not mask.any():
                        this_z_tone.append(np.full(tone_axis.shape, np.nan))
                        this_t_tone.append(np.full(tone_axis.shape, np.nan))
                        continue

                    rt   = ts[mask] - cue
                    sig  = zs[mask]
                    base = np.nanmean(sig[rt < 0]) if (rt < 0).any() else 0
                    corr = sig - base
                    this_z_tone.append(np.interp(tone_axis, rt, corr))
                    this_t_tone.append(tone_axis.copy())

            # — PE —
            this_z_PE = []
            this_t_PE = []
            if not isinstance(PEs, (list, np.ndarray)) or len(PEs) == 0:
                this_z_PE.append(np.full(PE_axis.shape, np.nan))
                this_t_PE.append(np.full(PE_axis.shape, np.nan))
            else:
                for i, PE in enumerate(PEs):
                    if PE is None or (isinstance(PE, float) and np.isnan(PE)):
                        this_z_PE.append(np.full(PE_axis.shape, np.nan))
                        this_t_PE.append(np.full(PE_axis.shape, np.nan))
                        continue

                    # pick baseline cue if available
                    baseline_cue = cues[i] if i < len(cues) else None
                    if baseline_cue is not None:
                        bm = (ts >= baseline_cue - pre_time) & (ts <= baseline_cue)
                        base_val = np.nanmean(zs[bm]) if bm.any() else 0
                    else:
                        base_val = 0

                    mask = (ts >= PE) & (ts <= PE + post_time)
                    if not mask.any():
                        this_z_PE.append(np.full(PE_axis.shape, np.nan))
                        this_t_PE.append(np.full(PE_axis.shape, np.nan))
                        continue

                    rt   = ts[mask] - PE
                    corr = zs[mask] - base_val
                    this_z_PE.append(np.interp(PE_axis, rt, corr))
                    this_t_PE.append(PE_axis.copy())

            tone_z.append(this_z_tone)
            tone_t.append(this_t_tone)
            PE_z.append(this_z_PE)
            PE_t.append(this_t_PE)

        # 4) Store back on da_df
        df['Tone_Time_Axis'] = tone_t
        df['Tone_Zscore']    = tone_z
        df['PE_Time_Axis'] = PE_t
        df['PE_Zscore']    = PE_z


    def compute_rtc_da_metrics(self, mode='EI', bout_duration=4):
        """
        Computes DA metrics (AUC, Max Peak, Time of Max Peak, Mean Z-score) for Tone and PE events
        over the 0→bout_duration window for each event.

        mode:
        • 'EI'       — uses baseline-corrected, peri-event traces in
                        `Tone_Zscore` / `PE_Zscore`.
        • 'standard' — extracts raw zscore around each cue or PE from
                        onset→onset + bout_duration and interpolates onto 0→bout_duration.
        """

        df = self.da_df
        use_ei = (mode == 'EI') and ('Tone_Zscore' in df.columns)

        # Map event names to the cue column, EI-trace column, and EI-time‐axis column
        event_info = {
            'Tone': {
                'cue_col':       'filtered_sound_cues',
                'ei_z_col':      'Tone_Zscore',
                'ei_t_col':      'Tone_Time_Axis'
            },
            'PE': {
                'cue_col':       'first_PE_after_sound_cue',
                'ei_z_col':      'PE_Zscore',
                'ei_t_col':      'PE_Time_Axis'
            }
        }

        # Helper: from (list-of-traces, list-of-axes) → per-event metrics
        def extract_metrics(traces, axes):
            aucs, maxs, times, means = [], [], [], []
            for tr, ax in zip(traces, axes):
                tr = np.asarray(tr); ax = np.asarray(ax)
                m = (ax >= 0) & (ax <= bout_duration)
                seg_t = ax[m]; seg = tr[m]
                if seg.size:
                    aucs.append(np.trapz(seg, seg_t))
                    idx = np.nanargmax(seg)
                    maxs.append(seg[idx])
                    times.append(seg_t[idx])
                    means.append(np.nanmean(seg))
                else:
                    aucs.append(np.nan); maxs.append(np.nan)
                    times.append(np.nan); means.append(np.nan)
            return aucs, maxs, times, means

        # If standard mode, build a common 0→bout_duration axis
        if not use_ei:
            min_dt = np.inf
            for _, row in df.iterrows():
                ts = np.array(row['trial'].timestamps)
                if ts.size > 1:
                    min_dt = min(min_dt, np.min(np.diff(ts)))
            if not np.isfinite(min_dt):
                raise RuntimeError("Cannot determine sampling dt for standard mode.")
            standard_axis = np.arange(0, bout_duration, min_dt)

        # Prepare storage
        metrics = {
            ev: {'auc': [], 'max': [], 'time': [], 'mean': []}
            for ev in event_info
        }

        # Loop over trials
        for _, row in df.iterrows():
            trial = row['trial']
            ts    = np.array(trial.timestamps)
            zs    = np.array(trial.zscore)

            for ev, info in event_info.items():
                # 1) gather traces & axes
                if use_ei:
                    traces = row.get(info['ei_z_col'], []) or []
                    axes   = row.get(info['ei_t_col'], []) or []
                else:
                    cues   = row.get(info['cue_col'], []) or []
                    traces, axes = [], []
                    for on in cues:
                        if on is None or (isinstance(on, float) and np.isnan(on)):
                            traces.append(np.full_like(standard_axis, np.nan))
                        else:
                            mask = (ts >= on) & (ts <= on + bout_duration)
                            rel  = ts[mask] - on
                            sig  = zs[mask]
                            interp = np.interp(standard_axis, rel, sig,
                                            left=np.nan, right=np.nan)
                            traces.append(interp)
                        axes.append(standard_axis)

                # 2) extract metrics over 0→bout_duration
                aucs, maxs, times, means = extract_metrics(traces, axes)
                metrics[ev]['auc'].append(aucs)
                metrics[ev]['max'].append(maxs)
                metrics[ev]['time'].append(times)
                metrics[ev]['mean'].append(means)

        # 3) write back into da_df
        for ev in event_info:
            df[f'{ev} AUC']               = metrics[ev]['auc']
            df[f'{ev} Max Peak']          = metrics[ev]['max']
            df[f'{ev} Time of Max Peak']  = metrics[ev]['time']
            df[f'{ev} Mean Z-score']      = metrics[ev]['mean']

        return df



    """******************************* AVERAGING FOR COMPARISON ********************************"""
    def find_overall_mean(self):
        """
        Computes the per-subject mean of all available DA metrics in self.trials_df.
        Automatically detects which scalar and array columns are present and skips missing ones.
        """
        df = self.trials_df

        def mean_arrays(group):
            result = {}

            # Always keep Time Axis
            if "Tone_Time_Axis" in group.columns:
                result["Tone_Time_Axis"] = group["Tone_Time_Axis"].iloc[0]

            # Optionally preserve metadata
            for meta_col in ["Rank", "Cage"]:
                if meta_col in group.columns:
                    result[meta_col] = group[meta_col].iloc[0]

            # Scalar numeric columns: take mean of list if needed, then mean across trials
            scalar_cols = [
                'PE AUC', 'PE Max Peak', 'PE Mean Z-score',
                'Tone AUC', 'Tone Max Peak', 'Tone Mean Z-score',
                'PE AUC First', 'PE AUC Last', 'PE Max Peak First', 'PE Max Peak Last',
                'PE Mean Z-score First', 'PE Mean Z-score Last',
                'Tone AUC First', 'Tone AUC Last', 'Tone Max Peak First', 'Tone Max Peak Last',
                'Tone Mean Z-score First', 'Tone Mean Z-score Last',
                'PE AUC EI', 'PE Max Peak EI', 'PE Mean Z-score EI',
                'Tone AUC EI', 'Tone Max Peak EI', 'Tone Mean Z-score EI',
                'PE AUC EI First', 'PE Max Peak EI First', 'Tone AUC EI First',
                'Tone Max Peak EI First', 'Tone Mean Z-score EI First',
                'PE AUC EI Last', 'PE Max Peak EI Last', 'Tone AUC EI Last',
                'Tone Max Peak EI Last', 'Tone Mean Z-score EI Last'
            ]
            for col in scalar_cols:
                if col in group.columns:
                    group[col] = group[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
                    group[col] = pd.to_numeric(group[col], errors='coerce')
                    result[col] = group[col].mean()

            # Array columns: element-wise mean
            array_cols = ["Tone_Zscore", "PE_Zscore"]
            for col in array_cols:
                if col in group.columns:
                    try:
                        stacked = np.vstack(group[col].dropna().values)
                        result[col] = np.mean(stacked, axis=0)
                    except:
                        result[col] = np.nan  # If stacking fails (e.g., inconsistent shapes)

            return pd.Series(result)

        return self.trials_df.groupby("subject_name").apply(mean_arrays).reset_index()


    """******************************* PLOTTING ********************************"""
    def plot_specific_event_psth(self, event_type, event_index, directory_path, brain_region, y_min, y_max, condition='Win', bin_size=100):
        """
        Plots the PSTH (mean and SEM) for a specific event bout (0→4 s after event onset)
        across trials, using the same averaging logic as for a bout response.

        Parameters:
            event_type (str): The event type (e.g. 'Tone' or 'PE').
            event_index (int): 1-indexed event number to plot.
            directory_path (str or None): Directory to save the plot (if None, the plot is not saved).
            brain_region (str): Brain region ('mPFC' or other) to filter subjects.
            y_min (float): Lower bound of the y-axis.
            y_max (float): Upper bound of the y-axis.
            df (DataFrame, optional): DataFrame to use (defaults to self.df).
            bin_size (int, optional): Bin size for downsampling.
        """
        df = self.da_df

        # Filter subjects by brain region.
        def split_by_subject(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)
        idx = event_index - 1
        # print(f"[DEBUG] PSTH: Plotting {event_type} event index {event_index} (0-indexed {idx}) for brain region {brain_region}")

        selected_traces = []
        for i, row in df.iterrows():
            event_z_list = row.get(f'{event_type}_Zscore', [])
            if isinstance(event_z_list, list) and len(event_z_list) > idx:
                trace = np.array(event_z_list[idx])
                selected_traces.append(trace)
        if len(selected_traces) == 0:
            print(f"No trials have an event at index {event_index} for {event_type}.")
            return

        # Use the common time axis from the first trial's bout.
        common_time_axis = df.iloc[0][f'{event_type}_Time_Axis'][idx]
        selected_traces = np.array(selected_traces)

        mean_trace = np.mean(selected_traces, axis=0)
        sem_trace = np.std(selected_traces, axis=0) / np.sqrt(selected_traces.shape[0])
        
        if hasattr(self, 'downsample_data'):
            mean_trace, downsampled_time_axis = self.downsample_data(mean_trace, common_time_axis, bin_size)
            sem_trace, _ = self.downsample_data(sem_trace, common_time_axis, bin_size)
        else:
            downsampled_time_axis = common_time_axis

        # # --- Debug: Check values in the 4–10 s window ---
        # post_window = (downsampled_time_axis >= 4) & (downsampled_time_axis <= 10)
        # post_vals = mean_trace[post_window]
        # print("[DEBUG] PSTH: Final mean trace shape:", mean_trace.shape)
        # print("[DEBUG] PSTH: Final mean trace first 10 values:", mean_trace[:10])
        # print("[DEBUG] PSTH: 4–10 s window values, first 10:", post_vals[:10])
        # print("[DEBUG] PSTH: Max in 4–10 s window:", np.max(post_vals))

        trace_color = '#FFAF00' if brain_region == 'mPFC' else '#15616F'

        plt.figure(figsize=(10, 6))
        plt.plot(downsampled_time_axis, mean_trace, color=trace_color, lw=3, label='Mean DA')
        plt.fill_between(downsampled_time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=trace_color, alpha=0.4, label='SEM')
        plt.axvline(0, color='black', linestyle='--', lw=2)
        plt.axvline(4, color='#FF69B4', linestyle='-', lw=2)

        # Force the x-axis to be from -4 to 10 seconds.
        plt.xlabel('Time from Tone Onset (s)', fontsize=30)
        plt.ylabel('Event-Induced z-scored ΔF/F', fontsize=30)
        plt.title(f'{event_type} Event {event_index} {condition} PSTH', fontsize=30, pad=30)
        plt.ylim(y_min, y_max)
        plt.xticks([-4, 0, 4, 10], fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim(-4, 10)  # Force x-axis limits

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)

        if directory_path is not None:
            # Ensure the directory exists.
            os.makedirs(directory_path, exist_ok=True)
            save_path = os.path.join(directory_path, f'{brain_region}_{event_type}_Event{event_index}_PSTH.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()