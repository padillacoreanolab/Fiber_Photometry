# This file creates a child class of Experiment that is specific to RT recordings. RTrecordings can either have one mouse or two mice in the same folder.
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)


from experiment_class import Experiment 
from trial_class import Trial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
from rtc_extension import RTC 
from scipy.stats import pearsonr


class Reward_Training(RTC):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)

    '''*********************************CREATE DF*************************************'''
    def create_base_df(self, directory_path):
        """
        Creates Dataframe that will contain all the data for analysis.
        """
        subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

        # Create the DataFrame with 'file name' column (subdirectories here)
        self.trials_df['file name'] = subdirectories
        self.trials_df['file name'] = self.trials_df['file name'].astype(str)
        
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


        # creating column for subject names
        if 'Cohort_1_2' not in directory_path: 
            self.trials_df['file name'] = self.trials_df['file name'].apply(expand_filenames)
            self.trials_df = self.trials_df.explode('file name', ignore_index=True)
        self.trials_df['trial'] = self.trials_df.apply(lambda row: find_matching_trial(row['file name']), axis=1)
        self.trials_df['subject_name'] = self.trials_df['file name'].str.split('-').str[0]
        self.trials_df['sound cues'] = self.trials_df['trial'].apply(lambda x: x.rtc_events.get('sound cues', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.trials_df['port entries'] = self.trials_df['trial'].apply(lambda x: x.rtc_events.get('port entries', None) if isinstance(x, object) and hasattr(x, 'rtc_events') else None)
        self.trials_df['sound cues onset'] = self.trials_df['sound cues'].apply(lambda x: x.onset_times if x else None)
        self.trials_df['port entries onset'] = self.trials_df['port entries'].apply(lambda x: x.onset_times if x else None)
        self.trials_df['port entries offset'] = self.trials_df['port entries'].apply(lambda x: x.offset_times if x else None)

        # This is added so both rt and rc can use the same functions
        self.trials_df['filtered_sound_cues'] = self.trials_df['sound cues onset']
        self.trials_df['filtered_port_entries'] = self.trials_df['port entries onset']
        self.trials_df['filtered_port_entry_offset'] = self.trials_df['port entries offset'] 



    '''********************************** LICKS **********************************'''
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
            df = self.da_df

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

                
    # def compute_da_metrics(self, mode='EI', bout_duration=4):
    #     """
    #     Computes DA metrics (AUC, Max Peak, Time of Max Peak, Mean Z-score) for tone and lick events
    #     **only** over the 0→bout_duration window for each event.

    #     Two modes supported:
    #     • mode='EI': uses event-induced, baseline-corrected traces stored in
    #         `Tone Event_Zscore` / `Lick Event_Zscore` (with axes in `Tone Event_Time_Axis` / `Lick Event_Time_Axis`).
    #     • mode='standard': extracts raw zscore around each event in a
    #         [event_time → event_time + bout_duration] window and interpolates onto 0→bout_duration.

    #     Overwrites these columns:
    #     'Tone AUC', 'Tone Max Peak', 'Tone Time of Max Peak', 'Tone Mean Z-score'
    #     'Lick AUC', 'Lick Max Peak', 'Lick Time of Max Peak', 'Lick Mean Z-score'
    #     """

    #     df = self.da_df

    #     use_ei = (mode == 'EI') and ('Tone Event_Zscore' in df.columns)

    #     # helper to extract metrics from (trace, axis) pairs **within** 0→bout_duration
    #     def extract_metrics(traces, axes):
    #         aucs, maxs, times, means = [], [], [], []
    #         for tr, ax in zip(traces, axes):
    #             tr = np.asarray(tr)
    #             ax = np.asarray(ax)
    #             # mask to 0→bout_duration
    #             m = (ax >= 0) & (ax <= bout_duration)
    #             seg = tr[m]
    #             seg_t = ax[m]
    #             if seg.size:
    #                 aucs .append(np.trapz(seg, seg_t))
    #                 idx   = np.nanargmax(seg)
    #                 maxs .append(seg[idx])
    #                 times.append(seg_t[idx])
    #                 means.append(np.nanmean(seg))
    #             else:
    #                 aucs .append(np.nan)
    #                 maxs .append(np.nan)
    #                 times.append(np.nan)
    #                 means.append(np.nan)
    #         return aucs, maxs, times, means

    #     # precompute dt for standard
    #     if not use_ei:
    #         min_dt = np.inf
    #         for _, row in df.iterrows():
    #             ts = np.array(row['trial'].timestamps)
    #             if ts.size>1:
    #                 min_dt = min(min_dt, np.min(np.diff(ts)))
    #         if not np.isfinite(min_dt):
    #             raise RuntimeError("Can't find sampling dt for standard mode")
    #         standard_axes = np.arange(0, bout_duration, min_dt)

    #     # storage
    #     t_auc, t_max, t_time, t_mean = [], [], [], []
    #     l_auc, l_max, l_time, l_mean = [], [], [], []

    #     for _, row in df.iterrows():
    #         if use_ei:
    #             tone_traces = row.get('Tone Event_Zscore',  []) or []
    #             tone_axes   = row.get('Tone Event_Time_Axis', []) or []
    #             lick_traces = row.get('Lick Event_Zscore',  []) or []
    #             lick_axes   = row.get('Lick Event_Time_Axis', []) or []
    #         else:
    #             # build standard traces
    #             trial = row['trial']
    #             ts    = np.array(trial.timestamps)
    #             zs    = np.array(trial.zscore)

    #             tone_traces, tone_axes = [], []
    #             for on in (row.get('sound cues onset') or []):
    #                 if on is None or np.isnan(on):
    #                     tone_traces.append(np.full_like(standard_axes, np.nan))
    #                 else:
    #                     mask = (ts >= on) & (ts <= on + bout_duration)
    #                     rel  = ts[mask] - on
    #                     sig  = zs[mask]
    #                     tone_traces.append(np.interp(standard_axes, rel, sig,
    #                                                 left=np.nan, right=np.nan))
    #                 tone_axes.append(standard_axes)

    #             lick_traces, lick_axes = [], []
    #             for on in (row.get('first_lick_after_sound_cue') or []):
    #                 if on is None or np.isnan(on):
    #                     lick_traces.append(np.full_like(standard_axes, np.nan))
    #                 else:
    #                     mask = (ts >= on) & (ts <= on + bout_duration)
    #                     rel  = ts[mask] - on
    #                     sig  = zs[mask]
    #                     lick_traces.append(np.interp(standard_axes, rel, sig,
    #                                                 left=np.nan, right=np.nan))
    #                 lick_axes.append(standard_axes)

    #         # now pull out metrics from the 0→bout_duration window
    #         ta, tm, tt, tme = extract_metrics(tone_traces, tone_axes)
    #         la, lm, lt, lme = extract_metrics(lick_traces, lick_axes)

    #         t_auc.append( ta); t_max.append( tm); t_time.append( tt); t_mean.append( tme)
    #         l_auc.append( la); l_max.append( lm); l_time.append( lt); l_mean.append( lme)

    #     # write back
    #     df['Tone AUC']              = t_auc
    #     df['Tone Max Peak']         = t_max
    #     df['Tone Time of Max Peak'] = t_time
    #     df['Tone Mean Z-score']     = t_mean

    #     df['Lick AUC']              = l_auc
    #     df['Lick Max Peak']         = l_max
    #     df['Lick Time of Max Peak'] = l_time
    #     df['Lick Mean Z-score']     = l_mean

    #     return df





    '''********************************** PSTH CODE  **********************************'''
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
            df (DataFrame, optional): DataFrame to use (defaults to self.da_df).
            bin_size (int, optional): Bin size for downsampling.
        """

        if df is None:
            df = self.da_df

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




    def plot_sequential_event(self,
                            column: str,
                            color='C0',
                            individual_dots=False,
                            xlabel: str = None,
                            ylabel: str = None,
                            title: str = None,
                            yrange: list = None):
        """
        For a column of per-trial arrays (e.g. 'Tone AUC'), this will:
        - find the minimum number of events across rows
        - for each event index 1..N, scatter all the individual values
        - plot mean±SEM over that
        - compute Pearson r,p between [1..N] and the mean trace
        - show r,p in the legend

        Args:
            column: name of the DataFrame column containing list/array per row
            color:  matplotlib color for dots+line
            individual_dots: whether to show scatter dots per subject
            xlabel/ylabel/title: override axis labels & title
            yrange: optional y-axis range [ymin, ymax]
        """
        df = self.da_df

        arrays = [np.asarray(v) for v in df[column]
                if isinstance(v, (list, np.ndarray)) and len(v) > 0]

        if not arrays:
            print(f"No data in column {column}")
            return

        min_len = min(arr.shape[0] for arr in arrays)
        means = np.zeros(min_len)
        sems  = np.zeros(min_len)
        indices = np.arange(1, min_len + 1)

        fig, ax = plt.subplots(figsize=(8, 5))  # Wider figure

        for i in range(min_len):
            ys = [arr[i] for arr in arrays if not np.isnan(arr[i])]
            if individual_dots:
                ax.scatter([i + 1] * len(ys), ys, color=color,
                        alpha=0.6, edgecolor='none', s=40)
            means[i] = np.mean(ys)
            sems[i] = np.std(ys, ddof=1) / np.sqrt(len(ys))

        ax.errorbar(indices, means, yerr=sems, fmt='-o', lw=2, capsize=5,
                    color=color, label='Mean ± SEM')

        r, p = pearsonr(indices, means)
        ax.plot([], [], ' ', label=f"r = {r:.2f}, p = {p:.3f}")

        ax.set_xlabel(xlabel or column, fontsize=14)
        ax.set_ylabel(ylabel or column, fontsize=14)
        ax.set_title(title or column, fontsize=16)
        ax.set_xticks(indices)

        if yrange and isinstance(yrange, (list, tuple)) and len(yrange) == 2:
            ax.set_ylim(yrange[0], yrange[1])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend outside plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Leave room for legend
        plt.show()
