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
from scipy.stats import pearsonr, linregress
from rtc_extension import RTC 


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

        # print(f"[DEBUG] Heatmap: Averaging {event_type} data for first {max_events} events in brain region {brain_region}")

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


    # Tone number scatter plots
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



    def plot_scatter(self,
                            column: str,
                            color='C0',
                            individual_dots=False,
                            xlabel: str = None,
                            ylabel: str = None,
                            title: str = None,
                            yrange: list = None):
        """
        …(docstring unchanged)…
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

        fig, ax = plt.subplots(figsize=(8, 5))

        for i in range(min_len):
            ys = [arr[i] for arr in arrays if not np.isnan(arr[i])]
            if individual_dots:
                ax.scatter([i + 1] * len(ys), ys,
                        color=color, alpha=0.6, edgecolor='none', s=40)
            means[i] = np.mean(ys)
            sems[i]  = np.std(ys, ddof=1) / np.sqrt(len(ys))

        # plot errorbars as markers only
        ax.errorbar(indices, means, yerr=sems,
                    fmt='o',         # <-- marker only
                    lw=2, capsize=5,
                    color=color,
                    label='Mean ± SEM')

        # compute Pearson on the means
        r, p = pearsonr(indices, means)
        ax.plot([], [], ' ', label=f"r = {r:.2f}, p = {p:.3f}")

        # compute and plot line of best fit through the mean points
        slope, intercept, r_val, p_val, stderr = linregress(indices, means)
        fit_y = intercept + slope * indices
        ax.plot(indices, fit_y,
                linestyle='--', linewidth=2,
                color=color, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

        ax.set_xlabel(xlabel or column, fontsize=14)
        ax.set_ylabel(ylabel or column, fontsize=14)
        ax.set_title(title or column, fontsize=16)
        ax.set_xticks(indices)

        if yrange and len(yrange)==2:
            ax.set_ylim(yrange)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.show()
