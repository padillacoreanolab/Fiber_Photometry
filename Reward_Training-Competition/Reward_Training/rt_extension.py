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
    def plot_events_heatmap(self, event_type, max_events, directory_path, brain_region, 
                             vmin, vmax, df=None, bin_size=125):
        """
        Plots a heatmap in which each row is the average DA trace (PSTH) for a given 
        event number (e.g., tone number) across subjects/trials, with tone #1 at the top.

        Parameters:
            event_type (str): The event type (e.g., 'Tone' or 'Lick').
            max_events (int): The maximum event number to include.
            directory_path (str or None): Directory to save the plot (if None, not saved).
            brain_region (str): Brain region filter ('mPFC' or 'NAc').
            vmin (float): Lower bound of the color scale (Z‑score).
            vmax (float): Upper bound of the color scale (Z‑score).
            df (pd.DataFrame, optional): DataFrame to use (defaults to self.da_df).
            bin_size (int): Bin size for optional downsampling.
        """

        if df is None:
            df = self.da_df

        # Filter by region prefix
        df = df[df['subject_name'].str.startswith('p')] if brain_region == 'mPFC' \
            else df[df['subject_name'].str.startswith('n')]
        if df.empty:
            print(f"No data for brain region '{brain_region}'.")
            return

        # Grab the common time axis
        axes_list = df.iloc[0].get(f'{event_type}_Time_Axis', [])
        if not isinstance(axes_list, list) or len(axes_list) == 0:
            print(f"No {event_type} Time_Axis found.")
            return
        common_time_axis = np.array(axes_list[0])

        # Collect and average traces for each event index
        averaged_event_traces = []
        for idx in range(max_events):
            traces = []
            for _, row in df.iterrows():
                zlist = row.get(f'{event_type}_Zscore', [])
                if isinstance(zlist, list) and len(zlist) > idx:
                    traces.append(np.array(zlist[idx]))
            if traces:
                averaged_event_traces.append(np.nanmean(np.vstack(traces), axis=0))
            else:
                averaged_event_traces.append(np.array([]))

        # Truncate to shortest length
        min_len = min((len(r) for r in averaged_event_traces if r.size), default=0)
        if min_len == 0:
            print("No valid event traces to plot.")
            return
        heatmap_data = np.vstack([r[:min_len] for r in averaged_event_traces])
        time_axis = common_time_axis[:min_len]

        # Plot heatmap
        x_min, x_max = time_axis[0], time_axis[-1]
        # extent: y from max_events→0 so row1 at top
        extent = [x_min, x_max, max_events, 0]

        fig, ax = plt.subplots(figsize=(12, 8))
        cmap_choice = 'inferno' if brain_region == 'mPFC' else 'viridis'
        im = ax.imshow(
            heatmap_data,
            aspect='auto',
            cmap=cmap_choice,
            origin='upper',
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        # Formatting
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('Time from Event Onset (s)', fontsize=24)
        ax.set_ylabel(f'{event_type} Number', fontsize=24)
        ax.set_xticks([-4, 0, 4, 10])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=20)

        # y‐ticks at each row (center of each block)
        y_centers = np.arange(max_events) + 0.5
        ax.set_yticks(y_centers)
        ax.set_yticklabels([str(i+1) for i in range(max_events)], fontsize=20)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.axvline(0, color='white', linestyle='--', linewidth=2)
        ax.axvline(4, color='#FF69B4', linestyle='-', linewidth=2)
        ax.set_title(f'{brain_region} {event_type} Heatmap 1–{max_events}', fontsize=28)

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Z-score', fontsize=20)

        if directory_path:
            fname = f'{brain_region}_{event_type}_Heatmap_1to{max_events}.png'
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', transparent=True)

        plt.tight_layout()
        plt.show()

    def plot_event_subject_heatmap(self,
                              event_type: str,
                              event_index: int,
                              directory_path: str,
                              brain_region: str,
                              vmin: float,
                              vmax: float,
                              df=None):
        """
        Plots a heatmap in which each row is the DA trace (PSTH) for a single specified
        event (e.g., Tone #3) across different subjects, with a crisp block per subject.
        """
        # 1) grab data
        if df is None:
            df = self.da_df.copy()

        # 2) filter by region
        df = df[df['subject_name'].str.startswith('p')] if brain_region == 'mPFC' \
            else df[df['subject_name'].str.startswith('n')]
        if df.empty:
            print(f"No rows for region '{brain_region}'.")
            return

        # 3) pull the common time‐axis for this event
        axes_list = df.iloc[0].get(f'{event_type}_Time_Axis', [])
        if not isinstance(axes_list, list) or len(axes_list) <= event_index:
            print(f"No Time_Axis for {event_type} index {event_index}.")
            return
        time_axis = np.array(axes_list[event_index])

        # 4) collect each subject’s Z‐scored trace
        subject_names = []
        traces = []
        for _, row in df.iterrows():
            zlist = row.get(f'{event_type}_Zscore', [])
            if isinstance(zlist, list) and len(zlist) > event_index:
                subject_names.append(row['subject_name'])
                traces.append(np.array(zlist[event_index]))
        if not traces:
            print("No traces found.")
            return

        # 5) truncate to shortest
        min_len = min(len(t) for t in traces)
        traces = [t[:min_len] for t in traces]
        time_axis = time_axis[:min_len]

        # 6) stack into (n_subjects × n_time) array
        heatmap_data = np.vstack(traces)
        n_subj = heatmap_data.shape[0]

        # 7) set extent so row 0 is at the top band, row n_subj-1 at bottom
        extent = [time_axis[0], time_axis[-1], n_subj, 0]

        # 8) plot
        fig, ax = plt.subplots(figsize=(8, max(2, n_subj * 0.4)))
        cmap = 'inferno' if brain_region == 'mPFC' else 'viridis'
        im = ax.imshow(
            heatmap_data,
            aspect='auto',
            origin='upper',
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        # 9) tidy up axes
        ax.set_xlabel('Time from Event Onset (s)', fontsize=14)
        ax.set_ylabel('Subject', fontsize=14)
        ax.set_title(f'{brain_region} {event_type} #{event_index+1} by Subject', fontsize=16)

        # vertical event markers
        for t in (0, 4):
            ax.axvline(t, color='white' if t==0 else '#FF69B4',
                    linestyle='--' if t==0 else '-', lw=1.5)

        # x‐ticks
        ax.set_xticks([-4, 0, 4, 10])
        ax.set_xticklabels(['-4', '0', '4', '10'])

        # y‐ticks at each band’s center
        ax.set_yticks(np.arange(n_subj) + 0.5)
        ax.set_yticklabels(subject_names, fontsize=8)

        # colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label('Z-score', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # 10) save if requested
        if directory_path:
            fname = f'{brain_region}_{event_type}_{event_index+1}_by_subject_heatmap.png'
            plt.savefig(os.path.join(directory_path, fname),
                        dpi=300, bbox_inches='tight', transparent=True)

        plt.tight_layout()
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
