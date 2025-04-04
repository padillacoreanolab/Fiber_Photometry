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

class Reward_Training(Experiment):
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

    def rt_processing1(self, time_segments_to_remove=None):
        """
        Batch processes rc with 1 box
        """
        print(self.trials.items())
        for trial_folder, trial in self.trials.items():
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Reward Training Processing {trial_folder}...")
            trial.remove_initial_LED_artifact(t=30)
            # trial.remove_final_data_segment(t = 10)
            
            trial.highpass_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            baseline_start, baseline_end = trial.find_baseline_period()  
            # trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            trial.compute_zscore(method = 'standard')
            # trial.compute_zscore(method = 'modified')
            trial.verify_signal()

            # PC0 = Tones
            """
            Using RIG DATA
            """
            trial.rtc_events['sound cues'] = trial.rtc_events.pop('PC0_')
            trial.rtc_events['port entries'] = trial.rtc_events.pop('PC2_', trial.rtc_events.pop('PC3_'))

            # Remove the first entry because it doesn't count
            trial.rtc_events['sound cues'].onset_times = trial.rtc_events['sound cues'].onset[1:]
            trial.rtc_events['sound cues'].offset_times = trial.rtc_events['sound cues'].offset[1:]
            trial.rtc_events['port entries'].onset_times = trial.rtc_events['port entries'].onset[1:]
            trial.rtc_events['port entries'].offset_times = trial.rtc_events['port entries'].offset[1:]

            
            # Finding instances after first tone is played
            port_entries_onset = np.array(trial.rtc_events['port entries'].onset_times)
            port_entries_offset = np.array(trial.rtc_events['port entries'].offset_times)
            first_sound_cue_onset = trial.rtc_events['sound cues'].onset_times[0]
            indices = np.where(port_entries_onset >= first_sound_cue_onset)[0]
            trial.rtc_events['port entries'].onset_times = port_entries_onset[indices].tolist()
            trial.rtc_events['port entries'].offset_times = port_entries_offset[indices].tolist()

            # self.combine_consecutive_rtc_events(behavior_name='all', bout_time_threshold=0.5)

    def rt_processing2(self, time_segments_to_remove=None):
        """
        Batch processes rc with 2 box
        """
        print(self.trials.items())
        for (trial_folder, trial), (trial_folder1, trial1) in zip(self.trials.items(), self.port_bnc.items()):
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Reward Training Processing {trial_folder}...")

            trial.remove_initial_LED_artifact(t=30)
            trial.highpass_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            baseline_start, baseline_end = trial.find_baseline_period()
            trial.compute_zscore(method='standard')
            trial.verify_signal()

            # Using RIG DATA
            print(f"Available behaviors in trial: {trial.rtc_events.keys()}")
            trial.rtc_events['sound cues'] = trial.rtc_events.pop('PC0_')
            
            # Correct the way 'port entries' is assigned for trial1 based on self.port_bnc
            trial_type1 = self.port_bnc.get(trial_folder1, None)  # Fetch trial type from self.port_bnc
            if trial_type1 == 3:
                trial.rtc_events['port entries'] = trial.rtc_events.pop('PC3_')
            elif trial_type1 == 2:
                trial.rtc_events['port entries'] = trial.rtc_events.pop('PC2_')

            # Remove the first entry because it doesn't count
            trial.rtc_events['sound cues'].onset_times = trial.rtc_events['sound cues'].onset[1:]
            trial.rtc_events['sound cues'].offset_times = trial.rtc_events['sound cues'].offset[1:]
            trial.rtc_events['port entries'].onset_times = trial.rtc_events['port entries'].onset[1:]
            trial.rtc_events['port entries'].offset_times = trial.rtc_events['port entries'].offset[1:]

            valid_sound_cues = [t for t in trial.rtc_events['sound cues'].onset_times if t >= 200]
            trial.rtc_events['sound cues'].onset_times = valid_sound_cues
            
            # Finding instances after the first tone is played
            port_entries_onset = np.array(trial.rtc_events['port entries'].onset_times)
            port_entries_offset = np.array(trial.rtc_events['port entries'].offset_times)
            first_sound_cue_onset = trial.rtc_events['sound cues'].onset_times[0]
            indices = np.where(port_entries_onset >= first_sound_cue_onset)[0]
            trial.rtc_events['port entries'].onset_times = port_entries_onset[indices].tolist()
            trial.rtc_events['port entries'].offset_times = port_entries_offset[indices].tolist()


    """********************************Combining consecutive entries************************************"""

    # def rtc_combine_consecutive_behaviors(self, behavior_name='all', bout_time_threshold=1, min_occurrences=1):
    #     """
    #     Applies the behavior combination logic to all trials within the experiment.
    #     """

    #     for trial_name, trial_obj in self.trials.items():
    #         # Ensure the trial has rtc_events attribute
    #         if not hasattr(trial_obj, 'rtc_events'):
    #             continue  # Skip if rtc_events is not available

    #         # Determine which behaviors to process
    #         if behavior_name == 'all':
    #             behaviors_to_process = trial_obj.rtc_events.keys()  # Process all behaviors
    #         else:
    #             behaviors_to_process = [behavior_name]  # Process a single behavior

    #         for behavior_event in behaviors_to_process:
    #             behavior_onsets = np.array(trial_obj.rtc_events[behavior_event].onset)
    #             behavior_offsets = np.array(trial_obj.rtc_events[behavior_event].offset)

    #             combined_onsets = []
    #             combined_offsets = []
    #             combined_durations = []

    #             if len(behavior_onsets) == 0:
    #                 continue  # Skip this behavior if there are no onsets

    #             start_idx = 0

    #             while start_idx < len(behavior_onsets):
    #                 # Initialize the combination window with the first behavior onset and offset
    #                 current_onset = behavior_onsets[start_idx]
    #                 current_offset = behavior_offsets[start_idx]

    #                 next_idx = start_idx + 1

    #                 # Check consecutive events and combine them if they fall within the threshold
    #                 while next_idx < len(behavior_onsets) and (behavior_onsets[next_idx] - current_offset) <= bout_time_threshold:
    #                     # Update the end of the combined bout
    #                     current_offset = behavior_offsets[next_idx]
    #                     next_idx += 1

    #                 # Add the combined onset, offset, and total duration to the list
    #                 combined_onsets.append(current_onset)
    #                 combined_offsets.append(current_offset)
    #                 combined_durations.append(current_offset - current_onset)

    #                 # Move to the next set of events
    #                 start_idx = next_idx

    #             # Filter out bouts with fewer than the minimum occurrences
    #             valid_indices = []
    #             for i in range(len(combined_onsets)):
    #                 num_occurrences = len([onset for onset in behavior_onsets if combined_onsets[i] <= onset <= combined_offsets[i]])
    #                 if num_occurrences >= min_occurrences:
    #                     valid_indices.append(i)

    #             # Update the behavior with the combined onsets, offsets, and durations
    #             trial_obj.rtc_events[behavior_event].onset = [combined_onsets[i] for i in valid_indices]
    #             trial_obj.rtc_events[behavior_event].offset = [combined_offsets[i] for i in valid_indices]
    #             trial_obj.rtc_events[behavior_event].Total_Duration = [combined_durations[i] for i in valid_indices]  # Update Total Duration

    #             trial_obj.bout_dict = {}  # Reset bout dictionary after processing

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
        self.df['sound cues onset'] = self.df['sound cues'].apply(lambda x: x.onset_times[:15] if x else None)
        self.df['port entries onset'] = self.df['port entries'].apply(lambda x: x.onset_times[:15] if x else None)
        self.df['port entries offset'] = self.df['port entries'].apply(lambda x: x.offset_times[:15] if x else None)

    def combining_cohorts(self, df1):
        df_combined = pd.concat([self.df, df1], ignore_index=True)
        # Filter rows where 'subject_name' is either 'n4', 'n7', or other specified 'n' values
        # List of subject names to remove
        subjects_to_remove = ["n4", "n3", "n2", "n1", 'p4']

        # Remove rows where 'subject_names' are in the list
        df_combined = df_combined[~df_combined['subject_name'].isin(subjects_to_remove)]

        # Display the result
        print(df_combined)
        
        self.df = df_combined
    '''********************************** PETH **********************************'''
    def rt_compute_peth_per_event(self, behavior_name='sound cues', n_events=None, pre_time=5, post_time=5, bin_size=0.1):
        """
        Computes the peri-event time histogram (PETH) data for each occurrence of a given behavior across all trials.
        Stores the peri-event data (zscore, time axis) for each event index in a dataframe

        Parameters:
        - behavior_name (str): The name of the behavior to generate the PETH for (e.g., 'sound cues').
        - n_events (int): The maximum number of events to analyze. If None, analyze all events.
        - pre_time (float): The time in seconds to include before the event.
        - post_time (float): The time in seconds to include after the event.
        - bin_size (float): The size of each bin in the histogram (in seconds).

        Returns:
        - None. Stores peri-event data for each event index across trials as a class variable.
        """

        # Initialize a dictionary to store peri-event data for each event index
        self.peri_event_data_per_event = {}

        # First, determine the maximum number of events across all trials if n_events is None
        if n_events is None:
            n_events = 0
            for block_data in self.trials.values():
                if behavior_name in block_data.rtc_events:
                    num_events = len(block_data.rtc_events[behavior_name].onset_times)
                    if num_events > n_events:
                        n_events = num_events

        # Define a common time axis
        time_axis = np.arange(-pre_time, post_time + bin_size, bin_size)
        self.time_axis = time_axis  # Store time_axis in the object

        # Initialize data structure
        for event_index in range(n_events):
            self.peri_event_data_per_event[event_index] = []

        # Loop through each block in self.trials
        print(self.trials.items())
        for block_name, block_data in self.trials.items():
            # Get the onset times for the behavior
            print(block_name)
            print(block_data)
            if behavior_name in block_data.rtc_events:
                event_onsets = block_data.rtc_events[behavior_name].onset_times
                # Limit to the first n_events if necessary
                event_onsets = event_onsets[:n_events]
                # For each event onset, compute the peri-event data
                for i, event_onset in enumerate(event_onsets):
                    # Define start and end times
                    start_time = event_onset - pre_time
                    end_time = event_onset + post_time

                    # Get indices for timestamps within this window
                    indices = np.where((block_data.timestamps >= start_time) & (block_data.timestamps <= end_time))[0]

                    if len(indices) == 0:
                        continue  # Skip if no data in this window

                    # Extract the corresponding zscore values
                    signal_segment = block_data.zscore[indices]

                    # Create a time axis relative to the event onset
                    timestamps_segment = block_data.timestamps[indices] - event_onset

                    # Interpolate the signal onto the common time axis
                    interpolated_signal = np.interp(time_axis, timestamps_segment, signal_segment)

                    # Store the interpolated signal in the data structure
                    self.peri_event_data_per_event[i].append(interpolated_signal)
            else:
                print(f"Behavior '{behavior_name}' not found in block '{block_name}'.")

    def plot_specific_peth(self, event_type, directory_path, brain_region, y_min, y_max, df=None):
        """
        Plots the PETH of the first and last bouts of either win or loss.
        """
        if df is None:
            df = self.df
        # Splitting either mPFC or NAc subjects
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            # Return filtered dataframes and subject_name column
            if region == 'mPFC':
                return df_p
            else:
                return df_n

        df = split_by_subject(df, brain_region)
        bin_size = 100
        if brain_region == 'mPFC':
            color = '#FFAF00'
        else:
            color = '#15616F'
        # Initialize data structures
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]
        first_events = []
        last_events = []
        for i, row in df.iterrows():
            print(f"Row {i}: Length of Z-score array: {len(row[f'{event_type} Event_Zscore'])}")

        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'])  # Shape: (num_1D_arrays, num_time_bins)

            # Ensure there is at least one 1D array in the row
            if len(z_scores) > 0:
                first_events.append(z_scores[0])   # First 1D array
                last_events.append(z_scores[14])   # Last 1D array

        # Convert lists to numpy arrays (num_trials, num_time_bins)
        first_events = np.array(first_events)
        last_events = np.array(last_events)

        # Compute mean and SEM
        mean_first = np.mean(first_events, axis=0)
        sem_first = np.std(first_events, axis=0) / np.sqrt(first_events.shape[0])

        mean_last = np.mean(last_events, axis=0)
        sem_last = np.std(last_events, axis=0) / np.sqrt(last_events.shape[0])

        mean_first, downsampled_time_axis = self.downsample_data(mean_first, common_time_axis, bin_size)
        sem_first, _ = self.downsample_data(sem_first, common_time_axis, bin_size)
        
        mean_last, _ = self.downsample_data(mean_last, common_time_axis, bin_size)
        sem_last, _ = self.downsample_data(sem_last, common_time_axis, bin_size)
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        axes[0].set_ylabel('Event Induced Z-scored ΔF/F', fontsize=18)
        axes[1].set_ylabel('Event Induced Z-scored ΔF/F', fontsize=18)

        # Show y-tick labels on both subplots
        axes[0].tick_params(axis='y', labelleft=True, labelsize=20)
        axes[1].tick_params(axis='y', labelleft=True, labelsize=20)

        for ax, mean_peth, sem_peth, title in zip(
            axes, [mean_first, mean_last], [sem_first, sem_last], 
            [f'First tone Z-Score', f'Last tone Z-Score']
        ):
            ax.plot(downsampled_time_axis, mean_peth, color=color, label='Mean DA')
            ax.fill_between(downsampled_time_axis, mean_peth - sem_peth, mean_peth + sem_peth, color=color, alpha=0.4)
            ax.axvline(0, color='black', linestyle='--', linewidth=3)  # Event onset
            ax.axvline(4, color='skyblue', linestyle='-', linewidth=3)  # Reward onset

            ax.set_title(title, fontsize=36)
            ax.set_xlabel('Time (s)', fontsize=24)
            
            ax.set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
            ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=20)
            
            ax.set_ylabel('Event Induced Z-scored ΔF/F', fontsize=24)
            ax.tick_params(axis='both', labelsize=20, width=3, length=6)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Increase thickness of bottom and left spines
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)

            ax.set_ylim(y_min, y_max)

        plt.subplots_adjust(wspace=0.3)
        save_path = os.path.join(str(directory_path) + '\\' + f'{brain_region}_{event_type}_PETH.png')
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        # plt.savefig(f'PETH.png', transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    """def rt_plot_peth_per_event(self, directory_path, title = 'PETH graph for n trials', signal_type = 'zscore', error_type='sem',
                            color='#00B7D7', display_pre_time=5, display_post_time=5, yticks_interval=2):
        
        Plots the PETH for each event index (e.g., each sound cue) across all trials in one figure with subplots.

        Parameters:
        - signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
        - error_type (str): The type of error to plot. Options are 'sem' for Standard Error of the Mean or 'std' for Standard Deviation.
        - title (str): Title for the figure.
        - color (str): Color for both the trace line and the error area (default is cyan '#00B7D7').
        - display_pre_time (float): How much time to show before the event on the x-axis (default is 5 seconds).
        - display_post_time (float): How much time to show after the event on the x-axis (default is 5 seconds).
        - yticks_interval (float): Interval for the y-ticks on the plots (default is 2).

        Returns:
        - None. Displays the PETH plot for each event index in one figure.
        
        # Get the time axis
        time_axis = self.time_axis

        # Determine the indices for the display range
        display_start_idx = np.searchsorted(time_axis, -display_pre_time)
        display_end_idx = np.searchsorted(time_axis, display_post_time)
        time_axis = time_axis[display_start_idx:display_end_idx]

        num_events = len(self.peri_event_data_per_event)
        if num_events == 0:
            print("No peri-event data available to plot.")
            return

        # Create subplots arranged horizontally
        fig, axes = plt.subplots(1, num_events, figsize=(5 * num_events, 5), sharey=True)

        # If there's only one event, make axes a list to keep the logic consistent
        if num_events == 1:
            axes = [axes]

        for idx, event_index in enumerate(range(num_events)):
            ax = axes[idx]
            event_traces = self.peri_event_data_per_event[event_index]
            if not event_traces:
                print(f"No data for event {event_index + 1}")
                continue
            # Convert list of traces to numpy array
            event_traces = np.array(event_traces)
            # Truncate the traces to the display range
            event_traces = event_traces[:, display_start_idx:display_end_idx]

            # Calculate the mean across trials
            mean_trace = np.mean(event_traces, axis=0)
            # Calculate SEM or Std across trials depending on the selected error type
            if error_type == 'sem':
                error_trace = np.std(event_traces, axis=0) / np.sqrt(len(event_traces))  # SEM
                error_label = 'SEM'
            elif error_type == 'std':
                error_trace = np.std(event_traces, axis=0)  # Standard Deviation
                error_label = 'Std'

            # Plot the mean trace with SEM/Std shaded area, with customizable color
            ax.plot(time_axis, mean_trace, color=color, label=f'Mean {signal_type.capitalize()}', linewidth=1.5)  # Trace color
            ax.fill_between(time_axis, mean_trace - error_trace, mean_trace + error_trace, color=color, alpha=0.3, label=error_label)  # Error color

            # Plot event onset line
            ax.axvline(0, color='black', linestyle='--', label='Event onset')

            # Set the x-ticks to show only the last time, 0, and the very end time
            ax.set_xticks([time_axis[0], 0, time_axis[-1]])
            ax.set_xticklabels([f'{time_axis[0]:.1f}', '0', f'{time_axis[-1]:.1f}'], fontsize=12)

            # Set the y-tick labels with specified interval
            if idx == 0:
                y_min, y_max = ax.get_ylim()
                y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval,
                                    np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval,
                                    yticks_interval)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=12)
            else:
                ax.set_yticks([])  # Hide y-ticks for other subplots

            ax.set_xlabel('Time (s)', fontsize=14)
            if idx == 0:
                ax.set_ylabel(f'{signal_type.capitalize()} dFF', fontsize=14)

            # Set the title for each event
            ax.set_title(f'Event {event_index + 1}', fontsize=14)

            # Remove the right and top spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Adjust layout and add a common title
        plt.suptitle(title, fontsize=16)
        save_path = os.path.join(str(directory_path) + '\\' + 'all_PETH.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        Plots the PETH for each event index (e.g., each sound cue) across all trials in one figure with subplots.

        Parameters:
        - signal_type (str): The type of signal to plot. Options are 'zscore' or 'dFF'.
        - error_type (str): The type of error to plot. Options are 'sem' for Standard Error of the Mean or 'std' for Standard Deviation.
        - title (str): Title for the figure.
        - color (str): Color for both the trace line and the error area (default is cyan '#00B7D7').
        - display_pre_time (float): How much time to show before the event on the x-axis (default is 5 seconds).
        - display_post_time (float): How much time to show after the event on the x-axis (default is 5 seconds).
        - yticks_interval (float): Interval for the y-ticks on the plots (default is 2).

        Returns:
        - None. Displays the PETH plot for each event index in one figure.
        """

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

        for index, row in df.iterrows():  # Use df, not self.df
            sound_cues_onsets = row['sound cues onset']
            port_entries_onsets = row['port entries onset']
            port_entries_offsets = row['port entries offset']

            first_licks_per_row = []

            for sc_onset in sound_cues_onsets:
                threshold_time = sc_onset + 4

                future_licks_indices = np.where(port_entries_onsets >= threshold_time)[0]

                if len(future_licks_indices) > 0:
                    first_licks_per_row.append(port_entries_onsets[future_licks_indices[0]])
                else:
                    ongoing_licks_indices = np.where((port_entries_onsets < threshold_time) & (port_entries_offsets > threshold_time))[0]

                    if len(ongoing_licks_indices) > 0:
                        first_licks_per_row.append(threshold_time)
                    else:
                        first_licks_per_row.append(None)

            first_licks.append(first_licks_per_row)

        df["first_lick_after_sound_cue"] = first_licks  # Add to the given DataFrame

        return df  # Return the modified DataFrame

    def compute_closest_port_offset(self, lick_column, offset_column, df=None):
        """
        Computes the closest port entry offsets after each lick time and adds them as a new column in the dataframe.
        
        Parameters:
            lick_column (str): The column name for the lick times (e.g., 'first_lick_after_sound_cue').
            offset_column (str): The column name for the port entry offset times (e.g., 'filtered_port_entry_offset').

        Returns:
            pd.DataFrame: Updated DataFrame with the new column of closest port entry offsets.
        """
        if df is None:
            df = self.df 

        def find_closest_port_entries(licks, port_entry_offsets):
            """Finds the closest port entry offsets greater than each lick in 'licks'."""
            closest_offsets = []
            
            # Ensure valid numerical values (filter out None/NaN)
            licks = [lick for lick in licks if lick is not None and not np.isnan(lick)]
            port_entry_offsets = [offset for offset in port_entry_offsets if offset is not None and not np.isnan(offset)]
            
            # Convert to numpy arrays
            licks = np.array(licks)
            port_entry_offsets = np.array(port_entry_offsets)
            
            if len(licks) == 0 or len(port_entry_offsets) == 0:
                return [np.nan] * len(licks)  # Return NaN list if either array is empty

            for lick in licks:
                valid_indices = np.where(port_entry_offsets > lick)[0]
                
                if len(valid_indices) == 0:
                    closest_offsets.append(np.nan)  # Append NaN if no valid offset is found
                else:
                    closest_offset = port_entry_offsets[valid_indices[0]]
                    closest_offsets.append(closest_offset)
            
            return closest_offsets

        def compute_lick_metrics(row):
            """Compute the closest port entry offsets for each trial."""
            # Extract lick times and port entry offsets safely
            first_licks = row[lick_column]
            port_entry_offsets = row[offset_column]

            if not isinstance(first_licks, list) or not isinstance(port_entry_offsets, list):
                return np.nan  # Return NaN if the data is not in the expected format

            # Get the closest port entry offsets for each lick
            return find_closest_port_entries(first_licks, port_entry_offsets)

        # Apply the function safely to avoid errors
        df['closest_lick_offset'] = df.apply(compute_lick_metrics, axis=1)

        return df

    def compute_event_induced_DA(self, df=None, pre_time=4, post_time=10):
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
                    if lick_event is None:
                        print("Warning: lick_event is None, skipping this row.")
                        continue  # Skip processing this row
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

                mask = (timestamps >= window_start) & (timestamps <= window_end)
                if not np.any(mask):
                    print(f"Warning: No timestamps found for event at {event_start}")
                    trial_event_zscores.append(np.full((1,), np.nan))
                    trial_event_times.append(np.full((1,), np.nan))
                    continue

                rel_time = timestamps[mask] - lick_start
                signal = zscore[mask]

                # Print to debug time range
                print(f"Relative Time Min = {np.min(rel_time)}, Max = {np.max(rel_time)}")

                pre_mask = (timestamps >= (cue_time - pre_time)) & (timestamps < cue_time)
                baseline = np.nanmean(zscore[pre_mask]) if np.any(pre_mask) else 0
                corrected_signal = signal - baseline

                common_time_axis = np.arange(-pre_time, post_time + min_dt, min_dt)

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
        def compute_da_metrics_for_trial(trial_obj, sound_cues):
            """Compute DA metrics (AUC, Max Peak, Time of Max Peak, Mean Z-score) for each sound cue, using adaptive peak-following."""
            """if not hasattr(trial_obj, "timestamps") or not hasattr(trial_obj, "zscore"):
                return np.nan"""  # Handle missing attributes

            timestamps = np.array(trial_obj.timestamps)  
            zscores = np.array(trial_obj.zscore)  

            computed_metrics = []
            for cue in sound_cues:
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
            if 'Tone Event_Time_Axis' not in df.columns or 'Mean Tone Event_Zscore' not in df.columns:
                print("Event-induced data not found in behaviors. Please run compute_event_induced_DA() first.")
                return df

            mean_zscores_all = []
            auc_values_all = []
            max_peaks_all = []
            peak_times_all = []

            for _, row in df.iterrows():
                final_time = np.array(row['Tone Event_Time_Axis'], dtype=float)[0]  # Ensure full array
                mean_event_zscore = np.array(row['Mean Tone Event_Zscore'], dtype=float)  # Convert to NumPy array

                if len(final_time) < len(mean_event_zscore):
                    print(f"Warning: final_time length {len(final_time)} < mean_event_zscore length {len(mean_event_zscore)}. Skipping.")
                    continue  

                final_time = final_time[-len(mean_event_zscore):]  # Trim to match z-score length

                # Apply mask for values between 0 and 4 seconds
                mask = (final_time >= 0) & (final_time <= 4)
                time_masked = final_time[mask]
                zscore_masked = mean_event_zscore[mask]

                if len(time_masked) == 0:
                    print("Warning: No values in the selected time range (0-4s). Skipping.")
                    continue  

                # Compute metrics
                mean_z = np.mean(zscore_masked)
                auc = np.trapz(zscore_masked, time_masked)  
                max_idx = np.argmax(zscore_masked)
                max_peak = zscore_masked[max_idx]
                peak_time = time_masked[max_idx]

                print(f"AUC (0-4s): {auc}")

                mean_zscores_all.append(mean_z)
                auc_values_all.append(auc)
                max_peaks_all.append(max_peak)
                peak_times_all.append(peak_time)

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
            if 'Lick Event_Time_Axis' not in df.columns or 'Mean Lick Event_Zscore' not in df.columns:
                print("Event-induced data not found in behaviors. Please run compute_event_induced_DA() first.")
                return df

            mean_zscores_all = []
            auc_values_all = []
            max_peaks_all = []
            peak_times_all = []

            for _, row in df.iterrows():
                final_time = np.array(row['Lick Event_Time_Axis'], dtype=float)[0] 
                mean_event_zscore = np.array(row['Mean Lick Event_Zscore'], dtype=float)  # Convert to NumPy array

                final_time = final_time[-len(mean_event_zscore):]  # Trim to match z-score length

                # Apply mask for values between 0 and 4 seconds
                mask = (final_time >= 0) & (final_time <= 4)
                time_masked = final_time[mask]
                
                mean_event_zscore = mean_event_zscore[-len(time_masked):]  
                zscore_masked = mean_event_zscore  # Now it matches time_masked
                # zscore_masked = mean_event_zscore[mask]

                if len(time_masked) == 0:
                    print("Warning: No values in the selected time range (0-4s). Skipping.")
                    continue  

                # Compute metrics
                mean_z = np.mean(zscore_masked)
                auc = np.trapz(zscore_masked, time_masked)  
                max_idx = np.argmax(zscore_masked)
                max_peak = zscore_masked[max_idx]
                peak_time = time_masked[max_idx]

                print(f"AUC (0-4s): {auc}")

                mean_zscores_all.append(mean_z)
                auc_values_all.append(auc)
                max_peaks_all.append(max_peak)
                peak_times_all.append(peak_time)

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

    def plot_trial_heatmaps(self, trial_numbers, event_type, directory_path, brain_region, df=None):
        """
        Plots heatmaps for the first 15 trials of a given condition (win/loss).
        Each heatmap represents one trial, showing Z-score variations over time.
        """
        if df is None:
            df = self.df
        # Function to filter data by brain region
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n
        
        df = split_by_subject(df, brain_region)

        # Extract data
        first_subject = df.iloc[0]
        common_time_axis = first_subject[f'{event_type} Event_Time_Axis'][0]

        trial_data_list = []
        
        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'])
            if len(z_scores) >= 15:
                trial_data_list = z_scores[:trial_numbers]  # Select the first 15 trials
                break  # Only need one subject's trials

        if len(trial_data_list) == 0:
            print(f"No valid trials found for {brain_region}")
            return
        
        # Downsample all trials
        bin_size = 125  
        downsampled_trials = []
        
        for trial in trial_data_list:
            downsampled_trial, new_time_axis = self.downsample_data(trial, common_time_axis, bin_size)
            downsampled_trials.append(downsampled_trial)

        downsampled_trials = np.array(downsampled_trials)  # Convert to 2D array (15 x time_bins)
        
        # Normalize color scale across trials
        vmin, vmax = downsampled_trials.min(), downsampled_trials.max()

        if brain_region == "mPFC":
            vmin, vmax = -3, 4
        else:
            vmin, vmax = -2, 10
        cmap = 'inferno' if brain_region == "mPFC" else 'viridis'

        # Set figure size dynamically based on the number of trials
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot heatmap
        cax = ax.imshow(downsampled_trials, aspect='auto', cmap=cmap, origin='upper',
                        extent=[common_time_axis[0], common_time_axis[-1], 0, len(downsampled_trials)],
                        vmin=vmin, vmax=vmax)

        # Formatting
        # ax.set_title(f'{brain_region}: all {trial_numbers} trials', fontsize=28)
        ax.set_yticks([])  # Remove y-axis labels for clean stacking
        ax.axvline(0, color='white', linestyle='--', linewidth=2)  # Mark event onset
        ax.axvline(4, color="#39FF14", linestyle='-', linewidth=2)

        # Set x-axis labels
        ax.set_xlabel('Time (s)', fontsize=26)
        ax.set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=26)

        # Add colorbar
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.7, label='Z-score')
        cbar.ax.tick_params(labelsize=26)
        cbar.set_label("Z-score", fontsize=26)

        # Save and show
        save_path = os.path.join(directory_path, f'{brain_region}_{trial_numbers}_Trials_Heatmap.png')
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_single_heatmaps(self, df, condition, event_type, directory_path, brain_region):
        """
        Plots heatmap of the mean Z-score across all trials of a given condition (win/loss).
        The heatmap represents the mean Z-score variations over time for all trials.
        """
        # Function to filter data by brain region
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            # Return filtered dataframes and subject_name column
            if region == 'mPFC':
                return df_p
            else:
                return df_n
        df = split_by_subject(df, brain_region)

        # Check if DataFrame is empty
        if df.empty:
            print("DataFrame is empty!")
            return  # Exit the function if no data is available

        # Check if the required column exists
        if f'{event_type} Event_Time_Axis' not in df.columns:
            print(f"Column '{event_type} Event_Time_Axis' not found in DataFrame!")
            return  # Exit if the column is missing

        # Extract data
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]
        all_trials = []
        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'], dtype=float)  # Convert to NumPy array
            
            if z_scores.ndim != 2:  # Ensure it's a 2D array
                print(f"Skipping row with invalid shape: {z_scores.shape}")
                continue
            
            mean_across_trials = np.mean(z_scores, axis=0)  # Mean across trials for this row
            all_trials.append(mean_across_trials)  # Store the result

        # Convert list to NumPy array and compute final mean across all rows
        all_trials = np.array(all_trials)
        mean_trial = np.mean(all_trials, axis=0)  # Mean across all subjects

        # Convert to 2D array (shape: (1, time_bins)) for heatmap
        bin_size = 125  
        mean_trial, new_time_axis = self.downsample_data(mean_trial, common_time_axis, bin_size)
        mean_trial = mean_trial[np.newaxis, :]  # Make it a 2D array for the heatmap

        # Normalize color scale
        vmin, vmax = mean_trial.min(), mean_trial.max()

        # Create a single figure (no subplots)
        fig, ax = plt.subplots(figsize=(10, 4))

        if brain_region == "mPFC":
            cmap = 'inferno'
        else:
            cmap = 'viridis'

        # Plot heatmap with mean trial data
        cax = ax.imshow(mean_trial, aspect='auto', cmap=cmap, origin='upper',
                        extent=[common_time_axis[0], common_time_axis[-1], 0, 1],
                        vmin=vmin, vmax=vmax)

        # Formatting
        ax.set_title(f'Mean {condition} Trial', fontsize=14)
        ax.set_yticks([])  # Remove y-axis ticks (since only one row)
        ax.axvline(0, color='white', linestyle='--', linewidth=2)  # Mark event onset

        # Set x-axis labels
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=10)

        # Add colorbar to represent Z-score intensity
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.7, label='Z-score')

        # Save and show
        save_path = os.path.join(directory_path, f'{brain_region}_{condition}_Mean_Trial_Heatmap.png')
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_linear_fit_with_error_bars(self, df, directory_path='directory_path', color='blue', y_limits=None, brain_region='mPFC'):
        """
        Plots the mean DA values with SEM error bars, fits a line of best fit,
        and computes the Pearson correlation coefficient.
        
        Parameters:
        - df: A pandas DataFrame containing trial numbers, mean DA signals, and SEMs.
        - color: The color of the error bars and data points.
        - y_limits: A tuple (y_min, y_max) to set the y-axis limits. If None, limits are set automatically.
        
        Returns:
        - slope: The slope of the line of best fit.
        - intercept: The intercept of the line of best fit.
        - r_value: The Pearson correlation coefficient.
        - p_value: The p-value for the correlation coefficient.
        """
        """# Sort the DataFrame by Trial
        df_sorted = self.df.sort_values('Trial')
        
        # Extract trial numbers, mean DA values, and SEMs
        x_data = df_sorted['Trial'].values
        y_data = df_sorted['Mean_DA'].values
        y_err = df_sorted['SEM_DA'].values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        y_fitted = intercept + slope * x_data
        
        # Plot the data with error bars and the fitted line
        # plt.figure(figsize=(12, 7))
        plt.figure(figsize=(30, 7))
        plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label = 'DA during Port Entry', color=color, 
                    capsize=10, markersize=20, elinewidth=4, capthick=3)
        plt.plot(x_data, y_fitted, 'r--', label=f'$R^2$ = {(r_value)**2:.2f}, p = {p_value:.3f}', linewidth=3)
        plt.xlabel('Tone Number', fontsize=36, labelpad=12)
        plt.ylabel('Global Z-scored ΔF/F', fontsize=36, labelpad=12)
        plt.title('', fontsize=10)
        plt.legend(fontsize=20)"""
        if df is None:
            df = self.df  # Use class DataFrame if no input is given

        # Function to filter subjects based on brain region
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)

        # Extract and truncate the first 15 elements of each array
        filtered_arrays = [np.array(arr[:15]) for arr in df['Mean Z-score EI'] if isinstance(arr, list)]

        # Stack into a 2D array (trials x 15 time points)
        stacked_arrays = np.vstack(filtered_arrays)  # Shape: (num_trials, 15)

        # Compute the mean across trials
        mean_array = np.nanmean(stacked_arrays, axis=0)  # Shape: (15,)

        # Check array length
        print(f"Mean array length: {len(mean_array)}")  # Should print 15

        # Generate trial numbers (assuming sequential indexing)
        x_data = np.arange(1, len(mean_array) + 1)
        y_data = mean_array

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        y_fitted = intercept + slope * x_data

        # Set figure size
        plt.figure(figsize=(30, 7))

        # Scatter plot
        plt.errorbar(x_data, y_data, yerr=np.nanstd(stacked_arrays, axis=0), fmt='o', label='DA during Port Entry', 
                    color=color, capsize=10, markersize=20, elinewidth=4, capthick=3)
        
        # Regression line
        plt.plot(x_data, y_fitted, 'r--', label=f'$R^2$ = {r_value**2:.2f}, p = {p_value:.3f}', linewidth=3)

        # Axis labels
        plt.xlabel("Tone Number", fontsize=36, labelpad=12)
        plt.ylabel("Global Z-scored ΔF/F", fontsize=36, labelpad=12)
        plt.title('', fontsize=10)
        plt.legend(fontsize=20)

        # Set custom x-ticks from 1 to 15 (every 2 steps)
        plt.xticks(np.arange(1, 16, 2), fontsize=26)

        # Define y-axis limits based on the brain region
        if "NAc" in str(directory_path):
            y_lower_limit, y_upper_limit = -1, 4
        else:  # mPFC
            y_lower_limit, y_upper_limit = -1, 3

        # Set y-axis ticks
        plt.yticks(np.arange(y_lower_limit, y_upper_limit), fontsize=26)

        # Remove the top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # Adjust tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=32, width=2)

        # Ensure a tight layout
        plt.tight_layout()

        # Save figure
        """save_path = os.path.join(str(directory_path), 'linear.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")"""

        # Show the plot
        plt.show()

        # Print regression stats
        print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")
        print(f"Pearson correlation coefficient (R): {r_value:.4f}, p-value: {p_value:.4e}")
        """# Set custom x-ticks from 2 to 16 (whole numbers)
        plt.xticks(np.arange(1, 15, 2), fontsize=26)

        if "NAc" in str(directory_path):
            y_lower_limit = -1
            y_upper_limit = 4
        else: #mPFC
            y_lower_limit = -1
            y_upper_limit = 3
        # Set y-axis limits if provided
        # if y_limits is not None:
        #     plt.ylim(y_limits)
        plt.yticks(np.arange(y_lower_limit, y_upper_limit), fontsize = 26)

        # Remove the top and right spines
        ax = plt.gca()  # Get current axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)    # Left axis line
        ax.spines['bottom'].set_linewidth(2)  # Bottom axis line

        
        # Optionally, adjust tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=32, width=2)  # Adjust tick label size and width


        plt.tight_layout()
        save_path = os.path.join(str(directory_path) + '\\linear.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # plt.savefig(f'linear.png', transparent=True, bbox_inches='tight', pad_inches=0.1)

        plt.show()
        
        print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")
        print(f"Pearson correlation coefficient (R): {r_value:.4f}, p-value: {p_value:.4e}")"""

    def rc_plot_peth_per_event(self, df, i, directory_path, title='PETH graph for n trials', signal_type='zscore', 
                            error_type='sem', display_pre_time=4, display_post_time=10, yticks_interval=2):
        # Plots the PETH for each event index (e.g., each sound cue) across all trials in one figure with subplots.
        
        if df is None:
            df = self.df
        # Determine the indices for the display range
        time_axis = df.iloc[i]['Tone Event_Time_Axis'][0]
        display_start_idx = np.searchsorted(time_axis, -display_pre_time)
        display_end_idx = np.searchsorted(time_axis, display_post_time)
        time_axis = time_axis[display_start_idx:display_end_idx]

        num_events = 15

        # Create subplots arranged horizontally
        fig, axes = plt.subplots(1, num_events, figsize=(5 * num_events, 5), sharey=True)
        print(len(df.iloc[i]['Tone Event_Zscore']))
        for idx, event_index in enumerate(range(num_events)):
            ax = axes[idx]

            tone_event_zscores = df.iloc[i]['Tone Event_Zscore']

            # Ensure it's a list/array and has enough elements
            if not isinstance(tone_event_zscores, (list, np.ndarray)) or len(tone_event_zscores) == 0:
                print(f"Skipping row {i}, 'Tone Event_Zscore' is empty or NaN.")
                continue

            if event_index >= len(tone_event_zscores):
                print(f"Skipping index {event_index}, out of range for row {i}.")
                continue  

            event_traces = tone_event_zscores[event_index]  # Safe access
            
            # Check if event_traces is empty
            if event_traces.size == 0:
                print(f"No data for event {event_index + 1}")
                continue
            
            # Convert list of traces to numpy array
            event_traces = np.array(event_traces)
            
            # Ensure event_traces is 2D (if only one trace, add a new axis to make it 2D)
            if event_traces.ndim == 1:
                event_traces = np.expand_dims(event_traces, axis=0)  # Convert 1D array to 2D
            
            # Truncate the traces to the display range
            event_traces = event_traces[:, display_start_idx:display_end_idx]

            # Plot the traces (without error bars)
            ax.plot(time_axis, event_traces.T, color='b', label=f'Event {event_index + 1}', linewidth=1.5)  # Transpose for proper plotting
            ax.axvline(0, color='black', linestyle='--', label='Event onset')
            ax.axvline(4, color='skyblue', linestyle='--', linewidth=2, label='t=4')

            # Set the x-ticks to show only the last time, 0, and the very end time
            ax.set_xticks([time_axis[0], 0, 4, time_axis[-1]])
            ax.set_xticklabels([f'{time_axis[0]:.1f}', '0', '4', f'{time_axis[-1]:.1f}'], fontsize=12)

            # Set the y-tick labels with specified interval
            # if idx == 0:
            y_min, y_max = ax.get_ylim()
            y_ticks = np.arange(np.floor(y_min / yticks_interval) * yticks_interval,
                                np.ceil(y_max / yticks_interval) * yticks_interval + yticks_interval,
                                yticks_interval)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{y:.0f}' for y in y_ticks], fontsize=12)
            """else:
                ax.set_yticks([])  # Hide y-ticks for other subplots"""

            ax.set_xlabel('Time (s)', fontsize=14)
            if idx == 0:
                ax.set_ylabel(f'{signal_type.capitalize()} dFF', fontsize=14)

            # Set the title for each event
            ax.set_title(f'Event {event_index + 1}', fontsize=14)

            # Remove the right and top spines for each subplot
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Adjust layout and add a common title
        plt.suptitle(title, fontsize=16)
        save_path = os.path.join(str(directory_path) + '\\' + str(i) + 'all_PETH.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


    def apply_rc_plot_peth(self, df, directory_path, title='PETH graph for n trials', 
                       signal_type='zscore', error_type='sem', display_pre_time=4, 
                       display_post_time=10, yticks_interval=2):
        
        # Applies the rc_plot_peth_per_event function to all rows in the DataFrame.
        
        # Loop through all rows in the DataFrame
        for i, row in df.iterrows():
            # Call the plotting function for each row
            self.rc_plot_peth_per_event(
                df, 
                i, 
                directory_path,
                title=title, 
                signal_type=signal_type, 
                error_type=error_type, 
                display_pre_time=display_pre_time, 
                display_post_time=display_post_time, 
                yticks_interval=yticks_interval
            )

    def plot_mean_psth(self, df, event_type, directory_path, brain_region, y_min, y_max):
        """
        Plots the PETH of all bouts averaged together.
        """
        if df is None:
            df = self.df

        # Splitting either mPFC or NAc subjects
        def split_by_subject(df1, region):            
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)
        bin_size = 100
        if brain_region == 'mPFC':
            color = '#FFAF00'
        else:
            color = '#15616F'
        # Initialize common time axis
        common_time_axis = df.iloc[0][f'{event_type} Event_Time_Axis'][0]

        row_means = []  # Store the mean PETH per row
        for _, row in df.iterrows():
            z_scores = np.array(row[f'{event_type} Event_Zscore'])  # Shape: (num_trials, num_time_bins)

            if z_scores.shape[0] > 0:  # Ensure there is data
                row_mean = np.mean(z_scores, axis=0)  # Mean across trials in a row
                row_means.append(row_mean)

        # Convert to numpy array and compute final mean and SEM
        row_means = np.array(row_means)  # Shape: (num_subjects, num_time_bins)
        mean_peth = np.mean(row_means, axis=0)  # Mean across subjects
        sem_peth = np.std(row_means, axis=0) / np.sqrt(row_means.shape[0])  # SEM

        # Downsample data
        mean_peth, downsampled_time_axis = self.downsample_data(mean_peth, common_time_axis, bin_size)
        sem_peth, _ = self.downsample_data(sem_peth, common_time_axis, bin_size)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.set_ylabel('Event Induced Z-scored ΔF/F', fontsize=20)
        ax.tick_params(axis='y', labelsize=16)

        # Plot mean and SEM
        ax.plot(downsampled_time_axis, mean_peth, color=color, label='Mean DA')
        ax.fill_between(downsampled_time_axis, mean_peth - sem_peth, mean_peth + sem_peth, color=color, alpha=0.4)
        ax.axvline(0, color='black', linestyle='--')  # Event onset
        ax.axvline(4, color="pink", linestyle='-')  # Reward onset

        # ax.set_title(f'{condition} bout Z-Score', fontsize=18)
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_xticks([common_time_axis[0], 0, 4, common_time_axis[-1]])
        ax.set_xticklabels(['-4', '0', '4', '10'], fontsize=16)

        ax.set_ylim(y_max, y_min)

        # Save figure
        save_path = os.path.join(str(directory_path), f'{brain_region}_PETH.png')
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()
    
    """*********************MISC.************************"""
    def downsample_data(self, data, time_axis, bin_size=10):
        """
        Downsamples the time series data by averaging over bins of 'bin_size' points.
        
        Parameters:
        - data (1D NumPy array): Original Z-score values.
        - time_axis (1D NumPy array): Corresponding time points.
        - bin_size (int): Number of original bins to merge into one.
        
        Returns:
        - downsampled_data (1D NumPy array): Smoothed Z-score values.
        - new_time_axis (1D NumPy array): Adjusted time points.
        """
        num_bins = len(data) // bin_size  # Number of new bins
        data = data[:num_bins * bin_size]  # Trim excess points
        time_axis = time_axis[:num_bins * bin_size]

        # Reshape and compute mean for each bin
        downsampled_data = data.reshape(num_bins, bin_size).mean(axis=1)
        new_time_axis = time_axis.reshape(num_bins, bin_size).mean(axis=1)

        return downsampled_data, new_time_axis
    def find_means(self, df=None):
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

    def find_overall_mean(self, df=None):
        if df is None:
            df = self.df
        
        # Function to compute mean for numerical values, preserving first for categorical ones
        def mean_arrays(group):
            result = {}
            result['Tone Event_Time_Axis'] = group['Tone Event_Time_Axis'].iloc[0]  # Time axis should be the same

            # Compute mean for scalar numerical values
            numerical_cols = [
                'Lick AUC', 'Lick Max Peak', 'Lick Mean Z-score',
                'Tone AUC', 'Tone Max Peak', 'Tone Mean Z-score',
                "Lick AUC EI", "Lick Max Peak EI", "Lick Mean Z-score EI",
                "Tone AUC EI", "Tone Max Peak EI", "Tone Mean Z-score EI"
            ]
            
            for col in numerical_cols:
                result[col] = group[col].mean()

            return pd.Series(result)

        df_mean = df.groupby('subject_name').apply(mean_arrays).reset_index()
        
        return df_mean
    
    def find_mean_event_zscore(self, df = None, behavior = 'Tone'):
        if df is None:
            df = self.df
        if 'Tone Event_Zscore' not in df.columns:
            print("Event-induced Z-score data not found in dataframe.")
            return df

        mean_zscores_all = []

        for _, row in df.iterrows():
            event_zscores = np.array(row['Tone Event_Zscore'])  # 2D array (trials × time points)

            if event_zscores.ndim == 2 and event_zscores.size > 0:
                mean_zscores = np.mean(event_zscores, axis=0)  # Compute mean across trials (row-wise)
            else:
                mean_zscores = np.nan  # Handle empty or invalid cases

            mean_zscores_all.append(mean_zscores)

        df[f'Mean {behavior} Event_Zscore'] = mean_zscores_all  # Store the mean 1D array in a new column
        return df