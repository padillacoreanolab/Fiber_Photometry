import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment
from trial_class import Trial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

class Reward_Training(Experiment):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path)
        self.trials = {}  # Reset trials to avoid loading from parent class
        self.load_rtc_trials()  # Load RTC trials instead

    def load_rtc_trials(self):
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
            self.trials[trial_name1] = trial_obj1
            self.trials[trial_name2] = trial_obj2

            # self.trials[trial_folder] = [trial_obj1, trial_obj2]

    def rt_processing(self, time_segments_to_remove=None):
        """
        Batch processes reward training
        """
        print(self.trials.items())
        for trial_folder, trial in self.trials.items():
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Reward Training Processing {trial_folder}...")
            trial.remove_initial_LED_artifact(t=30)
            trial.remove_final_data_segment(t = 10)
            
            trial.highpass_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            baseline_start, baseline_end = trial.find_baseline_period()  
            # trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            # trial.compute_zscore(method = 'standard')
            trial.compute_zscore(method = 'modified')
            trial.verify_signal()

            # PC0 = Tones
            # PC3 = Box 3
            # PC2 = Box 4
            # print(trial.behaviors1)
            trial.behaviors1['sound cues'] = trial.behaviors1.pop('PC0_')
            if int(trial_folder[-1]) % 2 == 0:
                trial.behaviors1['port entries'] = trial.behaviors1.pop('PC2_')
            else:    # use 'PC3_' instead
                trial.behaviors1['port entries'] = trial.behaviors1.pop('PC3_')


            # Remove the first entry because it doesn't count
            trial.behaviors1['sound cues'].onset_times = trial.behaviors1['sound cues'].onset[1:]
            trial.behaviors1['sound cues'].offset_times = trial.behaviors1['sound cues'].offset[1:]
            trial.behaviors1['port entries'].onset_times = trial.behaviors1['port entries'].onset[1:]
            trial.behaviors1['port entries'].offset_times = trial.behaviors1['port entries'].offset[1:]

            
            # Finding instances after first tone is played
            port_entries_onset = np.array(trial.behaviors1['port entries'].onset_times)
            port_entries_offset = np.array(trial.behaviors1['port entries'].offset_times)
            first_sound_cue_onset = trial.behaviors1['sound cues'].onset_times[0]
            indices = np.where(port_entries_onset >= first_sound_cue_onset)[0]
            trial.behaviors1['port entries'].onset_times = port_entries_onset[indices].tolist()
            trial.behaviors1['port entries'].offset_times = port_entries_offset[indices].tolist()

            self.combine_consecutive_behaviors1(behavior_name='all', bout_time_threshold=0.5)

    """********************************Combining consecutive entries************************************"""

    def combine_consecutive_behaviors1(self, behavior_name='all', bout_time_threshold=1, min_occurrences=1):
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

    '''********************************** PETH **********************************'''
    def rt_compute_peth_per_event(self, behavior_name='sound cues', n_events=None, pre_time=5, post_time=5, bin_size=0.1):
        """
        Computes the peri-event time histogram (PETH) data for each occurrence of a given behavior across all trials.
        Stores the peri-event data (zscore, time axis) for each event index as a class variable.

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
                if behavior_name in block_data.behaviors1:
                    num_events = len(block_data.behaviors1[behavior_name].onset_times)
                    if num_events > n_events:
                        n_events = num_events

        n_events = 40

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
            if behavior_name in block_data.behaviors1:
                event_onsets = block_data.behaviors1[behavior_name].onset_times
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


    def rt_plot_peth_per_event(self, directory_path, title = 'PETH graph for n trials', signal_type = 'zscore', error_type='sem',
                            color='#00B7D7', display_pre_time=5, display_post_time=5, yticks_interval=2):
        """
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


        """
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

    def plot_specific_peth(self, directory_path, brain_region):
        # Parameters
        selected_indices = [40]  # Specify which events to plot (1-based index)
        event_type = 'sound cues'  # Choose between 'port entries' or 'sound cues'
        pre_time = 4    # Time before event onset to include in PETH (seconds)
        post_time = 10   # Time after event onset to include in PETH (seconds)
        bin_size = 0.1  # Bin size for PETH (seconds)
        y_axis_limits = (-1.5, 2.5)  # Set y-axis limits as a tuple (min, max). Set to None for auto-scaling.

        # Initialize data structures
        peri_event_signals = [[] for _ in selected_indices]  # List to collect signals for each selected event
        common_time_axis = np.arange(-pre_time, post_time + bin_size, bin_size)

        # Iterate over all trials in exp
        for block_name, block_data in self.trials.items():
            print(f"Processing block: {block_name}")
            print(block_data)
            # Extract event onsets based on the chosen event type
            event_onsets = np.array(block_data.behaviors1[event_type].onset)
            
            # For each selected event
            for idx, event_index in enumerate(selected_indices):
                # Ensure the event_index is within the range of available events
                if event_index > len(event_onsets):
                    print(f"Event index {event_index} exceeds the number of {event_type} in block {block_name}. Skipping.")
                    continue
                
                # Get the onset of the specified event
                sc_onset = event_onsets[event_index - 1]  # 1-based index adjustment
                
                if event_type == 'port entries':
                    # For port entries, find the first port entry after the sound cue onset
                    pe_indices = np.where(block_data.behaviors1['sound cues'].onset > sc_onset)[0]
                    if len(pe_indices) == 0:
                        print(f"No sound cues found after {event_type} at {sc_onset} seconds in block {block_name}.")
                        continue
                    sc_onset = block_data.behaviors1['sound cues'].onset[pe_indices[0]]
                
                # Define time window around the event onset
                start_time = sc_onset - pre_time
                end_time = sc_onset + post_time
                
                # Get indices of DA signal within this window
                indices = np.where((block_data.timestamps >= start_time) & (block_data.timestamps <= end_time))[0]
                if len(indices) == 0:
                    print(f"No DA data found for {event_type} at {sc_onset} seconds in block {block_name}.")
                    continue
                
                # Extract DA signal and timestamps
                da_segment = block_data.zscore[indices]
                time_segment = block_data.timestamps[indices] - sc_onset  # Align time to event onset
                
                # Interpolate DA signal onto the common time axis
                interpolated_da = np.interp(common_time_axis, time_segment, da_segment)
                
                # Collect the interpolated DA signal
                peri_event_signals[idx].append(interpolated_da)

        # Now, peri_event_signals is a list where each element is a list of DA signals from each block for that event

        # Plot individual PETHs side by side for each selected event number
        num_events = len(selected_indices)
        fig, axes = plt.subplots(1, num_events, figsize=(4 * num_events, 7), sharey=True)
        if num_events == 1:
            axes = [axes]  # Ensure axes is iterable

        for i, ax in enumerate(axes):
            event_signals = peri_event_signals[i]
            if not event_signals:
                print(f"No data collected for {event_type} {i+1}.")
                continue
            # Convert to numpy array
            event_signals = np.array(event_signals)
            # Compute mean and SEM
            mean_peth = np.mean(event_signals, axis=0)
            sem_peth = np.std(event_signals, axis=0) / np.sqrt(len(event_signals))
            # Plot
            ax.plot(common_time_axis, mean_peth, color=brain_region, label='Mean DA')
            ax.fill_between(common_time_axis, mean_peth - sem_peth, mean_peth + sem_peth, color=brain_region, alpha=0.4)
            ax.axvline(0, color='black', linestyle='--')
            ax.set_title(f'Tone #{selected_indices[i]}', fontsize=24)
            ax.set_xlabel('Onset (s)', fontsize=26, labelpad=12)
            
            # Only show y-axis label and ticks on the first plot
            if i == 0:
                ax.set_ylabel('Event Induced Z-scored ΔF/F', fontsize=30, labelpad= 12)
            else:
                ax.tick_params(axis='y', labelleft=False)  # Hide y-ticks on subsequent plots

            # Set x-ticks and labels, including 6 seconds
            ax.set_xticks([common_time_axis[0], 0, 6, common_time_axis[-1]])
            ax.set_xticklabels([f'{common_time_axis[0]:.1f}', '0', '6.0', f'{common_time_axis[-1]:.1f}'], fontsize=20)
            ax.set_xticklabels(['-4', '0','6', '10'], fontsize=24)
            
            # Apply the same y-axis limits across all plots and ensure 0 is included
            if y_axis_limits:
                ax.set_ylim(y_axis_limits)
            else:
                # Adjust to include 0 in y-axis range
                min_y = min(0, np.min(mean_peth - sem_peth))
                max_y = max(0, np.max(mean_peth + sem_peth))
                ax.set_ylim(min_y, max_y)

            if "NAc" in str(directory_path):
                # NAc
                ax.set_yticks([-2, -1, 0, 1, 2, 3, 4, 5])
                ax.set_yticklabels(['-2.0', '-1.0', '0.0', '1.0', '2.0', '3.0', '4.0', '5.0'], fontsize=28)
            else:
                # mPFC
                ax.set_yticks([-2, -1, 0, 1, 2])
                ax.set_yticklabels(['-2.0', '-1.0', '0.0', '1.0', '2.0'], fontsize=28)

            # Manually ensure 0 is included in y-ticks
            y_ticks = ax.get_yticks()
            if 0 not in y_ticks:
                y_ticks = np.append(y_ticks, 0)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='both', which='major', labelsize=28, width=2)  # Adjust tick label size and width

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # **Adjust spine linewidths to make axes lines thicker**
            ax.spines['left'].set_linewidth(2)    # Left axis line
            ax.spines['bottom'].set_linewidth(2)  # Bottom axis line

        save_path = os.path.join(str(directory_path) + '\\' + 'PETH.png')
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # plt.savefig(f'PETH.png', transparent=True, bbox_inches='tight', pad_inches=0.1)

        # plt.suptitle(f'Mean PETH for Selected {event_type.capitalize()} Events', fontsize=16)
        plt.show()

    '''********************************** LICKS **********************************'''
    def find_first_lick_after_sound_cue(self):
        """
        Finds the first port entry occurring after 4 seconds following each sound cue.
        If a port entry starts before 4 seconds but extends past it, 
        the function selects the timestamp at 4 seconds after the sound cue.
        
        Stores the results as a list of timestamps in `self.first_lick_after_sound_cue`.
        """

        # Extract sound cue and port entry onset/offset times
        sound_cues_onsets = np.array(self.behaviors1['sound cues'].onset_times)
        port_entries_onsets = np.array(self.behaviors1['port entries'].onset_times)
        port_entries_offsets = np.array(self.behaviors1['port entries'].offset_times)

        first_licks = []  # List to store the first lick timestamp for each sound cue

        for sc_onset in sound_cues_onsets:
            threshold_time = sc_onset + 4  # Define the 4-second threshold after the sound cue

            # Find port entries that start AFTER 4 seconds post sound cue
            future_licks_indices = np.where(port_entries_onsets >= threshold_time)[0]

            if len(future_licks_indices) > 0:
                # If there are port entries after 4s, take the first one
                first_licks.append(port_entries_onsets[future_licks_indices[0]])
            else:
                # Find port entries that START before 4s but continue PAST it
                ongoing_licks_indices = np.where((port_entries_onsets < threshold_time) & (port_entries_offsets > threshold_time))[0]

                if len(ongoing_licks_indices) > 0:
                    # If a port entry spans past 4s, use the exact timepoint at 4s
                    first_licks.append(threshold_time)
                else:
                    # If no port entry occurs after 4s, append None (or NaN for consistency)
                    first_licks.append(None)

        # Store results in the object
        self.first_lick_after_sound_cue = first_licks

    def compute_mean_da_across_trials(self, n=40, pre_time=5, post_time=5, bin_size=0.1, mean_window=4):
        """
        Computes the mean DA signal across all trials for each of the first n sound cues.

        Parameters:
        - n: Number of sound cues to process.
        - pre_time: Time before port entry onset to include in PETH (seconds).
        - post_time: Time after port entry onset to include in PETH (seconds).
        - bin_size: Bin size for PETH (seconds).
        - mean_window: The time window (in seconds) from 0 to mean_window to compute the mean DA signal.

        Returns:
        - df: A pandas DataFrame containing trial numbers, mean DA signals, and SEMs.
        """
        # Initialize data structures
        peri_event_signals = [[] for _ in range(n)]  # List to collect signals for each of the first n port entries
        common_time_axis = np.arange(-pre_time, post_time + bin_size, bin_size)

        # Iterate over all trials
        for trial_name, trial_data in self.trials.items():
            print(f"Processing trial: {trial_name}")

            # Extract sound cue onsets and port entry onsets
            sound_cue_onsets = np.array(trial_data.behaviors1['sound cues'].onset)
            port_entry_onsets = np.array(trial_data.behaviors1['port entries'].onset)
            port_entry_offsets = np.array(trial_data.behaviors1['port entries'].offset)

            # Limit to the first n sound cues
            sound_cue_onsets = sound_cue_onsets[:n]

            # For each sound cue
            for sc_index, sc_onset in enumerate(sound_cue_onsets):
                reward_time = sc_onset + 6  # Reward issued at 6 seconds

                # Check if the subject was already in the port during the sound cue and stayed past 6s
                pe_indices_ongoing = np.where((port_entry_onsets < reward_time) & (port_entry_offsets > reward_time))[0]

                if len(pe_indices_ongoing) > 0:
                    # If a port entry was ongoing at 6s, set its time to exactly 6s
                    first_pe_index = pe_indices_ongoing[0]
                    pe_onset = reward_time
                else:
                    # Find the first port entry that starts after 6 seconds post sound cue
                    pe_indices_after = np.where(port_entry_onsets >= reward_time)[0]

                    if len(pe_indices_after) > 0:
                        # First port entry strictly after 6s
                        first_pe_index = pe_indices_after[0]
                        pe_onset = port_entry_onsets[first_pe_index]
                    else:
                        print(f"No valid port entry found after 6s for sound cue at {sc_onset} seconds in trial {trial_name}.")
                        continue

                # Define time window around the port entry onset
                start_time = pe_onset - pre_time
                end_time = pe_onset + post_time

                # Get indices of DA signal within this window
                indices = np.where((trial_data.timestamps >= start_time) & (trial_data.timestamps <= end_time))[0]
                if len(indices) == 0:
                    print(f"No DA data found for port entry at {pe_onset} seconds in trial {trial_name}.")
                    continue

                # Extract DA signal and timestamps
                da_segment = trial_data.zscore[indices]
                time_segment = trial_data.timestamps[indices] - pe_onset  # Align time to port entry onset

                # Interpolate DA signal onto the common time axis
                interpolated_da = np.interp(common_time_axis, time_segment, da_segment)

                # Collect the interpolated DA signal
                peri_event_signals[sc_index].append(interpolated_da)

        # Compute the mean DA signal and SEM across all trials for each port entry number
        trial_mean_da = []
        trial_sem_da = []
        mean_indices = np.where((common_time_axis >= 0) & (common_time_axis <= mean_window))[0]

        for event_signals in peri_event_signals:
            if event_signals:
                # Convert list of signals to numpy array
                event_signals = np.array(event_signals)
                # Compute mean PETH across all trials for this event
                mean_peth = np.mean(event_signals, axis=0)
                # Compute mean DA in the specified window
                mean_da = np.mean(mean_peth[mean_indices])
                sem_da = np.std(mean_peth[mean_indices]) / np.sqrt(len(event_signals))
                trial_mean_da.append(mean_da)
                trial_sem_da.append(sem_da)
            else:
                trial_mean_da.append(np.nan)  # Handle cases where no data is available
                trial_sem_da.append(np.nan)  # Handle cases where no data is available

        # Create a DataFrame to store the results
        df = pd.DataFrame({
            'Trial': np.arange(1, n + 1),
            'Mean_DA': trial_mean_da,
            'SEM_DA': trial_sem_da
        })

        return df


    
    def plot_linear_fit_with_error_bars(self, directory_path, df, color='blue', y_limits=None):
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
        # Sort the DataFrame by Trial
        df_sorted = df.sort_values('Trial')
        
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
        plt.legend(fontsize=20)
        
        # Set custom x-ticks from 2 to 16 (whole numbers)
        plt.xticks(np.arange(1, 43, 2), fontsize=26)

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
        print(f"Pearson correlation coefficient (R): {r_value:.4f}, p-value: {p_value:.4e}")
