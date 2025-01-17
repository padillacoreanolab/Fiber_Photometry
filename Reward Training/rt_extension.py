import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Reward_Training(Experiment):
    def rt_processing(self, time_segments_to_remove=None):
        """
        Batch processes reward training
        """
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
            trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            # trial.compute_zscore(method = 'standard')
            trial.verify_signal()

            trial.behaviors['sound cues'] = trial.behaviors.pop('PC0_')
            trial.behaviors['port entries'] = trial.behaviors.pop('PC3_')
            

            # Remove the first entry because it doesn't count
            trial.behaviors['sound cues'].onset_times = trial.behaviors['sound cues'].onset[1:]
            trial.behaviors['sound cues'].offset_times = trial.behaviors['sound cues'].offset[1:]
            trial.behaviors['port entries'].onset_times = trial.behaviors['port entries'].onset[1:]
            trial.behaviors['port entries'].offset_times = trial.behaviors['port entries'].offset[1:]

            
            # Finding instances after first tone is played
            port_entries_onset = np.array(trial.behaviors['port entries'].onset_times)
            port_entries_offset = np.array(trial.behaviors['port entries'].offset_times)
            first_sound_cue_onset = trial.behaviors['sound cues'].onset_times[0]
            indices = np.where(port_entries_onset >= first_sound_cue_onset)[0]
            trial.behaviors['port entries'].onset_times = port_entries_onset[indices].tolist()
            trial.behaviors['port entries'].offset_times = port_entries_offset[indices].tolist()

            trial.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=0.5)



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
                if behavior_name in block_data.behaviors:
                    num_events = len(block_data.behaviors[behavior_name].onset_times)
                    if num_events > n_events:
                        n_events = num_events

        # Define a common time axis
        time_axis = np.arange(-pre_time, post_time + bin_size, bin_size)
        self.time_axis = time_axis  # Store time_axis in the object

        # Initialize data structure
        for event_index in range(n_events):
            self.peri_event_data_per_event[event_index] = []

        # Loop through each block in self.trials
        for block_name, block_data in self.trials.items():
            # Get the onset times for the behavior
            if behavior_name in block_data.behaviors:
                event_onsets = block_data.behaviors[behavior_name].onset_times
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


    def rt_plot_peth_per_event(self, signal_type='zscore', error_type='sem', title='PETH for First n Sound Cues',
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
        sound_cues_onsets = np.array(self.behaviors['sound cues'].onset_times)
        port_entries_onsets = np.array(self.behaviors['port entries'].onset_times)
        port_entries_offsets = np.array(self.behaviors['port entries'].offset_times)

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




    def compute_mean_da_across_trials(self, n=15, pre_time=5, post_time=5, bin_size=0.1, mean_window=4):
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
            sound_cue_onsets = np.array(trial_data.behaviors['sound cues'].onset)
            port_entry_onsets = np.array(trial_data.behaviors['port entries'].onset)
            port_entry_offsets = np.array(trial_data.behaviors['port entries'].offset)

            # Limit to the first n sound cues
            sound_cue_onsets = sound_cue_onsets[:n]

            # For each sound cue
            for sc_index, sc_onset in enumerate(sound_cue_onsets):
                reward_time = sc_onset + 4  # Reward issued at 6 seconds

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


    
    def plot_linear_fit_with_error_bars(self, df, color='blue', y_limits=None):
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
        plt.figure(figsize=(12, 7))
        plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label = 'DA during Port Entry', color=color, 
                    capsize=10, markersize=20, elinewidth=4, capthick=3)
        plt.plot(x_data, y_fitted, 'r--', label=f'$R^2$ = {(r_value)**2:.2f}, p = {p_value:.3f}', linewidth=3)
        plt.xlabel('Tone Number', fontsize=36, labelpad=12)
        plt.ylabel('Global Z-scored ΔF/F', fontsize=36, labelpad=12)
        plt.title('', fontsize=10)
        plt.legend(fontsize=20)
        
        # Set custom x-ticks from 2 to 16 (whole numbers)
        plt.xticks(np.arange(1, 14, 2), fontsize=26)

        # Set y-axis limits if provided
        if y_limits is not None:
            plt.ylim(y_limits)

        # Remove the top and right spines
        ax = plt.gca()  # Get current axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)    # Left axis line
        ax.spines['bottom'].set_linewidth(2)  # Bottom axis line

        
        # Optionally, adjust tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=32, width=2)  # Adjust tick label size and width


        plt.tight_layout()
        plt.savefig(f'linear.png', transparent=True, bbox_inches='tight', pad_inches=0.1)

        plt.show()
        
        print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")
        print(f"Pearson correlation coefficient (R): {r_value:.4f}, p-value: {p_value:.4e}")