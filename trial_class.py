import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt

from sklearn.linear_model import LinearRegression

class Trial:
    def __init__(self, trial_path, stream_DA, stream_ISOS):
        """tdtdata = tdt.read_block(trial_path)
        
        # Extract the subject name from the folder or file name
        self.subject_name = os.path.basename(trial_path).split('-')[0]
        
        # Assume all streams have the same sampling frequency and length
        self.fs = tdtdata.streams["_465A"].fs
        self.timestamps = np.arange(len(tdtdata.streams['_465A'].data)) / self.fs

        self.streams = {}
        self.streams['DA'] = tdtdata.streams['_465A'].data
        self.streams['ISOS'] = tdtdata.streams['_405A'].data

        self.behaviors = None
        
        # To ensure that the most updated DA does not require specific preprocessing steps, each of the preprocessing functions will update these. Make sure your ordering is correct
        self.updated_DA = self.streams['DA']
        self.updated_ISOS = self.streams['ISOS']

        self.isosbestic_fitted = np.empty(1)

        self.dFF = np.empty(1)
        self.zscore = np.empty(1)
        """
        tdtdata = tdt.read_block(trial_path)

        self.subject_name = os.path.basename(trial_path).split('-')[0]
        self.fs = tdtdata.streams[stream_DA].fs
        self.timestamps = np.arange(len(tdtdata.streams[stream_DA].data)) / self.fs

        self.streams = {}
        self.streams['DA'] = tdtdata.streams[stream_DA].data
        self.streams['ISOS'] = tdtdata.streams[stream_ISOS].data

        self.behaviors = None
        self.behaviors1 = {key: value for key, value in tdtdata.epocs.items() if key not in ['Cam1', 'Cam2', 'Tick']}

        self.updated_DA = self.streams['DA']
        self.updated_ISOS = self.streams['ISOS']

        self.isosbestic_fitted = np.empty(1)
        self.dFF = np.empty(1)
        self.zscore = np.empty(1)


    '''********************************** PREPROCESSING **********************************'''
    def remove_initial_LED_artifact(self, t=30):
        '''
        This function removes intial LED artifact when starting a recording, assumed to be the first 't' seconds.
        It truncates the streams and timestamps accordingly.
        '''
        ind = np.where(self.timestamps > t)[0][0]
        
        for stream_name in ['DA', 'ISOS']:
            self.streams[stream_name] = self.streams[stream_name][ind:]
        
        self.updated_DA = self.streams['DA']
        self.updated_ISOS = self.streams['ISOS']
        self.timestamps = self.timestamps[ind:]

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None


    def remove_final_data_segment(self, t=30):
        '''
        This function removes the final segment of the data, assumed to be the last 't' seconds.
        It truncates the streams and timestamps accordingly.
        '''
        end_time = self.timestamps[-1] - t
        ind = np.where(self.timestamps <= end_time)[0][-1]
        
        for stream_name in ['DA', 'ISOS']:
            self.streams[stream_name] = self.streams[stream_name][:ind+1]
        
        self.timestamps = self.timestamps[:ind+1]
        self.updated_DA = self.streams['DA']
        self.updated_ISOS = self.streams['ISOS']

        # Clear dFF and zscore since the raw data has changed
        self.dFF = None
        self.zscore = None


    def remove_time_segment(self, start_time, end_time):
        """
        Remove the specified time segment between start_time and end_time
        from the timestamps and associated signal streams.
        """

        start_idx = np.searchsorted(self.timestamps, start_time)
        end_idx = np.searchsorted(self.timestamps, end_time)

        if start_idx >= end_idx:
            raise ValueError("Invalid time segment. start_time must be less than end_time.")

        self.timestamps = np.concatenate([self.timestamps[:start_idx], self.timestamps[end_idx:]])
        self.streams['DA'] = np.concatenate([self.streams['DA'][:start_idx], self.streams['DA'][end_idx:]])
        self.streams['ISOS'] = np.concatenate([self.streams['ISOS'][:start_idx], self.streams['ISOS'][end_idx:]])

        if hasattr(self, 'updated_DA'):
            self.updated_DA = np.concatenate([self.updated_DA[:start_idx], self.updated_DA[end_idx:]])
        if hasattr(self, 'updated_ISOS'):
            self.updated_ISOS = np.concatenate([self.updated_ISOS[:start_idx], self.updated_ISOS[end_idx:]])

        print(f"Removed time segment from {start_time}s to {end_time}s.")


    def smooth_and_apply(self, window_len=1):
        """Smooth both DA and ISOS signals using a window with requested size, and store them.
        """

        def smooth_signal(source, window_len):
            """Helper function to smooth a signal."""
            if source.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
            if source.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
            if window_len < 3:
                return source

            # Extend the signal by reflecting at the edges
            s = np.r_[source[window_len-1:0:-1], source, source[-2:-window_len-1:-1]]
            # Create a window for smoothing (using a flat window here)
            w = np.ones(window_len, 'd')
            # Convolve and return the smoothed signal
            return np.convolve(w / w.sum(), s, mode='valid')


        # Apply smoothing to DA and ISOS streams, then trim the excess padding
        if 'DA' in self.streams:
            smoothed_DA = smooth_signal(self.streams['DA'], window_len)
            # Trim the excess by slicing the array to match the original length
            self.updated_DA = smoothed_DA[window_len//2:-window_len//2+1]

        if 'ISOS' in self.streams:
            smoothed_ISOS = smooth_signal(self.streams['ISOS'], window_len)
            self.updated_ISOS = smoothed_ISOS[window_len//2:-window_len//2+1]


    def centered_moving_average_with_padding(self, source, window=1):
        """
        Applies a centered moving average to the input signal with edge padding to preserve the signal length.
        Used in apply_ma_baseline_correction
        
        Args:
            source (np.array): The signal for which the moving average is computed.
            window (int): The window size used to compute the moving average.

        Returns:
            np.array: The centered moving average of the input signal with the original length preserved.
        """
        source = np.array(source)

        if len(source.shape) == 1:
            # Pad the signal by reflecting the edges to avoid cutting
            padded_source = np.pad(source, (window // 2, window // 2), mode='reflect')

            # Calculate the cumulative sum and moving average
            cumsum = np.cumsum(padded_source)
            moving_avg = (cumsum[window:] - cumsum[:-window]) / float(window)
            
            return moving_avg[:len(source)]
        else:
            raise RuntimeError(f"Input array has too many dimensions. Input: {len(source.shape)}D, Required: 1D")
    

    def apply_ma_baseline_drift(self, window_len_seconds=30):
        """
        Applies centered moving average (MA) to both DA and ISOS signals and performs baseline correction,
        with padding to avoid shortening the signals. 

        Args:
            window_len_seconds (int): The window size in seconds for the moving average filter (default: 30 seconds).
        """
        # Adjust the window length in data points
        window_len = int(self.fs) * window_len_seconds

        # Apply centered moving average with padding to both DA and ISOS streams
        isosbestic_fc = self.centered_moving_average_with_padding(self.updated_ISOS, window_len)
        DA_fc = self.centered_moving_average_with_padding(self.updated_DA, window_len)

        self.updated_ISOS = (self.updated_ISOS - isosbestic_fc) / isosbestic_fc
        self.updated_DA = (self.updated_DA - DA_fc) / DA_fc


    def highpass_baseline_drift(self, cutoff=0.001):
        """
        Applies a high-pass Butterworth filter to remove slow drift from the DA and ISOS signals.
        https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
        """
        # Design a second-order Butterworth high-pass filter
        b, a = butter(N=2, Wn=cutoff, btype='high', fs=self.fs)

        # Apply zero-phase filtering to avoid phase distortion
        self.updated_DA = filtfilt(b, a, self.streams['DA'], padtype='even')
        self.updated_ISOS = filtfilt(b, a, self.streams['ISOS'], padtype='even')


    def align_channels(self):
        """
        Function that performs linear regression between isosbestic_corrected and DA_corrected signals, and aligns
        the fitted isosbestic with the DA signal. 

        Baseline correction must have occurred
        """
        reg = LinearRegression()
        
        n = len(self.updated_DA)
        reg.fit(self.updated_ISOS.reshape(n, 1), self.updated_DA.reshape(n, 1))
        self.isosbestic_fitted = reg.predict(self.updated_ISOS.reshape(n, 1)).reshape(n,)


    def compute_dFF(self):
        """
        Function that computes the dF/F of the fitted isosbestic and DA signals and saves it in self.dFF.
        """
        isosbestic = self.isosbestic_fitted
        da = self.updated_DA
        
        # Compute dF/F by subtracting the fitted isosbestic from the DA signal
        df_f = da - isosbestic
        
        # Save the computed dF/F into the class attribute
        self.dFF = df_f
        

    def compute_zscore(self, method='standard', baseline_start=None, baseline_end=None):
        """
        Computes the z-score of the delta F/F (dFF) signal and saves it as a class variable.
        The baseline period is used as a reference point, making the entire signal relative to it.

        Options are:
            'standard' - Computes the z-score using the standard method (z = (x - mean) / std).
            'baseline' - Computes the z-score using a baseline period, with the entire signal relative to this baseline.
            'modified' - Computes the z-score using a modified z-score method (z = 0.6745 * (x - median) / MAD).
    
        """
        dff = np.array(self.dFF)
        
        if method == 'standard':
            self.zscore = (dff - np.nanmean(dff)) / np.nanstd(dff)
        
        elif method == 'baseline':
            if baseline_start is None or baseline_end is None:
                raise ValueError("Baseline start and end times must be provided for baseline z-score computation.")
            
            baseline_indices = np.where((self.timestamps >= baseline_start) & (self.timestamps <= baseline_end))[0]
            if len(baseline_indices) == 0:
                raise ValueError("No baseline data found within the specified baseline period.")
            
            baseline_mean = np.nanmean(dff[baseline_indices])
            baseline_std = np.nanstd(dff[baseline_indices])

            self.zscore = (dff - baseline_mean) / baseline_std

        elif method == 'modified':
            median = np.nanmedian(dff)
            mad = np.nanmedian(np.abs(dff - median))
            self.zscore = 0.6745 * (dff - median) / mad
        
        else:
            raise ValueError("Invalid zscore_method. Choose from 'standard', 'baseline', or 'modified'.")


    def verify_signal(self):
        '''
        This function verifies that DA and ISOS are the same length. If not, it shortens it to
        whatever is the smallest
        '''
        da_length = len(self.streams['DA'])
        isos_length = len(self.streams['ISOS'])
        min_length = min(da_length, isos_length)

        if da_length != min_length or isos_length != min_length:
            print("Trimming")
            # Trim the streams to the shortest length
            self.streams[self.DA] = self.streams[self.DA][:min_length]
            self.streams[self.ISOS] = self.streams[self.ISOS][:min_length]
            self.timestamps = self.timestamps[:min_length]



    '''********************************** BEHAVIORS **********************************'''
    def extract_bouts_and_behaviors(self, csv_path, bout_definitions):
        """
        Reads an aggregated behavior CSV from csv_path, extracts behavior events occurring within bouts,
        and stores the result as a DataFrame in self.behaviors.

        Each bout is defined by an "introduced" event and a "removed" event. Multiple bout definitions
        can be provided to handle different naming conventions.

        Parameters:
        - csv_path (str): File path to the CSV containing the behavior data.
        - bout_definitions (list of dict): A list where each dict defines a bout type with keys:
            - 'prefix': A string used to label bouts (e.g., "s1", "s2", "x", etc.).
            - 'introduced': The name of the introduced event (e.g., "s1_Introduced", "X_Introduced", etc.).
            - 'removed': The name of the removed event (e.g., "s1_Removed", "X_Removed", etc.).

        The resulting DataFrame will have one row per behavior event (that is not a boundary event)
        with the following columns:
        - Bout: Bout label (e.g., "s1-1", "s2-1", etc.).
        - Behavior: The behavior name.
        - Event_Start: Event start time.
        - Event_End: Event end time.
        - Duration (s): Duration of the event (Stop - Start).
        - First Investigation: Indicates if this was the first investigation event in the bout.
        """

        # 1. Read CSV and ensure numeric columns
        data = pd.read_csv(csv_path)

        # Ensure 'Start (s)' and 'Stop (s)' are numeric
        data['Start (s)'] = pd.to_numeric(data['Start (s)'], errors='coerce')
        data['Stop (s)'] = pd.to_numeric(data['Stop (s)'], errors='coerce')

        # 2. Filter for the subject's data only
        data = data[data['Subject'] == 'Subject']

        # 3. Build a unique set of boundary behaviors from bout definitions
        boundary_behaviors = {bout_def['introduced'] for bout_def in bout_definitions} | {
            bout_def['removed'] for bout_def in bout_definitions
        }

        # 4. Helper function to extract events within each bout
        def extract_bout_events(df, introduced_behavior, removed_behavior, bout_prefix):
            introduced_df = df[df['Behavior'] == introduced_behavior].sort_values('Start (s)').reset_index(drop=True)
            removed_df = df[df['Behavior'] == removed_behavior].sort_values('Start (s)').reset_index(drop=True)
            num_bouts = min(len(introduced_df), len(removed_df))

            rows = []
            for i in range(num_bouts):
                bout_label = f"{bout_prefix}-{i+1}"
                bout_start = introduced_df.loc[i, 'Start (s)']
                bout_end = removed_df.loc[i, 'Start (s)']

                # Select only behaviors within this bout, excluding boundary behaviors
                subset = df[
                    (~df['Behavior'].isin(boundary_behaviors)) &
                    (df['Start (s)'] >= bout_start) & 
                    (df['Stop (s)'] <= bout_end)
                ]

                # Identify the first investigation event per bout
                first_investigation = (
                    subset[subset['Behavior'] == 'Investigation']
                    .sort_values('Start (s)')
                    .head(1)  # Keep only the first occurrence
                )

                for _, row in subset.iterrows():
                    rows.append({
                        'Bout': bout_label,
                        'Behavior': row['Behavior'],
                        'Event_Start': row['Start (s)'],
                        'Event_End': row['Stop (s)'],
                        'Duration (s)': row['Stop (s)'] - row['Start (s)'],
                        'First Investigation': int(row['Behavior'] == 'Investigation' and not first_investigation.empty and row['Start (s)'] == first_investigation['Start (s)'].values[0])
                    })

            return rows

        # 5. Extract behavior events for all bout definitions
        bout_rows = []
        for bout_def in bout_definitions:
            prefix = bout_def['prefix']
            introduced_behavior = bout_def['introduced']
            removed_behavior = bout_def['removed']
            bout_rows.extend(extract_bout_events(data, introduced_behavior, removed_behavior, prefix))

        # 6. Store the resulting DataFrame in the instance
        self.behaviors = pd.DataFrame(bout_rows)

    """def combine_consecutive_behaviors1(self, behavior_name='all', bout_time_threshold=1, min_occurrences=1):
        
        Combines consecutive behavior events if they occur within a specified time threshold,
        and updates the Total Duration.

        Parameters:
        - behavior_name (str): The name of the behavior to process. If 'all', process all behaviors.
        - bout_time_threshold (float): Maximum time gap (in seconds) between consecutive behaviors to be combined.
        - min_occurrences (int): Minimum number of occurrences required for a combined bout to be kept.
        

        # Determine which behaviors to process
        if behavior_name == 'all':
            behaviors_to_process = self.behaviors.keys()  # Process all behaviors
        else:
            behaviors_to_process = [behavior_name]  # Process a single behavior

        for behavior_event in behaviors_to_process:
            behavior_onsets = np.array(self.behaviors[behavior_event].onset)
            behavior_offsets = np.array(self.behaviors[behavior_event].offset)

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
            self.behaviors1[behavior_event].onset = [combined_onsets[i] for i in valid_indices]
            self.behaviors1[behavior_event].offset = [combined_offsets[i] for i in valid_indices]
            self.behaviors1[behavior_event].Total_Duration = [combined_durations[i] for i in valid_indices]  # Update Total Duration

            self.bout_dict = {}"""


    def combine_consecutive_behaviors(self, behavior_name='all', bout_time_threshold=1):
        """
        Combines consecutive behavior events if they occur within a specified time threshold
        and updates the self.behaviors DataFrame.

        Parameters:
        - behavior_name (str): The behavior type to process. If 'all', process all behaviors.
        - bout_time_threshold (float): Maximum time (in seconds) between consecutive events to merge them.
        """
        if self.behaviors.empty:
            return  # No behaviors to process

        df = self.behaviors.copy()  # Work on a copy to avoid modifying during iteration

        # Select behaviors to process
        if behavior_name != 'all':
            df = df[df['Behavior'] == behavior_name]

        # Sort by Bout and Event_Start to ensure proper merging
        df = df.sort_values(by=['Bout', 'Behavior', 'Event_Start']).reset_index(drop=True)

        # Storage for combined rows
        combined_rows = []
        
        # Iterate through groups of behaviors
        for (bout, behavior), group in df.groupby(['Bout', 'Behavior']):
            group = group.sort_values('Event_Start').reset_index(drop=True)
            
            # Initialize first behavior
            current_start = group.loc[0, 'Event_Start']
            current_end = group.loc[0, 'Event_End']

            for i in range(1, len(group)):
                next_start = group.loc[i, 'Event_Start']
                next_end = group.loc[i, 'Event_End']

                # If the next behavior starts within the threshold, merge it
                if next_start - current_end <= bout_time_threshold:
                    current_end = next_end  # Extend the current behavior
                else:
                    # Store merged event
                    combined_rows.append({
                        'Bout': bout,
                        'Behavior': behavior,
                        'Event_Start': current_start,
                        'Event_End': current_end,
                        'Duration (s)': current_end - current_start
                    })
                    # Reset to next behavior
                    current_start = next_start
                    current_end = next_end

            # Store the last behavior
            combined_rows.append({
                'Bout': bout,
                'Behavior': behavior,
                'Event_Start': current_start,
                'Event_End': current_end,
                'Duration (s)': current_end - current_start
            })

        # Convert back to DataFrame and update
        self.behaviors = pd.DataFrame(combined_rows)

    def remove_short_behaviors(self, behavior_name='all', min_duration=0):
        """
        Removes behaviors with a duration less than the specified minimum duration.

        Parameters:
        - behavior_name (str): The behavior type to filter. If 'all', process all behaviors.
        - min_duration (float): Minimum duration (in seconds) required to keep a behavior.
        """
        if self.behaviors.empty:
            return  # No behaviors to process

        df = self.behaviors.copy()

        # Filter by behavior if specified
        if behavior_name != 'all':
            df = df[df['Behavior'] == behavior_name]

        # Apply duration filter
        df = df[df['Duration (s)'] >= min_duration]

        # Update self.behaviors
        self.behaviors = df.reset_index(drop=True)




    '''********************************** PLOTTING **********************************'''
    def plot_behavior_event(self, behavior_name='all', ax=None):
        """
        Plot z-score signal with behavior event spans using the updated behaviors DataFrame.
        Adjusts x-limits to remove unnecessary blank space at the beginning and end.
        """
        y_data = self.zscore
        y_label = 'z-score'
        y_title = 'z-score Signal'

        # Create a new figure if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='black')

        # Define colors for behaviors
        behavior_colors = {
            'Investigation': 'dodgerblue', 
            'Approach': 'green', 
            'Defeat': 'red', 
            'Pinch': 'dodgerblue'
        }

        behavior_labels_plotted = set()

        # Determine x-axis limits based on behavior events
        if behavior_name == 'all':
            min_time = self.behaviors['Event_Start'].min() - 30
            max_time = self.behaviors['Event_End'].max() + 30
        else:
            behavior_df = self.behaviors[self.behaviors['Behavior'] == behavior_name]
            if behavior_df.empty:
                raise ValueError(f"Behavior event '{behavior_name}' not found in behaviors.")
            min_time = behavior_df['Event_Start'].min() - 30
            max_time = behavior_df['Event_End'].max() + 30

        # Iterate through the behaviors DataFrame
        if behavior_name == 'all':
            for _, row in self.behaviors.iterrows():
                event_name = row['Behavior']
                if event_name in behavior_colors:
                    color = behavior_colors[event_name]
                    on, off = row['Event_Start'], row['Event_End']
                    if event_name not in behavior_labels_plotted:
                        ax.axvspan(on, off, alpha=0.25, label=event_name, color=color)
                        behavior_labels_plotted.add(event_name)
                    else:
                        ax.axvspan(on, off, alpha=0.25, color=color)
        else:
            color = behavior_colors.get(behavior_name, 'dodgerblue')
            for _, row in behavior_df.iterrows():
                ax.axvspan(row['Event_Start'], row['Event_End'], alpha=0.25, color=color)

        ax.set_xlim(min_time, max_time)  # Adjust x-axis limits to remove blank space
        ax.set_ylabel(y_label)
        ax.set_xlabel('Seconds')
        ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

        if behavior_labels_plotted:
            ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()



    '''********************************** MISC **********************************'''
    def find_baseline_period(self):
            """
            Finds the baseline period from the beginning of the timestamps array to 2 minutes after.

            Returns:
            baseline_start (float): The start time of the baseline period (always 0).
            baseline_end (float): The end time of the baseline period (2 minutes after the start).
            """
            if self.timestamps is None or len(self.timestamps) == 0:
                raise ValueError("Timestamps data is missing or empty.")
            
            # Duration of the baseline period in seconds
            baseline_duration_in_seconds = 2 * 60  + 20 # 2 minutes 20 seconds

            # Calculate the end time for the baseline period
            baseline_end_time = self.timestamps[0] + baseline_duration_in_seconds

            # Ensure the baseline period does not exceed the data length
            if baseline_end_time > self.timestamps[-1]:
                baseline_end_time = self.timestamps[-1]

            baseline_start = self.timestamps[0]
            baseline_end = baseline_end_time

            return baseline_start, baseline_end
        
    def compute_da_metrics(self, 
                            use_fractional=False, 
                            max_bout_duration=30, 
                            use_adaptive=False, 
                            peak_fall_fraction=0.5,
                            allow_bout_extension=False,
                            first=False):
        """
        Computes DA metrics for each behavior event (row) in self.behaviors.

        If `first=True`, the behavior DataFrame is filtered to contain only the first investigation per bout before 
        computing DA metrics.

        Metrics computed:
        - AUC: Area under the z-score curve (using trapezoidal integration)
        - Max Peak: Maximum z-score in the window
        - Time of Max Peak: Timestamp corresponding to the maximum z-score
        - Mean Z-score: Mean z-score over the window

        Windowing logic:
        1. Fractional Bout Window (if use_fractional=True):  
            If the bout duration exceeds max_bout_duration seconds, only the first max_bout_duration 
            seconds are used.
        2. Adaptive Peak-Following (if use_adaptive=True):  
            If the peak is positive, the window is adjusted so that, starting from the peak,
            it continues until the z-score falls below (peak * peak_fall_fraction).  
            If no fall is found within the current window and allow_bout_extension is True, the search 
            extends to the end of the recording.
        3. If the peak is negative, adaptive processing is skipped and metrics are computed over the 
            current window.

        In addition, this function stores:
        - 'Original End': the original bout end time from the CSV.
        - 'Adjusted End': the end time after applying any window adjustments.

        Parameters:
        - use_fractional (bool): Whether to limit the window to max_bout_duration seconds.
        - max_bout_duration (float): Maximum duration (in seconds) for the fractional window.
        - use_adaptive (bool): Whether to apply the adaptive peak-following window.
        - peak_fall_fraction (float): Fraction of the peak to define the fall threshold.
        - allow_bout_extension (bool): Whether to extend the window past the bout’s official end if needed.
        - first (bool): If True, only the first investigation event per bout is considered.
        """
        if self.behaviors.empty:
            return

        # If first=True, filter the DataFrame to keep only the first investigation per bout
        if first:
            self.behaviors = (
                self.behaviors[self.behaviors["Behavior"] == "Investigation"]
                .sort_values("Event_Start")
                .groupby("Bout", as_index=False)
                .first()
            )

        # Ensure the metric columns exist
        for col in ['AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score', 'Original End', 'Adjusted End']:
            if col not in self.behaviors.columns:
                self.behaviors[col] = np.nan

        # Global end time (last timestamp) for potential extension
        global_end_time = self.timestamps[-1]

        # Process each behavior event (each row in self.behaviors)
        for i, row in self.behaviors.iterrows():
            start_time = row['Event_Start']
            orig_end_time = row['Event_End']
            end_time = orig_end_time  # default window end

            # Store the original bout end time
            self.behaviors.loc[i, 'Original End'] = orig_end_time

            # 1) Fractional Bout Window: truncate if duration exceeds max_bout_duration
            if use_fractional:
                bout_duration = orig_end_time - start_time
                if bout_duration > max_bout_duration:
                    end_time = start_time + max_bout_duration
            # (If not using fractional, end_time remains orig_end_time)

            # 2) Extract the current window from self.timestamps and self.zscore
            mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
            window_ts = self.timestamps[mask]
            window_z = self.zscore[mask]
            if len(window_ts) < 2:
                continue

            # 3) Adaptive Peak-Following Window: if enabled and peak is nonnegative
            if use_adaptive:
                max_idx = np.argmax(window_z)
                max_val = window_z[max_idx]
                peak_time = window_ts[max_idx]
                if max_val >= 0:
                    threshold = max_val * peak_fall_fraction
                    fall_idx = max_idx
                    while fall_idx < len(window_z) and window_z[fall_idx] > threshold:
                        fall_idx += 1

                    if fall_idx < len(window_z):
                        end_time = window_ts[fall_idx]
                    elif allow_bout_extension:
                        # Extend window to global end and search again
                        extended_mask = (self.timestamps >= start_time) & (self.timestamps <= global_end_time)
                        extended_ts = self.timestamps[extended_mask]
                        extended_z = self.zscore[extended_mask]
                        # Find the index closest to peak_time in the extended window
                        peak_idx_ext = np.argmin(np.abs(extended_ts - peak_time))
                        fall_idx_ext = peak_idx_ext
                        while fall_idx_ext < len(extended_z) and extended_z[fall_idx_ext] > threshold:
                            fall_idx_ext += 1
                        if fall_idx_ext < len(extended_ts):
                            end_time = extended_ts[fall_idx_ext]
                        else:
                            end_time = extended_ts[-1]
                    # Else: if adaptive is enabled but no fall is found and extension is not allowed,
                    # keep the current end_time.
                # If max_val is negative, adaptive processing is skipped.

            # 4) Re-extract the final window using the (possibly) updated end_time
            final_mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
            final_ts = self.timestamps[final_mask]
            final_z = self.zscore[final_mask]
            if len(final_ts) < 2:
                continue

            # 5) Compute final metrics
            auc = np.trapz(final_z, final_ts)
            mean_z = np.mean(final_z)
            final_max_idx = np.argmax(final_z)
            final_max_val = final_z[final_max_idx]
            final_peak_time = final_ts[final_max_idx]

            # 6) Save metrics and adjusted end time into the DataFrame
            self.behaviors.loc[i, 'AUC'] = auc
            self.behaviors.loc[i, 'Max Peak'] = final_max_val
            self.behaviors.loc[i, 'Time of Max Peak'] = final_peak_time
            self.behaviors.loc[i, 'Mean Z-score'] = mean_z
            self.behaviors.loc[i, 'Adjusted End'] = end_time
