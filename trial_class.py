import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from scipy.signal import butter, filtfilt

from sklearn.linear_model import LinearRegression
from behavior import Behavior

class Trial:
    def __init__(self, trial_path):
        tdtdata = tdt.read_block(trial_path)
        
        # Extract the subject name from the folder or file name
        self.subject_name = os.path.basename(trial_path).split('-')[0]
        
        # Assume all streams have the same sampling frequency and length
        self.fs = tdtdata.streams["_465A"].fs
        self.timestamps = np.arange(len(tdtdata.streams['_465A'].data)) / self.fs

        self.streams = {}
        self.streams['DA'] = tdtdata.streams['_465A'].data
        self.streams['ISOS'] = tdtdata.streams['_405A'].data

        self.behaviors = {key: value for key, value in tdtdata.epocs.items() if key not in ['Cam1', 'Tick']}
        
        # To ensure that the most updated DA does not require specific preprocessing steps, each of the preprocessing functions will update these. Make sure your ordering is correct
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
    def extract_single_behavior(self, behavior_name, bout_aggregated_df):
        """
        Extract a single behavior and add it to the behaviors dictionary.
        """
        behavior_df = bout_aggregated_df[bout_aggregated_df['Behavior'] == behavior_name]
        onset_times = behavior_df['Start (s)'].values.tolist()

        offset_times = behavior_df['Stop (s)'].values.tolist()

        self.behaviors[behavior_name] = Behavior(
            name=behavior_name,
            onset_times=onset_times,
            offset_times=offset_times,
        )


    def extract_manual_annotation_behaviors(self, bout_aggregated_csv_path):
        '''
        This function processes all behaviors of type 'STATE' in the CSV file and extracts them
        into the TDT recording.

        Parameters:
        bout_aggregated_csv_path: The file path to the CSV containing the behavior data.
        '''
        bout_aggregated_df = pd.read_csv(bout_aggregated_csv_path)

        # Filter the DataFrame to include only rows where the behavior type is 'STATE'
        state_behaviors_df = bout_aggregated_df[bout_aggregated_df['Behavior type'] == 'STATE']
        
        unique_behaviors = state_behaviors_df['Behavior'].unique()

        for behavior in unique_behaviors:
            # Filter the DataFrame for the current behavior
            behavior_df = state_behaviors_df[state_behaviors_df['Behavior'] == behavior]
            
            # Call the helper function to extract and add the behavior events
            self.extract_single_behavior(behavior, behavior_df)


    def combine_consecutive_behaviors(self, behavior_name='all', bout_time_threshold=1):
        """
        Combines consecutive behavior events if they occur within a specified time threshold
        and updates the behavior's data.
        """
        if behavior_name == 'all':
            behaviors_to_process = list(self.behaviors.keys())  
        else:
            behaviors_to_process = [behavior_name]  

        for behavior_event in behaviors_to_process:
            behavior = self.behaviors[behavior_event]
            behavior_onsets = np.array(behavior.onset_times)
            behavior_offsets = np.array(behavior.offset_times)

            combined_onsets = []
            combined_offsets = []

            if len(behavior_onsets) == 0:
                continue  

            start_idx = 0

            while start_idx < len(behavior_onsets):
                # Initialize the combination window with the first behavior onset and offset
                current_onset = behavior_onsets[start_idx]
                current_offset = behavior_offsets[start_idx]

                next_idx = start_idx + 1

                # Merge behaviors that start within `bout_time_threshold` of the previous offset
                while next_idx < len(behavior_onsets) and (behavior_onsets[next_idx] - current_offset) <= bout_time_threshold:
                    current_offset = behavior_offsets[next_idx]
                    next_idx += 1

                combined_onsets.append(current_onset)
                combined_offsets.append(current_offset)

                start_idx = next_idx

            # Update the behavior's onset and offset times
            behavior.onset_times = combined_onsets
            behavior.offset_times = combined_offsets



    def remove_short_behaviors(self, behavior_name='all', min_duration=0):
        """
        Removes behaviors with a duration less than the specified minimum duration.
        """
        if behavior_name == 'all':
            behaviors_to_process = list(self.behaviors.keys())
        else:
            behaviors_to_process = [behavior_name]

        for behavior_event in behaviors_to_process:
            behavior = self.behaviors[behavior_event]
            behavior_onsets = np.array(behavior.onset_times)
            behavior_offsets = np.array(behavior.offset_times)

            if len(behavior_onsets) == 0:
                continue

            behavior_durations = behavior_offsets - behavior_onsets
            valid_indices = np.where(behavior_durations >= min_duration)[0]

            behavior.onset_times = behavior_onsets[valid_indices].tolist()
            behavior.offset_times = behavior_offsets[valid_indices].tolist()
            behavior.data = [1] * len(valid_indices)



    '''********************************** PLOTTING **********************************'''
    def plot_behavior_event(self, behavior_name, ax=None):
        """
        Plot Delta F/F (dFF) or z-score with behavior events. Can be used to plot in a given Axes object or individually.
        """
        y_data = self.zscore
        y_label = 'z-score'
        y_title = 'z-score Signal'

        # Create plot if no axis provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='black')

        behavior_colors = {'Investigation': 'dodgerblue', 'Approach': 'green', 'Defeat': 'red', 'Pinch':'dodgerblue'}
        behavior_labels_plotted = set()

        if behavior_name == 'all':
            for behavior_event, behavior in self.behaviors.items():
                if behavior_event in behavior_colors:
                    color = behavior_colors[behavior_event]
                    for on, off in zip(behavior.onset_times, behavior.offset_times):
                        if behavior_event not in behavior_labels_plotted:
                            ax.axvspan(on, off, alpha=0.25, label=behavior_event, color=color)
                            behavior_labels_plotted.add(behavior_event)
                        else:
                            ax.axvspan(on, off, alpha=0.25, color=color)
        else:
            if behavior_name not in self.behaviors:
                raise ValueError(f"Behavior event '{behavior_name}' not found in behaviors.")
            behavior = self.behaviors[behavior_name]
            color = behavior_colors.get(behavior_name, 'dodgerblue')
            for on, off in zip(behavior.onset_times, behavior.offset_times):
                ax.axvspan(on, off, alpha=0.25, color=color)

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