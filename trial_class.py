import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, find_peaks


from sklearn.linear_model import LinearRegression

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

        self.behaviors = None
        
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
    def extract_bouts_and_behaviors(self, csv_path, bout_definitions, first_only=False):
        """
        Reads an aggregated behavior CSV from csv_path, extracts behavior events occurring within bouts,
        and stores the result as a DataFrame in self.behaviors.

        Each bout is defined by an "introduced" event and a "removed" event. Multiple bout definitions
        can be provided to handle different naming conventions.

        Parameters:
            csv_path (str): File path to the CSV containing the behavior data.
            bout_definitions (list of dict): A list where each dict defines a bout type with keys:
                - 'prefix': A string used to label bouts (e.g., "s1", "s2", "x").
                - 'introduced': The name of the introduced event (e.g., "s1_Introduced", "X_Introduced", etc.).
                - 'removed': The name of the removed event (e.g., "s1_Removed", "X_Removed", etc.).
            first_only (bool): If True, only the first event (by event start) in each bout is kept.
                            If False (default), all events within each bout are retained.

        The resulting DataFrame will have one row per behavior event (that is not a boundary event)
        unless first_only is True, in which case there will be one row per bout.
        """
        # 1. Read CSV and ensure numeric columns
        data = pd.read_csv(csv_path)
        data['Start (s)'] = pd.to_numeric(data['Start (s)'], errors='coerce')
        data['Stop (s)'] = pd.to_numeric(data['Stop (s)'], errors='coerce')

        # 2. Filter for the subject's data only
        data = data[data['Subject'] == 'Subject']

        # 3. Build a unique set of boundary behaviors from bout definitions
        boundary_behaviors = {bout_def['introduced'] for bout_def in bout_definitions} | {
            bout_def['removed'] for bout_def in bout_definitions
        }

        # 4. Helper function to extract events within each bout
        def extract_bout_events(df, introduced_behavior, removed_behavior, bout_prefix, first_only):
            introduced_df = df[df['Behavior'] == introduced_behavior].sort_values('Start (s)').reset_index(drop=True)
            removed_df = df[df['Behavior'] == removed_behavior].sort_values('Start (s)').reset_index(drop=True)
            num_bouts = min(len(introduced_df), len(removed_df))
            rows = []

            for i in range(num_bouts):
                bout_label = f"{bout_prefix}-{i+1}"
                bout_start = introduced_df.loc[i, 'Start (s)']
                bout_end = removed_df.loc[i, 'Start (s)']

                # Select only behaviors within this bout, excluding boundary behaviors.
                subset = df[
                    (~df['Behavior'].isin(boundary_behaviors)) &
                    (df['Start (s)'] >= bout_start) & 
                    (df['Stop (s)'] <= bout_end)
                ].sort_values('Start (s)')

                if first_only:
                    if not subset.empty:
                        first_row = subset.iloc[0]
                        rows.append({
                            'Bout': bout_label,
                            'Behavior': first_row['Behavior'],
                            'Event_Start': first_row['Start (s)'],
                            'Event_End': first_row['Stop (s)'],
                            'Duration (s)': first_row['Stop (s)'] - first_row['Start (s)']
                        })
                else:
                    for _, row in subset.iterrows():
                        rows.append({
                            'Bout': bout_label,
                            'Behavior': row['Behavior'],
                            'Event_Start': row['Start (s)'],
                            'Event_End': row['Stop (s)'],
                            'Duration (s)': row['Stop (s)'] - row['Start (s)']
                        })
            return rows

        # 5. Extract behavior events for all bout definitions.
        bout_rows = []
        for bout_def in bout_definitions:
            prefix = bout_def['prefix']
            introduced_behavior = bout_def['introduced']
            removed_behavior = bout_def['removed']
            bout_rows.extend(extract_bout_events(data, introduced_behavior, removed_behavior, prefix, first_only))

        # 6. Store the resulting DataFrame in the instance.
        self.behaviors = pd.DataFrame(bout_rows)

    def extract_bouts_and_behaviors_updated(self, csv_path, bout_definitions, first_only=True):
        """
            Extracts behavior bouts for a trial from Ethovision CSV, supporting flexible labels.
            
            Parameters:
            - csv_path (str): Path to the behavior CSV file.
            - bout_definitions (list of dict): Definitions of bout boundaries with flexible label support.
            - first_only (bool): If True, only extract the first bout per trial.
            """
        import pandas as pd

        data = pd.read_csv(csv_path)

        if data.empty or 'Behavior' not in data.columns or 'Subject' not in data.columns:
            print(f"⚠️ CSV at {csv_path} missing 'Behavior' or 'Subject' columns, or is empty.")
            self.behaviors = pd.DataFrame()
            return

        # Keep rows where Subject is 'Subject' or 'No focal subject' (case-insensitive)
        valid_subjects = ['subject', 'no focal subject']
        data = data[data['Subject'].str.lower().isin(valid_subjects)].copy()

        if data.empty:
            print(f"⚠️ No valid 'Subject' or 'No focal subject' rows found in {csv_path}.")
            self.behaviors = pd.DataFrame()
            return

        # Normalize 'Behavior' for matching
        data['Behavior_LC'] = data['Behavior'].str.lower()

        # Collect all introduced and removed labels from all bout_definitions
        all_intros = set()
        all_removes = set()
        for bout_def in bout_definitions:
            all_intros.update([label.lower() for label in bout_def['introduced']])
            all_removes.update([label.lower() for label in bout_def['removed']])

        events = []

        for bout_def in bout_definitions:
            prefix = bout_def['prefix']
            introduced_labels = [label.lower() for label in bout_def['introduced']]
            removed_labels = [label.lower() for label in bout_def['removed']]

            intro_events = data[data['Behavior_LC'].isin(introduced_labels)]
            remove_events = data[data['Behavior_LC'].isin(removed_labels)]

            for idx, intro_row in intro_events.iterrows():
                intro_time = intro_row['Start (s)']
                valid_removes = remove_events[remove_events['Start (s)'] > intro_time]

                if valid_removes.empty:
                    continue

                first_remove = valid_removes.iloc[0]
                remove_time = first_remove['Start (s)']

                bout_data = {
                    'Behavior': f"{prefix} Investigation",
                    'Start (s)': intro_time,
                    'Stop (s)': remove_time,
                    'Duration (s)': remove_time - intro_time
                }
                events.append(bout_data)

                if first_only:
                    break  # Stop after first bout for this trial

        self.behaviors = pd.DataFrame(events)

        if not self.behaviors.empty:
            print(f"✅ Extracted {len(self.behaviors)} bouts for trial from {csv_path}.")
        else:
            print(f"⚠️ No bouts extracted for trial from {csv_path}.")


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


    def plot_first_behavior_PETHs(self, behavior = "Investigation"):
            """
            Plot the first investigation of each bout as side-by-side subplots.
            Each subplot shows:
            - The investigation trace using the relative time axis.
            - A dashed black line at x=0 (start).
            - A dashed blue line at x=Duration (s) for the investigation end.
            - A dashed red line at x=Time of Max Peak.
            
            All plots share the same y-axis limits.
            RELATIVE DA STUFF MUST BE CALLED BEFORE YOU CAN PLOT
            """ 
            # 1. Filter to only 'Investigation' rows
            df_invest = self.behaviors[self.behaviors["Behavior"] == behavior].copy()
            
            # 2. Identify the first investigation of each bout.
            df_first_invest = df_invest.groupby("Bout", as_index=False).first()
            
            # Number of plots equals the number of first investigations
            n_plots = len(df_first_invest)
            
            # Create side-by-side subplots
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
            
            # If only one bout, ensure axes is iterable
            if n_plots == 1:
                axes = [axes]
            
            # Loop through each first investigation row and plot
            for i, (_, row) in enumerate(df_first_invest.iterrows()):
                ax = axes[i]
                
                # Extract arrays for plotting
                x = row["Relative_Time_Axis"]  
                y = row["Relative_Zscore"]  
                
                # Plot main investigation trace
                ax.plot(x, y, label=f"Bout: {row['Bout']}")
                
                # Dashed black line at investigation start (x=0)
                ax.axvline(x=0, color='black', linestyle='--', label="Investigation Start")
                
                # Dashed blue line at investigation end (x = duration)
                end_time = row["Duration (s)"]
                ax.axvline(x=end_time, color='blue', linestyle='--', label="Investigation End")
                
                # Dashed red line at time of max peak
                max_peak_time = row["Time of Max Peak"]
                ax.axvline(x=max_peak_time, color='red', linestyle='--', label="Time of Max Peak")
                
                # Set y-axis limits
                ax.set_ylim([-2.5, 14])
                
                # Titles and labels for each subplot
                ax.set_title(f"Bout {row['Bout']}")
                ax.set_xlabel("Relative Time (s)")
            
            # Set a common y-axis label on the leftmost subplot
            axes[0].set_ylabel("Z-score")
            
            # Display the legend on the first subplot (optional)
            axes[0].legend()
            
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
        


    def compute_behavior_relative_DA(self, pre_time=4, post_time=15, mode='EI'):
        """
        Computes the peri-event DA signal for each event in self.behaviors.
        
        Depending on the mode, the signal is processed differently:
        - mode='EI': The DA signal is baseline-corrected by subtracting the mean 
            z-score from the pre-event period (i.e. event-induced).
        - mode='standard': The raw DA signal is used without baseline correction.
        
        In both cases, the signal is interpolated onto a common time axis relative 
        to event onset.
        
        Two new columns are added to self.behaviors:
        - 'Relative_Time_Axis': A common time axis (relative to event onset).
        - 'Relative_Zscore': The processed z-score signal, interpolated onto that axis.
        
        Parameters:
        - pre_time (float): Seconds to include before event onset.
        - post_time (float): Seconds to include after event onset.
        - mode (str): Either 'EI' (baseline-corrected) or 'standard' (raw).
        """
        if self.behaviors is None or self.behaviors.empty:
            print(f"Trial {self.subject_name}: No behavior events available to compute behavior-relative DA.")
            return

        # Calculate a common time axis based on the average sampling interval.
        dt = np.mean(np.diff(self.timestamps))
        common_time_axis = np.arange(-pre_time, post_time, dt)

        # Lists to hold the computed axes and processed signals for each event.
        relative_time_list = []
        relative_zscore_list = []

        # Process each event.
        for idx, row in self.behaviors.iterrows():
            event_start = row['Event_Start']
            window_start = event_start - pre_time
            window_end = event_start + post_time

            # Select data within the peri-event window.
            mask = (self.timestamps >= window_start) & (self.timestamps <= window_end)
            if not np.any(mask):
                relative_time_list.append(np.full(common_time_axis.shape, np.nan))
                relative_zscore_list.append(np.full(common_time_axis.shape, np.nan))
                continue

            # Create a time axis relative to the event onset.
            rel_time = self.timestamps[mask] - event_start
            signal = self.zscore[mask]

            if mode == 'EI':
                # For event-induced mode, subtract the pre-event baseline.
                pre_mask = rel_time < 0
                baseline = np.nanmean(signal[pre_mask]) if np.any(pre_mask) else 0
                processed_signal = signal - baseline
            elif mode == 'standard':
                # For standard mode, leave the raw signal intact.
                processed_signal = signal
            else:
                raise ValueError("Mode must be either 'EI' or 'standard'")

            # Interpolate the processed signal onto the common time axis.
            interp_signal = np.interp(common_time_axis, rel_time, processed_signal)

            relative_time_list.append(common_time_axis)
            relative_zscore_list.append(interp_signal)

        # Save the computed arrays as new columns.
        self.behaviors['Relative_Time_Axis'] = relative_time_list
        self.behaviors['Relative_Zscore'] = relative_zscore_list


    def compute_da_metrics(self, 
                        use_max_length=False, 
                        max_bout_duration=30, 
                        use_adaptive=False, 
                        allow_bout_extension=False,
                        mode='standard', 
                        pre_time=4, 
                        post_time=15):
        """
        Computes DA metrics for each behavior event (row) in self.behaviors.
        
        Two modes are available:
        
        - mode='standard': Metrics are computed from self.timestamps and self.zscore,
          using the event’s absolute timing (window from Event_Start to Event_End, possibly truncated).
          In this branch, the "Time of Max Peak" is computed relative to the event onset.
        
        - mode='EI': Metrics are computed from behavior-relative data stored in 
          'Relative_Time_Axis' and 'Relative_Zscore'. These are computed using 
          compute_behavior_relative_DA (if not already computed) with the specified pre/post times.
          In this branch the window is defined from 0 (event onset) to the event duration (from the 
          'Duration (s)' column), with optional max length or adaptive adjustments.
        
        Metrics computed:
        - AUC: Area under the z-score curve.
        - Max Peak: Maximum z-score in the window.
        - Time of Max Peak: For standard mode, the time (relative to event onset) at which the maximum occurs;
                           for EI mode, this is directly obtained from the relative time axis.
        - Mean Z-score: Mean z-score over the window.
        - Adjusted End: The final effective window end after any adjustments.
        
        Parameters:
        - use_max_length (bool): If True, limit the window to max_bout_duration seconds.
        - max_bout_duration (float): Maximum allowed window duration (in seconds).
        - use_adaptive (bool): If True, adjust the window based on the first local minimum after the peak.
        - allow_bout_extension (bool): If True, extend the window if no local minimum is found.
        - mode (str): Either 'standard' or 'EI'.
        - pre_time (float): (For EI mode) Seconds before event onset used in computing the relative DA.
        - post_time (float): (For EI mode) Seconds after event onset used in computing the relative DA.
        """
        if self.behaviors.empty:
            return

        # Ensure metric columns exist.
        for col in ['AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score', 'Adjusted End']:
            if col not in self.behaviors.columns:
                self.behaviors[col] = np.nan

        standard_end_time = self.timestamps[-1]

        if mode == 'standard':
            if 'Relative_Time_Axis' not in self.behaviors.columns or 'Relative_Zscore' not in self.behaviors.columns:
                self.compute_behavior_relative_DA(pre_time=pre_time, post_time=post_time, mode='standard')
            # In standard mode, work with absolute timestamps and raw zscore.
            for i, row in self.behaviors.iterrows():
                start_time = row['Event_Start']
                orig_end_time = row['Event_End']
                end_time = orig_end_time  # default window end
                
                # Limit window if required.
                if use_max_length:
                    bout_duration = orig_end_time - start_time
                    if bout_duration > max_bout_duration:
                        end_time = start_time + max_bout_duration

                # Extract window data.
                mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
                window_ts = self.timestamps[mask]
                window_z = self.zscore[mask]
                if len(window_ts) < 2:
                    continue

                # Optionally apply adaptive adjustment.
                if use_adaptive:
                    peak_idx = np.argmax(window_z)
                    local_mins, _ = find_peaks(-window_z[peak_idx:])
                    if local_mins.size > 0:
                        end_time = window_ts[peak_idx + local_mins[0]]
                    elif allow_bout_extension:
                        extended_mask = (self.timestamps >= start_time) & (self.timestamps <= standard_end_time)
                        extended_ts = self.timestamps[extended_mask]
                        if len(extended_ts) > 0:
                            end_time = extended_ts[-1]

                # Re-extract final window.
                final_mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
                final_ts = self.timestamps[final_mask]
                final_z = self.zscore[final_mask]
                if len(final_ts) < 2:
                    continue

                auc = np.trapz(final_z, final_ts)
                mean_z = np.mean(final_z)
                final_max_idx = np.argmax(final_z)
                final_max_val = final_z[final_max_idx]
                final_peak_time = final_ts[final_max_idx]
                # Compute relative time of max peak (relative to event onset)
                relative_peak_time = final_peak_time - start_time

                self.behaviors.loc[i, 'AUC'] = auc
                self.behaviors.loc[i, 'Max Peak'] = final_max_val
                self.behaviors.loc[i, 'Time of Max Peak'] = relative_peak_time
                self.behaviors.loc[i, 'Mean Z-score'] = mean_z
                self.behaviors.loc[i, 'Adjusted End'] = end_time

        elif mode == 'EI':
            # For EI mode, ensure that the behavior-relative DA data exist.
            if 'Relative_Time_Axis' not in self.behaviors.columns or 'Relative_Zscore' not in self.behaviors.columns:
                self.compute_behavior_relative_DA(pre_time=pre_time, post_time=post_time, mode='EI')
            
            # Compute metrics using the relative (event-aligned) data.
            for i, row in self.behaviors.iterrows():
                time_axis = row['Relative_Time_Axis']  # relative to event onset
                event_zscore = row['Relative_Zscore']    # already processed (baseline-corrected if mode=='EI')
                # The initial effective end is taken from the event duration.
                effective_end = row['Duration (s)']
                if use_max_length and effective_end > max_bout_duration:
                    effective_end = max_bout_duration

                # Create mask for time_axis within [0, effective_end].
                time_axis_arr = np.array(time_axis)
                event_zscore_arr = np.array(event_zscore)
                mask = (time_axis_arr >= 0) & (time_axis_arr <= effective_end)
                if not np.any(mask):
                    continue
                final_time = time_axis_arr[mask]
                final_z = event_zscore_arr[mask]

                # Adaptive adjustment: adjust effective_end based on first local minimum after peak.
                if use_adaptive:
                    peak_idx = np.argmax(final_z)
                    local_mins, _ = find_peaks(-final_z[peak_idx:])
                    if local_mins.size > 0:
                        effective_end = final_time[peak_idx + local_mins[0]]
                    elif allow_bout_extension:
                        effective_end = np.max(time_axis_arr[time_axis_arr >= 0])
                    # Update final window.
                    mask = (time_axis_arr >= 0) & (time_axis_arr <= effective_end)
                    final_time = time_axis_arr[mask]
                    final_z = event_zscore_arr[mask]
                    if len(final_time) < 2:
                        continue

                auc = np.trapz(final_z, final_time)
                mean_z = np.mean(final_z)
                final_max_idx = np.argmax(final_z)
                final_max_val = final_z[final_max_idx]
                final_peak_time = final_time[final_max_idx]

                self.behaviors.loc[i, 'AUC'] = auc
                self.behaviors.loc[i, 'Max Peak'] = final_max_val
                self.behaviors.loc[i, 'Time of Max Peak'] = final_peak_time
                self.behaviors.loc[i, 'Mean Z-score'] = mean_z
                self.behaviors.loc[i, 'Adjusted End'] = effective_end

        else:
            raise ValueError("Mode must be either 'standard' or 'EI'")



