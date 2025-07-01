import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight
from scipy.optimize import curve_fit
from scipy.signal import welch


from sklearn.linear_model import LinearRegression

class Trial:
    def __init__(self, trial_path, stream_DA, stream_ISOS):
        tdtdata = tdt.read_block(trial_path)

        self.subject_name = os.path.basename(trial_path).split('-')[0]
        self.fs = tdtdata.streams[stream_DA].fs
        self.timestamps = np.arange(len(tdtdata.streams[stream_DA].data)) / self.fs

        self.streams = {}
        self.streams['DA'] = tdtdata.streams[stream_DA].data
        self.streams['ISOS'] = tdtdata.streams[stream_ISOS].data

        self.behaviors = None
        # For RTC
        self.rtc_events = {key: value for key, value in tdtdata.epocs.items() if key not in ['Cam1', 'Cam2', 'Tick']}

        self.updated_DA = self.streams['DA']
        self.updated_ISOS = self.streams['ISOS']

        self.isosbestic_fitted = np.empty(1)
        self.dFF = np.empty(1)
        self.zscore = np.empty(1)


    '''********************************** TIME REMOVAL **********************************'''
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


    '''********************************** SMOOTHING AND FITTING **********************************'''

    def smooth_and_apply_DA_ISOS(self, window_len_seconds: float = 1.0):
        """
        Smooth both DA and ISOS signals using a moving‐average window of the given length in seconds,
        and store the results in self.updated_DA / self.updated_ISOS.
        """
        def smooth_signal(source: np.ndarray, n_samples: int) -> np.ndarray:
            """Smooth a 1D array with a flat (uniform) window of n_samples size."""
            if source.ndim != 1:
                raise ValueError("smooth only accepts 1-dimensional arrays")
            if n_samples < 3 or source.size < n_samples:
                # too small to smooth meaningfully
                return source

            # Reflect padding to minimize edge artifacts
            pad = n_samples - 1
            s = np.r_[source[pad:0:-1], source, source[-2:-pad-1:-1]]
            w = np.ones(n_samples, dtype=float) / n_samples
            smoothed = np.convolve(s, w, mode='valid')
            # Trim back to original length
            start = pad // 2
            return smoothed[start:start + source.size]

        # Convert window length from seconds to samples
        n_samp = int(round(window_len_seconds * self.fs))
        if n_samp < 1:
            raise ValueError("window_len_seconds too small for your sampling rate")

        # Apply smoothing
        if 'DA' in self.streams:
            self.updated_DA = smooth_signal(self.streams['DA'], n_samp)
        if 'ISOS' in self.streams:
            self.updated_ISOS = smooth_signal(self.streams['ISOS'], n_samp)

    def smooth_and_apply(
        self,
        window_len_seconds: float = 1.0,
        smooth_zscore:      bool  = True
    ):
        """
        Smooth the self.dFF trace (and, if requested, self.zscore) using
        a moving‐average window of the given length in seconds.
        """
        def smooth_signal(source: np.ndarray, n_samples: int) -> np.ndarray:
            if source is None or source.ndim != 1 or source.size < n_samples or n_samples < 3:
                return source
            pad = n_samples - 1
            s = np.r_[source[pad:0:-1], source, source[-2:-pad-1:-1]]
            w = np.ones(n_samples, dtype=float) / n_samples
            v = np.convolve(s, w, mode='valid')
            start = pad // 2
            return v[start:start + source.size]

        # compute n_samples from seconds
        n_samp = int(round(window_len_seconds * self.fs))
        if n_samp < 1:
            raise ValueError("window_len_seconds too small for your sampling rate")

        # 1) smooth dFF
        if hasattr(self, 'dFF') and self.dFF is not None:
            self.dFF = smooth_signal(np.asarray(self.dFF, dtype=float), n_samp)
        else:
            raise RuntimeError("Cannot smooth: self.dFF is missing")

        # 2) optionally smooth zscore
        if smooth_zscore:
            if hasattr(self, 'zscore') and self.zscore is not None:
                self.zscore = smooth_signal(np.asarray(self.zscore, dtype=float), n_samp)
            else:
                # if no zscore yet, just skip or warn
                print("Warning: no self.zscore to smooth.")

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

    def lowpass_filter(self, cutoff_hz: float = 3.0):
            """
            2nd-order Butterworth low-pass @ cutoff_hz (Hz), applied to the already-updated DA & ISOS traces.
            """
            # design a 2nd-order low-pass (digital) at cutoff_hz
            b, a = butter(2, cutoff_hz, btype='low', fs=self.fs)
            # zero-phase filter
            self.updated_DA   = filtfilt(b, a, self.updated_DA,   padtype='even')
            self.updated_ISOS = filtfilt(b, a, self.updated_ISOS, padtype='even')
            print(f"Low-pass filtered @ {cutoff_hz} Hz")



    def highpass_baseline_drift_dFF(self, cutoff=0.001):
        """
        Applies a high-pass Butterworth filter to remove slow drift from the DA and ISOS signals.
        https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
        """
        # Design a second-order Butterworth high-pass filter
        b, a = butter(N=2, Wn=cutoff, btype='high', fs=self.fs)

        # Apply zero-phase filtering to avoid phase distortion
        self.dFF = filtfilt(b, a, self.dFF, padtype='even')


    def highpass_baseline_drift_Recentered(self, cutoff=0.001):
        """
        High-pass filter out the slow drift *but* keep the original
        DC-level by adding back the mean afterwards.
        Call this after any smoothing, on updated_DA/updated_ISOS.
        """
        # 1) remember pre-filter mean
        mu_DA   = np.mean(self.updated_DA)
        mu_ISOS = np.mean(self.updated_ISOS)

        # 2) design filter
        b, a = butter(N=2, Wn=cutoff, btype='high', fs=self.fs)

        # 3) filter
        hp_DA   = filtfilt(b, a, self.updated_DA,   padtype='even')
        hp_ISOS = filtfilt(b, a, self.updated_ISOS, padtype='even')

        # 4) add back original mean
        self.updated_DA   = hp_DA   + mu_DA
        self.updated_ISOS = hp_ISOS + mu_ISOS

    # def align_channels(self):
    #     """
    #     Function that performs linear regression between isosbestic_corrected and DA_corrected signals, and aligns
    #     the fitted isosbestic with the DA signal. 

    #     Baseline correction must have occurred
    #     """
    #     reg = LinearRegression()
        
    #     n = len(self.updated_DA)
    #     reg.fit(self.updated_ISOS.reshape(n, 1), self.updated_DA.reshape(n, 1))
    #     self.isosbestic_fitted = reg.predict(self.updated_ISOS.reshape(n, 1)).reshape(n,)

    def _double_exponential(t, c, A_fast, A_slow, tau_slow, tau_mul):
            tau_fast = tau_slow * tau_mul
            return c + A_slow * np.exp(-t / tau_slow) + A_fast * np.exp(-t / tau_fast)
    
    def correct_photobleach(self,
                            p0: dict | None = None,
                            bounds: dict | None = None,
                            maxfev: int = 2000):
        """
        Fit & subtract a double‐exp bleaching baseline from both updated_DA and updated_ISOS,
        then add back each signal’s original mean so the DC level is preserved.

        After this:
          self.bleach_baseline_DA/ISOS  = the fitted baselines
          self.updated_DA/ISOS          = (orig – baseline) + orig_mean

        Returns a dict { 'DA': popt_DA, 'ISOS': popt_ISOS } of fit params.
        """
        results = {}
        t = self.timestamps

        for ch in ('DA', 'ISOS'):
            key = 'updated_' + ch
            sig = getattr(self, key)
            mu  = np.nanmean(sig)  # remember original DC

            # pick or build initial guess & bounds
            m = np.nanmax(sig)
            p0_ch = p0.get(ch) if p0 and ch in p0 else [m/2, m/4, m/4, 3600, 0.1]
            if bounds and ch in bounds:
                bnds = bounds[ch]
            else:
                lo = [0,     0,     0,     600,    0]
                hi = [m,     m,     m,  36000,    1]
                bnds = (lo, hi)

            popt, _ = curve_fit(
                Trial._double_exponential,
                t, sig,
                p0=p0_ch,
                bounds=bnds,
                maxfev=maxfev
            )

            baseline = Trial._double_exponential(t, *popt)
            setattr(self, f'bleach_baseline_{ch}', baseline)

            # remove drift but preserve DC
            detrended = sig - baseline
            recentered = detrended + mu
            setattr(self, key, recentered)

            results[ch] = popt

        return results

    def align_channels_poly(self):
        """
        Fit a degree-1 polynomial (slope + intercept) to predict DA from the isosbestic channel.
        Stores the fitted control trace in self.isosbestic_fitted, which you can then use
        for dF/F calculation (e.g. (DA – fitted_control)/fitted_control).
        """
        # grab your two equal-length 1D arrays
        x = np.asarray(self.updated_ISOS, dtype=float)
        y = np.asarray(self.updated_DA,   dtype=float)

        # perform degree-1 polynomial fit: y ≃ m·x + b
        m, b = np.polyfit(x, y, deg=1)

        # build the fitted control trace
        self.isosbestic_fitted = m * x + b

        # optional: print fit parameters
        print(f"Align fit: DA ≃ {m:.4f}·ISOS + {b:.4f}")


    def align_channels_linReg(self):
        """
        Use ordinary least-squares LinearRegression to fit the isosbestic channel
        to the DA channel and store the fitted control trace in self.isosbestic_fitted.
        """
        # pull out your two equal-length 1D arrays
        reg = LinearRegression()
        
        n = len(self.updated_DA)
        reg.fit(self.updated_ISOS.reshape(n, 1), self.updated_DA.reshape(n, 1))
        self.isosbestic_fitted = reg.predict(self.updated_ISOS.reshape(n, 1)).reshape(n,)

    def align_channels_IRLS(self, IRLS_constant: float = 1.4):
        """
        Fit a robust (Tukey bisquare) regression of DA on ISOS,
        store the fitted control trace in self.isosbestic_fitted.
        """
        # grab your two equal‐length 1D arrays
        x = np.asarray(self.updated_ISOS, dtype=float)
        y = np.asarray(self.updated_DA,   dtype=float)

        # design matrix with intercept
        X = sm.add_constant(x)

        # RLM with Tukey’s bisquare M‐estimator
        rlm = sm.RLM(y, X, M=TukeyBiweight(IRLS_constant))
        res = rlm.fit()

        b0, b1 = res.params  # intercept, slope
        self.isosbestic_fitted = b0 + b1 * x

        # optional: log the fit
        print(f"IRLS fit: DA ≃ {b1:.4f}·ISOS + {b0:.4f}")



    def compute_dFF(self):
        """
        Function that computes the dF/F of the fitted isosbestic and DA signals and saves it in self.dFF.
        """
        isosbestic = self.isosbestic_fitted
        da = self.updated_DA
        
        # Compute dF/F by subtracting the fitted isosbestic from the DA signal
        df_f = (da - isosbestic) / isosbestic
        
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



    def compute_psd(self, channel: str = 'DA', nperseg: int = None):
        """
        Compute Welch PSD for the specified channel.
        channel: 'DA', 'ISOS', 'dFF', or 'zscore'
        nperseg: length of each segment for Welch; default ≃ 1/8 of the record
        Returns (f, Pxx).
        """
        if channel == 'DA':
            data = self.updated_DA
        elif channel == 'ISOS':
            data = self.updated_ISOS
        elif channel == 'dFF':
            data = self.dFF
        elif channel == 'zscore':
            data = self.zscore
        else:
            raise ValueError(f"Unknown channel '{channel}'")
        # pick a sensible default if none given
        if nperseg is None:
            nperseg = max(256, len(data)//8)
        f, Pxx = welch(data, fs=self.fs, nperseg=nperseg)
        return f, Pxx

    def plot_psd(self, channel: str = 'DA', nperseg: int = None, ax=None):
        """
        Plot PSD for this trial.
        Returns (f, Pxx).
        """
        f, Pxx = self.compute_psd(channel, nperseg)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,4))
        ax.semilogy(f, Pxx, lw=1.3)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"{self.subject_name} — PSD ({channel})")
        return f, Pxx

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
            first_only (bool): If True, only the first occurrence of **each distinct Behavior type**
                            within each bout is kept. If False (default), all events within each bout
                            are retained.

        The resulting DataFrame will have one row per behavior event (that is not a boundary event)
        unless first_only is True, in which case there will be one row per (bout × behavior‐type).
        """
        import pandas as pd

        # 1. Read CSV and ensure numeric columns
        data = pd.read_csv(csv_path)
        data['Start (s)'] = pd.to_numeric(data['Start (s)'], errors='coerce')
        data['Stop (s)']  = pd.to_numeric(data['Stop (s)'],  errors='coerce')

        # 2. Filter for the subject's data only
        data = data[data['Subject'] == 'Subject']

        # 3. Build a set of all boundary‐marker behaviors (introduced/removed) so we can exclude them later
        boundary_behaviors = {
            bout_def['introduced'] for bout_def in bout_definitions
        } | {
            bout_def['removed'] for bout_def in bout_definitions
        }

        # 4. Helper function: extract events inside a single bout (introduced→removed)
        def extract_bout_events(df, introduced_behavior, removed_behavior, bout_prefix, first_only_flag):
            """
            Looks at all rows where Behavior==introduced_behavior (sorted by Start),
            pairs them up with rows where Behavior==removed_behavior (also sorted),
            and for each matched pair [i], finds all non‐boundary events whose Start/Stop
            lie within that interval. Returns one row per event (or, if first_only_flag=True,
            one row per distinct Behavior type).
            """
            # Gather all introduction times (sorted) and removal times (sorted)
            introduced_df = (
                df[df['Behavior'] == introduced_behavior]
                .sort_values('Start (s)')
                .reset_index(drop=True)
            )
            removed_df = (
                df[df['Behavior'] == removed_behavior]
                .sort_values('Start (s)')
                .reset_index(drop=True)
            )

            # We only pair up as many bouts as we have both introduced & removed
            num_bouts = min(len(introduced_df), len(removed_df))
            rows = []

            for i in range(num_bouts):
                bout_label = f"{bout_prefix}-{i+1}"
                bout_start = introduced_df.loc[i, 'Start (s)']
                bout_end   = removed_df.loc[i, 'Start (s)']

                # Subset to all behaviors that:
                #   • are NOT one of the boundary names, and
                #   • start ≥ bout_start AND stop ≤ bout_end
                subset = df[
                    (~df['Behavior'].isin(boundary_behaviors)) &
                    (df['Start (s)'] >= bout_start) &
                    (df['Stop (s)']  <= bout_end)
                ].sort_values('Start (s)')

                if first_only_flag:
                    # Instead of grabbing only subset.iloc[0], we group by Behavior
                    # and take the first occurrence of each distinct Behavior type.
                    for behavior_type, group in subset.groupby('Behavior'):
                        first_row = group.iloc[0]
                        rows.append({
                            'Bout'        : bout_label,
                            'Behavior'    : behavior_type,
                            'Event_Start' : first_row['Start (s)'],
                            'Event_End'   : first_row['Stop (s)'],
                            'Duration (s)': first_row['Stop (s)'] - first_row['Start (s)']
                        })
                else:
                    # Keep all events (one row per event) inside this bout
                    for _, row in subset.iterrows():
                        rows.append({
                            'Bout'        : bout_label,
                            'Behavior'    : row['Behavior'],
                            'Event_Start' : row['Start (s)'],
                            'Event_End'   : row['Stop (s)'],
                            'Duration (s)': row['Stop (s)'] - row['Start (s)']
                        })

            return rows

        # 5. Loop over every bout_definition, extract events, and accumulate
        bout_rows = []
        for bout_def in bout_definitions:
            prefix             = bout_def['prefix']
            introduced_behavior = bout_def['introduced']
            removed_behavior    = bout_def['removed']

            bout_rows.extend(
                extract_bout_events(
                    data,
                    introduced_behavior,
                    removed_behavior,
                    prefix,
                    first_only_flag=first_only
                )
            )

        # 6. Convert to DataFrame and assign to self.behaviors
        self.behaviors = pd.DataFrame(bout_rows)


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
        Starts at 30 seconds and ensures every subplot shows x-axis ticks.
        """
        if self.behaviors is None or self.behaviors.empty:
            print(f"Warning: No behavior data for {self.subject_name}.")
            return

        y_data = self.zscore
        y_label = 'z-score'
        y_title = 'z-score Signal'

        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        ax.plot(self.timestamps, np.array(y_data), linewidth=2, color='black')

        # Behavior coloring
        behavior_colors = {
            'Investigation': 'dodgerblue', 
            'Approach': 'green', 
            'Defeat': 'red', 
            'Pinch': 'dodgerblue'
        }

        behavior_labels_plotted = set()

        # Fixed start at 30 seconds
        min_time = 30
        if behavior_name == 'all':
            max_time = self.behaviors['Event_End'].max() + 30
        else:
            behavior_df = self.behaviors[self.behaviors['Behavior'] == behavior_name]
            if behavior_df.empty:
                raise ValueError(f"Behavior event '{behavior_name}' not found in behaviors.")
            max_time = behavior_df['Event_End'].max() + 30

        for _, row in self.behaviors.iterrows():
            if behavior_name != 'all' and row['Behavior'] != behavior_name:
                continue
            event_name = row['Behavior']
            color = behavior_colors.get(event_name, 'dodgerblue')
            on, off = row['Event_Start'], row['Event_End']
            if event_name not in behavior_labels_plotted:
                ax.axvspan(on, off, alpha=0.25, label=event_name, color=color)
                behavior_labels_plotted.add(event_name)
            else:
                ax.axvspan(on, off, alpha=0.25, color=color)

        ax.set_xlim(min_time, max_time)
        ax.set_ylabel(y_label)
        ax.set_title(f'{self.subject_name}: {y_title} with {behavior_name.capitalize()} Bouts' if behavior_name != 'all' else f'{self.subject_name}: {y_title} with All Behavior Events')

        if behavior_labels_plotted:
            ax.legend()

        # Explicitly enable x-axis tick labels
        ax.tick_params(axis='x', labelbottom=True)



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

    def get_introductions_and_removals(self):
        """
        Returns two lists of times:
          - intros: all Event_Start times where Behavior endswith 'Introduced'
          - rems : all Event_End   times where Behavior endswith 'Removed'
        """
        if self.behaviors is None or self.behaviors.empty:
            return [], []

        # any Behavior name that ends with 'Introduced'
        intro_mask = self.behaviors["Behavior"].str.endswith("Introduced")
        # any Behavior name that ends with 'Removed'
        remove_mask = self.behaviors["Behavior"].str.endswith("Removed")

        intros = list(self.behaviors.loc[intro_mask, "Event_Start"])
        rems   = list(self.behaviors.loc[remove_mask, "Event_End"  ])
        return intros, rems

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
                        mode='standard', 
                        pre_time=4, 
                        post_time=15):
        """
        Computes DA metrics for each behavior event (row) in self.behaviors.
        
        Two modes are available:
        
        - mode='standard': Metrics are computed from self.timestamps and self.zscore,
        using the event’s absolute timing (window from Event_Start to Event_End, possibly truncated).
        The "Time of Max Peak" is computed relative to the event onset.
        
        - mode='EI': Metrics are computed from behavior-relative data stored in 
        'Relative_Time_Axis' and 'Relative_Zscore'. These are computed using 
        compute_behavior_relative_DA (if not already computed) with the specified pre/post times.
        In this branch the window is defined from 0 (event onset) to the event duration (from the 
        'Duration (s)' column), with optional max length adjustments.
        
        Additionally, if the behavior lasts less than 1 second, the window is allowed to extend
        past the bout end to search for the next peak.
        
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
            # Ensure behavior-relative DA is computed if needed.
            if 'Relative_Time_Axis' not in self.behaviors.columns or 'Relative_Zscore' not in self.behaviors.columns:
                self.compute_behavior_relative_DA(pre_time=pre_time, post_time=post_time, mode='standard')
            # In standard mode, work with absolute timestamps and raw zscore.
            for i, row in self.behaviors.iterrows():
                start_time = row['Event_Start']
                orig_end_time = row['Event_End']
                bout_duration = orig_end_time - start_time
                # Default window end is orig_end_time.
                end_time = orig_end_time

                # Limit window duration if required.
                if use_max_length and bout_duration > max_bout_duration:
                    end_time = start_time + max_bout_duration

                # If the behavior lasts less than 1 second, extend the window to look for the next peak.
                if bout_duration < 1:
                    extension_mask = (self.timestamps > orig_end_time)
                    if np.any(extension_mask):
                        extended_ts = self.timestamps[extension_mask]
                        extended_z = self.zscore[extension_mask]
                        # Find peaks in the extended portion.
                        peaks, _ = find_peaks(extended_z)
                        if peaks.size > 0:
                            end_time = extended_ts[peaks[0]]

                # Extract final window.
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
                event_zscore = row['Relative_Zscore']    # processed data
                # The initial effective end is taken from the event duration.
                effective_end = row['Duration (s)']
                if use_max_length and effective_end > max_bout_duration:
                    effective_end = max_bout_duration

                # If the behavior lasts less than 1 second, extend the window to search for the next peak.
                if effective_end < 1:
                    time_axis_arr = np.array(time_axis)
                    event_zscore_arr = np.array(event_zscore)
                    extension_mask = time_axis_arr > effective_end
                    if np.any(extension_mask):
                        extended_time = time_axis_arr[extension_mask]
                        extended_z = event_zscore_arr[extension_mask]
                        peaks, _ = find_peaks(extended_z)
                        if peaks.size > 0:
                            effective_end = extended_time[peaks[0]]

                # Create mask for time_axis within [0, effective_end].
                time_axis_arr = np.array(time_axis)
                event_zscore_arr = np.array(event_zscore)
                mask = (time_axis_arr >= 0) & (time_axis_arr <= effective_end)
                if not np.any(mask):
                    continue
                final_time = time_axis_arr[mask]
                final_z = event_zscore_arr[mask]

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


    def compute_event_induced_DA(self, pre_time=4, post_time=10):
        """
        Computes the event-induced DA (EI DA) signal for each behavior event in self.behaviors.
        
        For each event, this method:
        - Extracts DA data from (Event_Start - pre_time) to (Event_Start + post_time) using self.timestamps and self.zscore.
        - Computes a baseline from the pre-event period and subtracts it from the signal.
        - Interpolates the corrected signal onto a common time axis (relative to the event onset).
        
        The common time axis and the resulting baseline-corrected (event-induced) signal are stored
        in new columns 'Relative_Time_Axis' and 'Relative_Zscore' in self.behaviors.
        
        Parameters:
            pre_time (float): Seconds before event onset to include (default is 4 s).
            post_time (float): Seconds after event onset to include (default is 10 s).
        """
        if self.behaviors is None or self.behaviors.empty:
            print(f"Trial {self.subject_name}: No behavior events available for event-induced DA computation.")
            return

        # Determine the average sampling interval from the timestamps.
        dt = np.mean(np.diff(self.timestamps))
        # Create a common time axis from -pre_time to +post_time relative to event onset.
        common_time_axis = np.arange(-pre_time, post_time, dt)
        
        relative_time_list = []
        relative_zscore_list = []
        
        # Process each behavior event.
        for idx, row in self.behaviors.iterrows():
            event_start = row['Event_Start']
            window_start = event_start - pre_time
            window_end = event_start + post_time

            # Select the DA data within the extraction window.
            mask = (self.timestamps >= window_start) & (self.timestamps <= window_end)
            if not np.any(mask):
                relative_time_list.append(np.full(common_time_axis.shape, np.nan))
                relative_zscore_list.append(np.full(common_time_axis.shape, np.nan))
            else:
                # Convert to relative time (with respect to the event onset).
                rel_time = self.timestamps[mask] - event_start
                signal = self.zscore[mask]
                
                # Compute baseline from the pre-event period.
                pre_mask = rel_time < 0
                baseline = np.nanmean(signal[pre_mask]) if np.any(pre_mask) else 0
                corrected_signal = signal - baseline
                
                # Interpolate the corrected signal onto the common time axis.
                interp_signal = np.interp(common_time_axis, rel_time, corrected_signal)
                
                relative_time_list.append(common_time_axis)
                relative_zscore_list.append(interp_signal)
        
        # Store the computed arrays in new DataFrame columns.
        self.behaviors["Relative_Time_Axis"] = relative_time_list
        self.behaviors["Relative_Zscore"] = relative_zscore_list
