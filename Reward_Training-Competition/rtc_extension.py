# This file creates a child class of Experiment that is specific to RTC recordings. RTC recordings can either have one mouse or two mice in the same folder.
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment
from scipy.stats import pearsonr
from trial_class import Trial
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
from matplotlib.colors import LinearSegmentedColormap

class RTC(Experiment):
    def __init__(self, experiment_folder_path, behavior_folder_path):
        super().__init__(experiment_folder_path, behavior_folder_path, RTC=True)
        self.port_bnc = {} 
        self.trials_df = pd.DataFrame()
        self.da_df = pd.DataFrame()
        self.load_rtc_trials() 


    '''********************************** Initial Processing  **********************************'''
    def load_rtc_trials(self):
        """
        Unified trial loader for RTC recordings.

        For each folder in self.experiment_folder_path:
        - If the folder name contains an underscore, it is assumed to be multisubject.
            Two Trial objects are created using different channel pairs, and unique keys are generated.
        - Otherwise, a single Trial object is created.
        """
        trial_folders = [
            folder for folder in os.listdir(self.experiment_folder_path)
            if os.path.isdir(os.path.join(self.experiment_folder_path, folder))
        ]
        
        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            
            # Split the folder name by underscore.
            parts = trial_folder.split('_')
            
            # If there's an underscore, assume multisubject.
            if len(parts) > 1:
                # Create two Trial objects with different channel pairs.
                trial_obj1 = Trial(trial_path, '_465A', '_405A')
                trial_obj2 = Trial(trial_path, '_465C', '_405C')
                
                # Extract subject identifiers.
                subject1 = parts[0]  # e.g., "nn3"
                # For the second subject, split the second part by dash to extract the subject ID.
                subject2 = parts[1].split('-')[0]  # e.g., from "nn4-250124-064620" get "nn4"
                
                # Reconstruct a common identifier from the remainder (if present).
                if '-' in parts[1]:
                    # Get everything after the first dash.
                    rest = parts[1].split('-', 1)[1]  # e.g., "250124-064620"
                    trial_key1 = f"{subject1}-{rest}"
                    trial_key2 = f"{subject2}-{rest}"
                else:
                    trial_key1 = subject1
                    trial_key2 = subject2
                
                # Store the trial objects using the generated keys.
                self.trials[trial_key1] = trial_obj1
                self.trials[trial_key2] = trial_obj2
                
                # Record port information for multisubject.
                self.port_bnc[trial_key1] = 2
                self.port_bnc[trial_key2] = 3
            else:
                # Unisubject recording: Create one Trial object.
                trial_obj = Trial(trial_path, '_465A', '_405A')
                self.trials[trial_folder] = trial_obj


    def rtc_processing(self):
        """
        Unified processing for RTC recordings that handles both unisubject and multisubject trials.

        For each trial:
        1. Optionally remove designated time segments.
        2. Remove initial LED artifact.
        3. Highpass filter to remove baseline drift.
        4. Align channels, compute dFF, determine baseline period.
        5. Compute standard z-score and verify the signal.
        6. Reassign behavior channels for tone and port entries.
            - For multisubject, use self.port_bnc to decide whether to use PC3_ (port value 3) or PC2_ (port value 2).
            - For unisubject, try PC2_ first, then PC3_.
        7. Remove the first behavior entry (if it is not counting).
        8. Filter port entries so that only those after the first sound cue remain.
        """
        for trial_folder, trial in self.trials.items():
            print(f"Processing trial {trial_folder}...")

            # ----- Preprocessing Steps -----
            trial.remove_initial_LED_artifact(t=30)
            trial.highpass_baseline_drift()  # Used specifically for RTC to not smooth
            trial.align_channels()
            trial.compute_dFF()
            # baseline_start, baseline_end = trial.find_baseline_period()
            trial.compute_zscore(method='standard')
            trial.verify_signal()

            # ----- Reassign Behavior Channels -----
            # Sound cues always come from PC0_
            trial.rtc_events['sound cues'] = trial.rtc_events.pop('PC0_')

            # Determine if this trial is multisubject.
            if trial_folder in self.port_bnc:
                # Multisubject: use port info to select the proper channel.
                port_val = self.port_bnc[trial_folder]
                if port_val == 3:
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC3_')
                elif port_val == 2: 
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC2_')
                else:
                    print(f"Warning: Unexpected port value ({port_val}) for trial {trial_folder}")
            else:
                # Unisubject: try PC2_ first; if not available, try PC3_.
                if 'PC2_' in trial.rtc_events:
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC2_')
                elif 'PC3_' in trial.rtc_events:
                    trial.rtc_events['port entries'] = trial.rtc_events.pop('PC3_')
                else:
                    print(f"Warning: No port entries channel found for trial {trial_folder}")

            # ----- Post-Processing of Behaviors -----
            # Remove the first (non-counting) entry for both behaviors.
            trial.rtc_events['sound cues'].onset_times = trial.rtc_events['sound cues'].onset[1:]
            trial.rtc_events['sound cues'].offset_times = trial.rtc_events['sound cues'].offset[1:]
            trial.rtc_events['port entries'].onset_times = trial.rtc_events['port entries'].onset[1:]
            trial.rtc_events['port entries'].offset_times = trial.rtc_events['port entries'].offset[1:]
            
            valid_sound_cues = [t for t in trial.rtc_events['sound cues'].onset_times if t >= 200]
            trial.rtc_events['sound cues'].onset_times = valid_sound_cues



    def remove_specified_subjects(self):
        """
        Removes specified subjects not used in data analysis
        """
        # List of subject names to remove
        subjects_to_remove = ["n4", "n3", "n2", "n1", 'p4']

        # Remove rows where 'subject_names' are in the list
        df_combined = self.trials_df[~self.trials_df['subject_name'].isin(subjects_to_remove)]
        
        self.trials_df = df_combined


    def extract_da_columns(self):
        """
        Extracts dopamine-related columns from trials_df into da_df.
        """
        da_cols = [
            'subject_name', 'file name', 'trial',
            'filtered_sound_cues', 'filtered_port_entries', 'filtered_port_entry_offset',
            'first_PE_after_sound_cue',
            'Tone_Time_Axis', 'Tone_Zscore',
            'PE_Time_Axis', 'PE_Zscore', 'filtered_winner_array',
            'HVL_PreComp', 'HVL_Comp',
        ]


        # Only keep columns that actually exist in the DataFrame
        available_cols = [col for col in da_cols if col in self.trials_df.columns]
        self.da_df = self.trials_df[available_cols].copy()


    

    """******************************* PORT ENTRY CALCULATIONS ********************************"""
    def find_first_port_entry_after_sound_cue(self):
        """
        Finds the first port entry occurring ≥4 s after each sound cue.
        If a cue is np.nan, emits np.nan.  If an ongoing port-entry spans
        the 4 s threshold, uses threshold_time; otherwise picks the first
        port-onset ≥ threshold.  Always returns a list the same length
        as filtered_sound_cues.
        """
        import numpy as np

        df = self.da_df
        all_first_PEs = []

        for idx, row in df.iterrows():
            # pull in your placeholders (may be lists containing np.nan)
            cues   = np.asarray(row.get('filtered_sound_cues',   []), dtype=float)
            onsets = np.asarray(row.get('filtered_port_entries',  []), dtype=float)
            offs   = np.asarray(row.get('filtered_port_entry_offset', []), dtype=float)

            first_PEs = []
            for cue in cues:
                # if this cue was a placeholder, carry forward nan
                if np.isnan(cue):
                    first_PEs.append(np.nan)
                    continue

                threshold = cue + 4.0

                # check for any entry spanning the threshold
                ongoing_idx = np.where((onsets < threshold) & (offs >= threshold))[0]
                if ongoing_idx.size > 0:
                    first_PEs.append(threshold)
                else:
                    # otherwise first future onset
                    future_idx = np.where(onsets >= threshold)[0]
                    if future_idx.size > 0:
                        first_PEs.append(onsets[future_idx[0]])
                    else:
                        first_PEs.append(np.nan)

            all_first_PEs.append(first_PEs)

        # write back—each row gets a list of the same length as its cues
        df['first_PE_after_sound_cue'] = all_first_PEs
        return df



    def compute_closest_port_offset(self, PE_column, offset_column):
        """
        Computes the closest port entry offsets after each PE time and adds them as a new column in the dataframe.
        
        Parameters:
            PE_column (str): The column name for the PE times (e.g., 'first_PE_after_sound_cue').
            offset_column (str): The column name for the port entry offset times (e.g., 'filtered_port_entry_offset').
            new_column_name (str): The name of the new column to store the results. Default is 'closest_port_entry_offsets'.
        
        Returns:
            pd.DataFrame: Updated DataFrame with the new column of closest port entry offsets.
        """
        df = self.da_df 

        def find_closest_port_entries(PEs, port_entry_offsets):
            """Finds the closest port entry offsets greater than each PE in 'PEs'."""
            closest_offsets = []
            
            for PE in PEs:
                # Find the indices where port_entry_offset > PE
                valid_indices = np.where(port_entry_offsets > PE)[0]
                
                if len(valid_indices) == 0:
                    closest_offsets.append(np.nan)  # Append NaN if no valid offset is found
                else:
                    # Get the closest port entry offset (the first valid one in the array)
                    closest_offset = port_entry_offsets[valid_indices[0]]
                    closest_offsets.append(closest_offset)
            
            return closest_offsets

        def compute_PE_metrics(row):
            """Compute the closest port entry offsets for each trial."""
            # Extract first_PE_after_sound_cue and filtered_port_entry_offset
            first_PEs = np.array(row[PE_column])
            port_entry_offsets = np.array(row[offset_column])
            
            # Get the closest port entry offsets for each PE
            closest_offsets = find_closest_port_entries(first_PEs, port_entry_offsets)
            
            return closest_offsets

        # Apply the function to the DataFrame and create a new column with the results
        df['closest_PE_offset'] = df.apply(compute_PE_metrics, axis=1)

    def compute_duration(self, df=None):
        if df is None:
            df = self.df
        df["duration"] = df.apply(lambda row: np.array(row["closest_PE_offset"]) - np.array(row["first_PE_after_sound_cue"]), axis=1)


    """******************************* DOPAMINE CALCULATIONS ********************************"""
    def compute_standard_DA(self, pre_time=4, post_time=10):
        """
        Compute *raw* peri‐event z‐score traces for Tone and PE.
        - Tone_Time_Axis / Tone_Zscore:   -pre_time→+post_time around each cue
        - PE_Time_Axis    / PE_Zscore:     0→+post_time    around each PE
        """
        df = self.da_df

        # 1) find dt
        min_dt = np.inf
        for _, row in df.iterrows():
            ts = np.array(row['trial'].timestamps)
            if ts.size > 1:
                min_dt = min(min_dt, np.min(np.diff(ts)))
        if min_dt == np.inf:
            raise RuntimeError("No valid timestamps found for dt.")

        # 2) common axes
        tone_axis = np.arange(-pre_time, post_time, min_dt)
        pe_axis   = np.arange(0,       post_time, min_dt)

        # 3) containers
        tone_z, tone_t = [], []
        pe_z,   pe_t   = [], []

        # 4) loop trials
        for _, row in df.iterrows():
            trial = row['trial']
            ts    = np.array(trial.timestamps)
            zs    = np.array(trial.zscore)

            # — Tone —
            cues = row.get('filtered_sound_cues', [])
            tz, tt = [], []
            if not cues:
                tz.append(np.full_like(tone_axis, np.nan))
                tt.append(tone_axis.copy())
            else:
                for cue in cues:
                    mask = (ts >= cue - pre_time) & (ts <= cue + post_time)
                    rel  = ts[mask] - cue
                    sig  = zs[mask]
                    interp = np.interp(tone_axis, rel, sig,
                                    left=np.nan, right=np.nan)
                    tz.append(interp)
                    tt.append(tone_axis.copy())
            tone_z.append(tz)
            tone_t.append(tt)

            # — PE —
            pes = row.get('first_PE_after_sound_cue', [])
            pz, pt = [], []
            if not isinstance(pes, (list, np.ndarray)) or len(pes)==0:
                pz.append(np.full_like(pe_axis, np.nan))
                pt.append(pe_axis.copy())
            else:
                for i, pe in enumerate(pes):
                    if pe is None or (isinstance(pe,float) and np.isnan(pe)):
                        pz.append(np.full_like(pe_axis, np.nan))
                        pt.append(pe_axis.copy())
                    else:
                        mask = (ts >= pe) & (ts <= pe + post_time)
                        rel  = ts[mask] - pe
                        sig  = zs[mask]
                        interp = np.interp(pe_axis, rel, sig,
                                        left=np.nan, right=np.nan)
                        pz.append(interp)
                        pt.append(pe_axis.copy())
            pe_z.append(pz)
            pe_t.append(pt)

        # 5) save back
        df['Tone_Time_Axis'] = tone_t
        df['Tone_Zscore']    = tone_z
        df['PE_Time_Axis']   = pe_t
        df['PE_Zscore']      = pe_z

        return df



    def compute_EI_DA(self,
                  tone_window: tuple[float,float]    = (-4, 10),
                  pe_window:    tuple[float,float]    = (0, 10),
                  baseline_window: tuple[float,float] = (-4, 0)):
        """
        Compute baseline-corrected peri-event z-score traces for Tone and PE.
        
        Parameters
        ----------
        tone_window : (start, end) in seconds, relative to cue
        pe_window   : (start, end) in seconds, relative to first lick
        baseline_window : (start, end) in seconds, also relative to cue, 
                        from which to compute your baseline
        
        Returns
        -------
        Updates self.da_df in place, adding columns:
        'Tone_Time_Axis', 'Tone_Zscore',
        'PE_Time_Axis',   'PE_Zscore'
        """
        df = self.da_df

        # 1) find the finest dt across all trials
        min_dt = np.inf
        for _, row in df.iterrows():
            ts = np.array(row['trial'].timestamps)
            if ts.size > 1:
                min_dt = min(min_dt, np.min(np.diff(ts)))
        if not np.isfinite(min_dt):
            raise RuntimeError("No valid timestamps found to establish dt.")

        # 2) build common time-axes
        tone_start, tone_end = tone_window
        pe_start,   pe_end   = pe_window
        bl_start,   bl_end   = baseline_window

        tone_axis = np.arange(tone_start, tone_end, min_dt)
        pe_axis   = np.arange(pe_start,   pe_end,   min_dt)

        # 3) containers
        tone_z, tone_t = [], []
        pe_z,   pe_t   = [], []

        # 4) iterate trials
        for _, row in df.iterrows():
            trial = row['trial']
            ts    = np.array(trial.timestamps)
            zs    = np.array(trial.zscore)

            # get your cues & PEs, defaulting to empty list
            cues = row.get('filtered_sound_cues') or []
            pes  = row.get('first_PE_after_sound_cue') or []

            # —— Tone processing —— 
            tz_list, tt_list = [], []
            for i, cue in enumerate(cues):
                # if exactly 40 cues, skip the 40th one entirely - The last one sometimes didn't have enough data so it looked wonky
                if len(cues)==40 and i==39:
                    continue

                # mask out the window around that cue
                mask = (ts >= cue + tone_start) & (ts <= cue + tone_end)
                if not mask.any():
                    tz_list.append(np.full_like(tone_axis, np.nan))
                    tt_list.append(tone_axis.copy())
                    continue

                rel = ts[mask] - cue
                sig = zs[mask]

                # baseline from baseline_window
                blm = (rel >= bl_start) & (rel <= bl_end)
                base = np.nanmean(sig[blm]) if blm.any() else 0.0

                # subtract baseline and re-interpolate
                corr = sig - base
                tz_list.append(np.interp(tone_axis, rel, corr))
                tt_list.append(tone_axis.copy())

            tone_z.append(tz_list)
            tone_t.append(tt_list)

            # —— PE processing —— 
            pz_list, pt_list = [], []
            for i, pe in enumerate(pes):
                # same skip rule if you want to skip matching last PE
                if len(cues)==40 and i==39:
                    continue

                # skip if PE is None or NaN
                if pe is None or (isinstance(pe,float) and np.isnan(pe)):
                    pz_list.append(np.full_like(pe_axis, np.nan))
                    pt_list.append(pe_axis.copy())
                    continue

                # pick the same cue for baseline if exists
                cue = cues[i] if i < len(cues) else None
                if cue is not None:
                    bm = (ts >= cue + bl_start) & (ts <= cue + bl_end)
                    base_val = np.nanmean(zs[bm]) if bm.any() else 0.0
                else:
                    base_val = 0.0

                mask = (ts >= pe + pe_start) & (ts <= pe + pe_end)
                if not mask.any():
                    pz_list.append(np.full_like(pe_axis, np.nan))
                    pt_list.append(pe_axis.copy())
                    continue

                rel  = ts[mask] - pe
                corr = zs[mask] - base_val
                pz_list.append(np.interp(pe_axis, rel, corr))
                pt_list.append(pe_axis.copy())

            pe_z.append(pz_list)
            pe_t.append(pt_list)

        # 5) write back into your DataFrame
        df['Tone_Time_Axis'] = tone_t
        df['Tone_Zscore']    = tone_z
        df['PE_Time_Axis']   = pe_t
        df['PE_Zscore']      = pe_z

        # return for chaining if you like
        return df

    def compute_rtc_da_metrics(
        self,
        bout_duration: float = 4.0,       # still used for PE
        include_pretrial: bool = False,
        pretrial_duration: float = 10.0
    ):
        df = self.da_df.copy()

        if include_pretrial and 'Pretrial_Zscore' not in df.columns:
            self.compute_pretrial_EI_DA()
            df = self.da_df.copy()

        def _extract(arrs, t_arrs, window):
            # … your unchanged extraction code …
            aucs, maxs, times, means = [], [], [], []
            for arr, t in zip(arrs, t_arrs):
                a  = np.asarray(arr, dtype=float)
                t0 = np.asarray(t,   dtype=float)
                if a.size == 0 or t0.size == 0:
                    aucs.append(np.nan); maxs.append(np.nan)
                    times.append(np.nan); means.append(np.nan)
                    continue

                if window is None:
                    mask = np.ones_like(t0, dtype=bool)
                else:
                    start, end = window
                    mask = (t0 >= start) & (t0 <= end)

                seg   = a[mask]; seg_t = t0[mask]
                if seg.size and not np.all(np.isnan(seg)):
                    aucs.append(np.trapz(seg, seg_t))
                    idx = np.nanargmax(seg)
                    maxs.append(seg[idx])
                    times.append(seg_t[idx])
                    means.append(np.nanmean(seg))
                else:
                    aucs.append(np.nan); maxs.append(np.nan)
                    times.append(np.nan); means.append(np.nan)

            return aucs, maxs, times, means

        trial_metrics = {
            'Tone': {'auc': [], 'max': [], 'time': [], 'mean': []},
            'PE':   {'auc': [], 'max': [], 'time': [], 'mean': []}
        }
        pretrial_metrics = {'auc': [], 'max': [], 'time': [], 'mean': []}

        for _, row in df.iterrows():
            # 1) Tone — force window = first 4 seconds only
            z_tone = row.get('Tone_Zscore', []) or []
            t_tone = row.get('Tone_Time_Axis', []) or []
            a_t, M_t, t_t, mu_t = _extract(z_tone, t_tone, window=(0, bout_duration))
            trial_metrics['Tone']['auc'].append(a_t)
            trial_metrics['Tone']['max'].append(M_t)
            trial_metrics['Tone']['time'].append(t_t)
            trial_metrics['Tone']['mean'].append(mu_t)

            # 2) PE — still uses bout_duration
            z_pe = row.get('PE_Zscore', []) or []
            t_pe = row.get('PE_Time_Axis', []) or []
            a_pe, M_pe, t_pe2, mu_pe = _extract(z_pe, t_pe, window=(0, bout_duration))
            trial_metrics['PE']['auc'].append(a_pe)
            trial_metrics['PE']['max'].append(M_pe)
            trial_metrics['PE']['time'].append(t_pe2)
            trial_metrics['PE']['mean'].append(mu_pe)

            # 3) pretrial (unchanged)
            if include_pretrial:
                pre_z = row.get('Pretrial_Zscore', []) or []
                pre_t = row.get('Pretrial_Time_Axis', []) or []
                a_p, M_p, t_p, mu_p = _extract(pre_z, pre_t, window=None)
                pretrial_metrics['auc'].append(a_p)
                pretrial_metrics['max'].append(M_p)
                pretrial_metrics['time'].append(t_p)
                pretrial_metrics['mean'].append(mu_p)

        # write back trial metrics
        for ev, store in trial_metrics.items():
            df[f'{ev} AUC']              = store['auc']
            df[f'{ev} Max Peak']         = store['max']
            df[f'{ev} Time of Max Peak'] = store['time']
            df[f'{ev} Mean Z-score']     = store['mean']

        # write back pretrial metrics
        if include_pretrial:
            df['Pretrial AUC']              = pretrial_metrics['auc']
            df['Pretrial Max Peak']         = pretrial_metrics['max']
            df['Pretrial Time of Max Peak'] = pretrial_metrics['time']
            df['Pretrial Mean Z-score']     = pretrial_metrics['mean']

        self.da_df = df
        return df




    """******************************* AVERAGING FOR COMPARISON ********************************"""
    def find_overall_mean(self):
        """
        Computes the per-subject mean of all available DA metrics in self.trials_df.
        Automatically detects which scalar and array columns are present and skips missing ones.
        """
        df = self.trials_df

        def mean_arrays(group):
            result = {}

            # Always keep Time Axis
            if "Tone_Time_Axis" in group.columns:
                result["Tone_Time_Axis"] = group["Tone_Time_Axis"].iloc[0]

            # Optionally preserve metadata
            for meta_col in ["Rank", "Cage"]:
                if meta_col in group.columns:
                    result[meta_col] = group[meta_col].iloc[0]

            # Scalar numeric columns: take mean of list if needed, then mean across trials
            scalar_cols = [
                'PE AUC', 'PE Max Peak', 'PE Mean Z-score',
                'Tone AUC', 'Tone Max Peak', 'Tone Mean Z-score',
                'PE AUC First', 'PE AUC Last', 'PE Max Peak First', 'PE Max Peak Last',
                'PE Mean Z-score First', 'PE Mean Z-score Last',
                'Tone AUC First', 'Tone AUC Last', 'Tone Max Peak First', 'Tone Max Peak Last',
                'Tone Mean Z-score First', 'Tone Mean Z-score Last',
                'PE AUC EI', 'PE Max Peak EI', 'PE Mean Z-score EI',
                'Tone AUC EI', 'Tone Max Peak EI', 'Tone Mean Z-score EI',
                'PE AUC EI First', 'PE Max Peak EI First', 'Tone AUC EI First',
                'Tone Max Peak EI First', 'Tone Mean Z-score EI First',
                'PE AUC EI Last', 'PE Max Peak EI Last', 'Tone AUC EI Last',
                'Tone Max Peak EI Last', 'Tone Mean Z-score EI Last'
            ]
            for col in scalar_cols:
                if col in group.columns:
                    group[col] = group[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
                    group[col] = pd.to_numeric(group[col], errors='coerce')
                    result[col] = group[col].mean()

            # Array columns: element-wise mean
            array_cols = ["Tone_Zscore", "PE_Zscore"]
            for col in array_cols:
                if col in group.columns:
                    try:
                        stacked = np.vstack(group[col].dropna().values)
                        result[col] = np.mean(stacked, axis=0)
                    except:
                        result[col] = np.nan  # If stacking fails (e.g., inconsistent shapes)

            return pd.Series(result)

        return self.trials_df.groupby("subject_name").apply(mean_arrays).reset_index()


    """******************************* PLOTTING ********************************"""
    def plot_group_PETH(self,
                       df: pd.DataFrame = None,
                       event_type: str = 'Tone',
                       brain_region: str = 'NAc',
                       color: str = None,
                       title: str = None,
                       ylim: tuple = None,
                       bin_size: int = 100,
                       figsize: tuple = (6, 4),
                       save_path: str = None):
        """
        Plot a single PSTH by collapsing *all* event-induced traces (Tone or PE)
        across trials and subjects for the specified brain region.

        Parameters:
            df           : DataFrame to use (defaults to self.da_df)
            event_type   : 'Tone' or 'PE'
            brain_region : 'NAc' or 'mPFC' (filters by subject_name prefix)
            color        : hex color for trace (default per region)
            title        : figure title
            ylim         : (ymin, ymax) to set y-axis limits
            bin_size     : downsampling bin size
            figsize      : figure size tuple
            save_path    : if provided, path to save figure
        """
        # 1) choose input DataFrame
        if df is None:
            df = self.da_df

        # 2) filter by region prefix
        prefix = 'n' if brain_region == 'NAc' else 'p'
        df_reg = df[df['subject_name'].str.startswith(prefix)]
        if df_reg.empty:
            print(f"No data for region {brain_region}")
            return

        # 3) collect all per-event traces
        all_traces = []
        for _, row in df_reg.iterrows():
            evts = row.get(f"{event_type}_Zscore", [])
            if not isinstance(evts, (list, np.ndarray)):
                continue
            for tr in evts:
                arr = np.asarray(tr)
                if arr.size:
                    all_traces.append(arr)
        if not all_traces:
            print(f"No {event_type} traces found in {brain_region}")
            return

        # 4) stack & compute mean ± SEM
        M = np.vstack(all_traces)
        mean_trace = np.nanmean(M, axis=0)
        sem_trace = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])

        # 5) time axis from first event of first trial
        t0 = df_reg.iloc[0].get(f"{event_type}_Time_Axis", [])
        if not isinstance(t0, (list, np.ndarray)) or not t0:
            print("Cannot find a valid time axis.")
            return
        common_t = np.asarray(t0[0])

        # 6) downsample
        n = len(mean_trace) // bin_size
        ds_mean = mean_trace[:n*bin_size].reshape(n, bin_size).mean(axis=1)
        ds_sem = sem_trace[:n*bin_size].reshape(n, bin_size).mean(axis=1)
        ds_time = common_t[:n*bin_size].reshape(n, bin_size).mean(axis=1)

        # 7) plotting
        c = color or ('#15616F' if brain_region=='NAc' else '#FFAF00')
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ds_time, ds_mean, color=c, lw=3)
        ax.fill_between(ds_time, ds_mean-ds_sem, ds_mean+ds_sem, color=c, alpha=0.3)
        ax.axvline(0, color='k', ls='--', lw=2)
        ax.axvline(4, color='#FF69B4', ls='-', lw=2)

        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Event-induced Z-scored ΔF/F', fontsize=14)
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')
        else:
            ax.set_title(f"{brain_region} {event_type} PSTH", fontsize=16, fontweight='bold')

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xticks([-4, 0, 4, 10])
        ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def plot_specific_event_psth(self, event_type, event_index, directory_path, brain_region, y_min, y_max, df=None, condition='Win', bin_size=100):
        """
        Plots the PSTH (mean and SEM) for a specific event bout (0→4 s after event onset)
        across trials, using the same averaging logic as for a bout response.

        Parameters:
            event_type (str): The event type (e.g. 'Tone' or 'PE').
            event_index (int): 1-indexed event number to plot.
            directory_path (str or None): Directory to save the plot (if None, the plot is not saved).
            brain_region (str): Brain region ('mPFC' or other) to filter subjects.
            y_min (float): Lower bound of the y-axis.
            y_max (float): Upper bound of the y-axis.
            df (DataFrame, optional): DataFrame to use (defaults to self.df).
            bin_size (int, optional): Bin size for downsampling.
        """
        if df is None:
            df = self.da_df

        # Filter subjects by brain region.
        def split_by_subject(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region == 'mPFC' else df_n

        df = split_by_subject(df, brain_region)
        idx = event_index - 1
        # print(f"[DEBUG] PSTH: Plotting {event_type} event index {event_index} (0-indexed {idx}) for brain region {brain_region}")

        selected_traces = []
        for i, row in df.iterrows():
            event_z_list = row.get(f'{event_type}_Zscore', [])
            if isinstance(event_z_list, list) and len(event_z_list) > idx:
                trace = np.array(event_z_list[idx])
                selected_traces.append(trace)
        if len(selected_traces) == 0:
            print(f"No trials have an event at index {event_index} for {event_type}.")
            return

        # Use the common time axis from the first trial's bout.
        common_time_axis = df.iloc[0][f'{event_type}_Time_Axis'][idx]
        selected_traces = np.array(selected_traces)

        mean_trace = np.mean(selected_traces, axis=0)
        sem_trace = np.std(selected_traces, axis=0) / np.sqrt(selected_traces.shape[0])
        
        if hasattr(self, 'downsample_data'):
            mean_trace, downsampled_time_axis = self.downsample_data(mean_trace, common_time_axis, bin_size)
            sem_trace, _ = self.downsample_data(sem_trace, common_time_axis, bin_size)
        else:
            downsampled_time_axis = common_time_axis

        # # --- Debug: Check values in the 4–10 s window ---
        # post_window = (downsampled_time_axis >= 4) & (downsampled_time_axis <= 10)
        # post_vals = mean_trace[post_window]
        # print("[DEBUG] PSTH: Final mean trace shape:", mean_trace.shape)
        # print("[DEBUG] PSTH: Final mean trace first 10 values:", mean_trace[:10])
        # print("[DEBUG] PSTH: 4–10 s window values, first 10:", post_vals[:10])
        # print("[DEBUG] PSTH: Max in 4–10 s window:", np.max(post_vals))

        trace_color = '#FFAF00' if brain_region == 'mPFC' else '#15616F'

        plt.figure(figsize=(10, 6))
        plt.plot(downsampled_time_axis, mean_trace, color=trace_color, lw=3, label='Mean DA')
        plt.fill_between(downsampled_time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        color=trace_color, alpha=0.4, label='SEM')
        plt.axvline(0, color='black', linestyle='--', lw=2)
        plt.axvline(4, color='#FF69B4', linestyle='-', lw=2)

        # Force the x-axis to be from -4 to 10 seconds.
        plt.xlabel('Time from Tone Onset (s)', fontsize=30)
        plt.ylabel('Event-Induced z-scored ΔF/F', fontsize=30)
        plt.title(f'{event_type} Event {event_index} {condition} PSTH', fontsize=30, pad=30)
        plt.ylim(y_min, y_max)
        plt.xticks([-4, 0, 4, 10], fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlim(-4, 10)  # Force x-axis limits

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)

        if directory_path is not None:
            # Ensure the directory exists.
            os.makedirs(directory_path, exist_ok=True)
            save_path = os.path.join(directory_path, f'{brain_region}_{event_type}_Event{event_index}_PSTH.png')
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()

    

    def plot_PETH_index_grid(self,
                          df: pd.DataFrame,
                          event_type: str,
                          event_index: int,
                          brain_region: str,
                          bin_size: int = 100,
                          ncols: int = 4,
                          figsize_per_plot: tuple = (3, 2),
                          directory_path: str = None):
        """
        Plot each session's PSTH for a specific event_index in its own subplot,
        mark the first PE and the computed max peak, and return a DataFrame of times.
        Titles now show the 'file name' column.
        """
        import math, os, numpy as np, matplotlib.pyplot as plt

        # 1) pick only this region
        def split_by_subject(df1, region):
            df_n = df1[df1['subject_name'].str.startswith('n')]
            df_p = df1[df1['subject_name'].str.startswith('p')]
            return df_p if region=='mPFC' else df_n

        df_reg = split_by_subject(df, brain_region)
        idx    = event_index - 1

        traces    = []
        labels    = []
        peak_rows = []

        # 2) gather
        for _, row in df_reg.iterrows():
            tz = row.get(f'{event_type}_Zscore', [])
            ta = row.get(f'{event_type}_Time_Axis', [])
            if not isinstance(tz, (list, np.ndarray)) or len(tz) <= idx:
                continue

            trace     = np.array(tz[idx])
            time_axis = np.array(ta[idx])
            cue_abs   = row['filtered_sound_cues'][idx]
            first_pe  = row.get('first_PE_after_sound_cue', [None]*len(tz))[idx]
            peak_abs  = row.get(f'{event_type} Time of Max Peak', [None]*len(tz))[idx]

            # relative times
            lick_rel = (first_pe - cue_abs) if first_pe is not None else np.nan
            peak_rel = (peak_abs  - cue_abs) if peak_abs  is not None else np.nan

            traces.append((trace, time_axis, row['file name'], lick_rel, peak_rel))
            peak_rows.append({
                'file_name':     row['file name'],
                'event_type':    event_type,
                'event_index':   event_index,
                'brain_region':  brain_region,
                'first_PE_s':    lick_rel,
                'peak_time_s':   peak_rel
            })

        if not traces:
            print(f"No data for {event_type} event #{event_index} in {brain_region}")
            return pd.DataFrame()

        # 3) create grid
        N     = len(traces)
        nrows = math.ceil(N / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(figsize_per_plot[0]*ncols,
                                        figsize_per_plot[1]*nrows),
                                sharex=True, sharey=True)
        axes = axes.flatten()

        # 4) plotting parameters
        base_color = '#FFAF00' if brain_region=='mPFC' else '#15616F'
        lick_color = 'cyan'
        peak_color = 'gray'

        for i, (trace, ta, fname, lick_rel, peak_rel) in enumerate(traces):
            ds, dt = self.downsample_data(trace, ta, bin_size)

            # helper to find nearest downsampled time
            def _closest(t):
                return dt[np.abs(dt - t).argmin()] if not np.isnan(t) else np.nan

            lick_t = _closest(lick_rel)
            peak_t = _closest(peak_rel)

            ax = axes[i]
            ax.plot(dt, ds,        color=base_color, lw=1.5)
            ax.axvline(0,          color='k',      ls='--', lw=1)
            ax.axvline(4,          color='#FF69B4',ls='-',  lw=1)
            # only label legend on the first tile
            lbl_lick = '1st PE'   if i==0 else None
            lbl_peak = 'max peak' if i==0 else None
            ax.axvline(lick_t, color=lick_color, ls='-.', lw=1, label=lbl_lick)
            ax.axvline(peak_t, color=peak_color, ls=':',  lw=1, label=lbl_peak)

            ax.set_title(fname, fontsize=8)       # <-- use file name here
            ax.set_xlim(dt[0], dt[-1])
            ax.tick_params(labelsize=6)
            if i % ncols == 0:
                ax.set_ylabel('z ΔF/F', fontsize=6)
            if i // ncols == nrows - 1:
                ax.set_xlabel('Time (s)', fontsize=6)

        # 5) blank extras
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        # 6) shared legend
        handles, labels_ = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels_, fontsize=6, loc='upper right', frameon=False)

        plt.suptitle(f"{event_type} evt#{event_index} PSTH ({brain_region})", fontsize=10)
        plt.tight_layout(rect=[0,0,1,0.95])

        # 7) save if requested
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            out = os.path.join(directory_path,
                            f'{brain_region}_{event_type}_evt{event_index}_grid.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')

        plt.show()
        return pd.DataFrame(peak_rows)



    """********************************MISC*************************************"""
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
