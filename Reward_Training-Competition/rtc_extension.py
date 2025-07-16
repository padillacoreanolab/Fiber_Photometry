# This file creates a child class of Experiment that is specific to RTC recordings. RTC recordings can either have one mouse or two mice in the same folder.
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from experiment_class import Experiment
from trial_class import Trial
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


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
            # 1) Downsampling to 100 Hz
            trial.downsample(target_fs=100)

            # 2) trim LED/artifact
            trial.remove_initial_LED_artifact(t=30)
            trial.remove_final_data_segment(t=10)

            # 3) low‐pass
            trial.lowpass_filter(cutoff_hz=3.0)

            # 4) high‐pass recentered
            trial.baseline_drift_highpass_recentered(cutoff=0.001)

            # 5) IRLS fit
            trial.motion_correction_align_channels_IRLS(IRLS_constant=1.4)

            # 6) compute dF/F
            trial.compute_dFF()

            # 7) zscore
            trial.compute_zscore(method='standard')

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
        df = self.da_df.copy()

        def find_closest_port_entries(PEs, port_offsets):
            closest = []
            for pe in PEs:
                # only keep offsets strictly after this PE
                valid = [off for off in port_offsets if off > pe]
                closest.append(valid[0] if valid else np.nan)
            return closest

        def compute_for_row(row):
            # get the arrays (they may be lists or ndarrays)
            PEs = row[PE_column]
            offs = row[offset_column]

            # if it's None or empty, return an empty list
            if PEs is None or len(PEs) == 0:
                return []
            if offs is None or len(offs) == 0:
                # return NaN for each PE
                return [np.nan] * len(PEs)

            # make sure they’re proper Python lists
            PEs = list(PEs)
            offs = list(offs)

            return find_closest_port_entries(PEs, offs)

        df['closest_PE_offset'] = df.apply(compute_for_row, axis=1)
        self.da_df = df
        return df





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
                       xlim: tuple = None,
                       bin_size: int = 1,
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

    
        ax.set_xticks([-4, 0, 4, 10,20,30])
        if xlim is not None:
            ax.set_xlim(xlim)
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


    def plot_specific_event_psth(
        self,
        event_type: str,
        event_index: int,
        directory_path: str,
        brain_region: str,
        y_min: float,
        y_max: float,
        df=None,
        condition='Win',
        bin_size=1,
        xlim=None
    ):
        """
        Two‐step PSTH: top panel shows mean±SEM of z‐scored ΔF/F around a specific event.
        xlim can be a float (end time, start assumed -4) or a (xmin,xmax) tuple.
        """
        if df is None:
            df = self.da_df

        # 1) filter by region prefix
        mask = df['subject_name'].str.startswith('p' if brain_region=='mPFC' else 'n')
        df = df[mask]
        idx = event_index - 1

        # 2) collect each trial's Z‐score trace for that event
        traces = []
        for _, row in df.iterrows():
            ev = row.get(f'{event_type}_Zscore', [])
            if isinstance(ev, list) and len(ev) > idx:
                traces.append(np.array(ev[idx]))
        if not traces:
            print(f"No {event_type}#{event_index} in {brain_region}")
            return

        # 3) common time axis & stack
        common_t = df.iloc[0][f'{event_type}_Time_Axis'][idx]
        traces = np.vstack(traces)
        mean_tr = traces.mean(axis=0)
        sem_tr  = traces.std(axis=0, ddof=1) / np.sqrt(traces.shape[0])

        # 4) optional downsampling
        if hasattr(self, 'downsample_data'):
            mean_tr, t_ds = self.downsample_data(mean_tr, common_t, bin_size)
            sem_tr, _   = self.downsample_data(sem_tr,   common_t, bin_size)
        else:
            t_ds = common_t

        # 5) plotting
        fig, ax = plt.subplots(figsize=(10,6))

        color = '#FFAF00' if brain_region=='mPFC' else '#15616F'
        ax.plot(t_ds, mean_tr, color=color, lw=3, label='Mean DA')
        ax.fill_between(t_ds, mean_tr-sem_tr, mean_tr+sem_tr,
                        color=color, alpha=0.4, label='SEM')

        # event markers
        ax.axvline(0, color='black', linestyle='--', lw=2)
        ax.axvline(4, color='#FF69B4', linestyle='-', lw=2)

        # axes labels
        ax.set_xlabel('Time from Tone Onset (s)', fontsize=30)
        ax.set_ylabel('Event‐Induced z‐scored ΔF/F', fontsize=30)
        ax.set_title(f'{event_type} Event {event_index} {condition} PSTH', fontsize=30, pad=30)
        ax.set_ylim(y_min, y_max)

        # interpret xlim argument
        if xlim is not None:
            if isinstance(xlim, (list,tuple)) and len(xlim)==2:
                xmin, xmax = xlim
            else:
                xmin, xmax = -4, float(xlim)
            ax.set_xlim(xmin, xmax)
        else:
            # default window -4 to 30
            ax.set_xlim(-4, 30)

        # dynamic xticks
        xmin, xmax = ax.get_xlim()
        xt = np.unique(np.round(np.linspace(xmin, xmax, 5),1))
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{t:.0f}" for t in xt], fontsize=30)

        ax.tick_params(axis='y', labelsize=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)

        # save if requested
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
            fname = f"{brain_region}_{event_type}_Evt{event_index}_PSTH.png"
            plt.savefig(os.path.join(directory_path, fname),
                        transparent=True, dpi=300, bbox_inches="tight")

        plt.show()


    



    def plot_PETH_index_grid(self,
                            df: pd.DataFrame,
                            event_type: str,
                            event_index: int,
                            brain_region: str,
                            bin_size: int = 1,
                            ncols: int = 4,
                            figsize_per_plot: tuple = (3, 2),
                            directory_path: str = None):
        """
        Plot each session's PSTH for a specific event_index in its own subplot,
        mark the first PE and the computed closest‐offset for that PE,
        and return a DataFrame of those two times.
        Titles now show the 'file name' column.
        """
        # pick only this region
        def split_by_subject(df1, region):
            return (df1[df1['subject_name'].str.startswith('p')]
                    if region=='mPFC'
                    else df1[df1['subject_name'].str.startswith('n')])

        df_reg = split_by_subject(df, brain_region)
        idx    = event_index - 1

        traces    = []
        peak_rows = []

        # gather per‐session data
        for _, row in df_reg.iterrows():
            tz = row.get(f'{event_type}_Zscore', [])
            ta = row.get(f'{event_type}_Time_Axis', [])
            if not isinstance(tz, (list, np.ndarray)) or len(tz) <= idx:
                continue

            trace     = np.array(tz[idx])
            time_axis = np.array(ta[idx])
            cue_abs   = row.get('filtered_sound_cues', [np.nan])[idx]

            # first port entry after cue, and its closest offset
            first_pe  = row.get('first_PE_after_sound_cue', [np.nan]*len(tz))[idx]
            offset_abs = row.get('closest_PE_offset', [np.nan]*len(tz))[idx]

            # convert to time *relative* to cue
            pe_rel     = (first_pe  - cue_abs) if not np.isnan(first_pe)  else np.nan
            offset_rel = (offset_abs - cue_abs) if not np.isnan(offset_abs) else np.nan

            traces.append((trace, time_axis, row['file name'], pe_rel, offset_rel))
            peak_rows.append({
                'file_name':        row['file name'],
                'event_type':       event_type,
                'event_index':      event_index,
                'brain_region':     brain_region,
                'first_PE_rel_s':   pe_rel,
                'offset_rel_s':     offset_rel
            })

        if not traces:
            print(f"No data for {event_type} event #{event_index} in {brain_region}")
            return pd.DataFrame()

        # grid dimensions
        N     = len(traces)
        nrows = math.ceil(N / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(figsize_per_plot[0]*ncols,
                                        figsize_per_plot[1]*nrows),
                                sharex=True, sharey=True)
        axes = axes.flatten()

        base_color = '#FFAF00' if brain_region=='mPFC' else '#15616F'
        pe_color   = 'cyan'
        off_color  = 'magenta'

        for i, (trace, ta, fname, pe_rel, off_rel) in enumerate(traces):
            ds, dt = self.downsample_data(trace, ta, bin_size)

            # find nearest downsampled bin for each marker
            def _closest(t):
                return dt[np.abs(dt - t).argmin()] if not np.isnan(t) else np.nan

            pe_t  = _closest(pe_rel)
            off_t = _closest(off_rel)

            ax = axes[i]
            ax.plot(dt, ds, color=base_color, lw=1.5)
            ax.axvline(0, color='k',       ls='--', lw=1)
            ax.axvline(4, color='#FF69B4', ls='-',  lw=1)

            # only label legend on first subplot
            lbl_pe  = '1st PE'     if i==0 else None
            lbl_off = 'PE → offset' if i==0 else None

            ax.axvline(pe_t,  color=pe_color,  ls='-.', lw=1, label=lbl_pe)
            ax.axvline(off_t, color=off_color, ls=':',  lw=1, label=lbl_off)

            ax.set_title(fname, fontsize=8)
            ax.set_xlim(dt[0], dt[-1])
            ax.tick_params(labelsize=6)
            if i % ncols == 0:
                ax.set_ylabel('z ΔF/F', fontsize=6)
            if i // ncols == nrows - 1:
                ax.set_xlabel('Time (s)', fontsize=6)

        # turn off any unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        # shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, fontsize=6, loc='upper right', frameon=False)

        plt.suptitle(f"{event_type} evt#{event_index} PSTH ({brain_region})", fontsize=10)
        plt.tight_layout(rect=[0,0,1,0.95])

        # save
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


    def compute_spontaneous_events(self,
                               threshold_factor: float = 3.0,
                               min_distance_s:    float = 1.0,
                               min_width_s:       float = 0.5):
        """
        Runs detect_spontaneous_events on every trial's full z‐score trace,
        and stores a new DataFrame self.spont_peaks with one row per detected peak:
        ['subject_name','file name','event_time','amplitude','width_s']
        """
        def detect_spontaneous_events(
            timestamps: np.ndarray,
            signal: np.ndarray,
            threshold_factor: float = 3.0,
            min_distance_s: float = 1.0,
            min_width_s:    float = 0.5
        ):
            """
            Detect peaks using a MAD‐threshold, plus a minimum inter‐peak interval
            and a minimum peak width.
            Returns:
            peak_times (np.ndarray),
            peak_amps (np.ndarray),
            peak_widths (np.ndarray, in seconds)
            """
            # 1) compute baseline + MAD
            med = np.median(signal)
            mad = np.median(np.abs(signal - med))
            thresh = med + threshold_factor * mad

            # 2) convert time criteria into sample counts (assumes uniform dt)
            dt = np.median(np.diff(timestamps))
            min_dist_samples  = int(np.round(min_distance_s / dt))
            min_width_samples = int(np.round(min_width_s    / dt))

            # 3) detect peaks
            peaks, props = find_peaks(
                signal,
                height=thresh,
                distance=min_dist_samples,
                width=min_width_samples
            )

            # 4) collect outputs
            peak_times  = timestamps[peaks]
            peak_amps   = signal[peaks]
            peak_widths = props["widths"] * dt

            return peak_times, peak_amps, peak_widths
        rows = []
        for _, row in self.da_df.iterrows():
            subj   = row['subject_name']
            fname  = row['file name']
            # full‐session trace & timestamps:
            ts = np.asarray(row['trial'].timestamps, dtype=float)
            zs = np.asarray(row['trial'].zscore,     dtype=float)
            # detect
            peak_times, peak_amps, peak_widths = detect_spontaneous_events(
                ts, zs,
                threshold_factor=threshold_factor,
                min_distance_s=min_distance_s,
                min_width_s=min_width_s
            )
            for t,a,w in zip(peak_times, peak_amps, peak_widths):
                rows.append({
                    'subject_name': subj,
                    'file name':    fname,
                    'event_time':   t,
                    'amplitude':    a,
                    'width_s':      w
                })

        self.spont_peaks = pd.DataFrame(rows)
        return self.spont_peaks





    