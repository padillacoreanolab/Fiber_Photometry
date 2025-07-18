import numpy as np
import pandas as pd
import tdt
import os
import matplotlib.pyplot as plt

from trial_class2 import Trial2

class Experiment2:
    def __init__(self, experiment_folder_path, behavior_folder_path, RTC=False):
        self.experiment_folder_path = experiment_folder_path
        self.behavior_folder_path = behavior_folder_path
        self.trials = {}

        if not RTC:
            self.load_trials()
    

    '''********************************** GROUP PROCESSING **********************************'''
    def load_trials(self):
        """
        Loads each trial folder (block) as a TDTData object and extracts manual annotation behaviors.
        """
        trial_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                        if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            trial_obj = Trial2(trial_path, '_465A', '_405A')

            self.trials[trial_folder] = trial_obj


    def default_batch_process(self, time_segments_to_remove=None):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for trial_folder, trial in self.trials.items():
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Processing {trial_folder}...")
            trial.remove_initial_LED_artifact(t=30)
            trial.remove_final_data_segment(t = 10)
            
            trial.smooth_and_apply(window_len=int(trial.fs)*4)
            trial.apply_ma_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            baseline_start, baseline_end = trial.find_baseline_period()  
            # trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            trial.compute_zscore(method = 'standard')

            trial.verify_signal()

    
    def batch_process_exponential(self, time_segments_to_remove=None):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for trial_folder, trial in self.trials.items():
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Processing {trial_folder}...")
            trial.remove_initial_LED_artifact(t=30)
            trial.remove_final_data_segment(t=10)

            # 2) smooth (in‐place)
            trial.downsample(target_fs = 100)


            # 3) low‐pass
            trial.lowpass_filter(cutoff_hz=3.0)

            # 4) Double Exponential
            trial.basline_drift_double_exponential()
            # trial.highpass_baseline_drift_Recentered(cutoff=0.001)

            # 5) IRLS fit
            trial.motion_correction_align_channels_IRLS(IRLS_constant=1.4)

            # 6) compute dF/F
            trial.compute_dFF()

            # 7) zscore
            trial.compute_zscore(method='standard')


    def preprocessing_Melugin(self, max_time=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize  import curve_fit
        from scipy.signal    import butter, filtfilt
        from scipy.stats     import linregress

        def double_exponential(t, c, A_fast, A_slow, tau_slow, tau_mul):
            tau_fast = tau_slow * tau_mul
            return c + A_slow*np.exp(-t/tau_slow) + A_fast*np.exp(-t/tau_fast)

        for trial_folder, tr in self.trials.items():
            print(f"\n=== Processing {trial_folder} ===")

            # 1) trim & downsample
            tr.downsample(target_fs=100)

            tr.remove_initial_LED_artifact(t=30)
            tr.remove_final_data_segment(t=10)

            ts      = tr.timestamps
            raw_da  = tr.streams['DA']
            raw_iso = tr.streams['ISOS']
            if max_time is not None:
                m = ts <= max_time
                ts, raw_da, raw_iso = ts[m], raw_da[m], raw_iso[m]

            # fallback high-pass baseline (0.001 Hz)
            def hp_baseline(sig):
                b, a = butter(2, 0.001, btype='high', fs=tr.fs)
                return filtfilt(b, a, sig, padtype='even')

            # 2) fit & subtract for DA
            try:
                p0_da = [raw_da.max()/2, raw_da.max()/4, raw_da.max()/4, 3600, 0.1]
                bnds  = ([0,0,0,600,0], [raw_da.max()]*3 + [36000,1])
                popt_da, _ = curve_fit(
                    double_exponential, ts, raw_da,
                    p0=p0_da, bounds=bnds, maxfev=20000
                )
                da_baseline = double_exponential(ts, *popt_da)
            except RuntimeError as e:
                print("  ⚠️ DA exp‐fit failed, using high-pass instead:", e)
                da_baseline = hp_baseline(raw_da)

            da_detr = raw_da - da_baseline

            # 3) fit & subtract for ISO
            try:
                p0_iso = [raw_iso.max()/2, raw_iso.max()/4, raw_iso.max()/4, 3600, 0.1]
                bnds   = ([0,0,0,600,0], [raw_iso.max()]*3 + [36000,1])
                popt_iso, _ = curve_fit(
                    double_exponential, ts, raw_iso,
                    p0=p0_iso, bounds=bnds, maxfev=20000
                )
                iso_baseline = double_exponential(ts, *popt_iso)
            except RuntimeError as e:
                print("  ⚠️ ISO exp‐fit failed, using high-pass instead:", e)
                iso_baseline = hp_baseline(raw_iso)

            iso_detr = raw_iso - iso_baseline

            # 4) %ΔF/F
            da_pct  = 100 * da_detr  / da_baseline
            iso_pct = 100 * iso_detr / iso_baseline
            tr.dFF  = da_pct

            # 5) difference & z-score
            diff_pct = da_pct - iso_pct
            z        = (diff_pct - np.nanmean(diff_pct)) / np.nanstd(diff_pct)
            tr.zscore = z

            # 6) Plot exactly as before
            steps = [
                ((raw_da, raw_iso),        ("raw DA","raw ISO"),      "1) Raw"),
                ((da_baseline, iso_baseline),("DA fit","ISO fit"),      "2) Baseline"),
                ((da_pct,),                ("%ΔF/F",),                "3) %ΔF/F"),
                ((diff_pct,),             ("DA–ISO ΔF/F",),          "4) Diff %ΔF/F"),
                ((z,),                    ("z-score",),              "5) z-score"),
            ]

            fig, axes = plt.subplots(len(steps),1,
                                    figsize=(12,2.5*len(steps)),
                                    sharex=True)
            dual = {"1) Raw", "2) Baseline"}

            for ax, ((sig1,*rest),(lbl1,*r_lbl),title) in zip(axes,steps):
                if title in dual:
                    ax1, ax2 = ax, ax.twinx()
                    ax1.plot(ts, sig1,    'tab:blue',  lw=1.2,label=lbl1)
                    ax2.plot(ts, rest[0], 'tab:orange',lw=1.2,label=r_lbl[0])
                    ax1.spines['left'].set_position(('outward',10))
                    ax2.spines['right'].set_position(('outward',40))
                    ax1.set_ylabel(f"{title}\nDA",  color='tab:blue',   labelpad=8)
                    ax2.set_ylabel(f"{title}\nISO", color='tab:orange', labelpad=8)
                    h1,l1 = ax1.get_legend_handles_labels()
                    h2,l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize='small')
                else:
                    ax.plot(ts, sig1, lw=1.2, label=lbl1)
                    ax.set_ylabel(title, fontsize=10)
                    ax.legend(frameon=False, fontsize='small')

            axes[-1].set_xlabel("Time (s)", fontsize=12)
            fig.suptitle(f"{tr.subject_name} preprocessing steps", fontsize=14, y=0.99)
            plt.tight_layout(rect=[0,0,1,0.96])
            plt.show()


    def preprocessing_Simpson(self, max_time=None):
        """
        1) downsample → trim 30s/10s  
        2) low-pass @10Hz via Trial.lowpass_filter  
        3) double-exp bleach-fit & subtract  
        4) linear regression motion correction  
        5) compute %ΔF/F and store in tr.dFF  
        6) compute z-score via tr.compute_zscore()  
        then plot all six panels
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize  import curve_fit
        from scipy.stats     import linregress

        def double_exponential(t, c, A_fast, A_slow, tau_slow, tau_mul):
            tau_fast = tau_slow * tau_mul
            return c + A_slow * np.exp(-t/tau_slow) + A_fast * np.exp(-t/tau_fast)

        for trial_folder, tr in self.trials.items():
            print(f"\n=== Processing {trial_folder} ===")

            # 1) prep
            tr.downsample(target_fs=100)
            tr.remove_initial_LED_artifact(t=30)
            tr.remove_final_data_segment(t=10)

            # raw streams
            ts      = tr.timestamps
            raw_da  = tr.streams['DA']
            raw_iso = tr.streams['ISOS']
            if max_time is not None:
                m = ts <= max_time
                ts, raw_da, raw_iso = ts[m], raw_da[m], raw_iso[m]

            # 2) low-pass @10Hz
            tr.lowpass_filter(cutoff_hz=10.0)
            denoise_da  = tr.updated_DA.copy()
            denoise_iso = tr.updated_ISOS.copy()

            # 3) bleach-correction by double-exponential
            #   DA
            p0 = [np.max(denoise_da)/2, np.max(denoise_da)/4, np.max(denoise_da)/4, 3600, 0.1]
            bnds = ([0,0,0,600,0], [np.max(denoise_da)]*3 + [36000,1])
            popt_d, _ = curve_fit(double_exponential, ts, denoise_da, p0=p0, bounds=bnds, maxfev=2000)
            da_baseline  = double_exponential(ts, *popt_d)
            da_detrended = denoise_da - da_baseline

            #   ISO
            p0 = [np.max(denoise_iso)/2, np.max(denoise_iso)/4, np.max(denoise_iso)/4, 3600, 0.1]
            bnds = ([0,0,0,600,0], [np.max(denoise_iso)]*3 + [36000,1])
            popt_i, _ = curve_fit(double_exponential, ts, denoise_iso, p0=p0, bounds=bnds, maxfev=2000)
            iso_baseline   = double_exponential(ts, *popt_i)
            iso_detrended  = denoise_iso - iso_baseline

            # 4) motion correction via linear regression
            slope, intercept, _, _, _ = linregress(iso_detrended, da_detrended)
            est_motion   = intercept + slope * iso_detrended
            da_corrected = da_detrended - est_motion

            # 5) compute %ΔF/F and store in trial
            dFF = 100 * da_corrected / da_baseline
            tr.dFF = dFF

            # 6) compute z-score
            tr.compute_zscore(method='standard')
            zscore = tr.zscore.copy()

            # Assemble plotting steps
            steps = [
                ((raw_da,       raw_iso),      ("raw DA",    "raw ISO"),     "1) Raw"),
                ((denoise_da,   denoise_iso),  ("LP DA",     "LP ISO"),      "2) Low-pass (10Hz)"),
                ((da_detrended, iso_detrended),("DA detr",   "ISO detr"),    "3) Double-exp detrend"),
                ((da_detrended, est_motion),   ("DA",        "motion est"),  "4) LinReg fit"),
                ((dFF,),        ("%ΔF/F",),     "5) %ΔF/F"),
                ((zscore,),     ("z-score",),    "6) z-score"),
            ]

            fig, axes = plt.subplots(len(steps), 1, figsize=(12, 2.5*len(steps)), sharex=True)
            dual = {"1) Raw", "2) Low-pass (10Hz)", "3) Double-exp detrend"}

            for ax, ((sig1, *rest), (lbl1, *lbl_rest), title) in zip(axes, steps):
                if title in dual:
                    ax1, ax2 = ax, ax.twinx()
                    ax1.plot(ts, sig1,       color='tab:blue',  lw=1.2, label=lbl1)
                    ax2.plot(ts, rest[0],    color='tab:orange',lw=1.2, label=lbl_rest[0])
                    ax1.spines['left'].set_position(('outward', 10))
                    ax2.spines['right'].set_position(('outward',40))
                    ax1.set_ylabel(f"{title}\nDA",  color='tab:blue',   labelpad=8)
                    ax2.set_ylabel(f"{title}\nISO", color='tab:orange', labelpad=8)
                    h1,l1 = ax1.get_legend_handles_labels()
                    h2,l2 = ax2.get_legend_handles_labels()
                    ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize='small')
                else:
                    ax.plot(ts, sig1, lw=1.2, label=lbl1)
                    if rest:
                        ax.plot(ts, rest[0], lw=1.2, label=lbl_rest[0])
                    ax.set_ylabel(title, fontsize=10)
                    ax.legend(frameon=False, fontsize='small')

            axes[-1].set_xlabel("Time (s)", fontsize=12)
            fig.suptitle(f"{tr.subject_name} preprocessing steps", fontsize=14, y=0.99)
            plt.tight_layout(rect=[0,0,1,0.96])
            plt.show()


    
    def preprocessing_plotted_all_same_axis(self, max_time=None):
        for trial_folder, tr in self.trials.items():
            print(f"\n=== Processing {trial_folder} ===")
            
            # Downsample
            tr.downsample(target_fs=100)

            # 1) trim LED/artifact
            tr.remove_initial_LED_artifact(t=30)
            tr.remove_final_data_segment(t=10)

            # grab raw
            raw_da  = tr.streams['DA']
            raw_iso = tr.streams['ISOS']

            # 2) low-pass
            tr.lowpass_filter(cutoff_hz=3.0)
            lp_da   = tr.updated_DA.copy()
            lp_iso  = tr.updated_ISOS.copy()

            # 3) bleach-correct
            tr.basline_drift_double_exponential()
            hp_da   = tr.updated_DA.copy()
            hp_iso  = tr.updated_ISOS.copy()

            # 4) IRLS fit
            tr.motion_correction_align_channels_IRLS(IRLS_constant=1.4)
            fit_iso = tr.isosbestic_fitted.copy()

            # 5) compute dF/F
            tr.compute_dFF()
            raw_dff = tr.dFF.copy()

            # 6) z-score
            tr.compute_zscore(method='standard')
            z       = tr.zscore.copy()

            # time‐mask if desired
            ts = tr.timestamps
            if max_time is not None:
                m = ts <= max_time
                ts, raw_da, raw_iso, lp_da, lp_iso, hp_da, hp_iso, fit_iso, raw_dff, z = (
                    ts[m], raw_da[m], raw_iso[m], lp_da[m], lp_iso[m],
                    hp_da[m], hp_iso[m], fit_iso[m], raw_dff[m], z[m]
                )

            steps = [
                ((raw_da,  raw_iso),  ("raw DA",  "raw ISOS"),    "1) Raw"),
                ((lp_da,   lp_iso),   ("LP DA",   "LP ISOS"),     "2) Low-pass (3 Hz)"),
                ((hp_da,   hp_iso),   ("DA",      "ISOS"),         "3) Double Exponential"),
                ((hp_da,   fit_iso),  ("DA",      "fitted ISOS"),  "4) IRLS fit"),
                ((raw_dff,),          ("raw ΔF/F",),             "5) dF/F"),
                ((z,),                ("z-score",),               "6) z-score"),
            ]

            fig, axes = plt.subplots(len(steps), 1,
                                    figsize=(12, 2.5*len(steps)),
                                    sharex=True)

            for ax, ((sig1, *rest), (lbl1, *rest_lbls), title) in zip(axes, steps):
                ax.plot(ts, sig1, lw=1.2, label=lbl1)
                if rest:
                    ax.plot(ts, rest[0], lw=1.2, label=rest_lbls[0])
                ax.set_ylabel(title, fontsize=10)
                ax.legend(frameon=False, fontsize="small")
            
            xticks = range(0, int(ts.max()) + 50, 50)
            ax.set_xticks(xticks)

            axes[-1].set_xlabel("Time (s)", fontsize=12)
            fig.suptitle(f"{trial_folder} preprocessing steps", fontsize=14, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()


        def preprocessing_plotted_Urs(self, max_time=None):
            for trial_folder, tr in self.trials.items():
                print(f"\n=== Processing {trial_folder} ===")

                # 0) prepare: downsample + trim artifacts
                tr.downsample(target_fs=100)
                tr.remove_initial_LED_artifact(t=30)
                tr.remove_final_data_segment(t=10)

                # raw streams
                ts      = tr.timestamps
                raw_da  = tr.streams['DA']
                raw_iso = tr.streams['ISOS']

                # optional time cutoff
                if max_time is not None:
                    m = ts <= max_time
                    ts, raw_da, raw_iso = ts[m], raw_da[m], raw_iso[m]

                # 1) Raw
                da_raw, iso_raw = raw_da, raw_iso

                # 2) Linear detrend (polyfit) on [time → signal] for each channel
                m_da, b_da   = np.polyfit(ts, da_raw, 1)
                baseline_da  = m_da * ts + b_da
                da_detrended = da_raw - baseline_da + np.mean(baseline_da)

                m_iso, b_iso    = np.polyfit(ts, iso_raw, 1)
                baseline_iso    = m_iso * ts + b_iso
                iso_detrended   = iso_raw - baseline_iso + np.mean(baseline_iso)

                # 3) compute %ΔF/F from detrended traces
                #    NOTE: your compute_dFF() expects tr.isosbestic_fitted,
                #          so here we just do it inline:
                dF      = da_detrended - iso_detrended
                dFF     = 100 * (dF / iso_detrended)

                # 4) z-score
                zscore  = (dFF - np.nanmean(dFF)) / np.nanstd(dFF)
                tr.zscore = zscore

                # assemble plotting steps
                steps = [
                    ((da_raw,      iso_raw),    ("raw DA",    "raw ISOS"),      "1) Raw"),
                    ((da_detrended, iso_detrended), ("DA det.", "ISOS det."),   "2) Linear detrend"),
                    ((dFF,),       ("raw %ΔF/F",),   "3) %ΔF/F"),
                    ((zscore,),    ("z-score",),      "4) z-score"),
                ]

                fig, axes = plt.subplots(len(steps), 1,
                                        figsize=(12, 2.5*len(steps)),
                                        sharex=True)
                dual = {"1) Raw", "2) Linear detrend"}

                for ax, ((sig1, *rest), (lbl1, *lbl_rest), title) in zip(axes, steps):
                    if title in dual:
                        ax1, ax2 = ax, ax.twinx()
                        ax1.plot(ts, sig1,   color='tab:blue',  lw=1.2, label=lbl1)
                        ax2.plot(ts, rest[0], color='tab:orange', lw=1.2, label=lbl_rest[0])
                        ax1.spines['left'].set_position(('outward', 10))
                        ax2.spines['right'].set_position(('outward', 40))
                        ax1.set_ylabel(f"{title}\nDA",   color='tab:blue',   labelpad=8)
                        ax2.set_ylabel(f"{title}\nISOS", color='tab:orange', labelpad=8)
                        # combined legend
                        h1, l1 = ax1.get_legend_handles_labels()
                        h2, l2 = ax2.get_legend_handles_labels()
                        ax1.legend(h1+h2, l1+l2, loc='upper right', fontsize='small')
                    else:
                        ax.plot(ts, sig1, lw=1.2, label=lbl1)
                        if rest:
                            ax.plot(ts, rest[0], lw=1.2, label=lbl_rest[0])
                        ax.set_ylabel(title, fontsize=10)
                        ax.legend(frameon=False, fontsize='small')

                axes[-1].set_xlabel("Time (s)", fontsize=12)
                fig.suptitle(f"{tr.subject_name} preprocessing steps", fontsize=14, y=0.99)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()


    def preprocessing_plotted_hp(self, max_time=None):
        for trial_folder, tr in self.trials.items():
            print(f"\n=== Processing {trial_folder} ===")
            
            # Downsample
            tr.downsample(target_fs = 100)

            # 1) trim LED/artifact
            tr.remove_initial_LED_artifact(t=30)
            tr.remove_final_data_segment(t=10)


            # grab raw
            raw_da  = tr.streams['DA']
            raw_iso = tr.streams['ISOS']

            # 2) low-pass
            tr.lowpass_filter(cutoff_hz=3.0)
            lp_da   = tr.updated_DA.copy()
            lp_iso  = tr.updated_ISOS.copy()

            # 3) high-pass recentered
            tr.baseline_drift_highpass_recentered(cutoff=0.001)
            # tr.basline_drift_double_exponential()
            hp_da   = tr.updated_DA.copy()
            hp_iso  = tr.updated_ISOS.copy()

            # 4) IRLS fit
            tr.motion_correction_align_channels_IRLS(IRLS_constant=1.4)
            fit_iso = tr.isosbestic_fitted.copy()

            # 5) compute dF/F
            tr.compute_dFF()
            raw_dff = tr.dFF.copy()

            # 6) z-score
            tr.compute_zscore(method='standard')
            z       = tr.zscore.copy()

            # optional time mask
            ts = tr.timestamps
            if max_time is not None:
                m = ts <= max_time
                ts, raw_da, raw_iso, lp_da, lp_iso, hp_da, hp_iso, fit_iso, raw_dff, z = (
                    ts[m], raw_da[m], raw_iso[m], lp_da[m], lp_iso[m],
                    hp_da[m], hp_iso[m], fit_iso[m], raw_dff[m], z[m]
                )

            steps = [
                ((raw_da,  raw_iso),  ("raw DA",  "raw ISOS"),    "1) Raw"),
                ((lp_da,   lp_iso),   ("LP DA",   "LP ISOS"),     "2) Low-pass (3 Hz)"),
                ((hp_da,   hp_iso),   ("HP DA",   "HP ISOS"),     "3) High Pass (0.001 Hz)"),
                ((hp_da,   fit_iso),  ("DA",      "fitted ISOS"),  "4) IRLS fit"),
                ((raw_dff,),          ("raw ΔF/F",),             "5) dF/F"),
                ((z,),                ("z-score",),               "6) z-score"),
            ]

            fig, axes = plt.subplots(len(steps), 1,
                                    figsize=(12, 2.5*len(steps)),
                                    sharex=True)

            dual_axes_titles = {"1) Raw", "2) Low-pass (3 Hz)", "3) High Pass (0.001 Hz)"}

            for ax, ((sig1, *rest), (lbl1, *rest_lbls), title) in zip(axes, steps):
                if title in dual_axes_titles:
                    # split axes for DA vs ISOS
                    ax1 = ax
                    ax2 = ax1.twinx()
                    ax1.plot(ts, sig1,  color='tab:blue',  lw=1.2, label=lbl1)
                    ax2.plot(ts, rest[0], color='tab:orange', lw=1.2, label=rest_lbls[0])
                    ax2.spines['right'].set_position(('outward', 10))
                    ax1.set_ylabel(f"{title}\nDA",   color='tab:blue')
                    ax2.set_ylabel(f"{title}\nISOS", color='tab:orange')
                    # combined legend on the first axis
                    l1, t1 = ax1.get_legend_handles_labels()
                    l2, t2 = ax2.get_legend_handles_labels()
                    ax1.legend(l1+l2, t1+t2, loc='upper right', fontsize='small')

                else:
                    # single axis (IRLS fit, dF/F, z-score)
                    ax.plot(ts, sig1, lw=1.2, label=lbl1)
                    if rest:
                        ax.plot(ts, rest[0], lw=1.2, label=rest_lbls[0])
                    ax.set_ylabel(title, fontsize=10)
                    ax.legend(frameon=False, fontsize='small')

            axes[-1].set_xlabel("Time (s)", fontsize=12)
            fig.suptitle(f"{tr.subject_name} preprocessing steps", fontsize=14, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()


    def preprocessing_plotted_linreg(self, max_time=None):
        for trial_folder, tr in self.trials.items():
            print(f"\n=== Processing {trial_folder} ===")

            # 1) trim LED/artifact
            tr.remove_initial_LED_artifact(t=10)
            tr.remove_final_data_segment(t=10)

            # 2) smooth (in‐place)
            # tr.smooth_and_apply(window_len_seconds=4)
            raw_da   = tr.streams['DA']
            raw_iso  = tr.streams['ISOS']

            # 3) low‐pass
            tr.lowpass_filter(cutoff_hz=3.0)
            lp_da    = tr.updated_DA.copy()
            lp_iso   = tr.updated_ISOS.copy()

            # 4) high‐pass recentered
            # tr.highpass_baseline_drift_Recentered(cutoff=0.001)
            # hp_da    = tr.updated_DA.copy()
            # hp_iso   = tr.updated_ISOS.copy()

            # 5) IRLS fit
            # tr.align_channels_IRLS(IRLS_constant=3)
            tr.align_channels_linReg()
            fit_iso  = tr.isosbestic_fitted.copy()

            # 6) compute dF/F
            tr.compute_dFF()
            raw_dff  = tr.dFF.copy()

            # 6.5 baseline drift
            tr.highpass_baseline_drift_dFF(cutoff=0.001)
            hp_dff = tr.dFF.copy()

            # 7) zscore
            tr.compute_zscore(method='standard')
            z        = tr.zscore.copy()

            # Apply time masking if max_time is set
            ts = tr.timestamps
            if max_time is not None:
                mask = ts <= max_time
                ts = ts[mask]
                raw_da   = raw_da[mask]
                raw_iso  = raw_iso[mask]
                lp_da    = lp_da[mask]
                lp_iso   = lp_iso[mask]
                hp_da    = hp_da[mask]
                hp_iso   = hp_iso[mask]
                fit_iso  = fit_iso[mask]
                raw_dff  = raw_dff[mask]
                z        = z[mask]

            # now plot them all in one figure:
            steps = [
                ([raw_da, raw_iso], ["raw DA", "raw ISOS"], "1) Raw"),
                ([lp_da, lp_iso], ["LP DA", "LP ISOS"], "2) Low-pass"),
                # ([hp_da, hp_iso], ["HP DA", "HP ISOS"], "3) High-pass recentered"),
                ([lp_da, fit_iso], ["DA", "fitted ISOS"], "4) lin reg fit"),
                ([raw_dff], ["raw ΔF/F"], "5) dF/F"),
                ([hp_dff], ["raw ΔF/F"], "5) High passed dF/F"),
                ([z], ["z-score"], "6) z-score"),
            ]

            fig, axes = plt.subplots(len(steps), 1,
                                    figsize=(12, 2.5 * len(steps)),
                                    sharex=True)
            for ax, (sigs, labs, title) in zip(axes, steps):
                for sig, lab in zip(sigs, labs):
                    ax.plot(ts, sig, lw=1.2, label=lab)
                ax.set_ylabel(title, fontsize=10)
                ax.legend(frameon=False, fontsize="small")
            axes[-1].set_xlabel("Time (s)", fontsize=12)
            fig.suptitle(f"{tr.subject_name} preprocessing steps", fontsize=14, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    def preprocessing_plotted_old_pipeline(self, time_segments_to_remove=None):
        for trial_folder, tr in self.trials.items():
            print(f"\n=== Processing {trial_folder} ===")

            # 1) trim LED/artifact
            tr.remove_initial_LED_artifact(t=30)
            tr.remove_final_data_segment(t=10)


            raw_da   = tr.streams['DA']
            raw_iso  = tr.streams['ISOS']

            # 2) smooth (in‐place)
            tr.smooth_and_apply(window_len=int(tr.fs)*4)
            lp_da = tr.updated_DA.copy()
            lp_iso = tr.updated_ISOS.copy()

        
            # 4) MA baseline drift
            tr.apply_ma_baseline_drift()
            hp_da    = tr.updated_DA.copy()
            hp_iso   = tr.updated_ISOS.copy()

            # 5) lin regression fit
            tr.align_channels()
            fit_iso  = tr.isosbestic_fitted.copy()

            # 6) compute dF/F
            tr.compute_dFF()
            raw_dff  = tr.dFF.copy()

            # 7) zscore
            tr.compute_zscore(method='standard')
            z        = tr.zscore.copy()

            # now plot them all in one figure:
            steps = [
                ([raw_da,    raw_iso], ["raw DA",    "raw ISOS"],      "1) Raw"),
                ([lp_da,     lp_iso],  ["LP DA",     "LP ISOS"],       "2) smoothing"),
                ([hp_da,     hp_iso],  ["HP DA",     "HP ISOS"],       "3) Moving Average (Drift)"),
                ([hp_da,     fit_iso], ["DA",        "fitted ISOS"],    "4) Lin Reg fit"),
                ([raw_dff],           ["raw ΔF/F"],         "5) dF/F"),
                ([z],                 ["z-score"],           "6) z-score"),
            ]

            fig, axes = plt.subplots(len(steps), 1,
                                     figsize=(12, 2.5*len(steps)),
                                     sharex=True)
            for ax, (sigs, labs, title) in zip(axes, steps):
                for sig, lab in zip(sigs, labs):
                    ax.plot(tr.timestamps, sig, lw=1.2, label=lab)
                ax.set_ylabel(title, fontsize=10)
                ax.legend(frameon=False, fontsize="small")
            axes[-1].set_xlabel("Time (s)", fontsize=12)
            fig.suptitle(f"{tr.subject_name} preprocessing steps", fontsize=14, y=0.99)
            plt.tight_layout(rect=[0,0,1,0.96])
            plt.show()
    def plot_psd_and_drift(self,
                           channel='DA',
                           drift_cutoff=0.1,
                           zoom_cutoff=0.1,
                           nperseg_seconds=4):
        """
        1) Overlays the PSD of every trial (log–log),
        2) Zooms into the low-frequency band to show drift,
        3) Prints % power < drift_cutoff,
        4) Estimates the knee frequency via curvature.
        """
        plt.figure(figsize=(8,6))
        for trial_name, tr in self.trials.items():
            f, Pxx = tr.compute_psd(channel=channel,
                                    nperseg_seconds=nperseg_seconds)
            plt.loglog(f, Pxx, alpha=0.5, label=trial_name)
        plt.title(f"All trials PSD — {channel}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.legend(ncol=2, fontsize='small', loc='upper right')
        plt.tight_layout()
        plt.show()

        # Now zoom into the drift band
        plt.figure(figsize=(8,4))
        for trial_name, tr in self.trials.items():
            f, Pxx = tr.compute_psd(channel=channel,
                                    nperseg_seconds=nperseg_seconds)
            mask = f <= zoom_cutoff
            plt.loglog(f[mask], Pxx[mask], alpha=0.8, label=trial_name)
        plt.title(f"Drift band (<{zoom_cutoff} Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.tight_layout()
        plt.show()

        # Quantify percent power below drift_cutoff
        print(f"\nPower fraction below {drift_cutoff} Hz:")
        for trial_name, tr in self.trials.items():
            f, Pxx = tr.compute_psd(channel=channel,
                                    nperseg_seconds=nperseg_seconds)
            total_power = np.trapz(Pxx, f)
            drift_power = np.trapz(Pxx[f < drift_cutoff], f[f < drift_cutoff])
            frac = 100 * drift_power / total_power
            print(f"  {trial_name}: {frac:.1f}%")

        # Estimate knee frequency via curvature on log–log PSD (first trial only)
        # (just to give you an idea of where the spectrum bends)
        tr0 = next(iter(self.trials.values()))
        f0, P0 = tr0.compute_psd(channel=channel,
                                 nperseg_seconds=nperseg_seconds)
        # skip the zero-frequency bin
        lf = np.log10(f0[1:])
        lP = np.log10(P0[1:])
        # second difference (approx. curvature) on log–log curve
        curvature = np.diff(lP, n=2)
        knee_idx = np.argmax(curvature) + 1
        knee_freq = 10**lf[knee_idx]
        print(f"\nEstimated knee (drift→1/f transition) at ≈ {knee_freq:.3f} Hz")


    def preprocessing_plotted_keevers(self, time_segments_to_remove=None):
        for trial_folder, tr in self.trials.items():
            print(f"\n=== Processing {trial_folder} ===")

             # 1) trim LED/artifact
            tr.remove_initial_LED_artifact(t=120)
            tr.remove_final_data_segment(t=10)

            # 2) smooth (in‐place)
            tr.smooth_and_apply(window_len_seconds=2)
            raw_da   = tr.streams['DA']
            raw_iso  = tr.streams['ISOS']

            # 3) low‐pass
            tr.lowpass_filter(cutoff_hz=3.0)
            lp_da    = tr.updated_DA.copy()
            lp_iso   = tr.updated_ISOS.copy()


            # 5) IRLS fit
            tr.align_channels_IRLS(IRLS_constant=1.4)
            fit_iso  = tr.isosbestic_fitted.copy()
            hp_da    = tr.updated_DA.copy()

            # 6) compute dF/F
            tr.compute_dFF()
            raw_dff  = tr.dFF.copy()

            # 4) high‐pass dF/F
            tr.highpass_baseline_drift_dFF(cutoff=0.001)
            raw_dff  = tr.dFF.copy()

            # 7) zscore
            tr.compute_zscore(method='standard')
            z        = tr.zscore.copy()

            # now plot them all in one figure:
            steps = [
                ([raw_da,    raw_iso], ["raw DA",    "raw ISOS"],      "1) Raw"),
                ([lp_da,     lp_iso],  ["LP DA",     "LP ISOS"],       "2) Low-pass and smoothing"),
                ([hp_da,     fit_iso], ["DA",        "fitted ISOS"],    "4) IRLS fit"),
                ([raw_dff],           ["raw ΔF/F"],         "5) dF/F"),
                ([raw_dff],  ["ΔF/F"],       "3) High-pass"),
                ([z],                 ["z-score"],           "6) z-score"),
            ]

            fig, axes = plt.subplots(len(steps), 1,
                                     figsize=(12, 2.5*len(steps)),
                                     sharex=True)
            for ax, (sigs, labs, title) in zip(axes, steps):
                for sig, lab in zip(sigs, labs):
                    ax.plot(tr.timestamps, sig, lw=1.2, label=lab)
                ax.set_ylabel(title, fontsize=10)
                ax.legend(frameon=False, fontsize="small")
            axes[-1].set_xlabel("Time (s)", fontsize=12)
            fig.suptitle(f"{tr.subject_name} preprocessing steps", fontsize=14, y=0.99)
            plt.tight_layout(rect=[0,0,1,0.96])
            plt.show()
    

    def group_extract_manual_annotations(self, bout_definitions, first_only=True):
        """
        Extracts behavior bouts and annotations for all trials in the experiment.

        This function:
        1. Iterates through self.trials, looking for behavior CSV files in self.behavior_folder_path.
        2. Calls extract_bouts_and_behaviors for each trial.
        3. Stores the behavior data inside each Trial object.

        Parameters:
        - bout_definitions (list of dict): List defining each bout with:
            - 'prefix': Label used for the bout (e.g., "s1", "s2", "x").
            - 'introduced': Name of the behavior marking the start of the bout.
            - 'removed': Name of the behavior marking the end of the bout.
        - first_only (bool): If True, only the first event in each bout is kept;
                            if False, all events within each bout are retained.
        """
        for trial_name, trial in self.trials.items():
            csv_path = os.path.join(self.behavior_folder_path, f"{trial_name}.csv")
            if os.path.exists(csv_path):
                print(f"Processing behaviors for {trial_name}...")
                trial.extract_bouts_and_behaviors(csv_path, bout_definitions, first_only=first_only)
                trial.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=1)
                # Optionally, you can remove short behaviors:
                # trial.remove_short_behaviors(behavior_name='all', min_duration=0.3)
            else:
                print(f"Warning: No CSV found for {trial_name} in {self.behavior_folder_path}. Skipping.")


    '''********************************** PLOTTING **********************************'''
    def plot_all_traces(experiment, behavior_name='all'):
        """
        Plots behavior events for all trials with all subplots showing x-tick labels.
        """
        num_trials = len(experiment.trials)
        if num_trials == 0:
            print("No trials found in the experiment.")
            return

        fig, axes = plt.subplots(nrows=num_trials, figsize=(12, 3 * num_trials), sharex=False)

        if num_trials == 1:
            axes = [axes]

        for ax, (trial_name, trial) in zip(axes, experiment.trials.items()):
            if trial.behaviors is None or trial.behaviors.empty:
                ax.set_title(f"{trial_name} (no behavior data)")
                ax.axis("off")
                continue
            trial.plot_behavior_event(behavior_name, ax=ax)
            ax.set_title(trial_name)
            ax.tick_params(axis='x', labelbottom=True)  # Ensure x-tick labels are shown

        plt.tight_layout()
        plt.show()



    def plot_first_behavior_PETHs(self, selected_bouts=None, behavior="Investigation"):
        """
        Plots the first PETHs for the specified behavior for all trials in the experiment.
        
        For each trial, the method:
        - Filters rows where 'Behavior' equals the specified behavior.
        - (Optionally) Further filters the data to only include bouts listed in 'selected_bouts'.
        - Groups the data by 'Bout' and selects the first event in each bout.
        - Plots the behavior trace (Relative_Time_Axis vs. Relative_Zscore) with:
            • a dashed black line at x = 0 (Behavior Start),
            • a dashed blue line at x = Duration (s) (Behavior End),
            • a dashed red line at x = Time of Max Peak.
            
        The y-axis limits are determined dynamically based on the global minimum and maximum values 
        across all plotted bouts (with an extra margin of 1 added to each end). Each subplot displays 
        its own y-axis tick numbers.
        
        Parameters:
        selected_bouts (list, optional): A list of bout identifiers to include. If None, all bouts are plotted.
        behavior (str, optional): The behavior to plot. Defaults to "Investigation".
        """
        trial_first_data = []  # list to store each trial's first behavior DataFrame
        trial_names = []       # list to track trial names
        max_bouts = 0          # maximum number of bouts across trials

        # Loop over each trial and extract first-behavior events.
        for trial_name, trial in self.trials.items():
            if not hasattr(trial, 'behaviors'):
                continue
            # Filter for the specified behavior events.
            df_behavior = trial.behaviors[trial.behaviors["Behavior"] == behavior].copy()
            
            # If a selection of bouts is provided, filter to include only those bouts.
            if selected_bouts is not None:
                df_behavior = df_behavior[df_behavior["Bout"].isin(selected_bouts)]
            
            # Group by 'Bout' and take the first event in each group.
            df_first_behavior = df_behavior.groupby("Bout", as_index=False).first()
            trial_first_data.append(df_first_behavior)
            trial_names.append(trial_name)
            if len(df_first_behavior) > max_bouts:
                max_bouts = len(df_first_behavior)

        if len(trial_first_data) == 0:
            print("No trial data available for plotting first behavior PETHs.")
            return

        # Determine global y-axis limits from all Relative_Zscore data.
        global_min = np.inf
        global_max = -np.inf
        for df_first in trial_first_data:
            for _, row in df_first.iterrows():
                y_data = row["Relative_Zscore"]
                current_min = np.min(y_data)
                current_max = np.max(y_data)
                if current_min < global_min:
                    global_min = current_min
                if current_max > global_max:
                    global_max = current_max

        ymin = global_min - 1
        ymax = global_max + 1

        n_rows = len(trial_first_data)
        n_cols = max_bouts

        # Create a grid of subplots without sharing y axes so each shows its own y-axis numbers.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        # Ensure axes is a 2D array for consistent indexing.
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        # Loop over each trial (row) and each bout (column) to plot.
        for row_idx, (df_first, trial_name) in enumerate(zip(trial_first_data, trial_names)):
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                # Only plot if data for this bout exists.
                if col_idx < len(df_first):
                    data_row = df_first.iloc[col_idx]

                    # Extract time and Z-score arrays.
                    x = data_row["Relative_Time_Axis"]
                    y = data_row["Relative_Zscore"]

                    # Plot the behavior trace.
                    ax.plot(x, y, label=f"Bout: {data_row['Bout']}")
                    # Plot vertical dashed lines:
                    # Start (x = 0)
                    ax.axvline(x=0, color='black', linestyle='--', label="Start")
                    # End (x = Duration (s))
                    ax.axvline(x=data_row["Duration (s)"], color='blue', linestyle='--', label="End")
                    # Time of Max Peak (x = Time of Max Peak)
                    ax.axvline(x=data_row["Time of Max Peak"], color='red', linestyle='--', label="Max Peak")
                    
                    # Set y-axis limits based on computed global min and max.
                    ax.set_ylim([ymin, ymax])
                    ax.set_xlabel("Relative Time (s)")
                    ax.set_title(f"Trial {trial_name} - Bout {data_row['Bout']}")

                    # Ensure y-axis tick labels are visible on every subplot.
                    ax.tick_params(axis='y', labelleft=True)
                    # Add y-label and legend for the first column.
                    if col_idx == 0:
                        ax.set_ylabel("Z-score")
                        ax.legend()
                else:
                    ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_average_investigation_PETHs(
        self,
        n_subsequent_investigations=3,   # Number of subsequent investigation events to include
        behavior="Investigation",        # Behavior to filter for (e.g. "Investigation")
        plot_error_bars=True             # If True, plot SEM error bars
    ):
        """
        Computes and plots the average PETH trace for each subsequent investigation event
        across all trials/mice.

        Steps:
        1. For each trial, filter for events with the specified behavior.
        2. Sort events (e.g., by 'Event_Start' if available) and assign an InvestigationIndex.
        3. For each investigation index (up to n_subsequent_investigations), collect the PETH trace 
        (Relative_Zscore) and record the Relative_Time_Axis.
        4. Across all trials, compute the element-wise average and standard error (SEM) of the PETH traces.
        5. Plot the average trace for each investigation event with error bars if desired.

        Parameters:
        n_subsequent_investigations (int): Number of investigation events to include (e.g., first 3).
        behavior (str): Behavior label to filter events (default "Investigation").
        plot_error_bars (bool): Whether to overlay SEM error bars.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Dictionary to hold lists of PETH traces for each investigation index.
        peth_dict = {i: [] for i in range(1, n_subsequent_investigations + 1)}
        time_axis = None  # We assume all events share the same time axis.
        
        # Loop over trials (or mice) stored in self.trials.
        for trial_name, trial in self.trials.items():
            if not hasattr(trial, 'behaviors'):
                continue
            # Filter for the specified behavior.
            df_behavior = trial.behaviors[trial.behaviors["Behavior"] == behavior].copy()
            if df_behavior.empty:
                continue
            
            # Sort events by time; if "Event_Start" exists, use it.
            if "Event_Start" in df_behavior.columns:
                df_behavior.sort_values("Event_Start", inplace=True)
            else:
                df_behavior.sort_index(inplace=True)
            
            # Assign an InvestigationIndex (1 for the first event, 2 for the second, etc.)
            df_behavior["InvestigationIndex"] = np.arange(1, len(df_behavior) + 1)
            
            # Only consider events up to the specified number.
            df_behavior = df_behavior[df_behavior["InvestigationIndex"] <= n_subsequent_investigations]
            
            # Loop through each event and store its PETH trace.
            for _, row in df_behavior.iterrows():
                idx = int(row["InvestigationIndex"])
                # Each row should have a PETH trace (Relative_Zscore) and a time axis.
                # We assume these are numpy arrays (or lists) that are the same length for all events.
                peth_trace = np.array(row["Relative_Zscore"])
                peth_dict[idx].append(peth_trace)
                
                # Store the time axis (assumed common across events) from the first valid event.
                if time_axis is None:
                    time_axis = np.array(row["Relative_Time_Axis"])
        
        # Compute the average trace and SEM for each investigation index.
        avg_traces = {}
        sem_traces = {}
        for idx in range(1, n_subsequent_investigations + 1):
            traces = peth_dict[idx]
            if len(traces) > 0:
                stacked = np.vstack(traces)
                avg_traces[idx] = np.mean(stacked, axis=0)
                sem_traces[idx] = np.std(stacked, axis=0) / np.sqrt(stacked.shape[0])
            else:
                print(f"No data available for investigation {idx}.")
        
        # Plot the average traces.
        plt.figure(figsize=(12, 8))
        for idx in sorted(avg_traces.keys()):
            if plot_error_bars:
                plt.errorbar(
                    time_axis, 
                    avg_traces[idx], 
                    yerr=sem_traces[idx],
                    marker='o', linestyle='-',
                    capsize=5, label=f"Investigation {idx}"
                )
            else:
                plt.plot(time_axis, avg_traces[idx], marker='o', linestyle='-', label=f"Investigation {idx}")
        
        plt.xlabel("Relative Time (s)", fontsize=14)
        plt.ylabel("Average Z-score", fontsize=14)
        plt.title(f"Average PETH for First {n_subsequent_investigations} Investigations", fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        
    def plot_clean_single_PETH_for_poster(
        self,
        trial_name,
        bout_name,
        behavior="Investigation"
    ):
        import matplotlib.pyplot as plt
        import numpy as np

        trial = self.trials.get(trial_name, None)
        if trial is None or not hasattr(trial, 'behaviors'):
            print(f"Trial '{trial_name}' not found or missing behavior data.")
            return

        df = trial.behaviors.copy()
        df = df[(df["Behavior"] == behavior) & (df["Bout"] == bout_name)]

        if df.empty:
            print(f"No matching behavior '{behavior}' found for bout '{bout_name}' in trial '{trial_name}'.")
            return

        row = df.iloc[0]
        x = np.array(row["Relative_Time_Axis"])
        y = np.array(row["Relative_Zscore"])

        # Find index of peak
        peak_idx = np.argmax(y)
        peak_x = x[peak_idx]
        peak_y = y[peak_idx]

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, color="#15616F", linewidth=2)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)  # Start
        plt.axvline(x=row["Duration (s)"], color='black', linestyle='--', linewidth=1.5)  # End
        plt.scatter(peak_x, peak_y, color='red', zorder=5, s=60)  # Peak dot

        # Style adjustments
        plt.xlabel("Relative Time (s)", fontsize=12)
        plt.ylabel("Z-score", fontsize=12)
        plt.xlim([-4, 10])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_behavior_and_bouts(
        self,
        bout_bounds_df,
        behavior_name="all",
        start_time=30.0
    ):
        """
        Plots each trial's z-score trace overlaid with:
        - gray spans for each behavior in trial.behaviors (or only `behavior_name`)
        - dashed red lines at each Bout_Start_s and Bout_End_s from `bout_bounds_df`

        Parameters:
        - bout_bounds_df (pd.DataFrame): Output from get_bout_boundaries_df()
        - behavior_name (str): Name of behavior to span; "all" plots all behaviors
        - start_time (float): Minimum x-axis value for each subplot
        """
        n = len(self.trials)
        if n == 0:
            print("No trials to plot.")
            return

        fig, axes = plt.subplots(nrows=n, figsize=(12, 3*n), sharex=False)
        if n == 1:
            axes = [axes]

        for ax, (trial_name, trial) in zip(axes, self.trials.items()):
            ax.set_title(trial_name)
            ts = trial.timestamps
            ax.plot(ts, trial.zscore, color="black", lw=1)

            # 1) Plot behavior spans
            dfb = trial.behaviors
            if dfb is not None and not dfb.empty:
                if behavior_name != "all":
                    dfb = dfb[dfb["Behavior"] == behavior_name]
                for _, row in dfb.iterrows():
                    ax.axvspan(row["Event_Start"], row["Event_End"], color="gray", alpha=0.3)

            # 2) Plot dashed red lines from bout_bounds_df
            trial_subject = trial.subject_name
            trial_bounds = bout_bounds_df[bout_bounds_df["Subject"] == trial_subject]
            for _, row in trial_bounds.iterrows():
                ax.axvline(row["Bout_Start_s"], color="red", linestyle="--", lw=1.5)
                ax.axvline(row["Bout_End_s"], color="red", linestyle="--", lw=1.5)

            ax.set_xlim(start_time, ts[-1])
            ax.tick_params(axis="x", labelbottom=True)

        plt.tight_layout()
        plt.show()




    def get_bout_boundaries_df(self, bout_definitions):
        """
        Extracts bout start and end times from raw CSVs using bout_definitions and returns
        a unified DataFrame with columns: ['Subject', 'Bout', 'Bout_Start_s', 'Bout_End_s'].

        Parameters:
        - bout_definitions (list of dict): Each dict must have keys: 'prefix', 'introduced', 'removed'

        Returns:
        - pd.DataFrame containing bout timing information across all trials
        """
        all_rows = []

        for trial_name, trial in self.trials.items():
            csv_path = os.path.join(self.behavior_folder_path, f"{trial_name}.csv")
            if not os.path.exists(csv_path):
                print(f"Warning: No CSV found for {trial_name}. Skipping.")
                continue

            raw_df = pd.read_csv(csv_path)
            subject = trial.subject_name

            for bd in bout_definitions:
                prefix = bd['prefix']
                intro_df = raw_df[raw_df['Behavior'] == bd['introduced']]
                remove_df = raw_df[raw_df['Behavior'] == bd['removed']]

                # Sort and zip to pair up
                intro_df = intro_df.sort_values('Start (s)').reset_index(drop=True)
                remove_df = remove_df.sort_values('Start (s)').reset_index(drop=True)
                num_bouts = min(len(intro_df), len(remove_df))

                for idx in range(num_bouts):
                    irow = intro_df.iloc[idx]
                    rrow = remove_df.iloc[idx]
                    all_rows.append({
                        'Subject': subject,
                        'Bout': f"{prefix}-{idx+1}",
                        'Bout_Start_s': irow['Start (s)'],
                        'Bout_End_s': rrow['Start (s)']
                    })

        return pd.DataFrame(all_rows)


    '''********************************** DOPAMINE SHIZ **********************************'''
    def compute_all_da_metrics(self, use_max_length=False, max_bout_duration=10, mode='standard', post_time=15):
        """
        Iterates over all trials in the experiment and computes DA metrics with the specified windowing options.
        
        For each trial, computes AUC, Max Peak, Time of Max Peak, Mean Z-score, and Adjusted End for each behavior.
        If a behavior lasts less than 1 second, the window is extended beyond the bout end to search for the next peak.
        
        Parameters:
        - use_max_length (bool): Whether to limit the window to a maximum duration.
        - max_bout_duration (int): Maximum allowed window duration (in seconds).
        - mode (str): Either 'standard' to compute metrics using absolute timestamps and full z-score data,
                    or 'EI' to compute metrics using event-aligned relative data.
        """
        for trial_name, trial in self.trials.items():
            if hasattr(trial, 'compute_da_metrics'):
                print(f"Computing DA metrics for {trial_name} ...")
                trial.compute_da_metrics(
                    use_max_length=use_max_length,
                    max_bout_duration=max_bout_duration,
                    mode=mode,
                    post_time=post_time
                )
            else:
                print(f"Warning: Trial '{trial_name}' does not have compute_da_metrics method.")




    def compute_all_event_induced_DA(self, pre_time=4, post_time=15):
        """
        Iterates over all trials in the experiment and computes the event-induced DA signals
        for each trial by calling each Trial's compute_event_induced_DA() method.
        
        Parameters:
        - pre_time (float): Seconds to include before event onset.
        - post_time (float): Seconds to include after event onset.
        """
        for trial_name, trial in self.trials.items():
            print(f"Computing event-induced DA for trial {trial_name} ...")
            trial.compute_event_induced_DA(pre_time=pre_time, post_time=post_time)



    '''********************************** MISC **********************************'''
    def reset_all_behaviors(self):
        """
        Sets each trial's 'behaviors' DataFrame to empty, so you can re-run
        group_extract_manual_annotations with different parameters.
        """
        for trial in self.trials.values():
            trial.behaviors = pd.DataFrame()


    def compute_all_event_induced_DA(self, pre_time=4, post_time=10):
        """
        Iterates over all trials in the experiment and computes the event-induced DA signal 
        for each trial by calling each trial's compute_event_induced_DA() method.
        
        Parameters:
            pre_time (float): Seconds before event onset to include (default is 4 s).
            post_time (float): Seconds after event onset to include (default is 10 s).
        """
        for trial_name, trial in self.trials.items():
            print(f"Computing event-induced DA for trial {trial_name} ...")
            trial.compute_event_induced_DA(pre_time=pre_time, post_time=post_time)



    '''********************************** mPFC Peak Detection **********************************'''
    