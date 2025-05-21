import numpy as np
import matplotlib.pyplot as plt
import os


def plot_average_defeat_bout_psth(
    experiment,
    nth_defeat=1,
    save_path=None,
    save=False,
    bin_size=100,
    y_min=-5,
    y_max=3,
    brain_region=None
):
    """
    Standalone function to average and plot event-induced DA signal for the nth defeat bout across trials.

    Parameters:
        experiment (Experiment): The experiment object containing trials.
        nth_defeat (int): The 1-based index of the defeat bout to select (chronologically).
        save_path (str or None): Full file path (including filename and .png extension) to save the plot.
        save (bool): Whether to save the figure to disk.
        bin_size (int): Bin size for downsampling the averaged signal.
        y_min (float): Lower y-axis bound.
        y_max (float): Upper y-axis bound.
        brain_region (str): Hex color code for trace color. If None, uses default.
    
    Returns:
        dict or None: Contains time axis, mean trace, SEM, trial count, and subject IDs.
    """
    

    trace_color = brain_region if brain_region is not None else '#FFAF00'
    all_traces = []
    subject_ids = []
    missing_subjects = []

    for trial in experiment.trials.values():
        if trial.behaviors is None or trial.behaviors.empty:
            missing_subjects.append(trial.subject_name)
            continue

        df_defeat = trial.behaviors[trial.behaviors["Behavior"] == "Defeat"].copy()
        if df_defeat.empty:
            missing_subjects.append(trial.subject_name)
            continue

        if "Event_Start" in df_defeat.columns:
            df_defeat.sort_values(by="Event_Start", inplace=True)
        else:
            df_defeat.sort_index(inplace=True)

        if len(df_defeat) < nth_defeat:
            missing_subjects.append(trial.subject_name)
            continue

        row = df_defeat.iloc[nth_defeat - 1]
        if "Relative_Zscore" not in row or row["Relative_Zscore"] is None:
            missing_subjects.append(trial.subject_name)
            continue

        trace = np.array(row["Relative_Zscore"])
        all_traces.append(trace)
        subject_ids.append(trial.subject_name)

    if missing_subjects:
        print(f"Subjects missing {nth_defeat}-th defeat bout: {', '.join(missing_subjects)}")

    if len(all_traces) == 0:
        print(f"No valid data found for the {nth_defeat}-th defeat bout.")
        return None

    common_time_axis = np.array(row["Relative_Time_Axis"])
    traces_stacked = np.vstack(all_traces)
    mean_trace = np.mean(traces_stacked, axis=0)
    sem_trace = np.std(traces_stacked, axis=0) / np.sqrt(traces_stacked.shape[0])

    if hasattr(experiment, 'downsample_data'):
        mean_trace, downsampled_time_axis = experiment.downsample_data(mean_trace, common_time_axis, bin_size)
        sem_trace, _ = experiment.downsample_data(sem_trace, common_time_axis, bin_size)
    else:
        downsampled_time_axis = common_time_axis

    plt.figure(figsize=(14, 7))
    plt.plot(downsampled_time_axis, mean_trace, color=trace_color, lw=3, label='Mean Defeat DA')
    plt.fill_between(downsampled_time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                     color=trace_color, alpha=0.4, label='SEM')
    plt.axvline(0, color='black', linestyle='--', lw=2)
    plt.xlabel('Time from Defeat Onset (s)', fontsize=36)
    plt.ylabel('Event-Induced z-scored ΔF/F', fontsize=32, labelpad=20)
    plt.ylim(y_min, y_max)
    plt.xlim(-4, 10)

    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=40, width=3)
    plt.xticks(fontsize=44)
    plt.yticks(fontsize=44)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    plt.tight_layout()
    plt.subplots_adjust(left=0.35)

    if save:
        if save_path is None:
            raise ValueError("save_path must be specified when save=True.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.5)

    plt.show()

    return {
        "common_time_axis": downsampled_time_axis,
        "mean_trace": mean_trace,
        "sem_trace": sem_trace,
        "n_trials": len(all_traces),
        "subject_ids": subject_ids
    }
