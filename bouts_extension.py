# This is the general code used for any experiments that contain multiple bouts (Introduction and Removal of mouse mutltiple times)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, linregress
from sklearn.linear_model import LinearRegression
from trial_class import *
from itertools import combinations
import seaborn as sns


from scipy.optimize import curve_fit
# Behavior ---------------------------------------------------------------------------------------
# Behavior Processing
def get_trial_dataframes(experiment):
    """
    Given an Experiment object, return a dictionary where:
    - Keys are subject IDs (Trial.subject_name).
    - Values are DataFrames corresponding to the behaviors of each trial.
    """
    trial_data = {}

    for trial in experiment.trials.values():
        subject_id = trial.subject_name  # Extract subject ID
        trial_data[subject_id] = trial.behaviors  # Store behaviors DataFrame

    return trial_data


def create_metadata_dataframe(trial_data, behavior="Investigation", desired_bouts=None):
    """
    Parameters
    ----------
    trial_data : dict
        Dictionary of { subject_id : DataFrame }, 
        where each DataFrame has columns: 
        [Bout, Behavior, Event_Start, Event_End, Duration (s)], etc.
    
    behavior : str, optional
        The behavior type to filter for (default = "Investigation").

    desired_bouts : list or None, optional
        A list of bout labels to keep. If None, all bouts present in the subject's DataFrame are retained.

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame with columns:
        [Subject, Bout, Behavior, Total Investigation Time, Average Bout Duration]
        For each subject and bout in the specified list (or all bouts if None),
        if the subject never exhibits the specified behavior, a row is included with 
        Total Investigation Time and Average Bout Duration set to 0.
    """
    
    metadata_rows = []

    # Loop through each subject and its corresponding DataFrame
    for subject_id, df in trial_data.items():
        # Determine which bouts to include: use the desired list if provided,
        # otherwise use all unique bout labels from the DataFrame.
        if desired_bouts is not None:
            bouts = desired_bouts
        else:
            bouts = df["Bout"].unique()

        # Process each bout for the current subject.
        for bout in bouts:
            # Filter the subject's DataFrame for the current bout.
            df_bout = df[df["Bout"] == bout]
            # Then, filter for the specified behavior.
            df_behavior = df_bout[df_bout["Behavior"] == behavior]
            
            if df_behavior.empty:
                # If no investigation events are present in this bout, set metrics to 0.
                total_investigation_time = 0
                average_bout_duration = 0
            else:
                # Compute total investigation time and average bout duration.
                total_investigation_time = df_behavior["Duration (s)"].sum()
                count = df_behavior["Duration (s)"].count()
                average_bout_duration = total_investigation_time / count if count > 0 else 0
            
            metadata_rows.append({
                "Subject": subject_id,
                "Bout": bout,
                "Behavior": behavior,
                "Total Investigation Time": total_investigation_time,
                "Average Bout Duration": average_bout_duration
            })

    # Concatenate all rows into a single DataFrame.
    final_df = pd.DataFrame(metadata_rows)
    
    return final_df


def create_da_metrics_dataframe(trial_data, behavior="Investigation", desired_bouts=None):
    """
    Parameters
    ----------
    trial_data : dict
        Dictionary of { subject_id : DataFrame }, 
        where each DataFrame has columns: 
        [Bout, Behavior, AUC, Max Peak, Mean Z-score, etc.]

    behavior : str, optional
        The behavior type to filter for (default = "Investigation").

    desired_bouts : list or None, optional
        A list of bout labels to keep. If None, all bouts present in the subject's DataFrame are retained.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        [Subject, Bout, Behavior, AUC, Max Peak, Mean Z-score].
        - If multiple events occur in the same bout, the metrics are averaged.
        - If the bout has no rows for the specified behavior, all metrics are set to 0.
    """
    metric_rows = []

    # Loop over each subject and its corresponding DataFrame
    for subject_id, df in trial_data.items():
        
        # Determine which bouts to include
        if desired_bouts is not None:
            bouts = desired_bouts
        else:
            bouts = df["Bout"].unique()

        # Process each bout for the current subject
        for bout in bouts:
            # Filter the subject's DataFrame for the current bout
            df_bout = df[df["Bout"] == bout]
            # Further filter by the specified behavior
            df_behavior = df_bout[df_bout["Behavior"] == behavior]
            
            if df_behavior.empty:
                # No events of this behavior => metrics are 0
                auc_val = 0
                max_peak_val = 0
                mean_z_val = 0
            else:
                # If there are multiple rows, we average these metrics
                auc_val = df_behavior["AUC"].mean()
                max_peak_val = df_behavior["Max Peak"].mean()
                mean_z_val = df_behavior["Mean Z-score"].mean()

            metric_rows.append({
                "Subject": subject_id,
                "Bout": bout,
                "Behavior": behavior,
                "AUC": auc_val,
                "Max Peak": max_peak_val,
                "Mean Z-score": mean_z_val
            })

    # Concatenate all rows into a single DataFrame
    final_df = pd.DataFrame(metric_rows)
    
    return final_df


def create_da_metrics_first_instance(
    trial_data,
    behavior="Investigation",
    desired_bouts=None,
    time_col="Event_Start"
):
    """
    Like create_da_metrics_dataframe, but for each (Subject, Bout) we only
    pull the **first** row where Behavior==behavior (sorted by time_col).

    Parameters
    ----------
    trial_data : dict
        { subject_id : behaviors-DataFrame }
    behavior : str
        Which behavior to select (default "Investigation")
    desired_bouts : list or None
        If provided, only these bout labels are processed; else all in df.
    time_col : str
        Which column to sort by to determine "first" (default "Event_Start")

    Returns
    -------
    pd.DataFrame
        Columns: [Subject, Bout, Behavior, AUC, Max Peak, Mean Z-score]
    """
    rows = []

    for subj, df in trial_data.items():
        # pick which bouts to iterate
        if desired_bouts is not None:
            bouts = desired_bouts
        else:
            bouts = df["Bout"].unique()

        for bout in bouts:
            df_b = df[df["Bout"] == bout]
            # of those, take only rows matching the behavior
            df_beh = df_b[df_b["Behavior"] == behavior].copy()

            if df_beh.empty:
                # no instance → zeros
                auc_val = 0.0
                peak_val = 0.0
                z_val   = 0.0
            else:
                # sort by the time_col and grab the first row
                first = df_beh.sort_values(time_col, ascending=True).iloc[0]
                auc_val  = first["AUC"]
                peak_val = first["Max Peak"]
                z_val    = first["Mean Z-score"]

            rows.append({
                "Subject": subj,
                "Bout": bout,
                "Behavior": behavior,
                "AUC": auc_val,
                "Max Peak": peak_val,
                "Mean Z-score": z_val
            })

    return pd.DataFrame(rows)



def plot_behavior_across_bouts_no_identities(metadata_df,
                                          y_col="Total Investigation Time",
                                          behavior=None,
                                          title='Mean Across Bouts',
                                          ylabel=None,
                                          custom_xtick_labels=None,
                                          custom_xtick_colors=None,
                                          ylim=None,
                                          bar_color='#00B7D7',
                                          yticks_increment=None,
                                          xlabel='Agent',
                                          figsize=(12, 7),
                                          pad_inches=0.1,
                                          save=False,
                                          save_name=None):


    # 1) Optionally filter by behavior
    if behavior is not None:
        metadata_df = metadata_df[metadata_df["Behavior"] == behavior].copy()
        if metadata_df.empty:
            raise ValueError(f"No data found for behavior='{behavior}'.")

    # 2) Check if the desired y_col exists
    if y_col not in metadata_df.columns:
        raise ValueError(f"'{y_col}' not found in metadata_df columns.")

    # 3) Pivot the DataFrame: rows -> Subjects, columns -> Bout, values -> y_col
    pivot_df = metadata_df.pivot(index="Subject", columns="Bout", values=y_col)

    # 4) Calculate mean and SEM across subjects for each bout
    mean_values = pivot_df.mean()
    sem_values = pivot_df.sem()

    # 5) Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # 6) Bar plot with error bars (SEM)
    bars = ax.bar(
        pivot_df.columns,
        mean_values,
        yerr=sem_values,
        capsize=6,
        color=bar_color,
        edgecolor='black',
        linewidth=5,
        width=0.6,
        error_kw=dict(elinewidth=3, capthick=3, zorder=5)
    )

    # 7) Plot individual subject data in gray
    for subject in pivot_df.index:
        ax.plot(pivot_df.columns, pivot_df.loc[subject],
                linestyle='-', color='gray', alpha=0.5,
                linewidth=2.5, zorder=1)
        ax.scatter(pivot_df.columns, pivot_df.loc[subject],
                   facecolors='none', edgecolors='gray',
                   s=120, alpha=0.6, linewidth=4, zorder=2)

    # 8) Labels
    if ylabel is None:
        ylabel = y_col
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # 9) X-ticks
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    if custom_xtick_labels:
        ax.set_xticklabels(custom_xtick_labels, fontsize=28)
        if custom_xtick_colors:
            for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick.set_color(color)
    else:
        ax.set_xticklabels(pivot_df.columns, fontsize=26)

    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)

    # 10) Y-limits
    if ylim is None:
        all_vals = np.concatenate([pivot_df.values.flatten(), mean_values.values.flatten()])
        min_val = np.nanmin(all_vals)
        max_val = np.nanmax(all_vals)
        lower_ylim = 0 if min_val > 0 else min_val * 1.1
        upper_ylim = max_val * 1.1
        ax.set_ylim(lower_ylim, upper_ylim)
        if lower_ylim < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)
    else:
        ax.set_ylim(ylim)
        if ylim[0] < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)

    # 11) Custom Y-ticks
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # 12) Spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    plt.show()

    # 13) Paired t-tests
    print("\nPaired t-test results (all pairwise combinations):")
    bouts = pivot_df.columns.tolist()
    for a, b in combinations(bouts, 2):
        if a in pivot_df.columns and b in pivot_df.columns:
            paired = pivot_df[[a, b]].dropna()
            if len(paired) > 1:
                t_stat, p_val = ttest_rel(paired[a], paired[b])
                stars = "ns"
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                print(f"{a} vs {b}: p = {p_val:.4f} ({stars})")
            else:
                print(f"{a} vs {b}: Not enough data.")
        else:
            print(f"{a} vs {b}: Bout not found in data.")


    print("\nPaired t-test results New:")
    bouts = pivot_df.columns.tolist()
    for a, b in combinations(bouts, 2):
        if a in pivot_df.columns and b in pivot_df.columns:
            paired = pivot_df[[a, b]].dropna()
            if len(paired) > 1:
                t_stat, p_val = ttest_rel(paired[a], paired[b])

                # Effect size calculation (Cohen's d for paired samples)
                diff = paired[a] - paired[b]
                mean_diff = np.mean(diff)
                std_diff = np.std(diff, ddof=1)  # Sample standard deviation
                cohen_d = mean_diff / std_diff if std_diff != 0 else np.nan

                # Stars for significance
                stars = "ns"
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"

                print(f"{a} vs {b}: p = {p_val:.4f} ({stars}), d = {cohen_d:.3f}")
            else:
                print(f"{a} vs {b}: Not enough data.")
        else:
            print(f"{a} vs {b}: Bout not found in data.")


def plot_behavior_across_bouts_with_identities(metadata_df,
                                             y_col="Total Investigation Time",
                                             behavior=None,
                                             title='Mean Across Bouts',
                                             ylabel=None,
                                             custom_xtick_labels=None,
                                             custom_xtick_colors=None,
                                             ylim=None,
                                             bar_color='#00B7D7',
                                             yticks_increment=None,
                                             xlabel='Agent',
                                             figsize=(12,7),
                                             pad_inches=0.1,
                                             save=False,
                                             save_name=None):
    """
    Plots a bar chart with error bars (SEM) and individual subject lines in **color** (instead of gray),
    and provides a legend mapping subjects to their respective colors.
    """

    # 1) Optionally filter by behavior
    if behavior is not None:
        metadata_df = metadata_df[metadata_df["Behavior"] == behavior].copy()
        if metadata_df.empty:
            raise ValueError(f"No data found for behavior='{behavior}'.")

    # 2) Check if the desired y_col exists
    if y_col not in metadata_df.columns:
        raise ValueError(f"'{y_col}' not found in metadata_df columns.")

    # 3) Pivot the DataFrame: rows -> Subjects, columns -> Bout, values -> y_col
    pivot_df = metadata_df.pivot(index="Subject", columns="Bout", values=y_col)

    # 4) Generate unique colors for each subject
    subjects = pivot_df.index
    colors = sns.color_palette("husl", n_colors=len(subjects))  # Generate distinct colors
    subject_color_map = dict(zip(subjects, colors))  # Map each subject to a color

    # 5) Calculate mean and SEM across subjects for each bout
    mean_values = pivot_df.mean()
    sem_values = pivot_df.sem()

    # 6) Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # 7) Bar plot with error bars (SEM)
    bars = ax.bar(
        pivot_df.columns, 
        mean_values, 
        yerr=sem_values, 
        capsize=6,
        color=bar_color,
        edgecolor='black', 
        linewidth=5,
        width=0.6,
        error_kw=dict(elinewidth=3, capthick=3, zorder=5)
    )

    # 8) Plot each subject's data in **color** rather than gray
    for subject in pivot_df.index:
        ax.plot(pivot_df.columns, pivot_df.loc[subject],
                linestyle='-', color=subject_color_map[subject], alpha=0.8,
                linewidth=2.5, zorder=1, label=subject)
        ax.scatter(pivot_df.columns, pivot_df.loc[subject],
                   facecolors='none', edgecolors=subject_color_map[subject],
                   s=120, alpha=0.9, linewidth=4, zorder=2)

    # 9) Set axis labels and title
    if ylabel is None:
        ylabel = y_col
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # 10) Set x-ticks and labels
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    if custom_xtick_labels is not None:
        ax.set_xticklabels(custom_xtick_labels, fontsize=28)
        if custom_xtick_colors is not None:
            for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick.set_color(color)
    else:
        ax.set_xticklabels(pivot_df.columns, fontsize=26)

    # Increase tick label sizes
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)

    # 11) Set y-axis limits
    if ylim is None:
        all_values = np.concatenate([pivot_df.values.flatten(), mean_values.values.flatten()])
        min_val = np.nanmin(all_values)
        max_val = np.nanmax(all_values)
        lower_ylim = 0 if min_val > 0 else min_val * 1.1
        upper_ylim = max_val * 1.1
        ax.set_ylim(lower_ylim, upper_ylim)
        if lower_ylim < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)
    else:
        ax.set_ylim(ylim)
        if ylim[0] < 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)

    # 12) Set y-ticks if an increment is provided
    if yticks_increment is not None:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # 13) Remove right & top spines; thicken left & bottom spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # 14) Add legend for subjects
    ax.legend(title="Subjects", fontsize=18, title_fontsize=20, loc='upper right', bbox_to_anchor=(1.2, 1))

    # 15) Adjust layout and save the figure if requested
    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    
    plt.show()


def plot_peak_for_subsequent_behaviors(
    exp_da_dict,
    selected_bouts=None,
    behavior=None,
    n_subsequent_behaviors=3,
    peak_col="Max Peak",
    metric_type='slope',
    figsize=(14, 8),
    line_order=None,
    custom_colors=None,
    custom_legend_labels=None,
    custom_xtick_labels=None,
    ylim=None,
    ytick_increment=None,
    xlabel="Behavior Index",
    ylabel="Average Peak ΔF/F",
    plot_title="Average Peak per Behavior",
    save=False,
    save_path="peaks_for_subsequent_behaviors.png"
):

    # === Embedded helper: merges all subject DataFrames ===
    def create_big_df_from_exp_da_dict(exp_da_dict):
        return pd.concat(
            [df.assign(Subject=subj) for subj, df in exp_da_dict.items() if not df.empty],
            ignore_index=True
        )

    # === Exponential decay model ===
    def exponential_decay(x, A, B, tau):
        return A + B * np.exp(-x / tau)

    # === Build combined DataFrame ===
    big_df = create_big_df_from_exp_da_dict(exp_da_dict)

    if selected_bouts is not None:
        big_df = big_df[big_df["Bout"].isin(selected_bouts)].copy()

    if behavior is not None:
        big_df = big_df[big_df["Behavior"] == behavior].copy()

    if big_df.empty:
        print("No data left after filtering for bouts and behavior. Nothing to plot.")
        return pd.DataFrame()

    big_df.sort_values(["Subject", "Bout", "Event_Start"], inplace=True)
    big_df["BehaviorIndex"] = big_df.groupby(["Subject", "Bout"]).cumcount() + 1

    # Keep all groups, truncate after n_subsequent_behaviors
    big_df = big_df[big_df["BehaviorIndex"] <= n_subsequent_behaviors]

    agg_df = (
        big_df.groupby(["Bout", "BehaviorIndex"], as_index=False)
        .agg(
            SubjectCount=("Subject", "nunique"),
            AvgPeak=(peak_col, "mean"),
            StdPeak=(peak_col, "std")
        )
    )
    agg_df["SEM"] = agg_df["StdPeak"] / np.sqrt(agg_df["SubjectCount"])

    if custom_colors is None:
        custom_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(5)
    ax.spines["bottom"].set_linewidth(5)
    ax.tick_params(axis="both", which="major", labelsize=48)

    metrics_dict = {}

    unique_bouts = line_order if line_order else sorted(agg_df["Bout"].unique())

    for i, bout in enumerate(unique_bouts):
        df_line = agg_df[agg_df["Bout"] == bout].copy()
        df_line.sort_values("BehaviorIndex", inplace=True)

        x_vals = df_line["BehaviorIndex"].values
        y_vals = df_line["AvgPeak"].values
        y_err = df_line["SEM"].values

        if len(x_vals) == 0 or len(y_vals) == 0:
            print(f"Skipping bout '{bout}' due to no data.")
            continue

        # Fit metric
        if metric_type.lower() == 'slope':
            slope, intercept, r_val, p_val, std_err = linregress(x_vals, y_vals)
            metrics_dict[bout] = slope
            metric_label = f"slope: {slope:.3f}"
        elif metric_type.lower() == 'decay':
            p0 = (np.min(y_vals), np.max(y_vals) - np.min(y_vals), 1.0)
            try:
                popt, _ = curve_fit(exponential_decay, x_vals, y_vals, p0=p0)
                tau = popt[2]
                metrics_dict[bout] = tau
                metric_label = f"decay: {tau:.3f}"
            except RuntimeError:
                metrics_dict[bout] = np.nan
                metric_label = "decay: N/A"
                print(f"Warning: exponential fit failed for bout '{bout}'.")
        else:
            raise ValueError("metric_type must be 'slope' or 'decay'.")

        color = custom_colors[i % len(custom_colors)]
        subject_n = df_line["SubjectCount"].max()
        legend_label = (
            custom_legend_labels[i] if custom_legend_labels and i < len(custom_legend_labels)
            else f"{bout} ({metric_label}, n={subject_n})"
        )

        ax.errorbar(
            x_vals, y_vals,
            yerr=y_err,
            marker='o', linestyle='-',
            color=color,
            linewidth=5, markersize=30,
            capsize=10,
            elinewidth=8,
            capthick=8,
            label=legend_label
        )

    ax.set_xlabel(xlabel, fontsize=48, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=48, labelpad=12)

    if ylim is not None:
        ax.set_ylim(ylim)
        if ytick_increment is not None:
            y_ticks = np.arange(ylim[0], ylim[1] + ytick_increment, ytick_increment)
            ax.set_yticks(y_ticks)
            y_tick_labels = [f"{int(yt)}" if float(yt).is_integer() else f"{yt:.1f}" for yt in y_ticks]
            ax.set_yticklabels(y_tick_labels, fontsize=48)

    if custom_xtick_labels:
        ax.set_xticks(np.arange(1, len(custom_xtick_labels) + 1))
        ax.set_xticklabels(custom_xtick_labels, fontsize=48)
    else:
        unique_x = sorted(agg_df["BehaviorIndex"].unique())
        ax.set_xticks(unique_x)
        ax.set_xticklabels([str(x) for x in unique_x], fontsize=48)

    if plot_title:
        ax.set_title(plot_title, fontsize=20)

    ax.legend(fontsize=26)
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, transparent=True, dpi=300)

    plt.show()

    print(f"\n=== Computed Metric ({metric_type.upper()}): ===")
    for bout, val in metrics_dict.items():
        print(f"Bout: {bout}, {metric_type} = {val:.3f}")

    return agg_df

