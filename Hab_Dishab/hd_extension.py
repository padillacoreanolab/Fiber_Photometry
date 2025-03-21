import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from trial_class import *
import seaborn as sns


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




def plot_behavior_times_across_bouts_gray(metadata_df,
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
    Plots a bar chart with error bars (SEM) and individual subject lines in gray,
    based on a metadata DataFrame containing columns:
      [Subject, Bout, Behavior, Total Investigation Time, Average Bout Duration].
    
    Parameters:
        - metadata_df (pd.DataFrame): DataFrame with columns:
          [Subject, Bout, Behavior, Total Investigation Time, Average Bout Duration].
        - y_col (str): Which column to plot on the y-axis.
                       (e.g., "Total Investigation Time" or "Average Bout Duration")
        - behavior (str or None): If provided, filters the DataFrame to only rows with that behavior.
        - title (str): The title for the plot.
        - ylabel (str or None): Label for the y-axis. If None, defaults to y_col.
        - custom_xtick_labels (list or None): Custom x-tick labels; if not provided, uses the bout names.
        - custom_xtick_colors (list or None): A list of colors for the x-tick labels.
        - ylim (tuple or None): (min, max) for y-axis. If None, determined automatically.
        - bar_color (str): Color for the bars.
        - yticks_increment (float or None): Increment for y-axis ticks.
        - xlabel (str): Label for the x-axis.
        - figsize (tuple): Figure size.
        - pad_inches (float): Padding around the figure when saving.
        - save (bool): If True, saves the image to disk.
        - save_name (str or None): Full file path (with filename and extension) where the image should be saved.
                                   Required if save is True.
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

    # 7) Plot individual subject data in gray (lines and unfilled circles)
    for subject in pivot_df.index:
        ax.plot(pivot_df.columns, pivot_df.loc[subject],
                linestyle='-', color='gray', alpha=0.5,
                linewidth=2.5, zorder=1)
        ax.scatter(pivot_df.columns, pivot_df.loc[subject],
                   facecolors='none', edgecolors='gray',
                   s=120, alpha=0.6, linewidth=4, zorder=2)

    # 8) Set axis labels and title
    if ylabel is None:
        ylabel = y_col
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # 9) Set x-ticks and labels
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

    # 10) Set y-axis limits
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

    # 11) Set y-ticks if an increment is provided
    if yticks_increment is not None:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # 12) Remove right & top spines; thicken left & bottom spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # 13) Adjust layout, and save the figure if requested
    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    
    plt.show()




def plot_behavior_times_across_bouts_colored(metadata_df,
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

















# DA ---------------------------------------------------------------------------------------
def plot_all_da_metrics(self, 
                        metric_name="DA_peak", 
                        title="DA Metrics Across Trials", 
                        ylabel="DA Value", 
                        xlabel="Bout",
                        figsize=(12,7)):
    """
    Plots the specified DA metric across all trials.
    
    Assumes that after compute_all_da_metrics has been called, each trial in 
    self.trials that computed DA metrics stores its results in a DataFrame 
    attribute called 'da_metrics' with at least the following columns:
        - Bout: Bout label (e.g., "s1-1", "s2-1", etc.)
        - A column corresponding to the metric of interest (e.g., "DA_peak").
    
    Parameters:
        - metric_name (str): The name of the DA metric column to plot (default is "DA_peak").
        - title (str): The title for the plot.
        - ylabel (str): The label for the y-axis.
        - xlabel (str): The label for the x-axis.
        - figsize (tuple): The size of the figure.
    
    Returns:
        None. Displays a plot with one line per trial.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    all_metrics = []
    for trial_name, trial in self.trials.items():
        if hasattr(trial, "da_metrics"):
            # Assume trial.da_metrics is a DataFrame that contains columns "Bout" and metric_name.
            df = trial.da_metrics.copy()
            df["Trial"] = trial_name
            all_metrics.append(df)
        else:
            print(f"Trial '{trial_name}' does not have computed DA metrics.")
    
    if not all_metrics:
        print("No DA metrics available to plot.")
        return

    # Concatenate all trial metrics into one DataFrame.
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    # Pivot the DataFrame so that rows are Trials, columns are Bout labels, and values are the metric.
    pivot_df = metrics_df.pivot(index="Trial", columns="Bout", values=metric_name)
    pivot_df = pivot_df.fillna(0)  # Fill missing values with 0 (or you could choose to interpolate)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each trial as a separate line with markers.
    for trial in pivot_df.index:
        ax.plot(pivot_df.columns, pivot_df.loc[trial], marker="o", label=trial)
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.legend(title="Trial", fontsize=10, title_fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Dopamine Paired t-test Calculation. Use this to manually see the p-values
def perform_t_tests_on_bouts(experiment, metric_name):
    """
    Performs paired t-tests:
      - Between bout 1 and bout 2
      - Between bout 5 and bout 6

    Parameters:
        - experiment: The experiment object containing trial data.
        - metric_name (str): The DA metric to analyze (default: "Mean Z-score").

    Returns:
        - A dictionary containing t-statistics and p-values for both comparisons.
    """

    # Collect per-trial data grouped by Bout for the specified metric
    trial_data = []
    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, "behaviors") and not trial.behaviors.empty:
            df = trial.behaviors.copy()
            if metric_name not in df.columns:
                print(f"Warning: Trial '{trial_name}' does not contain metric '{metric_name}'. Skipping.")
                continue
            df_grouped = df.groupby("Bout", as_index=False)[metric_name].mean()
            df_grouped["Trial"] = trial_name
            trial_data.append(df_grouped)
        else:
            print(f"Warning: Trial '{trial_name}' has no behavior data.")

    if not trial_data:
        print("No data available for t-tests.")
        return None

    # Combine all trial data into a single DataFrame.
    combined_df = pd.concat(trial_data, ignore_index=True)

    # Ensure we have exactly 6 bouts: bout 1, bout 2, bout 5, bout 6
    selected_bouts = combined_df["Bout"].unique()
    if len(selected_bouts) < 6:
        print("Error: Not enough bouts available for t-tests.")
        return None

    # Extract data for the specific bouts
    bout_1_data = combined_df[combined_df["Bout"] == selected_bouts[0]][metric_name].values
    bout_2_data = combined_df[combined_df["Bout"] == selected_bouts[1]][metric_name].values
    bout_5_data = combined_df[combined_df["Bout"] == selected_bouts[4]][metric_name].values
    bout_6_data = combined_df[combined_df["Bout"] == selected_bouts[5]][metric_name].values

    # Ensure equal sample sizes (required for paired t-test)
    min_length_1 = min(len(bout_1_data), len(bout_2_data))
    min_length_2 = min(len(bout_5_data), len(bout_6_data))

    bout_1_data, bout_2_data = bout_1_data[:min_length_1], bout_2_data[:min_length_1]
    bout_5_data, bout_6_data = bout_5_data[:min_length_2], bout_6_data[:min_length_2]

    # Perform paired t-tests
    t_stat_1, p_value_1 = ttest_rel(bout_1_data, bout_2_data)
    t_stat_2, p_value_2 = ttest_rel(bout_5_data, bout_6_data)

    # Output results
    print(f'T-test between bout 1 and bout 2: t-statistic = {t_stat_1:.4f}, p-value = {p_value_1:.4f}')
    print(f'T-test between bout 5 and bout 6: t-statistic = {t_stat_2:.4f}, p-value = {p_value_2:.4f}')

    return {
        "t_stat_1": t_stat_1, "p_value_1": p_value_1,
        "t_stat_2": t_stat_2, "p_value_2": p_value_2
    }

# DA plotting. Includes significance markers.'
def plot_da_metrics_combined_oneplot_integrated(experiment, 
                                                metric_name="Mean Z-score", 
                                                title="Combined DA Metrics", 
                                                ylabel="DA Metric", 
                                                xlabel="Bout", 
                                                custom_xtick_labels=None, 
                                                custom_xtick_colors=None, 
                                                ylim=None, 
                                                bar_color="#00B7D7", 
                                                yticks_increment=None, 
                                                figsize=(14,8), 
                                                pad_inches=0.1):
    """
    Plots the computed DA metrics across 6 specific bouts for all trials in the experiment.
    If p-value < 0.05, it adds a horizontal significance line + asterisk above the bars.

    Updates:
    - Unfilled circle markers for individual trials
    - Thick grey outlines for visibility

    Parameters:
        - experiment: The experiment object (with a dictionary attribute `trials`).
        - metric_name (str): The DA metric to plot (e.g., "Mean Z-score").
        - title (str): The title for the plot.
        - ylabel (str): The label for the y-axis.
        - xlabel (str): The label for the x-axis.
        - custom_xtick_labels (list): A list of exactly 6 x-tick labels (default: ["i1", "i1", "i1", "i1", "i1", "i2"]).
        - custom_xtick_colors (list): A list of colors for the x-tick labels.
        - ylim (tuple): Y-axis limits.
        - bar_color (str): Color for bars.
        - yticks_increment (float): Increment for y-axis ticks.
        - figsize (tuple): Figure size.
        - pad_inches (float): Padding around the figure.
    """

    # Collect per-trial data (grouped by Bout for the chosen metric)
    trial_data = []
    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, "behaviors") and not trial.behaviors.empty:
            df = trial.behaviors.copy()
            if metric_name not in df.columns:
                print(f"Warning: Trial '{trial_name}' does not contain metric '{metric_name}'. Skipping.")
                continue
            df_grouped = df.groupby("Bout", as_index=False)[metric_name].mean()
            df_grouped["Trial"] = trial_name
            trial_data.append(df_grouped)
        else:
            print(f"Warning: Trial '{trial_name}' has no behavior data.")

    if not trial_data:
        print("No data available to plot.")
        return

    # Combine all trial data into a single DataFrame.
    combined_df = pd.concat(trial_data, ignore_index=True)

    # Select only 6 bouts: first 5 as "i1", last one as "i2"
    selected_bouts = combined_df["Bout"].unique()[:6]
    combined_df = combined_df[combined_df["Bout"].isin(selected_bouts)]

    # Pivot the data for the line plots: rows=Trial, columns=Bout, values=metric_name.
    try:
        pivot_df = combined_df.pivot(index="Trial", columns="Bout", values=metric_name)
    except Exception as e:
        print("Error pivoting data for line plots:", e)
        return
    pivot_df = pivot_df.fillna(0)

    # Compute overall average and SEM for each Bout across all trials.
    overall_stats = combined_df.groupby("Bout")[metric_name].agg(['mean', 'sem']).reset_index()

    # Create the figure and a single axis.
    fig, ax = plt.subplots(figsize=figsize, facecolor="none")  # Transparent background

    # Plot the overall average as a bar chart with error bars.
    ax.bar(overall_stats["Bout"], overall_stats["mean"], yerr=overall_stats["sem"],
           capsize=6, color=bar_color, edgecolor='black', linewidth=5, width=0.6,
           error_kw=dict(elinewidth=3, capthick=3, zorder=5))
    
    # Overlay individual trial lines (all in gray).
    for trial in pivot_df.index:
        ax.plot(pivot_df.columns, pivot_df.loc[trial], linestyle='-', color='gray', 
                alpha=0.5, linewidth=3, marker='o', markerfacecolor='none', markeredgecolor='gray', markeredgewidth=2, markersize=10)

    # Set labels and title.
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    if title is not None:
        ax.set_title(title, fontsize=28)

    # **Set exactly 6 x-tick labels**
    xtick_labels = ["i1", "i1", "i1", "i1", "i1", "i2"]
    xtick_colors = ["blue", "blue", "blue", "blue", "blue", "#E06928"]

    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(xtick_labels, fontsize=28)

    # Apply custom colors
    for tick, color in zip(ax.get_xticklabels(), xtick_colors):
        tick.set_color(color)

    # Increase tick label sizes.
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)
    
    # Set y-limits.
    if ylim is None:
        all_values = np.concatenate([pivot_df.values.flatten(), overall_stats["mean"].values])
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
    
    # Set y-ticks based on yticks_increment if provided.
    if yticks_increment is not None:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)
    
    # Remove right and top spines, adjust left and bottom spine width.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    
    # ---- Perform Paired T-tests ---- #
    def perform_t_tests_on_bouts(experiment, metric_name="Mean Z-score"):
        """Performs paired t-tests between Bout 1 & 2 and Bout 5 & 6."""
        trial_data = []
        for trial_name, trial in experiment.trials.items():
            if hasattr(trial, "behaviors") and not trial.behaviors.empty:
                df = trial.behaviors.copy()
                if metric_name not in df.columns:
                    continue
                df_grouped = df.groupby("Bout", as_index=False)[metric_name].mean()
                df_grouped["Trial"] = trial_name
                trial_data.append(df_grouped)

        if not trial_data:
            return None

        combined_df = pd.concat(trial_data, ignore_index=True)
        selected_bouts = combined_df["Bout"].unique()

        if len(selected_bouts) < 6:
            return None

        bout_1 = combined_df[combined_df["Bout"] == selected_bouts[0]][metric_name].values
        bout_2 = combined_df[combined_df["Bout"] == selected_bouts[1]][metric_name].values
        bout_5 = combined_df[combined_df["Bout"] == selected_bouts[4]][metric_name].values
        bout_6 = combined_df[combined_df["Bout"] == selected_bouts[5]][metric_name].values

        min_length_1 = min(len(bout_1), len(bout_2))
        min_length_2 = min(len(bout_5), len(bout_6))

        bout_1, bout_2 = bout_1[:min_length_1], bout_2[:min_length_1]
        bout_5, bout_6 = bout_5[:min_length_2], bout_6[:min_length_2]

        t_stat_1, p_value_1 = ttest_rel(bout_1, bout_2)
        t_stat_2, p_value_2 = ttest_rel(bout_5, bout_6)

        return {"t_stat_1": t_stat_1, "p_value_1": p_value_1, "t_stat_2": t_stat_2, "p_value_2": p_value_2}

    # Get t-test results
    t_test_results = perform_t_tests_on_bouts(experiment, metric_name)

    # ---- Plot significance markers ---- #
    if t_test_results:
        max_y = ax.get_ylim()[1]
        sig_y_offset = max_y * 0.05  # Offset above error bars

        if t_test_results["p_value_1"] < 0.05:
            x1, x2 = 0, 1  # Bout 1 and Bout 2 positions
            y = overall_stats["mean"].max() + sig_y_offset
            ax.plot([x1, x2], [y, y], color='black', linewidth=5)  # Horizontal line
            ax.text((x1 + x2) / 2, y + sig_y_offset / 2, "*", fontsize=40, ha='center', color='black')

        if t_test_results["p_value_2"] < 0.05:
            x1, x2 = 4, 5  # Bout 5 and Bout 6 positions
            y = overall_stats["mean"].max() + sig_y_offset
            ax.plot([x1, x2], [y, y], color='black', linewidth=5)  # Horizontal line
            ax.text((x1 + x2) / 2, y + sig_y_offset / 2, "*", fontsize=40, ha='center', color='black')

    plt.savefig(f'{title}{ylabel[0]}.png', transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    plt.tight_layout(pad=pad_inches)
    plt.show()

# DA plotting. Plots colored spaghetti plots corresponding to individual mouse ON TOP of average bar graphs. Use this to visualize mice.
def plot_da_metrics_color_oneplot(experiment, 
                                     metric_name="Mean Z-score", 
                                     title="Combined DA Metrics", 
                                     ylabel="DA Metric", 
                                     xlabel="Bout", 
                                     figsize=(14,8)):
    """
    Plots a combined figure with:
      - A bar chart showing the overall average of the specified DA metric for each bout.
      - Overlaid line plots for each trial (each in a unique color) showing the chosen metric across bouts.
    
    This function assumes that each trial in experiment.trials has its updated behaviors DataFrame 
    (with computed DA metrics) containing at least:
        - 'Bout': Bout label (e.g., "s1-1", "s2-1", etc.)
        - A column corresponding to the desired DA metric (e.g., "Mean Z-score").
    
    Parameters:
        - experiment: The experiment object (with a dictionary attribute `trials`).
        - metric_name (str): The DA metric to plot (e.g., "Mean Z-score").
        - title (str): The title for the plot.
        - ylabel (str): The y-axis label.
        - xlabel (str): The x-axis label.
        - figsize (tuple): The size of the figure.
    
    Returns:
        None. Displays a single plot that combines the overall average (bar chart)
        and the individual trial lines.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Collect per-trial data from updated behaviors.
    trial_data = []
    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, "behaviors") and not trial.behaviors.empty:
            df = trial.behaviors.copy()
            if metric_name not in df.columns:
                print(f"Warning: Trial '{trial_name}' does not have '{metric_name}'. Skipping.")
                continue
            # Group by Bout to get one numeric value per Bout for this trial.
            df_grouped = df.groupby("Bout", as_index=False)[metric_name].mean()
            df_grouped["Trial"] = trial_name
            trial_data.append(df_grouped)
        else:
            print(f"Warning: Trial '{trial_name}' has no behavior data.")
    
    if not trial_data:
        print("No data available to plot.")
        return
    
    # Combine data from all trials.
    combined_df = pd.concat(trial_data, ignore_index=True)
    
    # Pivot the combined DataFrame for the line plot: rows = Trial, columns = Bout.
    try:
        pivot_df = combined_df.pivot(index="Trial", columns="Bout", values=metric_name)
    except Exception as e:
        print("Error pivoting data:", e)
        return
    pivot_df = pivot_df.fillna(0)
    
    # Compute overall average for each Bout across all trials.
    overall_avg = combined_df.groupby("Bout", as_index=False)[metric_name].mean()
    
    # Create a single figure and axis.
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the overall average as a bar chart (with semi-transparent bars).
    ax.bar(overall_avg["Bout"], overall_avg[metric_name], color="skyblue", edgecolor="black", 
           alpha=0.5, label="Overall Average")
    
    # Plot each trial's data as a line plot.
    cmap = plt.cm.get_cmap("tab10", len(pivot_df.index))
    for i, trial in enumerate(pivot_df.index):
        ax.plot(pivot_df.columns, pivot_df.loc[trial], marker="o", color=cmap(i), label=trial)
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.legend(title="Trial", fontsize=10, title_fontsize=12)
    
    plt.tight_layout()
    plt.show()



# random stuff
# plot colors stuff
def hab_dishab_plot_y_across_bouts_colored(
    df, 
    title='Mean Across Bouts', 
    ylabel='Mean Value', 
    custom_xtick_labels=None, 
    custom_xtick_colors=None, 
    ylim=None, 
    bar_color='#00B7D7',
    yticks_increment=None, 
    xlabel='Intruder',
    figsize=(12,7), 
    pad_inches=1,
    cmap_name='tab10'
):
    """
    Plots the mean values (with SEM error bars) for each bout, plus each subject's data 
    in a unique color. The color map can be customized via `cmap_name` (e.g. 'tab10', 'tab20').
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate means and SEM across subjects
    means = df.mean(axis=0)
    sems = df.sem(axis=0)
    x = np.arange(len(df.columns))

    # Create a color map to get a unique color for each subject
    cmap = plt.cm.get_cmap(cmap_name, len(df.index))

    # -------------------------------------------------------
    # Plot each subject's data in a unique color + label
    # -------------------------------------------------------
    for i, idx in enumerate(df.index):
        subject_color = cmap(i)
        label = f"Subject {idx}"  # <--- ADDED: a label for the legend
        ax.plot(
            x, 
            df.loc[idx, :], 
            color=subject_color, 
            alpha=0.7, 
            marker='o', 
            linewidth=1,
            label=label             # <--- ADDED: pass the label here
        )

    # Bar chart for the means + error bars
    ax.bar(x, means, yerr=sems, color=bar_color, alpha=0.6, capsize=5)

    # Set x-axis ticks/labels
    if custom_xtick_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(custom_xtick_labels)
        if custom_xtick_colors is not None:
            for tick_label, c in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick_label.set_color(c)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(df.columns)

    # Labels and title
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Y-axis limit
    if ylim is not None:
        ax.set_ylim(ylim)

    # Custom y-ticks increment
    if yticks_increment is not None:
        start, end = ax.get_ylim()
        ax.set_yticks(np.arange(start, end + yticks_increment, yticks_increment))

    # -------------------------------------------------------
    # ADDED LEGEND: show the subject labels
    # -------------------------------------------------------
    # You can place the legend in many ways. For example, to the right:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # or simply:
    # ax.legend(loc='best')

    plt.tight_layout(pad=pad_inches)
    plt.show()

def plot_behavior_metric_colored(df, 
                                 behavior="Investigation", 
                                 metric="investigation_time",
                                 title='Mean Across Bouts',
                                 ylabel='Mean Value',
                                 ylim=None,
                                 cmap_name='tab10'):
    """
    Same idea as plot_behavior_metric, but calls the color-coded spaghetti plot.
    """
    # 1) Filter by Behavior
    filtered_df = df[df["Behavior"] == behavior].copy()

    # 2) Group & compute the chosen metric
    if metric == "investigation_time":
        grouped = filtered_df.groupby(["Subject", "Bout"], as_index=False)["Duration (s)"].sum()
        plot_label = "Investigation Time (Sum)"
    elif metric == "average_bout_duration":
        grouped = filtered_df.groupby(["Subject", "Bout"], as_index=False)["Duration (s)"].mean()
        plot_label = "Average Bout Duration (Mean)"
    else:
        raise ValueError("metric must be either 'investigation_time' or 'average_bout_duration'")

    # 3) Pivot so that each row = Subject, each column = Bout, values = numeric
    pivot_df = grouped.pivot(index="Subject", columns="Bout", values="Duration (s)")

    # 4) Call your color-coded function on the pivoted, numeric DataFrame
    hab_dishab_plot_y_across_bouts_colored(
        df=pivot_df,
        title=title,
        ylabel=ylabel if ylabel else plot_label,
        ylim=ylim,
        cmap_name=cmap_name
    )
