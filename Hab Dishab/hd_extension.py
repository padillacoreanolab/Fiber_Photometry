import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
from scipy.signal import butter, filtfilt
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from trial_class import *

import matplotlib.pyplot as plt
import re

# Behavior ---------------------------------------------------------------------------------------
# Behavior Processing
def get_trial_dataframes(experiment):
    """
    Given an Experiment object, return a list of DataFrames,
    where each DataFrame corresponds to the .behaviors of each trial.
    """
    # Extract all trial IDs from the experiment
    trial_ids = list(experiment.trials.keys())

    # Retrieve a DataFrame of behaviors for each trial
    trial_dataframes = [experiment.trials[tid].behaviors for tid in trial_ids]

    return trial_dataframes

def create_subject_summary_df(dfs):
    """
    Takes in a list of DataFrames (each CSV is one subject),
    and assigns a unique Subject ID (1 to N) to each DataFrame.
    
    For each subject:
      - Total Investigation Time = sum of "Duration (s)"
      - Average Bout Duration = total_investigation_time / number_of_bouts
    
    Returns a single DataFrame with columns:
      ['Bout', 'Subject', 'Behavior', 'Duration (s)',
       'Total Investigation Time', 'Average Bout Duration']
    """
    processed_list = []
    subject_id = 1
    
    for df in dfs:
        temp_df = df.copy()
        
        # Assign this entire CSV to one Subject
        temp_df["Subject"] = subject_id
        
        # Calculate sums and average for this subject
        total_invest_time = temp_df["Duration (s)"].sum()
        num_bouts = temp_df["Bout"].nunique()  # how many unique bouts in this CSV
        avg_bout_dur = total_invest_time / num_bouts if num_bouts else 0
        
        # Attach these values to every row
        temp_df["Total Investigation Time"] = total_invest_time
        temp_df["Average Bout Duration"] = avg_bout_dur
        
        processed_list.append(temp_df)
        subject_id += 1  # next CSV -> next Subject
    
    # Concatenate all into a single DataFrame
    final_df = pd.concat(processed_list, ignore_index=True)
    return final_df

def process_investigation_data(df, 
                               behavior_name='Investigation', 
                               gap_threshold=1.0, 
                               min_duration=0.5,
                               desired_bouts=None,
                               agg_func='sum'):
    """
    Merge consecutive Investigation events within 'gap_threshold' seconds,
    remove events shorter than 'min_duration', then group/pivot by Subject & Bout.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: [Subject, Behavior, Bout, Event_Start, Event_End, Duration (s)]
    behavior_name : str
        Which behavior to combine/filter (default 'Investigation').
    gap_threshold : float
        Max gap (in seconds) to consider consecutive events mergeable.
    min_duration : float
        Minimum duration below which events are removed.
    desired_bouts : list or None
        Which bouts to keep (if None, keep all).
    agg_func : {'sum', 'mean'}
        How to combine the durations in the final group step.
        
    Returns
    -------
    pivot_df : pd.DataFrame
        Pivoted DataFrame of aggregated durations by Subject × Bout.
    """
    
    #--- 1) Keep only rows matching the specified behavior ---
    df = df[df["Behavior"] == behavior_name].copy()
    
    #--- 2) Sort so consecutive events are truly consecutive by subject & start time ---
    df.sort_values(["Subject", "Event_Start"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    #--- 3) Identify which rows should *not* merge with their predecessor ---
    df["new_block"] = (
        (df["Subject"] != df["Subject"].shift(1)) |
        (df["Event_Start"] - df["Event_End"].shift(1) > gap_threshold)
    )
    df["group_id"] = df["new_block"].cumsum()
    
    #--- 4) Merge consecutive events in each group_id ---
    merged = (
        df.groupby("group_id", as_index=False)
          .agg({
              "Subject":      "first",
              "Behavior":     "first",
              "Bout":         "first",  # Uses the first bout label in the block
              "Event_Start":  "min",
              "Event_End":    "max",
              "Duration (s)": "sum"
          })
    )
    
    #--- 5) Remove events shorter than min_duration ---
    merged = merged[merged["Duration (s)"] >= min_duration].copy()
    
    #--- 6) Filter by desired bouts (if provided) ---
    if desired_bouts is not None:
        merged = merged[merged["Bout"].isin(desired_bouts)]
    
    #--- 7) Group by Subject & Bout with either sum or mean, then pivot ---
    if agg_func == 'sum':
        grouped_df = merged.groupby(["Subject", "Bout"], as_index=False)["Duration (s)"].sum()
    elif agg_func == 'mean':
        grouped_df = merged.groupby(["Subject", "Bout"], as_index=False)["Duration (s)"].mean()
    else:
        raise ValueError("agg_func must be either 'sum' or 'mean'")
    
    pivot_df = (
        grouped_df
        .pivot(index="Subject", columns="Bout", values="Duration (s)")
        .fillna(0)
    )
    
    return pivot_df


# Behavior Plotting
def plot_y_across_bouts_gray(df,  
                             title='Mean Across Bouts', 
                             ylabel='Mean Value', 
                             custom_xtick_labels=None, 
                             custom_xtick_colors=None, 
                             ylim=None, 
                             bar_color='#00B7D7',
                             yticks_increment=None, 
                             xlabel='Agent',
                             figsize=(12,7), 
                             pad_inches=0.1):
    """
    Plots the mean values during investigations or other events across bouts with error bars for SEM,
    and individual subject lines connecting the bouts. All subjects are plotted in gray.

    Parameters:
        - df (DataFrame): A DataFrame where rows are subjects, and bouts are columns.
                          Values should represent the mean values (e.g., mean DA, investigation times)
                          for each subject and bout.
        - title (str): The title for the plot.
        - ylabel (str): The label for the y-axis.
        - custom_xtick_labels (list): A list of custom x-tick labels. If not provided, defaults to the column names.
        - custom_xtick_colors (list): A list of colors for the x-tick labels. Must be the same length as custom_xtick_labels.
        - ylim (tuple): A tuple (min, max) to set the y-axis limits. If None, the limits are set automatically based on the data.
        - bar_color (str): The color to use for the bars (default is cyan).
        - yticks_increment (float): Increment amount for the y-axis ticks.
        - xlabel (str): The label for the x-axis.
        - figsize (tuple): The figure size.
        - pad_inches (float): Padding around the figure when saving.
    """
    # Calculate the mean and SEM for each bout (across all subjects)
    mean_values = df.mean()
    sem_values = df.sem()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the bar plot with error bars (mean and SEM) without adding it to the legend
    bars = ax.bar(
        df.columns, 
        mean_values, 
        yerr=sem_values, 
        capsize=6,               # Increase capsize for larger error bars
        color=bar_color,         # Customizable bar color
        edgecolor='black', 
        linewidth=5,             # Thicker and darker bar outlines
        width=0.6,
        error_kw=dict(elinewidth=3, capthick=3, zorder=5)  # Thicker error bars and make them appear above circles
    )

    # Plot all subject lines in gray with connecting markers
    for i, subject in enumerate(df.index):
        ax.plot(df.columns, df.loc[subject], linestyle='-', color='gray', alpha=0.5, linewidth=2.5, zorder=1)

    # Plot unfilled circle markers with larger size, in gray
    for i, subject in enumerate(df.index):
        ax.scatter(df.columns, df.loc[subject], facecolors='none', edgecolors='gray', s=120, alpha=0.6, linewidth=4, zorder=2)

    # Add labels, title, and format the axes
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # Set x-ticks to match the bout labels
    ax.set_xticks(np.arange(len(df.columns)))
    if custom_xtick_labels is not None:
        ax.set_xticklabels(custom_xtick_labels, fontsize=28)
        if custom_xtick_colors is not None:
            for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick.set_color(color)
    else:
        ax.set_xticklabels(df.columns, fontsize=26)

    # Increase the font size of y-axis and x-axis tick numbers
    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)

    # Automatically set the y-limits based on the data range if ylim is not provided
    if ylim is None:
        all_values = np.concatenate([df.values.flatten(), mean_values.values.flatten()])
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

    # Set y-ticks based on yticks_increment if provided
    if yticks_increment is not None:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # Remove the right and top spines, and adjust the left and bottom spines' linewidth
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # Save the figure and display the plot
    # plt.savefig(f'{title}{ylabel[0]}.png', transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    plt.tight_layout()
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
