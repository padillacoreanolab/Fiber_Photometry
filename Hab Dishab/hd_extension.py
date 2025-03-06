import numpy as np
import pandas as pd
import tdt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import os
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from trial_class import *

import matplotlib.pyplot as plt
import re

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

def plot_behavior_metric(df, 
                         behavior="Investigation", 
                         metric="investigation_time",
                         title='Mean Across Bouts',
                         ylabel='Mean Value',
                         ylim=None):
    """
    ...
    """
    # Filter by Behavior
    filtered_df = df[df["Behavior"] == behavior].copy()

    # Group & compute the chosen metric
    if metric == "investigation_time":
        grouped = filtered_df.groupby(["Subject", "Bout"], as_index=False)["Duration (s)"].sum()
        plot_label = "Investigation Time (Sum)"
    elif metric == "average_bout_duration":
        grouped = filtered_df.groupby(["Subject", "Bout"], as_index=False)["Duration (s)"].mean()
        plot_label = "Average Bout Duration (Mean)"
    else:
        raise ValueError("metric must be either 'investigation_time' or 'average_bout_duration'")

    # Pivot: rows = Subject, columns = Bout, values = numeric durations
    pivot_df = grouped.pivot(index="Subject", columns="Bout", values="Duration (s)")

    # ------------------------------------------------------
    # FIX: fill missing (NaN) values with 0
    # ------------------------------------------------------
    pivot_df = pivot_df.fillna(0)

    # Plot the pivoted DataFrame
    hab_dishab_plot_y_across_bouts_gray(
        df=pivot_df,
        title=title,
        ylabel=ylabel if ylabel else plot_label,
        ylim=ylim
    )

def automate_investigation_workflow(experiment, 
                                    trial_keys=None,
                                    combine_threshold=1, 
                                    min_duration=2,
                                    desired_bouts=None,
                                    calc_method="sum"):
    """
    Automates the investigation-data processing workflow.
    
    Parameters
    ----------
    experiment : object
        An object that holds trial data in `experiment.trials[...]`.
    trial_keys : list of str
        A list of keys that map to `experiment.trials`, e.g.
        ["n1-240507-080133", "n2-240507-093913", ...].
        If None, defaults to the 7 example keys.
    combine_threshold : float
        Maximum gap (seconds) to merge consecutive events.
    min_duration : float
        Minimum duration (seconds) to keep an event.
    desired_bouts : list of str
        A list of bout labels to keep in the final filtered summary.
        If None, defaults to ["s1-1", "s1-2", "s1-3", "s1-4", "s1-5", "s2-1"].
    calc_method : str
        The calculation method for the metric.
        Use "sum" (or "total") for total investigation time,
        or "average" for average bout duration.
    
    Returns
    -------
    filtered_summary_df : pd.DataFrame
        The final summary DataFrame (after merging, removing short events,
        creating the summary, selecting the desired metric, and filtering by bouts
        and excluded subjects).
    """

    # ----------------------------------------------------------------
    # 0) Define default keys and bouts if not provided
    # ----------------------------------------------------------------
    if trial_keys is None:
        trial_keys = [
            "n1-240507-080133", 
            "n2-240507-093913",
            "n3-240507-115440",
            "n4-240507-140651",
            "n5-240821-085040",
            "n6-240821-100116",
            "n7-240821-114717"
        ]
    if desired_bouts is None:
        desired_bouts = ["s1-1", "s1-2", "s1-3", "s1-4", "s1-5", "s2-1"]

    # ----------------------------------------------------------------
    # 1) Retrieve each trial and do the merging and removal
    # ----------------------------------------------------------------
    for key in trial_keys:
        nn = experiment.trials[key]
        nn.combine_consecutive_behaviors(
            behavior_name='Investigation',
            bout_time_threshold=combine_threshold
        )
        nn.remove_short_behaviors(
            behavior_name='Investigation',
            min_duration=min_duration
        )

    # ----------------------------------------------------------------
    # 2) Read each trial's behaviors into a list of DataFrames
    # ----------------------------------------------------------------
    trials = []
    for key in trial_keys:
        nn = experiment.trials[key]
        trials.append(nn.behaviors)  # Each .behaviors is now processed

    # ----------------------------------------------------------------
    # 3) Create the combined summary DataFrame
    # ----------------------------------------------------------------
    summary_df = create_subject_summary_df(trials)
    
    # ----------------------------------------------------------------
    # 4) Choose the calculation method for the metric.
    # ----------------------------------------------------------------
    if calc_method.lower() in ["average", "avg"]:
        summary_df["Duration (s)"] = summary_df["Average Bout Duration"]
    else:  # Default to total
        summary_df["Duration (s)"] = summary_df["Total Investigation Time"]

    # ----------------------------------------------------------------
    # 5) Filter the summary DataFrame by the desired bouts
    # ----------------------------------------------------------------
    filtered_summary_df = summary_df[summary_df["Bout"].isin(desired_bouts)]
    
    # ----------------------------------------------------------------
    # 6) Return the final DataFrame
    # ----------------------------------------------------------------
    return filtered_summary_df

# plot stuff
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
    #plt.savefig(f'{title}{ylabel[0]}.png', transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    plt.tight_layout()
    plt.show()

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

# plots colored spaghetti plots corresponding to individual mouse on top of average bar graphs 
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


