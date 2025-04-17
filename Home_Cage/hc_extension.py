from trial_class import *
import os
import re

import numpy as np
import pandas as pd
import tdt

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from scipy.signal import butter, filtfilt
from scipy.stats import ttest_rel, linregress
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
import seaborn as sns
from itertools import combinations

# Behavior processing -----------------------------------------------
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

def trim_short_term_to_5min(trial_data, short_term_bout='Short_Term-1', max_duration=300):
    """
    Trims the 'Short_Term-1' bout to only include behavior events within the first 5 minutes (300 seconds)
    for each subject. Returns a modified trial_data dictionary compatible with create_metadata_dataframe.

    Parameters
    ----------
    trial_data : dict
        Dictionary of {subject_id : DataFrame}, from get_trial_dataframes().
    
    short_term_bout : str
        The name of the bout to trim (default is 'Short_Term-1').
    
    max_duration : int or float
        The maximum duration in seconds to retain (default 300 seconds = 5 minutes).

    Returns
    -------
    trimmed_data : dict
        Updated trial_data dictionary with trimmed 'Short_Term-1' bout for each subject.
    """
    trimmed_data = {}

    for subject_id, df in trial_data.items():
        df_copy = df.copy()

        # Filter only Short_Term-1 rows
        st_mask = df_copy["Bout"] == short_term_bout
        df_st = df_copy[st_mask]

        if not df_st.empty:
            # Find the starting time for Short_Term-1
            start_time = df_st["Event_Start"].min()
            cutoff_time = start_time + max_duration  # 5 minutes after first event start

            # Trim to only events within the first 5 minutes
            df_st_trimmed = df_st[df_st["Event_Start"] <= cutoff_time].copy()

            # Combine with the rest of the DataFrame (non-Short_Term-1 rows)
            df_other = df_copy[~st_mask]
            df_combined = pd.concat([df_other, df_st_trimmed], ignore_index=True)
        else:
            # If no Short_Term-1 events, retain original DataFrame
            df_combined = df_copy

        trimmed_data[subject_id] = df_combined

    return trimmed_data 

# DA processing -----------------------------------------------------
def create_big_df_from_exp_da_dict(exp_da_dict):
    """
    Merges all subjects' DataFrames from exp_da_dict into one big DataFrame.
    Adds a 'Subject' column for each row.
    """
    all_list = []
    for subject_id, df_subj in exp_da_dict.items():
        df_copy = df_subj.copy()
        df_copy["Subject"] = subject_id
        all_list.append(df_copy)
    big_df = pd.concat(all_list, ignore_index=True)
    return big_df

def exponential_decay(x, A, B, tau):
    return A + B * np.exp(-x / tau)

# Behavior plotting -------------------------------------------------
def hc_plot_behavior_times_across_agents_gray(metadata_df,
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
                                             save_name=None,
                                             bout_order=None):
    
    """
    Plots a bar chart with error bars (SEM) and individual subject lines in grey,
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
        - bout_order (list or None): A list specifying the exact order of the bouts (x-axis categories).
                                     Example: ['Short_Term-1', 'Novel', 'Short_Term-2', 'Long_Term-1'].
    """

    # --- Optional filtering ---
    if behavior is not None:
        metadata_df = metadata_df[metadata_df["Behavior"] == behavior].copy()
        if metadata_df.empty:
            raise ValueError(f"No data found for behavior='{behavior}'.")

    if y_col not in metadata_df.columns:
        raise ValueError(f"'{y_col}' not found in metadata_df columns.")

    # --- Pivot Data ---
    pivot_df = metadata_df.pivot(index="Subject", columns="Bout", values=y_col)

    if bout_order is not None:
        pivot_df = pivot_df.reindex(columns=bout_order)

    # --- Compute stats ---
    mean_values = pivot_df.mean()
    sem_values = pivot_df.sem()

    # --- Perform all pairwise t-tests ---
    def perform_t_tests(pivot_df):
        results = {}
        bout_names = pivot_df.columns.tolist()
        for bout1, bout2 in combinations(bout_names, 2):
            paired_df = pivot_df[[bout1, bout2]].dropna()
            if len(paired_df) > 1:
                t_stat, p_value = ttest_rel(paired_df[bout1], paired_df[bout2])
                results[f"{bout1} vs {bout2}"] = {"t_stat": t_stat, "p_value": p_value}
        return results

    t_test_results = perform_t_tests(pivot_df)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

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

    for subject in pivot_df.index:
        ax.plot(pivot_df.columns, pivot_df.loc[subject],
                linestyle='-', color='gray', alpha=0.5,
                linewidth=2.5, zorder=1)
        ax.scatter(pivot_df.columns, pivot_df.loc[subject],
                   facecolors='none', edgecolors='gray',
                   s=120, alpha=0.6, linewidth=4, zorder=2)

    if ylabel is None:
        ylabel = y_col
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    if custom_xtick_labels is not None:
        ax.set_xticklabels(custom_xtick_labels, fontsize=28)
        if custom_xtick_colors is not None:
            for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick.set_color(color)
    else:
        ax.set_xticklabels(pivot_df.columns, fontsize=26)

    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)

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

    if yticks_increment is not None:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

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

    # --- Print T-test Results ---
    if t_test_results:
        print("\nPaired t-test results (all pairwise comparisons):")
        for comp, stats in t_test_results.items():
            p = stats["p_value"]
            stars = "ns"
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            print(f"{comp}: p = {p:.4f} ({stars})")

def hc_plot_behavior_times_across_agents_colored(metadata_df,
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
                                             save_name=None,
                                             bout_order=None):
    """
    Plots a bar chart with error bars (SEM) and individual subject lines in color,
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
        - bout_order (list or None): A list specifying the exact order of the bouts (x-axis categories).
                                     Example: ['Short_Term-1', 'Novel', 'Short_Term-2', 'Long_Term-1'].
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

    # 4) If a bout_order is provided, reorder the columns
    if bout_order is not None:
        pivot_df = pivot_df.reindex(columns=bout_order)

    # 5) Generate unique colors for each subject
    subjects = pivot_df.index
    colors = sns.color_palette("husl", n_colors=len(subjects))  # Generate distinct colors
    subject_color_map = dict(zip(subjects, colors))  # Map each subject to a color

    # 6) Calculate mean and SEM across subjects for each bout
    mean_values = pivot_df.mean()
    sem_values = pivot_df.sem()

    # 7) Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # 8) Bar plot with error bars (SEM)
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

    # 9) Plot each subject's data in color
    for subject in pivot_df.index:
        ax.plot(
            pivot_df.columns, pivot_df.loc[subject],
            linestyle='-', color=subject_color_map[subject], alpha=0.8,
            linewidth=2.5, zorder=1, label=subject
        )
        ax.scatter(
            pivot_df.columns, pivot_df.loc[subject],
            facecolors='none', edgecolors=subject_color_map[subject],
            s=120, alpha=0.9, linewidth=4, zorder=2
        )

    # 10) Set axis labels and title
    if ylabel is None:
        ylabel = y_col
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # 11) Set x-ticks and labels
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

    # 12) Set y-axis limits
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

    # 13) Set y-ticks if an increment is provided
    if yticks_increment is not None:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # 14) Remove right & top spines; thicken left & bottom spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # 15) Add legend for subjects
    ax.legend(title="Subjects", fontsize=18, title_fontsize=20, loc='upper right', bbox_to_anchor=(1.2, 1))

    # 16) Adjust layout and save the figure if requested
    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    
    plt.show()

# DA plotting -------------------------------------------------------
def hc_plot_da_across_agents_gray(precomputed_df, 
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
             pad_inches=0.1,
             save=False,
             save_name=None):
    """
    Plots DA metrics across specific bouts using a provided DataFrame.
    Prints paired t-test results comparing all pairwise combinations of bouts.
    """

    def perform_t_tests(pivot_df):
        results = {}
        bout_names = pivot_df.columns.tolist()
        for bout1, bout2 in combinations(bout_names, 2):  # all pairwise combos
            paired_df = pivot_df[[bout1, bout2]].dropna()
            if len(paired_df) > 1:
                t_stat, p_value = ttest_rel(paired_df[bout1], paired_df[bout2])
                results[f"{bout1} vs {bout2}"] = {"t_stat": t_stat, "p_value": p_value}
        return results

    df = precomputed_df.copy()

    try:
        pivot_df = df.pivot(index="Subject", columns="Bout", values=metric_name)
    except Exception as e:
        print("Error pivoting data:", e)
        return

    overall_stats = df.groupby("Bout")[metric_name].agg(['mean', 'sem']).reset_index()
    t_test_results = perform_t_tests(pivot_df)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    bar_positions = np.arange(len(overall_stats))

    ax.bar(bar_positions, overall_stats["mean"], yerr=overall_stats["sem"],
           capsize=6, color=bar_color, edgecolor='black', linewidth=5, width=0.6,
           error_kw=dict(elinewidth=3, capthick=3, zorder=5))

    # Individual subject lines
    for subject_id in pivot_df.index:
        ax.plot(bar_positions, pivot_df.loc[subject_id], linestyle='-', color='gray', 
                alpha=0.5, linewidth=3, marker='o', markerfacecolor='none', 
                markeredgecolor='gray', markeredgewidth=3, markersize=10)

    # Labels
    ax.set_ylabel(ylabel, fontsize=35, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=35, labelpad=12)
    ax.set_title(title, fontsize=28)

    ax.set_xticks(bar_positions)
    xtick_labels = custom_xtick_labels if custom_xtick_labels else overall_stats["Bout"].tolist()
    ax.set_xticklabels(xtick_labels, fontsize=35)
    if custom_xtick_colors:
        for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
            tick.set_color(color)

    ax.tick_params(axis='y', labelsize=35)
    ax.tick_params(axis='x', labelsize=35)

    if ylim:
        ax.set_ylim(ylim)
    else:
        all_vals = np.concatenate([pivot_df.values.flatten(), overall_stats["mean"].values])
        ax.set_ylim(0, np.nanmax(all_vals) * 1.2)

    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment))

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)

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

    # --- Print T-test Results ---
    if t_test_results:
        print("\nPaired t-test results (all pairwise comparisons):")
        for comp, stats in t_test_results.items():
            p = stats["p_value"]
            stars = "ns"
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            print(f"{comp}: p = {p:.4f} ({stars})")

def hc_plot_da_across_agents_colored(precomputed_df, 
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
             pad_inches=0.1,
             save=False,
             save_name=None):
    """
    Plots DA metrics across specific bouts using a provided DataFrame.
    Individual subjects are plotted in unique colors with a legend.
    Also prints paired t-test results for all pairwise comparisons of bouts.
    """

    def perform_t_tests(pivot_df):
        results = {}
        bout_names = pivot_df.columns.tolist()
        for bout1, bout2 in combinations(bout_names, 2):
            paired_df = pivot_df[[bout1, bout2]].dropna()
            if len(paired_df) > 1:
                t_stat, p_value = ttest_rel(paired_df[bout1], paired_df[bout2])
                results[f"{bout1} vs {bout2}"] = {"t_stat": t_stat, "p_value": p_value}
        return results

    df = precomputed_df.copy()

    try:
        pivot_df = df.pivot(index="Subject", columns="Bout", values=metric_name)
    except Exception as e:
        print("Error pivoting data:", e)
        return

    overall_stats = df.groupby("Bout")[metric_name].agg(['mean', 'sem']).reset_index()
    t_test_results = perform_t_tests(pivot_df)

    # --- Unique colors for subjects ---
    subjects = pivot_df.index.tolist()
    cmap = cm.get_cmap('tab20', len(subjects))
    color_map = {subject: cmap(i) for i, subject in enumerate(subjects)}

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    bar_positions = np.arange(len(overall_stats))

    ax.bar(bar_positions, overall_stats["mean"], yerr=overall_stats["sem"],
           capsize=6, color=bar_color, edgecolor='black', linewidth=5, width=0.6,
           error_kw=dict(elinewidth=3, capthick=3, zorder=5))

    # Individual subject lines with unique colors
    for subject_id in pivot_df.index:
        ax.plot(bar_positions, pivot_df.loc[subject_id], linestyle='-',
                color=color_map[subject_id], linewidth=3, marker='o',
                markerfacecolor='none', markeredgecolor=color_map[subject_id],
                markeredgewidth=3, markersize=10, label=str(subject_id))

    # Add legend (one entry per subject)
    ax.legend(title="Subjects", fontsize=14, title_fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Labels
    ax.set_ylabel(ylabel, fontsize=35, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=35, labelpad=12)
    ax.set_title(title, fontsize=28)

    ax.set_xticks(bar_positions)
    xtick_labels = custom_xtick_labels if custom_xtick_labels else overall_stats["Bout"].tolist()
    ax.set_xticklabels(xtick_labels, fontsize=35)
    if custom_xtick_colors:
        for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
            tick.set_color(color)

    ax.tick_params(axis='y', labelsize=35)
    ax.tick_params(axis='x', labelsize=35)

    if ylim:
        ax.set_ylim(ylim)
    else:
        all_vals = np.concatenate([pivot_df.values.flatten(), overall_stats["mean"].values])
        ax.set_ylim(0, np.nanmax(all_vals) * 1.2)

    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment))

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)

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

    # --- Print T-test Results ---
    if t_test_results:
        print("\nPaired t-test results (all pairwise comparisons):")
        for comp, stats in t_test_results.items():
            p = stats["p_value"]
            stars = "ns"
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            print(f"{comp}: p = {p:.4f} ({stars})")

def plot_peak_da_for_subsequent_investigations(
    exp_da_dict,
    selected_bouts=None,             # e.g. ["s1-1","s1-2"]
    n_subsequent_investigations=3,   # e.g. keep first 3 investigations per (Subject, Bout)
    peak_col="Max Peak",             # which column holds the per-event peak DA
    metric_type='slope',             # choose 'slope' or 'decay'
    figsize=(14, 8),
    line_order=None,
    custom_colors=None,
    custom_legend_labels=None,
    custom_xtick_labels=None,
    ylim=None,
    ytick_increment=None,
    xlabel="Investigation Index",
    ylabel="Avg " + "Max Peak",
    plot_title="Average Peak per Investigation"
):
    """
    1) Merges all DataFrames in exp_da_dict into one big DataFrame.
    2) Filters for the specified bouts (e.g. ["s1-1", "s1-2"]).
    3) Within each (Subject, Bout), sorts by Event_Start and assigns an 'InvestigationIndex'
       (1 for the first event, 2 for the second, etc.).
    4) Keeps only the first n_subsequent_investigations per (Subject, Bout).
    5) Groups by (Bout, InvestigationIndex) across subjects and computes the average of peak_col.
    6) For each Bout, fits either a linear regression (if metric_type='slope') or an exponential decay
       (if metric_type='decay') to the AvgPeak vs. InvestigationIndex data.
       The computed value (slope or decay constant) is shown in the legend.
    7) Plots each Bout as a line using full custom visual styling.
    
    Returns the aggregated DataFrame used for plotting.
    """
    # 1) Merge all subject data
    big_df = create_big_df_from_exp_da_dict(exp_da_dict)
    
    # 2) Filter for the chosen bouts if provided
    if selected_bouts is not None:
        big_df = big_df[big_df["Bout"].isin(selected_bouts)].copy()
    
    if big_df.empty:
        print("No data left after filtering for bouts. Nothing to plot.")
        return pd.DataFrame()
    
    # 3) Within each (Subject, Bout), sort by Event_Start and assign an InvestigationIndex
    big_df.sort_values(["Subject", "Bout", "Event_Start"], inplace=True)
    big_df["InvestigationIndex"] = big_df.groupby(["Subject", "Bout"]).cumcount() + 1
    
    # 4) Keep only the first n_subsequent_investigations per (Subject, Bout)
    big_df = big_df[big_df["InvestigationIndex"] <= n_subsequent_investigations]
    
    # 5) Group by (Bout, InvestigationIndex) and compute average peak and subject count
    agg_df = (
        big_df.groupby(["Bout", "InvestigationIndex"], as_index=False)
        .agg(
            SubjectCount=("Subject", "nunique"),
            AvgPeak=(peak_col, "mean")
        )
    )
    
    # 6) Create figure with custom styling
    if custom_colors is None:
        custom_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(5)
    ax.spines["bottom"].set_linewidth(5)
    ax.tick_params(axis="both", which="major", labelsize=48)
    
    metrics_dict = {}  # to store computed metric for each Bout
    
    # Determine unique bouts to plot (order by line_order if provided)
    if line_order is None:
        unique_bouts = sorted(agg_df["Bout"].unique())
    else:
        unique_bouts = line_order
    
    # 7) For each Bout, fit the data and plot the line
    for i, bout in enumerate(unique_bouts):
        df_line = agg_df[agg_df["Bout"] == bout].copy()
        df_line.sort_values("InvestigationIndex", inplace=True)
        
        x_vals = df_line["InvestigationIndex"].values
        y_vals = df_line["AvgPeak"].values
        
        if len(x_vals) == 0 or len(y_vals) == 0:
            print(f"Skipping bout '{bout}' due to no data.")
            continue
        
        if metric_type.lower() == 'slope':
            slope, intercept, r_val, p_val, std_err = linregress(x_vals, y_vals)
            metrics_dict[bout] = slope
            metric_label = f"slope: {slope:.3f}"
        elif metric_type.lower() == 'decay':
            p0 = (np.min(y_vals), np.max(y_vals)-np.min(y_vals), 1.0)  # initial guess: A, B, tau
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
        
        # Prepare legend text; incorporate custom legend labels if provided
        if custom_legend_labels and i < len(custom_legend_labels):
            legend_text = custom_legend_labels[i]
        else:
            legend_text = bout
        # Append computed metric and subject count (n) to legend text
        legend_text += f" ({metric_label}, n={df_line['SubjectCount'].max()})"
        
        color = custom_colors[i % len(custom_colors)]
        ax.plot(
            x_vals, y_vals,
            marker='o', linestyle='-',
            color=color,
            linewidth=5, markersize=30,
            label=legend_text
        )
    
    # 8) Set axis labels and formatting
    ax.set_xlabel(xlabel, fontsize=44, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=44, labelpad=12)
    
    if ylim is not None:
        ax.set_ylim(ylim)
        if ytick_increment is not None:
            y_ticks = np.arange(ylim[0], ylim[1] + ytick_increment, ytick_increment)
            ax.set_yticks(y_ticks)
            y_tick_labels = [f"{int(yt)}" if float(yt).is_integer() else f"{yt:.1f}" for yt in y_ticks]
            ax.set_yticklabels(y_tick_labels, fontsize=44)
    
    if custom_xtick_labels:
        ax.set_xticks(np.arange(1, len(custom_xtick_labels) + 1))
        ax.set_xticklabels(custom_xtick_labels, fontsize=44)
    else:
        unique_x = sorted(agg_df["InvestigationIndex"].unique())
        ax.set_xticks(unique_x)
        ax.set_xticklabels([str(x) for x in unique_x], fontsize=44)
    
    if plot_title:
        ax.set_title(plot_title, fontsize=20)
    
    ax.legend(fontsize=26)
    plt.tight_layout()
    plt.savefig("my_plot.png", transparent=True, dpi=300)
    plt.show()
    
    print(f"\n=== Computed Metric ({metric_type.upper()}): ===")
    for bout, val in metrics_dict.items():
        print(f"Bout: {bout}, {metric_type} = {val:.3f}")
    
    return agg_df

# -------------------------------------------------------------------
# Rank stuff. We'll be working on this more: not in documentation yet.
# Plots DA vs. bout duration plots
def plot_da_vs_duration_by_agent(experiment, 
                                 agents_of_interest, 
                                 agent_colors, 
                                 agent_labels, 
                                 title,
                                 da_metric='Mean Z-score',
                                 figsize=(10, 7),
                                 ylim=None,
                                 yticks_increment=None,
                                 xlabel = None,
                                 legend_loc='upper left',
                                 pad_inches=0.1,
                                 save=None,
                                 save_name=None):  # New parameter
    """
    Plot correlation between event-induced DA (Mean Z-score, AUC, or Max Peak) 
    and bout duration for selected agents.
    """
    valid_metrics = ['Mean Z-score', 'AUC', 'Max Peak']
    if da_metric not in valid_metrics:
        raise ValueError(f"Invalid da_metric. Choose from {valid_metrics}")

    trial_dfs = get_trial_dataframes(experiment)
    points = []

    for trial_id, df in zip(experiment.trials.keys(), trial_dfs):
        for bout_name in agents_of_interest:
            subset = df[(df["Bout"] == bout_name) & (df["Behavior"] == "Investigation")]
            if subset.empty:
                continue

            first_invest = subset.iloc[0]
            duration = first_invest["Duration (s)"]
            mean_z = first_invest.get("Mean Z-score", np.nan)
            auc = first_invest.get("AUC", np.nan)
            max_peak = first_invest.get("Max Peak", np.nan)

            prefix = bout_name.split('-')[0]
            agent_label = agent_labels.get(prefix, prefix)

            points.append({
                'Subject': trial_id,
                'Agent': agent_label,
                'Bout_Duration': duration,
                'Mean Z-score': mean_z,
                'AUC': auc,
                'Max Peak': max_peak
            })

    points_df = pd.DataFrame(points)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    all_x, all_y = [], []

    for agent, group in points_df.groupby("Agent"):
        x = group[da_metric].values
        y = group["Bout_Duration"].values
        all_x.extend(x)
        all_y.extend(y)

        color = agent_colors.get(agent, 'gray')
        ax.scatter(x, y, color=color, s=250, alpha=1.0, edgecolor='black', linewidth=3, label=agent, zorder=3)

    # --- Regression Line (Pooled) ---
    stats_text_lines = ["r = ---", "p = ---", "n = ---"]
    if len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_fit = np.linspace(min(all_x), max(all_x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='black', linewidth=2.5, linestyle='-', zorder=2)

        stats_text_lines = [
            f"r = {r_val:.3f}",
            f"p = {p_val:.3f}",
            f"n = {len(all_x)} events"
        ]

    # --- Labels ---
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel("Bout Duration (s)", fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=24)

    # X-axis label
    final_xlabel = xlabel if xlabel else da_metric
    ax.set_xlabel(final_xlabel, fontsize=24)


    # --- Y-axis formatting ---
    if ylim:
        ax.set_ylim(ylim)
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        yticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if x.is_integer() else f'{x}'))

    # --- Combined Legend ---
    handles, labels = ax.get_legend_handles_labels()
    stats_label = "\n".join(stats_text_lines)
    stats_handle = plt.Line2D([], [], color='none', label=stats_label)
    handles.append(stats_handle)
    labels.append(stats_label)

    legend = ax.legend(handles=handles, labels=labels, loc=legend_loc, fontsize=16, title='Agent', title_fontsize=18, 
                       frameon=True, facecolor='white', edgecolor='lightgray', fancybox=False)
    legend.get_frame().set_alpha(0.8)

    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    return points_df

# Plots event-induced DA plots in colors by subject
def plot_da_vs_duration_by_agent_colored(experiment, 
                                 agents_of_interest, 
                                 agent_labels, 
                                 title,
                                 da_metric='Mean Z-score',
                                 figsize=(10, 7),
                                 ylim=None,
                                 yticks_increment=None,
                                 xlabel=None,
                                 legend_loc='upper left',
                                 pad_inches=0.1,
                                 save=None,
                                 save_name=None):
    """
    Plot correlation between event-induced DA and bout duration for each subject with unique colors.
    """
    valid_metrics = ['Mean Z-score', 'AUC', 'Max Peak']
    if da_metric not in valid_metrics:
        raise ValueError(f"Invalid da_metric. Choose from {valid_metrics}")

    trial_dfs = get_trial_dataframes(experiment)
    points = []

    for trial_id, df in zip(experiment.trials.keys(), trial_dfs):
        for bout_name in agents_of_interest:
            subset = df[(df["Bout"] == bout_name) & (df["Behavior"] == "Investigation")]
            if subset.empty:
                continue

            first_invest = subset.iloc[0]
            duration = first_invest["Duration (s)"]
            mean_z = first_invest.get("Mean Z-score", np.nan)
            auc = first_invest.get("AUC", np.nan)
            max_peak = first_invest.get("Max Peak", np.nan)

            prefix = bout_name.split('-')[0]
            agent_label = agent_labels.get(prefix, prefix)

            points.append({
                'Subject': trial_id,
                'Agent': agent_label,
                'Bout_Duration': duration,
                'Mean Z-score': mean_z,
                'AUC': auc,
                'Max Peak': max_peak
            })

    points_df = pd.DataFrame(points)

    # --- Assign unique colors per Subject ---
    unique_subjects = points_df['Subject'].unique()
    color_map = cm.get_cmap('tab20', len(unique_subjects))  # Can change to 'nipy_spectral' etc.
    subject_color_dict = {subj: color_map(i) for i, subj in enumerate(unique_subjects)}

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    all_x, all_y = [], []

    for _, row in points_df.iterrows():
        x = row[da_metric]
        y = row["Bout_Duration"]
        all_x.append(x)
        all_y.append(y)
        subj = row["Subject"]
        color = subject_color_dict[subj]

        ax.scatter(x, y, color=color, s=250, alpha=1.0, edgecolor='black', linewidth=3, label=subj, zorder=3)

    # --- Regression Line (Pooled) ---
    stats_text_lines = ["r = ---", "p = ---", "n = ---"]
    if len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_fit = np.linspace(min(all_x), max(all_x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='black', linewidth=2.5, linestyle='-', zorder=2)

        stats_text_lines = [
            f"r = {r_val:.3f}",
            f"p = {p_val:.3f}",
            f"n = {len(all_x)} events"
        ]

    # --- Axis Labels ---
    final_xlabel = xlabel if xlabel else da_metric
    ax.set_xlabel(final_xlabel, fontsize=24)
    ax.set_ylabel("Bout Duration (s)", fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis='both', labelsize=24)

    # --- Y-axis formatting ---
    if ylim:
        ax.set_ylim(ylim)
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        yticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if x.is_integer() else f'{x}'))

    # --- Legend per Subject ---
    subject_handles = [
        plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, 
                   markeredgecolor='black', markerfacecolor=color, linewidth=0, label=str(subj))
        for subj, color in subject_color_dict.items()
    ]

    stats_label = "\n".join(stats_text_lines)
    stats_handle = plt.Line2D([], [], color='none', label=stats_label)

    handles_combined = subject_handles + [stats_handle]
    labels_combined = [h.get_label() for h in subject_handles] + [stats_label]

    legend = ax.legend(handles=handles_combined, labels=labels_combined, loc=legend_loc, fontsize=14, 
                       title='Subject ID', title_fontsize=16, frameon=True, facecolor='white', 
                       edgecolor='lightgray', fancybox=False)
    legend.get_frame().set_alpha(0.8)

    plt.tight_layout()
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    return points_df

def assign_subject_ranks_to_experiment(experiment, rank_csv_path):
    """
    Loads subject ranks from CSV and assigns each trial’s .behaviors DataFrame a new 'Rank' column.

    Parameters:
    - experiment : Experiment object
    - rank_csv_path : path to CSV with columns ['Subject', 'Rank']
    """
    # Load ranks
    rank_df = pd.read_csv(rank_csv_path)
    rank_dict = dict(zip(rank_df['Subject'].str.lower(), rank_df['Rank']))

    subjects_assigned = 0
    subjects_missing = []

    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            subject_prefix = trial_name[:3].lower()  # Match e.g., 'pp1', 'nn2'

            if subject_prefix in rank_dict:
                trial.behaviors["Rank"] = rank_dict[subject_prefix]
                subjects_assigned += 1
            else:
                trial.behaviors["Rank"] = None
                subjects_missing.append(trial_name)

    print(f"Ranks assigned to {subjects_assigned} trials.")
    if subjects_missing:
        print(f"No rank found for trials: {subjects_missing}")

def generate_investigation_per_agent_df(experiment, rank_csv_path=None, behavior_name='Investigation'):
    """
    Generates a DataFrame with Subject, Rank, and both Total Investigation Time
    and Average Bout Duration per Agent (Bout).

    Returns:
        DataFrame where rows = subjects, columns = [Total_Acq-ST, Avg_Acq-ST, ...], NaNs filled with 0.
    """

    # Load rank CSV if provided
    rank_dict = {}
    if rank_csv_path:
        rank_df = pd.read_csv(rank_csv_path)
        if 'Subject' not in rank_df.columns or 'Rank' not in rank_df.columns:
            rank_df = pd.read_csv(rank_csv_path, header=None, names=['Subject', 'Rank'])
        rank_dict = dict(zip(rank_df['Subject'].str.lower(), rank_df['Rank']))

    data = []
    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            df = trial.behaviors.copy()

            if behavior_name not in df['Behavior'].unique():
                continue

            df = df[df['Behavior'] == behavior_name].copy()
            if df.empty or 'Bout' not in df.columns:
                continue

            # Total duration per Agent
            total_per_agent = df.groupby('Bout')['Duration (s)'].sum()
            # Average bout duration per Agent
            avg_per_agent = df.groupby('Bout')['Duration (s)'].mean()

            # Subject ID
            subj_id = trial_name[:3].lower()
            rank = rank_dict.get(subj_id, None)

            row = {'Subject': subj_id, 'Rank': rank}
            for bout, total_val in total_per_agent.items():
                row[f'Total_{bout}'] = total_val
            for bout, avg_val in avg_per_agent.items():
                row[f'Avg_{bout}'] = avg_val

            data.append(row)

    # Final DataFrame with NaNs filled with 0
    agent_df = pd.DataFrame(data).set_index('Subject').fillna(0)
    return agent_df

def plot_y_across_bouts_ranks(df,  
                             title='Mean Across Bouts', 
                             ylabel='Mean Value', 
                             custom_xtick_labels=None, 
                             custom_xtick_colors=None, 
                             ylim=None, 
                             bar_fill_color='white',     # NEW
                             bar_edge_color='black',     # NEW
                             bar_linewidth=3,            # NEW
                             bar_hatch='///',            # NEW
                             yticks_increment=None, 
                             xlabel='Agent',
                             figsize=(12,7), 
                             pad_inches=0.1,
                             rank_filter=None,
                             metric='Total'):
    """
    Plots mean Total Investigation Time or Average Bout Duration per Agent, with SEM and individual lines.

    Parameters:
        - df (DataFrame): Includes columns like 'Total_X' or 'Avg_X' per agent, plus 'Rank'.
        - metric (str): 'Total' or 'Avg' - determines which columns to plot.
        - rank_filter (int or None): Plot only subjects with this rank (if provided).
        - bar_fill_color (str): Fill color of the bars.
        - bar_edge_color (str): Edge color of the bars.
        - bar_linewidth (float): Width of the bar edges.
        - bar_hatch (str): Hatch pattern for the bars.
        [other params same as before]
    """

    # --- Filter by Rank ---
    if rank_filter is not None:
        if "Rank" not in df.columns:
            print("Rank filtering requested, but 'Rank' column not found.")
            return
        df = df[df["Rank"] == rank_filter]
        if df.empty:
            print(f"No data for Rank {rank_filter}.")
            return
        print(f"Plotting Rank {rank_filter} subjects: {len(df)} entries.")

    # --- Select columns by metric ---
    if metric not in ['Total', 'Avg']:
        raise ValueError("metric must be 'Total' or 'Avg'")
    value_columns = [col for col in df.columns if col.startswith(f"{metric}_")]

    if not value_columns:
        print(f"No columns found starting with '{metric}_'.")
        return

    df_plot = df[value_columns].copy()
    df_plot.columns = [col.replace(f"{metric}_", "") for col in df_plot.columns]

    # --- T-tests ---
    def perform_t_tests(df_vals):
        comparisons = {
            "acq_st_vs_short_term": ("Acq-ST", "Short Term"),
            "acq_st_vs_long_term": ("Acq-ST", "Long Term"),
            "acq_st_vs_novel": ("Acq-ST", "Novel")
        }
        results = {}
        for key, (b1, b2) in comparisons.items():
            if b1 in df_vals.columns and b2 in df_vals.columns:
                paired = df_vals[[b1, b2]].dropna()
                if len(paired) > 1:
                    t_stat, p_value = ttest_rel(paired[b1], paired[b2])
                    results[key] = {"t_stat": t_stat, "p_value": p_value}
        return results

    t_test_results = perform_t_tests(df_plot)

    # --- Stats ---
    mean_vals = df_plot.mean()
    sem_vals = df_plot.sem()

    fig, ax = plt.subplots(figsize=figsize)

    # --- Bar Plot ---
    ax.bar(df_plot.columns, mean_vals, yerr=sem_vals, capsize=6,
           color=bar_fill_color, edgecolor=bar_edge_color, linewidth=bar_linewidth,
           width=0.6, hatch=bar_hatch,
           error_kw=dict(elinewidth=2, capthick=2, zorder=5))

     # --- Lines + Colored Markers ---
    for subject_id, row in df_plot.iterrows():
        subject_prefix = str(subject_id).lower().strip()
        if subject_prefix.startswith('n'):
            marker_color = '#15616F'  # NAc
        elif subject_prefix.startswith('p'):
            marker_color = '#FFAF00'  # mPFC
        else:
            marker_color = 'gray'  # fallback

        # Gray line
        ax.plot(df_plot.columns, row.values, linestyle='-', color='gray',
                alpha=0.5, linewidth=2.5, zorder=1)

        # Colored opaque filled circles, no border, behind error bars
        ax.scatter(df_plot.columns, row.values, color=marker_color,
                   s=120, alpha=1.0, zorder=1)

    # --- Labels ---
    ax.set_ylabel(ylabel, fontsize=30, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=30, labelpad=12)
    ax.set_title(title, fontsize=16)

    # --- X-ticks ---
    ax.set_xticks(np.arange(len(df_plot.columns)))
    if custom_xtick_labels:
        ax.set_xticklabels(custom_xtick_labels, fontsize=28)
        if custom_xtick_colors:
            for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
                tick.set_color(color)
    else:
        ax.set_xticklabels(df_plot.columns, fontsize=26)

    ax.tick_params(axis='y', labelsize=30)
    ax.tick_params(axis='x', labelsize=30)

    # --- Y-limits ---
    all_vals = np.concatenate([df_plot.values.flatten(), mean_vals.values])
    if ylim is None:
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

    # --- Y-ticks ---
    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # --- Aesthetic ---
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # --- Significance Markers ---
    if t_test_results:
        max_y = ax.get_ylim()[1]
        sig_y_offset = max_y * 0.05
        comparisons = {
            "acq_st_vs_short_term": (0, 1),
            "acq_st_vs_long_term": (0, 2),
            "acq_st_vs_novel": (0, 3)
        }
        line_spacing = sig_y_offset * 2.5
        current_y = mean_vals.max() + sig_y_offset

        for key, (x1, x2) in comparisons.items():
            if key in t_test_results:
                p_value = t_test_results[key]["p_value"]
                if p_value < 0.05:
                    significance = "**" if p_value < 0.01 else "*"
                    ax.plot([x1, x2], [current_y, current_y], color='black', linewidth=5)
                    ax.text((x1 + x2) / 2, current_y + sig_y_offset / 1.5, significance,
                            fontsize=40, ha='center', color='black')
                    current_y += line_spacing
                    
    # --- Legend for Region Colors with Dots ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='NAc',
            markerfacecolor='#15616F', markersize=12, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='none', label='mPFC',
            markerfacecolor='#FFAF00', markersize=12, markeredgewidth=0)
    ]

    ax.legend(handles=legend_elements, title="Region", fontsize=20, title_fontsize=22,
            loc='upper right', frameon=True)

    plt.savefig(f'{title}{ylabel[0]}.png', transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    plt.tight_layout(pad=pad_inches)
    plt.show()

def assign_ranks_and_combine_da_metrics(experiment, rank_csv_path):
    """
    Assigns subject ranks (from CSV) to each trial's behaviors DataFrame after DA metrics are computed.
    Then combines all trials into a single DataFrame with columns:
    ['Subject', 'Rank', 'Behavior', 'Bout', 'Event_Start', 'Event_End', 'Duration (s)',
     'AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score', 'Original End', 'Adjusted End']
    
    Returns:
        combined_df (DataFrame): All behaviors + DA metrics + Rank, with Subject column.
    """

    # Load rank CSV
    rank_df = pd.read_csv(rank_csv_path)
    if 'Subject' not in rank_df.columns or 'Rank' not in rank_df.columns:
        rank_df.columns = ['Subject', 'Rank']  # fallback if no headers
    rank_dict = dict(zip(rank_df['Subject'].str.lower(), rank_df['Rank']))

    combined_rows = []
    assigned = 0

    for trial_name, trial in experiment.trials.items():
        subj_id = trial_name[:3].lower()  # e.g., pp1, nn2
        rank = rank_dict.get(subj_id, None)

        if rank is not None and hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            df = trial.behaviors.copy()
            df['Subject'] = subj_id
            df['Rank'] = rank
            combined_rows.append(df)
            assigned += 1
        else:
            print(f"Skipped trial '{trial_name}' — no rank match or empty behaviors.")

    combined_df = pd.concat(combined_rows, ignore_index=True)
    print(f"Ranks assigned to {assigned} trials. Combined DataFrame shape: {combined_df.shape}")
    
    return combined_df


