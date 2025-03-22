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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

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

def plot_peak_for_subsequent_investigations_custom(
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
