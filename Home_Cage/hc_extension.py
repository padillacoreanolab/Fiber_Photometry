from trial_class import *
import os
import re

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
from scipy.stats import ttest_rel
from scipy.stats import linregress
import matplotlib.cm as cm



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

def plot_da_vs_duration_by_agent_flipped(experiment, 
                                         agents_of_interest, 
                                         agent_colors, 
                                         agent_labels, 
                                         title,
                                         da_metric='Mean Z-score',
                                         figsize=(10, 7),
                                         ylim=None,
                                         yticks_increment=None,
                                         ylabel=None,  # Custom Y-axis label
                                         legend_loc='upper left',
                                         pad_inches=0.1,
                                         save=False,
                                         save_name=None):
    """
    Plot correlation between bout duration (X) and event-induced DA (Y) 
    for selected agents, with customizable Y-axis label.

    Also prints skipped Subject × Bout types that had no valid data.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy.stats import linregress
    import pandas as pd

    valid_metrics = ['Mean Z-score', 'AUC', 'Max Peak']
    if da_metric not in valid_metrics:
        raise ValueError(f"Invalid da_metric. Choose from {valid_metrics}")

    trial_data = get_trial_dataframes(experiment)  # List of (trial_id, df)
    points = []
    skipped_points = []

    for trial_id, df in trial_data:
        for bout_name in agents_of_interest:
            subset = df[(df["Bout"] == bout_name) & (df["Behavior"] == "Investigation")]
            if subset.empty:
                skipped_points.append((trial_id, bout_name))
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

    # --- Print Skipped Data Points ---
    if skipped_points:
        print("⚠️ Skipped Data Points (No valid Investigation rows found):")
        for subj_id, bout in skipped_points:
            print(f"  - Subject: {subj_id}, Bout: {bout}")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)

    all_x, all_y = [], []

    for agent, group in points_df.groupby("Agent"):
        x = group["Bout_Duration"].values
        y = group[da_metric].values
        all_x.extend(x)
        all_y.extend(y)

        color = agent_colors.get(agent, 'gray')
        ax.scatter(x, y, color=color, s=250, alpha=1.0, edgecolor='black', linewidth=3, label=agent, zorder=3)

    # --- Regression Line ---
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
    ax.set_xlabel("Bout Duration (s)", fontsize=24)
    ax.set_ylabel(ylabel if ylabel else da_metric, fontsize=24)
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


