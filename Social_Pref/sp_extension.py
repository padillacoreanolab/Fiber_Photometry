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

# Behavior Plotting
def fix_behavior_data_for_experiment(experiment, csv_base_path):
    """
    Loads and converts raw CSVs into behavior DataFrames for each trial in the experiment.
    Replaces experiment.trials[trial_name].behaviors with the corrected DataFrame.
    """
    fixed_trials = 0
    skipped_trials = 0

    for trial_name in experiment.trials:
        csv_file = os.path.join(csv_base_path, trial_name + ".csv")

        if not os.path.exists(csv_file):
            print(f"CSV not found for trial '{trial_name}' — skipped.")
            skipped_trials += 1
            continue

        try:
            raw_df = pd.read_csv(csv_file)

            # Only process rows that have Start, Stop, Duration
            if {"Start (s)", "Stop (s)", "Duration (s)", "Behavior"}.issubset(raw_df.columns):
                behavior_df = pd.DataFrame({
                    "Behavior": raw_df["Behavior"],
                    "Event_Start": raw_df["Start (s)"],
                    "Event_End": raw_df["Stop (s)"],
                    "Duration (s)": raw_df["Duration (s)"],
                    "Bout": raw_df.get("Subject", "Subject-1")  # fallback if no 'Subject'
                })

                # Clean up NaNs, e.g., missing durations
                behavior_df.dropna(subset=["Event_Start", "Event_End", "Duration (s)"], inplace=True)

                # Inject into experiment object
                experiment.trials[trial_name].behaviors = behavior_df.reset_index(drop=True)
                fixed_trials += 1
            else:
                print(f"CSV for trial '{trial_name}' missing required columns — skipped.")
                skipped_trials += 1

        except Exception as e:
            print(f"Error processing trial '{trial_name}': {e}")
            skipped_trials += 1

    print(f"\nFinished processing CSVs.")
    print(f"Fixed trials: {fixed_trials}")
    print(f"Skipped trials: {skipped_trials}")
 
def plot_custom_sniff_cup_assignments(experiment, 
                                      assignment_csv_path,
                                      bar_color='#cccccc', 
                                      figsize=(6, 8),
                                      title="Mean Investigation Time (Long Term Sniff Cups)",
                                      pad_inches=0.1,
                                      save=False,
                                      save_name=None):
    """
    For each subject, extracts total investigation time for the sniff cup assigned to 'long_term',
    classifies them as Pref or No_Pref, and plots the mean + individual data points.

    Returns:
    - DataFrame with 'Preference' column
    - Prints lists of Pref and No_Pref subjects
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Load assignment CSV ---
    assign_df = pd.read_csv(assignment_csv_path)

    # --- Build subject -> sniff cup (assigned to long_term) mapping ---
    subject_to_behavior = {}

    for idx, row in assign_df.iterrows():
        subject = str(row['Subject']).strip()
        for cup_col in assign_df.columns:
            if 'sniff cup' in cup_col.lower():
                cup_value = str(row[cup_col]).strip().lower()
                if cup_value == 'long_term':
                    subject_to_behavior[subject] = cup_col.lower()
                    break  # Stop at first match for long_term

    subject_data = []
    total_trials = len(experiment.trials)
    valid_trials = 0
    skipped_empty = 0
    skipped_unmatched = 0
    skipped_no_behavior = 0

    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            df = trial.behaviors.copy()

            matched_key = None
            for key in subject_to_behavior:
                if trial_name.lower().startswith(key.lower()):
                    matched_key = key
                    break

            if not matched_key:
                print(f"Skipped trial '{trial_name}' — no subject assignment match.")
                skipped_unmatched += 1
                continue

            target_behavior = subject_to_behavior[matched_key]

            behavior_df = df[df["Behavior"].str.lower() == target_behavior]
            if behavior_df.empty:
                print(f"Skipped trial '{trial_name}' — no data for '{target_behavior}'.")
                skipped_no_behavior += 1
                continue

            total_time = behavior_df["Duration (s)"].sum()

            subject_data.append({
                "Subject": trial_name,
                "Total Investigation Time": total_time,
                "Assigned Behavior": target_behavior
            })
            valid_trials += 1
        else:
            skipped_empty += 1
            print(f"Skipped trial '{trial_name}' — empty behaviors DataFrame.")

    print(f"\nTotal trials: {total_trials}")
    print(f"Valid trials with matched behavior: {valid_trials}")
    print(f"Skipped (empty): {skipped_empty}, Skipped (no match): {skipped_unmatched}, Skipped (no behavior data): {skipped_no_behavior}")

    if not subject_data:
        print("No valid investigation data found for sniff cup assignments.")
        return

    data_df = pd.DataFrame(subject_data)

    # --- Classification ---
    mean_time = data_df["Total Investigation Time"].mean()
    sem_time = data_df["Total Investigation Time"].sem()
    threshold = mean_time + sem_time

    data_df["Preference"] = data_df["Total Investigation Time"].apply(
        lambda t: "Pref" if t > threshold else "No_Pref"
    )

    # Store subjects by classification
    pref_subjects = data_df[data_df["Preference"] == "Pref"]["Subject"].tolist()
    no_pref_subjects = data_df[data_df["Preference"] == "No_Pref"]["Subject"].tolist()

    print(f"\nPref Subjects ({len(pref_subjects)}): {pref_subjects}")
    print(f"No_Pref Subjects ({len(no_pref_subjects)}): {no_pref_subjects}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(0, mean_time, yerr=sem_time, capsize=10, color=bar_color, edgecolor='black', linewidth=3, width=0.6)

    for _, row in data_df.iterrows():
        subj = row["Subject"]
        time = row["Total Investigation Time"]

        color = "#15616F" if subj.lower().startswith('n') else "#FFAF00" if subj.lower().startswith('p') else "gray"
        ax.plot(0, time, 'o', color=color, markersize=10)

    ax.set_xticks([0])
    ax.set_xticklabels(["Long Term Sniff Cup"], fontsize=14)
    ax.set_ylabel("Total Investigation Time (s)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(-0.5, 0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout(pad=pad_inches)
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)
    
    plt.show()

    return data_df, pref_subjects, no_pref_subjects

# With Colors
def plot_grouped_sniff_cup_assignments(experiment, 
                                       assignment_csv_path,
                                       bar_color='#cccccc', 
                                       color_nn_pp="red", 
                                       color_n_p="blue", 
                                       figsize=(6, 8),
                                       title="Grouped Mean Investigation Time (Long Term Sniff Cups)",
                                       pad_inches=0.1,
                                       save=False,
                                       save_name=None):
    """
    Groups subjects by prefix:
      - nn/pp → one color
      - n/p → another color
    Dynamically extracts sniff cup assigned to 'long_term' from CSV,
    and plots their investigation times with group colors and a legend.

    Parameters:
    - assignment_csv_path (str): Path to CSV with Subject + sniff cup assignments.
    - title (str): Title for the plot.
    """
    # --- Load assignment CSV ---
    assign_df = pd.read_csv(assignment_csv_path)

    # --- Map subject → sniff cup assigned to long_term ---
    subject_to_behavior = {}

    for idx, row in assign_df.iterrows():
        subject = str(row['Subject']).strip()
        for col in assign_df.columns:
            if 'sniff cup' in col.lower():
                value = str(row[col]).strip().lower()
                if value == 'long_term':
                    subject_to_behavior[subject] = col.lower()
                    break

    subject_data = []
    total_trials = len(experiment.trials)
    valid_trials = 0
    skipped_empty = 0
    skipped_unmatched = 0
    skipped_no_behavior = 0

    for trial_name, trial in experiment.trials.items():
        if hasattr(trial, 'behaviors') and not trial.behaviors.empty:
            df = trial.behaviors.copy()

            matched_key = None
            for key in subject_to_behavior:
                if trial_name.lower().startswith(key.lower()):
                    matched_key = key
                    break

            if not matched_key:
                skipped_unmatched += 1
                continue

            target_behavior = subject_to_behavior[matched_key]

            behavior_df = df[df["Behavior"].str.lower() == target_behavior]
            if behavior_df.empty:
                skipped_no_behavior += 1
                continue

            total_time = behavior_df["Duration (s)"].sum()

            # Group by subject type
            if matched_key.lower().startswith(("nn", "pp")):
                color = color_nn_pp
                group_label = "nn / pp"
            else:
                color = color_n_p
                group_label = "n / p"

            subject_data.append({
                "Trial": trial_name,
                "Group": group_label,
                "Total Investigation Time": total_time,
                "Color": color
            })
            valid_trials += 1
        else:
            skipped_empty += 1

    print(f"\nTotal trials: {total_trials}")
    print(f"Valid trials with matched behavior: {valid_trials}")
    print(f"Skipped (empty): {skipped_empty}, Skipped (no match): {skipped_unmatched}, Skipped (no behavior data): {skipped_no_behavior}")

    if not subject_data:
        print("No valid investigation data found for sniff cup assignments.")
        return

    data_df = pd.DataFrame(subject_data)

    num_subjects = len(data_df)
    print(f"Number of mice (data points) plotted: {num_subjects}")

    # --- Plot ---
    mean_time = data_df["Total Investigation Time"].mean()
    sem_time = data_df["Total Investigation Time"].sem()

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(0, mean_time, yerr=sem_time, capsize=10, color=bar_color,
           edgecolor='black', linewidth=3, width=0.6)

    for _, row in data_df.iterrows():
        ax.plot(0, row["Total Investigation Time"], 'o', color=row["Color"], markersize=10)

    ax.set_xticks([0])
    ax.set_xticklabels(["Long Term Sniff Cup"], fontsize=14)
    ax.set_ylabel("Total Investigation Time (s)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(-0.5, 0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', labelsize=14)

    # --- Legend ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_n_p,
                   markersize=10, label="Cohort 1 + 2"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_nn_pp,
                   markersize=10, label="Cohort 3")
    ]
    ax.legend(handles=legend_elements, title="Groups", loc='upper left', fontsize=10, title_fontsize=12)

    plt.tight_layout(pad=pad_inches)
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    plt.show()

# Bimodal Distribution Behavior - Plotting Separate Modes
def create_metadata_dataframe_with_agent_mapping(trial_data_with_ids, sniff_cup_csv_path):
    """
    Computes total time and average bout duration per subject × agent type,
    using mapping from sniff cup to agent. Preserves full trial IDs.
    
    Parameters:
    - trial_data_with_ids: List of (full_trial_id, DataFrame) tuples.
    - sniff_cup_csv_path: Path to CSV with sniff cup → agent mappings.

    Returns:
    - DataFrame with rows = full trial IDs, columns = metrics per agent type.
    """
    import pandas as pd
    import numpy as np

    # Load sniff cup assignment CSV
    assign_df = pd.read_csv(sniff_cup_csv_path)
    assign_df['Subject'] = assign_df['Subject'].str.lower()

    all_records = []

    for full_trial_id, df in trial_data_with_ids:  # full_trial_id is e.g., 'p5-240826-091418'
        subject_prefix = full_trial_id.split('-')[0].lower()

        mapping_row = assign_df[assign_df['Subject'] == subject_prefix]
        if mapping_row.empty:
            print(f"⚠️ No mapping found for subject '{subject_prefix}'. Skipping trial '{full_trial_id}'.")
            continue

        # Build sniff cup → agent type mapping
        cup_to_agent = {}
        for col in assign_df.columns:
            if 'sniff cup' in col.lower():
                cup_val = str(mapping_row.iloc[0][col]).strip().lower()
                cup_to_agent[col.lower()] = cup_val

        # Map each behavior (sniff cup X) to agent
        df = df.copy()
        df['Agent Type'] = df['Behavior'].str.lower().map(cup_to_agent)
        df = df[df['Agent Type'].notnull()]

        if df.empty:
            print(f"⚠️ No valid sniff cup behaviors after mapping for '{full_trial_id}'.")
            continue

        grouped = df.groupby('Agent Type')['Duration (s)'].agg(
            Total='sum',
            Event_Count='count'
        ).reset_index()

        row_data = {'Subject': full_trial_id}
        for _, row in grouped.iterrows():
            agent = row['Agent Type']
            total = row['Total']
            count = row['Event_Count']
            avg = total / count if count > 0 else 0

            row_data[f'Total_{agent}'] = total
            row_data[f'Avg_{agent}'] = avg

        all_records.append(row_data)

    result_df = pd.DataFrame(all_records).set_index('Subject').fillna(0)
    return result_df

def get_trial_dataframes_with_ids(experiment):
    """
    Given an Experiment object, return a list of (subject_id, DataFrame) tuples,
    where each DataFrame corresponds to the .behaviors of each trial.
    The subject_id is extracted from the trial name.
    """
    trial_dataframes = []

    for trial_id, trial in experiment.trials.items():
        # Extract subject ID (e.g., 'n1', 'pp3') from the trial name
        subject_id = trial_id.split('-')[0].lower()  # Convert to lowercase for consistency
        df = trial.behaviors

        # Only include if behaviors exist and are not empty
        if df is not None and not df.empty:
            trial_dataframes.append((subject_id, df.copy()))
        else:
            print(f"Skipping {trial_id}: no behavior data.")

    return trial_dataframes

def plot_investigation_by_agent(df,
                                 subjects_to_include=None,
                                 metric="Total",
                                 title='Investigation Time by Agent (Total)',
                                 ylabel='Investigation Time (s)',
                                 xlabel='Social Agent',
                                 custom_xtick_labels=['Empty', 'Short Term', 'Long Term', 'Novel'],
                                 custom_xtick_colors=['teal', 'blue', 'purple', 'orange'],
                                 ylim=None,
                                 bar_fill_color='white',
                                 bar_edge_color='black',
                                 bar_linewidth=3,
                                 bar_hatch='///',
                                 yticks_increment=None,
                                 figsize=(12, 7),
                                 pad_inches=0.1,
                                 save=False,
                                 save_name=None):
    """
    Aesthetic-matched plot: Investigation Time across agents.
    Styled to match `plot_y_across_bouts_ranks` exactly but keeps region legend and colored markers.
    """
    df = df.copy()

    # --- Normalize Subject index ---
    if 'Subject' in df.columns:
        df['Subject'] = df['Subject'].astype(str).str.lower()
        df.set_index('Subject', inplace=True)
    else:
        df.index = df.index.astype(str).str.lower()

    # --- Filter Subjects ---
    if subjects_to_include:
        subjects_to_include = [s.lower() for s in subjects_to_include]
        df = df[df.index.isin(subjects_to_include)]

    # --- Column Prep ---
    agents = ['nothing', 'short_term', 'long_term', 'novel']
    columns_to_plot = [f"{metric}_{agent}" for agent in agents]
    missing_cols = [col for col in columns_to_plot if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        return

    df_plot = df[columns_to_plot].copy()
    df_plot.columns = agents

    if df_plot.empty or df_plot.isnull().all().all():
        print("⚠️ No valid data available to plot. Check your subjects_to_include and metric selection.")
        return

    # --- Paired T-tests ---
    def perform_t_tests(df_vals):
        comparisons = {
            "nothing_vs_short_term": ("nothing", "short_term"),
            "nothing_vs_long_term": ("nothing", "long_term"),
            "nothing_vs_novel": ("nothing", "novel")
        }
        results = {}
        for key, (a1, a2) in comparisons.items():
            paired = df_vals[[a1, a2]].dropna()
            if len(paired) > 1:
                t_stat, p_value = ttest_rel(paired[a1], paired[a2])
                results[key] = {"t_stat": t_stat, "p_value": p_value}
        return results

    t_test_results = perform_t_tests(df_plot)

    # --- Mean & SEM ---
    mean_vals = df_plot.mean()
    sem_vals = df_plot.sem()

    fig, ax = plt.subplots(figsize=figsize)

    # --- Bar Plot ---
    ax.bar(df_plot.columns, mean_vals, yerr=sem_vals, capsize=6,
           color=bar_fill_color, edgecolor=bar_edge_color, linewidth=bar_linewidth,
           width=0.6, hatch=bar_hatch,
           error_kw=dict(elinewidth=2, capthick=2, zorder=5))

    # --- Lines + Colored Markers (No Borders) ---
    for subject_id, row in df_plot.iterrows():
        marker_color = '#15616F' if subject_id.startswith('n') else '#FFAF00' if subject_id.startswith('p') else 'gray'
        ax.plot(df_plot.columns, row.values, linestyle='-', color='gray', alpha=0.5, linewidth=2.5, zorder=1)
        ax.scatter(df_plot.columns, row.values, color=marker_color,
                   s=120, alpha=1.0, zorder=1)  # No edgecolor

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

    # --- Spines ---
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # --- Significance Markers ---
    if t_test_results:
        max_y = ax.get_ylim()[1]
        sig_y_offset = max_y * 0.05
        comparisons = {
            "nothing_vs_short_term": (0, 1),
            "nothing_vs_long_term": (0, 2),
            "nothing_vs_novel": (0, 3)
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

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='NAc',
               markerfacecolor='#15616F', markersize=12, markeredgewidth=0),
        Line2D([0], [0], marker='o', color='none', label='mPFC',
               markerfacecolor='#FFAF00', markersize=12, markeredgewidth=0)
    ]
    ax.legend(handles=legend_elements, title="Region", fontsize=20, title_fontsize=22,
              loc='upper right', frameon=True)

    # --- Save ---
    plt.tight_layout(pad=pad_inches)
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    plt.show()
