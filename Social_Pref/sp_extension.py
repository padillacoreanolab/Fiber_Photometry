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
 
# Long Term Bar Graph
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

# Long Term Bar Graph with Colors
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

# Dopamine
def prep_combined_da_metrics(experiment, sniff_cup_csv_path, metric_list=None, first_only=False):

    # Normalize behavior label spacing
    def normalize_behavior_label(label):
        return re.sub(r'\s+', ' ', label.strip().lower().replace('\u00a0', ' '))

    assign_df = pd.read_csv(sniff_cup_csv_path)
    assign_df['Subject'] = assign_df['Subject'].astype(str).str.lower()

    # Build subject -> behavior name -> agent identity mapping
    subject_to_behavior_to_agent = {}
    for _, row in assign_df.iterrows():
        subj = row['Subject']
        subject_to_behavior_to_agent[subj] = {}
        for col in row.index:
            col_norm = normalize_behavior_label(str(col))
            if col_norm.startswith("sniff cup"):
                agent_label = normalize_behavior_label(str(row[col]))
                subject_to_behavior_to_agent[subj][col_norm] = agent_label

    all_rows = []

    for trial_name, trial in experiment.trials.items():
        if not hasattr(trial, 'behaviors') or trial.behaviors.empty:
            continue

        df = trial.behaviors.copy()
        df['Behavior'] = df['Behavior'].astype(str).apply(normalize_behavior_label)

        subject_id = trial_name.lower()

        if subject_id not in subject_to_behavior_to_agent:
            continue

        mapping = subject_to_behavior_to_agent[subject_id]

        # Keep only sniff cup behaviors
        df = df[df["Behavior"].str.startswith("sniff cup")]

        # Map behaviors to agents
        df["Agent"] = df["Behavior"].apply(lambda b: mapping.get(b))
        df["Subject"] = subject_id
        df["Trial"] = trial_name

        unmatched = df[df["Agent"].isna()]
        if not unmatched.empty:
            print(f"‼️ Unmatched behaviors for subject '{subject_id}':")
            print("Behaviors that failed to map:", unmatched["Behavior"].unique())
            print("Available mapping keys:", list(mapping.keys()))

        df = df.dropna(subset=["Agent"])

        # Choose metrics
        known_cols = ["Behavior", "Agent", "Subject", "Trial"]
        if metric_list:
            metric_cols = [m for m in metric_list if m in df.columns]
        else:
            metric_cols = [c for c in df.columns if c not in known_cols and pd.api.types.is_numeric_dtype(df[c])]

        if not metric_cols:
            continue

        df = df[["Subject", "Agent"] + metric_cols]

        if first_only:
            df = df.groupby(["Subject", "Agent"], as_index=False).first()

        all_rows.append(df)

    if not all_rows:
        print("⚠️ No rows added to DataFrame. Check if behavior labels match and mapping keys are clean.")
        print(f"Subjects in experiment: {list(experiment.trials.keys())}")
        print(f"Subjects in assignments file: {assign_df['Subject'].tolist()}")
        print("Sample mapping dictionary:")
        for subj, mapping in subject_to_behavior_to_agent.items():
            print(f"{subj} -> {mapping}")
        return pd.DataFrame()

    combined_df = pd.concat(all_rows, ignore_index=True)

    # --- Aggregate by Subject-Agent pair ---
    if first_only:
        grouped = combined_df  # already one row per subject-agent
    else:
        grouped = combined_df.groupby(["Subject", "Agent"], as_index=False)[metric_cols].mean()

    # --- Ensure each subject has all 4 agent rows ---
    all_agents = ['nothing', 'short_term', 'long_term', 'novel']
    all_subjects = sorted(grouped['Subject'].unique())
    full_index = pd.MultiIndex.from_product([all_subjects, all_agents], names=['Subject', 'Agent'])

    final_df = (
        grouped.set_index(['Subject', 'Agent'])
               .reindex(full_index)
               .fillna(0)
               .reset_index()
    )

    print(f"✅ Final DA metrics DataFrame created with {len(final_df)} rows from {len(all_subjects)} subjects.")
    return final_df

def create_da_metrics_dataframe(trial_data, behavior="Investigation", desired_bouts=None):
    """
    Extracts DA metrics per bout per subject for a specified behavior.

    Parameters
    ----------
    trial_data : dict
        Dictionary {subject_id: DataFrame} of behavior data.
    behavior : str
        Behavior to filter (e.g., 'Investigation').
    desired_bouts : list or None
        Bouts to keep (e.g., ['sniff cup 1', ...]); if None, keep all.

    Returns
    -------
    pd.DataFrame
        Summary metrics per subject × bout.
    """
    metric_rows = []

    behavior = behavior.strip().lower()
    desired_bouts = [b.strip().lower() for b in desired_bouts] if desired_bouts else None

    for subject_id, df in trial_data.items():
        df = df.copy()

        # Normalize string columns
        df["Bout"] = df["Bout"].astype(str).str.strip().str.lower()
        df["Behavior"] = df["Behavior"].astype(str).str.strip().str.lower()

        bouts_to_check = desired_bouts if desired_bouts else df["Bout"].unique()

        for bout in bouts_to_check:
            df_bout = df[df["Bout"].str.contains(bout, na=False)]
            df_behavior = df_bout[df_bout["Behavior"].str.contains(behavior, na=False)]

            if df_behavior.empty:
                auc_val = 0
                max_peak_val = 0
                mean_z_val = 0
            else:
                df_behavior["AUC"] = pd.to_numeric(df_behavior["AUC"], errors='coerce')
                df_behavior["Max Peak"] = pd.to_numeric(df_behavior["Max Peak"], errors='coerce')
                df_behavior["Mean Z-score"] = pd.to_numeric(df_behavior["Mean Z-score"], errors='coerce')

                auc_val = df_behavior["AUC"].mean(skipna=True)
                max_peak_val = df_behavior["Max Peak"].mean(skipna=True)
                mean_z_val = df_behavior["Mean Z-score"].mean(skipna=True)

                auc_val = 0 if pd.isna(auc_val) else auc_val
                max_peak_val = 0 if pd.isna(max_peak_val) else max_peak_val
                mean_z_val = 0 if pd.isna(mean_z_val) else mean_z_val

            metric_rows.append({
                "Subject": subject_id,
                "Bout": bout,
                "Behavior": behavior,
                "AUC": auc_val,
                "Max Peak": max_peak_val,
                "Mean Z-score": mean_z_val
            })

    return pd.DataFrame(metric_rows)

def plot_dopamine_bar_plots(precomputed_df, 
             metric_name="Mean Z-score", 
             title="Combined DA Metrics", 
             ylabel="DA Metric", 
             xlabel="Agent", 
             custom_xtick_labels=None, 
             custom_xtick_colors=None, 
             ylim=None, 
             bar_color="#00B7D7", 
             yticks_increment=None, 
             figsize=(14, 8), 
             pad_inches=0.1,
             save=False,
             save_name=None,
             subjects_to_include=None,
             highlight_subject=None):
    """
    Plots DA metrics across agents ("nothing", "short_term", "long_term", "novel")
    with subject-level spaghetti plots and paired t-tests.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel

    fixed_order = ["nothing", "short_term", "long_term", "novel"]
    bar_positions = np.arange(len(fixed_order))

    def perform_all_pairwise_t_tests(pivot_df):
        results = {}
        bout_names = pivot_df.columns.tolist()
        for i in range(len(bout_names)):
            for j in range(i + 1, len(bout_names)):
                bout1, bout2 = bout_names[i], bout_names[j]
                paired_df = pivot_df[[bout1, bout2]].dropna()
                if len(paired_df) > 1:
                    t_stat, p_value = ttest_rel(paired_df[bout1], paired_df[bout2])
                    results[f"{bout1} vs {bout2}"] = {"t_stat": t_stat, "p_value": p_value}
        return results

    df = precomputed_df.copy()

    # --- Filter by Subject ---
    if subjects_to_include is not None:
        subjects_to_include = [s.lower() for s in subjects_to_include]
        df['Subject'] = df['Subject'].astype(str).str.lower()
        df = df[df['Subject'].isin(subjects_to_include)]

    if df.empty:
        print("⚠️ No data to plot after filtering.")
        return

    # --- Pivot by Subject x Agent ---
    try:
        pivot_df = df.pivot(index="Subject", columns="Agent", values=metric_name)
    except Exception as e:
        print("Error pivoting data:", e)
        return

    pivot_df = pivot_df.reindex(columns=fixed_order)

    # --- Summary Stats ---
    stats = pivot_df.agg(['mean', 'sem']).T.reset_index()
    stats.columns = ['Agent', 'mean', 'sem']
    stats = stats.set_index('Agent').reindex(fixed_order).reset_index()

    means = stats['mean'].values
    sems = stats['sem'].values

    # --- Paired T-tests ---
    t_test_results = perform_all_pairwise_t_tests(pivot_df)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Bars
    ax.bar(
        bar_positions,
        means,
        yerr=sems,
        capsize=10,
        color=bar_color,
        edgecolor='black',
        linewidth=5,
        width=0.6
    )

    # Spaghetti lines (gray for all, black for highlight)
    for subject_id, row in pivot_df.iterrows():
        if highlight_subject and subject_id.lower() == highlight_subject.lower():
            ax.plot(bar_positions, row.values, linestyle='-', color='black', linewidth=4, zorder=3)
            ax.scatter(bar_positions, row.values, facecolors='black', edgecolors='black', 
                       s=160, linewidths=2, zorder=4)
        else:
            ax.plot(bar_positions, row.values, linestyle='-', color='gray', alpha=0.5, linewidth=2.5, zorder=1)
            ax.scatter(bar_positions, row.values, facecolors='none', edgecolors='gray', 
                       s=120, linewidths=3, zorder=2)

    # Labels
    ax.set_ylabel(ylabel, fontsize=35, labelpad=12)
    ax.set_xlabel(xlabel, fontsize=35, labelpad=12)
    if title:
        ax.set_title(title, fontsize=28)

    ax.set_xticks(bar_positions)
    xtick_labels = custom_xtick_labels if custom_xtick_labels else ["Empty", "Short Term", "Long Term", "Novel"]
    ax.set_xticklabels(xtick_labels, fontsize=35)
    if custom_xtick_colors:
        for tick, color in zip(ax.get_xticklabels(), custom_xtick_colors):
            tick.set_color(color)

    ax.tick_params(axis='y', labelsize=35)
    ax.tick_params(axis='x', labelsize=35)

    # Y-limits
    if ylim:
        ax.set_ylim(ylim)
    else:
        all_vals = np.concatenate([pivot_df.values.flatten(), means])
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

    # --- Print T-tests ---
    print(f"\nPlotted data from {pivot_df.shape[0]} subject(s).")
    if t_test_results:
        print("\nPaired t-test results (all agent comparisons):")
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

def get_all_per_event_df(experiment, sniff_cup_csv_path, metric_list=None):
    import re

    def normalize_label(label):
        return re.sub(r'\s+', ' ', str(label).strip().lower().replace('\u00a0', ' '))

    # Load mapping from cup -> agent
    assign_df = pd.read_csv(sniff_cup_csv_path)
    assign_df['Subject'] = assign_df['Subject'].str.lower()
    
    subj_map = {}
    for _, row in assign_df.iterrows():
        subj = row['Subject']
        subj_map[subj] = {}
        for col in row.index:
            col_norm = normalize_label(col)
            if col_norm.startswith("sniff cup"):
                agent = normalize_label(row[col])
                subj_map[subj][col_norm] = agent

    all_rows = []
    for trial_name, trial in experiment.trials.items():
        if not hasattr(trial, 'behaviors') or trial.behaviors.empty:
            continue

        df = trial.behaviors.copy()
        df['Behavior'] = df['Behavior'].astype(str).apply(normalize_label)

        subject_id = trial_name.lower()
        if subject_id not in subj_map:
            continue

        df = df[df['Behavior'].str.startswith("sniff cup")]
        df['Agent'] = df['Behavior'].apply(lambda b: subj_map[subject_id].get(b))
        df['Subject'] = subject_id
        df['Trial'] = trial_name

        df = df.dropna(subset=["Agent"])
        
        known = ['Subject', 'Agent', 'Trial', 'Behavior', 'Event_Start']
        if metric_list:
            cols = [col for col in metric_list if col in df.columns]
        else:
            cols = [c for c in df.columns if c not in known and pd.api.types.is_numeric_dtype(df[c])]

        df = df[['Subject', 'Agent', 'Trial', 'Event_Start'] + cols]
        all_rows.append(df)

    final_df = pd.concat(all_rows, ignore_index=True)
    print(f"✅ Per-event DataFrame created with {len(final_df)} rows from {final_df['Subject'].nunique()} subjects.")
    return final_df

def plot_peak_by_agent_from_df(
    df,
    sniff_cup_csv_path=None,              # optional if Agent column is already present
    selected_agents=None,                # e.g. ['novel', 'short_term']
    n_subsequent_investigations=3,
    peak_col="Max Peak",
    metric_type='slope',
    figsize=(14, 8),
    line_order=None,
    custom_colors=None,
    custom_legend_labels=None,
    custom_xtick_labels=None,
    ylim=None,
    ytick_increment=None,
    xlabel="Investigation Index",
    ylabel="Avg Max Peak",
    subjects_to_include=None,            # ✅ MISSING COMMA FIXED HERE
    plot_title="Average Peak per Agent",
    save=False,
    save_name="agent_peak_plot.png"
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    from scipy.optimize import curve_fit

    def exponential_decay(x, A, B, tau):
        return A + B * np.exp(-x / tau)

    def normalize_label(label):
        import re
        return re.sub(r'\s+', ' ', str(label).strip().lower().replace('\u00a0', ' '))

    def create_mapping(sniff_cup_csv_path):
        assign_df = pd.read_csv(sniff_cup_csv_path)
        assign_df['Subject'] = assign_df['Subject'].astype(str).str.lower()
        subject_to_behavior_to_agent = {}
        for _, row in assign_df.iterrows():
            subj = row['Subject']
            subject_to_behavior_to_agent[subj] = {}
            for col in row.index:
                col_norm = normalize_label(col)
                if col_norm.startswith("sniff cup"):
                    agent_label = normalize_label(row[col])
                    subject_to_behavior_to_agent[subj][col_norm] = agent_label
        return subject_to_behavior_to_agent

    df = df.copy()

    # --- Optional agent mapping ---
    if "Agent" not in df.columns:
        if sniff_cup_csv_path is None:
            raise ValueError("You must provide either an 'Agent' column or a sniff_cup_csv_path.")

        mapping = create_mapping(sniff_cup_csv_path)

        def get_agent(row):
            subj = str(row['Subject']).lower()
            bout = str(row['Bout']).lower()
            if '-' not in bout:
                return None
            cup_number = bout.split('-')[1]
            behavior = f"sniff cup {cup_number}"
            return mapping.get(subj, {}).get(behavior)

        df['Agent'] = df.apply(get_agent, axis=1)

    # --- Filter agents ---
    if selected_agents:
        df = df[df['Agent'].isin(selected_agents)]

    # --- Subject filtering ---
    if subjects_to_include:
        subjects_to_include = [s.lower() for s in subjects_to_include]
        df['Subject'] = df['Subject'].astype(str).str.lower()
        df = df[df['Subject'].isin(subjects_to_include)]

    # --- Investigation indexing ---
    df.sort_values(["Subject", "Agent", "Event_Start"], inplace=True)
    df["InvestigationIndex"] = df.groupby(["Subject", "Agent"]).cumcount() + 1
    df = df[df["InvestigationIndex"] <= n_subsequent_investigations]

    # --- Aggregate ---
    agg_df = (
        df.groupby(["Agent", "InvestigationIndex"], as_index=False)
        .agg(
            SubjectCount=("Subject", "nunique"),
            AvgPeak=(peak_col, "mean")
        )
    )

    # --- Plotting ---
    if custom_colors is None:
        custom_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(5)
    ax.spines["bottom"].set_linewidth(5)
    ax.tick_params(axis="both", which="major", labelsize=48)
    ax.tick_params(axis="y", labelsize=35)

    metrics_dict = {}
    unique_agents = line_order if line_order else sorted(agg_df["Agent"].dropna().unique())

    for i, agent in enumerate(unique_agents):
        df_line = agg_df[agg_df["Agent"] == agent].copy()
        df_line.sort_values("InvestigationIndex", inplace=True)

        x_vals = df_line["InvestigationIndex"].values
        y_vals = df_line["AvgPeak"].values

        if len(x_vals) == 0 or len(y_vals) == 0:
            print(f"Skipping agent '{agent}' due to no data.")
            continue

        if metric_type.lower() == 'slope':
            slope, _, _, _, _ = linregress(x_vals, y_vals)
            metrics_dict[agent] = slope
            metric_label = f"slope: {slope:.3f}"
        elif metric_type.lower() == 'decay':
            try:
                p0 = (np.min(y_vals), np.max(y_vals)-np.min(y_vals), 1.0)
                popt, _ = curve_fit(exponential_decay, x_vals, y_vals, p0=p0)
                tau = popt[2]
                metrics_dict[agent] = tau
                metric_label = f"decay: {tau:.3f}"
            except RuntimeError:
                metrics_dict[agent] = np.nan
                metric_label = "decay: N/A"
        else:
            raise ValueError("metric_type must be 'slope' or 'decay'.")

        legend_label = custom_legend_labels[i] if custom_legend_labels and i < len(custom_legend_labels) else agent
        legend_label += f" ({metric_label}, n={df_line['SubjectCount'].max()})"

        color = custom_colors[i % len(custom_colors)]
        ax.plot(
            x_vals, y_vals,
            marker='o', linestyle='-',
            color=color,
            linewidth=5, markersize=30,
            label=legend_label
        )

    ax.set_xlabel(xlabel, fontsize=35, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=35, labelpad=12)

    if ylim is not None:
        ax.set_ylim(ylim)
        if ytick_increment is not None:
            ticks = np.arange(ylim[0], ylim[1] + ytick_increment, ytick_increment)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:.0f}" if t.is_integer() else f"{t:.1f}" for t in ticks], fontsize=35)

    if custom_xtick_labels:
        ax.set_xticks(np.arange(1, len(custom_xtick_labels) + 1))
        ax.set_xticklabels(custom_xtick_labels, fontsize=35)
    else:
        x_vals = sorted(agg_df["InvestigationIndex"].unique())
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(x) for x in x_vals], fontsize=35)

    if plot_title:
        ax.set_title(plot_title, fontsize=24)

    ax.legend(fontsize=26)
    plt.tight_layout()

    if save:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches='tight')

    plt.show()

    print(f"\n=== Computed Metric ({metric_type.upper()}): ===")
    for agent, val in metrics_dict.items():
        print(f"Agent: {agent}, {metric_type} = {val:.3f}")

    return agg_df



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

    # Load sniff cup assignment CSV
    assign_df = pd.read_csv(sniff_cup_csv_path)
    assign_df['Subject'] = assign_df['Subject'].str.strip().str.lower()

    all_records = []

    for full_trial_id, df in trial_data_with_ids:
        subject_prefix = full_trial_id.lower().strip().split('-')[0]

        # Match first subject that starts with the prefix (e.g., 'pp3')
        matching_rows = assign_df[assign_df['Subject'].str.startswith(subject_prefix)]
        if matching_rows.empty:
            print(f"⚠️ No mapping found for subject '{subject_prefix}'. Skipping trial '{full_trial_id}'.")
            continue

        mapping_row = matching_rows.iloc[0]  # take first match if multiple
        cup_to_agent = {
            col.lower().strip(): str(mapping_row[col]).strip().lower()
            for col in assign_df.columns if 'sniff cup' in col.lower()
        }

        # Map behaviors in DataFrame to agent types
        df = df.copy()
        df['Behavior'] = df['Behavior'].str.strip().str.lower()
        df['Agent Type'] = df['Behavior'].map(cup_to_agent)
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

    if not all_records:
        print("❌ No records were created. Check subject matching.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_records)
    result_df = result_df.set_index('Subject').fillna(0)
    return result_df

def get_trial_dataframes(experiment):
    """
    Given an Experiment object, return a dictionary where:
    - Keys are subject IDs (Trial.subject_name).
    - Values are DataFrames corresponding to the behaviors of each trial.
    """
    trial_data = {}
    for trial in experiment.trials.values():
        subject_id = trial.subject_name
        trial_data[subject_id] = trial.behaviors
    return trial_data

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
                                 save_name=None,
                                 legend_loc='upper right'):  # 🆕 New parameter
    """
    Aesthetic-matched plot: Investigation Time across agents.
    Styled to match `plot_y_across_bouts_ranks` exactly but keeps region legend and colored markers.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_rel

    df = df.copy()

    if 'Subject' in df.columns:
        df['Subject'] = df['Subject'].astype(str).str.lower()
        df.set_index('Subject', inplace=True)
    else:
        df.index = df.index.astype(str).str.lower()

    if subjects_to_include:
        subjects_to_include = [s.lower() for s in subjects_to_include]
        df = df[df.index.isin(subjects_to_include)]

    agents = ['nothing', 'short_term', 'long_term', 'novel']
    columns_to_plot = [f"{metric}_{agent}" for agent in agents]
    missing_cols = [col for col in columns_to_plot if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        return

    df_plot = df[columns_to_plot].copy()
    df_plot.columns = agents

    if df_plot.empty or df_plot.isnull().all().all():
        print("⚠️ No valid data available to plot.")
        return

    # Paired t-tests
    def perform_t_tests(df_vals):
        from itertools import combinations
        results = {}
        for a1, a2 in combinations(df_vals.columns, 2):
            paired = df_vals[[a1, a2]].dropna()
            if len(paired) > 1:
                t_stat, p_val = ttest_rel(paired[a1], paired[a2])
                results[f"{a1} vs {a2}"] = p_val
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

    # --- Lines + Colored Markers with black edge
    for subject_id, row in df_plot.iterrows():
        marker_color = '#15616F' if subject_id.startswith('n') else '#FFAF00' if subject_id.startswith('p') else 'gray'
        ax.plot(df_plot.columns, row.values, linestyle='-', color='gray', alpha=0.5, linewidth=2.5, zorder=1)
        ax.scatter(df_plot.columns, row.values, color=marker_color,
                   s=160, alpha=1.0, zorder=2, edgecolors='black', linewidth=2.5)

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

    if yticks_increment:
        y_min, y_max = ax.get_ylim()
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + yticks_increment, yticks_increment)
        ax.set_yticks(y_ticks)

    # Spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='NAc',
               markerfacecolor='#15616F', markeredgecolor='black', markersize=12, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='none', label='mPFC',
               markerfacecolor='#FFAF00', markeredgecolor='black', markersize=12, markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, title="Region", fontsize=20, title_fontsize=22,
              loc=legend_loc, frameon=True)

    plt.tight_layout(pad=pad_inches)
    if save:
        if save_name is None:
            raise ValueError("save_name must be provided if save is True.")
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=pad_inches)

    plt.show()

    # --- Print t-test results ---
    print("\nPaired t-test results (across agents):")
    for comparison, p in t_test_results.items():
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = "ns"
        print(f"{comparison}: p = {p:.4f} ({stars})")