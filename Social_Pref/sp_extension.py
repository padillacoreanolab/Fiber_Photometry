# sp_extension.py
from __future__ import annotations
import re
from typing import Dict, Tuple, Iterable, Optional, List

import numpy as np
import pandas as pd

# =============================================================================
# Subject resolution (handles n*/p*, nn*/pp*, suffixes, punctuation)
# =============================================================================

_ALNUM = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

def _norm_subject(s: str) -> str:
    """Lowercase & strip whitespace (keeps characters)."""
    return str(s).strip().lower()

def _canon(s: str) -> str:
    """Lowercase and strip all non-alphanumerics (e.g., 'pp7-2409' -> 'pp72409')."""
    return _ALNUM.sub("", str(s).lower())

def resolve_subject_key(subj: str, mapping_keys: Iterable[str]) -> Optional[str]:
    """
    Resolve a trial subject to one of the assignment keys.
    Tries:
      (1) exact (lower/strip)
      (2) canonical (strip punctuation)
      (3) n/nn or p/pp + number variants
    """
    keys = list(mapping_keys)
    s_exact = _norm_subject(subj)
    if s_exact in keys:
        return s_exact

    s_canon = _canon(subj)
    for k in keys:
        if _canon(k) == s_canon:
            return k

    # match n/p or nn/pp + optional zeros + digits
    m = re.search(r"\b([np]{1,2})0*(\d+)\b", s_exact)
    if m:
        prefix = m.group(1).lower()
        num    = str(int(m.group(2)))  # normalize number
        candidates = [
            f"{prefix}{num}",
            (prefix*2)+num if len(prefix) == 1 else prefix[0]+num
        ]
        for cand in candidates:
            if cand in keys:
                return cand
    return None


# =============================================================================
# Agent normalization (handles spaces/hyphens/underscores/synonyms)
# =============================================================================

_WS = re.compile(r"[\s\-_]+")

def _norm_agent(s: Optional[str]) -> Optional[str]:
    """
    Canonicalize agent labels to tokens used downstream.
    Examples:
      'Short Term' / 'short-term' / 'short_term' / 'ST' -> 'short_term'
      'Long Term'  / 'long-term'  / 'long_term'  / 'LT' -> 'long_term'
      'Novel' -> 'novel'
      'Nothing' / 'acq' / 'acq st' / 'acq-st' -> 'nothing'
    """
    if s is None:
        return None
    x = str(s).strip().lower()
    x = _WS.sub("_", x)  # unify separators to underscores
    # common synonyms
    if x in {"short_term", "shortterm", "st"}:
        return "short_term"
    if x in {"long_term", "longterm", "lt"}:
        return "long_term"
    if x in {"novel"}:
        return "novel"
    if x in {"nothing", "acq", "acq_st", "acq-st", "acqst", "acq__st"}:
        return "nothing"
    return x  # already normalized enough


# =============================================================================
# Cup parsing (accepts 'sniff cup 3' or 'investigation cup 3' w/ flexible separators)
# =============================================================================

_CUP_RE = re.compile(
    r"(?:sniff|investigat\w*)\s*(?:-|_|\s)*cup\s*(\d+)\s*$",
    re.IGNORECASE
)

def infer_cup_from_behavior(behavior: str) -> Optional[int]:
    """Return cup index if behavior matches '(sniff|investigation) cup X'; else None."""
    if not isinstance(behavior, str):
        return None
    m = _CUP_RE.search(behavior.strip())
    return int(m.group(1)) if m else None


# =============================================================================
# Read assignments (Subject → {cup#: agent_token})
# =============================================================================

def read_cup_assignments(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Read the Social Preference cup-assignment CSV and build a mapping:
        subject -> {1: 'novel'|'short_term'|'long_term'|'nothing', ..., 4: ...}
    Returns (df, mapping).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    cup_cols = [c for c in df.columns if c.startswith("sniff cup")]
    if "subject" not in df.columns or len(cup_cols) != 4:
        raise ValueError("Expected columns: 'Subject', 'sniff cup 1'..'sniff cup 4'.")

    mapping: Dict[str, dict] = {}
    for _, row in df.iterrows():
        subj = _norm_subject(row["subject"])
        mapping[subj] = {}
        for c in cup_cols:
            cup_idx = int(c.split()[-1])
            mapping[subj][cup_idx] = _norm_agent(row[c])
    return df, mapping


# =============================================================================
# Behavior-column detection (robust to renamed/merged tables)
# =============================================================================

def _find_behavior_column(df: pd.DataFrame) -> str:
    """
    Return the column name that contains the behavior labels.
    Tolerates variants from BORIS exports and downstream merges.
    """
    if df is None or df.empty:
        raise ValueError("Trial.behaviors is empty or None")

    lowered  = {c.lower().strip(): c for c in df.columns}
    # priority list
    candidates = [
        "behavior", "behaviour", "event", "event_name", "event label", "label",
        "behavior ", " behaviour"
    ]
    for key in candidates:
        if key in lowered:
            return lowered[key]
    # fuzzy fallback
    for k, orig in lowered.items():
        if "behavior" in k or "behaviour" in k or "event" in k or "label" in k:
            return orig
    raise KeyError(f"Could not locate a behavior label column. Available: {list(df.columns)}")


# =============================================================================
# Annotate a Trial with Cup/Agent and store resolved subject
# =============================================================================

def annotate_trial_with_agents(trial_obj, subject_cup_map: Dict[str, dict]) -> None:
    """
    Adds 'Cup' and 'Agent' to trial_obj.behaviors.
    Stores trial_obj.resolved_subject_key so downstream uses the correct subject ID.
    """
    if getattr(trial_obj, "behaviors", None) is None or trial_obj.behaviors.empty:
        return

    resolved = resolve_subject_key(trial_obj.subject_name, subject_cup_map.keys())
    cup_map  = subject_cup_map.get(resolved)

    setattr(trial_obj, "resolved_subject_key", resolved)
    setattr(trial_obj, "cup_agent_map", cup_map)

    df = trial_obj.behaviors.copy()

    # detect behavior column robustly
    beh_col = _find_behavior_column(df)

    # infer Cup from that column
    beh_series = df[beh_col].astype(str)
    df["Cup"] = beh_series.apply(infer_cup_from_behavior)

    # map Agent if subject resolved; keep None otherwise
    if cup_map is not None:
        df["Agent"] = df["Cup"].apply(lambda c: cup_map.get(int(c)) if pd.notna(c) else None)
    else:
        df["Agent"] = None

    # normalize label AFTER parsing
    df.loc[df["Cup"].notna(), beh_col] = "Investigation"

    # ensure canonical 'Behavior' alias exists for downstream code
    if "Behavior" not in df.columns:
        df["Behavior"] = df[beh_col]

    trial_obj.behaviors = df


# =============================================================================
# First sniff per cup
# =============================================================================

def first_sniff_per_cup(experiment_obj, cup_map, metrics_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    For each trial, keep only the FIRST investigation event for each cup.
    Returns one row per (Subject, TrialName, Cup). Subject uses the resolved key.
    """
    if metrics_cols is None:
        metrics_cols = ['AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score']

    rows = []
    for trial_name, tr in experiment_obj.trials.items():
        annotate_trial_with_agents(tr, cup_map)
        df = getattr(tr, "behaviors", None)
        if df is None or df.empty:
            continue

        d = df[df['Cup'].notna()].copy()
        if d.empty:
            continue

        subj = getattr(tr, "resolved_subject_key", None) or _norm_subject(tr.subject_name)
        d['Subject']   = subj
        d['TrialName'] = trial_name

        d = d.sort_values('Event_Start')
        first = d.groupby(['Subject','TrialName','Cup'], as_index=False).first()

        extra = [c for c in (metrics_cols or []) if c in first.columns]
        cols  = ['Subject','TrialName','Cup','Agent','Behavior','Event_Start','Event_End','Duration (s)'] + extra
        rows.append(first[cols])

    if not rows:
        return pd.DataFrame(columns=['Subject','TrialName','Cup','Agent','Behavior','Event_Start','Event_End','Duration (s)'] + (metrics_cols or []))

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(['Subject','TrialName','Cup']).reset_index(drop=True)
    return out


# =============================================================================
# Plot-ready metadata (ALL metrics)
# =============================================================================

def first_sniff_da_metadata(
    experiment_obj,
    cup_map,
    behavior_label: str = "Investigation",
    desired_order = ("Acq-ST", "Short Term", "Long Term", "Novel"),
    agent_label_map = None,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert FIRST sniff-per-cup events into a 'metadata_df' that can be plotted.
    Returns: ['Subject','Bout','Behavior', <all present DA metrics>]
    """
    standard_metrics = ['AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score', 'Adjusted End']
    want_metrics = metrics if metrics is not None else standard_metrics

    first_df = first_sniff_per_cup(experiment_obj, cup_map, metrics_cols=want_metrics)
    if first_df.empty:
        return pd.DataFrame(columns=["Subject", "Bout", "Behavior"] + want_metrics)

    if agent_label_map is None:
        agent_label_map = {
            "short_term": "Short Term",
            "long_term":  "Long Term",
            "novel":      "Novel",
            "nothing":    "Acq-ST",
        }

    df = first_df.copy()
    df["Agent_norm"] = df["Agent"].apply(_norm_agent)
    df["Bout"] = df["Agent_norm"].map(agent_label_map)
    df = df[df["Bout"].notna()].copy()

    # Collect metric columns that exist; add NaN for requested-but-missing
    metric_cols = [m for m in want_metrics if m in df.columns]
    for m in want_metrics:
        if m not in df.columns:
            df[m] = np.nan
            metric_cols.append(m)

    cols = ["Subject", "Bout", "Behavior"] + metric_cols
    meta = pd.DataFrame({
        "Subject":  df["Subject"].values,
        "Bout":     df["Bout"].values,
        "Behavior": behavior_label,
        **{m: df[m].values for m in metric_cols}
    })[cols]

    if desired_order:
        meta["Bout"] = pd.Categorical(meta["Bout"], categories=list(desired_order), ordered=True)
        meta = meta.sort_values(["Subject","Bout"]).reset_index(drop=True)
    return meta


from typing import Optional, List
import pandas as pd
import numpy as np

def all_sniff_da_metadata(
    experiment_obj,
    cup_map,
    behavior_label: str = "Investigation",
    desired_order = ("Acq-ST", "Short Term", "Long Term", "Novel"),
    agent_label_map: Optional[dict] = None,
    metrics: Optional[List[str]] = None,
    *,
    # optional: compute metrics on the fly if a trial is missing them
    ensure_metrics: bool = False,
    mode: str = "standard",
    use_max_length: bool = False,
    max_bout_duration: float = 4.0,
    pre_time: float = 4.0,
    post_time: float = 15.0,
) -> pd.DataFrame:
    """
    Build a metadata_df for **ALL** investigation events (not just the first).
    Output columns: ['Subject','Bout','Behavior', <DA metric columns...>]
    Multiple rows per Subject×Bout are expected when multiple investigations occur.
    """
    # default metric set
    standard_metrics = ['AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score', 'Adjusted End']
    want_metrics = metrics if metrics is not None else standard_metrics

    # default label map
    if agent_label_map is None:
        agent_label_map = {
            "short_term": "Short Term",
            "long_term":  "Long Term",
            "novel":      "Novel",
            "nothing":    "Acq-ST",
        }

    rows = []
    for trial_name, tr in experiment_obj.trials.items():
        # add Cup/Agent columns (robust behavior-column detection lives inside this)
        annotate_trial_with_agents(tr, cup_map)
        df = getattr(tr, "behaviors", None)
        if df is None or df.empty:
            continue

        # compute DA metrics on the fly if any requested metric is missing
        if ensure_metrics and any(m not in df.columns for m in want_metrics):
            if hasattr(tr, "compute_da_metrics"):
                tr.compute_da_metrics(
                    use_max_length=use_max_length,
                    max_bout_duration=max_bout_duration,
                    mode=mode,
                    pre_time=pre_time,
                    post_time=post_time,
                )
                df = tr.behaviors  # refresh

        d = df[df["Cup"].notna()].copy()
        if d.empty:
            continue

        # resolved subject ID if available
        subj = getattr(tr, "resolved_subject_key", None) or str(tr.subject_name).strip().lower()
        d["Subject"] = subj

        # map Agent token -> Bout display label
        d["Agent_norm"] = d["Agent"].apply(_norm_agent)
        d["Bout"] = d["Agent_norm"].map(agent_label_map)

        # keep only rows we can label on x-axis
        d = d[d["Bout"].notna()].copy()
        if d.empty:
            continue

        # ensure all requested metric columns exist
        for m in want_metrics:
            if m not in d.columns:
                d[m] = np.nan

        # collect rows for metadata_df
        keep = ["Subject", "Bout", "Behavior"] + want_metrics
        rows.append(d[keep])

    if not rows:
        return pd.DataFrame(columns=["Subject", "Bout", "Behavior"] + want_metrics)

    meta = pd.concat(rows, ignore_index=True)

    # enforce desired plotting order
    if desired_order:
        meta["Bout"] = pd.Categorical(meta["Bout"], categories=list(desired_order), ordered=True)
        meta = meta.sort_values(["Subject", "Bout"]).reset_index(drop=True)

    # set a uniform behavior label
    meta["Behavior"] = behavior_label
    return meta


def average_within_subject_per_bout(
    metadata_df: pd.DataFrame,
    y_cols: Optional[List[str]] = None,
    behavior_label: str = "Investigation",
    desired_order = ("Acq-ST", "Short Term", "Long Term", "Novel"),
) -> pd.DataFrame:
    """
    Collapse multiple investigations to a single row per (Subject, Bout) by averaging.
    Returns a DataFrame shaped like your plotting function expects:
      ['Subject','Bout','Behavior', <y_cols...>]
    """
    if y_cols is None:
        # default to common DA metrics if not specified
        y_cols = ['AUC', 'Max Peak', 'Time of Max Peak', 'Mean Z-score', 'Adjusted End']
        y_cols = [c for c in y_cols if c in metadata_df.columns]

    # keep only columns we need
    cols = ["Subject", "Bout", "Behavior"] + y_cols
    df = metadata_df[cols].copy()

    # average within Subject × Bout
    agg = (
        df.groupby(["Subject", "Bout"], as_index=False)[y_cols]
          .mean()  # mean across all investigations for that subject×bout
    )

    # add a uniform Behavior label
    agg["Behavior"] = behavior_label

    # enforce plotting order
    if desired_order:
        agg["Bout"] = pd.Categorical(agg["Bout"], categories=list(desired_order), ordered=True)
        agg = agg.sort_values(["Subject", "Bout"]).reset_index(drop=True)

    return agg