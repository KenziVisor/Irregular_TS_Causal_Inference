from __future__ import annotations

import argparse
import os
import pickle
import warnings
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from preprocess_mimic_iii_large_contract import canonicalize_stay_id_series


# ============================================================
# Config
# ============================================================
DATASET_MODEL = "physionet"
LATENT_TAGS_PATH = "../../data/latent_tags.csv"
PHYSIONET_PKL_PATH = "../../data/processed/physionet2012_ts_oc_ids.pkl"
GRAPH_PKL_PATH = "../../data/causal_graph.pkl"

OUTCOME_COL = "in_hospital_mortality"
GRAPH_OUTCOME_NODE = "Death"

PHYSIONET_TREATMENTS = [
    "Severity", "Shock", "RespFail", "RenalFail", "HepFail", "HemeFail",
    "Inflam", "NeuroFail", "CardInj", "Metab"
]
MIMIC_TREATMENTS = [
    "Severity", "Inflammation", "Shock", "RespFail", "RenalDysfunction",
    "HepaticDysfunction", "CoagDysfunction", "NeuroDysfunction",
    "CardiacInjury", "MetabolicDerangement",
]
TREATMENTS = list(PHYSIONET_TREATMENTS)

OUTPUT_DIR = "./matching_outputs"
SEED = 42
DOWN_SAMPLE = False
USE_EXPANDED_SAFE_CONFOUNDERS = True

# Matching-specific globals
max_dist = 2                  # maximum allowed Hamming distance
MIN_MATCHED_PAIRS = 30        # absolute minimum number of matched pairs
MIN_MATCH_RATE = 0.50         # minimum fraction of treated units that should be matched
MATCH_WITH_REPLACEMENT = False

# Optional: keep only binary/categorical confounders for Hamming matching.
# If False, numeric variables are binarized / discretized automatically.
REQUIRE_BINARY_CONF = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate matched-pair causal effects for latent clinical treatments."
    )
    parser.add_argument(
        "--model",
        choices=["physionet", "mimic"],
        default=DATASET_MODEL,
        help=f"Dataset selector for path defaults. Default: {DATASET_MODEL}",
    )
    parser.add_argument("--latent-tags-path", default=None)
    parser.add_argument("--physionet-pkl-path", default=None)
    parser.add_argument("--graph-pkl-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def get_dataset_defaults(model: str) -> Dict[str, object]:
    if model == "physionet":
        return {
            "latent_tags_path": LATENT_TAGS_PATH,
            "physionet_pkl_path": PHYSIONET_PKL_PATH,
            "graph_pkl_path": GRAPH_PKL_PATH,
            "graph_outcome_node": "Death",
            "treatments": list(PHYSIONET_TREATMENTS),
        }
    if model == "mimic":
        return {
            "latent_tags_path": "mimiciii_latent_tags_output/latent_tags.csv",
            "physionet_pkl_path": "../data/processed/mimic_iii_ts_oc_ids.pkl",
            "graph_pkl_path": None,
            "graph_outcome_node": "InHospitalMortality",
            "treatments": list(MIMIC_TREATMENTS),
        }
    raise ValueError(f"Unsupported model: {model!r}")


def resolve_runtime_path(
    cli_value: str | None,
    default_value: str | None,
    field_name: str,
    *,
    must_exist: bool = True,
) -> str:
    raw_value = cli_value if cli_value is not None else default_value
    if raw_value is None:
        raise ValueError(
            f"{field_name} is not configured. Provide the matching CLI flag."
        )

    raw_value = raw_value.strip()
    if not raw_value:
        raise ValueError(f"{field_name} is empty. Provide a non-empty path.")

    resolved_path = os.path.abspath(os.path.expanduser(raw_value))
    if must_exist:
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"{field_name} does not exist: {resolved_path}")
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"{field_name} is not a file: {resolved_path}")
    elif os.path.exists(resolved_path) and not os.path.isdir(resolved_path):
        raise NotADirectoryError(
            f"{field_name} must be a directory path: {resolved_path}"
        )

    return resolved_path


# ============================================================
# Data loading
# ============================================================
def load_physionet_pickle(path: str):
    with open(path, "rb") as f:
        ts, oc, ts_ids = pickle.load(f)
    return ts, oc, ts_ids


def load_graph(path: str) -> nx.DiGraph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Loaded graph is not a networkx.DiGraph")
    return G


def build_background_features(
    ts: pd.DataFrame,
    dataset_model: str | None = None,
) -> pd.DataFrame:
    """
    Build patient-level observed background covariates from ts.
    PhysioNet preprocessing converts ICUType into ICUType_1..ICUType_4.
    MIMIC does not guarantee those columns, so only keep what is actually present.
    """
    current_model = DATASET_MODEL if dataset_model is None else dataset_model
    df = ts.copy().sort_values(["ts_id", "minute"])

    keep_vars = ["Age", "Gender", "Weight"]
    if current_model == "physionet":
        keep_vars += ["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]
    else:
        available_variables = set(df["variable"].astype(str).tolist())
        keep_vars += [
            col for col in ["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]
            if col in available_variables
        ]
    df = df[df["variable"].isin(keep_vars)].copy()

    first_vals = (
        df.groupby(["ts_id", "variable"], as_index=False)
          .first()[["ts_id", "variable", "value"]]
    )

    bg = first_vals.pivot(index="ts_id", columns="variable", values="value").reset_index()

    if current_model == "physionet":
        for col in ["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]:
            if col not in bg.columns:
                bg[col] = 0.0

    return bg


def load_analysis_dataframe(
    latent_tags_path: str,
    physionet_pkl_path: str,
) -> pd.DataFrame:
    print(f"[load_analysis_dataframe] Loading latent tags from: {latent_tags_path}")
    latent_df = pd.read_csv(latent_tags_path)
    if "ts_id" in latent_df.columns:
        latent_df = latent_df.copy()
    elif DATASET_MODEL == "mimic" and "icustay_id" in latent_df.columns:
        latent_df = latent_df.rename(columns={"icustay_id": "ts_id"}).copy()
    else:
        raise ValueError(
            "Latent tags CSV must contain 'ts_id', or contain 'icustay_id' when "
            f"--model mimic is used. Source: {latent_tags_path}"
        )
    latent_df["ts_id"] = canonicalize_stay_id_series(latent_df["ts_id"])
    if latent_df["ts_id"].isna().any():
        raise ValueError("Latent tags contain missing ts_id values after canonicalization.")

    print(f"[load_analysis_dataframe] Loading processed pickle from: {physionet_pkl_path}")
    ts, oc, _ = load_physionet_pickle(physionet_pkl_path)
    ts = ts.copy()
    ts["ts_id"] = canonicalize_stay_id_series(ts["ts_id"])
    if ts["ts_id"].isna().any():
        raise ValueError("Processed pickle ts contains missing ts_id values after canonicalization.")
    oc = oc.copy()
    oc["ts_id"] = canonicalize_stay_id_series(oc["ts_id"])
    if oc["ts_id"].isna().any():
        raise ValueError("Processed pickle oc contains missing ts_id values after canonicalization.")
    if OUTCOME_COL not in oc.columns:
        raise ValueError(
            f"Processed pickle is missing outcome column '{OUTCOME_COL}'. "
            f"Available oc columns: {list(oc.columns)}"
        )

    if OUTCOME_COL in latent_df.columns:
        latent_df = latent_df.drop(columns=[OUTCOME_COL])

    oc_small = oc[["ts_id", OUTCOME_COL]].copy().drop_duplicates(subset=["ts_id"])
    oc_small["ts_id"] = canonicalize_stay_id_series(oc_small["ts_id"])
    if oc_small["ts_id"].isna().any():
        raise ValueError("Processed pickle oc_small contains missing ts_id values after canonicalization.")

    bg_df = build_background_features(ts, dataset_model=DATASET_MODEL)
    bg_df["ts_id"] = canonicalize_stay_id_series(bg_df["ts_id"])
    if bg_df["ts_id"].isna().any():
        raise ValueError("Background features contain missing ts_id values after canonicalization.")

    latent_bg_overlap = [
        column for column in bg_df.columns
        if column != "ts_id" and column in latent_df.columns
    ]
    if latent_bg_overlap:
        latent_df = latent_df.drop(columns=latent_bg_overlap)

    overlapping_ids = set(latent_df["ts_id"].dropna().tolist()) & set(oc_small["ts_id"].dropna().tolist())
    if oc_small.empty or not overlapping_ids:
        if DATASET_MODEL == "mimic":
            raise ValueError(
                "Processed MIMIC pickle and latent tags are misaligned: there are no "
                "overlapping ts_id values between latent tags and oc. A known cause is "
                "float-style stay identifiers such as '12345.0' versus '12345'."
            )
        raise ValueError(
            "Processed pickle and latent tags are misaligned: there are no overlapping "
            "ts_id values between latent tags and oc."
        )

    df = latent_df.merge(oc_small, on="ts_id", how="inner")
    df = df.merge(bg_df, on="ts_id", how="left")

    df = df.dropna(subset=[OUTCOME_COL]).copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    print(
        f"[load_analysis_dataframe] Built analysis dataframe: shape={df.shape} | "
        f"outcome_rate={df[OUTCOME_COL].mean():.4f}"
    )

    return df


def downsample_majority_label(
    df: pd.DataFrame,
    outcome_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Down-sample the majority class of outcome=0 so that the number of
    outcome=0 rows matches the number of outcome=1 rows.

    Keeps all outcome=1 rows.
    Randomly samples outcome=0 rows.
    """
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in dataframe")

    df_pos = df[df[outcome_col] == 1].copy()
    df_neg = df[df[outcome_col] == 0].copy()

    n_pos = len(df_pos)
    n_neg = len(df_neg)

    print(f"[Down-sample] Before: label1={n_pos}, label0={n_neg}")

    if n_pos == 0:
        print("[Down-sample] No positive rows found. Skipping down-sampling.")
        return df.copy()

    if n_neg <= n_pos:
        print("[Down-sample] label0 is not larger than label1. Skipping down-sampling.")
        return df.copy()

    df_neg_sampled = df_neg.sample(n=n_pos, random_state=seed, replace=False)

    df_balanced = pd.concat([df_pos, df_neg_sampled], axis=0)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    new_pos = int((df_balanced[outcome_col] == 1).sum())
    new_neg = int((df_balanced[outcome_col] == 0).sum())

    print(f"[Down-sample] After:  label1={new_pos}, label0={new_neg}")

    return df_balanced


# ============================================================
# Graph logic: backdoor-style confounder discovery
# ============================================================
def dataframe_columns_to_graph_nodes(
    available_columns: List[str],
    G: nx.DiGraph,
) -> Set[str]:
    graph_nodes = set(G.nodes)
    mapped = set()

    for col in available_columns:
        if col == "ts_id":
            continue
        if col == OUTCOME_COL:
            continue
        if col.startswith("ICUType_"):
            if "ICUType" in graph_nodes:
                mapped.add("ICUType")
            continue
        if col in graph_nodes:
            mapped.add(col)

    return mapped


def map_graph_node_to_dataframe_columns(
    node: str,
    available_columns: Set[str],
) -> List[str]:
    if node == "ICUType":
        return [c for c in ["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"] if c in available_columns]
    return [node] if node in available_columns else []


def get_allowed_adjustment_nodes(
    G: nx.DiGraph,
    available_columns: List[str],
) -> Set[str]:
    available_graph_nodes = dataframe_columns_to_graph_nodes(available_columns, G)

    allowed = set()
    for n, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type in {"latent", "background"} and n in available_graph_nodes:
            allowed.add(n)

    return allowed


def remove_outgoing_edges_of_treatment(
    G: nx.DiGraph,
    treatment: str,
) -> nx.DiGraph:
    G_do = G.copy()
    G_do.remove_edges_from(list(G.out_edges(treatment)))
    return G_do


def is_collider_on_path(
    G: nx.DiGraph,
    left: str,
    middle: str,
    right: str,
) -> bool:
    return G.has_edge(left, middle) and G.has_edge(right, middle)


def ancestors_of_set(
    G: nx.DiGraph,
    nodes: Set[str],
) -> Set[str]:
    anc = set(nodes)
    for n in nodes:
        anc |= nx.ancestors(G, n)
    return anc


def get_backdoor_paths(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> List[List[str]]:
    UG = G.to_undirected()
    paths = []

    for path in nx.all_simple_paths(UG, source=treatment, target=outcome):
        if len(path) < 2:
            continue
        first_neighbor = path[1]
        if G.has_edge(first_neighbor, treatment):
            paths.append(path)

    return paths


def is_path_active_given_Z(
    G: nx.DiGraph,
    path: List[str],
    Z: Set[str],
) -> bool:
    if len(path) <= 2:
        return True

    ancestors_Z = ancestors_of_set(G, Z) if Z else set()

    for i in range(1, len(path) - 1):
        left = path[i - 1]
        middle = path[i]
        right = path[i + 1]

        collider = is_collider_on_path(G, left, middle, right)

        if collider:
            if middle not in ancestors_Z:
                return False
        else:
            if middle in Z:
                return False

    return True


def open_backdoor_paths(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    Z: Set[str],
) -> List[List[str]]:
    paths = get_backdoor_paths(G, treatment, outcome)
    return [p for p in paths if is_path_active_given_Z(G, p, Z)]


def blocks_all_backdoor_paths(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    Z: Set[str],
) -> bool:
    return len(open_backdoor_paths(G, treatment, outcome, Z)) == 0


def candidate_backdoor_pool(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    available_columns: List[str],
) -> Set[str]:
    if treatment not in G:
        raise ValueError(f"Treatment node '{treatment}' is not in graph")
    if outcome not in G:
        raise ValueError(f"Outcome node '{outcome}' is not in graph")

    allowed = get_allowed_adjustment_nodes(G, available_columns)
    descendants_t = nx.descendants(G, treatment)

    G_do = remove_outgoing_edges_of_treatment(G, treatment)

    anc_t = nx.ancestors(G, treatment)
    anc_y_do = nx.ancestors(G_do, outcome)

    pool = allowed & anc_t & anc_y_do
    pool -= descendants_t
    pool -= {treatment, outcome}

    return pool


def get_colliders_on_backdoor_paths(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> Set[str]:
    colliders = set()

    for path in get_backdoor_paths(G, treatment, outcome):
        if len(path) < 3:
            continue

        for i in range(1, len(path) - 1):
            left = path[i - 1]
            middle = path[i]
            right = path[i + 1]

            if is_collider_on_path(G, left, middle, right):
                colliders.add(middle)

    return colliders


def get_descendants_of_nodes(
    G: nx.DiGraph,
    nodes: Set[str],
) -> Set[str]:
    out = set()
    for n in nodes:
        out |= nx.descendants(G, n)
    return out


def safe_expanded_backdoor_adjustment_set(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    available_columns: List[str],
) -> Tuple[List[str], List[List[str]], Dict[str, List[str]]]:
    pool = candidate_backdoor_pool(G, treatment, outcome, available_columns)

    colliders = get_colliders_on_backdoor_paths(G, treatment, outcome)
    collider_descendants = get_descendants_of_nodes(G, colliders)

    forbidden = (colliders | collider_descendants) - {treatment, outcome}
    safe_pool = set(pool) - forbidden

    diagnostics = {
        "colliders_removed": sorted(pool & colliders),
        "collider_descendants_removed": sorted(pool & collider_descendants),
        "safe_pool_before_block_check": sorted(safe_pool),
    }

    if blocks_all_backdoor_paths(G, treatment, outcome, safe_pool):
        remaining_open_paths = open_backdoor_paths(G, treatment, outcome, safe_pool)
        return sorted(safe_pool), remaining_open_paths, diagnostics

    minimal_set, remaining_open_paths = minimal_backdoor_adjustment_set(
        G=G,
        treatment=treatment,
        outcome=outcome,
        available_columns=available_columns,
    )

    diagnostics["fallback_to_minimal"] = sorted(minimal_set)
    return minimal_set, remaining_open_paths, diagnostics


def minimal_backdoor_adjustment_set(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    available_columns: List[str],
) -> Tuple[List[str], List[List[str]]]:
    pool = candidate_backdoor_pool(G, treatment, outcome, available_columns)
    current = set(sorted(pool))

    if not blocks_all_backdoor_paths(G, treatment, outcome, current):
        remaining_open_paths = open_backdoor_paths(G, treatment, outcome, current)
        return [], remaining_open_paths

    for node in sorted(pool):
        trial = current - {node}
        if blocks_all_backdoor_paths(G, treatment, outcome, trial):
            current = trial

    remaining_open_paths = open_backdoor_paths(G, treatment, outcome, current)
    return sorted(current), remaining_open_paths


def find_backdoor_confounders(
    G: nx.DiGraph,
    treatment: str,
    outcome_graph_node: str,
    available_columns: List[str],
) -> Dict[str, List[str]]:
    available_set = set(available_columns)

    pool = sorted(candidate_backdoor_pool(
        G=G,
        treatment=treatment,
        outcome=outcome_graph_node,
        available_columns=available_columns,
    ))

    if USE_EXPANDED_SAFE_CONFOUNDERS:
        graph_candidates, remaining_open_paths, diagnostics = safe_expanded_backdoor_adjustment_set(
            G=G,
            treatment=treatment,
            outcome=outcome_graph_node,
            available_columns=available_columns,
        )
    else:
        graph_candidates, remaining_open_paths = minimal_backdoor_adjustment_set(
            G=G,
            treatment=treatment,
            outcome=outcome_graph_node,
            available_columns=available_columns,
        )
        diagnostics = {
            "colliders_removed": [],
            "collider_descendants_removed": [],
            "safe_pool_before_block_check": [],
        }

    observed_cols: List[str] = []
    missing_graph_nodes: List[str] = []

    for node in graph_candidates:
        mapped = map_graph_node_to_dataframe_columns(node, available_set)
        if mapped:
            observed_cols.extend(mapped)
        else:
            missing_graph_nodes.append(node)

    observed_cols = sorted(set(observed_cols))

    return {
        "candidate_pool": pool,
        "graph_candidates": graph_candidates,
        "observed_confounders": observed_cols,
        "missing_graph_nodes": missing_graph_nodes,
        "open_backdoor_paths_if_any": [" -> ".join(p) for p in remaining_open_paths],
        "identifiable_with_available_nodes": len(remaining_open_paths) == 0,
        "colliders_removed": diagnostics.get("colliders_removed", []),
        "collider_descendants_removed": diagnostics.get("collider_descendants_removed", []),
        "safe_pool_before_block_check": diagnostics.get("safe_pool_before_block_check", []),
    }


# ============================================================
# Matching preparation
# ============================================================
def prepare_work_df(
    df: pd.DataFrame,
    treatment: str,
    confounders: List[str],
) -> pd.DataFrame:
    work_df = df.dropna(subset=[treatment, OUTCOME_COL]).copy()

    work_df[treatment] = pd.to_numeric(work_df[treatment], errors="coerce")
    work_df[OUTCOME_COL] = pd.to_numeric(work_df[OUTCOME_COL], errors="coerce")
    work_df = work_df.dropna(subset=[treatment, OUTCOME_COL]).copy()

    work_df[treatment] = work_df[treatment].astype(int)
    work_df[OUTCOME_COL] = work_df[OUTCOME_COL].astype(int)

    confounders = [c for c in confounders if c in work_df.columns and c not in [treatment, OUTCOME_COL]]

    for col in confounders:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

        if work_df[col].isna().all():
            work_df[col] = 0.0
        else:
            work_df[col] = work_df[col].fillna(work_df[col].median())

    t_values = sorted(work_df[treatment].dropna().unique().tolist())
    if t_values != [0, 1]:
        raise ValueError(f"{treatment} must be binary 0/1. Found: {t_values}")

    y_values = sorted(work_df[OUTCOME_COL].dropna().unique().tolist())
    if not set(y_values).issubset({0, 1}):
        raise ValueError(f"{OUTCOME_COL} must be binary 0/1. Found: {y_values}")

    return work_df


def to_binary_matching_matrix(
    df: pd.DataFrame,
    confounders: List[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Convert confounders into a binary matching design matrix for Hamming distance.

    Strategy:
    - already binary columns stay binary
    - one-hot ICUType columns stay binary
    - other numeric columns are binarized by median threshold
    """
    out = pd.DataFrame(index=df.index)
    transform_info = {}

    for col in confounders:
        s = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if not df[col].isna().all() else 0)

        uniq = sorted(pd.Series(s).dropna().unique().tolist())

        if set(uniq).issubset({0, 1}):
            out[col] = s.astype(int)
            transform_info[col] = "kept_binary"
            continue

        if REQUIRE_BINARY_CONF:
            continue

        threshold = float(pd.Series(s).median())
        out[col] = (s > threshold).astype(int)
        transform_info[col] = f"binarized_by_median_gt_{threshold:.4f}"

    return out, transform_info


def hamming_distance_row_to_matrix(
    row_vec: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    return np.sum(mat != row_vec, axis=1)


def sufficient_matches(
    n_pairs: int,
    n_treated_total: int,
) -> bool:
    if n_treated_total == 0:
        return False

    match_rate = n_pairs / n_treated_total
    return (n_pairs >= MIN_MATCHED_PAIRS) and (match_rate >= MIN_MATCH_RATE)


def greedy_hamming_match(
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    conf_bin_cols: List[str],
    treatment_col: str,
    outcome_col: str,
    max_dist: int,
    with_replacement: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Greedy 1:1 matching from treated to control.
    We start at dist=0 and increase allowed distance until enough pairs are obtained
    or until max_dist is reached.

    Returns:
      pairs_df
      diagnostics
    """
    if treated_df.empty or control_df.empty:
        return pd.DataFrame(), {
            "matched_pairs": 0,
            "n_treated_total": int(len(treated_df)),
            "n_control_total": int(len(control_df)),
            "match_rate": 0.0,
            "final_allowed_distance": None,
            "reached_sufficient_pairs": False,
            "hit_max_dist_without_sufficient_pairs": False,
            "pairs_by_distance": {},
        }

    treated = treated_df.copy().reset_index(drop=True)
    control = control_df.copy().reset_index(drop=True)

    treated_bin = treated[conf_bin_cols].astype(int).to_numpy()
    control_bin = control[conf_bin_cols].astype(int).to_numpy()

    all_pairs = []
    used_control = set()
    pairs_by_distance: Dict[int, int] = {}

    for allowed_dist in range(max_dist + 1):
        print(
            f"      Matching round: allowed_distance<={allowed_dist} | "
            f"current_pairs={len(all_pairs):,}"
        )
        new_pairs_this_round = 0

        already_matched_treated = {p["treated_row_idx"] for p in all_pairs}

        for i in range(len(treated)):
            if i in already_matched_treated:
                continue

            if with_replacement:
                candidate_control_idx = np.arange(len(control))
            else:
                candidate_control_idx = np.array([j for j in range(len(control)) if j not in used_control])

            if len(candidate_control_idx) == 0:
                continue

            dists = hamming_distance_row_to_matrix(
                treated_bin[i],
                control_bin[candidate_control_idx]
            )

            min_dist = int(np.min(dists))
            if min_dist > allowed_dist:
                continue

            nearest_positions = np.where(dists == min_dist)[0]
            chosen_pos = int(nearest_positions[0])
            j = int(candidate_control_idx[chosen_pos])

            pair = {
                "treated_row_idx": i,
                "control_row_idx": j,
                "treated_ts_id": treated.loc[i, "ts_id"],
                "control_ts_id": control.loc[j, "ts_id"],
                "treated_outcome": int(treated.loc[i, outcome_col]),
                "control_outcome": int(control.loc[j, outcome_col]),
                "pair_effect": float(treated.loc[i, outcome_col] - control.loc[j, outcome_col]),
                "hamming_distance": min_dist,
            }

            all_pairs.append(pair)
            new_pairs_this_round += 1

            if not with_replacement:
                used_control.add(j)

        pairs_by_distance[allowed_dist] = new_pairs_this_round
        print(
            f"      Completed distance<={allowed_dist}: added {new_pairs_this_round:,} "
            f"pairs | total_pairs={len(all_pairs):,}"
        )

        if sufficient_matches(len(all_pairs), len(treated)):
            break

    pairs_df = pd.DataFrame(all_pairs)

    final_allowed_distance = None
    if not pairs_df.empty:
        final_allowed_distance = int(pairs_df["hamming_distance"].max())

    reached = sufficient_matches(len(all_pairs), len(treated))
    hit_max_without_sufficient = (not reached)

    diagnostics = {
        "matched_pairs": int(len(all_pairs)),
        "n_treated_total": int(len(treated)),
        "n_control_total": int(len(control)),
        "match_rate": float(len(all_pairs) / len(treated)) if len(treated) > 0 else 0.0,
        "final_allowed_distance": final_allowed_distance,
        "reached_sufficient_pairs": reached,
        "hit_max_dist_without_sufficient_pairs": hit_max_without_sufficient,
        "pairs_by_distance": pairs_by_distance,
    }

    return pairs_df, diagnostics


def add_pair_confounder_details(
    pairs_df: pd.DataFrame,
    treated_df: pd.DataFrame,
    control_df: pd.DataFrame,
    conf_bin_cols: List[str],
) -> pd.DataFrame:
    """
    Adds treated/control confounder values for easier debugging and analysis.
    """
    if pairs_df.empty:
        return pairs_df.copy()

    out = pairs_df.copy()

    for col in conf_bin_cols:
        out[f"treated_{col}"] = out["treated_row_idx"].map(treated_df[col].to_dict())
        out[f"control_{col}"] = out["control_row_idx"].map(control_df[col].to_dict())

    return out


def summarize_pair_effects(
    pairs_df: pd.DataFrame,
    full_df: pd.DataFrame,
    treatment: str,
    confounders_used: List[str],
    conf_bin_cols: List[str],
    matching_info: Dict[str, object],
) -> Dict[str, object]:
    if pairs_df.empty:
        return {
            "treatment": treatment,
            "n_total": int(len(full_df)),
            "outcome_rate": float(full_df[OUTCOME_COL].mean()),
            "treatment_rate": float(full_df[treatment].mean()),
            "n_pairs": 0,
            "match_rate": matching_info["match_rate"],
            "mean_pair_effect": np.nan,
            "std_pair_effect": np.nan,
            "min_pair_effect": np.nan,
            "max_pair_effect": np.nan,
            "mean_normalized_pair_effect": np.nan,
            "final_allowed_distance": matching_info["final_allowed_distance"],
            "reached_sufficient_pairs": matching_info["reached_sufficient_pairs"],
            "hit_max_dist_without_sufficient_pairs": matching_info["hit_max_dist_without_sufficient_pairs"],
            "observed_confounders": ", ".join(confounders_used),
            "binary_matching_columns": ", ".join(conf_bin_cols),
        }

    outcome_rate = float(full_df[OUTCOME_COL].mean())
    mean_eff = float(pairs_df["pair_effect"].mean())

    return {
        "treatment": treatment,
        "n_total": int(len(full_df)),
        "outcome_rate": outcome_rate,
        "treatment_rate": float(full_df[treatment].mean()),
        "n_pairs": int(len(pairs_df)),
        "match_rate": matching_info["match_rate"],
        "mean_pair_effect": mean_eff,
        "std_pair_effect": float(pairs_df["pair_effect"].std()) if len(pairs_df) > 1 else 0.0,
        "min_pair_effect": float(pairs_df["pair_effect"].min()),
        "max_pair_effect": float(pairs_df["pair_effect"].max()),
        "mean_normalized_pair_effect": float(mean_eff / outcome_rate) if outcome_rate > 0 else np.nan,
        "final_allowed_distance": matching_info["final_allowed_distance"],
        "reached_sufficient_pairs": matching_info["reached_sufficient_pairs"],
        "hit_max_dist_without_sufficient_pairs": matching_info["hit_max_dist_without_sufficient_pairs"],
        "observed_confounders": ", ".join(confounders_used),
        "binary_matching_columns": ", ".join(conf_bin_cols),
    }


# ============================================================
# Output writers
# ============================================================
def write_confounder_analysis(
    path: str,
    treatment: str,
    confounder_info: Dict[str, List[str]],
    transform_info: Dict[str, str],
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Matching Confounder Analysis ===\n\n")
        f.write(f"Treatment: {treatment}\n")
        f.write(f"Outcome (graph node): {GRAPH_OUTCOME_NODE}\n\n")

        f.write(f"Identifiable with available nodes: {confounder_info['identifiable_with_available_nodes']}\n\n")

        f.write("Observed dataframe confounders used:\n")
        if confounder_info["observed_confounders"]:
            for c in confounder_info["observed_confounders"]:
                f.write(f"  - {c}\n")
        else:
            f.write("  - None\n")

        f.write("\nOpen backdoor paths remaining (if any):\n")
        if confounder_info["open_backdoor_paths_if_any"]:
            for p in confounder_info["open_backdoor_paths_if_any"]:
                f.write(f"  - {p}\n")
        else:
            f.write("  - None\n")

        f.write("\nBinary matching transform info:\n")
        if transform_info:
            for k, v in transform_info.items():
                f.write(f"  - {k}: {v}\n")
        else:
            f.write("  - None\n")


def write_matching_summary(
    path: str,
    treatment: str,
    summary: Dict[str, object],
    matching_info: Dict[str, object],
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Hamming Matching Summary ===\n\n")
        f.write(f"Treatment: {treatment}\n")
        f.write(f"Outcome: {OUTCOME_COL}\n\n")

        f.write("Matching config:\n")
        f.write(f"max_dist = {max_dist}\n")
        f.write(f"MIN_MATCHED_PAIRS = {MIN_MATCHED_PAIRS}\n")
        f.write(f"MIN_MATCH_RATE = {MIN_MATCH_RATE}\n")
        f.write(f"MATCH_WITH_REPLACEMENT = {MATCH_WITH_REPLACEMENT}\n\n")

        f.write("Match progression:\n")
        for dist, count in matching_info["pairs_by_distance"].items():
            f.write(f"  - newly matched at dist <= {dist}: {count}\n")

        f.write("\nFinal diagnostics:\n")
        f.write(f"Matched pairs: {matching_info['matched_pairs']}\n")
        f.write(f"Treated total: {matching_info['n_treated_total']}\n")
        f.write(f"Control total: {matching_info['n_control_total']}\n")
        f.write(f"Match rate: {matching_info['match_rate']:.4f}\n")
        f.write(f"Final allowed distance used: {matching_info['final_allowed_distance']}\n")
        f.write(f"Reached sufficient pairs: {matching_info['reached_sufficient_pairs']}\n")
        f.write(f"Hit max_dist without sufficient pairs: {matching_info['hit_max_dist_without_sufficient_pairs']}\n\n")

        if matching_info["hit_max_dist_without_sufficient_pairs"]:
            f.write("WARNING: Matching reached max_dist before obtaining a sufficient number of pairs.\n")
            f.write("Interpret the baseline with caution because overlap/support may be weak.\n\n")

        f.write("Effect summary:\n")
        f.write(f"Outcome rate: {summary['outcome_rate']:.6f}\n")
        f.write(f"Treatment rate: {summary['treatment_rate']:.6f}\n")
        f.write(f"Mean matched pair effect: {summary['mean_pair_effect']}\n")
        f.write(f"Std matched pair effect: {summary['std_pair_effect']}\n")
        f.write(f"Min matched pair effect: {summary['min_pair_effect']}\n")
        f.write(f"Max matched pair effect: {summary['max_pair_effect']}\n")
        f.write(f"Mean normalized pair effect: {summary['mean_normalized_pair_effect']}\n\n")

        f.write("Interpretation note:\n")
        f.write("This is a matched-pair average outcome difference baseline.\n")
        f.write("It is closer to an ATT-style matched estimate than to a model-based per-patient CATE.\n")


# ============================================================
# Main loop
# ============================================================
def main():
    global DATASET_MODEL
    global LATENT_TAGS_PATH
    global PHYSIONET_PKL_PATH
    global GRAPH_PKL_PATH
    global GRAPH_OUTCOME_NODE
    global TREATMENTS
    global OUTPUT_DIR
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.random.seed(SEED)
    args = parse_args()
    DATASET_MODEL = args.model
    dataset_defaults = get_dataset_defaults(DATASET_MODEL)

    if DATASET_MODEL == "mimic" and args.graph_pkl_path is None:
        raise ValueError(
            "MIMIC mode requires --graph-pkl-path because this repo does not define "
            "a relative default MIMIC graph pickle path."
        )
    if DATASET_MODEL == "mimic" and args.output_dir is None:
        raise ValueError(
            "MIMIC mode requires --output-dir because this repo does not define "
            "a safe default MIMIC output directory and the PhysioNet default would collide."
        )

    GRAPH_OUTCOME_NODE = str(dataset_defaults["graph_outcome_node"])
    TREATMENTS = list(dataset_defaults["treatments"])
    LATENT_TAGS_PATH = resolve_runtime_path(
        args.latent_tags_path,
        str(dataset_defaults["latent_tags_path"]),
        "LATENT_TAGS_PATH",
    )
    PHYSIONET_PKL_PATH = resolve_runtime_path(
        args.physionet_pkl_path,
        str(dataset_defaults["physionet_pkl_path"]),
        "PHYSIONET_PKL_PATH",
    )
    GRAPH_PKL_PATH = resolve_runtime_path(
        args.graph_pkl_path,
        dataset_defaults["graph_pkl_path"],
        "GRAPH_PKL_PATH",
    )
    OUTPUT_DIR = resolve_runtime_path(
        args.output_dir,
        OUTPUT_DIR,
        "OUTPUT_DIR",
        must_exist=False,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Starting matched-pair causal effect run ===")
    print(
        "Runtime configuration: "
        f"model={DATASET_MODEL} | latent_tags_path={LATENT_TAGS_PATH} | "
        f"processed_pkl_path={PHYSIONET_PKL_PATH} | graph_pkl_path={GRAPH_PKL_PATH} | "
        f"output_dir={OUTPUT_DIR}"
    )
    print("[1/3] Loading dataframe and graph...")
    df = load_analysis_dataframe(LATENT_TAGS_PATH, PHYSIONET_PKL_PATH)
    G = load_graph(GRAPH_PKL_PATH)

    print(f"Loaded df shape: {df.shape}")
    print(f"Outcome rate before down-sampling: {df[OUTCOME_COL].mean():.4f}")
    print(f"DAG size: nodes={G.number_of_nodes()} | edges={G.number_of_edges()}")

    if DOWN_SAMPLE:
        df = downsample_majority_label(
            df=df,
            outcome_col=OUTCOME_COL,
            seed=SEED,
        )
        print(f"Loaded df shape after down-sampling: {df.shape}")
        print(f"Outcome rate after down-sampling: {df[OUTCOME_COL].mean():.4f}")
    else:
        print(f"Outcome rate: {df[OUTCOME_COL].mean():.4f}")

    print(f"[2/3] Starting treatment loop: {len(TREATMENTS)} treatments total")
    global_rows = []

    for treatment_index, treatment in enumerate(TREATMENTS, start=1):
        print(f"\n=== Treatment {treatment_index}/{len(TREATMENTS)}: {treatment} ===")

        if treatment not in df.columns:
            print(f"Skipping {treatment}: not found in dataframe")
            continue

        if treatment not in G.nodes:
            print(f"Skipping {treatment}: not found in DAG")
            continue

        confounder_info = find_backdoor_confounders(
            G=G,
            treatment=treatment,
            outcome_graph_node=GRAPH_OUTCOME_NODE,
            available_columns=list(df.columns),
        )

        confounders = confounder_info["observed_confounders"]
        print(
            f"[{treatment}] Confounder discovery finished: "
            f"observed_confounders={len(confounders)}"
        )

        treatment_dir = os.path.join(OUTPUT_DIR, treatment)
        os.makedirs(treatment_dir, exist_ok=True)

        confounder_txt = os.path.join(treatment_dir, "confounder_analysis.txt")
        summary_txt = os.path.join(treatment_dir, "summary_results.txt")
        pairs_csv = os.path.join(treatment_dir, "matched_pairs.csv")

        try:
            work_df = prepare_work_df(df, treatment, confounders)

            print(f"[{treatment}] rows used: {len(work_df)}")
            print(f"[{treatment}] treatment rate: {work_df[treatment].mean():.4f}")
            print(f"[{treatment}] outcome rate: {work_df[OUTCOME_COL].mean():.4f}")

            print(f"[{treatment}] Building binary matching matrix")
            match_design_df, transform_info = to_binary_matching_matrix(work_df, confounders)
            conf_bin_cols = list(match_design_df.columns)

            if len(conf_bin_cols) == 0:
                raise ValueError(
                    "No binary matching columns available after preprocessing. "
                    "Either disable REQUIRE_BINARY_CONF or revisit confounder encoding."
                )

            work_df = work_df.copy()
            for col in conf_bin_cols:
                work_df[col] = match_design_df[col].astype(int)

            treated_df = work_df[work_df[treatment] == 1].copy().reset_index(drop=True)
            control_df = work_df[work_df[treatment] == 0].copy().reset_index(drop=True)
            print(
                f"[{treatment}] Starting greedy Hamming matching | treated={len(treated_df):,} | "
                f"control={len(control_df):,} | binary_columns={len(conf_bin_cols)}"
            )

            pairs_df, matching_info = greedy_hamming_match(
                treated_df=treated_df,
                control_df=control_df,
                conf_bin_cols=conf_bin_cols,
                treatment_col=treatment,
                outcome_col=OUTCOME_COL,
                max_dist=max_dist,
                with_replacement=MATCH_WITH_REPLACEMENT,
            )

            pairs_df = add_pair_confounder_details(
                pairs_df=pairs_df,
                treated_df=treated_df,
                control_df=control_df,
                conf_bin_cols=conf_bin_cols,
            )

            summary = summarize_pair_effects(
                pairs_df=pairs_df,
                full_df=work_df,
                treatment=treatment,
                confounders_used=confounders,
                conf_bin_cols=conf_bin_cols,
                matching_info=matching_info,
            )
            print(f"[{treatment}] Matching complete. Next: writing outputs")

            pairs_df.to_csv(pairs_csv, index=False)

            write_confounder_analysis(
                path=confounder_txt,
                treatment=treatment,
                confounder_info=confounder_info,
                transform_info=transform_info,
            )

            write_matching_summary(
                path=summary_txt,
                treatment=treatment,
                summary=summary,
                matching_info=matching_info,
            )

            global_rows.append(summary)

            print(f"[{treatment}] matched pairs: {matching_info['matched_pairs']}")
            print(f"[{treatment}] match rate: {matching_info['match_rate']:.4f}")
            print(f"[{treatment}] final allowed distance: {matching_info['final_allowed_distance']}")
            if matching_info["hit_max_dist_without_sufficient_pairs"]:
                print(f"[{treatment}] WARNING: reached max_dist={max_dist} without sufficient pairs")

            print(f"Saved: {confounder_txt}")
            print(f"Saved: {summary_txt}")
            print(f"Saved: {pairs_csv}")

        except Exception as e:
            with open(summary_txt, "w", encoding="utf-8") as f:
                f.write("=== Hamming Matching Summary ===\n\n")
                f.write(f"Treatment: {treatment}\n")
                f.write("Run status: FAILED\n\n")
                f.write(f"Reason: {repr(e)}\n")

            print(f"Failed for {treatment}: {e}")
            print(f"Saved failure summary: {summary_txt}")

    global_summary_csv = os.path.join(OUTPUT_DIR, "global_summary.csv")
    if global_rows:
        global_df = pd.DataFrame(global_rows)
        global_df = global_df.sort_values(by="mean_pair_effect", ascending=False)
        global_df.to_csv(global_summary_csv, index=False)
        print(f"\nSaved global summary: {global_summary_csv}")
    print(
        f"[3/3] Matching run finished. Successful treatment summaries: "
        f"{len(global_rows)} / {len(TREATMENTS)}"
    )


if __name__ == "__main__":
    main()
