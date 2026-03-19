from __future__ import annotations

import os
import pickle
import warnings
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ============================================================
# Config
# ============================================================
LATENT_TAGS_PATH = "../../data/latent_tags.csv"
PHYSIONET_PKL_PATH = "../../data/processed/physionet2012_ts_oc_ids.pkl"
GRAPH_PKL_PATH = "../../data/causal_graph.pkl"

OUTCOME_COL = "in_hospital_mortality"
GRAPH_OUTCOME_NODE = "Death"

# All treatment candidates from your latent tagging pipeline
TREATMENTS = [
    "Severity", "Shock", "RespFail", "RenalFail", "HepFail", "HemeFail",
    "Inflam", "NeuroFail", "CardInj", "Metab",
    "ChronicRisk", "AcuteInsult"
]

OUTPUT_DIR = "./cate_outputs"
SEED = 42


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


def build_background_features(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Build patient-level observed background covariates from ts.
    Preprocessing already converts ICUType into ICUType_1..ICUType_4.
    """
    df = ts.copy().sort_values(["ts_id", "minute"])

    keep_vars = [
        "Age", "Gender", "Weight",
        "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"
    ]
    df = df[df["variable"].isin(keep_vars)].copy()

    first_vals = (
        df.groupby(["ts_id", "variable"], as_index=False)
          .first()[["ts_id", "variable", "value"]]
    )

    bg = first_vals.pivot(index="ts_id", columns="variable", values="value").reset_index()

    for col in ["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]:
        if col not in bg.columns:
            bg[col] = 0.0

    return bg


def load_analysis_dataframe(
    latent_tags_path: str,
    physionet_pkl_path: str,
) -> pd.DataFrame:
    latent_df = pd.read_csv(latent_tags_path)
    latent_df["ts_id"] = latent_df["ts_id"].astype(str)

    ts, oc, _ = load_physionet_pickle(physionet_pkl_path)

    oc_small = oc[["ts_id", OUTCOME_COL]].copy()
    oc_small["ts_id"] = oc_small["ts_id"].astype(str)

    bg_df = build_background_features(ts)
    bg_df["ts_id"] = bg_df["ts_id"].astype(str)

    df = latent_df.merge(oc_small, on="ts_id", how="inner")
    df = df.merge(bg_df, on="ts_id", how="left")

    df = df.dropna(subset=[OUTCOME_COL]).copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)

    return df


# ============================================================
# Graph logic: backdoor-style confounder discovery
# ============================================================


def dataframe_columns_to_graph_nodes(
    available_columns: List[str],
    G: nx.DiGraph,
) -> Set[str]:
    """
    Map dataframe columns to graph node names.

    Rules:
    - ICUType_1..4 in dataframe correspond to ICUType in graph
    - in_hospital_mortality is dataframe outcome, graph node is Death -> ignore here
    - ts_id is an identifier -> ignore
    - all other columns are kept only if they are actual graph nodes
    """
    graph_nodes = set(G.nodes)
    mapped = set()

    for col in available_columns:
        if col == "ts_id":
            continue
        if col == "in_hospital_mortality":
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
    """
    Map a graph node back to dataframe columns.
    """
    if node == "ICUType":
        return [c for c in ["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"] if c in available_columns]

    return [node] if node in available_columns else []


def get_allowed_adjustment_nodes(
    G: nx.DiGraph,
    available_columns: List[str],
) -> Set[str]:
    """
    Allowed adjustment variables:
    - latent nodes
    - background/meta nodes
    - and only if they are actually available in the dataframe
    """
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
    """
    Intervention-style graph for backdoor reasoning:
    remove all outgoing edges from treatment.
    """
    G_do = G.copy()
    G_do.remove_edges_from(list(G.out_edges(treatment)))
    return G_do


def is_collider_on_path(
    G: nx.DiGraph,
    left: str,
    middle: str,
    right: str,
) -> bool:
    """
    A node is a collider on a path if both arrows point into it:
    left -> middle <- right
    """
    return G.has_edge(left, middle) and G.has_edge(right, middle)


def ancestors_of_set(
    G: nx.DiGraph,
    nodes: Set[str],
) -> Set[str]:
    """
    All ancestors of a node set, including the set itself.
    """
    anc = set(nodes)
    for n in nodes:
        anc |= nx.ancestors(G, n)
    return anc


def get_backdoor_paths(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> List[List[str]]:
    """
    Enumerate all simple undirected paths between treatment and outcome
    that start with an arrow into treatment, i.e. real backdoor paths.
    """
    UG = G.to_undirected()
    paths = []

    for path in nx.all_simple_paths(UG, source=treatment, target=outcome):
        if len(path) < 2:
            continue

        first_neighbor = path[1]

        # Backdoor path must start with: first_neighbor -> treatment
        if G.has_edge(first_neighbor, treatment):
            paths.append(path)

    return paths

def is_path_active_given_Z(
    G: nx.DiGraph,
    path: List[str],
    Z: Set[str],
) -> bool:
    """
    Check whether a specific path is active (open) given conditioning set Z,
    using d-separation rules.

    Rules:
    - non-collider in Z => path blocked
    - collider not in An(Z) => path blocked
    """
    if len(path) <= 2:
        # direct edge path; if it's a backdoor direct path, it is active unless blocked
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
    """
    Return all open backdoor paths between treatment and outcome given Z.
    """
    paths = get_backdoor_paths(G, treatment, outcome)
    return [p for p in paths if is_path_active_given_Z(G, p, Z)]


def blocks_all_backdoor_paths(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    Z: Set[str],
) -> bool:
    """
    True iff Z blocks all backdoor paths from treatment to outcome.
    """
    return len(open_backdoor_paths(G, treatment, outcome, Z)) == 0


def candidate_backdoor_pool(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    available_columns: List[str],
) -> Set[str]:
    """
    Build a principled candidate pool.

    We keep only nodes that:
    1. are allowed adjustment variables (latent/background + available)
    2. are NOT descendants of treatment
    3. are ancestors of treatment in the original graph
    4. remain ancestors of outcome after removing outgoing edges from treatment

    Point (4) is important:
    it removes nodes whose effect on outcome goes only through treatment.
    """
    if treatment not in G:
        raise ValueError(f"Treatment node '{treatment}' is not in graph")
    if outcome not in G:
        raise ValueError(f"Outcome node '{outcome}' is not in graph")

    allowed = get_allowed_adjustment_nodes(G, available_columns)
    descendants_t = nx.descendants(G, treatment)

    G_do = remove_outgoing_edges_of_treatment(G, treatment)

    anc_t = nx.ancestors(G, treatment)
    anc_y_do = nx.ancestors(G_do, outcome)

    pool = (
        allowed
        & anc_t
        & anc_y_do
    )

    pool -= descendants_t
    pool -= {treatment, outcome}

    return pool


def minimal_backdoor_adjustment_set(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    available_columns: List[str],
) -> Tuple[List[str], List[List[str]]]:
    """
    Compute a minimal adjustment set over the allowed variable universe.

    Strategy:
    - Start from the principled candidate pool
    - If the whole pool still does not block all backdoor paths, return failure
    - Otherwise remove redundant variables one by one while blocking is preserved
    """
    pool = candidate_backdoor_pool(G, treatment, outcome, available_columns)

    # deterministic order
    current = set(sorted(pool))

    if not blocks_all_backdoor_paths(G, treatment, outcome, current):
        # even the full allowed pool cannot identify the effect
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
    """
    Main function to use in your pipeline.

    Returns:
    - graph_candidates: confounders as graph node names
    - observed_confounders: mapped dataframe columns
    - missing_graph_nodes: allowed graph nodes that were selected but do not map
    - candidate_pool: full pre-minimalization pool
    - open_backdoor_paths_if_any: remaining open paths if identification failed
    - identifiable_with_available_nodes: bool
    """
    available_set = set(available_columns)

    pool = sorted(candidate_backdoor_pool(
        G=G,
        treatment=treatment,
        outcome=outcome_graph_node,
        available_columns=available_columns,
    ))

    graph_candidates, remaining_open_paths = minimal_backdoor_adjustment_set(
        G=G,
        treatment=treatment,
        outcome=outcome_graph_node,
        available_columns=available_columns,
    )

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
    }


def choose_effect_modifiers(
    df: pd.DataFrame,
    treatment: str,
    confounders: List[str],
) -> List[str]:
    """
    Keep X compact and stable.
    You can change this later, but this is a sane default.
    """
    preferred = [
        "Age", "Gender", "Weight",
        "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4",
        "ChronicRisk", "AcuteInsult"
    ]
    return [c for c in preferred if c in df.columns and c != treatment and c != OUTCOME_COL]


# ============================================================
# Estimation
# ============================================================
def fit_one_treatment(
    df: pd.DataFrame,
    treatment: str,
    confounders: List[str],
    effect_modifiers: List[str],
) -> Tuple[CausalForestDML, pd.DataFrame, Dict[str, float], str]:
    """
    Fit CATE for one treatment.

    Returns:
      - fitted estimator
      - per-row CATE dataframe
      - summary stats dict
      - textual formula description
    """
    if treatment not in df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in dataframe")

    # Keep only rows with observed treatment and outcome
    work_df = df.dropna(subset=[treatment, OUTCOME_COL]).copy()

    # Force binary integer encoding
    work_df[treatment] = pd.to_numeric(work_df[treatment], errors="coerce")
    work_df[OUTCOME_COL] = pd.to_numeric(work_df[OUTCOME_COL], errors="coerce")
    work_df = work_df.dropna(subset=[treatment, OUTCOME_COL]).copy()

    work_df[treatment] = work_df[treatment].astype(int)
    work_df[OUTCOME_COL] = work_df[OUTCOME_COL].astype(int)

    # Keep only columns that actually exist
    confounders = [c for c in confounders if c in work_df.columns and c not in [treatment, OUTCOME_COL]]
    effect_modifiers = [c for c in effect_modifiers if c in work_df.columns and c not in [treatment, OUTCOME_COL]]

    used_cols = ["ts_id", treatment, OUTCOME_COL] + confounders + effect_modifiers
    used_cols = list(dict.fromkeys(used_cols))

    print(f"\n[{treatment}] missingness before filtering/imputation:")
    print(work_df[used_cols].isna().mean().sort_values(ascending=False))
    print(f"[{treatment}] rows before filtering: {len(work_df)}")

    model_df = work_df.copy()

    # Impute confounders and effect modifiers instead of dropping rows
    numeric_cols = list(dict.fromkeys(confounders + effect_modifiers))
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        if model_df[col].isna().all():
            model_df[col] = 0.0
        else:
            model_df[col] = model_df[col].fillna(model_df[col].median())

    print(f"[{treatment}] rows after filtering/imputation: {len(model_df)}")

    # Check treatment really is binary
    t_values = sorted(model_df[treatment].dropna().unique().tolist())
    if t_values != [0, 1]:
        raise ValueError(f"{treatment} must be binary 0/1. Found: {t_values}")

    # Outcome can be binary; keep as float for EconML
    y_values = sorted(model_df[OUTCOME_COL].dropna().unique().tolist())
    if not set(y_values).issubset({0, 1}):
        raise ValueError(f"{OUTCOME_COL} must be binary 0/1. Found: {y_values}")

    Y = model_df[OUTCOME_COL].astype(float).to_numpy()
    T = model_df[treatment].astype(int).to_numpy()
    W = model_df[confounders].astype(float).to_numpy() if confounders else None
    X = model_df[effect_modifiers].astype(float).to_numpy() if effect_modifiers else None

    est = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=SEED,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        discrete_treatment=True,
        n_estimators=400,
        min_samples_leaf=20,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1,
    )

    est.fit(Y=Y, T=T, X=X, W=W)
    cate = est.effect(X=X)

    out = model_df[["ts_id", treatment, OUTCOME_COL]].copy()
    out["CATE"] = cate

    try:
        lb, ub = est.effect_interval(X=X, alpha=0.05)
        out["CATE_lower_95"] = lb
        out["CATE_upper_95"] = ub
    except Exception:
        out["CATE_lower_95"] = np.nan
        out["CATE_upper_95"] = np.nan

    formula = (
        f"CATE_{treatment}(x) = E[{OUTCOME_COL}(1) - {OUTCOME_COL}(0) | X=x]\n"
        f"T = {treatment}\n"
        f"Y = {OUTCOME_COL}\n"
        f"W (backdoor confounders) = {confounders if confounders else 'None'}\n"
        f"X (effect modifiers) = {effect_modifiers if effect_modifiers else 'None'}"
    )

    summary = {
        "n": float(len(out)),
        "outcome_rate": float(model_df[OUTCOME_COL].mean()),
        "treatment_rate": float(model_df[treatment].mean()),
        "mean_cate": float(out["CATE"].mean()),
        "std_cate": float(out["CATE"].std()),
        "min_cate": float(out["CATE"].min()),
        "max_cate": float(out["CATE"].max()),
    }

    return est, out, summary, formula


# ============================================================
# Output writers
# ============================================================
def write_confounder_analysis(
    path: str,
    treatment: str,
    confounder_info: Dict[str, List[str]],
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Confounder Analysis ===\n\n")
        f.write(f"Treatment: {treatment}\n")
        f.write("Outcome (graph node): Death\n\n")

        f.write("Method used:\n")
        f.write("- Allowed adjustment variables: latent + background/meta only\n")
        f.write("- Excluded descendants of treatment\n")
        f.write("- Built candidate pool using ancestors of treatment and outcome in do(T) graph\n")
        f.write("- Minimalized set by blocking all backdoor paths via d-separation\n\n")

        f.write(f"Identifiable with available nodes: {confounder_info['identifiable_with_available_nodes']}\n\n")

        f.write("Candidate pool:\n")
        if confounder_info["candidate_pool"]:
            for c in confounder_info["candidate_pool"]:
                f.write(f"  - {c}\n")
        else:
            f.write("  - None\n")

        f.write("\nFinal graph-level confounders:\n")
        if confounder_info["graph_candidates"]:
            for c in confounder_info["graph_candidates"]:
                f.write(f"  - {c}\n")
        else:
            f.write("  - None\n")

        f.write("\nObserved dataframe confounders used:\n")
        if confounder_info["observed_confounders"]:
            for c in confounder_info["observed_confounders"]:
                f.write(f"  - {c}\n")
        else:
            f.write("  - None\n")

        f.write("\nMissing selected graph nodes:\n")
        if confounder_info["missing_graph_nodes"]:
            for c in confounder_info["missing_graph_nodes"]:
                f.write(f"  - {c}\n")
        else:
            f.write("  - None\n")

        f.write("\nOpen backdoor paths remaining (if not identifiable):\n")
        if confounder_info["open_backdoor_paths_if_any"]:
            for p in confounder_info["open_backdoor_paths_if_any"]:
                f.write(f"  - {p}\n")
        else:
            f.write("  - None\n")


def write_summary_results(
    path: str,
    treatment: str,
    formula: str,
    summary: Dict[str, float],
    confounder_info: Dict[str, List[str]],
    cate_csv_path: str,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== CATE Summary Results ===\n\n")
        f.write(f"Treatment: {treatment}\n")
        f.write(f"Outcome: {OUTCOME_COL}\n\n")

        f.write("Model formula / variables:\n")
        f.write(formula + "\n\n")

        f.write("Backdoor confounder analysis summary:\n")
        f.write(f"Observed confounders used: {confounder_info['observed_confounders']}\n")
        f.write(f"Missing graph candidates: {confounder_info['missing_graph_nodes']}\n\n")

        f.write("Results:\n")
        f.write(f"N used by model: {int(summary['n'])}\n")
        f.write(f"Outcome positive rate: {summary['outcome_rate']:.6f}\n")
        f.write(f"Treatment positive rate: {summary['treatment_rate']:.6f}\n")
        f.write(f"Mean CATE: {summary['mean_cate']:.6f}\n")
        f.write(f"Std CATE: {summary['std_cate']:.6f}\n")
        f.write(f"Min CATE: {summary['min_cate']:.6f}\n")
        f.write(f"Max CATE: {summary['max_cate']:.6f}\n\n")

        f.write(f"Per-patient CATE file: {cate_csv_path}\n")


# ============================================================
# Main loop
# ============================================================
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataframe and graph...")
    df = load_analysis_dataframe(LATENT_TAGS_PATH, PHYSIONET_PKL_PATH)
    G = load_graph(GRAPH_PKL_PATH)

    print(f"Loaded df shape: {df.shape}")
    print(f"Outcome rate: {df[OUTCOME_COL].mean():.4f}")

    global_summary_rows = []

    for treatment in TREATMENTS:
        print(f"\n=== Treatment: {treatment} ===")

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
        effect_modifiers = choose_effect_modifiers(df, treatment, confounders)

        treatment_dir = os.path.join(OUTPUT_DIR, treatment)
        os.makedirs(treatment_dir, exist_ok=True)

        confounder_txt = os.path.join(
            treatment_dir,
            "confounder_analysis.txt"
        )

        summary_txt = os.path.join(
            treatment_dir,
            "summary_results.txt"
        )

        cate_csv = os.path.join(
            treatment_dir,
            "cate.csv"
        )

        feature_importance_csv = os.path.join(
            treatment_dir,
            "feature_importance.csv"
        )

        write_confounder_analysis(
            path=confounder_txt,
            treatment=treatment,
            confounder_info=confounder_info,
        )

        try:
            est, cate_df, summary, formula = fit_one_treatment(
                df=df,
                treatment=treatment,
                confounders=confounders,
                effect_modifiers=effect_modifiers,
            )

            # ====================================================
            # Feature importance for CATE heterogeneity
            # ====================================================
            if effect_modifiers:
                importance_df = pd.DataFrame({
                    "variable": effect_modifiers,
                    "importance": est.feature_importances_
                }).sort_values("importance", ascending=False)

                importance_df.to_csv(feature_importance_csv, index=False)

            cate_df.to_csv(cate_csv, index=False)

            write_summary_results(
                path=summary_txt,
                treatment=treatment,
                formula=formula,
                summary=summary,
                confounder_info=confounder_info,
                cate_csv_path=cate_csv,
            )

            global_summary_rows.append({
                "treatment": treatment,
                "n": int(summary["n"]),
                "outcome_rate": summary["outcome_rate"],
                "treatment_rate": summary["treatment_rate"],
                "mean_cate": summary["mean_cate"],
                "std_cate": summary["std_cate"],
                "min_cate": summary["min_cate"],
                "max_cate": summary["max_cate"],
                "num_observed_confounders": len(confounder_info["observed_confounders"]),
                "num_missing_graph_candidates": len(confounder_info["missing_graph_nodes"]),
                "observed_confounders": ", ".join(confounder_info["observed_confounders"]),
                "missing_graph_candidates": ", ".join(confounder_info["missing_graph_nodes"]),
                "cate_csv_path": cate_csv,
            })

            print(f"Saved: {confounder_txt}")
            print(f"Saved: {summary_txt}")
            print(f"Saved: {cate_csv}")

        except Exception as e:
            with open(summary_txt, "w", encoding="utf-8") as f:
                f.write("=== CATE Summary Results ===\n\n")
                f.write(f"Treatment: {treatment}\n")
                f.write("Run status: FAILED\n\n")
                f.write(f"Reason: {repr(e)}\n\n")
                f.write(f"Observed confounders that would have been used: {confounders}\n")
                f.write(f"Effect modifiers that would have been used: {effect_modifiers}\n")

            print(f"Failed for {treatment}: {e}")
            print(f"Saved failure summary: {summary_txt}")

    global_summary_csv = os.path.join(OUTPUT_DIR, "global_summary.csv")

    if global_summary_rows:
        global_summary_df = pd.DataFrame(global_summary_rows)
        global_summary_df = global_summary_df.sort_values(
            by="mean_cate",
            ascending=False
        )
        global_summary_df.to_csv(global_summary_csv, index=False)
        print(f"\nSaved global summary: {global_summary_csv}")


if __name__ == "__main__":
    main()
