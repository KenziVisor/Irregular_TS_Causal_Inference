"""
CATE Estimation — by definition (Peng Ding, Chapter 10)
=========================================================

Definition (Theorem 10.1 / eq. 10.8):
--------------------------------------
Under ignorability Y(z) ⊥ Z | X, the ATE is identified by:

    τ = Σ_x  [E(Y | Z=1, X=x) - E(Y | Z=0, X=x)] · P(X=x)

That is:
    τ = E_X [ E(Y | Z=1, X) - E(Y | Z=0, X) ]
      = E_X [ τ(X) ]

where τ(X) = E(Y|Z=1,X) - E(Y|Z=0,X) is the CATE at covariate value X.

Implementation (nonparametric, no regression):
----------------------------------------------
Since all our variables (latents + background) are discrete/binary,
we implement the stratification estimator exactly as in eq. 10.8:

  For each stratum x (unique combination of confounder values):
    τ̂(x) = Ȳ_{Z=1, X=x}  -  Ȳ_{Z=0, X=x}   (difference in observed means)

  τ̂ = Σ_x  τ̂(x) · (n_x / n)                  (weighted by stratum size)

This is the nonparametric stratification estimator — no model assumptions.
Bootstrap is used for 95% CIs.

Pipeline per treatment T:
  1. Load the latent-only DAG
  2. Derive the backdoor adjustment set Z = parents(T) - descendants(T)
  3. Compute τ̂ by stratification over Z
"""

import pickle
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from itertools import product

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================
LATENT_TAGS_CSV  = "./../../data/latent_tags.csv"
PHYSIONET_PKL    = "../../../data/processed/physionet2012_ts_oc_ids.pkl"
CAUSAL_GRAPH_PKL = "../../../data/causal_graph.pkl"
RESULTS_TXT      = "cate_results.txt"

OUTCOME = "in_hospital_mortality"

LATENT_VARS = [
    "Shock", "RespFail", "RenalFail", "HepFail", "HemeFail",
    "Inflam", "NeuroFail", "CardInj", "Metab", "ChronicRisk", "AcuteInsult",
]
BACKGROUND_VARS = ["Age", "Gender", "ICUType"]


# ============================================================
# 1. LOAD CAUSAL DAG  →  latent-only subgraph
# ============================================================

def load_latent_dag() -> nx.DiGraph:
    """
    Keep only background + latent nodes + Death.
    Observed clinical vars are dropped — they are descendants of latents,
    so conditioning on them blocks causal paths or opens collider bias.
    """
    with open(CAUSAL_GRAPH_PKL, "rb") as f:
        G_full = pickle.load(f)

    keep_nodes = [
        n for n, d in G_full.nodes(data=True)
        if d.get("node_type") in {"background", "latent"}
    ] + ["Death"]

    G = G_full.subgraph(keep_nodes).copy()
    assert nx.is_directed_acyclic_graph(G)
    print(f"[DAG] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G


# ============================================================
# 2. BACKDOOR ADJUSTMENT SET
# ============================================================

def get_adjustment_set(G: nx.DiGraph, treatment: str) -> list:
    """
    Backdoor adjustment set for (treatment -> Death):
      Z = parents(T) - descendants(T)

    - Parents of T are the direct confounders (they causally precede T
      and also affect the outcome through the DAG).
    - We exclude descendants of T because:
        * Mediators (on the causal path T→...→Death) would block the effect.
        * Colliders (common children of T and another cause) would open
          spurious paths if conditioned on.
    """
    descendants_T = nx.descendants(G, treatment)
    parents_T     = set(G.predecessors(treatment))
    return sorted(parents_T - descendants_T - {treatment, "Death"})


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    latent_df = pd.read_csv(LATENT_TAGS_CSV)
    latent_df["ts_id"] = latent_df["ts_id"].astype(str)

    with open(PHYSIONET_PKL, "rb") as f:
        ts, oc, _ = pickle.load(f)

    # First-recorded background vars per patient
    ts_bg = ts[ts["variable"].isin(
        ["Age", "Gender", "ICUType", "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]
    )].copy()
    bg_wide = (
        ts_bg.sort_values("minute")
        .groupby(["ts_id", "variable"])["value"]
        .first()
        .unstack("variable")
        .reset_index()
    )
    # Reconstruct single ICUType column
    if "ICUType" not in bg_wide.columns:
        bg_wide["ICUType"] = np.nan
    for i in range(1, 5):
        col = f"ICUType_{i}"
        if col in bg_wide.columns:
            bg_wide.loc[bg_wide[col] == 1, "ICUType"] = i

    bg_wide["ts_id"] = bg_wide["ts_id"].astype(str)
    bg_keep = ["ts_id"] + [c for c in ["Age", "Gender", "ICUType"] if c in bg_wide.columns]

    oc_small = oc[["ts_id", OUTCOME]].copy()
    oc_small["ts_id"] = oc_small["ts_id"].astype(str)

    df = latent_df.merge(oc_small, on="ts_id", how="inner")
    df = df.merge(bg_wide[bg_keep], on="ts_id", how="left")
    df = df.dropna(subset=[OUTCOME])
    df[OUTCOME] = df[OUTCOME].astype(int)

    # Discretize Age into tertiles so stratification stays tractable
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Age"] = pd.qcut(df["Age"], q=3, labels=[0, 1, 2]).astype(float)

    for col in ["Gender", "ICUType"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"[Data] {len(df):,} patients | mortality = {df[OUTCOME].mean():.3f}")
    return df


# ============================================================
# 4. STRATIFICATION ESTIMATOR (Peng Ding eq. 10.8)
# ============================================================

def stratification_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list,
    n_bootstrap: int = 500,
    seed: int = 42,
    min_stratum_size: int = 5,   # drop strata with too few treated or control units
) -> tuple:
    """
    Nonparametric stratification estimator for ATE (Peng Ding, Theorem 10.1):

        τ̂ = Σ_x  [Ȳ(Z=1, X=x) - Ȳ(Z=0, X=x)] · P̂(X=x)

    Steps:
      1. Form strata by all unique combinations of confounder values.
      2. Within each stratum x:
           τ̂(x) = mean(Y | Z=1, X=x) - mean(Y | Z=0, X=x)
      3. Weight by stratum proportion in the full sample:
           τ̂ = Σ_x τ̂(x) · (n_x / n)

    Strata where either treated or control group is empty (or too small)
    are skipped — their contribution cannot be estimated (overlap violation).

    Returns: (ate, ci_lower, ci_upper, n_strata_used, coverage_fraction)
    """

    def _compute_ate(data):
        n_total = len(data)
        ate_sum = 0.0
        n_strata_used = 0

        if not confounders:
            # No confounders: simple difference in means
            y1 = data.loc[data[treatment] == 1, outcome]
            y0 = data.loc[data[treatment] == 0, outcome]
            if len(y1) < min_stratum_size or len(y0) < min_stratum_size:
                return np.nan, 0, 1.0
            return float(y1.mean() - y0.mean()), 1, 1.0

        # Group by all confounder combinations
        groups = data.groupby(confounders, observed=True)
        n_strata_total = len(groups)

        for stratum_vals, stratum_df in groups:
            n_x = len(stratum_df)
            y1 = stratum_df.loc[stratum_df[treatment] == 1, outcome]
            y0 = stratum_df.loc[stratum_df[treatment] == 0, outcome]

            # Skip strata without both treated and control (overlap violation)
            if len(y1) < min_stratum_size or len(y0) < min_stratum_size:
                continue

            tau_x   = float(y1.mean() - y0.mean())   # CATE at stratum x
            weight  = n_x / n_total                   # P̂(X = x)
            ate_sum += tau_x * weight
            n_strata_used += 1

        coverage = n_strata_used / n_strata_total if n_strata_total > 0 else 0.0
        return ate_sum, n_strata_used, coverage

    cols = [treatment, outcome] + confounders
    df_sub = df[cols].dropna()

    ate, n_strata_used, coverage = _compute_ate(df_sub)

    # Bootstrap CIs
    rng = np.random.default_rng(seed)
    boot_ates = []
    for _ in range(n_bootstrap):
        boot_df = df_sub.sample(n=len(df_sub), replace=True, random_state=int(rng.integers(1e6)))
        b_ate, _, _ = _compute_ate(boot_df)
        if not np.isnan(b_ate):
            boot_ates.append(b_ate)

    if len(boot_ates) < 10:
        ci_lower = ci_upper = np.nan
    else:
        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))

    return ate, ci_lower, ci_upper, n_strata_used, coverage, len(df_sub)


# ============================================================
# 5. ESTIMATE CATE FOR ONE TREATMENT
# ============================================================

def estimate_cate(df: pd.DataFrame, G: nx.DiGraph, treatment: str) -> dict:
    """
    1. Get backdoor adjustment set from DAG
    2. Compute ATE via stratification (Peng Ding eq. 10.8)
    """
    adj_set  = get_adjustment_set(G, treatment)
    conf_df  = [z for z in adj_set if z in df.columns]

    ate, ci_lower, ci_upper, n_strata, coverage, n = stratification_ate(
        df, treatment, OUTCOME, conf_df
    )

    return {
        "treatment": treatment,
        "ate": ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confounders": conf_df,
        "n_strata_used": n_strata,
        "stratum_coverage": coverage,
        "n": n,
    }


# ============================================================
# 6. MAIN
# ============================================================

def run():
    G  = load_latent_dag()
    df = load_data()

    all_treatments = LATENT_VARS

    results = []
    for treatment in all_treatments:
        if treatment not in df.columns:
            print(f"[SKIP] {treatment} not in dataframe")
            continue

        print(f"[CATE] {treatment}")
        r = estimate_cate(df, G, treatment)

        ci_str = (f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                  if r["ci_lower"] is not None and not np.isnan(r["ci_lower"])
                  else "N/A")
        print(f"  confounders    : {r['confounders']}")
        print(f"  strata used    : {r['n_strata_used']}  "
              f"(coverage {r['stratum_coverage']:.1%})")
        print(f"  ATE            : {r['ate']:.4f}  95% CI {ci_str}\n")
        results.append(r)

    # Save
    lines = [
        "=" * 60,
        "CATE RESULTS — PhysioNet 2012",
        f"Outcome : {OUTCOME}",
        "Method  : Stratification estimator (Peng Ding, Theorem 10.1)",
        "          ATE = sum_x [E(Y|Z=1,X=x) - E(Y|Z=0,X=x)] * P(X=x)",
        "CI      : Percentile bootstrap (500 resamples)",
        "=" * 60, "",
    ]
    for r in results:
        ci_str = (f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                  if r["ci_lower"] is not None and not np.isnan(r["ci_lower"])
                  else "N/A")
        lines += [
            f"Treatment : {r['treatment']}",
            f"  N                : {r['n']}",
            f"  Confounders      : {r['confounders']}",
            f"  Strata used      : {r['n_strata_used']}  "
            f"(coverage {r['stratum_coverage']:.1%})",
            f"  ATE              : {r['ate']:.4f}",
            f"  95% CI           : {ci_str}",
            "",
        ]

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved → {RESULTS_TXT}")
    return results


if __name__ == "__main__":
    run()