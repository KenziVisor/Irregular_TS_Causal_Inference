from __future__ import annotations

"""
Conservative ATE estimation for latent variables in PhysioNet 2012.

Why this version exists
-----------------------
The earlier script labeled its estimates as "CATE" but actually computed an
ATE-style stratification estimator. It also silently dropped required DAG
confounders that were missing from the dataframe (most importantly Severity),
which made several reported effects non-identified under the stated DAG.

This script fixes the main methodological problems:
1) Renames the estimand to ATE (not CATE).
2) Loads all observed background covariates used by the DAG: Age, Gender,
   Height, Weight, ICUType.
3) Refuses to estimate effects when the DAG requires unobserved confounders.
4) Uses a properly normalized stratification estimator when some strata are
   dropped for overlap violations.
5) Reports patient coverage in usable strata (not just number of strata).
6) Skips treatments with inadequate treated/control support.
7) Estimates only latent treatments, never background variables as treatments.

Interpretation
--------------
Under ignorability and overlap, the estimand is the total causal effect
(ATE): E[Y(1) - Y(0)] for a binary latent treatment T.
"""

import pickle
import warnings
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LATENT_TAGS_CSV = "../../../data/latent_tags.csv"
PHYSIONET_PKL = "../../../data/processed/physionet2012_ts_oc_ids.pkl"
CAUSAL_GRAPH_PKL = "../../../data/causal_graph.pkl"
RESULTS_TXT = "ate_results_fixed.txt"

OUTCOME = "in_hospital_mortality"

LATENT_VARS = [
    "Shock", "RespFail", "RenalFail", "HepFail", "HemeFail",
    "Inflam", "NeuroFail", "CardInj", "Metab", "ChronicRisk", "AcuteInsult",
]
BACKGROUND_VARS = ["Age", "Gender", "Height", "Weight", "ICUType"]
MIN_GROUP_SIZE = 5
N_BOOTSTRAP = 500
SEED = 42


@dataclass
class EstimateResult:
    treatment: str
    status: str
    reason: str | None
    n: int
    treated_n: int
    control_n: int
    treated_rate: float
    outcome_rate_treated: float | None
    outcome_rate_control: float | None
    observed_parents: list[str]
    missing_required_parents: list[str]
    ate: float | None
    ci_lower: float | None
    ci_upper: float | None
    strata_used: int | None
    patient_coverage: float | None


def load_latent_dag() -> nx.DiGraph:
    with open(CAUSAL_GRAPH_PKL, "rb") as f:
        g_full = pickle.load(f)
    keep_nodes = [
        n for n, d in g_full.nodes(data=True)
        if d.get("node_type") in {"background", "latent"}
    ] + ["Death"]
    g = g_full.subgraph(keep_nodes).copy()
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Subgraph is not a DAG.")
    print(f"[DAG] {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


def required_adjustment_set(g: nx.DiGraph, treatment: str) -> list[str]:
    descendants_t = nx.descendants(g, treatment)
    parents_t = set(g.predecessors(treatment))
    return sorted(parents_t - descendants_t - {treatment, "Death"})


def _safe_qcut(series: pd.Series, q: int, labels: list[int] | None = None) -> pd.Series:
    """Robust quantile binning that survives duplicate edges.

    pandas.qcut(..., duplicates="drop") may reduce the actual number of bins,
    which makes a fixed labels list invalid. To avoid that failure, we first ask
    qcut for integer bin ids (labels=False), then cast the result to float.
    """
    s = pd.to_numeric(series, errors="coerce")
    non_null = s.dropna()

    if non_null.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    if non_null.nunique() <= 1:
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[non_null.index] = 0.0
        return out

    try:
        bins = pd.qcut(s, q=q, labels=False, duplicates="drop")
    except ValueError:
        ranked = s.rank(method="average", pct=True)
        bins = pd.cut(ranked, bins=q, labels=False, include_lowest=True)

    return pd.Series(bins, index=series.index).astype(float)


def load_data() -> pd.DataFrame:
    latent_df = pd.read_csv(LATENT_TAGS_CSV)
    latent_df["ts_id"] = latent_df["ts_id"].astype(str)

    with open(PHYSIONET_PKL, "rb") as f:
        ts, oc, _ = pickle.load(f)

    bg_vars = BACKGROUND_VARS + ["ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"]
    ts_bg = ts[ts["variable"].isin(bg_vars)].copy()
    bg_wide = (
        ts_bg.sort_values("minute")
        .groupby(["ts_id", "variable"])["value"]
        .first()
        .unstack("variable")
        .reset_index()
    )

    if "ICUType" not in bg_wide.columns:
        bg_wide["ICUType"] = np.nan
    for i in range(1, 5):
        col = f"ICUType_{i}"
        if col in bg_wide.columns:
            bg_wide.loc[bg_wide[col] == 1, "ICUType"] = i

    bg_wide["ts_id"] = bg_wide["ts_id"].astype(str)
    oc_small = oc[["ts_id", OUTCOME]].copy()
    oc_small["ts_id"] = oc_small["ts_id"].astype(str)

    df = latent_df.merge(oc_small, on="ts_id", how="inner")
    df = df.merge(bg_wide[["ts_id"] + [c for c in BACKGROUND_VARS if c in bg_wide.columns]],
                  on="ts_id", how="left")
    df = df.dropna(subset=[OUTCOME]).copy()
    df[OUTCOME] = df[OUTCOME].astype(int)

    for col in ["Age", "Height", "Weight"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            df[col] = _safe_qcut(df[col], q=3, labels=[0, 1, 2])
    for col in ["Gender", "ICUType"]:
        if col in df.columns:
            median_val = df[col].median() if not df[col].dropna().empty else 0
            df[col] = df[col].fillna(median_val)
    for col in LATENT_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[Data] {len(df):,} patients | mortality={df[OUTCOME].mean():.3f}")
    return df


def _compute_stratified_ate(data: pd.DataFrame, treatment: str, outcome: str,
                            confounders: list[str], min_group_size: int):
    if not confounders:
        y1 = data.loc[data[treatment] == 1, outcome]
        y0 = data.loc[data[treatment] == 0, outcome]
        if len(y1) < min_group_size or len(y0) < min_group_size:
            return None, 0, 0.0
        return float(y1.mean() - y0.mean()), 1, 1.0

    groups = list(data.groupby(confounders, observed=True))
    usable = []
    usable_n = 0
    for _, g in groups:
        y1 = g.loc[g[treatment] == 1, outcome]
        y0 = g.loc[g[treatment] == 0, outcome]
        if len(y1) < min_group_size or len(y0) < min_group_size:
            continue
        tau_x = float(y1.mean() - y0.mean())
        n_x = len(g)
        usable.append((n_x, tau_x))
        usable_n += n_x
    if usable_n == 0:
        return None, 0, 0.0
    ate = sum((n_x / usable_n) * tau_x for n_x, tau_x in usable)
    return float(ate), len(usable), float(usable_n / len(data))


def stratified_ate_with_bootstrap(df: pd.DataFrame, treatment: str, outcome: str,
                                  confounders: list[str], n_bootstrap: int = N_BOOTSTRAP,
                                  seed: int = SEED, min_group_size: int = MIN_GROUP_SIZE):
    cols = [treatment, outcome] + confounders
    df_sub = df[cols].dropna().copy()
    ate, strata_used, patient_coverage = _compute_stratified_ate(
        df_sub, treatment, outcome, confounders, min_group_size
    )
    if ate is None:
        return None, None, None, strata_used, patient_coverage

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(df_sub), len(df_sub))
        boot_df = df_sub.iloc[sample_idx].copy()
        b_ate, _, _ = _compute_stratified_ate(
            boot_df, treatment, outcome, confounders, min_group_size
        )
        if b_ate is not None:
            boot.append(b_ate)
    if len(boot) < 50:
        return ate, None, None, strata_used, patient_coverage
    return ate, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), strata_used, patient_coverage


def estimate_one(df: pd.DataFrame, g: nx.DiGraph, treatment: str) -> EstimateResult:
    if treatment not in df.columns:
        return EstimateResult(treatment, "skipped", "Treatment column missing from dataframe.",
                              0, 0, 0, float("nan"), None, None, [], [], None, None, None, None, None)

    s = df[treatment].dropna()
    unique_vals = sorted(pd.unique(s))
    if set(unique_vals) - {0, 1}:
        return EstimateResult(treatment, "skipped",
                              f"Treatment is not binary 0/1. Unique values: {unique_vals}",
                              int(len(df)), 0, 0, float("nan"), None, None, [], [], None, None, None, None, None)

    required_parents = required_adjustment_set(g, treatment)
    observed_parents = [z for z in required_parents if z in df.columns]
    missing_required = [z for z in required_parents if z not in df.columns]

    n = int(df[treatment].notna().sum())
    treated_mask = df[treatment] == 1
    control_mask = df[treatment] == 0
    treated_n = int(treated_mask.sum())
    control_n = int(control_mask.sum())
    treated_rate = treated_n / max(n, 1)
    outcome_rate_treated = float(df.loc[treated_mask, OUTCOME].mean()) if treated_n > 0 else None
    outcome_rate_control = float(df.loc[control_mask, OUTCOME].mean()) if control_n > 0 else None

    if missing_required:
        return EstimateResult(
            treatment, "skipped",
            "Required DAG confounders are unobserved in the estimation dataframe, so the effect is not identified under the stated DAG.",
            n, treated_n, control_n, treated_rate, outcome_rate_treated, outcome_rate_control,
            observed_parents, missing_required, None, None, None, None, None
        )

    ate, ci_lower, ci_upper, strata_used, patient_coverage = stratified_ate_with_bootstrap(
        df, treatment, OUTCOME, observed_parents
    )
    if ate is None:
        return EstimateResult(
            treatment, "skipped", "Insufficient overlap/support after stratification.",
            n, treated_n, control_n, treated_rate, outcome_rate_treated, outcome_rate_control,
            observed_parents, [], None, None, None, strata_used, patient_coverage
        )

    return EstimateResult(
        treatment, "estimated", None,
        n, treated_n, control_n, treated_rate, outcome_rate_treated, outcome_rate_control,
        observed_parents, [], ate, ci_lower, ci_upper, strata_used, patient_coverage
    )


def format_result(r: EstimateResult):
    lines = [f"Treatment : {r.treatment}"]
    lines.append(f"  Status            : {r.status}")
    lines.append(f"  N                 : {r.n}")
    lines.append(f"  Treated / Control : {r.treated_n} / {r.control_n}")
    lines.append(f"  Treated rate      : {r.treated_rate:.4f}")
    if r.outcome_rate_treated is not None and r.outcome_rate_control is not None:
        lines.append(f"  Mortality treated : {r.outcome_rate_treated:.4f}")
        lines.append(f"  Mortality control : {r.outcome_rate_control:.4f}")
    lines.append(f"  Observed parents  : {r.observed_parents}")
    if r.missing_required_parents:
        lines.append(f"  Missing parents   : {r.missing_required_parents}")
    if r.reason:
        lines.append(f"  Reason            : {r.reason}")
    if r.ate is not None:
        lines.append(f"  ATE               : {r.ate:.4f}")
        ci_str = f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]" if r.ci_lower is not None and r.ci_upper is not None else "N/A"
        lines.append(f"  95% CI            : {ci_str}")
        lines.append(f"  Strata used       : {r.strata_used}")
        lines.append(f"  Patient coverage  : {r.patient_coverage:.1%}")
    lines.append("")
    return lines


def run():
    g = load_latent_dag()
    df = load_data()
    results = []
    for treatment in LATENT_VARS:
        print(f"[ATE] {treatment}")
        r = estimate_one(df, g, treatment)
        if r.status == "estimated":
            ci_str = f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]" if r.ci_lower is not None and r.ci_upper is not None else "N/A"
            print(f"  parents         : {r.observed_parents}")
            print(f"  ATE             : {r.ate:.4f} | CI {ci_str}")
            print(f"  patient coverage: {r.patient_coverage:.1%}\n")
        else:
            print(f"  skipped         : {r.reason}")
            if r.missing_required_parents:
                print(f"  missing         : {r.missing_required_parents}")
            print()
        results.append(r)

    lines = [
        "=" * 70,
        "ATE RESULTS — PhysioNet 2012 (fixed / conservative version)",
        f"Outcome : {OUTCOME}",
        "Method  : Exact stratification estimator with normalized weights.",
        "Policy  : Skip treatments whose required DAG confounders are unobserved.",
        f"CI      : Percentile bootstrap ({N_BOOTSTRAP} resamples)",
        "=" * 70,
        "",
    ]
    for r in results:
        lines.extend(format_result(r))

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved -> {RESULTS_TXT}")
    return results


if __name__ == "__main__":
    run()
