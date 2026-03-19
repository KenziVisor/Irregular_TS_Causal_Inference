"""
Bayesian optimization (Optuna) of latent-tagging thresholds + mortality classifier (PhysioNet 2012).

What this script does:
1) Loads your processed PhysioNet pickle (ts, oc, ts_ids)
2) Builds per-patient summary stats ONCE
3) For each Optuna trial:
   - tags latents (0/1) using tunable thresholds
   - trains LogisticRegression on TRAIN
   - returns VAL AUROC (Optuna optimizes this)
4) After Optuna finishes:
   - freezes best thresholds (picked by VAL AUROC)
   - trains LogisticRegression on TRAIN
   - evaluates on VAL and TEST
   - writes TWO text files in CURRENT WORKING DIRECTORY:
        (a) RESULTS_TXT_PATH: same format as your mortality_prediction_results.txt
        (b) BEST_PARAMS_TXT_PATH: best thresholds rounded to 2 decimals

Requirements:
- optuna installed:  pip install optuna

Run:
  python optimize_latent_thresholds.py
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import optuna  # assumes installed


# =========================
# Config
# =========================
PKL_PATH = "../../data/processed/physionet2012_ts_oc_ids.pkl"
SEED = 42
N_TRIALS = 100

# Output text files (saved to CURRENT WORKING DIRECTORY)
RESULTS_TXT_PATH = "mortality_prediction_results_optimized.txt"
BEST_PARAMS_TXT_PATH = "optimal_thresholds.txt"


# =========================
# 1) Load + pivot to wide
# =========================
def load_physionet_pickle(pkl_path: str):
    with open(pkl_path, "rb") as f:
        ts, oc, ts_ids = pickle.load(f)
    return ts, oc, ts_ids


def pivot_ts_to_wide(ts: pd.DataFrame) -> pd.DataFrame:
    df_wide = ts.pivot_table(
        index=["ts_id", "minute"],
        columns="variable",
        values="value"
    ).reset_index()
    return df_wide


# =========================
# 2) Patient summaries
# =========================
def build_patient_summaries(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Builds min/max/mean/first/last for each variable.
    Adds Urine_sum because your tree uses Urine_sum.
    """
    summary_funcs = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1],
    }

    variable_cols = [c for c in df_wide.columns if c not in ["ts_id", "minute"]]
    summaries = []

    # groupby is the heavy step, but happens ONCE (outside Optuna loop)
    for pid, g in df_wide.groupby("ts_id"):
        g = g.sort_values("minute")
        row = {"ts_id": str(pid)}

        for var in variable_cols:
            series = g[var].dropna()
            if len(series) == 0:
                for stat in summary_funcs:
                    row[f"{var}_{stat}"] = np.nan
                if var == "Urine":
                    row["Urine_sum"] = np.nan
            else:
                for stat, func in summary_funcs.items():
                    row[f"{var}_{stat}"] = func(series)
                if var == "Urine":
                    row["Urine_sum"] = float(series.sum())

        summaries.append(row)

    return pd.DataFrame(summaries)


# =========================
# 3) Latent tagging (0/1) with tunable thresholds
# =========================
def tag_latents(row: pd.Series, thr: dict) -> dict:
    """
    Returns dict latent->0/1.
    Logical structure matches your current trees, but thresholds are parameters in `thr`.
    """
    def g(key, default=np.nan):
        v = row.get(key, default)
        return v

    # Severity
    severity_flags = [
        int((g("MAP_first", np.inf) < thr["severity_map"]) or
            (g("SysABP_first", np.inf) <= thr["severity_sysabp"])),
        int(g("GCS_first", 15.0) < thr["severity_gcs"]),
        int((g("RespRate_first", -np.inf) >= thr["severity_resprate"]) or
            (g("SaO2_first", 100.0) < thr["severity_sao2"])),
        int((g("Lactate_first", 0.0) > thr["severity_lact"]) or
            (g("pH_first", 7.4) < thr["severity_ph"]) or
            (g("HCO3_first", 25.0) < thr["severity_hco3"])),
        int(g("Creatinine_first", 0.0) >= thr["severity_creat"]),
    ]
    severity = int(sum(severity_flags) >= thr["severity_min_count"])

    # Shock
    shock = int(
        (g("MAP_min", np.inf) < thr["shock_map"]) or
        (g("SysABP_min", np.inf) < thr["shock_sysabp"]) or
        (g("Lactate_max", 0.0) > thr["shock_lact"]) or
        (g("Urine_sum", np.inf) < thr["shock_urine_sum"])
    )

    # RespFail
    fio2 = g("FiO2_min", np.nan)
    pao2 = g("PaO2_min", np.nan)
    pf_ratio = (pao2 / fio2) if (not np.isnan(fio2) and fio2 > 0) else np.nan

    respfail = int(
        ((not np.isnan(pf_ratio)) and (pf_ratio < thr["resp_pf"])) or
        (g("SaO2_min", 100.0) < thr["resp_sao2"]) or
        (g("MechVent_max", 0.0) == 1.0)
    )

    # RenalFail
    renalfail = int(
        (g("Creatinine_max", 0.0) > thr["renal_creat"]) or
        (g("BUN_max", 0.0) > thr["renal_bun"]) or
        (g("Urine_sum", np.inf) < thr["renal_urine_sum"])
    )

    # HepFail
    hepfail = int(
        (g("Bilirubin_max", 0.0) > thr["hep_bili"]) or
        (g("AST_max", 0.0) > thr["hep_ast"]) or
        (g("ALT_max", 0.0) > thr["hep_alt"])
    )

    # HemeFail
    hemefail = int(
        (g("Platelets_min", np.inf) < thr["heme_plts"]) or
        (g("HCT_min", np.inf) < thr["heme_hct"])
    )

    # Inflam
    inflam = int(
        (g("WBC_max", 0.0) > thr["inflam_wbc_hi"]) or
        (g("WBC_min", 10.0) < thr["inflam_wbc_lo"]) or
        (g("Temp_max", 36.0) > thr["inflam_temp_hi"]) or
        (g("Temp_min", 37.0) < thr["inflam_temp_lo"])
    )

    # NeuroFail
    neurofail = int(g("GCS_min", 15.0) < thr["neuro_gcs"])

    # CardInj
    cardinj = int(
        (g("TropI_max", 0.0) > thr["card_tropi"]) or
        (g("TropT_max", 0.0) > thr["card_tropt"])
    )

    # Metab
    metab = int(
        (g("pH_min", 7.4) < thr["metab_ph_lo"]) or
        (g("pH_max", 7.4) > thr["metab_ph_hi"]) or
        (g("Glucose_min", 100.0) < thr["metab_glu_lo"]) or
        (g("Glucose_max", 100.0) > thr["metab_glu_hi"]) or
        (g("HCO3_min", 25.0) < thr["metab_hco3_lo"])
    )

    # ChronicRisk
    chronicrisk = int(
        (g("Age_first", 0.0) > thr["chronic_age"]) or
        (g("ICUType_first", 0.0) in [2, 3])  # may often be missing in your summary; kept as-is
    )

    # AcuteInsult
    acuteinsult = int(
        (g("Lactate_first", 0.0) > thr["acute_lact"]) or
        (g("MAP_first", 100.0) < thr["acute_map"]) or
        (g("GCS_first", 15.0) < thr["acute_gcs"])
    )

    return {
        "Severity": severity,
        "Shock": shock,
        "RespFail": respfail,
        "RenalFail": renalfail,
        "HepFail": hepfail,
        "HemeFail": hemefail,
        "Inflam": inflam,
        "NeuroFail": neurofail,
        "CardInj": cardinj,
        "Metab": metab,
        "ChronicRisk": chronicrisk,
        "AcuteInsult": acuteinsult,
    }


def make_latent_feature_matrix(summary_df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    rows = []
    for _, r in summary_df.iterrows():
        tags = tag_latents(r, thr)
        tags["ts_id"] = r["ts_id"]
        rows.append(tags)
    return pd.DataFrame(rows)


# =========================
# 4) Metrics helper
# =========================
def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "PosRate_true": float(y_true.mean()),
        "PosRate_pred": float(y_pred.mean()),
    }


# =========================
# 5) Single-split evaluation given thresholds
# =========================
def compute_val_auroc(summary_df: pd.DataFrame, oc: pd.DataFrame, thr: dict, seed: int) -> float:
    """
    Used by Optuna objective:
    - build latents with thr
    - split train/val/test (same as your pipeline)
    - train LogisticRegression on train
    - return VAL AUROC
    """
    latent_df = make_latent_feature_matrix(summary_df, thr)

    oc_small = oc[["ts_id", "in_hospital_mortality"]].copy()
    latent_df["ts_id"] = latent_df["ts_id"].astype(str)
    oc_small["ts_id"] = oc_small["ts_id"].astype(str)

    df = latent_df.merge(oc_small, on="ts_id", how="inner").dropna(subset=["in_hospital_mortality"])
    y = df["in_hospital_mortality"].astype(int).values
    X = df.drop(columns=["ts_id", "in_hospital_mortality"]).astype(float).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed
    )
    clf.fit(X_train_s, y_train)
    val_prob = clf.predict_proba(X_val_s)[:, 1]
    return float(roc_auc_score(y_val, val_prob))


def final_train_val_test_metrics(summary_df: pd.DataFrame, oc: pd.DataFrame, thr: dict, seed: int):
    """
    After Optuna finishes:
    - freeze best thresholds
    - train on TRAIN
    - report VAL + TEST metrics
    - returns (df_merged, val_metrics, test_metrics)
    """
    latent_df = make_latent_feature_matrix(summary_df, thr)
    oc_small = oc[["ts_id", "in_hospital_mortality"]].copy()

    latent_df["ts_id"] = latent_df["ts_id"].astype(str)
    oc_small["ts_id"] = oc_small["ts_id"].astype(str)

    df = latent_df.merge(oc_small, on="ts_id", how="inner").dropna(subset=["in_hospital_mortality"])
    y = df["in_hospital_mortality"].astype(int).values
    X = df.drop(columns=["ts_id", "in_hospital_mortality"]).astype(float).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed
    )
    clf.fit(X_train_s, y_train)

    val_prob = clf.predict_proba(X_val_s)[:, 1]
    test_prob = clf.predict_proba(X_test_s)[:, 1]

    val_metrics = evaluate_probs(y_val, val_prob)
    test_metrics = evaluate_probs(y_test, test_prob)

    return df, val_metrics, test_metrics


# =========================
# 6) Optuna search space
# =========================
def suggest_thresholds(trial: optuna.Trial) -> dict:
    return {
        # Severity
        "severity_map": trial.suggest_float("severity_map", 60, 75),
        "severity_sysabp": trial.suggest_float("severity_sysabp", 90, 105),
        "severity_gcs": trial.suggest_float("severity_gcs", 12, 15),
        "severity_resprate": trial.suggest_float("severity_resprate", 20, 30),
        "severity_sao2": trial.suggest_float("severity_sao2", 88, 95),
        "severity_lact": trial.suggest_float("severity_lact", 1.5, 4.0),
        "severity_ph": trial.suggest_float("severity_ph", 7.10, 7.35),
        "severity_hco3": trial.suggest_float("severity_hco3", 12, 22),
        "severity_creat": trial.suggest_float("severity_creat", 1.2, 3.0),
        "severity_min_count": trial.suggest_int("severity_min_count", 2, 3),

        # Shock
        "shock_map": trial.suggest_float("shock_map", 55, 80),
        "shock_sysabp": trial.suggest_float("shock_sysabp", 80, 110),
        "shock_lact": trial.suggest_float("shock_lact", 1.5, 5.0),
        "shock_urine_sum": trial.suggest_float("shock_urine_sum", 100, 1500),

        # RespFail
        "resp_pf": trial.suggest_float("resp_pf", 100, 400),
        "resp_sao2": trial.suggest_float("resp_sao2", 85, 95),

        # RenalFail
        "renal_creat": trial.suggest_float("renal_creat", 1.0, 4.0),
        "renal_bun": trial.suggest_float("renal_bun", 20, 100),
        "renal_urine_sum": trial.suggest_float("renal_urine_sum", 100, 1500),

        # HepFail
        "hep_bili": trial.suggest_float("hep_bili", 1.0, 8.0),
        "hep_ast": trial.suggest_float("hep_ast", 50, 400),
        "hep_alt": trial.suggest_float("hep_alt", 50, 400),

        # HemeFail
        "heme_plts": trial.suggest_float("heme_plts", 50, 200),
        "heme_hct": trial.suggest_float("heme_hct", 20, 40),

        # Inflam
        "inflam_wbc_hi": trial.suggest_float("inflam_wbc_hi", 10, 30),
        "inflam_wbc_lo": trial.suggest_float("inflam_wbc_lo", 2, 6),
        "inflam_temp_hi": trial.suggest_float("inflam_temp_hi", 37.5, 40.0),
        "inflam_temp_lo": trial.suggest_float("inflam_temp_lo", 34.0, 36.5),

        # NeuroFail
        "neuro_gcs": trial.suggest_float("neuro_gcs", 8, 15),

        # CardInj
        "card_tropi": trial.suggest_float("card_tropi", 0.05, 2.0),
        "card_tropt": trial.suggest_float("card_tropt", 0.01, 0.5),

        # Metab
        "metab_ph_lo": trial.suggest_float("metab_ph_lo", 7.05, 7.35),
        "metab_ph_hi": trial.suggest_float("metab_ph_hi", 7.45, 7.65),
        "metab_glu_lo": trial.suggest_float("metab_glu_lo", 50, 90),
        "metab_glu_hi": trial.suggest_float("metab_glu_hi", 140, 250),
        "metab_hco3_lo": trial.suggest_float("metab_hco3_lo", 10, 22),

        # ChronicRisk
        "chronic_age": trial.suggest_float("chronic_age", 50, 85),

        # AcuteInsult
        "acute_lact": trial.suggest_float("acute_lact", 1.5, 5.0),
        "acute_map": trial.suggest_float("acute_map", 55, 80),
        "acute_gcs": trial.suggest_float("acute_gcs", 8, 15),
    }


# =========================
# 7) Main
# =========================
def main():
    print("Loading PhysioNet pickle...")
    ts, oc, ts_ids = load_physionet_pickle(PKL_PATH)

    print("Pivoting to wide (long->wide)...")
    df_wide = pivot_ts_to_wide(ts)

    print("Building per-patient summaries (this runs once)...")
    summary_df = build_patient_summaries(df_wide)
    print(f"Summary shape: {summary_df.shape}")

    # ----- OPTUNA OBJECTIVE -----
    def objective(trial: optuna.Trial) -> float:
        thr = suggest_thresholds(trial)
        return compute_val_auroc(summary_df, oc, thr, seed=SEED)

    print("\nStarting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_val_auroc = float(study.best_value)
    best_thr = dict(study.best_params)

    print("\n=== BEST OPTUNA RESULT (picked by validation AUROC) ===")
    print(f"Best VAL AUROC: {best_val_auroc:.6f}")
    print("Best thresholds:")
    for k, v in sorted(best_thr.items()):
        print(f"  {k}: {v:.4f}")

    # ----- FINAL EVALUATION ON VAL + TEST USING BEST THR -----
    print("\nFinal evaluation on VAL and TEST using frozen best thresholds...")
    df_merged, val_metrics, test_metrics = final_train_val_test_metrics(summary_df, oc, best_thr, seed=SEED)

    # Write results file (same format as your previous mortality_prediction_results.txt)
    y_all = df_merged["in_hospital_mortality"].astype(int).values
    n_patients = len(df_merged)
    mortality_rate = float(y_all.mean())
    n_features = 12  # your latent set size

    with open(RESULTS_TXT_PATH, "w") as f:
        f.write("=== Mortality Prediction From Latent Variables ===\n\n")
        f.write(f"Dataset size: {n_patients} patients\n")
        f.write(f"Mortality rate: {mortality_rate:.4f}\n")
        f.write(f"Number of latent features: {n_features}\n\n")

        f.write("----- Logistic Regression -----\n")
        f.write("Validation metrics:\n")
        f.write(f"{val_metrics}\n\n")
        f.write("Test metrics:\n")
        f.write(f"{test_metrics}\n\n")

    print(f"Saved results to: {RESULTS_TXT_PATH}")

    # Write best parameters file (rounded to 2 decimals)
    with open(BEST_PARAMS_TXT_PATH, "w") as f:
        f.write("=== Optimal Thresholds (selected by validation AUROC) ===\n\n")
        f.write(f"Best VAL AUROC: {best_val_auroc:.6f}\n\n")
        for k, v in sorted(best_thr.items()):
            f.write(f"{k}: {v:.2f}\n")

    print(f"Saved best thresholds to: {BEST_PARAMS_TXT_PATH}")


if __name__ == "__main__":
    main()
