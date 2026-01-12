import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm


pkl_path = "../data/processed/physionet2012_ts_oc_ids.pkl"
output_csv_path = "../data/latent_tags.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_physionet_pickle(pkl_path):
    print("[1/5] Loading PhysioNet pickle...")

    with open(pkl_path, 'rb') as f:
        ts, oc, ts_ids = pickle.load(f)

    print(f"      Loaded ts: {len(ts):,} rows")
    print(f"      Loaded oc: {len(oc):,} rows")
    print(f"      Loaded ts_ids: {len(ts_ids):,} patients")
    return ts, oc, ts_ids


def pivot_ts_to_wide(ts):
    print("[2/5] Pivoting long time-series to wide format...")

    df_wide = ts.pivot_table(
        index=["ts_id", "minute"],
        columns="variable",
        values="value"
    ).reset_index()

    print(f"      Wide dataframe shape: {df_wide.shape}")
    return df_wide


# ============================================================
# 2. SUMMARY STATISTICS
# ============================================================

def build_patient_summaries(df_wide):
    print("[3/5] Building per-patient summary statistics...")

    summary_funcs = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1]
    }

    variable_cols = [c for c in df_wide.columns if c not in ["ts_id", "minute"]]

    summaries = []

    for pid, g in tqdm(df_wide.groupby("ts_id"),
                       desc="      Processing patients",
                       unit="patient"):

        row = {"ts_id": pid}
        g = g.sort_values("minute")

        for var in variable_cols:
            series = g[var].dropna()
            if len(series) == 0:
                for stat in summary_funcs:
                    row[f"{var}_{stat}"] = np.nan
            else:
                for stat, func in summary_funcs.items():
                    row[f"{var}_{stat}"] = func(series)

        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    print(f"      Created summary table: {summary_df.shape}")
    return summary_df


# ============================================================
# 3. DECISION TREES
# ============================================================


def tag_shock(row):
    return int(
        (row.get("MAP_min", np.inf) < 65) or
        (row.get("SysABP_min", np.inf) < 90) or
        (row.get("Lactate_max", 0) > 2.0) or
        (row.get("Urine_sum", np.inf) < 500)
    )


def tag_respfail(row):
    fio2 = row.get("FiO2_min", np.nan)
    pao2 = row.get("PaO2_min", np.nan)
    pf_ratio = pao2 / fio2 if fio2 and fio2 > 0 else np.nan

    return int(
        (not np.isnan(pf_ratio) and pf_ratio < 300) or
        (row.get("SaO2_min", 100) < 90) or
        (row.get("MechVent_max", 0) == 1)
    )


def tag_renalfail(row):
    return int(
        (row.get("Creatinine_max", 0) > 2.0) or
        (row.get("BUN_max", 0) > 40) or
        (row.get("Urine_sum", np.inf) < 500)
    )


def tag_hepfail(row):
    return int(
        (row.get("Bilirubin_max", 0) > 2.0) or
        (row.get("AST_max", 0) > 100) or
        (row.get("ALT_max", 0) > 100)
    )


def tag_hemefail(row):
    return int(
        (row.get("Platelets_min", np.inf) < 100) or
        (row.get("HCT_min", np.inf) < 30)
    )


def tag_inflam(row):
    return int(
        (row.get("WBC_max", 0) > 12) or
        (row.get("WBC_min", 10) < 4) or
        (row.get("Temp_max", 36) > 38.3) or
        (row.get("Temp_min", 37) < 36)
    )


def tag_neurofail(row):
    return int(row.get("GCS_min", 15) < 13)


def tag_cardinj(row):
    return int(
        (row.get("TropI_max", 0) > 0.4) or
        (row.get("TropT_max", 0) > 0.1)
    )


def tag_metab(row):
    return int(
        (row.get("pH_min", 7.4) < 7.30) or
        (row.get("pH_max", 7.4) > 7.50) or
        (row.get("Glucose_min", 100) < 70) or
        (row.get("Glucose_max", 100) > 180) or
        (row.get("HCO3_min", 25) < 18)
    )


def tag_chronicrisk(row):
    return int(
        (row.get("Age_first", 0) > 65) or
        (row.get("ICUType_first", 0) in [2, 3])
    )


def tag_acuteinsult(row):
    return int(
        (row.get("Lactate_first", 0) > 2.0) or
        (row.get("MAP_first", 100) < 65) or
        (row.get("GCS_first", 15) < 13)
    )


def get_latent_decision_trees():
    """
        Returns dict: latent_variable_name -> decision_function(row)->0/1
        All functions are top-level and pickle-safe.
    """
    print("[4/5] Initializing latent decision trees...")

    trees = {
        "Shock": tag_shock,
        "RespFail": tag_respfail,
        "RenalFail": tag_renalfail,
        "HepFail": tag_hepfail,
        "HemeFail": tag_hemefail,
        "Inflam": tag_inflam,
        "NeuroFail": tag_neurofail,
        "CardInj": tag_cardinj,
        "Metab": tag_metab,
        "ChronicRisk": tag_chronicrisk,
        "AcuteInsult": tag_acuteinsult,
    }

    print(f"      Loaded {len(trees)} latent definitions")
    return trees


# ============================================================
# 4. TAGGING
# ============================================================

def tag_all_patients(summary_df, decision_trees):
    print("[5/5] Applying latent tags to all patients...")

    tag_rows = []

    for _, row in tqdm(summary_df.iterrows(),
                       total=len(summary_df),
                       desc="      Tagging patients",
                       unit="patient"):
        pid = row["ts_id"]
        tag_row = {"ts_id": pid}

        for latent, func in decision_trees.items():
            tag_row[latent] = func(row)

        tag_rows.append(tag_row)

    latent_df = pd.DataFrame(tag_rows)
    print(f"      Tag table shape: {latent_df.shape}")
    return latent_df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def run_latent_tagging_pipeline(pkl_path, output_csv_path):

    print("\n=== Starting PhysioNet Latent Tagging Pipeline ===\n")

    ts, oc, ts_ids = load_physionet_pickle(pkl_path)

    df_wide = pivot_ts_to_wide(ts)

    summary_df = build_patient_summaries(df_wide)

    decision_trees = get_latent_decision_trees()

    latent_tags_df = tag_all_patients(summary_df, decision_trees)

    print("\nSaving results to CSV...")
    latent_tags_df.to_csv(output_csv_path, index=False)

    print("\n=== Pipeline completed successfully ✅ ===")
    print(f"Output saved to: {output_csv_path}\n")

    return latent_tags_df, decision_trees


latent_tags_df, decision_trees = run_latent_tagging_pipeline(pkl_path, output_csv_path)
print(latent_tags_df.head())


# Save decision trees dictionary
with open(output_csv_path.replace(".csv", "_trees.pkl"), "wb") as f:
    pickle.dump(decision_trees, f)

print(f"Decision trees dictionary saved to: {output_csv_path.replace('.csv', '_trees.pkl')}")

