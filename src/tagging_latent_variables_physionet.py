import argparse
import pickle
import sys
from pathlib import Path

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/tagging_latent_variables_physionet.py",
        fixed_dataset="physionet",
    )

import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial

from dataset_config import (
    get_config_bool,
    get_config_list,
    get_config_scalar,
    load_dataset_config,
    resolve_with_precedence,
)


pkl_path = "../../data/processed/physionet2012_ts_oc_ids.pkl"
output_csv_path = "latent_tags_optimized.csv"

OPTIMIZED = True
THRESHOLDS_PATH = "../../data/optimal_thresholds.txt"
DEFAULT_THRESHOLDS = {
    "severity_map": 70,
    "severity_sysabp": 100,
    "severity_gcs": 15,
    "severity_resprate": 22,
    "severity_sao2": 92,
    "severity_lact": 2.0,
    "severity_ph": 7.30,
    "severity_hco3": 18,
    "severity_creat": 2.0,
    "severity_min_count": 2,
    "shock_map": 65,
    "shock_sysabp": 90,
    "shock_lact": 2.0,
    "shock_urine_sum": 500,
    "resp_pf": 300,
    "resp_sao2": 90,
    "renal_creat": 2.0,
    "renal_bun": 40,
    "renal_urine_sum": 500,
    "hep_bili": 2.0,
    "hep_ast": 100,
    "hep_alt": 100,
    "heme_plts": 100,
    "heme_hct": 30,
    "inflam_wbc_hi": 12,
    "inflam_wbc_lo": 4,
    "inflam_temp_hi": 38.3,
    "inflam_temp_lo": 36,
    "neuro_gcs": 13,
    "card_tropi": 0.4,
    "card_tropt": 0.1,
    "metab_ph_lo": 7.30,
    "metab_ph_hi": 7.50,
    "metab_glu_lo": 70,
    "metab_glu_hi": 180,
    "metab_hco3_lo": 18,
    "chronic_age": 65,
    "acute_lact": 2.0,
    "acute_map": 65,
    "acute_gcs": 13,
}
LATENT_ORDER = [
    "Severity", "Shock", "RespFail", "RenalFail", "HepFail", "HemeFail",
    "Inflam", "NeuroFail", "CardInj", "Metab", "ChronicRisk", "AcuteInsult",
]


def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply rule-based PhysioNet latent variable tags."
    )
    parser.add_argument("--dataset-config-csv", default=None)
    parser.add_argument("--pkl-path", default=None)
    parser.add_argument("--output-csv-path", default=None)
    parser.add_argument("--optimized", type=str_to_bool, default=None)
    parser.add_argument("--thresholds-path", default=None)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args()

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


def load_optimal_thresholds(path):
    """
    Parses optimal_thresholds.txt into a dict.
    """
    thresholds = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if ":" not in line:
                continue

            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            try:
                thresholds[key] = float(val)
            except ValueError:
                continue

    print(f"[INFO] Loaded {len(thresholds)} optimized thresholds")
    return thresholds


def get_thr(thr_dict, key, default):
    if thr_dict is None:
        return default
    return thr_dict.get(key, default)

# ============================================================
# 3. DECISION TREES
# ============================================================


def tag_severity(row, thr=None):
    """
    Early global severity latent based on admission/early observed instability.
    Severity = 1 if at least severity_min_count early abnormal domains are present.
    """

    circulatory = int(
        (row.get("MAP_first", np.inf) < get_thr(thr, "severity_map", 70)) or
        (row.get("SysABP_first", np.inf) <= get_thr(thr, "severity_sysabp", 100))
    )

    neurologic = int(
        row.get("GCS_first", 15) < get_thr(thr, "severity_gcs", 15)
    )

    respiratory = int(
        (row.get("RespRate_first", -np.inf) >= get_thr(thr, "severity_resprate", 22)) or
        (row.get("SaO2_first", 100) < get_thr(thr, "severity_sao2", 92))
    )

    metabolic = int(
        (row.get("Lactate_first", 0) > get_thr(thr, "severity_lact", 2.0)) or
        (row.get("pH_first", 7.4) < get_thr(thr, "severity_ph", 7.30)) or
        (row.get("HCO3_first", 25) < get_thr(thr, "severity_hco3", 18))
    )

    renal = int(
        row.get("Creatinine_first", 0) >= get_thr(thr, "severity_creat", 2.0)
    )

    score = circulatory + neurologic + respiratory + metabolic + renal
    return int(score >= int(round(get_thr(thr, "severity_min_count", 2))))


def tag_shock(row, thr=None):
    return int(
        (row.get("MAP_min", np.inf) < get_thr(thr, "shock_map", 65)) or
        (row.get("SysABP_min", np.inf) < get_thr(thr, "shock_sysabp", 90)) or
        (row.get("Lactate_max", 0) > get_thr(thr, "shock_lact", 2.0)) or
        (row.get("Urine_sum", np.inf) < get_thr(thr, "shock_urine_sum", 500))
    )


def tag_respfail(row, thr=None):
    fio2 = row.get("FiO2_min", np.nan)
    pao2 = row.get("PaO2_min", np.nan)
    pf_ratio = pao2 / fio2 if pd.notna(fio2) and fio2 > 0 and pd.notna(pao2) else np.nan

    return int(
        ((not np.isnan(pf_ratio)) and (pf_ratio < get_thr(thr, "resp_pf", 300))) or
        (row.get("SaO2_min", 100) < get_thr(thr, "resp_sao2", 90)) or
        (row.get("MechVent_max", 0) == 1)
    )


def tag_renalfail(row, thr=None):
    return int(
        (row.get("Creatinine_max", 0) > get_thr(thr, "renal_creat", 2.0)) or
        (row.get("BUN_max", 0) > get_thr(thr, "renal_bun", 40)) or
        (row.get("Urine_sum", np.inf) < get_thr(thr, "renal_urine_sum", 500))
    )


def tag_hepfail(row, thr=None):
    return int(
        (row.get("Bilirubin_max", 0) > get_thr(thr, "hep_bili", 2.0)) or
        (row.get("AST_max", 0) > get_thr(thr, "hep_ast", 100)) or
        (row.get("ALT_max", 0) > get_thr(thr, "hep_alt", 100))
    )


def tag_hemefail(row, thr=None):
    return int(
        (row.get("Platelets_min", np.inf) < get_thr(thr, "heme_plts", 100)) or
        (row.get("HCT_min", np.inf) < get_thr(thr, "heme_hct", 30))
    )


def tag_inflam(row, thr=None):
    return int(
        (row.get("WBC_max", 0) > get_thr(thr, "inflam_wbc_hi", 12)) or
        (row.get("WBC_min", 10) < get_thr(thr, "inflam_wbc_lo", 4)) or
        (row.get("Temp_max", 36) > get_thr(thr, "inflam_temp_hi", 38.3)) or
        (row.get("Temp_min", 37) < get_thr(thr, "inflam_temp_lo", 36))
    )


def tag_neurofail(row, thr=None):
    return int(
        row.get("GCS_min", 15) < get_thr(thr, "neuro_gcs", 13)
    )


def tag_cardinj(row, thr=None):
    return int(
        (row.get("TropI_max", 0) > get_thr(thr, "card_tropi", 0.4)) or
        (row.get("TropT_max", 0) > get_thr(thr, "card_tropt", 0.1))
    )


def tag_metab(row, thr=None):
    return int(
        (row.get("pH_min", 7.4) < get_thr(thr, "metab_ph_lo", 7.30)) or
        (row.get("pH_max", 7.4) > get_thr(thr, "metab_ph_hi", 7.50)) or
        (row.get("Glucose_min", 100) < get_thr(thr, "metab_glu_lo", 70)) or
        (row.get("Glucose_max", 100) > get_thr(thr, "metab_glu_hi", 180)) or
        (row.get("HCO3_min", 25) < get_thr(thr, "metab_hco3_lo", 18))
    )


def tag_chronicrisk(row, thr=None):
    return int(
        (row.get("Age_first", 0) > get_thr(thr, "chronic_age", 65)) or
        (row.get("ICUType_first", 0) in [2, 3])
    )


def tag_acuteinsult(row, thr=None):
    return int(
        (row.get("Lactate_first", 0) > get_thr(thr, "acute_lact", 2.0)) or
        (row.get("MAP_first", 100) < get_thr(thr, "acute_map", 65)) or
        (row.get("GCS_first", 15) < get_thr(thr, "acute_gcs", 13))
    )


def get_latent_decision_trees(thr=None):
    """
    Returns dict: latent_variable_name -> decision_function(row)->0/1
    Uses functools.partial so functions remain pickle-safe.
    """
    print("[4/5] Initializing latent decision trees...")

    all_trees = {
        "Severity": partial(tag_severity, thr=thr),
        "Shock": partial(tag_shock, thr=thr),
        "RespFail": partial(tag_respfail, thr=thr),
        "RenalFail": partial(tag_renalfail, thr=thr),
        "HepFail": partial(tag_hepfail, thr=thr),
        "HemeFail": partial(tag_hemefail, thr=thr),
        "Inflam": partial(tag_inflam, thr=thr),
        "NeuroFail": partial(tag_neurofail, thr=thr),
        "CardInj": partial(tag_cardinj, thr=thr),
        "Metab": partial(tag_metab, thr=thr),
        "ChronicRisk": partial(tag_chronicrisk, thr=thr),
        "AcuteInsult": partial(tag_acuteinsult, thr=thr),
    }
    trees = {latent: all_trees[latent] for latent in LATENT_ORDER if latent in all_trees}

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

    if OPTIMIZED:
        print("[INFO] Using optimized thresholds")
        thr = load_optimal_thresholds(THRESHOLDS_PATH)
    else:
        print("[INFO] Using default thresholds")
        thr = dict(DEFAULT_THRESHOLDS)

    decision_trees = get_latent_decision_trees(thr)

    latent_tags_df = tag_all_patients(summary_df, decision_trees)

    print("\nSaving results to CSV...")
    latent_tags_df.to_csv(output_csv_path, index=False)

    print("\n=== Pipeline completed successfully ✅ ===")
    print(f"Output saved to: {output_csv_path}\n")

    return latent_tags_df, decision_trees


def main():
    global pkl_path
    global output_csv_path
    global OPTIMIZED
    global THRESHOLDS_PATH
    global DEFAULT_THRESHOLDS
    global LATENT_ORDER

    args = parse_args()
    config = load_dataset_config("physionet", args.dataset_config_csv)
    pkl_path = str(get_config_scalar(config, "TAGGING_PKL_PATH", pkl_path))
    output_csv_path = str(
        get_config_scalar(config, "TAGGING_OUTPUT_CSV_PATH", output_csv_path)
    )
    THRESHOLDS_PATH = str(get_config_scalar(config, "THRESHOLDS_PATH", THRESHOLDS_PATH))
    DEFAULT_THRESHOLDS = dict(
        get_config_scalar(config, "DEFAULT_THRESHOLDS", DEFAULT_THRESHOLDS)
    )
    LATENT_ORDER = list(get_config_list(config, "LATENT_ORDER", LATENT_ORDER) or [])
    OPTIMIZED = bool(
        resolve_with_precedence(
            args.optimized,
            config,
            "OPTIMIZED",
            get_config_bool(config, "OPTIMIZED", OPTIMIZED),
        )
    )

    if args.pkl_path is not None:
        pkl_path = args.pkl_path
    if args.output_csv_path is not None:
        output_csv_path = args.output_csv_path
    if args.thresholds_path is not None:
        THRESHOLDS_PATH = args.thresholds_path

    latent_tags_df, decision_trees = run_latent_tagging_pipeline(pkl_path, output_csv_path)
    print(latent_tags_df.head())

    trees_path = output_csv_path.replace(".csv", "_trees.pkl")
    with open(trees_path, "wb") as f:
        pickle.dump(decision_trees, f)

    print(f"Decision trees dictionary saved to: {trees_path}")


if __name__ == "__main__":
    main()

