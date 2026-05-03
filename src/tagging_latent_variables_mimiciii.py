
"""
tagging_latent_variables_mimiciii.py

Rule-based latent variable tagging for MIMIC-III ICU stays.

What this script does
---------------------
1. Loads either:
   - a pre-aggregated patient/ICU-stay summary CSV, or
   - raw concept-level CSVs/tables exported from MIMIC-III SQL queries.
   - a PhysioNet-compatible MIMIC pickle: [ts, oc, ts_ids]
2. Computes clinically motivated summary features.
3. Applies pickle-safe rule-based decision trees for latent physiologic states.
4. Saves:
   - latent_tags.csv
   - latent_tags_with_features.csv
   - latent_decision_trees.pkl
   - validation_summary.json
   - prevalence.csv
   - mortality_by_tag.csv
   - cooccurrence_phi.csv

Important note
--------------
This script intentionally stays rule-based and interpretable.
It does NOT train a model for labeling.

Recommended workflow
--------------------
Best practical use is:
A. extract concept-level tables from MIMIC-III using SQL (labs, vitals, urine, vent, vasopressors, etc.)
B. export those as CSVs
C. run this script to aggregate + tag

The script also supports a simpler path:
- pass a prebuilt summary CSV with columns such as MAP_min, Lactate_max, GCS_min, etc.

Authoring note
--------------
Some raw MIMIC-III extraction details depend on your local SQL pipeline / ITEMID mappings.
Therefore this file includes:
- fully implemented decision trees
- a complete summary/tagging/validation pipeline
- hooks for raw concept CSV inputs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/tagging_latent_variables_mimiciii.py",
        fixed_dataset="mimic",
    )

import numpy as np
import pandas as pd

from dataset_config import (
    get_config_int,
    get_config_list,
    get_config_scalar,
    load_dataset_config,
)
from preprocess_mimic_iii_large_contract import canonicalize_stay_id_series


# ============================================================
# Configuration
# ============================================================

LATENT_ORDER = [
    "ChronicBurden",
    "AcuteInsult",
    "Severity",
    "Inflammation",
    "Shock",
    "RespFail",
    "RenalDysfunction",
    "HepaticDysfunction",
    "CoagDysfunction",
    "NeuroDysfunction",
    "CardiacInjury",
    "MetabolicDerangement",
]

DEFAULT_THRESHOLDS = {
    # Chronic burden
    "chronic_age": 65,
    "chronic_albumin": 3.0,

    # Acute insult
    "acute_emergency_types": {"EMERGENCY", "URGENT"},

    # Severity
    "severity_lactate": 4.0,
    "severity_ph": 7.20,
    "severity_gcs": 9,
    "severity_platelets": 50,
    "severity_creatinine": 3.5,
    "severity_bilirubin": 6.0,

    # Inflammation / SIRS
    "sirs_temp_hi": 38.0,
    "sirs_temp_lo": 36.0,
    "sirs_hr": 90,
    "sirs_rr": 20,
    "sirs_paco2": 32,
    "sirs_wbc_hi": 12.0,
    "sirs_wbc_lo": 4.0,
    "sirs_min_count": 2,

    # Shock
    "shock_map": 65,
    "shock_sbp": 90,
    "shock_lactate": 2.0,
    "shock_urine_24h_ml": 500.0,
    "shock_urine_6h_mlkg": 0.5,

    # Respiratory failure
    "resp_pf": 300.0,
    "resp_sf": 315.0,
    "resp_paco2": 45.0,
    "resp_ph": 7.35,
    "resp_spo2": 90.0,

    # Renal dysfunction
    "renal_creatinine_delta": 0.3,
    "renal_creatinine_ratio": 1.5,
    "renal_creatinine_abs": 2.0,
    "renal_urine_24h_ml": 500.0,
    "renal_urine_6h_mlkg": 0.5,

    # Hepatic dysfunction
    "hep_bilirubin": 2.0,
    "hep_ast": 1000.0,
    "hep_alt": 1000.0,

    # Coagulation dysfunction
    "coag_platelets": 100.0,
    "coag_inr": 1.5,

    # Neuro dysfunction
    "neuro_gcs": 13,

    # Cardiac injury fallback thresholds
    "troponin_t_fallback": 0.1,
    "troponin_i_fallback": 0.4,

    # Metabolic derangement
    "metab_ph": 7.30,
    "metab_hco3": 18.0,
    "metab_anion_gap": 16.0,
    "metab_lactate": 4.0,
    "metab_glucose_lo": 70.0,
    "metab_glucose_hi": 180.0,
}

CHRONIC_ICD_KEYWORDS = [
    "CHF", "HEART FAILURE", "COPD", "CHRONIC KIDNEY", "CKD", "CIRRHOSIS",
    "MALIGNANC", "CANCER", "DIABETES", "DEMENTIA", "CAD", "CORONARY",
    "ATRIAL FIB", "HYPERTENSION", "LIVER DISEASE", "ESRD",
]

ACUTE_ICD_KEYWORDS = [
    "SEPSIS", "SEPTIC", "PNEUMONIA", "RESPIRATORY FAILURE", "ARDS",
    "MYOCARDIAL INFARCTION", "STEMI", "NSTEMI", "STROKE", "INTRACRANIAL",
    "TRAUMA", "HEMORRHAGE", "SHOCK", "PANCREATITIS", "GI BLEED",
]


PICKLE_TS_SUMMARY_SPECS = {
    "Age": {"aliases": ["Age"], "stats": {"first": "Age"}},
    "Albumin": {"aliases": ["Albumin"], "stats": {"first": "Albumin_first"}},
    "Lactate": {"aliases": ["Lactate"], "stats": {"max": "Lactate_max"}},
    "pH": {"aliases": ["pH", "pH Blood"], "stats": {"min": "pH_min"}},
    "Platelets": {"aliases": ["Platelets", "Platelet Count"], "stats": {"min": "Platelets_min"}},
    "Creatinine": {
        "aliases": ["Creatinine", "Creatinine Blood"],
        "stats": {"first": "Creatinine_first", "max": "Creatinine_max"},
    },
    "Bilirubin": {"aliases": ["Bilirubin", "Bilirubin (Total)"], "stats": {"max": "Bilirubin_max"}},
    "Temperature": {"aliases": ["Temperature"], "stats": {"min": "Temperature_min", "max": "Temperature_max"}},
    "HR": {"aliases": ["HR"], "stats": {"max": "HR_max"}},
    "RR": {"aliases": ["RR"], "stats": {"max": "RR_max"}},
    "PaCO2": {"aliases": ["PaCO2", "PCO2"], "stats": {"min": "PaCO2_min", "max": "PaCO2_max"}},
    "WBC": {"aliases": ["WBC"], "stats": {"min": "WBC_min", "max": "WBC_max"}},
    "MAP": {"aliases": ["MAP", "MBP"], "stats": {"min": "MAP_min"}},
    "SBP": {"aliases": ["SBP"], "stats": {"min": "SBP_min"}},
    "PaO2": {"aliases": ["PaO2", "PO2"], "stats": {"min": "PaO2_min"}},
    "FiO2": {"aliases": ["FiO2"], "stats": {"max": "FiO2_max"}},
    "SpO2": {"aliases": ["SpO2", "O2 Saturation"], "stats": {"min": "SpO2_min"}},
    "INR": {"aliases": ["INR"], "stats": {"max": "INR_max"}},
    "TroponinT": {"aliases": ["TroponinT", "Troponin T"], "stats": {"max": "TroponinT_max"}},
    "TroponinI": {"aliases": ["TroponinI", "Troponin I"], "stats": {"max": "TroponinI_max"}},
    "Bicarbonate": {"aliases": ["Bicarbonate"], "stats": {"min": "Bicarbonate_min"}},
    "AnionGap": {"aliases": ["AnionGap", "Anion Gap"], "stats": {"max": "AnionGap_max"}},
    "Glucose": {
        "aliases": ["Glucose", "Glucose (Blood)", "Glucose (Whole Blood)", "Glucose (Serum)"],
        "stats": {"min": "Glucose_min", "max": "Glucose_max"},
    },
    "AST": {"aliases": ["AST"], "stats": {"max": "AST_max"}},
    "ALT": {"aliases": ["ALT"], "stats": {"max": "ALT_max"}},
}

PICKLE_GCS_COMPONENTS = ["GCS_eye", "GCS_motor", "GCS_verbal"]
PICKLE_URINE_VARIABLE = "Urine"
PICKLE_WEIGHT_VARIABLE = "Weight"
PICKLE_TS_BINARY_HELPERS = {
    "MechanicalVentilation_any": ["MechanicalVentilation", "Intubated"],
    "Vasopressors_any": ["Vasopressin", "Norepinephrine", "Epinephrine", "Dopamine", "Neosynephrine"],
}
PICKLE_OC_OPTIONAL_FIELDS = {
    "InHospitalMortality": ["InHospitalMortality", "in_hospital_mortality"],
    "AdmissionType": ["AdmissionType", "admission_type", "ADMISSION_TYPE"],
    "ChronicICD_any": ["ChronicICD_any"],
    "AcuteICD_any": ["AcuteICD_any"],
    "SuspectedInfection_any": ["SuspectedInfection_any"],
    "TroponinPositive_any": ["TroponinPositive_any"],
}
PICKLE_EXPECTED_SUMMARY_COLUMNS = [
    "Age",
    "Albumin_first",
    "AdmissionType",
    "ChronicICD_any",
    "AcuteICD_any",
    "MechanicalVentilation_any",
    "Vasopressors_any",
    "Lactate_max",
    "pH_min",
    "GCS_min",
    "Platelets_min",
    "Creatinine_first",
    "Creatinine_max",
    "Bilirubin_max",
    "Temperature_min",
    "Temperature_max",
    "HR_max",
    "RR_max",
    "PaCO2_min",
    "PaCO2_max",
    "WBC_min",
    "WBC_max",
    "SIRS_count_max",
    "MAP_min",
    "SBP_min",
    "UrineOutput_sum_24h",
    "UrineOutput_mlkg_6h_min",
    "PaO2_min",
    "FiO2_max",
    "PF_ratio_min",
    "SF_ratio_min",
    "SpO2_min",
    "Creatinine_delta",
    "Creatinine_ratio",
    "INR_max",
    "TroponinPositive_any",
    "TroponinT_max",
    "TroponinI_max",
    "Bicarbonate_min",
    "AnionGap_max",
    "Glucose_min",
    "Glucose_max",
    "AST_max",
    "ALT_max",
    "SuspectedInfection_any",
    "InHospitalMortality",
]
PROGRESS_EVERY = 500


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def print_progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == total or current % PROGRESS_EVERY == 0:
        print(f"      {label}: {current:,} / {total:,}")


def safe_float(x) -> float:
    if x is None:
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def is_notna(x) -> bool:
    return pd.notna(x)


def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def normalize_fio2_value(x):
    """
    Converts FiO2 to fraction when values are given as percentages.
    Examples:
        0.40 -> 0.40
        40 -> 0.40
        100 -> 1.0
    """
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 0:
        return np.nan
    if x > 1.5:
        return x / 100.0
    return x


def binary_phi(a: pd.Series, b: pd.Series) -> float:
    """
    Phi coefficient for two binary vectors.
    Returns np.nan when undefined.
    """
    a = a.fillna(0).astype(int)
    b = b.fillna(0).astype(int)

    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())

    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0:
        return np.nan
    return (n11 * n00 - n10 * n01) / denom


def first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def last_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[-1] if len(s) else np.nan


def standard_stats(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if len(s) == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "first": np.nan,
            "last": np.nan,
        }
    return {
        "min": s.min(),
        "max": s.max(),
        "mean": s.mean(),
        "first": s.iloc[0],
        "last": s.iloc[-1],
    }


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# ============================================================
# Decision tree functions (pickle-safe)
# ============================================================

def tag_chronic_burden(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    age = row.get("Age")
    albumin = row.get("Albumin_first")
    chronic_icd = row.get("ChronicICD_any", 0)

    return int(
        (is_notna(age) and age >= thr["chronic_age"]) or
        (chronic_icd == 1) or
        (is_notna(albumin) and albumin < thr["chronic_albumin"])
    )


def tag_acute_insult(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    admission_type = str(row.get("AdmissionType", "")).upper()
    acute_icd = row.get("AcuteICD_any", 0)
    mv_any = row.get("MechanicalVentilation_any", 0)
    vaso_any = row.get("Vasopressors_any", 0)

    return int(
        (admission_type in thr["acute_emergency_types"]) or
        (acute_icd == 1) or
        (mv_any == 1) or
        (vaso_any == 1)
    )


def tag_severity(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        (row.get("Vasopressors_any", 0) == 1) or
        (row.get("MechanicalVentilation_any", 0) == 1) or
        (is_notna(row.get("Lactate_max")) and row["Lactate_max"] >= thr["severity_lactate"]) or
        (is_notna(row.get("pH_min")) and row["pH_min"] <= thr["severity_ph"]) or
        (is_notna(row.get("GCS_min")) and row["GCS_min"] <= thr["severity_gcs"]) or
        (is_notna(row.get("Platelets_min")) and row["Platelets_min"] < thr["severity_platelets"]) or
        (is_notna(row.get("Creatinine_max")) and row["Creatinine_max"] >= thr["severity_creatinine"]) or
        (is_notna(row.get("Bilirubin_max")) and row["Bilirubin_max"] >= thr["severity_bilirubin"])
    )


def tag_inflammation(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    suspected = row.get("SuspectedInfection_any", 0) == 1
    sirs_count = row.get("SIRS_count_max")
    return int(
        suspected or
        (is_notna(sirs_count) and sirs_count >= thr["sirs_min_count"])
    )


def tag_shock(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        (row.get("Vasopressors_any", 0) == 1) or
        (is_notna(row.get("MAP_min")) and row["MAP_min"] < thr["shock_map"]) or
        (is_notna(row.get("SBP_min")) and row["SBP_min"] < thr["shock_sbp"]) or
        (is_notna(row.get("Lactate_max")) and row["Lactate_max"] > thr["shock_lactate"]) or
        (is_notna(row.get("UrineOutput_sum_24h")) and row["UrineOutput_sum_24h"] < thr["shock_urine_24h_ml"]) or
        (is_notna(row.get("UrineOutput_mlkg_6h_min")) and row["UrineOutput_mlkg_6h_min"] < thr["shock_urine_6h_mlkg"])
    )


def tag_respfail(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        (row.get("MechanicalVentilation_any", 0) == 1) or
        (is_notna(row.get("PF_ratio_min")) and row["PF_ratio_min"] < thr["resp_pf"]) or
        (is_notna(row.get("SF_ratio_min")) and row["SF_ratio_min"] < thr["resp_sf"]) or
        (
            is_notna(row.get("PaCO2_max")) and
            is_notna(row.get("pH_min")) and
            row["PaCO2_max"] >= thr["resp_paco2"] and
            row["pH_min"] < thr["resp_ph"]
        ) or
        (is_notna(row.get("SpO2_min")) and row["SpO2_min"] < thr["resp_spo2"])
    )


def tag_renal_dysfunction(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        (is_notna(row.get("Creatinine_delta")) and row["Creatinine_delta"] >= thr["renal_creatinine_delta"]) or
        (is_notna(row.get("Creatinine_ratio")) and row["Creatinine_ratio"] >= thr["renal_creatinine_ratio"]) or
        (is_notna(row.get("Creatinine_max")) and row["Creatinine_max"] >= thr["renal_creatinine_abs"]) or
        (is_notna(row.get("UrineOutput_mlkg_6h_min")) and row["UrineOutput_mlkg_6h_min"] < thr["renal_urine_6h_mlkg"]) or
        (is_notna(row.get("UrineOutput_sum_24h")) and row["UrineOutput_sum_24h"] < thr["renal_urine_24h_ml"])
    )


def tag_hepatic_dysfunction(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        (is_notna(row.get("Bilirubin_max")) and row["Bilirubin_max"] >= thr["hep_bilirubin"]) or
        (is_notna(row.get("AST_max")) and row["AST_max"] >= thr["hep_ast"]) or
        (is_notna(row.get("ALT_max")) and row["ALT_max"] >= thr["hep_alt"])
    )


def tag_coag_dysfunction(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        (is_notna(row.get("Platelets_min")) and row["Platelets_min"] < thr["coag_platelets"]) or
        (is_notna(row.get("INR_max")) and row["INR_max"] >= thr["coag_inr"])
    )


def tag_neuro_dysfunction(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        is_notna(row.get("GCS_min")) and row["GCS_min"] < thr["neuro_gcs"]
    )


def tag_cardiac_injury(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    if row.get("TroponinPositive_any", 0) == 1:
        return 1

    tropt = row.get("TroponinT_max")
    tropi = row.get("TroponinI_max")

    return int(
        (is_notna(tropt) and tropt >= thr["troponin_t_fallback"]) or
        (is_notna(tropi) and tropi >= thr["troponin_i_fallback"])
    )


def tag_metabolic_derangement(row: pd.Series, thr: dict = None) -> int:
    thr = thr or DEFAULT_THRESHOLDS
    return int(
        (is_notna(row.get("pH_min")) and row["pH_min"] < thr["metab_ph"]) or
        (is_notna(row.get("Bicarbonate_min")) and row["Bicarbonate_min"] < thr["metab_hco3"]) or
        (is_notna(row.get("AnionGap_max")) and row["AnionGap_max"] > thr["metab_anion_gap"]) or
        (is_notna(row.get("Lactate_max")) and row["Lactate_max"] >= thr["metab_lactate"]) or
        (is_notna(row.get("Glucose_min")) and row["Glucose_min"] < thr["metab_glucose_lo"]) or
        (is_notna(row.get("Glucose_max")) and row["Glucose_max"] >= thr["metab_glucose_hi"])
    )


def get_latent_decision_trees(thr: dict = None) -> Dict[str, Callable[[pd.Series], int]]:
    thr = thr or DEFAULT_THRESHOLDS
    return {
        "ChronicBurden": partial(tag_chronic_burden, thr=thr),
        "AcuteInsult": partial(tag_acute_insult, thr=thr),
        "Severity": partial(tag_severity, thr=thr),
        "Inflammation": partial(tag_inflammation, thr=thr),
        "Shock": partial(tag_shock, thr=thr),
        "RespFail": partial(tag_respfail, thr=thr),
        "RenalDysfunction": partial(tag_renal_dysfunction, thr=thr),
        "HepaticDysfunction": partial(tag_hepatic_dysfunction, thr=thr),
        "CoagDysfunction": partial(tag_coag_dysfunction, thr=thr),
        "NeuroDysfunction": partial(tag_neuro_dysfunction, thr=thr),
        "CardiacInjury": partial(tag_cardiac_injury, thr=thr),
        "MetabolicDerangement": partial(tag_metabolic_derangement, thr=thr),
    }


# ============================================================
# Raw concept table loading
# ============================================================

@dataclass
class RawConceptTables:
    admissions: Optional[pd.DataFrame] = None
    diagnoses: Optional[pd.DataFrame] = None
    vitals: Optional[pd.DataFrame] = None
    labs: Optional[pd.DataFrame] = None
    urine: Optional[pd.DataFrame] = None
    vaso: Optional[pd.DataFrame] = None
    vent: Optional[pd.DataFrame] = None
    cultures_antibiotics: Optional[pd.DataFrame] = None
    troponin_map: Optional[pd.DataFrame] = None


def maybe_read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_raw_concept_tables(args) -> RawConceptTables:
    return RawConceptTables(
        admissions=maybe_read_csv(args.admissions_csv),
        diagnoses=maybe_read_csv(args.diagnoses_csv),
        vitals=maybe_read_csv(args.vitals_csv),
        labs=maybe_read_csv(args.labs_csv),
        urine=maybe_read_csv(args.urine_csv),
        vaso=maybe_read_csv(args.vaso_csv),
        vent=maybe_read_csv(args.vent_csv),
        cultures_antibiotics=maybe_read_csv(args.infection_csv),
        troponin_map=maybe_read_csv(args.troponin_map_csv),
    )


def load_mimic_pickle_payload(pkl_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    print(f"      Loading canonical MIMIC pickle payload from: {os.path.abspath(pkl_path)}")
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise ValueError(
            "Expected pickle payload [ts, oc, ts_ids]; got an object with a different structure."
        )

    ts, oc, ts_ids = payload
    if not isinstance(ts, pd.DataFrame) or not isinstance(oc, pd.DataFrame):
        raise ValueError("Pickle payload must contain pandas DataFrames for ts and oc.")
    if not isinstance(ts_ids, (list, tuple)):
        raise ValueError("Pickle payload must contain ts_ids as a list or tuple.")

    require_columns(ts, ["ts_id", "minute", "variable", "value"], "ts")
    require_columns(oc, ["ts_id"], "oc")

    ts = ts.loc[:, ["ts_id", "minute", "variable", "value"]].copy()
    ts["ts_id"] = canonicalize_stay_id_series(ts["ts_id"])
    if ts["ts_id"].isna().any():
        raise ValueError("Pickle ts contains missing ts_id values after canonicalization.")
    ts["minute"] = pd.to_numeric(ts["minute"], errors="raise").astype(int)
    ts["variable"] = ts["variable"].astype(str)
    ts["value"] = pd.to_numeric(ts["value"], errors="coerce")
    ts = ts.sort_values(["ts_id", "minute", "variable"]).reset_index(drop=True)

    oc = oc.copy()
    oc["ts_id"] = canonicalize_stay_id_series(oc["ts_id"])
    if oc["ts_id"].isna().any():
        raise ValueError("Pickle oc contains missing ts_id values after canonicalization.")

    ts_ids_series = canonicalize_stay_id_series(pd.Series(list(ts_ids), dtype="object"))
    if ts_ids_series.isna().any():
        raise ValueError("Pickle ts_ids contains missing values after canonicalization.")
    ts_ids = ts_ids_series.tolist()
    if ts_ids != sorted(ts_ids):
        raise ValueError("ts_ids in the pickle must be sorted.")

    ts_ids_from_ts = sorted(ts["ts_id"].unique().tolist())
    if ts_ids != ts_ids_from_ts:
        raise ValueError("ts_ids in the pickle must match sorted(ts.ts_id.unique()).")

    oc_ids = set(oc["ts_id"].astype(str))
    if not oc_ids.issubset(set(ts_ids)):
        raise ValueError("All oc.ts_id values must be contained in ts_ids.")

    print(
        f"      Loaded canonical payload: ts rows={len(ts):,}, oc rows={len(oc):,}, "
        f"stays={len(ts_ids):,}"
    )
    return ts, oc, ts_ids


def validate_mimic_pickle_summary(summary_df: pd.DataFrame) -> None:
    if "InHospitalMortality" not in summary_df.columns:
        raise ValueError(
            "Processed MIMIC pickle is broken: summary construction could not recover "
            "InHospitalMortality from oc. Regenerate the processed MIMIC pickle and "
            "then regenerate the MIMIC latent tags."
        )

    outcome = pd.to_numeric(summary_df["InHospitalMortality"], errors="coerce")
    if int(outcome.notna().sum()) == 0:
        raise ValueError(
            "Processed MIMIC pickle is broken: merged InHospitalMortality is entirely "
            "missing after aligning ts_ids with oc. A known cause is misaligned stay "
            "identifiers such as '12345.0' versus '12345'. Regenerate the processed "
            "MIMIC pickle and then regenerate the MIMIC latent tags."
        )


def _aggregate_minimal_summary_stats_from_ts(ts: pd.DataFrame) -> pd.DataFrame:
    alias_to_target = {}
    for target_name, spec in PICKLE_TS_SUMMARY_SPECS.items():
        for alias in spec["aliases"]:
            alias_to_target[alias] = target_name

    work = ts.loc[ts["variable"].isin(alias_to_target), ["ts_id", "minute", "variable", "value"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["icustay_id"])

    work["target_name"] = work["variable"].map(alias_to_target)
    rows = []
    total_stays = int(work["ts_id"].nunique())
    print(f"      Aggregating canonical summary stats for {total_stays:,} stays")
    for stay_index, (stay_id, g_stay) in enumerate(work.groupby("ts_id", sort=False), start=1):
        row = {"icustay_id": stay_id}
        for target_name, g_var in g_stay.groupby("target_name", sort=False):
            stats = standard_stats(g_var.sort_values("minute")["value"])
            for stat_name, out_col in PICKLE_TS_SUMMARY_SPECS[target_name]["stats"].items():
                row[out_col] = stats[stat_name]
        rows.append(row)
        print_progress("Canonical summary stats aggregated", stay_index, total_stays)

    return pd.DataFrame(rows)


def _aggregate_gcs_min_from_ts(ts: pd.DataFrame) -> pd.DataFrame:
    available_variables = set(ts["variable"].unique())
    missing_components = [col for col in PICKLE_GCS_COMPONENTS if col not in available_variables]
    if missing_components:
        raise ValueError(
            "Pickle mode requires GCS_eye, GCS_motor, and GCS_verbal in ts to build GCS_min. "
            f"Missing: {missing_components}"
        )

    gcs = ts.loc[ts["variable"].isin(PICKLE_GCS_COMPONENTS), ["ts_id", "minute", "variable", "value"]].copy()
    if gcs.empty:
        raise ValueError("Pickle mode could not find any GCS component rows in ts.")

    gcs_wide = gcs.pivot_table(
        index=["ts_id", "minute"],
        columns="variable",
        values="value",
        aggfunc="mean",
    )
    gcs_wide = gcs_wide.dropna(subset=PICKLE_GCS_COMPONENTS)
    if gcs_wide.empty:
        return pd.DataFrame(columns=["icustay_id", "GCS_min"])

    gcs_wide["GCS_total"] = (
        gcs_wide["GCS_eye"] + gcs_wide["GCS_motor"] + gcs_wide["GCS_verbal"]
    )
    out = (
        gcs_wide.reset_index()
        .groupby("ts_id", as_index=False)["GCS_total"]
        .min()
        .rename(columns={"ts_id": "icustay_id", "GCS_total": "GCS_min"})
    )
    return out


def _get_first_weight_by_stay(ts: pd.DataFrame) -> pd.Series:
    weights = ts.loc[ts["variable"] == PICKLE_WEIGHT_VARIABLE, ["ts_id", "minute", "value"]].copy()
    if weights.empty:
        return pd.Series(dtype=float)

    weights = weights.dropna(subset=["value"]).sort_values(["ts_id", "minute"])
    if weights.empty:
        return pd.Series(dtype=float)

    return weights.groupby("ts_id")["value"].first()


def _aggregate_urine_from_ts(ts: pd.DataFrame) -> pd.DataFrame:
    urine = ts.loc[ts["variable"] == PICKLE_URINE_VARIABLE, ["ts_id", "minute", "value"]].copy()
    if urine.empty:
        return pd.DataFrame(columns=["icustay_id", "UrineOutput_sum_24h", "UrineOutput_mlkg_6h_min"])

    urine = urine.dropna(subset=["minute", "value"]).sort_values(["ts_id", "minute"])
    if urine.empty:
        return pd.DataFrame(columns=["icustay_id", "UrineOutput_sum_24h", "UrineOutput_mlkg_6h_min"])

    first_weight = _get_first_weight_by_stay(ts)
    rows = []
    total_stays = int(urine["ts_id"].nunique())
    print(f"      Aggregating urine features for {total_stays:,} stays")
    for stay_index, (stay_id, g) in enumerate(urine.groupby("ts_id", sort=False), start=1):
        row = {"icustay_id": stay_id}

        first_day = g.loc[(g["minute"] >= 0) & (g["minute"] <= 24 * 60), "value"].dropna()
        row["UrineOutput_sum_24h"] = float(first_day.sum()) if len(first_day) else np.nan

        weight = first_weight.get(stay_id, np.nan)
        if is_notna(weight) and weight > 0:
            minutes = g["minute"].to_numpy(dtype=float)
            values = g["value"].to_numpy(dtype=float)
            cumsum = np.cumsum(values)
            window_starts = np.searchsorted(minutes, minutes - 360, side="left")
            window_sums = cumsum.copy()
            mask = window_starts > 0
            window_sums[mask] = window_sums[mask] - cumsum[window_starts[mask] - 1]
            urine_rates = window_sums / weight / 6.0
            row["UrineOutput_mlkg_6h_min"] = float(np.nanmin(urine_rates)) if len(urine_rates) else np.nan
        else:
            row["UrineOutput_mlkg_6h_min"] = np.nan

        rows.append(row)
        print_progress("Urine features aggregated", stay_index, total_stays)

    return pd.DataFrame(rows)


def _aggregate_binary_any_from_ts(
    ts: pd.DataFrame,
    source_variables: Iterable[str],
    out_name: str,
) -> Tuple[pd.DataFrame, bool]:
    source_variables = [var for var in source_variables if var in set(ts["variable"].unique())]
    if not source_variables:
        return pd.DataFrame(columns=["icustay_id", out_name]), False

    work = ts.loc[ts["variable"].isin(source_variables), ["ts_id", "value"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["icustay_id", out_name]), True

    work[out_name] = (work["value"].fillna(0) > 0).astype(int)
    out = (
        work.groupby("ts_id", as_index=False)[out_name]
        .max()
        .rename(columns={"ts_id": "icustay_id"})
    )
    return out, True


def _merge_optional_oc_fields(summary_df: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for out_col, aliases in PICKLE_OC_OPTIONAL_FIELDS.items():
        for alias in aliases:
            if alias in oc.columns:
                rename_map[alias] = out_col
                break

    if not rename_map:
        return summary_df

    oc_small = oc.loc[:, ["ts_id", *rename_map.keys()]].copy()
    oc_small = oc_small.rename(columns={"ts_id": "icustay_id", **rename_map})
    if "InHospitalMortality" in oc_small.columns:
        oc_small["InHospitalMortality"] = pd.to_numeric(
            oc_small["InHospitalMortality"], errors="coerce"
        )

    oc_small = oc_small.drop_duplicates(subset=["icustay_id"])
    return summary_df.merge(oc_small, on="icustay_id", how="left")


def build_summary_df_from_ts_oc(
    ts: pd.DataFrame,
    oc: pd.DataFrame,
    ts_ids: List[str],
) -> pd.DataFrame:
    print(f"      Building canonical summary dataframe for {len(ts_ids):,} stays")
    summary = pd.DataFrame({"icustay_id": ts_ids})

    print("      Stage A: summary statistics from canonical time-series")
    summary = summary.merge(_aggregate_minimal_summary_stats_from_ts(ts), on="icustay_id", how="left")
    print("      Stage B: deriving GCS minima from canonical time-series")
    summary = summary.merge(_aggregate_gcs_min_from_ts(ts), on="icustay_id", how="left")
    print("      Stage C: aggregating urine output features")
    summary = summary.merge(_aggregate_urine_from_ts(ts), on="icustay_id", how="left")
    print("      Stage D: merging optional outcome/context columns from oc")
    summary = _merge_optional_oc_fields(summary, oc)

    for out_name, source_variables in PICKLE_TS_BINARY_HELPERS.items():
        print(f"      Stage E: deriving helper flag '{out_name}'")
        helper_df, available = _aggregate_binary_any_from_ts(ts, source_variables, out_name)
        if available:
            summary = summary.merge(helper_df, on="icustay_id", how="left")
        elif out_name not in summary.columns:
            summary[out_name] = np.nan

    summary = _compute_sirs_features(summary)
    sirs_inputs = [
        "Temperature_min", "Temperature_max", "HR_max", "RR_max", "PaCO2_min", "WBC_min", "WBC_max"
    ]
    sirs_available = [col for col in sirs_inputs if col in summary.columns]
    if sirs_available:
        missing_all_sirs_inputs = summary[sirs_available].isna().all(axis=1)
        summary.loc[missing_all_sirs_inputs, "SIRS_count_max"] = np.nan

    summary = _compute_derived_features(summary)

    for col in PICKLE_EXPECTED_SUMMARY_COLUMNS:
        if col not in summary.columns:
            summary[col] = np.nan

    validate_mimic_pickle_summary(summary)
    print(f"      Canonical summary dataframe ready: {summary.shape}")
    return summary


def load_summary_from_mimic_pickle(pkl_path: str) -> pd.DataFrame:
    ts, oc, ts_ids = load_mimic_pickle_payload(pkl_path)
    return build_summary_df_from_ts_oc(ts, oc, ts_ids)


# ============================================================
# ICD helper flags
# ============================================================

def add_icd_flags(
    admissions_df: pd.DataFrame,
    diagnoses_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    df = admissions_df.copy()

    if diagnoses_df is None or diagnoses_df.empty:
        df["ChronicICD_any"] = 0
        df["AcuteICD_any"] = 0
        return df

    dx = diagnoses_df.copy()
    for col in ["long_title", "SHORT_TITLE", "short_title", "diagnosis"]:
        if col in dx.columns:
            dx["dx_text"] = dx[col].astype(str)
            break
    else:
        dx["dx_text"] = ""

    dx["dx_text_u"] = dx["dx_text"].str.upper()

    chronic = (
        dx.groupby("icustay_id")["dx_text_u"]
        .apply(lambda s: int(any(any(k in text for k in CHRONIC_ICD_KEYWORDS) for text in s)))
        .rename("ChronicICD_any")
        .reset_index()
    )

    acute = (
        dx.groupby("icustay_id")["dx_text_u"]
        .apply(lambda s: int(any(any(k in text for k in ACUTE_ICD_KEYWORDS) for text in s)))
        .rename("AcuteICD_any")
        .reset_index()
    )

    df = df.merge(chronic, on="icustay_id", how="left")
    df = df.merge(acute, on="icustay_id", how="left")
    df["ChronicICD_any"] = df["ChronicICD_any"].fillna(0).astype(int)
    df["AcuteICD_any"] = df["AcuteICD_any"].fillna(0).astype(int)
    return df


# ============================================================
# Summary building from concept-level data
# ============================================================

def _aggregate_named_variable_events(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    variable_col: str,
    value_col: str,
    variables: Iterable[str],
) -> pd.DataFrame:
    """
    Aggregates long-format events into summary columns:
    Variable_min, Variable_max, Variable_mean, Variable_first, Variable_last
    """
    if df is None or df.empty:
        return pd.DataFrame({id_col: []})

    variables = list(variables)
    work = df.copy()
    work = work[work[variable_col].isin(variables)].copy()
    if work.empty:
        return pd.DataFrame({id_col: []})

    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[id_col, time_col])

    rows = []
    total_stays = int(work[id_col].nunique())
    print(
        f"      Aggregating {len(variables)} variables from '{variable_col}' "
        f"for {total_stays:,} stays"
    )
    for stay_index, (stay_id, g) in enumerate(work.groupby(id_col), start=1):
        g = g.sort_values(time_col)
        row = {id_col: stay_id}

        for var, sub in g.groupby(variable_col):
            stats = standard_stats(sub[value_col])
            for stat_name, stat_val in stats.items():
                row[f"{var}_{stat_name}"] = stat_val

        rows.append(row)
        print_progress("Raw-table summary aggregation", stay_index, total_stays)

    return pd.DataFrame(rows)


def _aggregate_binary_any(
    df: pd.DataFrame,
    id_col: str,
    binary_col: str,
    out_name: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({id_col: []})
    tmp = df.groupby(id_col)[binary_col].max().reset_index()
    tmp = tmp.rename(columns={binary_col: out_name})
    return tmp


def _aggregate_urine(
    urine_df: Optional[pd.DataFrame],
    id_col: str = "icustay_id",
    time_col: str = "charttime",
    value_col: str = "value",
    weight_col: str = "weight_kg",
) -> pd.DataFrame:
    if urine_df is None or urine_df.empty:
        return pd.DataFrame({id_col: []})

    u = urine_df.copy()
    u[time_col] = pd.to_datetime(u[time_col], errors="coerce")
    u[value_col] = pd.to_numeric(u[value_col], errors="coerce")
    if weight_col in u.columns:
        u[weight_col] = pd.to_numeric(u[weight_col], errors="coerce")

    rows = []
    for stay_id, g in u.groupby(id_col):
        g = g.sort_values(time_col).dropna(subset=[time_col])
        row = {id_col: stay_id}

        if len(g) == 0:
            rows.append(row)
            continue

        # 24h sum from available rows
        row["UrineOutput_sum_24h"] = g[value_col].sum(skipna=True)

        # Approximate 6h rolling normalized urine output if weight exists
        if weight_col in g.columns and g[weight_col].notna().any():
            weight = g[weight_col].dropna().iloc[0]
        else:
            weight = np.nan

        if len(g) >= 1 and is_notna(weight) and weight > 0:
            gg = g[[time_col, value_col]].dropna().copy()
            if len(gg):
                gg = gg.set_index(time_col).sort_index()
                # Rolling 6h total / weight / 6
                roll = gg[value_col].rolling("6H").sum() / weight / 6.0
                row["UrineOutput_mlkg_6h_min"] = roll.min() if len(roll) else np.nan
            else:
                row["UrineOutput_mlkg_6h_min"] = np.nan
        else:
            row["UrineOutput_mlkg_6h_min"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _aggregate_infection(
    inf_df: Optional[pd.DataFrame],
    id_col: str = "icustay_id",
    suspicion_col: str = "suspected_infection",
) -> pd.DataFrame:
    if inf_df is None or inf_df.empty:
        return pd.DataFrame({id_col: []})

    if suspicion_col not in inf_df.columns:
        tmp = inf_df.copy()
        tmp[suspicion_col] = 1
    else:
        tmp = inf_df.copy()

    out = tmp.groupby(id_col)[suspicion_col].max().reset_index()
    out = out.rename(columns={suspicion_col: "SuspectedInfection_any"})
    return out


def _compute_sirs_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    def sirs_count(row):
        count = 0
        if is_notna(row.get("Temperature_max")) and row["Temperature_max"] > DEFAULT_THRESHOLDS["sirs_temp_hi"]:
            count += 1
        elif is_notna(row.get("Temperature_min")) and row["Temperature_min"] < DEFAULT_THRESHOLDS["sirs_temp_lo"]:
            count += 1

        if is_notna(row.get("HR_max")) and row["HR_max"] > DEFAULT_THRESHOLDS["sirs_hr"]:
            count += 1

        rr_flag = (
            is_notna(row.get("RR_max")) and row["RR_max"] > DEFAULT_THRESHOLDS["sirs_rr"]
        )
        paco2_flag = (
            is_notna(row.get("PaCO2_min")) and row["PaCO2_min"] < DEFAULT_THRESHOLDS["sirs_paco2"]
        )
        if rr_flag or paco2_flag:
            count += 1

        wbc_hi = is_notna(row.get("WBC_max")) and row["WBC_max"] > DEFAULT_THRESHOLDS["sirs_wbc_hi"]
        wbc_lo = is_notna(row.get("WBC_min")) and row["WBC_min"] < DEFAULT_THRESHOLDS["sirs_wbc_lo"]
        if wbc_hi or wbc_lo:
            count += 1

        return count

    df["SIRS_count_max"] = df.apply(sirs_count, axis=1)
    return df


def _compute_derived_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    # FiO2 normalization
    for col in [c for c in df.columns if c.startswith("FiO2_")]:
        df[col] = df[col].apply(normalize_fio2_value)

    # PF ratio: prefer worst oxygenation proxy using PaO2_min and FiO2_max
    df["PF_ratio_min"] = [
        safe_div(a, b)
        for a, b in zip(df.get("PaO2_min", pd.Series(np.nan, index=df.index)),
                        df.get("FiO2_max", pd.Series(np.nan, index=df.index)))
    ]

    # SF ratio
    fio2_for_sf = df["FiO2_max"] if "FiO2_max" in df.columns else pd.Series(np.nan, index=df.index)
    spo2_min = df["SpO2_min"] if "SpO2_min" in df.columns else pd.Series(np.nan, index=df.index)
    df["SF_ratio_min"] = [safe_div(a, b) for a, b in zip(spo2_min, fio2_for_sf)]

    # Creatinine KDIGO helper
    if "Creatinine_max" in df.columns and "Creatinine_first" in df.columns:
        df["Creatinine_delta"] = df["Creatinine_max"] - df["Creatinine_first"]
        df["Creatinine_ratio"] = [
            safe_div(a, b) for a, b in zip(df["Creatinine_max"], df["Creatinine_first"])
        ]
    else:
        df["Creatinine_delta"] = np.nan
        df["Creatinine_ratio"] = np.nan

    return df


def build_summary_from_raw_tables(raw: RawConceptTables) -> pd.DataFrame:
    if raw.admissions is None or raw.admissions.empty:
        raise ValueError(
            "Raw-table mode requires at least admissions_csv with one row per ICU stay "
            "and columns including icustay_id. "
            "You can also skip raw mode and pass --summary_csv."
        )

    admissions = raw.admissions.copy()
    required_id = "icustay_id"
    if required_id not in admissions.columns:
        raise ValueError("admissions_csv must include column 'icustay_id'.")

    summary = admissions.copy()
    print(f"      Admissions rows loaded: {len(summary):,}")

    # ICD-based helper flags
    print("      Stage A: adding ICD-derived helper flags")
    summary = add_icd_flags(summary, raw.diagnoses)

    # Vitals aggregation
    if raw.vitals is not None and not raw.vitals.empty:
        print(f"      Stage B: aggregating vitals rows={len(raw.vitals):,}")
        vitals_summary = _aggregate_named_variable_events(
            raw.vitals,
            id_col="icustay_id",
            time_col="charttime",
            variable_col="variable",
            value_col="value",
            variables=[
                "HR", "SBP", "DBP", "MAP", "RR", "SpO2", "Temperature", "GCS", "FiO2"
            ],
        )
        summary = summary.merge(vitals_summary, on="icustay_id", how="left")
        print(f"      Summary shape after vitals merge: {summary.shape}")

    # Labs aggregation
    if raw.labs is not None and not raw.labs.empty:
        print(f"      Stage C: aggregating labs rows={len(raw.labs):,}")
        labs_summary = _aggregate_named_variable_events(
            raw.labs,
            id_col="icustay_id",
            time_col="charttime",
            variable_col="variable",
            value_col="value",
            variables=[
                "Lactate", "PaO2", "PaCO2", "pH", "Bicarbonate", "Creatinine",
                "BUN", "Sodium", "Potassium", "AST", "ALT", "Bilirubin",
                "Albumin", "Platelets", "WBC", "Glucose", "AnionGap",
                "TroponinT", "TroponinI", "INR",
            ],
        )
        summary = summary.merge(labs_summary, on="icustay_id", how="left")
        print(f"      Summary shape after labs merge: {summary.shape}")

    # Urine aggregation
    print("      Stage D: aggregating urine features")
    urine_summary = _aggregate_urine(raw.urine)
    summary = summary.merge(urine_summary, on="icustay_id", how="left")
    print(f"      Summary shape after urine merge: {summary.shape}")

    # Vasopressors
    if raw.vaso is not None and not raw.vaso.empty:
        print(f"      Stage E: aggregating vasopressor flags from {len(raw.vaso):,} rows")
        vaso = raw.vaso.copy()
        if "vasopressor" not in vaso.columns:
            vaso["vasopressor"] = 1
        vaso_summary = _aggregate_binary_any(vaso, "icustay_id", "vasopressor", "Vasopressors_any")
        summary = summary.merge(vaso_summary, on="icustay_id", how="left")

    # Mechanical ventilation
    if raw.vent is not None and not raw.vent.empty:
        print(f"      Stage F: aggregating ventilation flags from {len(raw.vent):,} rows")
        vent = raw.vent.copy()
        if "mechanical_ventilation" not in vent.columns:
            vent["mechanical_ventilation"] = 1
        vent_summary = _aggregate_binary_any(vent, "icustay_id", "mechanical_ventilation", "MechanicalVentilation_any")
        summary = summary.merge(vent_summary, on="icustay_id", how="left")

    # Infection flag
    print("      Stage G: aggregating infection suspicion flags")
    infection_summary = _aggregate_infection(raw.cultures_antibiotics)
    summary = summary.merge(infection_summary, on="icustay_id", how="left")

    # Troponin positivity map
    if raw.troponin_map is not None and not raw.troponin_map.empty:
        print(f"      Stage H: aggregating troponin positivity from {len(raw.troponin_map):,} rows")
        tmap = raw.troponin_map.copy()
        cols_needed = {"icustay_id", "troponin_positive"}
        if cols_needed.issubset(set(tmap.columns)):
            tpos = (
                tmap.groupby("icustay_id")["troponin_positive"]
                .max()
                .reset_index()
                .rename(columns={"troponin_positive": "TroponinPositive_any"})
            )
            summary = summary.merge(tpos, on="icustay_id", how="left")

    # Fill absent binary helpers
    for col in ["Vasopressors_any", "MechanicalVentilation_any", "SuspectedInfection_any", "TroponinPositive_any"]:
        if col not in summary.columns:
            summary[col] = 0
        summary[col] = summary[col].fillna(0).astype(int)

    summary = _compute_sirs_features(summary)
    summary = _compute_derived_features(summary)
    print(f"      Raw-table summary dataframe ready: {summary.shape}")
    return summary


# ============================================================
# Summary CSV mode
# ============================================================

def load_summary_csv(summary_csv: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "icustay_id" not in df.columns:
        if "patient_id" in df.columns:
            df = df.rename(columns={"patient_id": "icustay_id"})
        elif "stay_id" in df.columns:
            df = df.rename(columns={"stay_id": "icustay_id"})
        else:
            raise ValueError(
                "summary_csv must contain one of: icustay_id, patient_id, stay_id"
            )

    # Normalize binary helpers if present
    for col in ["Vasopressors_any", "MechanicalVentilation_any", "SuspectedInfection_any", "TroponinPositive_any", "ChronicICD_any", "AcuteICD_any"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df = _compute_sirs_features(df) if "SIRS_count_max" not in df.columns else df
    df = _compute_derived_features(df)
    return df


# ============================================================
# Tagging pipeline
# ============================================================

def apply_decision_trees(
    summary_df: pd.DataFrame,
    decision_trees: Dict[str, Callable[[pd.Series], int]],
) -> pd.DataFrame:
    rows = []
    total_rows = len(summary_df)
    print(f"      Applying latent decision trees to {total_rows:,} ICU stays")
    for row_index, (_, row) in enumerate(summary_df.iterrows(), start=1):
        out = {"icustay_id": row["icustay_id"]}
        for latent_name, fn in decision_trees.items():
            out[latent_name] = int(fn(row))
        rows.append(out)
        print_progress("Latent tags applied", row_index, total_rows)
    return pd.DataFrame(rows)


# ============================================================
# Validation
# ============================================================

def prevalence_table(latent_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(latent_df)
    for latent in LATENT_ORDER:
        if latent not in latent_df.columns:
            continue
        s = latent_df[latent].fillna(0).astype(int)
        rows.append({
            "latent": latent,
            "n_positive": int(s.sum()),
            "prevalence": float(s.mean()) if n > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def mortality_by_tag_table(
    latent_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if "InHospitalMortality" not in summary_df.columns:
        return pd.DataFrame(columns=["latent", "mortality_tag0", "mortality_tag1", "risk_ratio"])

    merged = latent_df.merge(
        summary_df[["icustay_id", "InHospitalMortality"]],
        on="icustay_id",
        how="left",
    )

    rows = []
    for latent in LATENT_ORDER:
        if latent not in merged.columns:
            continue

        g0 = merged.loc[merged[latent] == 0, "InHospitalMortality"]
        g1 = merged.loc[merged[latent] == 1, "InHospitalMortality"]

        m0 = float(g0.mean()) if len(g0) else np.nan
        m1 = float(g1.mean()) if len(g1) else np.nan
        rr = safe_div(m1, m0)

        rows.append({
            "latent": latent,
            "n_tag0": int((merged[latent] == 0).sum()),
            "n_tag1": int((merged[latent] == 1).sum()),
            "mortality_tag0": m0,
            "mortality_tag1": m1,
            "risk_ratio": rr,
        })

    return pd.DataFrame(rows)


def cooccurrence_phi_table(latent_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LATENT_ORDER if c in latent_df.columns]
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            mat.loc[c1, c2] = binary_phi(latent_df[c1], latent_df[c2])

    return mat


def sanity_checks(latent_df: pd.DataFrame) -> Dict[str, dict]:
    results = {}
    for latent in LATENT_ORDER:
        if latent not in latent_df.columns:
            continue
        s = latent_df[latent].fillna(0).astype(int)
        p = float(s.mean()) if len(s) else np.nan
        results[latent] = {
            "all_zero": bool((s == 0).all()),
            "all_one": bool((s == 1).all()),
            "prevalence": p,
            "flag_too_rare_lt_0_5pct": bool(is_notna(p) and p < 0.005),
            "flag_too_common_gt_95pct": bool(is_notna(p) and p > 0.95),
        }
    return results


def build_validation_summary(
    latent_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> dict:
    prevalence = prevalence_table(latent_df)
    mortality = mortality_by_tag_table(latent_df, summary_df)
    checks = sanity_checks(latent_df)

    return {
        "n_stays": int(len(latent_df)),
        "available_latents": [c for c in LATENT_ORDER if c in latent_df.columns],
        "prevalence": prevalence.to_dict(orient="records"),
        "mortality_by_tag": mortality.to_dict(orient="records"),
        "sanity_checks": checks,
        "notes": [
            "High-prevalence tags may indicate too-soft thresholds or cohort-specific severity.",
            "Very low-prevalence tags may indicate too-harsh thresholds or missing concept extraction.",
            "Interpret cardiac injury carefully if TroponinPositive_any is unavailable and fallback thresholds are used.",
            "RespFail is more robust when PaO2/FiO2 and SpO2/FiO2 are both available.",
            "RenalDysfunction is more robust when urine output and weight are available.",
        ],
    }


# ============================================================
# Saving
# ============================================================

def _prepare_output_df_with_ts_id(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    if "icustay_id" not in df.columns:
        raise ValueError(f"{df_name} must contain icustay_id before saving outputs.")

    output_df = df.copy()
    if "ts_id" in output_df.columns:
        existing_ts_id = canonicalize_stay_id_series(output_df["ts_id"])
        internal_ids = canonicalize_stay_id_series(output_df["icustay_id"])
        mismatch = existing_ts_id.notna() & internal_ids.notna() & (existing_ts_id != internal_ids)
        if bool(mismatch.any()):
            raise ValueError(
                f"{df_name} contains both ts_id and icustay_id with conflicting values; "
                "cannot standardize output schema safely."
            )
        output_df = output_df.drop(columns=["ts_id"])

    return output_df.rename(columns={"icustay_id": "ts_id"})


def save_outputs(
    output_dir: str,
    summary_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    decision_trees: dict,
    validation_summary: dict,
) -> None:
    ensure_dir(output_dir)

    tags_path = os.path.join(output_dir, "latent_tags.csv")
    merged_path = os.path.join(output_dir, "latent_tags_with_features.csv")
    trees_path = os.path.join(output_dir, "latent_decision_trees.pkl")
    prevalence_path = os.path.join(output_dir, "prevalence.csv")
    mortality_path = os.path.join(output_dir, "mortality_by_tag.csv")
    cooccur_path = os.path.join(output_dir, "cooccurrence_phi.csv")
    validation_path = os.path.join(output_dir, "validation_summary.json")

    print(f"      Saving output files under: {os.path.abspath(output_dir)}")
    output_summary_df = _prepare_output_df_with_ts_id(summary_df, "summary_df")
    output_latent_df = _prepare_output_df_with_ts_id(latent_df, "latent_df")

    output_latent_df.to_csv(tags_path, index=False)
    output_summary_df.merge(output_latent_df, on="ts_id", how="left").to_csv(merged_path, index=False)
    prevalence_table(latent_df).to_csv(prevalence_path, index=False)
    mortality_by_tag_table(latent_df, summary_df).to_csv(mortality_path, index=False)
    cooccurrence_phi_table(latent_df).to_csv(cooccur_path)

    with open(trees_path, "wb") as f:
        pickle.dump(decision_trees, f)

    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)
    print("      Finished saving latent tags, merged features, validation tables, and decision trees")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Rule-based latent variable tagging for MIMIC-III ICU stays.")
    p.add_argument(
        "--dataset-config-csv",
        default=None,
        help=(
            "Path to the dataset global-variables CSV. If omitted, use the default "
            "MIMIC config."
        ),
    )

    # Input modes
    p.add_argument("--summary_csv", type=str, default=None,
                   help="Pre-aggregated summary CSV with one row per ICU stay.")
    p.add_argument("--pkl_path", type=str, default=None,
                   help="PhysioNet-compatible MIMIC pickle storing [ts, oc, ts_ids].")

    # Raw concept CSVs
    p.add_argument("--admissions_csv", type=str, default=None)
    p.add_argument("--diagnoses_csv", type=str, default=None)
    p.add_argument("--vitals_csv", type=str, default=None)
    p.add_argument("--labs_csv", type=str, default=None)
    p.add_argument("--urine_csv", type=str, default=None)
    p.add_argument("--vaso_csv", type=str, default=None)
    p.add_argument("--vent_csv", type=str, default=None)
    p.add_argument("--infection_csv", type=str, default=None)
    p.add_argument("--troponin_map_csv", type=str, default=None)

    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )

    return p.parse_args()


def main():
    global LATENT_ORDER
    global DEFAULT_THRESHOLDS
    global CHRONIC_ICD_KEYWORDS
    global ACUTE_ICD_KEYWORDS
    global PICKLE_TS_SUMMARY_SPECS
    global PICKLE_GCS_COMPONENTS
    global PICKLE_URINE_VARIABLE
    global PICKLE_WEIGHT_VARIABLE
    global PICKLE_TS_BINARY_HELPERS
    global PICKLE_OC_OPTIONAL_FIELDS
    global PICKLE_EXPECTED_SUMMARY_COLUMNS
    global PROGRESS_EVERY

    args = parse_args()
    config = load_dataset_config("mimic", args.dataset_config_csv)
    LATENT_ORDER = list(get_config_list(config, "LATENT_ORDER", LATENT_ORDER) or [])
    DEFAULT_THRESHOLDS = dict(
        get_config_scalar(config, "DEFAULT_THRESHOLDS", DEFAULT_THRESHOLDS)
    )
    if isinstance(DEFAULT_THRESHOLDS.get("acute_emergency_types"), list):
        DEFAULT_THRESHOLDS["acute_emergency_types"] = set(
            DEFAULT_THRESHOLDS["acute_emergency_types"]
        )
    CHRONIC_ICD_KEYWORDS = list(
        get_config_list(config, "CHRONIC_ICD_KEYWORDS", CHRONIC_ICD_KEYWORDS) or []
    )
    ACUTE_ICD_KEYWORDS = list(
        get_config_list(config, "ACUTE_ICD_KEYWORDS", ACUTE_ICD_KEYWORDS) or []
    )
    PICKLE_TS_SUMMARY_SPECS = dict(
        get_config_scalar(config, "PICKLE_TS_SUMMARY_SPECS", PICKLE_TS_SUMMARY_SPECS)
    )
    PICKLE_GCS_COMPONENTS = list(
        get_config_list(config, "PICKLE_GCS_COMPONENTS", PICKLE_GCS_COMPONENTS) or []
    )
    PICKLE_URINE_VARIABLE = str(
        get_config_scalar(config, "PICKLE_URINE_VARIABLE", PICKLE_URINE_VARIABLE)
    )
    PICKLE_WEIGHT_VARIABLE = str(
        get_config_scalar(config, "PICKLE_WEIGHT_VARIABLE", PICKLE_WEIGHT_VARIABLE)
    )
    PICKLE_TS_BINARY_HELPERS = dict(
        get_config_scalar(config, "PICKLE_TS_BINARY_HELPERS", PICKLE_TS_BINARY_HELPERS)
    )
    PICKLE_OC_OPTIONAL_FIELDS = dict(
        get_config_scalar(config, "PICKLE_OC_OPTIONAL_FIELDS", PICKLE_OC_OPTIONAL_FIELDS)
    )
    PICKLE_EXPECTED_SUMMARY_COLUMNS = list(
        get_config_list(
            config,
            "PICKLE_EXPECTED_SUMMARY_COLUMNS",
            PICKLE_EXPECTED_SUMMARY_COLUMNS,
        )
        or []
    )
    PROGRESS_EVERY = int(get_config_int(config, "PROGRESS_EVERY", PROGRESS_EVERY) or PROGRESS_EVERY)

    if args.pkl_path is None and not args.summary_csv and not any([
        args.admissions_csv,
        args.diagnoses_csv,
        args.vitals_csv,
        args.labs_csv,
        args.urine_csv,
        args.vaso_csv,
        args.vent_csv,
        args.infection_csv,
        args.troponin_map_csv,
    ]):
        configured_pkl_path = get_config_scalar(config, "TAGGING_PKL_PATH", None)
        if configured_pkl_path is not None:
            args.pkl_path = str(configured_pkl_path)

    output_dir = args.output_dir
    if output_dir is None:
        configured_output_csv = get_config_scalar(
            config,
            "TAGGING_OUTPUT_CSV_PATH",
            None,
        )
        if configured_output_csv is not None:
            output_dir = os.path.dirname(str(configured_output_csv)) or "."
        else:
            output_dir = "mimiciii_latent_tags_output"

    ensure_dir(output_dir)
    print("=== Starting MIMIC-III latent tagging ===")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    summary_mode = args.summary_csv is not None
    pkl_mode = args.pkl_path is not None
    raw_mode = any([
        args.admissions_csv,
        args.diagnoses_csv,
        args.vitals_csv,
        args.labs_csv,
        args.urine_csv,
        args.vaso_csv,
        args.vent_csv,
        args.infection_csv,
        args.troponin_map_csv,
    ])

    num_modes = int(summary_mode) + int(pkl_mode) + int(raw_mode)
    if num_modes != 1:
        raise ValueError(
            "Provide exactly one input mode: --summary_csv, --pkl_path, "
            "or raw concept CSV inputs (at minimum --admissions_csv for raw mode)."
        )

    if summary_mode:
        print("[1/5] Loading summary CSV...")
        summary_df = load_summary_csv(args.summary_csv)
    elif pkl_mode:
        print("[1/5] Loading MIMIC pickle...")
        print("[2/5] Building summary dataframe from canonical ts/oc...")
        summary_df = load_summary_from_mimic_pickle(args.pkl_path)
    else:
        print("[1/5] Loading raw concept CSVs...")
        raw = load_raw_concept_tables(args)
        print(
            "      Raw table availability: "
            f"admissions={0 if raw.admissions is None else len(raw.admissions):,}, "
            f"diagnoses={0 if raw.diagnoses is None else len(raw.diagnoses):,}, "
            f"vitals={0 if raw.vitals is None else len(raw.vitals):,}, "
            f"labs={0 if raw.labs is None else len(raw.labs):,}, "
            f"urine={0 if raw.urine is None else len(raw.urine):,}, "
            f"vaso={0 if raw.vaso is None else len(raw.vaso):,}, "
            f"vent={0 if raw.vent is None else len(raw.vent):,}, "
            f"infection={0 if raw.cultures_antibiotics is None else len(raw.cultures_antibiotics):,}, "
            f"troponin_map={0 if raw.troponin_map is None else len(raw.troponin_map):,}"
        )
        print("[2/5] Building summary dataframe from raw concept tables...")
        summary_df = build_summary_from_raw_tables(raw)

    if "icustay_id" not in summary_df.columns:
        raise ValueError("Summary dataframe must contain icustay_id.")

    print(f"[3/5] Summary dataframe shape: {summary_df.shape}")

    decision_trees = get_latent_decision_trees()
    print(f"      Latent definitions loaded: {len(decision_trees)}")

    print("[4/5] Applying decision trees...")
    latent_df = apply_decision_trees(summary_df, decision_trees)
    print(f"      Latent tag dataframe shape: {latent_df.shape}")

    print("[5/5] Running validation and saving outputs...")
    validation_summary = build_validation_summary(latent_df, summary_df)
    save_outputs(output_dir, summary_df, latent_df, decision_trees, validation_summary)

    print("\nDone.")
    print(f"Saved outputs to: {os.path.abspath(output_dir)}")
    print("Files:")
    for fname in [
        "latent_tags.csv",
        "latent_tags_with_features.csv",
        "latent_decision_trees.pkl",
        "validation_summary.json",
        "prevalence.csv",
        "mortality_by_tag.csv",
        "cooccurrence_phi.csv",
    ]:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()
