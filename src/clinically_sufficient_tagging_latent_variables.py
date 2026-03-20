from __future__ import annotations

import os
import pickle
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
pkl_path = "../../data/processed/physionet2012_ts_oc_ids.pkl"
output_csv_path = "latent_tags_clinical.csv"

OPTIMIZED = False
THRESHOLDS_PATH = "../../data/optimal_thresholds.txt"
SAVE_STAGE_DETAILS = False
STAGE_OUTPUT_PATH = output_csv_path.replace(".csv", "_stages.csv")

WINDOWS = {
    "w_0_6h": (0, 6 * 60),
    "w_6_12h": (6 * 60, 12 * 60),
    "w_12_24h": (12 * 60, 24 * 60),
    "w_0_24h": (0, 24 * 60),
    "w_24_48h": (24 * 60, 48 * 60 + 1),
    "w_0_48h": (0, 48 * 60 + 1),
}

TARGET_VARIABLES = [
    "Age", "Gender", "Height", "Weight",
    "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4",
    "SysABP", "DiasABP", "MAP", "NISysABP", "NIDiasABP", "NIMAP", "HR",
    "Lactate", "Urine",
    "RespRate", "PaO2", "SaO2", "PaCO2", "pH", "MechVent", "FiO2",
    "Creatinine", "BUN", "K", "Na", "Mg", "HCO3",
    "ALT", "AST", "Bilirubin", "ALP", "Albumin", "Cholesterol",
    "Platelets", "HCT", "WBC", "Temp", "GCS", "TropI", "TropT", "Glucose",
]


# ============================================================
# 1. Load data
# ============================================================
def load_physionet_pickle(path: str):
    print("[1/5] Loading PhysioNet pickle...")
    with open(path, "rb") as f:
        ts, oc, ts_ids = pickle.load(f)
    print(f"      Loaded ts: {len(ts):,} rows")
    print(f"      Loaded oc: {len(oc):,} rows")
    print(f"      Loaded ts_ids: {len(ts_ids):,} patients")
    return ts, oc, ts_ids


# ============================================================
# 2. Build patient contexts
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _to_numpy_sorted(g_var: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    g_var = g_var.sort_values("minute")
    return g_var["minute"].to_numpy(dtype=float), g_var["value"].to_numpy(dtype=float)


def infer_icu_type(static_values: Dict[str, float]) -> float:
    for k, v in static_values.items():
        if k.startswith("ICUType_") and _safe_float(v, 0.0) == 1.0:
            try:
                return float(k.split("_")[-1])
            except Exception:
                pass
    return np.nan


def build_patient_contexts(ts: pd.DataFrame) -> List[Dict[str, object]]:
    print("[2/5] Building patient contexts from long irregular time-series...")
    ts = ts.copy()
    ts["ts_id"] = ts["ts_id"].astype(str)
    ts = ts[ts["variable"].isin(TARGET_VARIABLES)].copy()
    ts = ts.sort_values(["ts_id", "variable", "minute"])

    contexts: List[Dict[str, object]] = []
    for pid, g_pid in tqdm(ts.groupby("ts_id"), desc="      Building contexts", unit="patient"):
        series: Dict[str, Dict[str, np.ndarray]] = {}
        static_values: Dict[str, float] = {}

        for var, g_var in g_pid.groupby("variable"):
            times, values = _to_numpy_sorted(g_var)
            series[var] = {"times": times, "values": values}
            if var in {"Age", "Gender", "Height", "Weight", "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4"}:
                static_values[var] = _safe_float(values[0]) if len(values) > 0 else np.nan

        static_values["ICUType"] = infer_icu_type(static_values)

        contexts.append({
            "ts_id": str(pid),
            "series": series,
            "static": static_values,
        })

    print(f"      Built {len(contexts):,} patient contexts")
    return contexts


# ============================================================
# 3. Time-series helpers
# ============================================================
def get_thr(thr_dict: Optional[Dict[str, float]], key: str, default: float) -> float:
    if thr_dict is None:
        return default
    return float(thr_dict.get(key, default))


def load_optimal_thresholds(path: str) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
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


def window_mask(times: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    start, end = window
    return (times >= start) & (times < end)


def get_window_values(context: Dict[str, object], var: str, window_name: str = "w_0_48h") -> np.ndarray:
    data = context["series"].get(var)
    if data is None:
        return np.array([], dtype=float)
    times = data["times"]
    values = data["values"]
    mask = window_mask(times, WINDOWS[window_name])
    return values[mask]


def get_window_times_values(
    context: Dict[str, object],
    var: str,
    window_name: str = "w_0_48h",
) -> Tuple[np.ndarray, np.ndarray]:
    data = context["series"].get(var)
    if data is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    times = data["times"]
    values = data["values"]
    mask = window_mask(times, WINDOWS[window_name])
    return times[mask], values[mask]


def first_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    _, values = get_window_times_values(context, var, window_name)
    return _safe_float(values[0], default) if len(values) > 0 else default


def last_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    _, values = get_window_times_values(context, var, window_name)
    return _safe_float(values[-1], default) if len(values) > 0 else default


def min_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    values = get_window_values(context, var, window_name)
    return _safe_float(np.min(values), default) if len(values) > 0 else default


def max_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    values = get_window_values(context, var, window_name)
    return _safe_float(np.max(values), default) if len(values) > 0 else default


def mean_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    values = get_window_values(context, var, window_name)
    return _safe_float(np.mean(values), default) if len(values) > 0 else default


def median_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    values = get_window_values(context, var, window_name)
    return _safe_float(np.median(values), default) if len(values) > 0 else default


def std_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    values = get_window_values(context, var, window_name)
    return _safe_float(np.std(values), default) if len(values) > 1 else default


def delta_value(context: Dict[str, object], var: str, window_name: str = "w_0_48h", default=np.nan) -> float:
    values = get_window_values(context, var, window_name)
    return _safe_float(values[-1] - values[0], default) if len(values) > 1 else default


def count_obs(context: Dict[str, object], var: str, window_name: str = "w_0_48h") -> int:
    return int(len(get_window_values(context, var, window_name)))


def count_above(context: Dict[str, object], var: str, cutoff: float, window_name: str = "w_0_48h") -> int:
    values = get_window_values(context, var, window_name)
    return int(np.sum(values > cutoff)) if len(values) > 0 else 0


def count_below(context: Dict[str, object], var: str, cutoff: float, window_name: str = "w_0_48h") -> int:
    values = get_window_values(context, var, window_name)
    return int(np.sum(values < cutoff)) if len(values) > 0 else 0


def proportion_above(context: Dict[str, object], var: str, cutoff: float, window_name: str = "w_0_48h") -> float:
    values = get_window_values(context, var, window_name)
    return float(np.mean(values > cutoff)) if len(values) > 0 else 0.0


def proportion_below(context: Dict[str, object], var: str, cutoff: float, window_name: str = "w_0_48h") -> float:
    values = get_window_values(context, var, window_name)
    return float(np.mean(values < cutoff)) if len(values) > 0 else 0.0


def urine_sum(context: Dict[str, object], window_name: str = "w_0_48h") -> float:
    values = get_window_values(context, "Urine", window_name)
    values = values[values >= 0]
    return float(np.sum(values)) if len(values) > 0 else np.nan


def worst_paired_pf_ratio(
    context: Dict[str, object],
    window_name: str = "w_0_48h",
    max_pair_gap_minutes: float = 120.0,
) -> float:
    t_pa, v_pa = get_window_times_values(context, "PaO2", window_name)
    t_fi, v_fi = get_window_times_values(context, "FiO2", window_name)
    if len(t_pa) == 0 or len(t_fi) == 0:
        return np.nan

    ratios: List[float] = []
    for tp, vp in zip(t_pa, v_pa):
        idx = int(np.argmin(np.abs(t_fi - tp)))
        gap = abs(float(t_fi[idx] - tp))
        fi = float(v_fi[idx])
        if gap <= max_pair_gap_minutes and fi > 0:
            ratios.append(float(vp / fi))
    return float(np.min(ratios)) if ratios else np.nan


def has_support(context: Dict[str, object], var: str, min_obs: int = 1, window_name: str = "w_0_48h") -> bool:
    return count_obs(context, var, window_name) >= min_obs


def stage_from_bins(value: float, bins: List[float], higher_worse: bool = True, default: int = 0) -> int:
    if pd.isna(value):
        return default
    if higher_worse:
        stage = 0
        for thr in bins:
            if value >= thr:
                stage += 1
        return stage
    stage = 0
    for thr in bins:
        if value <= thr:
            stage += 1
    return stage


# ============================================================
# 4. Latent stage functions (ordinal inside, binary outside)
# ============================================================
def stage_chronicrisk(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    age = context["static"].get("Age", np.nan)
    icu_type = context["static"].get("ICUType", np.nan)
    score = 0
    if pd.notna(age):
        if age >= get_thr(thr, "chronic_age_hi", 75):
            score += 2
        elif age >= get_thr(thr, "chronic_age_mid", 65):
            score += 1
    if icu_type in {2.0, 3.0}:
        score += 1
    weight = context["static"].get("Weight", np.nan)
    if pd.notna(weight) and weight < get_thr(thr, "chronic_weight_lo", 50):
        score += 1
    return int(min(score, 3))


def stage_acuteinsult(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    window = "w_0_6h"
    points = 0
    if min_value(context, "MAP", window, np.inf) < get_thr(thr, "acute_map_lo", 65):
        points += 1
    if max_value(context, "Lactate", window, 0.0) >= get_thr(thr, "acute_lact_hi", 2.0):
        points += 1
    if min_value(context, "GCS", window, 15.0) <= get_thr(thr, "acute_gcs_lo", 12):
        points += 1
    if min_value(context, "pH", window, 7.40) <= get_thr(thr, "acute_ph_lo", 7.30):
        points += 1
    return int(min(points, 3))


def stage_shock(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    map_min = min_value(context, "MAP", "w_0_48h", np.inf)
    sys_min = min_value(context, "SysABP", "w_0_48h", np.inf)
    lact_max = max_value(context, "Lactate", "w_0_48h", 0.0)
    lact_count = count_above(context, "Lactate", get_thr(thr, "shock_lact_mild", 2.0), "w_0_48h")
    map_low_count = count_below(context, "MAP", get_thr(thr, "shock_map_mild", 65), "w_0_48h")
    urine24 = urine_sum(context, "w_0_24h")
    urine48 = urine_sum(context, "w_0_48h")

    severe = (
        map_min < get_thr(thr, "shock_map_severe", 60) and
        lact_max >= get_thr(thr, "shock_lact_severe", 4.0)
    )
    moderate = (
        map_low_count >= int(round(get_thr(thr, "shock_map_low_count", 2))) or
        sys_min < get_thr(thr, "shock_sysabp_moderate", 90) or
        lact_count >= int(round(get_thr(thr, "shock_lact_count", 2)))
    )
    oliguria = (
        (pd.notna(urine24) and urine24 < get_thr(thr, "shock_urine24_lo", 500)) or
        (pd.notna(urine48) and urine48 < get_thr(thr, "shock_urine48_lo", 1000))
    )

    if severe or (moderate and oliguria):
        return 3
    if moderate or lact_max >= get_thr(thr, "shock_lact_mild", 2.0):
        return 2
    if map_min < get_thr(thr, "shock_map_borderline", 70) or sys_min < get_thr(thr, "shock_sysabp_borderline", 100):
        return 1
    return 0


def stage_respfail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    pf = worst_paired_pf_ratio(context, "w_0_48h")
    sao2_min = min_value(context, "SaO2", "w_0_48h", 100.0)
    paco2_max = max_value(context, "PaCO2", "w_0_48h", 0.0)
    ph_min = min_value(context, "pH", "w_0_48h", 7.40)
    mechvent = max_value(context, "MechVent", "w_0_48h", 0.0) >= 1.0

    if (pd.notna(pf) and pf < get_thr(thr, "resp_pf_very_severe", 100)):
        return 4
    if (pd.notna(pf) and pf < get_thr(thr, "resp_pf_severe", 200)) or (paco2_max > get_thr(thr, "resp_paco2_severe", 60) and ph_min < get_thr(thr, "resp_ph_severe", 7.25)):
        return 3
    if (pd.notna(pf) and pf < get_thr(thr, "resp_pf_moderate", 300)) or sao2_min < get_thr(thr, "resp_sao2_moderate", 90):
        return 2
    if mechvent or sao2_min < get_thr(thr, "resp_sao2_mild", 92) or (pd.notna(pf) and pf < get_thr(thr, "resp_pf_mild", 350)):
        return 1
    return 0


def stage_renalfail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    creat_max = max_value(context, "Creatinine", "w_0_48h", np.nan)
    creat_first = first_value(context, "Creatinine", "w_0_48h", np.nan)
    creat_delta = creat_max - creat_first if pd.notna(creat_max) and pd.notna(creat_first) else np.nan
    bun_max = max_value(context, "BUN", "w_0_48h", np.nan)
    urine24 = urine_sum(context, "w_0_24h")
    urine48 = urine_sum(context, "w_0_48h")

    stage = 0
    if pd.notna(creat_max):
        stage = max(stage, stage_from_bins(creat_max, [1.2, 2.0, 3.5, 5.0], higher_worse=True))
    if pd.notna(creat_delta):
        if creat_delta >= get_thr(thr, "renal_creat_delta_severe", 1.5):
            stage = max(stage, 3)
        elif creat_delta >= get_thr(thr, "renal_creat_delta_mild", 0.3):
            stage = max(stage, 1)
    if pd.notna(urine24) and urine24 < get_thr(thr, "renal_urine24_lo", 500):
        stage = max(stage, 2)
    if pd.notna(urine48) and urine48 < get_thr(thr, "renal_urine48_lo", 800):
        stage = max(stage, 3)
    if pd.notna(bun_max) and bun_max >= get_thr(thr, "renal_bun_hi", 40):
        stage = max(stage, 1)
    return int(min(stage, 4))


def stage_hepfail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    bili = max_value(context, "Bilirubin", "w_0_48h", np.nan)
    ast = max_value(context, "AST", "w_0_48h", np.nan)
    alt = max_value(context, "ALT", "w_0_48h", np.nan)

    stage = 0
    if pd.notna(bili):
        stage = max(stage, stage_from_bins(bili, [1.2, 2.0, 6.0, 12.0], higher_worse=True))
    if stage == 0:
        if (pd.notna(ast) and ast >= get_thr(thr, "hep_ast_hi", 200)) or (pd.notna(alt) and alt >= get_thr(thr, "hep_alt_hi", 200)):
            stage = 1
        if (pd.notna(ast) and ast >= get_thr(thr, "hep_ast_very_hi", 500)) or (pd.notna(alt) and alt >= get_thr(thr, "hep_alt_very_hi", 500)):
            stage = 2
    return int(min(stage, 4))


def stage_hemefail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    platelets = min_value(context, "Platelets", "w_0_48h", np.nan)
    hct = min_value(context, "HCT", "w_0_48h", np.nan)

    stage = 0
    if pd.notna(platelets):
        if platelets < get_thr(thr, "heme_plts_very_severe", 20):
            stage = 4
        elif platelets < get_thr(thr, "heme_plts_severe", 50):
            stage = 3
        elif platelets < get_thr(thr, "heme_plts_moderate", 100):
            stage = 2
        elif platelets < get_thr(thr, "heme_plts_mild", 150):
            stage = 1
    if pd.notna(hct):
        if hct < get_thr(thr, "heme_hct_severe", 24):
            stage = max(stage, 2)
        elif hct < get_thr(thr, "heme_hct_mild", 30):
            stage = max(stage, 1)
    return int(stage)


def stage_inflam(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    points = 0
    if max_value(context, "WBC", "w_0_48h", 0.0) > get_thr(thr, "inflam_wbc_hi", 12):
        points += 1
    if min_value(context, "WBC", "w_0_48h", 10.0) < get_thr(thr, "inflam_wbc_lo", 4):
        points += 1
    if max_value(context, "Temp", "w_0_48h", 36.5) > get_thr(thr, "inflam_temp_hi", 38.3):
        points += 1
    if min_value(context, "Temp", "w_0_48h", 37.0) < get_thr(thr, "inflam_temp_lo", 36.0):
        points += 1
    if points >= 3:
        return 3
    if points == 2:
        return 2
    if points == 1:
        return 1
    return 0


def stage_neurofail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    gcs = min_value(context, "GCS", "w_0_48h", np.nan)
    if pd.isna(gcs):
        return 0
    if gcs < get_thr(thr, "neuro_gcs_very_severe", 6):
        return 4
    if gcs < get_thr(thr, "neuro_gcs_severe", 10):
        return 3
    if gcs < get_thr(thr, "neuro_gcs_moderate", 13):
        return 2
    if gcs < get_thr(thr, "neuro_gcs_mild", 15):
        return 1
    return 0


def stage_cardinj(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    tropi = max_value(context, "TropI", "w_0_48h", np.nan)
    tropt = max_value(context, "TropT", "w_0_48h", np.nan)
    stage = 0
    if pd.notna(tropi):
        if tropi >= get_thr(thr, "card_tropi_severe", 2.0):
            stage = max(stage, 3)
        elif tropi >= get_thr(thr, "card_tropi_moderate", 0.4):
            stage = max(stage, 2)
        elif tropi >= get_thr(thr, "card_tropi_mild", 0.04):
            stage = max(stage, 1)
    if pd.notna(tropt):
        if tropt >= get_thr(thr, "card_tropt_severe", 0.5):
            stage = max(stage, 3)
        elif tropt >= get_thr(thr, "card_tropt_moderate", 0.1):
            stage = max(stage, 2)
        elif tropt >= get_thr(thr, "card_tropt_mild", 0.03):
            stage = max(stage, 1)
    return int(stage)


def stage_metab(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    ph_min = min_value(context, "pH", "w_0_48h", np.nan)
    ph_max = max_value(context, "pH", "w_0_48h", np.nan)
    hco3_min = min_value(context, "HCO3", "w_0_48h", np.nan)
    lact_max = max_value(context, "Lactate", "w_0_48h", np.nan)
    glu_min = min_value(context, "Glucose", "w_0_48h", np.nan)
    glu_max = max_value(context, "Glucose", "w_0_48h", np.nan)

    severe = False
    moderate = False
    mild = False

    if pd.notna(ph_min) and ph_min < get_thr(thr, "metab_ph_severe_lo", 7.20):
        severe = True
    if pd.notna(ph_max) and ph_max > get_thr(thr, "metab_ph_severe_hi", 7.55):
        severe = True
    if pd.notna(lact_max) and lact_max >= get_thr(thr, "metab_lact_severe", 4.0):
        severe = True

    if pd.notna(ph_min) and ph_min < get_thr(thr, "metab_ph_moderate_lo", 7.30):
        moderate = True
    if pd.notna(ph_max) and ph_max > get_thr(thr, "metab_ph_moderate_hi", 7.50):
        moderate = True
    if pd.notna(hco3_min) and hco3_min < get_thr(thr, "metab_hco3_moderate", 18):
        moderate = True
    if pd.notna(glu_min) and glu_min < get_thr(thr, "metab_glu_lo", 70):
        moderate = True
    if pd.notna(glu_max) and glu_max > get_thr(thr, "metab_glu_hi", 180):
        moderate = True
    if pd.notna(lact_max) and lact_max >= get_thr(thr, "metab_lact_moderate", 2.0):
        moderate = True

    if pd.notna(hco3_min) and hco3_min < get_thr(thr, "metab_hco3_mild", 22):
        mild = True
    if pd.notna(glu_max) and glu_max > get_thr(thr, "metab_glu_mild_hi", 140):
        mild = True

    if severe:
        return 3
    if moderate:
        return 2
    if mild:
        return 1
    return 0


def stage_severity(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    organ_stages = [
        stage_shock(context, thr),
        stage_respfail(context, thr),
        stage_renalfail(context, thr),
        stage_hepfail(context, thr),
        stage_hemefail(context, thr),
        stage_inflam(context, thr),
        stage_neurofail(context, thr),
        stage_cardinj(context, thr),
        stage_metab(context, thr),
    ]
    acute = stage_acuteinsult(context, thr)
    chronic = stage_chronicrisk(context, thr)
    total = float(np.sum(organ_stages)) + 0.5 * acute + 0.5 * chronic

    if total >= get_thr(thr, "severity_total_very_high", 10.0):
        return 4
    if total >= get_thr(thr, "severity_total_high", 7.0):
        return 3
    if total >= get_thr(thr, "severity_total_moderate", 4.0):
        return 2
    if total >= get_thr(thr, "severity_total_mild", 2.0):
        return 1
    return 0


# ============================================================
# 5. Binary wrappers (CSV output keeps the original structure)
# ============================================================
def binary_from_stage(stage: int, positive_stage: int = 2) -> int:
    return int(stage >= positive_stage)


def tag_chronicrisk(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_chronicrisk(context, thr), positive_stage=int(round(get_thr(thr, "bin_chronicrisk_stage", 1))))


def tag_acuteinsult(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_acuteinsult(context, thr), positive_stage=int(round(get_thr(thr, "bin_acuteinsult_stage", 1))))


def tag_shock(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_shock(context, thr), positive_stage=int(round(get_thr(thr, "bin_shock_stage", 2))))


def tag_respfail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_respfail(context, thr), positive_stage=int(round(get_thr(thr, "bin_respfail_stage", 2))))


def tag_renalfail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_renalfail(context, thr), positive_stage=int(round(get_thr(thr, "bin_renalfail_stage", 2))))


def tag_hepfail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_hepfail(context, thr), positive_stage=int(round(get_thr(thr, "bin_hepfail_stage", 2))))


def tag_hemefail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_hemefail(context, thr), positive_stage=int(round(get_thr(thr, "bin_hemefail_stage", 2))))


def tag_inflam(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_inflam(context, thr), positive_stage=int(round(get_thr(thr, "bin_inflam_stage", 2))))


def tag_neurofail(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_neurofail(context, thr), positive_stage=int(round(get_thr(thr, "bin_neurofail_stage", 2))))


def tag_cardinj(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_cardinj(context, thr), positive_stage=int(round(get_thr(thr, "bin_cardinj_stage", 2))))


def tag_metab(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_metab(context, thr), positive_stage=int(round(get_thr(thr, "bin_metab_stage", 2))))


def tag_severity(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> int:
    return binary_from_stage(stage_severity(context, thr), positive_stage=int(round(get_thr(thr, "bin_severity_stage", 2))))


# ============================================================
# 6. Decision trees dict
# ============================================================
def get_latent_decision_trees(thr: Optional[Dict[str, float]] = None):
    print("[3/5] Initializing clinically grounded latent decision trees...")
    trees = {
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
    print(f"      Loaded {len(trees)} latent definitions")
    return trees


# ============================================================
# 7. Tagging
# ============================================================
def compute_stage_details(context: Dict[str, object], thr: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    return {
        "Severity_stage": stage_severity(context, thr),
        "Shock_stage": stage_shock(context, thr),
        "RespFail_stage": stage_respfail(context, thr),
        "RenalFail_stage": stage_renalfail(context, thr),
        "HepFail_stage": stage_hepfail(context, thr),
        "HemeFail_stage": stage_hemefail(context, thr),
        "Inflam_stage": stage_inflam(context, thr),
        "NeuroFail_stage": stage_neurofail(context, thr),
        "CardInj_stage": stage_cardinj(context, thr),
        "Metab_stage": stage_metab(context, thr),
        "ChronicRisk_stage": stage_chronicrisk(context, thr),
        "AcuteInsult_stage": stage_acuteinsult(context, thr),
    }


def tag_all_patients(contexts: List[Dict[str, object]], decision_trees, thr: Optional[Dict[str, float]] = None):
    print("[4/5] Applying latent tags to all patients...")
    tag_rows = []
    stage_rows = []

    for context in tqdm(contexts, desc="      Tagging patients", unit="patient"):
        pid = context["ts_id"]
        tag_row = {"ts_id": pid}
        for latent, func in decision_trees.items():
            tag_row[latent] = int(func(context))
        tag_rows.append(tag_row)

        if SAVE_STAGE_DETAILS:
            srow = {"ts_id": pid}
            srow.update(compute_stage_details(context, thr))
            stage_rows.append(srow)

    latent_df = pd.DataFrame(tag_rows)
    stage_df = pd.DataFrame(stage_rows) if SAVE_STAGE_DETAILS else None
    print(f"      Tag table shape: {latent_df.shape}")
    return latent_df, stage_df


# ============================================================
# 8. Main pipeline
# ============================================================
def run_latent_tagging_pipeline(pkl_path: str, output_csv_path: str):
    print("\n=== Starting Clinically Grounded PhysioNet Latent Tagging Pipeline ===\n")

    ts, oc, ts_ids = load_physionet_pickle(pkl_path)
    contexts = build_patient_contexts(ts)

    if OPTIMIZED:
        print("[INFO] Using optimized thresholds")
        thr = load_optimal_thresholds(THRESHOLDS_PATH)
    else:
        print("[INFO] Using default thresholds")
        thr = None

    decision_trees = get_latent_decision_trees(thr)
    latent_tags_df, stage_df = tag_all_patients(contexts, decision_trees, thr=thr)

    print("[5/5] Saving results...")
    latent_tags_df.to_csv(output_csv_path, index=False)

    with open(output_csv_path.replace(".csv", "_trees.pkl"), "wb") as f:
        pickle.dump(decision_trees, f)

    if SAVE_STAGE_DETAILS and stage_df is not None:
        stage_df.to_csv(STAGE_OUTPUT_PATH, index=False)
        print(f"      Saved stage details to: {STAGE_OUTPUT_PATH}")

    print("\n=== Pipeline completed successfully ✅ ===")
    print(f"Output saved to: {output_csv_path}")
    print(f"Decision trees dictionary saved to: {output_csv_path.replace('.csv', '_trees.pkl')}\n")

    return latent_tags_df, decision_trees


if __name__ == "__main__":
    latent_tags_df, decision_trees = run_latent_tagging_pipeline(pkl_path, output_csv_path)
    print(latent_tags_df.head())
