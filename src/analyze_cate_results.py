from __future__ import annotations

import argparse
import csv
import pickle
import sys
import traceback
import types
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/analyze_cate_results.py",
        default_dataset="physionet",
    )

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from dataset_config import (
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_list,
    get_config_scalar,
    get_first_available,
    load_dataset_config,
)
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_MODEL = "physionet"
LATENT_TAGS_PATH = "../../data/predicted_latent_tags_230326_absolute_tags.csv"
PHYSIONET_PKL_PATH = "../../data/processed/physionet2012_ts_oc_ids.pkl"
CATE_RESULTS_DIR = "../../data/relevant_outputs/cate_outputs_predicted_230326"
PREFERRED_ENV_NAME = "econml310"

TOP_K_BENCHMARK_CONFOUNDERS = 1
SEED = 42
SAVE_CONTOUR_PLOT = True

OUTCOME_COL = "in_hospital_mortality"
DEFAULT_SENSITIVITY_ALPHA = 0.05
DEFAULT_SENSITIVITY_C_Y = 0.05
DEFAULT_SENSITIVITY_C_T = 0.05
DEFAULT_SENSITIVITY_RHO = 1.0
SENSITIVITY_GRID_STEPS = 21
ALLOWED_TOP_K_VALUES = {1, 3}
BACKGROUND_FEATURE_COLUMNS = [
    "Age", "Gender", "Weight",
    "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4",
]

BENCHMARK_SCORE_COLUMNS = [
    "rank",
    "confounder",
    "proxy_cf_y",
    "proxy_cf_d",
    "proxy_strength_score",
    "selected_as_candidate",
    "is_primary_candidate",
]

CLEAN_RUN_SUMMARY_COLUMNS = [
    "row_id",
    "treatment",
    "model_type",
    "estimator_class",
    "RV",
    "direct_rv",
    "sensitivity_interval",
    "sensitivity_interval_lb",
    "sensitivity_interval_ub",
    "direct_sensitivity_interval",
    "direct_sensitivity_interval_lb",
    "direct_sensitivity_interval_ub",
    "direct_sensitivity_summary_available",
    "proxy_primary_candidate",
    "proxy_cf_y",
    "proxy_cf_d",
    "proxy_strength_score",
    "proxy_robustness_ratio",
    "selected_benchmark_confounder",
    "real_benchmark_strength_score",
    "robustness_ratio",
    "analysis_rows",
    "residual_rows",
]

CONTROL_RUN_SUMMARY_COLUMNS = [
    "row_id",
    "treatment",
    "model_type",
    "estimator_class",
    "model_loaded_in_econml310",
    "cache_values_used",
    "saved_training_residuals_available",
    "saved_training_residuals_source",
    "rv_source",
    "sensitivity_interval_source",
    "sensitivity_summary_source",
    "estimator_summary_source",
    "residual_source",
    "real_benchmark_available",
    "real_benchmark_cf_y",
    "real_benchmark_cf_d",
    "real_benchmark_source",
    "contour_source",
    "contour_plot_path",
    "run_status",
    "report_path",
    "benchmark_scores_path",
    "warnings",
]

SensitivityParams = namedtuple("SensitivityParams", ["theta", "sigma", "nu", "cov"])

DIRECT_SENSITIVITY_SOURCE_ORDER = [
    "saved_training_direct",
    "loaded_estimator_direct",
    "compatibility_refit_direct",
    "saved_training_params_reconstructed",
    "fallback_manual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved CATE artifacts and benchmark sensitivity outputs."
    )
    parser.add_argument(
        "--model",
        choices=["physionet", "mimic"],
        default=DATASET_MODEL,
        help=f"Dataset selector for path defaults. Default: {DATASET_MODEL}",
    )
    parser.add_argument(
        "--dataset-config-csv",
        default=None,
        help=(
            "Path to the dataset global-variables CSV. If omitted, use the default "
            "config for --model."
        ),
    )
    parser.add_argument("--latent-tags-path", default=None)
    parser.add_argument("--physionet-pkl-path", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for all analysis outputs. Default: write into --results-dir, "
            "matching the current behavior."
        ),
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args()


def resolve_script_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def build_treatment_output_path(
    treatment_dir: Path,
    treatment: str,
    suffix: str,
    extension: str,
) -> Path:
    return treatment_dir / f"{treatment}_{suffix}.{extension}"


def build_run_output_csv(output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{output_dir.name}_{suffix}.csv"


def build_summary_row_id(model_type: str, treatment: str) -> str:
    safe_model_type = model_type or "unknown_model"
    return f"{safe_model_type}__{treatment}"


def finalize_rows(
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> List[Dict[str, Any]]:
    finalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        finalized_rows.append({
            field: row.get(field)
            for field in fieldnames
        })
    return finalized_rows


def format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return float(value.reshape(-1)[0])
        raise ValueError("Expected a scalar-like value but got an array")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected a scalar-like value but got a sequence")
        return coerce_float(value[0])
    return float(value)


def serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return value


def write_rows_to_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: serialize_csv_value(row.get(field))
                for field in fieldnames
            })


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def install_legacy_sensitivity_pickle_shim() -> None:
    try:
        import econml.validate.sensitivity_analysis  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    validate_module = sys.modules.get("econml.validate")
    if validate_module is None:
        validate_module = types.ModuleType("econml.validate")
        sys.modules["econml.validate"] = validate_module

    shim_module = types.ModuleType("econml.validate.sensitivity_analysis")
    shim_module.SensitivityParams = SensitivityParams
    sys.modules["econml.validate.sensitivity_analysis"] = shim_module
    setattr(validate_module, "sensitivity_analysis", shim_module)


def load_physionet_pickle(path: Path) -> Tuple[Any, Any, Any]:
    with open(path, "rb") as f:
        ts, oc, ts_ids = pickle.load(f)
    return ts, oc, ts_ids


def build_background_features(
    ts: pd.DataFrame,
    dataset_model: str | None = None,
    background_feature_columns: List[str] | None = None,
) -> pd.DataFrame:
    current_model = DATASET_MODEL if dataset_model is None else dataset_model
    configured_columns = list(background_feature_columns or BACKGROUND_FEATURE_COLUMNS)
    df = ts.copy().sort_values(["ts_id", "minute"])

    if current_model == "physionet":
        keep_vars = list(configured_columns)
    else:
        available_variables = set(df["variable"].astype(str).tolist())
        keep_vars = [col for col in configured_columns if col in available_variables]
    df = df[df["variable"].isin(keep_vars)].copy()

    first_vals = (
        df.groupby(["ts_id", "variable"], as_index=False)
        .first()[["ts_id", "variable", "value"]]
    )

    bg = first_vals.pivot(index="ts_id", columns="variable", values="value").reset_index()

    if current_model == "physionet":
        for col in [c for c in configured_columns if c.startswith("ICUType_")]:
            if col not in bg.columns:
                bg[col] = 0.0

    return bg


def load_analysis_dataframe(
    latent_tags_path: Path,
    physionet_pkl_path: Path,
) -> pd.DataFrame:
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
    latent_df["ts_id"] = latent_df["ts_id"].astype(str)

    ts, oc, _ = load_physionet_pickle(physionet_pkl_path)
    if OUTCOME_COL not in oc.columns:
        raise ValueError(
            f"Processed pickle is missing outcome column '{OUTCOME_COL}'. "
            f"Available oc columns: {list(oc.columns)}"
        )

    if OUTCOME_COL in latent_df.columns:
        latent_df = latent_df.drop(columns=[OUTCOME_COL])

    oc_small = oc[["ts_id", OUTCOME_COL]].copy().drop_duplicates(subset=["ts_id"])
    oc_small["ts_id"] = oc_small["ts_id"].astype(str)

    bg_df = build_background_features(
        ts,
        dataset_model=DATASET_MODEL,
        background_feature_columns=BACKGROUND_FEATURE_COLUMNS,
    )
    bg_df["ts_id"] = bg_df["ts_id"].astype(str)

    latent_bg_overlap = [
        column for column in bg_df.columns
        if column != "ts_id" and column in latent_df.columns
    ]
    if latent_bg_overlap:
        latent_df = latent_df.drop(columns=latent_bg_overlap)

    df = latent_df.merge(oc_small, on="ts_id", how="inner")
    df = df.merge(bg_df, on="ts_id", how="left")
    df = df.dropna(subset=[OUTCOME_COL]).copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    return df


def load_model_artifact(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    artifact = None
    warning_messages: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
        except ModuleNotFoundError as exc:
            if exc.name != "econml.validate.sensitivity_analysis":
                raise
            install_legacy_sensitivity_pickle_shim()
            warning_messages.append(
                "Used legacy pickle shim for econml.validate.sensitivity_analysis during artifact load."
            )
            with open(path, "rb") as f:
                artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise TypeError(f"Model artifact is not a dict: {path}")

    for item in caught:
        warning_messages.append(str(item.message))
    return artifact, dedupe_preserve_order(warning_messages)


def artifact_value_with_fallback(
    artifact: Dict[str, Any],
    key: str,
    *fallback_keys: str,
) -> Any:
    direct = dict(artifact.get("direct_diagnostics", {}))
    for candidate in (key, *fallback_keys):
        if candidate in artifact:
            return artifact.get(candidate)
        if candidate in direct:
            return direct.get(candidate)
    return None


def validate_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = [
        "estimator",
        "model_type",
        "treatment",
        "outcome_col",
        "confounders",
        "effect_modifiers",
        "feature_fill_values",
        "formula",
        "summary",
    ]
    missing = [key for key in required_keys if key not in artifact]
    if missing:
        raise KeyError(f"Artifact missing keys: {missing}")

    direct_diagnostics = dict(artifact.get("direct_diagnostics", {}))
    return {
        "estimator": artifact["estimator"],
        "artifact_schema_version": int(artifact.get("artifact_schema_version", 1)),
        "python_version": str(artifact.get("python_version", "")),
        "platform": str(artifact.get("platform", "")),
        "econml_version": str(artifact.get("econml_version", "")),
        "sklearn_version": str(artifact.get("sklearn_version", "")),
        "numpy_version": str(artifact.get("numpy_version", "")),
        "pandas_version": str(artifact.get("pandas_version", "")),
        "scipy_version": str(artifact.get("scipy_version", "")),
        "training_timestamp": str(artifact.get("training_timestamp", "")),
        "model_type": str(artifact["model_type"]),
        "treatment": str(artifact["treatment"]),
        "outcome_col": str(artifact["outcome_col"]),
        "confounders": list(artifact.get("confounders", [])),
        "effect_modifiers": list(artifact.get("effect_modifiers", [])),
        "cache_values_used": bool(artifact.get("cache_values_used", False)),
        "estimator_module": str(
            artifact.get("estimator_module", type(artifact["estimator"]).__module__)
        ),
        "estimator_class": str(
            artifact.get("estimator_class", type(artifact["estimator"]).__name__)
        ),
        "confounders_order": list(
            artifact.get("confounders_order", artifact.get("confounders", []))
        ),
        "effect_modifiers_order": list(
            artifact.get("effect_modifiers_order", artifact.get("effect_modifiers", []))
        ),
        "feature_fill_values": dict(artifact.get("feature_fill_values", {})),
        "formula": str(artifact.get("formula", "")),
        "summary": dict(artifact.get("summary", {})),
        "has_method_robustness_value": bool(
            artifact_value_with_fallback(artifact, "has_method_robustness_value")
        ),
        "has_method_sensitivity_interval": bool(
            artifact_value_with_fallback(artifact, "has_method_sensitivity_interval")
        ),
        "has_method_sensitivity_summary": bool(
            artifact_value_with_fallback(artifact, "has_method_sensitivity_summary")
        ),
        "has_method_summary": bool(
            artifact_value_with_fallback(artifact, "has_method_summary")
        ),
        "has_attr_residuals": bool(
            artifact_value_with_fallback(artifact, "has_attr_residuals")
        ),
        "saved_direct_rv": artifact_value_with_fallback(
            artifact, "saved_direct_rv", "direct_robustness_value"
        ),
        "saved_direct_rv_source": maybe_string(
            artifact_value_with_fallback(
                artifact, "saved_direct_rv_source", "direct_robustness_value_source"
            )
        ),
        "saved_direct_rv_error": maybe_string(
            artifact_value_with_fallback(
                artifact, "saved_direct_rv_error", "direct_robustness_value_error"
            )
        ),
        "saved_direct_sensitivity_interval": artifact_value_with_fallback(
            artifact, "saved_direct_sensitivity_interval", "direct_sensitivity_interval"
        ),
        "saved_direct_sensitivity_interval_source": maybe_string(
            artifact_value_with_fallback(
                artifact,
                "saved_direct_sensitivity_interval_source",
                "direct_sensitivity_interval_source",
            )
        ),
        "saved_direct_sensitivity_interval_error": maybe_string(
            artifact_value_with_fallback(
                artifact,
                "saved_direct_sensitivity_interval_error",
                "direct_sensitivity_interval_error",
            )
        ),
        "saved_direct_sensitivity_summary": artifact_value_with_fallback(
            artifact, "saved_direct_sensitivity_summary", "direct_sensitivity_summary"
        ),
        "saved_direct_sensitivity_summary_source": maybe_string(
            artifact_value_with_fallback(
                artifact,
                "saved_direct_sensitivity_summary_source",
                "direct_sensitivity_summary_source",
            )
        ),
        "saved_direct_sensitivity_summary_error": maybe_string(
            artifact_value_with_fallback(
                artifact,
                "saved_direct_sensitivity_summary_error",
                "direct_sensitivity_summary_error",
            )
        ),
        "saved_direct_estimator_summary_text": maybe_string(
            artifact_value_with_fallback(
                artifact,
                "saved_direct_estimator_summary_text",
                "direct_estimator_summary_text",
            )
        ),
        "saved_direct_estimator_summary_source": maybe_string(
            artifact_value_with_fallback(
                artifact,
                "saved_direct_estimator_summary_source",
                "direct_estimator_summary_source",
            )
        ),
        "saved_direct_estimator_summary_error": maybe_string(
            artifact_value_with_fallback(
                artifact,
                "saved_direct_estimator_summary_error",
                "direct_estimator_summary_error",
            )
        ),
        "saved_sensitivity_params_available": bool(
            artifact_value_with_fallback(artifact, "saved_sensitivity_params_available")
        ),
        "saved_sensitivity_params_source": maybe_string(
            artifact_value_with_fallback(artifact, "saved_sensitivity_params_source")
        ),
        "saved_sensitivity_params_error": maybe_string(
            artifact_value_with_fallback(artifact, "saved_sensitivity_params_error")
        ),
        "saved_sensitivity_params_serialized": artifact_value_with_fallback(
            artifact, "saved_sensitivity_params_serialized"
        ),
        "saved_training_residuals_available": bool(
            artifact_value_with_fallback(artifact, "saved_training_residuals_available")
        ),
        "saved_training_residuals_source": maybe_string(
            artifact_value_with_fallback(artifact, "saved_training_residuals_source")
        ),
        "saved_training_residuals_error": maybe_string(
            artifact_value_with_fallback(artifact, "saved_training_residuals_error")
        ),
        "saved_training_residuals_tuple_length": artifact_value_with_fallback(
            artifact, "saved_training_residuals_tuple_length"
        ),
        "direct_diagnostics": direct_diagnostics,
    }


def normalize_interval_or_none(
    value: Any,
) -> Tuple[Tuple[float | None, float | None] | None, str | None]:
    if not metric_has_value(value):
        return None, None
    try:
        return normalize_interval_value(value), None
    except Exception as exc:
        return None, format_exception(exc)


def sensitivity_params_from_serialized(value: Any) -> SensitivityParams:
    payload = value
    if isinstance(payload, SensitivityParams):
        return payload
    if isinstance(payload, dict) and "attributes" in payload:
        payload = payload["attributes"]
    if not isinstance(payload, dict):
        raise TypeError("Serialized sensitivity params must be a dict-like object")

    theta = payload.get("theta")
    sigma = payload.get("sigma", payload.get("sigma2"))
    nu = payload.get("nu", payload.get("nu2"))
    cov = payload.get("cov", payload.get("covariance"))

    missing = [
        key for key, item in {
            "theta": theta,
            "sigma": sigma,
            "nu": nu,
            "cov": cov,
        }.items()
        if item is None
    ]
    if missing:
        raise KeyError(f"Serialized sensitivity params missing fields: {missing}")

    return SensitivityParams(
        theta=np.asarray(theta, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        nu=np.asarray(nu, dtype=float),
        cov=np.asarray(cov, dtype=float),
    )


def extract_saved_training_direct_metrics(artifact: Dict[str, Any]) -> Dict[str, Any]:
    interval_value, interval_error = normalize_interval_or_none(
        artifact.get("saved_direct_sensitivity_interval")
    )

    out = {
        "rv": None,
        "rv_source": "",
        "rv_error": maybe_string(artifact.get("saved_direct_rv_error")),
        "interval": interval_value,
        "interval_source": "",
        "interval_error": interval_error or maybe_string(
            artifact.get("saved_direct_sensitivity_interval_error")
        ),
        "summary_text": maybe_string(artifact.get("saved_direct_sensitivity_summary")),
        "summary_source": "",
        "summary_error": maybe_string(
            artifact.get("saved_direct_sensitivity_summary_error")
        ),
        "estimator_summary_text": maybe_string(
            artifact.get("saved_direct_estimator_summary_text")
        ),
        "estimator_summary_source": "",
        "estimator_summary_error": maybe_string(
            artifact.get("saved_direct_estimator_summary_error")
        ),
    }

    if metric_has_value(artifact.get("saved_direct_rv")):
        try:
            out["rv"] = coerce_float(artifact.get("saved_direct_rv"))
        except Exception as exc:
            out["rv_error"] = format_exception(exc)
        else:
            out["rv_source"] = maybe_string(artifact.get("saved_direct_rv_source"))
            out["rv_error"] = ""

    if metric_has_value(interval_value):
        out["interval_source"] = maybe_string(
            artifact.get("saved_direct_sensitivity_interval_source")
        )

    if out["summary_text"]:
        out["summary_source"] = maybe_string(
            artifact.get("saved_direct_sensitivity_summary_source")
        )

    if out["estimator_summary_text"]:
        out["estimator_summary_source"] = maybe_string(
            artifact.get("saved_direct_estimator_summary_source")
        )

    return out


def empty_sensitivity_metrics() -> Dict[str, Any]:
    return {
        "rv": None,
        "rv_source": "",
        "rv_error": "",
        "interval": None,
        "interval_source": "",
        "interval_error": "",
        "summary_text": "",
        "summary_source": "",
        "summary_error": "",
        "estimator_summary_text": "",
        "estimator_summary_source": "",
        "estimator_summary_error": "",
        "rv_theta": None,
        "rv_theta_source": "",
    }


def metric_selected_from_source(
    candidate: Dict[str, Any],
    value_key: str,
    source_key: str,
    allowed_sources: Sequence[str],
) -> bool:
    return (
        metric_has_value(candidate.get(value_key))
        and maybe_string(candidate.get(source_key)) in set(allowed_sources)
    )


def prepare_treatment_matrices_from_artifact(
    df: pd.DataFrame,
    model_artifact: Dict[str, Any],
) -> Dict[str, Any]:
    treatment = model_artifact["treatment"]
    outcome_col = model_artifact["outcome_col"]
    confounders = list(model_artifact.get("confounders_order", model_artifact["confounders"]))
    effect_modifiers = list(
        model_artifact.get("effect_modifiers_order", model_artifact["effect_modifiers"])
    )
    fill_values = dict(model_artifact.get("feature_fill_values", {}))

    if treatment not in df.columns:
        raise ValueError(f"Treatment column '{treatment}' not found in analysis dataframe")
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in analysis dataframe")

    work_df = df.dropna(subset=[treatment, outcome_col]).copy()
    work_df[treatment] = pd.to_numeric(work_df[treatment], errors="coerce")
    work_df[outcome_col] = pd.to_numeric(work_df[outcome_col], errors="coerce")
    work_df = work_df.dropna(subset=[treatment, outcome_col]).copy()
    work_df[treatment] = work_df[treatment].astype(int)
    work_df[outcome_col] = work_df[outcome_col].astype(int)

    model_df = work_df.copy()
    ordered_features = list(dict.fromkeys(confounders + effect_modifiers))

    created_missing_columns: List[str] = []
    fill_map: Dict[str, float] = {}

    for col in ordered_features:
        if col not in model_df.columns:
            model_df[col] = np.nan
            created_missing_columns.append(col)

        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        fill_value = coerce_float(fill_values.get(col, 0.0))
        if fill_value is None:
            fill_value = 0.0
        fill_map[col] = float(fill_value)
        model_df[col] = model_df[col].fillna(fill_value)

    t_values = sorted(model_df[treatment].dropna().unique().tolist())
    if t_values != [0, 1]:
        raise ValueError(f"{treatment} must be binary 0/1. Found: {t_values}")

    y_values = sorted(model_df[outcome_col].dropna().unique().tolist())
    if not set(y_values).issubset({0, 1}):
        raise ValueError(f"{outcome_col} must be binary 0/1. Found: {y_values}")

    Y = model_df[outcome_col].astype(float).to_numpy()
    T = model_df[treatment].astype(int).to_numpy()
    W = model_df[confounders].astype(float).to_numpy() if confounders else None
    X = model_df[effect_modifiers].astype(float).to_numpy() if effect_modifiers else None

    return {
        "Y": Y,
        "T": T,
        "W": W,
        "X": X,
        "analysis_rows": int(len(model_df)),
        "outcome_rate": float(model_df[outcome_col].mean()),
        "treatment_rate": float(model_df[treatment].mean()),
        "confounders": confounders,
        "effect_modifiers": effect_modifiers,
        "created_missing_columns": created_missing_columns,
        "fill_map": fill_map,
    }


def metric_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(metric_has_value(item) for item in value)
    return True


def normalize_interval_value(value: Any) -> Tuple[float | None, float | None] | None:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (coerce_float(value[0]), coerce_float(value[1]))

    raise ValueError("Expected sensitivity interval to be a 2-item tuple/list")


def interval_bounds_or_none(
    value: Any,
) -> Tuple[float | None, float | None]:
    if not metric_has_value(value):
        return None, None
    try:
        normalized = normalize_interval_value(value)
    except Exception:
        return None, None
    if normalized is None:
        return None, None
    return normalized


def build_clean_run_summary_row(summary_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        field: summary_row.get(field)
        for field in CLEAN_RUN_SUMMARY_COLUMNS
    }


def build_control_run_summary_row(summary_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        field: summary_row.get(field)
        for field in CONTROL_RUN_SUMMARY_COLUMNS
    }


def maybe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def make_dml_estimator(model_type: str) -> Any:
    if model_type == "CausalForest":
        return CausalForestDML(
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

    if model_type == "LinearDML":
        return LinearDML(
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
            random_state=SEED,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_compatibility_estimator(
    model_type: str,
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray | None,
    X: np.ndarray | None,
) -> Any:
    est = make_dml_estimator(model_type)
    est.fit(Y=Y, T=T, X=X, W=W, cache_values=True)
    return est


def try_method_calls(
    obj: Any,
    method_name: str,
    candidate_kwargs: Sequence[Dict[str, Any]],
) -> Tuple[Any | None, str | None]:
    if not hasattr(obj, method_name):
        return None, f"Estimator does not expose '{method_name}'"

    method = getattr(obj, method_name)
    if not callable(method):
        return None, f"Estimator attribute '{method_name}' is not callable"

    last_error: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            return method(**kwargs), None
        except Exception as exc:
            last_error = exc

    if last_error is None:
        return None, f"Unable to call '{method_name}'"
    return None, format_exception(last_error)


def extract_estimator_direct_metrics(
    est: Any,
    effect_modifiers: Sequence[str],
    source_label: str,
) -> Dict[str, Any]:
    rv_loaded, rv_loaded_error = try_method_calls(
        est,
        "robustness_value",
        [{}, {"null_hypothesis": 0.0, "alpha": DEFAULT_SENSITIVITY_ALPHA}],
    )
    summary_loaded, summary_loaded_error = try_method_calls(
        est,
        "sensitivity_summary",
        [{}, {"null_hypothesis": 0.0, "alpha": DEFAULT_SENSITIVITY_ALPHA}],
    )
    interval_loaded, interval_loaded_error = try_method_calls(
        est,
        "sensitivity_interval",
        [{}, {"alpha": DEFAULT_SENSITIVITY_ALPHA, "interval_type": "ci"}],
    )

    interval_value = None
    if interval_loaded is not None:
        try:
            interval_value = normalize_interval_value(interval_loaded)
            interval_loaded_error = None
        except Exception as exc:
            interval_loaded_error = format_exception(exc)

    estimator_summary_text = ""
    estimator_summary_error = None
    if hasattr(est, "summary") and callable(getattr(est, "summary", None)):
        try:
            estimator_summary_text = str(
                est.summary(feature_names=list(effect_modifiers) or None)
            )
        except Exception as exc:
            estimator_summary_error = format_exception(exc)
            try:
                estimator_summary_text = str(est.summary())
                estimator_summary_error = None
            except Exception as nested_exc:
                estimator_summary_error = format_exception(nested_exc)
    else:
        estimator_summary_error = "Estimator does not expose 'summary'"

    rv_value = None
    if rv_loaded is not None:
        try:
            rv_value = coerce_float(rv_loaded)
            rv_loaded_error = None
        except Exception as exc:
            rv_loaded_error = format_exception(exc)

    return {
        "rv": rv_value,
        "rv_source": source_label if rv_value is not None else "",
        "rv_error": maybe_string(rv_loaded_error),
        "interval": interval_value,
        "interval_source": source_label if interval_value is not None else "",
        "interval_error": maybe_string(interval_loaded_error),
        "summary_text": str(summary_loaded) if summary_loaded is not None else "",
        "summary_source": source_label if summary_loaded is not None else "",
        "summary_error": maybe_string(summary_loaded_error),
        "estimator_summary_text": estimator_summary_text,
        "estimator_summary_source": source_label if estimator_summary_text else "",
        "estimator_summary_error": maybe_string(estimator_summary_error),
    }


def to_1d_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def to_2d_float_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after coercion; got shape {arr.shape}")
    return arr


def extract_dml_residuals(est: Any) -> Dict[str, Any]:
    try:
        residuals = est.residuals_
    except Exception as exc:
        raise AttributeError(str(exc)) from exc

    if not isinstance(residuals, (tuple, list)):
        raise TypeError("Estimator residuals_ is not a tuple/list")
    if len(residuals) < 4:
        raise ValueError("Estimator residuals_ did not include (y_res, t_res, X, W)")

    # EconML documents that residual rows are not guaranteed to preserve the
    # original input order, so we never merge these arrays back to patient IDs.
    y_res = to_1d_float_array(residuals[0])
    t_res = to_1d_float_array(residuals[1])
    w_res = residuals[3]
    if w_res is None:
        raise ValueError("Estimator residuals_ returned W=None")

    W_res = to_2d_float_array(w_res, "W_residual_order")
    if W_res.shape[0] != y_res.shape[0] or W_res.shape[0] != t_res.shape[0]:
        raise ValueError(
            "Residual arrays have inconsistent row counts: "
            f"y={y_res.shape[0]}, t={t_res.shape[0]}, W={W_res.shape[0]}"
        )

    return {
        "y_res": y_res,
        "t_res": t_res,
        "W_res": W_res,
        "residual_rows": int(W_res.shape[0]),
    }


def sensitivity_interval_from_params(
    params: SensitivityParams,
    alpha: float,
    c_y: float,
    c_t: float,
    rho: float,
    interval_type: str = "ci",
) -> Tuple[float, float]:
    if interval_type not in {"theta", "ci"}:
        raise ValueError(
            "interval_type must be 'theta' or 'ci' for sensitivity_interval"
        )

    if not (0 <= c_y < 1 and 0 <= c_t < 1):
        raise ValueError("c_y and c_t must be between 0 and 1")
    if rho < -1 or rho > 1:
        raise ValueError("rho must be between -1 and 1")

    theta = float(np.asarray(params.theta).reshape(-1)[0])
    sigma = float(np.asarray(params.sigma).reshape(-1)[0])
    nu = float(np.asarray(params.nu).reshape(-1)[0])
    cov = np.asarray(params.cov, dtype=float)

    if sigma < 0 or nu < 0:
        raise ValueError("sigma and nu must be non-negative")

    C = abs(rho) * np.sqrt(c_y) * np.sqrt(c_t / (1 - c_t)) / 2
    ests = np.array([theta, sigma, nu], dtype=float)

    coefs_p = np.array([1, C * np.sqrt(nu / sigma), C * np.sqrt(sigma / nu)], dtype=float)
    coefs_n = np.array([1, -C * np.sqrt(nu / sigma), -C * np.sqrt(sigma / nu)], dtype=float)

    lb = float(ests @ coefs_n)
    ub = float(ests @ coefs_p)

    if interval_type == "ci":
        sigma_p = float(coefs_p @ cov @ coefs_p)
        sigma_n = float(coefs_n @ cov @ coefs_n)
        lb = float(norm.ppf(alpha / 2, loc=lb, scale=np.sqrt(max(sigma_n, 0.0))))
        ub = float(norm.ppf(1 - alpha / 2, loc=ub, scale=np.sqrt(max(sigma_p, 0.0))))

    return lb, ub


def robustness_value_from_params(
    params: SensitivityParams,
    alpha: float,
    null_hypothesis: float = 0.0,
    interval_type: str = "ci",
) -> float:
    r = 0.0
    r_up = 1.0
    r_down = 0.0

    lb, ub = sensitivity_interval_from_params(
        params=params,
        alpha=alpha,
        c_y=0.0,
        c_t=0.0,
        rho=1.0,
        interval_type=interval_type,
    )

    if lb < null_hypothesis < ub:
        return 0.0
    if lb > null_hypothesis:
        target_ind = 0
        multiplier = 1.0
        distance = lb - null_hypothesis
    else:
        target_ind = 1
        multiplier = -1.0
        distance = ub - null_hypothesis

    while abs(distance) > 1e-6 and r_up - r_down > 1e-10:
        interval = sensitivity_interval_from_params(
            params=params,
            alpha=alpha,
            c_y=r,
            c_t=r,
            rho=1.0,
            interval_type=interval_type,
        )
        bound = interval[target_ind]
        distance = multiplier * (bound - null_hypothesis)

        if distance > 0:
            r_down = r
        else:
            r_up = r

        r = (r_down + r_up) / 2

    return float(r)


def dml_sensitivity_values(
    t_res: np.ndarray,
    y_res: np.ndarray,
) -> SensitivityParams:
    t_res = t_res.reshape(-1, 1)
    y_res = y_res.reshape(-1, 1)

    theta = np.mean(y_res * t_res) / np.mean(t_res ** 2)
    sigma2 = np.mean((y_res - theta * t_res) ** 2)
    nu2 = 1 / np.mean(t_res ** 2)

    ls = np.concatenate([t_res ** 2, np.ones_like(t_res), t_res ** 2], axis=1)
    G = np.diag(np.mean(ls, axis=0))
    G_inv = np.linalg.inv(G)

    residuals = np.concatenate([
        y_res * t_res - theta * t_res * t_res,
        (y_res - theta * t_res) ** 2 - sigma2,
        t_res ** 2 * nu2 - 1,
    ], axis=1)
    omega = residuals.T @ residuals / len(residuals)
    cov = G_inv @ omega @ G_inv / len(residuals)

    return SensitivityParams(
        theta=theta,
        sigma=sigma2,
        nu=nu2,
        cov=cov,
    )


def build_sensitivity_summary_text(
    params: SensitivityParams,
    null_hypothesis: float = 0.0,
    alpha: float = DEFAULT_SENSITIVITY_ALPHA,
    c_y: float = DEFAULT_SENSITIVITY_C_Y,
    c_t: float = DEFAULT_SENSITIVITY_C_T,
    rho: float = DEFAULT_SENSITIVITY_RHO,
    source_label: str = "fallback_manual",
) -> str:
    ci_lb, ci_ub = sensitivity_interval_from_params(
        params=params,
        alpha=alpha,
        c_y=c_y,
        c_t=c_t,
        rho=rho,
        interval_type="ci",
    )
    theta_lb, theta_ub = sensitivity_interval_from_params(
        params=params,
        alpha=alpha,
        c_y=c_y,
        c_t=c_t,
        rho=rho,
        interval_type="theta",
    )
    theta = float(np.asarray(params.theta).reshape(-1)[0])
    rv_theta = robustness_value_from_params(
        params=params,
        alpha=alpha,
        null_hypothesis=null_hypothesis,
        interval_type="theta",
    )
    rv_ci = robustness_value_from_params(
        params=params,
        alpha=alpha,
        null_hypothesis=null_hypothesis,
        interval_type="ci",
    )

    header = {
        "saved_training_params_reconstructed": (
            "Sensitivity summary reconstructed from saved training sensitivity parameters"
        ),
        "fallback_manual": (
            "Fallback sensitivity summary from residual-space params"
        ),
    }.get(source_label, f"Sensitivity summary from {source_label}")

    lines = [
        f"{header} (c_y={c_y}, c_t={c_t}, rho={rho}, alpha={alpha})",
        f"CI Lower: {ci_lb:.6f}",
        f"Theta Lower: {theta_lb:.6f}",
        f"Theta: {theta:.6f}",
        f"Theta Upper: {theta_ub:.6f}",
        f"CI Upper: {ci_ub:.6f}",
        f"Robustness Value (theta): {rv_theta:.6f}",
        f"Robustness Value (ci): {rv_ci:.6f}",
        f"Null hypothesis: {null_hypothesis:.6f}",
    ]
    return "\n".join(lines)


def reconstruct_saved_training_sensitivity_params(
    artifact: Dict[str, Any],
) -> Tuple[SensitivityParams | None, str | None]:
    if not artifact.get("saved_sensitivity_params_available", False):
        return None, maybe_string(artifact.get("saved_sensitivity_params_error"))

    serialized = artifact.get("saved_sensitivity_params_serialized")
    if serialized is None:
        return None, "Saved sensitivity params were marked available but no serialized payload was stored"

    try:
        return sensitivity_params_from_serialized(serialized), None
    except Exception as exc:
        return None, format_exception(exc)


def build_params_based_metrics(
    sensitivity_params: SensitivityParams | None,
    source_label: str,
) -> Dict[str, Any]:
    if sensitivity_params is None:
        return {
            "rv": None,
            "rv_source": "",
            "rv_error": "",
            "interval": None,
            "interval_source": "",
            "interval_error": "",
            "summary_text": "",
            "summary_source": "",
            "summary_error": "",
            "estimator_summary_text": "",
            "estimator_summary_source": "",
            "estimator_summary_error": "",
            "rv_theta": None,
            "rv_theta_source": "",
        }

    return {
        "rv": robustness_value_from_params(
            params=sensitivity_params,
            alpha=DEFAULT_SENSITIVITY_ALPHA,
            null_hypothesis=0.0,
            interval_type="ci",
        ),
        "rv_source": source_label,
        "rv_error": "",
        "interval": sensitivity_interval_from_params(
            params=sensitivity_params,
            alpha=DEFAULT_SENSITIVITY_ALPHA,
            c_y=DEFAULT_SENSITIVITY_C_Y,
            c_t=DEFAULT_SENSITIVITY_C_T,
            rho=DEFAULT_SENSITIVITY_RHO,
            interval_type="ci",
        ),
        "interval_source": source_label,
        "interval_error": "",
        "summary_text": build_sensitivity_summary_text(
            sensitivity_params,
            source_label=source_label,
        ),
        "summary_source": source_label,
        "summary_error": "",
        "estimator_summary_text": "",
        "estimator_summary_source": "",
        "estimator_summary_error": "",
        "rv_theta": robustness_value_from_params(
            params=sensitivity_params,
            alpha=DEFAULT_SENSITIVITY_ALPHA,
            null_hypothesis=0.0,
            interval_type="theta",
        ),
        "rv_theta_source": source_label,
    }


def select_authoritative_metric(
    candidates: Sequence[Dict[str, Any]],
    value_key: str,
    source_key: str,
    error_key: str,
) -> Dict[str, Any]:
    first_error = ""
    for candidate in candidates:
        if metric_has_value(candidate.get(value_key)) and candidate.get(source_key):
            return {
                "value": candidate.get(value_key),
                "source": candidate.get(source_key),
                "error": "",
            }
        if not first_error and maybe_string(candidate.get(error_key)):
            first_error = maybe_string(candidate.get(error_key))
    return {"value": None, "source": "", "error": first_error}


def extract_sensitivity_outputs(
    saved_training_direct: Dict[str, Any],
    loaded_direct: Dict[str, Any],
    compatibility_direct: Dict[str, Any],
    saved_training_params_metrics: Dict[str, Any],
    fallback_manual_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = [
        saved_training_direct,
        loaded_direct,
        compatibility_direct,
        saved_training_params_metrics,
        fallback_manual_metrics,
    ]

    rv = select_authoritative_metric(candidates, "rv", "rv_source", "rv_error")
    interval = select_authoritative_metric(
        candidates, "interval", "interval_source", "interval_error"
    )
    summary_text = select_authoritative_metric(
        candidates, "summary_text", "summary_source", "summary_error"
    )
    estimator_summary = select_authoritative_metric(
        [saved_training_direct, loaded_direct, compatibility_direct],
        "estimator_summary_text",
        "estimator_summary_source",
        "estimator_summary_error",
    )
    rv_theta = select_authoritative_metric(
        [saved_training_params_metrics, fallback_manual_metrics],
        "rv_theta",
        "rv_theta_source",
        "rv_error",
    )

    return {
        "rv": rv["value"],
        "rv_source": rv["source"],
        "rv_error": rv["error"],
        "rv_theta": rv_theta["value"],
        "rv_theta_source": rv_theta["source"],
        "summary_text": summary_text["value"] or "",
        "summary_source": summary_text["source"],
        "summary_error": summary_text["error"],
        "interval": interval["value"],
        "interval_source": interval["source"],
        "interval_error": interval["error"],
        "estimator_summary_text": estimator_summary["value"] or "",
        "estimator_summary_source": estimator_summary["source"],
        "estimator_summary_error": estimator_summary["error"],
    }


def single_feature_r_squared(feature: np.ndarray, target: np.ndarray) -> float:
    feature = np.asarray(feature, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)

    mask = np.isfinite(feature) & np.isfinite(target)
    if mask.sum() < 3:
        return 0.0

    feature = feature[mask]
    target = target[mask]
    if np.allclose(feature.std(), 0.0) or np.allclose(target.std(), 0.0):
        return 0.0

    corr = np.corrcoef(feature, target)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr ** 2, 0.0, 1.0))


def compute_single_confounder_strengths(
    y_res: np.ndarray,
    t_res: np.ndarray,
    W: np.ndarray,
    confounder_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if W.shape[1] != len(confounder_names):
        raise ValueError(
            "Residual-space W column count does not match artifact confounders: "
            f"W.shape[1]={W.shape[1]}, n_confounders={len(confounder_names)}"
        )

    rows: List[Dict[str, Any]] = []
    for idx, confounder in enumerate(confounder_names):
        z = W[:, idx]
        proxy_cf_y = single_feature_r_squared(z, y_res)
        proxy_cf_d = single_feature_r_squared(z, t_res)
        rows.append({
            "confounder": confounder,
            "proxy_cf_y": proxy_cf_y,
            "proxy_cf_d": proxy_cf_d,
            "proxy_strength_score": proxy_cf_y * proxy_cf_d,
        })

    rows.sort(
        key=lambda row: (
            -row["proxy_strength_score"],
            -row["proxy_cf_y"],
            -row["proxy_cf_d"],
            row["confounder"],
        )
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def select_benchmark_confounders(
    scores_rows: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    if top_k not in ALLOWED_TOP_K_VALUES:
        raise ValueError(
            f"TOP_K_BENCHMARK_CONFOUNDERS must be one of {sorted(ALLOWED_TOP_K_VALUES)}"
        )

    selected_rows = list(scores_rows[:top_k])
    primary_row = selected_rows[0] if selected_rows else None
    selected_names = {row["confounder"] for row in selected_rows}
    primary_name = primary_row["confounder"] if primary_row else None

    annotated_rows = []
    for row in scores_rows:
        new_row = dict(row)
        new_row["selected_as_candidate"] = row["confounder"] in selected_names
        new_row["is_primary_candidate"] = row["confounder"] == primary_name
        annotated_rows.append(new_row)

    return {
        "all_rows": annotated_rows,
        "selected_rows": selected_rows,
        "primary_row": primary_row,
        "aggregation_rule": (
            "Primary candidate is the shortlisted confounder with the largest "
            "proxy_strength_score among the top-k proxy-screened confounders."
        ),
    }


def compute_proxy_robustness_ratio(
    rv: float | None,
    cf_y: float | None,
    cf_d: float | None,
) -> float | None:
    if rv is None or cf_y is None or cf_d is None:
        return None

    denom = max(cf_y, cf_d)
    if np.isclose(denom, 0.0):
        if np.isclose(rv, 0.0):
            return 0.0
        return float("inf")
    return float(rv / denom)


def compute_real_benchmark_values(
    est: Any,
    artifact_direct: Dict[str, Any],
    shortlisted_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    _ = est
    _ = artifact_direct

    if not shortlisted_rows:
        return {
            "available": False,
            "source": "unavailable_no_proxy_candidates",
            "selected_confounder": "",
            "cf_y": None,
            "cf_d": None,
            "strength_score": None,
            "warnings": ["No shortlisted confounders were available for Stage 2 real benchmark evaluation."],
        }

    return {
        "available": False,
        "source": "unimplemented_by_design",
        "selected_confounder": "",
        "cf_y": None,
        "cf_d": None,
        "strength_score": None,
        "warnings": [
            (
                "Stage 2 real benchmark extraction is currently unimplemented by design in this "
                f"analysis pipeline for econml in {PREFERRED_ENV_NAME}; this is not a loaded-artifact "
                "extraction failure."
            )
        ],
    }


def sensitivity_margin(lb: float | None, ub: float | None) -> float:
    if lb is None or ub is None:
        return float("nan")
    if lb > 0:
        return float(lb)
    if ub < 0:
        return float(-ub)
    return float(-min(ub, -lb))


def save_custom_sensitivity_contour(
    sensitivity_params: SensitivityParams | None,
    treatment_dir: Path,
    treatment: str,
    benchmark_rows: Sequence[Dict[str, Any]],
    source_label: str,
) -> Tuple[str | None, str, List[str]]:
    # In the pinned econml310 env with econml 0.16.0, fitted LinearDML and
    # CausalForestDML expose sensitivity statistics but not sensitivity_plot(),
    # so contour generation here is intentionally custom and parameter-based.
    notes: List[str] = []

    if not SAVE_CONTOUR_PLOT:
        notes.append("Contour plotting disabled by SAVE_CONTOUR_PLOT=False.")
        return None, "custom_not_available", notes
    if sensitivity_params is None:
        notes.append("Custom contour unavailable because sensitivity parameters were unavailable.")
        return None, "custom_not_available", notes

    grid = np.linspace(0.0, 1.0, SENSITIVITY_GRID_STEPS)
    margins = np.full((len(grid), len(grid)), np.nan, dtype=float)

    for row_idx, c_y in enumerate(grid):
        for col_idx, c_t in enumerate(grid):
            try:
                lb, ub = sensitivity_interval_from_params(
                    params=sensitivity_params,
                    alpha=DEFAULT_SENSITIVITY_ALPHA,
                    c_y=float(c_y),
                    c_t=float(c_t),
                    rho=DEFAULT_SENSITIVITY_RHO,
                    interval_type="ci",
                )
                margins[row_idx, col_idx] = sensitivity_margin(lb, ub)
            except Exception:
                continue

    if np.isnan(margins).all():
        notes.append("Custom contour unavailable because the sensitivity grid could not be evaluated.")
        return None, "custom_not_available", notes

    finite = margins[np.isfinite(margins)]
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    levels = np.linspace(vmin, vmax, 15)
    c_t_grid, c_y_grid = np.meshgrid(grid, grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.contourf(c_t_grid, c_y_grid, margins, levels=levels, cmap="coolwarm")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Signed null-exclusion margin")

    if np.nanmin(margins) <= 0 <= np.nanmax(margins):
        ax.contour(
            c_t_grid,
            c_y_grid,
            margins,
            levels=[0.0],
            colors="black",
            linewidths=1.2,
        )

    for idx, row in enumerate(benchmark_rows, start=1):
        marker = "*" if idx == 1 else "o"
        size = 140 if idx == 1 else 70
        color = "gold" if idx == 1 else "black"
        ax.scatter(
            row["proxy_cf_d"],
            row["proxy_cf_y"],
            marker=marker,
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{idx}. {row['confounder']}",
            (row["proxy_cf_d"], row["proxy_cf_y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("c_t benchmark proxy")
    ax.set_ylabel("c_y benchmark proxy")
    ax.set_title(f"{treatment} sensitivity contour")
    fig.tight_layout()

    output_path = build_treatment_output_path(
        treatment_dir=treatment_dir,
        treatment=treatment,
        suffix="sensitivity_contour",
        extension="png",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    notes.append(
        "Contour plot used a custom reconstructed matplotlib grid from sensitivity parameters "
        f"with source '{source_label}'."
    )
    return str(output_path), source_label, notes


def write_benchmark_report(path: Path, report_data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Benchmark-Based CATE Sensitivity Report ===\n\n")
        f.write(f"Run status: {report_data.get('run_status', 'UNKNOWN')}\n")
        f.write(f"Preferred runtime env: {PREFERRED_ENV_NAME}\n")
        f.write(f"Python executable: {sys.executable}\n")
        f.write(f"Treatment: {report_data.get('treatment', '')}\n")
        f.write(f"Model type: {report_data.get('model_type', '')}\n")
        f.write(f"Estimator class: {report_data.get('estimator_class', '')}\n")
        f.write(
            "Model loaded successfully in econml310: "
            f"{report_data.get('model_loaded_in_econml310', False)}\n"
        )
        f.write(f"Model artifact: {report_data.get('model_artifact_path', '')}\n")
        f.write(f"Analysis dataframe source: {report_data.get('analysis_dataframe_paths', '')}\n\n")

        if report_data.get("error"):
            f.write("Failure / primary error:\n")
            f.write(report_data["error"] + "\n\n")

        if report_data.get("traceback"):
            f.write("Traceback:\n")
            f.write(report_data["traceback"] + "\n")

        f.write("Artifact metadata:\n")
        f.write(f"Artifact schema version: {report_data.get('artifact_schema_version', '')}\n")
        f.write(f"Training timestamp: {report_data.get('training_timestamp', '')}\n")
        f.write(f"Training python version: {report_data.get('training_python_version', '')}\n")
        f.write(f"Training platform: {report_data.get('training_platform', '')}\n")
        f.write(f"Training econml version: {report_data.get('training_econml_version', '')}\n")
        f.write(f"Training sklearn version: {report_data.get('training_sklearn_version', '')}\n")
        f.write(f"Training numpy version: {report_data.get('training_numpy_version', '')}\n")
        f.write(f"Training pandas version: {report_data.get('training_pandas_version', '')}\n")
        f.write(f"Training scipy version: {report_data.get('training_scipy_version', '')}\n")
        f.write(f"Outcome column: {report_data.get('outcome_col', '')}\n")
        f.write(f"cache_values_used during training: {report_data.get('cache_values_used', False)}\n")
        f.write(f"Confounders: {report_data.get('confounders', [])}\n")
        f.write(f"Effect modifiers: {report_data.get('effect_modifiers', [])}\n")
        f.write(f"Formula:\n{report_data.get('formula', '')}\n\n")

        f.write("Saved training-time direct diagnostics:\n")
        f.write(f"saved_direct_rv: {report_data.get('saved_direct_rv', '')}\n")
        f.write(
            f"saved_direct_rv_source: {report_data.get('saved_direct_rv_source', '')}\n"
        )
        f.write(
            f"saved_direct_rv_error: {report_data.get('saved_direct_rv_error', '')}\n"
        )
        f.write(
            "saved_direct_sensitivity_interval: "
            f"{report_data.get('saved_direct_sensitivity_interval', '')}\n"
        )
        f.write(
            "saved_direct_sensitivity_interval_source: "
            f"{report_data.get('saved_direct_sensitivity_interval_source', '')}\n"
        )
        f.write(
            "saved_direct_sensitivity_interval_error: "
            f"{report_data.get('saved_direct_sensitivity_interval_error', '')}\n"
        )
        f.write(
            "saved_direct_sensitivity_summary_source: "
            f"{report_data.get('saved_direct_sensitivity_summary_source', '')}\n"
        )
        f.write(
            "saved_direct_sensitivity_summary_error: "
            f"{report_data.get('saved_direct_sensitivity_summary_error', '')}\n"
        )
        f.write(
            "saved_direct_estimator_summary_source: "
            f"{report_data.get('saved_direct_estimator_summary_source', '')}\n"
        )
        f.write(
            "saved_sensitivity_params_available: "
            f"{report_data.get('saved_sensitivity_params_available', False)}\n"
        )
        f.write(
            "saved_sensitivity_params_source: "
            f"{report_data.get('saved_sensitivity_params_source', '')}\n"
        )
        f.write(
            "saved_sensitivity_params_error: "
            f"{report_data.get('saved_sensitivity_params_error', '')}\n\n"
        )
        f.write(
            "saved_training_residuals_available: "
            f"{report_data.get('saved_training_residuals_available', False)}\n"
        )
        f.write(
            "saved_training_residuals_source: "
            f"{report_data.get('saved_training_residuals_source', '')}\n"
        )
        f.write(
            "saved_training_residuals_error: "
            f"{report_data.get('saved_training_residuals_error', '')}\n\n"
        )

        analysis_summary = report_data.get("analysis_summary", {})
        if analysis_summary:
            f.write("Reconstructed analysis subset:\n")
            f.write(f"Rows: {analysis_summary.get('analysis_rows', '')}\n")
            f.write(f"Outcome rate: {analysis_summary.get('outcome_rate', '')}\n")
            f.write(f"Treatment rate: {analysis_summary.get('treatment_rate', '')}\n")
            f.write(
                "Created missing columns then filled from artifact values: "
                f"{analysis_summary.get('created_missing_columns', [])}\n"
            )
            f.write(f"Fill map: {analysis_summary.get('fill_map', {})}\n\n")

        f.write("Direct model extraction:\n")
        f.write(
            f"Direct residuals available on loaded estimator: "
            f"{report_data.get('direct_residuals_available', False)}\n"
        )
        f.write(f"Residual source used by analysis: {report_data.get('residual_source', '')}\n")
        f.write(
            "Direct RV from saved/loaded/compatibility path if available: "
            f"{report_data.get('direct_rv', '')}\n"
        )
        f.write(
            "Direct sensitivity interval from saved/loaded/compatibility path if available: "
            f"{report_data.get('direct_sensitivity_interval', '')}\n"
        )
        direct_summary_text = maybe_string(report_data.get("direct_sensitivity_summary_text"))
        f.write(
            "Direct sensitivity summary from saved/loaded/compatibility path available: "
            f"{bool(direct_summary_text)}\n"
        )
        f.write(f"Authoritative RV: {report_data.get('rv', '')}\n")
        f.write(f"RV source: {report_data.get('rv_source', '')}\n")
        f.write(f"Fallback theta RV: {report_data.get('rv_theta', '')}\n")
        f.write(f"Fallback theta RV source: {report_data.get('rv_theta_source', '')}\n")
        f.write(
            "Authoritative sensitivity interval: "
            f"{report_data.get('sensitivity_interval', '')}\n"
        )
        f.write(
            "Sensitivity interval source: "
            f"{report_data.get('sensitivity_interval_source', '')}\n"
        )
        f.write(
            "Authoritative sensitivity summary source: "
            f"{report_data.get('sensitivity_summary_source', '')}\n"
        )
        f.write(
            "Direct estimator summary source: "
            f"{report_data.get('estimator_summary_source', '')}\n\n"
        )

        if report_data.get("sensitivity_summary_text"):
            f.write("Sensitivity summary used in report:\n")
            f.write(report_data["sensitivity_summary_text"] + "\n\n")

        if report_data.get("estimator_summary_text"):
            f.write("Direct estimator summary:\n")
            f.write(report_data["estimator_summary_text"] + "\n\n")

        f.write("Residual handling note:\n")
        f.write(
            "EconML documents that residual rows are not guaranteed to be in the "
            "original input order, so this script ranks confounders directly in "
            "residual-array space and does not merge residual rows back to ts_id.\n\n"
        )

        if report_data.get("residual_rows") is not None:
            f.write(f"Residual rows available: {report_data.get('residual_rows')}\n\n")

        scores_rows = list(report_data.get("scores_rows", []))
        if scores_rows:
            f.write("Stage 1 proxy confounder screening:\n")
            f.write("rank | confounder | proxy_cf_y | proxy_cf_d | proxy_strength_score\n")
            for row in scores_rows:
                f.write(
                    f"{row['rank']} | {row['confounder']} | "
                    f"{row['proxy_cf_y']:.6f} | {row['proxy_cf_d']:.6f} | "
                    f"{row['proxy_strength_score']:.6f}\n"
                )
            f.write("\n")
        else:
            f.write("Stage 1 proxy confounder screening:\nNone\n\n")

        selected_rows = list(report_data.get("selected_rows", []))
        if selected_rows:
            f.write("Selected candidate confounders for Stage 2 real benchmark:\n")
            for row in selected_rows:
                f.write(
                    f"- {row['confounder']} "
                    f"(proxy_cf_y={row['proxy_cf_y']:.6f}, "
                    f"proxy_cf_d={row['proxy_cf_d']:.6f}, "
                    f"proxy_strength_score={row['proxy_strength_score']:.6f})\n"
                )
            f.write(
                "\nAggregation rule for proxy screening:\n"
                f"{report_data.get('aggregation_rule', '')}\n\n"
            )
        else:
            f.write("Selected candidate confounders for Stage 2 real benchmark:\nNone\n\n")

        real_benchmark = dict(report_data.get("real_benchmark", {}))
        f.write("Stage 2 real benchmark results:\n")
        f.write(f"Available: {real_benchmark.get('available', False)}\n")
        f.write(f"Source: {real_benchmark.get('source', '')}\n")
        f.write(f"Selected benchmark confounder: {real_benchmark.get('selected_confounder', '')}\n")
        f.write(f"benchmark_cf_y: {real_benchmark.get('cf_y', '')}\n")
        f.write(f"benchmark_cf_d: {real_benchmark.get('cf_d', '')}\n")
        f.write(f"benchmark_strength_score: {real_benchmark.get('strength_score', '')}\n\n")

        f.write("Ratios:\n")
        f.write(
            "Proxy robustness ratio (RV / max(proxy_cf_y, proxy_cf_d)): "
            f"{report_data.get('proxy_robustness_ratio', '')}\n"
        )
        f.write(
            "Real benchmark robustness ratio: "
            f"{report_data.get('robustness_ratio', '')}\n\n"
        )

        f.write("Contour output:\n")
        f.write(f"Contour source: {report_data.get('contour_source', '')}\n")
        f.write(f"Contour plot path: {report_data.get('contour_plot_path', '')}\n\n")

        direct_artifact = dict(report_data.get("direct_artifact_diagnostics", {}))
        if direct_artifact:
            f.write("Saved direct artifact diagnostics:\n")
            for key in sorted(direct_artifact):
                f.write(f"- {key}: {direct_artifact[key]}\n")
            f.write("\n")

        fallbacks_used = list(report_data.get("fallbacks_used", []))
        if fallbacks_used:
            f.write("Fallbacks used:\n")
            for item in fallbacks_used:
                f.write(f"- {item}\n")
            f.write("\n")

        warnings_list = list(report_data.get("warnings", []))
        if warnings_list:
            f.write("Warnings / unavailable APIs:\n")
            for item in warnings_list:
                f.write(f"- {item}\n")
            f.write("\n")

        training_summary = dict(report_data.get("training_summary", {}))
        if training_summary:
            f.write("Saved training summary from artifact:\n")
            for key in sorted(training_summary):
                f.write(f"- {key}: {training_summary[key]}\n")


def empty_summary_row(treatment: str, report_path: Path, scores_path: Path) -> Dict[str, Any]:
    return {
        "row_id": build_summary_row_id("", treatment),
        "treatment": treatment,
        "model_type": "",
        "estimator_class": "",
        "model_loaded_in_econml310": False,
        "cache_values_used": False,
        "saved_training_residuals_available": False,
        "saved_training_residuals_source": "",
        "RV": None,
        "rv_source": "",
        "sensitivity_interval_source": "",
        "sensitivity_summary_source": "",
        "estimator_summary_source": "",
        "direct_rv": None,
        "sensitivity_interval": None,
        "sensitivity_interval_lb": None,
        "sensitivity_interval_ub": None,
        "direct_sensitivity_interval": None,
        "direct_sensitivity_interval_lb": None,
        "direct_sensitivity_interval_ub": None,
        "direct_sensitivity_summary_available": False,
        "residual_source": "",
        "proxy_primary_candidate": "",
        "proxy_cf_y": None,
        "proxy_cf_d": None,
        "proxy_strength_score": None,
        "proxy_robustness_ratio": None,
        "selected_benchmark_confounder": "",
        "real_benchmark_available": False,
        "real_benchmark_cf_y": None,
        "real_benchmark_cf_d": None,
        "real_benchmark_strength_score": None,
        "real_benchmark_source": "",
        "robustness_ratio": None,
        "contour_source": "",
        "contour_plot_path": "",
        "run_status": "FAILED",
        "report_path": str(report_path),
        "benchmark_scores_path": str(scores_path),
        "analysis_rows": None,
        "residual_rows": None,
        "warnings": "",
    }


def analyze_one_treatment(
    artifact_treatment_dir: Path,
    output_treatment_dir: Path,
    analysis_df: pd.DataFrame | None,
    analysis_df_error: str | None,
    latent_tags_path: Path,
    physionet_pkl_path: Path,
) -> Dict[str, Any]:
    output_treatment_dir.mkdir(parents=True, exist_ok=True)

    treatment_hint = artifact_treatment_dir.name
    print(f"      [{treatment_hint}] Loading saved model artifact and diagnostics")
    model_path = artifact_treatment_dir / f"{treatment_hint}_model.pkl"
    report_path = build_treatment_output_path(
        output_treatment_dir,
        treatment_hint,
        "benchmark_report",
        "txt",
    )
    scores_path = build_treatment_output_path(
        output_treatment_dir,
        treatment_hint,
        "benchmark_scores",
        "csv",
    )

    summary_row = empty_summary_row(treatment_hint, report_path, scores_path)
    warnings_list: List[str] = []
    fallbacks_used: List[str] = []
    report_data: Dict[str, Any] = {
        "run_status": "FAILED",
        "treatment": treatment_hint,
        "model_type": "",
        "estimator_class": "",
        "model_loaded_in_econml310": False,
        "cache_values_used": False,
        "model_artifact_path": str(model_path),
        "analysis_dataframe_paths": (
            f"latent_tags={latent_tags_path}; processed_pkl={physionet_pkl_path}"
        ),
        "artifact_schema_version": "",
        "training_timestamp": "",
        "training_python_version": "",
        "training_platform": "",
        "training_econml_version": "",
        "training_sklearn_version": "",
        "training_numpy_version": "",
        "training_pandas_version": "",
        "training_scipy_version": "",
        "outcome_col": "",
        "confounders": [],
        "effect_modifiers": [],
        "formula": "",
        "analysis_summary": {},
        "saved_direct_rv": None,
        "saved_direct_rv_source": "",
        "saved_direct_rv_error": "",
        "saved_direct_sensitivity_interval": None,
        "saved_direct_sensitivity_interval_source": "",
        "saved_direct_sensitivity_interval_error": "",
        "saved_direct_sensitivity_summary_source": "",
        "saved_direct_sensitivity_summary_error": "",
        "saved_direct_estimator_summary_source": "",
        "saved_sensitivity_params_available": False,
        "saved_sensitivity_params_source": "",
        "saved_sensitivity_params_error": "",
        "saved_training_residuals_available": False,
        "saved_training_residuals_source": "",
        "saved_training_residuals_error": "",
        "rv": None,
        "rv_source": "",
        "rv_theta": None,
        "rv_theta_source": "",
        "direct_rv": None,
        "direct_sensitivity_interval": None,
        "direct_sensitivity_summary_text": "",
        "sensitivity_interval": None,
        "sensitivity_interval_source": "",
        "sensitivity_summary_text": "",
        "sensitivity_summary_source": "",
        "estimator_summary_text": "",
        "estimator_summary_source": "",
        "direct_residuals_available": False,
        "residual_source": "",
        "proxy_robustness_ratio": None,
        "robustness_ratio": None,
        "real_benchmark": {},
        "contour_source": "",
        "contour_plot_path": "",
        "scores_rows": [],
        "selected_rows": [],
        "aggregation_rule": "",
        "fallbacks_used": fallbacks_used,
        "warnings": warnings_list,
        "training_summary": {},
        "direct_artifact_diagnostics": {},
        "residual_rows": None,
        "error": "",
        "traceback": "",
    }

    scores_rows: List[Dict[str, Any]] = []

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model artifact not found: {model_path}")

        raw_artifact, artifact_warnings = load_model_artifact(model_path)
        warnings_list.extend(artifact_warnings)
        artifact = validate_artifact(raw_artifact)

        treatment = artifact["treatment"]
        if treatment != treatment_hint:
            warnings_list.append(
                f"Artifact treatment '{treatment}' does not match directory '{treatment_hint}'."
            )

        report_data.update({
            "treatment": treatment,
            "model_type": artifact["model_type"],
            "estimator_class": artifact["estimator_class"],
            "model_loaded_in_econml310": PREFERRED_ENV_NAME in sys.executable,
            "cache_values_used": artifact["cache_values_used"],
            "artifact_schema_version": artifact["artifact_schema_version"],
            "training_timestamp": artifact["training_timestamp"],
            "training_python_version": artifact["python_version"],
            "training_platform": artifact["platform"],
            "training_econml_version": artifact["econml_version"],
            "training_sklearn_version": artifact["sklearn_version"],
            "training_numpy_version": artifact["numpy_version"],
            "training_pandas_version": artifact["pandas_version"],
            "training_scipy_version": artifact["scipy_version"],
            "outcome_col": artifact["outcome_col"],
            "confounders": artifact["confounders"],
            "effect_modifiers": artifact["effect_modifiers"],
            "formula": artifact["formula"],
            "saved_direct_rv": artifact["saved_direct_rv"],
            "saved_direct_rv_source": artifact["saved_direct_rv_source"],
            "saved_direct_rv_error": artifact["saved_direct_rv_error"],
            "saved_direct_sensitivity_interval": artifact[
                "saved_direct_sensitivity_interval"
            ],
            "saved_direct_sensitivity_interval_source": artifact[
                "saved_direct_sensitivity_interval_source"
            ],
            "saved_direct_sensitivity_interval_error": artifact[
                "saved_direct_sensitivity_interval_error"
            ],
            "saved_direct_sensitivity_summary_source": artifact[
                "saved_direct_sensitivity_summary_source"
            ],
            "saved_direct_sensitivity_summary_error": artifact[
                "saved_direct_sensitivity_summary_error"
            ],
            "saved_direct_estimator_summary_source": artifact[
                "saved_direct_estimator_summary_source"
            ],
            "saved_sensitivity_params_available": artifact[
                "saved_sensitivity_params_available"
            ],
            "saved_sensitivity_params_source": artifact[
                "saved_sensitivity_params_source"
            ],
            "saved_sensitivity_params_error": artifact[
                "saved_sensitivity_params_error"
            ],
            "saved_training_residuals_available": artifact[
                "saved_training_residuals_available"
            ],
            "saved_training_residuals_source": artifact[
                "saved_training_residuals_source"
            ],
            "saved_training_residuals_error": artifact[
                "saved_training_residuals_error"
            ],
            "training_summary": artifact["summary"],
            "direct_artifact_diagnostics": artifact["direct_diagnostics"],
        })

        prepared = None
        if analysis_df is None:
            warnings_list.append(
                "Analysis dataframe was unavailable, so treatment matrices could not be reconstructed: "
                f"{analysis_df_error}"
            )
        else:
            print(f"      [{treatment}] Reconstructing treatment matrices from saved artifact")
            prepared = prepare_treatment_matrices_from_artifact(analysis_df, artifact)
            report_data["analysis_summary"] = prepared
            summary_row["analysis_rows"] = prepared["analysis_rows"]

        loaded_estimator = artifact["estimator"]
        saved_training_direct = extract_saved_training_direct_metrics(artifact)
        loaded_direct = extract_estimator_direct_metrics(
            est=loaded_estimator,
            effect_modifiers=artifact.get(
                "effect_modifiers_order", artifact["effect_modifiers"]
            ),
            source_label="loaded_estimator_direct",
        )

        residual_source = ""
        residual_info = None
        try:
            residual_info = extract_dml_residuals(loaded_estimator)
            residual_source = "loaded_estimator_direct"
            report_data["direct_residuals_available"] = True
        except Exception as exc:
            warnings_list.append(
                f"Loaded estimator residuals_ unavailable: {format_exception(exc)}"
            )
            report_data["direct_residuals_available"] = False

        need_compatibility_refit = prepared is not None and (
            residual_info is None
            or (
                not metric_selected_from_source(
                    saved_training_direct,
                    "rv",
                    "rv_source",
                    ["saved_training_direct"],
                )
                and not metric_selected_from_source(
                    loaded_direct,
                    "rv",
                    "rv_source",
                    ["loaded_estimator_direct"],
                )
            )
            or (
                not metric_selected_from_source(
                    saved_training_direct,
                    "interval",
                    "interval_source",
                    ["saved_training_direct"],
                )
                and not metric_selected_from_source(
                    loaded_direct,
                    "interval",
                    "interval_source",
                    ["loaded_estimator_direct"],
                )
            )
            or (
                not metric_selected_from_source(
                    saved_training_direct,
                    "summary_text",
                    "summary_source",
                    ["saved_training_direct"],
                )
                and not metric_selected_from_source(
                    loaded_direct,
                    "summary_text",
                    "summary_source",
                    ["loaded_estimator_direct"],
                )
            )
        )

        compatibility_estimator = None
        compatibility_direct = empty_sensitivity_metrics()
        if need_compatibility_refit:
            print(f"      [{treatment}] Starting compatibility refit for missing direct sensitivity APIs")
            compatibility_note = (
                "Used compatibility refit with cache_values=True to test direct sensitivity APIs in "
                f"{PREFERRED_ENV_NAME} and recover residual-space diagnostics when needed."
            )
            warnings_list.append(compatibility_note)
            compatibility_estimator = fit_compatibility_estimator(
                model_type=artifact["model_type"],
                Y=prepared["Y"],
                T=prepared["T"],
                W=prepared["W"],
                X=prepared["X"],
            )
            compatibility_direct = extract_estimator_direct_metrics(
                est=compatibility_estimator,
                effect_modifiers=artifact.get(
                    "effect_modifiers_order", artifact["effect_modifiers"]
                ),
                source_label="compatibility_refit_direct",
            )
            if residual_info is None:
                residual_info = extract_dml_residuals(compatibility_estimator)
                residual_source = "compatibility_refit_direct"
                report_data["direct_residuals_available"] = True

        report_data["residual_source"] = residual_source
        summary_row["residual_source"] = residual_source

        sensitivity_params = None
        if residual_info is not None:
            sensitivity_params = dml_sensitivity_values(
                t_res=residual_info["t_res"],
                y_res=residual_info["y_res"],
            )
            report_data["residual_rows"] = residual_info["residual_rows"]
            summary_row["residual_rows"] = residual_info["residual_rows"]

            scores_rows = compute_single_confounder_strengths(
                y_res=residual_info["y_res"],
                t_res=residual_info["t_res"],
                W=residual_info["W_res"],
                confounder_names=artifact.get("confounders_order", artifact["confounders"]),
            )

            benchmark_selection = select_benchmark_confounders(
                scores_rows=scores_rows,
                top_k=TOP_K_BENCHMARK_CONFOUNDERS,
            )
            scores_rows = list(benchmark_selection["all_rows"])
            selected_rows = list(benchmark_selection["selected_rows"])
            primary_row = benchmark_selection["primary_row"]

            report_data["scores_rows"] = scores_rows
            report_data["selected_rows"] = selected_rows
            report_data["aggregation_rule"] = benchmark_selection["aggregation_rule"]
        else:
            selected_rows = []
            primary_row = None

        saved_training_params, saved_training_params_error = (
            reconstruct_saved_training_sensitivity_params(artifact)
        )
        if saved_training_params_error and artifact["saved_sensitivity_params_available"]:
            warnings_list.append(
                "Saved training sensitivity params could not be reconstructed: "
                f"{saved_training_params_error}"
            )

        saved_training_params_metrics = build_params_based_metrics(
            saved_training_params,
            "saved_training_params_reconstructed",
        )
        fallback_manual_metrics = build_params_based_metrics(
            sensitivity_params,
            "fallback_manual",
        )

        sensitivity_outputs = extract_sensitivity_outputs(
            saved_training_direct=saved_training_direct,
            loaded_direct=loaded_direct,
            compatibility_direct=compatibility_direct,
            saved_training_params_metrics=saved_training_params_metrics,
            fallback_manual_metrics=fallback_manual_metrics,
        )
        report_data["rv"] = sensitivity_outputs["rv"]
        report_data["rv_source"] = sensitivity_outputs["rv_source"]
        report_data["rv_theta"] = sensitivity_outputs["rv_theta"]
        report_data["rv_theta_source"] = sensitivity_outputs["rv_theta_source"]
        report_data["direct_rv"] = (
            sensitivity_outputs["rv"]
            if sensitivity_outputs["rv_source"] in {
                "saved_training_direct",
                "loaded_estimator_direct",
                "compatibility_refit_direct",
            }
            else None
        )
        report_data["direct_sensitivity_interval"] = (
            sensitivity_outputs["interval"]
            if sensitivity_outputs["interval_source"] in {
                "saved_training_direct",
                "loaded_estimator_direct",
                "compatibility_refit_direct",
            }
            else None
        )
        report_data["direct_sensitivity_summary_text"] = (
            sensitivity_outputs["summary_text"]
            if sensitivity_outputs["summary_source"] in {
                "saved_training_direct",
                "loaded_estimator_direct",
                "compatibility_refit_direct",
            }
            else ""
        )
        report_data["sensitivity_interval"] = sensitivity_outputs["interval"]
        report_data["sensitivity_interval_source"] = sensitivity_outputs["interval_source"]
        report_data["sensitivity_summary_text"] = sensitivity_outputs["summary_text"]
        report_data["sensitivity_summary_source"] = sensitivity_outputs["summary_source"]
        report_data["estimator_summary_text"] = sensitivity_outputs["estimator_summary_text"]
        report_data["estimator_summary_source"] = sensitivity_outputs["estimator_summary_source"]

        if loaded_direct["rv_error"]:
            warnings_list.append(
                f"robustness_value() unavailable on loaded estimator: {loaded_direct['rv_error']}"
            )
        if loaded_direct["summary_error"]:
            warnings_list.append(
                f"sensitivity_summary() unavailable on loaded estimator: {loaded_direct['summary_error']}"
            )
        if loaded_direct["interval_error"]:
            warnings_list.append(
                "sensitivity_interval() unavailable on loaded estimator: "
                f"{loaded_direct['interval_error']}"
            )
        if compatibility_estimator is not None and compatibility_direct["rv_error"]:
            warnings_list.append(
                "robustness_value() unavailable on compatibility refit estimator: "
                f"{compatibility_direct['rv_error']}"
            )
        if compatibility_estimator is not None and compatibility_direct["summary_error"]:
            warnings_list.append(
                "sensitivity_summary() unavailable on compatibility refit estimator: "
                f"{compatibility_direct['summary_error']}"
            )
        if compatibility_estimator is not None and compatibility_direct["interval_error"]:
            warnings_list.append(
                "sensitivity_interval() unavailable on compatibility refit estimator: "
                f"{compatibility_direct['interval_error']}"
            )
        if sensitivity_outputs["rv_source"] == "saved_training_params_reconstructed":
            fallbacks_used.append(
                "Used saved_training_params_reconstructed RV from serialized training-time sensitivity params."
            )
        elif sensitivity_outputs["rv_source"] == "fallback_manual":
            fallbacks_used.append(
                "Used fallback_manual RV from residual-space sensitivity parameters."
            )
        if sensitivity_outputs["summary_source"] == "saved_training_params_reconstructed":
            fallbacks_used.append(
                "Used saved_training_params_reconstructed sensitivity summary from serialized training-time sensitivity params."
            )
        elif sensitivity_outputs["summary_source"] == "fallback_manual":
            fallbacks_used.append(
                "Used fallback_manual sensitivity summary from residual-space sensitivity parameters."
            )
        if sensitivity_outputs["interval_source"] == "saved_training_params_reconstructed":
            fallbacks_used.append(
                "Used saved_training_params_reconstructed sensitivity interval from serialized training-time sensitivity params."
            )
        elif sensitivity_outputs["interval_source"] == "fallback_manual":
            fallbacks_used.append(
                "Used fallback_manual sensitivity interval from residual-space sensitivity parameters."
            )

        real_benchmark = compute_real_benchmark_values(
            est=loaded_estimator,
            artifact_direct=artifact["direct_diagnostics"],
            shortlisted_rows=selected_rows,
        )
        warnings_list.extend(real_benchmark["warnings"])
        report_data["real_benchmark"] = real_benchmark

        contour_params = None
        contour_params_source = "custom_not_available"
        if saved_training_params is not None:
            contour_params = saved_training_params
            contour_params_source = "custom_reconstructed_from_saved_params"
        elif sensitivity_params is not None:
            contour_params = sensitivity_params
            contour_params_source = "custom_reconstructed_from_residual_params"
        contour_plot_path, contour_source, contour_notes = save_custom_sensitivity_contour(
            sensitivity_params=contour_params,
            treatment_dir=output_treatment_dir,
            treatment=treatment,
            benchmark_rows=selected_rows,
            source_label=contour_params_source,
        )
        warnings_list.extend(contour_notes)
        report_data["contour_source"] = contour_source
        report_data["contour_plot_path"] = contour_plot_path or ""
        summary_row["contour_source"] = contour_source
        summary_row["contour_plot_path"] = contour_plot_path or ""

        proxy_cf_y = primary_row["proxy_cf_y"] if primary_row else None
        proxy_cf_d = primary_row["proxy_cf_d"] if primary_row else None
        proxy_strength = primary_row["proxy_strength_score"] if primary_row else None

        proxy_robustness_ratio = compute_proxy_robustness_ratio(
            rv=sensitivity_outputs["rv"],
            cf_y=proxy_cf_y,
            cf_d=proxy_cf_d,
        )
        report_data["proxy_robustness_ratio"] = proxy_robustness_ratio

        robustness_ratio = None
        if real_benchmark["available"]:
            robustness_ratio = compute_proxy_robustness_ratio(
                rv=sensitivity_outputs["rv"],
                cf_y=real_benchmark["cf_y"],
                cf_d=real_benchmark["cf_d"],
            )
        report_data["robustness_ratio"] = robustness_ratio

        run_status = "SUCCESS"
        if not real_benchmark["available"] or primary_row is None or sensitivity_outputs["rv"] is None:
            run_status = "PARTIAL"
        if residual_info is None and sensitivity_outputs["rv"] is None:
            run_status = "FAILED"

        report_data["run_status"] = run_status
        sensitivity_interval_lb, sensitivity_interval_ub = interval_bounds_or_none(
            report_data["sensitivity_interval"]
        )
        direct_interval_lb, direct_interval_ub = interval_bounds_or_none(
            report_data["direct_sensitivity_interval"]
        )
        summary_row.update({
            "row_id": build_summary_row_id(artifact["model_type"], treatment),
            "treatment": treatment,
            "model_type": artifact["model_type"],
            "estimator_class": artifact["estimator_class"],
            "model_loaded_in_econml310": PREFERRED_ENV_NAME in sys.executable,
            "cache_values_used": artifact["cache_values_used"],
            "saved_training_residuals_available": artifact[
                "saved_training_residuals_available"
            ],
            "saved_training_residuals_source": artifact[
                "saved_training_residuals_source"
            ],
            "RV": sensitivity_outputs["rv"],
            "rv_source": sensitivity_outputs["rv_source"],
            "sensitivity_interval": report_data["sensitivity_interval"],
            "sensitivity_interval_lb": sensitivity_interval_lb,
            "sensitivity_interval_ub": sensitivity_interval_ub,
            "sensitivity_interval_source": sensitivity_outputs["interval_source"],
            "sensitivity_summary_source": sensitivity_outputs["summary_source"],
            "estimator_summary_source": sensitivity_outputs["estimator_summary_source"],
            "direct_rv": report_data["direct_rv"],
            "direct_sensitivity_interval": report_data["direct_sensitivity_interval"],
            "direct_sensitivity_interval_lb": direct_interval_lb,
            "direct_sensitivity_interval_ub": direct_interval_ub,
            "direct_sensitivity_summary_available": bool(report_data["direct_sensitivity_summary_text"]),
            "proxy_primary_candidate": primary_row["confounder"] if primary_row else "",
            "proxy_cf_y": proxy_cf_y,
            "proxy_cf_d": proxy_cf_d,
            "proxy_strength_score": proxy_strength,
            "proxy_robustness_ratio": proxy_robustness_ratio,
            "selected_benchmark_confounder": real_benchmark["selected_confounder"],
            "real_benchmark_available": real_benchmark["available"],
            "real_benchmark_cf_y": real_benchmark["cf_y"],
            "real_benchmark_cf_d": real_benchmark["cf_d"],
            "real_benchmark_strength_score": real_benchmark["strength_score"],
            "real_benchmark_source": real_benchmark["source"],
            "robustness_ratio": robustness_ratio,
            "run_status": run_status,
            "warnings": " | ".join(dedupe_preserve_order(warnings_list + fallbacks_used)),
        })

    except Exception as exc:
        report_data["error"] = format_exception(exc)
        report_data["traceback"] = traceback.format_exc()
        warnings_list.append(report_data["error"])
        summary_row["warnings"] = " | ".join(warnings_list)

    finally:
        report_data["warnings"] = dedupe_preserve_order(warnings_list)
        report_data["fallbacks_used"] = dedupe_preserve_order(fallbacks_used)
        report_data["scores_rows"] = scores_rows
        write_rows_to_csv(scores_path, scores_rows, BENCHMARK_SCORE_COLUMNS)
        write_benchmark_report(report_path, report_data)
        summary_row["report_path"] = str(report_path)
        summary_row["benchmark_scores_path"] = str(scores_path)
        print(
            f"      [{treatment_hint}] Finished with status={summary_row['run_status']} | "
            f"report={report_path}"
        )

    return summary_row


def main() -> None:
    global DATASET_MODEL
    global LATENT_TAGS_PATH
    global PHYSIONET_PKL_PATH
    global CATE_RESULTS_DIR
    global PREFERRED_ENV_NAME
    global TOP_K_BENCHMARK_CONFOUNDERS
    global SEED
    global SAVE_CONTOUR_PLOT
    global OUTCOME_COL
    global DEFAULT_SENSITIVITY_ALPHA
    global DEFAULT_SENSITIVITY_C_Y
    global DEFAULT_SENSITIVITY_C_T
    global DEFAULT_SENSITIVITY_RHO
    global SENSITIVITY_GRID_STEPS
    global ALLOWED_TOP_K_VALUES
    global BACKGROUND_FEATURE_COLUMNS
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    DATASET_MODEL = args.model
    config = load_dataset_config(DATASET_MODEL, args.dataset_config_csv)

    PREFERRED_ENV_NAME = str(
        get_config_scalar(config, "PREFERRED_ENV_NAME", PREFERRED_ENV_NAME)
    )
    TOP_K_BENCHMARK_CONFOUNDERS = int(
        get_config_int(
            config,
            "TOP_K_BENCHMARK_CONFOUNDERS",
            TOP_K_BENCHMARK_CONFOUNDERS,
        )
        or TOP_K_BENCHMARK_CONFOUNDERS
    )
    SEED = int(get_config_int(config, "SEED", SEED) or SEED)
    SAVE_CONTOUR_PLOT = bool(
        get_config_bool(config, "SAVE_CONTOUR_PLOT", SAVE_CONTOUR_PLOT)
    )
    OUTCOME_COL = str(get_config_scalar(config, "OUTCOME_COL", OUTCOME_COL))
    DEFAULT_SENSITIVITY_ALPHA = float(
        get_config_float(config, "DEFAULT_SENSITIVITY_ALPHA", DEFAULT_SENSITIVITY_ALPHA)
        or DEFAULT_SENSITIVITY_ALPHA
    )
    DEFAULT_SENSITIVITY_C_Y = float(
        get_config_float(config, "DEFAULT_SENSITIVITY_C_Y", DEFAULT_SENSITIVITY_C_Y)
        or DEFAULT_SENSITIVITY_C_Y
    )
    DEFAULT_SENSITIVITY_C_T = float(
        get_config_float(config, "DEFAULT_SENSITIVITY_C_T", DEFAULT_SENSITIVITY_C_T)
        or DEFAULT_SENSITIVITY_C_T
    )
    DEFAULT_SENSITIVITY_RHO = float(
        get_config_float(config, "DEFAULT_SENSITIVITY_RHO", DEFAULT_SENSITIVITY_RHO)
        or DEFAULT_SENSITIVITY_RHO
    )
    SENSITIVITY_GRID_STEPS = int(
        get_config_int(config, "SENSITIVITY_GRID_STEPS", SENSITIVITY_GRID_STEPS)
        or SENSITIVITY_GRID_STEPS
    )
    ALLOWED_TOP_K_VALUES = set(
        int(value)
        for value in (get_config_list(config, "ALLOWED_TOP_K_VALUES", list(ALLOWED_TOP_K_VALUES)) or [])
    )
    BACKGROUND_FEATURE_COLUMNS = list(
        get_config_list(config, "BACKGROUND_FEATURE_COLUMNS", BACKGROUND_FEATURE_COLUMNS) or []
    )
    np.random.seed(SEED)

    latent_tags_default = get_first_available(
        config,
        ["ANALYZE_LATENT_TAGS_PATH", "LATENT_TAGS_PATH"],
        LATENT_TAGS_PATH,
    )
    physionet_pkl_default = get_first_available(
        config,
        ["ANALYZE_PKL_PATH", "DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
        PHYSIONET_PKL_PATH,
    )
    results_dir_default = get_first_available(
        config,
        ["CATE_RESULTS_DIR"],
        CATE_RESULTS_DIR,
    )
    output_dir_default = get_first_available(
        config,
        ["ANALYZE_OUTPUT_DIR", "OUTPUT_DIR", "CATE_RESULTS_DIR"],
        None,
    )

    if TOP_K_BENCHMARK_CONFOUNDERS not in ALLOWED_TOP_K_VALUES:
        raise ValueError(
            "TOP_K_BENCHMARK_CONFOUNDERS must be one of "
            f"{sorted(ALLOWED_TOP_K_VALUES)}. Found: {TOP_K_BENCHMARK_CONFOUNDERS}"
        )

    results_dir = resolve_script_path(
        args.results_dir if args.results_dir is not None else results_dir_default
    )
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory is not a directory: {results_dir}")

    output_dir = results_dir
    if args.output_dir is not None:
        output_dir = resolve_script_path(args.output_dir)
    elif output_dir_default is not None:
        output_dir = resolve_script_path(output_dir_default)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output directory is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_tags_path = resolve_script_path(
        args.latent_tags_path
        if args.latent_tags_path is not None
        else latent_tags_default
    )
    physionet_pkl_path = resolve_script_path(
        args.physionet_pkl_path
        if args.physionet_pkl_path is not None
        else physionet_pkl_default
    )
    print("=== Starting saved CATE analysis ===")
    print(
        "Runtime configuration: "
        f"model={DATASET_MODEL} | latent_tags_path={latent_tags_path} | "
        f"processed_pkl_path={physionet_pkl_path} | results_dir={results_dir} | "
        f"output_dir={output_dir}"
    )

    treatment_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    run_summary_csv = output_dir / "benchmark_summary.csv"
    control_messages_csv = output_dir / "control_messages_analyze_cate_results.csv"
    print(f"[1/3] Found {len(treatment_dirs)} treatment result directories to analyze")

    analysis_df = None
    analysis_df_error = None
    try:
        print("[2/3] Loading analysis dataframe for artifact reconstruction")
        analysis_df = load_analysis_dataframe(latent_tags_path, physionet_pkl_path)
        print(f"      Analysis dataframe ready: {analysis_df.shape}")
    except Exception as exc:
        analysis_df_error = format_exception(exc)
        print(f"      Analysis dataframe unavailable: {analysis_df_error}")

    summary_rows: List[Dict[str, Any]] = []
    success_count = 0
    partial_count = 0
    failed_count = 0

    print("[3/3] Starting per-treatment analysis loop")
    for treatment_index, treatment_dir in enumerate(treatment_dirs, start=1):
        print(
            f"\n=== Analyze treatment {treatment_index}/{len(treatment_dirs)}: "
            f"{treatment_dir.name} ==="
        )
        output_treatment_dir = output_dir / treatment_dir.name
        summary_row = analyze_one_treatment(
            artifact_treatment_dir=treatment_dir,
            output_treatment_dir=output_treatment_dir,
            analysis_df=analysis_df,
            analysis_df_error=analysis_df_error,
            latent_tags_path=latent_tags_path,
            physionet_pkl_path=physionet_pkl_path,
        )
        summary_rows.append(summary_row)

        if summary_row["run_status"] == "SUCCESS":
            success_count += 1
        elif summary_row["run_status"] == "PARTIAL":
            partial_count += 1
        else:
            failed_count += 1

    summary_rows.sort(key=lambda row: row["treatment"])
    clean_summary_rows = finalize_rows(
        [build_clean_run_summary_row(row) for row in summary_rows],
        CLEAN_RUN_SUMMARY_COLUMNS,
    )
    control_summary_rows = finalize_rows(
        [build_control_run_summary_row(row) for row in summary_rows],
        CONTROL_RUN_SUMMARY_COLUMNS,
    )
    write_rows_to_csv(run_summary_csv, clean_summary_rows, CLEAN_RUN_SUMMARY_COLUMNS)
    write_rows_to_csv(
        control_messages_csv,
        control_summary_rows,
        CONTROL_RUN_SUMMARY_COLUMNS,
    )

    print(f"Treatments processed: {len(summary_rows)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {failed_count}")
    if partial_count:
        print(f"Partial: {partial_count}")
    print(f"Run-level summary CSV: {run_summary_csv}")
    print(f"Control messages CSV: {control_messages_csv}")
    print("Saved CATE analysis run completed.")


if __name__ == "__main__":
    main()
