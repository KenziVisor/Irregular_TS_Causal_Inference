from __future__ import annotations

import argparse
import os
import pickle
import platform
import re
import sys
import warnings
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from preprocess_mimic_iii_large_contract import canonicalize_stay_id_series
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ============================================================
# Config
# ============================================================
DATASET_MODEL = "physionet"
LATENT_TAGS_PATH = "../../data/predicted_latent_tags_230326_absolute_tags.csv"
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
BACKGROUND_FEATURE_COLUMNS = [
    "Age", "Gender", "Weight",
    "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4",
]

OUTPUT_DIR = "../../data/relevant_outputs/cate_outputs_predicted_230326"
SEED = 42
DOWN_SAMPLE = False
USE_EXPANDED_SAFE_CONFOUNDERS = True
MODEL_TYPE = "CausalForest"   # LinearDML or CausalForest
DEFAULT_SENSITIVITY_ALPHA = 0.05
ARTIFACT_SCHEMA_VERSION = 3
ESTIMATOR_STACK_DEVICE_NOTE = (
    "Running on CPU (EconML/scikit-learn estimators are CPU-based here); "
    "CUDA availability is recorded for provenance only."
)


def str_to_bool(value: str) -> bool:
    """
    Parse common string forms for booleans.
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use true/false."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate CATEs for latent clinical treatments."
    )
    parser.add_argument(
        "--model",
        choices=["physionet", "mimic"],
        default=DATASET_MODEL,
        help=f"Dataset selector for path defaults. Default: {DATASET_MODEL}",
    )
    parser.add_argument(
        "--latent-tags-path",
        default=None,
        help=(
            "Path to the latent tags CSV. Default: use script-level "
            f"LATENT_TAGS_PATH ({LATENT_TAGS_PATH}) if set."
        ),
    )
    parser.add_argument(
        "--physionet-pkl-path",
        default=None,
        help=(
            "Path to the processed PhysioNet pickle. Default: use script-level "
            f"PHYSIONET_PKL_PATH ({PHYSIONET_PKL_PATH}) if set."
        ),
    )
    parser.add_argument(
        "--graph-pkl-path",
        default=None,
        help=(
            "Path to the pickled causal graph. Default: use script-level "
            f"GRAPH_PKL_PATH ({GRAPH_PKL_PATH}) if set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for run outputs. Default: use script-level "
            f"OUTPUT_DIR ({OUTPUT_DIR}) if set."
        ),
    )
    parser.add_argument(
        "--down-sample",
        type=str_to_bool,
        default=DOWN_SAMPLE,
        help=f"Whether to down-sample the majority outcome class. Default: {DOWN_SAMPLE}",
    )
    parser.add_argument(
        "--use-expanded-safe-confounders",
        type=str_to_bool,
        default=USE_EXPANDED_SAFE_CONFOUNDERS,
        help=(
            "Whether to use the expanded safe confounder set instead of the minimal one. "
            f"Default: {USE_EXPANDED_SAFE_CONFOUNDERS}"
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["LinearDML", "CausalForest"],
        default=MODEL_TYPE,
        help=f"Estimator family to use. Default: {MODEL_TYPE}",
    )
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
    global_value: str | None,
    field_name: str,
    *,
    must_exist: bool = True,
) -> str:
    cli_flag_name = {
        "LATENT_TAGS_PATH": "--latent-tags-path",
        "PHYSIONET_PKL_PATH": "--physionet-pkl-path",
        "GRAPH_PKL_PATH": "--graph-pkl-path",
        "OUTPUT_DIR": "--output-dir",
    }.get(field_name, field_name)

    raw_value = cli_value if cli_value is not None else global_value
    if raw_value is None:
        raise ValueError(
            f"{field_name} is not configured. Provide {cli_flag_name} or set the "
            f"script-level {field_name}."
        )

    if not isinstance(raw_value, str):
        raise TypeError(f"{field_name} must be a string path. Got: {type(raw_value)!r}")

    raw_value = raw_value.strip()
    if not raw_value:
        raise ValueError(
            f"{field_name} is empty. Provide a non-empty path via {cli_flag_name} "
            f"or set the script-level {field_name}."
        )

    resolved_path = os.path.abspath(os.path.expanduser(raw_value))

    if must_exist:
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"{field_name} does not exist: {resolved_path}. Provide a valid "
                f"path via {cli_flag_name} or set the script-level {field_name}."
            )
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"{field_name} is not a file: {resolved_path}. Provide a valid file "
                f"path via {cli_flag_name} or set the script-level {field_name}."
            )
    else:
        if os.path.exists(resolved_path) and not os.path.isdir(resolved_path):
            raise NotADirectoryError(
                f"{field_name} must be a directory path: {resolved_path}"
            )

    return resolved_path


def detect_runtime_device() -> Dict[str, object]:
    info: Dict[str, object] = {
        "runtime_device_selected": "cpu",
        "torch_cuda_available": False,
        "torch_cuda_device_count": None,
        "torch_cuda_device_name": None,
        "runtime_device_note": "Running on CPU (CUDA unavailable).",
    }

    try:
        import torch
    except Exception as exc:
        info["runtime_device_note"] = (
            "PyTorch import failed; running on CPU. "
            f"Detail: {type(exc).__name__}: {exc}"
        )
        return info

    try:
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["runtime_device_note"] = (
            "torch.cuda.is_available() failed; running on CPU. "
            f"Detail: {type(exc).__name__}: {exc}"
        )
        return info

    if not info["torch_cuda_available"]:
        return info

    info["runtime_device_note"] = ESTIMATOR_STACK_DEVICE_NOTE

    try:
        info["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception:
        info["torch_cuda_device_count"] = None

    return info


def build_treatment_output_csv(
    treatment_dir: str,
    treatment: str,
    suffix: str,
) -> str:
    return os.path.join(treatment_dir, f"{treatment}_{suffix}.csv")


def build_treatment_output_pkl(
    treatment_dir: str,
    treatment: str,
    suffix: str,
) -> str:
    return os.path.join(treatment_dir, f"{treatment}_{suffix}.pkl")


def build_run_output_csv(output_dir: str, suffix: str) -> str:
    run_name = os.path.basename(os.path.normpath(output_dir))
    return os.path.join(output_dir, f"{run_name}_{suffix}.csv")


def build_summary_row_id(model_type: str, treatment: str) -> str:
    safe_model_type = model_type or "unknown_model"
    return f"{safe_model_type}__{treatment}"


def coerce_float(value: object) -> float | None:
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


def normalize_interval_value(
    value: object,
) -> Tuple[float | None, float | None] | None:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (coerce_float(value[0]), coerce_float(value[1]))

    raise ValueError("Expected sensitivity interval to be a 2-item tuple/list")


CLEAN_GLOBAL_SUMMARY_COLUMNS = [
    "row_id",
    "treatment",
    "model_type",
    "estimator_class",
    "n",
    "outcome_rate",
    "treatment_rate",
    "treated_outcome_positive_rate",
    "mean_cate",
    "std_cate",
    "min_cate",
    "max_cate",
    "mean_normalized_cate",
    "std_normalized_cate",
    "min_normalized_cate",
    "max_normalized_cate",
    "num_observed_confounders",
    "num_missing_graph_candidates",
    "observed_confounders",
    "missing_graph_candidates",
    "saved_direct_rv",
    "saved_direct_sensitivity_interval",
    "saved_direct_sensitivity_interval_lb",
    "saved_direct_sensitivity_interval_ub",
]


CONTROL_GLOBAL_SUMMARY_COLUMNS = [
    "row_id",
    "treatment",
    "model_type",
    "estimator_class",
    "artifact_schema_version",
    "econml_version",
    "sklearn_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "training_timestamp",
    "platform",
    "runtime_device_selected",
    "torch_cuda_available",
    "torch_cuda_device_count",
    "torch_cuda_device_name",
    "runtime_device_note",
    "estimator_module",
    "cache_values_used",
    "has_method_robustness_value",
    "has_method_sensitivity_interval",
    "has_method_sensitivity_summary",
    "has_method_summary",
    "has_attr_residuals",
    "saved_direct_rv_source",
    "saved_direct_rv_error",
    "saved_direct_sensitivity_interval_source",
    "saved_direct_sensitivity_interval_error",
    "saved_direct_sensitivity_summary_source",
    "saved_direct_sensitivity_summary_error",
    "saved_direct_estimator_summary_source",
    "saved_direct_estimator_summary_error",
    "saved_sensitivity_params_available",
    "saved_sensitivity_params_source",
    "saved_sensitivity_params_error",
    "saved_training_residuals_available",
    "saved_training_residuals_source",
    "saved_training_residuals_error",
    "saved_training_residuals_tuple_length",
    "latent_tags_path",
    "physionet_pkl_path",
    "graph_pkl_path",
    "output_dir",
    "cate_csv_path",
    "model_artifact_path",
]


MANAGER_GLOBAL_SUMMARY_COLUMNS = [
    "model_type",
    "treatment",
    "n",
    "outcome_rate",
    "treatment_rate",
    "treated_outcome_positive_rate",
    "mean_cate",
    "mean_normalized_cate",
]


def finalize_ordered_dataframe(
    rows: List[Dict[str, object]],
    columns: List[str],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df[columns]


def build_clean_global_summary_row(
    row: Dict[str, object],
) -> Dict[str, object]:
    interval_lb = None
    interval_ub = None
    interval_value = row.get("saved_direct_sensitivity_interval")
    if interval_value is not None:
        try:
            normalized_interval = normalize_interval_value(interval_value)
        except Exception:
            normalized_interval = None
        if normalized_interval is not None:
            interval_lb, interval_ub = normalized_interval

    return {
        "row_id": row["row_id"],
        "treatment": row["treatment"],
        "model_type": row["model_type"],
        "estimator_class": row["estimator_class"],
        "n": row["n"],
        "outcome_rate": row["outcome_rate"],
        "treatment_rate": row["treatment_rate"],
        "treated_outcome_positive_rate": row["treated_outcome_positive_rate"],
        "mean_cate": row["mean_cate"],
        "std_cate": row["std_cate"],
        "min_cate": row["min_cate"],
        "max_cate": row["max_cate"],
        "mean_normalized_cate": row["mean_normalized_cate"],
        "std_normalized_cate": row["std_normalized_cate"],
        "min_normalized_cate": row["min_normalized_cate"],
        "max_normalized_cate": row["max_normalized_cate"],
        "num_observed_confounders": row["num_observed_confounders"],
        "num_missing_graph_candidates": row["num_missing_graph_candidates"],
        "observed_confounders": row["observed_confounders"],
        "missing_graph_candidates": row["missing_graph_candidates"],
        "saved_direct_rv": row["saved_direct_rv"],
        "saved_direct_sensitivity_interval": row["saved_direct_sensitivity_interval"],
        "saved_direct_sensitivity_interval_lb": interval_lb,
        "saved_direct_sensitivity_interval_ub": interval_ub,
    }


def build_control_global_summary_row(
    row: Dict[str, object],
) -> Dict[str, object]:
    return {
        column: row.get(column)
        for column in CONTROL_GLOBAL_SUMMARY_COLUMNS
    }


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


def save_model_artifact(path: str, artifact: Dict[str, object]) -> None:
    with open(path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_artifact(path: str) -> Dict[str, object]:
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict):
        raise TypeError("Loaded model artifact is not a dict")
    return artifact


def get_installed_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_environment_metadata(
    runtime_device_info: Dict[str, object] | None = None,
) -> Dict[str, object]:
    if runtime_device_info is None:
        runtime_device_info = detect_runtime_device()

    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "econml_version": get_installed_version("econml"),
        "sklearn_version": get_installed_version("scikit-learn"),
        "numpy_version": get_installed_version("numpy"),
        "pandas_version": get_installed_version("pandas"),
        "scipy_version": get_installed_version("scipy"),
        "matplotlib_version": get_installed_version("matplotlib"),
        "networkx_version": get_installed_version("networkx"),
        "optuna_version": get_installed_version("optuna"),
        "torch_version": get_installed_version("torch"),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_device_selected": runtime_device_info.get("runtime_device_selected"),
        "torch_cuda_available": runtime_device_info.get("torch_cuda_available"),
        "torch_cuda_device_count": runtime_device_info.get("torch_cuda_device_count"),
        "torch_cuda_device_name": runtime_device_info.get("torch_cuda_device_name"),
        "runtime_device_note": runtime_device_info.get("runtime_device_note"),
    }


def log_environment_metadata(env_metadata: Dict[str, object]) -> None:
    print("Runtime environment:")
    for key in [
        "python_version",
        "platform",
        "econml_version",
        "sklearn_version",
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "matplotlib_version",
        "networkx_version",
        "optuna_version",
        "torch_version",
        "runtime_device_selected",
        "torch_cuda_available",
        "torch_cuda_device_count",
        "torch_cuda_device_name",
        "runtime_device_note",
    ]:
        print(f"  {key}: {env_metadata.get(key)}")


def log_runtime_configuration(
    *,
    latent_tags_path: str,
    processed_pkl_path: str,
    graph_pkl_path: str,
    output_dir: str,
    runtime_device_info: Dict[str, object],
) -> None:
    print("Runtime configuration:")
    print(f"  latent_tags_path: {latent_tags_path}")
    print(f"  processed_pkl_path: {processed_pkl_path}")
    print(f"  graph_pkl_path: {graph_pkl_path}")
    print(f"  output_dir: {output_dir}")
    print(
        "  runtime_device_selected: "
        f"{runtime_device_info.get('runtime_device_selected')}"
    )
    print(f"  runtime_device_note: {runtime_device_info.get('runtime_device_note')}")


def has_attribute(obj: object, attr_name: str) -> bool:
    try:
        getattr(obj, attr_name)
    except Exception:
        return False
    return True


def has_callable_attribute(obj: object, attr_name: str) -> bool:
    try:
        value = getattr(obj, attr_name)
    except Exception:
        return False
    return callable(value)


def collect_estimator_method_availability(est: object) -> Dict[str, bool]:
    return {
        "has_method_robustness_value": has_callable_attribute(est, "robustness_value"),
        "has_method_sensitivity_interval": has_callable_attribute(est, "sensitivity_interval"),
        "has_method_sensitivity_summary": has_callable_attribute(est, "sensitivity_summary"),
        "has_method_summary": has_callable_attribute(est, "summary"),
        "has_attr_residuals": has_attribute(est, "residuals_"),
    }


def format_estimator_method_availability(availability: Dict[str, bool]) -> str:
    return ", ".join([
        f"robustness_value={availability.get('has_method_robustness_value')}",
        f"sensitivity_interval={availability.get('has_method_sensitivity_interval')}",
        f"sensitivity_summary={availability.get('has_method_sensitivity_summary')}",
        f"summary={availability.get('has_method_summary')}",
        f"residuals_={availability.get('has_attr_residuals')}",
    ])


def build_effect_modifier_matrix(
    df: pd.DataFrame,
    model_artifact: Dict[str, object],
) -> np.ndarray | None:
    """
    Rebuild X in the saved training order for est.effect(X=...).
    """
    effect_modifiers = list(model_artifact.get("effect_modifiers", []))
    if not effect_modifiers:
        return None

    fill_values = {
        key: float(value)
        for key, value in dict(model_artifact.get("feature_fill_values", {})).items()
    }

    X_df = df.copy()
    for col in effect_modifiers:
        if col not in X_df.columns:
            X_df[col] = np.nan
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df[col] = X_df[col].fillna(fill_values.get(col, 0.0))

    return X_df[effect_modifiers].astype(float).to_numpy()


def log_dataframe_columns(name: str, df: pd.DataFrame) -> None:
    print(f"[{name}] columns ({len(df.columns)}): {list(df.columns)}")


def log_non_null_counts(
    name: str,
    df: pd.DataFrame,
    columns: List[str],
) -> None:
    print(f"[{name}] non-null counts:")
    for column in columns:
        if column in df.columns:
            print(f"  {column}: {int(df[column].notna().sum())}")
        else:
            print(f"  {column}: MISSING")


def sample_ts_ids(values: Set[str], limit: int = 5) -> List[str]:
    return sorted(str(value) for value in values)[:limit]


def find_merge_style_variants(
    available_columns: List[str],
    expected_column: str,
) -> List[str]:
    pattern = re.compile(
        rf"^{re.escape(expected_column)}(?:_[xy]|\.\d+|__[A-Za-z0-9]+)$"
    )
    return [
        column for column in available_columns
        if column == expected_column or pattern.fullmatch(column)
    ]


def normalize_expected_columns(
    df: pd.DataFrame,
    expected_columns: List[str],
    *,
    source_name: str,
) -> pd.DataFrame:
    out = df.copy()

    for expected_column in expected_columns:
        candidate_columns = find_merge_style_variants(
            list(out.columns),
            expected_column,
        )
        if not candidate_columns:
            continue
        if candidate_columns == [expected_column]:
            continue

        non_null_counts = {
            column: int(out[column].notna().sum())
            for column in candidate_columns
        }

        if (
            expected_column in non_null_counts
            and non_null_counts[expected_column] > 0
        ):
            selected_column = expected_column
        else:
            selected_column = max(
                candidate_columns,
                key=lambda column: (
                    non_null_counts[column],
                    column == expected_column,
                    column.endswith("_x"),
                    column.endswith("__latent"),
                    -len(column),
                ),
            )

        print(
            f"[{source_name}] Normalizing '{expected_column}' from candidates "
            f"{candidate_columns} with non-null counts {non_null_counts}. "
            f"Selected: {selected_column}"
        )
        out[expected_column] = out[selected_column]

        duplicate_columns = [
            column for column in candidate_columns
            if column != expected_column
        ]
        if duplicate_columns:
            print(
                f"[{source_name}] Dropping duplicate/suffixed columns for "
                f"'{expected_column}': {duplicate_columns}"
            )
            out = out.drop(columns=duplicate_columns)

    return out


def build_background_features(
    ts: pd.DataFrame,
    dataset_model: str | None = None,
) -> pd.DataFrame:
    """
    Build patient-level observed background covariates from ts.
    PhysioNet preprocessing already converts ICUType into ICUType_1..ICUType_4.
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
    model: str | None = None,
) -> pd.DataFrame:
    current_model = DATASET_MODEL if model is None else model
    latent_df = pd.read_csv(latent_tags_path)
    log_dataframe_columns("latent_df_raw", latent_df)

    if "ts_id" in latent_df.columns:
        latent_df = latent_df.copy()
    elif current_model == "mimic" and "icustay_id" in latent_df.columns:
        latent_df = latent_df.rename(columns={"icustay_id": "ts_id"}).copy()
    else:
        raise ValueError(
            "Latent tags CSV must contain 'ts_id', or contain 'icustay_id' when "
            f"--model mimic is used. Source: {latent_tags_path}"
        )
    latent_df["ts_id"] = canonicalize_stay_id_series(latent_df["ts_id"])
    if latent_df["ts_id"].isna().any():
        raise ValueError("Latent tags contain missing ts_id values after canonicalization.")
    latent_df = normalize_expected_columns(
        latent_df,
        list(TREATMENTS),
        source_name="latent_df",
    )
    log_dataframe_columns("latent_df_normalized", latent_df)

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
    log_dataframe_columns("oc", oc)

    if OUTCOME_COL in latent_df.columns:
        print(
            f"[load_analysis_dataframe] Dropping '{OUTCOME_COL}' from latent tags "
            "so the canonical outcome from oc is used."
        )
        latent_df = latent_df.drop(columns=[OUTCOME_COL])

    oc_small = oc[["ts_id", OUTCOME_COL]].copy().drop_duplicates(subset=["ts_id"])
    oc_small["ts_id"] = canonicalize_stay_id_series(oc_small["ts_id"])
    if oc_small["ts_id"].isna().any():
        raise ValueError("Processed pickle oc_small contains missing ts_id values after canonicalization.")

    bg_df = build_background_features(ts, dataset_model=current_model)
    bg_df["ts_id"] = canonicalize_stay_id_series(bg_df["ts_id"])
    if bg_df["ts_id"].isna().any():
        raise ValueError("Background features contain missing ts_id values after canonicalization.")
    log_dataframe_columns("bg_df", bg_df)

    latent_bg_overlap = [
        column for column in bg_df.columns
        if column != "ts_id" and column in latent_df.columns
    ]
    if latent_bg_overlap:
        print(
            "[load_analysis_dataframe] Dropping background columns from latent tags "
            "so ts-derived background features remain canonical: "
            f"{latent_bg_overlap}"
        )
        latent_df = latent_df.drop(columns=latent_bg_overlap)

    latent_ids = set(latent_df["ts_id"].dropna().tolist())
    oc_ids = set(oc_small["ts_id"].dropna().tolist())
    overlapping_ids = latent_ids & oc_ids
    only_latent_ids = latent_ids - oc_ids
    only_oc_ids = oc_ids - latent_ids
    print("[load_analysis_dataframe] ts_id overlap diagnostics:")
    print(f"  latent_df unique ids: {len(latent_ids)}")
    print(f"  oc_small unique ids: {len(oc_ids)}")
    print(f"  overlapping ids: {len(overlapping_ids)}")
    print(f"  sample only in latent_df: {sample_ts_ids(only_latent_ids)}")
    print(f"  sample only in oc_small: {sample_ts_ids(only_oc_ids)}")

    if oc_small.empty or not overlapping_ids:
        if current_model == "mimic":
            raise ValueError(
                "Processed MIMIC pickle and latent tags are misaligned: there are no "
                "overlapping ts_id values between latent tags and oc. A known cause is "
                "float-style stay identifiers such as '12345.0' versus '12345'. "
                "Regenerate the processed MIMIC pickle and then regenerate the MIMIC "
                "latent tags CSV."
            )
        raise ValueError(
            "Processed pickle and latent tags are misaligned: there are no overlapping "
            "ts_id values between latent tags and oc."
        )

    df = latent_df.merge(oc_small, on="ts_id", how="inner")
    df = df.merge(bg_df, on="ts_id", how="left")
    df = normalize_expected_columns(
        df,
        [OUTCOME_COL, *BACKGROUND_FEATURE_COLUMNS, *TREATMENTS],
        source_name="analysis_df",
    )

    rows_before_outcome_dropna = len(df)
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors="coerce")
    df = df.dropna(subset=[OUTCOME_COL]).copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    print(
        "[analysis_df] rows before dropping missing outcomes: "
        f"{rows_before_outcome_dropna}"
    )
    log_dataframe_columns("analysis_df", df)
    print(f"[analysis_df] shape: {df.shape}")
    log_non_null_counts(
        "analysis_df",
        df,
        ["ts_id", OUTCOME_COL, *TREATMENTS],
    )

    return df


def validate_analysis_dataframe(
    df: pd.DataFrame,
    treatments: List[str],
    outcome_col: str,
    model: str,
) -> None:
    outcome_exists = outcome_col in df.columns
    outcome_all_nan = True
    if outcome_exists:
        outcome_all_nan = int(df[outcome_col].notna().sum()) == 0

    missing_treatments = [
        treatment for treatment in treatments
        if treatment not in df.columns
    ]
    all_nan_treatments = [
        treatment for treatment in treatments
        if treatment in df.columns and int(df[treatment].notna().sum()) == 0
    ]

    if outcome_exists and not outcome_all_nan and not missing_treatments and not all_nan_treatments:
        return

    raise ValueError(
        f"Invalid analysis dataframe for model={model}. "
        f"Outcome column present: {outcome_exists}. "
        f"Outcome all-NaN: {outcome_all_nan}. "
        f"Missing treatment columns: {missing_treatments}. "
        f"All-NaN treatment columns: {all_nan_treatments}. "
        f"Shape: {df.shape}. "
        f"Columns: {list(df.columns)}"
    )


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
    """
    Map dataframe columns to graph node names.

    Rules:
    - ICUType_1..4 in dataframe correspond to ICUType in graph
    - in_hospital_mortality is the dataframe outcome -> ignore here
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


def get_colliders_on_backdoor_paths(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> Set[str]:
    """
    Find all collider nodes that appear on any backdoor path.
    """
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
    """
    Union of descendants of a node set.
    """
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
    """
    Expanded but safe adjustment set.

    Strategy:
    - Start from the current principled candidate pool
    - Remove any node that is:
        (a) a collider on any backdoor path
        (b) a descendant of such a collider
    - Keep the remaining nodes if they still block all backdoor paths
    - If not, fall back to minimal_backdoor_adjustment_set
    """
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

    # fallback: keep identification first
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

    If USE_EXPANDED_SAFE_CONFOUNDERS=True:
      - use a larger safe adjustment set
      - explicitly exclude colliders and descendants of colliders
    Otherwise:
      - use the old minimal adjustment set
    """
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


def choose_effect_modifiers(
    df: pd.DataFrame,
    treatment: str,
    confounders: List[str],
) -> List[str]:
    """
    Keep X compact and stable.
    You can change this later, but this is a sane default.
    """
    if DATASET_MODEL == "mimic":
        preferred = [
            "Age", "Gender", "Weight",
            "ChronicBurden", "AcuteInsult",
        ]
    else:
        preferred = [
            "Age", "Gender", "Weight",
            "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4",
            "ChronicRisk", "AcuteInsult",
        ]
    return [c for c in preferred if c in df.columns and c != treatment and c != OUTCOME_COL]


# ============================================================
# Estimation
# ============================================================
def make_dml_estimator():
    """
    Build the estimator according to MODEL_TYPE.
    Supported values:
      - "CausalForest"
      - "LinearDML"
    """
    if MODEL_TYPE == "CausalForest":
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

    if MODEL_TYPE == "LinearDML":
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

    raise ValueError(
        f"Unsupported MODEL_TYPE: {MODEL_TYPE}. "
        "Use 'CausalForest' or 'LinearDML'."
    )


def format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def serialize_scalar_or_array(value: object) -> object:
    if value is None:
        return None

    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = arr.item()
        if isinstance(scalar, (np.integer, int)) and not isinstance(scalar, bool):
            return int(scalar)
        if isinstance(scalar, (np.floating, float)):
            return float(scalar)
        return scalar

    return arr.tolist()


def serialize_json_safe(value: Any) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): serialize_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_json_safe(item) for item in value]
    if hasattr(value, "_asdict"):
        try:
            return serialize_json_safe(value._asdict())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            public_items = {
                key: serialize_json_safe(val)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
            if public_items:
                return public_items
        except Exception:
            pass
    return repr(value)


def serialize_sensitivity_object(value: object) -> object:
    known_attrs = {}
    for attr_name in ["theta", "sigma", "sigma2", "nu", "nu2", "cov", "covariance"]:
        if not has_attribute(value, attr_name):
            continue
        try:
            known_attrs[attr_name] = serialize_json_safe(getattr(value, attr_name))
        except Exception as exc:
            known_attrs[f"{attr_name}_error"] = format_exception(exc)

    if known_attrs:
        return {
            "type": type(value).__name__,
            "module": type(value).__module__,
            "attributes": known_attrs,
        }

    return serialize_json_safe(value)


def collect_sensitivity_params_snapshot(est: object) -> Dict[str, object]:
    candidate_names = [
        "sensitivity_params",
        "sensitivity_params_",
        "_sensitivity_params",
        "sensitivity_elements",
        "_sensitivity_elements",
    ]
    extra_names = sorted(
        name for name in dir(est)
        if "sensitivity" in name.lower() and name not in candidate_names
    )

    errors: List[str] = []
    for attr_name in candidate_names + extra_names:
        try:
            value = getattr(est, attr_name)
        except AttributeError:
            continue
        except Exception as exc:
            errors.append(f"{attr_name}: {format_exception(exc)}")
            continue

        if callable(value):
            errors.append(f"{attr_name}: callable attribute, not raw sensitivity params")
            continue

        try:
            serialized = serialize_sensitivity_object(value)
        except Exception as exc:
            return {
                "saved_sensitivity_params_available": False,
                "saved_sensitivity_params_source": f"estimator.{attr_name}",
                "saved_sensitivity_params_error": format_exception(exc),
                "saved_sensitivity_params_serialized": None,
            }

        return {
            "saved_sensitivity_params_available": True,
            "saved_sensitivity_params_source": f"estimator.{attr_name}",
            "saved_sensitivity_params_error": None,
            "saved_sensitivity_params_serialized": serialized,
        }

    error_message = "No non-callable sensitivity parameter attribute found on estimator"
    if errors:
        error_message = f"{error_message}; checked attributes: {' | '.join(errors)}"

    return {
        "saved_sensitivity_params_available": False,
        "saved_sensitivity_params_source": "not_exposed_in_training_env",
        "saved_sensitivity_params_error": error_message,
        "saved_sensitivity_params_serialized": None,
    }


def attempt_direct_estimator_call(
    est: object,
    method_name: str,
    candidate_kwargs: List[Dict[str, object]],
) -> Tuple[object | None, str, str | None]:
    if not has_attribute(est, method_name):
        return None, "api_absent_in_training_env", (
            f"Estimator does not expose '{method_name}'"
        )

    method = getattr(est, method_name)
    if not callable(method):
        return None, "attribute_not_callable_in_training_env", (
            f"Estimator attribute '{method_name}' is not callable"
        )

    last_error: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            return method(**kwargs), "saved_training_direct", None
        except Exception as exc:
            last_error = exc

    if last_error is None:
        return None, "training_direct_call_failed", (
            f"Unable to call '{method_name}'"
        )
    return None, "training_direct_call_failed", format_exception(last_error)


def extract_estimator_summary_text(
    est: object,
    effect_modifiers: List[str],
) -> Tuple[str | None, str, str | None]:
    if not has_callable_attribute(est, "summary"):
        return None, "api_absent_in_training_env", "Estimator does not expose 'summary'"

    method = getattr(est, "summary")
    candidate_kwargs: List[Dict[str, object]] = []
    if effect_modifiers:
        candidate_kwargs.append({"feature_names": effect_modifiers})
    candidate_kwargs.append({})

    last_error: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            return str(method(**kwargs)), "saved_training_direct", None
        except Exception as exc:
            last_error = exc

    if last_error is None:
        return None, "training_direct_call_failed", "Unable to call 'summary'"
    return None, "training_direct_call_failed", format_exception(last_error)


def collect_direct_diagnostics(
    est: object,
    effect_modifiers: List[str],
) -> Dict[str, object]:
    diagnostics: Dict[str, object] = {}
    availability = collect_estimator_method_availability(est)
    diagnostics.update(availability)

    method_specs = {
        "robustness_value": [
            {},
            {"null_hypothesis": 0.0, "alpha": DEFAULT_SENSITIVITY_ALPHA},
        ],
        "sensitivity_summary": [
            {},
            {"null_hypothesis": 0.0, "alpha": DEFAULT_SENSITIVITY_ALPHA},
        ],
        "sensitivity_interval": [
            {},
            {"alpha": DEFAULT_SENSITIVITY_ALPHA, "interval_type": "ci"},
        ],
    }

    key_map = {
        "robustness_value": ("saved_direct_rv", "direct_robustness_value"),
        "sensitivity_interval": (
            "saved_direct_sensitivity_interval",
            "direct_sensitivity_interval",
        ),
        "sensitivity_summary": (
            "saved_direct_sensitivity_summary",
            "direct_sensitivity_summary",
        ),
    }

    for method_name, candidate_kwargs in method_specs.items():
        value, source, error = attempt_direct_estimator_call(
            est=est,
            method_name=method_name,
            candidate_kwargs=candidate_kwargs,
        )
        artifact_key, legacy_key = key_map[method_name]
        diagnostics[artifact_key] = None
        diagnostics[f"{artifact_key}_source"] = source
        diagnostics[f"{artifact_key}_error"] = error
        diagnostics[legacy_key] = None
        diagnostics[f"{legacy_key}_source"] = source
        diagnostics[f"{legacy_key}_error"] = error

        if value is None:
            continue

        if method_name == "sensitivity_summary":
            serialized_value = str(value)
        elif method_name == "sensitivity_interval":
            try:
                lb, ub = value
                serialized_value = [
                    serialize_scalar_or_array(lb),
                    serialize_scalar_or_array(ub),
                ]
            except Exception as exc:
                serialized_value = None
                error = format_exception(exc)
                diagnostics[f"{artifact_key}_error"] = error
                diagnostics[f"{legacy_key}_error"] = error
        else:
            serialized_value = serialize_scalar_or_array(value)

        diagnostics[artifact_key] = serialized_value
        diagnostics[legacy_key] = serialized_value

    try:
        residuals = est.residuals_
        diagnostics["direct_residuals_available"] = True
        diagnostics["direct_residuals_tuple_length"] = len(residuals) if isinstance(
            residuals, (tuple, list)
        ) else None
        diagnostics["direct_residuals_error"] = None
        diagnostics["saved_training_residuals_available"] = True
        diagnostics["saved_training_residuals_source"] = "saved_training_direct"
        diagnostics["saved_training_residuals_error"] = None
        diagnostics["saved_training_residuals_tuple_length"] = diagnostics[
            "direct_residuals_tuple_length"
        ]
    except Exception as exc:
        diagnostics["direct_residuals_available"] = False
        diagnostics["direct_residuals_tuple_length"] = None
        diagnostics["direct_residuals_error"] = format_exception(exc)
        diagnostics["saved_training_residuals_available"] = False
        diagnostics["saved_training_residuals_source"] = "training_direct_call_failed"
        diagnostics["saved_training_residuals_error"] = format_exception(exc)
        diagnostics["saved_training_residuals_tuple_length"] = None

    summary_text, summary_source, summary_error = extract_estimator_summary_text(
        est=est,
        effect_modifiers=effect_modifiers,
    )
    diagnostics["saved_direct_estimator_summary_text"] = summary_text
    diagnostics["saved_direct_estimator_summary_source"] = summary_source
    diagnostics["saved_direct_estimator_summary_error"] = summary_error
    diagnostics["direct_estimator_summary_text"] = summary_text
    diagnostics["direct_estimator_summary_source"] = summary_source
    diagnostics["direct_estimator_summary_error"] = summary_error

    diagnostics.update(collect_sensitivity_params_snapshot(est))

    if isinstance(est, LinearDML):
        try:
            diagnostics["linear_coef"] = serialize_scalar_or_array(est.coef_)
        except Exception as exc:
            diagnostics["linear_coef"] = None
            diagnostics["linear_coef_error"] = format_exception(exc)

        try:
            diagnostics["linear_intercept"] = serialize_scalar_or_array(est.intercept_)
        except Exception as exc:
            diagnostics["linear_intercept"] = None
            diagnostics["linear_intercept_error"] = format_exception(exc)

        try:
            coef_inference = est.coef__inference()
            diagnostics["linear_coef_inference_table"] = (
                coef_inference.summary_frame().reset_index().to_dict(orient="records")
            )
            diagnostics["linear_coef_inference_error"] = None
        except Exception as exc:
            diagnostics["linear_coef_inference_table"] = None
            diagnostics["linear_coef_inference_error"] = format_exception(exc)

    if isinstance(est, CausalForestDML):
        try:
            importances = np.asarray(est.feature_importances_, dtype=float).reshape(-1)
            diagnostics["causal_forest_feature_importances"] = importances.tolist()
            diagnostics["causal_forest_feature_importance_by_modifier"] = [
                {"variable": variable, "importance": float(importance)}
                for variable, importance in zip(effect_modifiers, importances)
            ]
            diagnostics["causal_forest_feature_importances_error"] = None
        except Exception as exc:
            diagnostics["causal_forest_feature_importances"] = None
            diagnostics["causal_forest_feature_importance_by_modifier"] = None
            diagnostics["causal_forest_feature_importances_error"] = format_exception(exc)

    return diagnostics


def fit_one_treatment(
    df: pd.DataFrame,
    treatment: str,
    confounders: List[str],
    effect_modifiers: List[str],
    runtime_device_info: Dict[str, object],
) -> Tuple[object, pd.DataFrame, Dict[str, float], str, Dict[str, object]]:
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

    total_rows = int(len(df))
    treatment_non_null = int(df[treatment].notna().sum())
    outcome_non_null = int(df[OUTCOME_COL].notna().sum())

    work_df = df.copy()
    work_df[treatment] = pd.to_numeric(work_df[treatment], errors="coerce")
    work_df[OUTCOME_COL] = pd.to_numeric(work_df[OUTCOME_COL], errors="coerce")
    rows_after_dropna = int(
        work_df.dropna(subset=[treatment, OUTCOME_COL]).shape[0]
    )

    print(
        f"[{treatment}] row counts before fit: total_rows={total_rows}, "
        f"{treatment}_non_null={treatment_non_null}, "
        f"{OUTCOME_COL}_non_null={outcome_non_null}, "
        f"rows_after_dropna={rows_after_dropna}"
    )
    if rows_after_dropna == 0:
        raise ValueError(
            f"No rows remain for treatment {treatment} after dropping rows with "
            f"missing treatment/outcome. total_rows={total_rows}, "
            f"{treatment}_non_null={treatment_non_null}, "
            f"{OUTCOME_COL}_non_null={outcome_non_null}, "
            f"rows_after_dropna={rows_after_dropna}"
        )

    work_df = work_df.dropna(subset=[treatment, OUTCOME_COL]).copy()

    work_df[treatment] = work_df[treatment].astype(int)
    work_df[OUTCOME_COL] = work_df[OUTCOME_COL].astype(int)

    # Keep only columns that actually exist
    confounders = [c for c in confounders if c in work_df.columns and c not in [treatment, OUTCOME_COL]]
    effect_modifiers = [c for c in effect_modifiers if c in work_df.columns and c not in [treatment, OUTCOME_COL]]
    print(
        f"[{treatment}] selected variables: confounders={len(confounders)} | "
        f"effect_modifiers={len(effect_modifiers)}"
    )

    used_cols = ["ts_id", treatment, OUTCOME_COL] + confounders + effect_modifiers
    used_cols = list(dict.fromkeys(used_cols))

    print(f"\n[{treatment}] missingness before filtering/imputation:")
    print(work_df[used_cols].isna().mean().sort_values(ascending=False))
    print(f"[{treatment}] rows before filtering: {len(work_df)}")

    model_df = work_df.copy()

    # Impute confounders and effect modifiers instead of dropping rows
    numeric_cols = list(dict.fromkeys(confounders + effect_modifiers))
    fill_values: Dict[str, float] = {}
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

        if model_df[col].isna().all():
            fill_value = 0.0
        else:
            fill_value = float(model_df[col].median())

        fill_values[col] = fill_value
        model_df[col] = model_df[col].fillna(fill_value)

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

    est = make_dml_estimator()
    pre_fit_availability = collect_estimator_method_availability(est)
    print(
        f"[{treatment}] estimator method availability before fit: "
        f"{format_estimator_method_availability(pre_fit_availability)}"
    )
    print(
        f"[{treatment}] Starting estimator fit | n={len(model_df):,} | "
        f"W_shape={None if W is None else W.shape} | X_shape={None if X is None else X.shape}"
    )

    est.fit(Y=Y, T=T, X=X, W=W, cache_values=True)
    post_fit_availability = collect_estimator_method_availability(est)
    print(
        f"[{treatment}] estimator method availability after fit: "
        f"{format_estimator_method_availability(post_fit_availability)}"
    )
    cate = est.effect(X=X)
    print(f"[{treatment}] Generated CATE estimates for {len(cate):,} rows")

    out = model_df[["ts_id", treatment, OUTCOME_COL]].copy()
    out["CATE"] = cate

    # Normalize by target rate (outcome prevalence)
    target_rate = float(model_df[OUTCOME_COL].mean())

    if target_rate > 0:
        out["normalized_CATE"] = out["CATE"] / target_rate
    else:
        out["normalized_CATE"] = np.nan

    try:
        lb, ub = est.effect_interval(X=X, alpha=0.05)
        out["CATE_lower_95"] = lb
        out["CATE_upper_95"] = ub

        if target_rate > 0:
            out["normalized_CATE_lower_95"] = out["CATE_lower_95"] / target_rate
            out["normalized_CATE_upper_95"] = out["CATE_upper_95"] / target_rate
        else:
            out["normalized_CATE_lower_95"] = np.nan
            out["normalized_CATE_upper_95"] = np.nan

    except Exception:
        out["CATE_lower_95"] = np.nan
        out["CATE_upper_95"] = np.nan
        out["normalized_CATE_lower_95"] = np.nan
        out["normalized_CATE_upper_95"] = np.nan

    formula = (
        f"Model = {MODEL_TYPE}\n"
        f"CATE_{treatment}(x) = E[{OUTCOME_COL}(1) - {OUTCOME_COL}(0) | X=x]\n"
        f"T = {treatment}\n"
        f"Y = {OUTCOME_COL}\n"
        f"W (backdoor confounders) = {confounders if confounders else 'None'}\n"
        f"X (effect modifiers) = {effect_modifiers if effect_modifiers else 'None'}\n"
        f"Normalized CATE = CATE / outcome_rate"
    )

    summary = {
        "model_type": MODEL_TYPE,
        "n": float(len(out)),
        "outcome_rate": float(model_df[OUTCOME_COL].mean()),
        "treatment_rate": float(model_df[treatment].mean()),
        "treated_outcome_positive_rate": float(
            ((model_df[treatment] == 1) & (model_df[OUTCOME_COL] == 1)).mean()
        ),
        "mean_cate": float(out["CATE"].mean()),
        "std_cate": float(out["CATE"].std()),
        "min_cate": float(out["CATE"].min()),
        "max_cate": float(out["CATE"].max()),
        "mean_normalized_cate": float(out["normalized_CATE"].mean()) if out[
            "normalized_CATE"].notna().any() else np.nan,
        "std_normalized_cate": float(out["normalized_CATE"].std()) if out["normalized_CATE"].notna().any() else np.nan,
        "min_normalized_cate": float(out["normalized_CATE"].min()) if out["normalized_CATE"].notna().any() else np.nan,
        "max_normalized_cate": float(out["normalized_CATE"].max()) if out["normalized_CATE"].notna().any() else np.nan,
    }

    direct_diagnostics = collect_direct_diagnostics(
        est=est,
        effect_modifiers=effect_modifiers,
    )
    print(
        f"[{treatment}] direct sensitivity extraction sources: "
        f"rv={direct_diagnostics['saved_direct_rv_source']}, "
        f"interval={direct_diagnostics['saved_direct_sensitivity_interval_source']}, "
        f"summary={direct_diagnostics['saved_direct_sensitivity_summary_source']}"
    )
    for label, error_key in [
        ("rv", "saved_direct_rv_error"),
        ("sensitivity_interval", "saved_direct_sensitivity_interval_error"),
        ("sensitivity_summary", "saved_direct_sensitivity_summary_error"),
        ("estimator_summary_text", "saved_direct_estimator_summary_error"),
    ]:
        if direct_diagnostics.get(error_key):
            print(f"[{treatment}] {label} detail: {direct_diagnostics[error_key]}")

    print(
        f"[{treatment}] raw sensitivity params saved: "
        f"{direct_diagnostics['saved_sensitivity_params_available']} "
        f"(source={direct_diagnostics['saved_sensitivity_params_source']})"
    )
    if direct_diagnostics.get("saved_sensitivity_params_error"):
        print(
            f"[{treatment}] raw sensitivity params detail: "
            f"{direct_diagnostics['saved_sensitivity_params_error']}"
        )

    env_metadata = collect_environment_metadata(runtime_device_info=runtime_device_info)

    model_artifact = {
        "estimator": est,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        **env_metadata,
        "estimator_module": type(est).__module__,
        "model_type": MODEL_TYPE,
        "treatment": treatment,
        "outcome_col": OUTCOME_COL,
        "confounders": confounders,
        "effect_modifiers": effect_modifiers,
        "cache_values_used": True,
        "estimator_class": type(est).__name__,
        "has_method_robustness_value": post_fit_availability["has_method_robustness_value"],
        "has_method_sensitivity_interval": post_fit_availability["has_method_sensitivity_interval"],
        "has_method_sensitivity_summary": post_fit_availability["has_method_sensitivity_summary"],
        "has_method_summary": post_fit_availability["has_method_summary"],
        "has_attr_residuals": post_fit_availability["has_attr_residuals"],
        "estimator_method_availability_pre_fit": pre_fit_availability,
        "estimator_method_availability_post_fit": post_fit_availability,
        "confounders_order": list(confounders),
        "effect_modifiers_order": list(effect_modifiers),
        "feature_fill_values": fill_values,
        "formula": formula,
        "summary": summary,
        "saved_direct_rv": direct_diagnostics["saved_direct_rv"],
        "saved_direct_rv_source": direct_diagnostics["saved_direct_rv_source"],
        "saved_direct_rv_error": direct_diagnostics["saved_direct_rv_error"],
        "saved_direct_sensitivity_interval": direct_diagnostics[
            "saved_direct_sensitivity_interval"
        ],
        "saved_direct_sensitivity_interval_source": direct_diagnostics[
            "saved_direct_sensitivity_interval_source"
        ],
        "saved_direct_sensitivity_interval_error": direct_diagnostics[
            "saved_direct_sensitivity_interval_error"
        ],
        "saved_direct_sensitivity_summary": direct_diagnostics[
            "saved_direct_sensitivity_summary"
        ],
        "saved_direct_sensitivity_summary_source": direct_diagnostics[
            "saved_direct_sensitivity_summary_source"
        ],
        "saved_direct_sensitivity_summary_error": direct_diagnostics[
            "saved_direct_sensitivity_summary_error"
        ],
        "saved_direct_estimator_summary_text": direct_diagnostics[
            "saved_direct_estimator_summary_text"
        ],
        "saved_direct_estimator_summary_source": direct_diagnostics[
            "saved_direct_estimator_summary_source"
        ],
        "saved_direct_estimator_summary_error": direct_diagnostics[
            "saved_direct_estimator_summary_error"
        ],
        "saved_sensitivity_params_available": direct_diagnostics[
            "saved_sensitivity_params_available"
        ],
        "saved_sensitivity_params_source": direct_diagnostics[
            "saved_sensitivity_params_source"
        ],
        "saved_sensitivity_params_error": direct_diagnostics[
            "saved_sensitivity_params_error"
        ],
        "saved_sensitivity_params_serialized": direct_diagnostics[
            "saved_sensitivity_params_serialized"
        ],
        "saved_training_residuals_available": direct_diagnostics[
            "saved_training_residuals_available"
        ],
        "saved_training_residuals_source": direct_diagnostics[
            "saved_training_residuals_source"
        ],
        "saved_training_residuals_error": direct_diagnostics[
            "saved_training_residuals_error"
        ],
        "saved_training_residuals_tuple_length": direct_diagnostics[
            "saved_training_residuals_tuple_length"
        ],
        "direct_diagnostics": direct_diagnostics,
    }

    return est, out, summary, formula, model_artifact


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
        f.write(f"Outcome (graph node): {GRAPH_OUTCOME_NODE}\n\n")

        f.write("Method used:\n")
        f.write("- Allowed adjustment variables: latent + background/meta only\n")
        f.write("- Excluded descendants of treatment\n")
        f.write("- Built candidate pool using ancestors of treatment and outcome in do(T) graph\n")
        if USE_EXPANDED_SAFE_CONFOUNDERS:
            f.write("- Excluded colliders on backdoor paths\n")
            f.write("- Excluded descendants of those colliders\n")
            f.write("- Used expanded safe blocking set when possible\n")
            f.write("- Fell back to minimal blocking set only if needed for identification\n\n")
        else:
            f.write("- Minimalized set by blocking all backdoor paths via d-separation\n\n")

        f.write(f"Identifiable with available nodes: {confounder_info['identifiable_with_available_nodes']}\n\n")

        f.write("Candidate pool:\n")
        if confounder_info["candidate_pool"]:
            for c in confounder_info["candidate_pool"]:
                f.write(f"  - {c}\n")
        else:
            f.write("  - None\n")

        f.write("\nColliders removed from candidate pool:\n")
        if confounder_info["colliders_removed"]:
            for c in confounder_info["colliders_removed"]:
                f.write(f"  - {c}\n")
        else:
            f.write("  - None\n")

        f.write("\nDescendants of colliders removed from candidate pool:\n")
        if confounder_info["collider_descendants_removed"]:
            for c in confounder_info["collider_descendants_removed"]:
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
    model_artifact_path: str,
    model_artifact: Dict[str, object],
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== CATE Summary Results ===\n\n")
        f.write(f"Model type: {summary['model_type']}\n")
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
        f.write(
            "Treatment & outcome positive rate: "
            f"{summary['treated_outcome_positive_rate']:.6f}\n"
        )
        f.write(f"Mean CATE: {summary['mean_cate']:.6f}\n")
        f.write(f"Std CATE: {summary['std_cate']:.6f}\n")
        f.write(f"Min CATE: {summary['min_cate']:.6f}\n")
        f.write(f"Max CATE: {summary['max_cate']:.6f}\n")
        f.write(f"Mean normalized CATE: {summary['mean_normalized_cate']:.6f}\n")
        f.write(f"Std normalized CATE: {summary['std_normalized_cate']:.6f}\n")
        f.write(f"Min normalized CATE: {summary['min_normalized_cate']:.6f}\n")
        f.write(f"Max normalized CATE: {summary['max_normalized_cate']:.6f}\n\n")

        f.write("Training-time extraction diagnostics:\n")
        f.write(f"Artifact schema version: {model_artifact.get('artifact_schema_version')}\n")
        f.write(f"Training timestamp: {model_artifact.get('training_timestamp')}\n")
        f.write(f"Python version: {model_artifact.get('python_version')}\n")
        f.write(f"Platform: {model_artifact.get('platform')}\n")
        f.write(
            "Runtime device selected: "
            f"{model_artifact.get('runtime_device_selected')}\n"
        )
        f.write(
            "torch CUDA available: "
            f"{model_artifact.get('torch_cuda_available')}\n"
        )
        f.write(
            "torch CUDA device count: "
            f"{model_artifact.get('torch_cuda_device_count')}\n"
        )
        f.write(
            "torch CUDA device name: "
            f"{model_artifact.get('torch_cuda_device_name')}\n"
        )
        f.write(
            "Runtime device note: "
            f"{model_artifact.get('runtime_device_note')}\n"
        )
        f.write(f"EconML version: {model_artifact.get('econml_version')}\n")
        f.write(f"scikit-learn version: {model_artifact.get('sklearn_version')}\n")
        f.write(f"NumPy version: {model_artifact.get('numpy_version')}\n")
        f.write(f"Pandas version: {model_artifact.get('pandas_version')}\n")
        f.write(f"SciPy version: {model_artifact.get('scipy_version')}\n")
        f.write(f"Estimator module: {model_artifact.get('estimator_module')}\n")
        f.write(f"Estimator class: {model_artifact.get('estimator_class')}\n")
        f.write(f"cache_values_used: {model_artifact.get('cache_values_used')}\n")
        f.write(
            "Method availability after fit: "
            f"robustness_value={model_artifact.get('has_method_robustness_value')}, "
            f"sensitivity_interval={model_artifact.get('has_method_sensitivity_interval')}, "
            f"sensitivity_summary={model_artifact.get('has_method_sensitivity_summary')}, "
            f"summary={model_artifact.get('has_method_summary')}, "
            f"residuals_={model_artifact.get('has_attr_residuals')}\n"
        )
        f.write(f"Saved direct RV value: {model_artifact.get('saved_direct_rv')}\n")
        f.write(
            f"Saved direct RV source: {model_artifact.get('saved_direct_rv_source')}\n"
        )
        f.write(
            f"Saved direct RV error: {model_artifact.get('saved_direct_rv_error')}\n"
        )
        f.write(
            "Saved direct sensitivity interval value: "
            f"{model_artifact.get('saved_direct_sensitivity_interval')}\n"
        )
        f.write(
            "Saved direct sensitivity interval source: "
            f"{model_artifact.get('saved_direct_sensitivity_interval_source')}\n"
        )
        f.write(
            "Saved direct sensitivity interval error: "
            f"{model_artifact.get('saved_direct_sensitivity_interval_error')}\n"
        )
        f.write(
            "Saved direct sensitivity summary text: "
            f"{model_artifact.get('saved_direct_sensitivity_summary')}\n"
        )
        f.write(
            "Saved direct sensitivity summary source: "
            f"{model_artifact.get('saved_direct_sensitivity_summary_source')}\n"
        )
        f.write(
            "Saved direct sensitivity summary error: "
            f"{model_artifact.get('saved_direct_sensitivity_summary_error')}\n"
        )
        f.write(
            "Saved direct estimator summary text source: "
            f"{model_artifact.get('saved_direct_estimator_summary_source')}\n"
        )
        f.write(
            "Saved direct estimator summary text error: "
            f"{model_artifact.get('saved_direct_estimator_summary_error')}\n"
        )
        f.write(
            "Saved sensitivity params available: "
            f"{model_artifact.get('saved_sensitivity_params_available')}\n"
        )
        f.write(
            "Saved sensitivity params source: "
            f"{model_artifact.get('saved_sensitivity_params_source')}\n"
        )
        f.write(
            "Saved sensitivity params error: "
            f"{model_artifact.get('saved_sensitivity_params_error')}\n\n"
        )
        f.write(
            "Saved training residuals available: "
            f"{model_artifact.get('saved_training_residuals_available')}\n"
        )
        f.write(
            "Saved training residuals source: "
            f"{model_artifact.get('saved_training_residuals_source')}\n"
        )
        f.write(
            "Saved training residuals error: "
            f"{model_artifact.get('saved_training_residuals_error')}\n"
        )
        f.write(
            "Saved training residuals tuple length: "
            f"{model_artifact.get('saved_training_residuals_tuple_length')}\n\n"
        )

        f.write(f"Per-patient CATE file: {cate_csv_path}\n")
        f.write(f"Saved model artifact: {model_artifact_path}\n")


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
    global DOWN_SAMPLE
    global USE_EXPANDED_SAFE_CONFOUNDERS
    global MODEL_TYPE

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
    DOWN_SAMPLE = args.down_sample
    USE_EXPANDED_SAFE_CONFOUNDERS = args.use_expanded_safe_confounders
    MODEL_TYPE = args.model_type

    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runtime_device_info = detect_runtime_device()
    log_runtime_configuration(
        latent_tags_path=LATENT_TAGS_PATH,
        processed_pkl_path=PHYSIONET_PKL_PATH,
        graph_pkl_path=GRAPH_PKL_PATH,
        output_dir=OUTPUT_DIR,
        runtime_device_info=runtime_device_info,
    )
    startup_env_metadata = collect_environment_metadata(
        runtime_device_info=runtime_device_info,
    )
    log_environment_metadata(startup_env_metadata)

    print("=== Starting CATE estimation run ===")
    print("[1/3] Loading dataframe and graph...")
    df = load_analysis_dataframe(
        LATENT_TAGS_PATH,
        PHYSIONET_PKL_PATH,
        model=DATASET_MODEL,
    )
    validate_analysis_dataframe(
        df=df,
        treatments=TREATMENTS,
        outcome_col=OUTCOME_COL,
        model=DATASET_MODEL,
    )
    G = load_graph(GRAPH_PKL_PATH)

    print(f"Loaded df shape: {df.shape}")
    print(f"Outcome rate before down-sampling: {df[OUTCOME_COL].mean():.4f}")
    print(f"DAG size: nodes={G.number_of_nodes()} | edges={G.number_of_edges()}")

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Down-sample: {DOWN_SAMPLE}")
    print(f"Use expanded safe confounders: {USE_EXPANDED_SAFE_CONFOUNDERS}")
    print(f"Model type: {MODEL_TYPE}")

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
    global_summary_rows = []

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
        effect_modifiers = choose_effect_modifiers(df, treatment, confounders)
        print(
            f"[{treatment}] Confounder discovery finished: observed={len(confounders)} | "
            f"effect_modifiers={len(effect_modifiers)}"
        )

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

        cate_csv = build_treatment_output_csv(
            treatment_dir=treatment_dir,
            treatment=treatment,
            suffix="cate",
        )
        model_pkl = build_treatment_output_pkl(
            treatment_dir=treatment_dir,
            treatment=treatment,
            suffix="model",
        )

        feature_importance_csv = build_treatment_output_csv(
            treatment_dir=treatment_dir,
            treatment=treatment,
            suffix="feature_importance",
        )

        write_confounder_analysis(
            path=confounder_txt,
            treatment=treatment,
            confounder_info=confounder_info,
        )

        try:
            print(f"[{treatment}] Starting model fit")
            est, cate_df, summary, formula, model_artifact = fit_one_treatment(
                df=df,
                treatment=treatment,
                confounders=confounders,
                effect_modifiers=effect_modifiers,
                runtime_device_info=runtime_device_info,
            )

            model_artifact["latent_tags_path"] = LATENT_TAGS_PATH
            model_artifact["physionet_pkl_path"] = PHYSIONET_PKL_PATH
            model_artifact["graph_pkl_path"] = GRAPH_PKL_PATH
            model_artifact["output_dir"] = OUTPUT_DIR

            # ====================================================
            # Feature importance for CATE heterogeneity
            # ====================================================
            if effect_modifiers and MODEL_TYPE == "CausalForest" and hasattr(est, "feature_importances_"):
                importance_df = pd.DataFrame({
                    "variable": effect_modifiers,
                    "importance": est.feature_importances_
                }).sort_values("importance", ascending=False)

                importance_df.to_csv(feature_importance_csv, index=False)

            elif effect_modifiers and MODEL_TYPE == "LinearDML":
                # Optional: save linear final-stage coefficients if available
                try:
                    coef = np.ravel(est.coef_)
                    if len(coef) == len(effect_modifiers):
                        coef_df = pd.DataFrame({
                            "variable": effect_modifiers,
                            "coefficient": coef
                        }).sort_values("coefficient", ascending=False)
                        coef_df.to_csv(feature_importance_csv, index=False)
                except Exception:
                    pass

            print(f"[{treatment}] Writing per-treatment outputs")
            cate_df.to_csv(cate_csv, index=False)
            save_model_artifact(model_pkl, model_artifact)

            write_summary_results(
                path=summary_txt,
                treatment=treatment,
                formula=formula,
                summary=summary,
                confounder_info=confounder_info,
                cate_csv_path=cate_csv,
                model_artifact_path=model_pkl,
                model_artifact=model_artifact,
            )

            global_summary_rows.append({
                "row_id": build_summary_row_id(summary["model_type"], treatment),
                "artifact_schema_version": model_artifact["artifact_schema_version"],
                "model_type": summary["model_type"],
                "treatment": treatment,
                "n": int(summary["n"]),
                "outcome_rate": summary["outcome_rate"],
                "treatment_rate": summary["treatment_rate"],
                "treated_outcome_positive_rate": summary["treated_outcome_positive_rate"],
                "mean_cate": summary["mean_cate"],
                "std_cate": summary["std_cate"],
                "min_cate": summary["min_cate"],
                "max_cate": summary["max_cate"],
                "mean_normalized_cate": summary["mean_normalized_cate"],
                "std_normalized_cate": summary["std_normalized_cate"],
                "min_normalized_cate": summary["min_normalized_cate"],
                "max_normalized_cate": summary["max_normalized_cate"],
                "num_observed_confounders": len(confounder_info["observed_confounders"]),
                "num_missing_graph_candidates": len(confounder_info["missing_graph_nodes"]),
                "observed_confounders": ", ".join(confounder_info["observed_confounders"]),
                "missing_graph_candidates": ", ".join(confounder_info["missing_graph_nodes"]),
                "econml_version": model_artifact["econml_version"],
                "sklearn_version": model_artifact["sklearn_version"],
                "numpy_version": model_artifact["numpy_version"],
                "pandas_version": model_artifact["pandas_version"],
                "scipy_version": model_artifact["scipy_version"],
                "training_timestamp": model_artifact["training_timestamp"],
                "platform": model_artifact["platform"],
                "runtime_device_selected": model_artifact["runtime_device_selected"],
                "torch_cuda_available": model_artifact["torch_cuda_available"],
                "torch_cuda_device_count": model_artifact["torch_cuda_device_count"],
                "torch_cuda_device_name": model_artifact["torch_cuda_device_name"],
                "runtime_device_note": model_artifact["runtime_device_note"],
                "estimator_module": model_artifact["estimator_module"],
                "estimator_class": model_artifact["estimator_class"],
                "cache_values_used": model_artifact["cache_values_used"],
                "has_method_robustness_value": model_artifact["has_method_robustness_value"],
                "has_method_sensitivity_interval": model_artifact["has_method_sensitivity_interval"],
                "has_method_sensitivity_summary": model_artifact["has_method_sensitivity_summary"],
                "has_method_summary": model_artifact["has_method_summary"],
                "has_attr_residuals": model_artifact["has_attr_residuals"],
                "saved_direct_rv": model_artifact["saved_direct_rv"],
                "saved_direct_rv_source": model_artifact["saved_direct_rv_source"],
                "saved_direct_rv_error": model_artifact["saved_direct_rv_error"],
                "saved_direct_sensitivity_interval": model_artifact[
                    "saved_direct_sensitivity_interval"
                ],
                "saved_direct_sensitivity_interval_source": model_artifact[
                    "saved_direct_sensitivity_interval_source"
                ],
                "saved_direct_sensitivity_interval_error": model_artifact[
                    "saved_direct_sensitivity_interval_error"
                ],
                "saved_direct_sensitivity_summary_source": model_artifact[
                    "saved_direct_sensitivity_summary_source"
                ],
                "saved_direct_sensitivity_summary_error": model_artifact[
                    "saved_direct_sensitivity_summary_error"
                ],
                "saved_direct_estimator_summary_source": model_artifact[
                    "saved_direct_estimator_summary_source"
                ],
                "saved_direct_estimator_summary_error": model_artifact[
                    "saved_direct_estimator_summary_error"
                ],
                "saved_sensitivity_params_available": model_artifact[
                    "saved_sensitivity_params_available"
                ],
                "saved_sensitivity_params_source": model_artifact[
                    "saved_sensitivity_params_source"
                ],
                "saved_sensitivity_params_error": model_artifact[
                    "saved_sensitivity_params_error"
                ],
                "saved_training_residuals_available": model_artifact[
                    "saved_training_residuals_available"
                ],
                "saved_training_residuals_source": model_artifact[
                    "saved_training_residuals_source"
                ],
                "saved_training_residuals_error": model_artifact[
                    "saved_training_residuals_error"
                ],
                "saved_training_residuals_tuple_length": model_artifact[
                    "saved_training_residuals_tuple_length"
                ],
                "latent_tags_path": model_artifact["latent_tags_path"],
                "physionet_pkl_path": model_artifact["physionet_pkl_path"],
                "graph_pkl_path": model_artifact["graph_pkl_path"],
                "output_dir": model_artifact["output_dir"],
                "cate_csv_path": cate_csv,
                "model_artifact_path": model_pkl,
            })

            print(f"Saved: {confounder_txt}")
            print(f"Saved: {summary_txt}")
            print(f"Saved: {cate_csv}")
            print(f"Saved: {model_pkl}")

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
    manager_global_summary_csv = os.path.join(
        OUTPUT_DIR,
        "manager_global_summary.csv",
    )
    control_messages_csv = os.path.join(
        OUTPUT_DIR,
        "control_messages_cate_estimation.csv",
    )

    if global_summary_rows:
        global_summary_df = finalize_ordered_dataframe(
            [build_clean_global_summary_row(row) for row in global_summary_rows],
            CLEAN_GLOBAL_SUMMARY_COLUMNS,
        )
        global_summary_df = global_summary_df.sort_values(
            by="mean_cate",
            ascending=False
        )
        global_summary_df.to_csv(global_summary_csv, index=False)
        print(f"\nSaved global summary: {global_summary_csv}")

        control_messages_df = finalize_ordered_dataframe(
            [build_control_global_summary_row(row) for row in global_summary_rows],
            CONTROL_GLOBAL_SUMMARY_COLUMNS,
        )
        control_messages_df["row_sort_order"] = control_messages_df["row_id"].map({
            row_id: idx for idx, row_id in enumerate(global_summary_df["row_id"].tolist())
        })
        control_messages_df = control_messages_df.sort_values(
            by="row_sort_order"
        ).drop(columns="row_sort_order")
        control_messages_df.to_csv(control_messages_csv, index=False)
        print(f"Saved control messages: {control_messages_csv}")

        manager_global_summary_df = global_summary_df[
            MANAGER_GLOBAL_SUMMARY_COLUMNS
        ].copy()
        manager_global_summary_df.to_csv(manager_global_summary_csv, index=False)
        print(f"Saved manager global summary: {manager_global_summary_csv}")
    print(
        f"[3/3] CATE estimation run finished. Successful treatment summaries: "
        f"{len(global_summary_rows)} / {len(TREATMENTS)}"
    )


if __name__ == "__main__":
    main()
