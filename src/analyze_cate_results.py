from __future__ import annotations

import csv
import pickle
import sys
import traceback
import types
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


SCRIPT_DIR = Path(__file__).resolve().parent

LATENT_TAGS_PATH = "../../data/predicted_latent_tags_230326_absolute_tags.csv"
PHYSIONET_PKL_PATH = "../../data/processed/physionet2012_ts_oc_ids.pkl"
CATE_RESULTS_DIR = "../../data/relevant_outputs/cate_outputs_predicted_230326"

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

BENCHMARK_SCORE_COLUMNS = [
    "rank",
    "confounder",
    "cf_y",
    "cf_d",
    "strength_score",
    "selected_as_benchmark",
    "is_primary_benchmark",
]

RUN_SUMMARY_COLUMNS = [
    "treatment",
    "model_type",
    "RV",
    "selected_benchmark_confounder",
    "benchmark_cf_y",
    "benchmark_cf_d",
    "benchmark_strength_score",
    "robustness_ratio",
    "contour_plot_path",
    "run_status",
    "report_path",
    "benchmark_scores_path",
    "analysis_rows",
    "residual_rows",
    "warnings",
]

SensitivityParams = namedtuple("SensitivityParams", ["theta", "sigma", "nu", "cov"])


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


def install_sensitivity_pickle_shim() -> None:
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


def build_background_features(ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy().sort_values(["ts_id", "minute"])

    keep_vars = [
        "Age", "Gender", "Weight",
        "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4",
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
    latent_tags_path: Path,
    physionet_pkl_path: Path,
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


def load_model_artifact(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    install_sensitivity_pickle_shim()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(path, "rb") as f:
            artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise TypeError(f"Model artifact is not a dict: {path}")

    warning_messages = []
    for item in caught:
        warning_messages.append(str(item.message))
    return artifact, dedupe_preserve_order(warning_messages)


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

    return {
        "estimator": artifact["estimator"],
        "model_type": str(artifact["model_type"]),
        "treatment": str(artifact["treatment"]),
        "outcome_col": str(artifact["outcome_col"]),
        "confounders": list(artifact.get("confounders", [])),
        "effect_modifiers": list(artifact.get("effect_modifiers", [])),
        "feature_fill_values": dict(artifact.get("feature_fill_values", {})),
        "formula": str(artifact.get("formula", "")),
        "summary": dict(artifact.get("summary", {})),
    }


def prepare_treatment_matrices_from_artifact(
    df: pd.DataFrame,
    model_artifact: Dict[str, Any],
) -> Dict[str, Any]:
    treatment = model_artifact["treatment"]
    outcome_col = model_artifact["outcome_col"]
    confounders = list(model_artifact["confounders"])
    effect_modifiers = list(model_artifact["effect_modifiers"])
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

    lines = [
        (
            "Fallback sensitivity summary from residual-space params "
            f"(c_y={c_y}, c_t={c_t}, rho={rho}, alpha={alpha})"
        ),
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


def extract_sensitivity_outputs(
    est: Any,
    sensitivity_params: SensitivityParams | None = None,
) -> Dict[str, Any]:
    rv, rv_error = try_method_calls(
        est,
        "robustness_value",
        [{}, {"null_hypothesis": 0.0, "alpha": DEFAULT_SENSITIVITY_ALPHA}],
    )
    summary_obj, summary_error = try_method_calls(
        est,
        "sensitivity_summary",
        [{}, {"null_hypothesis": 0.0, "alpha": DEFAULT_SENSITIVITY_ALPHA}],
    )
    interval_obj, interval_error = try_method_calls(
        est,
        "sensitivity_interval",
        [{}, {"alpha": DEFAULT_SENSITIVITY_ALPHA, "interval_type": "ci"}],
    )

    out = {
        "rv": None,
        "rv_theta": None,
        "rv_error": rv_error,
        "summary_text": str(summary_obj) if summary_obj is not None else "",
        "summary_error": summary_error,
        "interval": None,
        "interval_error": interval_error,
        "source": "official_api",
    }

    if rv is not None:
        try:
            out["rv"] = coerce_float(rv)
        except Exception as exc:
            out["rv_error"] = format_exception(exc)

    if interval_obj is not None:
        try:
            lb, ub = interval_obj
            out["interval"] = (coerce_float(lb), coerce_float(ub))
        except Exception as exc:
            out["interval_error"] = format_exception(exc)

    if sensitivity_params is None:
        return out

    fallback_interval = sensitivity_interval_from_params(
        params=sensitivity_params,
        alpha=DEFAULT_SENSITIVITY_ALPHA,
        c_y=DEFAULT_SENSITIVITY_C_Y,
        c_t=DEFAULT_SENSITIVITY_C_T,
        rho=DEFAULT_SENSITIVITY_RHO,
        interval_type="ci",
    )
    fallback_rv = robustness_value_from_params(
        params=sensitivity_params,
        alpha=DEFAULT_SENSITIVITY_ALPHA,
        null_hypothesis=0.0,
        interval_type="ci",
    )
    fallback_rv_theta = robustness_value_from_params(
        params=sensitivity_params,
        alpha=DEFAULT_SENSITIVITY_ALPHA,
        null_hypothesis=0.0,
        interval_type="theta",
    )
    fallback_summary = build_sensitivity_summary_text(sensitivity_params)

    if out["rv"] is None:
        out["rv"] = fallback_rv
        out["rv_error"] = None
        out["source"] = "residual_space_fallback"

    if out["interval"] is None:
        out["interval"] = fallback_interval
        out["interval_error"] = None
        out["source"] = "residual_space_fallback"

    if not out["summary_text"]:
        out["summary_text"] = fallback_summary
        out["summary_error"] = None
        out["source"] = "residual_space_fallback"

    out["rv_theta"] = fallback_rv_theta
    return out


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
        cf_y = single_feature_r_squared(z, y_res)
        cf_d = single_feature_r_squared(z, t_res)
        rows.append({
            "confounder": confounder,
            "cf_y": cf_y,
            "cf_d": cf_d,
            "strength_score": cf_y * cf_d,
        })

    rows.sort(
        key=lambda row: (
            -row["strength_score"],
            -row["cf_y"],
            -row["cf_d"],
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
        new_row["selected_as_benchmark"] = row["confounder"] in selected_names
        new_row["is_primary_benchmark"] = row["confounder"] == primary_name
        annotated_rows.append(new_row)

    return {
        "all_rows": annotated_rows,
        "selected_rows": selected_rows,
        "primary_row": primary_row,
        "aggregation_rule": (
            "Primary benchmark is the selected confounder with the largest "
            "strength_score among the top-k selected confounders."
        ),
    }


def compute_robustness_ratio(
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


def sensitivity_margin(lb: float | None, ub: float | None) -> float:
    if lb is None or ub is None:
        return float("nan")
    if lb > 0:
        return float(lb)
    if ub < 0:
        return float(-ub)
    return float(-min(ub, -lb))


def save_sensitivity_contour(
    sensitivity_params: SensitivityParams | None,
    treatment_dir: Path,
    treatment: str,
    benchmark_rows: Sequence[Dict[str, Any]],
) -> Tuple[str | None, List[str]]:
    notes: List[str] = []

    if not SAVE_CONTOUR_PLOT:
        notes.append("Contour plotting disabled by SAVE_CONTOUR_PLOT=False.")
        return None, notes
    if sensitivity_params is None:
        notes.append("Contour plot unavailable because sensitivity parameters were unavailable.")
        return None, notes

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
        notes.append("Contour plot unavailable because the sensitivity grid could not be evaluated.")
        return None, notes

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
            row["cf_d"],
            row["cf_y"],
            marker=marker,
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{idx}. {row['confounder']}",
            (row["cf_d"], row["cf_y"]),
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
        "Contour plot used a fallback matplotlib grid from residual-space sensitivity parameters."
    )
    return str(output_path), notes


def write_benchmark_report(path: Path, report_data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Benchmark-Based CATE Sensitivity Report ===\n\n")
        f.write(f"Run status: {report_data.get('run_status', 'UNKNOWN')}\n")
        f.write(f"Treatment: {report_data.get('treatment', '')}\n")
        f.write(f"Model type: {report_data.get('model_type', '')}\n")
        f.write(f"Estimator class (loaded artifact): {report_data.get('loaded_estimator_class', '')}\n")
        f.write(f"Benchmark estimator source: {report_data.get('benchmark_estimator_source', '')}\n")
        f.write(f"Sensitivity source: {report_data.get('sensitivity_source', '')}\n")
        f.write(f"Model artifact: {report_data.get('model_artifact_path', '')}\n")
        f.write(f"Analysis dataframe source: {report_data.get('analysis_dataframe_paths', '')}\n\n")

        if report_data.get("error"):
            f.write("Failure / primary error:\n")
            f.write(report_data["error"] + "\n\n")

        if report_data.get("traceback"):
            f.write("Traceback:\n")
            f.write(report_data["traceback"] + "\n")

        f.write("Artifact metadata:\n")
        f.write(f"Outcome column: {report_data.get('outcome_col', '')}\n")
        f.write(f"Confounders: {report_data.get('confounders', [])}\n")
        f.write(f"Effect modifiers: {report_data.get('effect_modifiers', [])}\n")
        f.write(f"Formula:\n{report_data.get('formula', '')}\n\n")

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

        f.write("Sensitivity outputs:\n")
        f.write(f"RV: {report_data.get('rv', '')}\n")
        f.write(f"RV (theta): {report_data.get('rv_theta', '')}\n")
        f.write(f"Default sensitivity interval: {report_data.get('sensitivity_interval', '')}\n")
        f.write(
            "Robustness ratio (RV / max(primary cf_y, primary cf_d)): "
            f"{report_data.get('robustness_ratio', '')}\n"
        )
        f.write(f"Contour plot path: {report_data.get('contour_plot_path', '')}\n\n")

        if report_data.get("sensitivity_summary_text"):
            f.write("Sensitivity summary:\n")
            f.write(report_data["sensitivity_summary_text"] + "\n\n")

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
            f.write("Per-confounder benchmark scores:\n")
            f.write("rank | confounder | cf_y | cf_d | strength_score\n")
            for row in scores_rows:
                f.write(
                    f"{row['rank']} | {row['confounder']} | "
                    f"{row['cf_y']:.6f} | {row['cf_d']:.6f} | "
                    f"{row['strength_score']:.6f}\n"
                )
            f.write("\n")
        else:
            f.write("Per-confounder benchmark scores:\nNone\n\n")

        selected_rows = list(report_data.get("selected_rows", []))
        if selected_rows:
            f.write("Selected benchmark confounders:\n")
            for row in selected_rows:
                f.write(
                    f"- {row['confounder']} "
                    f"(cf_y={row['cf_y']:.6f}, cf_d={row['cf_d']:.6f}, "
                    f"strength_score={row['strength_score']:.6f})\n"
                )
            f.write(
                "\nAggregation rule for benchmark ratio:\n"
                f"{report_data.get('aggregation_rule', '')}\n\n"
            )
        else:
            f.write("Selected benchmark confounders:\nNone\n\n")

        warnings_list = list(report_data.get("warnings", []))
        if warnings_list:
            f.write("Warnings / fallbacks / unavailable APIs:\n")
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
        "treatment": treatment,
        "model_type": "",
        "RV": None,
        "selected_benchmark_confounder": "",
        "benchmark_cf_y": None,
        "benchmark_cf_d": None,
        "benchmark_strength_score": None,
        "robustness_ratio": None,
        "contour_plot_path": "",
        "run_status": "FAILED",
        "report_path": str(report_path),
        "benchmark_scores_path": str(scores_path),
        "analysis_rows": None,
        "residual_rows": None,
        "warnings": "",
    }


def analyze_one_treatment(
    treatment_dir: Path,
    analysis_df: pd.DataFrame | None,
    analysis_df_error: str | None,
    latent_tags_path: Path,
    physionet_pkl_path: Path,
) -> Dict[str, Any]:
    treatment_hint = treatment_dir.name
    model_path = treatment_dir / f"{treatment_hint}_model.pkl"
    report_path = build_treatment_output_path(treatment_dir, treatment_hint, "benchmark_report", "txt")
    scores_path = build_treatment_output_path(treatment_dir, treatment_hint, "benchmark_scores", "csv")

    summary_row = empty_summary_row(treatment_hint, report_path, scores_path)
    warnings_list: List[str] = []
    report_data: Dict[str, Any] = {
        "run_status": "FAILED",
        "treatment": treatment_hint,
        "model_type": "",
        "loaded_estimator_class": "",
        "benchmark_estimator_source": "",
        "sensitivity_source": "",
        "model_artifact_path": str(model_path),
        "analysis_dataframe_paths": f"latent_tags={latent_tags_path}; physionet={physionet_pkl_path}",
        "outcome_col": "",
        "confounders": [],
        "effect_modifiers": [],
        "formula": "",
        "analysis_summary": {},
        "rv": None,
        "rv_theta": None,
        "sensitivity_interval": None,
        "sensitivity_summary_text": "",
        "robustness_ratio": None,
        "contour_plot_path": "",
        "scores_rows": [],
        "selected_rows": [],
        "aggregation_rule": "",
        "warnings": warnings_list,
        "training_summary": {},
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
            "loaded_estimator_class": type(artifact["estimator"]).__name__,
            "outcome_col": artifact["outcome_col"],
            "confounders": artifact["confounders"],
            "effect_modifiers": artifact["effect_modifiers"],
            "formula": artifact["formula"],
            "training_summary": artifact["summary"],
        })

        prepared = None
        if analysis_df is None:
            warnings_list.append(
                "Analysis dataframe was unavailable, so treatment matrices could not be reconstructed: "
                f"{analysis_df_error}"
            )
        else:
            prepared = prepare_treatment_matrices_from_artifact(analysis_df, artifact)
            report_data["analysis_summary"] = prepared
            summary_row["analysis_rows"] = prepared["analysis_rows"]

        benchmark_estimator = artifact["estimator"]
        benchmark_estimator_source = "loaded_artifact"

        residual_info = None
        try:
            residual_info = extract_dml_residuals(benchmark_estimator)
        except Exception as exc:
            warnings_list.append(
                f"Loaded estimator residuals_ unavailable: {format_exception(exc)}"
            )

        if residual_info is None and prepared is not None:
            warnings_list.append(
                "Re-fitting a compatibility estimator with cache_values=True to recover residual-space diagnostics."
            )
            benchmark_estimator = fit_compatibility_estimator(
                model_type=artifact["model_type"],
                Y=prepared["Y"],
                T=prepared["T"],
                W=prepared["W"],
                X=prepared["X"],
            )
            benchmark_estimator_source = "compatibility_refit"
            residual_info = extract_dml_residuals(benchmark_estimator)

        report_data["benchmark_estimator_source"] = benchmark_estimator_source

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
                confounder_names=artifact["confounders"],
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

        sensitivity_outputs = extract_sensitivity_outputs(
            est=artifact["estimator"],
            sensitivity_params=sensitivity_params,
        )
        report_data["sensitivity_source"] = sensitivity_outputs["source"]
        report_data["rv"] = sensitivity_outputs["rv"]
        report_data["rv_theta"] = sensitivity_outputs["rv_theta"]
        report_data["sensitivity_interval"] = sensitivity_outputs["interval"]
        report_data["sensitivity_summary_text"] = sensitivity_outputs["summary_text"]

        if sensitivity_outputs["rv_error"]:
            warnings_list.append(
                f"robustness_value() unavailable on loaded estimator: {sensitivity_outputs['rv_error']}"
            )
        if sensitivity_outputs["summary_error"]:
            warnings_list.append(
                f"sensitivity_summary() unavailable on loaded estimator: {sensitivity_outputs['summary_error']}"
            )
        if sensitivity_outputs["interval_error"]:
            warnings_list.append(
                "sensitivity_interval() unavailable on loaded estimator: "
                f"{sensitivity_outputs['interval_error']}"
            )

        contour_plot_path, contour_notes = save_sensitivity_contour(
            sensitivity_params=sensitivity_params,
            treatment_dir=treatment_dir,
            treatment=treatment,
            benchmark_rows=selected_rows,
        )
        warnings_list.extend(contour_notes)
        report_data["contour_plot_path"] = contour_plot_path or ""
        summary_row["contour_plot_path"] = contour_plot_path or ""

        benchmark_cf_y = primary_row["cf_y"] if primary_row else None
        benchmark_cf_d = primary_row["cf_d"] if primary_row else None
        benchmark_strength = primary_row["strength_score"] if primary_row else None

        robustness_ratio = compute_robustness_ratio(
            rv=sensitivity_outputs["rv"],
            cf_y=benchmark_cf_y,
            cf_d=benchmark_cf_d,
        )
        report_data["robustness_ratio"] = robustness_ratio

        run_status = "SUCCESS"
        if primary_row is None or sensitivity_outputs["rv"] is None:
            run_status = "PARTIAL"
        if residual_info is None and sensitivity_outputs["rv"] is None:
            run_status = "FAILED"

        report_data["run_status"] = run_status
        summary_row.update({
            "treatment": treatment,
            "model_type": artifact["model_type"],
            "RV": sensitivity_outputs["rv"],
            "selected_benchmark_confounder": primary_row["confounder"] if primary_row else "",
            "benchmark_cf_y": benchmark_cf_y,
            "benchmark_cf_d": benchmark_cf_d,
            "benchmark_strength_score": benchmark_strength,
            "robustness_ratio": robustness_ratio,
            "run_status": run_status,
            "warnings": " | ".join(warnings_list),
        })

    except Exception as exc:
        report_data["error"] = format_exception(exc)
        report_data["traceback"] = traceback.format_exc()
        warnings_list.append(report_data["error"])
        summary_row["warnings"] = " | ".join(warnings_list)

    finally:
        report_data["warnings"] = warnings_list
        report_data["scores_rows"] = scores_rows
        write_rows_to_csv(scores_path, scores_rows, BENCHMARK_SCORE_COLUMNS)
        write_benchmark_report(report_path, report_data)
        summary_row["report_path"] = str(report_path)
        summary_row["benchmark_scores_path"] = str(scores_path)

    return summary_row


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.random.seed(SEED)

    if TOP_K_BENCHMARK_CONFOUNDERS not in ALLOWED_TOP_K_VALUES:
        raise ValueError(
            f"TOP_K_BENCHMARK_CONFOUNDERS must be 1 or 3. Found: {TOP_K_BENCHMARK_CONFOUNDERS}"
        )

    results_dir = resolve_script_path(CATE_RESULTS_DIR)
    latent_tags_path = resolve_script_path(LATENT_TAGS_PATH)
    physionet_pkl_path = resolve_script_path(PHYSIONET_PKL_PATH)

    treatment_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    run_summary_csv = build_run_output_csv(results_dir, "benchmark_summary")

    analysis_df = None
    analysis_df_error = None
    try:
        analysis_df = load_analysis_dataframe(latent_tags_path, physionet_pkl_path)
    except Exception as exc:
        analysis_df_error = format_exception(exc)

    summary_rows: List[Dict[str, Any]] = []
    success_count = 0
    partial_count = 0
    failed_count = 0

    for treatment_dir in treatment_dirs:
        summary_row = analyze_one_treatment(
            treatment_dir=treatment_dir,
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
    write_rows_to_csv(run_summary_csv, summary_rows, RUN_SUMMARY_COLUMNS)

    print(f"Treatments processed: {len(summary_rows)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {failed_count}")
    if partial_count:
        print(f"Partial: {partial_count}")
    print(f"Run-level summary CSV: {run_summary_csv}")


if __name__ == "__main__":
    main()
