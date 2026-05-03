from __future__ import annotations

import argparse
import glob
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Sequence, Tuple

if "--validate-config-only" in sys.argv:
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/permutations_test.py",
        default_dataset="physionet",
    )

import numpy as np
import pandas as pd
from dataset_config import (
    get_config_float,
    get_config_int,
    get_config_scalar,
    get_first_available,
    load_dataset_config,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
CATE_ESTIMATION_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "cate_estimation.py")


# ============================================================
# Config
# ============================================================
DATASET_MODEL = "physionet"
TRIALS = 10
EXPERIMENT_DIR = os.path.join(DATA_ROOT, "relevant_outputs", "permutations_test")
LATENT_TAGS_PATH = os.path.join(DATA_ROOT, "predicted_latent_tags_230326_absolute_tags.csv")
PHYSIONET_PKL_PATH = os.path.join(DATA_ROOT, "processed", "physionet2012_ts_oc_ids.pkl")
GRAPH_PKL_PATH = os.path.join(DATA_ROOT, "causal_graph.pkl")
MODEL_TYPE = "CausalForest"
SEED = 42

OUTCOME_COL = "in_hospital_mortality"
EPSILON = 1e-12

SUMMARY_REQUIRED_COLUMNS = [
    "treatment",
    "model_type",
    "mean_cate",
    "mean_normalized_cate",
]

RESULT_COLUMNS = [
    "treatment",
    "model_type",
    "real_mean_cate",
    "real_mean_normalized_cate",
    "num_trials",
    "permutation_target",
    "permuted_mean_cate_mean",
    "permuted_mean_cate_std",
    "permuted_mean_normalized_cate_mean",
    "permuted_mean_normalized_cate_std",
    "snr_cate",
    "zscore_cate",
    "snr_normalized_cate",
    "zscore_normalized_cate",
    "seed",
    "experiment_dir",
    "warnings",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run treatment and outcome permutation sanity checks by repeatedly "
            "calling cate_estimation.py and aggregating run-level summaries."
        )
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
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=(
            "Number of permutation trials per experiment group. Default: use "
            f"script-level TRIALS ({TRIALS}) if set."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help=(
            "Directory for final permutation outputs. Default: use script-level "
            f"EXPERIMENT_DIR ({EXPERIMENT_DIR}) if set."
        ),
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
        "--model-type",
        choices=["LinearDML", "CausalForest"],
        default=None,
        help=(
            "Estimator family to use when calling cate_estimation.py. Default: "
            f"use script-level MODEL_TYPE ({MODEL_TYPE}) if set."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Base seed for deterministic shuffling. Default: use SEED ({SEED}).",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args()


def resolve_runtime_path(
    cli_value: str | None,
    global_value: str | None,
    field_name: str,
    *,
    must_exist: bool = True,
) -> str:
    raw_value = cli_value if cli_value is not None else global_value
    if raw_value is None:
        raise ValueError(
            f"{field_name} is not configured. Provide the matching CLI flag or set "
            f"the script-level {field_name}."
        )
    if not isinstance(raw_value, str):
        raise TypeError(f"{field_name} must be a string path. Got: {type(raw_value)!r}")

    raw_value = raw_value.strip()
    if not raw_value:
        raise ValueError(f"{field_name} is empty. Provide a non-empty path.")

    resolved_path = os.path.abspath(os.path.expanduser(raw_value))

    if must_exist:
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"{field_name} does not exist: {resolved_path}")
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"{field_name} is not a file: {resolved_path}")
    elif os.path.exists(resolved_path) and not os.path.isdir(resolved_path):
        raise NotADirectoryError(
            f"{field_name} must be a directory path: {resolved_path}"
        )

    return resolved_path


def resolve_runtime_int(
    cli_value: int | None,
    global_value: int | None,
    field_name: str,
    *,
    minimum: int = 1,
) -> int:
    value = cli_value if cli_value is not None else global_value
    if value is None:
        raise ValueError(
            f"{field_name} is not configured. Provide the matching CLI flag or set "
            f"the script-level {field_name}."
        )
    if int(value) < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}. Got: {value}")
    return int(value)


def resolve_runtime_choice(
    cli_value: str | None,
    global_value: str | None,
    field_name: str,
    valid_values: Iterable[str],
) -> str:
    value = cli_value if cli_value is not None else global_value
    if value is None:
        raise ValueError(
            f"{field_name} is not configured. Provide the matching CLI flag or set "
            f"the script-level {field_name}."
        )
    if value not in set(valid_values):
        raise ValueError(
            f"{field_name} must be one of {sorted(valid_values)}. Got: {value!r}"
        )
    return value


def build_trial_seed(
    base_seed: int,
    experiment_code: int,
    outer_index: int,
    trial_index: int,
) -> int:
    return int(base_seed + experiment_code * 10_000_000 + outer_index * 10_000 + trial_index)


def summarize_completed_process(process: subprocess.CompletedProcess[str]) -> str:
    lines: List[str] = []
    if process.stdout:
        lines.extend(process.stdout.strip().splitlines()[-10:])
    if process.stderr:
        stderr_lines = process.stderr.strip().splitlines()[-10:]
        if stderr_lines:
            lines.append("stderr:")
            lines.extend(stderr_lines)
    return "\n".join(lines)


def find_summary_csv(output_dir: str) -> str:
    candidates = sorted(
        glob.glob(os.path.join(output_dir, "*global_summary*.csv")),
        key=lambda path: ("manager_global_summary" in os.path.basename(path), path),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No global summary CSV found under output directory: {output_dir}"
        )

    required = set(SUMMARY_REQUIRED_COLUMNS)
    for path in candidates:
        header = pd.read_csv(path, nrows=0)
        if required.issubset(set(header.columns)):
            return path

    raise ValueError(
        "Could not find a summary CSV with the required columns "
        f"{SUMMARY_REQUIRED_COLUMNS} under {output_dir}. Checked: {candidates}"
    )


def extract_summary_metrics(summary_csv_path: str) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_csv_path, usecols=SUMMARY_REQUIRED_COLUMNS)
    if summary_df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_csv_path}")

    for column in ["mean_cate", "mean_normalized_cate"]:
        summary_df[column] = pd.to_numeric(summary_df[column], errors="raise")

    if summary_df["treatment"].duplicated().any():
        duplicated = summary_df.loc[
            summary_df["treatment"].duplicated(),
            "treatment",
        ].tolist()
        raise ValueError(
            f"Summary CSV contains duplicate treatment rows: {duplicated}. "
            f"Source: {summary_csv_path}"
        )

    return summary_df


def load_latent_tags_dataframe(latent_tags_path: str) -> pd.DataFrame:
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
    return latent_df


def run_cate_estimation_once(
    *,
    latent_tags_path: str,
    physionet_pkl_path: str,
    graph_pkl_path: str,
    output_dir: str,
    model_type: str,
    dataset_config_csv: str,
) -> Tuple[pd.DataFrame, str]:
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        CATE_ESTIMATION_SCRIPT_PATH,
        "--dataset-config-csv",
        dataset_config_csv,
        "--latent-tags-path",
        latent_tags_path,
        "--physionet-pkl-path",
        physionet_pkl_path,
        "--model",
        DATASET_MODEL,
        "--graph-pkl-path",
        graph_pkl_path,
        "--output-dir",
        output_dir,
        "--model-type",
        model_type,
    ]

    print(f"      Launching cate_estimation.py with output dir: {output_dir}")
    process = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "cate_estimation.py failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {process.returncode}\n"
            f"{summarize_completed_process(process)}"
        )

    summary_csv_path = find_summary_csv(output_dir)
    print(f"      Completed cate_estimation.py run. Summary CSV: {summary_csv_path}")
    return extract_summary_metrics(summary_csv_path), summary_csv_path


def build_treatment_metrics_map(
    summary_df: pd.DataFrame,
    *,
    context: str,
    expected_treatments: Sequence[str] | None = None,
) -> Dict[str, Dict[str, float | str]]:
    metrics_map: Dict[str, Dict[str, float | str]] = {}
    for row in summary_df.itertuples(index=False):
        mean_cate = float(row.mean_cate)
        mean_normalized_cate = float(row.mean_normalized_cate)
        if not np.isfinite(mean_cate) or not np.isfinite(mean_normalized_cate):
            raise ValueError(
                "Non-finite summary value found for "
                f"{row.treatment} in {context}: mean_cate={mean_cate}, "
                f"mean_normalized_cate={mean_normalized_cate}"
            )
        metrics_map[str(row.treatment)] = {
            "treatment": str(row.treatment),
            "model_type": str(row.model_type),
            "mean_cate": mean_cate,
            "mean_normalized_cate": mean_normalized_cate,
        }

    if expected_treatments is not None:
        missing = [t for t in expected_treatments if t not in metrics_map]
        if missing:
            raise ValueError(
                f"Missing expected treatment rows in {context}: {missing}"
            )

    return metrics_map


def shuffle_treatment_column(
    latent_df: pd.DataFrame,
    treatment: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if treatment not in latent_df.columns:
        raise ValueError(f"Treatment column not found in latent tags CSV: {treatment}")

    shuffled_df = latent_df.copy()
    shuffled_df[treatment] = rng.permutation(shuffled_df[treatment].to_numpy(copy=True))
    return shuffled_df


def shuffle_outcome_column(
    physionet_pkl_path: str,
    destination_pkl_path: str,
    rng: np.random.Generator,
) -> None:
    with open(physionet_pkl_path, "rb") as f:
        ts, oc, ts_ids = pickle.load(f)

    if OUTCOME_COL not in oc.columns:
        raise ValueError(
            f"Outcome column {OUTCOME_COL!r} was not found in the processed pickle: "
            f"{physionet_pkl_path}"
        )

    shuffled_oc = oc.copy()
    shuffled_oc[OUTCOME_COL] = rng.permutation(
        shuffled_oc[OUTCOME_COL].to_numpy(copy=True)
    )

    with open(destination_pkl_path, "wb") as f:
        pickle.dump((ts, shuffled_oc, ts_ids), f, protocol=pickle.HIGHEST_PROTOCOL)


def compute_permutation_metrics(
    real_effect: float,
    perm_values: Sequence[float],
) -> Dict[str, float | str]:
    values = np.asarray(perm_values, dtype=float)
    if values.size == 0:
        return {
            "perm_mean": np.nan,
            "perm_std": np.nan,
            "snr": np.nan,
            "zscore": np.nan,
            "warnings": "no_permutation_values",
        }

    perm_mean = float(np.mean(values))
    perm_std = float(np.std(values))
    mean_abs_perm = float(np.mean(np.abs(values)))

    warnings_list: List[str] = []
    if mean_abs_perm <= EPSILON:
        snr = np.nan
        warnings_list.append("snr_denominator_near_zero")
    else:
        snr = float(abs(real_effect) / mean_abs_perm)

    if perm_std <= EPSILON:
        zscore = np.nan
        warnings_list.append("perm_std_near_zero")
    else:
        zscore = float((real_effect - perm_mean) / perm_std)

    return {
        "perm_mean": perm_mean,
        "perm_std": perm_std,
        "snr": snr,
        "zscore": zscore,
        "warnings": ";".join(warnings_list),
    }


def join_warnings(*warning_values: str) -> str:
    seen: List[str] = []
    for warning_value in warning_values:
        if not warning_value:
            continue
        for item in warning_value.split(";"):
            if item and item not in seen:
                seen.append(item)
    return ";".join(seen)


def build_result_row(
    *,
    treatment: str,
    model_type: str,
    real_mean_cate: float,
    real_mean_normalized_cate: float,
    num_trials: int,
    permutation_target: str,
    cate_metrics: Dict[str, float | str],
    normalized_metrics: Dict[str, float | str],
    seed: int,
    experiment_dir: str,
) -> Dict[str, object]:
    return {
        "treatment": treatment,
        "model_type": model_type,
        "real_mean_cate": real_mean_cate,
        "real_mean_normalized_cate": real_mean_normalized_cate,
        "num_trials": num_trials,
        "permutation_target": permutation_target,
        "permuted_mean_cate_mean": cate_metrics["perm_mean"],
        "permuted_mean_cate_std": cate_metrics["perm_std"],
        "permuted_mean_normalized_cate_mean": normalized_metrics["perm_mean"],
        "permuted_mean_normalized_cate_std": normalized_metrics["perm_std"],
        "snr_cate": cate_metrics["snr"],
        "zscore_cate": cate_metrics["zscore"],
        "snr_normalized_cate": normalized_metrics["snr"],
        "zscore_normalized_cate": normalized_metrics["zscore"],
        "seed": seed,
        "experiment_dir": experiment_dir,
        "warnings": join_warnings(
            str(cate_metrics["warnings"]),
            str(normalized_metrics["warnings"]),
        ),
    }


def run_treatment_permutation_experiment(
    *,
    trials: int,
    experiment_dir: str,
    latent_tags_path: str,
    physionet_pkl_path: str,
    graph_pkl_path: str,
    model_type: str,
    dataset_config_csv: str,
    seed: int,
    baseline_metrics_map: Dict[str, Dict[str, float | str]],
    baseline_treatments: Sequence[str],
) -> pd.DataFrame:
    latent_df = load_latent_tags_dataframe(latent_tags_path)

    missing_treatment_cols = [t for t in baseline_treatments if t not in latent_df.columns]
    if missing_treatment_cols:
        raise ValueError(
            "Latent tags CSV is missing treatment columns required for the "
            f"permutation test: {missing_treatment_cols}"
        )

    results: List[Dict[str, object]] = []
    total_trial_runs = len(baseline_treatments) * trials

    for treatment_index, treatment in enumerate(baseline_treatments):
        print(f"[Treatment permutation] {treatment} ({treatment_index + 1}/{len(baseline_treatments)})")
        permuted_mean_cates: List[float] = []
        permuted_mean_normalized_cates: List[float] = []

        for trial_index in range(trials):
            overall_trial_index = treatment_index * trials + trial_index + 1
            print(
                f"  trial {trial_index + 1}/{trials} | overall run "
                f"{overall_trial_index}/{total_trial_runs}"
            )
            trial_seed = build_trial_seed(seed, 1, treatment_index, trial_index)
            rng = np.random.default_rng(trial_seed)

            with tempfile.TemporaryDirectory(
                prefix=f"treatment_perm_{treatment}_{trial_index:04d}_",
                dir=experiment_dir,
            ) as temp_dir:
                temp_latent_path = os.path.join(temp_dir, "latent_tags_permuted.csv")
                temp_output_dir = os.path.join(temp_dir, "cate_run")
                print(f"    temp latent CSV: {temp_latent_path}")

                shuffled_df = shuffle_treatment_column(latent_df, treatment, rng)
                shuffled_df.to_csv(temp_latent_path, index=False)

                trial_summary_df, summary_csv_path = run_cate_estimation_once(
                    latent_tags_path=temp_latent_path,
                    physionet_pkl_path=physionet_pkl_path,
                    graph_pkl_path=graph_pkl_path,
                    output_dir=temp_output_dir,
                    model_type=model_type,
                    dataset_config_csv=dataset_config_csv,
                )
                trial_metrics_map = build_treatment_metrics_map(
                    trial_summary_df,
                    context=(
                        f"treatment permutation summary for treatment={treatment}, "
                        f"trial={trial_index}"
                    ),
                    expected_treatments=[treatment],
                )
                permuted_mean_cates.append(float(trial_metrics_map[treatment]["mean_cate"]))
                permuted_mean_normalized_cates.append(
                    float(trial_metrics_map[treatment]["mean_normalized_cate"])
                )
                print(
                    f"    extracted metrics for {treatment}: "
                    f"mean_cate={trial_metrics_map[treatment]['mean_cate']:.6f} | "
                    f"mean_normalized_cate={trial_metrics_map[treatment]['mean_normalized_cate']:.6f}"
                )
                print(f"    metrics source CSV: {summary_csv_path}")
            print("    cleanup complete")

        real_metrics = baseline_metrics_map[treatment]
        cate_metrics = compute_permutation_metrics(
            float(real_metrics["mean_cate"]),
            permuted_mean_cates,
        )
        normalized_metrics = compute_permutation_metrics(
            float(real_metrics["mean_normalized_cate"]),
            permuted_mean_normalized_cates,
        )
        results.append(
            build_result_row(
                treatment=treatment,
                model_type=str(real_metrics["model_type"]),
                real_mean_cate=float(real_metrics["mean_cate"]),
                real_mean_normalized_cate=float(real_metrics["mean_normalized_cate"]),
                num_trials=trials,
                permutation_target="treatment",
                cate_metrics=cate_metrics,
                normalized_metrics=normalized_metrics,
                seed=seed,
                experiment_dir=experiment_dir,
            )
        )

    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def run_outcome_permutation_experiment(
    *,
    trials: int,
    experiment_dir: str,
    latent_tags_path: str,
    physionet_pkl_path: str,
    graph_pkl_path: str,
    model_type: str,
    dataset_config_csv: str,
    seed: int,
    baseline_metrics_map: Dict[str, Dict[str, float | str]],
    baseline_treatments: Sequence[str],
) -> pd.DataFrame:
    permuted_values_by_treatment = {
        treatment: {
            "mean_cate": [],
            "mean_normalized_cate": [],
        }
        for treatment in baseline_treatments
    }

    for trial_index in range(trials):
        print(f"[Outcome permutation] trial {trial_index + 1}/{trials}")
        trial_seed = build_trial_seed(seed, 2, 0, trial_index)
        rng = np.random.default_rng(trial_seed)

        with tempfile.TemporaryDirectory(
            prefix=f"outcome_perm_{trial_index:04d}_",
            dir=experiment_dir,
        ) as temp_dir:
            temp_physionet_pkl_path = os.path.join(temp_dir, "processed_outcome_permuted.pkl")
            temp_output_dir = os.path.join(temp_dir, "cate_run")
            print(f"    temp shuffled pickle: {temp_physionet_pkl_path}")

            shuffle_outcome_column(
                physionet_pkl_path=physionet_pkl_path,
                destination_pkl_path=temp_physionet_pkl_path,
                rng=rng,
            )

            trial_summary_df, summary_csv_path = run_cate_estimation_once(
                latent_tags_path=latent_tags_path,
                physionet_pkl_path=temp_physionet_pkl_path,
                graph_pkl_path=graph_pkl_path,
                output_dir=temp_output_dir,
                model_type=model_type,
                dataset_config_csv=dataset_config_csv,
            )
            trial_metrics_map = build_treatment_metrics_map(
                trial_summary_df,
                context=f"outcome permutation summary for trial={trial_index}",
                expected_treatments=baseline_treatments,
            )

            for treatment in baseline_treatments:
                permuted_values_by_treatment[treatment]["mean_cate"].append(
                    float(trial_metrics_map[treatment]["mean_cate"])
                )
                permuted_values_by_treatment[treatment]["mean_normalized_cate"].append(
                    float(trial_metrics_map[treatment]["mean_normalized_cate"])
                )
            print(f"    metrics source CSV: {summary_csv_path}")
        print("    cleanup complete")

    results: List[Dict[str, object]] = []
    for treatment in baseline_treatments:
        real_metrics = baseline_metrics_map[treatment]
        treatment_values = permuted_values_by_treatment[treatment]
        cate_metrics = compute_permutation_metrics(
            float(real_metrics["mean_cate"]),
            treatment_values["mean_cate"],
        )
        normalized_metrics = compute_permutation_metrics(
            float(real_metrics["mean_normalized_cate"]),
            treatment_values["mean_normalized_cate"],
        )
        results.append(
            build_result_row(
                treatment=treatment,
                model_type=str(real_metrics["model_type"]),
                real_mean_cate=float(real_metrics["mean_cate"]),
                real_mean_normalized_cate=float(real_metrics["mean_normalized_cate"]),
                num_trials=trials,
                permutation_target="outcome",
                cate_metrics=cate_metrics,
                normalized_metrics=normalized_metrics,
                seed=seed,
                experiment_dir=experiment_dir,
            )
        )

    return pd.DataFrame(results, columns=RESULT_COLUMNS)


def main() -> None:
    global DATASET_MODEL
    global OUTCOME_COL
    global EPSILON
    args = parse_args()
    DATASET_MODEL = args.model
    config = load_dataset_config(DATASET_MODEL, args.dataset_config_csv)
    dataset_config_csv = str(config["__config_csv_path__"])

    OUTCOME_COL = str(get_config_scalar(config, "OUTCOME_COL", OUTCOME_COL))
    EPSILON = float(get_config_float(config, "EPSILON", EPSILON) or EPSILON)

    trials = resolve_runtime_int(
        args.trials,
        get_config_int(config, "TRIALS", TRIALS),
        "TRIALS",
        minimum=1,
    )
    experiment_dir_default = get_first_available(
        config,
        ["EXPERIMENT_DIR"],
        EXPERIMENT_DIR,
    )
    experiment_dir = resolve_runtime_path(
        args.experiment_dir,
        str(experiment_dir_default),
        "EXPERIMENT_DIR",
        must_exist=False,
    )
    latent_tags_default = get_first_available(
        config,
        ["PERMUTATIONS_LATENT_TAGS_PATH", "LATENT_TAGS_PATH"],
        LATENT_TAGS_PATH,
    )
    physionet_pkl_default = get_first_available(
        config,
        ["PERMUTATIONS_PKL_PATH", "DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
        PHYSIONET_PKL_PATH,
    )
    graph_pkl_default = get_first_available(config, ["GRAPH_PKL_PATH"], GRAPH_PKL_PATH)
    latent_tags_path = resolve_runtime_path(
        args.latent_tags_path,
        str(latent_tags_default),
        "LATENT_TAGS_PATH",
    )
    physionet_pkl_path = resolve_runtime_path(
        args.physionet_pkl_path,
        str(physionet_pkl_default),
        "PHYSIONET_PKL_PATH",
    )
    graph_pkl_path = resolve_runtime_path(
        args.graph_pkl_path,
        graph_pkl_default,
        "GRAPH_PKL_PATH",
    )
    model_type = resolve_runtime_choice(
        args.model_type,
        str(get_config_scalar(config, "MODEL_TYPE", MODEL_TYPE)),
        "MODEL_TYPE",
        valid_values=["LinearDML", "CausalForest"],
    )
    seed = resolve_runtime_int(
        args.seed,
        get_config_int(config, "SEED", SEED),
        "SEED",
        minimum=0,
    )

    if not os.path.isfile(CATE_ESTIMATION_SCRIPT_PATH):
        raise FileNotFoundError(
            f"cate_estimation.py was not found at {CATE_ESTIMATION_SCRIPT_PATH}"
        )

    os.makedirs(experiment_dir, exist_ok=True)
    print("=== Starting permutation sanity checks ===")
    print(
        "Runtime configuration: "
        f"model={DATASET_MODEL} | trials={trials} | model_type={model_type} | seed={seed} | "
        f"latent_tags_path={latent_tags_path} | processed_pkl_path={physionet_pkl_path} | "
        f"graph_pkl_path={graph_pkl_path} | experiment_dir={experiment_dir}"
    )

    print("Running baseline cate_estimation.py on original inputs...")
    with tempfile.TemporaryDirectory(
        prefix="baseline_perm_",
        dir=experiment_dir,
    ) as temp_dir:
        baseline_output_dir = os.path.join(temp_dir, "cate_run")
        baseline_summary_df, baseline_summary_csv = run_cate_estimation_once(
            latent_tags_path=latent_tags_path,
            physionet_pkl_path=physionet_pkl_path,
            graph_pkl_path=graph_pkl_path,
            output_dir=baseline_output_dir,
            model_type=model_type,
            dataset_config_csv=dataset_config_csv,
        )

        baseline_treatments = baseline_summary_df["treatment"].astype(str).tolist()
        baseline_metrics_map = build_treatment_metrics_map(
            baseline_summary_df,
            context=f"baseline summary from {baseline_summary_csv}",
            expected_treatments=baseline_treatments,
        )
        print(
            f"Baseline completed: treatments={len(baseline_treatments)} | "
            f"summary_csv={baseline_summary_csv}"
        )

    print("Running treatment permutation tests...")
    treatment_results_df = run_treatment_permutation_experiment(
        trials=trials,
        experiment_dir=experiment_dir,
        latent_tags_path=latent_tags_path,
        physionet_pkl_path=physionet_pkl_path,
        graph_pkl_path=graph_pkl_path,
        model_type=model_type,
        dataset_config_csv=dataset_config_csv,
        seed=seed,
        baseline_metrics_map=baseline_metrics_map,
        baseline_treatments=baseline_treatments,
    )

    print("Running outcome permutation tests...")
    outcome_results_df = run_outcome_permutation_experiment(
        trials=trials,
        experiment_dir=experiment_dir,
        latent_tags_path=latent_tags_path,
        physionet_pkl_path=physionet_pkl_path,
        graph_pkl_path=graph_pkl_path,
        model_type=model_type,
        dataset_config_csv=dataset_config_csv,
        seed=seed,
        baseline_metrics_map=baseline_metrics_map,
        baseline_treatments=baseline_treatments,
    )

    treatment_results_path = os.path.join(
        experiment_dir,
        "treatment_permutation_results.csv",
    )
    outcome_results_path = os.path.join(
        experiment_dir,
        "outcome_permutation_results.csv",
    )

    treatment_results_df.to_csv(treatment_results_path, index=False)
    outcome_results_df.to_csv(outcome_results_path, index=False)

    print(f"Saved treatment permutation results: {treatment_results_path}")
    print(f"Saved outcome permutation results: {outcome_results_path}")
    print("Done.")


if __name__ == "__main__":
    main()
