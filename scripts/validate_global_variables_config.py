#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_config import (  # noqa: E402
    KNOWN_BOOL_KEYS,
    KNOWN_FLOAT_KEYS,
    KNOWN_INT_KEYS,
    KNOWN_LIST_KEYS,
    MIMIC_ONLY_KEYS,
    REQUIRED_CANONICAL_KEYS,
    SCRIPT_CONFIG_CONTRACTS,
    get_config_bool,
    get_config_float,
    get_config_int,
    load_dataset_config,
    validate_script_config,
)


CONFIGS = {
    "physionet": REPO_ROOT / "configs" / "physionet-global-variables.csv",
    "mimic": REPO_ROOT / "configs" / "mimic-global-variables.csv",
}

DATASET_SCRIPTS = {
    "physionet": [
        "main.py",
        "src/cate_estimation.py",
        "src/matching_causal_effect.py",
        "src/mortality_prediction_using_latents.py",
        "src/analyze_cate_results.py",
        "src/permutations_test.py",
        "src/split_predicted_latent_tags.py",
        "src/physionet2012_causal_graph.py",
        "src/tagging_latent_variables_physionet.py",
        "src/preprocess_physionet_2012.py",
    ],
    "mimic": [
        "main.py",
        "src/cate_estimation.py",
        "src/matching_causal_effect.py",
        "src/mortality_prediction_using_latents.py",
        "src/analyze_cate_results.py",
        "src/permutations_test.py",
        "src/split_predicted_latent_tags.py",
        "src/mimiciii_causal_graph.py",
        "src/tagging_latent_variables_mimiciii.py",
        "src/preprocess_mimic_iii_large.py",
    ],
}

EMPTY_ALLOWED_CANONICAL = {
    "ALT_ID_COL",
    "PHYSIONET_SET_NAMES",
    "THRESHOLDS_PATH",
}

SUSPICIOUS_DUPLICATE_GROUPS = [
    [
        "LATENT_TAGS_PATH",
        "CATE_LATENT_TAGS_PATH",
        "MATCHING_LATENT_TAGS_PATH",
        "MORTALITY_LATENT_TAGS_PATH",
        "ANALYZE_LATENT_TAGS_PATH",
        "PERMUTATIONS_LATENT_TAGS_PATH",
    ],
    [
        "PHYSIONET_PKL_PATH",
        "DATASET_PKL_PATH",
        "CATE_PKL_PATH",
        "MATCHING_PKL_PATH",
        "MORTALITY_PKL_PATH",
        "ANALYZE_PKL_PATH",
        "PERMUTATIONS_PKL_PATH",
    ],
    [
        "OUTPUT_DIR",
        "CATE_OUTPUT_DIR",
        "MATCHING_OUTPUT_DIR",
        "ANALYZE_OUTPUT_DIR",
    ],
    ["RESULTS_TXT_PATH", "MORTALITY_RESULTS_TXT_PATH"],
    ["GRAPH_PKL_PATH", "DEFAULT_GRAPH_PKL_PATH"],
]

ALLOWED_SAME_VALUE_PAIRS = {
    frozenset(("GRAPH_PKL_PATH", "DEFAULT_GRAPH_PKL_PATH")),
}


def read_raw_csv(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row")
        fieldnames = [field.strip() for field in reader.fieldnames if field]
        values = {key: [] for key in fieldnames}
        for row in reader:
            for key in fieldnames:
                value = (row.get(key) or "").strip()
                if value:
                    values[key].append(value)
    return fieldnames, values


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(has_value(item) for item in value)
    return True


def normalized_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip()


def validate_required_canonical_keys(
    dataset: str,
    fieldnames: list[str],
    config: dict[str, object],
) -> list[str]:
    missing = [key for key in REQUIRED_CANONICAL_KEYS if key not in fieldnames]
    empty = [
        key
        for key in REQUIRED_CANONICAL_KEYS
        if key in fieldnames
        and key not in EMPTY_ALLOWED_CANONICAL
        and not has_value(config.get(key))
    ]
    if dataset != "mimic":
        empty = [key for key in empty if key not in MIMIC_ONLY_KEYS]
    return [f"missing:{key}" for key in missing] + [f"empty:{key}" for key in empty]


def validate_dataset_specific_keys(
    dataset: str,
    fieldnames: list[str],
    config: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    if dataset == "mimic":
        for key in MIMIC_ONLY_KEYS:
            if key not in fieldnames:
                errors.append(f"missing mimic-only:{key}")
            elif not has_value(config.get(key)):
                errors.append(f"empty mimic-only:{key}")
    else:
        present = [key for key in MIMIC_ONLY_KEYS if key in fieldnames and has_value(config.get(key))]
        if present:
            errors.append("physionet has mimic-only keys:" + ",".join(present))
    return errors


def validate_typed_values(config: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for key in sorted(KNOWN_BOOL_KEYS):
        if has_value(config.get(key)):
            try:
                get_config_bool(config, key)
            except Exception as exc:
                errors.append(f"{key}:bool:{exc}")
    for key in sorted(KNOWN_INT_KEYS):
        if has_value(config.get(key)):
            try:
                get_config_int(config, key)
            except Exception as exc:
                errors.append(f"{key}:int:{exc}")
    for key in sorted(KNOWN_FLOAT_KEYS):
        if has_value(config.get(key)):
            try:
                get_config_float(config, key)
            except Exception as exc:
                errors.append(f"{key}:float:{exc}")
    for key in sorted(KNOWN_LIST_KEYS):
        value = config.get(key)
        if has_value(value) and not isinstance(value, list):
            errors.append(f"{key}:list:expected list, got {type(value).__name__}")
    return errors


def validate_non_empty_contract_lists(
    script: str,
    config: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    contract = SCRIPT_CONFIG_CONTRACTS[script]
    for key in contract.get("required_keys", []):
        if key in KNOWN_LIST_KEYS and not has_value(config.get(str(key))):
            errors.append(str(key))
    if errors:
        return [f"empty required list:{','.join(errors)}"]
    return []


def validate_duplicate_policy(
    fieldnames: list[str],
    config: dict[str, object],
    docs_text: str,
) -> list[str]:
    errors: list[str] = []
    for group in SUSPICIOUS_DUPLICATE_GROUPS:
        present = [key for key in group if key in fieldnames and has_value(config.get(key))]
        for left_index, left in enumerate(present):
            for right in present[left_index + 1 :]:
                pair = frozenset((left, right))
                if normalized_value(config[left]) != normalized_value(config[right]):
                    continue
                if pair in ALLOWED_SAME_VALUE_PAIRS and left in docs_text and right in docs_text:
                    continue
                errors.append(f"{left}={right}")
    return errors


def format_resolved_aliases(resolved: dict[str, dict[str, object]]) -> str:
    parts = []
    for name in sorted(resolved):
        item = resolved[name]
        candidate_keys = list(item.get("candidate_keys", []))
        if len(candidate_keys) > 1:
            parts.append(f"{name}->{item.get('source_key')}")
    return "; ".join(parts) if parts else "-"


def print_table(rows: list[dict[str, str]]) -> None:
    headers = ["dataset", "script", "status", "missing_keys", "resolved_aliases"]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def main() -> int:
    docs_path = REPO_ROOT / "docs" / "global-variables-parameters.txt"
    docs_text = docs_path.read_text(encoding="utf-8") if docs_path.exists() else ""

    rows: list[dict[str, str]] = []
    global_errors: list[str] = []

    for dataset, path in CONFIGS.items():
        fieldnames, _ = read_raw_csv(path)
        try:
            config = load_dataset_config(dataset, path)
        except Exception as exc:
            global_errors.append(f"{dataset}: load failed: {type(exc).__name__}: {exc}")
            config = {"__dataset__": dataset, "__config_csv_path__": str(path)}

        global_errors.extend(
            f"{dataset}: {error}"
            for error in validate_required_canonical_keys(dataset, fieldnames, config)
        )
        global_errors.extend(
            f"{dataset}: {error}"
            for error in validate_dataset_specific_keys(dataset, fieldnames, config)
        )
        global_errors.extend(f"{dataset}: {error}" for error in validate_typed_values(config))
        global_errors.extend(
            f"{dataset}: duplicate:{error}"
            for error in validate_duplicate_policy(fieldnames, config, docs_text)
        )

        for script in DATASET_SCRIPTS[dataset]:
            status = "PASS"
            missing = "-"
            aliases = "-"
            try:
                resolved = validate_script_config(script, config)
                contract_list_errors = validate_non_empty_contract_lists(script, config)
                if contract_list_errors:
                    raise ValueError("; ".join(contract_list_errors))
                aliases = format_resolved_aliases(resolved)
            except Exception as exc:
                status = "FAIL"
                missing = str(exc)
                global_errors.append(f"{dataset} {script}: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "dataset": dataset,
                    "script": script,
                    "status": status,
                    "missing_keys": missing,
                    "resolved_aliases": aliases,
                }
            )

    print_table(rows)
    if global_errors:
        print("\nConfig validation errors:")
        for error in global_errors:
            print(f"- {error}")
        return 1

    print("\nAll global variable config checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
