from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


DATASET_CHOICES = {"physionet", "mimic"}

KNOWN_LIST_KEYS = {
    "TREATMENTS",
    "LATENT_ORDER",
    "BACKGROUND_FEATURE_COLUMNS",
    "EFFECT_MODIFIER_COLUMNS",
    "PHYSIONET_SET_NAMES",
    "CHRONIC_ICD_KEYWORDS",
    "ACUTE_ICD_KEYWORDS",
    "PICKLE_GCS_COMPONENTS",
    "PICKLE_EXPECTED_SUMMARY_COLUMNS",
    "CANONICAL_TS_COLUMNS",
    "CANONICAL_OC_COLUMNS",
    "REQUIRED_OC_COLUMNS",
    "ALLOWED_TOP_K_VALUES",
}

KNOWN_BOOL_KEYS = {
    "DOWN_SAMPLE",
    "USE_EXPANDED_SAFE_CONFOUNDERS",
    "SAVE_CONTOUR_PLOT",
    "MATCH_WITH_REPLACEMENT",
    "REQUIRE_BINARY_CONF",
    "OPTIMIZED",
}

KNOWN_INT_KEYS = {
    "SEED",
    "TRIALS",
    "SENSITIVITY_GRID_STEPS",
    "TOP_K_BENCHMARK_CONFOUNDERS",
    "ARTIFACT_SCHEMA_VERSION",
    "MAX_DIST",
    "MIN_MATCHED_PAIRS",
    "TOTAL_STAGES",
    "PROGRESS_EVERY",
}

KNOWN_FLOAT_KEYS = {
    "DEFAULT_SENSITIVITY_ALPHA",
    "DEFAULT_SENSITIVITY_C_Y",
    "DEFAULT_SENSITIVITY_C_T",
    "DEFAULT_SENSITIVITY_RHO",
    "EPSILON",
    "MIN_MATCH_RATE",
}

REQUIRED_CANONICAL_KEYS = [
    "DATASET_MODEL",
    "DATASET_NAME",
    "ID_COL",
    "ALT_ID_COL",
    "OUTCOME_COL",
    "GRAPH_OUTCOME_NODE",
    "LATENT_TAGS_PATH",
    "PHYSIONET_PKL_PATH",
    "GRAPH_PKL_PATH",
    "OUTPUT_DIR",
    "RESULTS_TXT_PATH",
    "CATE_RESULTS_DIR",
    "EXPERIMENT_DIR",
    "TREATMENTS",
    "LATENT_ORDER",
    "BACKGROUND_FEATURE_COLUMNS",
    "EFFECT_MODIFIER_COLUMNS",
    "MODEL_TYPE",
    "SEED",
    "TRIALS",
    "DOWN_SAMPLE",
    "USE_EXPANDED_SAFE_CONFOUNDERS",
    "DEFAULT_SENSITIVITY_ALPHA",
    "MAX_DIST",
    "MIN_MATCHED_PAIRS",
    "MIN_MATCH_RATE",
    "MATCH_WITH_REPLACEMENT",
    "REQUIRE_BINARY_CONF",
    "DEFAULT_GRAPH_PKL_PATH",
    "DEFAULT_GRAPH_PNG_PATH",
    "PREPROCESS_RAW_DATA_PATH",
    "PREPROCESS_OUTPUT_PATH",
    "PREPROCESS_OUTPUT_DIR",
    "PHYSIONET_SET_NAMES",
    "TAGGING_PKL_PATH",
    "TAGGING_OUTPUT_CSV_PATH",
    "OPTIMIZED",
    "THRESHOLDS_PATH",
    "DEFAULT_THRESHOLDS",
]

MIMIC_ONLY_KEYS = [
    "CHRONIC_ICD_KEYWORDS",
    "ACUTE_ICD_KEYWORDS",
    "PICKLE_TS_SUMMARY_SPECS",
    "PICKLE_GCS_COMPONENTS",
    "PICKLE_URINE_VARIABLE",
    "PICKLE_WEIGHT_VARIABLE",
    "PICKLE_TS_BINARY_HELPERS",
    "PICKLE_OC_OPTIONAL_FIELDS",
    "PICKLE_EXPECTED_SUMMARY_COLUMNS",
    "PROGRESS_EVERY",
]

SCRIPT_CONFIG_CONTRACTS: dict[str, dict[str, object]] = {
    "main.py": {
        "required_keys": ["MODEL_TYPE", "TRIALS"],
        "alias_groups": {
            "latent_tags_path": ["LATENT_TAGS_PATH"],
            "dataset_pkl_path": ["DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
            "output_dir": ["OUTPUT_DIR"],
        },
    },
    "src/cate_estimation.py": {
        "required_keys": [
            "OUTCOME_COL",
            "GRAPH_OUTCOME_NODE",
            "TREATMENTS",
            "BACKGROUND_FEATURE_COLUMNS",
            "EFFECT_MODIFIER_COLUMNS",
            "SEED",
            "DOWN_SAMPLE",
            "USE_EXPANDED_SAFE_CONFOUNDERS",
            "DEFAULT_SENSITIVITY_ALPHA",
            "MODEL_TYPE",
        ],
        "alias_groups": {
            "latent_tags_path": ["CATE_LATENT_TAGS_PATH", "LATENT_TAGS_PATH"],
            "pkl_path": ["CATE_PKL_PATH", "DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
            "graph_pkl_path": ["GRAPH_PKL_PATH"],
            "output_dir": ["CATE_OUTPUT_DIR", "OUTPUT_DIR"],
        },
    },
    "src/matching_causal_effect.py": {
        "required_keys": [
            "OUTCOME_COL",
            "GRAPH_OUTCOME_NODE",
            "TREATMENTS",
            "BACKGROUND_FEATURE_COLUMNS",
            "SEED",
            "DOWN_SAMPLE",
            "USE_EXPANDED_SAFE_CONFOUNDERS",
            "MAX_DIST",
            "MIN_MATCHED_PAIRS",
            "MIN_MATCH_RATE",
            "MATCH_WITH_REPLACEMENT",
            "REQUIRE_BINARY_CONF",
        ],
        "alias_groups": {
            "latent_tags_path": ["MATCHING_LATENT_TAGS_PATH", "LATENT_TAGS_PATH"],
            "pkl_path": ["MATCHING_PKL_PATH", "DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
            "graph_pkl_path": ["GRAPH_PKL_PATH"],
            "output_dir": ["MATCHING_OUTPUT_DIR", "OUTPUT_DIR"],
        },
    },
    "src/mortality_prediction_using_latents.py": {
        "required_keys": ["OUTCOME_COL", "SEED"],
        "alias_groups": {
            "latent_tags_path": ["MORTALITY_LATENT_TAGS_PATH", "LATENT_TAGS_PATH"],
            "pkl_path": ["MORTALITY_PKL_PATH", "DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
            "results_txt_path": ["MORTALITY_RESULTS_TXT_PATH", "RESULTS_TXT_PATH"],
        },
    },
    "src/analyze_cate_results.py": {
        "required_keys": [
            "OUTCOME_COL",
            "BACKGROUND_FEATURE_COLUMNS",
            "SEED",
            "DEFAULT_SENSITIVITY_ALPHA",
        ],
        "alias_groups": {
            "latent_tags_path": ["ANALYZE_LATENT_TAGS_PATH", "LATENT_TAGS_PATH"],
            "pkl_path": ["ANALYZE_PKL_PATH", "DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
            "results_dir": ["CATE_RESULTS_DIR"],
            "output_dir": ["ANALYZE_OUTPUT_DIR", "OUTPUT_DIR", "CATE_RESULTS_DIR"],
        },
    },
    "src/permutations_test.py": {
        "required_keys": ["OUTCOME_COL", "TRIALS", "MODEL_TYPE", "SEED"],
        "alias_groups": {
            "experiment_dir": ["EXPERIMENT_DIR"],
            "latent_tags_path": ["PERMUTATIONS_LATENT_TAGS_PATH", "LATENT_TAGS_PATH"],
            "pkl_path": ["PERMUTATIONS_PKL_PATH", "DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
            "graph_pkl_path": ["GRAPH_PKL_PATH"],
        },
    },
    "src/split_predicted_latent_tags.py": {
        "required_keys": ["DATASET_MODEL"],
        "alias_groups": {
            "input_csv": ["SPLIT_INPUT_CSV", "INPUT_CSV"],
        },
    },
    "src/physionet2012_causal_graph.py": {
        "required_keys": ["GRAPH_OUTCOME_NODE"],
        "alias_groups": {
            "graph_pkl_path": ["DEFAULT_GRAPH_PKL_PATH", "GRAPH_PKL_PATH"],
            "graph_png_path": ["DEFAULT_GRAPH_PNG_PATH"],
        },
    },
    "src/mimiciii_causal_graph.py": {
        "required_keys": ["GRAPH_OUTCOME_NODE"],
        "alias_groups": {
            "graph_pkl_path": ["DEFAULT_GRAPH_PKL_PATH", "GRAPH_PKL_PATH"],
            "graph_png_path": ["DEFAULT_GRAPH_PNG_PATH"],
        },
    },
    "src/tagging_latent_variables_physionet.py": {
        "required_keys": [
            "LATENT_ORDER",
            "TAGGING_PKL_PATH",
            "TAGGING_OUTPUT_CSV_PATH",
            "OPTIMIZED",
            "DEFAULT_THRESHOLDS",
        ],
        "alias_groups": {
            "thresholds_path": ["THRESHOLDS_PATH"],
        },
    },
    "src/tagging_latent_variables_mimiciii.py": {
        "required_keys": [
            "LATENT_ORDER",
            "TAGGING_PKL_PATH",
            "TAGGING_OUTPUT_CSV_PATH",
            "DEFAULT_THRESHOLDS",
            "CHRONIC_ICD_KEYWORDS",
            "ACUTE_ICD_KEYWORDS",
            "PICKLE_TS_SUMMARY_SPECS",
            "PICKLE_GCS_COMPONENTS",
            "PICKLE_URINE_VARIABLE",
            "PICKLE_WEIGHT_VARIABLE",
            "PICKLE_TS_BINARY_HELPERS",
            "PICKLE_OC_OPTIONAL_FIELDS",
            "PICKLE_EXPECTED_SUMMARY_COLUMNS",
            "PROGRESS_EVERY",
        ],
        "alias_groups": {},
    },
    "src/preprocess_physionet_2012.py": {
        "required_keys": [
            "PREPROCESS_RAW_DATA_PATH",
            "PREPROCESS_OUTPUT_PATH",
            "PREPROCESS_OUTPUT_DIR",
            "PHYSIONET_SET_NAMES",
        ],
        "alias_groups": {},
    },
    "src/preprocess_mimic_iii_large.py": {
        "required_keys": [
            "PREPROCESS_RAW_DATA_PATH",
            "PREPROCESS_OUTPUT_PATH",
            "TOTAL_STAGES",
        ],
        "alias_groups": {},
    },
}

_TRUE_VALUES = {"true", "yes", "on", "1"}
_FALSE_VALUES = {"false", "no", "off", "0"}
_INT_RE = re.compile(r"^[+-]?(0|[1-9]\d*)$")
_FLOAT_RE = re.compile(
    r"^[+-]?((\d+\.\d*)|(\.\d+)|(\d+e[+-]?\d+)|(\d+\.\d*e[+-]?\d+)|(\.\d+e[+-]?\d+))$",
    re.IGNORECASE,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_dataset(dataset: str) -> str:
    normalized = str(dataset).strip().lower()
    if normalized not in DATASET_CHOICES:
        raise ValueError(
            f"Invalid dataset {dataset!r}. Expected one of {sorted(DATASET_CHOICES)}."
        )
    return normalized


def default_config_path(dataset: str) -> Path:
    dataset = _normalize_dataset(dataset)
    filename = {
        "physionet": "physionet-global-variables.csv",
        "mimic": "mimic-global-variables.csv",
    }[dataset]
    return repo_root() / "configs" / filename


def _context_message(dataset: str, config_csv_path: str | Path, key: str) -> str:
    return (
        f"dataset={dataset!r}, config file={Path(config_csv_path)}, "
        f"config key={key!r}"
    )


def _parse_value(raw_value: str, *, dataset: str, path: Path, key: str) -> Any:
    value = raw_value.strip()

    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON value in dataset config "
                f"({_context_message(dataset, path, key)}): {exc}"
            ) from exc

    lowered = value.lower()
    if key in KNOWN_BOOL_KEYS and lowered in (_TRUE_VALUES | _FALSE_VALUES):
        return lowered in _TRUE_VALUES

    if _INT_RE.match(value):
        try:
            return int(value)
        except ValueError:
            pass

    if _FLOAT_RE.match(value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


def load_dataset_config(
    dataset: str,
    config_csv_path: str | Path | None = None,
) -> dict[str, object]:
    dataset = _normalize_dataset(dataset)
    path = (
        default_config_path(dataset)
        if config_csv_path is None
        else Path(config_csv_path).expanduser()
    )
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Dataset config CSV does not exist "
            f"({_context_message(dataset, path, '<config_csv_path>')})."
        )
    if not path.is_file():
        raise FileNotFoundError(
            "Dataset config path is not a file "
            f"({_context_message(dataset, path, '<config_csv_path>')})."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                "Dataset config CSV is missing a header row "
                f"({_context_message(dataset, path, '<header>')})."
            )

        columns = [field.strip() for field in reader.fieldnames if field and field.strip()]
        values_by_key: dict[str, list[str]] = {key: [] for key in columns}

        for row in reader:
            normalized_row = {
                (key.strip() if key else ""): ("" if value is None else str(value).strip())
                for key, value in row.items()
            }
            if not any(normalized_row.get(key, "") for key in columns):
                continue
            for key in columns:
                value = normalized_row.get(key, "")
                if value:
                    values_by_key[key].append(value)

    config: dict[str, object] = {
        "__dataset__": dataset,
        "__config_csv_path__": str(path),
    }
    for key, raw_values in values_by_key.items():
        if not raw_values:
            config[key] = None
            continue

        if key in KNOWN_LIST_KEYS:
            config[key] = [
                _parse_value(raw, dataset=dataset, path=path, key=key)
                for raw in raw_values
            ]
            continue

        if len(raw_values) == 1:
            config[key] = _parse_value(raw_values[0], dataset=dataset, path=path, key=key)
            continue

        config[key] = [
            _parse_value(raw, dataset=dataset, path=path, key=key)
            for raw in raw_values
        ]

    return config


def _config_dataset(config: dict[str, object]) -> str:
    return str(config.get("__dataset__", "<unknown>"))


def _config_path(config: dict[str, object]) -> str:
    return str(config.get("__config_csv_path__", "<unknown>"))


def get_config_value(config: dict[str, object], key: str, default: Any = None) -> Any:
    value = config.get(key)
    return default if value is None else value


def _has_config_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(_has_config_value(item) for item in value)
    return True


def get_first_available(
    config: dict[str, object],
    keys: list[str],
    default: Any = None,
) -> Any:
    for key in keys:
        value = config.get(key)
        if _has_config_value(value):
            return value
    return default


def get_first_available_with_key(
    config: dict[str, object],
    keys: list[str],
    default: Any = None,
) -> tuple[Any, str | None]:
    for key in keys:
        value = config.get(key)
        if _has_config_value(value):
            return value, key
    return default, None


def get_config_required(config: dict[str, object], key: str) -> Any:
    value = config.get(key)
    if value is None:
        raise KeyError(
            "Missing required dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)})."
        )
    return value


def get_config_list(
    config: dict[str, object],
    key: str,
    default: list[Any] | None = None,
) -> list[Any] | None:
    value = get_config_value(config, key, default)
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def get_config_scalar(config: dict[str, object], key: str, default: Any = None) -> Any:
    value = get_config_value(config, key, default)
    if isinstance(value, list):
        raise TypeError(
            "Expected scalar dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got list."
        )
    return value


def _parse_bool(value: Any, *, config: dict[str, object], key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    raise TypeError(
        "Expected boolean dataset config value "
        f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
        f"got {value!r}."
    )


def get_config_bool(
    config: dict[str, object],
    key: str,
    default: bool | None = None,
) -> bool | None:
    value = get_config_scalar(config, key, default)
    if value is None:
        return None
    return _parse_bool(value, config=config, key=key)


def get_config_int(
    config: dict[str, object],
    key: str,
    default: int | None = None,
) -> int | None:
    value = get_config_scalar(config, key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            "Expected integer dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got bool."
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Expected integer dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got {value!r}."
        ) from exc
    if not numeric_value.is_integer():
        raise ValueError(
            "Expected integral dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got {value!r}."
        )
    return int(numeric_value)


def get_config_float(
    config: dict[str, object],
    key: str,
    default: float | None = None,
) -> float | None:
    value = get_config_scalar(config, key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            "Expected float dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got bool."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Expected float dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got {value!r}."
        ) from exc


def resolve_with_precedence(
    cli_value: Any,
    config: dict[str, object],
    key: str,
    fallback: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    config_value = config.get(key)
    if config_value is not None:
        return config_value
    return fallback


def normalize_script_name(script_name: str | Path) -> str:
    value = str(script_name).replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]
    if "/src/" in value:
        value = "src/" + value.rsplit("/src/", 1)[1]
    elif value.endswith("/main.py"):
        value = "main.py"
    return value


def validate_script_config(
    script_name: str | Path,
    config: dict[str, object],
) -> dict[str, dict[str, object]]:
    normalized_script = normalize_script_name(script_name)
    if normalized_script not in SCRIPT_CONFIG_CONTRACTS:
        raise KeyError(f"No dataset config contract is defined for {normalized_script!r}.")

    contract = SCRIPT_CONFIG_CONTRACTS[normalized_script]
    resolved: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    for key in list(contract.get("required_keys", [])):
        value, source_key = get_first_available_with_key(config, [str(key)])
        if source_key is None:
            missing.append(str(key))
        else:
            resolved[str(key)] = {
                "value": value,
                "source_key": source_key,
                "candidate_keys": [str(key)],
            }

    alias_groups = dict(contract.get("alias_groups", {}))
    for name, keys in alias_groups.items():
        candidate_keys = [str(key) for key in keys]
        value, source_key = get_first_available_with_key(config, candidate_keys)
        if source_key is None:
            missing.append(f"{name} ({', '.join(candidate_keys)})")
        else:
            resolved[str(name)] = {
                "value": value,
                "source_key": source_key,
                "candidate_keys": candidate_keys,
            }

    if missing:
        raise KeyError(
            "Missing required config values for "
            f"{normalized_script}: {', '.join(missing)}"
        )

    return resolved


def _format_summary_value(value: Any, limit: int = 120) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def print_resolved_config_summary(
    script_name: str | Path,
    config: dict[str, object],
    resolved: dict[str, dict[str, object]],
) -> None:
    normalized_script = normalize_script_name(script_name)
    print("VALIDATE-CONFIG-ONLY PASS")
    print(f"  dataset: {_config_dataset(config)}")
    print(f"  script: {normalized_script}")
    print(f"  config_csv: {_config_path(config)}")
    for name in sorted(resolved):
        item = resolved[name]
        value = _format_summary_value(item["value"])
        print(f"  {name}: {value}  [source={item['source_key']}]")


def maybe_run_validate_config_only(
    script_name: str,
    *,
    fixed_dataset: str | None = None,
    default_dataset: str | None = None,
    argv: list[str] | None = None,
) -> None:
    argv = list(sys.argv if argv is None else argv)
    if "--validate-config-only" not in argv:
        return

    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--dataset-config-csv", default=None)
    parser.add_argument("--model", choices=sorted(DATASET_CHOICES), default=None)
    parser.add_argument("--dataset", choices=sorted(DATASET_CHOICES), default=None)
    args, _ = parser.parse_known_args(argv[1:])

    dataset = fixed_dataset or args.dataset or args.model or default_dataset
    if dataset is None:
        raise SystemExit(
            "Validation mode requires --dataset/--model or a fixed script dataset."
        )

    try:
        config = load_dataset_config(dataset, args.dataset_config_csv)
        resolved = validate_script_config(script_name, config)
        print_resolved_config_summary(script_name, config, resolved)
    except Exception as exc:
        print(f"VALIDATE-CONFIG-ONLY FAIL: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(0)
