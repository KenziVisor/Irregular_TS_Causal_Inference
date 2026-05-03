from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_config import (  # noqa: E402
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_list,
    load_dataset_config,
    resolve_with_precedence,
)


def test_load_physionet_config() -> None:
    config = load_dataset_config("physionet")
    assert isinstance(config["TREATMENTS"], list)
    assert isinstance(config["BACKGROUND_FEATURE_COLUMNS"], list)
    assert isinstance(config["DEFAULT_THRESHOLDS"], dict)
    assert "Severity" in config["TREATMENTS"]
    assert "Age" in config["BACKGROUND_FEATURE_COLUMNS"]


def test_load_mimic_config() -> None:
    config = load_dataset_config("mimic")
    assert isinstance(config["TREATMENTS"], list)
    assert isinstance(config["BACKGROUND_FEATURE_COLUMNS"], list)
    assert isinstance(config["DEFAULT_THRESHOLDS"], dict)
    assert isinstance(config["PICKLE_TS_SUMMARY_SPECS"], dict)
    assert "Severity" in config["TREATMENTS"]


def test_typed_helpers_and_precedence() -> None:
    config = {
        "__dataset__": "test",
        "__config_csv_path__": "/tmp/test.csv",
        "FLAG": True,
        "COUNT": 10,
        "RATE": 0.25,
        "ITEM": "one",
        "CSV_VALUE": "from_csv",
    }

    assert get_config_bool(config, "FLAG") is True
    assert get_config_int(config, "COUNT") == 10
    assert get_config_float(config, "RATE") == 0.25
    assert get_config_list(config, "ITEM") == ["one"]
    assert resolve_with_precedence("from_cli", config, "CSV_VALUE", "fallback") == "from_cli"
    assert resolve_with_precedence(None, config, "CSV_VALUE", "fallback") == "from_csv"
    assert resolve_with_precedence(None, config, "MISSING", "fallback") == "fallback"


def main() -> None:
    test_load_physionet_config()
    test_load_mimic_config()
    test_typed_helpers_and_precedence()
    print("dataset_config checks passed.")


if __name__ == "__main__":
    main()
