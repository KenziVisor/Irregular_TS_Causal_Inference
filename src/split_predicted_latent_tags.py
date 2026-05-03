from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/split_predicted_latent_tags.py",
        default_dataset="physionet",
    )

from dataset_config import get_first_available, load_dataset_config


DATASET_MODEL = "physionet"
INPUT_CSV = Path("../../data/predicted_latent_tags_230326.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a combined predicted-latents CSV into probability and tag tables."
    )
    parser.add_argument(
        "--model",
        choices=["physionet", "mimic"],
        default=DATASET_MODEL,
        help=f"Dataset selector for input defaults. Default: {DATASET_MODEL}",
    )
    parser.add_argument(
        "--dataset-config-csv",
        default=None,
        help=(
            "Path to the dataset global-variables CSV. If omitted, use the default "
            "config for --model."
        ),
    )
    parser.add_argument("--input-csv", default=None)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args()


def default_output_paths(input_csv: Path) -> tuple[Path, Path]:
    prob_output = input_csv.with_name(f"{input_csv.stem}_probabilities.csv")
    tag_output = input_csv.with_name(f"{input_csv.stem}_absolute_tags.csv")
    return prob_output, tag_output


def write_csv(output_path: Path, header: list[str], rows: list[list[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def split_predicted_latent_tags(
    input_csv: Path,
    prob_output: Path,
    tag_output: Path,
) -> tuple[list[str], list[str], int]:
    print(f"[2/4] Reading combined latent CSV from: {input_csv.resolve()}")
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {input_csv}") from exc

        if not header:
            raise ValueError(f"Input CSV has an empty header row: {input_csv}")
        if header[0] != "ts_id":
            raise ValueError(
                f"Expected the first column to be 'ts_id', got '{header[0]}' instead."
            )

        non_key_columns = header[1:]
        if len(non_key_columns) == 0 or len(non_key_columns) % 2 != 0:
            raise ValueError(
                "Expected a non-zero, even number of non-key columns after 'ts_id'."
            )

        midpoint = len(non_key_columns) // 2
        prob_columns = non_key_columns[:midpoint]
        tag_columns = non_key_columns[midpoint:]

        prob_rows: list[list[str]] = []
        tag_rows: list[list[str]] = []
        processed_rows = 0

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns, expected {len(header)}."
                )

            ts_id = row[0]
            values = row[1:]
            prob_rows.append([ts_id, *values[:midpoint]])
            tag_rows.append([ts_id, *values[midpoint:]])
            processed_rows += 1
            if processed_rows % 50000 == 0:
                print(f"      Processed {processed_rows:,} data rows")

    print(
        f"[3/4] Writing split outputs ({processed_rows:,} rows | "
        f"{len(prob_columns)} probability columns | {len(tag_columns)} tag columns)"
    )
    write_csv(prob_output, ["ts_id", *prob_columns], prob_rows)
    write_csv(tag_output, ["ts_id", *tag_columns], tag_rows)

    return prob_columns, tag_columns, processed_rows


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.model, args.dataset_config_csv)
    input_default = get_first_available(
        config,
        ["SPLIT_INPUT_CSV", "INPUT_CSV"],
        str(INPUT_CSV),
    )
    input_csv_value = args.input_csv if args.input_csv is not None else input_default
    if input_csv_value is None:
        raise ValueError(
            "Provide --input-csv because this repo does not define a safe checked-in "
            "default combined predicted-latents CSV."
        )

    input_csv = Path(str(input_csv_value))

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print("=== Splitting predicted latent tags ===")
    print(f"[1/4] Runtime configuration: model={args.model} | input={input_csv.resolve()}")
    prob_output, tag_output = default_output_paths(input_csv)
    prob_columns, tag_columns, row_count = split_predicted_latent_tags(
        input_csv=input_csv,
        prob_output=prob_output,
        tag_output=tag_output,
    )

    print("[4/4] Finished writing split CSV files")
    print(f"Saved probabilities table to: {prob_output}")
    print(f"Saved absolute tags table to: {tag_output}")
    print(f"Rows processed: {row_count:,}")
    print(f"Probability columns: {len(prob_columns)}")
    print(f"Absolute tag columns: {len(tag_columns)}")


if __name__ == "__main__":
    main()
