from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_OUTPUT_PATH = Path("./latent_tags_majority_vote.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one majority-vote latent CSV from a folder of aligned latent-tag voter CSV files."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the latent-tag voter CSV files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def resolve_input_dir(path_like: str) -> Path:
    input_dir = Path(path_like).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return input_dir


def resolve_output_path(path_like: str) -> Path:
    output_path = Path(path_like).expanduser().resolve()
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"Output path points to a directory, not a CSV file: {output_path}")
    return output_path


def discover_csv_files(input_dir: Path, output_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in input_dir.iterdir():
        if not child.is_file():
            continue
        if child.name.startswith("."):
            continue
        if child.suffix.lower() != ".csv":
            continue
        child_resolved = child.resolve()
        if child_resolved == output_path:
            continue
        candidates.append(child_resolved)

    candidates = sorted(candidates, key=lambda path: str(path))
    if not candidates:
        raise ValueError(
            f"No candidate CSV files were found in input directory: {input_dir}"
        )
    return candidates


def load_latent_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file: {path} ({exc})") from exc

    stripped_columns = [str(column).strip() for column in df.columns]
    if any(column == "" for column in stripped_columns):
        raise ValueError(
            f"CSV file has an empty column name after stripping whitespace: {path}"
        )
    if len(set(stripped_columns)) != len(stripped_columns):
        raise ValueError(
            f"CSV file has duplicate column names after stripping whitespace: {path}. "
            f"Columns={stripped_columns}"
        )

    df = df.copy()
    df.columns = stripped_columns

    if "ts_id" not in df.columns:
        raise ValueError(f"CSV file is missing required 'ts_id' column: {path}")

    df["ts_id"] = df["ts_id"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    empty_ts_id_mask = df["ts_id"] == ""
    if empty_ts_id_mask.any():
        first_bad_row = next(
            row_index for row_index, is_bad in enumerate(empty_ts_id_mask.tolist(), start=1)
            if is_bad
        )
        raise ValueError(
            f"CSV file contains an empty ts_id value after stripping whitespace: {path}. "
            f"First offending data row={first_bad_row}"
        )

    duplicate_mask = df["ts_id"].duplicated(keep=False)
    if duplicate_mask.any():
        first_duplicate_row = next(
            row_index for row_index, is_dup in enumerate(duplicate_mask.tolist(), start=1)
            if is_dup
        )
        duplicate_ts_id = df.loc[duplicate_mask, "ts_id"].iloc[0]
        raise ValueError(
            f"CSV file contains duplicate ts_id values: {path}. "
            f"First duplicated data row={first_duplicate_row}, ts_id={duplicate_ts_id!r}"
        )

    latent_columns = [column for column in df.columns if column != "ts_id"]
    if not latent_columns:
        raise ValueError(
            f"CSV file does not contain any latent columns beyond 'ts_id': {path}"
        )

    return df


def coerce_binary_latent_columns(
    df: pd.DataFrame,
    file_path: Path,
    latent_columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()

    for column in latent_columns:
        numeric_values = pd.to_numeric(out[column], errors="coerce")
        invalid_mask = numeric_values.isna() | ~numeric_values.isin([0, 1])
        if invalid_mask.any():
            first_bad_position = next(
                position for position, is_bad in enumerate(invalid_mask.tolist(), start=1)
                if is_bad
            )
            offending_values = out.loc[invalid_mask, column].head(5).tolist()
            raise ValueError(
                f"Non-binary latent values found in file {file_path}, column {column!r}. "
                f"First offending data row={first_bad_position}. "
                f"Example offending values={offending_values}"
            )
        out[column] = numeric_values.astype(int)

    return out


def validate_reference_dataframe(
    df: pd.DataFrame,
    file_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    latent_columns = [column for column in df.columns if column != "ts_id"]
    validated_df = coerce_binary_latent_columns(df, file_path, latent_columns)
    return validated_df, latent_columns


def validate_against_reference(
    df: pd.DataFrame,
    file_path: Path,
    reference_df: pd.DataFrame,
    reference_file_path: Path,
    reference_latent_columns: Sequence[str],
) -> pd.DataFrame:
    current_latent_columns = [column for column in df.columns if column != "ts_id"]
    current_latent_set = set(current_latent_columns)
    reference_latent_set = set(reference_latent_columns)

    if current_latent_set != reference_latent_set:
        missing_columns = [
            column for column in reference_latent_columns if column not in current_latent_set
        ]
        extra_columns = sorted(current_latent_set - reference_latent_set)
        raise ValueError(
            f"Latent column set mismatch in file {file_path}. "
            f"Reference file={reference_file_path}. "
            f"Missing columns={missing_columns}. Extra columns={extra_columns}."
        )

    validated_df = coerce_binary_latent_columns(df, file_path, current_latent_columns)

    if len(validated_df) != len(reference_df):
        raise ValueError(
            f"Row-count mismatch in file {file_path}. "
            f"Reference rows={len(reference_df):,}, current rows={len(validated_df):,}."
        )

    reference_ts_ids = reference_df["ts_id"].tolist()
    current_ts_ids = validated_df["ts_id"].tolist()
    for row_index, (reference_ts_id, current_ts_id) in enumerate(
        zip(reference_ts_ids, current_ts_ids),
        start=1,
    ):
        if reference_ts_id != current_ts_id:
            raise ValueError(
                f"ts_id sequence mismatch in file {file_path}. "
                f"First mismatched data row={row_index}. "
                f"Reference ts_id={reference_ts_id!r}, current ts_id={current_ts_id!r}."
            )

    return validated_df.loc[:, ["ts_id", *reference_latent_columns]].copy()


def build_majority_vote_dataframe(
    reference_df: pd.DataFrame,
    voter_dfs: Sequence[pd.DataFrame],
    latent_columns: Sequence[str],
) -> pd.DataFrame:
    n_voters = len(voter_dfs)
    ones_count = pd.DataFrame(
        0,
        index=reference_df.index,
        columns=list(latent_columns),
        dtype=int,
    )

    for voter_df in voter_dfs:
        ones_count = ones_count.add(voter_df.loc[:, latent_columns], fill_value=0)

    majority_values = (2 * ones_count >= n_voters).astype(int)
    output_df = pd.concat([reference_df.loc[:, ["ts_id"]].copy(), majority_values], axis=1)
    return output_df


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_path = resolve_output_path(args.output_path)

    print("=== Building majority-vote latent CSV ===")
    print(f"[1/5] Discovering voter CSV files in: {input_dir}")
    csv_files = discover_csv_files(input_dir, output_path)
    print(f"Found {len(csv_files)} candidate CSV files")

    reference_file = csv_files[0]
    print(f"[2/5] Loading reference file: {reference_file}")
    reference_df = load_latent_csv(reference_file)
    reference_df, reference_latent_columns = validate_reference_dataframe(
        reference_df,
        reference_file,
    )
    print(
        f"Reference rows: {len(reference_df):,} | "
        f"latent columns: {len(reference_latent_columns)}"
    )

    voter_dfs: list[pd.DataFrame] = [reference_df.loc[:, ["ts_id", *reference_latent_columns]].copy()]

    print("[3/5] Validating remaining voter files")
    if len(csv_files) == 1:
        print(f"Only one voter CSV found: {reference_file}")
    else:
        for voter_index, voter_file in enumerate(csv_files[1:], start=2):
            print(f"Validating voter {voter_index}/{len(csv_files)}: {voter_file}")
            voter_df = load_latent_csv(voter_file)
            aligned_voter_df = validate_against_reference(
                voter_df,
                voter_file,
                reference_df,
                reference_file,
                reference_latent_columns,
            )
            voter_dfs.append(aligned_voter_df)
            print(f"Validation passed for: {voter_file}")

    print("[4/5] Computing majority vote")
    print(
        f"Voting across {len(voter_dfs)} files, {len(reference_df):,} patients, "
        f"{len(reference_latent_columns)} latent columns"
    )
    output_df = build_majority_vote_dataframe(
        reference_df=reference_df,
        voter_dfs=voter_dfs,
        latent_columns=reference_latent_columns,
    )

    print(f"[5/5] Saving output to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved majority-vote latent CSV: {output_path}")
    print("Done")


if __name__ == "__main__":
    main()
