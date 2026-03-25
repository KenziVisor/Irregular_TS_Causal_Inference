from __future__ import annotations

import csv
from pathlib import Path


INPUT_CSV = Path("../../data/predicted_latent_tags_230326.csv")


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
) -> tuple[list[str], list[str]]:
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

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns, expected {len(header)}."
                )

            ts_id = row[0]
            values = row[1:]
            prob_rows.append([ts_id, *values[:midpoint]])
            tag_rows.append([ts_id, *values[midpoint:]])

    write_csv(prob_output, ["ts_id", *prob_columns], prob_rows)
    write_csv(tag_output, ["ts_id", *tag_columns], tag_rows)

    return prob_columns, tag_columns


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    prob_output, tag_output = default_output_paths(INPUT_CSV)
    prob_columns, tag_columns = split_predicted_latent_tags(
        input_csv=INPUT_CSV,
        prob_output=prob_output,
        tag_output=tag_output,
    )

    print(f"Saved probabilities table to: {prob_output}")
    print(f"Saved absolute tags table to: {tag_output}")
    print(f"Probability columns: {len(prob_columns)}")
    print(f"Absolute tag columns: {len(tag_columns)}")


if __name__ == "__main__":
    main()
