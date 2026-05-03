#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_main.sh \
    --dataset physionet \
    --latent-tags-dir /path/to/latent_tag_voters \
    --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl \
    --output-dir /path/to/run_output

  ./scripts/run_main.sh \
    --dataset mimic \
    --latent-tags-dir /path/to/latent_tag_voters \
    --dataset-pkl-path /path/to/mimic_iii_ts_oc_ids.pkl \
    --output-dir /path/to/run_output

Required arguments:
  --dataset             Dataset name: physionet or mimic.
  --latent-tags-dir     Directory containing latent-tag voter CSV files.
  --dataset-pkl-path    Processed dataset pickle path.
  --output-dir          Output directory for the orchestrated run.

Optional arguments:
  --dataset-config-csv  Dataset config CSV. Defaults to configs/<dataset>-global-variables.csv.
  -h, --help            Show this help text.
EOF
}

die() {
  echo "Error: $*" >&2
  echo >&2
  usage >&2
  exit 1
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    die "Missing value for $flag"
  fi
}

resolve_existing_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    die "File does not exist: $path"
  fi

  local dir
  dir="$(cd "$(dirname "$path")" && pwd -P)"
  printf '%s/%s\n' "$dir" "$(basename "$path")"
}

resolve_existing_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    die "Directory does not exist: $path"
  fi

  cd "$path" && pwd -P
}

resolve_output_dir() {
  local path="$1"
  mkdir -p "$path"
  if [[ ! -d "$path" ]]; then
    die "Could not create output directory: $path"
  fi

  cd "$path" && pwd -P
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

DATASET=""
LATENT_TAGS_DIR=""
DATASET_PKL_PATH=""
OUTPUT_DIR=""
DATASET_CONFIG_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dataset)
      require_value "$1" "${2:-}"
      DATASET="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
    --latent-tags-dir)
      require_value "$1" "${2:-}"
      LATENT_TAGS_DIR="$2"
      shift 2
      ;;
    --latent-tags-dir=*)
      LATENT_TAGS_DIR="${1#*=}"
      shift
      ;;
    --dataset-pkl-path)
      require_value "$1" "${2:-}"
      DATASET_PKL_PATH="$2"
      shift 2
      ;;
    --dataset-pkl-path=*)
      DATASET_PKL_PATH="${1#*=}"
      shift
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --dataset-config-csv)
      require_value "$1" "${2:-}"
      DATASET_CONFIG_CSV="$2"
      shift 2
      ;;
    --dataset-config-csv=*)
      DATASET_CONFIG_CSV="${1#*=}"
      shift
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$DATASET" ]] || die "Missing required argument: --dataset"
[[ -n "$LATENT_TAGS_DIR" ]] || die "Missing required argument: --latent-tags-dir"
[[ -n "$DATASET_PKL_PATH" ]] || die "Missing required argument: --dataset-pkl-path"
[[ -n "$OUTPUT_DIR" ]] || die "Missing required argument: --output-dir"

case "$DATASET" in
  physionet)
    : "${DATASET_CONFIG_CSV:=$REPO_ROOT/configs/physionet-global-variables.csv}"
    ;;
  mimic)
    : "${DATASET_CONFIG_CSV:=$REPO_ROOT/configs/mimic-global-variables.csv}"
    ;;
  *)
    die "--dataset must be either 'physionet' or 'mimic'"
    ;;
esac

DATASET_CONFIG_CSV="$(resolve_existing_file "$DATASET_CONFIG_CSV")"
LATENT_TAGS_DIR="$(resolve_existing_dir "$LATENT_TAGS_DIR")"
DATASET_PKL_PATH="$(resolve_existing_file "$DATASET_PKL_PATH")"
OUTPUT_DIR="$(resolve_output_dir "$OUTPUT_DIR")"

cat <<EOF
Resolved runtime configuration:
  repo_root: $REPO_ROOT
  dataset: $DATASET
  dataset_config_csv: $DATASET_CONFIG_CSV
  latent_tags_dir: $LATENT_TAGS_DIR
  dataset_pkl_path: $DATASET_PKL_PATH
  output_dir: $OUTPUT_DIR
EOF

cd "$REPO_ROOT"

python main.py \
  --dataset "$DATASET" \
  --dataset-config-csv "$DATASET_CONFIG_CSV" \
  --latent-tags-dir "$LATENT_TAGS_DIR" \
  --dataset-pkl-path "$DATASET_PKL_PATH" \
  --output-dir "$OUTPUT_DIR"
