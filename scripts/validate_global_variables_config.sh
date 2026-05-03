#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_EXE="${CONDA_EXE:-/home/kobik/miniconda3/bin/conda}"
if [[ ! -x "${CONDA_EXE}" ]]; then
  CONDA_EXE="conda"
fi
CONDA_ENV="${CONDA_ENV:-econml310}"
PYTHON_CMD=("${CONDA_EXE}" run -n "${CONDA_ENV}" python)

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

cd "${REPO_ROOT}"

failures=0

run_one() {
  local dataset="$1"
  local script="$2"
  shift 2

  if "${PYTHON_CMD[@]}" "$@"; then
    printf 'PASS | %s | %s\n' "${dataset}" "${script}"
  else
    printf 'FAIL | %s | %s\n' "${dataset}" "${script}"
    failures=$((failures + 1))
  fi
}

PHYSIONET_CONFIG="configs/physionet-global-variables.csv"
MIMIC_CONFIG="configs/mimic-global-variables.csv"

run_one "both" "scripts/validate_global_variables_config.py" \
  scripts/validate_global_variables_config.py

for dataset in physionet mimic; do
  if [[ "${dataset}" == "physionet" ]]; then
    config="${PHYSIONET_CONFIG}"
  else
    config="${MIMIC_CONFIG}"
  fi

  run_one "${dataset}" "main.py" \
    main.py --dataset "${dataset}" --dataset-config-csv "${config}" --validate-config-only
  run_one "${dataset}" "src/cate_estimation.py" \
    src/cate_estimation.py --model "${dataset}" --dataset-config-csv "${config}" --validate-config-only
  run_one "${dataset}" "src/matching_causal_effect.py" \
    src/matching_causal_effect.py --model "${dataset}" --dataset-config-csv "${config}" --validate-config-only
  run_one "${dataset}" "src/mortality_prediction_using_latents.py" \
    src/mortality_prediction_using_latents.py --model "${dataset}" --dataset-config-csv "${config}" --validate-config-only
  run_one "${dataset}" "src/analyze_cate_results.py" \
    src/analyze_cate_results.py --model "${dataset}" --dataset-config-csv "${config}" --validate-config-only
  run_one "${dataset}" "src/permutations_test.py" \
    src/permutations_test.py --model "${dataset}" --dataset-config-csv "${config}" --validate-config-only
  run_one "${dataset}" "src/split_predicted_latent_tags.py" \
    src/split_predicted_latent_tags.py --model "${dataset}" --dataset-config-csv "${config}" --validate-config-only
done

run_one "physionet" "src/physionet2012_causal_graph.py" \
  src/physionet2012_causal_graph.py --dataset-config-csv "${PHYSIONET_CONFIG}" --validate-config-only
run_one "physionet" "src/tagging_latent_variables_physionet.py" \
  src/tagging_latent_variables_physionet.py --dataset-config-csv "${PHYSIONET_CONFIG}" --validate-config-only
run_one "physionet" "src/preprocess_physionet_2012.py" \
  src/preprocess_physionet_2012.py --dataset-config-csv "${PHYSIONET_CONFIG}" --validate-config-only

run_one "mimic" "src/mimiciii_causal_graph.py" \
  src/mimiciii_causal_graph.py --dataset-config-csv "${MIMIC_CONFIG}" --validate-config-only
run_one "mimic" "src/tagging_latent_variables_mimiciii.py" \
  src/tagging_latent_variables_mimiciii.py --dataset-config-csv "${MIMIC_CONFIG}" --validate-config-only
run_one "mimic" "src/preprocess_mimic_iii_large.py" \
  src/preprocess_mimic_iii_large.py --dataset-config-csv "${MIMIC_CONFIG}" --validate-config-only

if [[ "${failures}" -ne 0 ]]; then
  printf 'Global variables config validation failed: %s failure(s).\n' "${failures}"
  exit 1
fi

printf 'Global variables config validation passed.\n'
