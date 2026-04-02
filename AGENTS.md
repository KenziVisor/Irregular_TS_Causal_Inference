# Project Details

## Summary

This repository is thesis research code for causal inference on irregular multivariate ICU time series from PhysioNet / CinC 2012.

The main idea is:

1. preprocess raw ICU files into a reusable patient-level time-series dataset
2. define a clinician-authored causal DAG
3. convert observed variables into interpretable latent clinical states such as `Shock`, `RespFail`, `RenalFail`, and `Severity`
4. use those latent states for mortality prediction, matching-based effect summaries, and DML/CausalForest effect estimation

This is script-first research code, not a package:

- `main.py` is empty
- `requirements.txt` now exists and pins the WSL analysis stack used for the current sensitivity pipeline
- there is no test suite

## Read This First

- Almost all real logic lives in `src/`.
- Relative paths are fragile. Do not assume running from repo root is safe.
- Most `src/*.py` files use paths like `../../data/...` and `../../physionet2012/...` relative to the process working directory.
- `src/draft/*.py` usually assume `../../../data/...`.
- `src/physionet2012_causal_graph.py` is inconsistent with the rest of the repo and saves to `../data/causal_graph.pkl`.
- Generated result folders under `src/run_*` are archived experiment outputs, not source code.

## Repo Map

- `src/preprocess_physionet_2012.py`: raw PhysioNet files -> processed pickle. Auto-runs on import.
- `src/physionet2012_causal_graph.py`: builds DAG, saves graph pickle and PNG. Auto-runs on import.
- `src/tagging_latent_variables.py`: older summary-statistics latent tagger. Auto-runs on import.
- `src/clinically_sufficient_tagging_latent_variables.py`: newer clinical/windowed latent tagger. Guarded by `if __name__ == "__main__":`.
- `src/optimize_latent_thresholds.py`: Optuna threshold search for the older tagger only. Guarded.
- `src/mortality_prediction_using_latents.py`: mortality prediction from latent tags using Logistic Regression and a small PyTorch MLP. Auto-runs on import.
- `src/matching_causal_effect.py`: DAG-guided confounder selection + greedy Hamming matching baseline. Guarded.
- `src/cate_estimation.py`: DAG-guided confounder selection + EconML `CausalForestDML` / `LinearDML`. Guarded. This is the current main causal script.
- `src/draft/`: older stratification-based ATE/CATE experiments plus a treatment-splitting helper.
- `src/run_*`: archived `cate_estimation.py` outputs from past runs.

## Core Data Contracts

### Processed pickle

`physionet2012_ts_oc_ids.pkl` stores a 3-item object:

- `ts`: long-format time-series dataframe
- `oc`: patient-level outcomes dataframe
- `ts_ids`: sorted list of patient IDs

### `ts` dataframe

Expected columns:

- `ts_id`: patient / record identifier as string
- `minute`: integer minutes since admission
- `variable`: measurement name
- `value`: numeric observation

### `oc` dataframe

Expected columns:

- `ts_id`
- `length_of_stay`
- `in_hospital_mortality`
- `subset`

### Latent tag CSVs

Both latent tagging pipelines output the same binary latent columns:

- `Severity`
- `Shock`
- `RespFail`
- `RenalFail`
- `HepFail`
- `HemeFail`
- `Inflam`
- `NeuroFail`
- `CardInj`
- `Metab`
- `ChronicRisk`
- `AcuteInsult`

The CSV schema is `ts_id` plus those 12 binary columns.

### Background features used by the main causal scripts

`matching_causal_effect.py` and `cate_estimation.py` build patient-level background covariates from the first recorded values of:

- `Age`
- `Gender`
- `Weight`
- `ICUType_1`, `ICUType_2`, `ICUType_3`, `ICUType_4`

Important: the DAG contains `Height`, but the current main causal scripts do not load it.

## Actual Pipeline To Assume By Default

If a future thread asks for the main current flow, assume:

1. `src/preprocess_physionet_2012.py`
2. `src/physionet2012_causal_graph.py`
3. `src/clinically_sufficient_tagging_latent_variables.py`
4. one of:
   - `src/mortality_prediction_using_latents.py`
   - `src/cate_estimation.py`
   - `src/matching_causal_effect.py` only after confirming which latent CSV it should use

Why this is the best default:

- downstream mortality and CATE scripts default to `latent_tags_clinical.csv`
- the clinical tagger is more detailed than the older summary-statistics tagger
- the threshold optimizer still tunes the older tagger, not the clinical one

## What Each Main Script Actually Does

### `src/preprocess_physionet_2012.py`

- Reads raw PhysioNet sets `set-a`, `set-b`, `set-c` plus the matching outcomes files.
- Drops the first row of each patient CSV via `.iloc[1:]`.
- Removes rows with missing `Parameter` and rows where `Value < 0`.
- Skips patient files with 5 or fewer valid rows.
- Converts `Time` from `HH:MM` to integer minutes since admission.
- Rewrites `ICUType` into `ICUType_1` ... `ICUType_4` and sets `value = 1`.
- Saves `../../data/processed/physionet2012_ts_oc_ids.pkl`.
- Auto-runs on import.

### `src/physionet2012_causal_graph.py`

- Builds a hand-crafted clinical `networkx.DiGraph`.
- Background nodes: `Age`, `Gender`, `Height`, `Weight`, `ICUType`.
- Latent nodes: `ChronicRisk`, `AcuteInsult`, `Severity`, `Shock`, `RespFail`, `RenalFail`, `HepFail`, `HemeFail`, `Inflam`, `NeuroFail`, `CardInj`, `Metab`.
- Observed nodes are vitals, labs, interventions, and the outcome node `Death`.
- Main structure:
  - background -> `ChronicRisk`
  - `ChronicRisk` and `AcuteInsult` -> `Severity`
  - `Severity` -> organ-failure latent states
  - latent states -> observed measurements
  - `Severity`, several organ failures, and `Age` -> `Death`
- Saves `../data/causal_graph.pkl` and `../PhysioNet 2012 - Causal DAG.png`.
- Auto-runs on import.

### `src/tagging_latent_variables.py` (older tagger)

- Loads the processed pickle.
- Pivots the long series to a wide patient-minute dataframe.
- Builds per-patient summary features using `min`, `max`, `mean`, `first`, `last`.
- Applies binary threshold rules to the 12 latent variables.
- Defaults to `OPTIMIZED = True` and loads thresholds from `../../data/optimal_thresholds.txt`.
- Writes `latent_tags_optimized.csv` plus `latent_tags_optimized_trees.pkl` in the current working directory.
- Auto-runs on import.

Known issue:

- The tag rules use `Urine_sum`, but `build_patient_summaries()` in this file never computes it. Urine-based logic is therefore incomplete here.

### `src/clinically_sufficient_tagging_latent_variables.py` (preferred tagger)

- Keeps each patient's irregular time series intact longer by building patient contexts instead of immediately collapsing to one summary row.
- Uses explicit windows:
  - `w_0_6h`
  - `w_6_12h`
  - `w_12_24h`
  - `w_0_24h`
  - `w_24_48h`
  - `w_0_48h`
- Uses clinically motivated ordinal stage functions and then converts stages to binary tags.
- Important helpers:
  - `worst_paired_pf_ratio()`: pairs `PaO2` with nearest `FiO2` within 120 minutes
  - `urine_sum()`: actually aggregates urine output over a chosen window
  - `infer_icu_type()`: reconstructs a single `ICUType` from one-hot columns
- Defaults to `OPTIMIZED = False` and `SAVE_STAGE_DETAILS = False`.
- Writes `latent_tags_clinical.csv` plus `latent_tags_clinical_trees.pkl` in the current working directory.
- Optional stage details file: `latent_tags_clinical_stages.csv`.

### `src/optimize_latent_thresholds.py`

- Builds summary statistics once, then runs Optuna.
- Objective = validation AUROC from downstream mortality prediction with `LogisticRegression`.
- Adds `Urine_sum` correctly during summary construction.
- Writes `mortality_prediction_results_optimized.txt` and `optimal_thresholds.txt` to the current working directory.
- Tunes the older simple tagger only; it does not optimize the clinical/windowed tagger.

### `src/mortality_prediction_using_latents.py`

- Defaults to `../../data/latent_tags_clinical.csv`.
- Merges latent tags with `in_hospital_mortality` from the processed pickle.
- Trains:
  - class-balanced `LogisticRegression`
  - a small PyTorch MLP
- Scales features with `StandardScaler`.
- In practice uses an 80/10/10 train/val/test split because the temp split is hard-coded to `0.5`; the `val_size` argument is not actually used.
- Writes `clinical_mortality_prediction_results.txt`.
- Auto-runs on import.

### `src/matching_causal_effect.py`

- Defaults to `../../data/latent_tags.csv`, not the clinical tags.
- Loads latent tags, outcome, and background covariates.
- Uses the DAG to build a backdoor adjustment set.
- Converts confounders to a binary matching matrix:
  - binary columns stay binary
  - other numeric confounders are binarized by median threshold unless `REQUIRE_BINARY_CONF = True`
- Performs greedy 1:1 treated/control matching with Hamming distance.
- Main matching config:
  - `max_dist = 2`
  - `MIN_MATCHED_PAIRS = 30`
  - `MIN_MATCH_RATE = 0.50`
  - `MATCH_WITH_REPLACEMENT = False`
- Writes:
  - `matching_outputs/<Treatment>/confounder_analysis.txt`
  - `matching_outputs/<Treatment>/summary_results.txt`
  - `matching_outputs/<Treatment>/matched_pairs.csv`
  - `matching_outputs/global_summary.csv`
- This is a matched-pair baseline, closer to an ATT-style summary than a per-patient CATE model.

### `src/cate_estimation.py`

- Defaults to `../../data/latent_tags_clinical.csv`.
- Supports `CausalForestDML` and `LinearDML` via `--model-type`.
- CLI flags:
  - `--output-dir`
  - `--down-sample`
  - `--use-expanded-safe-confounders`
  - `--model-type`
- Loads latent tags, outcome, and background features.
- Uses the DAG for confounder discovery.
- Uses a fixed preferred set of effect modifiers:
  - `Age`, `Gender`, `Weight`
  - `ICUType_1` ... `ICUType_4`
  - `ChronicRisk`, `AcuteInsult`
- Missing confounders / modifiers are median-imputed instead of dropping rows.
- Current code allows overlap between `W` (confounders) and `X` (effect modifiers).
- The fit call now passes `cache_values=True`, and saved artifacts include `cache_values_used`, estimator class, exact confounder / effect-modifier order, and a `direct_diagnostics` block for easy post-hoc reuse.
- Default output dir is now `../../data/relevant_outputs/cate_outputs_predicted_230326` when the script is run from `src/`, which aligns it with `analyze_cate_results.py`.
- Writes per treatment:
  - `confounder_analysis.txt`
  - `summary_results.txt`
  - `cate.csv`
  - `<Treatment>_model.pkl` with the fitted estimator plus saved confounders, effect modifiers, and fill values for reuse in another script
  - sometimes `feature_importance.csv`
- Writes run-level summaries:
  - `global_summary.csv`: cleaned run-level results table with stable `row_id`
  - `control_messages_cate_estimation.csv`: provenance / diagnostic / path fields joinable on `row_id`, `treatment`, `model_type`
  - `manager_global_summary.csv`

### `src/analyze_cate_results.py`

- Reads the saved `<Treatment>_model.pkl` artifacts under the CATE output directory and recomputes / validates sensitivity outputs, benchmark proxy scores, and contour artifacts.
- Writes per treatment:
  - `<Treatment>_benchmark_scores.csv`
  - `<Treatment>_benchmark_report.txt`
- Writes run-level summaries:
  - `benchmark_summary.csv`: cleaned numeric analysis results table with stable `row_id`
  - `control_messages_analyze_cate_results.csv`: source labels / warnings / path fields joinable on `row_id`, `treatment`, `model_type`

## Shared DAG / Confounder Logic In `matching_causal_effect.py` And `cate_estimation.py`

Both scripts implement essentially the same graph logic:

- map dataframe `ICUType_1..4` back to graph node `ICUType`
- map dataframe outcome `in_hospital_mortality` to graph outcome node `Death`
- keep only available background / latent nodes as adjustment candidates
- build a candidate backdoor pool as:
  - allowed nodes
  - intersect ancestors of treatment
  - intersect ancestors of outcome in the `do(T)` graph
  - remove descendants of treatment
- if `USE_EXPANDED_SAFE_CONFOUNDERS = True`, also remove colliders and descendants of colliders on backdoor paths
- if the expanded safe set fails to block all backdoor paths, fall back to a minimal blocking set

## Archived Run Folders

The folders under `src/run_*` are saved outputs from `cate_estimation.py`, not code.

Observed naming convention:

- `run_cf_*`: `CausalForest`
- `run_ld_*`: `LinearDML`
- `d1`: down-sampled outcome runs (`outcome_rate = 0.5` in the saved summaries)
- `d0`: original class imbalance
- `e1`: expanded safe confounder mode
- `e0`: minimal confounder mode

Each run folder contains per-treatment outputs plus `global_summary.csv` and `manager_global_summary.csv`.

## Biggest Footguns

- Relative paths are not normalized from `__file__`; execution directory matters a lot.
- The graph uses `Death`, but the dataframe uses `in_hospital_mortality`.
- Preprocessing converts `ICUType` into one-hot-style variables, but the DAG still uses a single `ICUType` node.
- `Height` exists in the DAG but is ignored by the current main causal scripts.
- The simple tagger and the Optuna optimizer belong to the older summary-statistics pipeline.
- `matching_causal_effect.py` still defaults to the older `latent_tags.csv` path.
- Several scripts auto-run when imported: preprocess, DAG construction, simple tagging, mortality prediction, and `draft/treatment_split.py`.
- Default output directories are relative to the process cwd; this is why archived run folders ended up directly under `src/`.
- Saved CATE model pickles under `data/relevant_outputs/cate_outputs_predicted_230326` were created with a newer stack than the current unpinned Python 3.8 requirements environment: they reference `econml.validate.sensitivity_analysis.SensitivityParams` and `scikit-learn 1.5.1`, so artifact-side validation may require a compatibility shim or a one-time refit fallback.
- In the older WSL `econml310` env with `econml 0.15.1`, loaded `LinearDML` / `CausalForestDML` estimators exposed `residuals_` and `summary()` when trained with `cache_values=True`, but they did not expose estimator-native `robustness_value()`, `sensitivity_summary()`, or `sensitivity_interval()`. That version mismatch was the main reason the old sensitivity pipeline fell back to manual calculations.
- Sensitivity-analysis APIs (`robustness_value`, `sensitivity_interval`, `sensitivity_summary`) are available on fresh and loaded `LinearDML` / `CausalForestDML` estimators in the pinned `econml310` WSL env only after upgrading to `econml==0.16.0`; they are absent in `econml==0.15.1`.
- Current schema-v3 CATE artifacts written by `src/cate_estimation.py` save training-time direct RV / interval / summary values plus residual metadata explicitly, and `src/analyze_cate_results.py` prefers those saved training-direct values before trying loaded-estimator direct calls, compatibility refits, or manual fallback paths.
- In the current `econml==0.16.0` env, fitted estimators still do not expose raw non-callable sensitivity-parameter attributes or a direct `sensitivity_plot()` API, so contour plots and the secondary theta-RV path still come from clearly labeled residual-space fallback logic rather than serialized native sensitivity params; Stage 2 real benchmark remains `unimplemented_by_design`.

## Likely Dependencies

Based on imports, the project likely needs at least:

- `pandas`
- `numpy`
- `tqdm`
- `networkx`
- `matplotlib`
- `scikit-learn`
- `torch`
- `optuna`
- `econml`

## Practical Default For Future Codex Threads

When you start a new task in this repo:

1. confirm the current working directory and resolve all relative paths first
2. check whether the processed pickle, graph pickle, and intended latent CSV actually exist
3. assume the clinical tagger is the current representation unless the user explicitly wants the older threshold-optimized pipeline
4. ignore `src/run_*` unless the task is about analyzing prior results
5. treat this repo as research scripts with side effects, not as import-safe library code
