# Thesis Repository: Causal Inference on Irregular Clinical Time Series

![Thesis Workflow](Thesis%20code%20Workflow.png)

## Introduction

This repository contains code for thesis research on causal inference with irregular clinical time-series data. It is organized as a research-oriented, script-first codebase and includes components for data preprocessing, latent clinical variable tagging, causal graph construction, confounder logic, and downstream causal effect estimation and analysis.

The current workflow is centered on patient-level ICU time-series processing and subsequent causal analysis stages. The repository preserves multiple experimental scripts and pipelines, so it is best read as an evolving research workspace rather than a packaged software library.

## Installation

A typical local setup is:

```bash
git clone <repo-url>
cd <repo-folder>
conda create -n <env-name> python=3.10 -y
conda activate <env-name>
pip install -r requirements.txt
```

Some scripts rely on relative paths and expected local data locations, so it is advisable to keep the repository structure unchanged and run scripts within the workflow conventions already used in the project.

## Project Flow / Pipeline

The repository follows a staged research workflow:

1. **Preprocessing**
   Raw clinical records are transformed into processed patient-level time-series and outcome tables that can be reused across later analyses.

2. **Causal graph definition**
   A clinician-authored directed acyclic graph is used to encode assumed relationships among background variables, latent clinical states, observed measurements, and mortality.

3. **Latent variable tagging**
   Observed measurements are converted into interpretable latent clinical states, such as severity and organ-failure-related indicators. The repository includes both older summary-based tagging code and a newer clinically motivated windowed tagging pipeline.

4. **Downstream modeling and causal analysis**
   The latent representations are used in mortality prediction, matching-based causal summaries, and heterogeneous treatment effect estimation with econometric machine-learning methods.

5. **Post hoc analysis and checks**
   Additional scripts support analysis of saved causal effect outputs and permutation-based sanity checks for research runs.

---
> **Work in progress — structure and contents will expand as the research advances.**
