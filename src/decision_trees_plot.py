#!/usr/bin/env python3
"""
Plot explicit rule diagrams for saved latent-tagging decision-tree pickles.

These pickles store rule callables such as ``functools.partial(tag_xxx, thr=...)``
or plain function objects. They are not sklearn trees, so this utility does not
use sklearn plotting helpers. Instead, it mirrors the dataset-specific rule logic
from the latent-tagging scripts and renders one matplotlib figure per latent.

Usage examples
--------------
python decision_trees_plot.py --dataset mimic --pickle-path mimiciii_latent_tags_output/latent_decision_trees.pkl --output-dir mimic_tree_plots
python decision_trees_plot.py --dataset physionet --pickle-path latent_tags_optimized_trees.pkl --output-dir physionet_tree_plots
"""

from __future__ import annotations

import argparse
import pickle
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from dataset_config import get_config_list, get_config_scalar, load_dataset_config


# ---------------------------------------------------------------------------
# Compatibility placeholders for pickle.load
# ---------------------------------------------------------------------------

def _make_placeholder(name: str):
    def _placeholder(*args, **kwargs):
        return 0

    _placeholder.__name__ = name
    return _placeholder


for _name in {
    "tag_chronic_burden",
    "tag_acute_insult",
    "tag_severity",
    "tag_inflammation",
    "tag_shock",
    "tag_respfail",
    "tag_renal_dysfunction",
    "tag_hepatic_dysfunction",
    "tag_coag_dysfunction",
    "tag_neuro_dysfunction",
    "tag_cardiac_injury",
    "tag_metabolic_derangement",
    "tag_renalfail",
    "tag_hepfail",
    "tag_hemefail",
    "tag_inflam",
    "tag_neurofail",
    "tag_cardinj",
    "tag_metab",
    "tag_chronicrisk",
    "tag_acuteinsult",
}:
    globals()[_name] = _make_placeholder(_name)


# ---------------------------------------------------------------------------
# Defaults mirrored from the source scripts
# ---------------------------------------------------------------------------

MIMIC_DEFAULT_THRESHOLDS = {
    "chronic_age": 65,
    "chronic_albumin": 3.0,
    "acute_emergency_types": {"EMERGENCY", "URGENT"},
    "severity_lactate": 4.0,
    "severity_ph": 7.20,
    "severity_gcs": 9,
    "severity_platelets": 50,
    "severity_creatinine": 3.5,
    "severity_bilirubin": 6.0,
    "sirs_temp_hi": 38.0,
    "sirs_temp_lo": 36.0,
    "sirs_hr": 90,
    "sirs_rr": 20,
    "sirs_paco2": 32,
    "sirs_wbc_hi": 12.0,
    "sirs_wbc_lo": 4.0,
    "sirs_min_count": 2,
    "shock_map": 65,
    "shock_sbp": 90,
    "shock_lactate": 2.0,
    "shock_urine_24h_ml": 500.0,
    "shock_urine_6h_mlkg": 0.5,
    "resp_pf": 300.0,
    "resp_sf": 315.0,
    "resp_paco2": 45.0,
    "resp_ph": 7.35,
    "resp_spo2": 90.0,
    "renal_creatinine_delta": 0.3,
    "renal_creatinine_ratio": 1.5,
    "renal_creatinine_abs": 2.0,
    "renal_urine_24h_ml": 500.0,
    "renal_urine_6h_mlkg": 0.5,
    "hep_bilirubin": 2.0,
    "hep_ast": 1000.0,
    "hep_alt": 1000.0,
    "coag_platelets": 100.0,
    "coag_inr": 1.5,
    "neuro_gcs": 13,
    "troponin_t_fallback": 0.1,
    "troponin_i_fallback": 0.4,
    "metab_ph": 7.30,
    "metab_hco3": 18.0,
    "metab_anion_gap": 16.0,
    "metab_lactate": 4.0,
    "metab_glucose_lo": 70.0,
    "metab_glucose_hi": 180.0,
}

PHYSIONET_DEFAULT_THRESHOLDS = {
    "severity_map": 70,
    "severity_sysabp": 100,
    "severity_gcs": 15,
    "severity_resprate": 22,
    "severity_sao2": 92,
    "severity_lact": 2.0,
    "severity_ph": 7.30,
    "severity_hco3": 18,
    "severity_creat": 2.0,
    "severity_min_count": 2,
    "shock_map": 65,
    "shock_sysabp": 90,
    "shock_lact": 2.0,
    "shock_urine_sum": 500,
    "resp_pf": 300,
    "resp_sao2": 90,
    "renal_creat": 2.0,
    "renal_bun": 40,
    "renal_urine_sum": 500,
    "hep_bili": 2.0,
    "hep_ast": 100,
    "hep_alt": 100,
    "heme_plts": 100,
    "heme_hct": 30,
    "inflam_wbc_hi": 12,
    "inflam_wbc_lo": 4,
    "inflam_temp_hi": 38.3,
    "inflam_temp_lo": 36,
    "neuro_gcs": 13,
    "card_tropi": 0.4,
    "card_tropt": 0.1,
    "metab_ph_lo": 7.30,
    "metab_ph_hi": 7.50,
    "metab_glu_lo": 70,
    "metab_glu_hi": 180,
    "metab_hco3_lo": 18,
    "chronic_age": 65,
    "acute_lact": 2.0,
    "acute_map": 65,
    "acute_gcs": 13,
}

DEFAULT_THRESHOLDS = {
    "mimic": MIMIC_DEFAULT_THRESHOLDS,
    "physionet": PHYSIONET_DEFAULT_THRESHOLDS,
}

KNOWN_LATENTS = {
    "mimic": {
        "ChronicBurden",
        "AcuteInsult",
        "Severity",
        "Inflammation",
        "Shock",
        "RespFail",
        "RenalDysfunction",
        "HepaticDysfunction",
        "CoagDysfunction",
        "NeuroDysfunction",
        "CardiacInjury",
        "MetabolicDerangement",
    },
    "physionet": {
        "Severity",
        "Shock",
        "RespFail",
        "RenalFail",
        "HepFail",
        "HemeFail",
        "Inflam",
        "NeuroFail",
        "CardInj",
        "Metab",
        "ChronicRisk",
        "AcuteInsult",
    },
}


# ---------------------------------------------------------------------------
# Plot model
# ---------------------------------------------------------------------------

@dataclass
class PlotNode:
    node_id: str
    label: str
    level: int
    kind: str


@dataclass
class PlotEdge:
    source: str
    target: str
    label: str = ""


@dataclass
class PlotSpec:
    title: str
    nodes: List[PlotNode]
    edges: List[PlotEdge]


NODE_STYLES = {
    "condition": {"facecolor": "#E7F0FA", "edgecolor": "#4C78A8"},
    "domain": {"facecolor": "#E8F7EE", "edgecolor": "#4E9F71"},
    "combine": {"facecolor": "#FFF1CC", "edgecolor": "#B58900"},
    "terminal_true": {"facecolor": "#DFF5E1", "edgecolor": "#2E7D32"},
    "terminal_false": {"facecolor": "#FDE7E9", "edgecolor": "#C62828"},
    "fallback": {"facecolor": "#F3F4F6", "edgecolor": "#6B7280"},
}
BOX_PAD = 0.012
AXIS_PAD_X = 0.05
AXIS_PAD_Y = 0.06
EDGE_ZORDER = 1
NODE_ZORDER = 3
TEXT_ZORDER = 4
ARROW_TARGET_GAP = 0.01


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot explicit rule diagrams for latent decision-tree pickle files. "
            "The pickle is expected to store dict[latent_name] = callable."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["mimic", "physionet"])
    parser.add_argument(
        "--dataset-config-csv",
        default=None,
        help=(
            "Path to the dataset global-variables CSV. If omitted, use the default "
            "config for --dataset."
        ),
    )
    parser.add_argument("--pickle-path", required=True, help="Path to the saved latent decision-tree pickle.")
    parser.add_argument(
        "--output-dir",
        default="decision_tree_figures",
        help="Directory where one figure per latent will be written.",
    )
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fig-width", type=float, default=None)
    parser.add_argument("--fig-height", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, help="Plot only the selected latent names.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def validate_args(args: argparse.Namespace) -> None:
    pickle_path = Path(args.pickle_path)
    if not pickle_path.exists():
        fail(f"pickle file does not exist: {pickle_path}")
    if not pickle_path.is_file():
        fail(f"pickle path is not a file: {pickle_path}")
    if args.dpi <= 0:
        fail("--dpi must be a positive integer.")
    if args.fig_width is not None and args.fig_width <= 0:
        fail("--fig-width must be positive when provided.")
    if args.fig_height is not None and args.fig_height <= 0:
        fail("--fig-height must be positive when provided.")


def wrap_label(text: str, width: int = 24) -> str:
    wrapped_parts = []
    for part in text.splitlines():
        if not part:
            wrapped_parts.append("")
            continue
        wrapped_parts.append(
            textwrap.fill(
                part,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped_parts)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, set):
        return "{" + ", ".join(sorted(str(item) for item in value)) + "}"
    return str(value)


def safe_filename(dataset: str, latent_name: str, output_format: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", latent_name).strip("._")
    return f"{dataset}_{cleaned or 'latent'}.{output_format}"


def describe_callable(obj: Any) -> str:
    if isinstance(obj, partial):
        func_name = getattr(obj.func, "__name__", repr(obj.func))
        return f"functools.partial({func_name})"
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    return repr(obj)


def extract_thresholds(dataset: str, obj: Any) -> Dict[str, Any]:
    thresholds = dict(DEFAULT_THRESHOLDS[dataset])
    if isinstance(obj, partial):
        partial_keywords = getattr(obj, "keywords", {}) or {}
        partial_thresholds = partial_keywords.get("thr")
        if isinstance(partial_thresholds, dict):
            thresholds.update(partial_thresholds)
    return thresholds


# ---------------------------------------------------------------------------
# Pickle loading
# ---------------------------------------------------------------------------

def load_pickle_dict(pickle_path: Path) -> Dict[str, Any]:
    try:
        with pickle_path.open("rb") as handle:
            payload = pickle.load(handle)
    except FileNotFoundError:
        fail(f"pickle file does not exist: {pickle_path}")
    except pickle.UnpicklingError as exc:
        fail(f"pickle file is corrupted or not a valid pickle: {pickle_path} ({exc})")
    except AttributeError as exc:
        fail(
            "pickle references callable names that are not supported by this utility. "
            f"Expected latent-tagging callables from the known scripts. Original error: {exc}"
        )
    except Exception as exc:
        fail(f"failed to load pickle file {pickle_path}: {exc}")

    if not isinstance(payload, dict):
        fail(f"pickle must contain a dict of latent_name -> callable, got {type(payload).__name__}.")
    if not payload:
        fail("pickle dict is empty; no latent names were found.")
    return payload


# ---------------------------------------------------------------------------
# Plot spec builders
# ---------------------------------------------------------------------------

def make_or_spec(title: str, conditions: Sequence[str]) -> PlotSpec:
    nodes: List[PlotNode] = []
    edges: List[PlotEdge] = []

    for index, label in enumerate(conditions):
        node_id = f"c{index}"
        nodes.append(PlotNode(node_id=node_id, label=label, level=0, kind="condition"))
        edges.append(PlotEdge(source=node_id, target="or"))

    nodes.append(PlotNode(node_id="or", label="OR", level=1, kind="combine"))
    nodes.append(PlotNode(node_id="true", label=f"{title} = 1", level=2, kind="terminal_true"))
    nodes.append(PlotNode(node_id="false", label=f"{title} = 0", level=2, kind="terminal_false"))
    edges.append(PlotEdge(source="or", target="true", label="yes"))
    edges.append(PlotEdge(source="or", target="false", label="no"))
    return PlotSpec(title=title, nodes=nodes, edges=edges)


def make_single_rule_spec(title: str, condition: str) -> PlotSpec:
    nodes = [
        PlotNode(node_id="c0", label=condition, level=0, kind="condition"),
        PlotNode(node_id="true", label=f"{title} = 1", level=1, kind="terminal_true"),
        PlotNode(node_id="false", label=f"{title} = 0", level=1, kind="terminal_false"),
    ]
    edges = [
        PlotEdge(source="c0", target="true", label="yes"),
        PlotEdge(source="c0", target="false", label="no"),
    ]
    return PlotSpec(title=title, nodes=nodes, edges=edges)


def make_nested_or_spec(
    title: str,
    direct_conditions: Sequence[str],
    grouped_conditions: Sequence[Dict[str, Any]],
) -> PlotSpec:
    nodes: List[PlotNode] = []
    edges: List[PlotEdge] = []

    for index, label in enumerate(direct_conditions):
        node_id = f"direct_{index}"
        nodes.append(PlotNode(node_id=node_id, label=label, level=0, kind="condition"))
        edges.append(PlotEdge(source=node_id, target="or"))

    for group_index, group in enumerate(grouped_conditions):
        group_node_id = f"group_{group_index}"
        for condition_index, label in enumerate(group["conditions"]):
            node_id = f"{group_node_id}_c{condition_index}"
            nodes.append(PlotNode(node_id=node_id, label=label, level=0, kind="condition"))
            edges.append(PlotEdge(source=node_id, target=group_node_id))

        nodes.append(
            PlotNode(
                node_id=group_node_id,
                label=group["combine_label"],
                level=1,
                kind="combine",
            )
        )
        edges.append(PlotEdge(source=group_node_id, target="or"))

    nodes.append(PlotNode(node_id="or", label="OR", level=2, kind="combine"))
    nodes.append(PlotNode(node_id="true", label=f"{title} = 1", level=3, kind="terminal_true"))
    nodes.append(PlotNode(node_id="false", label=f"{title} = 0", level=3, kind="terminal_false"))
    edges.append(PlotEdge(source="or", target="true", label="yes"))
    edges.append(PlotEdge(source="or", target="false", label="no"))
    return PlotSpec(title=title, nodes=nodes, edges=edges)


def make_score_spec(title: str, domain_boxes: Sequence[str], score_label: str, decision_label: str) -> PlotSpec:
    nodes: List[PlotNode] = []
    edges: List[PlotEdge] = []

    for index, label in enumerate(domain_boxes):
        node_id = f"d{index}"
        nodes.append(PlotNode(node_id=node_id, label=label, level=0, kind="domain"))
        edges.append(PlotEdge(source=node_id, target="score"))

    nodes.append(PlotNode(node_id="score", label=score_label, level=1, kind="combine"))
    nodes.append(PlotNode(node_id="decision", label=decision_label, level=2, kind="combine"))
    nodes.append(PlotNode(node_id="true", label=f"{title} = 1", level=3, kind="terminal_true"))
    nodes.append(PlotNode(node_id="false", label=f"{title} = 0", level=3, kind="terminal_false"))
    edges.append(PlotEdge(source="score", target="decision"))
    edges.append(PlotEdge(source="decision", target="true", label="yes"))
    edges.append(PlotEdge(source="decision", target="false", label="no"))
    return PlotSpec(title=title, nodes=nodes, edges=edges)


def make_fallback_spec(title: str, callable_text: str) -> PlotSpec:
    message = (
        f"No explicit plotting spec available for {title}.\n"
        f"Pickle contains callable: {callable_text}"
    )
    nodes = [PlotNode(node_id="fallback", label=message, level=0, kind="fallback")]
    return PlotSpec(title=title, nodes=nodes, edges=[])


# ---------------------------------------------------------------------------
# Dataset-specific plotting specs
# ---------------------------------------------------------------------------

def mimic_plot_spec(latent_name: str, thresholds: Dict[str, Any]) -> PlotSpec | None:
    if latent_name == "ChronicBurden":
        return make_or_spec(
            latent_name,
            [
                f"Age >= {format_value(thresholds['chronic_age'])}",
                "ChronicICD_any == 1",
                f"Albumin_first < {format_value(thresholds['chronic_albumin'])}",
            ],
        )

    if latent_name == "AcuteInsult":
        return make_or_spec(
            latent_name,
            [
                f"AdmissionType in {format_value(thresholds['acute_emergency_types'])}",
                "AcuteICD_any == 1",
                "MechanicalVentilation_any == 1",
                "Vasopressors_any == 1",
            ],
        )

    if latent_name == "Severity":
        return make_or_spec(
            latent_name,
            [
                "Vasopressors_any == 1",
                "MechanicalVentilation_any == 1",
                f"Lactate_max >= {format_value(thresholds['severity_lactate'])}",
                f"pH_min <= {format_value(thresholds['severity_ph'])}",
                f"GCS_min <= {format_value(thresholds['severity_gcs'])}",
                f"Platelets_min < {format_value(thresholds['severity_platelets'])}",
                f"Creatinine_max >= {format_value(thresholds['severity_creatinine'])}",
                f"Bilirubin_max >= {format_value(thresholds['severity_bilirubin'])}",
            ],
        )

    if latent_name == "Inflammation":
        return make_or_spec(
            latent_name,
            [
                "SuspectedInfection_any == 1",
                (
                    f"SIRS_count_max >= {format_value(thresholds['sirs_min_count'])}\n"
                    "where SIRS_count uses temp, HR, RR/PaCO2, and WBC criteria"
                ),
            ],
        )

    if latent_name == "Shock":
        return make_or_spec(
            latent_name,
            [
                "Vasopressors_any == 1",
                f"MAP_min < {format_value(thresholds['shock_map'])}",
                f"SBP_min < {format_value(thresholds['shock_sbp'])}",
                f"Lactate_max > {format_value(thresholds['shock_lactate'])}",
                f"UrineOutput_sum_24h < {format_value(thresholds['shock_urine_24h_ml'])}",
                f"UrineOutput_mlkg_6h_min < {format_value(thresholds['shock_urine_6h_mlkg'])}",
            ],
        )

    if latent_name == "RespFail":
        return make_nested_or_spec(
            latent_name,
            direct_conditions=[
                "MechanicalVentilation_any == 1",
                f"PF_ratio_min < {format_value(thresholds['resp_pf'])}",
                f"SF_ratio_min < {format_value(thresholds['resp_sf'])}",
                f"SpO2_min < {format_value(thresholds['resp_spo2'])}",
            ],
            grouped_conditions=[
                {
                    "combine_label": "AND",
                    "conditions": [
                        f"PaCO2_max >= {format_value(thresholds['resp_paco2'])}",
                        f"pH_min < {format_value(thresholds['resp_ph'])}",
                    ],
                }
            ],
        )

    if latent_name == "RenalDysfunction":
        return make_or_spec(
            latent_name,
            [
                f"Creatinine_delta >= {format_value(thresholds['renal_creatinine_delta'])}",
                f"Creatinine_ratio >= {format_value(thresholds['renal_creatinine_ratio'])}",
                f"Creatinine_max >= {format_value(thresholds['renal_creatinine_abs'])}",
                f"UrineOutput_mlkg_6h_min < {format_value(thresholds['renal_urine_6h_mlkg'])}",
                f"UrineOutput_sum_24h < {format_value(thresholds['renal_urine_24h_ml'])}",
            ],
        )

    if latent_name == "HepaticDysfunction":
        return make_or_spec(
            latent_name,
            [
                f"Bilirubin_max >= {format_value(thresholds['hep_bilirubin'])}",
                f"AST_max >= {format_value(thresholds['hep_ast'])}",
                f"ALT_max >= {format_value(thresholds['hep_alt'])}",
            ],
        )

    if latent_name == "CoagDysfunction":
        return make_or_spec(
            latent_name,
            [
                f"Platelets_min < {format_value(thresholds['coag_platelets'])}",
                f"INR_max >= {format_value(thresholds['coag_inr'])}",
            ],
        )

    if latent_name == "NeuroDysfunction":
        return make_single_rule_spec(
            latent_name,
            f"GCS_min < {format_value(thresholds['neuro_gcs'])}",
        )

    if latent_name == "CardiacInjury":
        return make_or_spec(
            latent_name,
            [
                "TroponinPositive_any == 1",
                f"TroponinT_max >= {format_value(thresholds['troponin_t_fallback'])}",
                f"TroponinI_max >= {format_value(thresholds['troponin_i_fallback'])}",
            ],
        )

    if latent_name == "MetabolicDerangement":
        return make_or_spec(
            latent_name,
            [
                f"pH_min < {format_value(thresholds['metab_ph'])}",
                f"Bicarbonate_min < {format_value(thresholds['metab_hco3'])}",
                f"AnionGap_max > {format_value(thresholds['metab_anion_gap'])}",
                f"Lactate_max >= {format_value(thresholds['metab_lactate'])}",
                f"Glucose_min < {format_value(thresholds['metab_glucose_lo'])}",
                f"Glucose_max >= {format_value(thresholds['metab_glucose_hi'])}",
            ],
        )

    return None


def physionet_plot_spec(latent_name: str, thresholds: Dict[str, Any]) -> PlotSpec | None:
    if latent_name == "Severity":
        return make_score_spec(
            latent_name,
            domain_boxes=[
                (
                    f"Circulatory domain\n"
                    f"MAP_first < {format_value(thresholds['severity_map'])} OR\n"
                    f"SysABP_first <= {format_value(thresholds['severity_sysabp'])}"
                ),
                f"Neurologic domain\nGCS_first < {format_value(thresholds['severity_gcs'])}",
                (
                    f"Respiratory domain\n"
                    f"RespRate_first >= {format_value(thresholds['severity_resprate'])} OR\n"
                    f"SaO2_first < {format_value(thresholds['severity_sao2'])}"
                ),
                (
                    f"Metabolic domain\n"
                    f"Lactate_first > {format_value(thresholds['severity_lact'])} OR\n"
                    f"pH_first < {format_value(thresholds['severity_ph'])} OR\n"
                    f"HCO3_first < {format_value(thresholds['severity_hco3'])}"
                ),
                f"Renal domain\nCreatinine_first >= {format_value(thresholds['severity_creat'])}",
            ],
            score_label="score = sum(domain indicators)",
            decision_label=f"score >= {format_value(thresholds['severity_min_count'])}",
        )

    if latent_name == "Shock":
        return make_or_spec(
            latent_name,
            [
                f"MAP_min < {format_value(thresholds['shock_map'])}",
                f"SysABP_min < {format_value(thresholds['shock_sysabp'])}",
                f"Lactate_max > {format_value(thresholds['shock_lact'])}",
                f"Urine_sum < {format_value(thresholds['shock_urine_sum'])}",
            ],
        )

    if latent_name == "RespFail":
        return make_or_spec(
            latent_name,
            [
                f"PF_ratio = PaO2_min / FiO2_min < {format_value(thresholds['resp_pf'])}",
                f"SaO2_min < {format_value(thresholds['resp_sao2'])}",
                "MechVent_max == 1",
            ],
        )

    if latent_name == "RenalFail":
        return make_or_spec(
            latent_name,
            [
                f"Creatinine_max > {format_value(thresholds['renal_creat'])}",
                f"BUN_max > {format_value(thresholds['renal_bun'])}",
                f"Urine_sum < {format_value(thresholds['renal_urine_sum'])}",
            ],
        )

    if latent_name == "HepFail":
        return make_or_spec(
            latent_name,
            [
                f"Bilirubin_max > {format_value(thresholds['hep_bili'])}",
                f"AST_max > {format_value(thresholds['hep_ast'])}",
                f"ALT_max > {format_value(thresholds['hep_alt'])}",
            ],
        )

    if latent_name == "HemeFail":
        return make_or_spec(
            latent_name,
            [
                f"Platelets_min < {format_value(thresholds['heme_plts'])}",
                f"HCT_min < {format_value(thresholds['heme_hct'])}",
            ],
        )

    if latent_name == "Inflam":
        return make_or_spec(
            latent_name,
            [
                f"WBC_max > {format_value(thresholds['inflam_wbc_hi'])}",
                f"WBC_min < {format_value(thresholds['inflam_wbc_lo'])}",
                f"Temp_max > {format_value(thresholds['inflam_temp_hi'])}",
                f"Temp_min < {format_value(thresholds['inflam_temp_lo'])}",
            ],
        )

    if latent_name == "NeuroFail":
        return make_single_rule_spec(
            latent_name,
            f"GCS_min < {format_value(thresholds['neuro_gcs'])}",
        )

    if latent_name == "CardInj":
        return make_or_spec(
            latent_name,
            [
                f"TropI_max > {format_value(thresholds['card_tropi'])}",
                f"TropT_max > {format_value(thresholds['card_tropt'])}",
            ],
        )

    if latent_name == "Metab":
        return make_or_spec(
            latent_name,
            [
                f"pH_min < {format_value(thresholds['metab_ph_lo'])}",
                f"pH_max > {format_value(thresholds['metab_ph_hi'])}",
                f"Glucose_min < {format_value(thresholds['metab_glu_lo'])}",
                f"Glucose_max > {format_value(thresholds['metab_glu_hi'])}",
                f"HCO3_min < {format_value(thresholds['metab_hco3_lo'])}",
            ],
        )

    if latent_name == "ChronicRisk":
        return make_or_spec(
            latent_name,
            [
                f"Age_first > {format_value(thresholds['chronic_age'])}",
                "ICUType_first in {2, 3}",
            ],
        )

    if latent_name == "AcuteInsult":
        return make_or_spec(
            latent_name,
            [
                f"Lactate_first > {format_value(thresholds['acute_lact'])}",
                f"MAP_first < {format_value(thresholds['acute_map'])}",
                f"GCS_first < {format_value(thresholds['acute_gcs'])}",
            ],
        )

    return None


SPEC_BUILDERS = {
    "mimic": mimic_plot_spec,
    "physionet": physionet_plot_spec,
}


def build_plot_spec(dataset: str, latent_name: str, obj: Any) -> PlotSpec:
    thresholds = extract_thresholds(dataset, obj)
    explicit_spec = SPEC_BUILDERS[dataset](latent_name, thresholds)
    if explicit_spec is not None:
        return explicit_spec
    return make_fallback_spec(latent_name, describe_callable(obj))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def infer_figure_size(spec: PlotSpec, fig_width: float | None, fig_height: float | None) -> tuple[float, float]:
    levels = defaultdict(int)
    for node in spec.nodes:
        levels[node.level] += 1

    max_nodes_in_level = max(levels.values()) if levels else 1
    auto_width = max(8.0, min(18.0, 2.4 * max_nodes_in_level))
    auto_height = max(5.5, 2.0 + 1.65 * (len(levels) + 1))
    return fig_width or auto_width, fig_height or auto_height


def compute_layout(spec: PlotSpec) -> Dict[str, tuple[float, float, float, float]]:
    levels: Dict[int, List[PlotNode]] = defaultdict(list)
    for node in spec.nodes:
        levels[node.level].append(node)

    level_ids = sorted(levels)
    n_levels = len(level_ids)
    layout: Dict[str, tuple[float, float, float, float]] = {}

    if n_levels == 1:
        vertical_positions = [0.55]
    else:
        vertical_positions = [
            0.88 - (index * (0.76 / max(n_levels - 1, 1)))
            for index in range(n_levels)
        ]

    for level_index, level_id in enumerate(level_ids):
        nodes = levels[level_id]
        count = len(nodes)
        if count == 1:
            x_positions = [0.5]
        else:
            x_positions = [
                0.08 + (index * (0.84 / max(count - 1, 1)))
                for index in range(count)
            ]

        box_width = min(0.22, max(0.12, 0.80 / max(count, 1)))
        for x_pos, node in zip(x_positions, nodes):
            wrapped_label = wrap_label(node.label, width=24)
            line_count = max(1, len(wrapped_label.splitlines()))
            box_height = min(0.22, max(0.08, 0.04 + line_count * 0.022))
            layout[node.node_id] = (x_pos, vertical_positions[level_index], box_width, box_height)

    return layout


def compute_layout_extents(
    layout: Dict[str, tuple[float, float, float, float]]
) -> tuple[float, float, float, float]:
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for x_center, y_center, width, height in layout.values():
        min_x = min(min_x, x_center - width / 2.0 - BOX_PAD)
        max_x = max(max_x, x_center + width / 2.0 + BOX_PAD)
        min_y = min(min_y, y_center - height / 2.0 - BOX_PAD)
        max_y = max(max_y, y_center + height / 2.0 + BOX_PAD)

    return min_x, max_x, min_y, max_y


def draw_node(ax: plt.Axes, node: PlotNode, position: tuple[float, float, float, float]) -> None:
    x_center, y_center, width, height = position
    style = NODE_STYLES[node.kind]

    patch = FancyBboxPatch(
        (x_center - width / 2.0, y_center - height / 2.0),
        width,
        height,
        boxstyle=f"round,pad={BOX_PAD},rounding_size=0.02",
        linewidth=1.6,
        facecolor=style["facecolor"],
        edgecolor=style["edgecolor"],
        zorder=NODE_ZORDER,
    )
    ax.add_patch(patch)
    ax.text(
        x_center,
        y_center,
        wrap_label(node.label, width=24),
        ha="center",
        va="center",
        fontsize=10,
        zorder=TEXT_ZORDER,
    )


def draw_edge(ax: plt.Axes, edge: PlotEdge, layout: Dict[str, tuple[float, float, float, float]]) -> None:
    source_x, source_y, _, _ = layout[edge.source]
    target_x, target_y, target_w, target_h = layout[edge.target]

    dx = target_x - source_x
    dy = target_y - source_y
    distance = (dx ** 2 + dy ** 2) ** 0.5
    if distance == 0:
        return

    ux = dx / distance
    uy = dy / distance

    target_half_w = target_w / 2.0 + BOX_PAD
    target_half_h = target_h / 2.0 + BOX_PAD
    tx = float("inf") if abs(ux) < 1e-9 else target_half_w / abs(ux)
    ty = float("inf") if abs(uy) < 1e-9 else target_half_h / abs(uy)
    target_boundary_distance = min(tx, ty)

    start = (source_x, source_y)
    end = (
        target_x - ux * (target_boundary_distance + ARROW_TARGET_GAP),
        target_y - uy * (target_boundary_distance + ARROW_TARGET_GAP),
    )

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.3,
            "color": "#4B5563",
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=EDGE_ZORDER,
    )

    if edge.label:
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        ax.text(
            mid_x,
            mid_y,
            edge.label,
            fontsize=9,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            zorder=TEXT_ZORDER,
        )


def save_plot(spec: PlotSpec, output_path: Path, dpi: int, fig_width: float | None, fig_height: float | None) -> None:
    width, height = infer_figure_size(spec, fig_width, fig_height)
    fig, ax = plt.subplots(figsize=(width, height))
    layout = compute_layout(spec)
    min_x, max_x, min_y, max_y = compute_layout_extents(layout)

    fig.suptitle(spec.title, fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.90)

    ax.set_xlim(min_x - AXIS_PAD_X, max_x + AXIS_PAD_X)
    ax.set_ylim(min_y - AXIS_PAD_Y, max_y + AXIS_PAD_Y)
    ax.axis("off")

    for edge in spec.edges:
        draw_edge(ax, edge, layout)

    for node in spec.nodes:
        draw_node(ax, node, layout[node.node_id])

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def select_latents(payload: Dict[str, Any], requested_latents: Iterable[str] | None) -> List[str]:
    available_latents = list(payload.keys())
    if requested_latents is None:
        return available_latents

    missing = [latent for latent in requested_latents if latent not in payload]
    if missing:
        fail(
            "selected latent names were not found in the pickle: "
            f"{missing}. Available latent names: {available_latents}"
        )
    return list(requested_latents)


def ensure_output_paths(latents: Sequence[str], dataset: str, output_dir: Path, output_format: str, overwrite: bool) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / safe_filename(dataset, latent, output_format) for latent in latents]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        fail(
            "output files already exist. Use --overwrite to replace them or choose another --output-dir. "
            f"Existing paths: {[str(path) for path in existing]}"
        )
    return paths


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_dataset_config(args.dataset, args.dataset_config_csv)
    configured_thresholds = get_config_scalar(config, "DEFAULT_THRESHOLDS", None)
    if isinstance(configured_thresholds, dict):
        DEFAULT_THRESHOLDS[args.dataset] = dict(configured_thresholds)
        if isinstance(
            DEFAULT_THRESHOLDS[args.dataset].get("acute_emergency_types"),
            list,
        ):
            DEFAULT_THRESHOLDS[args.dataset]["acute_emergency_types"] = set(
                DEFAULT_THRESHOLDS[args.dataset]["acute_emergency_types"]
            )
    configured_latents = get_config_list(config, "LATENT_ORDER", None)
    if configured_latents:
        KNOWN_LATENTS[args.dataset] = set(str(latent) for latent in configured_latents)

    pickle_path = Path(args.pickle_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    payload = load_pickle_dict(pickle_path)
    selected_latents = select_latents(payload, args.only)
    output_paths = ensure_output_paths(
        latents=selected_latents,
        dataset=args.dataset,
        output_dir=output_dir,
        output_format=args.format,
        overwrite=args.overwrite,
    )

    explicit_count = 0
    saved_paths: List[Path] = []

    for latent_name, output_path in zip(selected_latents, output_paths):
        plot_spec = build_plot_spec(args.dataset, latent_name, payload[latent_name])
        if latent_name in KNOWN_LATENTS[args.dataset]:
            explicit_count += 1
        save_plot(
            spec=plot_spec,
            output_path=output_path,
            dpi=args.dpi,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
        )
        saved_paths.append(output_path)

    print(f"Loaded {len(payload)} latent callables from: {pickle_path}")
    print(f"Selected latents ({len(selected_latents)}): {', '.join(selected_latents)}")
    if explicit_count == 0:
        print(
            "Warning: none of the selected latent names matched an explicit plotting spec; "
            "fallback figures were generated."
        )
    print(f"Saved {len(saved_paths)} figure(s) under: {output_dir}")
    for path in saved_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("interrupted by user.")
