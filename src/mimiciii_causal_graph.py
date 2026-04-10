
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_GRAPH_PKL_PATH = "../data/mimiciii_causal_graph.pkl"
DEFAULT_GRAPH_PNG_PATH = "../data/mimiciii_causal_dag.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the MIMIC-III causal DAG and save the graph artifacts."
    )
    parser.add_argument(
        "--graph-pkl-path",
        default=DEFAULT_GRAPH_PKL_PATH,
        help=f"Output path for the graph pickle. Default: {DEFAULT_GRAPH_PKL_PATH}",
    )
    parser.add_argument(
        "--graph-png-path",
        default=DEFAULT_GRAPH_PNG_PATH,
        help=f"Output path for the rendered graph PNG. Default: {DEFAULT_GRAPH_PNG_PATH}",
    )
    return parser.parse_args()


def resolve_output_path(path_like: str) -> str:
    raw_path = path_like.strip()
    if not raw_path:
        raise ValueError("Output path must be a non-empty string.")
    return os.path.abspath(os.path.expanduser(raw_path))


def create_mimiciii_causal_graph(
    save=0,
    graph_pkl_path: str | None = None,
) -> nx.DiGraph:
    """
    Research-ready causal DAG for a clinically aggregated subset of MIMIC-III.
    This graph models background variables, latent physiologic states, observed
    vitals/labs/interventions, and mortality.

    Notes
    -----
    - This is not a DAG over all raw MIMIC-III ITEMIDs.
    - It is a compact causal abstraction over commonly used aggregated variables.
    - Edges encode clinically plausible causal directions for causal-inference work.
    """
    G = nx.DiGraph()

    # ------------------------------------------------------------------
    # Observed background variables
    # ------------------------------------------------------------------
    background_vars = [
        "Age", "Gender", "Ethnicity", "BMI", "AdmissionType", "FirstCareUnit"
    ]

    # ------------------------------------------------------------------
    # Latent variables
    # ------------------------------------------------------------------
    latent_vars = [
        "ChronicBurden",       # baseline chronic disease burden / physiologic reserve
        "AcuteInsult",         # acute diagnosis / precipitating insult
        "Severity",            # overall illness severity
        "Inflammation",        # infection / inflammatory burden
        "Shock",               # hemodynamic failure / hypoperfusion
        "RespFail",            # respiratory failure
        "RenalDysfunction",    # kidney dysfunction
        "HepaticDysfunction",  # liver dysfunction
        "CoagDysfunction",     # coagulation / hematologic dysfunction
        "NeuroDysfunction",    # neurologic dysfunction
        "CardiacInjury",       # myocardial injury / cardiac dysfunction
        "MetabolicDerangement" # acid-base / glucose / metabolic stress
    ]

    # ------------------------------------------------------------------
    # Observed variables (clinically aggregated MIMIC-style variables)
    # ------------------------------------------------------------------
    observed_vars = [
        # Hemodynamics / perfusion
        "HR", "SBP", "DBP", "MAP", "Lactate", "UrineOutput",

        # Respiratory
        "RR", "SpO2", "PaO2", "PaCO2", "pH", "FiO2", "MechanicalVentilation",

        # Renal / electrolytes
        "Creatinine", "BUN", "Sodium", "Potassium", "Bicarbonate",

        # Hepatic
        "AST", "ALT", "Bilirubin", "Albumin",

        # Hematologic / inflammatory
        "Platelets", "WBC", "Temperature",

        # Neurologic
        "GCS",

        # Cardiac
        "Troponin",

        # Metabolic
        "Glucose", "AnionGap",

        # ICU interventions
        "Vasopressors", "FluidBolus",

        # Outcome
        "InHospitalMortality",
    ]

    G.add_nodes_from(background_vars, node_type="background")
    G.add_nodes_from(latent_vars, node_type="latent")
    G.add_nodes_from(observed_vars, node_type="observed")

    # ------------------------------------------------------------------
    # Background -> latent baseline / context
    # ------------------------------------------------------------------
    for var in ["Age", "Gender", "Ethnicity", "BMI"]:
        G.add_edge(var, "ChronicBurden")

    for var in ["AdmissionType", "FirstCareUnit"]:
        G.add_edge(var, "AcuteInsult")

    G.add_edge("ChronicBurden", "Severity")
    G.add_edge("AcuteInsult", "Severity")
    G.add_edge("AcuteInsult", "Inflammation")

    # ------------------------------------------------------------------
    # Severity -> organ/system states
    # ------------------------------------------------------------------
    for state in [
        "Inflammation", "Shock", "RespFail", "RenalDysfunction",
        "HepaticDysfunction", "CoagDysfunction", "NeuroDysfunction",
        "CardiacInjury", "MetabolicDerangement"
    ]:
        G.add_edge("Severity", state)

    # ------------------------------------------------------------------
    # Clinically plausible cross-organ links
    # ------------------------------------------------------------------
    G.add_edge("Inflammation", "Shock")
    G.add_edge("Inflammation", "CoagDysfunction")
    G.add_edge("Inflammation", "MetabolicDerangement")

    G.add_edge("Shock", "RenalDysfunction")
    G.add_edge("Shock", "HepaticDysfunction")
    G.add_edge("Shock", "MetabolicDerangement")

    G.add_edge("RespFail", "MetabolicDerangement")
    G.add_edge("RenalDysfunction", "MetabolicDerangement")
    G.add_edge("CardiacInjury", "Shock")

    # ------------------------------------------------------------------
    # Latent -> observed measurements
    # ------------------------------------------------------------------
    # Inflammation
    G.add_edge("Inflammation", "WBC")
    G.add_edge("Inflammation", "Temperature")

    # Shock / perfusion
    for var in ["SBP", "DBP", "MAP", "HR", "Lactate", "UrineOutput", "Vasopressors", "FluidBolus"]:
        G.add_edge("Shock", var)

    # Respiratory failure
    for var in ["RR", "SpO2", "PaO2", "PaCO2", "pH", "FiO2", "MechanicalVentilation"]:
        G.add_edge("RespFail", var)

    # Renal dysfunction
    for var in ["Creatinine", "BUN", "UrineOutput", "Potassium", "Sodium", "Bicarbonate", "pH", "AnionGap"]:
        G.add_edge("RenalDysfunction", var)

    # Hepatic dysfunction
    for var in ["AST", "ALT", "Bilirubin", "Albumin"]:
        G.add_edge("HepaticDysfunction", var)

    # Coagulation dysfunction
    G.add_edge("CoagDysfunction", "Platelets")

    # Neurologic dysfunction
    G.add_edge("NeuroDysfunction", "GCS")

    # Cardiac injury
    G.add_edge("CardiacInjury", "Troponin")

    # Metabolic derangement
    for var in ["Lactate", "Glucose", "Bicarbonate", "pH", "AnionGap"]:
        G.add_edge("MetabolicDerangement", var)

    # ------------------------------------------------------------------
    # Interventions affect downstream measurements
    # ------------------------------------------------------------------
    G.add_edge("MechanicalVentilation", "PaO2")
    G.add_edge("MechanicalVentilation", "PaCO2")
    G.add_edge("MechanicalVentilation", "pH")
    G.add_edge("MechanicalVentilation", "SpO2")

    G.add_edge("FiO2", "PaO2")
    G.add_edge("FiO2", "SpO2")

    G.add_edge("Vasopressors", "MAP")
    G.add_edge("Vasopressors", "SBP")
    G.add_edge("FluidBolus", "MAP")
    G.add_edge("FluidBolus", "UrineOutput")

    # ------------------------------------------------------------------
    # Outcome
    # ------------------------------------------------------------------
    G.add_edge("Severity", "InHospitalMortality")
    G.add_edge("Age", "InHospitalMortality")
    for state in [
        "Shock", "RespFail", "RenalDysfunction", "HepaticDysfunction",
        "CoagDysfunction", "NeuroDysfunction", "CardiacInjury",
        "MetabolicDerangement"
    ]:
        G.add_edge(state, "InHospitalMortality")

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Constructed graph is not a DAG")

    print(
        f"      Built MIMIC-III DAG with {G.number_of_nodes()} nodes and "
        f"{G.number_of_edges()} edges."
    )

    if save:
        output_path = resolve_output_path(graph_pkl_path or DEFAULT_GRAPH_PKL_PATH)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"      Saving graph pickle to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(G, f)

    return G


def draw_graph(
    G: nx.DiGraph,
    save=0,
    graph_png_path: str | None = None,
    figsize=(24, 18),
    node_size=1500,
    font_size=8,
):
    color_map = {
        "background": "#9ecae1",
        "latent": "#fdae6b",
        "observed": "#a1d99b",
    }

    background_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "background"]
    latent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "latent"]
    observed_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "observed"]

    pos = {}

    def assign_layer(nodes, y, x_spacing):
        x_offset = -(len(nodes) - 1) * x_spacing / 2
        for i, node in enumerate(nodes):
            pos[node] = (x_offset + i * x_spacing, y)

    assign_layer(background_nodes, y=3.1, x_spacing=2.0)
    assign_layer(latent_nodes, y=2.0, x_spacing=1.7)
    assign_layer(observed_nodes, y=0.9, x_spacing=0.8)

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=15,
        edge_color="gray",
        width=1.1,
        alpha=0.65,
        connectionstyle="arc3,rad=0.04",
        min_source_margin=10,
        min_target_margin=15,
    )

    for group, nodes in [
        ("background", background_nodes),
        ("latent", latent_nodes),
        ("observed", observed_nodes),
    ]:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=color_map[group],
            node_size=node_size,
            edgecolors="black",
        )

    nx.draw_networkx_labels(G, pos, font_size=font_size)

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=11,
        )
        for label, color in color_map.items()
    ]

    plt.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    plt.title("MIMIC-III – Clinically Aggregated Causal DAG", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    if save:
        output_path = resolve_output_path(graph_png_path or DEFAULT_GRAPH_PNG_PATH)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"      Saving DAG figure to: {output_path}")
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    print("=== Building MIMIC-III causal DAG ===")
    print("[1/2] Creating graph structure")
    g = create_mimiciii_causal_graph(
        save=1,
        graph_pkl_path=args.graph_pkl_path,
    )
    print("[2/2] Rendering graph figure")
    draw_graph(
        g,
        save=1,
        graph_png_path=args.graph_png_path,
    )
    print("MIMIC-III causal DAG build completed.")
