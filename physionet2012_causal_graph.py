import matplotlib.pyplot as plt
import networkx as nx
import os


def create_physionet2012_causal_graph() -> nx.DiGraph:
    """
    Create a directed acyclic graph (DAG) representing a clinically-plausible
    causal structure for PhysioNet / CinC Challenge 2012 variables.

    Returns
    -------
    nx.DiGraph
        A directed graph where nodes are variables (observed + latent)
        and edges represent assumed causal relationships.
    """

    G = nx.DiGraph()

    # ------------------------------------------------------------------
    # Observed background variables
    # ------------------------------------------------------------------
    background_vars = [
        "Age", "Gender", "Height", "Weight", "ICUType"
    ]

    # ------------------------------------------------------------------
    # Latent variables (explicitly modeled to keep DAG clean)
    # ------------------------------------------------------------------
    latent_vars = [
        "ChronicRisk",        # Chronic health / baseline risk
        "AcuteInsult",        # Acute insult / diagnosis mix
        "Severity",           # Overall illness severity
        "Shock",              # Hemodynamic failure
        "RespFail",           # Respiratory failure
        "RenalFail",          # Renal dysfunction
        "HepFail",            # Hepatic dysfunction
        "HemeFail",           # Hematologic dysfunction
        "Inflam",             # Inflammation / infection burden
        "NeuroFail",          # Neurologic dysfunction
        "CardInj",            # Cardiac injury
        "Metab",              # Metabolic derangement
    ]

    # ------------------------------------------------------------------
    # Observed PhysioNet 2012 variables
    # ------------------------------------------------------------------
    observed_vars = [
        # Hemodynamics
        "SysABP", "DiasABP", "MAP",
        "NISysABP", "NIDiasABP", "NIMAP",
        "HR",

        # Perfusion / output
        "Lactate", "Urine",

        # Respiratory
        "RespRate", "PaO2", "SaO2", "PaCO2", "pH",
        "MechVent", "FiO2",

        # Renal / electrolytes
        "Creatinine", "BUN", "K", "Na", "Mg", "HCO3",

        # Hepatic / nutrition
        "ALT", "AST", "Bilirubin", "ALP",
        "Albumin", "Cholesterol",

        # Hematologic
        "Platelets", "HCT",

        # Inflammation
        "WBC", "Temp",

        # Neurologic
        "GCS",

        # Cardiac
        "TropI", "TropT",

        # Metabolic
        "Glucose",

        # Outcome
        "Death",
    ]

    # ------------------------------------------------------------------
    # Add all nodes
    # ------------------------------------------------------------------
    G.add_nodes_from(background_vars, node_type="background")
    G.add_nodes_from(latent_vars, node_type="latent")
    G.add_nodes_from(observed_vars, node_type="observed")

    # ------------------------------------------------------------------
    # Background -> latent
    # ------------------------------------------------------------------
    for var in background_vars:
        G.add_edge(var, "ChronicRisk")

    G.add_edge("ChronicRisk", "Severity")
    G.add_edge("AcuteInsult", "Severity")

    # ------------------------------------------------------------------
    # Severity -> organ failures / states
    # ------------------------------------------------------------------
    organ_states = [
        "Shock", "RespFail", "RenalFail", "HepFail",
        "HemeFail", "Inflam", "NeuroFail", "CardInj", "Metab"
    ]

    for state in organ_states:
        G.add_edge("Severity", state)

    # ------------------------------------------------------------------
    # Shock -> measurements
    # ------------------------------------------------------------------
    shock_outputs = [
        "SysABP", "DiasABP", "MAP",
        "NISysABP", "NIDiasABP", "NIMAP",
        "HR", "Lactate", "Urine"
    ]

    for var in shock_outputs:
        G.add_edge("Shock", var)

    # ------------------------------------------------------------------
    # Respiratory failure -> measurements & interventions
    # ------------------------------------------------------------------
    resp_outputs = ["RespRate", "PaO2", "SaO2", "PaCO2", "pH"]
    for var in resp_outputs:
        G.add_edge("RespFail", var)

    G.add_edge("RespFail", "MechVent")
    G.add_edge("RespFail", "FiO2")

    # Interventions affect gas exchange
    G.add_edge("MechVent", "PaCO2")
    G.add_edge("MechVent", "pH")
    G.add_edge("FiO2", "PaO2")
    G.add_edge("FiO2", "SaO2")

    # ------------------------------------------------------------------
    # Renal dysfunction
    # ------------------------------------------------------------------
    renal_outputs = [
        "Creatinine", "BUN", "Urine",
        "K", "Na", "Mg", "HCO3", "pH"
    ]

    for var in renal_outputs:
        G.add_edge("RenalFail", var)

    # ------------------------------------------------------------------
    # Hepatic dysfunction
    # ------------------------------------------------------------------
    hepatic_outputs = [
        "ALT", "AST", "Bilirubin", "ALP",
        "Albumin", "Cholesterol"
    ]

    for var in hepatic_outputs:
        G.add_edge("HepFail", var)

    # ------------------------------------------------------------------
    # Hematologic dysfunction
    # ------------------------------------------------------------------
    for var in ["Platelets", "HCT"]:
        G.add_edge("HemeFail", var)

    # ------------------------------------------------------------------
    # Inflammation
    # ------------------------------------------------------------------
    G.add_edge("Inflam", "WBC")
    G.add_edge("Inflam", "Temp")
    G.add_edge("Inflam", "Shock")  # sepsis pathway

    # ------------------------------------------------------------------
    # Neurologic dysfunction
    # ------------------------------------------------------------------
    G.add_edge("NeuroFail", "GCS")

    # ------------------------------------------------------------------
    # Cardiac injury
    # ------------------------------------------------------------------
    G.add_edge("CardInj", "TropI")
    G.add_edge("CardInj", "TropT")
    G.add_edge("CardInj", "Shock")

    # ------------------------------------------------------------------
    # Metabolic derangement
    # ------------------------------------------------------------------
    for var in ["Glucose", "Lactate", "HCO3", "pH"]:
        G.add_edge("Metab", var)

    # ------------------------------------------------------------------
    # Outcome
    # ------------------------------------------------------------------
    G.add_edge("Severity", "Death")
    for state in [
        "Shock", "RespFail", "RenalFail", "HepFail",
        "HemeFail", "NeuroFail", "CardInj"
    ]:
        G.add_edge(state, "Death")

    G.add_edge("Age", "Death")

    # ------------------------------------------------------------------
    # Sanity check: ensure DAG
    # ------------------------------------------------------------------
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Constructed graph is not a DAG")

    return G


def draw_graph(
    G: nx.DiGraph,
    save = 0,
    figsize=(22, 18),
    node_size=1400,
    font_size=8,
):
    """
    Draw a hierarchical causal DAG with visible directed arrows.
    Node colors:
      - background : light blue
      - latent     : orange
      - observed   : light green
    """

    # ------------------------------------------------------------
    # Color mapping
    # ------------------------------------------------------------
    color_map = {
        "background": "#9ecae1",  # light blue
        "latent": "#fdae6b",      # orange
        "observed": "#a1d99b",    # light green
    }

    # ------------------------------------------------------------
    # Group nodes
    # ------------------------------------------------------------
    background_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "background"
    ]
    latent_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "latent"
    ]
    observed_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "observed"
    ]

    # ------------------------------------------------------------
    # Hierarchical layout (manual & deterministic)
    # ------------------------------------------------------------
    pos = {}

    def _assign_layer(nodes, y, x_spacing):
        x_offset = -(len(nodes) - 1) * x_spacing / 2
        for i, node in enumerate(nodes):
            pos[node] = (x_offset + i * x_spacing, y)

    _assign_layer(background_nodes, y=3.0, x_spacing=2.2)
    _assign_layer(latent_nodes, y=2.0, x_spacing=2.0)
    _assign_layer(observed_nodes, y=1.0, x_spacing=0.9)

    # ------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------
    plt.figure(figsize=figsize)

    # --- EDGES (with real arrows) ---
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="gray",
        width=1.2,
        alpha=0.7,
        connectionstyle="arc3,rad=0.05",
        min_source_margin=10,
        min_target_margin=15,
    )

    # --- NODES ---
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=background_nodes,
        node_color=color_map["background"],
        node_size=node_size,
        edgecolors="black",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=latent_nodes,
        node_color=color_map["latent"],
        node_size=node_size,
        edgecolors="black",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=observed_nodes,
        node_color=color_map["observed"],
        node_size=node_size,
        edgecolors="black",
    )

    # --- LABELS ---
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
    )

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=12,
        )
        for label, color in color_map.items()
    ]

    plt.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    plt.title("PhysioNet 2012 – Causal DAG", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    if save:
        file_name = "../PhysioNet 2012 – Causal DAG.png"
        plt.savefig(file_name)
        if os.path.exists(file_name):
            size = os.path.getsize(file_name)
            if size > 0:
                print(f"✅ File '{file_name}' exists and has size {size} bytes.")
    else:
        plt.show()


g = create_physionet2012_causal_graph()
draw_graph(g, save=1)
