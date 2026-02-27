"""KG visualisation via PyVis with entity-type colours and shapes."""
from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# Colour / shape mapping per entity type
_TYPE_STYLE = {
    "person":   {"color": "#4A90D9", "shape": "dot"},
    "location": {"color": "#50C878", "shape": "diamond"},
    "item":     {"color": "#FFB347", "shape": "triangle"},
    "creature": {"color": "#FF6B6B", "shape": "star"},
    "event":    {"color": "#9B59B6", "shape": "square"},
    "unknown":  {"color": "#C0C0C0", "shape": "dot"},
}


def render_kg_html(graph: nx.MultiDiGraph, output_path: str = "kg_vis.html") -> str:
    """Return an HTML string of an interactive PyVis visualisation.

    Falls back to a plain HTML table if PyVis is not installed.
    """
    try:
        from pyvis.network import Network  # type: ignore[import-untyped]
    except ImportError:
        return _fallback_html(graph)

    net = Network(height="480px", width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="white")
    net.set_options("""
    {"physics": {"forceAtlas2Based": {"gravitationalConstant": -50,
      "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08},
      "solver": "forceAtlas2Based", "stabilization": {"iterations": 100}},
     "edges": {"arrows": {"to": {"enabled": true}}, "smooth": {"type": "curvedCW", "roundness": 0.2}}}
    """)

    for node, data in graph.nodes(data=True):
        etype = data.get("entity_type", "unknown")
        style = _TYPE_STYLE.get(etype, _TYPE_STYLE["unknown"])
        label = data.get("name", node)
        net.add_node(node, label=label, color=style["color"], shape=style["shape"],
                     title=f"{label} [{etype}]", size=20)

    for src, tgt, data in graph.edges(data=True):
        rel = data.get("relation", "related_to")
        net.add_edge(src, tgt, label=rel, title=rel, width=2, color="#95A5A6")

    try:
        net.save_graph(output_path)
        with open(output_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return _fallback_html(graph)


def _fallback_html(graph: nx.MultiDiGraph) -> str:
    rows = ""
    for src, tgt, data in graph.edges(data=True):
        rel = data.get("relation", "related_to")
        rows += f"<tr><td>{src}</td><td>{rel}</td><td>{tgt}</td></tr>"
    return (
        "<table border='1' style='color:white;background:#1a1a2e;'>"
        "<tr><th>Source</th><th>Relation</th><th>Target</th></tr>"
        f"{rows}</table>"
    )
