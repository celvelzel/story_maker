"""KG visualisation via PyVis with entity-type colours and shapes."""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import quote

import networkx as nx

logger = logging.getLogger(__name__)

# Colour / shape mapping per entity type — cyberpunk neon palette
_TYPE_STYLE = {
    "person":   {"color": "#00f0ff", "shape": "dot"},
    "location": {"color": "#39ff14", "shape": "dot"},
    "item":     {"color": "#ffd700", "shape": "dot"},
    "creature": {"color": "#ff00aa", "shape": "dot"},
    "event":    {"color": "#7b2fff", "shape": "dot"},
    "unknown":  {"color": "#5a6a8a", "shape": "dot"},
}


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    c = hex_color.strip().lstrip("#")
    if len(c) != 6:
        return (90, 106, 138)
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def _mix(color: tuple[int, int, int], toward: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return (
        _clamp(int(color[0] + (toward[0] - color[0]) * amount)),
        _clamp(int(color[1] + (toward[1] - color[1]) * amount)),
        _clamp(int(color[2] + (toward[2] - color[2]) * amount)),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _gradient_svg_data_uri(base_color: str) -> str:
    """Build a radial-gradient circular SVG image encoded as data URI.

    Used by vis-network `circularImage` nodes so each entity appears with
    center-to-edge gradient fill.
    """
    base = _hex_to_rgb(base_color)
    center = _rgb_to_hex(_mix(base, (255, 255, 255), 0.35))
    edge = _rgb_to_hex(_mix(base, (0, 0, 0), 0.30))
    ring = _rgb_to_hex(_mix(base, (255, 255, 255), 0.15))

    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128' viewBox='0 0 128 128'>
  <defs>
    <radialGradient id='g' cx='38%' cy='35%' r='65%'>
      <stop offset='0%' stop-color='{center}'/>
      <stop offset='58%' stop-color='{base_color}'/>
      <stop offset='100%' stop-color='{edge}'/>
    </radialGradient>
  </defs>
  <circle cx='64' cy='64' r='58' fill='url(#g)' stroke='{ring}' stroke-width='4'/>
</svg>
""".strip()

    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def render_kg_html(graph: nx.MultiDiGraph, output_path: str = "kg_vis.html") -> str:
    """Return an HTML string of an interactive PyVis visualisation.

    Falls back to a plain HTML table if PyVis is not installed.
    """
    try:
        from pyvis.network import Network  # type: ignore[import-untyped]
    except ImportError:
        return _fallback_html(graph)

    net = Network(height="480px", width="100%", directed=True,
                  bgcolor="#06080f", font_color="#e0e8ff")
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
        node_image = _gradient_svg_data_uri(style["color"])
        net.add_node(
            node,
            label=label,
            shape="circularImage",
            image=node_image,
            title=f"{label} [{etype}]",
            size=24,
        )

    for src, tgt, data in graph.edges(data=True):
        rel = data.get("relation", "related_to")
        net.add_edge(src, tgt, label=rel, title=rel, width=2, color="#00f0ff55")

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
        "<table border='1' style='color:#e0e8ff;background:#06080f;border-color:#00f0ff33;'>"
        "<tr><th>Source</th><th>Relation</th><th>Target</th></tr>"
        f"{rows}</table>"
    )
