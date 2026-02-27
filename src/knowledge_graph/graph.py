"""KnowledgeGraph: MultiDiGraph-based world state tracker.

Node keys are always **lowercased** names.  Supports multiple edges between the
same pair (NetworkX ``MultiDiGraph``).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Real-time world-state graph backed by ``nx.MultiDiGraph``."""

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ── node helpers ──────────────────────────────────────
    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    # ── entity management ─────────────────────────────────
    def add_entity(self, name: str, entity_type: str, **attrs: Any) -> str:
        """Upsert a node.  Returns the lowered key."""
        key = self._key(name)
        if self.graph.has_node(key):
            # merge attributes
            self.graph.nodes[key].update(attrs)
            self.graph.nodes[key]["entity_type"] = entity_type
        else:
            self.graph.add_node(key, name=name, entity_type=entity_type, **attrs)
            self._enforce_limit()
        return key

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        key = self._key(name)
        if key in self.graph:
            return dict(self.graph.nodes[key])
        return None

    def remove_entity(self, name: str) -> None:
        key = self._key(name)
        if key in self.graph:
            self.graph.remove_node(key)

    # ── relation management ───────────────────────────────
    def add_relation(self, source: str, target: str, relation: str, **attrs: Any) -> None:
        """Add an edge.  Auto-creates missing nodes as ``unknown`` type.

        Duplicate (source, target, relation) triples are silently skipped.
        """
        src = self._key(source)
        tgt = self._key(target)
        if not self.graph.has_node(src):
            self.add_entity(source, "unknown")
        if not self.graph.has_node(tgt):
            self.add_entity(target, "unknown")

        # skip if exact same relation already exists
        for _k, data in self.graph[src][tgt].items() if self.graph.has_edge(src, tgt) else []:
            if data.get("relation") == relation:
                return
        self.graph.add_edge(src, tgt, relation=relation, **attrs)

    def get_relations(self, name: str) -> List[Dict[str, Any]]:
        key = self._key(name)
        results: List[Dict[str, Any]] = []
        if key not in self.graph:
            return results
        for _, tgt, data in self.graph.out_edges(key, data=True):
            results.append({"target": tgt, **data})
        for src, _, data in self.graph.in_edges(key, data=True):
            results.append({"source": src, **data})
        return results

    # ── summary for LLM context ──────────────────────────
    def to_summary(self, max_entities: int = 30) -> str:
        """Produce a textual world-state block consumable by the LLM."""
        nodes = list(self.graph.nodes(data=True))[:max_entities]
        if not nodes:
            return "=== World State ===\n(empty)"

        lines = ["=== World State ==="]
        for key, data in nodes:
            etype = data.get("entity_type", "unknown")
            name = data.get("name", key)
            extra = {k: v for k, v in data.items() if k not in ("name", "entity_type")}
            extra_str = f" {extra}" if extra else ""
            lines.append(f"- {name} [{etype}]{extra_str}")

        lines.append("\n=== Relations ===")
        for src, tgt, data in list(self.graph.edges(data=True)):
            rel = data.get("relation", "related_to")
            lines.append(f"- {src} --[{rel}]--> {tgt}")

        return "\n".join(lines)

    # ── size enforcement ──────────────────────────────────
    def _enforce_limit(self) -> None:
        """Remove least-connected node when exceeding ``KG_MAX_NODES``."""
        while self.graph.number_of_nodes() > settings.KG_MAX_NODES:
            least = min(self.graph.nodes, key=lambda n: self.graph.degree(n))
            self.graph.remove_node(least)
            logger.debug("KG limit: removed %s", least)

    # ── stats ─────────────────────────────────────────────
    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
