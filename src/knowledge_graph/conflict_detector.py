"""Rule-based + LLM conflict detection for the knowledge graph.

Layer 1 — deterministic rules (exclusive relation pairs, attribute contradictions).
Layer 2 — LLM reasoning against the KG summary.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Pairs of relations that cannot coexist between the same source→target
EXCLUSIVE_PAIRS: List[tuple[str, str]] = [
    ("ally_of", "enemy_of"),
    ("alive", "dead"),
]


class ConflictDetector:
    """Detect contradictions in a :class:`KnowledgeGraph`."""

    def __init__(self, kg: Any) -> None:  # avoid circular import typing
        self.kg = kg

    # ── public entry point ────────────────────────────────
    def check_all(self, new_text: str = "") -> List[Dict[str, str]]:
        """Return a list of conflict dicts (may be empty)."""
        conflicts: List[Dict[str, str]] = []
        conflicts.extend(self._rule_based_check())
        if new_text:
            conflicts.extend(self._llm_check(new_text))
        return conflicts

    # ── layer 1: rule-based ───────────────────────────────
    def _rule_based_check(self) -> List[Dict[str, str]]:
        conflicts: List[Dict[str, str]] = []
        g = self.kg.graph

        # 1. exclusive relation pairs
        for src, tgt, data in g.edges(data=True):
            rel = data.get("relation", "")
            for a, b in EXCLUSIVE_PAIRS:
                opposite = b if rel == a else (a if rel == b else None)
                if opposite is None:
                    continue
                # check if opposite edge exists
                if g.has_edge(src, tgt):
                    for _k, edata in g[src][tgt].items():
                        if edata.get("relation") == opposite:
                            conflicts.append({
                                "type": "exclusive_relation",
                                "description": (
                                    f"{src} has both '{rel}' and '{opposite}' towards {tgt}."
                                ),
                            })

        # 2. dead entity with positive-health relation
        for node, ndata in g.nodes(data=True):
            status = ndata.get("status", "")
            if status == "dead":
                for _, tgt, edata in g.out_edges(node, data=True):
                    rel = edata.get("relation", "")
                    if rel in ("possesses", "located_at", "ally_of"):
                        conflicts.append({
                            "type": "dead_active",
                            "description": f"{node} is dead but has relation '{rel}' → {tgt}.",
                        })
        return conflicts

    # ── layer 2: LLM ─────────────────────────────────────
    def _llm_check(self, new_text: str) -> List[Dict[str, str]]:
        try:
            from src.utils.api_client import llm_client

            kg_summary = self.kg.to_summary()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a consistency checker for a text adventure knowledge graph.\n"
                        "Given the current world state and a new story passage, identify any "
                        "logical contradictions.  Return JSON: "
                        '{"conflicts": [{"description": str}]}  '
                        "Return an empty list if there are none."
                    ),
                },
                {
                    "role": "user",
                    "content": f"World state:\n{kg_summary}\n\nNew text:\n{new_text}",
                },
            ]
            data = llm_client.chat_json(messages, temperature=0.1, max_tokens=256)
            raw = data.get("conflicts", [])
            return [{"type": "llm", "description": c.get("description", str(c))} for c in raw]
        except Exception as exc:
            logger.warning("LLM conflict check failed: %s", exc)
            return []
