"""Thin wrapper that runs the ConflictDetector over accumulated story text."""
from __future__ import annotations

from typing import List

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.conflict_detector import ConflictDetector


def evaluate_consistency(kg: KnowledgeGraph, story_texts: List[str]) -> dict:
    """Return a summary dict with per-turn conflict counts and overall rate.

    Parameters
    ----------
    kg : KnowledgeGraph
        The world-state graph built during the session.
    story_texts : list[str]
        One string per turn (generated narration).

    Returns
    -------
    dict  with keys ``turn_conflicts`` (list[int]) and ``consistency_rate`` (float).
    """
    detector = ConflictDetector(kg)
    turn_conflicts: list[int] = []
    for text in story_texts:
        conflicts = detector.check_all(text)
        turn_conflicts.append(len(conflicts))

    clean = sum(1 for c in turn_conflicts if c == 0)
    rate = clean / len(turn_conflicts) if turn_conflicts else 1.0

    return {
        "turn_conflicts": turn_conflicts,
        "consistency_rate": rate,
    }
