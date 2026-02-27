"""Lightweight automatic evaluation metrics for StoryWeaver.

Metrics included:
- distinct_n  (lexical diversity)
- self_bleu   (intra-corpus diversity – lower ⇒ more diverse)
- entity_coverage (KG entities mentioned in generated text)
- consistency_rate (percentage of turns without KG conflicts)
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List


# ── Distinct-n ──────────────────────────────────────────────────────────


def distinct_n(texts: List[str], n: int = 2) -> float:
    """Ratio of unique *n*-grams to total *n*-grams across all *texts*."""
    total: Counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        for i in range(len(tokens) - n + 1):
            total[tuple(tokens[i : i + n])] += 1
    if not total:
        return 0.0
    return len(total) / sum(total.values())


# ── Self-BLEU ───────────────────────────────────────────────────────────


def self_bleu(texts: List[str], max_n: int = 4) -> float:
    """Average sentence-BLEU of each text against all others (NLTK).

    Lower ⇒ more diverse.  Returns 0.0 if < 2 texts.
    """
    if len(texts) < 2:
        return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return -1.0

    smoother = SmoothingFunction().method1
    weights = tuple(1.0 / max_n for _ in range(max_n))
    scores: list[float] = []

    for i, hyp in enumerate(texts):
        refs = [texts[j].split() for j in range(len(texts)) if j != i]
        hyp_tok = hyp.split()
        if not hyp_tok or not refs:
            continue
        try:
            scores.append(
                sentence_bleu(refs, hyp_tok, weights=weights, smoothing_function=smoother)
            )
        except Exception:
            continue
    return sum(scores) / len(scores) if scores else 0.0


# ── Entity coverage ────────────────────────────────────────────────────


def entity_coverage(texts: List[str], entity_names: List[str]) -> float:
    """Fraction of *entity_names* that appear at least once in *texts*."""
    if not entity_names:
        return 1.0
    combined = " ".join(texts).lower()
    found = sum(1 for e in entity_names if e.lower() in combined)
    return found / len(entity_names)


# ── Consistency rate ────────────────────────────────────────────────────


def consistency_rate(turn_conflict_counts: List[int]) -> float:
    """Fraction of turns with **zero** conflicts.

    *turn_conflict_counts* is a list where each element is the number of
    conflicts detected after that turn.
    """
    if not turn_conflict_counts:
        return 1.0
    clean = sum(1 for c in turn_conflict_counts if c == 0)
    return clean / len(turn_conflict_counts)


# ── Bundle helper ───────────────────────────────────────────────────────


def full_evaluation(
    texts: List[str],
    entity_names: List[str] | None = None,
    turn_conflict_counts: List[int] | None = None,
) -> Dict[str, float]:
    """Run all lightweight metrics and return a flat dict."""
    results: Dict[str, float] = {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "self_bleu": self_bleu(texts),
    }
    if entity_names is not None:
        results["entity_coverage"] = entity_coverage(texts, entity_names)
    if turn_conflict_counts is not None:
        results["consistency_rate"] = consistency_rate(turn_conflict_counts)
    return results
