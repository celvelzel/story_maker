"""Lightweight automatic evaluation metrics for StoryWeaver.

StoryWeaver 轻量级自动评估指标模块。

包含的指标：
- distinct_n（词汇多样性）：独特 n-gram 与总 n-gram 的比率
- self_bleu（语内多样性）：越低说明文本越多样化
- entity_coverage（实体覆盖率）：知识图谱实体在生成文本中的提及比例
- consistency_rate（一致性率）：没有冲突的回合占比

这些指标用于快速评估故事生成的质量，无需人工标注。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Iterable, Set, cast


# ── Distinct-n ──────────────────────────────────────────────────────────


def distinct_n(texts: List[str], n: int = 2) -> float:
    """计算独特 n-gram 与总 n-gram 的比率。

    用于衡量词汇多样性。值越高表示文本越丰富多样。

    参数：
        texts: 文本列表
        n: n-gram 的 n 值（1=单词, 2=二元组, 3=三元组）

    返回：
        float: 独特 n-gram 数量 / 总 n-gram 数量，范围 [0, 1]
    """
    total: Counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        for i in range(len(tokens) - n + 1):
            total[tuple(tokens[i : i + n])] += 1
    if not total:
        return 0.0
    return len(total) / sum(total.values())


# ── Self-BLEU ──────────────────────────────────────────────────────────


def self_bleu(texts: List[str], max_n: int = 4) -> float:
    """计算每个文本相对于其他所有文本的平均 Sentence-BLEU 分数（NLTK）。

    越低说明文本越多样化。BLEU 分数衡量候选文本与参考文本的 n-gram 重叠度。

    参数：
        texts: 文本列表（至少需要 2 个文本）
        max_n: 最大 n-gram 阶数

    返回：
        float: 平均 Self-BLEU 分数 [0, 1]，少于 2 个文本返回 0.0
    """
    if len(texts) < 2:
        return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return -1.0  # NLTK 不可用时返回 -1

    smoother = SmoothingFunction().method1  # 平滑处理
    weights = tuple(1.0 / max_n for _ in range(max_n))  # 均匀权重
    scores: list[float] = []

    for i, hyp in enumerate(texts):
        # 将其他文本作为参考
        refs = [texts[j].split() for j in range(len(texts)) if j != i]
        hyp_tok = hyp.split()
        if not hyp_tok or not refs:
            continue
        try:
            score = cast(float, sentence_bleu(refs, hyp_tok, weights=weights, smoothing_function=smoother))
            scores.append(score)
        except Exception:
            continue
    return sum(scores) / len(scores) if scores else 0.0


# ── Entity coverage ────────────────────────────────────────────────────


def entity_coverage(texts: List[str], entity_names: List[str]) -> float:
    """计算知识图谱实体在生成文本中的覆盖率。

    衡量故事生成是否提及了知识图谱中的重要实体。
    高覆盖率表示生成内容与构建的世界状态保持一致。

    参数：
        texts: 生成的故事文本列表
        entity_names: 知识图谱中的实体名称列表

    返回：
        float: 被提及的实体比例 [0, 1]
    """
    if not entity_names:
        return 1.0  # 没有实体时默认为完美覆盖
    combined = " ".join(texts).lower()  # 合并所有文本
    found = sum(1 for e in entity_names if e.lower() in combined)
    return found / len(entity_names)


# ── Consistency rate ────────────────────────────────────────────────────


def consistency_rate(turn_conflict_counts: List[int]) -> float:
    """计算没有冲突的回合占比。

    衡量故事生成的一致性质量。
    高一致性率表示知识图谱维护良好，世界状态保持连贯。

    参数：
        turn_conflict_counts: 每个回合检测到的冲突数量列表

    返回：
        float: 无冲突回合的比例 [0, 1]
    """
    if not turn_conflict_counts:
        return 1.0  # 没有数据时默认为完美一致
    clean = sum(1 for c in turn_conflict_counts if c == 0)
    return clean / len(turn_conflict_counts)


# ── Reference-free evaluators ───────────────────────────────────────────


_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
_VOWEL_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _tokenize_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text)]


def lexical_overlap(text_a: str, text_b: str) -> float:
    """Jaccard overlap of unique word sets between two texts."""
    a_set = set(_tokenize_words(text_a))
    b_set = set(_tokenize_words(text_b))
    union = a_set | b_set
    if not union:
        return 0.0
    return len(a_set & b_set) / len(union)


def type_token_ratio(text: str) -> float:
    """Vocabulary richness ratio: unique tokens / total tokens."""
    tokens = _tokenize_words(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = _VOWEL_PATTERN.findall(w)
    count = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    """Lightweight Flesch Reading Ease score (English heuristic)."""
    sentence_units = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = _tokenize_words(text)
    if not words:
        return 0.0
    sentence_count = len(sentence_units) if sentence_units else 1
    syllable_count = sum(_estimate_syllables(w) for w in words)
    words_per_sentence = len(words) / sentence_count
    syllables_per_word = syllable_count / len(words)
    score = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    return round(score, 2)


# ── Graph density evolution ─────────────────────────────────────────────


def _graph_density(node_count: int, edge_count: int) -> float:
    if node_count < 2:
        return 0.0
    denominator = node_count * (node_count - 1)
    if denominator <= 0:
        return 0.0
    return edge_count / denominator


def graph_density_evolution(kg_turn_stats: List[Dict[str, int]]) -> Dict[str, float | List[float]]:
    """Compute density trend from per-turn KG node/edge snapshots.

    Input items expect at least: turn_id, node_count, edge_count.
    """
    densities: List[float] = []
    for row in kg_turn_stats:
        nodes = int(row.get("node_count", 0))
        edges = int(row.get("edge_count", 0))
        densities.append(round(_graph_density(nodes, edges), 6))

    if not densities:
        return {
            "density_average": 0.0,
            "density_delta": 0.0,
            "density_start": 0.0,
            "density_end": 0.0,
            "density_series": [],
        }

    density_start = densities[0]
    density_end = densities[-1]
    density_average = round(sum(densities) / len(densities), 6)
    density_delta = round(density_end - density_start, 6)
    return {
        "density_average": density_average,
        "density_delta": density_delta,
        "density_start": density_start,
        "density_end": density_end,
        "density_series": densities,
    }


# ── Bundle helper ──────────────────────────────────────────────────────


def full_evaluation(
    texts: List[str],
    entity_names: List[str] | None = None,
    turn_conflict_counts: List[int] | None = None,
    kg_turn_stats: List[Dict[str, int]] | None = None,
) -> Dict[str, float]:
    """运行所有轻量级指标评估，返回扁平字典。

    综合评估故事生成的多个维度：
    - 词汇多样性（Distinct-1/2/3）
    - 语内多样性（Self-BLEU）
    - 实体覆盖率（可选）
    - 一致性率（可选）

    参数：
        texts: 故事文本列表
        entity_names: 知识图谱实体名称（可选）
        turn_conflict_counts: 回合冲突计数（可选）

    返回：
        Dict[str, float]: 指标名称到分数的映射
    """
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
    if texts:
        combined_text = "\n".join(texts)
        results["type_token_ratio"] = type_token_ratio(combined_text)
        results["flesch_reading_ease"] = flesch_reading_ease(combined_text)
        if len(texts) >= 2:
            overlaps = [lexical_overlap(texts[i - 1], texts[i]) for i in range(1, len(texts))]
            results["lexical_overlap"] = sum(overlaps) / len(overlaps)
        else:
            results["lexical_overlap"] = 0.0
    else:
        results["type_token_ratio"] = 0.0
        results["flesch_reading_ease"] = 0.0
        results["lexical_overlap"] = 0.0
    if kg_turn_stats is not None:
        density_metrics = graph_density_evolution(kg_turn_stats)
        density_average = density_metrics.get("density_average", 0.0)
        density_delta = density_metrics.get("density_delta", 0.0)
        results["graph_density_average"] = float(density_average) if isinstance(density_average, (int, float)) else 0.0
        results["graph_density_delta"] = float(density_delta) if isinstance(density_delta, (int, float)) else 0.0
    return results


# ── Wave-A regression helpers ───────────────────────────────────────────


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision/recall/F1 from confusion counts.

    Returns a stable dict with keys: precision, recall, f1.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match_accuracy(predicted: List[str], expected: List[str]) -> float:
    """Compute exact-match accuracy for aligned label lists."""
    if not expected:
        return 1.0
    n = min(len(predicted), len(expected))
    if n == 0:
        return 0.0
    correct = sum(1 for i in range(n) if str(predicted[i]).strip().lower() == str(expected[i]).strip().lower())
    return correct / len(expected)


def overlap_prf(predicted_items: Iterable[str], expected_items: Iterable[str]) -> Dict[str, float]:
    """Compute set-overlap precision/recall/F1 for normalized string items."""
    p_set: Set[str] = {str(x).strip().lower() for x in predicted_items if str(x).strip()}
    e_set: Set[str] = {str(x).strip().lower() for x in expected_items if str(x).strip()}

    tp = len(p_set & e_set)
    fp = len(p_set - e_set)
    fn = len(e_set - p_set)
    return precision_recall_f1(tp, fp, fn)


def entity_signature(name: str, entity_type: str) -> str:
    """Canonical entity signature for deterministic comparisons."""
    return f"{str(name).strip().lower()}::{str(entity_type).strip().lower()}"


def relation_signature(source: str, target: str, relation: str) -> str:
    """Canonical relation signature for deterministic comparisons."""
    return "::".join(
        [str(source).strip().lower(), str(target).strip().lower(), str(relation).strip().lower()]
    )


def conflict_signature(conflict_type: str, entity: str) -> str:
    """Canonical conflict signature for deterministic comparisons."""
    return f"{str(conflict_type).strip().lower()}::{str(entity).strip().lower()}"
