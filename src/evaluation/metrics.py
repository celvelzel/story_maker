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

from collections import Counter
from typing import Dict, List


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
            scores.append(
                sentence_bleu(refs, hyp_tok, weights=weights, smoothing_function=smoother)
            )
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


# ── Bundle helper ──────────────────────────────────────────────────────


def full_evaluation(
    texts: List[str],
    entity_names: List[str] | None = None,
    turn_conflict_counts: List[int] | None = None,
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
    return results
