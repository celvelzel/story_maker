"""Thin wrapper that runs the ConflictDetector over accumulated story text.

一致性评估薄包装模块。

对累积的故事文本运行冲突检测器，生成每回合的冲突计数和总体一致性率。
这是对 ConflictDetector 的简单封装，用于批量评估。
"""
from __future__ import annotations

from typing import List

from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.conflict_detector import ConflictDetector


def evaluate_consistency(kg: KnowledgeGraph, story_texts: List[str]) -> dict:
    """对累积的故事文本进行一致性评估。

    使用冲突检测器逐回合检查故事文本，返回每回合的冲突数量和总体一致性率。

    参数：
        kg: 知识图谱对象，包含会话期间构建的世界状态
        story_texts: 故事文本列表，每个元素对应一个回合的生成叙述

    返回：
        dict: 包含以下键的字典：
            - turn_conflicts (List[int]): 每回合检测到的冲突数量列表
            - consistency_rate (float): 一致性率，无冲突回合的占比 [0, 1]
    """
    detector = ConflictDetector(kg)
    turn_conflicts: list[int] = []
    # 逐回合检测冲突
    for text in story_texts:
        conflicts = detector.check_all(text)
        turn_conflicts.append(len(conflicts))

    # 计算一致性率
    clean = sum(1 for c in turn_conflicts if c == 0)
    rate = clean / len(turn_conflicts) if turn_conflicts else 1.0

    return {
        "turn_conflicts": turn_conflicts,
        "consistency_rate": rate,
    }
