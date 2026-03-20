"""LLM-as-Judge evaluation: rate a full story session on 5 dimensions.

LLM 评判评估模块：使用 LLM 作为评判者对完整故事会话进行多维度评分。

评估维度（每项 1-10 分）：
- narrative_quality（叙事质量）：文笔质量、生动描述、吸引人的语言
- consistency（一致性）：角色、地点和事实保持连贯
- player_agency（玩家能动性）：玩家的选择对故事的影响程度
- creativity（创意）：情节、设定和角色的原创性
- pacing（节奏）：适当的故事张力和节奏管理
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# LLM 评判者系统提示：定义评判者角色和评分标准
_JUDGE_SYSTEM = (
    "You are an expert literary critic evaluating an interactive fiction session. "
    "Score each dimension from 1 (terrible) to 10 (masterful). "
    "Return ONLY valid JSON with the five keys."
)

# LLM 评判者用户提示：包含待评估的故事文本和评分维度说明
_JUDGE_USER = """\
Below is the full transcript of an interactive story session.

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Rate the session on these five dimensions (1-10):
1. narrative_quality – prose quality, vivid descriptions, engaging language
2. consistency – characters, locations, and facts remain coherent
3. player_agency – how meaningfully the player's choices affected the story
4. creativity – originality of plot, settings, and characters
5. pacing – appropriate story momentum and tension management

Return JSON:
{{"narrative_quality": <int>, "consistency": <int>, "player_agency": <int>, "creativity": <int>, "pacing": <int>}}
"""

# 评估维度列表
DIMENSIONS = [
    "narrative_quality",  # 叙事质量
    "consistency",         # 一致性
    "player_agency",      # 玩家能动性
    "creativity",         # 创意
    "pacing",             # 节奏
]


def judge(transcript: str) -> Dict[str, int | float]:
    """发送故事文本到 LLM 并返回各维度评分。

    将完整的故事会话记录发送给 LLM 评判者，从 5 个维度进行评估。
    LLM 返回 JSON 格式的评分，分数范围 1-10。

    参数：
        transcript: 完整的故事会话文本记录

    返回：
        Dict[str, int | float]: 包含各维度评分的字典，
            每个维度为 1-10 的整数，额外包含 "average" 平均分（浮点数）
            评估失败时所有分数为 0
    """
    from src.utils.api_client import llm_client

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER.format(transcript=transcript[:6000])},
    ]
    try:
        # 调用 LLM API 获取评分（适度温度以保持一致性）
        data = llm_client.chat_json(messages, temperature=0.3, max_tokens=256)
        scores: Dict[str, int | float] = {}
        for dim in DIMENSIONS:
            val = data.get(dim, 0)
            # 确保分数在 1-10 范围内
            scores[dim] = max(1, min(10, int(val)))
        # 计算平均分
        scores["average"] = round(sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)
        return scores
    except Exception as exc:
        logger.warning("LLM 评判失败: %s", exc)
        return {d: 0 for d in DIMENSIONS} | {"average": 0.0}
