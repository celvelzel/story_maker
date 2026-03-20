"""Player option generation via LLM with hardcoded fallback.

玩家选项生成模块：通过 LLM 生成玩家选项，支持硬编码回退。

每个选项包含：
- ``text``: 选项文本
- ``intent_hint``: 意图提示
- ``risk_level``: 风险等级
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from config import settings
from src.nlg.prompt_templates import OPTION_GENERATION_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class StoryOption:
    """A single player choice.
    
    单个玩家选项。
    包含选项文本、意图提示、风险等级。
    """
    text: str  # 选项文本
    intent_hint: str = "other"  # 意图提示
    risk_level: str = "medium"  # 风险等级（low, medium, high）


# Hardcoded safety net
_FALLBACK_OPTIONS: List[StoryOption] = [
    StoryOption("Look around and assess the situation.", "explore", "low"),
    StoryOption("Move cautiously forward.", "action", "medium"),
    StoryOption("Try to speak with someone nearby.", "dialogue", "low"),
]


class OptionGenerator:
    """Generate contextual options for the player using the LLM.
    
    选项生成器：使用 LLM 为玩家生成上下文相关的选项。
    如果 LLM 调用失败，使用硬编码的默认选项。
    """

    def generate(
        self,
        story_text: str,
        kg_summary: str,
        num_options: int | None = None,
    ) -> List[StoryOption]:
        """Return a list of ``StoryOption``.  Falls back to defaults on error.
        
        生成玩家选项列表。如果出错，回退到默认选项。
        
        参数:
            story_text: 故事文本
            kg_summary: 知识图谱摘要
            num_options: 要生成的选项数量（可选）
            
        返回:
            List[StoryOption]: 选项列表
        """
        num_options = num_options or settings.NUM_OPTIONS
        try:
            from src.utils.api_client import llm_client

            user_msg = OPTION_GENERATION_PROMPT.format(
                num_options=num_options,
                story_text=story_text[-1500:],
                kg_summary=kg_summary[-1000:],
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            data = llm_client.chat_json(messages, temperature=0.8, max_tokens=512)
            raw_options = data.get("options", [])
            options = [
                StoryOption(
                    text=o.get("text", "Continue…"),
                    intent_hint=o.get("intent_hint", "other"),
                    risk_level=o.get("risk_level", "medium"),
                )
                for o in raw_options
            ]
            if options:
                return options[:num_options]
        except Exception as exc:
            logger.warning("Option generation failed (%s) – using fallback.", exc)

        return list(_FALLBACK_OPTIONS[:num_options])
