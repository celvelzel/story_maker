"""Story generation via OpenAI chat completions.

故事生成模块：通过 OpenAI 聊天补全生成故事。

提供 ``generate_opening()`` 和 ``continue_story()`` 两个方法，
都是围绕共享的 ``llm_client`` 单例的薄包装器。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from src.nlg.prompt_templates import (
    OPENING_PROMPT,
    STORY_CONTINUE_PROMPT,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class StoryGenerator:
    """LLM-powered narrator for the text adventure.
    
    LLM 驱动的文本冒险叙述者。
    负责生成游戏开场和续写故事。
    """

    def generate_opening(self, genre: str = "fantasy") -> str:
        """Generate the opening scene of a new game.
        
        生成新游戏的开场场景。
        
        参数:
            genre: 故事类型（如 "fantasy"）
            
        返回:
            str: 开场故事文本
        """
        from src.utils.api_client import llm_client

        user_msg = OPENING_PROMPT.format(genre=genre)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return llm_client.chat(messages)

    def continue_story(
        self,
        player_input: str,
        intent: str,
        kg_summary: str,
        history: str,
        emotion: str = "neutral",
    ) -> str:
        """Continue the story based on the player's action and emotion.
        
        根据玩家行动和情感续写故事。
        
        参数:
            player_input: 玩家输入文本
            intent: 玩家意图
            kg_summary: 知识图谱摘要
            history: 最近的故事历史
            emotion: 情感标签
            
        返回:
            str: 续写的故事文本
        """
        from src.utils.api_client import llm_client

        user_msg = STORY_CONTINUE_PROMPT.format(
            kg_summary=kg_summary,
            history=history,
            intent=intent,
            player_input=player_input,
            emotion=emotion,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return llm_client.chat(messages)
