"""Game state data structures for StoryWeaver (hybrid architecture).

StoryWeaver 游戏状态数据结构。
管理回合信息、类型上下文、玩家与叙述者之间的完整交互历史。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GameState:
    """
    游戏状态类：管理交互式故事游戏的叙事进展。
    
    维护回合信息、故事类型上下文、玩家与叙述者之间的完整交互历史。
    
    属性:
        turn_id (int): 当前回合编号计数器。每次叙述者行动后递增。
            默认值: 0
        genre (str): 故事类型或主题（如 "fantasy", "sci-fi", "mystery"）。
            影响叙述的风格和上下文。
            默认值: "fantasy"
        story_history (List[Dict[str, str]]): 完整的故事交互记录。
            每个条目是一个字典，包含：
            - "role": "player"（玩家输入）或 "narrator"（故事叙述）
            - "text": 该次交互的实际内容
            默认值: 空列表
    
    方法:
        add_player_input(text: str) -> None:
            将玩家输入添加到故事历史。
            参数:
                text (str): 要添加的玩家输入文本。
        add_narration(text: str) -> None:
            将叙述者回复添加到故事历史，并递增回合计数器。
            应在处理叙述者行动后调用此方法以推进游戏状态。
            参数:
                text (str): 要添加的叙述者回复文本。
        recent_history(n: int = 6) -> str:
            检索并格式化故事历史中的最后 n 条记录，用于 LLM 上下文。
            这对于为语言模型提供最近的故事上下文很有用。
            参数:
                n (int): 要检索的最近条目数。默认值: 6
            返回:
                str: 格式化字符串，条目用换行符分隔，
                     每个条目前缀有 "[Player]" 或 "[Narrator]" 标签。
    """
    """Minimal game state carried across turns."""

    turn_id: int = 0  # 当前回合编号
    genre: str = "fantasy"  # 故事类型
    story_history: List[Dict[str, str]] = field(default_factory=list)  # 故事历史记录
    # 每个条目格式: {"role": "player"|"narrator", "text": "..."}

    def add_player_input(self, text: str) -> None:
        """添加玩家输入到故事历史。"""
        self.story_history.append({"role": "player", "text": text})

    def add_narration(self, text: str) -> None:
        """添加叙述者回复到故事历史，并递增回合计数器。"""
        self.story_history.append({"role": "narrator", "text": text})
        self.turn_id += 1

    
    def recent_history(self, n: int = 6) -> str:
        """Return the last *n* entries formatted for LLM context.
        
        返回最后 n 条记录，格式化为 LLM 上下文。
        每条记录前缀有 "[Player]" 或 "[Narrator]" 标签。
        """
        entries = self.story_history[-n:]
        parts: List[str] = []
        for e in entries:
            prefix = "Player" if e["role"] == "player" else "Narrator"
            parts.append(f"[{prefix}] {e['text']}")
        return "\n".join(parts)
