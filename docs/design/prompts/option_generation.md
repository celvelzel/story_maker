# 选项生成 — 提示词规范

> **最后更新：** 2026-04-01  
> **来源：** `src/nlg/prompt_templates.py`

本文档定义了基于最新叙事段落和世界状态生成分支玩家选择的提示词结构。

## 1. 系统提示词

系统提示词定义了叙述者的人格和所有生成任务的通用规则。

```text
You are an expert interactive-fiction narrator for a text-adventure game.
# 你是一位文字冒险游戏的交互式小说叙述专家

Rules:
# 规则：
1. Always narrate in **second person** ("You see…", "You feel…").
   # 始终使用**第二人称**叙述
2. Keep each response to **exactly 1 paragraph** (3-5 sentences max).
   # 每次回复严格限制为**恰好 1 段**
3. Maintain absolute consistency with the world state provided.
   # 与世界状态保持绝对一致
4. Be **concrete and specific**: name objects, locations, and NPCs explicitly.
   # **具体明确**：明确命名物体、地点和 NPC
5. Explain **cause and effect**: every story beat must follow logically from previous events.
   # 解释**因果关系**：每个故事节拍必须从先前事件中逻辑推导
6. Use **sensory details** only when describing actual things in the world.
   # 仅在描述实际存在的事物时使用**感官细节**
7. Never mention game mechanics, stats, or that you are an AI.
   # 绝不提及游戏机制、数值或你是 AI
8. Seamlessly incorporate the player's action into the narrative.
   # 将玩家行动无缝融入叙事
9. End the passage at a moment that invites the player to act next.
   # 在邀请玩家下一步行动的时刻结束
```

## 2. 用户提示词模板

用户提示词请求以结构化 JSON 格式生成指定数量的选项。

```text
Given the latest story passage and world state below, generate exactly {num_options} player options as a JSON array.
# 根据以下最新故事段落和世界状态，生成恰好 {num_options} 个玩家选项，以 JSON 数组格式返回

Story passage:
# 故事段落：
{story_text}

World state:
# 世界状态：
{kg_summary}

Return ONLY a JSON object:
# 仅返回一个 JSON 对象：
{"options": [{"text": "...", "intent_hint": "action|dialogue|explore|use_item|ask_info|rest|trade|other", "risk_level": "low|medium|high"}]}
# text: 选项文本, intent_hint: 意图提示, risk_level: 风险等级（low/低 | medium/中 | high/高）
```

## 3. 输出模式

每个选项是一个 `StoryOption` 数据类：

```python
@dataclass
class StoryOption:
    text: str            # 向用户显示的选项文本
    intent_hint: str     # 建议的意图类别
    risk_level: str      # 风险等级："low"（低）| "medium"（中）| "high"（高）
```

## 4. 约束清单

- **JSON 格式**：助手必须仅返回有效的 JSON 对象，无前缀或后缀文本
- **意图提示**：必须是 `action`（行动）、`dialogue`（对话）、`explore`（探索）、`use_item`（使用物品）、`ask_info`（询问信息）、`rest`（休息）、`trade`（交易）或 `other`（其他）之一
- **风险等级**：必须是 `low`（低）、`medium`（中）或 `high`（高）之一
- **上下文契合**：选项必须基于提供的 `story_text` 和 `kg_summary`

## 5. 兜底行为

如果 LLM 选项生成失败，系统回退到硬编码默认值：

```python
_FALLBACK_OPTIONS = [
    # 兜底选项：观察周围并评估情况（探索，低风险）
    StoryOption("Look around and assess the situation.", "explore", "low"),
    # 兜底选项：谨慎前进（行动，中风险）
    StoryOption("Move cautiously forward.", "action", "medium"),
    # 兜底选项：尝试与附近的人交谈（对话，低风险）
    StoryOption("Try to speak with someone nearby.", "dialogue", "low"),
]
```

## 6. 使用方式

```python
from src.nlg.option_generator import OptionGenerator
from src.nlg.prompt_templates import OPTION_GENERATION_PROMPT, SYSTEM_PROMPT

option_gen = OptionGenerator()  # 创建选项生成器实例
# 生成选项：传入故事文本、知识图谱摘要和选项数量
options = option_gen.generate(story_text, kg_summary, num_options=3)
```

---
*实现说明：实际模板存储在 `src/nlg/prompt_templates.py` 中。*
