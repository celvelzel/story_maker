# 故事续写生成 — 提示词规范

> **最后更新：** 2026-04-01  
> **来源：** `src/nlg/prompt_templates.py`

本文档定义了基于玩家输入和世界状态继续叙事的提示词结构。

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
   # 每次回复严格限制为**恰好 1 段**（最多 3-5 句）
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

用户提示词提供当前上下文（知识图谱摘要、近期历史和玩家状态）以指导下一步叙事。

```text
{kg_summary}
# 知识图谱摘要（当前世界状态）

Recent history:
# 近期历史：
{history}

The player's intent is "{intent}".
# 玩家的意图是"{intent}"
The player's emotional tone is: {emotion}
# 玩家的情绪基调是：{emotion}
The player says: "{player_input}"
# 玩家说："{player_input}"

Continue the story by:
# 继续故事：
1. **React directly** to what the player did—explain the immediate, concrete consequence in 1-2 sentences.
   # **直接回应**玩家的行动——用 1-2 句解释即时具体的后果
2. **Maintain consistency** with the world state above. Only describe things that exist in the KG.
   # **保持一致性**，仅描述 KG 中存在的事物
3. **Advance the plot**: Introduce the next situation or challenge. Be specific about what the player encounters.
   # **推进情节**：引入下一个情境或挑战，具体描述玩家遇到的事物

Write exactly **1 paragraph** (3-4 sentences total). End with a clear moment where the player must decide what to do next.
# 写**恰好 1 段**（共 3-4 句）。在玩家必须决定下一步的清晰时刻结束。
```

## 3. 上下文变量

| 变量 | 来源 | 描述 |
|----------|--------|-------------|
| `kg_summary` | `KnowledgeGraph.to_summary()` | 当前世界状态（扁平或分层模式） |
| `history` | `GameState.recent_history(6)` | 最近 6 条对话记录，格式为 `[玩家]/[叙述者]` 行 |
| `intent` | `IntentClassifier.predict()` | 分类意图（action/行动、dialogue/对话、explore/探索、use_item/使用物品、ask_info/询问信息、rest/休息、trade/交易、other/其他） |
| `emotion` | `SentimentAnalyzer.analyze()` | 检测到的情绪（anger/愤怒、disgust/厌恶、fear/恐惧、joy/喜悦、sadness/悲伤、surprise/惊讶、neutral/中性） |
| `player_input` | `CoreferenceResolver.resolve()` | 共指消解后的玩家输入文本 |

## 4. 使用方式

```python
from src.nlg.prompt_templates import STORY_CONTINUE_PROMPT, SYSTEM_PROMPT
from src.utils.api_client import llm_client

# 格式化用户消息，注入所有上下文变量
user_msg = STORY_CONTINUE_PROMPT.format(
    kg_summary=kg_summary,    # 知识图谱摘要
    history=history,          # 近期历史
    intent=intent,            # 玩家意图
    emotion=emotion,          # 玩家情绪
    player_input=resolved_input,  # 消解后的玩家输入
)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示词
    {"role": "user", "content": user_msg},  # 用户提示词
]
story = llm_client.chat(messages)  # 调用 LLM 生成续写
```

## 5. 训练数据生成（ChatML 格式）

用于微调本地模型时，样本以 JSONL 格式生成：

```json
{
  "messages": [
    {"role": "system", "content": "...（第 1 节的系统提示词）..."},
    {"role": "user", "content": "...（第 2 节填充后的模板）..."},
    {"role": "assistant", "content": "You grip the ancient sword tightly..."}
    # 助手回复：生成的续写叙事
  ]
}
```

## 6. 上下文数据池

### KG_SUMMARIES（知识图谱摘要池）

1. `Entities: hero (person/人物, importance: 0.9), dark_forest (location/地点, importance: 0.7), ancient_sword (item/物品, importance: 0.8)...`
2. `Entities: captain (person/人物, importance: 0.85), starship (item/物品, importance: 0.8), nebula_station (location/地点, importance: 0.6)...`
3. `Entities: detective (person/人物, importance: 0.9), warehouse (location/地点, importance: 0.5), mysterious_note (item/物品, importance: 0.7)...`

### HISTORIES（最近对话历史池，可取 1-3 段拼接）

1. `[Player] I step into the dark forest.\n[Narrator] The canopy above swallows the daylight...`
   # [玩家] 我走进黑暗森林。\n[叙述者] 上方的树冠吞噬了日光…
2. `[Player] I examine the control panel.\n[Narrator] The console hums to life...`
   # [玩家] 我检查控制面板。\n[叙述者] 控制台嗡嗡启动…

### INTENTS（玩家意图池）

- action（行动）、dialogue（对话）、explore（探索）、use_item（使用物品）、ask_info（询问信息）、rest（休息）、trade（交易）、other（其他）

### EMOTIONS（玩家情绪池）

- neutral（中性）、excited（兴奋）、anxious（焦虑）、angry（愤怒）、sad（悲伤）、curious（好奇）、determined（坚定）、fearful（恐惧）、joyful（喜悦）、suspicious（怀疑）、desperate（绝望）、hopeful（充满希望）

---
*实现说明：实际模板存储在 `src/nlg/prompt_templates.py` 中。*
