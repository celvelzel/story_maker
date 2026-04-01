# 情感分析设计

> **最后更新：** 2026-04-01  
> **模块：** `src/nlu/sentiment_analyzer.py`

## 1. 概述

情感分析器检测玩家输入中的情绪基调，使叙事能够根据玩家的情绪状态调整其情绪、节奏和措辞。

## 2. 架构

```
玩家输入 ──► SentimentAnalyzer.analyze()
                  │
          ┌───────┴───────┐
          │  模型已加载？  │
          └───────┬───────┘
                  │
     ┌────────────┴────────────┐
     │                         │
┌────▼────┐             ┌──────▼──────┐
│ 神经网络 │             │ 基于规则    │
│ 模型     │             │ 兜底方案    │
│          │             │             │
│ distil-  │             │ 关键词      │
│ roberta  │             │ 匹配        │
└────┬────┘             └──────┬──────┘
     │                         │
     └────────────┬────────────┘
                  ▼
     {emotion/情绪, confidence/置信度, scores/得分}
```

## 3. 情绪模型

### 3.1 神经网络后端

- **模型：** `j-hartmann/emotion-english-distilroberta-base`
- **标签：** 7 种 Ekman 基本情绪 + 中性
- **加载策略：** 最多 3 次重试，每次间隔 1 秒

### 3.2 情绪标签

| 标签 | 示例触发词 |
|-------|-----------------|
| `anger`（愤怒） | "attack"（攻击）、"kill"（杀死）、"furious"（狂怒）、"destroy"（毁灭）、"revenge"（复仇） |
| `disgust`（厌恶） | "disgusting"（恶心）、"vile"（卑鄙）、"repulsive"（令人反感）、"filthy"（肮脏） |
| `fear`（恐惧） | "afraid"（害怕）、"scared"（恐惧）、"danger"（危险）、"flee"（逃跑）、"panic"（恐慌） |
| `joy`（喜悦） | "happy"（快乐）、"wonderful"（精彩）、"love"（爱）、"celebrate"（庆祝）、"victory"（胜利） |
| `sadness`（悲伤） | "sad"（悲伤）、"lost"（失去）、"cry"（哭泣）、"mourn"（哀悼）、"lonely"（孤独）、"despair"（绝望） |
| `surprise`（惊讶） | "wow"（哇）、"unexpected"（意外）、"shocked"（震惊）、"incredible"（难以置信） |
| `neutral`（中性） | 未检测到强烈情绪时的默认值 |

## 4. 基于规则的兜底方案

当神经网络模型不可用时，使用关键词匹配：

```python
# 每种情绪有一个关键词列表
# 计算匹配数量并归一化得分
matches = sum(1 for kw in keywords if kw in text_lower)
score = min(matches / 3.0, 1.0)  # 归一化到 0-1
```

如果没有情绪得分超过 0.1，则返回 `neutral`（中性），置信度 0.5。

## 5. 输出格式

```python
{
    "emotion": "determined",     # 主导情绪标签
    "confidence": 0.7834,        # 置信度评分（0.0-1.0）
    "scores": {                  # 所有情绪得分
        "anger": 0.05,           # 愤怒
        "disgust": 0.02,         # 厌恶
        "fear": 0.15,            # 恐惧
        "joy": 0.10,             # 喜悦
        "sadness": 0.03,         # 悲伤
        "surprise": 0.08,        # 惊讶
        "neutral": 0.57          # 中性
    }
}
```

## 6. 与游戏引擎的集成

情绪结果贯穿整个流水线传递：

1. **故事生成：** `emotion` 参数包含在 `STORY_CONTINUE_PROMPT` 中，使 LLM 能够调整叙事情绪
2. **KG 追踪：** 实体节点可存储 `last_emotion` 用于情绪上下文
3. **调试输出：** 可在 `TurnResult.nlu_debug["emotion"]` 中查看

```python
# 在 GameEngine.process_turn() 中：
emotion_result = self.sentiment.analyze(resolved)  # 分析消解后文本的情绪
emotion = emotion_result.get("emotion", "neutral")  # 获取主导情绪，默认中性

story_text = self.story_gen.continue_story(
    player_input=resolved,
    intent=intent,
    kg_summary=kg_summary,
    history=history,
    emotion=emotion,  # 传递给 LLM 用于情绪调整
)
```

## 7. 配置

SentimentAnalyzer 没有专用的配置参数。它使用共享的 transformers 库，并进行版本兼容性检查（如果 transformers >= 4.50 则发出警告）。

## 8. 错误处理

- **模型加载失败：** 回退到基于关键词的情绪检测
- **Transformers 版本：** 如果版本 >= 4.50 则发出警告（测试范围：4.40–4.49）
- **运行时错误：** 任何分析失败时返回 `neutral`（中性），置信度 0.5

---
*相关文档：KG 情绪追踪请见 [entity-importance.md](entity-importance.md)*
