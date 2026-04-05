# KG 摘要模式

> **最后更新：** 2026-04-01  
> **模块：** `src/knowledge_graph/graph.py`

## 1. 概述

知识图谱摘要是世界当前状态的文本表示，在故事和选项生成期间被 LLM 消费。StoryWeaver 支持两种摘要模式：`flat`（扁平模式，向后兼容）和 `layered`（分层模式，按重要性排序）。

## 2. 模式对比

| 特性 | 扁平模式 (Flat) | 分层模式 (Layered) |
|---------|-----------|--------------|
| **组织方式** | 顺序列表 | 按重要性分层的区块 |
| **实体信息** | 基础信息（名称、类型、属性） | 丰富信息（描述、状态、历史、关系） |
| **排序方式** | 插入顺序 | 重要性评分（降序） |
| **时间线** | 不包含 | 包含（近期事件） |
| **Token 效率** | 较低（所有实体平等对待） | 较高（优先重要实体） |
| **适用场景** | 小型 KG、调试 | 生产环境、大型 KG |

## 3. 扁平模式

生成简单的实体和关系顺序列表。

**输出格式：**
```
=== World State ===（世界状态）
- Dragon [creature/生物] {'created_turn': 0, 'last_mentioned_turn': 3, ...}
- Forest [location/地点] {'created_turn': 0, ...}
- Sword [item/物品] {'created_turn': 1, ...}

=== Relations ===（关系）
- dragon --[located_at/位于]--> forest
- dragon --[enemy_of/敌对]--> player
- player --[possesses/拥有]--> sword
```

**适用场景：**
- 调试与开发
- 实体数量可控的小型 KG
- 与现有提示词的向后兼容

## 4. 分层模式

生成按重要性排序的摘要，包含丰富的实体详情和时间线。

**输出格式：**
```
=== Core Entities (High Importance) ===（核心实体 - 高重要性）
- Dragon [creature/生物] (importance/重要性: 0.85, turn/回合 3)
  Description/描述: A fearsome red dragon guarding the treasure（一条守护宝藏的可怕红龙）
  Status/状态: {health: injured/受伤, mood: aggressive/攻击性}
  Emotion/情绪: fearful（恐惧）
  History/历史: turn 2: health=healthy→injured; turn 3: mood=calm→aggressive
  Relations/关系: located_at→forest (0.9), enemy_of→player (0.8)

=== Secondary Entities ===（次要实体）
- Forest [location/地点] (importance/重要性: 0.52, turn/回合 3)
  Description/描述: A dark, ancient forest（一片黑暗的古老森林）
  Relations/关系: contains→dragon (0.7)

=== Background ===（背景实体）
- Tavern [location/地点] (importance/重要性: 0.15, last seen/最后出现 turn/回合 0)
- Innkeeper [person/人物] (importance/重要性: 0.12, last seen/最后出现 turn/回合 0)

=== Recent Timeline ===（近期时间线）
- Turn 3: Player attacked the dragon with their sword（玩家用剑攻击了龙）
- Turn 2: Dragon breathed fire at the player（龙向玩家喷火）
- Turn 1: Player entered the dark forest（玩家进入了黑暗森林）
```

**层级阈值：**
- **核心（Core）：** 重要性 ≥ 0.6
- **次要（Secondary）：** 0.3 ≤ 重要性 < 0.6
- **背景（Background）：** 重要性 < 0.3

**实体块组件：**
- 名称和类型，附带重要性评分
- 描述（如有）
- 当前状态（如有）
- 最后关联情绪（如有）
- 近期状态历史（最近 3 条）
- 出边关系，附带置信度评分

**时间线：**
- 显示最近的 `KG_MAX_TIMELINE_ENTRIES`（默认：5）条关系确认记录
- 按 `last_confirmed_turn` 降序排列

## 5. 配置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `KG_SUMMARY_MODE` | `layered` | `flat`（扁平）或 `layered`（分层） |
| `KG_MAX_TIMELINE_ENTRIES` | `5` | 分层时间线中的最大条目数 |

## 6. 切换模式

### 通过配置文件
```env
KG_SUMMARY_MODE=layered  # 设置为分层模式
```

### 通过代码
```python
from src.engine.game_engine import GameEngine
# 创建引擎时指定摘要模式为分层模式
engine = GameEngine(summary_mode="layered")
```

### 通过前端
在 Streamlit 侧边栏 **"⚙ KG 策略设置"** 面板中，选择 **"摘要模式"** 下拉菜单。

## 7. 性能考量

- **扁平模式** 生成速度更快（简单迭代），但可能在无关实体上浪费 token
- **分层模式** 需要按重要性排序并格式化丰富的实体块，但为 LLM 生成上下文效率更高的摘要
- 两种模式都尊重 `max_entities` 参数（默认：30）以限制输出大小

## 8. 摘要缓存

当 `KG_ENABLE_SUMMARY_CACHE` 为 `True`（默认）时，KG 摘要每回合只计算一次并缓存，避免在同一 `process_turn()` 调用期间重复遍历图谱。

---
*相关文档：重要性评分详情请见 [entity-importance.md](entity-importance.md)*
