# 冲突检测与解决

> **最后更新：** 2026-04-01  
> **模块：** `src/knowledge_graph/conflict_detector.py`

## 1. 概述

冲突检测与解决系统确保 StoryWeaver 知识图谱中的叙事一致性。它作为多层检测流水线运行，支持可配置的解决策略。

## 2. 检测架构

三个检测层按顺序运行：

```
┌─────────────────────────────────────────────────────┐
│              ConflictDetector.check_all()            │
│                                                     │
│  第 1 层：基于规则（确定性）                           │
│  ├── 互斥关系对（ally_of ↔ enemy_of）                │
│  └── 死亡活跃检测                                    │
│                                                     │
│  第 1b 层：时序检测（确定性）                          │
│  ├── 死后实体行为                                    │
│  └── 因果倒置（causes/enables）                      │
│                                                     │
│  第 2 层：基于 LLM（概率性）                           │
│  └── 逻辑矛盾分析                                    │
│      └── 置信度分区：                                │
│          ≥ 0.75 → 接受                               │
│          0.45-0.74 → 延迟处理                        │
│          < 0.45 → 丢弃                               │
└─────────────────────────────────────────────────────┘
```

### 2.1 基于规则的检测

检测图谱中的确定性矛盾：

| 冲突类型 | 检测逻辑 |
|---------------|----------------|
| `exclusive_relation`（互斥关系） | 同一 source→target 同时存在 `ally_of`（盟友）和 `enemy_of`（敌人），或 `alive`（存活）和 `dead`（死亡） |
| `dead_active`（死亡活跃） | 标记为 `dead`（死亡）的实体仍存在活跃关系（`possesses`/拥有、`located_at`/位于、`ally_of`/盟友） |

### 2.2 时序检测

检测基于时间的矛盾：

| 冲突类型 | 检测逻辑 |
|---------------|----------------|
| `dead_entity_action`（死后行为） | 实体在死亡回合之后创建了关系 |
| `causal_inversion`（因果倒置） | 对于 `causes`/`enables` 关系，结果实体先于原因实体创建 |

### 2.3 基于 LLM 的检测

将当前世界状态和新故事文本发送给 LLM 进行逻辑矛盾分析。返回包含冲突描述和置信度评分的 JSON。

**置信度分区：**
- **≥ 0.75**：接受并进入解决流程
- **0.45–0.74**：延迟处理（跟踪但不解决）
- **< 0.45**：丢弃（噪声过大）

## 3. 解决策略

通过 `KG_CONFLICT_RESOLUTION` 设置可选择两种策略：

### 3.1 `keep_latest`（保留最新，确定性）

| 冲突类型 | 解决方式 |
|---------------|------------|
| `exclusive_relation`（互斥关系） | 移除 `last_confirmed_turn`（最后确认回合）较旧的关系 |
| `dead_active`（死亡活跃） | 移除来自死亡实体的活跃关系 |
| `temporal`（时序） | 不解决（需要人工干预） |
| `llm`（LLM 检测） | 不解决（无法确定性解决） |

### 3.2 `llm_arbitrate`（LLM 仲裁，混合）

**确定性优先通道：** 使用 `KeepLatestResolver` 逻辑解决 `exclusive_relation` 和 `dead_active` 冲突。

**LLM 仲裁：** 对于置信度 ≥ 0.75 的 LLM 检测冲突，调用 LLM 决定解决方式：
- `keep_new` — 移除旧关系
- `keep_old` — 移除新关系
- `remove_relation` — 移除特定关系
- `update_entity` — 更新实体状态
- `no_action` — 非真实冲突

**时序冲突：** 始终不解决，留待后续明确处理。

## 4. 冲突输出

每个检测到的冲突是一个字典：

```python
{
    "type": "exclusive_relation" | "dead_active" | "temporal" | "llm",  # 冲突类型
    "source": str,        # 源实体
    "target": str,        # 目标实体
    "description": str,   # 人类可读的描述
    "confidence": str,    # 置信度评分（仅 LLM 冲突）
    # 每种类型的附加字段：
    "relation_a": str,    # 用于互斥关系
    "relation_b": str,    # 用于互斥关系
    "subtype": str,       # 用于时序冲突："dead_entity_action"（死后行为）| "causal_inversion"（因果倒置）
    "death_turn": int,    # 用于时序冲突
    "relation_turn": int, # 用于时序冲突
}
```

## 5. 配置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `KG_CONFLICT_RESOLUTION` | `llm_arbitrate` | `keep_latest`（保留最新）或 `llm_arbitrate`（LLM 仲裁） |

### 内部阈值

| 常量 | 值 | 描述 |
|----------|-------|-------------|
| `LLM_CONFLICT_ACCEPT_THRESHOLD` | `0.75` | 接受 LLM 冲突的最低置信度 |
| `LLM_CONFLICT_DEFER_LOW` | `0.45` | 延迟处理区间下限 |
| `LLM_CONFLICT_DEFER_HIGH` | `0.74` | 延迟处理区间上限 |

## 6. 使用方式

```python
from src.knowledge_graph.conflict_detector import ConflictDetector, get_resolver

# 创建检测器
detector = ConflictDetector(kg)

# 运行所有检测层
conflicts = detector.check_all(new_story_text)

# 使用配置的策略解决
resolver = get_resolver("llm_arbitrate")  # 获取 LLM 仲裁解决器
unresolved = resolver.resolve(conflicts, kg)
```

---
*相关文档：KG 重要性评分请见 [entity-importance.md](entity-importance.md)*
