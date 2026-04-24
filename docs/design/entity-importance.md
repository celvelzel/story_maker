# 实体重要性策略

本文档详细说明 `story_weaver` 项目中知识图谱（KG）实体剪枝与重要性评分机制。该功能由配置参数 `KG_IMPORTANCE_MODE` 控制。

## 1. 功能概述

在长篇叙事中，知识图谱会随着回合数增加而不断增长。为确保 LLM 上下文的有效性，系统需要识别哪些实体是当前故事的核心，哪些是过时的背景信息。实体剪枝策略通过计算实体的 `importance_score`（0.0 到 1.0）来决定其在上下文窗口构建和摘要生成中的优先级。

## 2. 策略模式

目前支持三种模式：`composite`（复合模式，默认）、`incremental`（增量模式）和 `degree_only`（仅度数模式）。

### 2.1 Composite（复合模式，推荐）

这是系统默认的智能模式，结合了图结构、时序衰减和玩家交互。

**计算公式：**
```
重要性 = 0.3 * norm(degree/度数) + 0.3 * recency/近期性 + 0.2 * norm(mention_count/提及次数) + 0.2 * norm(player_mention_count/玩家提及次数)
```

- **norm(degree/度数)**: 节点度数的归一化值（连接越多，基础权重越高）
- **recency/近期性**: 使用指数衰减公式 `0.95 ^ 距上次提及的回合数`。距上次提及越久，权重下降越快
- **mention_count/提及次数**: 实体在故事文本中出现的总频率
- **player_mention_count/玩家提及次数**: 玩家在输入中直接提及该实体的次数

**优势：**
- **动态剪枝**: 长时间未出现的角色（如序章角色）会自然降低重要性
- **面向玩家**: 玩家反复提及的物品或角色会获得显著的权重提升
- **上下文友好**: 确保有限的上下文空间始终保留给最活跃的实体

### 2.2 Incremental（增量模式，性能优化）

复合模式的优化版本，仅对"脏"（已修改）节点重新计算重要性评分，并定期进行完整重计算作为保障。

**核心逻辑：**
- 仅当前回合被修改（新增、提及或状态变更）的实体才会重新计算重要性
- 每 `KG_INCREMENTAL_FULL_RECALC_INTERVAL` 回合（默认：10）执行一次完整重计算，防止偏差累积
- 由配置中的 `KG_ENABLE_INCREMENTAL_IMPORTANCE` 开关控制

**适用场景：**
- 会话较长、KG 规模较大（>100 个节点），完整重计算开销较大时
- 每回合延迟是关键考量时

### 2.3 Degree Only（仅度数模式）

传统的图论评估方法，主要用于向后兼容或极简场景。

**核心逻辑：**
实体的权重完全取决于其在图中的边数。

**局限性：**
- **无法剪枝**: 早期建立大量关系的"已故"角色将永远保持高分数
- **缺乏时效性**: 无法区分当前焦点与历史背景

## 3. 配置参数

您可以在 `.env` 文件中微调以下参数来调整剪枝行为：

| 参数 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `KG_IMPORTANCE_MODE` | `composite` | 切换模式：`composite`（复合）或 `degree_only`（仅度数） |
| `KG_IMPORTANCE_DECAY_FACTOR` | `0.95` | 每回合未被提及时的重要性衰减因子 |
| `KG_IMPORTANCE_MENTION_BOOST` | `0.15` | 故事中被提及一次的重要性提升值 |
| `KG_IMPORTANCE_PLAYER_BOOST` | `0.3` | 被玩家直接提及时的额外权重提升值 |

## 4. 切换方式

### 1. 配置文件
在 `.env` 中修改：
```env
KG_IMPORTANCE_MODE=composite  # 设置为复合模式
```

### 2. 前端界面
在 Streamlit 侧边栏 **"⚙ KG 策略设置"** 面板中，选择 **"实体剪枝策略"** 下拉菜单。

### 3. 代码调用
```python
from src.engine.game_engine import GameEngine
# 创建引擎时指定重要性模式为复合模式
engine = GameEngine(importance_mode="composite")
```

## 5. 实现细节

重要性评分在每次 `update_graph` 调用期间由 `KnowledgeGraphManager` 计算。评分低于阈值（默认 0.1）的实体可能被排除在即时 LLM 上下文窗口之外以节省 token，但仍保留在完整图谱中以便未来可能的召回。

---
*相关参考：详细技术实现请见 [kg-optimization.md](../reports/optimization/kg-optimization.md)*
