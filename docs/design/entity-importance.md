# 实体淘汰策略 (Entity Importance Strategy)

本文档详细介绍了 `story_maker` 项目中知识图谱 (KG) 的实体淘汰与重要性评分机制。该功能主要通过配置文件中的 `KG_IMPORTANCE_MODE` 参数进行管理。

## 一、 功能概述

在长篇叙事中，知识图谱会随着回合数增加而变得庞大。为了保证 LLM 上下文的有效性，系统需要识别哪些实体是当前故事的核心，哪些是过时的背景。实体淘汰策略决定了实体的 `importance_score`（0.0 到 1.0），从而影响摘要生成时的优先级。

## 二、 策略对比

目前支持两种模式：`composite` (默认) 和 `degree_only`。

### 1. Composite (复合模式) - 推荐使用
这是系统默认的智能模式，结合了图论结构、时间衰减和玩家交互。

**计算公式：**
`Importance = 0.3 * norm(degree) + 0.3 * recency + 0.2 * norm(mention_count) + 0.2 * norm(player_mention_count)`

*   **norm(degree)**: 节点的归一化度数（连接的关系边越多，基础权重越高）。
*   **recency (新鲜度)**: 采用指数衰减公式 `0.95 ^ turns_since_last_mention`。距离上次被提及的时间越长，权重下降越快。
*   **mention_count**: 实体在故事文本中出现的总频率。
*   **player_mention_count**: 玩家在输入中直接提到该实体的次数。

**优点：**
- **动态淘汰**：很久不出现的角色（如序章人物）会自然失去重要性。
- **玩家导向**：玩家反复提到的道具或人物会获得显著的权重提升（`KG_IMPORTANCE_PLAYER_BOOST`）。
- **上下文友好**：确保生成摘要时，有限的空间始终留给最活跃的实体。

### 2. Degree Only (仅度数模式)
这是传统的图论评价方式，主要用于向后兼容或极简场景。

**核心逻辑：**
实体的权重完全取决于它在图谱中拥有的边数。

**局限性：**
- **无法淘汰**：一个早期建立大量关系的“死掉”的角色，其分值将永远保持高位。
- **缺乏时效性**：无法区分当前焦点和历史背景。

## 三、 相关配置参数

你可以在 `.env` 文件中微调以下参数以改变淘汰行为：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `KG_IMPORTANCE_MODE` | `composite` | 切换模式：`composite` 或 `degree_only` |
| `KG_IMPORTANCE_DECAY_FACTOR` | `0.95` | 每回合不被提及的重要性衰减系数 |
| `KG_IMPORTANCE_MENTION_BOOST` | `0.15` | 每次被故事提及的重要性提升值 |
| `KG_IMPORTANCE_PLAYER_BOOST` | `0.3` | 被玩家直接提及的额外权重加成 |

## 四、 如何切换

### 1. 配置文件
在 `.env` 中修改：
```env
KG_IMPORTANCE_MODE=composite
```

### 2. 前端界面
在 Streamlit 侧边栏的 **"⚙ KG 策略设置"** 面板中，选择 **"实体淘汰策略"** 下拉框。

### 3. 代码调用
```python
engine = GameEngine(importance_mode="composite")
```

---
*相关参考：详细的技术实现请查阅 [docs/kg_optimization_report.md](./kg_optimization_report.md)*