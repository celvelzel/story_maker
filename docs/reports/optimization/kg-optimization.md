# 知识图谱优化报告

**Commit Hash**：`d1d59c7` + `b1799ed` | **最后更新**：2026-03-31

---

## 1. 概述

本报告详细说明了知识图谱（KG）子系统的全面增强，涵盖数据模型丰富度、每回合更新逻辑、冲突解决策略以及前端配置面板。新增了 **76 个单元测试**，全部通过。

### 1.1 变更摘要

| 文件 | 变更行数 | 类型 | 描述 |
|------|--------------|------|-------------|
| `config.py` | +21 行 | 更新 | 添加策略配置和调优参数 |
| `src/knowledge_graph/graph.py` | +300 行 | 重写 | 数据结构、新方法、分层摘要 |
| `src/knowledge_graph/relation_extractor.py` | +144 行 | 重写 | 增强提示词和双重抽取 |
| `src/knowledge_graph/conflict_detector.py` | +195 行 | 重写 | 策略模式，两种实现 |
| `src/engine/game_engine.py` | +135 行 | 重写 | 7 步回合处理，含策略注入 |
| `app.py` | +58 行 | 更新 | 添加 KG 策略设置面板 |
| `tests/` | +900+ 行 | 新增 | 为图谱、抽取器和引擎添加 76 个单元测试 |

---

## 2. 功能详情

### 2.1 数据模型增强

**节点（实体）** 现在追踪：
- `name`（名称）、`entity_type`（实体类型）、`description`（描述，叙事细节）
- `status`（状态）：动态字典（如 `{"health": "injured"/受伤}`）
- **指标**：`created_turn`（创建回合）、`last_mentioned_turn`（最后提及回合）、`mention_count`（提及次数）、`player_mention_count`（玩家提及次数），以及计算得出的 `importance_score`（重要性评分，0-1）

**边（关系）** 现在追踪：
- `relation`（关系类型）、`context`（上下文，关系存在的原因）
- `created_turn`（创建回合）、`last_confirmed_turn`（最后确认回合）、`confidence`（置信度评分，0-1）

### 2.2 新增核心方法
- `update_entity_state(...)`：更新实体状态字典
- `refresh_mentions(...)`：批量更新提及次数并提升重要性
- `apply_decay(...)`：降低旧关系的置信度，低于阈值时修剪
- `recalculate_importance()`：基于度数、近期性和提及次数计算重要性

### 2.3 分层摘要输出

通过 `KG_SUMMARY_MODE` 支持两种模式：
1. **分层模式（默认）**：按重要性将实体分组（核心、次要、背景），并提供近期时间线
2. **扁平模式（旧版）**：简单的实体和关系列表，用于向后兼容

### 2.4 双重实体抽取

同时从**玩家输入**和**故事文本**中抽取实体和关系：
- 合并结果并去重
- 优先使用故事文本中更丰富的描述
- 更准确地捕获玩家直接交互

### 2.5 冲突解决策略

使用策略模式灵活处理冲突：
- **`keep_latest`（保留最新）**：快速、基于规则的解决，偏向最新确认的信息。无需 LLM 调用
- **`llm_arbitrate`（LLM 仲裁，默认）**：将冲突详情发送给 LLM 进行高精度仲裁

---

## 3. 配置与使用

### 3.1 前端配置面板

Streamlit 侧边栏中新增 **"⚙ KG 策略设置"** 面板，支持实时调整：
- 冲突解决策略
- 抽取模式
- 摘要格式
- 重要性计算策略

### 3.2 全局配置（.env）
```env
KG_CONFLICT_RESOLUTION=llm_arbitrate  # 冲突解决：LLM 仲裁
KG_EXTRACTION_MODE=dual_extract       # 抽取模式：双重抽取
KG_SUMMARY_MODE=layered               # 摘要模式：分层
KG_IMPORTANCE_DECAY_FACTOR=0.95       # 重要性衰减因子
KG_RELATION_DECAY_FACTOR=0.90         # 关系衰减因子
```

---

## 4. 架构与逻辑

### 4.1 回合处理流水线（更新版）
1. NLU（共指消解、意图分类、情感分析、实体抽取）
2. 故事生成
3. **KG 更新**：
   - 抽取（双重或仅故事）
   - 应用状态变更
   - 刷新提及并应用衰减
   - 重新计算重要性
4. **冲突检测与解决**
5. 选项生成

### 4.2 重要性评分公式
`importance（重要性） = 0.3 * norm(degree/度数) + 0.3 * recency/近期性 + 0.2 * norm(mentions/提及次数) + 0.2 * norm(player_mentions/玩家提及次数)`

---

## 5. 测试与验证

| 测试套件 | 数量 | 覆盖范围 |
|------------|-------|----------|
| `test_graph_enhanced.py` | 35 | 节点/边属性、衰减、重要性、时间线 |
| `test_relation_extractor_enhanced.py` | 11 | 丰富抽取、标准化、合并 |
| `test_conflict_resolution.py` | 14 | 检测、保留最新、LLM 仲裁 |
| `test_engine_enhanced.py` | 16 | 策略注入、集成、时间追踪 |

**总计**：76 通过

---

## 6. 已知限制
- **LLM 成本**：`llm_arbitrate` 和 `dual_extract` 每回合增加 1-2 次 API 调用
- **持久化**：虽然已优化，图谱仍驻留在内存中（更新：参见 [持久化报告](runtime-persistence.md) 了解最新修复）
