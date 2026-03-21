# 知识图谱优化 — 变更报告

> 提交: `d1d59c7` + `b1799ed` | 日期: 2026-03-19

---

## 一、改动概览

本次提交对知识图谱子系统进行了全面增强，涵盖数据模型丰富度、每轮对话更新逻辑、冲突解决策略和前端配置面板。同时新增 76 个单元测试，全部通过。

| 文件 | 改动量 | 类型 |
|------|--------|------|
| `config.py` | +21 行 | 新增策略配置 + 调优参数 |
| `src/knowledge_graph/graph.py` | +300 行 | 重写：数据结构 + 新方法 + 分层摘要 |
| `src/knowledge_graph/relation_extractor.py` | +144 行 | 重写：增强 prompt + 双重提取 |
| `src/knowledge_graph/conflict_detector.py` | +195 行 | 重写：策略模式 + 两种实现 |
| `src/engine/game_engine.py` | +135 行 | 重写：7 步流程改造 + 策略注入 |
| `app.py` | +58 行 | 新增 KG 策略设置面板 |
| `tests/test_graph_enhanced.py` | +290 行 (新建) | graph.py 单元测试 (35 个) |
| `tests/test_relation_extractor_enhanced.py` | +155 行 (新建) | extractor 单元测试 (11 个) |
| `tests/test_conflict_resolution.py` | +190 行 (新建) | 冲突检测+解决测试 (14 个) |
| `tests/test_engine_enhanced.py` | +270 行 (新建) | 引擎集成测试 (16 个) |

---

## 二、修改功能详解

### 2.1 知识图谱数据模型增强

#### 节点（实体）属性

| 属性 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `name` | str | 原始显示名 | `"Hero"` |
| `entity_type` | str | 实体类型 | `"person"` |
| `description` | str | 叙事描述 | `"A brave warrior with a scarred face"` |
| `status` | dict | 动态状态字典 | `{"health": "injured", "mood": "determined"}` |
| `created_turn` | int | 首次创建回合 | `1` |
| `last_mentioned_turn` | int | 最近被提及回合 | `12` |
| `mention_count` | int | 总提及次数 | `8` |
| `player_mention_count` | int | 被玩家直接提及次数 | `3` |
| `importance_score` | float | 综合重要性评分 (0-1) | `0.87` |

#### 边（关系）属性

| 属性 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `relation` | str | 关系类型 | `"possesses"` |
| `context` | str | 建立此关系的上下文 | `"Hero picked up the ancient sword"` |
| `created_turn` | int | 创建回合 | `5` |
| `last_confirmed_turn` | int | 最后确认回合 | `12` |
| `confidence` | float | 置信度 (0-1) | `0.85` |

### 2.2 知识图谱新增方法

| 方法 | 说明 |
|------|------|
| `update_entity_state(name, state_updates, turn_id)` | 更新实体的 status 字典字段 |
| `refresh_mentions(mentioned_names, turn_id, player_mentioned_names)` | 批量刷新提及计数和重要性，未提及实体衰减 |
| `apply_decay(turn_id)` | 对久未确认的关系降低 confidence，低于阈值自动删除 |
| `recalculate_importance()` | 综合 degree + recency + mentions 重算重要性 |
| `get_timeline(n)` | 返回最近 n 个回合的关系事件（按时间排序） |
| `set_turn(turn_id)` | 设置当前回合号供时间追踪使用 |

### 2.3 分层摘要输出

`to_summary()` 支持两种模式，通过 `config.py` 中 `KG_SUMMARY_MODE` 控制：

**layered 模式（默认）**：
```
=== Core Entities (High Importance) ===
- Hero [person] (importance: 0.92, turn 12)
  Description: A brave warrior with a scarred face
  Status: {health: injured, mood: determined}
  Relations: possesses→sword (1.0), ally_of→mage (0.9)

=== Secondary Entities ===
- Sword [item] (importance: 0.65)
  Description: An ancient glowing blade

=== Background ===
- Old Man [person] (importance: 0.21, last seen turn 3)

=== Recent Timeline ===
- Turn 12: Hero entered the dark cave
- Turn 11: Hero defeated the dragon
```

**flat 模式（向后兼容）**：
```
=== World State ===
- Hero [person] {status: injured}
- Sword [item]

=== Relations ===
- hero --[possesses]--> sword
```

### 2.4 双重实体提取

支持从**玩家输入**和**故事文本**同时提取实体与关系，并自动合并去重：

- 玩家输入提取侧重于新实体和直接提及
- 故事文本提取侧重于叙事细节、状态变化
- 合并时优先保留信息更丰富的版本

### 2.5 每轮对话更新逻辑

改造后的每轮处理流程：

```
Step 1-3: coref → intent → entity extraction（不变）
Step 4:   Story generation（不变）

Step 5:   KG 更新（改造）
  5a. 根据配置决定是否从玩家输入提取实体/关系
  5b. 从故事文本提取（含 description/status/state_changes/context）
  5c. 将 state_changes 应用到已有实体
  5d. 刷新提及计数：mentioned 实体 + importance boost，未提及实体 decay
  5e. 应用关系时间衰减：未确认关系 confidence 递减，弱关系自动删除
  5f. 重算所有实体 importance_score

Step 6:   冲突检测 + 策略化解决
  6a. 检测冲突（规则 + LLM）
  6b. 根据配置的策略自动解决冲突
  6c. 记录解决结果

Step 7:   Option generation（不变）
```

### 2.6 冲突解决策略

采用策略模式（Strategy Pattern），支持以下实现：

#### keep_latest — 保留最新信息
- 对互斥关系对（如 ally_of / enemy_of），保留 `last_confirmed_turn` 更新的边
- 对死亡实体的活跃关系，自动删除
- 不消耗 LLM 调用，速度快

#### llm_arbitrate — LLM 仲裁（默认）
- 将冲突详情 + KG 摘要发送给 LLM
- LLM 返回：`keep_new` / `keep_old` / `remove_relation` / `no_action`
- 根据 LLM 判断自动修复 KG
- 效果最好，但每轮多一次 LLM 调用

### 2.7 前端策略设置面板

在 Streamlit 侧边栏新增 **"⚙ KG 策略设置"** 面板，包含 4 个下拉选择框：

| 设置项 | 选项 | 默认值 |
|--------|------|--------|
| 冲突解决策略 | `llm_arbitrate` / `keep_latest` | `llm_arbitrate` |
| 实体提取模式 | `dual_extract` / `story_only` | `dual_extract` |
| KG 摘要格式 | `layered` / `flat` | `layered` |
| 实体淘汰策略 | `composite` / `degree_only` | `composite` |

> 策略变更在下一次"开始新游戏"后生效。

### 2.8 调试日志

所有关键步骤均添加了结构化日志，格式为 `[模块][方法] 描述 | 关键数据`：

```
[KG][add_entity] Created 'hero' type=person turn=3 importance=0.80 | total_nodes=15
[KG][add_relation] hero --[possesses]--> sword turn=1 confidence=0.95 | edges=8
[KG][refresh_mentions] Mentioned=3 player=1 | total_nodes=12
[KG][apply_decay] Pruned 1 low-confidence edges
[Engine][process_turn] === Turn 5 START === | input='I draw my sword'
[Engine][kg_update] Dual extract: 3 entities, 2 relations
[Engine][conflicts] Detected=1 resolved=1 remaining=0 via llm_arbitrate
```

---

## 三、使用方法

### 3.1 启动项目

```bash
# 方式一：直接运行
python -m streamlit run app.py --server.port 7860

# 方式二：使用启动脚本
start_project_prod.bat    # Windows
```

浏览器访问 http://localhost:7860

### 3.2 配置策略

#### 方式一：通过 .env 文件（全局默认）

```env
# .env
KG_CONFLICT_RESOLUTION=llm_arbitrate
KG_EXTRACTION_MODE=dual_extract
KG_IMPORTANCE_MODE=composite
KG_SUMMARY_MODE=layered
KG_IMPORTANCE_DECAY_FACTOR=0.95
KG_RELATION_DECAY_FACTOR=0.90
KG_RELATION_MIN_CONFIDENCE=0.2
KG_IMPORTANCE_MENTION_BOOST=0.15
KG_IMPORTANCE_PLAYER_BOOST=0.3
KG_MAX_TIMELINE_ENTRIES=5
```

#### 方式二：通过前端界面（每局游戏）

1. 打开 http://localhost:7860
2. 在左侧边栏展开 **"⚙ KG 策略设置"**
3. 选择所需的策略组合
4. 点击 **"🎮 开始新游戏"** 使策略生效

#### 方式三：通过代码（编程调用）

```python
from src.engine.game_engine import GameEngine

engine = GameEngine(
    genre="fantasy",
    conflict_resolution="llm_arbitrate",
    extraction_mode="dual_extract",
    importance_mode="composite",
    summary_mode="layered",
)
result = engine.start_game()
```

### 3.3 策略对比实验

推荐的对比组：

| 实验组 | 冲突策略 | 提取模式 | 摘要格式 | 淘汰策略 | 适用场景 |
|--------|----------|----------|----------|----------|----------|
| A（默认） | llm_arbitrate | dual_extract | layered | composite | 最佳效果 |
| B | keep_latest | dual_extract | layered | composite | 不消耗额外 LLM 调用 |
| C | llm_arbitrate | story_only | flat | degree_only | 原始行为基线 |
| D | keep_latest | story_only | flat | degree_only | 最简配置 |

通过前端设置面板切换策略后开始新游戏，运行若干轮后对比评测面板中的指标。

### 3.4 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行新增的 KG 相关测试
python -m pytest tests/test_graph_enhanced.py tests/test_relation_extractor_enhanced.py tests/test_conflict_resolution.py tests/test_engine_enhanced.py -v

# 运行单个测试文件
python -m pytest tests/test_graph_enhanced.py -v
```

---

## 四、配置参数参考

### 策略配置

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `KG_CONFLICT_RESOLUTION` | str | `llm_arbitrate` | `keep_latest`, `llm_arbitrate` | 冲突解决策略 |
| `KG_EXTRACTION_MODE` | str | `dual_extract` | `story_only`, `dual_extract` | 实体提取模式 |
| `KG_IMPORTANCE_MODE` | str | `composite` | `degree_only`, `composite` | 实体淘汰策略 |
| `KG_SUMMARY_MODE` | str | `layered` | `flat`, `layered` | KG 摘要格式 |

### 调优参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `KG_IMPORTANCE_DECAY_FACTOR` | float | `0.95` | 每轮不被提及的重要性衰减系数 |
| `KG_RELATION_DECAY_FACTOR` | float | `0.90` | 每轮不被确认的关系衰减系数 |
| `KG_RELATION_MIN_CONFIDENCE` | float | `0.2` | 关系置信度低于此值时自动删除 |
| `KG_IMPORTANCE_MENTION_BOOST` | float | `0.15` | 每次被提及的重要性提升值 |
| `KG_IMPORTANCE_PLAYER_BOOST` | float | `0.3` | 被玩家直接提及的额外提升值 |
| `KG_MAX_TIMELINE_ENTRIES` | int | `5` | 摘要中时间线显示的最大条目数 |

---

## 五、架构说明

### 5.1 策略模式（Strategy Pattern）

```
ConflictResolutionStrategy (ABC)
├── KeepLatestResolver    — 保留最新，无需 LLM
└── LLMArbitrateResolver  — LLM 仲裁，效果最好

get_resolver(mode) → ConflictResolutionStrategy  ← 工厂函数
```

新增策略只需：
1. 继承 `ConflictResolutionStrategy`
2. 实现 `resolve(conflicts, kg)` 方法
3. 在 `get_resolver()` 中注册
4. 在 `config.py` 中添加选项值
5. 在 `app.py` 的 selectbox 中添加选项

### 5.2 数据流

```
玩家输入
  │
  ├─→ NLU 实体提取 ──────────────────┐
  │                                    │
  ├─→ 核心消解 → 意图分类 → 故事生成    │
  │                                    │
  │  故事文本 ─→ 双重提取(LLM) ────────┤
  │                                    │
  │  玩家输入 ─→ 玩家提取(LLM) ────────┤
  │                                    ▼
  │                          ┌─────────────────┐
  │                          │   KG 更新流程    │
  │                          │ 1. 添加实体      │
  │                          │ 2. 添加关系      │
  │                          │ 3. 状态更新      │
  │                          │ 4. 提及刷新      │
  │                          │ 5. 时间衰减      │
  │                          │ 6. 重要性重算    │
  │                          └────────┬────────┘
  │                                   │
  │  冲突检测 ─→ 策略化解决 ──────────┘
  │
  └─→ 选项生成 ← KG 摘要
```

### 5.3 重要性评分公式（composite 模式）

```
importance = 0.3 × norm(degree)
           + 0.3 × recency
           + 0.2 × norm(mention_count)
           + 0.2 × norm(player_mention_count)

其中:
- norm(x) = x / max(all_x)  (归一化到 0-1)
- recency = 0.95 ^ turns_since_last_mention  (指数衰减)
```

---

## 六、测试覆盖

| 测试文件 | 测试数 | 覆盖范围 |
|----------|--------|----------|
| `test_graph_enhanced.py` | 35 | 节点属性、边属性、状态更新、提及刷新、衰减、重要性、时间线、分层摘要、向后兼容 |
| `test_relation_extractor_enhanced.py` | 11 | 富属性提取、字段归一化、错误处理、双提取、实体合并 |
| `test_conflict_resolution.py` | 14 | 冲突检测、KeepLatest 解决、LLM 仲裁、工厂函数 |
| `test_engine_enhanced.py` | 16 | 引擎初始化、策略注入、双提取模式、冲突解决集成、摘要模式、时间追踪 |
| **合计** | **76** | **全部通过** |

运行结果示例：

```
tests/test_graph_enhanced.py ............................. 35 passed
tests/test_relation_extractor_enhanced.py ................ 11 passed
tests/test_conflict_resolution.py ....................... 14 passed
tests/test_engine_enhanced.py ........................... 16 passed
```

---

## 七、已知限制

1. **LLM 依赖**：`llm_arbitrate` 和 `dual_extract` 模式每轮多消耗 1-2 次 LLM API 调用
2. **实体类型未严格校验**：LLM 返回的 type 可能与 config 中定义的不完全一致
3. **无持久化**：KG 仍在内存中，刷新页面或重启后丢失
4. **测试环境**：原始 `test_integration.py` 中有 2 个测试因 DistilBERT 模型兼容性问题失败（与本次改动无关）
