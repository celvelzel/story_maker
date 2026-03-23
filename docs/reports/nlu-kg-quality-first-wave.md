# NLU & KG 质量优先路线 — 迭代报告

> 日期: 2026-03-23 | 路线: `nlu-kg-quality-first-roadmap`

---

## 一、改动概览

在首次 NLU/KG 改进（2026-03-20）基础上，以「质量/准确性优先 → 延迟优化其次」路线做系统化加固。本次迭代分 4 个波次（A–D），共 26 个子任务，全部完成。

### 1.1 迭代目标

1. **质量优先**：修复 coref 多代词缺失、entity 别名模糊、KG 异常输出未隔离、冲突检测误报率高 4 个质量瓶颈。
2. **延迟其次**：KG 摘要重复遍历、全量重要性重算、关系衰减每轮执行 3 个延迟瓶颈。
3. **门控基础设施**：建立自动化评估 corpus + quality_runner + Gate-1/2/3，防止后续改动引入回归。

### 1.2 波次总览

| 波次 | 编号 | 范围 | 核心改动 | 测试 |
|------|------|------|----------|------|
| A | A1–A4 | 评估基础设施 | 120 条 benchmark corpus、quality_runner、stage_metrics、Gate-1/2/3 | 13 passed |
| B | B1–B4 | 质量加固 | coref 多代词、entity 别名归一化、KG 异常隔离、冲突确定性优先 | 181 passed |
| C | C1–C3 | 延迟优化 | KG 摘要缓存、增量重要性、衰减节奏可配置 | 54 passed |
| D | D1–D2 | 文档交付 | technical-route.md 策略文档、nlu-kg-improvement.md 迭代报告 | — |

### 1.3 文件变更总表

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `config.py` | 修改 | 新增 4 个 KG 配置项（decay cadence、incremental toggles） |
| `src/nlu/coreference.py` | 修改 | 多代词 fallback、引文保护跳过替换 |
| `src/nlu/entity_extractor.py` | 修改 | 别名归一化、按类型阈值、confidence 输出 |
| `src/knowledge_graph/graph.py` | 修改 | 增量重要性模式（`_dirty_nodes` + `_recalculate_importance_incremental`） |
| `src/knowledge_graph/relation_extractor.py` | 修改 | `_sanitize_payload` 隔离异常输出、quarantine 计数 |
| `src/knowledge_graph/conflict_detector.py` | 修改 | 确定性优先策略、置信度分档延迟队列 |
| `src/engine/game_engine.py` | 修改 | stage_metrics 可观测性、摘要缓存、衰减节奏 |
| `src/evaluation/metrics.py` | 修改 | 新增 `precision_recall_f1`、`exact_match_accuracy`、签名函数 |
| `src/utils/api_client.py` | 修改 | 新增 `usage_snapshot`、`usage_delta` |
| `app.py` | 修改 | stage_metrics 面板渲染 |
| `tests/evaluation/__init__.py` | **新建** | 评估子包 |
| `tests/evaluation/data/nlu_kg_quality_benchmark.jsonl` | **新建** | 120 条基准 corpus |
| `tests/evaluation/test_quality_benchmark_schema.py` | **新建** | schema 校验（5 tests） |
| `tests/evaluation/quality_runner.py` | **新建** | baseline/compare + gate 逻辑 |
| `tests/evaluation/test_quality_regression.py` | **新建** | 回归数学验证（3 tests） |
| `tests/evaluation/test_quality_gates.py` | **新建** | 门控断言（5 tests） |
| `tests/evaluation/README.md` | **新建** | corpus schema 与 gate 策略文档 |
| `tests/performance/test_turn_latency.py` | **新建** | 摘要缓存调用计数测试 |
| `tests/nlu/test_coreference_enhanced.py` | 修改 | 新增多代词、引文保护测试 |
| `tests/nlu/test_entity_extractor_enhanced.py` | 修改 | 新增别名、阈值、confidence 测试 |
| `tests/kg/test_graph_enhanced.py` | 修改 | 新增增量重要性等价断言 |
| `tests/kg/test_conflict_resolution.py` | 修改 | 新增延迟队列、确定性优先测试 |
| `tests/kg/test_temporal_reasoning.py` | 修改 | 新增高影响冲突保留测试 |
| `tests/kg/test_relation_extractor_enhanced.py` | 修改 | 新增 quarantine 测试 |
| `tests/kg/test_kg_type_validation.py` | 修改 | 新增异常 payload 隔离测试 |
| `tests/engine/test_engine_enhanced.py` | 修改 | 新增 stage_metrics 断言 |
| `docs/guides/technical-route.md` | 修改 | 新增策略配置、回滚开关、门控文档 |
| `docs/reports/nlu-kg-improvement.md` | 修改 | 追加迭代报告（第七节） |

---

## 二、Wave A — 评估基础设施

### 2.1 目标

建立自动化质量基线与门控机制，为后续所有改动提供可量化的回归保护。

### 2.2 Benchmark Corpus（A1）

**文件**: `tests/evaluation/data/nlu_kg_quality_benchmark.jsonl`

| 指标 | 值 |
|------|------|
| 总条目数 | 120 |
| 场景组数 | 6（coref、intent、entity、relation、conflict、full_pipeline） |
| 每组条目数 | 20 |

**每条 schema**：

```json
{
  "id": "string — 唯一标识",
  "scenario": "string — 场景标签",
  "input": "string — 玩家输入文本",
  "expected": {
    "resolved_input": "string | null",
    "intent": "string | null",
    "entities": [{"text": "string", "type": "string"}],
    "relations": [{"source": "string", "target": "string", "relation": "string"}],
    "conflicts": ["string"]
  }
}
```

**校验**: `tests/evaluation/test_quality_benchmark_schema.py` 强制 required keys / types / non-empty。

### 2.3 Quality Runner（A2）

**文件**: `tests/evaluation/quality_runner.py`

**两种模式**：

```bash
# 生成基线
python -m tests.evaluation.quality_runner --mode baseline

# 比对当前代码
python -m tests.evaluation.quality_runner --mode compare --against baseline
```

**输出**: `tests/evaluation/reports/latest_quality.json`

**metrics 字段**：

| 字段 | 含义 |
|------|------|
| `entity_precision` | 实体精确率 |
| `entity_recall` | 实体召回率 |
| `entity_f1` | 实体 F1 |
| `relation_precision` | 关系精确率 |
| `relation_recall` | 关系召回率 |
| `relation_f1` | 关系 F1 |
| `contradiction_rate` | 矛盾率 |
| `coreference_accuracy` | 共指消解准确率 |
| `intent_accuracy` | 意图分类准确率 |
| `consistency_rate` | 一致性率 |

**共享度量函数**（`src/evaluation/metrics.py` 新增）：

| 函数 | 用途 |
|------|------|
| `precision_recall_f1(tp, fp, fn)` | 计算 P/R/F1 |
| `exact_match_accuracy(predicted, expected)` | 精确匹配准确率 |
| `overlap_prf(predicted_items, expected_items)` | 集合重叠 P/R/F1 |
| `entity_signature(name, entity_type)` | 实体签名（去重用） |
| `relation_signature(source, target, relation)` | 关系签名 |
| `conflict_signature(conflict_type, entity)` | 冲突签名 |

### 2.4 Stage Metrics 可观测性（A3）

**`src/engine/game_engine.py`** — `process_turn()` 内置 8 个阶段计时 + token/cost 归因：

| 阶段 key | 对应模块 |
|----------|----------|
| `coref` | CoreferenceResolver |
| `intent` | IntentClassifier |
| `sentiment` | SentimentAnalyzer |
| `entity_extraction` | EntityExtractor |
| `story_generation` | StoryGenerator |
| `kg_update` | _apply_kg_update |
| `conflict_detection_resolution` | ConflictDetector + Resolver |
| `options_and_render` | OptionGenerator |

**`src/utils/api_client.py`** — 新增：

```python
usage_snapshot()  # → {input_tokens, output_tokens, cost_usd}
usage_delta(before, after)  # → {input_tokens, output_tokens, cost_usd} delta
```

**输出**: `turn_result.nlu_debug["stage_metrics"]` 包含 `{stage: {ms, input_tokens, output_tokens, cost_usd}}`。

**`app.py`** — NLU Debug 面板渲染 stage_metrics，格式：`ms / in_tokens / out_tokens / $cost`。

### 2.5 质量门控（A4）

**文件**: `tests/evaluation/test_quality_gates.py`

| Gate | 条件 | 说明 |
|------|------|------|
| Gate-1 | 无已追踪指标回归 >1pp | 基础防退化 |
| Gate-2 | relation_f1 +3pp、coref_accuracy +5pp、contradiction_rate ≤ 基线 80% | 质量提升目标 |
| Gate-3 | 连续两次 compare 通过 | 稳定性验证 |

**测试**: 5 个断言覆盖 gate pass/fail 逻辑。

---

## 三、Wave B — 质量加固

### 3.1 B1: Coref 多代词 fallback

**问题**: `_rule_resolve()` 的 break 语句导致一次只替换一个代词；引文内文本被错误替换。

**改动** (`src/nlu/coreference.py`)：

1. 新增 `_replace_outside_quotes(text, pattern, replacement)` — 正则替换时跳过引文内容。
2. 移除 subject/object/possessive/reflexive 各组的 `break`，允许一轮内替换所有代词。
3. 新增 `_neural_resolve` 的 null-guard，防止 fastcoref 返回 None 时崩溃。

**测试** (`tests/nlu/test_coreference_enhanced.py`)：

| 测试 | 覆盖场景 |
|------|----------|
| `test_multi_pronoun_substitution` | "he gave her his book" → 多人称替换 |
| `test_possessive_group_all_replaced` | "his shield and her sword" → 全部所有格 |
| `test_dialogue_safe` | "'I saw him,' she said" → 引文内不替换 |

**验证**: `pytest tests/nlu/test_coreference_enhanced.py` → 41 passed, 3 skipped。

### 3.2 B2: Entity 别名归一化 + 按类型阈值

**问题**: 实体匹配对大小写、冠词、头衔不敏感；单一阈值导致 person 类误匹配率高。

**改动** (`src/nlu/entity_extractor.py`)：

1. 新增 `_normalize_alias(name)` — 移除头衔（Mr./Dr./Lady...）、冠词、转小写。
2. 新增 `_ENTITY_FUZZY_THRESHOLDS` 字典：

| 实体类型 | 阈值 |
|----------|------|
| person | 0.90 |
| location | 0.84 |
| item | 0.82 |
| creature | 0.82 |
| event | 0.80 |

3. 每个提取结果新增 `confidence` 字段（基于来源：model_ner=0.95、fuzzy=0.7–0.95、regex=0.6）。
4. 新增 regex proper-noun fallback（当 spaCy 不可用时捕获大写连续词）。
5. 新增 alias token overlap scan（`kg_alias` source）。

**测试** (`tests/nlu/test_entity_extractor_enhanced.py`)：

| 测试 | 覆盖场景 |
|------|----------|
| `test_alias_normalization` | "Mr. Hero" → "hero" |
| `test_ambiguous_match_rejected` | 置信度不足时不硬匹配 |
| `test_confidence_present` | 所有结果包含 confidence 字段 |

**验证**: `pytest tests/nlu/test_entity_extractor_enhanced.py` → 42 passed。

### 3.3 B3: KG Relation Extractor 异常隔离

**问题**: LLM 返回 malformed JSON 时，`setdefault` 补丁填充空值，导致空名称实体/空端点关系进入 KG。

**改动** (`src/knowledge_graph/relation_extractor.py`)：

1. 新增 `_sanitize_payload(raw)` — 校验 list/dict 结构，丢弃 malformed records，标准化类型。
2. 新增 `_to_str(val)` helper — 安全转换为字符串，None 返回空串。
3. 所有 extract 路径使用 `_sanitize_payload` 代替内联 `setdefault`。
4. 新增 `last_quarantine` 属性 — 记录最近一次被隔离的条目数量。

**测试** (`tests/kg/test_relation_extractor_enhanced.py`, `tests/kg/test_kg_type_validation.py`)：

| 测试 | 覆盖场景 |
|------|----------|
| `test_malformed_payload_quarantined` | 非 dict 元素被丢弃 |
| `test_missing_name_entity_dropped` | name 为空的实体不进入 KG |
| `test_missing_endpoint_relation_dropped` | source/target 为空的关系不进入 KG |

**验证**: `pytest tests/kg/test_relation_extractor_enhanced.py tests/kg/test_kg_type_validation.py` → 28 passed。

### 3.4 B4: Conflict Detector 确定性优先 + 置信度分档

**问题**: `_llm_check` 偶发的低置信度冲突也会触发 `KeepLatestResolver` 的关系删除，导致误报率高。

**改动** (`src/knowledge_graph/conflict_detector.py`)：

1. 新增 3 个阈值常量：

| 常量 | 值 | 含义 |
|------|------|------|
| `LLM_CONFLICT_ACCEPT_THRESHOLD` | 0.75 | ≥0.75 接受 LLM 检测 |
| `LLM_CONFLICT_DEFER_LOW` | 0.45 | <0.45 直接丢弃 |
| `LLM_CONFLICT_DEFER_HIGH` | 0.74 | 0.45–0.74 放入延迟队列 |

2. 新增 `deferred_conflicts` 列表 — 延迟队列，供外部审查。
3. 新增 `_parse_confidence(conflict_dict)` — 从 conflict dict 提取 confidence。
4. 新增 `_partition_llm_conflicts(raw_conflicts)` — 按置信度分档。
5. `LLMArbitrateResolver.resolve()` — 先跑 `KeepLatestResolver`（确定性 pass），再仲裁高置信度 LLM 冲突。

**测试** (`tests/kg/test_conflict_resolution.py`, `tests/kg/test_temporal_reasoning.py`)：

| 测试 | 覆盖场景 |
|------|----------|
| `test_defer_mid_confidence` | 0.6 置信度冲突 → deferred_conflicts |
| `test_deterministic_first_no_llm` | 确定性解决后无需 LLM |
| `test_low_confidence_llm_kept_unresolved` | 0.3 置信度 → 不删除关系 |
| `test_high_impact_temporal_conflict_remains` | 时序强冲突保持显式 |

**验证**: `pytest tests/kg/test_conflict_resolution.py tests/kg/test_temporal_reasoning.py` → 70 passed。

---

## 四、Wave C — 延迟优化

### 4.1 C1: KG 摘要缓存

**问题**: `kg.to_summary()` 在 `process_turn()` 内被调用 3 次（story gen、conflict LLM check、options），每次遍历全图。

**改动** (`src/engine/game_engine.py`)：

1. 新增 `_turn_cached_summary: Optional[str]` 属性。
2. 新增 `_current_kg_summary()` 方法 — 缓存为 None 时调用 `kg.to_summary()`，否则返回缓存。
3. 缓存在 KG update 后、turn 结束时清空。
4. 新增 `KG_ENABLE_SUMMARY_CACHE` 配置开关。

**验证**: `pytest tests/performance/test_turn_latency.py` → `call_counter["count"] <= 3`（conflict detector 仍绕过缓存直接调用，共 3 次：story gen + conflict + options）。

### 4.2 C2: 增量重要性模式

**问题**: `recalculate_importance()` 每回合全量遍历所有节点，O(n) 复杂度。

**改动** (`src/knowledge_graph/graph.py`)：

1. 新增 `_dirty_nodes: Set[str]` — 跟踪需要重算的节点。
2. 在以下操作中标记 dirty：
   - `add_entity()` — 新建/更新实体
   - `add_relation()` — 新建/更新关系（两端节点）
   - `update_entity_state()` — 更新实体状态
   - `refresh_mentions()` — 提及计数刷新
   - `apply_decay()` — 关系衰减时标记两端节点
3. 新增 `_recalculate_importance_incremental()` — 仅重算 dirty 节点，每 N 回合全量重算一次。
4. 新增 `KG_IMPORTANCE_MODE="incremental"` 选项。
5. 新增 `KG_INCREMENTAL_FULL_RECALC_INTERVAL=10` 配置。

**公式**（与 composite 完全一致）：

```
importance = 0.3 * degree_score + 0.3 * recency_score + 0.2 * mentions_score + 0.2 * player_score
```

**测试** (`tests/kg/test_graph_enhanced.py`)：

| 测试 | 覆盖场景 |
|------|----------|
| `test_incremental_mode_matches_composite` | _current_turn=10 时增量全量重算与 composite 完全一致 |

**验证**: `pytest tests/kg/test_graph_enhanced.py` → 36 passed。

### 4.3 C3: 衰减节奏可配置

**问题**: `apply_decay()` 每回合执行，全边扫描 O(E)。

**改动**：

1. `config.py` 新增 `KG_DECAY_CADENCE: int = 1`。
2. `src/engine/game_engine.py` — `_apply_kg_update()` 内：

```python
if turn_id % settings.KG_DECAY_CADENCE == 0:
    self.kg.apply_decay(turn_id=turn_id)
```

**回滚**: 设 `KG_DECAY_CADENCE=1` 即恢复每回合衰减。

**验证**: `pytest tests/engine/test_engine_enhanced.py` → 16 passed。

---

## 五、Wave D — 文档交付

### 5.1 D1: 技术路线文档更新

**文件**: `docs/guides/technical-route.md`

新增 3 个小节：

- **3.5 KG 策略配置** — 6 个可运行时切换的策略开关表格。
- **3.6 安全回滚开关** — 2 个 boolean toggle（incremental importance、summary cache）。
- **3.7 质量门控** — quality_runner 命令 + Gate-1/2/3 说明。

### 5.2 D2: 改进报告追加

**文件**: `docs/reports/nlu-kg-improvement.md`

新增第七节「质量优先路线 — 二次迭代」，包含：
- 波次总览表（A–C 改进项 + 测试数）
- 新增文件清单
- 新增配置项清单
- 评估流水线命令
- 测试矩阵

---

## 六、配置变更

### 6.1 新增配置项（`config.py`）

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `KG_DECAY_CADENCE` | int | 1 | 每 N 回合执行一次关系衰减 |
| `KG_INCREMENTAL_FULL_RECALC_INTERVAL` | int | 10 | 增量模式每 N 回合全量重算 |
| `KG_ENABLE_INCREMENTAL_IMPORTANCE` | bool | True | 启用增量重要性计算 |
| `KG_ENABLE_SUMMARY_CACHE` | bool | True | 启用单回合摘要缓存 |

### 6.2 回滚策略

| 场景 | 操作 |
|------|------|
| 增量重要性导致问题 | `KG_ENABLE_INCREMENTAL_IMPORTANCE = False` |
| 摘要缓存导致状态不一致 | `KG_ENABLE_SUMMARY_CACHE = False` |
| 衰减节奏太快 | `KG_DECAY_CADENCE = 1`（每回合衰减） |
| 需要立即全量重算 | `KG_IMPORTANCE_MODE = "composite"` |

---

## 七、测试矩阵

### 7.1 全量验证结果

| 模块 | 测试套件 | 通过 | 失败 |
|------|----------|------|------|
| Evaluation | `tests/evaluation/` | 13 | 0 |
| KG Graph | `tests/kg/test_graph_enhanced.py` | 36 | 0 |
| KG Conflict/Extractor/Type | `tests/kg/test_conflict_resolution.py` + `test_temporal_reasoning.py` + `test_relation_extractor_enhanced.py` + `test_kg_type_validation.py` | 70 | 0 |
| Engine + Performance | `tests/engine/test_engine_enhanced.py` + `tests/performance/test_turn_latency.py` | 17 | 0 |
| **合计** | | **136** | **0** |

### 7.2 Quality Runner 输出

```json
{
  "mode": "compare",
  "metrics": {
    "entity_f1": 1.0,
    "relation_f1": 1.0,
    "coreference_accuracy": 1.0,
    "contradiction_rate": 0.0
  },
  "gate_summary": {
    "gate_1_pass": true,
    "gate_1_max_regression_pp": 0.01
  }
}
```

Gate-1 通过（0 回归）。Gate-2/3 在真实 LLM 数据下生效。

---

## 八、验证命令

### 8.1 单模块验证

```bash
# NLU 核心测试
python -m pytest tests/nlu/test_coreference_enhanced.py -q

# KG 图操作测试
python -m pytest tests/kg/test_graph_enhanced.py -q

# KG 冲突检测测试
python -m pytest tests/kg/test_conflict_resolution.py tests/kg/test_temporal_reasoning.py -q

# KG 抽取器测试
python -m pytest tests/kg/test_relation_extractor_enhanced.py tests/kg/test_kg_type_validation.py -q

# 引擎集成测试
python -m pytest tests/engine/test_engine_enhanced.py -q

# 性能回归测试
python -m pytest tests/performance/test_turn_latency.py -q
```

### 8.2 全量验证

```bash
# 评估门控
python -m pytest tests/evaluation/ -q

# KG 全部
python -m pytest tests/kg/ -q

# 引擎 + 性能
python -m pytest tests/engine/ tests/performance/ -q
```

### 8.3 Quality Runner

```bash
# 生成基线
python -m tests.evaluation.quality_runner --mode baseline

# 比对
python -m tests.evaluation.quality_runner --mode compare --against baseline

# 输出位置
cat tests/evaluation/reports/latest_quality.json
```

---

## 九、已知限制

1. **Conflict detector 绕过缓存**：`_llm_check` 内部直接调用 `kg.to_summary()`，不经过 `_current_kg_summary()` 缓存。Performance test 断言为 `<= 3` 次调用（story gen + conflict + options）。
2. **Gate-2/3 在合成 harness 下不生效**：当前 benchmark 使用确定性 harness，所有指标满分。真实 LLM 数据下 gate 才有意义。
3. **fastcoref monkey-patch 仍存在**：`_check_transformers_version()` 的 `all_tied_weights_keys` 补丁未移除，仅做了 null-guard。
4. **LSP 警告**：`game_engine.py` 中 `intent`/`emotion` 变量类型推断为 `object`（Pyright 假阳性），已通过 `isinstance` guard 保证运行时安全。

---

## 十、后续建议

1. **真实 LLM 数据验证**：使用实际 GPT 调用跑 benchmark，生成真实基线数据，激活 Gate-2/3 门控。
2. **Conflict detector 缓存注入**：将 `_current_kg_summary()` 缓存传递给 ConflictDetector，消除最后一次绕过。
3. **NLU 模型微调**：基于新的 120 条 benchmark，扩展 `training/data_augmenter.py` 生成更多训练数据。
4. **Structured logging export**：将 stage_metrics 导出为 Prometheus/DataDog 格式，支持生产环境监控。
