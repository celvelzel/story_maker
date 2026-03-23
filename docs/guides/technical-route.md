# StoryWeaver 技术路线（中文）

## 1. 总体目标

本项目面向多轮互动叙事，采用 NLU（本地）+ NLG（API）+ KG（世界状态）混合架构，实现以下目标：

1. 玩家输入可解释：输出意图、实体、共指消解结果。
2. 剧情延续可控：结合历史与知识图谱生成后续叙事。
3. 世界一致性可追踪：每轮更新 KG 并检测冲突。
4. 退化可用：模型不可用时自动 fallback，保证服务不中断。

## 2. NLU 设计

### 2.1 模块组成

1. CoreferenceResolver（fastcoref + 规则兜底）
2. IntentClassifier（DistilBERT + 关键词 rule_fallback）
3. EntityExtractor（spaCy NER + 名词短语/正则兜底）

### 2.2 输入输出

1. 输入：玩家原始文本 + 最近历史文本
2. 输出：
   - `resolved_input`
   - `intent`
   - `confidence`
   - `entities`
   - `intent_backend`
   - `intent_model_loaded`

### 2.3 核心算法

1. 共指：优先神经模型，失败转规则替换最近实体名。
2. 意图：优先本地 checkpoint（`models/intent_classifier`），失败转关键词匹配。
3. 实体：优先 spaCy，缺失时退化到启发式词表 + regex。

### 2.4 降级策略

1. `fastcoref` 不可用：coref 规则模式。
2. intent 模型目录不存在或加载失败：`rule_fallback`。
3. spaCy 模型不可用：实体抽取 regex 模式。

## 3. KG 设计

### 3.1 目标

1. 维护实体和关系的结构化世界状态。
2. 支持可视化与冲突检测。

### 3.2 输入输出

1. 输入：故事文本、玩家实体、关系抽取结果
2. 输出：
   - `kg_summary`（供 NLG 继续生成）
   - `kg_html`（前端可视化）
   - `conflicts`（冲突列表）

### 3.3 核心算法

1. 关系抽取：LLM 结构化输出 entities/relations。
2. 图更新：NetworkX MultiDiGraph 增量写入。
3. 冲突检测：规则 + LLM 校验双通道。

### 3.4 降级策略

1. LLM 抽取异常：跳过当轮 KG 增量并记录 warning。
2. 可视化异常：不影响主流程，返回空图或上次图。

### 3.5 KG 策略配置（`config.py`）

可运行时切换的策略开关：

| 配置项 | 可选值 | 默认值 | 说明 |
|--------|--------|--------|------|
| `KG_CONFLICT_RESOLUTION` | `keep_latest` \| `llm_arbitrate` | `llm_arbitrate` | 冲突解决策略 |
| `KG_EXTRACTION_MODE` | `story_only` \| `dual_extract` | `dual_extract` | 关系抽取模式 |
| `KG_IMPORTANCE_MODE` | `degree_only` \| `composite` \| `incremental` | `composite` | 重要性计算模式 |
| `KG_SUMMARY_MODE` | `flat` \| `layered` | `layered` | KG 摘要输出模式 |
| `KG_DECAY_CADENCE` | 整数 ≥1 | `1` | 每 N 回合执行一次关系衰减 |
| `KG_INCREMENTAL_FULL_RECALC_INTERVAL` | 整数 ≥1 | `10` | 增量模式全量重算间隔 |

### 3.6 安全回滚开关

| 配置项 | 默认值 | 回滚说明 |
|--------|--------|----------|
| `KG_ENABLE_INCREMENTAL_IMPORTANCE` | `True` | 设为 `False` 强制 composite 全量重算 |
| `KG_ENABLE_SUMMARY_CACHE` | `True` | 设为 `False` 禁用单回合摘要缓存 |

### 3.7 质量门控（Wave A 基础设施）

自动化评估脚本：`tests/evaluation/quality_runner.py`

```bash
# 生成基线
python -m tests.evaluation.quality_runner --mode baseline

# 比对当前代码
python -m tests.evaluation.quality_runner --mode compare --against baseline
```

门控策略：
- Gate-1：已追踪指标无回归 >1pp
- Gate-2：relation_f1 +3pp、coref_accuracy +5pp、contradiction_rate ≤ 基线 80%
- Gate-3：连续两次通过才算达标

## 4. NLG 设计

### 4.1 目标

1. 根据当前意图、历史与世界状态生成叙事。
2. 输出可行动选项并维持玩家代理感。

### 4.2 输入输出

1. 输入：`player_input`、`intent`、`kg_summary`、`history`
2. 输出：
   - `story_text`
   - `options`（text / intent_hint / risk_level）

### 4.3 核心算法

1. Prompt 模板化：将意图与世界摘要注入上下文。
2. 选项生成：约束返回结构，便于 UI 直接渲染。

### 4.4 降级策略

1. API 抖动：客户端重试。
2. 结构化输出异常：使用保底默认选项或上轮选项。

## 5. 兼容与约束

1. Intent 接口保持不变：`predict() -> {"intent", "confidence"}`。
2. fallback 路径保留，不移除规则能力。
3. 游戏主流水线不改输入输出契约，保证 UI 与测试兼容。
