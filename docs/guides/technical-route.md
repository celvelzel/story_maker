# StoryWeaver 技术路线（中文）

> **Last Updated**: 2026-04-01

## 1. 总体目标

本项目面向多轮互动叙事，采用 NLU（本地）+ NLG（API / 本地混合）+ KG（世界状态）混合架构，实现以下目标：

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

### 3.5 KG 增强特性（2026-03 更新）

- **重要性评分**：复合评分（度数中心性 + 提及频率 + 玩家交互权重）
- **时间线追踪**：维护最近 5 条关键事件时间线
- **关系衰减**：每回合关系强度衰减 10%，防止图谱无限膨胀
- **持久化**：每 5 回合自动保存快照至 `saves/` 目录

## 4. NLG 设计

### 4.1 目标

1. 根据当前意图、历史与世界状态生成叙事。
2. 输出可行动选项并维持玩家代理感。

### 4.2 输入输出

1. 输入：`player_input`、`intent`、`kg_summary`、`history`
2. 输出：
   - `story_text`
   - `options`（text / intent_hint / risk_level）

### 4.3 后端模式

项目支持三种 NLG 后端（通过 `config.py` 的 `NLG_MODE` 控制）：

| 模式 | 说明 |
|------|------|
| `api` | 远程 OpenAI 兼容 API（Mimo API） |
| `local` | 本地 llama.cpp 服务器（Qwen3-4B GGUF） |
| `hybrid` | 混合模式：创意任务用本地 Qwen3，结构化任务用远程 API（默认） |

### 4.4 核心算法

1. Prompt 模板化：将意图与世界摘要注入上下文。
2. 选项生成：约束返回结构，便于 UI 直接渲染。

### 4.5 降级策略

1. API 抖动：客户端重试。
2. 结构化输出异常：使用保底默认选项或上轮选项。
3. 本地模型不可用：自动回退到远程 API（hybrid 模式）。

## 5. 兼容与约束

1. Intent 接口保持不变：`predict() -> {"intent", "confidence"}`。
2. fallback 路径保留，不移除规则能力。
3. 游戏主流水线不改输入输出契约，保证 UI 与测试兼容。

## 6. 配置总览

核心配置项见 `config.py`（Pydantic Settings），支持 `.env` 文件覆盖：

| 类别 | 关键字段 | 默认值 |
|------|---------|--------|
| LLM | `NLG_MODE` | `"hybrid"` |
| LLM | `OPENAI_BASE_URL` | 远程 API URL |
| LLM | `OPENAI_MODEL` | `"mimo-v2-flash"` |
| LLM | `OPENAI_TIMEOUT_READ` | `60.0` |
| NLU | `INTENT_MODEL_PATH` | `models/intent_classifier` |
| NLU | `SPACY_MODEL` | `"en_core_web_sm"` |
| KG | `KG_MAX_NODES` | `200` |
| KG | `KG_SUMMARY_MODE` | `"layered"` |
| Game | `NARRATIVE_HISTORY_WINDOW` | `6` |
