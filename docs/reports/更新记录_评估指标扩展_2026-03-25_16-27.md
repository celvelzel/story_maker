# 更新记录 - 评估指标扩展

**日期：** 2026年03月25日 16:27  
**核心目标：** 建立多维度自动化评估体系，涵盖语言质量、知识图谱演化与 LLM 主观评分。

---

## 1. 变更摘要

本次更新对 StoryWeaver 的评估体系进行了深度扩展。新增了 3 个 LLM 评判维度（总计 8 维）、3 个轻量级 reference-free 自动指标，以及知识图谱（KG）密度演化指标。同时，引擎层补齐了每回合 KG 规模快照能力，为长线叙事的连贯性分析提供了数据基础。

## 2. 修改详情

### 2.1 关键文件变更

| 模块 | 文件路径 | 变更说明 |
|---------|---------|------|
| **引擎** | `src/engine/state.py` | `GameState` 新增 `kg_turn_stats` 记录每回合节点/边数量快照 |
| **引擎** | `src/engine/game_engine.py` | 实现快照自动触发、`TurnResult` 指标透传及持久化支持 |
| **评估** | `src/evaluation/metrics.py` | 新增 `lexical_overlap`、`type_token_ratio`、`flesch_reading_ease` 等指标 |
| **评判** | `src/evaluation/llm_judge.py` | 维度从 5 扩展至 8，增强了对异常 JSON 响应的解析鲁棒性 |
| **测试** | `tests/evaluation/*.py` | 新增针对扩展指标、图密度边界值及 LLM 评判解析的专项测试 |

### 2.2 功能增强

- **LLM 评判维度扩展**:
  - `option_relevance`: 选项与当前剧情的相关性。
  - `causal_link`: 剧情发展的因果链条合理性。
  - `local_coherence`: 相邻回合间的叙事连贯性。
- **自动指标增强**:
  - `lexical_overlap`: 相邻文本词汇交集，衡量局部承接。
  - `type_token_ratio`: 词汇丰富度指标。
  - `flesch_reading_ease`: 文本可读性评分。
- **知识图谱演化分析**:
  - 引入 `graph_density_evolution`，支持输出图密度的起始、均值、增量及完整演化序列。

## 3. 技术规范

- **异常处理**: 针对空文本、空 Token 或极简图（节点数 < 2）实现了严格的边界处理，确保评估过程不崩溃。
- **解析鲁棒性**: LLM Judge 支持从带有噪声文字的响应中提取 JSON 内容，并对分值进行 1-10 范围的归一化处理。
- **性能保持**: 评估逻辑保持轻量化设计，未引入额外的重量级 NLP 模型依赖。

## 4. 测试与验证

- **专项测试运行**:
  `pytest tests/engine/test_engine_enhanced.py tests/evaluation/test_metrics_extended.py tests/evaluation/test_llm_judge.py`
  - **结果**: 26 passed, 0 failed.
- **全量回归测试**:
  `python -m pytest -q`
  - **结果**: 286 passed, 0 failed.

## 5. 结论

评估体系的补齐为项目提供了“量化基准”，使得后续对 NLU 或 NLG 模型的任何微调都能得到即时的质量反馈，显著提升了开发迭代的科学性。
