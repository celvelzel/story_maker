# 设计文档

本目录包含 StoryWeaver 项目的技术设计文档与规范。

## 文档索引

### 核心架构
- **[entity-importance.md](entity-importance.md)** — 知识图谱实体重要性评分策略，用于图谱剪枝。支持 `composite`（复合）、`incremental`（增量）和 `degree_only`（仅度数）三种模式。
- **[nlg-local-model-finetuning.md](nlg-local-model-finetuning.md)** — NLG 模块本地 LLM 集成与微调部署方案。
- **[hybrid-nlg-architecture.md](hybrid-nlg-architecture.md)** — 混合 NLG 路由架构：本地 Qwen3 + Mimo API 的基于任务路由。
- **[conflict-detection-resolution.md](conflict-detection-resolution.md)** — 多层冲突检测（规则 + 时序 + LLM）与解决策略。
- **[sentiment-analysis.md](sentiment-analysis.md)** — 玩家输入情感/情绪分析设计（distilroberta + 关键词兜底）。
- **[kg-summary-modes.md](kg-summary-modes.md)** — KG 摘要生成模式：扁平模式 vs 分层重要性排序摘要。

### 提示词模板 (`prompts/`)
- **[story_opening.md](prompts/story_opening.md)** — 生成新故事开场白的系统提示词。
- **[story_continuation.md](prompts/story_continuation.md)** — 基于玩家输入继续叙事的系统提示词。
- **[option_generation.md](prompts/option_generation.md)** — 生成分支玩家选择的提示词。

## 设计原则

1. **模块化架构** — NLU、NLG 和 KG 模块相互独立，通过定义良好的接口通信。
2. **优雅降级** — 所有模块均支持回退机制（如模型失效时使用基于关键词的 NLU），确保服务连续性。
3. **可扩展性** — 支持通过配置热切换模型和动态调整策略。
4. **世界一致性** — 通过知识图谱和冲突检测进行状态管理，维护叙事完整性。
5. **混合智能** — 本地模型处理创意任务，云端 API 处理结构化任务，优化成本与质量。
