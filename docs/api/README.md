# API 文档

本目录包含 StoryWeaver 项目的 API 文档。

## 文件列表

- **[API_REFERENCE.md](API_REFERENCE.md)** - 完整 API 参考文档 (v1.2.0)，包含：
  - 前端交互模型
  - 数据结构定义 (TurnResult、StoryOption、GameState、Emotion Result)
  - 后端编排器 (GameEngine，含惰性 NLU 加载、存档/读档)
  - NLU 模块 API (IntentClassifier、EntityExtractor、CoreferenceResolver、SentimentAnalyzer)
  - NLG 模块 API (StoryGenerator、OptionGenerator、混合 NLG 路由)
  - 知识图谱 API (KnowledgeGraph、RelationExtractor、ConflictDetector)
  - 评估 API (8 项自动指标 + 8 维度 LLM 评审)
  - 完整配置参考 (所有 Pydantic Settings 参数)
  - 错误处理与使用示例

## 使用说明

本文档是前后端集成与内部模块通信的主要参考文档，会随代码库状态保持同步更新。
