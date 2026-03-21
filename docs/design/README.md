# 设计文档

本目录包含 StoryWeaver 项目的技术设计文档和规范。

## 文件说明

### 核心设计
- **entity-importance.md** - 实体重要性评分策略，包含复合模式和度数模式两种算法
- **nlg-local-model-finetuning.md** - NLG 模块本地大模型微调与部署方案

### 提示词模板 (`prompts/`)
- **story_opening.md** - 故事开场生成提示词
- **story_continuation.md** - 故事延续生成提示词
- **option_generation.md** - 选项生成提示词

## 设计原则

1. **模块化设计** - NLU、NLG、KG 三大模块独立开发，通过标准接口交互
2. **降级可用** - 所有模块支持 fallback 机制，确保服务不中断
3. **可扩展性** - 支持模型热切换和策略动态调整
4. **一致性维护** - 通过知识图谱和冲突检测保证故事世界状态一致