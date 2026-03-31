# StoryWeaver 文档中心

本目录包含 StoryWeaver 项目的所有技术文档，按类别组织如下：

## 目录结构

- **api/** - API 接口文档
- **design/** - 设计文档与规范
- **fixes/** - 问题修复报告与解决方案
- **guides/** - 使用指南与部署文档
- **reports/** - 优化与改进报告
- **project/** - 项目规范与计划

## 文档分类

### API 文档 (`api/`)
包含前端接口规范、数据模型和模块 API 说明。

### 设计文档 (`design/`)
包含系统架构设计、算法策略和微调方案等技术设计文档。

### 修复报告 (`fixes/`)
记录各类问题的修复过程、根因分析和解决方案。

### 使用指南 (`guides/`)
提供部署指南、技术路线和数据流说明等实用文档。

### 优化报告 (`reports/`)
包含系统优化、性能改进和功能增强的详细报告。

### 项目文档 (`project/`)
包含项目规范、实现计划和提示词模板。

## ⭐ 快速导航

### 🚀 首次部署（强烈推荐）
- **[从零部署指南](guides/zero-to-hero-deployment.md)** - 完整的多平台部署指南，包含 llama.cpp 本地模型推理
- **[本地模型启动指南](guides/local-model-startup.md)** - llama.cpp 服务器快速启动

### 基础文档
- [技术路线图](guides/technical-route.md)
- [API 参考文档](api/API_REFERENCE.md)

### 按系统部署
- [部署指南 (Windows)](guides/deployment-windows.md)
- [部署指南 (macOS)](guides/deployment-macos.md)

### 备选推理方案
- [vLLM 集成指南](../VLLM_INTEGRATION.md) - 本地 GPU 推理
- [CPU 推理指南](../CPU_INFERENCE.md) - 旧版 CPU 推理

### 🛠️ 自动化与测试
- **[自动化测试报告](reports/automated_test_report.md)** - 全模块自动化测试结果
- **[知识图谱开关测试报告](kg_on_off_report.md)** - KG 对生成质量影响的对比测试

### 🔧 故障排除与修复
- [DistilBERT 分词器修复](fixes/distilbert-tokenizer-fix.md)
- [LLM JSON 截断修复](fixes/llm-json-truncation-fix.md)
- [Fastcoref 内存优化](fixes/fastcoref-fix.md)
- [DistilBERT 兼容性修复](fixes/distilbert-compatibility-fix.md)
- [DistilBERT 故障排查指南](fixes/distilbert-troubleshooting.md)

### 📈 知识图谱相关
- [知识图谱优化报告](reports/kg-optimization.md)
- [NLU & KG 改进报告](reports/nlu-kg-improvement.md)
- [运行时持久化文档](reports/runtime-persistence.md)
- [实体重要性评分设计](design/entity-importance.md)

### 🤖 本地模型集成
- [本地模型推理集成报告](reports/本地模型推理集成_2026-03-27_21-07.md)
- [本地模型调优报告](reports/local-model-tuning_2026-03-27.md)
- [NLG 本地模型微调方案](design/nlg-local-model-finetuning.md)

### 📝 提示词工程
- [提示词重建计划](project/rebuilt_prompt.md)
- [智能体提示词设计](project/agent_prompt.md)
- [开场白提示词模板](design/prompts/story_opening.md)
- [剧情延续提示词模板](design/prompts/story_continuation.md)
- [选项生成提示词模板](design/prompts/option_generation.md)