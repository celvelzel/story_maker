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
- **[本地模型启动指南](guides/local-model-startup.md)** - 本地推理服务快速启动（含 KoboldCpp Vulkan + llama.cpp 两种方案）

### 基础文档
- [技术路线图](guides/technical-route.md)
- [API 参考文档](api/API_REFERENCE.md)

### 按系统部署
- [部署指南 (Windows)](guides/deployment-windows.md)
- [部署指南 (macOS)](guides/deployment-macos.md)

### 备选推理方案
- [vLLM 集成指南](../VLLM_INTEGRATION.md) - 本地 GPU 推理
- [CPU 推理指南](../CPU_INFERENCE.md) - 旧版 CPU 推理

### 知识图谱相关
- [知识图谱优化报告](reports/kg-optimization.md)
- [NLU & KG 改进报告](reports/nlu-kg-improvement.md)
- [运行时持久化文档](reports/runtime-persistence.md)

### 本地模型集成
- [本地模型推理集成报告](reports/本地模型推理集成_2026-03-27_21-07.md)
- [本地模型调优报告](reports/local-model-tuning_2026-03-27.md)