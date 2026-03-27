# 使用指南

本目录包含 StoryWeaver 项目的使用指南和部署文档。

## 文件说明

### 🚀 快速部署（推荐新手）
- **[zero-to-hero-deployment.md](zero-to-hero-deployment.md)** - ⭐ 从零开始完整部署指南，包含 llama.cpp 本地模型推理配置，兼容 Windows/macOS
- **[local-model-startup.md](local-model-startup.md)** - llama.cpp 本地服务器快速启动指南

### 技术文档
- **technical-route.md** - 项目技术路线图，包含 NLU/NLG/KG 架构设计
- **data-flow.md** - 模块间数据传递详细说明（字段级）

### 部署指南（按系统）
- **deployment-windows.md** - Windows 系统高可用部署指南
- **deployment-macos.md** - macOS 系统高可用部署指南

### 推理配置（备选方案）
- **../VLLM_INTEGRATION.md** - vLLM 本地 GPU 推理集成指南（需要 NVIDIA GPU）
- **../CPU_INFERENCE.md** - 旧版 CPU 推理优化指南（使用 vLLM）

## 快速开始

1. **首次部署** → 阅读 [从零部署指南](zero-to-hero-deployment.md)
2. **已部署用户** → 使用 [本地模型启动指南](local-model-startup.md) 快速启动
3. **新开发者** → 阅读 [技术路线图](technical-route.md) 了解整体架构
4. **理解数据流** → 查看 [数据流文档](data-flow.md) 了解模块交互

## 部署模式

项目支持两种运行模式：
- **开发模式** - 快速迭代，使用 `start_project.sh` 或 `start_project.bat`
- **生产模式** - 高可用部署，使用 `start_project_prod.sh` 或 `start_project_prod.bat`

## 推理后端选择

| 后端 | 硬件要求 | 推荐场景 |
|------|---------|---------|
| **llama.cpp (本地)** | CPU 即可 | ⭐ 推荐，无需 GPU，适合演示 |
| **vLLM (本地)** | NVIDIA GPU | 高性能部署，需要 GPU |
| **远程 API** | 无 | 快速测试，质量最高 |