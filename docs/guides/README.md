# 使用指南

本目录包含 StoryWeaver 项目的使用指南和部署文档。

## 文件说明

### 技术文档
- **technical-route.md** - 项目技术路线图，包含 NLU/NLG/KG 架构设计
- **data-flow.md** - 模块间数据传递详细说明（字段级）

### 部署指南
- **deployment-macos.md** - macOS 系统高可用部署指南
- **deployment-windows.md** - Windows 系统高可用部署指南

## 快速开始

1. **新开发者** - 阅读 [技术路线图](technical-route.md) 了解整体架构
2. **部署项目** - 根据系统选择对应的部署指南
3. **理解数据流** - 查看 [数据流文档](data-flow.md) 了解模块交互

## 部署模式

项目支持两种运行模式：
- **开发模式** - 快速迭代，使用 `start_project.sh` 或 `start_project.bat`
- **生产模式** - 高可用部署，使用 `start_project_prod.sh` 或 `start_project_prod.bat`