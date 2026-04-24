# StoryWeaver：AI 文本冒险引擎（NLU + LLM + 知识图谱）

> **最后更新**: 2026-04-24

**[English](README.md) | [中文](README_zh.md)**

StoryWeaver 是一个基于 Streamlit 的交互式文本冒险引擎，融合本地 NLU、LLM 生成与动态知识图谱，用于在多回合叙事中保持世界状态一致。

## 项目亮点

- 支持混合 NLG 路由（`NLG_MODE=local|api|hybrid`），可平衡质量、成本与延迟。
- 支持运行时会话持久化（`runtime_session.json`）与活跃状态恢复逻辑。
- 提供知识图谱关系抽取、冲突检测与分层摘要模式。
- 提供评估工具链：自动化指标、KG 开关基准、LLM-as-Judge、三配置对比文档。

## 快速开始

### 1）创建环境并安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
```

### 2）配置 `.env`

最少需要：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=mimo-v2-flash
```

当前代码中常用开关：

- `NLG_MODE`（默认 `hybrid`）
- `MIMO_API_KEY`、`MIMO_BASE_URL`、`MIMO_MODEL`
- `EVAL_LLM_API_KEY`、`EVAL_LLM_BASE_URL`、`EVAL_LLM_MODEL`
- `KG_SUMMARY_MODE`、`KG_EXTRACTION_MODE`、`KG_IMPORTANCE_MODE`

### 3）启动项目

- Windows：`scripts/start/start_project_prod.bat`
- macOS/Linux：`scripts/start/start_project_prod.sh`

根目录下的 `start_project_prod.bat/.sh` 为便捷副本。

## 每回合处理流程

1. 指代消解（可用 fastcoref 则启用，否则规则回退）
2. 意图分类（DistilBERT + 回退）
3. 实体抽取（spaCy + 启发式）
4. 剧情续写生成
5. 关系抽取并更新知识图谱
6. 冲突检测与状态协调
7. 生成 3 个玩家选项（含风险标签）

## 目录结构

```text
story_maker/
├── app.py
├── config.py
├── src/
│   ├── engine/              # 游戏主循环、runtime_session、状态管理
│   ├── nlu/                 # 意图/实体/指代/情感
│   ├── nlg/                 # 剧情/选项生成与提示词
│   ├── knowledge_graph/     # 图谱、关系抽取、冲突检测、可视化
│   ├── evaluation/          # 自动指标与 LLM 评审
│   ├── ui/                  # Streamlit 分区与状态管理
│   └── utils/
├── scripts/
│   ├── start/               # 启动与部署脚本
│   ├── eval/                # 评估运行脚本
│   ├── inference/           # 本地推理辅助脚本
│   ├── data/                # 数据集处理脚本
│   └── quantize/
├── training/
├── tests/
├── docs/
├── reports/                 # 独立对比/评估/混合策略报告
├── models/
├── config/
└── lib/
```

## 测试与评估

```bash
# 单元/集成测试
pytest tests/ -v

# 自动评估
python scripts/eval/run_automated_eval.py

# KG 开关对比
python scripts/eval/run_kg_on_off_benchmark.py
```

## 文档导航

- 文档总入口：[docs/README.md](docs/README.md)
- 指南索引：[docs/guides/README.md](docs/guides/README.md)
- API 参考：[docs/api/API_REFERENCE.md](docs/api/API_REFERENCE.md)
- 设计文档：[docs/design/README.md](docs/design/README.md)
- 修复与排障：[docs/fixes/README.md](docs/fixes/README.md)
- 报告索引：[docs/reports/README.md](docs/reports/README.md)

## Git 近期新增能力（摘要）

- 三配置评估文档（local / api / hybrid）已补齐。
- 运行时恢复与活跃会话生命周期处理已增强。
- 混合 NLG 路由与评估模型解耦已落地。
- 文档/报告目录整理与部署脚本一致性持续修正。

## 技术栈

- **前端**：Streamlit
- **NLU**：DistilBERT、spaCy、fastcoref（含回退）
- **NLG**：OpenAI 兼容 API + 本地模型路径
- **知识图谱**：NetworkX + PyVis
- **评估**：Distinct-n、Self-BLEU、LLM-as-Judge
