# StoryWeaver：动态情节生成的 AI 驱动文本冒险游戏

**[English](README.md) | [中文](README_zh.md)**

## COMP5423 自然语言处理小组项目

一个交互式文本冒险游戏引擎，结合了**本地 NLU 模型**、**LLM 驱动的故事生成**和**动态知识图**，以实现叙事一致性。

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit 前端界面                            │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │ 聊天界面 │  │ NLU 调试     │  │ 知识图谱可视化          ││
│  └──────────┘  └──────────────┘  └────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    游戏引擎（主控制器）                           │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ NLU (本地) │→ │  游戏状 │→ │ 故事生成     │→ │ 选项生  │ │
│  │ DistilBERT │  │  态管理 │  │ (OpenAI API) │  │ 成(API) │ │
│  │ spaCy +    │  │         │  │              │  │         │ │
│  │ fastcoref  │  └────┬────┘  └──────────────┘  └─────────┘ │
│  └─────────────┘       │                                       │
│              ┌─────────▼───────┐                                │
│              │ 知识图谱        │← 关系提取 (API)               │
│              │ + 冲突检测      │                                 │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### 处理流程（每个回合）

1. **指代消解** — fastcoref 使用最近历史解决代词
2. **意图分类** — DistilBERT 微调分类器（带关键字回退）
3. **实体提取** — spaCy NER + 名词短语启发式方法
4. **故事生成** — LLM API 继续叙述
5. **知识图更新** — LLM 将实体和关系提取到 NetworkX 图中
6. **冲突检测** — 基于规则 + LLM 一致性检查
7. **选项生成** — LLM 生成 3 个玩家选择及其风险等级

### 项目结构

```
story_maker/
├── app.py                    # Streamlit 应用程序入口
├── config.py                 # Pydantic 设置支持 .env
├── requirements.txt          # 依赖项
├── config/
│   └── .env.example          # 环境配置模板
├── src/                      # 源代码模块
│   ├── engine/               # 游戏引擎控制器
│   │   ├── game_engine.py    # 主管道协调器 (NLU → NLG → KG)
│   │   ├── runtime_session.py # 会话持久化管理
│   │   └── state.py          # 游戏状态和历史追踪
│   ├── nlu/                  # 自然语言理解
│   │   ├── intent_classifier.py   # DistilBERT + 关键字回退
│   │   ├── entity_extractor.py    # spaCy NER + 正则表达式
│   │   ├── coreference.py         # fastcoref + 规则消解
│   │   └── sentiment_analyzer.py  # 情感/语气分析 (Ekman 6 类)
│   ├── nlg/                  # 自然语言生成
│   │   ├── story_generator.py     # OpenAI API 故事生成
│   │   ├── option_generator.py    # 玩家选择生成 (API)
│   │   └── prompt_templates.py    # 提示工程模板
│   ├── knowledge_graph/      # 动态世界状态管理
│   │   ├── graph.py               # NetworkX MultiDiGraph 包装器
│   │   ├── relation_extractor.py  # LLM 基础关系提取
│   │   ├── conflict_detector.py   # 规则 + LLM 一致性检查
│   │   └── visualizer.py          # PyVis HTML 可视化
│   ├── evaluation/           # 质量评估指标
│   │   ├── metrics.py             # Distinct-n, Self-BLEU, 覆盖率
│   │   ├── llm_judge.py           # LLM 作为评判者评分
│   │   └── consistency_eval.py    # 知识图一致性评估
│   └── utils/                # 共享工具
│       └── api_client.py          # 单例 LLM 客户端（含重试）
├── data/                     # 数据资产和处理
│   ├── intent_labels.json         # 意图标签定义
│   ├── raw/                       # 原始数据集 (git-ignored)
│   └── scripts/                   # 数据处理脚本
│       ├── download_data.py       # 数据集下载自动化
│       └── preprocess.py          # 数据预处理管道
├── training/                 # 模型训练脚本
│   ├── train_intent.py            # DistilBERT 意图分类器训练
│   ├── train_generator.py         # GPT-2 LoRA 微调 (遗留/可选)
│   └── data_augmenter.py          # 训练数据增强
├── tests/                    # 测试套件 (按模块组织)
│   ├── engine/              # 引擎组件测试
│   ├── nlu/                 # NLU 模块测试
│   ├── nlg/                 # NLG 模块测试
│   ├── kg/                  # 知识图谱测试
│   ├── integration/         # 跨模块集成测试
│   └── training/            # 训练管道测试
├── scripts/                  # 工具和部署脚本
│   ├── start_project_prod.bat     # Windows 生产启动器
│   ├── start_project_prod.sh      # macOS/Linux 生产启动器
│   ├── health_check.py            # 部署前健康检查
│   └── generate_dataset.py        # 数据集生成工具
├── docs/                     # 全面文档
│   ├── api/                 # API 参考文档
│   ├── design/              # 架构和设计文档
│   ├── guides/              # 部署和使用指南
│   ├── fixes/               # 问题修复报告
│   ├── reports/             # 优化和改进报告
│   └── project/             # 项目规范和计划
├── models/                   # 训练模型文件 (git-ignored)
│   └── intent_classifier/   # 微调 DistilBERT 检查点
├── lib/                      # 第三方前端库
│   ├── vis-9.1.2/           # Vis.js 网络可视化
│   ├── tom-select/          # 增强选择组件
│   └── bindings/            # JavaScript 工具绑定
├── logs/                     # 应用日志 (git-ignored)
├── saves/                    # 游戏存档文件 (git-ignored)
└── .env                      # 环境变量 (git-ignored)
```

### 部署与启动 (Windows / macOS)

使用与你的操作系统匹配的启动脚本进行部署：

- Windows: `scripts/start_project_prod.bat`
- macOS/Linux: `scripts/start_project_prod.sh`

生产启动脚本提供以下功能：

1. **端口占用检测和进程识别** — 自动检测现有 Streamlit 进程
2. **安全重启策略** — 智能处理现有 Streamlit 应用进程
3. **依赖项安装** — 具有显式网络超时控制的依赖安装
4. **启动失败处理** — 结构化退出代码便于诊断
5. **日志记录** — `logs/` 下带时间戳的完整日志文件输出

#### 引导和启动序列

脚本执行以下步骤：

1. 检查 `7860` 端口上是否有现有 StoryWeaver 进程（有则安全重启）
2. 自动创建 `.venv` 虚拟环境（如果不存在）
3. 升级 `pip` 到最新版本
4. 安装 `requirements.txt` 中的所有依赖
5. 启动 Streamlit 应用（默认 URL: `http://127.0.0.1:7860`）
6. 输出完整日志到 `logs/storyweaver_prod_<timestamp>.log`

#### 脚本使用指南

- **首次运行（全新机器/项目克隆）：**
  - Windows: 在项目根目录打开 PowerShell，运行 `./scripts/start_project_prod.bat`
  - macOS/Linux: 运行一次 `chmod +x ./scripts/start_project_prod.sh`，然后 `./scripts/start_project_prod.sh`
  - 等待直到在控制台看到 Streamlit 启动 URL

- **后续运行：**
  - 为你的操作系统运行相同命令
  - 现有 `.venv` 被重用，依赖项被检查/更新
  - 如果 StoryWeaver 实例已在 `7860` 上运行，脚本将自动重启该进程
  - 完整执行日志保存至 `logs/` 目录供后续检查

#### 日志和诊断

- 每次启动都在 `logs/` 目录生成带时间戳的日志文件（格式：`storyweaver_prod_YYYYMMDD_HHMMSS.log`）
- 日志记录引导过程的每一步，便于诊断启动问题
- 如果启动失败，查看日志文件获取详细错误信息

### 意图模型（CPU 友好默认值）

意图分类在训练期间默认为 `distilbert-base-uncased`，推理时从本地检查点目录加载：

- 默认检查点目录: `models/intent_classifier`
- 如果模型目录缺失或加载失败，系统自动回退到 `rule_fallback`
- API 不变: `predict(text) -> {"intent": str, "confidence": float}`

推荐的 CPU 设置：

1. `batch_size=8`
2. `max_length=128`
3. 典型意图推理延迟: 约 20-80ms/回合（取决于 CPU 和模型大小）
4. 意图模型预期内存: 约 300-700MB（包括运行时开销）

- **API 配置（推荐）：**
  - 编辑 `.env`
  - 至少设置：
    - `OPENAI_API_KEY=sk-...`
    - `OPENAI_BASE_URL=https://api.openai.com/v1`（或您兼容的端点）

#### 可选：运行测试

脚本引导后，你可以运行：

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

#### 故障排查

- **脚本运行后端口仍被占用？**
  - 脚本首先检查 StoryWeaver 是否已在 `7860/7861` 上运行；如果是，它重用该实例。
  - 如果两个端口都被其他程序占用，启动停止并出现错误。
  - 手动检查 (Windows): `netstat -ano | findstr :7860` 或 `:7861`，然后 `taskkill /PID <PID> /F`
  - 手动检查 (macOS/Linux): `lsof -i :7860` 或 `:7861`，然后 `kill -9 <PID>`
  
- **spaCy 模型下载超时？**
  - 这在较慢的网络上是预期行为。脚本将警告但继续 — 实体提取回退到正则表达式模式。
  - 你可以稍后手动下载：
    - Windows: `.venv\Scripts\python.exe -m spacy download en_core_web_sm`
    - macOS/Linux: `.venv/bin/python -m spacy download en_core_web_sm`

- **`.env` 缺失且 API 调用失败？**
  - 将 `.env.example` 复制到 `.env` 并设置 `OPENAI_API_KEY`
  - 当两者都缺失时，脚本会自动执行此操作

### 技术栈

- **NLU**: DistilBERT（意图分类）+ spaCy（命名实体识别）+ fastcoref（指代消解）
- **NLG**: OpenAI gpt-4o-mini API（故事生成、选项生成、关系提取）
- **知识图**: NetworkX MultiDiGraph + PyVis 可视化
- **前端**: Streamlit（聊天 UI + KG 面板 + NLU 调试）
- **评估**: Distinct-n、Self-BLEU、实体覆盖、一致性率、LLM 作为评判者

### 意图训练快速入门

使用默认 DistilBERT 基础模型并保存到标准检查点目录：

```bash
python training/train_intent.py --output_dir models/intent_classifier --model_name distilbert-base-uncased --epochs 6 --batch_size 8 --max_length 128
```

运行时，如果 `models/intent_classifier` 不可用，系统自动回退到关键字 `rule_fallback` 模式。

### 中文文档索引

1. `docs/guides/technical-route.md` - NLU/KG/NLG 技术路线和回退策略
2. `docs/guides/data-flow.md` - 每回合字段级数据流和模块映射
3. `docs/guides/deployment-windows.md` - Windows 高可用性部署指南
4. `docs/guides/deployment-macos.md` - macOS 高可用性部署指南
5. `docs/design/entity-importance.md` - 实体重要性评分策略
6. `docs/design/nlg-local-model-finetuning.md` - NLG 模块本地大模型微调方案
7. `docs/reports/kg-optimization.md` - 知识图谱优化报告
8. `docs/reports/nlu-kg-improvement.md` - NLU & KG 模块改进报告
