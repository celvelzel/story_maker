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
├── src/
│   ├── engine/
│   │   ├── game_engine.py    # 主控制器 (NLU → NLG → KG 管道)
│   │   └── state.py          # 游戏状态和历史追踪
│   ├── nlu/
│   │   ├── intent_classifier.py   # DistilBERT + 关键字回退
│   │   ├── entity_extractor.py    # spaCy NER + 正则表达式
│   │   └── coreference.py         # fastcoref + 基于规则
│   ├── nlg/
│   │   ├── story_generator.py     # OpenAI API 故事生成
│   │   ├── option_generator.py    # 玩家选择生成 (API)
│   │   └── prompt_templates.py    # 提示工程模板
│   ├── knowledge_graph/
│   │   ├── graph.py               # NetworkX MultiDiGraph 包装器
│   │   ├── relation_extractor.py  # LLM 基础关系提取
│   │   ├── conflict_detector.py   # 基于规则 + LLM 一致性
│   │   └── visualizer.py          # PyVis HTML 可视化
│   ├── evaluation/
│   │   ├── metrics.py             # Distinct-n, Self-BLEU, 覆盖率
│   │   ├── llm_judge.py           # LLM 作为评判者评分
│   │   └── consistency_eval.py    # 知识图一致性评估
│   └── utils/
│       └── api_client.py          # 单例 LLM 客户端（含重试）
├── data/
│   ├── intent_labels.json         # 意图标签定义
│   └── scripts/
│       ├── download_data.py       # 数据集下载脚本
│       └── preprocess.py          # 数据预处理管道
├── training/
│   ├── train_intent.py            # DistilBERT 意图分类器训练
│   └── train_generator.py         # GPT-2 LoRA 微调 (遗留/可选)
├── tests/
│   ├── test_nlu.py
│   ├── test_nlg.py
│   ├── test_knowledge_graph.py
│   └── test_integration.py
└── info/                          # 项目文档
```

### 部署与启动 (Windows / macOS)

使用与你的操作系统匹配的一键脚本：

- Windows: `start_project.bat`
- macOS/Linux: `start_project.sh`

两个脚本执行相同的引导和启动序列：

1. 自动创建 `.venv`（如果不存在）
2. 升级 `pip`
3. 安装 `requirements.txt`
4. 安装 Streamlit 运行时依赖
5. 尝试下载 `en_core_web_sm`（如果超时，应用仍会以回退提取启动）
6. 缺少时从 `.env.example` 创建 `.env`
7. 检查 `7860/7861` 上的运行实例并避免重复启动
8. 通过 `python app.py` 启动应用（默认 URL: `http://127.0.0.1:7860` 或回退 `7861`）

#### 脚本使用记录

- **首次运行（全新机器/项目克隆）：**
  - Windows: 在项目根目录打开 PowerShell，运行 `./start_project.bat`
  - macOS/Linux: 运行一次 `chmod +x ./start_project.sh`，然后 `./start_project.sh`
  - 等待直到在控制台看到 Streamlit 启动 URL

- **后续运行：**
  - 为你的操作系统运行相同命令
  - 现有 `.venv` 被重用，依赖项被检查/更新
  - 如果实例已在 `7860` 或 `7861` 上运行，脚本将打印 URL 并退出（无重复进程）

- **强制重启模式（停止旧实例，然后启动新实例）：**
  - Windows: `./start_project.bat --force-restart` 或 `-f`
  - macOS/Linux: `./start_project.sh --force-restart` 或 `-f`

### 生产启动脚本

原始脚本已保留。为高可用性操作添加了生产脚本：

- Windows: `start_project_prod.bat`
- macOS/Linux: `start_project_prod.sh`

生产脚本提供：

1. 端口占用检测和进程识别
2. 现有 Streamlit 应用进程的安全重启策略
3. 具有显式网络超时控制的依赖项安装
4. 环境变量验证（`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`）
5. 启动失败的结构化退出代码
6. `logs/` 下带时间戳的日志文件输出

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

1. `docs/technical_route_zh.md` - NLU/KG/NLG 技术路线和回退策略
2. `docs/data_flow_zh.md` - 每回合字段级数据流和模块映射
3. `docs/deployment_windows_zh.md` - Windows 高可用性部署指南
4. `docs/deployment_macos_zh.md` - macOS 高可用性部署指南
