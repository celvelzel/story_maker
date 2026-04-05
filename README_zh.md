# StoryWeaver：动态情节生成的 AI 驱动文本冒险游戏

> **最后更新**: 2026-04-05

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
├── app.py                          # Streamlit 应用程序入口
├── config.py                       # Pydantic 设置支持 .env
├── requirements.txt                # Python 依赖项
├── .env                            # 环境变量 (git-ignored)
├── .env.llama                      # llama.cpp 服务器配置
├── .env.vllm                       # vLLM GPU 推理配置
├── .env.vllm.cpu                   # vLLM CPU 推理配置
├── .env.vllm.example               # vLLM 配置模板
├── start_project_prod.bat          # Windows 生产启动器（根目录快捷方式）
├── start_project_prod.sh           # macOS/Linux 生产启动器（根目录快捷方式）
│
├── src/                            # 源代码模块
│   ├── __init__.py
│   ├── engine/                     # 游戏引擎主控制器
│   │   ├── __init__.py
│   │   ├── game_engine.py          # 主管道协调器 (NLU → NLG → KG)
│   │   ├── runtime_session.py      # 会话持久化管理器
│   │   ├── state.py                # 游戏状态和历史追踪
│   │   └── naming.py               # 角色/地点命名系统
│   ├── nlu/                        # 自然语言理解
│   │   ├── __init__.py
│   │   ├── intent_classifier.py    # DistilBERT + 关键字回退
│   │   ├── entity_extractor.py     # spaCy NER + 正则表达式
│   │   ├── coreference.py          # fastcoref + 规则消解
│   │   └── sentiment_analyzer.py   # 情感/语气分析 (Ekman 6 类)
│   ├── nlg/                        # 自然语言生成
│   │   ├── __init__.py
│   │   ├── story_generator.py      # OpenAI API 故事生成
│   │   ├── option_generator.py     # 玩家选择生成 (API)
│   │   └── prompt_templates.py     # 提示工程模板
│   ├── knowledge_graph/            # 动态世界状态管理
│   │   ├── __init__.py
│   │   ├── graph.py                # NetworkX MultiDiGraph 包装器
│   │   ├── relation_extractor.py   # LLM 基础关系提取
│   │   ├── conflict_detector.py    # 规则 + LLM 一致性检查
│   │   └── visualizer.py           # PyVis HTML 可视化
│   ├── evaluation/                 # 质量评估指标
│   │   ├── __init__.py
│   │   ├── metrics.py              # Distinct-n, Self-BLEU, 覆盖率
│   │   ├── llm_judge.py            # LLM 作为评判者评分
│   │   └── consistency_eval.py     # 知识图一致性评估
│   ├── ui/                         # Streamlit UI 组件
│   │   ├── __init__.py
│   │   ├── layout/                 # 页面布局和主题
│   │   │   ├── __init__.py
│   │   │   └── theme.py
│   │   ├── sections/               # UI 部分模块
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── evaluation.py
│   │   │   └── sidebar.py
│   │   └── state_manager.py        # UI 状态管理
│   └── utils/                      # 共享工具
│       ├── __init__.py
│       └── api_client.py           # 单例 LLM 客户端（含重试）
│
├── scripts/                        # 工具和部署脚本
│   ├── start/                      # 启动脚本
│   │   ├── start_project_prod.bat  # Windows 生产启动器
│   │   ├── start_project_prod.sh   # macOS/Linux 生产启动器
│   │   ├── start_llama_server.bat  # llama.cpp 服务器启动器
│   │   ├── start_inference_server.sh
│   │   └── start_streamlit.sh
│   ├── config/                     # 环境配置模板
│   │   ├── .env.llama
│   │   ├── .env.vllm
│   │   ├── .env.vllm.cpu
│   │   └── .env.vllm.example
│   ├── data/                       # 数据集生成工具
│   │   ├── generate_dataset.py
│   │   ├── extract_pdfs.py
│   │   ├── read_pdfs.py
│   │   ├── fix_and_merge.py
│   │   └── validate_and_merge.py
│   ├── eval/                       # 评估运行器
│   │   ├── run_automated_eval.py
│   │   ├── run_eval_benchmark.py
│   │   ├── run_kg_on_off_benchmark.py
│   │   ├── run_llm_judge.py
│   │   └── simple_model_eval.py
│   ├── inference/                  # 推理工具
│   │   ├── local_inference_server.py
│   │   └── test_openai_api.py
│   └── quantize/                   # 模型量化
│       └── quantize_gguf.bat
│
├── training/                       # 模型训练脚本
│   ├── train_intent.py             # DistilBERT 意图分类器训练
│   ├── train_generator.py          # GPT-2 LoRA 微调 (遗留/可选)
│   ├── train_llama.sh              # Llama.cpp 训练脚本
│   ├── train_qwen.sh               # Qwen 训练脚本
│   ├── data_augmenter.py           # 训练数据增强
│   └── nlg_dataset/                # NLG 训练数据集
│       ├── combined_data.jsonl     # 合并训练数据集
│       └── combined_data_generate_prompt.md  # 数据生成提示词
│
├── tests/                          # 测试套件（按模块组织）
│   ├── __init__.py
│   ├── engine/                     # 引擎组件测试
│   ├── nlu/                        # NLU 模块测试
│   ├── nlg/                        # NLG 模块测试
│   ├── kg/                         # 知识图谱测试
│   ├── integration/                # 跨模块集成测试
│   ├── evaluation/                 # 质量评估测试
│   ├── performance/                # 性能基准测试
│   ├── ui/                         # UI 组件测试
│   └── utils/                      # 工具函数测试
│
├── docs/                           # 全面文档
│   ├── README.md                   # 文档索引
│   ├── api/                        # API 参考文档
│   │   ├── README.md
│   │   └── API_REFERENCE.md
│   ├── design/                     # 架构和设计文档
│   │   ├── README.md
│   │   ├── prompts/                # 提示词模板
│   │   ├── conflict-detection-resolution.md
│   │   ├── entity-importance.md
│   │   ├── hybrid-nlg-architecture.md
│   │   ├── implementation_plan.md
│   │   ├── kg-summary-modes.md
│   │   ├── nlg-local-model-finetuning.md
│   │   ├── sentiment-analysis.md
│   │   └── storyweaver_pipeline.*  # 管道图 (drawio/svg/html)
│   ├── guides/                     # 部署和使用指南
│   │   ├── README.md
│   │   ├── CPU_INFERENCE.md
│   │   ├── data-flow.md
│   │   ├── deployment-macos.md
│   │   ├── deployment-windows.md
│   │   ├── local-model-startup.md
│   │   ├── technical-route.md
│   │   └── zero-to-hero-deployment.md
│   ├── fixes/                      # 问题修复报告
│   │   ├── README.md
│   │   ├── distilbert-compatibility-fix.md
│   │   ├── distilbert-tokenizer-fix.md
│   │   ├── distilbert-troubleshooting.md
│   │   ├── fastcoref-fix.md
│   │   └── llm-json-truncation-fix.md
│   ├── reports/                    # 优化和评估报告
│   │   ├── README.md
│   │   ├── changelog/              # 自动生成的更新日志
│   │   ├── evaluation/             # 模型评估结果
│   │   ├── local-model/            # 本地模型报告
│   │   ├── optimization/           # 优化报告
│   │   └── test-results/           # 测试运行结果
│   ├── project/                    # 项目规范和材料
│   │   ├── COMP5423 NLP Group Project Specification-2026.pdf
│   │   └── project intro.pdf
│   └── final_submit/               # 最终提交材料
│       └── final_report/
│           └── Final_Project_Report.md
│
├── models/                         # 训练模型文件 (git-ignored)
│   ├── intent_classifier/          # 微调 DistilBERT 检查点
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── checkpoint-*/           # 训练检查点
│   └── nlg/                        # NLG 模型检查点
│       └── README.md
│
├── lib/                            # 第三方前端库
│   ├── vis-9.1.2/                  # Vis.js 网络可视化
│   │   ├── vis-network.min.js
│   │   └── vis-network.css
│   ├── tom-select/                 # 增强选择组件
│   │   ├── tom-select.complete.min.js
│   │   └── tom-select.css
│   └── bindings/                   # JavaScript 工具
│       └── utils.js
│
├── reports/                        # 独立评估报告
│   ├── comparison/                 # 模型对比报告
│   │   └── model-comparison.md
│   ├── evaluation/                 # 评估结果
│   │   ├── automated_eval_report.md
│   │   ├── local-model-eval.md
│   │   └── mimo_eval_report.md
│   └── hybrid/                     # 混合策略报告
│       ├── hybrid_eval_report.md
│       ├── hybrid_strategy_guide.md
│       └── hybrid_vs_standalone_comparison.md
│
├── saves/                          # 游戏存档文件 (git-ignored)
│   ├── runtime_engine.json         # 运行时引擎状态
│   ├── runtime_session.json        # 会话持久化
│   └── *.json                      # 独立游戏存档
│
├── config/                         # 配置模板
│   └── .env.example                # 环境配置模板
│
├── logs/                           # 应用日志 (git-ignored)
└── .gitignore                      # Git 忽略规则
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

#### 使用指南
1. **[技术路线](docs/guides/technical-route.md)** - NLU/KG/NLG 技术路线和回退策略
2. **[数据流](docs/guides/data-flow.md)** - 每回合字段级数据流和模块映射
3. **[从零到一部署](docs/guides/zero-to-hero-deployment.md)** - 完整设置指南
4. **[Windows 部署](docs/guides/deployment-windows.md)** - Windows 高可用性部署指南
5. **[macOS 部署](docs/guides/deployment-macos.md)** - macOS 高可用性部署指南
6. **[CPU 推理](docs/guides/CPU_INFERENCE.md)** - CPU 推理优化指南
7. **[本地模型启动](docs/guides/local-model-startup.md)** - 本地模型启动指南

#### 架构与设计
8. **[实体重要性](docs/design/entity-importance.md)** - 实体重要性评分策略
9. **[混合 NLG 架构](docs/design/hybrid-nlg-architecture.md)** - 混合 NLG 架构设计
10. **[NLG 本地微调](docs/design/nlg-local-model-finetuning.md)** - NLG 模块本地大模型微调方案
11. **[知识图摘要模式](docs/design/kg-summary-modes.md)** - 知识图谱摘要模式
12. **[情感分析](docs/design/sentiment-analysis.md)** - 情感/语气分析策略
13. **[冲突检测](docs/design/conflict-detection-resolution.md)** - 冲突检测与解决方案

#### API 与报告
14. **[API 参考](docs/api/API_REFERENCE.md)** - 完整 API 参考文档
15. **[知识图优化](docs/reports/optimization/kg-optimization.md)** - 知识图谱优化报告
16. **[NLU & KG 改进](docs/reports/optimization/nlu-kg-improvement.md)** - NLU & KG 模块改进报告
17. **[会话持久化](docs/reports/optimization/runtime-persistence.md)** - 运行时会话持久化文档
18. **[评估报告](docs/reports/evaluation/)** - 模型评估结果 (API、本地、混合)
19. **[测试结果](docs/reports/test-results/)** - 自动化测试与 KG 开关基准测试

#### 其他
20. **[修复报告](docs/fixes/)** - Bug 修复文档 (DistilBERT、fastcoref、LLM JSON)
21. **[更新日志](docs/reports/changelog/)** - 自动生成的更新日志
