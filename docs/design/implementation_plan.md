# StoryWeaver 混合架构实施计划

> **项目**: COMP5423 NLP — 交互式文字冒险故事生成器  
> **架构**: 混合方案（本地 NLU + API NLG）  
> **最后更新**: 2026-03-31

> **注意**: 本文档是项目级实施计划。部分路径和示例为设计草案。有关当前运行时说明，请参考根目录 `README.md`。

---

## I. 架构概览

```
用户输入
  │
  ▼
┌──────────────────────────────────────────────────┐
│  NLU 流水线（本地）                                │
│  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ 意图        │  │ 实体     │  │ 共指消解    │ │
│  │ 分类器      │  │ 抽取器   │  │ 器          │ │
│  │ (DistilBERT)│  │ (spaCy)  │  │ (fastcoref) │ │
│  └──────┬──────┘  └────┬─────┘  └──────┬──────┘ │
│         └───────┬──────┘───────────────┘         │
└─────────────────┼────────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│  知识图谱（本地 NetworkX）                          │
│  ┌───────────────┐  ┌──────────────────────────┐ │
│  │ 图谱管理器     │  │ 冲突检测器               │ │
│  │ (MultiDiGraph) │  │ (规则 + LLM 兜底)        │ │
│  └───────┬───────┘  └───────────┬──────────────┘ │
│          └──────────┬───────────┘                 │
└─────────────────────┼────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│  NLG 流水线（API / 本地混合）                       │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ 故事        │  │ 选项         │               │
│  │ 生成器      │  │ 生成器       │               │
│  │ (GPT/Qwen)  │  │ (GPT/Qwen)   │               │
│  └──────┬──────┘  └──────┬───────┘               │
│         └───────┬────────┘                        │
└─────────────────┼────────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│  Streamlit UI                                     │
│  聊天 + 选项按钮 + KG 可视化 (PyVis)               │
└──────────────────────────────────────────────────┘
```

---

## II. 目录结构

```
story_maker/
├── .env                          # API 密钥（git 忽略）
├── config/
│   └── .env.example              # 环境变量示例
├── .gitignore
├── config.py                     # 全局配置（Pydantic Settings）
├── app.py                        # Streamlit 应用入口
├── requirements.txt              # 依赖列表
├── README.md
│
├── docs/project/                 # 项目文档
│   ├── implementation_plan.md    # 本文件
│   ├── agent_prompt.md           # 智能体提示词
│   └── *.pdf                     # 课程项目规范
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── api_client.py         # 统一 LLM API 客户端
│   │
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py  # DistilBERT 意图分类
│   │   ├── entity_extractor.py   # spaCy NER + 名词短语
│   │   ├── sentiment_analyzer.py # 情感分析
│   │   └── coreference.py        # fastcoref 共指消解
│   │
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── graph.py              # NetworkX MultiDiGraph 管理
│   │   ├── relation_extractor.py # 基于 LLM 的关系抽取
│   │   ├── conflict_detector.py  # 基于规则 + LLM 的冲突检测
│   │   └── visualizer.py         # PyVis 可视化
│   │
│   ├── nlg/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py   # 版本化提示词模板
│   │   ├── story_generator.py    # 故事续写
│   │   └── option_generator.py   # 选项生成
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_engine.py        # 游戏循环编排
│   │   ├── runtime_session.py    # 持久化与会话管理
│   │   ├── state.py              # GameState 数据类
│   │   └── naming.py             # 实体命名与 ID 生成
│   │
│   ├── ui/                       # Streamlit UI 模块
│   │   ├── __init__.py
│   │   ├── state_manager.py      # UI 状态管理
│   │   ├── layout/               # 主题与样式
│   │   └── sections/             # 聊天、侧边栏、评估 UI
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py            # Distinct-n、Self-BLEU 等
│       ├── consistency_eval.py   # 一致性评估指标
│       └── llm_judge.py          # LLM-as-Judge 评估
│
├── tests/                        # 综合测试套件
├── scripts/                      # 启动与工具脚本
├── training/                     # 微调脚本
└── models/                       # 本地模型产物
```

---

## III. 模块详细设计

### 3.1 config.py — 全局配置

**核心特性**:
- 使用 `pydantic-settings` 从 `.env` 加载
- 集中管理 API 配置、NLU 模型、KG 限制和游戏设置
- 支持 `NLG_MODE` 切换（local、api、hybrid）

---

### 3.2 src/utils/api_client.py — 统一 LLM 客户端

**核心特性**:
- 单例模式，全局访问
- 指数退避重试逻辑（3 次尝试）
- 支持 JSON 模式和结构化输出
- 使用量追踪（token 数和预估成本）

---

### 3.3 NLU 模块

#### 3.3.1 intent_classifier.py
- 微调的 `distilbert-base-uncased`
- 8 个意图标签：`action`（行动）、`dialogue`（对话）、`explore`（探索）、`use_item`（使用物品）、`ask_info`（询问信息）、`rest`（休息）、`trade`（交易）、`other`（其他）
- 基于关键词的兜底机制，确保鲁棒性

#### 3.3.2 entity_extractor.py
- `spaCy` NER（`en_core_web_sm`）
- 名词短语启发式方法，捕获游戏特定实体
- 实体类型：`person`（人物）、`location`（地点）、`item`（物品）、`creature`（生物）、`event`（事件）

#### 3.3.3 coreference.py
- 使用 `fastcoref` 进行代词消解
- 解析上下文相关的输入（如 "Look at him" → "Look at the dragon"）

---

### 3.4 知识图谱模块

#### 3.4.1 graph.py — MultiDiGraph 管理
- 基于 `NetworkX` 实现，支持实体间的多重关系
- 增量重要性评分与衰减机制
- 生成摘要用于 LLM 上下文注入

#### 3.4.2 relation_extractor.py — 基于 LLM 的抽取
- 从叙事文本中抽取结构化的实体与关系
- 支持双重抽取（同时从玩家输入和故事文本中抽取）

#### 3.4.3 conflict_detector.py — 双层检测
- 基于规则的检测（如互斥关系 `ally_of` vs `enemy_of`）
- 基于 LLM 的逻辑推理，处理复杂矛盾

---

### 3.5 NLG 模块

#### 3.5.1 story_generator.py
- 处理开场场景与叙事续写
- 利用历史和 KG 摘要进行上下文感知生成

#### 3.5.2 option_generator.py
- 每回合生成 3 个分支选择
- 分配意图提示和风险等级（低/中/高）

---

### 3.6 评估模块

#### 3.6.1 自动指标 (metrics.py)
- `Distinct-n`: 生成文本的多样性
- `Self-BLEU`: 重复性检查
- `Entity Coverage`: KG 集成深度
- `Consistency Rate`: 逻辑冲突频率

#### 3.6.2 LLM-as-Judge (llm_judge.py)
- 多维度评分：叙事质量、一致性、玩家代理感、创意、节奏
- 提供定性反馈与总体评分

---

## IV. 实施时间线

| 周次 | 里程碑 | 交付物 |
|----|--------|--------|
| **W1** | 基础设施 | config.py、api_client.py、初始 KG 结构、测试框架 |
| **W2** | 核心模块 | NLU 流水线（DistilBERT/spaCy）、KG 抽取、NLG 模板 |
| **W3** | 集成 | 游戏引擎编排、Streamlit UI、冲突检测、持久化 |
| **W4** | 评估 | LLM-as-Judge、自动指标、最终 Bug 修复、文档 |

---

## V. 评估标准

| 指标 | 目标 |
|------|------|
| Distinct-2 | ≥ 0.75 |
| Self-BLEU-4 | ≤ 0.35 |
| Consistency Rate（一致性比率） | ≥ 0.85 |
| Narrative Quality（叙事质量，评审） | ≥ 7/10 |
| Intent Accuracy（意图准确率） | ≥ 85% |

---

*实施计划结束*
