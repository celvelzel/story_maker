# StoryWeaver API 参考文档

> **版本：** 1.2.0  
> **最后更新：** 2026-04-01  
> **基础 URL：** `http://localhost:7860`  
> **框架：** Streamlit + Python 后端  
> **NLG 模式：** 混合模式 (本地 Qwen3 + Mimo API)  

---

## 目录

1. [概述](#1-概述)
2. [UI 交互模型](#2-ui-交互模型)
3. [数据模型](#3-数据模型)
4. [后端 API 参考](#4-后端-api-参考)
5. [NLU 模块 API](#5-nlu-模块-api)
6. [NLG 模块 API](#6-nlg-模块-api)
7. [知识图谱 API](#7-知识图谱-api)
8. [评估 API](#8-评估-api)
9. [配置](#9-配置)
10. [错误处理](#10-错误处理)
11. [示例](#11-示例)

---

## 1. 概述

StoryWeaver 是一个交互式文字冒险游戏引擎，结合了以下核心能力：

- **NLU（自然语言理解）：** 意图分类（DistilBERT + 关键词兜底）、实体抽取（spaCy + 名词短语 + KG 上下文）、共指消解（fastcoref + 规则兜底）、情感/情绪分析（distilroberta + 关键词兜底）。
- **NLG（自然语言生成）：** 基于 LLM 的混合式故事与选项生成。支持通过可配置路由切换 `api`、`local` 和 `hybrid` 三种模式。
- **KG（知识图谱）：** 动态世界状态追踪，支持冲突检测（规则 + 时序 + LLM）、多策略解决、重要性评分（composite/incremental/degree_only）以及分层摘要生成。

### 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit 前端                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ 聊天界面  │  │ 选项按钮  │  │ KG 可视化 │  │ 评估仪表盘    │  │
│  │ 输入      │  │          │  │ 面板      │  │              │  │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └───────────────┘  │
│       │              │                                          │
│       └──────────────┴──────────────────────────────────────────┘
│                              │
│                    ┌─────────▼─────────┐
│                    │   GameEngine      │
│                    │   (编排器)         │
│                    └─────────┬─────────┘
│                              │
│         ┌────────────────────┼────────────────────┐
│         │                    │                    │
│  ┌──────▼──────┐    ┌───────▼───────┐    ┌───────▼───────┐
│  │  NLU 层     │    │  NLG 层       │    │  KG 层        │
│  │ ┌─────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │
│  │ │ 意图    │ │    │ │ 故事生成   │ │    │ │ 图谱      │ │
│  │ │ 实体    │ │    │ │ 选项生成   │ │    │ │ 关系      │ │
│  │ │ 共指    │ │    │ └───────────┘ │    │ │ 冲突      │ │
│  │ │ 情感    │ │    │               │    │ └───────────┘ │
│  │ └─────────┘ │    └───────────────┘    └───────────────┘
│  └─────────────┘
│         │                                      │
│  ┌──────▼──────┐                        ┌──────▼──────┐
│  │ DistilBERT  │                        │ NetworkX    │
│  │ spaCy       │                        │ MultiDiGraph│
│  │ fastcoref   │                        │ PyVis       │
│  │ distilroberta│                       └─────────────┘
│  └─────────────┘
│                    ┌─────────▼─────────┐
│                    │   LLM 客户端       │
│                    │ (混合: 本地 +     │
│                    │  OpenAI 兼容)      │
│                    └───────────────────┘
```

---

## 2. UI 交互模型

### 2.1 页面布局

| 区域 | 组件 | 描述 |
|------|-----------|-------------|
| **主标题** | 横幅 | 项目标题与简介 |
| **主区域** | 类型输入 | 故事类型输入（如 `fantasy`、`sci-fi`、`mystery`） |
| **主区域** | 新游戏按钮 | 🎮 开始新的游戏会话 |
| **主区域** | 聊天历史 | 交互历史，支持折叠回合 |
| **主区域** | 选项按钮 | 🧭 可点击的分支行动（通常 3 个） |
| **主区域** | 聊天输入 | 自由文本行动输入 |
| **主区域** | 评估面板 | 📊 会话评估指标与仪表盘 |
| **侧边栏** | NLU 模型配置 | 🧠 NLU 模型路径与后端设置 |
| **侧边栏** | KG 可视化 | 📊 交互式知识图谱视图 |
| **侧边栏** | 一致性趋势 | 📈 故事一致性可视化趋势 |
| **侧边栏** | NLU 调试信息 | 🔍 详细 NLU 解析结果（意图、情绪、实体、共指） |
| **侧边栏** | 统计 | 回合数、实体数、冲突数计数器 |
| **侧边栏** | 下载 | 📥 导出完整故事为文本文件 |

### 2.2 用户交互流程

```
┌──────────────────┐
│    用户访问       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────────┐
│    输入类型       │────▶│ GameEngine.start_game │
│    点击新游戏     │     └──────────┬───────────┘
└──────────────────┘                │
         │                          ▼
         │              ┌───────────────────────┐
         │              │ 生成开场白 +           │
         │              │ 3 个选项 + 初始化 KG   │
         │              └──────────┬────────────┘
         │                         │
         ▼                         ▼
┌──────────────────────────────────────────────┐
│              游戏进行中                        │
│                                              │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  自由文本输入    │  │  点击选项         │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           │                    │             │
│           └────────┬───────────┘             │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │    GameEngine.process_turn(input)    │    │
│  │                                     │    │
│  │  1. 共指消解                         │    │
│  │  2. 意图分类                         │    │
│  │  3. 情感/情绪分析                    │    │
│  │  4. 实体抽取                         │    │
│  │  5. 故事生成（通过 LLM）             │    │
│  │  6. KG 更新（双重/故事抽取）         │    │
│  │  7. 冲突检测与解决                   │    │
│  │  8. 选项生成（通过 LLM）             │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │           返回 TurnResult            │    │
│  │  - story_text（故事文本）            │    │
│  │  - options（选项列表）               │    │
│  │  - nlu_debug（含情绪信息）           │    │
│  │  - kg_html（图谱可视化）             │    │
│  │  - conflicts（冲突列表）             │    │
│  │  - kg_node_count, kg_edge_count     │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────┐                        │
│  │    运行评估      │                        │
│  └────────┬────────┘                        │
│           │                                  │
│           ▼                                  │
│  ┌─────────────────────────────────────┐    │
│  │           评估结果                   │    │
│  │  - 自动指标（Distinct-n 等）         │    │
│  │  - LLM 评审得分（8 个维度）          │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────┐                        │
│  │   下载故事       │──▶ 导出 .txt 文件      │
│  └─────────────────┘                        │
└──────────────────────────────────────────────┘
```

---

## 3. 数据模型

### 3.1 TurnResult（回合结果）

每回合处理后返回的核心数据结构。

```python
@dataclass
class TurnResult:
    story_text: str                    # 生成的叙事文本
    options: List[StoryOption]         # 玩家分支行动选项
    nlu_debug: Dict = {}               # NLU 调试信息（意图、情绪、实体、阶段指标）
    kg_html: str = ""                  # KG 可视化 HTML
    conflicts: List[str] = []          # 一致性冲突描述
    kg_node_count: int = 0             # 回合结束时 KG 节点数
    kg_edge_count: int = 0             # 回合结束时 KG 边数
```

### 3.2 StoryOption（故事选项）

玩家可选择的行动分支。

```python
@dataclass
class StoryOption:
    text: str            # 向用户显示的选项文本
    intent_hint: str     # 建议的意图类别
    risk_level: str      # 风险等级："low"（低）| "medium"（中）| "high"（高）
```

### 3.3 GameState（游戏状态）

会话状态管理。

```python
@dataclass
class GameState:
    turn_id: int = 0                           # 当前回合（从 0 开始）
    genre: str = "fantasy"                      # 故事类型
    story_history: List[Dict[str, str]] = []    # 对话历史 [{"role": "player"|"narrator", "text": "..."}]
    kg_turn_stats: List[Dict[str, int]] = []    # 每回合 KG 节点/边快照
```

### 3.4 Entity（实体）

NLU 抽取的实体。

```python
{
    "text": str,        # 实体文本
    "type": str,        # 实体类型（person/人物, location/地点, item/物品, creature/生物, event/事件）
    "start": int,       # 在文本中的起始位置
    "end": int,         # 在文本中的结束位置
    "source": str,      # 来源（spacy | noun_phrase/名词短语 | possessive/所有格 | regex/正则 | kg_context/KG上下文 | kg_alias/KG别名）
    "confidence": float # 置信度（0.0-1.0）
}
```

### 3.5 Emotion Result（情绪结果）

情感分析输出。

```python
{
    "emotion": str,         # 主导情绪标签
    "confidence": float,    # 置信度（0.0-1.0）
    "scores": Dict[str, float]  # 所有情绪得分
}
```

情绪标签：`anger`（愤怒）、`disgust`（厌恶）、`fear`（恐惧）、`joy`（喜悦）、`sadness`（悲伤）、`surprise`（惊讶）、`neutral`（中性）。

### 3.6 Intent Labels（意图标签）

支持的意图类别：`action`（行动）、`dialogue`（对话）、`explore`（探索）、`use_item`（使用物品）、`ask_info`（询问信息）、`rest`（休息）、`trade`（交易）、`other`（其他）。

---

## 4. 后端 API 参考

### 4.1 GameEngine（游戏引擎）

编排 NLU → NLG → KG 流水线。

#### `GameEngine.__init__(genre="fantasy", intent_model_path=None, auto_load_nlu=False, conflict_resolution=None, extraction_mode=None, importance_mode=None, summary_mode=None)`

- `genre`：默认故事类型
- `intent_model_path`：DistilBERT 模型的自定义路径
- `auto_load_nlu`：如果为 True，在初始化时加载本地模型（默认：在首次 `process_turn` 时惰性加载）
- `conflict_resolution`：覆盖 `KG_CONFLICT_RESOLUTION` 设置
- `extraction_mode`：覆盖 `KG_EXTRACTION_MODE` 设置
- `importance_mode`：覆盖 `KG_IMPORTANCE_MODE` 设置
- `summary_mode`：覆盖 `KG_SUMMARY_MODE` 设置

#### `GameEngine.start_game() -> TurnResult`

初始化会话并生成开场叙事。从开场文本中初始化 KG。

#### `GameEngine.process_turn(player_input: str) -> TurnResult`

执行单回合的完整 8 阶段流水线：
1. **共指消解** — fastcoref 使用近期历史消解代词（含实体类型感知）
2. **意图分类** — DistilBERT 微调分类器（含关键词兜底）
3. **情感分析** — distilroberta 情绪分类器（含关键词兜底）
4. **实体抽取** — spaCy NER + 名词短语启发式 + KG 上下文模糊匹配
5. **故事生成** — LLM 继续叙事（由 NLG_MODE 路由）
6. **KG 更新** — LLM 抽取实体与关系（dual_extract 或 story_only 模式）
7. **冲突检测与解决** — 规则 + 时序 + LLM 检测，配合可配置的解决策略
8. **选项生成** — LLM 生成 3 个带风险等级的玩家选择

#### `GameEngine.save_game(filepath=None) -> str`

将当前游戏状态（KG + 故事历史）保存为 JSON。支持新游戏的语义命名。

#### `GameEngine.load_game(filepath: str) -> None`

从 JSON 文件加载已保存的游戏状态。

#### 评估辅助方法

- `GameEngine.all_story_texts` → 所有叙述者文本列表
- `GameEngine.kg_entity_names` → 所有 KG 实体显示名称列表
- `GameEngine.kg_density_inputs` → 每回合 KG 大小快照

---

## 5. NLU 模块 API

### 5.1 IntentClassifier（意图分类器）

基于 DistilBERT 的分类器，含关键词兜底。后端可通过 `nlu_debug.intent_backend` 识别（`distilbert` | `rule_fallback`）。

- `load()` — 从路径加载模型，支持重试（最多 3 次）。失败时回退到规则匹配
- `predict(text: str) -> Dict[str, object]` — 返回 `{"intent": str, "confidence": float}`
- `rule_fallback(text: str) -> Dict[str, object]` — 基于关键词的分类（始终可用）

### 5.2 EntityExtractor（实体抽取器）

混合 spaCy NER + 名词短语启发式 + KG 上下文感知抽取器。

- `load()` — 加载 spaCy 模型。失败时回退到仅名词短语
- `extract(text: str, known_entities=None) -> List[Dict]` — 返回去重后的实体列表，含 KG 上下文增强
- 支持与已知 KG 实体的模糊匹配以解析别名

### 5.3 CoreferenceResolver（共指消解器）

使用 `fastcoref` FCoref 基于近期历史消解代词。增强实体类型感知。

- `load()` — 加载 fastcoref 模型。失败时回退到规则消解
- `resolve(text: str, context=None, known_entities=None) -> str` — 返回代词被先行词替换后的文本
- 支持人称代词、非人称代词、所有格代词和反身代词

### 5.4 SentimentAnalyzer（情感分析器）

基于 DistilRoBERTa 的情绪分类器，含关键词兜底。

- `load()` — 加载 `j-hartmann/emotion-english-distilroberta-base`，支持重试。失败时回退到关键词匹配
- `analyze(text: str) -> Dict[str, object]` — 返回 `{"emotion": str, "confidence": float, "scores": Dict[str, float]}`
- 情绪标签：`anger`（愤怒）、`disgust`（厌恶）、`fear`（恐惧）、`joy`（喜悦）、`sadness`（悲伤）、`surprise`（惊讶）、`neutral`（中性）

---

## 6. NLG 模块 API

### 6.1 StoryGenerator（故事生成器）

通过 LLM 处理叙事生成。所有路由由 `api_client` 管理。

- `generate_opening(genre: str) -> str` — 生成开场场景
- `continue_story(player_input, intent, kg_summary, history, emotion) -> str` — 基于玩家行动和世界状态继续叙事

### 6.2 OptionGenerator（选项生成器）

通过 LLM 以 JSON 格式生成 3 个分支选择。失败时回退到硬编码选项。

- `generate(story_text, kg_summary, num_options=None) -> List[StoryOption]` — 返回上下文相关的玩家选项

### 6.3 混合 NLG 路由

`src/utils/api_client.py` 中的 `LLMClient` 和 `HybridClientManager` 支持三种模式：

| NLG_MODE | 故事生成 | 选项/关系生成 |
|----------|-----------------|---------------------------|
| `api` | Mimo/OpenAI API | Mimo/OpenAI API |
| `local` | 本地 Qwen3 (llama.cpp) | 本地 Qwen3 (llama.cpp) |
| `hybrid` | 本地 Qwen3 (llama.cpp) | Mimo/OpenAI API |

---

## 7. 知识图谱 API

### 7.1 KnowledgeGraph（知识图谱）

使用 NetworkX MultiDiGraph 管理世界状态。特性：
- **丰富实体属性：** 描述、状态、状态历史、情绪追踪、时序元数据
- **重要性评分：** 三种模式 — `composite`（默认）、`incremental`、`degree_only`
- **分层摘要生成：** `flat`（向后兼容）和 `layered`（按重要性排序含描述）
- **时序衰减：** 每回合关系置信度衰减，衰减周期可配置
- **持久化：** 支持快照和自动保存到 `saves/`

#### 关键方法
- `add_entity(name, entity_type, description, status, turn_id, is_player_mentioned, emotion)` — 新增或更新实体节点
- `add_relation(source, target, relation, context, turn_id, confidence)` — 添加带丰富属性的边
- `update_entity_state(name, state_updates, turn_id)` — 更新实体状态字段
- `refresh_mentions(mentioned_names, turn_id, player_mentioned_names)` — 批量更新提及追踪
- `apply_decay(turn_id)` — 降低未确认关系的置信度
- `recalculate_importance()` — 重新计算重要性评分（支持增量模式）
- `to_summary(max_entities)` — 生成文本世界状态摘要（flat 或 layered）
- `get_timeline(n)` — 以时间线形式返回近期事件
- `to_dict() / from_dict(data)` — 序列化/反序列化
- `save(filepath) / load(filepath)` — 文件持久化

### 7.2 RelationExtractor（关系抽取器）

基于 LLM 的实体与关系抽取。

- `extract(text: str)` — 增强模式：抽取含描述、状态、状态变化的实体 + 含上下文的关系
- `extract_dual(player_input, story_text, existing_entities)` — 双重抽取：在单次 LLM 调用中同时从玩家输入和故事文本中抽取
- 模块级便捷函数：`extract()`、`extract_dual()`、`extract_legacy()`

### 7.3 ConflictDetector（冲突检测器）

混合规则与 LLM 的一致性检查，支持多策略解决。

**检测层级：**
1. **规则检测：** 互斥关系对（ally_of↔enemy_of、alive↔dead）、死亡活跃检测
2. **时序检测：** 死后行为、因果倒置
3. **LLM 检测：** 通过 LLM 分析检测逻辑矛盾

**解决策略：**
- `keep_latest` — 确定性策略：保留更新信息，移除旧的冲突数据
- `llm_arbitrate` — 确定性优先，然后 LLM 仲裁剩余冲突

---

## 8. 评估 API

### 8.1 自动指标

| 指标 | 描述 |
|--------|-------------|
| `distinct_n` | 衡量 n-gram 多样性（Distinct-1, 2, 3） |
| `self_bleu` | 衡量叙事冗余度（越低 = 越多样） |
| `entity_coverage` | 文本中提及的 KG 实体比例 |
| `consistency_rate` | 无冲突回合的比例 |
| `type_token_ratio` | 词汇丰富度指标 |
| `flesch_reading_ease` | 可读性评分（英语启发式） |
| `lexical_overlap` | 相邻回合词汇重叠度 |
| `graph_density_evolution` | KG 密度随回合变化趋势 |

### 8.2 LLM 评审

以 **8 个维度**（1-10 分制）评估完整故事会话：

| 维度 | 描述 |
|-----------|-------------|
| `narrative_quality` | 散文质量、生动描述、引人入胜的语言 |
| `consistency` | 角色、地点和事实保持一致 |
| `player_agency` | 玩家选择对故事的影响程度 |
| `creativity` | 情节、设定和角色的原创性 |
| `pacing` | 故事节奏和张力管理是否恰当 |
| `option_relevance` | 提供的选项与当前上下文的契合度 |
| `causal_link` | 玩家行为是否引起可信的状态变化 |
| `local_coherence` | 相邻回合之间的连贯性 |

使用独立的评估 LLM 配置（`EVAL_LLM_*` 设置，默认：通过智谱 API 的 `glm-5`）。

---

## 9. 配置

通过 `config.py`（Pydantic Settings）和 `.env` 配置。

### LLM API 设置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `OPENAI_API_KEY` | `""` | 本地/OpenAI 兼容端点的 API 密钥 |
| `OPENAI_BASE_URL` | `""` | OpenAI 兼容 API 的基础 URL |
| `OPENAI_MODEL` | `mimo-v2-flash` | NLG 生成的模型名称 |
| `OPENAI_MAX_TOKENS` | `1024` | 最大生成 token 数 |
| `OPENAI_TEMPERATURE` | `0.85` | 生成温度 |
| `OPENAI_TIMEOUT_CONNECT` | `10.0` | 连接超时（秒） |
| `OPENAI_TIMEOUT_READ` | `60.0` | 读取超时（秒） |

### 评估 LLM 设置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `EVAL_LLM_API_KEY` | `""` | LLM 评审的独立 API 密钥 |
| `EVAL_LLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | 评估 LLM 端点 |
| `EVAL_LLM_MODEL` | `glm-5` | 评估模型 |
| `EVAL_LLM_MAX_TOKENS` | `256` | 评估最大 token 数 |
| `EVAL_LLM_TEMPERATURE` | `0.3` | 评估温度 |

### NLU 设置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `INTENT_MODEL_NAME` | `distilbert-base-uncased` | 基础模型名称 |
| `INTENT_MODEL_PATH` | `models/intent_classifier` | 微调模型路径 |
| `INTENT_MAX_LENGTH` | `128` | 最大 token 长度 |
| `INTENT_LABELS` | 8 个标签 | `action, dialogue, explore, use_item, ask_info, rest, trade, other` |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy NER 模型 |

### NLG 设置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `NLG_MODE` | `hybrid` | `api`、`local` 或 `hybrid` |
| `NUM_OPTIONS` | `3` | 每回合玩家选项数 |

### 知识图谱设置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `KG_MAX_NODES` | `200` | KG 最大节点数 |
| `KG_ENTITY_TYPES` | 5 种类型 | `person`（人物）、`location`（地点）、`item`（物品）、`creature`（生物）、`event`（事件） |
| `KG_RELATION_TYPES` | 12 种类型 | `located_at`、`possesses`、`ally_of`、`enemy_of`、`knows`、`part_of`、`caused_by`、`has_attribute`、`causes`、`prevents`、`enables`、`follows` |
| `KG_CONFLICT_RESOLUTION` | `llm_arbitrate` | `keep_latest` 或 `llm_arbitrate` |
| `KG_EXTRACTION_MODE` | `dual_extract` | `story_only` 或 `dual_extract` |
| `KG_IMPORTANCE_MODE` | `composite` | `degree_only`、`composite` 或 `incremental` |
| `KG_SUMMARY_MODE` | `layered` | `flat` 或 `layered` |
| `KG_IMPORTANCE_DECAY_FACTOR` | `0.95` | 每回合重要性衰减因子 |
| `KG_RELATION_DECAY_FACTOR` | `0.90` | 关系置信度衰减因子 |
| `KG_RELATION_MIN_CONFIDENCE` | `0.2` | 最小关系置信度阈值 |
| `KG_IMPORTANCE_MENTION_BOOST` | `0.15` | 提及重要性提升值 |
| `KG_IMPORTANCE_PLAYER_BOOST` | `0.3` | 玩家提及额外提升值 |
| `KG_MAX_TIMELINE_ENTRIES` | `5` | 最大时间线条目数 |
| `KG_DECAY_CADENCE` | `1` | 每 N 回合应用衰减 |
| `KG_INCREMENTAL_FULL_RECALC_INTERVAL` | `10` | 增量模式完整重计算间隔 |
| `KG_ENABLE_INCREMENTAL_IMPORTANCE` | `True` | 启用增量重要性 |
| `KG_ENABLE_SUMMARY_CACHE` | `True` | 启用每回合 KG 摘要缓存 |

### 持久化设置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `KG_SAVE_DIR` | `saves/` | 保存目录 |
| `KG_AUTO_SAVE` | `True` | 启用自动保存 |
| `KG_SNAPSHOT_INTERVAL` | `5` | 每 N 回合创建快照 |

### 游戏设置

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `NARRATIVE_HISTORY_WINDOW` | `6` | LLM 上下文的近期历史条目数 |
| `MAX_CONTEXT_TOKENS` | `512` | 最大上下文 token 数 |
| `STREAMLIT_PORT` | `7860` | Streamlit 服务器端口 |

---

## 10. 错误处理

- **LLM API 失败：** 自动重试，指数退避（最多 3 次）。UI 显示错误提示
- **NLU 模型缺失：** 优雅回退到规则匹配（意图、实体、共指、情感）
- **KG 冲突：** 记录在 `TurnResult.conflicts` 中，不阻塞回合。根据配置的策略解决
- **JSON 解析失败：** 多阶段修复（Markdown 围栏剥离、平衡 JSON 提取、尾逗号移除、严格重试）
- **Transformers 版本警告：** 如果 transformers >= 4.50，记录警告（测试范围：4.40–4.49）

---

## 11. 示例

API 连通性测试请参考 `scripts/test_openai_api.py`，完整流水线集成测试请参考 `tests/integration/test_integration.py`。

### 快速开始

```python
from src.engine.game_engine import GameEngine

# 创建引擎，使用惰性 NLU 加载（默认）
engine = GameEngine(genre="fantasy")

# 开始新游戏
result = engine.start_game()
print(result.story_text)
print([opt.text for opt in result.options])

# 处理玩家回合
result = engine.process_turn("I draw my sword and attack the dragon")
print(result.story_text)
print(f"意图: {result.nlu_debug['intent']}, 情绪: {result.nlu_debug['emotion']}")
print(f"知识图谱: {result.kg_node_count} 个节点, {result.kg_edge_count} 条边")
print(f"冲突: {result.conflicts}")
```
