# StoryWeaver 前端接口文档 (API Reference)

> **Version:** 1.0.0  
> **Last Updated:** 2026-03-19  
> **Base URL:** `http://localhost:7860`  
> **Framework:** Streamlit + Python Backend  

---

## Table of Contents

1. [Overview](#1-overview)
2. [UI Interaction Model](#2-ui-interaction-model)
3. [Data Models](#3-data-models)
4. [Backend API Reference](#4-backend-api-reference)
5. [NLU Module API](#5-nlu-module-api)
6. [NLG Module API](#6-nlg-module-api)
7. [Knowledge Graph API](#7-knowledge-graph-api)
8. [Evaluation API](#8-evaluation-api)
9. [Configuration](#9-configuration)
10. [Error Handling](#10-error-handling)
11. [Examples](#11-examples)

---

## 1. Overview

StoryWeaver is an interactive text adventure game engine that combines:

- **NLU (Natural Language Understanding):** Intent classification, entity extraction, coreference resolution
- **NLG (Natural Language Generation):** LLM-powered story and option generation
- **KG (Knowledge Graph):** Dynamic world-state tracking with conflict detection

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Chat UI  │  │ Option   │  │ KG Vis   │  │ Evaluation    │  │
│  │ Input    │  │ Buttons  │  │ Panel    │  │ Dashboard     │  │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └───────────────┘  │
│       │              │                                          │
│       └──────────────┴──────────────────────────────────────────┘
│                              │
│                    ┌─────────▼─────────┐
│                    │   GameEngine      │
│                    │  (Orchestrator)   │
│                    └─────────┬─────────┘
│                              │
│         ┌────────────────────┼────────────────────┐
│         │                    │                    │
│  ┌──────▼──────┐    ┌───────▼───────┐    ┌───────▼───────┐
│  │  NLU Layer  │    │  NLG Layer    │    │  KG Layer     │
│  │ ┌─────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │
│  │ │ Intent  │ │    │ │ Story Gen │ │    │ │ Graph     │ │
│  │ │ Entity  │ │    │ │ Option Gen│ │    │ │ Relations │ │
│  │ │ Coref   │ │    │ └───────────┘ │    │ │ Conflict  │ │
│  │ └─────────┘ │    └──────────────┘    │ └───────────┘ │
│  └─────────────┘                        └───────────────┘
│         │                                      │
│  ┌──────▼──────┐                        ┌──────▼──────┐
│  │ DistilBERT  │                        │ NetworkX    │
│  │ spaCy       │                        │ MultiDiGraph│
│  │ fastcoref   │                        │ PyVis       │
│  └─────────────┘                        └─────────────┘
│                              │
│                    ┌─────────▼─────────┐
│                    │   LLM Client      │
│                    │ (OpenAI Compatible)│
│                    └───────────────────┘
```

---

## 2. UI Interaction Model

### 2.1 Page Layout

| Zone | Component | Description |
|------|-----------|-------------|
| **Main Header** | Hero Banner | 项目标题与简介 |
| **Main Area** | Genre Input | 故事类型输入框 (e.g., `fantasy`, `sci-fi`, `mystery`) |
| **Main Area** | New Game Button | 🎮 开始新游戏 按钮 |
| **Main Area** | Chat History | 对话历史 (支持按轮次折叠) |
| **Main Area** | Option Buttons | 🧭 分支选项 (3个可点击选项) |
| **Main Area** | Chat Input | 自由行动输入框 |
| **Main Area** | Evaluation Panel | 📊 会话评测面板 |
| **Sidebar** | NLU Model Config | 🧠 NLU 模型路径配置 |
| **Sidebar** | KG Visualization | 📊 知识图谱可视化 |
| **Sidebar** | Consistency Trend | 📈 一致性趋势图 |
| **Sidebar** | NLU Debug Info | 🔍 NLU 解析详情 |
| **Sidebar** | Stats | 轮次/实体/冲突 计数 |
| **Sidebar** | Download | 📥 下载完整故事 |

### 2.2 User Interaction Flow

```
┌──────────────────┐
│  用户访问页面     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────────┐
│  输入 Genre      │────▶│ GameEngine.start_game │
│  点击"开始新游戏" │     └──────────┬───────────┘
└──────────────────┘                │
         │                          ▼
         │              ┌───────────────────────┐
         │              │ 生成开场白 + 3个选项   │
         │              │ 初始化知识图谱          │
         │              └──────────┬────────────┘
         │                         │
         ▼                         ▼
┌──────────────────────────────────────────────┐
│              游戏进行中                       │
│                                              │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │ 用户输入自由行动 │  │ 用户点击选项按钮  │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           │                    │             │
│           └────────┬───────────┘             │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │    GameEngine.process_turn(input)    │    │
│  │                                     │    │
│  │  1. 指代消解 (Coreference)          │    │
│  │  2. 意图识别 (Intent)               │    │
│  │  3. 实体抽取 (Entity)               │    │
│  │  4. 故事生成 (Story Gen via LLM)    │    │
│  │  5. 知识图谱更新 (KG Update)        │    │
│  │  6. 冲突检测 (Conflict Detection)   │    │
│  │  7. 选项生成 (Option Gen via LLM)   │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │           返回 TurnResult            │    │
│  │  - story_text (叙述文本)             │    │
│  │  - options (分支选项)               │    │
│  │  - nlu_debug (NLU 调试信息)        │    │
│  │  - kg_html (图谱可视化)             │    │
│  │  - conflicts (冲突列表)             │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────┐                        │
│  │  点击"运行评测"  │                        │
│  └────────┬────────┘                        │
│           │                                  │
│           ▼                                  │
│  ┌─────────────────────────────────────┐    │
│  │        评测结果展示                  │    │
│  │  - 自动指标 (Distinct-n, BLEU...)   │    │
│  │  - LLM Judge 维度评分              │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────┐                        │
│  │ 点击"下载故事"  │──▶ 导出 full_story.txt │
│  └─────────────────┘                        │
└──────────────────────────────────────────────┘
```

---

## 3. Data Models

### 3.1 TurnResult

每次游戏轮次处理后返回的核心数据结构。

```python
@dataclass
class TurnResult:
    story_text: str                    # 生成的叙述文本
    options: List[StoryOption]         # 玩家可选分支 (通常3个)
    nlu_debug: Dict = {}               # NLU 处理调试信息
    kg_html: str = ""                  # 知识图谱可视化 HTML
    conflicts: List[str] = []          # 世界观一致性冲突描述
```

**JSON Schema:**

```json
{
  "type": "object",
  "properties": {
    "story_text": {
      "type": "string",
      "description": "LLM 生成的叙述文本，包含故事情节发展"
    },
    "options": {
      "type": "array",
      "items": { "$ref": "#/definitions/StoryOption" },
      "description": "玩家可选的行动分支列表"
    },
    "nlu_debug": {
      "type": "object",
      "properties": {
        "resolved_input": { "type": "string", "description": "指代消解后的玩家输入" },
        "intent": { "type": "string", "description": "识别的意图类别" },
        "confidence": { "type": "number", "description": "意图识别置信度 (0-1)" },
        "entities": { "type": "array", "items": { "$ref": "#/definitions/Entity" }, "description": "抽取的实体列表" },
        "intent_backend": { "type": "string", "description": "使用的意图识别后端" },
        "intent_model_loaded": { "type": "boolean", "description": "意图模型是否已加载" },
        "coref_loaded": { "type": "boolean", "description": "指代消解模型是否已加载" },
        "entity_model_loaded": { "type": "boolean", "description": "实体抽取模型是否已加载" }
      }
    },
    "kg_html": {
      "type": "string",
      "description": "PyVis 生成的知识图谱可视化 HTML 内容"
    },
    "conflicts": {
      "type": "array",
      "items": { "type": "string" },
      "description": "检测到的世界观一致性冲突描述列表"
    }
  },
  "required": ["story_text", "options"]
}
```

**Example:**

```json
{
  "story_text": "你推开古堡沉重的橡木门，灰尘在阳光中飞舞。大厅中央矗立着一座石雕喷泉，水面映出你的倒影。远处传来低沉的吟唱声...",
  "options": [
    {
      "text": "走向石雕喷泉仔细观察",
      "intent_hint": "explore",
      "risk_level": "low"
    },
    {
      "text": "循着吟唱声前进",
      "intent_hint": "explore",
      "risk_level": "medium"
    },
    {
      "text": "大声询问是谁在吟唱",
      "intent_hint": "dialogue",
      "risk_level": "medium"
    }
  ],
  "nlu_debug": {
    "resolved_input": "走向石雕喷泉仔细观察",
    "intent": "explore",
    "confidence": 0.87,
    "entities": [
      { "text": "石雕喷泉", "type": "item", "start": 2, "end": 6, "source": "spacy" }
    ],
    "intent_backend": "rule_fallback",
    "intent_model_loaded": false,
    "coref_loaded": false,
    "entity_model_loaded": true
  },
  "kg_html": "<html>...</html>",
  "conflicts": []
}
```

### 3.2 StoryOption

玩家可选的行动分支。

```python
@dataclass
class StoryOption:
    text: str            # 选项文本 (显示给玩家)
    intent_hint: str     # 建议意图类别
    risk_level: str      # 风险等级: "low" | "medium" | "high"
```

**JSON Schema:**

```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "选项文本内容，直接显示给玩家"
    },
    "intent_hint": {
      "type": "string",
      "enum": ["action", "dialogue", "explore", "use_item", "ask_info", "rest", "trade", "other"],
      "description": "该选项的建议意图类别"
    },
    "risk_level": {
      "type": "string",
      "enum": ["low", "medium", "high"],
      "description": "选项的风险等级，影响故事走向难度"
    }
  },
  "required": ["text", "intent_hint", "risk_level"]
}
```

### 3.3 GameState

游戏状态管理。

```python
@dataclass
class GameState:
    turn_id: int = 0                           # 当前轮次 (从 0 开始)
    genre: str = "fantasy"                      # 故事类型
    story_history: List[Dict[str, str]] = []    # 对话历史
```

**story_history 条目格式:**

```json
{
  "role": "player | narrator",
  "text": "对话内容"
}
```

### 3.4 Entity

NLU 抽取的实体。

```python
# Entity dict structure
{
    "text": str,        # 实体文本
    "type": str,        # 实体类型
    "start": int,       # 在原文中的起始位置
    "end": int,         # 在原文中的结束位置
    "source": str       # 抽取来源 ("spacy" | "noun_phrase")
}
```

**JSON Schema:**

```json
{
  "type": "object",
  "properties": {
    "text": { "type": "string", "description": "实体文本内容" },
    "type": {
      "type": "string",
      "enum": ["person", "location", "item", "creature", "event"],
      "description": "实体类型"
    },
    "start": { "type": "integer", "description": "起始字符位置" },
    "end": { "type": "integer", "description": "结束字符位置" },
    "source": { "type": "string", "description": "抽取来源" }
  },
  "required": ["text", "type"]
}
```

### 3.5 NLU Debug Info

NLU 处理的调试信息。

```json
{
  "resolved_input": "string - 指代消解后的输入文本",
  "intent": "string - 意图类别",
  "confidence": "number - 置信度 (0-1)",
  "entities": "array<Entity> - 实体列表",
  "intent_backend": "string - 'distilbert' | 'rule_fallback'",
  "intent_model_loaded": "boolean",
  "coref_loaded": "boolean",
  "entity_model_loaded": "boolean"
}
```

### 3.6 Intent Labels

意图分类标签 (共 8 类):

| Label | Description | Example Input |
|-------|-------------|---------------|
| `action` | 执行动作 | "攻击怪物" |
| `dialogue` | 对话交流 | "和村民交谈" |
| `explore` | 探索环境 | "走进森林" |
| `use_item` | 使用物品 | "使用治疗药水" |
| `ask_info` | 询问信息 | "这里有什么危险?" |
| `rest` | 休息恢复 | "原地休息" |
| `trade` | 交易买卖 | "购买装备" |
| `other` | 其他 | 无法归类的输入 |

### 3.7 KG Entity Types

知识图谱实体类型:

| Type | Description |
|------|-------------|
| `person` | 人物角色 |
| `location` | 地点位置 |
| `item` | 物品道具 |
| `creature` | 生物怪物 |
| `event` | 事件 |

### 3.8 KG Relation Types

知识图谱关系类型:

| Relation | Description | Example |
|----------|-------------|---------|
| `located_at` | 位于 | 勇士 -[位于]-> 古堡 |
| `possesses` | 拥有 | 勇士 -[拥有]-> 宝剑 |
| `ally_of` | 同盟 | 勇士 -[同盟]-> 法师 |
| `enemy_of` | 敌对 | 勇士 -[敌对]-> 恶龙 |
| `knows` | 知晓 | 法师 -[知晓]-> 秘密 |
| `part_of` | 属于 | 大厅 -[属于]-> 古堡 |
| `caused_by` | 由...引起 | 火灾 -[由...引起]-> 闪电 |
| `has_attribute` | 具有属性 | 宝剑 -[具有属性]-> 锋利 |

---

## 4. Backend API Reference

### 4.1 GameEngine

核心游戏引擎，协调整个 NLU → NLG → KG 处理管线。

#### `GameEngine.__init__(genre, intent_model_path, auto_load_nlu)`

初始化游戏引擎。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genre` | `str` | `"fantasy"` | 故事类型 (fantasy, sci-fi, mystery, etc.) |
| `intent_model_path` | `Optional[str]` | `None` | 意图分类模型路径，None 使用默认路径 |
| `auto_load_nlu` | `bool` | `True` | 是否自动加载 NLU 组件 |

**Example:**

```python
engine = GameEngine(
    genre="fantasy",
    intent_model_path="models/intent_classifier"
)
```

#### `GameEngine.start_game() -> TurnResult`

开始新游戏，生成开场白。

**Returns:** `TurnResult` - 包含开场叙述、初始选项、知识图谱

**Example:**

```python
result = engine.start_game()
print(result.story_text)   # 开场白
print(result.options)       # 初始3个选项
print(result.kg_html)       # 知识图谱可视化
```

#### `GameEngine.process_turn(player_input: str) -> TurnResult`

处理玩家输入，执行完整的 7 阶段管线。

**Processing Pipeline:**

```
player_input
    │
    ▼
┌──────────────────────┐
│ 1. Coreference       │  指代消解 (将 "他" 解析为具体实体)
│    Resolution        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 2. Intent            │  意图分类 (action/dialogue/explore...)
│    Classification    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 3. Entity            │  实体抽取 (人物/地点/物品)
│    Extraction        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 4. Story Generation  │  LLM 生成叙述文本
│    (LLM)             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 5. KG Update         │  从叙述中抽取实体和关系
│    (Relation Extr.)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 6. Conflict          │  检测世界观一致性冲突
│    Detection         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 7. Option            │  LLM 生成 3 个玩家可选分支
│    Generation        │
└──────────┬───────────┘
           │
           ▼
     TurnResult
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `player_input` | `str` | 玩家的自由文本输入 |

**Returns:** `TurnResult`

**Example:**

```python
result = engine.process_turn("走进森林深处")
print(result.story_text)    # 生成的叙述
print(result.options)        # 3个选项
print(result.nlu_debug)      # NLU 调试信息
print(result.conflicts)      # 冲突列表 (如有)
```

#### `GameEngine.all_story_texts -> List[str]`

只读属性，返回当前会话的所有叙述文本 (用于评测)。

#### `GameEngine.kg_entity_names -> List[str]`

只读属性，返回知识图谱中所有实体名称。

---

## 5. NLU Module API

### 5.1 IntentClassifier

意图分类器 (DistilBERT + 规则降级)。

#### `IntentClassifier.__init__(model_path, max_length=128)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `Optional[str]` | `None` | 模型目录路径 |
| `max_length` | `int` | `128` | 最大输入长度 |

#### `IntentClassifier.load() -> None`

加载模型。模型不存在时自动降级为规则匹配。

#### `IntentClassifier.predict(text: str) -> Dict`

预测意图。

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | 输入文本 |

**Returns:**

```json
{
  "intent": "explore",
  "confidence": 0.87
}
```

#### `IntentClassifier.rule_fallback(text: str) -> Dict`

规则降级意图识别。

**Keyword Map:**

| Intent | Keywords |
|--------|----------|
| `action` | attack, hit, strike, fight, kill, kick, punch, throw, shoot, break, push, pull |
| `dialogue` | talk, speak, say, tell, ask, greet, shout, whisper, call, yell, chat |
| `explore` | go, walk, move, enter, exit, climb, run, jump, search, look, examine, inspect, investigate |
| `use_item` | use, drink, eat, open, unlock, read, wear, equip, wield, cast, activate |
| `ask_info` | what, where, who, when, how, why, which, question, explain, describe, about |
| `rest` | rest, sleep, wait, sit, relax, recover, heal, nap, pause |
| `trade` | buy, sell, trade, exchange, bargain, shop, purchase, merchant |

### 5.2 EntityExtractor

实体抽取器 (spaCy NER + 名词短语启发式)。

#### `EntityExtractor.load() -> None`

加载 spaCy 模型 (`en_core_web_sm`)。

#### `EntityExtractor.extract(text: str) -> List[Dict]`

抽取实体。

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | 输入文本 |

**Returns:**

```json
[
  {
    "text": "黑森林",
    "type": "location",
    "start": 2,
    "end": 5,
    "source": "spacy"
  }
]
```

**Entity Type Mapping (spaCy label → StoryWeaver type):**

| spaCy Label | StoryWeaver Type |
|-------------|------------------|
| `PERSON` | `person` |
| `GPE`, `LOC`, `FAC` | `location` |
| `NORP`, `ORG` | `person` |
| 其他 | 由名词短语启发式判断 |

### 5.3 CoreferenceResolver

指代消解器 (fastcoref + 规则降级)。

#### `CoreferenceResolver.load() -> None`

加载 fastcoref 模型。

#### `CoreferenceResolver.resolve(text: str, context: List[str]) -> str`

消解文本中的代词指代。

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | 当前输入文本 |
| `context` | `List[str]` | 最近的上下文文本列表 |

**Returns:** `str` - 消解后的文本

**Example:**

```
Context: ["你遇到了一位老法师", "他向你微笑"]
Input:   "向他询问古堡的秘密"
Output:  "向老法师询问古堡的秘密"
```

---

## 6. NLG Module API

### 6.1 StoryGenerator

故事生成器 (LLM)。

#### `StoryGenerator.generate_opening(genre: str = "fantasy") -> str`

生成故事开场白。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genre` | `str` | `"fantasy"` | 故事类型 |

**Returns:** `str` - 开场叙述文本

#### `StoryGenerator.continue_story(player_input, intent, kg_summary, history) -> str`

继续故事叙述。

| Parameter | Type | Description |
|-----------|------|-------------|
| `player_input` | `str` | 玩家输入 (已消解) |
| `intent` | `str` | 识别的意图 |
| `kg_summary` | `str` | 知识图谱摘要文本 |
| `history` | `str` | 最近的对话历史 (格式化字符串) |

**Returns:** `str` - 生成的叙述文本

### 6.2 OptionGenerator

选项生成器 (LLM, JSON mode)。

#### `OptionGenerator.generate(story_text, kg_summary, num_options=None) -> List[StoryOption]`

生成玩家可选分支。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `story_text` | `str` | - | 当前叙述文本 |
| `kg_summary` | `str` | - | 知识图谱摘要 |
| `num_options` | `Optional[int]` | `None` | 选项数量 (默认 3) |

**Returns:** `List[StoryOption]` - 选项列表

**Default num_options:** 由 `config.NUM_OPTIONS` 控制，默认 `3`

---

## 7. Knowledge Graph API

### 7.1 KnowledgeGraph

知识图谱管理 (NetworkX MultiDiGraph)。

#### `KnowledgeGraph.add_entity(name, entity_type, **attrs) -> str`

添加实体节点。

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | 实体名称 |
| `entity_type` | `str` | 实体类型 |
| `**attrs` | `dict` | 附加属性 |

**Returns:** `str` - 标准化的节点 key (小写)

#### `KnowledgeGraph.get_entity(name) -> Optional[Dict]`

获取实体信息。

**Returns:**

```json
{
  "name": "黑森林",
  "type": "location",
  "key": "黑森林"
}
```

#### `KnowledgeGraph.remove_entity(name) -> None`

删除实体节点及其关联关系。

#### `KnowledgeGraph.add_relation(source, target, relation, **attrs) -> None`

添加实体间关系。

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `str` | 源实体名称 |
| `target` | `str` | 目标实体名称 |
| `relation` | `str` | 关系类型 |
| `**attrs` | `dict` | 附加属性 |

#### `KnowledgeGraph.get_relations(name) -> List[Dict]`

获取实体的所有关系。

**Returns:**

```json
[
  {
    "source": "勇士",
    "target": "古堡",
    "relation": "located_at"
  }
]
```

#### `KnowledgeGraph.to_summary(max_entities=30) -> str`

生成知识图谱摘要文本 (用于 LLM prompt)。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_entities` | `int` | `30` | 最大实体数量 |

**Returns:** `str` - 格式化的摘要文本

**Example Output:**

```
Entities: 勇士(person), 古堡(location), 黑森林(location), 宝剑(item)
Relations: 勇士 -[located_at]-> 古堡, 勇士 -[possesses]-> 宝剑, 古堡 -[part_of]-> 黑森林
```

### 7.2 RelationExtractor

关系抽取器 (LLM, JSON mode)。

#### `extract(text: str) -> Dict`

从文本中抽取实体和关系。

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | 输入文本 |

**Returns:**

```json
{
  "entities": [
    { "name": "勇士", "type": "person" },
    { "name": "古堡", "type": "location" }
  ],
  "relations": [
    {
      "source": "勇士",
      "target": "古堡",
      "relation": "located_at"
    }
  ]
}
```

### 7.3 ConflictDetector

世界观一致性冲突检测器。

**Exclusive Pairs (互斥关系对):**

| Pair | Description |
|------|-------------|
| `ally_of` / `enemy_of` | 同盟与敌对不能同时存在 |
| `alive` / `dead` | 生存与死亡不能同时存在 |

#### `ConflictDetector.check_all(new_text="") -> List[Dict]`

检测所有冲突。

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `new_text` | `str` | `""` | 新生成的文本 (可选，用于 LLM 检测) |

**Returns:**

```json
[
  {
    "type": "exclusive_relation",
    "description": "勇士同时与法师存在 ally_of 和 enemy_of 关系"
  }
]
```

### 7.4 Visualizer

知识图谱可视化 (PyVis)。

#### `render_kg_html(graph) -> str`

将 NetworkX 图渲染为可交互 HTML。

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.MultiDiGraph` | 知识图谱 |

**Returns:** `str` - HTML 内容 (可嵌入 iframe)

---

## 8. Evaluation API

### 8.1 Automatic Metrics

#### `full_evaluation(texts, entity_names, turn_conflict_counts) -> Dict[str, float]`

运行所有自动评估指标。

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `List[str]` | 所有叙述文本 |
| `entity_names` | `List[str]` | KG 实体名称列表 |
| `turn_conflict_counts` | `List[int]` | 每轮冲突计数 |

**Returns:**

```json
{
  "distinct_1": 0.4523,
  "distinct_2": 0.7891,
  "distinct_3": 0.9124,
  "self_bleu": 0.2341,
  "entity_coverage": 0.8567,
  "consistency_rate": 0.9231
}
```

**Metrics Description:**

| Metric | Range | Description | Better Direction |
|--------|-------|-------------|-----------------|
| `distinct_1` | 0-1 | 1-gram 多样性 | ↑ Higher |
| `distinct_2` | 0-1 | 2-gram 多样性 | ↑ Higher |
| `distinct_3` | 0-1 | 3-gram 多样性 | ↑ Higher |
| `self_bleu` | 0-1 | 叙述间相似度 | ↓ Lower |
| `entity_coverage` | 0-1 | KG 实体在文本中的覆盖率 | ↑ Higher |
| `consistency_rate` | 0-1 | 无冲突轮次比例 | ↑ Higher |

#### `distinct_n(texts, n=2) -> float`

计算 n-gram 多样性。

#### `self_bleu(texts, max_n=4) -> float`

计算 Self-BLEU 分数。

#### `entity_coverage(texts, entity_names) -> float`

计算实体覆盖率。

#### `consistency_rate(turn_conflict_counts) -> float`

计算一致性率。

### 8.2 LLM Judge

#### `judge(transcript: str) -> Dict[str, int | float]`

使用 LLM 对故事进行多维度评分。

| Parameter | Type | Description |
|-----------|------|-------------|
| `transcript` | `str` | 完整故事文本 |

**Evaluation Dimensions:**

| Dimension | Score Range | Description |
|-----------|-------------|-------------|
| `narrative_quality` | 1-10 | 叙事质量：情节连贯、描写生动 |
| `consistency` | 1-10 | 世界观一致性：无矛盾 |
| `player_agency` | 1-10 | 玩家代理感：选择有意义 |
| `creativity` | 1-10 | 创意性：情节新颖 |
| `pacing` | 1-10 | 节奏：发展速度合理 |
| `average` | 1-10 | 综合平均分 |

**Returns:**

```json
{
  "narrative_quality": 8,
  "consistency": 9,
  "player_agency": 7,
  "creativity": 8,
  "pacing": 7,
  "average": 7.8
}
```

---

## 9. Configuration

### 9.1 Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `""` | LLM API 密钥 |
| `OPENAI_BASE_URL` | `""` | LLM API 基础 URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | 使用的 LLM 模型 |
| `OPENAI_MAX_TOKENS` | `1024` | 最大生成 token 数 |
| `OPENAI_TEMPERATURE` | `0.85` | 生成温度 |
| `OPENAI_TOP_P` | `0.95` | Top-p 采样参数 |
| `NUM_OPTIONS` | `3` | 每轮生成选项数 |
| `KG_MAX_NODES` | `200` | 知识图谱最大节点数 |
| `STREAMLIT_PORT` | `7860` | Streamlit 服务端口 |

### 9.2 Settings Class

```python
class Settings(BaseSettings):
    # Project
    PROJECT_ROOT: Path
    DATA_DIR: Path

    # LLM API
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 1024
    OPENAI_TEMPERATURE: float = 0.85
    OPENAI_TOP_P: float = 0.95

    # NLU
    INTENT_MODEL_NAME: str = "distilbert-base-uncased"
    INTENT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "intent_classifier"
    INTENT_MAX_LENGTH: int = 128
    INTENT_CPU_BATCH_SIZE: int = 8
    INTENT_LABELS: List[str] = ["action", "dialogue", "explore", "use_item",
                                 "ask_info", "rest", "trade", "other"]
    SPACY_MODEL: str = "en_core_web_sm"

    # NLG
    NUM_OPTIONS: int = 3

    # KG
    KG_MAX_NODES: int = 200
    KG_ENTITY_TYPES: List[str] = ["person", "location", "item", "creature", "event"]
    KG_RELATION_TYPES: List[str] = ["located_at", "possesses", "ally_of",
                                      "enemy_of", "knows", "part_of",
                                      "caused_by", "has_attribute"]

    # Game
    NARRATIVE_HISTORY_WINDOW: int = 6
    MAX_CONTEXT_TOKENS: int = 512

    # Streamlit
    STREAMLIT_PORT: int = 7860
```

---

## 10. Error Handling

### 10.1 NLU 降级策略

系统采用渐进式降级策略，确保在模型加载失败时仍可运行：

```
┌─────────────────────────┐
│  IntentClassifier       │
│  ┌───────────────────┐  │
│  │ DistilBERT 加载   │  │
│  └────────┬──────────┘  │
│           │              │
│     成功? │              │
│     ┌─────┴─────┐       │
│     │ Yes       │ No    │
│     ▼           ▼       │
│  使用模型    使用规则     │
│  (高精度)    (低精度)    │
└─────────────────────────┘

┌─────────────────────────┐
│  CoreferenceResolver    │
│  ┌───────────────────┐  │
│  │ fastcoref 加载    │  │
│  └────────┬──────────┘  │
│           │              │
│     成功? │              │
│     ┌─────┴─────┐       │
│     │ Yes       │ No    │
│     ▼           ▼       │
│  使用模型    跳过消解    │
│  (高精度)    (原文不变)  │
└─────────────────────────┘

┌─────────────────────────┐
│  EntityExtractor        │
│  ┌───────────────────┐  │
│  │ spaCy 加载        │  │
│  └────────┬──────────┘  │
│           │              │
│     成功? │              │
│     ┌─────┴─────┐       │
│     │ Yes       │ No    │
│     ▼           ▼       │
│  使用 NER    使用名词    │
│  (高精度)    短语启发式  │
└─────────────────────────┘
```

### 10.2 nlu_debug 后端标识

前端可通过 `nlu_debug.intent_backend` 判断当前使用的意图识别后端：

| Value | Description |
|-------|-------------|
| `distilbert` | 使用训练好的 DistilBERT 模型 |
| `rule_fallback` | 使用关键词规则匹配 |

### 10.3 常见错误场景

| Scenario | Behavior |
|----------|----------|
| LLM API 不可达 | 抛出异常，Streamlit 显示错误信息 |
| NLU 模型缺失 | 自动降级为规则匹配，不中断服务 |
| spaCy 模型缺失 | 使用名词短语启发式抽取 |
| fastcoref 加载失败 | 跳过指代消解，使用原文 |
| KG 节点超限 | 不再添加新节点，保留现有图谱 |
| 知识图谱冲突 | 记录到 `conflicts` 列表，不阻断流程 |

---

## 11. Examples

### 11.1 Complete Game Session Example

```python
from src.engine.game_engine import GameEngine

# 1. 初始化引擎
engine = GameEngine(genre="fantasy")

# 2. 开始新游戏
result = engine.start_game()
print("开场白:", result.story_text)
print("初始选项:")
for i, opt in enumerate(result.options):
    print(f"  {i+1}. [{opt.intent_hint}] {opt.text} (风险: {opt.risk_level})")

# 3. 玩家选择选项
result = engine.process_turn(result.options[0].text)
print("\n叙事:", result.story_text)
print("NLU 调试:", result.nlu_debug)

# 4. 玩家自由输入
result = engine.process_turn("使用照明术照亮前方的道路")
print("\n叙事:", result.story_text)
print("冲突:", result.conflicts)

# 5. 运行评测
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge as llm_judge

auto_scores = full_evaluation(
    texts=engine.all_story_texts,
    entity_names=engine.kg_entity_names,
    turn_conflict_counts=engine.turn_conflict_counts,
)
llm_scores = llm_judge("\n".join(engine.all_story_texts))

print("自动指标:", auto_scores)
print("LLM 评分:", llm_scores)
```

### 11.2 Streamlit UI State Management

```python
# Session State Keys (前端需维护的状态)

st.session_state.engine              # GameEngine 实例
st.session_state.history             # 对话历史 [{"role", "content"}]
st.session_state.consistency_history # 一致性分数列表
st.session_state.kg_html             # 知识图谱 HTML
st.session_state.options             # 当前选项列表
st.session_state.nlu_debug           # NLU 调试信息
st.session_state.eval_result         # 评测报告 (Markdown)
st.session_state.eval_auto           # 自动指标
st.session_state.eval_llm            # LLM Judge 评分
st.session_state.eval_prev_auto      # 上次自动指标 (用于 delta)
st.session_state.eval_prev_llm       # 上次 LLM 评分 (用于 delta)
st.session_state.eval_at             # 评测时间
st.session_state.chat_fold_mode      # 聊天折叠模式
st.session_state.last_elapsed        # 上轮耗时 (秒)
st.session_state.intent_model_path   # 意图模型路径
```

### 11.3 Frontend Data Flow

```
用户操作 ──▶ Streamlit Widget ──▶ Python Function ──▶ GameEngine
                                                        │
                                                        ▼
                                                     TurnResult
                                                        │
              ┌─────────────────────────────────────────┤
              │                                         │
              ▼                                         ▼
     st.session_state 更新                     UI 组件渲染
     - history                                  - st.chat_message
     - options                                  - st.button (选项)
     - kg_html                                  - components.html (KG)
     - nlu_debug                                - st.expander (调试)
     - conflicts                                - st.warning (冲突)
```

---

## Appendix: File Structure Reference

```
story_maker/
├── app.py                          # Streamlit 前端入口
├── config.py                       # 全局配置
├── src/
│   ├── engine/
│   │   ├── game_engine.py          # 游戏引擎 (核心)
│   │   └── state.py                # 游戏状态
│   ├── nlu/
│   │   ├── intent_classifier.py    # 意图分类
│   │   ├── entity_extractor.py     # 实体抽取
│   │   └── coreference.py          # 指代消解
│   ├── nlg/
│   │   ├── story_generator.py      # 故事生成
│   │   ├── option_generator.py     # 选项生成
│   │   └── prompt_templates.py     # Prompt 模板
│   ├── knowledge_graph/
│   │   ├── graph.py                # 知识图谱
│   │   ├── relation_extractor.py   # 关系抽取
│   │   ├── conflict_detector.py    # 冲突检测
│   │   └── visualizer.py           # 可视化
│   ├── evaluation/
│   │   ├── metrics.py              # 自动指标
│   │   ├── llm_judge.py            # LLM 评测
│   │   └── consistency_eval.py     # 一致性评测
│   └── utils/
│       └── api_client.py           # LLM API 客户端
├── models/                         # 训练好的模型
├── tests/                          # 测试用例
└── docs/                           # 文档
```
