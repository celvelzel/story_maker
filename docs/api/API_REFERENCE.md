# StoryWeaver API Reference

> **Version:** 1.1.0  
> **Last Updated:** 2026-03-31  
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

- **NLU (Natural Language Understanding):** Intent classification, entity extraction, and coreference resolution.
- **NLG (Natural Language Generation):** Hybrid LLM-powered story and option generation (Local Qwen + Mimo API).
- **KG (Knowledge Graph):** Dynamic world-state tracking with conflict detection and importance scoring.

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
| **Main Header** | Hero Banner | Project title and introduction. |
| **Main Area** | Genre Input | Story type input (e.g., `fantasy`, `sci-fi`, `mystery`). |
| **Main Area** | New Game Button | 🎮 Starts a new game session. |
| **Main Area** | Chat History | Interaction history with foldable turns. |
| **Main Area** | Option Buttons | 🧭 Clickable action branches (usually 3). |
| **Main Area** | Chat Input | Free-text action input. |
| **Main Area** | Evaluation Panel | 📊 Session evaluation metrics and dashboard. |
| **Sidebar** | NLU Model Config | 🧠 NLU model path and backend settings. |
| **Sidebar** | KG Visualization | 📊 Interactive knowledge graph view. |
| **Sidebar** | Consistency Trend | 📈 Visual trend of story consistency. |
| **Sidebar** | NLU Debug Info | 🔍 Detailed NLU parsing results. |
| **Sidebar** | Stats | Counters for turns, entities, and conflicts. |
| **Sidebar** | Download | 📥 Export full story as text. |

### 2.2 User Interaction Flow

```
┌──────────────────┐
│  User Accesses   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────────┐
│  Enter Genre     │────▶│ GameEngine.start_game │
│  Click New Game  │     └──────────┬───────────┘
└──────────────────┘                │
         │                          ▼
         │              ┌───────────────────────┐
         │              │ Generate Opening +    │
         │              │ 3 Options + Init KG   │
         │              └──────────┬────────────┘
         │                         │
         ▼                         ▼
┌──────────────────────────────────────────────┐
│              Game in Progress                │
│                                              │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │ Free-text Input │  │ Click Option     │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           │                    │             │
│           └────────┬───────────┘             │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │    GameEngine.process_turn(input)    │    │
│  │                                     │    │
│  │  1. Coreference Resolution          │    │
│  │  2. Intent Classification           │    │
│  │  3. Entity Extraction               │    │
│  │  4. Story Generation (via LLM)      │    │
│  │  5. KG Update (Relation Extr.)      │    │
│  │  6. Conflict Detection              │    │
│  │  7. Option Generation (via LLM)     │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │           Return TurnResult          │    │
│  │  - story_text                       │    │
│  │  - options                          │    │
│  │  - nlu_debug                        │    │
│  │  - kg_html                          │    │
│  │  - conflicts                        │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────┐                        │
│  │ Run Evaluation  │                        │
│  └────────┬────────┘                        │
│           │                                  │
│           ▼                                  │
│  ┌─────────────────────────────────────┐    │
│  │        Evaluation Results           │    │
│  │  - Auto Metrics (Distinct-n, etc.)  │    │
│  │  - LLM Judge Scores                 │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────┐                        │
│  │ Download Story  │──▶ Export .txt file    │
│  └─────────────────┘                        │
└──────────────────────────────────────────────┘
```

---

## 3. Data Models

### 3.1 TurnResult

Core data structure returned after each turn processing.

```python
@dataclass
class TurnResult:
    story_text: str                    # Generated narrative text
    options: List[StoryOption]         # Player action branches
    nlu_debug: Dict = {}               # NLU debugging information
    kg_html: str = ""                  # KG visualization HTML
    conflicts: List[str] = []          # Consistency conflict descriptions
```

### 3.2 StoryOption

Selectable action branch for the player.

```python
@dataclass
class StoryOption:
    text: str            # Option text displayed to user
    intent_hint: str     # Suggested intent category
    risk_level: str      # Risk level: "low" | "medium" | "high"
```

### 3.3 GameState

Session state management.

```python
@dataclass
class GameState:
    turn_id: int = 0                           # Current turn (starts at 0)
    genre: str = "fantasy"                      # Story genre
    story_history: List[Dict[str, str]] = []    # Dialogue history
```

### 3.4 Entity

NLU extracted entity.

```python
{
    "text": str,        # Entity text
    "type": str,        # Entity type (person, location, item, creature, event)
    "start": int,       # Start position in text
    "end": int,         # End position in text
    "source": str       # Source (spacy | noun_phrase)
}
```

### 3.5 Intent Labels

Supported intent categories: `action`, `dialogue`, `explore`, `use_item`, `ask_info`, `rest`, `trade`, `other`.

---

## 4. Backend API Reference

### 4.1 GameEngine

Orchestrates the NLU → NLG → KG pipeline.

#### `GameEngine.__init__(genre="fantasy", intent_model_path=None, auto_load_nlu=True)`

- `genre`: Default story type.
- `intent_model_path`: Custom path for DistilBERT model.
- `auto_load_nlu`: If True, loads local models on initialization.

#### `GameEngine.start_game() -> TurnResult`

Initializes the session and generates the opening narrative.

#### `GameEngine.process_turn(player_input: str) -> TurnResult`

Executes the full 7-stage pipeline for a single turn.

---

## 5. NLU Module API

### 5.1 IntentClassifier

DistilBERT-based classifier with keyword fallback.
Backend identifiable via `nlu_debug.intent_backend` (`distilbert` | `rule_fallback`).

### 5.2 EntityExtractor

Hybrid spaCy NER and noun-phrase heuristic extractor.

### 5.3 CoreferenceResolver

Uses `fastcoref` to resolve pronouns based on recent history.

---

## 6. NLG Module API

### 6.1 StoryGenerator

Handles narrative generation. Supports `api` (Mimo/OpenAI), `local` (Local Qwen), and `hybrid` modes.

### 6.2 OptionGenerator

Generates 3 branching choices in JSON format.

---

## 7. Knowledge Graph API

### 7.1 KnowledgeGraph

Manages the world state using a NetworkX MultiDiGraph. Features:
- **Importance Scoring:** Nodes are ranked by mention frequency and player focus.
- **Incremental Updates:** Efficiently updates only new information.
- **Persistence:** Supports snapshots and auto-saving to `saves/`.

### 7.2 RelationExtractor

LLM-powered extraction of entities and relations from text.

### 7.3 ConflictDetector

Hybrid rule-based and LLM-based consistency checking.

---

## 8. Evaluation API

### 8.1 Automatic Metrics

- `distinct_n`: Measures n-gram diversity.
- `self_bleu`: Measures narrative redundancy.
- `entity_coverage`: Ratio of KG entities mentioned in text.
- `consistency_rate`: Ratio of conflict-free turns.

### 8.2 LLM Judge

Dimensions: `narrative_quality`, `consistency`, `player_agency`, `creativity`, `pacing`.

---

## 9. Configuration

Configured via `config.py` and `.env`.

- `OPENAI_MODEL`: Defaults to `mimo-v2-flash`.
- `NLG_MODE`: `api`, `local`, or `hybrid`.
- `KG_IMPORTANCE_MODE`: `composite` for advanced ranking.

---

## 10. Error Handling

- **LLM API Failure:** Raises exception; UI displays error toast.
- **NLU Model Missing:** Graceful fallback to rule-based matching.
- **KG Conflict:** Logged in `TurnResult.conflicts` without blocking the turn.

---

## 11. Examples

Refer to `scripts/test_openai_api.py` or `tests/integration/test_integration.py` for code examples.

