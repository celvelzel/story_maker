# StoryWeaver API Reference

> **Version:** 1.2.0  
> **Last Updated:** 2026-04-01  
> **Base URL:** `http://localhost:7860`  
> **Framework:** Streamlit + Python Backend  
> **NLG Mode:** Hybrid (Local Qwen3 + Mimo API)  

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

- **NLU (Natural Language Understanding):** Intent classification (DistilBERT + keyword fallback), entity extraction (spaCy + noun-phrase + KG context), coreference resolution (fastcoref + rule fallback), and sentiment/emotion analysis (distilroberta + keyword fallback).
- **NLG (Natural Language Generation):** Hybrid LLM-powered story and option generation. Supports `api`, `local`, and `hybrid` modes via configurable routing.
- **KG (Knowledge Graph):** Dynamic world-state tracking with conflict detection (rule-based + temporal + LLM), multi-strategy resolution, importance scoring (composite/incremental/degree_only), and layered summary generation.

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
│  │ │ Sentiment│ │                   │ └───────────┘ │
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
│                    │   LLM Client      │
│                    │ (Hybrid: Local +  │
│                    │  OpenAI Compatible)│
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
| **Sidebar** | NLU Debug Info | 🔍 Detailed NLU parsing results (intent, emotion, entities, coref). |
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
│  │  3. Sentiment/Emotion Analysis      │    │
│  │  4. Entity Extraction               │    │
│  │  5. Story Generation (via LLM)      │    │
│  │  6. KG Update (Dual/Story Extract)  │    │
│  │  7. Conflict Detection + Resolution │    │
│  │  8. Option Generation (via LLM)     │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │           Return TurnResult          │    │
│  │  - story_text                       │    │
│  │  - options                          │    │
│  │  - nlu_debug (incl. emotion)        │    │
│  │  - kg_html                          │    │
│  │  - conflicts                        │    │
│  │  - kg_node_count, kg_edge_count     │    │
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
│  │  - LLM Judge Scores (8 dimensions)  │    │
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
    nlu_debug: Dict = {}               # NLU debugging (intent, emotion, entities, stage_metrics)
    kg_html: str = ""                  # KG visualization HTML
    conflicts: List[str] = []          # Consistency conflict descriptions
    kg_node_count: int = 0             # KG node count at turn end
    kg_edge_count: int = 0             # KG edge count at turn end
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
    story_history: List[Dict[str, str]] = []    # Dialogue history [{"role": "player"|"narrator", "text": "..."}]
    kg_turn_stats: List[Dict[str, int]] = []    # Per-turn KG node/edge snapshots
```

### 3.4 Entity

NLU extracted entity.

```python
{
    "text": str,        # Entity text
    "type": str,        # Entity type (person, location, item, creature, event)
    "start": int,       # Start position in text
    "end": int,         # End position in text
    "source": str,      # Source (spacy | noun_phrase | possessive | regex | kg_context | kg_alias)
    "confidence": float # Confidence score (0.0-1.0)
}
```

### 3.5 Emotion Result

Sentiment analysis output.

```python
{
    "emotion": str,         # Dominant emotion label
    "confidence": float,    # Confidence score (0.0-1.0)
    "scores": Dict[str, float]  # All emotion scores
}
```

Emotion labels: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `neutral`.

### 3.6 Intent Labels

Supported intent categories: `action`, `dialogue`, `explore`, `use_item`, `ask_info`, `rest`, `trade`, `other`.

---

## 4. Backend API Reference

### 4.1 GameEngine

Orchestrates the NLU → NLG → KG pipeline.

#### `GameEngine.__init__(genre="fantasy", intent_model_path=None, auto_load_nlu=False, conflict_resolution=None, extraction_mode=None, importance_mode=None, summary_mode=None)`

- `genre`: Default story type.
- `intent_model_path`: Custom path for DistilBERT model.
- `auto_load_nlu`: If True, loads local models on initialization (default: lazy load on first `process_turn`).
- `conflict_resolution`: Override for `KG_CONFLICT_RESOLUTION` setting.
- `extraction_mode`: Override for `KG_EXTRACTION_MODE` setting.
- `importance_mode`: Override for `KG_IMPORTANCE_MODE` setting.
- `summary_mode`: Override for `KG_SUMMARY_MODE` setting.

#### `GameEngine.start_game() -> TurnResult`

Initializes the session and generates the opening narrative. Seeds KG from opening text.

#### `GameEngine.process_turn(player_input: str) -> TurnResult`

Executes the full 8-stage pipeline for a single turn:
1. **Coreference Resolution** — fastcoref resolves pronouns using recent history (with entity-type awareness).
2. **Intent Classification** — DistilBERT fine-tuned classifier (with keyword fallback).
3. **Sentiment Analysis** — distilroberta emotion classifier (with keyword fallback).
4. **Entity Extraction** — spaCy NER + noun-phrase heuristics + KG context fuzzy matching.
5. **Story Generation** — LLM continues the narrative (routed by NLG_MODE).
6. **KG Update** — LLM extracts entities & relations (dual_extract or story_only mode).
7. **Conflict Detection + Resolution** — Rule-based + temporal + LLM detection with configurable resolution strategy.
8. **Option Generation** — LLM generates 3 player choices with risk levels.

#### `GameEngine.save_game(filepath=None) -> str`

Saves current game state (KG + story history) to JSON. Supports semantic naming for new games.

#### `GameEngine.load_game(filepath: str) -> None`

Loads a saved game state from JSON file.

#### Evaluation Helpers

- `GameEngine.all_story_texts` → List of all narrator texts.
- `GameEngine.kg_entity_names` → List of all KG entity display names.
- `GameEngine.kg_density_inputs` → Per-turn KG size snapshots.

---

## 5. NLU Module API

### 5.1 IntentClassifier

DistilBERT-based classifier with keyword fallback. Backend identifiable via `nlu_debug.intent_backend` (`distilbert` | `rule_fallback`).

- `load()` — Loads model from path with retry (up to 3 attempts). Falls back to rule-based on failure.
- `predict(text: str) -> Dict[str, object]` — Returns `{"intent": str, "confidence": float}`.
- `rule_fallback(text: str) -> Dict[str, object]` — Keyword-based classification (always available).

### 5.2 EntityExtractor

Hybrid spaCy NER + noun-phrase heuristic + KG context-aware extractor.

- `load()` — Loads spaCy model. Falls back to noun-phrase only on failure.
- `extract(text: str, known_entities=None) -> List[Dict]` — Returns deduplicated entity list with KG context enrichment.
- Supports fuzzy matching against known KG entities for alias resolution.

### 5.3 CoreferenceResolver

Uses `fastcoref` FCoref to resolve pronouns based on recent history. Enhanced with entity-type awareness.

- `load()` — Loads fastcoref model. Falls back to rule-based resolution.
- `resolve(text: str, context=None, known_entities=None) -> str` — Returns text with pronouns replaced by antecedents.
- Supports personal, non-personal, possessive, and reflexive pronouns.

### 5.4 SentimentAnalyzer

DistilRoBERTa-based emotion classifier with keyword fallback.

- `load()` — Loads `j-hartmann/emotion-english-distilroberta-base` with retry. Falls back to keyword matching.
- `analyze(text: str) -> Dict[str, object]` — Returns `{"emotion": str, "confidence": float, "scores": Dict[str, float]}`.
- Emotion labels: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `neutral`.

---

## 6. NLG Module API

### 6.1 StoryGenerator

Handles narrative generation via LLM. All routing handled by `api_client`.

- `generate_opening(genre: str) -> str` — Generates opening scene.
- `continue_story(player_input, intent, kg_summary, history, emotion) -> str` — Continues narrative based on player action and world state.

### 6.2 OptionGenerator

Generates 3 branching choices in JSON format with LLM. Falls back to hardcoded options on failure.

- `generate(story_text, kg_summary, num_options=None) -> List[StoryOption]` — Returns contextual player options.

### 6.3 Hybrid NLG Routing

The `LLMClient` and `HybridClientManager` in `src/utils/api_client.py` support three modes:

| NLG_MODE | Story Generation | Option/Relation Generation |
|----------|-----------------|---------------------------|
| `api` | Mimo/OpenAI API | Mimo/OpenAI API |
| `local` | Local Qwen3 (llama.cpp) | Local Qwen3 (llama.cpp) |
| `hybrid` | Local Qwen3 (llama.cpp) | Mimo/OpenAI API |

---

## 7. Knowledge Graph API

### 7.1 KnowledgeGraph

Manages the world state using a NetworkX MultiDiGraph. Features:
- **Rich Entity Attributes:** Description, status, status history, emotion tracking, temporal metadata.
- **Importance Scoring:** Three modes — `composite` (default), `incremental`, `degree_only`.
- **Layered Summary Generation:** `flat` (backward compatible) and `layered` (importance-ranked with descriptions).
- **Temporal Decay:** Relationship confidence decay per turn with configurable cadence.
- **Persistence:** Supports snapshots and auto-saving to `saves/`.

#### Key Methods
- `add_entity(name, entity_type, description, status, turn_id, is_player_mentioned, emotion)` — Upsert entity node.
- `add_relation(source, target, relation, context, turn_id, confidence)` — Add edge with rich attributes.
- `update_entity_state(name, state_updates, turn_id)` — Update entity status fields.
- `refresh_mentions(mentioned_names, turn_id, player_mentioned_names)` — Batch-update mention tracking.
- `apply_decay(turn_id)` — Reduce confidence of unconfirmed relations.
- `recalculate_importance()` — Recalculate importance scores (supports incremental mode).
- `to_summary(max_entities)` — Generate textual world-state summary (flat or layered).
- `get_timeline(n)` — Return recent events as timeline.
- `to_dict() / from_dict(data)` — Serialization/deserialization.
- `save(filepath) / load(filepath)` — File persistence.

### 7.2 RelationExtractor

LLM-powered extraction of entities and relations from text.

- `extract(text: str)` — Enhanced mode: extracts entities with description, status, state_changes + relations with context.
- `extract_dual(player_input, story_text, existing_entities)` — Dual extraction from both player input and story text in single LLM call.
- Module-level convenience functions: `extract()`, `extract_dual()`, `extract_legacy()`.

### 7.3 ConflictDetector

Hybrid rule-based and LLM-based consistency checking with multi-strategy resolution.

**Detection Layers:**
1. **Rule-based:** Exclusive relation pairs (ally_of↔enemy_of, alive↔dead), dead-active detection.
2. **Temporal:** Post-death actions, causal inversion.
3. **LLM:** Logical contradiction detection via LLM analysis.

**Resolution Strategies:**
- `keep_latest` — Deterministic: keeps newer information, removes older conflicting data.
- `llm_arbitrate` — Deterministic-first pass, then LLM arbitration for remaining conflicts.

---

## 8. Evaluation API

### 8.1 Automatic Metrics

| Metric | Description |
|--------|-------------|
| `distinct_n` | Measures n-gram diversity (Distinct-1, 2, 3). |
| `self_bleu` | Measures narrative redundancy (lower = more diverse). |
| `entity_coverage` | Ratio of KG entities mentioned in text. |
| `consistency_rate` | Ratio of conflict-free turns. |
| `type_token_ratio` | Vocabulary richness metric. |
| `flesch_reading_ease` | Readability score (English heuristic). |
| `lexical_overlap` | Adjacent turn lexical similarity. |
| `graph_density_evolution` | KG density trend over turns. |

### 8.2 LLM Judge

Evaluates full story sessions on **8 dimensions** (1-10 scale):

| Dimension | Description |
|-----------|-------------|
| `narrative_quality` | Prose quality, vivid descriptions, engaging language. |
| `consistency` | Characters, locations, and facts remain coherent. |
| `player_agency` | How meaningfully player choices affected the story. |
| `creativity` | Originality of plot, settings, and characters. |
| `pacing` | Appropriate story momentum and tension management. |
| `option_relevance` | How well offered options align with current context. |
| `causal_link` | Whether player actions cause believable state changes. |
| `local_coherence` | Continuity between adjacent turns. |

Uses separate evaluation LLM config (`EVAL_LLM_*` settings, default: `glm-5` via Zhipu API).

---

## 9. Configuration

Configured via `config.py` (Pydantic Settings) and `.env`.

### LLM API Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPENAI_API_KEY` | `""` | API key for local/OpenAI-compatible endpoint. |
| `OPENAI_BASE_URL` | `""` | Base URL for OpenAI-compatible API. |
| `OPENAI_MODEL` | `mimo-v2-flash` | Model name for NLG generation. |
| `OPENAI_MAX_TOKENS` | `1024` | Maximum generation tokens. |
| `OPENAI_TEMPERATURE` | `0.85` | Temperature for generation. |
| `OPENAI_TIMEOUT_CONNECT` | `10.0` | Connection timeout (seconds). |
| `OPENAI_TIMEOUT_READ` | `60.0` | Read timeout (seconds). |

### Evaluation LLM Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVAL_LLM_API_KEY` | `""` | Separate API key for LLM judge. |
| `EVAL_LLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | Evaluation LLM endpoint. |
| `EVAL_LLM_MODEL` | `glm-5` | Model for evaluation. |
| `EVAL_LLM_MAX_TOKENS` | `256` | Max tokens for evaluation. |
| `EVAL_LLM_TEMPERATURE` | `0.3` | Temperature for evaluation. |

### NLU Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INTENT_MODEL_NAME` | `distilbert-base-uncased` | Base model name. |
| `INTENT_MODEL_PATH` | `models/intent_classifier` | Fine-tuned model path. |
| `INTENT_MAX_LENGTH` | `128` | Max token length. |
| `INTENT_LABELS` | 8 labels | `action, dialogue, explore, use_item, ask_info, rest, trade, other` |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy NER model. |

### NLG Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NLG_MODE` | `hybrid` | `api`, `local`, or `hybrid`. |
| `NUM_OPTIONS` | `3` | Number of player options per turn. |

### Knowledge Graph Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KG_MAX_NODES` | `200` | Maximum nodes in KG. |
| `KG_ENTITY_TYPES` | 5 types | `person, location, item, creature, event` |
| `KG_RELATION_TYPES` | 12 types | `located_at, possesses, ally_of, enemy_of, knows, part_of, caused_by, has_attribute, causes, prevents, enables, follows` |
| `KG_CONFLICT_RESOLUTION` | `llm_arbitrate` | `keep_latest` or `llm_arbitrate`. |
| `KG_EXTRACTION_MODE` | `dual_extract` | `story_only` or `dual_extract`. |
| `KG_IMPORTANCE_MODE` | `composite` | `degree_only`, `composite`, or `incremental`. |
| `KG_SUMMARY_MODE` | `layered` | `flat` or `layered`. |
| `KG_IMPORTANCE_DECAY_FACTOR` | `0.95` | Per-turn importance decay. |
| `KG_RELATION_DECAY_FACTOR` | `0.90` | Relation confidence decay. |
| `KG_RELATION_MIN_CONFIDENCE` | `0.2` | Minimum relation confidence threshold. |
| `KG_IMPORTANCE_MENTION_BOOST` | `0.15` | Mention importance boost. |
| `KG_IMPORTANCE_PLAYER_BOOST` | `0.3` | Player mention extra boost. |
| `KG_MAX_TIMELINE_ENTRIES` | `5` | Maximum timeline entries. |
| `KG_DECAY_CADENCE` | `1` | Every N turns apply decay. |
| `KG_INCREMENTAL_FULL_RECALC_INTERVAL` | `10` | Full recalc interval for incremental mode. |
| `KG_ENABLE_INCREMENTAL_IMPORTANCE` | `True` | Enable incremental importance. |
| `KG_ENABLE_SUMMARY_CACHE` | `True` | Enable per-turn KG summary caching. |

### Persistence Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KG_SAVE_DIR` | `saves/` | Save directory. |
| `KG_AUTO_SAVE` | `True` | Auto-save enabled. |
| `KG_SNAPSHOT_INTERVAL` | `5` | Snapshot every N turns. |

### Game Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NARRATIVE_HISTORY_WINDOW` | `6` | Recent history entries for LLM context. |
| `MAX_CONTEXT_TOKENS` | `512` | Maximum context tokens. |
| `STREAMLIT_PORT` | `7860` | Streamlit server port. |

---

## 10. Error Handling

- **LLM API Failure:** Automatic retry with exponential backoff (up to 3 attempts). UI displays error toast.
- **NLU Model Missing:** Graceful fallback to rule-based matching for intent, entity, coref, and sentiment.
- **KG Conflict:** Logged in `TurnResult.conflicts` without blocking the turn. Resolved based on configured strategy.
- **JSON Parse Failure:** Multi-stage repair (markdown fence stripping, balanced JSON extraction, trailing comma removal, strict retry).
- **Transformers Version Warning:** Logs warning if transformers >= 4.50 (tested range: 4.40–4.49).

---

## 11. Examples

Refer to `scripts/test_openai_api.py` for API connectivity testing or `tests/integration/test_integration.py` for full pipeline integration tests.

### Quick Start

```python
from src.engine.game_engine import GameEngine

# Create engine with lazy NLU loading (default)
engine = GameEngine(genre="fantasy")

# Start a new game
result = engine.start_game()
print(result.story_text)
print([opt.text for opt in result.options])

# Process a player turn
result = engine.process_turn("I draw my sword and attack the dragon")
print(result.story_text)
print(f"Intent: {result.nlu_debug['intent']}, Emotion: {result.nlu_debug['emotion']}")
print(f"KG: {result.kg_node_count} nodes, {result.kg_edge_count} edges")
print(f"Conflicts: {result.conflicts}")
```

