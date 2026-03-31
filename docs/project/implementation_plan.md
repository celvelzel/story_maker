# StoryWeaver Hybrid Architecture Implementation Plan

> **Project**: COMP5423 NLP — Interactive Text Adventure Story Generator
> **Architecture**: Hybrid Approach (Local NLU + API NLG)
> **Last Update**: 2026-03-31

> **Note**: This document is the project-level implementation plan. Some paths and examples are design drafts. For current runtime instructions, refer to the root `README.md`.

---

## I. Architecture Overview

```
User Input
  │
  ▼
┌──────────────────────────────────────────────────┐
│  NLU Pipeline (Local)                             │
│  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Intent      │  │ Entity   │  │ Coreference │ │
│  │ Classifier  │  │ Extractor│  │ Resolver    │ │
│  │ (DistilBERT)│  │ (spaCy)  │  │ (fastcoref) │ │
│  └──────┬──────┘  └────┬─────┘  └──────┬──────┘ │
│         └───────┬──────┘───────────────┘         │
└─────────────────┼────────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│  Knowledge Graph (Local NetworkX)                 │
│  ┌───────────────┐  ┌──────────────────────────┐ │
│  │ Graph Manager  │  │ Conflict Detector       │ │
│  │ (MultiDiGraph) │  │ (Rules + LLM fallback)  │ │
│  └───────┬───────┘  └───────────┬──────────────┘ │
│          └──────────┬───────────┘                 │
└─────────────────────┼────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│  NLG Pipeline (API / Local Hybrid)                │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ Story        │  │ Option       │               │
│  │ Generator    │  │ Generator    │               │
│  │ (GPT/Qwen)   │  │ (GPT/Qwen)   │               │
│  └──────┬──────┘  └──────┬───────┘               │
│         └───────┬────────┘                        │
└─────────────────┼────────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│  Streamlit UI                                     │
│  Chat + Option Buttons + KG Visualization (PyVis) │
└──────────────────────────────────────────────────┘
```

---

## II. Directory Structure

```
story_maker/
├── .env                          # API keys (git-ignored)
├── config/
│   └── .env.example              # Example environment variables
├── .gitignore
├── config.py                     # Global configuration (Pydantic Settings)
├── app.py                        # Streamlit application entry point
├── requirements.txt              # Dependency list
├── README.md
│
├── docs/project/                 # Project documentation
│   ├── implementation_plan.md    # This file
│   ├── agent_prompt.md           # Agent prompts
│   └── *.pdf                     # Course project specs
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── api_client.py         # Unified LLM API client
│   │
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py  # DistilBERT intent classification
│   │   ├── entity_extractor.py   # spaCy NER + noun phrases
│   │   ├── sentiment_analyzer.py # Sentiment analysis
│   │   └── coreference.py        # fastcoref coreference resolution
│   │
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── graph.py              # NetworkX MultiDiGraph management
│   │   ├── relation_extractor.py # LLM-based relation extraction
│   │   ├── conflict_detector.py  # Rule-based + LLM conflict detection
│   │   └── visualizer.py         # PyVis visualization
│   │
│   ├── nlg/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py   # Versioned prompt templates
│   │   ├── story_generator.py    # Story continuation
│   │   └── option_generator.py   # Option generation
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_engine.py        # Game loop orchestration
│   │   ├── runtime_session.py    # Persistence and session management
│   │   └── state.py              # GameState data classes
│   │
│   ├── ui/                       # Streamlit UI modules
│   │   ├── layout/               # Theme and styling
│   │   └── sections/             # Chat, sidebar, evaluation UI
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py            # Distinct-n, Self-BLEU, etc.
│       └── llm_judge.py          # LLM-as-Judge evaluation
│
├── tests/                        # Comprehensive test suite
├── scripts/                      # Startup and utility scripts
├── training/                     # Fine-tuning scripts
└── models/                       # Local model artifacts
```

---

## III. Detailed Module Design

### 3.1 config.py — Global Configuration

**Key Features**:
- Uses `pydantic-settings` to load from `.env`
- Centralized management for API config, NLU models, KG limits, and game settings
- Supports `NLG_MODE` switching (local, api, hybrid)

---

### 3.2 src/utils/api_client.py — Unified LLM Client

**Key Features**:
- Singleton pattern for global access
- Exponential backoff retry logic (3 attempts)
- Support for JSON mode and structured outputs
- Usage tracking (tokens and estimated cost)

---

### 3.3 NLU Module

#### 3.3.1 intent_classifier.py
- Fine-tuned `distilbert-base-uncased`
- 8 intent labels: `action, dialogue, explore, use_item, ask_info, rest, trade, other`
- Keyword-based fallback for robustness

#### 3.3.2 entity_extractor.py
- `spaCy` NER (`en_core_web_sm`)
- Noun phrase heuristics to capture game-specific entities
- Entity types: `person, location, item, creature, event`

#### 3.3.3 coreference.py
- `fastcoref` for pronoun resolution
- Resolves context-dependent inputs (e.g., "Look at him" → "Look at the dragon")

---

### 3.4 Knowledge Graph Module

#### 3.4.1 graph.py — MultiDiGraph Management
- `NetworkX` implementation supporting multiple relations between entities
- Incremental importance scoring and decay mechanisms
- Summary generation for LLM context injection

#### 3.4.2 relation_extractor.py — LLM-based Extraction
- Extracts structured entities and relations from narrative text
- Supports dual-extraction (from both player input and story text)

#### 3.4.3 conflict_detector.py — Dual-layer Detection
- Rule-based detection (e.g., mutually exclusive relations like `ally_of` vs `enemy_of`)
- LLM-based logical reasoning for complex contradictions

---

### 3.5 NLG Module

#### 3.5.1 story_generator.py
- Handles opening scenes and narrative continuation
- Context-aware generation using history and KG summary

#### 3.5.2 option_generator.py
- Generates 3 branching choices per turn
- Assigns intent hints and risk levels (low/medium/high)

---

### 3.6 Evaluation Module

#### 3.6.1 Automated Metrics (metrics.py)
- `Distinct-n`: Diversity of generated text
- `Self-BLEU`: Repetition check
- `Entity Coverage`: KG integration depth
- `Consistency Rate`: Frequency of logical conflicts

#### 3.6.2 LLM-as-Judge (llm_judge.py)
- Multidimensional scoring: Narrative Quality, Consistency, Agency, Creativity, Pacing
- Provides qualitative feedback and overall rating

---

## IV. Implementation Timeline

| Week | Milestone | Deliverables |
|----|--------|--------|
| **W1** | Infrastructure | config.py, api_client.py, initial KG structure, testing framework |
| **W2** | Core Modules | NLU pipeline (DistilBERT/spaCy), KG extraction, NLG templates |
| **W3** | Integration | Game Engine orchestration, Streamlit UI, Conflict detection, Persistence |
| **W4** | Evaluation | LLM-as-Judge, Automated metrics, Final bug fixes, Documentation |

---

## V. Evaluation Standards

| Metric | Target |
|------|------|
| Distinct-2 | ≥ 0.75 |
| Self-BLEU-4 | ≤ 0.35 |
| Consistency Rate | ≥ 0.85 |
| Narrative Quality (Judge) | ≥ 7/10 |
| Intent Accuracy | ≥ 85% |

---

*End of Implementation Plan*
