# StoryWeaver: AI-Powered Text Adventure Game with Dynamic Plot Generation

**[English](README.md) | [中文](README_zh.md)**

## COMP5423 NLP Group Project

An interactive text adventure game engine that combines **local NLU models** with **LLM-powered story generation** and a **dynamic knowledge graph** for narrative consistency.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Pipeline (per turn)](#pipeline-per-turn)
- [Project Structure](#project-structure)
- [Deployment & Startup](#deployment--startup)
- [NLU Module Status](#nlu-module-status)
- [Tech Stack](#tech-stack)
- [Documentation Index](#documentation-index)

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit Frontend                        │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │ Chat UI  │  │ NLU Debug    │  │ Knowledge Graph Visualizer ││
│  └──────────┘  └──────────────┘  └────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Game Engine (Orchestrator)                   │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ NLU (local) │→ │  Game    │→ │ Story Gen    │→ │ Option  │ │
│  │ DistilBERT+ │  │  State   │  │ (OpenAI API) │  │   Gen   │ │
│  │ spaCy +     │  │          │  │              │  │  (API)  │ │
│  │ fastcoref   │  └────┬─────┘  └──────────────┘  └─────────┘ │
│  └─────────────┘       │                                       │
│              ┌─────────▼───────┐                                │
│              │ Knowledge Graph │← Relation Extraction (API)     │
│              │ + Conflict Det. │                                 │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline (per turn)

1. **Coreference Resolution** — fastcoref resolves pronouns using recent history.
2. **Intent Classification** — DistilBERT fine-tuned classifier (with keyword fallback).
3. **Entity Extraction** — spaCy NER + noun-phrase heuristics.
4. **Story Generation** — LLM API continues the narrative.
5. **KG Update** — LLM extracts entities & relations into a NetworkX graph.
6. **Conflict Detection** — Rule-based + LLM consistency checking.
7. **Option Generation** — LLM generates 3 player choices with risk levels.

### Project Structure

```
story_maker/
├── app.py                    # Streamlit application entry point
├── config.py                 # Pydantic Settings with .env support
├── requirements.txt          # Dependencies
├── config/
│   └── .env.example          # Environment configuration template
├── src/                      # Source code modules
│   ├── engine/               # Game engine orchestrator
│   ├── nlu/                  # Natural Language Understanding (Intent, Entity, Coref)
│   ├── nlg/                  # Natural Language Generation (Story, Options)
│   ├── knowledge_graph/      # Dynamic world state management
│   ├── evaluation/           # Quality assessment metrics
│   ├── ui/                   # Streamlit UI components
│   └── utils/                # Shared utilities
├── training/                 # Model training scripts (Intent, Qwen, Llama)
├── tests/                    # Test suite (Engine, NLU, NLG, KG, Evaluation)
├── scripts/                  # Utility and deployment scripts
├── docs/                     # Comprehensive documentation
├── models/                   # Trained model artifacts (git-ignored)
├── lib/                      # Third-party frontend libraries
├── logs/                     # Application logs (git-ignored)
├── saves/                    # Game save files (git-ignored)
└── .env                      # Environment variables (git-ignored)
```

### Deployment & Startup

Use the startup script that matches your OS for production deployment:

- **Windows**: `scripts/start_project_prod.bat`
- **macOS/Linux**: `scripts/start_project_prod.sh`

#### Bootstrap Sequence
1. Detect existing StoryWeaver process on port `7860`.
2. Create `.venv` virtual environment if missing.
3. Install dependencies from `requirements.txt`.
4. Start Streamlit app (Default: `http://127.0.0.1:7860`).

### NLU Module Status

| Module | Backend | Status |
|--------|---------|--------|
| Intent | DistilBERT | ✅ Active |
| Entity | spaCy (en_core_web_sm) | ✅ Active |
| Coref | fastcoref FCoref | ✅ Active |

### Tech Stack

- **NLU**: DistilBERT + spaCy + fastcoref
- **NLG**: OpenAI GPT-4o-mini (API) / Local Qwen (llama.cpp)
- **Knowledge Graph**: NetworkX + PyVis
- **Frontend**: Streamlit
- **Evaluation**: Distinct-n, Self-BLEU, LLM-as-Judge

### Documentation Index

1. **[Technical Route](docs/guides/technical-route.md)** - NLU/KG/NLG strategy
2. **[Data Flow](docs/guides/data-flow.md)** - Field-level data mapping
3. **[Deployment Guide](docs/guides/zero-to-hero-deployment.md)** - Full setup guide
4. **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
5. **[Optimization Report](docs/reports/kg-optimization.md)** - KG enhancement details
6. **[Persistence Doc](docs/reports/runtime-persistence.md)** - Session persistence
