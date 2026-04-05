# StoryWeaver: AI-Powered Text Adventure Game with Dynamic Plot Generation

> **Last Updated**: 2026-04-05

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
  - [Guides](#documentation-index)
  - [Design & Architecture](#documentation-index)
  - [API & Reports](#documentation-index)

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
├── app.py                          # Streamlit application entry point
├── config.py                       # Pydantic Settings with .env support
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (git-ignored)
├── .env.llama                      # llama.cpp server configuration
├── .env.vllm                       # vLLM GPU inference configuration
├── .env.vllm.cpu                   # vLLM CPU inference configuration
├── .env.vllm.example               # vLLM configuration template
├── start_project_prod.bat          # Windows production launcher (root shortcut)
├── start_project_prod.sh           # macOS/Linux production launcher (root shortcut)
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── engine/                     # Game engine orchestrator
│   │   ├── __init__.py
│   │   ├── game_engine.py          # Main pipeline coordinator (NLU → NLG → KG)
│   │   ├── runtime_session.py      # Session persistence manager
│   │   ├── state.py                # Game state & history tracking
│   │   └── naming.py               # Character/location naming system
│   ├── nlu/                        # Natural Language Understanding
│   │   ├── __init__.py
│   │   ├── intent_classifier.py    # DistilBERT + keyword fallback
│   │   ├── entity_extractor.py     # spaCy NER + regex patterns
│   │   ├── coreference.py          # fastcoref + rule-based resolution
│   │   └── sentiment_analyzer.py   # Sentiment/tone analysis (Ekman 6-class)
│   ├── nlg/                        # Natural Language Generation
│   │   ├── __init__.py
│   │   ├── story_generator.py      # OpenAI API story generation
│   │   ├── option_generator.py     # Player choice generation (API)
│   │   └── prompt_templates.py     # Prompt engineering templates
│   ├── knowledge_graph/            # Dynamic world state management
│   │   ├── __init__.py
│   │   ├── graph.py                # NetworkX MultiDiGraph wrapper
│   │   ├── relation_extractor.py   # LLM-based relation extraction
│   │   ├── conflict_detector.py    # Rule + LLM consistency checking
│   │   └── visualizer.py           # PyVis HTML visualization
│   ├── evaluation/                 # Quality assessment metrics
│   │   ├── __init__.py
│   │   ├── metrics.py              # Distinct-n, Self-BLEU, coverage
│   │   ├── llm_judge.py            # LLM-as-judge scoring
│   │   └── consistency_eval.py     # Knowledge graph consistency
│   ├── ui/                         # Streamlit UI components
│   │   ├── __init__.py
│   │   ├── layout/                 # Page layout & theme
│   │   │   ├── __init__.py
│   │   │   └── theme.py
│   │   ├── sections/               # UI section modules
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   ├── evaluation.py
│   │   │   └── sidebar.py
│   │   └── state_manager.py        # UI state management
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       └── api_client.py           # Singleton LLM client (with retry)
│
├── scripts/                        # Utility and deployment scripts
│   ├── start/                      # Startup scripts
│   │   ├── start_project_prod.bat  # Windows production launcher
│   │   ├── start_project_prod.sh   # macOS/Linux production launcher
│   │   ├── start_llama_server.bat  # llama.cpp server launcher
│   │   ├── start_inference_server.sh
│   │   └── start_streamlit.sh
│   ├── config/                     # Environment config templates
│   │   ├── .env.llama
│   │   ├── .env.vllm
│   │   ├── .env.vllm.cpu
│   │   └── .env.vllm.example
│   ├── data/                       # Dataset generation tools
│   │   ├── generate_dataset.py
│   │   ├── extract_pdfs.py
│   │   ├── read_pdfs.py
│   │   ├── fix_and_merge.py
│   │   └── validate_and_merge.py
│   ├── eval/                       # Evaluation runners
│   │   ├── run_automated_eval.py
│   │   ├── run_eval_benchmark.py
│   │   ├── run_kg_on_off_benchmark.py
│   │   ├── run_llm_judge.py
│   │   └── simple_model_eval.py
│   ├── inference/                  # Inference utilities
│   │   ├── local_inference_server.py
│   │   └── test_openai_api.py
│   └── quantize/                   # Model quantization
│       └── quantize_gguf.bat
│
├── training/                       # Model training scripts
│   ├── train_intent.py             # DistilBERT intent classifier
│   ├── train_generator.py          # GPT-2 LoRA fine-tuning (legacy)
│   ├── train_llama.sh              # Llama.cpp training script
│   ├── train_qwen.sh               # Qwen training script
│   ├── data_augmenter.py           # Training data augmentation
│   └── nlg_dataset/                # NLG training dataset
│       ├── combined_data.jsonl
│       └── combined_data_generate_prompt.md
│
├── tests/                          # Test suite (organized by module)
│   ├── __init__.py
│   ├── engine/                     # Engine component tests
│   ├── nlu/                        # NLU module tests
│   ├── nlg/                        # NLG module tests
│   ├── kg/                         # Knowledge graph tests
│   ├── integration/                # Cross-module integration tests
│   ├── evaluation/                 # Quality evaluation tests
│   ├── performance/                # Performance benchmark tests
│   ├── ui/                         # UI component tests
│   └── utils/                      # Utility function tests
│
├── docs/                           # Comprehensive documentation
│   ├── README.md                   # Documentation index
│   ├── api/                        # API reference docs
│   │   ├── README.md
│   │   └── API_REFERENCE.md
│   ├── design/                     # Architecture & design docs
│   │   ├── README.md
│   │   ├── prompts/                # Prompt templates
│   │   ├── conflict-detection-resolution.md
│   │   ├── entity-importance.md
│   │   ├── hybrid-nlg-architecture.md
│   │   ├── implementation_plan.md
│   │   ├── kg-summary-modes.md
│   │   ├── nlg-local-model-finetuning.md
│   │   ├── sentiment-analysis.md
│   │   └── storyweaver_pipeline.*  # Pipeline diagrams (drawio/svg/html)
│   ├── guides/                     # Deployment & usage guides
│   │   ├── README.md
│   │   ├── CPU_INFERENCE.md
│   │   ├── data-flow.md
│   │   ├── deployment-macos.md
│   │   ├── deployment-windows.md
│   │   ├── local-model-startup.md
│   │   ├── technical-route.md
│   │   └── zero-to-hero-deployment.md
│   ├── fixes/                      # Bug fix reports
│   │   ├── README.md
│   │   ├── distilbert-compatibility-fix.md
│   │   ├── distilbert-tokenizer-fix.md
│   │   ├── distilbert-troubleshooting.md
│   │   ├── fastcoref-fix.md
│   │   └── llm-json-truncation-fix.md
│   ├── reports/                    # Optimization & evaluation reports
│   │   ├── README.md
│   │   ├── changelog/              # Auto-generated changelogs
│   │   ├── evaluation/             # Model evaluation results
│   │   ├── local-model/            # Local model reports
│   │   ├── optimization/           # Optimization reports
│   │   └── test-results/           # Test run results
│   ├── project/                    # Project specs & materials
│   │   ├── COMP5423 NLP Group Project Specification-2026.pdf
│   │   └── project intro.pdf
│   └── final_submit/               # Final submission materials
│       └── final_report/
│           └── Final_Project_Report.md
│
├── models/                         # Trained model artifacts (git-ignored)
│   ├── intent_classifier/          # Fine-tuned DistilBERT checkpoints
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── checkpoint-*/           # Training checkpoints
│   └── nlg/                        # NLG model checkpoints
│       └── README.md
│
├── lib/                            # Third-party frontend libraries
│   ├── vis-9.1.2/                  # Vis.js network visualization
│   │   ├── vis-network.min.js
│   │   └── vis-network.css
│   ├── tom-select/                 # Enhanced select component
│   │   ├── tom-select.complete.min.js
│   │   └── tom-select.css
│   └── bindings/                   # JavaScript utilities
│       └── utils.js
│
├── reports/                        # Standalone evaluation reports
│   ├── comparison/                 # Model comparison reports
│   │   └── model-comparison.md
│   ├── evaluation/                 # Evaluation results
│   │   ├── automated_eval_report.md
│   │   ├── local-model-eval.md
│   │   └── mimo_eval_report.md
│   └── hybrid/                     # Hybrid strategy reports
│       ├── hybrid_eval_report.md
│       ├── hybrid_strategy_guide.md
│       └── hybrid_vs_standalone_comparison.md
│
├── saves/                          # Game save files (git-ignored)
│   ├── runtime_engine.json         # Runtime engine state
│   ├── runtime_session.json        # Session persistence
│   └── *.json                      # Individual game saves
│
├── config/                         # Configuration templates
│   └── .env.example                # Environment configuration template
│
├── logs/                           # Application logs (git-ignored)
└── .gitignore                      # Git ignore rules
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

#### Guides
1. **[Technical Route](docs/guides/technical-route.md)** - NLU/KG/NLG strategy & fallback policies
2. **[Data Flow](docs/guides/data-flow.md)** - Turn-by-turn field-level data mapping
3. **[Zero-to-Hero Deployment](docs/guides/zero-to-hero-deployment.md)** - Complete setup guide
4. **[Windows Deployment](docs/guides/deployment-windows.md)** - Windows HA deployment guide
5. **[macOS Deployment](docs/guides/deployment-macos.md)** - macOS HA deployment guide
6. **[CPU Inference](docs/guides/CPU_INFERENCE.md)** - CPU inference optimization
7. **[Local Model Startup](docs/guides/local-model-startup.md)** - Local model startup guide

#### Design & Architecture
8. **[Entity Importance](docs/design/entity-importance.md)** - Entity importance scoring
9. **[Hybrid NLG Architecture](docs/design/hybrid-nlg-architecture.md)** - Hybrid NLG design
10. **[NLG Local Model Fine-tuning](docs/design/nlg-local-model-finetuning.md)** - Local LLM fine-tuning plan
11. **[KG Summary Modes](docs/design/kg-summary-modes.md)** - Knowledge graph summary modes
12. **[Sentiment Analysis](docs/design/sentiment-analysis.md)** - Sentiment/tone analysis strategy
13. **[Conflict Detection](docs/design/conflict-detection-resolution.md)** - Conflict detection & resolution

#### API & Reports
14. **[API Reference](docs/api/API_REFERENCE.md)** - Complete API documentation
15. **[KG Optimization](docs/reports/optimization/kg-optimization.md)** - Knowledge graph enhancement
16. **[NLU & KG Improvement](docs/reports/optimization/nlu-kg-improvement.md)** - NLU & KG module improvements
17. **[Runtime Persistence](docs/reports/optimization/runtime-persistence.md)** - Session persistence docs
18. **[Evaluation Reports](docs/reports/evaluation/)** - Model evaluation results (API, Local, Hybrid)
19. **[Test Results](docs/reports/test-results/)** - Automated test & KG on/off benchmark results

#### Other
20. **[Fix Reports](docs/fixes/)** - Bug fix documentation (DistilBERT, fastcoref, LLM JSON)
21. **[Changelog](docs/reports/changelog/)** - Auto-generated update changelogs
