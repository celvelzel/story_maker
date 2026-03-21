# StoryWeaver: AI-Powered Text Adventure Game with Dynamic Plot Generation

**[English](README.md) | [中文](README_zh.md)**

## COMP5423 NLP Group Project

An interactive text adventure game engine that combines **local NLU models** with **LLM-powered story generation** and a **dynamic knowledge graph** for narrative consistency.

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

1. **Coreference Resolution** — fastcoref resolves pronouns using recent history
2. **Intent Classification** — DistilBERT fine-tuned classifier (with keyword fallback)
3. **Entity Extraction** — spaCy NER + noun-phrase heuristics
4. **Story Generation** — LLM API continues the narrative
5. **KG Update** — LLM extracts entities & relations into a NetworkX graph
6. **Conflict Detection** — Rule-based + LLM consistency checking
7. **Option Generation** — LLM generates 3 player choices with risk levels

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
│   │   ├── game_engine.py    # Main pipeline coordinator (NLU → NLG → KG)
│   │   ├── runtime_session.py # Session persistence management
│   │   └── state.py          # Game state & history tracking
│   ├── nlu/                  # Natural Language Understanding
│   │   ├── intent_classifier.py   # DistilBERT + keyword fallback
│   │   ├── entity_extractor.py    # spaCy NER + regex patterns
│   │   ├── coreference.py         # fastcoref + rule-based resolution
│   │   └── sentiment_analyzer.py  # Emotion/tone analysis (Ekman 6-class)
│   ├── nlg/                  # Natural Language Generation
│   │   ├── story_generator.py     # OpenAI API story generation
│   │   ├── option_generator.py    # Player choice generation (API)
│   │   └── prompt_templates.py    # Prompt engineering templates
│   ├── knowledge_graph/      # Dynamic world state management
│   │   ├── graph.py               # NetworkX MultiDiGraph wrapper
│   │   ├── relation_extractor.py  # LLM-based relation extraction
│   │   ├── conflict_detector.py   # Rule-based + LLM consistency checking
│   │   └── visualizer.py          # PyVis HTML visualization
│   ├── evaluation/           # Quality assessment metrics
│   │   ├── metrics.py             # Distinct-n, Self-BLEU, coverage
│   │   ├── llm_judge.py           # LLM-as-Judge scoring
│   │   └── consistency_eval.py    # KG consistency evaluation
│   └── utils/                # Shared utilities
│       └── api_client.py          # Singleton LLM client with retry
├── data/                     # Data assets and processing
│   ├── intent_labels.json         # Intent label definitions
│   ├── raw/                       # Raw datasets (git-ignored)
│   └── scripts/                   # Data processing scripts
│       ├── download_data.py       # Dataset download automation
│       └── preprocess.py          # Data preprocessing pipeline
├── training/                 # Model training scripts
│   ├── train_intent.py            # DistilBERT intent classifier training
│   ├── train_generator.py         # GPT-2 LoRA fine-tuning (legacy/optional)
│   └── data_augmenter.py          # Training data augmentation
├── tests/                    # Test suite (organized by module)
│   ├── engine/              # Engine component tests
│   ├── nlu/                 # NLU module tests
│   ├── nlg/                 # NLG module tests
│   ├── kg/                  # Knowledge graph tests
│   ├── integration/         # Cross-module integration tests
│   └── training/            # Training pipeline tests
├── scripts/                  # Utility and deployment scripts
│   ├── start_project_prod.bat     # Windows production launcher
│   ├── start_project_prod.sh      # macOS/Linux production launcher
│   ├── health_check.py            # Pre-deployment health validation
│   └── generate_dataset.py        # Dataset generation utility
├── docs/                     # Comprehensive documentation
│   ├── api/                 # API reference documentation
│   ├── design/              # Architecture and design documents
│   ├── guides/              # Deployment and usage guides
│   ├── fixes/               # Issue resolution reports
│   ├── reports/             # Optimization and improvement reports
│   └── project/             # Project specifications and plans
├── models/                   # Trained model artifacts (git-ignored)
│   └── intent_classifier/   # Fine-tuned DistilBERT checkpoints
├── lib/                      # Third-party frontend libraries
│   ├── vis-9.1.2/           # Vis.js network visualization
│   ├── tom-select/          # Enhanced select component
│   └── bindings/            # JavaScript utility bindings
├── logs/                     # Application logs (git-ignored)
├── saves/                    # Game save files (git-ignored)
└── .env                      # Environment variables (git-ignored)
```

### Deployment & Startup (Windows / macOS)

Use the startup script that matches your OS for production deployment:

- Windows: `scripts/start_project_prod.bat`
- macOS/Linux: `scripts/start_project_prod.sh`

Production startup scripts are optimized and provide:

1. **Port occupancy detection and process identification** — Automatically detects existing Streamlit processes
2. **Safe restart policy** — Intelligently handles existing Streamlit app processes
3. **Dependency installation** — Installation with explicit network timeout controls
4. **Startup failure handling** — Structured exit codes for easy diagnostics
5. **Logging** — Complete log files with timestamps under `logs/` directory

#### Bootstrap and Launch Sequence

The scripts perform the following steps:

1. Detect existing StoryWeaver process on port `7860` (safely restart if found)
2. Create `.venv` virtual environment automatically (if missing)
3. Upgrade `pip` to latest version
4. Install all dependencies from `requirements.txt`
5. Start Streamlit app (default URL: `http://127.0.0.1:7860`)
6. Output complete logs to `logs/storyweaver_prod_<timestamp>.log`

#### Script Usage Guide

- **First run (fresh machine/project clone):**
  - Windows: open PowerShell in project root, run `./scripts/start_project_prod.bat`
  - macOS/Linux: run `chmod +x ./scripts/start_project_prod.sh` once, then `./scripts/start_project_prod.sh`
  - Wait until you see the Streamlit startup URL in console

- **Subsequent runs:**
  - Run the same command for your OS
  - Existing `.venv` is reused and dependencies are checked/updated
  - If a StoryWeaver instance is already running on `7860`, the script will automatically restart the process
  - Complete execution logs are saved to `logs/` directory for later inspection

#### Logging and Diagnostics

- Each startup generates a timestamped log file in `logs/` directory (format: `storyweaver_prod_YYYYMMDD_HHMMSS.log`)
- Logs record each step of the bootstrap process for easy troubleshooting
- If startup fails, check the log file for detailed error information

### Intent Model (CPU-Friendly Default)

Intent classification defaults to `distilbert-base-uncased` during training, while inference loads from local checkpoint directory:

- Default checkpoint directory: `models/intent_classifier`
- If model directory is missing or loading fails, system automatically falls back to `rule_fallback`
- API is unchanged: `predict(text) -> {"intent": str, "confidence": float}`

Recommended CPU settings:

1. `batch_size=8`
2. `max_length=128`
3. Typical intent inference latency: around 20-80ms/turn (depends on CPU and model size)
4. Expected memory for intent model: roughly 300-700MB including runtime overhead

### NLU Module Status

All 3 NLU components are **fully initialized**:

| Module | Backend | Status |
|--------|---------|--------|
| Intent | DistilBERT (distilbert-base-uncased) | ✅ Active |
| Entity | spaCy (en_core_web_sm) | ✅ Active |
| Coref | fastcoref FCoref | ✅ Active* |

\* **Fastcoref compatibility:** A transformers 5.2.0 compatibility patch is applied in `src/nlu/coreference.py` to enable fastcoref 2.x. See [FASTCOREF_PATCH.md](FASTCOREF_PATCH.md) for technical details.

**Verification:** Run `python verify_nlu_load.py` to confirm all modules load successfully.

- **API configuration (recommended):**
  - Edit `.env`
  - Set at least:
    - `OPENAI_API_KEY=sk-...`
    - `OPENAI_BASE_URL=https://api.openai.com/v1` (or your compatible endpoint)

#### Optional: run tests

After script bootstrap, you can run:

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

#### Troubleshooting

- **Port still occupied after script runs?**
  - The script first checks if StoryWeaver is already running on `7860/7861`; if yes, it reuses that instance.
  - If both ports are occupied by other programs, startup stops with an error.
  - Manual check (Windows): `netstat -ano | findstr :7860` or `:7861`, then `taskkill /PID <PID> /F`.
  - Manual check (macOS/Linux): `lsof -i :7860` or `:7861`, then `kill -9 <PID>`.
  
- **spaCy model download times out?**
  - This is expected on slower networks. The script will warn but continue — entity extraction falls back to regex patterns.
  - You can download manually later:
    - Windows: `.venv\Scripts\python.exe -m spacy download en_core_web_sm`
    - macOS/Linux: `.venv/bin/python -m spacy download en_core_web_sm`

- **`.env` is missing and API calls fail?**
  - Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
  - The script does this automatically if both are missing.

### Tech Stack

- **NLU**: DistilBERT (intent classification) + spaCy (NER) + fastcoref (coreference resolution)
- **NLG**: OpenAI gpt-4o-mini via API (story generation, option generation, relation extraction)
- **Knowledge Graph**: NetworkX MultiDiGraph + PyVis visualization
- **Frontend**: Streamlit (chat UI + KG panel + NLU debug)
- **Evaluation**: Distinct-n, Self-BLEU, Entity Coverage, Consistency Rate, LLM-as-Judge

### Intent Training Quick Start

Use the default DistilBERT base model and save to the standard checkpoint directory:

```bash
python training/train_intent.py --output_dir models/intent_classifier --model_name distilbert-base-uncased --epochs 6 --batch_size 8 --max_length 128
```

At runtime, if `models/intent_classifier` is unavailable, the system automatically falls back to keyword `rule_fallback` mode.

### Documentation Index

1. `docs/guides/technical-route.md` - NLU/KG/NLG technical route and fallback strategy
2. `docs/guides/data-flow.md` - Per-turn field-level data flow and module mapping
3. `docs/guides/deployment-windows.md` - Windows high-availability deployment guide
4. `docs/guides/deployment-macos.md` - macOS high-availability deployment guide
5. `docs/design/entity-importance.md` - Entity importance scoring strategy
6. `docs/design/nlg-local-model-finetuning.md` - NLG local model fine-tuning plan
7. `docs/reports/kg-optimization.md` - Knowledge graph optimization report
8. `docs/reports/nlu-kg-improvement.md` - NLU & KG improvement report
9. `docs/api/API_REFERENCE.md` - Complete API reference documentation
