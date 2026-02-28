# StoryWeaver: AI-Powered Text Adventure Game with Dynamic Plot Generation

## COMP5423 NLP Group Project

An interactive text adventure game engine that combines **local NLU models** with **LLM-powered story generation** and a **dynamic knowledge graph** for narrative consistency.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Frontend                          │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │ Chat UI  │  │ NLU Debug    │  │ Knowledge Graph Visualizer ││
│  └──────────┘  └──────────────┘  └────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Game Engine (Orchestrator)                   │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────┐ │
│  │ NLU (local) │→ │  Game    │→ │ Story Gen    │→ │ Option  │ │
│  │ RoBERTa +   │  │  State   │  │ (OpenAI API) │  │   Gen   │ │
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
2. **Intent Classification** — RoBERTa fine-tuned classifier (with keyword fallback)
3. **Entity Extraction** — spaCy NER + noun-phrase heuristics
4. **Story Generation** — OpenAI gpt-4o-mini continues the narrative
5. **KG Update** — LLM extracts entities & relations into a NetworkX graph
6. **Conflict Detection** — Rule-based + LLM consistency checking
7. **Option Generation** — LLM generates 3 player choices with risk levels

### Project Structure

```
story_maker/
├── app.py                    # Gradio application entry point
├── config.py                 # Pydantic Settings with .env support
├── requirements.txt          # Dependencies
├── src/
│   ├── engine/
│   │   ├── game_engine.py    # Main orchestrator (NLU → NLG → KG pipeline)
│   │   └── state.py          # Game state & history tracking
│   ├── nlu/
│   │   ├── intent_classifier.py   # RoBERTa + keyword fallback
│   │   ├── entity_extractor.py    # spaCy NER + regex
│   │   └── coreference.py         # fastcoref + rule-based
│   ├── nlg/
│   │   ├── story_generator.py     # OpenAI API story generation
│   │   ├── option_generator.py    # Player choice generation (API)
│   │   └── prompt_templates.py    # Prompt engineering templates
│   ├── knowledge_graph/
│   │   ├── graph.py               # NetworkX MultiDiGraph wrapper
│   │   ├── relation_extractor.py  # LLM-based relation extraction
│   │   ├── conflict_detector.py   # Rule-based + LLM consistency
│   │   └── visualizer.py          # PyVis HTML visualization
│   ├── evaluation/
│   │   ├── metrics.py             # Distinct-n, Self-BLEU, coverage
│   │   ├── llm_judge.py           # LLM-as-Judge scoring
│   │   └── consistency_eval.py    # KG consistency evaluation
│   └── utils/
│       └── api_client.py          # Singleton LLM client with retry
├── data/
│   ├── intent_labels.json         # Intent label definitions
│   └── scripts/
│       ├── download_data.py       # Dataset download scripts
│       └── preprocess.py          # Data preprocessing pipeline
├── training/
│   ├── train_intent.py            # RoBERTa intent classifier training
│   └── train_generator.py         # GPT-2 LoRA fine-tuning (legacy/optional)
├── tests/
│   ├── test_nlu.py
│   ├── test_nlg.py
│   ├── test_knowledge_graph.py
│   └── test_integration.py
└── info/                          # Project documentation
```

### Deployment & Startup (Windows / macOS)

Use the one-click script that matches your OS:

- Windows: `start_project.bat`
- macOS/Linux: `start_project.sh`

Both scripts perform the same bootstrap and launch sequence:

1. Creates `.venv` automatically (if missing)
2. Upgrades `pip`
3. Installs `requirements.txt`
4. Applies runtime compatibility fix: `gradio<6` (required by current UI code)
5. Tries downloading `en_core_web_sm` (if timeout occurs, app still starts with fallback extraction)
6. Creates `.env` from `.env.example` when missing
7. Checks running instances on `7860/7861` and avoids duplicate launches
8. Starts app via `python app.py` (default URL: `http://127.0.0.1:7860` or fallback `7861`)

#### Script usage record

- **First run (fresh machine/project clone):**
  - Windows: open PowerShell in project root, run `./start_project.bat`
  - macOS/Linux: run `chmod +x ./start_project.sh` once, then `./start_project.sh`
  - Wait until you see the Gradio startup URL in console

- **Subsequent runs:**
  - Run the same command for your OS
  - Existing `.venv` is reused and dependencies are checked/updated
  - If an instance is already running on `7860` or `7861`, the script will print the URL and exit (no duplicate process)

- **Force restart mode (stop old instance, then launch new one):**
  - Windows: `./start_project.bat --force-restart` (or `-f`)
  - macOS/Linux: `./start_project.sh --force-restart` (or `-f`)

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

- **NLU**: RoBERTa (intent classification) + spaCy (NER) + fastcoref (coreference resolution)
- **NLG**: OpenAI gpt-4o-mini via API (story generation, option generation, relation extraction)
- **Knowledge Graph**: NetworkX MultiDiGraph + PyVis visualization
- **Frontend**: Gradio Blocks (chat UI + KG panel + NLU debug)
- **Evaluation**: Distinct-n, Self-BLEU, Entity Coverage, Consistency Rate, LLM-as-Judge
