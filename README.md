# StoryWeaver: AI Text Adventure Engine (NLU + LLM + Knowledge Graph)

> **Last Updated**: 2026-04-24

**[English](README.md) | [中文](README_zh.md)**

StoryWeaver is a Streamlit-based interactive text adventure engine that combines local NLU, LLM generation, and a dynamic knowledge graph to keep narrative state consistent across turns.

## Highlights

- Hybrid NLG routing (`NLG_MODE=local|api|hybrid`) for quality/cost/latency trade-offs.
- Runtime session persistence (`runtime_session.json`) with active-state restore logic.
- Knowledge graph extraction + conflict detection + layered summary mode.
- Evaluation toolkit: automated metrics, KG on/off benchmark, LLM-as-judge, tri-config comparison docs.

## Quick Start

### 1) Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
```

### 2) Configure `.env`

At minimum:

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=mimo-v2-flash
```

Optional key switches used in current codebase:

- `NLG_MODE` (`hybrid` by default)
- `MIMO_API_KEY`, `MIMO_BASE_URL`, `MIMO_MODEL`
- `EVAL_LLM_API_KEY`, `EVAL_LLM_BASE_URL`, `EVAL_LLM_MODEL`
- `KG_SUMMARY_MODE`, `KG_EXTRACTION_MODE`, `KG_IMPORTANCE_MODE`

### 3) Launch

- Windows: `scripts/start/start_project_prod.bat`
- macOS/Linux: `scripts/start/start_project_prod.sh`

Root-level `start_project_prod.bat/.sh` are convenience wrappers/copies.

## Turn Pipeline

1. Coreference resolution (fastcoref when available; rule fallback otherwise)
2. Intent classification (DistilBERT + fallback)
3. Entity extraction (spaCy + heuristics)
4. Story continuation generation
5. Relation extraction and KG update
6. Conflict detection and state reconciliation
7. Option generation (3 choices + risk labels)

## Project Structure

```text
story_maker/
├── app.py
├── config.py
├── src/
│   ├── engine/              # game loop, runtime_session, state
│   ├── nlu/                 # intent/entity/coreference/sentiment
│   ├── nlg/                 # story/option generation + prompts
│   ├── knowledge_graph/     # graph/relation/conflict/visualizer
│   ├── evaluation/          # metrics + LLM judge
│   ├── ui/                  # Streamlit sections + state management
│   └── utils/
├── scripts/
│   ├── start/               # production/startup scripts
│   ├── eval/                # evaluation runners
│   ├── inference/           # local inference utilities
│   ├── data/                # dataset utilities
│   └── quantize/
├── training/
├── tests/
├── docs/
├── reports/                 # standalone comparison/eval/hybrid reports
├── models/
├── config/
└── lib/
```

## Testing & Evaluation

```bash
# unit/integration
pytest tests/ -v

# automated evaluation
python scripts/eval/run_automated_eval.py

# KG on/off benchmark
python scripts/eval/run_kg_on_off_benchmark.py
```

## Documentation Index

- Main docs hub: [docs/README.md](docs/README.md)
- Guides: [docs/guides/README.md](docs/guides/README.md)
- API reference: [docs/api/API_REFERENCE.md](docs/api/API_REFERENCE.md)
- Design docs: [docs/design/README.md](docs/design/README.md)
- Fixes/troubleshooting: [docs/fixes/README.md](docs/fixes/README.md)
- Reports index: [docs/reports/README.md](docs/reports/README.md)

## Recent Additions (from Git History)

- Tri-config evaluation documentation (local vs API vs hybrid).
- Runtime restore robustness and active-session lifecycle handling.
- Hybrid NLG routing and evaluation model decoupling.
- Enhanced docs/report organization and deployment script cleanup.

## Tech Stack

- **Frontend**: Streamlit
- **NLU**: DistilBERT, spaCy, fastcoref (with fallback)
- **NLG**: OpenAI-compatible APIs + local model path
- **Knowledge Graph**: NetworkX + PyVis
- **Evaluation**: Distinct-n, Self-BLEU, LLM-as-judge
