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

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Configure API key:** Copy `.env.example` to `.env` and set your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_BASE_URL=https://api.openai.com/v1   # or compatible endpoint
   ```

3. **Run the app:**
   ```bash
   python app.py
   ```

4. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```

### Tech Stack

- **NLU**: RoBERTa (intent classification) + spaCy (NER) + fastcoref (coreference resolution)
- **NLG**: OpenAI gpt-4o-mini via API (story generation, option generation, relation extraction)
- **Knowledge Graph**: NetworkX MultiDiGraph + PyVis visualization
- **Frontend**: Gradio Blocks (chat UI + KG panel + NLU debug)
- **Evaluation**: Distinct-n, Self-BLEU, Entity Coverage, Consistency Rate, LLM-as-Judge
