# StoryWeaver: AI-Powered Text Adventure Game with Dynamic Plot Generation

## COMP5423 NLP Group Project

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Frontend                          │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │ Chat UI  │  │ Status Panel │  │ Knowledge Graph Visualizer ││
│  └──────────┘  └──────────────┘  └────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Game Engine (Orchestrator)                   │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │   NLU   │→ │ Dialog State │→ │ Story Gen    │→ │ Options │ │
│  │ Module  │  │   Tracker    │  │   (NLG)      │  │   Gen   │ │
│  └─────────┘  └──────┬───────┘  └──────────────┘  └─────────┘ │
│                       │                                         │
│              ┌────────▼────────┐                                │
│              │ Knowledge Graph │                                │
│              │   (Consistency) │                                │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
story_maker/
├── app.py                    # Gradio application entry point
├── requirements.txt          # Dependencies
├── config.py                 # Global configurations
├── src/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_engine.py    # Main orchestrator
│   │   └── state.py          # Game state dataclass
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py   # RoBERTa intent classification
│   │   ├── entity_extractor.py    # spaCy + custom NER
│   │   └── coreference.py         # Coreference resolution
│   ├── nlg/
│   │   ├── __init__.py
│   │   ├── story_generator.py     # GPT-2 story generation
│   │   ├── option_generator.py    # Player choice generation
│   │   └── prompt_templates.py    # Prompt engineering
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── graph.py               # KG construction & update
│   │   ├── conflict_detector.py   # Consistency checking
│   │   ├── relation_extractor.py  # Relation extraction
│   │   └── visualizer.py          # PyVis visualization
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py             # Auto metrics (BERTScore, etc.)
│       └── consistency_eval.py    # KG-based consistency eval
├── data/
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Cleaned datasets
│   └── scripts/
│       ├── download_data.py       # Dataset download scripts
│       └── preprocess.py          # Data cleaning pipeline
├── training/
│   ├── train_intent.py            # Intent classifier training
│   ├── train_generator.py         # GPT-2 fine-tuning (LoRA)
│   └── configs/                   # Training hyperparameters
├── tests/
│   ├── test_nlu.py
│   ├── test_nlg.py
│   ├── test_knowledge_graph.py
│   └── test_integration.py
└── info/                          # Project specs
```

### Quick Start

```bash
pip install -r requirements.txt
python app.py
```

### Tech Stack

- **NLU**: RoBERTa (intent) + spaCy (NER) + fastcoref (coreference)
- **NLG**: GPT-2 Medium (LoRA fine-tuned) + Constrained Decoding
- **Knowledge Graph**: NetworkX + PyVis
- **Frontend**: Gradio Blocks
- **Evaluation**: BERTScore, Distinct-n, Self-BLEU, human evaluation
