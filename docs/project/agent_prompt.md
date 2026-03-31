# StoryWeaver Agent Prompt

```
You are a senior software engineer implementing "StoryWeaver" — an interactive text adventure game engine for an NLP course project (COMP5423). The architecture is a hybrid design: local NLU + API-based/Local NLG.

## Project Location
/hpc/puhome/25116696g/NLP/story_maker/

## Architecture
- NLU (local): DistilBERT intent classification, spaCy NER, fastcoref coreference resolution
- Knowledge Graph (local): NetworkX MultiDiGraph with entity/relation management and relation extraction
- NLG (API/Hybrid): OpenAI-compatible API (Mimo/Qwen) for story generation and option generation
- UI: Streamlit app with chat, option buttons, KG visualization (PyVis)
- Evaluation: Distinct-n, Self-BLEU, Consistency Rate, Entity Coverage, LLM-as-Judge

## Complete File Structure
```
story_maker/
├── .env                          # API keys (git-ignored)
├── config/
│   └── .env.example              # Template
├── .gitignore
├── config.py                     # Pydantic Settings, reads .env
├── app.py                        # Streamlit frontend entry
├── requirements.txt
├── README.md
├── src/
│   ├── utils/
│   │   └── api_client.py         # Singleton API wrapper, retry, cost tracking
│   ├── nlu/
│   │   ├── intent_classifier.py  # DistilBERT fine-tuned, 8 intents + rule_fallback
│   │   ├── entity_extractor.py   # spaCy NER + noun phrase
│   │   ├── sentiment_analyzer.py # Sentiment analyzer
│   │   └── coreference.py        # fastcoref FCoref
│   ├── knowledge_graph/
│   │   ├── graph.py              # KnowledgeGraph: MultiDiGraph, to_summary()
│   │   ├── relation_extractor.py # LLM-based extraction
│   │   ├── conflict_detector.py  # Rule + LLM conflict detection
│   │   └── visualizer.py         # PyVis visualization
│   ├── nlg/
│   │   ├── prompt_templates.py   # All prompt templates
│   │   ├── story_generator.py    # Story generation logic
│   │   └── option_generator.py   # Option generation logic
│   ├── engine/
│   │   ├── game_engine.py        # Orchestration pipeline
│   │   ├── runtime_session.py    # Session persistence
│   │   └── state.py              # GameState dataclass
│   ├── ui/                       # Streamlit UI modules
│   └── evaluation/
│       ├── metrics.py            # Automated metrics
│       └── llm_judge.py          # LLM-as-Judge
├── tests/                        # Comprehensive tests
├── scripts/                      # Startup scripts (start_project_prod.sh)
├── training/                     # Fine-tuning scripts
└── models/                       # Local model artifacts
```

## Detailed Pipeline (per turn)
1. Player types or clicks an option → `player_input: str`
2. **Coreference**: `CoreferenceResolver.resolve(player_input)` → `resolved_input`
3. **Intent**: `IntentClassifier.predict(resolved_input)` → `{"intent": str, "confidence": float}`
   - 8 labels: action, dialogue, explore, use_item, ask_info, rest, trade, other
4. **Entity**: `EntityExtractor.extract(resolved_input)` → list of entities
   - spaCy NER with LABEL_MAP
5. **Story Generation**: `StoryGenerator.continue_story(...)` → story_text
   - Context-aware via `kg.to_summary()` and narrative history.
6. **KG Update**: `RelationExtractor.extract(...)` → entities + relations
   - Applied to KG via `add_entity()` and `add_relation()`
7. **Conflict Detection**: `ConflictDetector.check_all(...)` → conflicts
   - Layer 1: Rule-based (exclusive relations)
   - Layer 2: LLM-based reasoning
8. **Option Generation**: `OptionGenerator.generate(...)` → list of StoryOptions
9. Return `TurnResult` to UI

## Key Design Decisions

### config.py
- Use `pydantic-settings` with BaseSettings.
- Support `NLG_MODE` (local/api/hybrid).
- Centralized constants for NLU, KG, and NLG.

### graph.py
- `nx.MultiDiGraph` for rich relationships.
- Importance scoring with decay.
- Layered summary for LLM context.

### game_engine.py
- Full pipeline orchestration.
- Integration with `RuntimeSession` for persistence.

## Implementation Rules
1. Independently testable modules with pytest.
2. Mock all API calls in tests.
3. Type hints everywhere.
4. Graceful error handling and fallbacks.

## Running the Project
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Configure .env
./scripts/start_project_prod.sh
```
