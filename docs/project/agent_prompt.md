# StoryWeaver Agent Prompt

```
You are a senior software engineer implementing "StoryWeaver" — an interactive text adventure game engine for an NLP course project (COMP5423). The architecture is a hybrid design: local NLU + API-based NLG.

## Project Location
c:\Develop\python_projects\COMP5423_NLP\story_maker\

## Architecture
- NLU (local): RoBERTa intent classification, spaCy NER, fastcoref coreference resolution
- Knowledge Graph (local): NetworkX MultiDiGraph with entity/relation management
- NLG (API): OpenAI gpt-4o-mini for story generation, option generation, relation extraction
- UI: Gradio Blocks with chat, option buttons, KG visualization (PyVis)
- Evaluation: Distinct-n, Self-BLEU, Consistency Rate, Entity Coverage, LLM-as-Judge

## Complete File Structure
```
story_maker/
├── .env                          # OPENAI_API_KEY=sk-... (git-ignored)
├── .env.example                  # Template
├── .gitignore
├── config.py                     # Pydantic Settings, reads .env
├── app.py                        # Gradio frontend
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── api_client.py         # Singleton OpenAI wrapper, retry, cost tracking
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py  # RoBERTa fine-tuned, 8 intents + rule_fallback
│   │   ├── entity_extractor.py   # spaCy NER + noun phrase, LABEL_MAP
│   │   └── coreference.py        # fastcoref FCoref, pronoun→antecedent
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── graph.py              # KnowledgeGraph: MultiDiGraph, add/get/remove, to_summary()
│   │   ├── relation_extractor.py # LLM-based entity+relation extraction from story text
│   │   ├── conflict_detector.py  # Rule layer (exclusive pairs, attr conflicts) + LLM layer
│   │   └── visualizer.py         # PyVis HTML, entity type colors/shapes
│   ├── nlg/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py   # SYSTEM_PROMPT, STORY_CONTINUE_PROMPT, OPTION_GENERATION_PROMPT, OPENING_PROMPT
│   │   ├── story_generator.py    # generate_opening(), continue_story() via LLM
│   │   └── option_generator.py   # StoryOption dataclass, generate() → list[StoryOption], fallback
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── game_engine.py        # GameEngine: start_game(), process_turn() → TurnResult
│   │   └── state.py              # GameState dataclass
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py            # distinct_n, self_bleu, entity_coverage, consistency_rate
│       └── llm_judge.py          # LLMJudge: 5 dimensions (narrative, consistency, agency, creativity, pacing)
├── tests/
│   ├── __init__.py
│   ├── test_nlu.py
│   ├── test_knowledge_graph.py
│   ├── test_nlg.py
│   ├── test_engine.py
│   └── test_integration.py
├── data/
│   ├── intent_labels.json
│   └── scripts/
│       └── download_data.py
└── training/
    └── train_intent.py
```

## Detailed Pipeline (per turn)

1. Player types or clicks an option → `player_input: str`
2. **Coreference**: `CoreferenceResolver.resolve(player_input)` → `resolved_input`
3. **Intent**: `IntentClassifier.predict(resolved_input)` → `{"intent": str, "confidence": float}`
   - 8 labels: action, dialogue, explore, use_item, ask_info, rest, trade, other
   - Has `rule_fallback()` keyword matcher for when model is unavailable
4. **Entity**: `EntityExtractor.extract(resolved_input)` → `list[{"text", "type", "start", "end", "source"}]`
   - spaCy NER with LABEL_MAP (PERSON→person, GPE/LOC/FAC→location)
   - Noun phrase extraction with `_infer_type()` for creature/location/item/person
5. **Story Generation**: `StoryGenerator.continue_story(resolved_input, intent, kg.to_summary(), history)` → story_text
   - Uses SYSTEM_PROMPT + STORY_CONTINUE_PROMPT template
   - `kg.to_summary()` converts KG to text for LLM context
6. **KG Update**: `RelationExtractor.extract(combined_text)` → entities + relations
   - LLM extracts structured data in JSON mode
   - Applied to KG via `add_entity()` and `add_relation()`
7. **Conflict Detection**: `ConflictDetector.check_all(story_text)` → conflicts
   - Layer 1: Rule-based (exclusive relation pairs, attribute contradictions)
   - Layer 2: LLM reasoning against KG summary
8. **Option Generation**: `OptionGenerator.generate(story_text, kg.to_summary())` → list[StoryOption]
   - Each option has text, intent_hint, risk_level
   - Fallback returns 3 default options on error
9. Return `TurnResult(story_text, options, nlu_debug, kg_html, conflicts)` to UI

## Key Design Decisions

### config.py
- Use `pydantic-settings` with BaseSettings, reads from `.env`
- All constants centralized: API config, intent labels, KG limits, gradio port
- Intent labels: ["action", "dialogue", "explore", "use_item", "ask_info", "rest", "trade", "other"]

### api_client.py
- Singleton pattern (LLMClient._instance)
- `chat()`: general text response with retry (3 attempts, exponential backoff)
- `chat_json()`: calls chat with json_mode=True, returns parsed dict
- Cost tracking: input/output tokens, USD calculation (gpt-4o-mini pricing)
- Global instance: `llm_client = LLMClient()`

### graph.py — KnowledgeGraph
- `nx.MultiDiGraph` (supports multiple edges between same node pair)
- Node keys are lowercased names; store original name in `name` attribute
- `add_entity(name, entity_type, **attrs)`: upsert node
- `add_relation(source, target, relation, **attrs)`: check for duplicate relation before adding
- `to_summary(max_entities=30)`: produces "=== World State ===" + "=== Relations ===" text block
- `_enforce_limit()`: removes least-connected node when exceeding KG_MAX_NODES (200)

### conflict_detector.py
- `EXCLUSIVE_PAIRS = [("ally_of", "enemy_of"), ("alive", "dead")]`
- `_rule_based_check()`: iterate edges, check for opposing relations; check dead+positive health
- `_llm_check(new_text)`: send KG summary + new text, ask LLM to identify contradictions in JSON

### prompt_templates.py
- SYSTEM_PROMPT: 7 rules for the storyteller (second person, 2-4 paragraphs, consistency, vivid, etc.)
- STORY_CONTINUE_PROMPT: template with {kg_summary}, {history}, {intent}, {player_input}
- OPTION_GENERATION_PROMPT: template requesting {num_options} options in JSON with text/intent_hint/risk_level
- OPENING_PROMPT: instructions for generating opening scene

### option_generator.py
- `StoryOption` dataclass: text, intent_hint, risk_level
- On any exception, return 3 hardcoded fallback options
- Uses json_mode for structured response

### game_engine.py
- `TurnResult` dataclass: story_text, options, nlu_debug, kg_html, conflicts
- `start_game()`: generate opening → extract KG → generate options → return TurnResult
- `process_turn(player_input)`: full 7-step pipeline as described above
- `_apply_extraction(extraction)`: iterate entities/relations and add to KG

### app.py — Gradio UI
- `gr.Blocks` with `gr.themes.Soft()`
- Layout: Left column (3/5) = Chatbot + TextInput + Radio options + buttons; Right column (2/5) = KG HTML + NLU debug
- `start_new_game()`: creates GameEngine, calls start_game()
- `player_action()`: uses typed message or selected radio option
- Chatbot type="messages" (OpenAI-style message format)

### llm_judge.py
- 5 dimensions: narrative_quality, consistency, player_agency, creativity, pacing (1-10 each)
- `evaluate_session(session_turns)`: formats turns, sends to LLM with JSON mode
- Returns scores dict + overall score + feedback text

## Implementation Rules

1. Every module must be independently testable with pytest
2. All LLM calls go through `llm_client` singleton from `src/utils/api_client.py`
3. No direct `openai` imports outside of `api_client.py`
4. KG node keys are always lowercased
5. All external imports have try/except with graceful fallback where possible
6. Tests should mock API calls (never make real API calls in tests)
7. Use type hints everywhere
8. Keep each file focused — one class/concern per file
9. Gradio UI must work with both typed input and clicked options
10. Error handling: never crash the game; log errors and provide fallback

## Testing Strategy

### Unit Tests (mock all external dependencies)
- test_nlu.py: intent classification (model + rule_fallback), entity extraction, coreference
- test_knowledge_graph.py: graph CRUD, to_summary, relation extraction (mock LLM), conflict detection, visualization
- test_nlg.py: prompt template formatting, story generation (mock LLM), option generation (mock LLM + fallback)
- test_engine.py: GameEngine pipeline with all mocked dependencies

### Integration Tests
- test_integration.py: Full pipeline start_game → process_turn × N with mocked API client

### Evaluation Tests
- Test distinct_n, self_bleu, entity_coverage, consistency_rate with known inputs
- Test LLMJudge with mocked responses

## Running the Project

```bash
# Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # Then add your OPENAI_API_KEY

# Run tests
python -m pytest tests/ -v

# Launch UI
python app.py
```

## Current State
There is existing scaffold code from a pure-local architecture. You need to REPLACE the existing files to implement the hybrid architecture as specified above. Key changes:
- config.py: Add API settings, use pydantic-settings
- NEW: src/utils/api_client.py
- NEW: src/knowledge_graph/relation_extractor.py
- NEW: src/evaluation/llm_judge.py
- REWRITE: All NLG modules (API-based instead of GPT-2)
- REWRITE: game_engine.py (TurnResult-based, new pipeline)
- REWRITE: app.py (option buttons, new layout)
- REWRITE: conflict_detector.py (add LLM layer)
- UPDATE: entity_extractor.py (add noun phrases)
- UPDATE: All tests (mock API, new interfaces)

Please implement all files in order: config → api_client → NLU modules → KG modules → NLG modules → engine → evaluation → app → tests. Ensure all tests pass.
```

---

## 使用说明

1. 将上方 ` ``` ` 中的全部内容复制
2. 作为 System Prompt 或 User Prompt 提供给 Coding Agent (如 GitHub Copilot Agent, Cursor, Windsurf 等)
3. Agent 将按照指定顺序自动创建/重写所有文件
4. 实现完成后运行 `python -m pytest tests/ -v` 验证

---

## 注意事项

- 确保 `.env` 文件中已配置有效的 `OPENAI_API_KEY`
- Agent 实现时应 mock 所有 API 调用的测试
- 如需调整模型或参数，修改 `config.py` 中的 Settings 即可
- 详细的模块设计参见 `implementation_plan.md`
