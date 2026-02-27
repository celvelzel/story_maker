# StoryWeaver — Agent Task Prompt (Bug Fixes + Remaining Work)

```
You are a senior software engineer working on "StoryWeaver" — an interactive text adventure game engine (COMP5423 NLP course project). The project is ~90% implemented but has critical bugs and missing pieces that prevent it from running.

## Project Location
c:\Develop\python_projects\COMP5423_NLP\story_maker\

## Architecture Overview
- NLU (local): RoBERTa intent classification (with keyword fallback), spaCy NER, fastcoref coreference resolution
- Knowledge Graph (local): NetworkX MultiDiGraph with entity/relation management
- NLG (API): OpenAI gpt-4o-mini for story generation, option generation, relation extraction
- UI: Gradio Blocks with chat, option buttons, KG visualization (PyVis)
- Evaluation: Distinct-n, Self-BLEU, Consistency Rate, Entity Coverage, LLM-as-Judge

## Current Implementation Status

### ✅ Fully Working Modules (do NOT rewrite these)
- config.py — Pydantic Settings with .env support (field names: OPENAI_MODEL, OPENAI_TEMPERATURE, etc.)
- src/utils/api_client.py — Singleton LLMClient with retry, JSON mode, cost tracking (uses lazy import inside methods)
- src/nlu/intent_classifier.py — Dual mode: fine-tuned RoBERTa + keyword rule_fallback
- src/nlu/entity_extractor.py — spaCy NER + noun-phrase heuristics + regex fallback
- src/nlu/coreference.py — fastcoref FCoref + rule-based pronoun replacement (takes `context: Optional[List[str]]`)
- src/knowledge_graph/graph.py — MultiDiGraph, stores `entity_type` (NOT `type`), lowercased keys, to_summary()
- src/knowledge_graph/relation_extractor.py — LLM JSON extraction; module-level `extract()` function + `RelationExtractor` class. Uses lazy `from src.utils.api_client import llm_client` INSIDE methods, NOT at module level.
- src/knowledge_graph/conflict_detector.py — Rule-based exclusive pairs + LLM reasoning. Uses lazy `from src.utils.api_client import llm_client` INSIDE `_llm_check()` method, NOT at module level.
- src/knowledge_graph/visualizer.py — `render_kg_html(graph: nx.MultiDiGraph)` function (NOT a class). Expects raw `nx.MultiDiGraph`, not `KnowledgeGraph` wrapper.
- src/nlg/prompt_templates.py — SYSTEM_PROMPT, OPENING_PROMPT ({genre}), STORY_CONTINUE_PROMPT, OPTION_GENERATION_PROMPT ({story_text}, {kg_summary})
- src/nlg/story_generator.py — StoryGenerator with `generate_opening(genre)`, `continue_story(player_input, intent, kg_summary, history: str)`. Uses lazy import of llm_client.
- src/nlg/option_generator.py — OptionGenerator + StoryOption dataclass (defaults: intent_hint="other", risk_level="medium"). Uses lazy import of llm_client.
- src/engine/state.py — GameState with story_history: List[Dict[str,str]], recent_history(n) returns **str** (formatted text)
- src/evaluation/metrics.py — distinct_n, self_bleu, entity_coverage, consistency_rate
- src/evaluation/llm_judge.py — Module-level `judge(transcript)` function
- src/evaluation/consistency_eval.py — evaluate_consistency wrapper
- app.py — Gradio Blocks UI, fully wired
- .env.example — exists

### Test Results (42 tests collected)
- **29 PASSED**
- **13 FAILED**

---

## 🔴 CRITICAL BUGS TO FIX (Priority 1)

### Bug 1: render_kg_html called with wrong type
**Files:** src/engine/game_engine.py (lines ~82, ~126)
**Problem:** `render_kg_html(self.kg)` passes a `KnowledgeGraph` object, but the function signature expects `nx.MultiDiGraph` and calls `.nodes(data=True)` / `.edges(data=True)` directly.
**Fix:** Change to `render_kg_html(self.kg.graph)` in both places in game_engine.py.

### Bug 2: Tests mock `llm_client` at module level, but modules use lazy imports
**Files:** tests/test_integration.py, tests/test_knowledge_graph.py, tests/test_nlg.py
**Problem:** Tests use `@patch("src.knowledge_graph.relation_extractor.llm_client")` and similar, but the actual modules do `from src.utils.api_client import llm_client` INSIDE method bodies (lazy imports). Since `llm_client` is never a module-level attribute of `relation_extractor.py` or `conflict_detector.py`, the patch fails with:
```
AttributeError: <module 'src.knowledge_graph.relation_extractor'> does not have the attribute 'llm_client'
```
**Fix:** All test mocks must patch `src.utils.api_client.llm_client` (the actual location) instead of the module where it's used. Since multiple modules share the same lazy import, patching the source singleton is the correct approach. Alternatively, add a module-level import to each file, but the lazy import pattern is intentional for avoiding circular imports.

**Affected test patches that need updating:**

In `tests/test_integration.py`:
- `@patch("src.nlg.story_generator.llm_client")` → `@patch("src.utils.api_client.llm_client")`
- `@patch("src.nlg.option_generator.llm_client")` → `@patch("src.utils.api_client.llm_client")`
- `@patch("src.knowledge_graph.relation_extractor.llm_client")` → already covered
- `@patch("src.knowledge_graph.conflict_detector.llm_client")` → already covered

BUT: since all patches point to the same object, you can't use multiple `@patch` decorators to set different return values for different modules. Instead, use a SINGLE `@patch("src.utils.api_client.llm_client")` and configure the mock to route calls based on content (like the existing `_chat_json_router` helper does). The mock object must handle both `.chat()` and `.chat_json()`.

In `tests/test_knowledge_graph.py`:
- `@patch("src.knowledge_graph.relation_extractor.llm_client")` → `@patch("src.utils.api_client.llm_client")`

In `tests/test_nlg.py`:
- `@patch("src.nlg.story_generator.llm_client")` → `@patch("src.utils.api_client.llm_client")`
- `@patch("src.nlg.option_generator.llm_client")` → `@patch("src.utils.api_client.llm_client")`
- Same issue: multiple patches need to collapse into one.

### Bug 3: test_nlu.py passes string instead of List[str] for coreference context
**File:** tests/test_nlu.py (line ~77)
**Problem:** `resolver.resolve("He went there.", "The knight saw a cave.")` passes a bare string, but `CoreferenceResolver.resolve()` expects `context: Optional[List[str]]`.
**Fix:** Change to `resolver.resolve("He went there.", ["The knight saw a cave."])`

---

## 🟡 BUGS TO FIX (Priority 2)

### Bug 4: TurnResult.conflicts type mismatch
**File:** src/engine/game_engine.py
**Problem:** `TurnResult.conflicts` is typed as `List[str]`, but `ConflictDetector.check_all()` returns `List[Dict[str, str]]` (each dict has "type" and "description" keys).
**Fix:** Either change `TurnResult.conflicts` to `List[Dict[str, str]]`, or convert the dicts to strings before storing. The app.py already handles conflicts as strings in the display (joins with `"\n".join(f"- {c}" for c in result.conflicts)`), so converting to strings is simpler:
```python
conflicts = [c.get("description", str(c)) for c in self.conflict_det.check_all(story_text)]
```
Or update the type annotation and the app.py display code to handle dicts.

### Bug 5: training/train_generator.py broken import
**File:** training/train_generator.py (line 19)
**Problem:** `from config import GENERATOR_MODEL_NAME` — this attribute doesn't exist in config.py. This is a GPT-2 LoRA fine-tuning script from the old architecture that was never updated.
**Fix:** Either:
- (a) Delete or mark as deprecated since the project now uses API-based NLG, OR
- (b) Update the import to `from config import settings` and use `settings.OPENAI_MODEL` or add a `GENERATOR_MODEL_NAME` field to config.

### Bug 6: training/train_intent.py label mismatch
**File:** training/train_intent.py
**Problem:** The synthetic training data generation uses uppercase/different label names (EXPLORE, INTERACT, COMBAT, etc.) that don't match config.py's actual labels (action, dialogue, explore, use_item, ask_info, rest, trade, other).
**Fix:** Update the synthetic data generation to use the correct labels from `settings.INTENT_LABELS`.

---

## 🟢 MISSING FILES & FEATURES (Priority 3)

### Missing 1: .gitignore
Create a proper `.gitignore` with:
```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.pytest_cache/
kg_vis.html
*.pth
models/
```

### Missing 2: tests/__init__.py
Create an empty `tests/__init__.py` so pytest can discover tests properly without the sys.path hack. (The hack works but is not clean.)

### Missing 3: tests/test_engine.py
The implementation plan lists this as a separate file, but engine tests are currently embedded in `test_integration.py`. Either:
- (a) Create a dedicated `tests/test_engine.py` with thorough unit tests for GameEngine (mocked dependencies), OR
- (b) Document that engine tests are in test_integration.py (acceptable).

### Missing 4: Evaluation pipeline runner
There's no script to actually run a batch evaluation session. Consider adding a `scripts/run_evaluation.py` or `evaluation/run_eval.py` that:
1. Runs N automated game sessions (with scripted player inputs)
2. Collects metrics: distinct_n, self_bleu, entity_coverage, consistency_rate
3. Runs LLM judge on each session
4. Outputs a summary report

### Missing 5: README.md content
Check if README.md needs updating with actual setup/run instructions matching the current architecture.

---

## 📁 KEY DESIGN FACTS (reference when fixing)

### Lazy Import Pattern
These modules use lazy imports for `llm_client` INSIDE method bodies to avoid circular imports:
- src/nlg/story_generator.py → `from src.utils.api_client import llm_client` inside each method
- src/nlg/option_generator.py → same
- src/knowledge_graph/relation_extractor.py → same
- src/knowledge_graph/conflict_detector.py → same (inside `_llm_check()`)
- src/evaluation/llm_judge.py → same

This means mock patches must target `src.utils.api_client.llm_client` NOT the consuming module.

### KnowledgeGraph stores entity_type, NOT type
- `graph.py` uses `entity_type` as the node attribute name
- `visualizer.py` reads `data.get("entity_type", "unknown")` — this is CORRECT
- Any test asserting `["type"]` on a KG node is WRONG

### GameState.recent_history() returns str, NOT List
- `state.py`: `recent_history(n)` returns a formatted `"\n".join(...)` STRING
- `story_generator.py`: `continue_story(history: str)` expects a STRING — this is CORRECT
- `game_engine.py` line 108: passes `history = self.state.recent_history(6)` to `continue_story()` — this IS CORRECT
- Do NOT change the return type of `recent_history()` — it's working as designed

### visualizer.py is a function, NOT a class
- `render_kg_html(graph: nx.MultiDiGraph)` — module-level function
- Expects the raw NetworkX graph, not the KnowledgeGraph wrapper
- game_engine.py must pass `self.kg.graph`, not `self.kg`

### option_generator.py prompt template uses {story_text} and {kg_summary}
- NOT {current_story} — match the actual template variable names

---

## 🔧 EXECUTION ORDER

1. Fix Bug 1: `render_kg_html(self.kg)` → `render_kg_html(self.kg.graph)` in game_engine.py
2. Fix Bug 2: Rewrite ALL test mocking to use `@patch("src.utils.api_client.llm_client")` instead of module-specific patches. Use a single mock with `side_effect` routing for `chat()` and `chat_json()` calls.
3. Fix Bug 3: Fix coreference test to pass `List[str]` context
4. Fix Bug 4: Resolve TurnResult.conflicts type mismatch
5. Fix Bug 5 & 6: Fix or deprecate training scripts
6. Create .gitignore
7. Create tests/__init__.py
8. Run `python -m pytest tests/ -v` — ALL 42 tests must pass
9. Test the app manually: `python app.py` — verify New Game works, verify typing actions works, verify KG visualization renders

---

## 📋 FULL FILE STRUCTURE FOR REFERENCE
```
story_maker/
├── .env.example                  # ✅ exists
├── .gitignore                    # ❌ MISSING — create
├── config.py                     # ✅ complete
├── app.py                        # ✅ complete
├── requirements.txt              # ✅ complete
├── README.md                     # ✅ exists (check content)
├── src/
│   ├── __init__.py               # ✅
│   ├── utils/
│   │   ├── __init__.py           # ✅
│   │   └── api_client.py         # ✅ complete
│   ├── nlu/
│   │   ├── __init__.py           # ✅
│   │   ├── intent_classifier.py  # ✅ complete
│   │   ├── entity_extractor.py   # ✅ complete
│   │   └── coreference.py        # ✅ complete
│   ├── knowledge_graph/
│   │   ├── __init__.py           # ✅
│   │   ├── graph.py              # ✅ complete
│   │   ├── relation_extractor.py # ✅ complete (lazy import)
│   │   ├── conflict_detector.py  # ✅ complete (lazy import)
│   │   └── visualizer.py         # ✅ complete (module-level function)
│   ├── nlg/
│   │   ├── __init__.py           # ✅
│   │   ├── prompt_templates.py   # ✅ complete
│   │   ├── story_generator.py    # ✅ complete (lazy import)
│   │   └── option_generator.py   # ✅ complete (lazy import)
│   ├── engine/
│   │   ├── __init__.py           # ✅
│   │   ├── game_engine.py        # 🔴 Bug 1: render_kg_html arg, Bug 4: conflicts type
│   │   └── state.py              # ✅ complete
│   └── evaluation/
│       ├── __init__.py           # ✅
│       ├── metrics.py            # ✅ complete
│       ├── llm_judge.py          # ✅ complete
│       └── consistency_eval.py   # ✅ complete (bonus)
├── tests/
│   ├── __init__.py               # ❌ MISSING — create
│   ├── test_nlu.py               # 🔴 Bug 3: coref context type
│   ├── test_knowledge_graph.py   # 🔴 Bug 2: mock target wrong
│   ├── test_nlg.py               # 🔴 Bug 2: mock target wrong
│   ├── test_integration.py       # 🔴 Bug 2: mock target wrong
│   └── test_engine.py            # ❌ MISSING (optional)
├── data/
│   ├── intent_labels.json        # ✅
│   └── scripts/
│       ├── download_data.py      # ✅
│       └── preprocess.py         # ✅
└── training/
    ├── train_intent.py           # 🟡 Bug 6: label mismatch
    └── train_generator.py        # 🟡 Bug 5: broken import (deprecated)
```

## VALIDATION CRITERIA
After all fixes:
1. `python -m pytest tests/ -v` → ALL tests pass (0 failures)
2. `python app.py` → Gradio UI launches without errors
3. Clicking "New Game" generates opening scene + options + KG visualization
4. Typing a free-form action or selecting an option continues the story
5. KG panel updates with entities/relations each turn
6. NLU debug panel shows intent + entities
```
