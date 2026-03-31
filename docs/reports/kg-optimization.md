# Knowledge Graph Optimization Report

**Commit Hash**: `d1d59c7` + `b1799ed` | **Last Updated**: 2026-03-31

---

## 1. Overview

This report details the comprehensive enhancements made to the Knowledge Graph (KG) subsystem, covering data model richness, per-turn update logic, conflict resolution strategies, and a frontend configuration panel. **76 new unit tests** were added and are all passing.

### 1.1 Summary of Changes

| File | Change Count | Type | Description |
|------|--------------|------|-------------|
| `config.py` | +21 lines | Update | Added strategy configurations and tuning parameters. |
| `src/knowledge_graph/graph.py` | +300 lines | Rewrite | Data structures, new methods, layered summaries. |
| `src/knowledge_graph/relation_extractor.py` | +144 lines | Rewrite | Enhanced prompts and dual extraction. |
| `src/knowledge_graph/conflict_detector.py` | +195 lines | Rewrite | Strategy pattern with two implementations. |
| `src/engine/game_engine.py` | +135 lines | Rewrite | 7-step turn processing with strategy injection. |
| `app.py` | +58 lines | Update | Added KG strategy settings panel. |
| `tests/` | +900+ lines | New | Added 76 unit tests for graph, extractor, and engine. |

---

## 2. Feature Details

### 2.1 Data Model Enhancements
**Nodes (Entities)** now track:
- `name`, `entity_type`, `description` (narrative details).
- `status`: A dynamic dictionary (e.g., `{"health": "injured"}`).
- **Metrics**: `created_turn`, `last_mentioned_turn`, `mention_count`, `player_mention_count`, and a calculated `importance_score` (0-1).

**Edges (Relations)** now track:
- `relation` type, `context` (why the relation exists).
- `created_turn`, `last_confirmed_turn`, and `confidence` score (0-1).

### 2.2 New Core Methods
- `update_entity_state(...)`: Updates entity status dictionary.
- `refresh_mentions(...)`: Batch updates mention counts and boosts importance.
- `apply_decay(...)`: Reduces confidence of old relations and prunes them if below threshold.
- `recalculate_importance()`: Calculates importance based on degree, recency, and mentions.

### 2.3 Layered Summary Output
Supports two modes via `KG_SUMMARY_MODE`:
1. **Layered (Default)**: Groups entities by importance (Core, Secondary, Background) and provides a recent timeline.
2. **Flat (Legacy)**: Simple list of entities and relations for backward compatibility.

### 2.4 Dual Entity Extraction
Extracts entities and relations from both **Player Input** and **Story Text** simultaneously:
- Merges results and deduplicates.
- Prioritizes richer descriptions from the story text.
- Captures direct player interactions more accurately.

### 2.5 Conflict Resolution Strategies
Uses the Strategy Pattern for flexible conflict handling:
- **`keep_latest`**: Fast, rule-based resolution favoring the most recent confirmation. No LLM calls.
- **`llm_arbitrate` (Default)**: Sends conflict details to the LLM for high-accuracy arbitration.

---

## 3. Configuration & Usage

### 3.1 Frontend Configuration Panel
A new **"⚙ KG Strategy Settings"** panel in the Streamlit sidebar allows real-time adjustment of:
- Conflict Resolution Strategy
- Extraction Mode
- Summary Format
- Importance Calculation Strategy

### 3.2 Global Configuration (.env)
```env
KG_CONFLICT_RESOLUTION=llm_arbitrate
KG_EXTRACTION_MODE=dual_extract
KG_SUMMARY_MODE=layered
KG_IMPORTANCE_DECAY_FACTOR=0.95
KG_RELATION_DECAY_FACTOR=0.90
```

---

## 4. Architecture & Logic

### 4.1 Turn Processing Pipeline (Updated)
1. NLU (Coref, Intent, Sentiment, Entity).
2. Story Generation.
3. **KG Update**:
   - Extraction (Dual or Story-only).
   - Apply state changes.
   - Refresh mentions & apply decay.
   - Recalculate importance.
4. **Conflict Detection & Resolution**.
5. Option Generation.

### 4.2 Importance Score Formula
`importance = 0.3 * norm(degree) + 0.3 * recency + 0.2 * norm(mentions) + 0.2 * norm(player_mentions)`

---

## 5. Testing & Verification

| Test Suite | Count | Coverage |
|------------|-------|----------|
| `test_graph_enhanced.py` | 35 | Node/Edge props, decay, importance, timeline. |
| `test_relation_extractor_enhanced.py` | 11 | Rich extraction, normalization, merging. |
| `test_conflict_resolution.py` | 14 | Detection, KeepLatest, LLM Arbitrate. |
| `test_engine_enhanced.py` | 16 | Strategy injection, integration, time tracking. |

**Total**: 76 Passed.

---

## 6. Known Limitations
- **LLM Cost**: `llm_arbitrate` and `dual_extract` add 1-2 API calls per turn.
- **Persistence**: While optimized, the graph resides in memory (Update: See [Persistence Report](runtime-persistence.md) for recent fixes).
